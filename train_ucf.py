"""Train CST-Flow on paired volumes using a frozen coarse generator."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from mlrd.engine import scaled_optimizer_step
from ucf.anchor import FrozenAnchor, AnchorCache
from ucf.data import PairedDataset, read_metadata, training_fingerprint, sha
from ucf.losses import objective
from ucf.model import build_model
from ucf.runtime import (atomic_json, save_checkpoint, setup, autocast, evaluate,
                         source_fingerprint)
from wave_tools import haar


def validate_config(cfg):
    tr, loss = cfg['training'], cfg['loss']
    if cfg.get('architecture') == 'original_ct' and (
            cfg['model']['sources'] != ['mri'] or cfg['model']['targets'] != ['ct']
            or (loss['steps'], loss['solver']) != (2, 'euler')):
        raise ValueError('Original CT adaptation requires MRI, CT and two Euler steps')
    for key in ('epochs','batch_size','accumulation_steps','patience','cpu_threads'):
        if type(tr[key]) is not int or tr[key]<1:
            raise ValueError(key+' must be a positive integer')
    if tr['precision'] not in ('fp32','fp16','bf16'):
        raise ValueError('Unknown precision')
    if len(cfg['spatial_size'])!=3 or any(type(n) is not int or n<8 or n%4 for n in cfg['spatial_size']):
        raise ValueError('Anchor input requires 3 dimensions divisible by 4, each >=8')
    if loss['solver'] not in ('euler','heun') or type(loss['steps']) is not int or loss['steps']<1:
        raise ValueError('Invalid rollout settings')
    if not 0<=loss['anchor_probability']<=1:
        raise ValueError('Invalid anchor probability')
    if loss['huber_bands'] not in ([],[3,5,6,7],list(range(8))):
        raise ValueError('Huber bands must be [] (MSE), [3,5,6,7], or all eight bands')
    required_weights = {'fm','endpoint','rollout','ssim','cfam','spectrum','schedule'}
    optional_weights = {'final_detail', 'final_high', 'gradient', 'trajectory', 'bone', 'edge'}
    if not required_weights <= set(loss['weights']) <= required_weights|optional_weights:
        raise ValueError('Incorrect loss weights')
    if loss['weights'].get('trajectory', 0) and (loss['solver'] != 'euler' or loss['steps'] < 2
                                                or cfg.get('architecture', 'ucf') != 'ucf'):
        raise ValueError('Trajectory supervision requires UCF with at least two Euler steps')
    if (loss['weights'].get('bone', 0) or loss['weights'].get('edge', 0)) and cfg['model']['targets'] != ['ct']:
        raise ValueError('Bone and edge thresholds require the configured CT HU scale')
    if any(not math.isfinite(v) or v<0 for v in loss['weights'].values()):
        raise ValueError('Loss weights must be finite and nonnegative')
    if cfg['model']['dynamic'] and loss['weights']['rollout']<=0:
        raise ValueError('Learned schedule needs rollout supervision')
    if any(tr[k]<=0 or not math.isfinite(tr[k]) for k in ('lr','grad_clip')):
        raise ValueError('Invalid optimization settings')


@torch.no_grad()
def fit_thresholds(dataset, cache, device, precision, model):
    sq = torch.zeros(8,dtype=torch.float64,device=device)
    condition_sq = torch.zeros_like(model.condition_rms, dtype=torch.float64) if hasattr(model, 'condition_rms') else None
    count = 0.
    for i, batch in enumerate(DataLoader(dataset,batch_size=1,num_workers=0)):
        mri, target = batch['mri'].to(device), batch['target'].to(device)
        mask = batch['support'].to(device).float() if 'support' in batch else (mri.abs()>1e-6).float()
        with autocast(device,precision):
            anchor = cache.get(mri,batch['mri_path'])*mask
        residual = haar((target*mask-anchor)/model.scale)
        weight = F.avg_pool3d(mask,2)
        sq += (residual.double().square()*weight).sum((0,2,3,4))
        if condition_sq is not None:
            for index, image in enumerate((mri, anchor)):
                condition_sq[index] += (haar(image).double().square()*weight).sum((0,2,3,4))
        count += float(weight.sum())
        if i%25==0:
            print(json.dumps(dict(event='train_only_statistics',scan=i+1,total=len(dataset))),flush=True)
    if not count:
        raise ValueError('Training data contain no source support')
    if condition_sq is not None:
        model.condition_rms.copy_((condition_sq/count).sqrt().clamp_min(1e-3))
    return (sq/count).sqrt().clamp_min(1e-3).float()


def train(args):
    cfg = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.seed is not None:
        cfg['seed'] = args.seed
    if args.no_cfam:
        cfg['model']['cfam'] = False
    if args.no_dynamic:
        cfg['model']['dynamic'] = False
    validate_config(cfg)
    tr = cfg['training']
    device = setup(cfg['seed'],args.device,tr['precision'],tr['cpu_threads'],tr.get('deterministic_warn_only',False))
    frame = read_metadata(args.metadata,args.dataset_root)
    train_ds = PairedDataset(frame,'train',cfg['spatial_size'],args.tensor_cache)
    val_ds = PairedDataset(frame,'val',cfg['spatial_size'],args.tensor_cache)
    run = Path(args.run_dir)
    if args.resume and not (run/'last.pt').is_file():
        raise FileNotFoundError('Resume requires last.pt')
    if not args.resume and run.exists() and any(run.iterdir()):
        raise FileExistsError('Use a new run directory or --resume')
    run.mkdir(parents=True,exist_ok=True)
    anchor = FrozenAnchor(args.anchor).to(device)
    if list(anchor.cfg['data']['spatial_size']) != cfg['spatial_size']:
        raise ValueError('Use the anchor training spatial_size')
    # Checkpoints can be relocated; identity comes from content rather than path.
    manifest = dict(config=cfg, anchor_sha256=anchor.digest,
                    metadata_sha256=sha(args.metadata), source=source_fingerprint(),
                    images=training_fingerprint(frame),
                    runtime=dict(torch=str(torch.__version__),device=str(device),
                                 cuda=torch.version.cuda),
                    test_images_read=False)
    if (run/'manifest.json').exists():
        if json.loads((run/'manifest.json').read_text(encoding='utf-8'))!=manifest:
            raise ValueError('Data, anchor, source, config or runtime changed; start a new run')
    atomic_json(run/'manifest.json',manifest)
    atomic_json(run/'config.json',cfg)
    cache = AnchorCache(anchor,args.anchor_cache or run/'anchor_cache',tr['precision'])
    model = build_model(cfg).to(device)
    initialization = None
    if args.init_checkpoint and args.resume:
        recorded = run/'initialization.json'
        if not recorded.is_file() or json.loads(recorded.read_text())['checkpoint_sha256'] != sha(args.init_checkpoint):
            raise ValueError('Resumed initialization differs from the recorded checkpoint')
    if args.init_checkpoint and not args.resume:
        initial = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
        if initial.get('format_version') != 'ucf-mri2pet-v1' or initial['anchor_sha256'] != anchor.digest:
            raise ValueError('Initialization must use the same frozen anchor')
        if initial['config']['model'] != cfg['model'] or initial['config']['spatial_size'] != cfg['spatial_size']:
            raise ValueError('Initialization model configuration differs')
        if set(initial['training_subjects']) != set(train_ds.rows.subject_id) or set(initial['validation_subjects']) != set(val_ds.rows.subject_id):
            raise ValueError('Initialization split membership differs')
        model.load_state_dict(initial['model'], strict=True)
        initialization = dict(checkpoint_sha256=sha(args.init_checkpoint), source_epoch=initial['epoch'])
        if (run/'initialization.json').exists() and json.loads((run/'initialization.json').read_text()) != initialization:
            raise ValueError('Initialization checkpoint changed')
        atomic_json(run/'initialization.json', initialization)
    optimizer = torch.optim.AdamW(model.parameters(),lr=tr['lr'],weight_decay=tr['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,
                             patience=tr['lr_patience'],min_lr=1e-6)
    scaler = torch.amp.GradScaler('cuda',enabled=(device.type=='cuda' and tr['precision']=='fp16'),
                                 init_scale=tr.get('amp_initial_scale',65536.))
    start, stale, best, history = 1, 0, math.inf, []
    if args.resume:
        saved = torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        thresholds = saved['thresholds'].to(device)
        start, stale, best = saved['epoch']+1,saved['stale'],saved['best']
        history = saved['history']
        torch.set_rng_state(saved['rng_cpu'])
        if device.type=='cuda':
            torch.cuda.set_rng_state_all(saved['rng_cuda'])
    elif initialization is not None:
        thresholds = initial['thresholds'].to(device)
    else:
        thresholds = fit_thresholds(train_ds,cache,device,tr['precision'],model)
    initialization_path = run/'initialization.json'
    if initialization_path.exists():
        initialization = json.loads(initialization_path.read_text())
    statistics = dict(fit_split='train', thresholds=thresholds.tolist(),
                      sample_ids=train_ds.rows.sample_id.tolist(), residual_scale=model.scale)
    if hasattr(model, 'condition_rms'):
        statistics['condition_rms'] = dict(zip(('source', 'anchor'), model.condition_rms.tolist()))
    atomic_json(run/'statistics.json', statistics)
    val_loader = DataLoader(val_ds,batch_size=1,num_workers=0)
    full_volume = cfg.get('full_volume_validation', False)

    def weights(epoch):
        return dict(format_version='ucf-mri2pet-v1',model=model.state_dict(),config=cfg,
                    epoch=epoch,anchor_sha256=anchor.digest,thresholds=thresholds.cpu(),
                    training_subjects=sorted(train_ds.rows.subject_id.unique().tolist()),
                    validation_subjects=sorted(val_ds.rows.subject_id.unique().tolist()),
                    training_cohort=anchor.cfg.get('cohort','unknown'), initialization=initialization)

    if not args.resume:
        summary, rows = evaluate(model,anchor,val_loader,device,tr['precision'],
                                 cfg['loss']['steps'],cfg['loss']['solver'],cache,full_volume=full_volume)
        best = summary['subject']['mae']
        save_checkpoint(run/'best.pt',dict(**weights(0),validation=summary))
        pd.DataFrame(rows).to_csv(run/'best_val_per_scan.csv',index=False)
        atomic_json(run/'initial_validation.json',summary)
    for epoch in range(start,tr['epochs']+1):
        if stale>=tr['patience'] or (run/'STOP_REQUESTED').exists():
            break
        started = time.monotonic()
        model.train()
        loader = DataLoader(train_ds,batch_size=tr['batch_size'],shuffle=True,num_workers=0,
                  generator=torch.Generator().manual_seed(cfg['seed']+epoch))
        total = {}
        updates, skipped, consecutive = 0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for i,batch in enumerate(loader):
            mri, target = batch['mri'].to(device), batch['target'].to(device)
            with autocast(device,tr['precision']):
                coarse = cache.get(mri,batch['mri_path'])
                support = batch['support'].to(device) if 'support' in batch else None
                loss, terms = objective(model,mri,target,coarse,thresholds,cfg['loss'],support=support)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            # Weight by actual examples, including a partial final microbatch.
            group_start = (i//tr['accumulation_steps'])*tr['accumulation_steps']*tr['batch_size']
            group_size = min(tr['accumulation_steps']*tr['batch_size'],len(train_ds)-group_start)
            scaler.scale(loss*len(mri)/group_size).backward()
            if (i+1)%tr['accumulation_steps']==0 or i+1==len(loader):
                success = scaled_optimizer_step(model.parameters(),optimizer,scaler,tr['grad_clip'])
                updates += int(success)
                skipped += int(not success)
                consecutive = 0 if success else consecutive+1
                if consecutive>=8:
                    raise FloatingPointError('Eight consecutive AMP overflows')
                optimizer.zero_grad(set_to_none=True)
            for name,value in dict(total=float(loss.detach()),**terms).items():
                total[name] = total.get(name,0.)+value*len(mri)
            if i%25==0:
                print(json.dumps(dict(event='train',epoch=epoch,batch=i+1,batches=len(loader),loss=float(loss.detach()))),flush=True)
        if updates==0:
            raise FloatingPointError('No successful optimizer updates in epoch')
        summary, rows = evaluate(model,anchor,val_loader,device,tr['precision'],
                                 cfg['loss']['steps'],cfg['loss']['solver'],cache,full_volume=full_volume)
        score = summary['subject']['mae']
        scheduler.step(score)
        if score < best-tr['min_delta']:
            best, stale = score, 0
            save_checkpoint(run/'best.pt',dict(**weights(epoch),validation=summary))
            pd.DataFrame(rows).to_csv(run/'best_val_per_scan.csv',index=False)
        else:
            stale += 1
        record = dict(epoch=epoch,train={k:v/len(train_ds) for k,v in total.items()},
                      val_subject_mae=score,best=best,seconds=time.monotonic()-started,
                      lr=optimizer.param_groups[0]['lr'],optimizer_updates=updates,
                      amp_skipped_updates=skipped,amp_scale=scaler.get_scale())
        history.append(record)
        save_checkpoint(run/'last.pt',dict(**weights(epoch),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),stale=stale,best=best,
            history=history,rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else []))
        atomic_json(run/'history.json',history)
        print(json.dumps(record),flush=True)
        if args.stop_after_epoch is not None and epoch>=args.stop_after_epoch:
            break
    interrupted = (run/'STOP_REQUESTED').exists() or (args.stop_after_epoch is not None and
                    bool(history) and history[-1]['epoch']<tr['epochs'] and stale<tr['patience'])
    atomic_json(run/'status.json',dict(status='stopped' if interrupted else 'complete',
                completed_epochs=history[-1]['epoch'] if history else 0,best_val_subject_mae=best,
                early_stopped=stale>=tr['patience'],test_evaluated=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('metadata','dataset-root','anchor','run-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--config',default=str(Path(__file__).parent/'configs/cst_mri2ct.json'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--seed',type=int)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--init-checkpoint',help='Initialize a fresh fine-tuning run with matching model, anchor and subjects')
    p.add_argument('--tensor-cache',help='Shared lossless FP32 paired-volume cache')
    p.add_argument('--anchor-cache',help='Shared content-addressed frozen anchor cache')
    p.add_argument('--no-cfam',action='store_true')
    p.add_argument('--no-dynamic',action='store_true')
    p.add_argument('--stop-after-epoch',type=int,help='Save and stop at an epoch boundary; resume with the same config')
    train(p.parse_args())


if __name__=='__main__':
    main()
