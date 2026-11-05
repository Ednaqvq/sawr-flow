import contextlib
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from mlrd.losses import ssim3d
from wave_tools import BANDS, band_metrics, haar
from .data import sha


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def save_checkpoint(path, state):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(state,temp)
    temp.replace(path)


def setup(seed, device, precision, threads=8, deterministic_warn_only=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    device = torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use --device cpu for smoke tests')
    if device.type!='cuda' and precision!='fp32':
        raise ValueError('Use fp32 on CPU')
    if precision=='bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 unsupported; choose fp16 or fp32 in the config')
    torch.use_deterministic_algorithms(True,warn_only=deterministic_warn_only)
    torch.backends.cudnn.benchmark = False
    return device


def autocast(device, precision):
    if precision=='fp32':
        return contextlib.nullcontext()
    return torch.autocast(device.type,dtype={'bf16':torch.bfloat16,'fp16':torch.float16}[precision])


def source_fingerprint():
    src = Path(__file__).resolve().parents[1]
    paths = list(src.rglob('*.py'))+list(src.parent.glob('*ucf.py'))
    return {str(p.relative_to(src.parent)):sha(p) for p in sorted(paths)}


def metrics(pred, target, mri, ids, subjects, prefix='', evaluation_mask=None):
    mask = (mri.abs()>1e-6).float() if evaluation_mask is None else evaluation_mask.float()
    if not torch.isfinite(pred).all():
        raise FloatingPointError('Non-finite prediction')
    bands, _ = band_metrics(pred,target,mask)
    ss = ssim3d(pred,target,2.,mask=mask)
    full_ss = ssim3d(pred,target,2.)
    rows = []
    for i in range(len(pred)):
        error, weight = pred[i]-target[i], mask[i]
        mse = (error.square()*weight).sum()/weight.sum()
        row = dict(sample_id=ids[i],subject_id=subjects[i])
        values = dict(mae=float((error.abs()*weight).sum()/weight.sum()),
                      psnr=float(10*torch.log10(4/mse.clamp_min(1e-12))),ssim=float(ss[i]),
                      full_mae=float(error.abs().mean()),full_ssim=float(full_ss[i]),
                      full_psnr=float(10*torch.log10(4/error.square().mean().clamp_min(1e-12))))
        values.update({f'wave_{b}_mae':float(bands[i,j]) for j,b in enumerate(BANDS)})
        row.update({prefix+k:v for k,v in values.items()})
        rows.append(row)
    return rows


def summarize(rows):
    frame = pd.DataFrame(rows)
    cols = [c for c in frame if c not in ('sample_id','subject_id')]
    if not np.isfinite(frame[cols].to_numpy()).all():
        raise ValueError('Non-finite evaluation metrics')
    grouped = frame.groupby('subject_id')[cols].mean()
    return dict(scans=len(frame),subjects=len(grouped),scan=frame[cols].mean().to_dict(),
                subject=grouped.mean().to_dict())


@torch.no_grad()
def evaluate(model, anchor, loader, device, precision, steps, solver, cache=None, *, residual_stats=False,
             full_volume=False):
    model.eval()
    rows = []
    for batch in loader:
        mri, target = batch['mri'].to(device), batch['target'].to(device)
        support = batch['support'].to(device) if 'support' in batch else None
        roi = batch['evaluation_mask'].to(device) if 'evaluation_mask' in batch else support
        if full_volume:
            roi = torch.ones_like(target)
        with autocast(device,precision):
            coarse = cache.get(mri,batch['mri_path']) if cache else anchor(mri)
            cond = model.conditions(mri,coarse,support=support)
            pred = model.generate(cond,steps,solver)
        anchor_image = cond['anchor']+model.background_value*(1-cond['mask'])
        result = metrics(pred,target,mri,batch['sample_id'],batch['subject_id'],evaluation_mask=roi)
        base = metrics(anchor_image,target,mri,batch['sample_id'],batch['subject_id'],'anchor_',roi)
        if residual_stats:
            interior = torch.nn.functional.avg_pool3d(cond['mask'].float(), 2) == 1
            proposed = haar((pred - anchor_image) * cond['mask'])
            required = haar((target - anchor_image) * cond['mask'])
        for i, (row, original) in enumerate(zip(result,base)):
            row.update({k:v for k,v in original.items() if k.startswith('anchor_')})
            if full_volume:
                for metric in ('mae', 'psnr', 'ssim'):
                    row[metric] = row['full_' + metric]
                    row['anchor_' + metric] = row['anchor_full_' + metric]
            if cond['target']=='ct':
                row.update(mae_hu=row['mae']*2012.,anchor_mae_hu=row['anchor_mae']*2012.)
                if roi is not None and not full_volume:
                    row['roi_source_coverage'] = float((roi[i]*cond['mask'][i]).sum()/roi[i].sum())
            row.update({f'route_{b}':float(cond['k'][i,j]) for j,b in enumerate(BANDS)})
            if residual_stats:
                inside = interior[i, 0]
                if not inside.any():
                    raise ValueError('No interior blocks for residual diagnostics')
                p, r = proposed[i][:, inside], required[i][:, inside]
                active = (p.abs() > 1e-8) & (r.abs() > 1e-8)
                error = (p-r).abs()
                values = dict(
                    predicted_mean=p.mean(1), required_mean=r.mean(1),
                    predicted_abs=p.abs().mean(1), required_abs=r.abs().mean(1),
                    mae=error.mean(1), harm=(error-r.abs()).clamp_min(0).mean(1),
                    active_fraction=active.float().mean(1),
                    sign_agreement=((p*r > 0) & active).sum(1)/active.sum(1).clamp_min(1),
                )
                for key, by_band in values.items():
                    row.update({f'residual_{band}_{key}':value for band,value in zip(BANDS,by_band.tolist())})
        rows.extend(result)
    return summarize(rows), rows
