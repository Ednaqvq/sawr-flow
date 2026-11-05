"""Separate held-out evaluation, with subject-equal summaries and band errors."""
import argparse
import json
from pathlib import Path
from inference_ucf import Predictor
import pandas as pd
from torch.utils.data import DataLoader
from ucf.data import read_metadata, PairedDataset
from ucf.runtime import evaluate, atomic_json
from ucf.anchor import AnchorCache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','anchor','metadata','dataset-root','output-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--precision',choices=['fp32','fp16','bf16'])
    p.add_argument('--partition',choices=['val','test'],default='test')
    p.add_argument('--full-volume',action=argparse.BooleanOptionalAction,default=None,
                   help='Evaluate all voxels and all Haar blocks (default: use checkpoint config)')
    p.add_argument('--steps',type=int)
    p.add_argument('--solver',choices=['euler','heun'])
    p.add_argument('--operator-intervention',choices=['none','diagonal','spatial_flip'],default='none')
    p.add_argument('--tensor-cache')
    p.add_argument('--anchor-cache')
    a = p.parse_args()
    out = Path(a.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    pred = Predictor(a.checkpoint,a.anchor,a.device,a.precision)
    if a.operator_intervention!='none' and pred.model.transport!='coupled':
        raise ValueError('Operator intervention requires coupled transport')
    pred.model.operator_intervention = a.operator_intervention
    frame = read_metadata(a.metadata,a.dataset_root)
    dataset = PairedDataset(frame,a.partition,pred.cfg['spatial_size'],a.tensor_cache)
    if a.partition=='test':
        used = set(pred.state['training_subjects'])|set(pred.state['validation_subjects'])
        if used & set(dataset.rows.subject_id):
            raise ValueError('Test subjects overlap training/validation; use cohort-prefixed subject IDs')
    steps = a.steps if a.steps is not None else pred.cfg['loss']['steps']
    solver = a.solver or pred.cfg['loss']['solver']
    cache = AnchorCache(pred.anchor,a.anchor_cache,pred.precision) if a.anchor_cache else None
    full_volume = a.full_volume if a.full_volume is not None else pred.cfg.get('full_volume_validation',False)
    summary, rows = evaluate(pred.model,pred.anchor,DataLoader(dataset,batch_size=1),
                             pred.device,pred.precision,steps,solver,cache,full_volume=full_volume)
    summary.update(partition=a.partition,steps=steps,solver=solver,operator_intervention=a.operator_intervention,
                   residual_nfe=steps*(2 if solver=='heun' else 1),
                   anchor_sha256=pred.anchor.digest,checkpoint_sha256=pred.checkpoint_sha256)
    if list(pred.model.targets)==['ct']:
        summary.update(source_modality=pred.model.sources[0],target_modality='ct',
                       reference_ct_hu_range=[-1024,3000],
                       mask='full volume' if full_volume else 'registered patient outline or source FOV')
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'per_scan.csv',index=False)
    pd.DataFrame(rows).groupby('subject_id').mean(numeric_only=True).to_csv(out/'per_subject.csv')
    atomic_json(out/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
