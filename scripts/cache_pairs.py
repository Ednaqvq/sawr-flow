"""Validate and cache training/validation pairs only, without opening test images."""
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from ucf.data import PairedDataset, read_metadata, training_fingerprint
from ucf.runtime import atomic_json

def main():
    p=argparse.ArgumentParser()
    for name in ('metadata','dataset-root','cache-dir','report'):p.add_argument('--'+name,required=True)
    p.add_argument('--shape',type=int,nargs=3,default=[160,196,160])
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args(); torch.set_num_threads(1); started=time.monotonic()
    frame=read_metadata(a.metadata,a.dataset_root); report=dict(test_images_read=False,partitions={})
    for split in ('train','val'):
        ds=PairedDataset(frame,split,a.shape,a.cache_dir)
        def item(i):
            v=ds[i]
            return dict(sample_id=v['sample_id'],mri_min=float(v['mri'].min()),mri_max=float(v['mri'].max()),
                        pet_min=float(v['target'].min()),pet_max=float(v['target'].max()))
        result=[]
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for row in pool.map(item,range(len(ds))):
                result.append(row)
                if len(result)%25==0: print(json.dumps(dict(partition=split,cached=len(result),total=len(ds))),flush=True)
        report['partitions'][split]=result
    report['images']=training_fingerprint(frame)
    report['seconds']=time.monotonic()-started
    atomic_json(a.report,report)
    print(json.dumps(dict(event='cache_complete',seconds=report['seconds'])),flush=True)

if __name__=='__main__':main()
