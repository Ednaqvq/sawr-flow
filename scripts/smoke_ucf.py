"""Exercise the all-band MRI-to-CT workflow on synthetic volumes."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from mlrd.fusion_flow import FusedMRIGenerator


def run(*args):
    subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT/'runs/ucf_synthetic_smoke'))
    a = parser.parse_args()
    root = Path(a.output).resolve()
    if root.exists():
        raise FileExistsError('Choose a fresh smoke output directory')
    root.mkdir(parents=True)
    torch.manual_seed(7)
    torch.set_num_threads(2)
    rng = np.random.default_rng(7)
    shape = (16,20,16)
    grid = np.indices(shape).astype(np.float32)
    mask = sum(((grid[j]-(shape[j]-1)/2)/(shape[j]/2-1))**2 for j in range(3))<1
    records = []
    for i in range(4):
        mri = ((rng.random(shape)*1.2-.6)*mask).astype(np.float32)
        ct = np.where(mask, .25*mri+.12, -1.).astype(np.float32)
        for kind,data in [('mri',mri),('ct',ct),('fov',mask.astype(np.uint8))]:
            nib.save(nib.Nifti1Image(data,np.diag([1.2,1.5,1.2,1.])),root/f'{kind}_{i}.nii.gz')
        records.append(dict(subject_id=f'synthetic_{i}',sample_id=f'scan_{i}',
            t1_path=f'mri_{i}.nii.gz',fdg_pet_path=f'ct_{i}.nii.gz',has_fdg_pet=True,
            source_mask_path=f'fov_{i}.nii.gz',
            split='final_val' if i==3 else 'train',fold=0 if i==2 else (1 if i<2 else -1)))
    metadata = root/'metadata.csv'
    pd.DataFrame(records).to_csv(metadata,index=False)
    anchor_cfg = json.loads((ROOT/'configs/default.json').read_text())
    anchor_cfg['cohort'] = 'SYNTHETIC'
    anchor_cfg['data']['spatial_size'] = list(shape)
    anchor_cfg['model'].update(ae_base=4,latent_channels=4,unet_channels=[4,8,8,8],
                               time_dim=16,attention_heads=1,checkpointing=False,dropout=0.)
    anchor_cfg['flow'].update(sample_steps=1,sample_solver='euler')
    gen = FusedMRIGenerator(anchor_cfg['model'],anchor_cfg['fusion_flow'])
    anchor = root/'synthetic_anchor.pt'
    torch.save(dict(format_version='mri-fusion-flow-v1',config=anchor_cfg,generator=gen.state_dict(),
                    normalization=dict(fit_split='train',mean=[0.]*4,std=[1.]*4)),anchor)
    cfg = json.loads((ROOT/'configs/cst_mri2ct.json').read_text())
    cfg['spatial_size'] = list(shape)
    cfg['model'].update(channels=[4,8,8,8],time_dim=16,token_dim=4,heads=1,checkpointing=False,coupling_pool=2)
    cfg['training'].update(epochs=2,precision='fp32',cpu_threads=2,accumulation_steps=3)
    config = root/'config.json'
    config.write_text(json.dumps(cfg,indent=2))
    command = [ROOT/'train_ucf.py','--metadata',metadata,'--dataset-root',root,
               '--anchor',anchor,'--run-dir',root/'run','--config',config,'--device','cpu']
    run(*command,'--stop-after-epoch','1')
    # A real interrupted run must match an uninterrupted run tensor for tensor.
    run(*command,'--resume')
    uninterrupted = command.copy()
    uninterrupted[uninterrupted.index('--run-dir')+1] = root/'uninterrupted'
    run(*uninterrupted)
    resumed = torch.load(root/'run/last.pt',map_location='cpu',weights_only=True)
    full = torch.load(root/'uninterrupted/last.pt',map_location='cpu',weights_only=True)
    for key,value in resumed['model'].items():
        torch.testing.assert_close(value,full['model'][key],rtol=0,atol=0)
    for a,b in zip(resumed['history'],full['history']):
        assert a['train']==b['train'] and a['val_subject_mae']==b['val_subject_mae']
    checkpoint = root/'run/last.pt'
    run(ROOT/'evaluate_ucf.py','--metadata',metadata,'--dataset-root',root,'--anchor',anchor,
        '--checkpoint',checkpoint,'--device','cpu','--output-dir',root/'evaluation')
    run(ROOT/'inference_ucf.py','--mri',root/'mri_3.nii.gz','--source-mask',root/'fov_3.nii.gz','--anchor',anchor,
        '--checkpoint',checkpoint,'--device','cpu','--output',root/'generated_ct.nii.gz')
    output = nib.load(root/'generated_ct.nii.gz')
    assert output.shape==shape and np.isfinite(output.get_fdata()).all()
    assert np.all(output.get_fdata()[~mask]==-1024.)
    np.testing.assert_allclose(output.affine,nib.load(root/'mri_3.nii.gz').affine)
    held_out = root/'ct_3.nii.gz'
    renamed = root/'held_out_ct_hidden.nii.gz'
    held_out.rename(renamed)
    try:
        run(ROOT/'inference_ucf.py','--mri',root/'mri_3.nii.gz','--source-mask',root/'fov_3.nii.gz','--anchor',anchor,
            '--checkpoint',checkpoint,'--device','cpu','--output',root/'mri_only_repeat.nii.gz')
        np.testing.assert_array_equal(output.get_fdata(),nib.load(root/'mri_only_repeat.nii.gz').get_fdata())
    finally:
        renamed.rename(held_out)
    report = dict(status='passed',data='synthetic only',output=str(root),
                  interrupted_resume_bitwise_equal=True,source_only_inference=True,
                  nifti_geometry=True,background_air_hu=True,training_epochs=2)
    (root/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':
    main()
