"""Source-only NIfTI inference for CST-Flow."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
import nibabel as nib
import numpy as np
import torch
from ucf.anchor import FrozenAnchor
from ucf.data import load_mask, load_mri, sha
from ucf.model import build_model
from ucf.runtime import setup, autocast, atomic_json
from ucf.selective import BenefitGate, adaptive_gains, candidate, render


class Predictor:
    def __init__(self,checkpoint,anchor,device='cuda:0',precision=None,gate=None,*,anchor_interior_lll=False):
        if anchor_interior_lll and gate is None:
            raise ValueError('Anchoring interior LLL requires a selective gate checkpoint')
        self.state = torch.load(checkpoint,map_location='cpu',weights_only=True)
        if self.state.get('format_version')!='ucf-mri2pet-v1':
            raise ValueError('Expected UCF checkpoint')
        self.cfg = self.state['config']
        self.precision = precision or self.cfg['training']['precision']
        self.device = setup(self.cfg['seed'],device,self.precision,self.cfg['training']['cpu_threads'])
        self.anchor = FrozenAnchor(anchor).to(self.device)
        if self.anchor.digest!=self.state['anchor_sha256']:
            raise ValueError('Anchor checkpoint differs from the one used for training')
        self.model = build_model(self.cfg).to(self.device).eval()
        self.model.load_state_dict(self.state['model'],strict=True)
        self.checkpoint_sha256 = sha(checkpoint)
        self.gate = None
        self.gate_sha256 = None
        self.anchor_interior_lll = anchor_interior_lll
        if gate is not None:
            state = torch.load(gate,map_location='cpu',weights_only=True)
            if (state.get('format_version') != 'selective-wavelet-v1'
                    or state['candidate_sha256'] != self.checkpoint_sha256):
                raise ValueError('Selective gate requires its original candidate checkpoint')
            self.gate = BenefitGate().to(self.device).eval()
            self.gate.load_state_dict(state['model'],strict=True)
            self.gate_sha256 = sha(gate)

    @torch.no_grad()
    def tensor(self,mri,steps=None,solver=None,support=None):
        if list(mri.shape[2:])!=self.cfg['spatial_size']:
            raise ValueError('Incorrect input grid shape')
        if not torch.isfinite(mri).all() or mri.min() < -1.001 or mri.max() > 1.001:
            raise ValueError('Expected preprocessed [-1,1] MRI')
        mri = mri.to(self.device)
        if list(self.model.targets)==['ct'] and support is None:
            raise ValueError('CT inference requires the registered source FOV mask')
        if support is not None:
            support = support.to(self.device)
        with autocast(self.device,self.precision):
            coarse = self.anchor(mri)
            steps = steps if steps is not None else self.cfg['loss']['steps']
            solver = solver or self.cfg['loss']['solver']
            cond = self.model.conditions(mri,coarse,support=support)
            if self.gate is None:
                out = self.model.generate(cond,steps,solver)
            else:
                if (steps, solver) != (self.cfg['loss']['steps'], self.cfg['loss']['solver']):
                    raise ValueError('Selective gate requires its original candidate integration settings')
                features, residual, _, interior = candidate(self.model,mri,cond,steps,solver)
        if self.gate is not None:
            out = render(self.model,cond,residual,interior,adaptive_gains(self.gate(features)),
                         anchor_interior_lll=self.anchor_interior_lll)
        if not torch.isfinite(out).all():
            raise FloatingPointError('Non-finite generated PET')
        return out

    def nifti(self,mri_path,output,steps=None,solver=None,source_mask=None):
        output = Path(output)
        sidecar = Path(str(output)+'.json')
        if output.exists() or sidecar.exists():
            raise FileExistsError(output)
        mri, native = load_mri(mri_path,self.cfg['spatial_size'])
        support = load_mask(source_mask,native,self.cfg['spatial_size'])[None] if source_mask else None
        out = self.tensor(mri[None],steps,solver,support=support)[0,0].cpu().numpy()
        is_ct = list(self.model.targets)==['ct']
        if is_ct:
            out = (out+1)*2012.-1024.
        scale = np.asarray(native.shape)/np.asarray(self.cfg['spatial_size'])
        transform = np.eye(4)
        transform[:3,:3] = np.diag(scale)
        transform[:3,3] = (scale-1)/2
        affine = native.affine@transform
        output.parent.mkdir(parents=True,exist_ok=True)
        nib.save(nib.Nifti1Image(out,affine),output)
        effective_steps = steps if steps is not None else self.cfg['loss']['steps']
        effective_solver = solver or self.cfg['loss']['solver']
        method = f'CST-Flow {self.model.sources[0].upper()}2{self.model.targets[0].upper()}'
        info = dict(method=method,shape=list(out.shape),affine=affine.tolist(),
                    steps=effective_steps,solver=effective_solver,
                    residual_nfe=effective_steps*(2 if effective_solver=='heun' else 1),
                    anchor_sha256=self.anchor.digest,checkpoint_sha256=self.checkpoint_sha256,
                    precision=self.precision,intensity='HU' if is_ct else 'normalized training PET scale; not absolute SUVR',
                    support='source FOV; air outside' if is_ct else 'MRI only; background zero',
                    output_grid='resized source grid')
        if self.gate_sha256 is not None:
            info['selective_gate_sha256'] = self.gate_sha256
            info['anchor_interior_lll'] = self.anchor_interior_lll
        atomic_json(sidecar,info)
        return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','anchor','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--mri','--source',dest='mri',required=True)
    p.add_argument('--source-mask',help='Registered source FOV NIfTI for CT generation')
    p.add_argument('--gate',help='Selective residual gate trained on this exact checkpoint')
    p.add_argument('--anchor-interior-lll',action='store_true',
                   help='Preserve coarse LLL coefficients in fully supported Haar blocks')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--precision',choices=['fp32','fp16','bf16'])
    p.add_argument('--steps',type=int)
    p.add_argument('--solver',choices=['euler','heun'])
    a = p.parse_args()
    print(json.dumps(Predictor(a.checkpoint,a.anchor,a.device,a.precision,a.gate,
                               anchor_interior_lll=a.anchor_interior_lll).nifti(
        a.mri,a.output,a.steps,a.solver,source_mask=a.source_mask),indent=2))


if __name__=='__main__':
    main()
