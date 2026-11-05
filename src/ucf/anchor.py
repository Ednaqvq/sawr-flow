"""Portable, frozen adapter for the original two-branch latent Flow anchor."""
import hashlib
import json
import os
import uuid
from pathlib import Path
import torch
from torch import nn
from mlrd.fusion_flow import FusedMRIGenerator
from mlrd.flow import integrate
from .data import sha


class FrozenAnchor(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.digest = sha(path)
        # Upstream training artifacts contain optimizer/RNG Python objects.
        # Load only your own or trusted upstream checkpoint files.
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state.get('format_version') != 'mri-fusion-flow-v1':
            raise ValueError('Expected original anchor.pt or frozen/fold_0/flow/best.pt')
        self.cfg = state['config']
        if self.cfg['model'].get('clinical_dim',8) != 0:
            raise ValueError('MRI-only anchor requires clinical_dim=0')
        if self.cfg['flow']['source_sigma'] != 0:
            raise ValueError('This adapter requires the released deterministic source_sigma=0 anchor')
        self.gen = FusedMRIGenerator(self.cfg['model'], self.cfg['fusion_flow'])
        self.gen.load_state_dict(state['generator'], strict=True)
        stats = state['normalization']
        if stats['fit_split'] != 'train':
            raise ValueError('Anchor normalization must be fitted on training data')
        mu, std = torch.tensor(stats['mean']), torch.tensor(stats['std'])
        if not torch.isfinite(mu).all() or not torch.isfinite(std).all() or (std<0).any():
            raise ValueError('Invalid anchor statistics')
        self.register_buffer('mu',mu.view(1,-1,1,1,1))
        self.register_buffer('std',std.clamp_min(self.cfg['fusion_flow']['std_floor']).view(1,-1,1,1,1))
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, mri):
        _, _, fused = self.gen.source(mri)
        state = (fused.float()-self.mu)/self.std
        cond = self.gen.velocity.conditions(fused)
        clinical = mri.new_empty(len(mri),0)
        flow = self.cfg['flow']
        state, _ = integrate(self.gen.velocity,state,cond,clinical,flow['sample_steps'],flow['sample_solver'])
        image = self.gen.pet.decode(state*self.std+self.mu).float()
        if not torch.isfinite(image).all():
            raise FloatingPointError('Non-finite anchor output')
        return image


class AnchorCache:
    def __init__(self, anchor, directory, precision):
        self.anchor, self.directory, self.precision = anchor, Path(directory), precision
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hashes = {}

    @torch.no_grad()
    def get(self, mri, paths):
        result = []
        for image, path in zip(mri, paths):
            stat = Path(path).stat()
            identity = (str(path),stat.st_size,stat.st_mtime_ns)
            if identity not in self.hashes:
                self.hashes[identity] = sha(path)
            key = dict(mri_sha256=self.hashes[identity], anchor_sha256=self.anchor.digest,
                       shape=list(image.shape), precision=self.precision,
                       torch=str(torch.__version__), device=str(mri.device), version=1)
            digest = hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
            out = self.directory/(digest+'.pt')
            if out.exists():
                item = torch.load(out,map_location='cpu',weights_only=True)
                if item['key'] != key:
                    raise ValueError('Anchor cache mismatch')
                value = item['anchor'].to(mri.device)
            else:
                value = self.anchor(image[None])[0]
                temp = out.with_name(out.name+'.'+str(os.getpid())+'.'+uuid.uuid4().hex+'.tmp')
                torch.save(dict(key=key,anchor=value.cpu()),temp)
                temp.replace(out)
            if value.shape != image.shape or not torch.isfinite(value).all():
                raise ValueError('Invalid cached anchor')
            result.append(value)
        return torch.stack(result)
