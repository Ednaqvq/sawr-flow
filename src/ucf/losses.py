"""Paired supervision; targets never enter the deployed condition path."""
import torch
from torch.nn import functional as F
from mlrd.losses import gradient_l1, masked_l1, ssim3d
from multih_huber_loss import selective_huber_fm_loss
from wave_tools import haar, inverse
from .model import project


DETAIL_BANDS = (3, 5, 6, 7)


def terminal_detail_loss(predicted, required, mask, train_rms):
    """Equal-band signed residual L1 on complete Haar blocks inside support."""
    interior = (F.avg_pool3d(mask.float(), 2) == 1).float()
    error = (predicted[:, DETAIL_BANDS].float()-required[:, DETAIL_BANDS].float()).abs()
    scale = train_rms[list(DETAIL_BANDS)].float().clamp_min(1e-3).view(1, -1, 1, 1, 1)
    per_band = (error/scale*interior).sum((2, 3, 4))/interior.sum((2, 3, 4)).clamp_min(1)
    valid = interior.flatten(1).any(dim=1)
    return (per_band.mean(dim=1)*valid).sum()/valid.sum().clamp_min(1)


def terminal_high_loss(predicted, required, mask, train_rms):
    """Supervise signed endpoints in all seven high-frequency bands equally."""
    interior = (F.avg_pool3d(mask.float(), 2) == 1).float()
    error = (predicted[:, 1:].float()-required[:, 1:].float()).abs()
    scale = train_rms[1:].float().clamp_min(1e-3).view(1, 7, 1, 1, 1)
    count = interior.sum((2, 3, 4)).clamp_min(1)
    per_sample = ((error/scale*interior).sum((2, 3, 4))/count).mean(1)
    valid = interior.flatten(1).any(1)
    return (per_sample*valid).sum()/valid.sum().clamp_min(1)


def ct_structure_losses(predicted, target, mask):
    """Target-only supervision on bone neighbourhoods and signed CT edges."""
    bone_threshold = (300.+1024.)/2012.-1.
    bone = (target > bone_threshold).float()*mask
    neighbourhood = F.max_pool3d(bone, 3, stride=1, padding=1)*mask
    bone_error = masked_l1(predicted, target, neighbourhood)
    edge_errors = []
    for dim in (2, 3, 4):
        before, after = [slice(None)]*5, [slice(None)]*5
        before[dim], after[dim] = slice(None, -1), slice(1, None)
        valid = mask[tuple(before)]*mask[tuple(after)]
        reference = torch.diff(target, dim=dim)
        edges = (reference.abs() > 200./2012.).float()*valid
        error = (torch.diff(predicted, dim=dim)-reference).abs()
        numerator = (error*edges).flatten(1).sum(1)
        count = edges.flatten(1).sum(1)
        edge_errors.append(((numerator/count.clamp_min(1))*(count>0)).sum()/(count>0).sum().clamp_min(1))
    return bone_error, torch.stack(edge_errors).mean()


def spectrum(waves, mask):
    weight = F.avg_pool3d(mask.float(), 2)
    return (waves.float().square()*weight).sum((2,3,4))/weight.sum((2,3,4)).clamp_min(1)


def objective(model, mri, target, anchor, thresholds, cfg, t=None, support=None):
    cond = model.conditions(mri, anchor) if support is None else model.conditions(mri, anchor, support=support)
    mask, scale = cond['mask'], model.scale
    target = target.float()*mask
    residual = haar((target-cond['anchor'])/scale).detach()
    if t is None:
        t = torch.rand(len(mri), device=mri.device)
        t = torch.where(torch.rand_like(t)<cfg['anchor_probability'], 0., t)
    # The matching problem uses a fixed current path. The router itself is
    # learned by differentiating the actual deployed rollout plus regularizer.
    state = project(model.path_apply(residual,t,cond,detach=True), mask).detach()
    desired = project(model.path_apply(residual,t,cond,derivative=True,detach=True), mask).detach()
    direction = model.raw_direction(state, t, cond)
    velocity = project(model.path_apply(direction,t,cond,derivative=True,detach=True), mask)
    image_error = inverse(velocity-desired)
    selected = cfg['huber_bands']
    if selected:
        fm = selective_huber_fm_loss(image_error, mask, thresholds[selected],
                                     selected=selected, protect_boundary=False)
    else:
        fm = ((image_error.square()*mask).flatten(1).sum(1)/mask.flatten(1).sum(1)).mean()
    # For a non-linear per-band path, (1-t)*v is NOT an endpoint estimator.
    endpoint = cond['anchor']+scale*inverse(state+project(direction-model.path_apply(direction,t,cond,detach=True), mask))
    trajectory = cfg['weights'].get('trajectory', 0) > 0
    integrated = model.integrate(cond, cfg['steps'], cfg['solver'], return_endpoints=True) if trajectory else model.integrate(cond, cfg['steps'], cfg['solver'])
    final_state, endpoints = integrated if trajectory else (integrated, [])
    pred = (cond['anchor']+scale*inverse(final_state))*mask
    terms = dict(fm=fm, endpoint=masked_l1(endpoint,target,mask)/scale,
                 rollout=masked_l1(pred,target,mask)/scale,
                 ssim=1-ssim3d(pred,target,2.,mask=mask).mean(),
                 schedule=cond['k'].square().mean())
    if 'spatial_schedule' in cond:
        terms['schedule'] = cond['spatial_schedule'].square().mean()
    elif 'operator' in cond:
        terms['schedule'] = cond['operator'].square().mean()
    if model.use_cfam and (cfg['weights']['cfam'] or cfg['weights']['spectrum']):
        # Signed coefficient supervision prevents energy-only sign ambiguity.
        wp, wt = project(cond['adapted'], mask), haar(target)
        terms['cfam'] = (wp-wt).abs().mean()
        terms['spectrum'] = (torch.log1p(spectrum(wp,mask))-
                             torch.log1p(spectrum(wt,mask))).abs().mean()
    else:
        terms['cfam'] = terms['spectrum'] = fm.new_zeros(())
    if cfg['weights'].get('final_detail', 0) > 0:
        # The integrated state is in residual/scale units, like the training RMS.
        # Each velocity is already projected onto the source support.
        terms['final_detail'] = terminal_detail_loss(final_state,residual,mask,thresholds)
    elif 'final_detail' in cfg['weights']:
        terms['final_detail'] = fm.new_zeros(())
    if cfg['weights'].get('final_high', 0) > 0:
        terms['final_high'] = terminal_high_loss(final_state, residual, mask, thresholds)
    elif 'final_high' in cfg['weights']:
        terms['final_high'] = fm.new_zeros(())
    if cfg['weights'].get('gradient', 0) > 0:
        terms['gradient'] = gradient_l1(pred, target, mask)/scale
    elif 'gradient' in cfg['weights']:
        terms['gradient'] = fm.new_zeros(())
    if cfg['weights'].get('bone', 0) or cfg['weights'].get('edge', 0):
        bone, edge = ct_structure_losses(pred, target, mask)
        terms.update(bone=bone/scale, edge=edge/scale)
        # Each structural term can be ablated independently.
        terms = {key: value for key, value in terms.items() if key in cfg['weights']}
    else:
        for key in ('bone', 'edge'):
            if key in cfg['weights']:
                terms[key] = fm.new_zeros(())
    if trajectory:
        # Endpoints start from the model's own rollout states, including the
        # predicted state at t=0.5 in a two-step deployment.
        terms['trajectory'] = torch.stack([
            masked_l1(inverse(value-residual), torch.zeros_like(target), mask) for value in endpoints
        ]).mean()
    elif 'trajectory' in cfg['weights']:
        terms['trajectory'] = fm.new_zeros(())
    loss = sum(cfg['weights'][key]*value for key, value in terms.items())
    return loss, {key:float(value.detach()) for key,value in terms.items()}
