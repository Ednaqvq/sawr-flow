"""Predict the incremental utility of applying a frozen wavelet residual."""
import torch
from torch import nn
from torch.nn import functional as F

from wave_tools import haar, inverse


GAINS = (0., .1, .25, .5, 1.)
HIGH4 = (2, 4, 5, 6)  # LHH, HLH, HHL, HHH among the seven details


class BenefitGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(24, 24, 3, padding=1), nn.SiLU(),
            nn.Conv3d(24, 24, 1), nn.SiLU(), nn.Conv3d(24, 28, 1),
        )

    def forward(self, features):
        return self.net(features).unflatten(1, (4, 7)).tanh()


def candidate(model, source, condition, steps, solver):
    residual = model.scale * model.integrate(condition, steps, solver)
    source_wave, anchor_wave = haar(source), haar(condition['anchor'])
    # Signed log compresses the wide range between LLL and the fine details.
    features = torch.cat((source_wave, anchor_wave, residual), dim=1)
    features = features.sign() * torch.log1p(100 * features.abs())
    interior = F.avg_pool3d(condition['mask'].float(), 2) == 1
    return features, residual, anchor_wave, interior


def benefit_labels(truth, detail):
    gains = detail.new_tensor(GAINS[1:]).view(1, -1, 1, 1, 1, 1)
    cost = detail.abs().unsqueeze(1) * gains
    truth = truth.unsqueeze(1)
    benefit = truth.abs() - (truth - gains * detail.unsqueeze(1)).abs()
    # Error reduction per unit correction is bounded by the triangle inequality.
    return (benefit / cost.clamp_min(1e-8)).clamp(-1, 1), benefit


def adaptive_gains(scores):
    gains = scores.new_tensor(GAINS).view(1, -1, 1, 1, 1, 1)
    value = torch.cat((torch.zeros_like(scores[:, :1]), scores), dim=1) * gains
    return scores.new_tensor(GAINS)[value.argmax(dim=1)]


def budget_gains(rank, weight, fraction):
    """Allocate an exact L1 budget separately for each scan and detail band."""
    weight_flat = weight.flatten(2)
    order = rank.flatten(2).argsort(dim=-1, descending=True, stable=True)
    # CUDA cumsum is unavailable under strict determinism in PyTorch 2.3.
    ordered = weight_flat.gather(-1, order).to(device='cpu', dtype=torch.float64)
    before = ordered.cumsum(-1) - ordered
    budget = fraction * ordered.sum(-1, keepdim=True)
    gain = ((budget - before) / ordered.clamp_min(1e-30)).clamp(0, 1)
    return torch.zeros_like(weight_flat).scatter_(-1, order, gain.to(weight_flat)).view_as(weight)


def render(model, condition, residual, interior, gains, *, anchor_interior_lll=False):
    detail = residual[:, 1:] * torch.where(interior, gains, 1.)
    lll = residual[:, :1]
    if anchor_interior_lll:
        lll = lll.masked_fill(interior, 0.)
    corrected = torch.cat((lll, detail), dim=1)
    mask = condition['mask']
    return (condition['anchor'] + inverse(corrected)) * mask + model.background_value * (1 - mask)
