"""Source-conditioned local band operator, shared by conditioning and FM path.

A[target,source] is row stochastic; B is signed and bounded. This is a learned
coefficient interaction operator, not a physical or optimal-transport plan.
"""
import torch
from torch import nn
from torch.nn import functional as F


class LocalBandOperator(nn.Module):
    def __init__(self, token_dim, pool=4, mode='full', locality='local'):
        super().__init__()
        if mode not in ('full','diagonal','identity') or locality not in ('local','global') or pool<1:
            raise ValueError('Invalid coupling configuration')
        self.pool, self.mode, self.locality = pool, mode, locality
        self.net = nn.Sequential(nn.Conv3d(16,32,3,padding=1),nn.SiLU(),nn.Conv3d(32,72,1))
        self.task = nn.Linear(token_dim,72)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.zeros_(self.task.weight)
        nn.init.zeros_(self.task.bias)
        self.register_buffer('identity',torch.eye(8).view(1,8,8,1,1,1))

    def forward(self, source, anchor, token):
        value = torch.cat([source,anchor],1)
        if self.locality=='global':
            value = value.mean((2,3,4),keepdim=True)
        else:
            size = tuple(max(1,n//self.pool) for n in value.shape[2:])
            value = F.adaptive_avg_pool3d(value,size)
        out = (self.net(value)+self.task(token)[:,:,None,None,None]).float()
        logits = out[:,:64].reshape(len(out),8,8,*out.shape[2:])+4*self.identity
        a = logits.softmax(2)
        k = .4*torch.tanh(out[:,64:])
        b = .2*(a-self.identity)+self.identity*k[:,:,None]
        if self.mode=='diagonal':
            b = b*self.identity
        elif self.mode=='identity':
            b = b*0
        return b, a


def apply_operator(matrix, value):
    """Interpolate one output row at a time, avoiding a full 64-channel field."""
    rows = []
    for j in range(8):
        weights = F.interpolate(matrix[:,j],size=value.shape[2:],mode='trilinear',align_corners=False)
        rows.append((weights*value.float()).sum(1,keepdim=True))
    return torch.cat(rows,1)


class SpatialBandSchedule(nn.Module):
    """Predict bounded local time warps with optional LLL modulation."""

    def __init__(self, token_dim, pool, schedule_lll=False):
        super().__init__()
        self.pool = pool
        self.net = nn.Sequential(nn.Conv3d(16, 16, 3, padding=1), nn.SiLU(), nn.Conv3d(16, 7, 1))
        self.task = nn.Linear(token_dim, 7)
        self.low = nn.Conv3d(16, 1, 1) if schedule_lll else None
        self.low_task = nn.Linear(token_dim, 1) if schedule_lll else None
        outputs = (self.net[-1], self.task) if self.low is None else (self.net[-1], self.task, self.low, self.low_task)
        for output in outputs:
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def forward(self, source, anchor, token):
        size = tuple(max(1, length // self.pool) for length in source.shape[2:])
        inputs = F.adaptive_avg_pool3d(torch.cat((source, anchor), dim=1), size)
        features = self.net[1](self.net[0](inputs))
        high = self.net[2](features) + self.task(token)[:, :, None, None, None]
        low = torch.zeros_like(high[:, :1])
        if self.low is not None:
            low = self.low(features) + self.low_task(token)[:, :, None, None, None]
        return .5 * torch.tanh(torch.cat((low, high), dim=1).float())
