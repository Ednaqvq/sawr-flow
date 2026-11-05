"""Source-only CFAM and mathematically matched, projected subband paths.

The shared trunk reuses the original ResidualUNet. Modality-specific stems and
decoders live outside it; only MRI -> PET is enabled in the supplied trainer.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from mlrd.models import ResidualUNet, norm
from supported_flow import mri_support
from wave_tools import haar, inverse


def build_model(config):
    architecture = config.get('architecture', 'ucf')
    if architecture == 'original_ct':
        from .original_ct import OriginalCTFlow
        return OriginalCTFlow()
    if architecture != 'ucf':
        raise ValueError(f'Unknown architecture: {architecture}')
    return UCFFlow(**config['model'])


def project(waves, mask):
    return haar(inverse(waves) * mask)


def scale_interior_hhh(waves, mask, gain):
    """Shrink only the added HHH correction on complete source-support blocks."""
    if gain == 1.:
        return waves
    interior = (F.avg_pool3d(mask.float(), 2) == 1).float()
    hhh = waves[:, 7:8]*(1+(gain-1)*interior)
    return torch.cat((waves[:, :7], hhh), dim=1)


def schedule(t, k):
    """g(0)=0, g(1)=1, dg/dt in [0.2,1.8] for |k|<=0.8."""
    t = t.float().view(-1, 1, 1, 1, 1)
    k = k.float().view(-1, 8, 1, 1, 1)
    return t + k * torch.sin(math.pi*t)/math.pi, 1+k*torch.cos(math.pi*t)


class CFAM(nn.Module):
    """Predict target-like signed coefficients from MRI and frozen anchor only.

    Local and pooled context operate at two resolutions of the first-level
    Haar lattice (not two independent DWT levels). Bounded gain and residual
    bias initialize to the identity MRI transform.
    """
    def __init__(self, token_dim):
        super().__init__()
        self.local = nn.Sequential(nn.Conv3d(16, 16, 3, padding=1, groups=16),
                                   nn.SiLU(), nn.Conv3d(16, 16, 1))
        self.context = nn.Conv3d(16, 16, 1)
        self.task = nn.Linear(token_dim, 16)
        for layer in (self.local[-1], self.context, self.task):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, source, anchor, token):
        joined = torch.cat([source, anchor], 1)
        pooled = F.avg_pool3d(joined, 2, ceil_mode=True)
        coarse = F.interpolate(self.context(pooled), size=source.shape[2:],
                               mode='trilinear', align_corners=False)
        gain, bias = (self.local(joined)+coarse+
                      self.task(token)[:, :, None, None, None]).chunk(2, 1)
        return source * (1+.5*torch.tanh(gain)) + .5*torch.tanh(bias)


class UCFFlow(nn.Module):
    def __init__(self, channels=(24,48,72,96), time_dim=128, token_dim=32,
                 heads=4, checkpointing=True, cfam=True, dynamic=True,
                 residual_scale=.1, sources=('mri',), targets=('pet',),
                 anatomies=('brain',), transport='legacy', coupling_mode='full',
                 coupling_locality='local', coupling_pool=4, coupled_bridge=True,
                 background_value=0., interior_hhh_gain=1., conditioning_input='matrix',
                 spatial_schedule=False, schedule_lll=False):
        super().__init__()
        if len(channels) != 4 or channels[-1] % heads:
            raise ValueError('Four trunk resolutions and divisible attention heads required')
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError('residual_scale must be positive')
        self.scale, self.use_cfam, self.dynamic = residual_scale, cfam, dynamic
        if transport not in ('legacy','coupled'):
            raise ValueError('Unknown transport')
        if transport=='coupled' and (not cfam or not dynamic):
            raise ValueError('Use coupling_mode/coupled_bridge ablations for coupled transport')
        if conditioning_input not in ('matrix', 'concat', 'residual_matrix'):
            raise ValueError('Unknown conditioning input')
        if (conditioning_input != 'matrix' or spatial_schedule) and (transport != 'coupled' or coupled_bridge):
            raise ValueError('Independent CST conditioning and scheduling require coupled transport with coupled_bridge disabled')
        if schedule_lll and not spatial_schedule:
            raise ValueError('LLL scheduling requires the spatial schedule')
        self.transport, self.coupled_bridge = transport, coupled_bridge
        self.conditioning_input = conditioning_input
        self.spatial_schedule = spatial_schedule
        self.schedule_lll = schedule_lll
        self.background_value = background_value
        if not math.isfinite(interior_hhh_gain) or not 0 <= interior_hhh_gain <= 1:
            raise ValueError('interior_hhh_gain must be in [0,1]')
        self.interior_hhh_gain = interior_hhh_gain
        self.sources, self.targets, self.anatomies = list(sources), list(targets), list(anatomies)
        self.source_token = nn.Embedding(len(sources), token_dim)
        self.target_token = nn.Embedding(len(targets), token_dim)
        self.anatomy_token = nn.Embedding(len(anatomies), token_dim)
        self.cfam = CFAM(token_dim) if cfam else None
        self.stems = nn.ModuleDict({s: nn.Conv3d(16, channels[0], 1) for s in sources})
        # The upstream clinical projection is reused as a task-token projection;
        # no subject-level clinical variables are loaded or accepted here.
        self.trunk = ResidualUNet(8, channels, time_dim, heads, 0., checkpointing, token_dim)
        del self.trunk.condition
        self.trunk.output = nn.Identity()
        self.decoders = nn.ModuleDict({s: nn.Sequential(norm(channels[0]), nn.SiLU(),
                                  nn.Conv3d(channels[0], 8, 3, padding=1)) for s in targets})
        for decoder in self.decoders.values():
            nn.init.zeros_(decoder[-1].weight)
            nn.init.zeros_(decoder[-1].bias)
        self.router = nn.Sequential(nn.Linear(channels[0]+token_dim, 64), nn.SiLU(),
                                    nn.Linear(64, 8)) if dynamic else None
        if self.router is not None:
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)
        if transport=='coupled':
            from .coupling import LocalBandOperator, SpatialBandSchedule
            self.cfam, self.router = None, None
            if conditioning_input in ('matrix', 'residual_matrix'):
                self.band_operator = LocalBandOperator(token_dim,coupling_pool,coupling_mode,coupling_locality)
            if conditioning_input == 'residual_matrix':
                self.condition_increment = nn.ModuleDict({
                    s: nn.Conv3d(8, channels[0], 1, bias=False) for s in sources
                })
                for projection in self.condition_increment.values():
                    nn.init.zeros_(projection.weight)
            if spatial_schedule:
                self.schedule_head = SpatialBandSchedule(token_dim, coupling_pool, schedule_lll)
            if conditioning_input == 'residual_matrix' or spatial_schedule:
                self.register_buffer('condition_rms', torch.ones(2, 8))
        # Optional source-only interventions for mechanism evaluation.
        self.operator_intervention = 'none'

    def conditions(self, mri, anchor, source=None, target=None, anatomy=None, support=None):
        if mri.shape != anchor.shape or not torch.isfinite(anchor).all():
            raise ValueError('Finite, aligned source and anchor volumes required')
        if any(n < 8 or n % 2 for n in mri.shape[2:]):
            raise ValueError('Spatial dimensions must be even and at least 8')
        source = source or self.sources[0]
        target = target or self.targets[0]
        anatomy = anatomy or self.anatomies[0]
        if support is not None and support.shape != mri.shape:
            raise ValueError('Source support must match input shape')
        mask = mri_support(mri) if support is None else support.float()
        anchor = anchor.float() * mask
        token = (self.source_token.weight[self.sources.index(source)]+
                 self.target_token.weight[self.targets.index(target)]+
                 self.anatomy_token.weight[self.anatomies.index(anatomy)])
        token = token[None].expand(len(mri), -1)
        sw, aw = haar(mri), haar(anchor)
        adapted = self.cfam(sw, aw, token) if self.cfam else sw
        extra = {}
        if hasattr(self, 'condition_rms'):
            normalized_source = sw / self.condition_rms[0].view(1, 8, 1, 1, 1)
            normalized_anchor = aw / self.condition_rms[1].view(1, 8, 1, 1, 1)
        if self.transport == 'coupled' and self.conditioning_input != 'concat':
            from .coupling import apply_operator
            residual_condition = self.conditioning_input == 'residual_matrix'
            operator, coupling = (self.band_operator(normalized_source, normalized_anchor, token)
                                  if residual_condition else self.band_operator(sw, aw, token))
            if self.operator_intervention=='diagonal':
                operator = operator*self.band_operator.identity
            elif self.operator_intervention=='spatial_flip':
                operator = operator.flip(-1)
            elif self.operator_intervention!='none':
                raise ValueError('Unknown operator intervention')
            if not residual_condition:
                adapted = aw+apply_operator(operator,sw-aw)
            extra = dict(operator=operator,coupling=coupling)
        levels = [self.stems[source](torch.cat([adapted, aw], 1))]
        if self.conditioning_input == 'residual_matrix':
            increment = apply_operator(extra['operator'], normalized_source)
            levels[0] = levels[0] + self.condition_increment[source](increment)
        if self.spatial_schedule:
            extra['spatial_schedule'] = self.schedule_head(normalized_source, normalized_anchor, token)
        # The route sees only the source-conditioned features. Prevent the
        # schedule objective from altering the shared feature representation.
        route_input = torch.cat([levels[0].mean((2,3,4)), token], -1).detach()
        k = .8*torch.tanh(self.router(route_input).float()) if self.router else sw.new_zeros(len(mri),8)
        if 'operator' in extra:
            k = torch.stack([extra['operator'][:,j,j].mean((1,2,3)) for j in range(8)],1)
        for down in self.trunk.cond_down:
            levels.append(down(levels[-1]))
        return dict(levels=levels, token=token, k=k, adapted=adapted, anchor=anchor,
                    mask=mask, target=target, **extra)

    def path_apply(self, value, t, cond, derivative=False, detach=False):
        if self.transport=='legacy':
            k = cond['k'].detach() if detach else cond['k']
            g, speed = schedule(t,k)
            return (speed if derivative else g)*value
        t = t.float().view(-1,1,1,1,1)
        if self.spatial_schedule:
            k = cond['spatial_schedule'].detach() if detach else cond['spatial_schedule']
            k = F.interpolate(k, size=value.shape[2:], mode='trilinear', align_corners=False)
            return (1+k*torch.cos(math.pi*t) if derivative else t+k*torch.sin(math.pi*t)/math.pi)*value
        if not self.coupled_bridge:
            return value if derivative else t*value
        from .coupling import apply_operator
        b = cond['operator'].detach() if detach else cond['operator']
        interaction = apply_operator(b,value)
        return value+torch.cos(math.pi*t)*interaction if derivative else t*value+torch.sin(math.pi*t)/math.pi*interaction

    def raw_direction(self, state, t, cond):
        features = self.trunk(state, t*1000, cond['levels'], cond['token'])
        return self.decoders[cond['target']](features).float()

    def velocity(self, state, t, cond, detach_schedule=False):
        direction = self.raw_direction(state,t,cond)
        return project(self.path_apply(direction,t,cond,derivative=True,detach=detach_schedule),cond['mask'])

    def integrate(self, cond, steps=2, solver='euler', return_endpoints=False):
        if not isinstance(steps, int) or steps < 1 or solver not in ('euler','heun'):
            raise ValueError('Positive integer steps and euler/heun solver required')
        if return_endpoints and (solver != 'euler' or steps < 2):
            raise ValueError('Trajectory endpoint supervision requires at least two Euler steps')
        state = torch.zeros_like(cond['adapted'], dtype=torch.float32)
        endpoints = []
        for i in range(steps):
            t = state.new_full((len(state),), i/steps)
            if return_endpoints:
                direction = self.raw_direction(state, t, cond)
                v = project(self.path_apply(direction, t, cond, derivative=True), cond['mask'])
                if i > 0:
                    endpoints.append(state + project(direction-self.path_apply(direction, t, cond), cond['mask']))
            else:
                v = self.velocity(state, t, cond)
            proposal = state+v/steps
            if solver == 'heun':
                v2 = self.velocity(proposal, t+1/steps, cond)
                proposal = state+(v+v2)/(2*steps)
            state = proposal
        return (state, endpoints) if return_endpoints else state

    def predict(self, mri, anchor, steps=2, solver='euler', support=None):
        cond = self.conditions(mri, anchor, support=support)
        return self.generate(cond,steps,solver)

    def generate(self, cond, steps=2, solver='euler'):
        state = self.integrate(cond, steps, solver)
        state = scale_interior_hhh(state,cond['mask'],self.interior_hhh_gain)
        image = cond['anchor']+self.scale*inverse(state)
        return image*cond['mask']+self.background_value*(1-cond['mask'])
