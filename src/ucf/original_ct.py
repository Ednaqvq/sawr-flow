"""Adapt the published two-step residual Flow to a CT source-FOV mask."""
import torch
from supported_flow import SupportedFlow


class OriginalCTFlow(SupportedFlow):
    scale = .1
    use_cfam = False
    background_value = -1.
    sources = ('mri',)
    targets = ('ct',)

    def __init__(self):
        super().__init__('mri')

    def conditions(self, mri, anchor, *, support=None):
        if support is None or support.shape != mri.shape:
            raise ValueError('Original CT adaptation requires the registered source FOV')
        bands = self.encode(anchor)
        return dict(levels=super().conditions(mri, anchor), anchor=anchor.float()*support,
                    mask=support.float(), k=bands.new_zeros(len(mri), 8), shape=bands.shape,
                    clinical=mri.new_empty(len(mri), 0), target='ct')

    def path_apply(self, value, t, cond, derivative=False, detach=False):
        return value if derivative else t.float().view(-1, 1, 1, 1, 1)*value

    def raw_direction(self, state, t, cond):
        return self(state, t, cond['levels'], cond['clinical'])

    def integrate(self, cond, steps, solver):
        if solver != 'euler':
            raise ValueError('Published residual Flow requires two Euler steps')
        return self.integrate_supported(cond['shape'], cond['levels'], cond['clinical'], cond['mask'], steps)

    def generate(self, cond, steps, solver):
        residual = self.decode(self.integrate(cond, steps, solver))
        return (cond['anchor'] + self.scale*residual)*cond['mask'] + self.background_value*(1-cond['mask'])
