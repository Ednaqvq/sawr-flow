import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from ucf.model import UCFFlow,project,scale_interior_hhh
from ucf.coupling import apply_operator
from ucf.losses import objective,terminal_detail_loss
from wave_tools import haar
import json

torch.set_num_threads(2)

class CoupledTests(unittest.TestCase):
    def make(self,**kwargs):
        return UCFFlow(channels=(4,8,8,8),heads=1,time_dim=16,token_dim=4,
                       checkpointing=False,transport='coupled',coupling_pool=2,**kwargs)

    def test_bound_row_stochastic_and_offdiagonal(self):
        m=self.make(); x=torch.rand(2,1,8,12,8)
        c=m.conditions(x,x*.2)
        torch.testing.assert_close(c['coupling'].sum(2),torch.ones_like(c['coupling'].sum(2)))
        self.assertLessEqual(float(c['operator'].abs().sum(2).max().detach()),.80001)
        self.assertGreater(float(c['operator'][:,0,1].abs().sum().detach()),0)
        value=torch.zeros(2,8,4,6,4);value[:,1]=1
        self.assertGreater(float(apply_operator(c['operator'],value)[:,0].abs().sum().detach()),0)

    def test_direct_concatenation_keeps_source_with_linear_path(self):
        m = self.make(conditioning_input='concat', coupled_bridge=False)
        x = torch.rand(1, 1, 8, 12, 8)
        anchor = x * .2
        cond = m.conditions(x, anchor)
        torch.testing.assert_close(cond['levels'][0], m.stems['mri'](torch.cat((haar(x), haar(anchor)), 1)))
        self.assertNotIn('operator', cond)
        r = torch.randn_like(cond['adapted'])
        torch.testing.assert_close(m.path_apply(r, torch.tensor([.4]), cond), r * .4)
        cfg = json.loads((Path(__file__).resolve().parents[1]/'configs/cst_mri2pet.json').read_text())['loss']
        cfg['weights'].update(cfam=0., spectrum=0., schedule=0.)
        loss, _ = objective(m, x, x * .3, anchor, torch.ones(8), cfg)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(m.decoders['pet'][-1].weight.grad)

    def test_independent_matrix_and_schedule_start_at_concat_linear(self):
        x = torch.rand(2, 1, 8, 12, 8)
        baseline = self.make(conditioning_input='concat', coupled_bridge=False)
        base_state = baseline.state_dict()
        for mode, spatial in (('residual_matrix', False), ('concat', True), ('residual_matrix', True)):
            model = self.make(conditioning_input=mode, coupled_bridge=False, spatial_schedule=spatial)
            state = model.state_dict()
            state.update(base_state)
            model.load_state_dict(state, strict=True)
            cond = model.conditions(x, x * .2)
            torch.testing.assert_close(cond['levels'][0], baseline.conditions(x, x * .2)['levels'][0])
            residual = torch.randn_like(cond['adapted'])
            t = torch.tensor([.3, .7])
            torch.testing.assert_close(model.path_apply(residual, t, cond), residual * t[:, None, None, None, None])

    def test_spatial_schedule_endpoints_derivative_and_lll(self):
        model = self.make(conditioning_input='concat', coupled_bridge=False, spatial_schedule=True)
        with torch.no_grad():
            model.schedule_head.net[-1].weight.normal_(std=.2)
            model.schedule_head.net[-1].bias.fill_(.3)
        x = torch.rand(2, 1, 8, 12, 8)
        cond = model.conditions(x, x * .2)
        self.assertEqual(int(torch.count_nonzero(cond['spatial_schedule'][:, 0])), 0)
        self.assertGreater(float(cond['spatial_schedule'][:, 1:].std().detach()), 0.)
        residual = torch.randn_like(cond['adapted'])
        t = torch.tensor([.25, .6])
        start = model.path_apply(residual, torch.zeros(2), cond)
        stop = model.path_apply(residual, torch.ones(2), cond)
        torch.testing.assert_close(start, torch.zeros_like(residual))
        torch.testing.assert_close(stop, residual, atol=1e-6, rtol=1e-6)
        speed = model.path_apply(residual, t, cond, derivative=True)
        torch.testing.assert_close(speed[:, 0], residual[:, 0])
        delta = (model.path_apply(residual, t + .001, cond)-model.path_apply(residual, t - .001, cond))/.002
        torch.testing.assert_close(speed, delta, atol=.001, rtol=.002)
        self.assertFalse(model.path_apply(residual, t, cond, detach=True).requires_grad)
        speed.square().mean().backward()
        self.assertGreater(float(model.schedule_head.net[-1].weight.grad.abs().sum()), 0.)

    def test_optional_lll_schedule_preserves_high_frequency_head(self):
        kwargs = dict(conditioning_input='residual_matrix', coupled_bridge=False, spatial_schedule=True)
        torch.manual_seed(42)
        high = self.make(**kwargs)
        torch.manual_seed(42)
        all_bands = self.make(**kwargs, schedule_lll=True)
        for name, value in high.state_dict().items():
            torch.testing.assert_close(value, all_bands.state_dict()[name], atol=0, rtol=0)
        with torch.no_grad():
            all_bands.schedule_head.low.bias.fill_(.5)
        x = torch.rand(1, 1, 8, 12, 8)
        cond = all_bands.conditions(x, x*.2)
        residual = torch.randn_like(cond['adapted'])
        t = torch.tensor([.25])
        z = all_bands.path_apply(residual, t, cond)
        self.assertFalse(torch.allclose(z[:, :1], .25*residual[:, :1]))
        torch.testing.assert_close(z[:, 1:], .25*residual[:, 1:])
        torch.testing.assert_close(all_bands.path_apply(residual, torch.ones(1), cond), residual)
        z[:, :1].square().mean().backward()
        self.assertGreater(float(all_bands.schedule_head.low.bias.grad.abs().sum()), 0.)

    def test_exact_path_derivative_endpoint_and_support(self):
        m=self.make(); x=torch.rand(2,1,8,12,8);x[:,:,:2]=0
        c=m.conditions(x,x*.2);r=project(torch.randn_like(c['adapted']),c['mask'])
        t=torch.tensor([.25,.6]);eps=.001
        f=lambda tt:project(m.path_apply(r,tt,c),c['mask'])
        torch.testing.assert_close(f(torch.zeros(2)),torch.zeros_like(r))
        torch.testing.assert_close(f(torch.ones(2)),r,atol=3e-6,rtol=3e-6)
        torch.testing.assert_close((f(t+eps)-f(t-eps))/(2*eps),
            project(m.path_apply(r,t,c,derivative=True),c['mask']),atol=.001,rtol=.002)
        endpoint=f(t)+project(r-m.path_apply(r,t,c),c['mask'])
        torch.testing.assert_close(endpoint,r,atol=3e-6,rtol=3e-6)

    def test_gradients_and_capacity_matched_controls(self):
        m=self.make();x=torch.rand(2,1,8,12,8)
        cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/cst_mri2pet.json').read_text())['loss']
        cfg['weights']['cfam'] = .1
        loss,_=objective(m,x,x*.3,x*.2,torch.ones(8),cfg)
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.band_operator.parameters() if p.grad is not None),0)
        count=sum(p.numel() for p in m.parameters())
        for mode in ('diagonal','identity'):
            other=self.make(coupling_mode=mode)
            self.assertEqual(count,sum(p.numel() for p in other.parameters()))
        identity=self.make(coupling_mode='identity');c=identity.conditions(x,x*.2)
        self.assertEqual(int(torch.count_nonzero(c['operator'])),0)

    def test_operator_interventions(self):
        m=self.make();x=torch.rand(1,1,8,12,8)
        m.operator_intervention='diagonal';c=m.conditions(x,x*.2)
        off=c['operator']*(1-m.band_operator.identity)
        self.assertEqual(int(torch.count_nonzero(off)),0)

    def test_terminal_detail_distinguishes_wrong_sign_in_interior(self):
        target=torch.zeros(1,8,1,1,2)
        target[:,7]=1
        mask=torch.ones(1,1,2,2,4)
        mask[...,3]=0
        pred=target.clone()
        pred[:,7,:,:,1]=-3
        self.assertEqual(float(terminal_detail_loss(pred,target,mask,torch.ones(8))),0.)
        pred[:,7,:,:,0]=-1
        pred.requires_grad_()
        terminal_detail_loss(pred,target,mask,torch.ones(8)).backward()
        self.assertLess(float(pred.grad[0,7,0,0,0]),0.)
        self.assertEqual(float(pred.grad[0,7,0,0,1]),0.)

    def test_calibration_changes_only_interior_hhh(self):
        waves=torch.ones(1,8,1,1,2)
        mask=torch.ones(1,1,2,2,4)
        mask[...,3]=0
        result=scale_interior_hhh(waves,mask,.2)
        torch.testing.assert_close(result[:,:7],waves[:,:7])
        torch.testing.assert_close(result[0,7,0,0],torch.tensor([.2,1.]))

if __name__=='__main__':unittest.main()
