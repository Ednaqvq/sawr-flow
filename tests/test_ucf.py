"""Analytic path, leakage, gradient, and real checkpoint adapter contracts."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from ucf.model import UCFFlow, project, schedule
from ucf.losses import objective
from ucf.data import read_metadata, PairedDataset
from wave_tools import haar, inverse

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


class UCFTests(unittest.TestCase):
    def model(self,**kwargs):
        return UCFFlow(channels=(4,8,8,8),heads=1,time_dim=16,token_dim=4,
                       checkpointing=False,**kwargs)

    def sample(self):
        torch.manual_seed(11)
        mri = torch.rand(2,1,8,12,8)*2-1
        mri[:,:,:2] = 0
        return mri, .3*mri+.1*(mri!=0).float(), .2*mri

    def test_schedule_endpoints_derivative_and_monotonicity(self):
        k = torch.linspace(-.8,.8,8)[None].expand(2,-1)
        zero, _ = schedule(torch.zeros(2),k)
        one, _ = schedule(torch.ones(2),k)
        torch.testing.assert_close(zero,torch.zeros_like(zero))
        torch.testing.assert_close(one,torch.ones_like(one))
        for t in (0.,.2,.7,1.):
            times = torch.full((2,),t)
            g, derivative = schedule(times,k)
            plus, _ = schedule(times+1e-3,k)
            minus, _ = schedule(times-1e-3,k)
            torch.testing.assert_close((plus-minus)/.002,derivative,rtol=2e-4,atol=1e-4)
            self.assertTrue((derivative>=.19999).all())

    def test_projected_path_and_exact_endpoint(self):
        x, y, a = self.sample()
        mask = (x!=0).float()
        r = haar((y-a)/.1)
        k = torch.linspace(-.7,.7,8)[None].expand(2,-1)
        t = torch.tensor([.2,.7])
        g, speed = schedule(t,k)
        rt = project(g*r,mask)
        endpoint = rt+project((1-g)*r,mask)
        torch.testing.assert_close(inverse(endpoint), (y-a)/.1,rtol=3e-6,atol=3e-6)
        gp,_ = schedule(t+1e-3,k)
        gm,_ = schedule(t-1e-3,k)
        finite_difference = (project(gp*r,mask)-project(gm*r,mask))/.002
        torch.testing.assert_close(finite_difference,project(speed*r,mask),rtol=1e-3,atol=5e-4)

    def test_zero_decoder_equals_anchor_and_mri_only_support(self):
        model = self.model()
        x,y,a = self.sample()
        with torch.no_grad():
            for solver in ('euler','heun'):
                out = model.predict(x,a,2,solver)
                torch.testing.assert_close(out,a)
                self.assertEqual(int(torch.count_nonzero(out[x==0])),0)
        # Target is absent from both conditions() and predict() signatures.
        import inspect
        self.assertNotIn('pet',inspect.signature(model.predict).parameters)
        self.assertNotIn('target_image',inspect.signature(model.conditions).parameters)

    def test_all_components_receive_gradients_after_head_opens(self):
        model = self.model()
        x,y,a = self.sample()
        cfg = json.loads((ROOT/'configs/ucf_mri2pet.json').read_text())['loss']
        opt = torch.optim.Adam(model.parameters(),lr=.001)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            loss,_ = objective(model,x,y,a,torch.ones(8),cfg,t=torch.tensor([.25,.65]))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            opt.step()
        for name, module in [('cfam',model.cfam),('router',model.router),
                             ('stem',model.stems),('trunk',model.trunk),('decoder',model.decoders)]:
            total = sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
            self.assertGreater(total,0,name)

    def test_ablation_contracts_and_invalid_steps(self):
        x,_,a = self.sample()
        for cfam in (False,True):
            for dynamic in (False,True):
                model = self.model(cfam=cfam,dynamic=dynamic)
                cond = model.conditions(x,a)
                self.assertEqual(cond['k'].shape,(2,8))
                if not cfam:
                    torch.testing.assert_close(cond['adapted'],haar(x))
                with self.assertRaises(ValueError):
                    model.integrate(cond,0)

    def test_subject_and_path_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'metadata.csv'
            rows = [dict(subject_id='same',sample_id='a',t1_path='a.nii.gz',fdg_pet_path='pa.nii.gz',
                         has_fdg_pet=True,split='train',fold=1),
                    dict(subject_id='same',sample_id='b',t1_path='b.nii.gz',fdg_pet_path='pb.nii.gz',
                         has_fdg_pet=True,split='train',fold=0)]
            pd.DataFrame(rows).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'Subject leakage'):
                read_metadata(path,root)
            rows[1].update(subject_id='other',t1_path='a.nii.gz')
            pd.DataFrame(rows).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'Image path leakage'):
                read_metadata(path,root)

    def test_paired_geometry_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            image = np.ones((8,8,8),dtype=np.float32)*.5
            nib.save(nib.Nifti1Image(image,np.eye(4)),root/'mri.nii.gz')
            affine = np.eye(4)
            affine[0,3] = 5
            nib.save(nib.Nifti1Image(image,affine),root/'pet.nii.gz')
            frame = pd.DataFrame([dict(subject_id='a',sample_id='a',t1_path=str(root/'mri.nii.gz'),
                    fdg_pet_path=str(root/'pet.nii.gz'),partition='train')])
            with self.assertRaisesRegex(ValueError,'grid mismatch'):
                PairedDataset(frame,'train',(8,8,8))[0]

    def test_activation_checkpointing_matches_gradients(self):
        torch.manual_seed(2)
        direct = self.model()
        checked = copy.deepcopy(direct)
        checked.trunk.checkpointing = True
        # Open the head so gradients exercise every checkpointed block.
        torch.nn.init.normal_(direct.decoders['pet'][-1].weight,std=.01)
        checked.load_state_dict(direct.state_dict())
        x,_,a = self.sample()
        for model in (direct,checked):
            model.predict(x,a,2).square().mean().backward()
        for (name,p), (_,q) in zip(direct.named_parameters(),checked.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-7,msg=name)

    def test_anchor_adapter_matches_upstream_generation(self):
        from mlrd.fusion_flow import FusionFlowTrainer, FusedMRIGenerator
        from ucf.anchor import FrozenAnchor
        with tempfile.TemporaryDirectory() as folder:
            cfg = json.loads((ROOT/'configs/default.json').read_text())
            cfg['output_dir'] = folder
            cfg['model'].update(ae_base=4,latent_channels=4,unet_channels=[4,8,8,8],
                time_dim=16,attention_heads=1,checkpointing=False,dropout=0.)
            cfg['runtime'].update(activation_checkpointing=False,cpu_threads=2)
            cfg['training']['amp'] = False
            cfg['data']['spatial_size'] = [8,12,8]
            cfg['flow'].update(sample_steps=2,sample_solver='heun')
            stats = dict(fit_split='train',fold=0,metadata_hash='synthetic',
                         mean=[.1,-.1,.2,-.2],std=[.8,1.,1.2,.7])
            original = FusionFlowTrainer(copy.deepcopy(cfg),0,'cpu',inference_only=True)
            original.set_normalization(stats)
            original.gen.eval()
            path = Path(folder)/'anchor.pt'
            torch.save(dict(format_version='mri-fusion-flow-v1',config=cfg,
                             generator=original.gen.state_dict(),normalization=stats),path)
            adapted = FrozenAnchor(path)
            x,_,_ = self.sample()
            with torch.no_grad():
                expected = original.generate(x,torch.empty(2,0),['a','b'])[0]
                actual = adapted(x)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
