import gc, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from ucf.data import PairedDataset
from mlrd.data import TrainingDataset, ClinicalTransform

class ServerCacheTests(unittest.TestCase):
    def test_lossless_shared_cache_and_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); x=np.ones((8,12,8),dtype=np.float32)*.2
            for name,value in [('mri',x),('pet',x*.4)]:nib.save(nib.Nifti1Image(value,np.eye(4)),root/(name+'.nii.gz'))
            frame=pd.DataFrame([dict(sample_id='a',subject_id='s',partition='train',has_fdg_pet=True,
                t1_path=str(root/'mri.nii.gz'),fdg_pet_path=str(root/'pet.nii.gz'))])
            uncached=PairedDataset(frame,'train',(8,12,8))[0]
            ds=PairedDataset(frame,'train',(8,12,8),root/'cache'); first=ds[0]; second=ds[0]
            for name in ('mri','target'):
                torch.testing.assert_close(first[name],uncached[name],rtol=0,atol=0)
                torch.testing.assert_close(second[name],uncached[name],rtol=0,atol=0)
            cfg=dict(dataset_root=str(root),spatial_size=[8,12,8],paired_tensor_cache=str(root/'cache'))
            legacy=TrainingDataset(frame,cfg,ClinicalTransform.fit(None))[0]
            torch.testing.assert_close(legacy['input'],uncached['mri'],rtol=0,atol=0)
            torch.testing.assert_close(legacy['target'],uncached['target'],rtol=0,atol=0)
            self.assertEqual(len(list((root/'cache').glob('*.pt'))),1)
            nib.save(nib.Nifti1Image(x*.7,np.eye(4)),root/'pet.nii.gz')
            updated=ds[0]
            self.assertFalse(torch.equal(updated['target'],first['target']))
            self.assertEqual(len(list((root/'cache').glob('*.pt'))),2)
            del first,second,legacy,updated,uncached
            gc.collect()

    def test_geometry_mismatch_rejected_before_cache_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); x=np.ones((8,8,8),dtype=np.float32)
            nib.save(nib.Nifti1Image(x,np.eye(4)),root/'mri.nii')
            affine=np.eye(4);affine[0,3]=1
            nib.save(nib.Nifti1Image(x,affine),root/'pet.nii')
            frame=pd.DataFrame([dict(sample_id='a',subject_id='s',partition='train',
                t1_path=str(root/'mri.nii'),fdg_pet_path=str(root/'pet.nii'))])
            with self.assertRaisesRegex(ValueError,'grid mismatch'):
                PairedDataset(frame,'train',(8,8,8),root/'cache')[0]
            self.assertFalse((root/'cache').exists())

if __name__=='__main__':unittest.main()
