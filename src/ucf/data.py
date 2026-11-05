"""Strict subject splits and paired geometry; no target-derived inference mask."""
import hashlib
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from paired_t1_fdg_dataset import _resize_volume
from torch.nn.functional import interpolate

MASK_COLUMNS = {'source_mask_path':'support', 'evaluation_mask_path':'evaluation_mask'}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_metadata(path, root):
    frame = pd.read_csv(path, dtype={'subject_id':str, 'sample_id':str})
    required = ['subject_id','sample_id','t1_path','fdg_pet_path','has_fdg_pet','split','fold']
    required += [col for col in MASK_COLUMNS if col in frame]
    if not set(required) <= set(frame) or frame[required].isna().any().any():
        raise ValueError('Missing metadata columns or values')
    if frame.sample_id.duplicated().any():
        raise ValueError('Duplicate sample_id')
    if not set(frame['split']) <= {'train','final_val'}:
        raise ValueError('Use train/fold!=0, train/fold=0, final_val')
    fold = pd.to_numeric(frame['fold'], errors='raise')
    if (fold != fold.astype(int)).any() or (fold[frame['split']=='train']<0).any():
        raise ValueError('Training folds must be nonnegative integers')
    frame['partition'] = np.where(frame['split']=='final_val', 'test',
                                  np.where(fold==0, 'val', 'train'))
    if (frame.groupby('subject_id').partition.nunique()>1).any():
        raise ValueError('Subject leakage across train/val/test')
    valid = frame.has_fdg_pet.astype(str).str.lower()
    if not valid.isin(['true','false','1','0']).all():
        raise ValueError('Invalid has_fdg_pet values')
    frame = frame[valid.isin(['true','1'])].copy()
    root = Path(root).resolve()
    for col in ('t1_path','fdg_pet_path',*(col for col in MASK_COLUMNS if col in frame)):
        frame[col] = frame[col].map(lambda p: str((root/Path(p)).resolve()))
        if not all(Path(p).is_relative_to(root) for p in frame[col]):
            raise ValueError('Image outside dataset root')
    # Catch shared images assigned different IDs before fitting any model.
    for col in ('t1_path','fdg_pet_path'):
        if (frame.groupby(col).partition.nunique()>1).any():
            raise ValueError('Image path leakage across partitions')
    return frame.sort_values(['subject_id','sample_id']).reset_index(drop=True)


def load_mri(path, shape):
    native = nib.load(path)
    array = native.get_fdata(dtype=np.float32)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError(f'Expected finite 3D NIfTI: {path}')
    if array.min() < -1.001 or array.max() > 1.001:
        raise ValueError(f'Expected preprocessed [-1,1] MRI: {path}')
    volume = _resize_volume(torch.from_numpy(array)[None], tuple(shape))
    if not (volume.abs()>1e-6).any():
        raise ValueError(f'Empty MRI: {path}')
    return volume, native


def load_mask(path, native, shape):
    image = nib.load(path)
    if image.shape != native.shape or not np.allclose(image.affine,native.affine,rtol=0,atol=1e-4):
        raise ValueError(f'Mask grid mismatch: {path}')
    array = image.get_fdata(dtype=np.float32)
    if not np.isin(array,(0.,1.)).all() or not array.any():
        raise ValueError(f'Expected nonempty binary mask: {path}')
    value = torch.from_numpy(array)[None]
    if tuple(image.shape)!=tuple(shape):
        value = interpolate(value[None],size=shape,mode='nearest')[0]
    if not value.any():
        raise ValueError(f'Empty resized mask: {path}')
    return value.bool()


class PairedDataset(Dataset):
    def __init__(self, frame, partition, shape, cache_dir=None):
        self.rows = frame[frame.partition==partition].reset_index(drop=True)
        self.shape = tuple(shape)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.rows.empty:
            raise ValueError(f'Empty partition: {partition}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        mask_paths = {key:row[col] for col,key in MASK_COLUMNS.items() if col in row}
        cache = None
        if self.cache_dir:
            signature = []
            for path in (row.t1_path,row.fdg_pet_path,*mask_paths.values()):
                stat = Path(path).stat()
                signature.append([str(Path(path).resolve()),stat.st_size,stat.st_mtime_ns])
            key_parts = [signature,self.shape,'strict-pair-fp32-v1']
            if mask_paths:
                key_parts.append(list(mask_paths))
            key = hashlib.sha256(json.dumps(key_parts).encode()).hexdigest()
            cache = self.cache_dir/(key+'.pt')
            if cache.is_file():
                values = torch.load(cache,map_location='cpu',weights_only=True,mmap=True)
                return dict(**values,sample_id=row.sample_id,subject_id=row.subject_id,mri_path=row.t1_path)
        mri, src = load_mri(row.t1_path, self.shape)
        dst = nib.load(row.fdg_pet_path)
        if src.shape != dst.shape or not np.allclose(src.affine,dst.affine,rtol=0,atol=1e-4):
            raise ValueError(f'MRI/PET grid mismatch: {row.sample_id}; register first')
        array = dst.get_fdata(dtype=np.float32)
        if not np.isfinite(array).all() or array.min() < -1.001 or array.max() > 1.001:
            raise ValueError(f'Expected finite preprocessed [-1,1] PET: {row.sample_id}')
        pet = _resize_volume(torch.from_numpy(array)[None], self.shape)
        values = dict(mri=mri,target=pet)
        values.update({key:load_mask(path,src,self.shape) for key,path in mask_paths.items()})
        if cache:
            cache.parent.mkdir(parents=True,exist_ok=True)
            tmp = cache.with_name(cache.name+'.'+str(os.getpid())+'.'+uuid.uuid4().hex+'.tmp')
            torch.save(values,tmp)
            tmp.replace(cache)
        return dict(**values, sample_id=row.sample_id,
                    subject_id=row.subject_id, mri_path=row.t1_path)


def training_fingerprint(frame):
    """Content hashes of train/val only; test images are never opened here."""
    def record(row):
        result = dict(sample_id=row.sample_id,subject_id=row.subject_id,
                      partition=row.partition,mri=sha(row.t1_path),pet=sha(row.fdg_pet_path))
        result.update({col:sha(getattr(row,col)) for col in MASK_COLUMNS if hasattr(row,col)})
        return result
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(record,frame[frame.partition!='test'].itertuples()))
