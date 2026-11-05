"""Compare matched original and UCF residual models on full SynthRAD volumes."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from inference_ucf import Predictor
from ucf.anchor import AnchorCache
from ucf.data import PairedDataset, read_metadata, sha
from ucf.runtime import atomic_json, evaluate
from wave_tools import BANDS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('anchor', 'metadata', 'dataset-root', 'tensor-cache', 'anchor-cache', 'run-root', 'output-dir'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--revised-run', type=Path, help='Include a matched revised CST checkpoint')
    p.add_argument('--all-band-run', type=Path, help='Include revised CST with LLL scheduling enabled')
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if a.all_band_run and not a.revised_run:
        p.error('--all-band-run requires the high-frequency-only --revised-run control')
    if (a.output_dir/'summary.json').exists():
        raise FileExistsError(a.output_dir/'summary.json')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    frame = read_metadata(a.metadata, a.dataset_root)
    tables, summaries, checkpoints = {}, {}, {}
    keys = ['mae_hu', 'psnr', 'ssim', *(f'wave_{band}_mae' for band in BANDS)]
    shared_cfg = None
    arms = [('original_ct', 'original_ct', a.run_root/'original_ct'),
            ('concat_linear', 'ucf', a.run_root/'concat_linear')]
    if a.revised_run:
        arms.append(('revised_cst', 'ucf', a.revised_run))
    if a.all_band_run:
        arms.append(('all_band_cst', 'ucf', a.all_band_run))
    for name, architecture, run in arms:
        pred = Predictor(run/'last.pt', a.anchor, a.device)
        cfg, state = pred.cfg, pred.state
        if (cfg.get('architecture', 'ucf') != architecture or state['epoch'] != a.epoch
                or cfg['training']['epochs'] != a.epoch or not cfg.get('full_volume_validation')):
            raise ValueError('Unexpected architecture, budget or validation metric')
        common = {k: v for k, v in cfg.items() if k != 'architecture'}
        if architecture == 'ucf':
            revised = name in ('revised_cst', 'all_band_cst')
            if (pred.model.conditioning_input != ('residual_matrix' if revised else 'concat')
                    or pred.model.coupled_bridge or pred.model.spatial_schedule != revised
                    or pred.model.schedule_lll != (name == 'all_band_cst')
                    or pred.model.targets != ['ct'] or pred.model.interior_hhh_gain != 1):
                raise ValueError(f'Unexpected conditioning or path for {name}')
        if name in ('revised_cst', 'all_band_cst'):
            common['model'] = dict(common['model'], conditioning_input='concat')
            common['model'].pop('spatial_schedule')
            common['model'].pop('schedule_lll', None)
        if shared_cfg is not None and common != shared_cfg:
            raise ValueError('Training configurations differ outside the architecture')
        shared_cfg = common
        for split, key in [('train', 'training_subjects'), ('val', 'validation_subjects')]:
            if set(state[key]) != set(frame.loc[frame.partition == split, 'subject_id']):
                raise ValueError('Training or validation split mismatch')
        dataset = PairedDataset(frame, 'test', cfg['spatial_size'], a.tensor_cache)
        if len(dataset) != 18 or dataset.rows.subject_id.nunique() != 18:
            raise ValueError('Expected the locked 18-case held-out set')
        cache = AnchorCache(pred.anchor, a.anchor_cache, pred.precision)
        summary, rows = evaluate(pred.model, pred.anchor, DataLoader(dataset, batch_size=1),
                                 pred.device, pred.precision, cfg['loss']['steps'], cfg['loss']['solver'],
                                 cache, full_volume=True)
        table = pd.DataFrame(rows).set_index(['subject_id', 'sample_id'], verify_integrity=True).sort_index()
        table.to_csv(a.output_dir/f'{name}_per_scan.csv')
        summaries[name] = summary
        tables[name] = table
        checkpoints[name] = pred.checkpoint_sha256
        print(f'{name}: evaluated {len(dataset)} whole volumes', flush=True)
        del pred, cache

    original, concat = tables['original_ct'], tables['concat_linear']
    coarse_keys = ['anchor_' + key for key in keys]
    for table in tables.values():
        if not original.index.equals(table.index) or not np.allclose(original[coarse_keys], table[coarse_keys],
                                                                     rtol=0., atol=1e-7):
            raise ValueError('Held-out scan pairs or coarse predictions differ')
    coarse = original[coarse_keys].set_axis(keys, axis=1)
    draws = np.random.default_rng(shared_cfg['seed']).integers(18, size=(10000, 18))
    paired = {}
    contrasts = [('concat_minus_original', concat[keys]-original[keys]),
                 ('original_minus_coarse', original[keys]-coarse),
                 ('concat_minus_coarse', concat[keys]-coarse)]
    if a.revised_run:
        revised = tables['revised_cst'][keys]
        contrasts += [('revised_minus_original', revised-original[keys]),
                      ('revised_minus_concat', revised-concat[keys]),
                      ('revised_minus_coarse', revised-coarse)]
    if a.all_band_run:
        all_band = tables['all_band_cst'][keys]
        contrasts += [('all_band_minus_revised', all_band-revised),
                      ('all_band_minus_original', all_band-original[keys]),
                      ('all_band_minus_concat', all_band-concat[keys]),
                      ('all_band_minus_coarse', all_band-coarse)]
    for name, delta in contrasts:
        ci = np.quantile(delta.to_numpy()[draws].mean(axis=1), [.025, .975], axis=0)
        paired[name] = {key: dict(delta=float(delta[key].mean()), ci95=ci[:, i].tolist())
                        for i, key in enumerate(keys)}
    atomic_json(a.output_dir/'summary.json', dict(split='test', evaluation='full_volume', subjects=18,
                epoch=a.epoch, anchor_sha256=sha(a.anchor), metadata_sha256=sha(a.metadata),
                checkpoint_sha256=checkpoints, configuration=shared_cfg, by_variant=summaries,
                coarse=coarse.mean().to_dict(), paired=paired))


if __name__ == '__main__':
    main()
