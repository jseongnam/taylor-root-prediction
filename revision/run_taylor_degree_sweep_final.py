#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--degree', type=int, default=None)
    ap.add_argument('--degrees', type=int, nargs='*', default=None)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--base-dir', type=str, default=str(Path(__file__).resolve().parent))
    ap.add_argument('--interval-script', type=str, default=str(Path(__file__).resolve().parent / 'model_degree_final.py'))
    ap.add_argument('--ann-script', type=str, default=str(Path(__file__).resolve().parent / 'ann_degree_final.py'))
    ap.add_argument('--mlp-script', type=str, default=str(Path(__file__).resolve().parent / 'mlp_degree_final.py'))
    ap.add_argument('--lstm-script', type=str, default=str(Path(__file__).resolve().parent / 'lstm_degree_final.py'))
    ap.add_argument('--skip-interval', action='store_true')
    ap.add_argument('--skip-ann', action='store_true')
    ap.add_argument('--skip-mlp', action='store_true')
    ap.add_argument('--skip-lstm', action='store_true')
    ap.add_argument('--interval-dir-template', type=str, default='{base}/taylor_data_physchem_v4_interval_deg{degree}')
    ap.add_argument('--root-dir-template', type=str, default='{base}/taylor_data_physchem_v4_deg{degree}')
    return ap.parse_args()


def candidate_paths(data_dir: Path, degree: int, split: str) -> List[Path]:
    return [
        data_dir / f'taylor_deg{degree}_{split}.npz',
        data_dir / f'{split}.npz',
        data_dir / f'{split}_deg{degree}.npz',
        data_dir / f'deg{degree}_{split}.npz',
        data_dir / f'taylor_{split}_deg{degree}.npz',
        data_dir / f'{degree}_{split}.npz',
    ]


def resolve_npz(data_dir: Path, degree: int, split: str) -> str:
    for cand in candidate_paths(data_dir, degree, split):
        if cand.exists():
            return str(cand)
    if data_dir.exists():
        matches = sorted(data_dir.glob(f'*{split}*.npz'))
        matches_deg = [m for m in matches if f'deg{degree}' in m.name or str(degree) in m.name]
        if len(matches_deg) == 1:
            return str(matches_deg[0])
        if len(matches) == 1:
            return str(matches[0])
    raise FileNotFoundError(f'Could not resolve {split}.npz in {data_dir} for degree={degree}')


def run_one(script: str, env: dict):
    cmd = [sys.executable, script]
    print('[RUN]', ' '.join(cmd))
    print('      TAYLOR_ORDER=', env.get('TAYLOR_ORDER'))
    subprocess.run(cmd, check=True, env=env)


def main():
    args = parse_args()
    if args.degrees:
        degrees = args.degrees
    elif args.degree is not None:
        degrees = [args.degree]
    else:
        degrees = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    base = Path(args.base_dir).resolve()
    for degree in degrees:
        env_common = os.environ.copy()
        env_common['TAYLOR_ORDER'] = str(degree)
        env_common['DEVICE'] = args.device

        interval_dir = Path(args.interval_dir_template.format(base=str(base), degree=degree)).resolve()
        root_dir = Path(args.root_dir_template.format(base=str(base), degree=degree)).resolve()

        if not args.skip_interval:
            env_i = env_common.copy()
            env_i['TRAIN_NPZ'] = resolve_npz(interval_dir, degree, 'train')
            env_i['VAL_NPZ'] = resolve_npz(interval_dir, degree, 'val')
            env_i['TEST_NPZ'] = resolve_npz(interval_dir, degree, 'test')
            env_i['OUT_DIR'] = str((base.parent / 'results' / 'transformer_interval' / f'deg{degree}').resolve())
            run_one(args.interval_script, env_i)

        for skip, script, name in [
            (args.skip_mlp, args.mlp_script, 'mlp'),
            (args.skip_ann, args.ann_script, 'ann'),
            
            (args.skip_lstm, args.lstm_script, 'lstm'),
        ]:
            if skip:
                continue
            env_r = env_common.copy()
            env_r['TRAIN_NPZ'] = resolve_npz(root_dir, degree, 'train')
            env_r['VAL_NPZ'] = resolve_npz(root_dir, degree, 'val')
            env_r['TEST_NPZ'] = resolve_npz(root_dir, degree, 'test')
            env_r['OUT_DIR'] = str((base.parent / 'results' / 'taylor_nn' / name / f'deg{degree}').resolve())
            run_one(script, env_r)

    print('[ALL DONE]')


if __name__ == '__main__':
    main()
