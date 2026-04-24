#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run ANN / MLP / LSTM Taylor root models only, degree by degree.
- Interval model training is skipped.
- Intended for cases where interval checkpoints are already trained/saved.
- Root-model datasets are auto-resolved from the user's revision folder layout.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def find_repo_root(start: str | Path) -> Path:
    p = Path(start).resolve()
    if p.is_file():
        p = p.parent
    for cur in [p, *p.parents]:
        if (cur / "configs").exists() and (cur / "revision").exists():
            return cur
    return Path(start).resolve().parent if Path(start).is_file() else Path(start).resolve()


def resolve_npz(data_dir: Path, degree: int, split: str) -> str:
    split = split.lower()
    cand = [
        data_dir / f"taylor_deg{degree}_{split}.npz",
        data_dir / f"{split}.npz",
        data_dir / f"{split}_deg{degree}.npz",
        data_dir / f"deg{degree}_{split}.npz",
        data_dir / f"taylor_{split}_deg{degree}.npz",
        data_dir / f"{degree}_{split}.npz",
        data_dir / split / f"taylor_deg{degree}_{split}.npz",
        data_dir / split / f"{split}.npz",
    ]
    for p in cand:
        if p.exists():
            return str(p)
    tried = ", ".join(str(x.name) if x.parent == data_dir else str(x.relative_to(data_dir)) for x in cand)
    raise FileNotFoundError(
        f"Could not resolve {split}.npz in {data_dir} for degree={degree}. Tried: {tried}"
    )


def run_one(script: str, env: dict) -> None:
    cmd = [sys.executable, script]
    print(f"[RUN] {' '.join(cmd)}")
    print(f"      TAYLOR_ORDER= {env.get('TAYLOR_ORDER')}")
    print(f"      TRAIN_NPZ   = {env.get('TRAIN_NPZ')}")
    print(f"      VAL_NPZ     = {env.get('VAL_NPZ')}")
    print(f"      TEST_NPZ    = {env.get('TEST_NPZ')}")
    subprocess.run(cmd, check=True, env=env)


def parse_degrees(args: argparse.Namespace) -> List[int]:
    if args.degree is not None:
        return [int(args.degree)]
    if args.degrees:
        return [int(x) for x in args.degrees]
    return [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", type=int, default=None, help="Run a single degree")
    ap.add_argument("--degrees", nargs="*", default=None, help="Run multiple degrees")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--repo-root", type=str, default=None)
    ap.add_argument("--revision-dir", type=str, default=None)

    ap.add_argument("--ann-script", type=str, default=None)
    ap.add_argument("--mlp-script", type=str, default=None)
    ap.add_argument("--lstm-script", type=str, default=None)

    ap.add_argument("--root-dir-template", type=str, default="taylor_data_physchem_v4_deg{degree}")

    ap.add_argument("--no-ann", action="store_true")
    ap.add_argument("--no-mlp", action="store_true")
    ap.add_argument("--no-lstm", action="store_true")

    ap.add_argument("--train-npz", type=str, default=None)
    ap.add_argument("--val-npz", type=str, default=None)
    ap.add_argument("--test-npz", type=str, default=None)

    args = ap.parse_args()

    repo = Path(args.repo_root).resolve() if args.repo_root else find_repo_root(__file__)
    revision = Path(args.revision_dir).resolve() if args.revision_dir else (repo / "revision")

    ann_script = args.ann_script or str(revision / "ann_degree_final.py")
    mlp_script = args.mlp_script or str(revision / "mlp_degree_final.py")
    lstm_script = args.lstm_script or str(revision / "lstm_degree_final.py")

    degrees = parse_degrees(args)

    print(f"[INFO] repo     = {repo}")
    print(f"[INFO] revision = {revision}")
    print(f"[INFO] degrees  = {degrees}")

    for degree in degrees:
        print("\n" + "=" * 80)
        print(f"[DEGREE] {degree}")
        print("=" * 80)

        root_dir = revision / args.root_dir_template.format(degree=degree)
        if not root_dir.exists() and not (args.train_npz and args.val_npz and args.test_npz):
            raise FileNotFoundError(f"Root data directory not found: {root_dir}")

        train_npz = args.train_npz or resolve_npz(root_dir, degree, "train")
        val_npz = args.val_npz or resolve_npz(root_dir, degree, "val")
        test_npz = args.test_npz or resolve_npz(root_dir, degree, "test")

        common_env = os.environ.copy()
        common_env["TAYLOR_ORDER"] = str(degree)
        common_env["DEVICE"] = args.device
        common_env["TRAIN_NPZ"] = train_npz
        common_env["VAL_NPZ"] = val_npz
        common_env["TEST_NPZ"] = test_npz
        if not args.no_mlp:
            run_one(mlp_script, common_env.copy())
        if not args.no_ann:
            run_one(ann_script, common_env.copy())
        
        if not args.no_lstm:
            run_one(lstm_script, common_env.copy())

    print("\n[DONE] root-model sweep finished.")


if __name__ == "__main__":
    main()
