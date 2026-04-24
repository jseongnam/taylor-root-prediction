#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_dataset_physchem_v4_cli.py

기존 YAML-only 생성기
  scripts/data/generate_dataset_physchem_v4.py

를 감싸는 CLI 래퍼.

예시:
  python generate_dataset_physchem_v4_cli.py \
    --script scripts/data/generate_dataset_physchem_v4.py \
    --config configs/dataset_physchem_v4_deg25.yaml \
    --degree 30
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import yaml
except Exception as e:
    raise ImportError("PyYAML이 필요합니다. `pip install pyyaml`") from e


ALLOWED_DEGREES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _replace_degree_token(text: str, degree: int) -> str:
    if not text:
        return text
    text = re.sub(r"deg\d+", f"deg{degree}", text)
    text = re.sub(r"degree\d+", f"degree{degree}", text)
    return text


def infer_out_dir(cfg: dict, degree: int, user_out_dir: str | None) -> str:
    if user_out_dir:
        return user_out_dir

    dataset = cfg.setdefault("dataset", {})
    out_dir = dataset.get("out_dir", "./taylor_data_physchem_v4_deg25")
    out_dir = _replace_degree_token(str(out_dir), degree)

    if out_dir == dataset.get("out_dir"):
        out_dir = str(Path(out_dir).parent / f"{Path(out_dir).name}_deg{degree}")
    return out_dir


def patch_config(cfg: dict, degree: int, n_total: int | None, seed: int | None, out_dir: str):
    cfg = dict(cfg)
    dataset = dict(cfg.get("dataset", {}))
    dataset["degree"] = int(degree)
    dataset["out_dir"] = str(out_dir)

    if n_total is not None:
        dataset["n_total"] = int(n_total)
    if seed is not None:
        dataset["seed"] = int(seed)

    cfg["dataset"] = dataset
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description="Root regression dataset generator CLI wrapper")
    p.add_argument("--script", required=True,
                   help="원본 생성기 경로 (예: scripts/data/generate_dataset_physchem_v4.py)")
    p.add_argument("--config", required=True,
                   help="원본 YAML config 경로")
    p.add_argument("--degree", type=int, required=True, choices=ALLOWED_DEGREES,
                   help="생성할 Taylor/Maclaurin 차수")
    p.add_argument("--out_dir", type=str, default=None,
                   help="출력 폴더 override")
    p.add_argument("--n_total", type=int, default=None,
                   help="총 샘플 수 override")
    p.add_argument("--seed", type=int, default=None,
                   help="seed override")
    p.add_argument("--python", type=str, default=sys.executable,
                   help="사용할 python 실행 파일")
    return p.parse_args()


def main():
    args = parse_args()

    script_path = Path(args.script).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()

    if not script_path.exists():
        raise FileNotFoundError(f"script not found: {script_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    cfg = load_yaml(str(config_path))
    out_dir = infer_out_dir(cfg, args.degree, args.out_dir)
    patched = patch_config(cfg, args.degree, args.n_total, args.seed, out_dir)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tf:
        temp_cfg_path = tf.name
    save_yaml(temp_cfg_path, patched)

    env = os.environ.copy()
    env["DATASET_CFG"] = temp_cfg_path
    env["OUT_DIR"] = out_dir

    cmd = [args.python, str(script_path)]

    print("[INFO] Running root regression dataset generator")
    print(f"[INFO] script   : {script_path}")
    print(f"[INFO] config   : {config_path}")
    print(f"[INFO] temp_cfg : {temp_cfg_path}")
    print(f"[INFO] degree   : {args.degree}")
    print(f"[INFO] out_dir  : {out_dir}")
    if args.n_total is not None:
        print(f"[INFO] n_total  : {args.n_total}")
    if args.seed is not None:
        print(f"[INFO] seed     : {args.seed}")
    print()

    try:
        subprocess.run(cmd, env=env, check=True)
    finally:
        try:
            os.remove(temp_cfg_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
