#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/data/make_test_dataset_multi.py

역할
- 기존 single-degree 테스트 데이터 생성기(make_test_dataset.py)를 여러 degree에 대해 반복 실행
- configs/make_test_dataset.yaml 의 degree_list를 읽어 degree별 NPZ 생성
- 기존 생성기 로직은 그대로 재사용하고, 이 스크립트는 orchestration만 담당

필수 YAML 예시
dataset:
  degree_list: [10, 15, 20, 25]

output:
  out_dir: data/test_multi
  filename_template: taylor_test_physchem_v3_allroots_deg{degree}_10000.npz

runtime:
  single_degree_script: scripts/data/make_test_dataset.py
"""
from __future__ import annotations

import copy
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    cur = start if start.is_dir() else start.parent
    for _ in range(20):
        if (cur / "configs").is_dir() and (cur / "models").is_dir():
            return cur
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return (start.parent if start.is_file() else start).resolve()


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid YAML: {path}")
    return obj


def save_yaml(obj: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def get(cfg: Dict[str, Any], key: str, default=None):
    cur = cfg
    for p in key.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def resolve_repo_path(repo: Path, p: str) -> Path:
    pp = Path(str(p))
    return pp if pp.is_absolute() else (repo / pp).resolve()


def main():
    repo = find_repo_root(Path(__file__))
    default_cfg = repo / "configs" / "make_test_dataset.yaml"
    cfg_path = Path(os.environ.get("CFG_PATH", str(default_cfg))).resolve()
    cfg = load_yaml(cfg_path)

    degree_list = get(cfg, "dataset.degree_list", None)
    degree_single = get(cfg, "dataset.degree", None)
    if degree_list is None:
        if degree_single is None:
            raise ValueError("dataset.degree_list or dataset.degree must exist")
        degree_list = [int(degree_single)]
    degree_list = [int(x) for x in degree_list]

    out_dir = resolve_repo_path(repo, str(get(cfg, "output.out_dir", "data/test_multi")))
    filename_template = str(get(cfg, "output.filename_template", "taylor_test_physchem_v3_allroots_deg{degree}_10000.npz"))
    single_script = resolve_repo_path(repo, str(get(cfg, "runtime.single_degree_script", "scripts/data/make_test_dataset.py")))

    if not single_script.exists():
        raise FileNotFoundError(f"single_degree_script not found: {single_script}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CFG] {cfg_path}")
    print(f"[REPO] {repo}")
    print(f"[SCRIPT(single)] {single_script}")
    print(f"[OUT_DIR] {out_dir}")
    print(f"[DEGREES] {degree_list}")

    with tempfile.TemporaryDirectory(prefix="multi_test_dataset_cfgs_") as tmpdir_s:
        tmpdir = Path(tmpdir_s)

        for degree in degree_list:
            subcfg = copy.deepcopy(cfg)
            subcfg.setdefault("dataset", {})
            subcfg["dataset"]["degree"] = int(degree)
            # single-generator가 degree_list를 잘못 읽지 않게 제거
            if "degree_list" in subcfg["dataset"]:
                subcfg["dataset"].pop("degree_list", None)

            subcfg.setdefault("output", {})
            out_npz = out_dir / filename_template.format(degree=int(degree))
            subcfg["output"]["npz_path"] = str(out_npz.relative_to(repo) if out_npz.is_relative_to(repo) else out_npz)

            subcfg_path = tmpdir / f"make_test_dataset_deg{degree}.yaml"
            save_yaml(subcfg, subcfg_path)

            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo)
            env["CFG_PATH"] = str(subcfg_path)
            env["OUT_NPZ"] = str(out_npz)

            print(f"\n[RUN] degree={degree} -> {out_npz}")
            ret = subprocess.run(
                [sys.executable, str(single_script)],
                cwd=str(repo),
                env=env,
                check=False,
            )
            if ret.returncode != 0:
                raise RuntimeError(f"degree={degree} generation failed with code={ret.returncode}")
            if not out_npz.exists():
                raise FileNotFoundError(f"degree={degree} output not found: {out_npz}")

    print("[DONE] all degree test datasets generated.")


if __name__ == "__main__":
    main()
