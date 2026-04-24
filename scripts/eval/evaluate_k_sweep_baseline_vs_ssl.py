#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline vs Semi-Supervised only evaluation wrapper.

Assumptions
-----------
1) You already have the original evaluation module at:
   scripts/eval/evaluate_k_sweep.py
2) Your semi-supervised root regressor checkpoint can be loaded with the same
   loader used for the old anchored backend (i.e. MLP-like checkpoint, possibly
   anchor-logits or direct scalar output).
3) AST interval predictor and baseline solver are unchanged.

What changed
------------
- Removed ann/lstm/mlp style multi-backend comparison.
- Keeps only two methods:
  * ssl      : semi-supervised backend
  * baseline : numerical baseline
- Prints head-to-head comparison for each K.
- Saves per-K comparison CSV.

Run example
-----------
PYTHONPATH=. EVAL_CFG=configs/eval_k_sweep_ssl.yaml \
OUTDIR=results/runs_baseline_vs_ssl \
python scripts/eval/evaluate_k_sweep_baseline_vs_ssl.py
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.eval import evaluate_k_sweep as ek


def _print_head_to_head(K: int, compare_thr: float, ssl_pack: dict, baseline_pack: dict):
    ssl_abs = np.asarray(ssl_pack["abs"], dtype=np.float64)
    base_abs = np.asarray(baseline_pack["abs"], dtype=np.float64)
    ssl_t = np.asarray(ssl_pack["time_ms"], dtype=np.float64)
    base_t = np.asarray(baseline_pack["time_ms"], dtype=np.float64)

    ssl_ok = np.isfinite(ssl_abs) & (ssl_abs <= compare_thr)
    base_ok = np.isfinite(base_abs) & (base_abs <= compare_thr)

    both_ok = ssl_ok & base_ok
    ssl_only = ssl_ok & (~base_ok)
    base_only = (~ssl_ok) & base_ok
    both_fail = (~ssl_ok) & (~base_ok)

    ssl_better = both_ok & (ssl_abs < base_abs)
    base_better = both_ok & (base_abs < ssl_abs)
    tie = both_ok & np.isclose(ssl_abs, base_abs, rtol=0.0, atol=1e-30)

    print(f"\n===== HEAD-TO-HEAD @ K={K} =====")
    print(f"threshold          : {compare_thr:.1e}")
    print(f"ssl ok rate        : {ssl_ok.mean()*100:8.3f}%")
    print(f"baseline ok rate   : {base_ok.mean()*100:8.3f}%")
    print(f"ssl only success   : {ssl_only.mean()*100:8.3f}%  ({int(ssl_only.sum())})")
    print(f"baseline only succ.: {base_only.mean()*100:8.3f}%  ({int(base_only.sum())})")
    print(f"both fail          : {both_fail.mean()*100:8.3f}%  ({int(both_fail.sum())})")
    print(f"both ok            : {both_ok.mean()*100:8.3f}%  ({int(both_ok.sum())})")
    print(f"ssl better |f(x)|  : {ssl_better.mean()*100:8.3f}%  ({int(ssl_better.sum())})")
    print(f"base better |f(x)| : {base_better.mean()*100:8.3f}%  ({int(base_better.sum())})")
    print(f"tie                : {tie.mean()*100:8.3f}%  ({int(tie.sum())})")

    ssl_abs_f = ssl_abs[np.isfinite(ssl_abs)]
    base_abs_f = base_abs[np.isfinite(base_abs)]
    ssl_t_f = ssl_t[np.isfinite(ssl_t)]
    base_t_f = base_t[np.isfinite(base_t)]

    if ssl_abs_f.size:
        print(f"ssl |f(x)| mean/p50/p90 : {ssl_abs_f.mean():.3e} / {np.percentile(ssl_abs_f,50):.3e} / {np.percentile(ssl_abs_f,90):.3e}")
    if base_abs_f.size:
        print(f"base|f(x)| mean/p50/p90 : {base_abs_f.mean():.3e} / {np.percentile(base_abs_f,50):.3e} / {np.percentile(base_abs_f,90):.3e}")
    if ssl_t_f.size:
        print(f"ssl time ms mean/p50    : {ssl_t_f.mean():.3f} / {np.percentile(ssl_t_f,50):.3f}")
    if base_t_f.size:
        print(f"base time ms mean/p50   : {base_t_f.mean():.3f} / {np.percentile(base_t_f,50):.3f}")
    print("===============================\n")


def _save_casewise_csv(path: Path, K: int, compare_thr: float, expr, func_id, ssl_pack: dict, baseline_pack: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    ssl_abs = np.asarray(ssl_pack["abs"], dtype=np.float64)
    base_abs = np.asarray(baseline_pack["abs"], dtype=np.float64)
    ssl_root = np.asarray(ssl_pack["root"], dtype=np.float64)
    base_root = np.asarray(baseline_pack["root"], dtype=np.float64)
    ssl_t = np.asarray(ssl_pack["time_ms"], dtype=np.float64)
    base_t = np.asarray(baseline_pack["time_ms"], dtype=np.float64)

    ssl_ok = np.isfinite(ssl_abs) & (ssl_abs <= compare_thr)
    base_ok = np.isfinite(base_abs) & (base_abs <= compare_thr)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "K", "index", "func_id", "expr",
            "ssl_root", "ssl_abs", "ssl_ok", "ssl_time_ms",
            "baseline_root", "baseline_abs", "baseline_ok", "baseline_time_ms",
            "winner"
        ])
        for i in range(len(expr)):
            if ssl_ok[i] and base_ok[i]:
                if ssl_abs[i] < base_abs[i]:
                    winner = "ssl"
                elif base_abs[i] < ssl_abs[i]:
                    winner = "baseline"
                else:
                    winner = "tie"
            elif ssl_ok[i]:
                winner = "ssl"
            elif base_ok[i]:
                winner = "baseline"
            else:
                winner = "none"
            w.writerow([
                K,
                i,
                "" if func_id is None else int(func_id[i]),
                str(expr[i]),
                ssl_root[i], ssl_abs[i], bool(ssl_ok[i]), ssl_t[i],
                base_root[i], base_abs[i], bool(base_ok[i]), base_t[i],
                winner,
            ])


def main():
    cfg_path = os.environ.get("EVAL_CFG", "configs/eval_k_sweep_ssl.yaml")
    outdir = Path(os.environ.get("OUTDIR", "results/runs_baseline_vs_ssl")).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = ek.load_yaml(Path(cfg_path))
    repo = ek.find_repo_root(Path(__file__))

    device_str = ek.resolve_device(str(ek._get(cfg, "eval.device", "auto")))
    device = torch.device(device_str)
    batch_size = int(ek._get(cfg, "eval.batch_size", 512))
    topk_list = ek.parse_csv_list(str(ek._get(cfg, "eval.topk_list", "5,10,15,20,25")), cast=int)
    compare_thr = float(ek._get(cfg, "eval.compare_thr", 1e-10))
    thr_sweep = str(ek._get(cfg, "eval.thr_sweep", "1e-6,1e-8,1e-10,1e-12"))

    test_path = ek.resolve_repo_path(str(ek._get(cfg, "data.test_npz", "")), repo)
    ast_ckpt = ek.resolve_repo_path(str(ek._get(cfg, "models.ast_ckpt", "")), repo)
    ssl_ckpt = ek.resolve_repo_path(str(ek._get(cfg, "models.ssl_ckpt", "")), repo)

    if not test_path.exists():
        raise FileNotFoundError(f"test_npz not found: {test_path}")
    if not ast_ckpt.exists():
        raise FileNotFoundError(f"ast_ckpt not found: {ast_ckpt}")
    if not ssl_ckpt.exists():
        raise FileNotFoundError(f"ssl_ckpt not found: {ssl_ckpt}")

    args = ek.EvalArgs(
        compare_thr=compare_thr,
        thr_sweep=thr_sweep,
        topk_list=topk_list,
        batch_size=batch_size,
        device=device_str,
        solver_mode=str(ek._get(cfg, "solver.solver_mode", "newton_bisect")),
        max_tries=int(ek._get(cfg, "solver.max_tries", 10)),
        stop_after_first_success=bool(ek._get(cfg, "solver.stop_after_first_success", False)),
        proxy=str(ek._get(cfg, "solver.proxy", "poly")),
        tol_f=float(ek._get(cfg, "solver.tol_f", 1e-10)),
        newton_iters=int(ek._get(cfg, "solver.newton_iters", 30)),
        newton_max_step=float(ek._get(cfg, "solver.newton_max_step", 2.0)),
        bisect_iters=int(ek._get(cfg, "solver.bisect_iters", 60)),
        local_radius=float(ek._get(cfg, "solver.local_radius", 1.0)),
        local_scan_n=int(ek._get(cfg, "solver.local_scan_n", 101)),
        local_max_brackets=int(ek._get(cfg, "solver.local_max_brackets", 10)),
        stable_radius=float(ek._get(cfg, "solver.stable_radius", 1.0)),
        stable_scan_n=int(ek._get(cfg, "solver.stable_scan_n", 80)),
        stable_valid_min=float(ek._get(cfg, "solver.stable_valid_min", 0.7)),
        stable_dfx_min=float(ek._get(cfg, "solver.stable_dfx_min", 1e-10)),
        baseline_mode=str(ek._get(cfg, "baseline.mode", "all")),
        baseline_topk=int(ek._get(cfg, "baseline.topk", 10)),
        base_scan_xmin=float(ek._get(cfg, "baseline.scan_xmin", -20.0)),
        base_scan_xmax=float(ek._get(cfg, "baseline.scan_xmax", 20.0)),
        base_scan_n=int(ek._get(cfg, "baseline.scan_n", 250)),
        anchored_fb="off",
        baseline_cache=str(ek._get(cfg, "baseline.cache", "")),
        baseline_cache_save=bool(ek._get(cfg, "baseline.cache_save", False)),
        thr_winner_okonly=False,
        report_funcid_winner=False,
        report_funcid_winner_mode="both",
        report_funcid_winner_expr_k=3,
        report_funcid_winner_topn=50,
        report_fail_funcid=False,
        report_fail_mode="baseline",
        report_fail_funcid_topn=30,
        report_fail_expr_k=3,
        report_fail_save=False,
        plot_residual_hist=False,
        plot_funcid_box=False,
        plot_topn_funcid=15,
    )

    data = np.load(test_path, allow_pickle=True)
    coeffs = data["coeffs"].astype(np.float32)
    expr = data["func_expr"] if "func_expr" in data else data["expr"]
    func_id = data["func_id"] if "func_id" in data else None

    N, D = coeffs.shape
    print(f"[NPZ] path={test_path}")
    print(f"[DATA] N={N} coeff_dim={D}")

    ast_model, ast_cfg, ast_scale, ast_sanitize = ek.load_ast_topk_model(ast_ckpt, device=device)
    K_all = int(ast_cfg.get("num_candidates", ast_cfg.get("K", 10)))
    max_len = int(ast_cfg.get("max_len", 128))
    K_need = min(max(topk_list), K_all)
    print(f"[AST] ckpt={ast_ckpt} K_all={K_all} K_need={K_need} max_len={max_len} scale={ast_scale} sanitize={ast_sanitize}")

    ds_expr = ek.ExprASTOnlyDataset(expr, max_len=max_len, sanitize=ast_sanitize)
    dl = DataLoader(ds_expr, batch_size=batch_size, shuffle=False, num_workers=0)

    centers_all = np.zeros((N, K_need), dtype=np.float64)
    idx0 = 0
    for ids, numvals, attn in tqdm(dl, desc="AST forward", ncols=110):
        B = ids.size(0)
        ids = ids.to(device)
        numvals = numvals.to(device)
        attn = attn.to(device)
        with torch.no_grad():
            y = ast_model(ids, numvals, attn).double()[:, :K_need].contiguous()
            c = (float(ast_scale) * torch.sinh(y)).cpu().numpy().astype(np.float64)
        centers_all[idx0:idx0 + B, :] = c
        idx0 += B

    ssl_model, ssl_anchors, ssl_scaler = ek.load_backend_anchored(ssl_ckpt, device=device)
    backends = [
        ("ssl", {"type": "anchored", "model": ssl_model, "anchors": ssl_anchors, "scaler": ssl_scaler})
    ]
    print(f"[SSL] {ssl_ckpt}")

    def _infer_first_linear_in_dim(model: nn.Module):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                return int(m.in_features)
        return None

    in_dim = _infer_first_linear_in_dim(ssl_model)
    if in_dim is not None and in_dim != coeffs.shape[1]:
        raise ValueError(f"[ssl] input dim mismatch: model expects {in_dim}, but test coeff dim is {coeffs.shape[1]}")

    rr = ek.ReasonRegistry()
    base_cache_obj = None
    baseline_cache_path = Path(args.baseline_cache) if str(args.baseline_cache).strip() else None
    if baseline_cache_path is not None:
        base_cache_obj = ek.load_baseline_cache(baseline_cache_path, coeffs.shape[0])
        if base_cache_obj is not None:
            print(f"[BASELINE CACHE] loaded: {baseline_cache_path} | done={int(base_cache_obj['done'].sum())}/{coeffs.shape[0]}")

    summary_rows = []

    print("\n==================== BASELINE vs SSL K SWEEP START ====================")
    for K in topk_list:
        t0 = time.perf_counter()
        out_methods, baseline_pack = ek.run_eval_for_K(
            K=int(K),
            centers_all=centers_all,
            coeffs=coeffs,
            expr=expr,
            func_id=func_id,
            data=data,
            backends=backends,
            device=device,
            rr=rr,
            args=args,
            base_cache_obj=base_cache_obj,
        )
        elapsed = time.perf_counter() - t0

        if base_cache_obj is None:
            base_cache_obj = {
                "root": baseline_pack["root"].copy(),
                "abs": baseline_pack["abs"].copy(),
                "time_ms": baseline_pack["time_ms"].copy(),
                "method": baseline_pack["method"].copy(),
                "done": baseline_pack["done"].copy(),
                "reason_id": baseline_pack["reason_id"].copy(),
            }
            if baseline_cache_path is not None and args.baseline_cache_save:
                ek.save_baseline_cache(
                    baseline_cache_path,
                    base_cache_obj["root"],
                    base_cache_obj["abs"],
                    base_cache_obj["time_ms"],
                    base_cache_obj["method"],
                    base_cache_obj["done"],
                    base_cache_obj["reason_id"],
                )

        ssl_pack = out_methods["ssl"]
        _print_head_to_head(K, compare_thr, ssl_pack, baseline_pack)

        csv_path = outdir / f"compare_cases_K{int(K)}.csv"
        _save_casewise_csv(csv_path, int(K), compare_thr, expr, func_id, ssl_pack, baseline_pack)

        ssl_abs = np.asarray(ssl_pack["abs"], dtype=np.float64)
        base_abs = np.asarray(baseline_pack["abs"], dtype=np.float64)
        ssl_ok = np.isfinite(ssl_abs) & (ssl_abs <= compare_thr)
        base_ok = np.isfinite(base_abs) & (base_abs <= compare_thr)
        both_ok = ssl_ok & base_ok
        ssl_better = both_ok & (ssl_abs < base_abs)
        base_better = both_ok & (base_abs < ssl_abs)

        summary_rows.append({
            "K": int(K),
            "ssl_ok_rate": float(ssl_ok.mean()),
            "baseline_ok_rate": float(base_ok.mean()),
            "ssl_only": int((ssl_ok & (~base_ok)).sum()),
            "baseline_only": int(((~ssl_ok) & base_ok).sum()),
            "both_ok": int(both_ok.sum()),
            "both_fail": int(((~ssl_ok) & (~base_ok)).sum()),
            "ssl_better_among_both_ok": int(ssl_better.sum()),
            "baseline_better_among_both_ok": int(base_better.sum()),
            "elapsed_sec": float(elapsed),
            "case_csv": str(csv_path),
        })

    summary_csv = outdir / "summary_baseline_vs_ssl.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"[DONE] summary csv saved to: {summary_csv}")
    print("==================== BASELINE vs SSL K SWEEP END ====================")


if __name__ == "__main__":
    main()
