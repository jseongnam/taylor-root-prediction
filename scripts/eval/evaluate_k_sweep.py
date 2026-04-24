#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/eval/evaluate_k_sweep.py

YAML-driven evaluation (paper/GitHub release):
- AST interval (center) predictor -> top-K centers
- Root regressors: anchored / ann / lstm
- Baseline solver: global scan + bracket + bisection + postcheck
- anchored_fb: anchored fails -> baseline fallback
- K-sweep, threshold sweep, winner(any/ok-only)
- func_id winner ratio + expr examples
- fail concentration by func_id (baseline fail / all-method fail)
- residual hist + func_id boxplot

Run:
  PYTHONPATH=. EVAL_CFG=configs/eval_k_sweep.yaml OUTDIR=results/runs_k_sweep_viz python scripts/eval/evaluate_k_sweep.py
"""

from __future__ import annotations

import os
import re
import ast
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict, OrderedDict
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import yaml
except Exception as e:
    raise ImportError("PyYAML required: pip install pyyaml") from e


# =========================================================
# Repo path helpers
# =========================================================

def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    cur = start if start.is_dir() else start.parent
    for _ in range(12):
        if (cur / ".git").exists() or (cur / "configs").exists():
            return cur
        cur = cur.parent
    return (start.parent if start.is_file() else start).resolve()

def resolve_repo_path(p: str, repo_root: Path) -> Path:
    p = str(p).strip()
    if not p:
        return Path("")
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (repo_root / pp).resolve()

def resolve_device(device_str: str) -> str:
    s = str(device_str).strip().lower()
    if s in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if s.startswith("cuda") and (not torch.cuda.is_available()):
        return "cpu"
    return s


# =========================================================
# YAML helpers
# =========================================================

def _get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj if isinstance(obj, dict) else {}

def parse_csv_list(s: str, cast=int) -> List:
    out = []
    for t in str(s).split(","):
        t = t.strip()
        if not t:
            continue
        out.append(cast(t))
    return out

def parse_thr_list(s: str) -> List[float]:
    xs = parse_csv_list(s, cast=float)
    if not xs:
        xs = [1e-10]
    xs = sorted(xs, reverse=True)
    return xs

def shorten(s: str, max_len: int = 220) -> str:
    s = str(s).replace("\n", " ").strip()
    return s if len(s) <= max_len else s[:max_len - 3] + "..."


# =========================================================
# Stats
# =========================================================

def time_stats_ms_full(x: np.ndarray):
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": np.nan, "std": np.nan, "p50": np.nan, "p90": np.nan, "p99": np.nan, "max": np.nan, "n": 0}
    return {
        "mean": float(a.mean()),
        "std":  float(a.std(ddof=0)),
        "p50":  float(np.percentile(a, 50.0)),
        "p90":  float(np.percentile(a, 90.0)),
        "p99":  float(np.percentile(a, 99.0)),
        "max":  float(a.max()),
        "n":    int(a.size),
    }

def abs_stats(x: np.ndarray):
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": np.nan, "p90": np.nan, "p99": np.nan, "max": np.nan, "n": 0}
    return {
        "mean": float(a.mean()),
        "p90":  float(np.percentile(a, 90.0)),
        "p99":  float(np.percentile(a, 99.0)),
        "max":  float(a.max()),
        "n":    int(a.size),
    }


# =========================================================
# Safe TRUE f(x) eval
# =========================================================

_FUNC_MAP = [
    ("exp", "np.exp"),
    ("sin", "np.sin"),
    ("cos", "np.cos"),
    ("tan", "np.tan"),
    ("tanh", "np.tanh"),
    ("sinh", "np.sinh"),
    ("cosh", "np.cosh"),
    ("log", "np.log"),
    ("log10", "np.log10"),
    ("sqrt", "np.sqrt"),
    ("abs", "np.abs"),
]

def sanitize_expr_for_eval(raw: str) -> str:
    s = str(raw).strip()
    if "= 0" in s:
        s = s.split("= 0")[0].strip()
    elif "=0" in s:
        s = s.split("=0")[0].strip()
    s = re.sub(r"\s*\([^()]*\)\s*$", "", s).strip()
    s = s.replace("^", "**")
    s = re.sub(r"\bnp\.", "", s)
    for src, dst in _FUNC_MAP:
        s = re.sub(rf"(?<!np\.)\b{src}\s*\(", f"{dst}(", s)
    s = re.sub(r"\bln\s*\(", "np.log(", s)
    return s

def make_callable(expr_sanitized: str):
    code = compile(expr_sanitized, "<expr>", "eval")
    allowed = {"np": np, "math": math}
    def f(x: float) -> float:
        x = float(x)
        with np.errstate(all="raise"):
            v = eval(code, {"__builtins__": {}}, {**allowed, "x": x})
        return float(v)
    return f

def safe_f_eval(f, x: float):
    try:
        v = f(float(x))
        v = float(v)
        if not np.isfinite(v):
            return False, float("nan")
        return True, v
    except FloatingPointError:
        return False, float("nan")
    except Exception:
        return False, float("nan")


# =========================================================
# Domain helpers (NPZ)
# =========================================================

def _norm_iv(a, b):
    a = float(a); b = float(b)
    if not np.isfinite(a): a = -float("inf")
    if not np.isfinite(b): b =  float("inf")
    if b < a: a, b = b, a
    return a, b

def extract_allowed_intervals(npz: np.lib.npyio.NpzFile, i: int):
    if ("domains" in npz) and ("domain_count" in npz):
        dom = npz["domains"]; cnt = npz["domain_count"]
        try:
            m = int(cnt[i])
        except Exception:
            m = 0
        out = []
        for k in range(max(0, m)):
            a = dom[i, k, 0]; b = dom[i, k, 1]
            if not (np.isfinite(a) or np.isfinite(b)):
                continue
            out.append(_norm_iv(a, b))
        if out:
            return out
    if ("x_min" in npz) and ("x_max" in npz):
        return [_norm_iv(npz["x_min"][i], npz["x_max"][i])]
    return [(-float("inf"), float("inf"))]

def in_any_interval(x: float, intervals):
    x = float(x)
    for a, b in intervals:
        if x >= a and x <= b:
            return True
    return False

def interval_containing_x(x: float, intervals):
    x = float(x)
    best = None
    for a, b in intervals:
        if x >= a and x <= b:
            w = (b - a) if (np.isfinite(a) and np.isfinite(b)) else float("inf")
            if best is None or w < best[0]:
                best = (w, a, b)
    if best is None:
        return None
    _, a, b = best
    return float(a), float(b)


# =========================================================
# Polynomial helpers
# =========================================================

def poly_eval_asc(coeffs_asc: np.ndarray, x: float) -> float:
    x = float(x)
    c = coeffs_asc.astype(np.float64)
    p = 0.0
    for a in reversed(c):
        p = p * x + float(a)
    return float(p)

def poly_shift_to_z(coeffs_asc: np.ndarray, c: float) -> np.ndarray:
    c = float(c)
    a = coeffs_asc.astype(np.float64)
    n = a.shape[0] - 1
    b = np.zeros((n + 1,), dtype=np.float64)
    cp = np.ones((n + 1,), dtype=np.float64)
    for t in range(1, n + 1):
        cp[t] = cp[t - 1] * c
    for i in range(0, n + 1):
        ai = float(a[i])
        if ai == 0.0:
            continue
        for k in range(0, i + 1):
            b[k] += ai * math.comb(i, k) * cp[i - k]
    return b.astype(np.float32)


# =========================================================
# Solver: bracket scan + bisection + Newton + postcheck
# =========================================================

def numeric_derivative(f, x, h_scale=1e-6):
    x = float(x)
    h = h_scale * (1.0 + abs(x))
    ok1, f1 = safe_f_eval(f, x + h)
    ok2, f2 = safe_f_eval(f, x - h)
    if not (ok1 and ok2):
        return np.nan
    return (f1 - f2) / (2.0 * h)

def find_brackets_by_scan(f, xmin, xmax, n=250):
    xmin = float(xmin); xmax = float(xmax)
    if not (np.isfinite(xmin) and np.isfinite(xmax)) or xmax <= xmin:
        return []
    xs = np.linspace(xmin, xmax, int(n), dtype=np.float64)
    fs = np.empty_like(xs, dtype=np.float64)
    valid = np.ones_like(xs, dtype=bool)
    for i in range(xs.size):
        ok, v = safe_f_eval(f, float(xs[i]))
        if ok:
            fs[i] = v
            valid[i] = True
        else:
            fs[i] = np.nan
            valid[i] = False
    brs = []
    for i in range(xs.size - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        f1, f2 = fs[i], fs[i + 1]
        if f1 == 0.0:
            a = b = float(xs[i]); mid = a
            brs.append((0.0, a, b, mid))
            continue
        if f2 == 0.0:
            a = b = float(xs[i + 1]); mid = a
            brs.append((0.0, a, b, mid))
            continue
        if np.sign(f1) * np.sign(f2) < 0.0:
            a = float(xs[i]); b = float(xs[i + 1]); mid = 0.5 * (a + b)
            okm, fm = safe_f_eval(f, mid)
            mid_abs = abs(fm) if okm else float("inf")
            brs.append((float(mid_abs), a, b, mid))
    brs.sort(key=lambda t: t[0])
    return brs

def bisection(f, a, b, max_iter=60, tol_f=1e-10, tol_x=1e-12):
    a = float(a); b = float(b)
    ok_a, fa = safe_f_eval(f, a)
    ok_b, fb = safe_f_eval(f, b)
    if not (ok_a and ok_b):
        return False, float("nan"), float("nan")
    if abs(fa) <= tol_f:
        return True, a, fa
    if abs(fb) <= tol_f:
        return True, b, fb
    if np.sign(fa) * np.sign(fb) > 0.0:
        return False, 0.5 * (a + b), float("nan")
    lo, hi = a, b
    flo, fhi = fa, fb
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        ok_m, fmid = safe_f_eval(f, mid)
        if not ok_m:
            return False, mid, float("nan")
        if abs(fmid) <= tol_f or abs(hi - lo) <= tol_x:
            return True, mid, fmid
        if np.sign(flo) * np.sign(fmid) < 0.0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    mid = 0.5 * (lo + hi)
    ok_m, fmid = safe_f_eval(f, mid)
    if not ok_m:
        return False, mid, float("nan")
    return True, mid, fmid

def newton_domainaware(f, x0, intervals, max_iter=30, tol_f=1e-10, tol_step=1e-12, max_step=2.0, dfx_eps=1e-14):
    x = float(x0)
    iv = interval_containing_x(x, intervals)
    ok0, fx = safe_f_eval(f, x)
    if not ok0:
        return False, x, float("nan")
    if abs(fx) <= tol_f:
        return True, x, fx
    for _ in range(int(max_iter)):
        dfx = numeric_derivative(f, x)
        if not np.isfinite(dfx) or abs(dfx) < dfx_eps:
            return False, x, fx
        step = fx / dfx
        if abs(step) > max_step:
            step = math.copysign(max_step, step)
        x_new = x - step
        if iv is not None:
            a, b = iv
            if np.isfinite(a): x_new = max(x_new, a)
            if np.isfinite(b): x_new = min(x_new, b)
        if not np.isfinite(x_new):
            return False, x, fx
        if abs(x_new - x) <= tol_step:
            ok1, f1 = safe_f_eval(f, x_new)
            if ok1 and abs(f1) <= tol_f:
                return True, x_new, f1
            return False, x_new, f1 if ok1 else float("nan")
        x = x_new
        ok, fx = safe_f_eval(f, x)
        if not ok:
            return False, x, float("nan")
        if abs(fx) <= tol_f:
            return True, x, fx
    return False, x, fx

def postcheck_root_stable(f, root: float, intervals, stable_radius: float, stable_scan_n: int, stable_valid_min: float, stable_dfx_min: float):
    if not np.isfinite(root):
        return False
    if not in_any_interval(root, intervals):
        return False
    iv = interval_containing_x(root, intervals)
    if iv is None:
        return False
    a, b = iv
    lo = max(root - stable_radius, a if np.isfinite(a) else root - stable_radius)
    hi = min(root + stable_radius, b if np.isfinite(b) else root + stable_radius)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return False
    xs = np.linspace(lo, hi, int(stable_scan_n), dtype=np.float64)
    valid = []
    has_bracket = False
    prev_ok = None
    prev_f = None
    for x in xs:
        ok, v = safe_f_eval(f, float(x))
        valid.append(bool(ok))
        if prev_ok is not None and prev_ok and ok:
            if (prev_f == 0.0) or (v == 0.0) or (np.sign(prev_f) * np.sign(v) < 0.0):
                has_bracket = True
        prev_ok, prev_f = bool(ok), float(v) if ok else None
    valid_ratio = float(np.mean(valid)) if valid else 0.0
    if (not has_bracket) or (valid_ratio < stable_valid_min):
        return False
    dfx = numeric_derivative(f, float(root))
    if not (np.isfinite(dfx) and abs(dfx) >= stable_dfx_min):
        return False
    return True

def solve_one(
    f, x0, intervals, solver_mode: str,
    tol_f: float, newton_iters: int, newton_max_step: float,
    local_radius: float, local_scan_n: int, local_max_brackets: int, bisect_iters: int,
    stable_radius: float, stable_scan_n: int, stable_valid_min: float, stable_dfx_min: float,
):
    solver_mode = str(solver_mode).lower()
    if solver_mode in ("newton", "newton_bisect"):
        okN, xN, fxN = newton_domainaware(
            f, x0, intervals,
            max_iter=newton_iters, tol_f=tol_f, max_step=newton_max_step,
        )
        if okN and postcheck_root_stable(f, xN, intervals, stable_radius, stable_scan_n, stable_valid_min, stable_dfx_min):
            ok_fx, fx = safe_f_eval(f, xN)
            if ok_fx:
                return True, float(xN), abs(float(fx))
        if solver_mode == "newton":
            return False, float("nan"), float("inf")

    brs = find_brackets_by_scan(f, float(x0) - local_radius, float(x0) + local_radius, n=local_scan_n)
    if not brs:
        return False, float("nan"), float("inf")

    best_fx = float("inf")
    best_x = float("nan")
    for _, a, b, _mid in brs[:int(local_max_brackets)]:
        okB, xB, _fxB = bisection(f, a, b, max_iter=bisect_iters, tol_f=tol_f)
        if not okB:
            continue
        if not postcheck_root_stable(f, xB, intervals, stable_radius, stable_scan_n, stable_valid_min, stable_dfx_min):
            continue
        ok_fx, fx = safe_f_eval(f, xB)
        if not ok_fx:
            continue
        fx_abs = abs(float(fx))
        if fx_abs < best_fx:
            best_fx = fx_abs
            best_x = float(xB)
        if best_fx <= tol_f:
            break
    if np.isfinite(best_x) and np.isfinite(best_fx):
        return True, best_x, best_fx
    return False, float("nan"), float("inf")


# =========================================================
# Baseline (global ranked topk brackets)
# =========================================================

def intersect_with_global_bounds(iv, gmin, gmax):
    a, b = iv
    lo = a if np.isfinite(a) else float(gmin)
    hi = b if np.isfinite(b) else float(gmax)
    lo = max(lo, float(gmin))
    hi = min(hi, float(gmax))
    return float(lo), float(hi)

def baseline_ranked_retry_topk(
    f, intervals,
    base_scan_xmin: float, base_scan_xmax: float, base_scan_n: int,
    tol_f: float, bisect_iters: int, baseline_topk: int,
    stable_radius: float, stable_scan_n: int, stable_valid_min: float, stable_dfx_min: float,
):
    candidates = []
    for iv in intervals:
        lo, hi = intersect_with_global_bounds(iv, base_scan_xmin, base_scan_xmax)
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            continue
        candidates.extend(find_brackets_by_scan(f, lo, hi, n=base_scan_n))
    if not candidates:
        return False, float("nan"), float("inf"), 0, 0, "no_bracket"
    candidates.sort(key=lambda t: t[0])
    candK = candidates[:int(baseline_topk)]
    tested = 0
    passed = 0
    best_fx = float("inf")
    best_x = float("nan")
    reason = "fail"
    for (mid_abs, a, b, mid) in candK:
        tested += 1
        okB, xB, _fxB = bisection(f, a, b, max_iter=bisect_iters, tol_f=tol_f)
        if not okB:
            reason = "bisect_fail"
            continue
        if not postcheck_root_stable(f, xB, intervals, stable_radius, stable_scan_n, stable_valid_min, stable_dfx_min):
            reason = "post_fail"
            continue
        ok_fx, fx = safe_f_eval(f, xB)
        if not ok_fx:
            reason = "fx_nan"
            continue
        passed += 1
        fx_abs = abs(float(fx))
        if fx_abs < best_fx:
            best_fx = fx_abs
            best_x = float(xB)
            reason = "ok"
        if best_fx <= tol_f:
            break
    if np.isfinite(best_x) and np.isfinite(best_fx):
        return True, best_x, best_fx, tested, passed, "ok"
    return False, float("nan"), float("inf"), tested, passed, reason


# =========================================================
# AST model (centers)
# =========================================================

_ALLOWED_FUNCS_AST = {"sin","cos","tan","tanh","sinh","cosh","exp","log","log10","sqrt","abs","ln"}
_BASE_TOKENS = ["<PAD>","<UNK>","<CLS>","x","NUM","+","-","*","/","**","neg","pos"]
_FUNC_TOKENS = sorted(list({"sin","cos","tan","tanh","sinh","cosh","exp","log","log10","sqrt","abs"}))
VOCAB = _BASE_TOKENS + _FUNC_TOKENS
STOI = {t:i for i,t in enumerate(VOCAB)}
PAD_ID = STOI["<PAD>"]; UNK_ID = STOI["<UNK>"]; CLS_ID = STOI["<CLS>"]; NUM_ID = STOI["NUM"]

def sanitize_expr_for_ast(raw: str) -> str:
    s = str(raw).strip()
    if "= 0" in s: s = s.split("= 0")[0].strip()
    elif "=0" in s: s = s.split("=0")[0].strip()
    s = re.sub(r"\s*\([^()]*\)\s*$", "", s).strip()
    s = s.replace("^","**")
    s = re.sub(r"\bnp\.", "", s)
    s = re.sub(r"\bln\s*\(", "log(", s)
    return s

def _tok_id(tok: str) -> int:
    return STOI.get(tok, UNK_ID)

def ast_to_prefix(node):
    tokens=[]; nums=[]
    def emit(tok, num=0.0):
        tokens.append(tok); nums.append(float(num))
    def visit(n):
        if isinstance(n, ast.Expression): return visit(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value,(int,float)) and np.isfinite(float(n.value)):
                emit("NUM", float(n.value)); return
            emit("<UNK>",0.0); return
        if isinstance(n, ast.Name):
            emit("x",0.0) if n.id=="x" else emit("<UNK>",0.0); return
        if isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.USub): emit("neg",0.0)
            elif isinstance(n.op, ast.UAdd): emit("pos",0.0)
            else: emit("<UNK>",0.0)
            visit(n.operand); return
        if isinstance(n, ast.BinOp):
            if isinstance(n.op, ast.Add): emit("+",0.0)
            elif isinstance(n.op, ast.Sub): emit("-",0.0)
            elif isinstance(n.op, ast.Mult): emit("*",0.0)
            elif isinstance(n.op, ast.Div): emit("/",0.0)
            elif isinstance(n.op, ast.Pow): emit("**",0.0)
            else: emit("<UNK>",0.0)
            visit(n.left); visit(n.right); return
        if isinstance(n, ast.Call):
            fname=None
            if isinstance(n.func, ast.Name): fname=n.func.id
            elif isinstance(n.func, ast.Attribute): fname=n.func.attr
            if fname=="ln": fname="log"
            emit(fname,0.0) if fname in _ALLOWED_FUNCS_AST else emit("<UNK>",0.0)
            if len(n.args)>=1: visit(n.args[0])
            else: emit("<UNK>",0.0)
            return
        emit("<UNK>",0.0)
    visit(node)
    return tokens, nums

def encode_prefix(tokens, nums, max_len: int):
    toks = ["<CLS>"] + tokens
    nvs  = [0.0] + nums
    if len(toks) > max_len:
        toks = toks[:max_len]
        nvs  = nvs[:max_len]
    ids = np.array([_tok_id(t) for t in toks], dtype=np.int64)
    numvals = np.array(nvs, dtype=np.float32)
    attn = np.ones((len(ids),), dtype=np.bool_)
    if len(ids) < max_len:
        pad_n = max_len - len(ids)
        ids = np.concatenate([ids, np.full((pad_n,), PAD_ID, dtype=np.int64)], axis=0)
        numvals = np.concatenate([numvals, np.zeros((pad_n,), dtype=np.float32)], axis=0)
        attn = np.concatenate([attn, np.zeros((pad_n,), dtype=np.bool_)], axis=0)
    return ids, numvals, attn

class ExprASTOnlyDataset(Dataset):
    def __init__(self, expr_arr, max_len: int, sanitize: bool=True):
        self.expr = expr_arr
        self.max_len = int(max_len)
        self.sanitize = bool(sanitize)
    def __len__(self): return len(self.expr)
    def __getitem__(self, idx):
        e = str(self.expr[idx])
        if self.sanitize:
            e = sanitize_expr_for_ast(e)
        try:
            node = ast.parse(e, mode="eval")
            toks, nums = ast_to_prefix(node)
        except Exception:
            toks, nums = ["<UNK>"], [0.0]
        ids, numvals, attn = encode_prefix(toks, nums, self.max_len)
        return (torch.from_numpy(ids),
                torch.from_numpy(numvals),
                torch.from_numpy(attn.astype(np.uint8)))

class ASTPrefixTransformerTopK(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, num_candidates: int,
                 d_model: int=256, nhead: int=8, num_layers: int=4):
        super().__init__()
        self.K = int(num_candidates)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.num_mlp = nn.Sequential(nn.Linear(1, d_model), nn.Tanh(), nn.Linear(d_model, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.y_head = nn.Linear(d_model, self.K)
    def forward(self, ids, numvals, attn_u8):
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        is_num = (ids == NUM_ID).unsqueeze(-1)
        x = x + self.num_mlp(numvals.unsqueeze(-1)) * is_num
        key_padding_mask = (attn_u8 == 0)
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        cls = h[:, 0, :]
        return self.y_head(cls)

def load_ast_topk_model(ckpt_path: Path, device: torch.device):
    obj = torch.load(ckpt_path, map_location=device)
    cfg = obj.get("config", {})
    if not isinstance(cfg, dict):
        raise RuntimeError("AST ckpt must contain dict 'config'")
    K = int(cfg.get("num_candidates", cfg.get("K", 10)))
    max_len = int(cfg.get("max_len", 128))
    d_model = int(cfg.get("d_model", 256))
    nhead = int(cfg.get("nhead", 8))
    num_layers = int(cfg.get("num_layers", 4))
    scale = float(cfg.get("scale", 1.0))
    sanitize_inputs = bool(cfg.get("sanitize_inputs", True))
    model = ASTPrefixTransformerTopK(
        vocab_size=len(VOCAB),
        max_len=max_len,
        num_candidates=K,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    ).to(device)
    sd = obj.get("model_state", None)
    if not isinstance(sd, dict):
        raise RuntimeError("AST ckpt must contain dict 'model_state'")
    model.load_state_dict(sd)
    model.eval()
    return model, cfg, scale, sanitize_inputs


# =========================================================
# Backends: anchored / ann / lstm
# =========================================================

def _safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location)


def anchored_predict_z(model: nn.Module, X: torch.Tensor, anchors: Optional[np.ndarray]):
    y = model(X)
    if isinstance(y, (tuple, list)):
        y = y[0]
    if y.dim() == 1:
        y = y.unsqueeze(0)
    if y.dim() == 2 and y.size(1) == 1:
        return y
    if anchors is None:
        raise RuntimeError("Anchored: logits output but anchors missing.")
    a = torch.from_numpy(np.array(anchors, dtype=np.float32)).to(y.device).view(1, -1)
    w = torch.softmax(y, dim=1)
    z = (w * a).sum(dim=1, keepdim=True)
    return z

def _extract_anchors(sd_or_obj):
    anchors = None
    if isinstance(sd_or_obj, dict) and ("anchors" in sd_or_obj):
        a = sd_or_obj["anchors"]
        try:
            if torch.is_tensor(a):
                anchors = a.detach().cpu().numpy().astype(np.float32)
            else:
                anchors = np.array(a, dtype=np.float32)
        except Exception:
            anchors = None
    return anchors

def _infer_prefix_from_state_dict(sd: dict):
    if any(k.startswith("net.") for k in sd.keys()):
        return "net"
    for cand in ("model.", "backbone.", "mlp.", "layers.", "seq."):
        if any(k.startswith(cand) for k in sd.keys()):
            return cand[:-1]
    for k in sd.keys():
        m = re.match(r"^([A-Za-z0-9_]+)\.\d+\.weight$", k)
        if m:
            return m.group(1)
    return None

def _collect_linear_keys_any(sd: dict):
    groups = {}
    for k, v in sd.items():
        if (not torch.is_tensor(v)) or v.ndim != 2 or (not k.endswith(".weight")):
            continue
        prefix = k.rsplit(".", 2)[0]
        groups.setdefault(prefix, []).append(k)
    return groups

def build_mlp_from_state_dict(sd, prefix=None, act_name="relu", dropout_p=0.0):
    idxs = []

    if prefix is not None:
        for k in sd.keys():
            m = re.match(rf"^{re.escape(prefix)}\.(\d+)\.weight$", k)
            if m:
                idxs.append(int(m.group(1)))
        idxs = sorted(set(idxs))

    if len(idxs) > 0:
        if act_name in ("relu",):
            Act = nn.ReLU
        elif act_name in ("tanh",):
            Act = nn.Tanh
        elif act_name in ("gelu",):
            Act = nn.GELU
        elif act_name in ("leakyrelu", "leaky_relu"):
            Act = lambda: nn.LeakyReLU(0.01)
        else:
            Act = nn.ReLU

        layers = []
        for i, li in enumerate(idxs):
            w = sd[f"{prefix}.{li}.weight"]
            if not torch.is_tensor(w):
                w = torch.tensor(w)
            out_dim, in_dim = int(w.shape[0]), int(w.shape[1])
            has_bias = (f"{prefix}.{li}.bias" in sd)
            layers.append((f"linear_{li}", nn.Linear(in_dim, out_dim, bias=has_bias)))
            if i < len(idxs) - 1:
                layers.append((f"act_{li}", Act()))
                if float(dropout_p) > 0:
                    layers.append((f"drop_{li}", nn.Dropout(p=float(dropout_p))))
        model = nn.Sequential(OrderedDict(layers))
        new_sd = {}
        for li in idxs:
            new_sd[f"linear_{li}.weight"] = sd[f"{prefix}.{li}.weight"]
            b_key = f"{prefix}.{li}.bias"
            if b_key in sd:
                new_sd[f"linear_{li}.bias"] = sd[b_key]
        return model, new_sd

    groups = _collect_linear_keys_any(sd)
    if not groups:
        raise RuntimeError("[anchored] state_dict에서 선형층(weight ndim=2)을 찾지 못했습니다.")

    best_prefix, best_keys = max(groups.items(), key=lambda kv: (len(kv[1]), kv[0]))
    best_keys = sorted(best_keys)
    if act_name in ("relu",):
        Act = nn.ReLU
    elif act_name in ("tanh",):
        Act = nn.Tanh
    elif act_name in ("gelu",):
        Act = nn.GELU
    elif act_name in ("leakyrelu", "leaky_relu"):
        Act = lambda: nn.LeakyReLU(0.01)
    else:
        Act = nn.ReLU

    layers = []
    new_sd = {}
    for i, w_key in enumerate(best_keys):
        w = sd[w_key]
        if not torch.is_tensor(w):
            w = torch.tensor(w)
        out_dim, in_dim = int(w.shape[0]), int(w.shape[1])
        base = w_key[:-len(".weight")]
        b_key = base + ".bias"
        has_bias = b_key in sd
        lname = f"linear_{i}"
        layers.append((lname, nn.Linear(in_dim, out_dim, bias=has_bias)))
        new_sd[f"{lname}.weight"] = sd[w_key]
        if has_bias:
            new_sd[f"{lname}.bias"] = sd[b_key]
        if i < len(best_keys) - 1:
            layers.append((f"act_{i}", Act()))
            if float(dropout_p) > 0:
                layers.append((f"drop_{i}", nn.Dropout(p=float(dropout_p))))
    model = nn.Sequential(OrderedDict(layers))
    return model, new_sd

def load_backend_anchored(ckpt_path: Path, device: torch.device):
    obj = _safe_torch_load(ckpt_path, map_location=device)
    cfg = obj.get("config", {}) if isinstance(obj, dict) and isinstance(obj.get("config", {}), dict) else {}

    if isinstance(obj, dict) and ("model_state" in obj) and isinstance(obj["model_state"], dict):
        sd = dict(obj["model_state"])
    elif isinstance(obj, dict) and ("state_dict" in obj) and isinstance(obj["state_dict"], dict):
        sd = dict(obj["state_dict"])
    elif isinstance(obj, dict):
        sd = {k: v for k, v in obj.items() if torch.is_tensor(v)}
        if not sd:
            sd = dict(obj)
    else:
        raise RuntimeError(f"[anchored] Unsupported checkpoint type: {type(obj)}")

    anchors = None
    if isinstance(cfg, dict):
        anchors = _extract_anchors(cfg)
    if anchors is None and isinstance(obj, dict):
        anchors = _extract_anchors(obj)
    if anchors is None:
        anchors = _extract_anchors(sd)

    if "anchors" in sd:
        sd.pop("anchors", None)

    prefix = _infer_prefix_from_state_dict(sd)

    act_name = str(cfg.get("activation", "relu")).lower() if isinstance(cfg, dict) else "relu"
    dropout_p = 0.0
    if isinstance(cfg, dict) and ("dropout" in cfg):
        try:
            dropout_p = float(cfg.get("dropout", 0.0))
        except Exception:
            dropout_p = 0.0

    model, new_sd = build_mlp_from_state_dict(sd, prefix=prefix, act_name=act_name, dropout_p=dropout_p)
    model = model.to(device)
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing or unexpected:
        print(f"[anchored] load_state_dict strict=False | missing={list(missing)} unexpected={list(unexpected)}")
    model.eval()
    scaler = None
    return model, anchors, scaler



class ANNRootRegressor(nn.Module):
    def __init__(self, in_dim: int, num_roots: int, arch_cfg: dict):
        super().__init__()
        hidden_dim = int(arch_cfg.get("hidden_dim", 25))
        layers = arch_cfg.get("layers", "auto")
        activation = str(arch_cfg.get("activation", "tanh")).lower()
        dropout = float(arch_cfg.get("dropout", 0.0))
        bounded_output = bool(arch_cfg.get("bounded_output", False))
        root_range = float(arch_cfg.get("root_range", 10.0))

        if isinstance(layers, str) and layers.lower() == "auto":
            if in_dim <= 64:
                hlist = [hidden_dim, hidden_dim, hidden_dim]
            elif in_dim <= 256:
                hlist = [hidden_dim] * 4
            else:
                hlist = [hidden_dim] * 5
        elif isinstance(layers, int):
            hlist = [hidden_dim] * int(layers)
        elif isinstance(layers, (list, tuple)):
            hlist = [int(x) for x in layers]
        else:
            hlist = [hidden_dim, hidden_dim, hidden_dim]

        if activation == "tanh":
            def make_act(): return nn.Tanh()
        elif activation == "relu":
            def make_act(): return nn.ReLU(inplace=True)
        elif activation == "gelu":
            def make_act(): return nn.GELU()
        elif activation in ("silu", "swish"):
            def make_act(): return nn.SiLU(inplace=True)
        else:
            def make_act(): return nn.Tanh()

        mods = []
        prev = in_dim
        for h in hlist:
            mods.append(nn.Linear(prev, h))
            mods.append(make_act())
            if dropout > 0:
                mods.append(nn.Dropout(dropout))
            prev = h

        self.backbone = nn.Sequential(*mods) if mods else nn.Identity()
        self.head = nn.Linear(prev, num_roots)
        self.bounded_output = bounded_output
        self.root_range = root_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.head(self.backbone(x))
        if self.bounded_output:
            y = torch.tanh(y) * float(self.root_range)
        return y


def _load_json_if_exists(p: Path):
    try:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _resolve_optional_json(base_path: Path, path_str: str, repo_root: Path):
    s = str(path_str).strip()
    if not s:
        return None
    p = Path(s)
    if p.is_absolute() and p.exists():
        return p
    cands = [
        (base_path.parent / p).resolve(),
        (repo_root / p).resolve(),
    ]
    for c in cands:
        if c.exists():
            return c
    return None


def _extract_tensor_state_dict(obj):
    if isinstance(obj, dict):
        if "model" in obj and isinstance(obj["model"], dict):
            if all(torch.is_tensor(v) for v in obj["model"].values()):
                return dict(obj["model"]), "model"
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            if all(torch.is_tensor(v) for v in obj["state_dict"].values()):
                return dict(obj["state_dict"]), "state_dict"
        if "model_state" in obj and isinstance(obj["model_state"], dict):
            if all(torch.is_tensor(v) for v in obj["model_state"].values()):
                return dict(obj["model_state"]), "model_state"
        tensor_items = {k: v for k, v in obj.items() if torch.is_tensor(v)}
        if tensor_items:
            return tensor_items, "flat"
    raise RuntimeError("ANN checkpoint에서 state_dict를 찾지 못했습니다.")


def _infer_ann_kind(sd: dict):
    keys = list(sd.keys())
    if "fc1.weight" in sd and "fc2.weight" in sd:
        return "shallow"
    if any(k.startswith("backbone.") for k in keys) and any(k.startswith("head.") for k in keys):
        return "modern"
    if "head.weight" in sd:
        return "modern"
    return "shallow"

class ShallowFNN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden, out_dim)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
def _build_shallow_ann_from_sd(sd: dict, device: torch.device):
    if "fc1.weight" not in sd or "fc2.weight" not in sd:
        raise RuntimeError("Shallow ANN state_dict에 fc1/fc2가 없습니다.")
    in_dim = int(sd["fc1.weight"].shape[1])
    hidden = int(sd["fc1.weight"].shape[0])
    out_dim = int(sd["fc2.weight"].shape[0])
    model = ShallowFNN(in_dim=in_dim, out_dim=out_dim, hidden=hidden).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, in_dim, out_dim


def _build_modern_ann_from_sd(sd: dict, ck: dict, repo_root: Path, ckpt_path: Path, device: torch.device):
    cfg = {}
    if isinstance(ck, dict) and isinstance(ck.get("config", None), dict):
        cfg = ck["config"]

    cfg_json_path = None
    if isinstance(ck, dict):
        cfg_json_path = _resolve_optional_json(ckpt_path, ck.get("config_json", ""), repo_root)
    if cfg_json_path is not None:
        loaded = _load_json_if_exists(cfg_json_path)
        if isinstance(loaded, dict):
            cfg = loaded

    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    arch_cfg = cfg.get("architecture", {}) if isinstance(cfg, dict) else {}

    # infer in_dim / num_roots if not present
    if "head.weight" in sd:
        num_roots = int(sd["head.weight"].shape[0])
        prev_dim = int(sd["head.weight"].shape[1])
    else:
        last_linear_key = None
        for k, v in sd.items():
            if k.endswith(".weight") and torch.is_tensor(v) and v.ndim == 2:
                last_linear_key = k
        if last_linear_key is None:
            raise RuntimeError("Modern ANN state_dict에서 마지막 linear weight를 찾지 못했습니다.")
        num_roots = int(sd[last_linear_key].shape[0])
        prev_dim = int(sd[last_linear_key].shape[1])

    first_linear_key = None
    for k, v in sd.items():
        if k.endswith(".weight") and torch.is_tensor(v) and v.ndim == 2:
            first_linear_key = k
            break
    if first_linear_key is None:
        raise RuntimeError("Modern ANN state_dict에서 첫 linear weight를 찾지 못했습니다.")
    inferred_in_dim = int(sd[first_linear_key].shape[1])

    in_dim = int(ck.get("in_dim", model_cfg.get("input", {}).get("dimension", model_cfg.get("input", {}).get("order", inferred_in_dim - 1) + 1) if isinstance(model_cfg, dict) else inferred_in_dim))
    num_roots = int(ck.get("num_roots", model_cfg.get("output", {}).get("num_roots", num_roots) if isinstance(model_cfg, dict) else num_roots))

    # infer hidden/layers if config missing
    if not arch_cfg:
        hidden_guess = prev_dim
        backbone_linears = []
        for k, v in sd.items():
            if k.startswith("backbone.") and k.endswith(".weight") and torch.is_tensor(v) and v.ndim == 2:
                backbone_linears.append(k)
        layers_guess = len(backbone_linears)
        arch_cfg = {
            "hidden_dim": hidden_guess,
            "layers": max(1, layers_guess) if layers_guess > 0 else "auto",
            "activation": "tanh",
            "dropout": 0.0,
            "bounded_output": False,
            "root_range": 10.0,
        }

    model = ANNRootRegressor(in_dim=in_dim, num_roots=num_roots, arch_cfg=arch_cfg).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[ann] load_state_dict strict=False | missing={list(missing)} unexpected={list(unexpected)}")
    model.eval()
    return model, in_dim, num_roots


def load_backend_ann_mdpi(ckpt_path: Path, device: torch.device, repo_root: Path):
    ck = _safe_torch_load(ckpt_path, map_location=device)

    if isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], dict) and "in_dim" in ck and "out_dim" in ck:
        model = ShallowFNN(
            in_dim=int(ck["in_dim"]),
            out_dim=int(ck["out_dim"]),
            hidden=int(ck.get("hidden", 10)),
        ).to(device)
        model.load_state_dict(ck["model"], strict=False)
        model.eval()
    else:
        sd, _src = _extract_tensor_state_dict(ck)
        kind = _infer_ann_kind(sd)
        if kind == "shallow":
            model, in_dim, out_dim = _build_shallow_ann_from_sd(sd, device=device)
        else:
            model, in_dim, out_dim = _build_modern_ann_from_sd(
                sd, ck if isinstance(ck, dict) else {}, repo_root, ckpt_path, device=device
            )

    scaler_path = ""
    if isinstance(ck, dict):
        scaler_path = str(ck.get("scaler_json", "")).strip()

    spath = None
    if scaler_path:
        spath = _resolve_optional_json(ckpt_path, scaler_path, repo_root)

    if spath is None:
        for cand in [
            ckpt_path.parent / "scaler.json",
            ckpt_path.parent / "scaler_minmax.json",
            ckpt_path.parent / "config_resolved.json",
        ]:
            if cand.exists():
                spath = cand
                break

    if spath is None or (not spath.exists()):
        raise RuntimeError(f"ANN scaler_json not found near checkpoint: {ckpt_path}")

    sc = _load_json_if_exists(spath)
    if not isinstance(sc, dict) or ("x_min" not in sc) or ("x_max" not in sc):
        raise RuntimeError(f"ANN scaler file invalid or missing x_min/x_max: {spath}")

    x_min = torch.tensor(sc["x_min"], dtype=torch.float32, device=device)
    x_max = torch.tensor(sc["x_max"], dtype=torch.float32, device=device)

    y_min = None
    y_max = None
    if ("y_min" in sc) and ("y_max" in sc):
        y_min = torch.tensor(sc["y_min"], dtype=torch.float32, device=device)
        y_max = torch.tensor(sc["y_max"], dtype=torch.float32, device=device)
    else:
        print(f"[ann] scaler without y_min/y_max -> output is treated as already original scale: {spath}")

    return model, (x_min, x_max, y_min, y_max)
def _minmax_to_minus1_1_torch(x: torch.Tensor, mn: torch.Tensor, mx: torch.Tensor, eps: float = 1e-12):
    den = torch.clamp(mx - mn, min=eps)
    return (2.0 * (x - mn) / den - 1.0)

def _inv_minmax_from_minus1_1_torch(x_scaled: torch.Tensor, mn: Optional[torch.Tensor], mx: Optional[torch.Tensor]):
    if mn is None or mx is None:
        return x_scaled
    return ((x_scaled + 1.0) * 0.5 * (mx - mn) + mn)
def ann_predict_z(model: nn.Module, X: torch.Tensor, scalers):
    x_min, x_max, y_min, y_max = scalers
    x_scaled = _minmax_to_minus1_1_torch(X, x_min, x_max)
    with torch.no_grad():
        y_scaled = model(x_scaled)
    y_org = _inv_minmax_from_minus1_1_torch(y_scaled, y_min, y_max)
    return y_org[:, 0:1]

class LSTMRootRegressor(nn.Module):
    def __init__(
        self,
        hidden: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        num_roots: int = 1,
        bounded_output: bool = False,
        root_range: float = 10.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, int(num_roots)),
        )
        self.bounded_output = bool(bounded_output)
        self.root_range = float(root_range)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        y = self.head(h_n[-1])
        if self.bounded_output:
            y = torch.tanh(y) * self.root_range
        return y
def _extract_lstm_state_dict(ck):
    if isinstance(ck, dict):
        for key in ("model", "state_dict", "model_state"):
            if key in ck and isinstance(ck[key], dict):
                if all(torch.is_tensor(v) for v in ck[key].values()):
                    return dict(ck[key]), key
        tensor_items = {k: v for k, v in ck.items() if torch.is_tensor(v)}
        if tensor_items:
            return tensor_items, "flat"
    raise RuntimeError("LSTM checkpoint에서 state_dict를 찾지 못했습니다.")
def load_backend_lstm(ckpt_path: Path, device: torch.device):
    ck = _safe_torch_load(ckpt_path, map_location=device)
    sd, src_kind = _extract_lstm_state_dict(ck)

    cfg = ck.get("config", {}) if isinstance(ck, dict) and isinstance(ck.get("config", {}), dict) else {}

    config_json_path = None
    if isinstance(ck, dict):
        config_json_path = _resolve_optional_json(
            ckpt_path,
            ck.get("config_json", ""),
            find_repo_root(ckpt_path),
        )
    if config_json_path is not None:
        loaded = _load_json_if_exists(config_json_path)
        if isinstance(loaded, dict):
            cfg = loaded

    layer_ids = set()
    for k in sd.keys():
        m = re.match(r"lstm\.weight_ih_l(\d+)$", k)
        if m:
            layer_ids.add(int(m.group(1)))
    inferred_layers = max(layer_ids) + 1 if layer_ids else 2

    hidden = int(sd["lstm.weight_ih_l0"].shape[0] // 4) if "lstm.weight_ih_l0" in sd else 128
    dropout = 0.0
    num_roots = 1
    bounded_output = False
    root_range = 10.0

    if isinstance(cfg, dict):
        arch = cfg.get("architecture", {}) if isinstance(cfg.get("architecture", {}), dict) else {}
        model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}

        if "hidden_dim" in arch:
            hidden = int(arch["hidden_dim"])

        layers_raw = arch.get("layers", inferred_layers)
        if isinstance(layers_raw, str) and layers_raw.strip().lower() == "auto":
            layers = inferred_layers
        else:
            layers = int(layers_raw)

        dropout = float(arch.get("dropout", 0.0))
        bounded_output = bool(arch.get("bounded_output", False))
        root_range = float(arch.get("root_range", 10.0))

        if isinstance(model_cfg, dict) and ("num_roots" in model_cfg):
            num_roots = int(model_cfg["num_roots"])
        elif isinstance(model_cfg.get("output", {}), dict) and ("num_roots" in model_cfg["output"]):
            num_roots = int(model_cfg["output"]["num_roots"])
        elif isinstance(ck, dict) and ("num_roots" in ck):
            num_roots = int(ck["num_roots"])
        elif "head.2.weight" in sd:
            num_roots = int(sd["head.2.weight"].shape[0])
        elif "head.weight" in sd:
            num_roots = int(sd["head.weight"].shape[0])
    else:
        layers = inferred_layers
        if "head.2.weight" in sd:
            num_roots = int(sd["head.2.weight"].shape[0])

    model = LSTMRootRegressor(
        hidden=hidden,
        num_layers=layers,
        dropout=dropout,
        num_roots=num_roots,
        bounded_output=bounded_output,
        root_range=root_range,
    ).to(device)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[lstm] load_state_dict strict=False | missing={list(missing)} unexpected={list(unexpected)} | src={src_kind}")

    model.eval()
    return model


def lstm_predict_z(model: nn.Module, X: torch.Tensor):
    x_seq = X.unsqueeze(-1)
    with torch.no_grad():
        z = model(x_seq)
    return z[:, 0:1] if z.dim() == 2 else z


# =========================================================
# Reason registry (for fail reports)
# =========================================================

class ReasonRegistry:
    def __init__(self):
        self.reason_to_id: Dict[str, int] = {}
        self.id_to_reason: List[str] = []
        self.get_id("ok")
    def get_id(self, reason: str) -> int:
        r = str(reason)
        if r in self.reason_to_id:
            return self.reason_to_id[r]
        rid = len(self.id_to_reason)
        self.reason_to_id[r] = rid
        self.id_to_reason.append(r)
        return rid

def _decode_reason_ids(reason_id_arr: np.ndarray, rr: ReasonRegistry):
    rid = np.asarray(reason_id_arr, dtype=np.int64)
    out = np.array([""] * rid.shape[0], dtype=object)
    for i in range(rid.shape[0]):
        r = int(rid[i])
        if 0 <= r < len(rr.id_to_reason):
            out[i] = rr.id_to_reason[r]
        else:
            out[i] = f"unknown_reason_id({r})"
    return out


# =========================================================
# Baseline cache
# =========================================================

def load_baseline_cache(cache_path: Path, N: int):
    if cache_path is None or (not cache_path.exists()):
        return None
    try:
        obj = np.load(cache_path, allow_pickle=True)
    except Exception:
        return None
    need = ["baseline_root","baseline_abs","baseline_time_ms","baseline_method","baseline_done","baseline_reason_id"]
    for k in need:
        if k not in obj:
            return None
    root = obj["baseline_root"]
    absv = obj["baseline_abs"]
    tms  = obj["baseline_time_ms"]
    meth = obj["baseline_method"]
    done = obj["baseline_done"]
    rsid = obj["baseline_reason_id"]
    if root.shape[0] != N:
        return None
    return {
        "root": root.astype(np.float64),
        "abs":  absv.astype(np.float64),
        "time_ms": tms.astype(np.float64),
        "method": meth.astype(object),
        "done": done.astype(bool),
        "reason_id": rsid.astype(np.int32),
    }

def save_baseline_cache(cache_path: Path, base_root, base_abs, base_time, base_method, base_done, base_reason_id):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        baseline_root=base_root,
        baseline_abs=base_abs,
        baseline_time_ms=base_time,
        baseline_method=base_method,
        baseline_done=base_done.astype(np.bool_),
        baseline_reason_id=base_reason_id.astype(np.int32),
    )


# =========================================================
# Winner/threshold/reports/plots
# =========================================================

def threshold_sweep_table(methods: List[str], abs_by: Dict[str, np.ndarray], thr_list: List[float]):
    ok_rate = {}
    ok_mask_by = {}
    for thr in thr_list:
        ok_rate_thr = {}
        ok_mask_thr = {}
        for m in methods:
            a = np.asarray(abs_by[m], dtype=np.float64)
            ok = np.isfinite(a) & (a <= float(thr))
            ok_rate_thr[m] = float(ok.mean())
            ok_mask_thr[m] = ok
        ok_rate[thr] = ok_rate_thr
        ok_mask_by[thr] = ok_mask_thr
    return ok_rate, ok_mask_by

def print_threshold_sweep(methods: List[str], ok_rate: dict, thr_list: List[float]):
    print("\n==================== THRESHOLD SWEEP (ok%) ====================")
    header = "thr".ljust(12) + " | " + " | ".join([m.rjust(16) for m in methods])
    print(header)
    print("-" * len(header))
    for thr in thr_list:
        row = f"{thr:.1e}".ljust(12) + " | " + " | ".join([f"{ok_rate[thr][m]*100:14.2f}%" for m in methods])
        print(row)
    print("===============================================================\n")

def compute_winner(methods: List[str], abs_by: Dict[str, np.ndarray], ok_by: Dict[str, np.ndarray], ok_only: bool):
    mats = []
    for m in methods:
        a = np.asarray(abs_by[m], dtype=np.float64)
        a = np.where(np.isfinite(a), a, np.inf)
        if ok_only:
            ok = np.asarray(ok_by[m], dtype=bool)
            a = np.where(ok, a, np.inf)
        mats.append(a)
    mat = np.stack(mats, axis=1)
    idx = np.argmin(mat, axis=1)
    minv = np.min(mat, axis=1)
    w = np.array(["none"] * mat.shape[0], dtype=object)
    has = np.isfinite(minv) & (minv < np.inf)
    for i in range(mat.shape[0]):
        if has[i]:
            w[i] = methods[int(idx[i])]
    return w

def print_winner_summary(methods: List[str], winner: np.ndarray, title: str):
    print(f"\n==================== {title} ====================")
    N = int(winner.shape[0])
    c = Counter([str(x) for x in winner])
    for m in methods:
        cnt = int(c.get(m, 0))
        print(f"{m:16s}: {cnt:8d} ({cnt/max(1,N)*100.0:6.2f}%)")
    none_cnt = int(c.get("none", 0))
    if none_cnt > 0:
        print(f"{'none':16s}: {none_cnt:8d} ({none_cnt/max(1,N)*100.0:6.2f}%)")
    print("=============================================================\n")

def print_funcid_winner_ratio_with_expr(
    func_id: np.ndarray,
    func_expr: np.ndarray,
    winner_methods: List[str],
    winner_any: np.ndarray,
    winner_ok: np.ndarray,
    ok_by_method: Dict[str, np.ndarray],
    thr: float,
    mode: str = "both",
    expr_k: int = 3,
    topn: int = 50,
):
    fid = np.asarray(func_id)
    uniq = np.unique(fid)
    counts = {u: int((fid == u).sum()) for u in uniq}
    uniq_sorted = sorted(list(uniq), key=lambda u: (-counts[u], int(u)))

    print("\n==================== FUNC_ID WINNER RATIO (with expr examples) ====================")
    print(f"[thr={thr:.1e}] mode={mode} | expr_k={expr_k} | show topn={topn}")
    print("-----------------------------------------------------------------------------------")

    shown = 0
    for u in uniq_sorted:
        idx_mask = (fid == u)
        n = int(idx_mask.sum())
        if n <= 0:
            continue

        ok_parts = []
        for m in winner_methods:
            okm = ok_by_method[m][idx_mask]
            ok_parts.append(f"{m}:{okm.mean()*100:5.1f}%")
        print(f"func_id={int(u):4d} | n={n:6d} | " + " ".join(ok_parts))

        def _ratio(warr: np.ndarray):
            w = warr[idx_mask]
            c = Counter([str(x) for x in w])
            denom = max(1, n)
            return {m: float(c.get(m, 0) / denom) for m in (winner_methods + ["none"])}

        if mode in ("any","both"):
            r = _ratio(winner_any)
            line = "  winner(any)   : " + " ".join([f"{m}={r[m]*100:5.1f}%" for m in winner_methods]) + f" none={r['none']*100:5.1f}%"
            print(line)
        if mode in ("okonly","both"):
            r = _ratio(winner_ok)
            line = "  winner(okonly): " + " ".join([f"{m}={r[m]*100:5.1f}%" for m in winner_methods]) + f" none={r['none']*100:5.1f}%"
            print(line)

        seen = set()
        ex = []
        for gi in np.where(idx_mask)[0].tolist():
            s = shorten(func_expr[gi], 220)
            if s in seen:
                continue
            seen.add(s)
            ex.append(s)
            if len(ex) >= int(expr_k):
                break
        for j, s in enumerate(ex):
            print(f"   expr{j+1}: {s}")
        print("-----------------------------------------------------------------------------------")

        shown += 1
        if shown >= int(topn):
            break

    print("================== END FUNC_ID WINNER RATIO REPORT ==================\n")

def report_fail_concentration_by_funcid(
    *,
    func_id: np.ndarray,
    func_expr: np.ndarray,
    fail_mask: np.ndarray,
    reason_str: Optional[np.ndarray],
    topn: int = 30,
    expr_k: int = 3,
    title: str = "FAIL CONCENTRATION BY FUNC_ID",
    save_csv_path: Optional[Path] = None,
    save_json_path: Optional[Path] = None,
):
    fid = np.asarray(func_id)
    fail_mask = np.asarray(fail_mask, dtype=bool)
    uniq = np.unique(fid)
    rows = []

    for u in uniq.tolist():
        u = int(u)
        m_all = (fid == u)
        n_all = int(m_all.sum())
        if n_all == 0:
            continue
        m_fail = m_all & fail_mask
        n_fail = int(m_fail.sum())
        fail_rate = n_fail / max(1, n_all)

        samples = []
        seen = set()
        idx_fail = np.where(m_fail)[0]
        for gi in idx_fail.tolist():
            e = shorten(func_expr[gi], 220)
            if e in seen:
                continue
            seen.add(e)
            samples.append(e)
            if len(samples) >= int(expr_k):
                break

        top_reasons = []
        if reason_str is not None:
            rs = reason_str[m_fail]
            c = Counter([str(x) for x in rs])
            top_reasons = c.most_common(3)

        rows.append({
            "func_id": u,
            "n_all": n_all,
            "n_fail": n_fail,
            "fail_rate": fail_rate,
            "top_reasons": top_reasons,
            "expr_examples": samples,
        })

    rows.sort(key=lambda r: (-r["n_fail"], -r["fail_rate"], r["func_id"]))

    print(f"\n==================== {title} ====================")
    print(f"N_total={int(fid.shape[0])}, N_fail={int(fail_mask.sum())} ({fail_mask.mean()*100:.2f}%)")
    print("--------------------------------------------------------------------------")
    print("rank | func_id | n_fail / n_all | fail_rate | top_reasons")
    print("--------------------------------------------------------------------------")

    shown = 0
    for rank, r in enumerate(rows, start=1):
        if r["n_fail"] <= 0:
            continue
        tr = ""
        if r["top_reasons"]:
            tr = " ; ".join([f"{k}:{v}" for (k, v) in r["top_reasons"]])
        print(f"{rank:4d} | {r['func_id']:7d} | {r['n_fail']:6d}/{r['n_all']:<6d} | {r['fail_rate']*100:8.2f}% | {tr}")
        for j, e in enumerate(r["expr_examples"][:expr_k]):
            print(f"       expr{j+1}: {e}")
        print("--------------------------------------------------------------------------")
        shown += 1
        if shown >= int(topn):
            break

    print("================== END FAIL CONCENTRATION REPORT ==================\n")

    if save_csv_path is not None:
        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with open(save_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["func_id", "n_fail", "n_all", "fail_rate", "top_reasons"])
            for r in rows:
                tr = ""
                if r["top_reasons"]:
                    tr = " ; ".join([f"{k}:{v}" for (k, v) in r["top_reasons"]])
                w.writerow([r["func_id"], r["n_fail"], r["n_all"], f"{r['fail_rate']:.6f}", tr])

    if save_json_path is not None:
        save_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

def plot_residual_histogram(abs_arr: np.ndarray, out_png: Path, title: str, log10: bool = True):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(abs_arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return
    y = np.log10(a + 1e-30) if log10 else a
    plt.figure(figsize=(10, 5))
    plt.hist(y, bins=60)
    plt.xlabel("log10(|f|+eps)" if log10 else "|f|")
    plt.ylabel("count")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def plot_funcid_boxplot(abs_arr: np.ndarray, func_id: np.ndarray, out_png: Path, topn: int = 15, log10: bool = True, title: str = ""):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(abs_arr, dtype=np.float64)
    fid = np.asarray(func_id)
    uniq = np.unique(fid)
    counts = {u: int((fid == u).sum()) for u in uniq}
    order = sorted(list(uniq), key=lambda u: (-counts[u], int(u)))
    if int(topn) > 0:
        order = order[:int(topn)]
    data = []
    labels = []
    for u in order:
        mask = (fid == u) & np.isfinite(a)
        v = a[mask]
        if v.size == 0:
            continue
        if log10:
            v = np.log10(v + 1e-30)
        data.append(v)
        labels.append(f"{int(u)}\n(n={v.size})")
    if not data:
        return
    plt.figure(figsize=(max(12, 0.8 * len(data)), 6))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("log10(|f|+eps)" if log10 else "|f|")
    plt.title(title if title else "Residual by func_id (boxplot)")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


# =========================================================
# Evaluation core
# =========================================================

@dataclass
class EvalArgs:
    compare_thr: float
    thr_sweep: str
    topk_list: List[int]
    batch_size: int
    device: str

    solver_mode: str
    max_tries: int
    stop_after_first_success: bool
    proxy: str

    tol_f: float
    newton_iters: int
    newton_max_step: float
    bisect_iters: int

    local_radius: float
    local_scan_n: int
    local_max_brackets: int

    stable_radius: float
    stable_scan_n: int
    stable_valid_min: float
    stable_dfx_min: float

    baseline_mode: str
    baseline_topk: int
    base_scan_xmin: float
    base_scan_xmax: float
    base_scan_n: int

    anchored_fb: str

    baseline_cache: str
    baseline_cache_save: bool

    thr_winner_okonly: bool
    report_funcid_winner: bool
    report_funcid_winner_mode: str
    report_funcid_winner_expr_k: int
    report_funcid_winner_topn: int

    report_fail_funcid: bool
    report_fail_mode: str
    report_fail_funcid_topn: int
    report_fail_expr_k: int
    report_fail_save: bool

    plot_residual_hist: bool
    plot_funcid_box: bool
    plot_topn_funcid: int

def make_out_struct(N: int):
    return {
        "root": np.full((N,), np.nan, dtype=np.float64),
        "abs":  np.full((N,), np.inf, dtype=np.float64),
        "ok":   np.zeros((N,), dtype=np.bool_),
        "time_ms": np.full((N,), np.nan, dtype=np.float64),
        "method": np.array([""] * N, dtype=object),
        "tested": np.zeros((N,), dtype=np.int16),
        "passed": np.zeros((N,), dtype=np.int16),
        "reason_id": np.zeros((N,), dtype=np.int32),
    }

def run_eval_for_K(
    *,
    K: int,
    centers_all: np.ndarray,        # (N, K_need)
    coeffs: np.ndarray,             # (N, D)
    expr: np.ndarray,               # (N,)
    func_id: np.ndarray | None,
    data: np.lib.npyio.NpzFile,
    backends: list,                 # list[(name, pack)]
    device: torch.device,
    rr: ReasonRegistry,
    args: EvalArgs,
    base_cache_obj: dict | None,
):
    N = coeffs.shape[0]
    K_all = centers_all.shape[1]
    K_use = min(int(K), int(K_all))
    centers = centers_all[:, :K_use].astype(np.float64)

    out = {name: make_out_struct(N) for (name, _) in backends}

    enable_anchored_fb = False
    if args.anchored_fb == "off":
        enable_anchored_fb = False
    elif args.anchored_fb == "on":
        enable_anchored_fb = True
    else:
        enable_anchored_fb = (args.baseline_mode != "none")
    if enable_anchored_fb:
        out["anchored_fb"] = make_out_struct(N)

    base_root = np.full((N,), np.nan, dtype=np.float64)
    base_abs  = np.full((N,), np.inf, dtype=np.float64)
    base_time = np.full((N,), np.nan, dtype=np.float64)
    base_method = np.array([""] * N, dtype=object)
    base_done = np.zeros((N,), dtype=np.bool_)
    base_reason_id = np.full((N,), rr.get_id("baseline_not_run"), dtype=np.int32)

    if base_cache_obj is not None and base_cache_obj["done"].shape[0] == N:
        base_root[:] = base_cache_obj["root"]
        base_abs[:]  = base_cache_obj["abs"]
        base_time[:] = base_cache_obj["time_ms"]
        base_method[:] = base_cache_obj["method"]
        base_done[:] = base_cache_obj["done"]
        if "reason_id" in base_cache_obj:
            base_reason_id[:] = base_cache_obj["reason_id"]
        else:
            base_reason_id[:] = rr.get_id("baseline_cached")

    thr = float(args.compare_thr)

    f_cache: Dict[str, Any] = {}

    def maybe_compute_baseline(gi: int, f, intervals, force: bool):
        if args.baseline_mode == "none":
            return
        if base_done[gi]:
            return
        if args.baseline_mode == "on_demand" and (not force):
            return

        tb0 = time.perf_counter()
        okB, xB, fxB, tested, passed, reason = baseline_ranked_retry_topk(
            f=f,
            intervals=intervals,
            base_scan_xmin=float(args.base_scan_xmin),
            base_scan_xmax=float(args.base_scan_xmax),
            base_scan_n=int(args.base_scan_n),
            tol_f=float(args.tol_f),
            bisect_iters=int(args.bisect_iters),
            baseline_topk=int(args.baseline_topk),
            stable_radius=float(args.stable_radius),
            stable_scan_n=int(args.stable_scan_n),
            stable_valid_min=float(args.stable_valid_min),
            stable_dfx_min=float(args.stable_dfx_min),
        )
        tb1 = time.perf_counter()

        base_time[gi] = (tb1 - tb0) * 1000.0
        base_method[gi] = f"baseline:ranked{int(args.baseline_topk)}"
        base_done[gi] = True

        if okB:
            base_root[gi] = float(xB)
            base_abs[gi]  = float(fxB)
            base_reason_id[gi] = rr.get_id("ok")
        else:
            base_root[gi] = np.nan
            base_abs[gi]  = np.inf
            base_reason_id[gi] = rr.get_id(f"baseline_fail:{reason}")

    for gi in tqdm(range(N), desc=f"eval K={K_use}", ncols=110):
        raw_expr = str(expr[gi])
        san = sanitize_expr_for_eval(raw_expr)
        if san in f_cache:
            f = f_cache[san]
        else:
            try:
                f = make_callable(san)
            except Exception:
                f = None
            f_cache[san] = f

        if f is None:
            for name in out.keys():
                out[name]["method"][gi] = "fail_expr"
                out[name]["abs"][gi] = np.inf
                out[name]["ok"][gi] = False
                out[name]["reason_id"][gi] = rr.get_id("fail_expr")
            base_done[gi] = True
            base_abs[gi] = np.inf
            base_reason_id[gi] = rr.get_id("fail_expr")
            continue

        intervals = extract_allowed_intervals(data, gi)

        if args.baseline_mode == "all":
            maybe_compute_baseline(gi, f, intervals, force=True)

        for (bname, pack) in backends:
            t0 = time.perf_counter()
            cand = []

            for k in range(K_use):
                c = float(centers[gi, k])
                if not np.isfinite(c):
                    continue
                if not in_any_interval(c, intervals):
                    continue

                q = poly_shift_to_z(coeffs[gi], c)
                Xnp = q.astype(np.float32)

                try:
                    if pack["type"] == "anchored":
                        X = torch.from_numpy(Xnp[None, :]).to(device)
                        z = anchored_predict_z(pack["model"], X, pack.get("anchors", None))
                        z0 = float(z[0, 0].item())
                    elif pack["type"] == "ann":
                        X = torch.from_numpy(Xnp[None, :]).to(device)
                        z = ann_predict_z(pack["model"], X, pack["scalers"])
                        z0 = float(z[0, 0].item())
                    else:
                        X = torch.from_numpy(Xnp[None, :]).to(device)
                        z = lstm_predict_z(pack["model"], X)
                        z0 = float(z[0, 0].item())
                except Exception:
                    continue

                x0 = c + z0
                if args.proxy == "true":
                    ok0, fx0 = safe_f_eval(f, x0)
                    score = abs(float(fx0)) if ok0 else abs(poly_eval_asc(coeffs[gi], x0))
                else:
                    score = abs(poly_eval_asc(coeffs[gi], x0))

                cand.append((float(score), float(x0)))

            if not cand:
                t1 = time.perf_counter()
                out[bname]["time_ms"][gi] = (t1 - t0) * 1000.0
                out[bname]["method"][gi] = "no_candidate"
                out[bname]["abs"][gi] = np.inf
                out[bname]["ok"][gi] = False
                out[bname]["reason_id"][gi] = rr.get_id("no_candidate")
                continue

            cand.sort(key=lambda t: t[0])
            tries = min(int(args.max_tries), len(cand))

            best_fx = float("inf")
            best_x = float("nan")
            tested = 0
            passed = 0
            last_reason = "all_candidates_failed"

            for r in range(tries):
                tested += 1
                _, x0 = cand[r]
                okS, xS, fxS = solve_one(
                    f=f, x0=float(x0), intervals=intervals, solver_mode=args.solver_mode,
                    tol_f=float(args.tol_f),
                    newton_iters=int(args.newton_iters),
                    newton_max_step=float(args.newton_max_step),
                    local_radius=float(args.local_radius),
                    local_scan_n=int(args.local_scan_n),
                    local_max_brackets=int(args.local_max_brackets),
                    bisect_iters=int(args.bisect_iters),
                    stable_radius=float(args.stable_radius),
                    stable_scan_n=int(args.stable_scan_n),
                    stable_valid_min=float(args.stable_valid_min),
                    stable_dfx_min=float(args.stable_dfx_min),
                )
                if not okS:
                    last_reason = "solve_fail"
                    continue
                passed += 1
                if fxS < best_fx:
                    best_fx = float(fxS)
                    best_x = float(xS)
                if args.stop_after_first_success:
                    break
                if best_fx <= float(args.tol_f):
                    break

            t1 = time.perf_counter()
            out[bname]["time_ms"][gi] = (t1 - t0) * 1000.0
            out[bname]["tested"][gi] = int(tested)
            out[bname]["passed"][gi] = int(passed)

            if np.isfinite(best_x) and np.isfinite(best_fx):
                out[bname]["root"][gi] = best_x
                out[bname]["abs"][gi] = best_fx
                out[bname]["ok"][gi] = bool(best_fx <= thr)
                out[bname]["method"][gi] = f"{bname}:tested={tested},passed={passed}"
                out[bname]["reason_id"][gi] = rr.get_id("ok")
            else:
                out[bname]["abs"][gi] = np.inf
                out[bname]["ok"][gi] = False
                out[bname]["method"][gi] = f"{bname}:fail|tested={tested}"
                out[bname]["reason_id"][gi] = rr.get_id(last_reason)

        if enable_anchored_fb and ("anchored" in out):
            anchored_ok = bool(np.isfinite(out["anchored"]["abs"][gi]) and out["anchored"]["abs"][gi] <= thr)
            if anchored_ok:
                out["anchored_fb"]["root"][gi] = out["anchored"]["root"][gi]
                out["anchored_fb"]["abs"][gi]  = out["anchored"]["abs"][gi]
                out["anchored_fb"]["ok"][gi]   = True
                out["anchored_fb"]["time_ms"][gi] = out["anchored"]["time_ms"][gi]
                out["anchored_fb"]["method"][gi] = "anchored_fb:use_anchored"
                out["anchored_fb"]["reason_id"][gi] = rr.get_id("ok")
            else:
                maybe_compute_baseline(gi, f, intervals, force=True)
                base_ok = bool(base_done[gi] and np.isfinite(base_abs[gi]) and base_abs[gi] <= thr)
                if base_ok:
                    out["anchored_fb"]["root"][gi] = float(base_root[gi])
                    out["anchored_fb"]["abs"][gi]  = float(base_abs[gi])
                    out["anchored_fb"]["ok"][gi]   = True
                    at = out["anchored"]["time_ms"][gi]
                    bt = base_time[gi]
                    at = float(at) if np.isfinite(at) else 0.0
                    bt = float(bt) if np.isfinite(bt) else 0.0
                    out["anchored_fb"]["time_ms"][gi] = at + bt
                    out["anchored_fb"]["method"][gi] = "anchored_fb:anchored_fail->baseline_ok"
                    out["anchored_fb"]["reason_id"][gi] = rr.get_id("ok")
                else:
                    out["anchored_fb"]["abs"][gi] = np.inf
                    out["anchored_fb"]["ok"][gi] = False
                    out["anchored_fb"]["time_ms"][gi] = out["anchored"]["time_ms"][gi]
                    out["anchored_fb"]["method"][gi] = "anchored_fb:both_fail"
                    out["anchored_fb"]["reason_id"][gi] = out["anchored"]["reason_id"][gi]

    baseline_pack = {
        "root": base_root, "abs": base_abs, "time_ms": base_time,
        "method": base_method, "done": base_done, "reason_id": base_reason_id,
    }
    return out, baseline_pack


def run_all_from_prepared(
    *,
    topk_list: List[int],
    centers_all: np.ndarray,
    coeffs: np.ndarray,
    expr: np.ndarray,
    func_id: np.ndarray | None,
    data: np.lib.npyio.NpzFile,
    backends: list,
    device: torch.device,
    rr: ReasonRegistry,
    args: EvalArgs,
    outdir: Path,
):
    thr = float(args.compare_thr)
    thr_list = parse_thr_list(args.thr_sweep)

    baseline_cache_path = Path(args.baseline_cache) if str(args.baseline_cache).strip() else None
    base_cache_obj = None
    if baseline_cache_path is not None:
        base_cache_obj = load_baseline_cache(baseline_cache_path, coeffs.shape[0])
        if base_cache_obj is not None:
            print(f"[BASELINE CACHE] loaded: {baseline_cache_path} | done={int(base_cache_obj['done'].sum())}/{coeffs.shape[0]}")

    perK_results = {}

    print("\n==================== K SWEEP START ====================")
    for K in topk_list:
        out_methods, baseline_pack = run_eval_for_K(
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

        if base_cache_obj is None:
            base_cache_obj = {
                "root": baseline_pack["root"].copy(),
                "abs": baseline_pack["abs"].copy(),
                "time_ms": baseline_pack["time_ms"].copy(),
                "method": baseline_pack["method"].copy(),
                "done": baseline_pack["done"].copy(),
                "reason_id": baseline_pack["reason_id"].copy(),
            }
        else:
            msk = (~base_cache_obj["done"]) & baseline_pack["done"]
            base_cache_obj["root"][msk] = baseline_pack["root"][msk]
            base_cache_obj["abs"][msk]  = baseline_pack["abs"][msk]
            base_cache_obj["time_ms"][msk] = baseline_pack["time_ms"][msk]
            base_cache_obj["method"][msk] = baseline_pack["method"][msk]
            base_cache_obj["done"][msk] = True
            if "reason_id" in base_cache_obj:
                base_cache_obj["reason_id"][msk] = baseline_pack["reason_id"][msk]

        perK_results[int(K)] = (out_methods, baseline_pack)

        print(f"\n-------------------- [K={int(K)}] SUMMARY --------------------")
        b_ok = np.isfinite(baseline_pack["abs"]) & (baseline_pack["abs"] <= thr)
        bt = time_stats_ms_full(baseline_pack["time_ms"])
        ba = abs_stats(baseline_pack["abs"])
        print(f"[baseline] ok@{thr:.1e} = {b_ok.mean()*100:6.2f}% | "
              f"|f| mean={ba['mean']:.3e} p90={ba['p90']:.3e} p99={ba['p99']:.3e} | "
              f"time(ms) mean={bt['mean']:.3f} std={bt['std']:.3f} p50={bt['p50']:.3f} p90={bt['p90']:.3f} p99={bt['p99']:.3f}")

        for mname in out_methods.keys():
            ok = np.isfinite(out_methods[mname]["abs"]) & (out_methods[mname]["abs"] <= thr)
            tt = time_stats_ms_full(out_methods[mname]["time_ms"])
            aa = abs_stats(out_methods[mname]["abs"])
            print(f"[{mname}] ok@{thr:.1e} = {ok.mean()*100:6.2f}% | "
                  f"|f| mean={aa['mean']:.3e} p90={aa['p90']:.3e} p99={aa['p99']:.3e} | "
                  f"time(ms) mean={tt['mean']:.3f} std={tt['std']:.3f} p50={tt['p50']:.3f} p90={tt['p90']:.3f} p99={tt['p99']:.3f}")
        print("----------------------------------------------------------")

    print("==================== K SWEEP END ====================\n")

    if baseline_cache_path is not None and bool(args.baseline_cache_save) and base_cache_obj is not None:
        save_baseline_cache(
            baseline_cache_path,
            base_cache_obj["root"], base_cache_obj["abs"], base_cache_obj["time_ms"],
            base_cache_obj["method"], base_cache_obj["done"],
            base_cache_obj.get("reason_id", np.full_like(base_cache_obj["done"], rr.get_id("baseline_cached"), dtype=np.int32)),
        )
        print(f"[BASELINE CACHE] saved/updated: {baseline_cache_path}")

    K_ref = int(max(topk_list))
    out_methods, baseline_pack = perK_results[K_ref]

    abs_by = {m: out_methods[m]["abs"] for m in out_methods.keys()}
    ok_by  = {m: (np.isfinite(out_methods[m]["abs"]) & (out_methods[m]["abs"] <= thr)) for m in out_methods.keys()}

    abs_by["baseline"] = np.where(baseline_pack["done"], baseline_pack["abs"], np.inf)
    ok_by["baseline"]  = np.where(baseline_pack["done"], (np.isfinite(baseline_pack["abs"]) & (baseline_pack["abs"] <= thr)), False)

    winner_methods = list(out_methods.keys()) + ["baseline"]

    winner_any = compute_winner(winner_methods, abs_by, ok_by, ok_only=False)
    print_winner_summary(winner_methods, winner_any, f"WINNER (best |f|) @K={K_ref}")

    winner_ok = compute_winner(winner_methods, abs_by, ok_by, ok_only=True)
    print_winner_summary(winner_methods, winner_ok, f"WINNER OK-ONLY @thr={thr:.1e} @K={K_ref}")

    ok_rate, ok_mask_by_thr = threshold_sweep_table(winner_methods, abs_by, thr_list)
    print_threshold_sweep(winner_methods, ok_rate, thr_list)

    if bool(args.thr_winner_okonly):
        for tthr in thr_list:
            tmp_ok_by = {m: ok_mask_by_thr[tthr][m] for m in winner_methods}
            w_ok_thr = compute_winner(winner_methods, abs_by, tmp_ok_by, ok_only=True)
            print_winner_summary(winner_methods, w_ok_thr, f"WINNER OK-ONLY @thr={tthr:.1e} @K={K_ref}")

    if bool(args.report_fail_funcid) and (func_id is not None):
        baseline_abs = np.where(baseline_pack["done"], baseline_pack["abs"], np.inf)
        baseline_ok = np.isfinite(baseline_abs) & (baseline_abs <= thr)

        if str(args.report_fail_mode) == "baseline":
            fail_mask = ~baseline_ok
            title = f"BASELINE FAIL CONCENTRATION BY FUNC_ID @thr={thr:.1e} @K={K_ref}"
            reason_str = _decode_reason_ids(baseline_pack["reason_id"], rr) if ("reason_id" in baseline_pack) else None
        else:
            fail_mask = (winner_ok == "none")
            title = f"ALL-METHOD FAIL CONCENTRATION (winner_ok==none) @thr={thr:.1e} @K={K_ref}"
            reason_str = _decode_reason_ids(baseline_pack["reason_id"], rr) if ("reason_id" in baseline_pack) else None

        save_csv = None
        save_json = None
        if bool(args.report_fail_save):
            mode = str(args.report_fail_mode)
            save_csv = outdir / f"fail_by_funcid_{mode}_K{K_ref}_thr{thr:.0e}.csv"
            save_json = outdir / f"fail_by_funcid_{mode}_K{K_ref}_thr{thr:.0e}.json"

        report_fail_concentration_by_funcid(
            func_id=func_id,
            func_expr=expr,
            fail_mask=fail_mask,
            reason_str=reason_str,
            topn=int(args.report_fail_funcid_topn),
            expr_k=int(args.report_fail_expr_k),
            title=title,
            save_csv_path=save_csv,
            save_json_path=save_json,
        )

    if bool(args.report_funcid_winner) and (func_id is not None):
        print_funcid_winner_ratio_with_expr(
            func_id=func_id,
            func_expr=expr,
            winner_methods=winner_methods,
            winner_any=winner_any,
            winner_ok=winner_ok,
            ok_by_method=ok_by,
            thr=thr,
            mode=str(args.report_funcid_winner_mode),
            expr_k=int(args.report_funcid_winner_expr_k),
            topn=int(args.report_funcid_winner_topn),
        )

    if bool(args.plot_residual_hist):
        for m in out_methods.keys():
            p = outdir / f"hist_residual_{m}_K{K_ref}.png"
            plot_residual_histogram(out_methods[m]["abs"], p, title=f"{m} residual histogram (K={K_ref})", log10=True)
        p = outdir / f"hist_residual_baseline_K{K_ref}.png"
        plot_residual_histogram(baseline_pack["abs"], p, title=f"baseline residual histogram (K={K_ref})", log10=True)

    if bool(args.plot_funcid_box) and (func_id is not None):
        for m in out_methods.keys():
            p = outdir / f"box_funcid_{m}_K{K_ref}.png"
            plot_funcid_boxplot(out_methods[m]["abs"], func_id, p, topn=int(args.plot_topn_funcid), log10=True,
                                title=f"{m} residual by func_id (K={K_ref})")
        p = outdir / f"box_funcid_baseline_K{K_ref}.png"
        plot_funcid_boxplot(baseline_pack["abs"], func_id, p, topn=int(args.plot_topn_funcid), log10=True,
                            title=f"baseline residual by func_id (K={K_ref})")

    print(f"\n[FINISHED] outputs saved to: {outdir.resolve()}")


# =========================================================
# Main
# =========================================================

def main():
    repo = find_repo_root(Path(__file__))

    cfg_path = os.environ.get("EVAL_CFG", "configs/eval_k_sweep.yaml")
    cfg_path_p = resolve_repo_path(cfg_path, repo)
    cfg = load_yaml(cfg_path_p)

    outdir_str = os.environ.get("OUTDIR", str(_get(cfg, "outdir", "results/runs_k_sweep_viz")))
    outdir = resolve_repo_path(outdir_str, repo)
    outdir.mkdir(parents=True, exist_ok=True)

    suppress_runtime_warnings = bool(_get(cfg, "runtime.suppress_runtime_warnings", False))
    if suppress_runtime_warnings:
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    device_str = resolve_device(os.environ.get("DEVICE", str(_get(cfg, "device", "auto"))))
    device = torch.device(device_str)
    batch_size = int(_get(cfg, "batch_size", 256))

    test_npz = str(_get(cfg, "data.test_npz", "data/taylor_test_physchem_v3_allroots_10000.npz"))
    test_path = resolve_repo_path(test_npz, repo)
    if not test_path.exists():
        raise FileNotFoundError(f"test_npz not found: {test_path}")

    ast_ckpt = resolve_repo_path(str(_get(cfg, "models.ast_ckpt", "")), repo)
    anchored_ckpt = resolve_repo_path(str(_get(cfg, "models.anchored_ckpt", "")), repo)
    ann_ckpt = resolve_repo_path(str(_get(cfg, "models.ann_ckpt", "")), repo)
    lstm_ckpt = resolve_repo_path(str(_get(cfg, "models.lstm_ckpt", "")), repo)

    topk_list = parse_csv_list(_get(cfg, "k_sweep.topk_list", "5,10,15,20,25"), cast=int)
    topk_list = [k for k in topk_list if k > 0]
    if not topk_list:
        topk_list = [10]

    compare_thr = float(_get(cfg, "thresholds.compare_thr", 1e-10))
    thr_sweep = str(_get(cfg, "thresholds.thr_sweep", "1e-6,1e-8,1e-10,1e-12"))

    args = EvalArgs(
        compare_thr=compare_thr,
        thr_sweep=thr_sweep,
        topk_list=topk_list,
        batch_size=batch_size,
        device=device_str,

        solver_mode=str(_get(cfg, "solver.solver_mode", "newton_bisect")),
        max_tries=int(_get(cfg, "solver.max_tries", 10)),
        stop_after_first_success=bool(_get(cfg, "solver.stop_after_first_success", False)),
        proxy=str(_get(cfg, "solver.proxy", "poly")),

        tol_f=float(_get(cfg, "solver.tol_f", 1e-10)),
        newton_iters=int(_get(cfg, "solver.newton_iters", 30)),
        newton_max_step=float(_get(cfg, "solver.newton_max_step", 2.0)),
        bisect_iters=int(_get(cfg, "solver.bisect_iters", 60)),

        local_radius=float(_get(cfg, "solver.local_radius", 1.0)),
        local_scan_n=int(_get(cfg, "solver.local_scan_n", 101)),
        local_max_brackets=int(_get(cfg, "solver.local_max_brackets", 10)),

        stable_radius=float(_get(cfg, "solver.stable_radius", 1.0)),
        stable_scan_n=int(_get(cfg, "solver.stable_scan_n", 80)),
        stable_valid_min=float(_get(cfg, "solver.stable_valid_min", 0.7)),
        stable_dfx_min=float(_get(cfg, "solver.stable_dfx_min", 1e-10)),

        baseline_mode=str(_get(cfg, "baseline.mode", "all")),
        baseline_topk=int(_get(cfg, "baseline.topk", 10)),
        base_scan_xmin=float(_get(cfg, "baseline.scan_xmin", -20.0)),
        base_scan_xmax=float(_get(cfg, "baseline.scan_xmax", +20.0)),
        base_scan_n=int(_get(cfg, "baseline.scan_n", 250)),
        anchored_fb=str(_get(cfg, "baseline.anchored_fb", "auto")),

        baseline_cache=str(_get(cfg, "baseline.cache", "")),
        baseline_cache_save=bool(_get(cfg, "baseline.cache_save", False)),

        thr_winner_okonly=bool(_get(cfg, "reports.thr_winner_okonly", False)),
        report_funcid_winner=bool(_get(cfg, "reports.report_funcid_winner", False)),
        report_funcid_winner_mode=str(_get(cfg, "reports.report_funcid_winner_mode", "both")),
        report_funcid_winner_expr_k=int(_get(cfg, "reports.report_funcid_winner_expr_k", 3)),
        report_funcid_winner_topn=int(_get(cfg, "reports.report_funcid_winner_topn", 50)),

        report_fail_funcid=bool(_get(cfg, "reports.report_fail_funcid", False)),
        report_fail_mode=str(_get(cfg, "reports.report_fail_mode", "baseline")),
        report_fail_funcid_topn=int(_get(cfg, "reports.report_fail_funcid_topn", 30)),
        report_fail_expr_k=int(_get(cfg, "reports.report_fail_expr_k", 3)),
        report_fail_save=bool(_get(cfg, "reports.report_fail_save", False)),

        plot_residual_hist=bool(_get(cfg, "plots.plot_residual_hist", False)),
        plot_funcid_box=bool(_get(cfg, "plots.plot_funcid_box", False)),
        plot_topn_funcid=int(_get(cfg, "plots.plot_topn_funcid", 15)),
    )

    if args.anchored_fb not in ("auto", "on", "off"):
        args.anchored_fb = "auto"
    if args.baseline_mode not in ("all", "on_demand", "none"):
        args.baseline_mode = "all"

    print(f"[CFG] repo_root={repo}")
    print(f"[CFG] cfg={cfg_path_p}")
    print(f"[CFG] outdir={outdir.resolve()}")
    print(f"[CFG] device={device} batch={batch_size} K_list={topk_list}")
    print(f"[CFG] test_npz={test_path}")
    print(f"[CFG] compare_thr={compare_thr:.1e} thr_list={parse_thr_list(thr_sweep)}")
    print(f"[CFG] solver_mode={args.solver_mode} baseline_mode={args.baseline_mode} anchored_fb={args.anchored_fb}")

    data = np.load(test_path, allow_pickle=True)
    coeffs = data["coeffs"].astype(np.float32)

    if "func_expr" in data:
        expr = data["func_expr"]
    elif "expr_str" in data:
        expr = data["expr_str"]
    else:
        raise KeyError("NPZ must contain 'func_expr' or 'expr_str'.")

    N, D = coeffs.shape
    func_id = data["func_id"] if ("func_id" in data and data["func_id"].shape[0] == N) else None
    print(f"[NPZ] N={N} degree={D-1} has_func_id={func_id is not None}")
    test_npz = str(_get(cfg, "data.test_npz", "data/taylor_test_physchem_v3_allroots_10000.npz"))
    test_path = resolve_repo_path(test_npz, repo)
    if not test_path.exists():
        raise FileNotFoundError(f"test_npz not found: {test_path}")
    if test_path.is_dir():
        raise IsADirectoryError(f"test_npz is a directory, not a file: {test_path}")

    eval_degree = _get(cfg, "data.eval_degree", None)
    env_eval_degree = os.environ.get("EVAL_DEGREE", "").strip()
    if env_eval_degree:
        eval_degree = int(env_eval_degree)
    elif eval_degree is not None:
        eval_degree = int(eval_degree)

    data = np.load(test_path, allow_pickle=True)

    raw_coeffs = data["coeffs"].astype(np.float32)
    if raw_coeffs.ndim != 2:
        raise ValueError(f"coeffs must be 2D, got shape={raw_coeffs.shape}")

    raw_D = raw_coeffs.shape[1]

    if eval_degree is None:
        coeffs = raw_coeffs
    else:
        need_dim = int(eval_degree) + 1
        if raw_D < need_dim:
            raise ValueError(
                f"master coeff dim too small: raw_dim={raw_D}, need={need_dim} for eval_degree={eval_degree}"
            )
        coeffs = raw_coeffs[:, :need_dim].copy()

    if "func_expr" in data:
        expr = data["func_expr"]
    elif "expr_str" in data:
        expr = data["expr_str"]
    else:
        raise KeyError("NPZ must contain 'func_expr' or 'expr_str'.")

    N, D = coeffs.shape
    func_id = data["func_id"] if ("func_id" in data and data["func_id"].shape[0] == N) else None

    print(f"[NPZ] path={test_path}")
    print(f"[NPZ] raw_degree={raw_D - 1} eval_degree={D - 1} has_func_id={func_id is not None}")
    if (ast_ckpt is None) or (not ast_ckpt.exists()):
        raise FileNotFoundError(f"AST ckpt not found: {ast_ckpt}")

    ast_model, ast_cfg, ast_scale, ast_sanitize = load_ast_topk_model(ast_ckpt, device=device)
    K_all = int(ast_cfg.get("num_candidates", ast_cfg.get("K", 10)))
    max_len = int(ast_cfg.get("max_len", 128))
    K_need = min(max(topk_list), K_all)
    print(f"[AST] ckpt={ast_ckpt} K_all={K_all} K_need={K_need} max_len={max_len} scale={ast_scale} sanitize={ast_sanitize}")

    ds_expr = ExprASTOnlyDataset(expr, max_len=max_len, sanitize=ast_sanitize)
    dl = DataLoader(ds_expr, batch_size=batch_size, shuffle=False, num_workers=0)

    centers_all = np.zeros((N, K_need), dtype=np.float64)
    idx0 = 0
    for ids, numvals, attn in tqdm(dl, desc="AST forward", ncols=110):
        B = ids.size(0)
        ids = ids.to(device); numvals = numvals.to(device); attn = attn.to(device)
        with torch.no_grad():
            y = ast_model(ids, numvals, attn).double()[:, :K_need].contiguous()
            c = (float(ast_scale) * torch.sinh(y)).cpu().numpy().astype(np.float64)
        centers_all[idx0:idx0 + B, :] = c
        idx0 += B

    backends = []
    if anchored_ckpt is not None and anchored_ckpt.exists():
        m, anchors, scaler = load_backend_anchored(anchored_ckpt, device=device)
        backends.append(("anchored", {"type": "anchored", "model": m, "anchors": anchors, "scaler": scaler}))
        print(f"[anchored] {anchored_ckpt}")
    if ann_ckpt is not None and ann_ckpt.exists():
        m, scalers = load_backend_ann_mdpi(ann_ckpt, device=device, repo_root=repo)
        backends.append(("ann", {"type": "ann", "model": m, "scalers": scalers}))
        print(f"[ann] {ann_ckpt}")
    if lstm_ckpt is not None and lstm_ckpt.exists():
        m = load_backend_lstm(lstm_ckpt, device=device)
        backends.append(("lstm", {"type": "lstm", "model": m}))
        print(f"[lstm] {lstm_ckpt}")
    expected_in_dim = coeffs.shape[1]

    def _infer_first_linear_in_dim(model: nn.Module):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                return int(m.in_features)
        return None

    for bname, pack in backends:
        if pack["type"] in ("anchored", "ann"):
            in_dim = _infer_first_linear_in_dim(pack["model"])
            if in_dim is not None and in_dim != expected_in_dim:
                raise ValueError(
                    f"[{bname}] input dim mismatch: model expects {in_dim}, "
                    f"but test coeff dim is {expected_in_dim}"
            )
    if not backends:
        raise RuntimeError("No backend checkpoints found. Set models.*_ckpt in YAML to existing files.")

    rr = ReasonRegistry()

    run_all_from_prepared(
        topk_list=topk_list,
        centers_all=centers_all,
        coeffs=coeffs,
        expr=expr,
        func_id=func_id,
        data=data,
        backends=backends,
        device=device,
        rr=rr,
        args=args,
        outdir=outdir,
    )


if __name__ == "__main__":
    main()