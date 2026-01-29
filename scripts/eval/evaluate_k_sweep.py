#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/eval/evaluate_k_sweep.py

GitHub(논문 공개용) 정리 버전: K-sweep 평가 + baseline 포함
- anchored / ann / lstm / anchored_fb(anchored 실패 시 baseline 구제) + baseline
- K sweep (ex: 5,10,15,20,25)
- threshold sweep(ok% table)
- winner(any / ok-only)
- func_id winner ratio + expr 예시
- fail concentration by func_id (baseline fail / all-method fail)
- residual hist + func_id boxplot

✅ argparse 최소화(=없음). YAML 기반 실행.
- 기본 설정 파일: configs/eval_k_sweep.yaml
- 환경변수로 override 가능:
    EVAL_CFG=/path/to/eval_k_sweep.yaml
    OUTDIR=/path/to/outdir

실행:
  python scripts/eval/evaluate_k_sweep.py
"""

from __future__ import annotations

import os, re, ast, json, math, time, warnings
from dataclasses import dataclass
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import yaml
except Exception as e:
    raise ImportError("PyYAML 필요: pip install pyyaml") from e


# =========================================================
# Config
# =========================================================

def _get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_csv_list(s: str, cast=int) -> List:
    out = []
    for t in str(s).split(","):
        t = t.strip()
        if not t:
            continue
        out.append(cast(t))
    return out

def parse_csv_float_list(s: str) -> List[float]:
    return parse_csv_list(s, cast=float)

def time_stats_ms(x: np.ndarray):
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

def shorten(s: str, max_len: int = 220) -> str:
    s = str(s).replace("\n", " ").strip()
    return s if len(s) <= max_len else s[:max_len-3] + "..."


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
    # trailing "(...)" param dump 제거(너가 쓰던 습관 유지)
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
# Domain helpers (NPZ에 domains/domain_count 또는 x_min/x_max 있으면 사용)
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
    # q(z) = p(z+c)
    c = float(c)
    a = coeffs_asc.astype(np.float64)
    n = a.shape[0] - 1
    b = np.zeros((n+1,), dtype=np.float64)
    cp = np.ones((n+1,), dtype=np.float64)
    for t in range(1, n+1):
        cp[t] = cp[t-1] * c
    for i in range(0, n+1):
        ai = float(a[i])
        if ai == 0.0:
            continue
        for k in range(0, i+1):
            b[k] += ai * math.comb(i, k) * cp[i-k]
    return b.astype(np.float32)


# =========================================================
# Solver: bracket scan + bisection + domain-aware Newton + postcheck
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
        if not (valid[i] and valid[i+1]):
            continue
        f1, f2 = fs[i], fs[i+1]
        if f1 == 0.0:
            a=b=float(xs[i]); mid=a
            brs.append((0.0, a, b, mid))
            continue
        if f2 == 0.0:
            a=b=float(xs[i+1]); mid=a
            brs.append((0.0, a, b, mid))
            continue
        if np.sign(f1) * np.sign(f2) < 0.0:
            a=float(xs[i]); b=float(xs[i+1]); mid=0.5*(a+b)
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
        return False, 0.5*(a+b), float("nan")

    lo, hi = a, b
    flo, fhi = fa, fb
    for _ in range(int(max_iter)):
        mid = 0.5*(lo+hi)
        ok_m, fmid = safe_f_eval(f, mid)
        if not ok_m:
            return False, mid, float("nan")
        if abs(fmid) <= tol_f:
            return True, mid, fmid
        if abs(hi-lo) <= tol_x:
            return True, mid, fmid
        if np.sign(flo) * np.sign(fmid) < 0.0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    mid = 0.5*(lo+hi)
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

    prev_x = None
    prev_ok = None
    prev_f = None

    for x in xs:
        ok, v = safe_f_eval(f, float(x))
        valid.append(bool(ok))
        if prev_x is not None and prev_ok and ok:
            if (prev_f == 0.0) or (v == 0.0) or (np.sign(prev_f) * np.sign(v) < 0.0):
                has_bracket = True
        prev_x, prev_ok, prev_f = float(x), bool(ok), float(v) if ok else None

    valid_ratio = float(np.mean(valid)) if valid else 0.0
    if (not has_bracket) or (valid_ratio < stable_valid_min):
        return False

    dfx = numeric_derivative(f, float(root))
    if not (np.isfinite(dfx) and abs(dfx) >= stable_dfx_min):
        return False
    return True

def solve_one(f, x0, intervals, solver_mode: str,
              tol_f: float, newton_iters: int, newton_max_step: float,
              local_radius: float, local_scan_n: int, local_max_brackets: int, bisect_iters: int,
              stable_radius: float, stable_scan_n: int, stable_valid_min: float, stable_dfx_min: float):
    solver_mode = str(solver_mode).lower()

    if solver_mode in ("newton", "newton_bisect"):
        okN, xN, fxN = newton_domainaware(
            f, x0, intervals,
            max_iter=newton_iters,
            tol_f=tol_f,
            max_step=newton_max_step,
        )
        if okN and postcheck_root_stable(f, xN, intervals, stable_radius, stable_scan_n, stable_valid_min, stable_dfx_min):
            ok_fx, fx = safe_f_eval(f, xN)
            if ok_fx:
                return True, float(xN), abs(float(fx))

        if solver_mode == "newton":
            return False, float("nan"), float("inf")

    # bisect (local scan)
    brs = find_brackets_by_scan(f, float(x0) - local_radius, float(x0) + local_radius, n=local_scan_n)
    if not brs:
        return False, float("nan"), float("inf")

    best_fx = float("inf")
    best_x = float("nan")
    for (mid_abs, a, b, mid) in brs[:int(local_max_brackets)]:
        okB, xB, fxB = bisection(f, a, b, max_iter=bisect_iters, tol_f=tol_f)
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
        return False, float("nan"), float("inf"), 0, 0

    candidates.sort(key=lambda t: t[0])
    candK = candidates[:int(baseline_topk)]

    tested = 0
    passed = 0
    best_fx = float("inf")
    best_x = float("nan")

    for (mid_abs, a, b, mid) in candK:
        tested += 1
        okB, xB, fxB = bisection(f, a, b, max_iter=bisect_iters, tol_f=tol_f)
        if not okB:
            continue
        if not postcheck_root_stable(f, xB, intervals, stable_radius, stable_scan_n, stable_valid_min, stable_dfx_min):
            continue
        ok_fx, fx = safe_f_eval(f, xB)
        if not ok_fx:
            continue
        passed += 1
        fx_abs = abs(float(fx))
        if fx_abs < best_fx:
            best_fx = fx_abs
            best_x = float(xB)
        if best_fx <= tol_f:
            break

    if np.isfinite(best_x) and np.isfinite(best_fx):
        return True, best_x, best_fx, tested, passed
    return False, float("nan"), float("inf"), tested, passed


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
        if isinstance(n, ast.Expression):
            return visit(n.body)

        if isinstance(n, ast.Constant):
            if isinstance(n.value,(int,float)) and np.isfinite(float(n.value)):
                emit("NUM", float(n.value)); return
            emit("<UNK>",0.0); return

        if isinstance(n, ast.Name):
            emit("x",0.0) if n.id=="x" else emit("<UNK>",0.0)
            return

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
        return (
            torch.from_numpy(ids),
            torch.from_numpy(numvals),
            torch.from_numpy(attn.astype(np.uint8))
        )

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
    cfg = obj["config"]
    K = int(cfg.get("num_candidates", cfg.get("K", 10)))
    max_len = int(cfg.get("max_len", 128))
    d_model = int(cfg.get("d_model", 256))
    nhead = int(cfg.get("nhead", 8))
    num_layers = int(cfg.get("num_layers", 4))
    scale = float(cfg["scale"])
    sanitize_inputs = bool(cfg.get("sanitize_inputs", True))

    model = ASTPrefixTransformerTopK(
        vocab_size=len(VOCAB),
        max_len=max_len,
        num_candidates=K,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    ).to(device)
    model.load_state_dict(obj["model_state"])
    model.eval()
    return model, cfg, scale, sanitize_inputs


# =========================================================
# Backends: anchored / ann(mdpi) / lstm
# =========================================================

def _safe_torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def _load_meanstd_scaler_if_exists(obj):
    scaler = None
    if isinstance(obj, dict):
        if "scaler" in obj and isinstance(obj["scaler"], dict) and ("mean" in obj["scaler"]) and ("std" in obj["scaler"]):
            scaler = obj["scaler"]
        elif "config" in obj and isinstance(obj["config"], dict) and ("scaler" in obj["config"]):
            scaler = obj["config"]["scaler"]
    if scaler is None:
        return None
    mean = np.array(scaler["mean"], dtype=np.float32)
    std  = np.array(scaler["std"], dtype=np.float32)
    std  = np.where(std == 0, 1.0, std).astype(np.float32)
    return (mean, std)

def apply_meanstd(X: np.ndarray, scaler):
    if scaler is None:
        return X
    mean, std = scaler
    return ((X - mean) / std).astype(np.float32)

def _infer_prefix(sd: dict):
    if any(k.startswith("net.") for k in sd.keys()):
        return "net"
    return "net"

def build_mlp_from_state(sd: dict, prefix="net"):
    idxs=[]
    for k in sd.keys():
        m = re.match(rf"^{re.escape(prefix)}\.(\d+)\.weight$", k)
        if m:
            idxs.append(int(m.group(1)))
    idxs = sorted(set(idxs))
    if not idxs:
        raise RuntimeError("Anchored ckpt: net.*.weight not found.")

    layers = []
    for i, li in enumerate(idxs):
        w = sd[f"{prefix}.{li}.weight"]
        if not torch.is_tensor(w):
            w = torch.tensor(w)
        out_dim, in_dim = int(w.shape[0]), int(w.shape[1])
        has_bias = (f"{prefix}.{li}.bias" in sd)
        layers.append((f"L{li}", nn.Linear(in_dim, out_dim, bias=has_bias)))
        if i < len(idxs) - 1:
            layers.append((f"R{li}", nn.ReLU()))
    return nn.Sequential(dict(layers))

def load_backend_anchored(ckpt_path: Path, device: torch.device):
    obj = _safe_torch_load(ckpt_path, map_location=device)
    cfg = obj.get("config", {}) if isinstance(obj, dict) else {}
    scaler = _load_meanstd_scaler_if_exists(obj)

    if isinstance(obj, dict) and ("model_state" in obj) and isinstance(obj["model_state"], dict):
        sd = dict(obj["model_state"])
    elif isinstance(obj, dict):
        sd = dict(obj)
    else:
        raise RuntimeError("Anchored ckpt unsupported format")

    # anchors optional
    anchors = None
    if isinstance(cfg, dict) and "anchors" in cfg:
        try:
            a = cfg["anchors"]
            if torch.is_tensor(a):
                anchors = a.detach().cpu().numpy().astype(np.float32)
            else:
                anchors = np.array(a, dtype=np.float32)
        except Exception:
            anchors = None

    prefix = _infer_prefix(sd)
    model = build_mlp_from_state(sd, prefix=prefix).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, anchors, scaler

def anchored_predict_z(model: nn.Module, X: torch.Tensor, anchors: Optional[np.ndarray]):
    y = model(X)
    if y.dim() == 2 and y.size(1) == 1:
        return y
    if anchors is None:
        raise RuntimeError("Anchored: logits output but anchors missing.")
    a = torch.from_numpy(np.array(anchors, dtype=np.float32)).to(y.device).view(1, -1)
    w = torch.softmax(y, dim=1)
    z = (w * a).sum(dim=1, keepdim=True)
    return z

class ShallowFNN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden, out_dim)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

def _minmax_to_minus1_1_torch(x: torch.Tensor, mn: torch.Tensor, mx: torch.Tensor, eps: float = 1e-12):
    den = torch.clamp(mx - mn, min=eps)
    return (2.0 * (x - mn) / den - 1.0)

def _inv_minmax_from_minus1_1_torch(x_scaled: torch.Tensor, mn: torch.Tensor, mx: torch.Tensor):
    return ((x_scaled + 1.0) * 0.5 * (mx - mn) + mn)

def load_backend_ann_mdpi(ckpt_path: Path, device: torch.device):
    ck = _safe_torch_load(ckpt_path, map_location=device)
    if not isinstance(ck, dict) or "model" not in ck:
        raise RuntimeError("ANN ckpt must be dict with key 'model'")

    in_dim = int(ck["in_dim"])
    out_dim = int(ck["out_dim"])
    hidden = int(ck.get("hidden", 10))

    model = ShallowFNN(in_dim=in_dim, out_dim=out_dim, hidden=hidden).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    scaler_path = ck.get("scaler_json", "")
    if not scaler_path:
        raise RuntimeError("ANN ckpt missing scaler_json(path)")

    spath = Path(scaler_path)
    if not spath.exists():
        alt = ckpt_path.parent / spath.name
        if alt.exists():
            spath = alt
        else:
            raise RuntimeError(f"ANN scaler_json not found: {spath}")

    with open(spath, "r", encoding="utf-8") as f:
        sc = json.load(f)

    x_min = torch.tensor(sc["x_min"], dtype=torch.float32, device=device)
    x_max = torch.tensor(sc["x_max"], dtype=torch.float32, device=device)
    y_min = torch.tensor(sc["y_min"], dtype=torch.float32, device=device)
    y_max = torch.tensor(sc["y_max"], dtype=torch.float32, device=device)
    return model, (x_min, x_max, y_min, y_max)

def ann_predict_z(model: nn.Module, X: torch.Tensor, scalers):
    x_min, x_max, y_min, y_max = scalers
    x_scaled = _minmax_to_minus1_1_torch(X, x_min, x_max)
    with torch.no_grad():
        y_scaled = model(x_scaled)
    y_org = _inv_minmax_from_minus1_1_torch(y_scaled, y_min, y_max)
    return y_org[:, 0:1]

class LSTMRootRegressor(nn.Module):
    def __init__(self, hidden: int = 128, num_layers: int = 2, dropout: float = 0.0):
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
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])

def load_backend_lstm(ckpt_path: Path, device: torch.device):
    ck = _safe_torch_load(ckpt_path, map_location=device)
    if not isinstance(ck, dict) or "model" not in ck:
        raise RuntimeError("LSTM ckpt must be dict with key 'model'")
    args = ck.get("args", {})
    hidden = int(args.get("hidden", 128))
    layers = int(args.get("layers", 2))
    dropout = float(args.get("dropout", 0.0))
    model = LSTMRootRegressor(hidden=hidden, num_layers=layers, dropout=dropout).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model

def lstm_predict_z(model: nn.Module, X: torch.Tensor):
    x_seq = X.unsqueeze(-1)
    with torch.no_grad():
        z = model(x_seq)
    return z


# =========================================================
# Reports: winner / funcid / fail concentration
# =========================================================

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

def report_funcid_winner(
    func_id: np.ndarray,
    expr: np.ndarray,
    methods: List[str],
    ok_by: Dict[str, np.ndarray],
    winner_any: np.ndarray,
    winner_ok: np.ndarray,
    thr: float,
    mode: str = "both",
    expr_k: int = 3,
    topn: int = 50,
):
    fid = np.asarray(func_id)
    uniq = np.unique(fid)
    counts = {u: int((fid == u).sum()) for u in uniq}
    order = sorted(list(uniq), key=lambda u: (-counts[u], int(u)))

    print("\n==================== FUNC_ID WINNER RATIO (with expr) ====================")
    print(f"[thr={thr:.1e}] mode={mode} | expr_k={expr_k} | topn={topn}")
    print("--------------------------------------------------------------------------")

    shown = 0
    for u in order:
        idx = np.where(fid == u)[0]
        n = idx.size
        if n <= 0:
            continue

        # ok rate per method
        ok_parts = []
        for m in methods:
            okm = ok_by[m][idx]
            ok_parts.append(f"{m}:{okm.mean()*100:5.1f}%")

        print(f"func_id={int(u):4d} | n={n:6d} | " + " ".join(ok_parts))

        if mode in ("any","both"):
            c = Counter([str(x) for x in winner_any[idx]])
            denom = max(1, n)
            line = "  winner(any)   : " + " ".join([f"{m}={c.get(m,0)/denom*100:5.1f}%" for m in methods]) + f" none={c.get('none',0)/denom*100:5.1f}%"
            print(line)

        if mode in ("okonly","both"):
            c = Counter([str(x) for x in winner_ok[idx]])
            denom = max(1, n)
            line = "  winner(okonly): " + " ".join([f"{m}={c.get(m,0)/denom*100:5.1f}%" for m in methods]) + f" none={c.get('none',0)/denom*100:5.1f}%"
            print(line)

        # expr examples
        seen = set()
        ex = []
        for gi in idx.tolist():
            s = shorten(expr[gi], 220)
            if s in seen:
                continue
            seen.add(s)
            ex.append(s)
            if len(ex) >= int(expr_k):
                break
        for j, s in enumerate(ex):
            print(f"   expr{j+1}: {s}")

        print("--------------------------------------------------------------------------")
        shown += 1
        if shown >= int(topn):
            break

    print("================== END FUNC_ID WINNER REPORT ==================\n")

def report_fail_concentration(
    func_id: np.ndarray,
    expr: np.ndarray,
    fail_mask: np.ndarray,
    topn: int = 30,
    expr_k: int = 3,
    title: str = "FAIL CONCENTRATION BY FUNC_ID",
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
        if n_fail == 0:
            continue
        rows.append((u, n_fail, n_all, n_fail/max(1,n_all)))

    rows.sort(key=lambda t: (-t[1], -t[3], t[0]))

    print(f"\n==================== {title} ====================")
    print(f"N_fail={int(fail_mask.sum())}/{int(fail_mask.shape[0])} ({fail_mask.mean()*100:.2f}%)")
    print("rank | func_id | n_fail/n_all | fail_rate | expr_examples")
    print("--------------------------------------------------------------------------")

    shown = 0
    for rank, (u, n_fail, n_all, fr) in enumerate(rows, start=1):
        idx_fail = np.where((fid == u) & fail_mask)[0]
        seen = set()
        ex = []
        for gi in idx_fail.tolist():
            s = shorten(expr[gi], 220)
            if s in seen:
                continue
            seen.add(s)
            ex.append(s)
            if len(ex) >= int(expr_k):
                break

        print(f"{rank:4d} | {u:7d} | {n_fail:6d}/{n_all:<6d} | {fr*100:8.2f}%")
        for j, s in enumerate(ex):
            print(f"       expr{j+1}: {s}")
        print("--------------------------------------------------------------------------")

        shown += 1
        if shown >= int(topn):
            break

    print("================== END FAIL CONCENTRATION REPORT ==================\n")


# =========================================================
# Plotting (hist / box only)
# =========================================================

def plot_residual_hist(abs_arr: np.ndarray, out_png: Path, title: str, log10: bool = True):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib not installed. skip plotting.")
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

def plot_funcid_box(abs_arr: np.ndarray, func_id: np.ndarray, out_png: Path, topn: int = 15, log10: bool = True, title: str = ""):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib not installed. skip plotting.")
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(abs_arr, dtype=np.float64)
    fid = np.asarray(func_id)

    uniq = np.unique(fid)
    counts = {u: int((fid == u).sum()) for u in uniq}
    order = sorted(list(uniq), key=lambda u: (-counts[u], int(u)))[:int(topn)]

    data = []
    labels = []
    for u in order:
        m = (fid == u) & np.isfinite(a)
        v = a[m]
        if v.size == 0:
            continue
        if log10:
            v = np.log10(v + 1e-30)
        data.append(v)
        labels.append(f"{int(u)}\n(n={v.size})")

    if not data:
        return

    plt.figure(figsize=(max(12, 0.8*len(data)), 6))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("log10(|f|+eps)" if log10 else "|f|")
    plt.title(title if title else "Residual by func_id (boxplot)")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


# =========================================================
# Main evaluation
# =========================================================

def main():
    import os
    import warnings
    from pathlib import Path
    import numpy as np
    import torch
    from tqdm import tqdm

    from src.path_utils import find_repo_root, resolve_repo_path, resolve_device

    repo = find_repo_root(__file__)

    # 1) config load (repo-root 기준)
    cfg_path = os.environ.get("EVAL_CFG", "configs/eval_k_sweep.yaml")
    cfg_path_p = resolve_repo_path(cfg_path, repo)
    cfg = load_yaml(str(cfg_path_p))

    # 2) outdir (repo-root 기준)
    outdir_str = os.environ.get("OUTDIR", str(_get(cfg, "outdir", "results/eval_k_sweep")))
    outdir = resolve_repo_path(outdir_str, repo)
    outdir.mkdir(parents=True, exist_ok=True)

    # 3) warnings
    suppress_runtime_warnings = bool(_get(cfg, "runtime.suppress_runtime_warnings", False))
    if suppress_runtime_warnings:
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    # 4) device/batch
    device_str = resolve_device(os.environ.get("DEVICE", str(_get(cfg, "device", "auto"))))
    device = torch.device(device_str)
    batch_size = int(_get(cfg, "batch_size", 256))

    # 5) data path (repo-root 기준 상대경로)
    test_npz = str(_get(cfg, "data.test_npz", "data/taylor_test_physchem_v3_allroots_10000.npz"))
    test_path = resolve_repo_path(test_npz, repo)
    if test_path is None or (not test_path.exists()):
        raise FileNotFoundError(f"test_npz not found: {test_path}")

    # 6) ckpt paths (repo-root 기준 상대경로)
    ast_ckpt = resolve_repo_path(str(_get(cfg, "models.ast_ckpt", "")), repo)
    anchored_ckpt = resolve_repo_path(str(_get(cfg, "models.anchored_ckpt", "")), repo)
    ann_ckpt = resolve_repo_path(str(_get(cfg, "models.ann_ckpt", "")), repo)
    lstm_ckpt = resolve_repo_path(str(_get(cfg, "models.lstm_ckpt", "")), repo)

    # 7) K list
    topk_list = parse_csv_list(_get(cfg, "k_sweep.topk_list", "5,10,15,20,25"), cast=int)
    topk_list = [k for k in topk_list if k > 0]
    if not topk_list:
        topk_list = [10]

    # 8) toggles
    thr_winner_okonly   = bool(_get(cfg, "reports.thr_winner_okonly", False))
    report_funcid_winner = bool(_get(cfg, "reports.report_funcid_winner", False))
    report_funcid_mode   = str(_get(cfg, "reports.report_funcid_winner_mode", "both"))
    report_funcid_expr_k = int(_get(cfg, "reports.report_funcid_winner_expr_k", 3))
    report_funcid_topn   = int(_get(cfg, "reports.report_funcid_winner_topn", 50))

    report_fail_funcid = bool(_get(cfg, "reports.report_fail_funcid", False))
    report_fail_mode   = str(_get(cfg, "reports.report_fail_mode", "baseline"))
    report_fail_topn   = int(_get(cfg, "reports.report_fail_funcid_topn", 30))
    report_fail_expr_k = int(_get(cfg, "reports.report_fail_expr_k", 3))

    plot_residual_hist = bool(_get(cfg, "plots.plot_residual_hist", False))
    plot_funcid_box    = bool(_get(cfg, "plots.plot_funcid_box", False))
    plot_topn_funcid   = int(_get(cfg, "plots.plot_topn_funcid", 15))

    # 9) thresholds
    compare_thr = float(_get(cfg, "thresholds.compare_thr", 1e-10))
    thr_sweep = str(_get(cfg, "thresholds.thr_sweep", "1e-6,1e-8,1e-10,1e-12"))
    thr_list = sorted(parse_csv_float_list(thr_sweep), reverse=True) if thr_sweep else [compare_thr]

    # 10) solver config
    solver_mode = str(_get(cfg, "solver.solver_mode", "newton_bisect"))
    max_tries = int(_get(cfg, "solver.max_tries", 10))
    stop_after_first_success = bool(_get(cfg, "solver.stop_after_first_success", False))
    proxy = str(_get(cfg, "solver.proxy", "poly"))

    tol_f = float(_get(cfg, "solver.tol_f", 1e-10))
    newton_iters = int(_get(cfg, "solver.newton_iters", 30))
    newton_max_step = float(_get(cfg, "solver.newton_max_step", 2.0))
    bisect_iters = int(_get(cfg, "solver.bisect_iters", 60))

    local_radius = float(_get(cfg, "solver.local_radius", 1.0))
    local_scan_n = int(_get(cfg, "solver.local_scan_n", 101))
    local_max_brackets = int(_get(cfg, "solver.local_max_brackets", 10))

    stable_radius = float(_get(cfg, "solver.stable_radius", 1.0))
    stable_scan_n = int(_get(cfg, "solver.stable_scan_n", 80))
    stable_valid_min = float(_get(cfg, "solver.stable_valid_min", 0.7))
    stable_dfx_min = float(_get(cfg, "solver.stable_dfx_min", 1e-10))

    # baseline config
    baseline_mode = str(_get(cfg, "baseline.mode", "all"))
    baseline_topk = int(_get(cfg, "baseline.topk", 10))
    base_scan_xmin = float(_get(cfg, "baseline.scan_xmin", -20.0))
    base_scan_xmax = float(_get(cfg, "baseline.scan_xmax", +20.0))
    base_scan_n = int(_get(cfg, "baseline.scan_n", 250))
    anchored_fb = str(_get(cfg, "baseline.anchored_fb", "auto"))
    if anchored_fb not in ("auto", "on", "off"):
        anchored_fb = "auto"

    # log
    print(f"[CFG] repo_root={repo}")
    print(f"[CFG] cfg={cfg_path_p}")
    print(f"[CFG] outdir={outdir.resolve()}")
    print(f"[CFG] device={device} batch={batch_size} K_list={topk_list}")
    print(f"[CFG] test_npz={test_path}")
    print(f"[CFG] compare_thr={compare_thr:.1e} thr_list={thr_list}")
    print(f"[CFG] solver_mode={solver_mode} baseline_mode={baseline_mode} anchored_fb={anchored_fb}")

    # load npz
    data = np.load(test_path, allow_pickle=True)
    coeffs = data["coeffs"].astype(np.float32)

    if "func_expr" in data:
        expr = data["func_expr"]
    elif "expr_str" in data:
        expr = data["expr_str"]
    else:
        raise KeyError("NPZ must contain 'func_expr' or 'expr_str'.")

    N, D = coeffs.shape
    func_id = data["func_id"] if "func_id" in data and data["func_id"].shape[0] == N else None
    print(f"[NPZ] N={N} degree={D-1} has_func_id={func_id is not None}")

    # load AST
    if ast_ckpt is None or (not ast_ckpt.exists()):
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
            c = (ast_scale * torch.sinh(y)).cpu().numpy().astype(np.float64)
        centers_all[idx0:idx0+B, :] = c
        idx0 += B

    # load backends
    backends = []
    if anchored_ckpt is not None and anchored_ckpt.exists():
        m, anchors, scaler = load_backend_anchored(anchored_ckpt, device=device)
        backends.append(("anchored", {"type":"anchored", "model":m, "anchors":anchors, "scaler":scaler}))
        print(f"[anchored] {anchored_ckpt}")

    if ann_ckpt is not None and ann_ckpt.exists():
        m, scalers = load_backend_ann_mdpi(ann_ckpt, device=device)
        backends.append(("ann", {"type":"ann", "model":m, "scalers":scalers}))
        print(f"[ann] {ann_ckpt}")

    if lstm_ckpt is not None and lstm_ckpt.exists():
        m = load_backend_lstm(lstm_ckpt, device=device)
        backends.append(("lstm", {"type":"lstm", "model":m}))
        print(f"[lstm] {lstm_ckpt}")

    if not backends:
        raise RuntimeError("No backend checkpoints found. Set models.*_ckpt in YAML to existing files.")

    # anchored_fb enable
    if anchored_fb == "off":
        enable_anchored_fb = False
    elif anchored_fb == "on":
        enable_anchored_fb = True
        if baseline_mode == "none":
            baseline_mode = "on_demand"
    else:
        enable_anchored_fb = (baseline_mode != "none")

    # ---- 여기부터는 기존 evaluate 로직 그대로 사용 ----
    # (네 evaluate_k_sweep.py 안에 이미 있는 run_eval_for_K / winner / plot 함수들을 호출하도록 이어붙이면 됨)
    #
    # 즉, main에서 cfg/paths/device/npz/ckpt만 "repo-root 상대경로 resolve"로 고쳐주면
    # 나머지 평가지표/시각화/리포트 로직은 그대로 유지 가능.

    # 아래는 예시: 너 파일 구조에 맞춰 run_eval_for_K를 이미 만들어 둔 버전이면 그걸 호출
    run_all_from_prepared(
        outdir=outdir,
        data=data,
        coeffs=coeffs,
        expr=expr,
        func_id=func_id,
        centers_all=centers_all,
        backends=backends,
        device=device,
        topk_list=topk_list,
        compare_thr=compare_thr,
        thr_list=thr_list,
        thr_winner_okonly=thr_winner_okonly,
        report_funcid_winner=report_funcid_winner,
        report_funcid_mode=report_funcid_mode,
        report_funcid_expr_k=report_funcid_expr_k,
        report_funcid_topn=report_funcid_topn,
        report_fail_funcid=report_fail_funcid,
        report_fail_mode=report_fail_mode,
        report_fail_topn=report_fail_topn,
        report_fail_expr_k=report_fail_expr_k,
        plot_residual_hist=plot_residual_hist,
        plot_funcid_box=plot_funcid_box,
        plot_topn_funcid=plot_topn_funcid,
        solver_mode=solver_mode,
        max_tries=max_tries,
        stop_after_first_success=stop_after_first_success,
        proxy=proxy,
        tol_f=tol_f,
        newton_iters=newton_iters,
        newton_max_step=newton_max_step,
        bisect_iters=bisect_iters,
        local_radius=local_radius,
        local_scan_n=local_scan_n,
        local_max_brackets=local_max_brackets,
        stable_radius=stable_radius,
        stable_scan_n=stable_scan_n,
        stable_valid_min=stable_valid_min,
        stable_dfx_min=stable_dfx_min,
        baseline_mode=baseline_mode,
        baseline_topk=baseline_topk,
        base_scan_xmin=base_scan_xmin,
        base_scan_xmax=base_scan_xmax,
        base_scan_n=base_scan_n,
        enable_anchored_fb=enable_anchored_fb,
    )


if __name__ == "__main__":
    main()
