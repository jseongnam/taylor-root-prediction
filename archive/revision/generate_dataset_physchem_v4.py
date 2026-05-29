#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/data/generate_dataset_physchem_v4.py

✅ YAML(configs/dataset_physchem_v4_deg25.yaml) 기반 데이터셋 생성기
- argparse 없음
- 기본 실행: python scripts/data/generate_dataset_physchem_v4.py
- 환경변수 override:
    DATASET_CFG=path/to/config.yaml
    OUT_DIR=override/output/dir
"""

from __future__ import annotations

import os
import json
import math
from pathlib import Path
from itertools import product
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

try:
    import yaml  # PyYAML
except Exception as e:
    raise ImportError("PyYAML이 필요합니다. `pip install pyyaml`") from e


# ======================================
#  Config loader
# ======================================
def _get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_cfg(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ======================================
#  문자열 포맷 유틸 (expr_str 길이 폭주 방지)
# ======================================
def fmt_float(x: float, sig: int = 6) -> str:
    return f"{float(x):.{sig}g}"


def poly_to_string(coeffs_0_to_d, var="x", sig: int = 6) -> str:
    terms = []
    for k, c in enumerate(coeffs_0_to_d):
        c = float(c)
        if abs(c) < 1e-12:
            continue
        cs = fmt_float(c, sig=sig)
        if k == 0:
            terms.append(f"{cs}")
        elif k == 1:
            terms.append(f"{cs}*{var}")
        else:
            terms.append(f"{cs}*{var}**{k}")
    if not terms:
        return "0"
    return "(" + " + ".join(terms) + ")"


# ======================================
#  Taylor 계수 유틸
# ======================================
def taylor_sin(b, degree):
    coeffs = np.zeros(degree + 1, dtype=float)
    max_n = (degree - 1) // 2
    for n in range(max_n + 1):
        k = 2 * n + 1
        if k > degree:
            break
        coeffs[k] = ((-1) ** n) * (b ** (2 * n + 1)) / math.factorial(2 * n + 1)
    return coeffs

def taylor_cos(b, degree):
    coeffs = np.zeros(degree + 1, dtype=float)
    max_n = degree // 2
    for n in range(max_n + 1):
        k = 2 * n
        if k > degree:
            break
        coeffs[k] = ((-1) ** n) * (b ** (2 * n)) / math.factorial(2 * n)
    return coeffs

def taylor_exp(b, degree):
    coeffs = np.zeros(degree + 1, dtype=float)
    for n in range(degree + 1):
        coeffs[n] = (b ** n) / math.factorial(n)
    return coeffs

def taylor_sinh(b, degree):
    coeffs = np.zeros(degree + 1, dtype=float)
    max_n = (degree - 1) // 2
    for n in range(max_n + 1):
        k = 2 * n + 1
        if k > degree:
            break
        coeffs[k] = (b ** (2 * n + 1)) / math.factorial(2 * n + 1)
    return coeffs

def taylor_cosh(b, degree):
    coeffs = np.zeros(degree + 1, dtype=float)
    max_n = degree // 2
    for n in range(max_n + 1):
        k = 2 * n
        if k > degree:
            break
        coeffs[k] = (b ** (2 * n)) / math.factorial(2 * n)
    return coeffs

def taylor_log1p(b, degree):
    coeffs = np.zeros(degree + 1, dtype=float)
    for n in range(1, degree + 1):
        coeffs[n] = ((-1) ** (n + 1)) * (b ** n) / n
    return coeffs

def series_divide(num, den, degree):
    q = np.zeros(degree + 1, dtype=float)
    if abs(den[0]) < 1e-12:
        return num.copy()
    for k in range(degree + 1):
        acc = num[k]
        for i in range(k):
            acc -= q[i] * den[k - i]
        q[k] = acc / den[0]
    return q

def taylor_tanh(b, degree):
    num = taylor_sinh(b, degree)
    den = taylor_cosh(b, degree)
    return series_divide(num, den, degree)

def taylor_inv1p(b, degree):
    coeffs = np.zeros(degree + 1, dtype=float)
    for n in range(degree + 1):
        coeffs[n] = ((-1) ** n) * (b ** n)
    return coeffs

def taylor_sqrt1p(b, degree):
    coeffs = np.zeros(degree + 1, dtype=float)
    alpha = 0.5
    c = 1.0
    coeffs[0] = 1.0
    for n in range(1, degree + 1):
        c *= (alpha - (n - 1)) / n
        coeffs[n] = c * (b ** n)
    return coeffs


# ======================================
#  다항식 연산 & 합성
# ======================================
def poly_mul_trunc(a: np.ndarray, b: np.ndarray, degree: int) -> np.ndarray:
    res = np.zeros(degree + 1, dtype=float)
    na = min(len(a) - 1, degree)
    nb = min(len(b) - 1, degree)
    for i in range(na + 1):
        for j in range(nb + 1):
            k = i + j
            if k > degree:
                break
            res[k] += a[i] * b[j]
    return res

def series_compose(F: np.ndarray, G: np.ndarray, degree: int) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    G = np.asarray(G, dtype=float)
    res = np.zeros(degree + 1, dtype=float)

    power = np.zeros(degree + 1, dtype=float)
    power[0] = 1.0

    max_n = min(degree, len(F) - 1)
    for n in range(max_n + 1):
        res[:degree + 1] += F[n] * power[:degree + 1]
        power = poly_mul_trunc(power, G, degree)

    return res


# ======================================
#  다항식 값/미분/근 찾기
# ======================================
def poly_eval(coeffs, x):
    res = 0.0
    for c in reversed(coeffs):
        res = res * x + c
    return res

def poly_derivative(coeffs):
    n = len(coeffs) - 1
    if n <= 0:
        return np.array([0.0], dtype=float)
    der = np.zeros(n, dtype=float)
    for k in range(1, n + 1):
        der[k - 1] = k * coeffs[k]
    return der

def find_all_roots_poly(coeffs, x_min=-1.0, x_max=1.0, num_intervals=800, tol=1e-6, max_iter=80):
    h = (x_max - x_min) / num_intervals
    roots = []

    x_left = x_min
    f_left = poly_eval(coeffs, x_left)

    def append_root(r):
        if not roots:
            roots.append(r)
            return
        if abs(r - roots[-1]) < 1e-5:
            return
        roots.append(r)

    for _ in range(num_intervals):
        x_right = x_left + h
        f_right = poly_eval(coeffs, x_right)
        root = None

        if f_left == 0.0:
            root = x_left
        elif f_left * f_right < 0.0 or f_right == 0.0:
            a, b = x_left, x_right
            fa, fb = f_left, f_right
            for _ in range(max_iter):
                m = 0.5 * (a + b)
                fm = poly_eval(coeffs, m)
                if abs(fm) < tol or (b - a) * 0.5 < tol:
                    root = m
                    break
                if fa * fm <= 0.0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
            else:
                root = 0.5 * (a + b)

        if root is not None:
            append_root(root)

        x_left = x_right
        f_left = f_right

    roots.sort()
    return roots

def find_root_closest_to_zero(coeffs, x_min=-1.0, x_max=1.0, num_intervals=800, tol=1e-6, max_iter=80):
    roots = find_all_roots_poly(coeffs, x_min=x_min, x_max=x_max,
                                num_intervals=num_intervals, tol=tol, max_iter=max_iter)
    if not roots:
        return None
    return min(roots, key=lambda r: abs(r))

def newton_refine_root(coeffs, x0, x_min=-1.0, x_max=1.0, max_iter=7, tol=1e-12):
    der = poly_derivative(coeffs)
    x = float(x0)

    for _ in range(max_iter):
        fx = poly_eval(coeffs, x)
        dfx = poly_eval(der, x)
        if abs(dfx) < 1e-12:
            break
        step = fx / dfx
        x_new = x - step
        if x_new < x_min:
            x_new = x_min
        elif x_new > x_max:
            x_new = x_max
        if abs(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x


# ======================================
#  합성용 파라미터 샘플링 & 계수 + expr_str 생성
# ======================================
def sample_ab_for_outer(term: str, rng):
    if term in ("ln", "log", "inv", "sqrt1p"):
        a = rng.uniform(-2.0, 2.0)
        if abs(a) < 0.05:
            a = math.copysign(0.05, a if a != 0 else 0.5)
    else:
        a = rng.uniform(-4.0, 4.0)
        if abs(a) < 0.1:
            a = math.copysign(0.1, a if a != 0 else 0.5)

    if term in ("sin", "cos"):
        b = rng.uniform(-2.0, 2.0)
        if abs(b) < 0.2:
            b = math.copysign(0.2, b if b != 0 else 0.5)
    elif term in ("exp", "sinh", "cosh", "tanh"):
        b = rng.uniform(-1.2, 1.2)
        if abs(b) < 0.08:
            b = math.copysign(0.08, b if b != 0 else 0.5)
    elif term in ("ln", "log", "inv", "sqrt1p"):
        b = rng.uniform(-0.35, 0.35)
        if abs(b) < 0.03:
            b = math.copysign(0.03, b if b != 0 else 0.5)
    else:
        raise ValueError(f"Unknown term: {term}")
    return a, b

def outer_taylor_in_z_with_params(term: str, degree: int, rng):
    a, b = sample_ab_for_outer(term, rng)

    if term == "sin":
        base = taylor_sin(b, degree)
    elif term == "cos":
        base = taylor_cos(b, degree)
    elif term == "exp":
        base = taylor_exp(b, degree)
    elif term == "sinh":
        base = taylor_sinh(b, degree)
    elif term == "cosh":
        base = taylor_cosh(b, degree)
    elif term == "tanh":
        base = taylor_tanh(b, degree)
    elif term == "ln":
        base = taylor_log1p(b, degree)
    elif term == "log":
        base = (1.0 / math.log(10.0)) * taylor_log1p(b, degree)
    elif term == "inv":
        base = taylor_inv1p(b, degree)
    elif term == "sqrt1p":
        base = taylor_sqrt1p(b, degree)
    else:
        raise ValueError(f"Unknown outer term: {term}")

    return a * base, (a, b)

def apply_outer_expr_string(term: str, a: float, b: float, inner_str: str, sig: int = 6) -> str:
    as_ = fmt_float(a, sig=sig)
    bs_ = fmt_float(b, sig=sig)

    if term == "ln":
        return f"({as_})*ln(1 + ({bs_})*({inner_str}))"
    if term == "log":
        return f"({as_})*log10(1 + ({bs_})*({inner_str}))"
    if term == "inv":
        return f"({as_})/(1 + ({bs_})*({inner_str}))"
    if term == "sqrt1p":
        return f"({as_})*sqrt(1 + ({bs_})*({inner_str}))"

    return f"({as_})*{term}(({bs_})*({inner_str}))"


# ======================================
#  term 문자열 파싱 & 계수 생성 (+ expr_str 생성)
# ======================================
def parse_outer_inner(expr: str):
    expr = expr.strip()
    if "(" not in expr:
        return expr, None
    idx = expr.find("(")
    outer = expr[:idx]
    if not expr.endswith(")"):
        raise ValueError(f"Invalid composite expr: {expr}")
    inner = expr[idx + 1:-1]
    return outer, inner

def build_poly_base_with_expr(expr: str, degree: int, rng, TRANSC_TERMS, poly_degs: List[int], float_sig: int):
    expr = expr.strip()

    if expr.startswith("poly"):
        if expr == "poly_rand":
            d = int(rng.integers(1, degree + 1))
        else:
            d = int(expr[4:])
            d = max(1, min(d, degree))

        coefs = rng.uniform(-2.0, 2.0, size=(d + 1,))
        poly = np.zeros(degree + 1, dtype=float)
        for k in range(d + 1):
            poly[k] += coefs[k]

        expr_str = poly_to_string(coefs, var="x", sig=float_sig)
        return poly, expr_str

    if expr in TRANSC_TERMS:
        inner = np.zeros(degree + 1, dtype=float)
        inner[1] = 1.0  # x

        Fz, (a, b) = outer_taylor_in_z_with_params(expr, degree, rng)
        poly = series_compose(Fz, inner, degree)

        expr_str = apply_outer_expr_string(expr, a, b, "x", sig=float_sig)
        return poly, expr_str

    raise ValueError(f"Unknown base expr: {expr}")

def build_term_poly_expr_with_expr(expr: str, degree: int, rng, TRANSC_TERMS, poly_degs: List[int], float_sig: int):
    outer, inner = parse_outer_inner(expr)

    if inner is None:
        return build_poly_base_with_expr(outer, degree, rng, TRANSC_TERMS, poly_degs, float_sig)

    inner_poly, inner_str = build_term_poly_expr_with_expr(inner, degree, rng, TRANSC_TERMS, poly_degs, float_sig)

    if outer not in TRANSC_TERMS:
        raise ValueError(f"Outer must be transcendental term, got: {outer}")

    Fz, (a, b) = outer_taylor_in_z_with_params(outer, degree, rng)
    poly = series_compose(Fz, inner_poly, degree)

    expr_str = apply_outer_expr_string(outer, a, b, inner_str, sig=float_sig)
    return poly, expr_str

def build_poly_from_template(template, degree, rng, TRANSC_TERMS, poly_degs: List[int], min_norm: float, float_sig: int):
    poly = np.zeros(degree + 1, dtype=float)
    expr_terms = []

    for term_expr in template:
        term_poly, term_str = build_term_poly_expr_with_expr(term_expr, degree, rng, TRANSC_TERMS, poly_degs, float_sig)
        poly += term_poly
        expr_terms.append(term_str)

    max_abs = float(np.max(np.abs(poly)))
    if (not np.isfinite(max_abs)) or (max_abs < min_norm):
        return None, None, None, None

    poly_normed = poly / max_abs

    template_str = " + ".join(template)
    expr_sum_str = " + ".join([f"({s})" for s in expr_terms])

    return poly_normed, max_abs, template_str, expr_sum_str


# ======================================
#  템플릿 생성
# ======================================
def build_atomic_exprs(TRANSC_TERMS, poly_degs: List[int], max_depth: int = 2):
    poly_terms = [f"poly{d}" for d in poly_degs] + ["poly_rand"]

    depth_exprs = {}
    depth_exprs[1] = set(list(TRANSC_TERMS) + poly_terms)

    for depth in range(2, max_depth + 1):
        cur = set()
        for outer in TRANSC_TERMS:
            for inner in depth_exprs[depth - 1]:
                cur.add(f"{outer}({inner})")
        depth_exprs[depth] = cur

    all_exprs = set()
    for d in range(1, max_depth + 1):
        all_exprs |= depth_exprs[d]

    exprs_sorted = sorted(all_exprs)
    print(f"[INFO] #atomic term exprs (<= depth {max_depth}) = {len(exprs_sorted)}")
    return exprs_sorted

def build_templates(TRANSC_TERMS, poly_degs: List[int], target_num: int = 5000, max_terms: int = 3, max_depth: int = 2):
    atoms = build_atomic_exprs(TRANSC_TERMS, poly_degs, max_depth=max_depth)
    templates = []

    for a in atoms:
        templates.append((a,))
        if len(templates) >= target_num:
            break

    if len(templates) < target_num:
        for L in range(2, max_terms + 1):
            for combo in product(atoms, repeat=L):
                templates.append(combo)
                if len(templates) >= target_num:
                    break
            if len(templates) >= target_num:
                break

    print(f"[INFO] Generated {len(templates)} templates (target={target_num}, max_terms={max_terms}, max_depth={max_depth})")
    return templates


# ======================================
#  출력 범위 필터
# ======================================
def poly_value_filter(poly_coeffs, x_min, x_max, grid_n, y_abs_max, y_ptp_min):
    xs = np.linspace(x_min, x_max, grid_n, dtype=float)
    ys = np.array([poly_eval(poly_coeffs, float(x)) for x in xs], dtype=float)

    if not np.all(np.isfinite(ys)):
        return False
    if float(np.max(np.abs(ys))) > y_abs_max:
        return False
    if float(np.ptp(ys)) < y_ptp_min:
        return False
    return True


# ======================================
#  데이터 생성
# ======================================
def generate_dataset(
    num_samples: int,
    degree: int,
    seed: int,
    templates,
    TRANSC_TERMS,
    poly_degs: List[int],
    min_poly_norm: float,
    float_sig: int,
    root_range: float = 1.0,
    num_intervals: int = 800,
    max_retry_factor: int = 80,
    save_expr_str: bool = False,
    y_abs_max: float = 50.0,
    y_ptp_min: float = 1e-3,
    y_grid_n: int = 201,
    max_roots_keep: int = 8,
):
    rng = np.random.default_rng(seed)

    coeffs_list = []
    r0_list = []
    r1_list = []
    r2_list = []
    func_id_list = []

    template_str_list = []
    expr_str_list = []
    norm_scale_list = []

    root_count_list = []
    roots_pad_list = []

    num_templates = len(templates)
    print(f"[INFO] Using {num_templates} templates")

    max_tries = num_samples * max_retry_factor
    tries = 0

    skipped_tiny = 0
    skipped_no_root = 0
    skipped_filter_y = 0
    multi_root_count = 0

    x_min = -root_range
    x_max = root_range

    while len(coeffs_list) < num_samples and tries < max_tries:
        tries += 1

        fid = int(rng.integers(0, num_templates))
        template = templates[fid]

        poly_normed, norm_scale, template_str, expr_sum_str = build_poly_from_template(
            template, degree, rng, TRANSC_TERMS, poly_degs, min_poly_norm, float_sig
        )

        if poly_normed is None:
            skipped_tiny += 1
            continue

        if not poly_value_filter(poly_normed, x_min, x_max, y_grid_n, y_abs_max, y_ptp_min):
            skipped_filter_y += 1
            continue

        roots = find_all_roots_poly(
            poly_normed,
            x_min=x_min, x_max=x_max,
            num_intervals=num_intervals, tol=1e-6, max_iter=80
        )
        if not roots:
            skipped_no_root += 1
            continue

        if len(roots) > 1:
            multi_root_count += 1

        r0_init = min(roots, key=lambda r: abs(r))
        r0_refined = newton_refine_root(poly_normed, r0_init, x_min=x_min, x_max=x_max, max_iter=7, tol=1e-12)

        c1 = poly_derivative(poly_normed)
        c2 = poly_derivative(c1)
        r1 = find_root_closest_to_zero(c1, x_min=x_min, x_max=x_max, num_intervals=num_intervals, tol=1e-6, max_iter=80)
        r2 = find_root_closest_to_zero(c2, x_min=x_min, x_max=x_max, num_intervals=num_intervals, tol=1e-6, max_iter=80)
        if r1 is None or r2 is None:
            skipped_no_root += 1
            continue

        roots_sorted = sorted(roots)
        root_count = len(roots_sorted)
        roots_pad = np.full((max_roots_keep,), np.nan, dtype=np.float32)
        take = min(root_count, max_roots_keep)
        roots_pad[:take] = np.array(roots_sorted[:take], dtype=np.float32)

        coeffs_list.append(poly_normed.astype(np.float32))
        r0_list.append([float(r0_refined)])
        r1_list.append([float(r1)])
        r2_list.append([float(r2)])
        func_id_list.append(fid)

        root_count_list.append(root_count)
        roots_pad_list.append(roots_pad)

        template_str_list.append(template_str)
        norm_scale_list.append(float(norm_scale))

        if save_expr_str:
            expr_scaled = f"({expr_sum_str}) / ({fmt_float(norm_scale, sig=float_sig)})"
            expr_str_list.append(expr_scaled)

        if len(coeffs_list) % 1000 == 0:
            print(f"[INFO] collected {len(coeffs_list)} / {num_samples}  (tries={tries})")

    if len(coeffs_list) < num_samples:
        print(f"[WARN] 목표 {num_samples}개 중 {len(coeffs_list)}개만 생성되었습니다. (max_tries={max_tries})")

    print(f"[STATS] skipped_tiny_or_nonfinite = {skipped_tiny}")
    print(f"[STATS] skipped_no_root           = {skipped_no_root}")
    print(f"[STATS] skipped_filter_y          = {skipped_filter_y}")
    print(f"[STATS] multi_root_samples        = {multi_root_count}  (accepted)")

    coeffs_arr = np.stack(coeffs_list, axis=0)
    r0_arr = np.array(r0_list, dtype=np.float32)
    r1_arr = np.array(r1_list, dtype=np.float32)
    r2_arr = np.array(r2_list, dtype=np.float32)
    fid_arr = np.array(func_id_list, dtype=np.int32)

    root_count_arr = np.array(root_count_list, dtype=np.int32)
    roots_pad_arr = np.stack(roots_pad_list, axis=0).astype(np.float32)

    template_str_arr = np.array(template_str_list, dtype=f"<U{max(1, max(len(s) for s in template_str_list))}")
    norm_scale_arr = np.array(norm_scale_list, dtype=np.float32)

    if save_expr_str and expr_str_list:
        expr_str_arr = np.array(expr_str_list, dtype=f"<U{max(1, max(len(s) for s in expr_str_list))}")
    else:
        expr_str_arr = None

    return (coeffs_arr, r0_arr, r1_arr, r2_arr, fid_arr,
            template_str_arr, expr_str_arr, norm_scale_arr,
            root_count_arr, roots_pad_arr)


def save_splits(out_dir, degree,
                coeffs, r0, r1, r2, fid,
                template_str, expr_str, norm_scale,
                root_count, roots,
                train_ratio=0.8, val_ratio=0.1, seed: int = 1234):
    N = coeffs.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)

    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train + n_val]
    idx_test = idx[n_train + n_val:]

    splits = {"train": idx_train, "val": idx_val, "test": idx_test}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_idx in splits.items():
        path = out_dir / f"taylor_deg{degree}_{split_name}.npz"
        payload = dict(
            coeffs=coeffs[split_idx],
            root0=r0[split_idx],
            root1=r1[split_idx],
            root2=r2[split_idx],
            func_id=fid[split_idx],
            degree=np.full((split_idx.shape[0],), degree, dtype=np.int32),
            template_str=template_str[split_idx],
            norm_scale=norm_scale[split_idx],
            root_count=root_count[split_idx],
            roots=roots[split_idx],
        )
        if expr_str is not None:
            payload["expr_str"] = expr_str[split_idx]

        np.savez_compressed(path, **payload)
        print(f"[SAVE] {split_name}: {path}  (N={split_idx.shape[0]})")


# ======================================
#  main (no argparse)
# ======================================
def main():
    cfg_path = os.environ.get("DATASET_CFG", "configs/dataset_physchem_v4_deg25.yaml")
    cfg = load_cfg(cfg_path)

    degree = int(_get(cfg, "dataset.degree", 25))
    n_total = int(_get(cfg, "dataset.n_total", 1000000))
    seed = int(_get(cfg, "dataset.seed", 42))
    out_dir = str(_get(cfg, "dataset.out_dir", "./taylor_data_physchem_v4_deg25"))

    # 환경변수로 out_dir override 허용
    out_dir = os.environ.get("OUT_DIR", out_dir)

    n_templates = int(_get(cfg, "templates.n_templates", 5000))
    max_terms = int(_get(cfg, "templates.max_terms", 3))
    max_depth = int(_get(cfg, "templates.max_depth", 2))

    root_range = float(_get(cfg, "roots.root_range", 1.0))
    num_intervals = int(_get(cfg, "roots.num_intervals", 800))
    max_roots_keep = int(_get(cfg, "roots.max_roots_keep", 8))

    save_expr_str = bool(_get(cfg, "strings.save_expr_str", True))
    float_sig = int(_get(cfg, "strings.float_sig", 6))

    y_abs_max = float(_get(cfg, "filters.y_abs_max", 50.0))
    y_ptp_min = float(_get(cfg, "filters.y_ptp_min", 1e-3))
    y_grid_n = int(_get(cfg, "filters.y_grid_n", 201))

    include_tan = bool(_get(cfg, "terms.include_tan", False))
    base_transc_terms = list(_get(cfg, "terms.base_transc_terms", []))

    poly_degs = list(_get(cfg, "poly.poly_degs", [1,2,3,4,5,7,10]))
    min_poly_norm = float(_get(cfg, "poly.min_poly_norm", 1e-8))

    max_retry_factor = int(_get(cfg, "runtime.max_retry_factor", 80))

    TRANSC_TERMS = list(base_transc_terms)
    if include_tan:
        TRANSC_TERMS = TRANSC_TERMS + ["tan"]

    print(f"[INFO] cfg_path={cfg_path}")
    print(f"[INFO] degree={degree}, n_total={n_total}, seed={seed}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] n_templates={n_templates}, max_terms={max_terms}, max_depth={max_depth}")
    print(f"[INFO] root_range={root_range} → [{-root_range}, {root_range}]")
    print(f"[INFO] TRANSC_TERMS={TRANSC_TERMS}")
    print(f"[INFO] poly_degs={poly_degs}, min_poly_norm={min_poly_norm}")
    print(f"[INFO] (filter) y_abs_max={y_abs_max}, y_ptp_min={y_ptp_min}, y_grid_n={y_grid_n}")
    print(f"[INFO] max_roots_keep={max_roots_keep}")
    print(f"[INFO] save_expr_str={save_expr_str}, float_sig={float_sig}")
    print(f"[INFO] max_retry_factor={max_retry_factor}")
    print("[INFO] 각 샘플은 최종적으로 max |coeff| = 1 로 정규화됩니다.\n")

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # templates
    templates = build_templates(
        TRANSC_TERMS=TRANSC_TERMS,
        poly_degs=poly_degs,
        target_num=n_templates,
        max_terms=max_terms,
        max_depth=max_depth,
    )

    templates_path = out_dir_p / "templates.json"
    with open(templates_path, "w", encoding="utf-8") as f:
        json.dump([list(t) for t in templates], f, ensure_ascii=False, indent=2)
    print(f"[SAVE] templates list -> {templates_path}")

    meta_path = out_dir_p / "meta.json"
    meta = {
        "degree": degree,
        "n_total": n_total,
        "seed": seed,
        "n_templates": n_templates,
        "max_terms": max_terms,
        "max_depth": max_depth,
        "root_range": root_range,
        "num_intervals": num_intervals,
        "min_poly_norm": min_poly_norm,
        "save_expr_str": save_expr_str,
        "transc_terms": TRANSC_TERMS,
        "poly_degs": poly_degs,
        "filter": {
            "y_abs_max": y_abs_max,
            "y_ptp_min": y_ptp_min,
            "y_grid_n": y_grid_n,
        },
        "max_roots_keep": max_roots_keep,
        "max_retry_factor": max_retry_factor,
        "note": "coeffs는 template 기반 원함수의 Maclaurin truncation (degree)이며, norm_scale로 샘플별 max|coeff_raw|로 정규화됨. roots는 스캔+이분법 기반 실근(부호변화)이며 중근은 누락될 수 있음."
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] meta -> {meta_path}")

    # generate
    (coeffs, r0, r1, r2, fid,
     template_str, expr_str, norm_scale,
     root_count, roots_pad) = generate_dataset(
        num_samples=n_total,
        degree=degree,
        seed=seed,
        templates=templates,
        TRANSC_TERMS=TRANSC_TERMS,
        poly_degs=poly_degs,
        min_poly_norm=min_poly_norm,
        float_sig=float_sig,
        root_range=root_range,
        num_intervals=num_intervals,
        max_retry_factor=max_retry_factor,
        save_expr_str=save_expr_str,
        y_abs_max=y_abs_max,
        y_ptp_min=y_ptp_min,
        y_grid_n=y_grid_n,
        max_roots_keep=max_roots_keep,
    )

    print("\n[CHECK] 생성된 coeffs 통계:")
    print("  shape           =", coeffs.shape)
    print("  max |coeff|     =", float(np.max(np.abs(coeffs))))
    print("  mean |coeff|    =", float(np.mean(np.abs(coeffs))))
    print("  mean root_count =", float(np.mean(root_count)))

    # save splits
    save_splits(
        out_dir=out_dir,
        degree=degree,
        coeffs=coeffs,
        r0=r0,
        r1=r1,
        r2=r2,
        fid=fid,
        template_str=template_str,
        expr_str=expr_str,
        norm_scale=norm_scale,
        root_count=root_count,
        roots=roots_pad,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=seed,
    )

    print("[DONE] dataset generation finished (YAML-based, no-arg).")


if __name__ == "__main__":
    main()
