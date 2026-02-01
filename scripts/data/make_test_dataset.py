#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_test_dataset.py

Physics/Chemistry test dataset generator (v3_allroots)
- Multi-domains extraction (domain scan on TRUE expression)
- Labels:
  - root0_all/root1_all: all real roots inside domains (NaN padded)
  - root0/root1: compatibility single root = nearest-to-zero from *_all

This script is YAML-driven (no argparse).
All paths are resolved relative to the repository root so reviewers can run after git clone.

Default:
  CFG_PATH=configs/make_test_dataset.yaml
Run:
  PYTHONPATH=. python data_generation/make_test_dataset.py
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

try:
    import yaml
except Exception as e:
    yaml = None


# ============================================================
# Repo root helpers
# ============================================================

def find_repo_root(start: Path) -> Path:
    """
    Walk upward from `start` to find repo root by checking typical markers.
    """
    cur = start.resolve()
    for _ in range(20):
        if (cur / "configs").is_dir() and (cur / "models").is_dir():
            return cur
        if (cur / ".git").exists():
            # in case configs/models are not checked, still accept .git as root
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # fallback: use start's parent
    return start.resolve()


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is not installed. Please install with: pip install pyyaml"
        )
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid YAML (must be dict at top-level): {path}")
    return obj


def get(cfg: Dict[str, Any], key: str, default=None):
    """
    Dot-access: get(cfg, "dataset.degree")
    """
    cur = cfg
    parts = key.split(".")
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


# ============================================================
#  Basic Taylor / series utils
# ============================================================

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
    # ln(1 + b x) = sum_{n>=1} (-1)^{n+1} (b x)^n / n
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

def series_mul(a, b, degree):
    res = np.zeros(degree + 1, dtype=float)
    for i in range(degree + 1):
        ai = a[i]
        if ai == 0.0:
            continue
        for j in range(degree - i + 1):
            res[i + j] += ai * b[j]
    return res


# ============================================================
#  Polynomial eval / derivative
# ============================================================

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


# ============================================================
#  numpy.roots 기반 실근 전체
# ============================================================

def real_roots_numpy(coeffs_asc, imag_tol=1e-8):
    c = np.array(coeffs_asc, dtype=float)
    while len(c) > 1 and abs(c[-1]) < 1e-14:
        c = c[:-1]
    if len(c) <= 1:
        return np.array([], dtype=float)

    roots = np.roots(c[::-1])  # descending
    mask = np.isfinite(roots.real) & np.isfinite(roots.imag) & (np.abs(roots.imag) < imag_tol)
    reals = roots.real[mask]
    reals.sort()
    return reals

def merge_close_sorted(xs, tol=1e-6):
    if xs.size == 0:
        return xs
    out = [float(xs[0])]
    for v in xs[1:]:
        if abs(float(v) - out[-1]) <= tol:
            out[-1] = 0.5 * (out[-1] + float(v))
        else:
            out.append(float(v))
    return np.asarray(out, dtype=float)

def roots_in_interval(sorted_roots, a, b, eps=0.0):
    if sorted_roots.size == 0:
        return sorted_roots
    lo = min(a, b) - float(eps)
    hi = max(a, b) + float(eps)
    m = (sorted_roots >= lo) & (sorted_roots <= hi)
    return sorted_roots[m]

def pick_nearest_zero_in_list(xs):
    if xs.size == 0:
        return None
    return float(xs[int(np.argmin(np.abs(xs)))])


# ============================================================
#  exp(h(x)) series (Arrhenius)
# ============================================================

def series_exp_of_poly(h, degree):
    y = np.zeros(degree + 1, dtype=float)
    y[0] = 1.0
    for n in range(1, degree + 1):
        s = 0.0
        for k in range(1, n + 1):
            s += k * h[k] * y[n - k]
        y[n] = s / n
    return y


# ============================================================
#  Templates / roles
# ============================================================

PHYSICS_TEMPLATES = [
    "damped_osc_1",        # 0
    "damped_osc_2",        # 1
    "double_well",         # 2
    "spring_tanh",         # 3
    "rc_step",             # 4
    "damped_sine_current", # 5
    "heat_fin",            # 6
    "boltzmann_prob",      # 7
    "ising_magnet",        # 8
    "activation_log",      # 9
    "arrhenius",           # 10
    "diode_iv",            # 11
    "damped_wave",         # 12
    "reaction_coord",      # 13
    "michaelis_menten",    # 14
]

ROLE_NAMES = ["real", "time", "temperature_K", "concentration", "angle"]

TEMPLATE_ROLE = {
    "damped_osc_1": "time",
    "damped_osc_2": "time",
    "rc_step": "time",
    "damped_sine_current": "time",
    "heat_fin": "time",
    "damped_wave": "time",

    "arrhenius": "temperature_K",
    "michaelis_menten": "concentration",
    "activation_log": "concentration",
    "diode_iv": "real",

    "double_well": "real",
    "reaction_coord": "real",
    "spring_tanh": "real",
    "boltzmann_prob": "real",
    "ising_magnet": "real",
}

def role_id_of(template_name: str) -> int:
    role = TEMPLATE_ROLE.get(template_name, "real")
    return ROLE_NAMES.index(role)

def role_global_domain(role_id: int, cli_x_min: float, cli_x_max: float):
    role = ROLE_NAMES[role_id]
    if role == "real":
        return cli_x_min, cli_x_max
    if role in ("time", "temperature_K", "concentration"):
        return max(0.0, cli_x_min), cli_x_max
    if role == "angle":
        a, b = -math.pi, math.pi
        return max(a, cli_x_min), min(b, cli_x_max)
    raise ValueError("unknown role")


# ============================================================
#  analytic expr safe eval for domain extraction
# ============================================================

SAFE_GLOBALS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "log": np.log,
    "sqrt": np.sqrt,
    "pi": np.pi,
    "e": np.e,
}

def expr_to_eval_core(func_expr: str) -> str:
    s = func_expr.strip()
    if "=" in s:
        s = s.split("=")[0].strip()
    s = s.replace("^", "**")
    return s

def safe_eval_expr(expr_core: str, xs: np.ndarray) -> np.ndarray:
    try:
        with np.errstate(all="ignore"):
            y = eval(expr_core, {"__builtins__": {}}, dict(SAFE_GLOBALS, x=xs))
        return np.asarray(y, dtype=float)
    except Exception:
        return np.full_like(xs, np.nan, dtype=float)

def mask_to_intervals(xs: np.ndarray, mask: np.ndarray, gap_eps: float = 0.0):
    intervals = []
    n = len(xs)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        a = float(xs[i]) + float(gap_eps)
        b = float(xs[j]) - float(gap_eps)
        if b > a:
            intervals.append((a, b))
        i = j + 1
    return intervals


# ============================================================
#  Build polynomial + expression by template
# ============================================================

def build_poly_physics(template_id, degree, rng):
    name = PHYSICS_TEMPLATES[template_id]
    poly = np.zeros(degree + 1, dtype=float)
    expr_core = None

    if name == "damped_osc_1":
        A = rng.uniform(0.5, 3.0)
        gamma = rng.uniform(0.05, 0.8)
        omega = rng.uniform(0.5, 3.0)
        x_th = rng.uniform(-1.0, 1.0)

        c_exp = taylor_exp(-gamma, degree)
        c_cos = taylor_cos(omega, degree)
        poly = A * series_mul(c_exp, c_cos, degree)
        poly[0] -= x_th
        expr_core = f"{A:.6g} * exp(-{gamma:.6g}*x) * cos({omega:.6g}*x) - {x_th:.6g}"

    elif name == "damped_osc_2":
        A = rng.uniform(0.3, 2.0)
        B = rng.uniform(0.3, 2.0)
        gamma = rng.uniform(0.05, 0.5)
        omega0 = rng.uniform(0.5, 2.0)
        omegad = rng.uniform(0.5, 2.5)
        x_th = rng.uniform(-1.0, 1.0)

        c_exp = taylor_exp(-gamma, degree)
        c_cos0 = taylor_cos(omega0, degree)
        c_cosd = taylor_cos(omegad, degree)

        poly = A * series_mul(c_exp, c_cos0, degree)
        poly += B * c_cosd
        poly[0] -= x_th
        expr_core = (
            f"{A:.6g}*exp(-{gamma:.6g}*x)*cos({omega0:.6g}*x) + "
            f"{B:.6g}*cos({omegad:.6g}*x) - {x_th:.6g}"
        )

    elif name == "double_well":
        k = rng.uniform(0.5, 5.0)
        alpha = rng.uniform(-2.0, 2.0)
        beta = rng.uniform(0.1, 2.0)
        E = rng.uniform(-3.0, 3.0)

        poly[2] += 0.5 * k
        poly[3] += alpha
        poly[4] += beta
        poly[0] -= E
        expr_core = f"0.5*{k:.6g}*x^2 + {alpha:.6g}*x^3 + {beta:.6g}*x^4 - {E:.6g}"

    elif name == "spring_tanh":
        k1 = rng.uniform(0.5, 5.0)
        k2 = rng.uniform(0.5, 5.0)
        k3 = rng.uniform(0.3, 1.5)
        F0 = rng.uniform(-3.0, 3.0)

        c_tanh = taylor_tanh(k3, degree)
        poly = k2 * c_tanh
        poly[1] += k1
        poly[0] -= F0
        expr_core = f"{k1:.6g}*x + {k2:.6g}*tanh({k3:.6g}*x) - {F0:.6g}"

    elif name == "rc_step":
        V0 = rng.uniform(0.5, 5.0)
        tau = rng.uniform(0.2, 2.0)
        V_th = rng.uniform(-2.0, 2.0)

        c_exp = taylor_exp(-1.0 / tau, degree)
        poly = V0 * (1.0 - c_exp)
        poly[0] -= V_th
        expr_core = f"{V0:.6g}*(1-exp(-x/{tau:.6g})) - {V_th:.6g}"

    elif name == "damped_sine_current":
        I0 = rng.uniform(0.5, 5.0)
        gamma = rng.uniform(0.05, 0.8)
        omega = rng.uniform(0.5, 3.0)
        I_th = rng.uniform(-2.0, 2.0)

        c_exp = taylor_exp(-gamma, degree)
        c_sin = taylor_sin(omega, degree)
        poly = I0 * series_mul(c_exp, c_sin, degree)
        poly[0] -= I_th
        expr_core = f"{I0:.6g}*exp(-{gamma:.6g}*x)*sin({omega:.6g}*x) - {I_th:.6g}"

    elif name == "heat_fin":
        T0 = rng.uniform(20.0, 100.0)
        T_inf = rng.uniform(0.0, 40.0)
        alpha = rng.uniform(0.05, 0.5)
        beta = rng.uniform(0.1, 2.0)
        T_star = rng.uniform(0.0, 100.0)

        delta = T0 - T_inf
        c_exp = taylor_exp(-alpha, degree)
        c_cosh = taylor_cosh(beta, degree)
        poly = delta * series_mul(c_exp, c_cosh, degree)
        poly[0] += T_inf
        poly[0] -= T_star
        expr_core = (
            f"{T_inf:.6g} + ({T0:.6g}-{T_inf:.6g})*exp(-{alpha:.6g}*x)*cosh({beta:.6g}*x) - {T_star:.6g}"
        )

    elif name == "boltzmann_prob":
        Z = rng.uniform(0.5, 5.0)
        p_target = rng.uniform(1e-3, 1.0)
        poly[1] -= 1.0
        poly[0] += -math.log(Z) - math.log(p_target)
        expr_core = f"-x - log({Z:.6g}) - log({p_target:.6g})"

    elif name == "ising_magnet":
        m_s = rng.uniform(0.5, 1.5)
        a = rng.uniform(0.5, 2.0)
        m_target = rng.uniform(-1.0, 1.0)

        c_tanh = taylor_tanh(a, degree)
        poly = m_s * c_tanh
        poly[0] -= m_target
        expr_core = f"{m_s:.6g}*tanh({a:.6g}*x) - {m_target:.6g}"

    elif name == "activation_log":
        A = rng.uniform(-2.0, 2.0)
        B = rng.uniform(-3.0, 3.0)
        C = rng.uniform(-3.0, 3.0)
        b = rng.uniform(-0.4, 0.4)
        if abs(b) < 0.05:
            b = math.copysign(0.05, b if b != 0 else 0.5)
        F_target = rng.uniform(-3.0, 3.0)

        c_log = taylor_log1p(b, degree)
        poly = C * c_log
        poly[0] += A
        poly[1] += B
        poly[0] -= F_target
        expr_core = f"{A:.6g} + {B:.6g}*x + {C:.6g}*log(1+{b:.6g}*x) - {F_target:.6g}"

    elif name == "arrhenius":
        A_param = rng.uniform(1e1, 1e3)
        Ea_norm = rng.uniform(1.0, 10.0)
        k_target = rng.uniform(1e-2, 1e2)

        h = np.zeros(degree + 1, dtype=float)
        for n in range(degree + 1):
            h[n] = -Ea_norm * ((-1.0) ** n)

        h0 = h[0]
        h_tilde = h.copy()
        h_tilde[0] = 0.0

        exp_h_tilde = series_exp_of_poly(h_tilde, degree)
        exp_h = math.exp(h0) * exp_h_tilde

        poly = A_param * exp_h
        poly[0] -= k_target
        expr_core = f"{A_param:.6g}*exp(-{Ea_norm:.6g}/(1+x)) - {k_target:.6g}"

    elif name == "diode_iv":
        I_s = rng.uniform(1e-12, 1e-6) * 1e6
        beta = rng.uniform(0.5, 5.0)
        I_load = rng.uniform(-2.0, 2.0)

        c_exp = taylor_exp(beta, degree)
        one = np.zeros(degree + 1, dtype=float)
        one[0] = 1.0
        poly = I_s * (c_exp - one)
        poly[0] -= I_load
        expr_core = f"{I_s:.6g}*(exp({beta:.6g}*x)-1) - {I_load:.6g}"

    elif name == "damped_wave":
        E0 = rng.uniform(0.5, 5.0)
        alpha = rng.uniform(0.05, 0.8)
        k = rng.uniform(0.5, 3.0)
        E_ref = rng.uniform(-2.0, 2.0)

        c_exp = taylor_exp(-alpha, degree)
        c_sin = taylor_sin(k, degree)
        poly = E0 * series_mul(c_exp, c_sin, degree)
        poly[0] -= E_ref
        expr_core = f"{E0:.6g}*exp(-{alpha:.6g}*x)*sin({k:.6g}*x) - {E_ref:.6g}"

    elif name == "reaction_coord":
        k = rng.uniform(0.5, 5.0)
        a = rng.uniform(-2.0, 2.0)
        b = rng.uniform(0.1, 2.0)
        dG = rng.uniform(-3.0, 3.0)

        poly[2] += 0.5 * k
        poly[3] += a
        poly[4] += b
        poly[0] -= dG
        expr_core = f"0.5*{k:.6g}*x^2 + {a:.6g}*x^3 + {b:.6g}*x^4 - {dG:.6g}"

    elif name == "michaelis_menten":
        Vmax = rng.uniform(0.5, 5.0)
        v_target = rng.uniform(0.0, 5.0)

        frac = np.zeros(degree + 1, dtype=float)
        for n in range(1, degree + 1):
            frac[n] = ((-1.0) ** (n - 1))
        poly = Vmax * frac
        poly[0] -= v_target
        expr_core = f"{Vmax:.6g}*(x/(1+x)) - {v_target:.6g}"

    else:
        raise ValueError(f"Unknown template: {name}")

    # constant term normalization
    a0 = float(poly[0])
    if abs(a0) > 1.0:
        scale = abs(a0)
        poly /= scale
        expr = f"({expr_core})/{scale:.6g} = 0"
    else:
        expr = f"{expr_core} = 0"

    return poly, expr


# ============================================================
# Domain extraction
# ============================================================

def extract_valid_domains_from_expr(
    func_expr: str,
    role_id: int,
    cli_x_min: float,
    cli_x_max: float,
    scan_n: int,
    y_abs_max_true: float,
    gap_eps: float,
    min_width: float,
    max_domains_keep: int,
):
    gmin, gmax = role_global_domain(role_id, cli_x_min, cli_x_max)
    if not (np.isfinite(gmin) and np.isfinite(gmax)) or gmin >= gmax:
        return []

    xs = np.linspace(gmin, gmax, scan_n, dtype=float)
    expr_core = expr_to_eval_core(func_expr)
    ys = safe_eval_expr(expr_core, xs)

    mask = np.isfinite(ys) & (np.abs(ys) <= y_abs_max_true)
    intervals = mask_to_intervals(xs, mask, gap_eps=gap_eps)

    intervals = [(a, b) for (a, b) in intervals if (b - a) >= min_width]
    intervals = intervals[:max_domains_keep]
    return intervals


# ============================================================
# All-roots collection
# ============================================================

def collect_all_real_roots_for_coeffs_in_intervals(
    coeffs, intervals, imag_tol, merge_tol, in_eps=0.0
):
    c1 = poly_derivative(coeffs)

    rr0 = real_roots_numpy(coeffs, imag_tol=imag_tol)
    rr1 = real_roots_numpy(c1, imag_tol=imag_tol)

    out0 = []
    out1 = []
    for (a, b) in intervals:
        r0i = roots_in_interval(rr0, a, b, eps=in_eps)
        r1i = roots_in_interval(rr1, a, b, eps=in_eps)
        if r0i.size:
            out0.append(r0i)
        if r1i.size:
            out1.append(r1i)

    if out0:
        out0 = np.concatenate(out0, axis=0)
        out0.sort()
        out0 = merge_close_sorted(out0, tol=merge_tol)
    else:
        out0 = np.array([], dtype=float)

    if out1:
        out1 = np.concatenate(out1, axis=0)
        out1.sort()
        out1 = merge_close_sorted(out1, tol=merge_tol)
    else:
        out1 = np.array([], dtype=float)

    return out0, out1


def choose_interval_index(intervals, rr0_all, prefer="most_roots_then_width"):
    if not intervals:
        return None
    if prefer == "first":
        return 0

    best = None
    for i, (a, b) in enumerate(intervals):
        r = roots_in_interval(rr0_all, a, b, eps=0.0)
        cnt = int(r.size)
        width = float(b - a)
        key = (cnt, width)
        if best is None or key > best[0]:
            best = (key, i)
    return best[1] if best is not None else 0


def pad_roots(xs, max_roots):
    arr = np.full((max_roots,), np.nan, dtype=np.float32)
    n = min(int(xs.size), int(max_roots))
    if n > 0:
        arr[:n] = xs[:n].astype(np.float32)
    return arr, int(xs.size)


# ============================================================
# Dataset generation
# ============================================================

def generate_physics_test_dataset_v3_allroots(
    *,
    n_total: int,
    degree: int,
    seed: int,
    cli_x_min: float,
    cli_x_max: float,
    imag_tol: float,
    merge_tol: float,
    in_eps: float,
    max_tries_factor: int,

    domain_scan_n: int,
    y_abs_max_true: float,
    domain_gap_eps: float,
    domain_min_width: float,
    max_domains_keep: int,

    roots_scope: str,
    max_roots: int,
    choose_policy: str,
):
    rng = np.random.default_rng(seed)

    coeffs_list = []
    fid_list = []
    func_expr_list = []

    x_min_list = []
    x_max_list = []

    role_id_list = []
    role_name_list = []

    domain_count_list = []
    domains_list = []
    domain_choice_idx_list = []

    root0_all_list = []
    root1_all_list = []
    root0_count_list = []
    root1_count_list = []

    r0_single_list = []
    r1_single_list = []

    num_templates = len(PHYSICS_TEMPLATES)
    print(f"[INFO] #physics templates = {num_templates}")
    print(f"[INFO] CLI x range = [{cli_x_min}, {cli_x_max}]")
    print(f"[INFO] imag_tol={imag_tol} merge_tol={merge_tol} in_eps={in_eps}")
    print(f"[INFO] domain_scan_n={domain_scan_n} y_abs_max_true={y_abs_max_true} gap_eps={domain_gap_eps} "
          f"min_width={domain_min_width} max_domains_keep={max_domains_keep}")
    print(f"[INFO] roots_scope={roots_scope} max_roots={max_roots} choose_policy={choose_policy}")

    max_tries = int(n_total) * int(max_tries_factor)
    tries = 0
    skipped_no_domain = 0
    skipped_no_roots = 0

    while len(coeffs_list) < n_total and tries < max_tries:
        tries += 1

        fid = int(rng.integers(0, num_templates))
        coeffs, func_expr = build_poly_physics(fid, degree, rng)

        if np.all(np.abs(coeffs) < 1e-12):
            continue

        template_name = PHYSICS_TEMPLATES[fid]
        rid = role_id_of(template_name)
        rname = ROLE_NAMES[rid]

        intervals = extract_valid_domains_from_expr(
            func_expr=func_expr,
            role_id=rid,
            cli_x_min=cli_x_min,
            cli_x_max=cli_x_max,
            scan_n=domain_scan_n,
            y_abs_max_true=y_abs_max_true,
            gap_eps=domain_gap_eps,
            min_width=domain_min_width,
            max_domains_keep=max_domains_keep,
        )
        if not intervals:
            skipped_no_domain += 1
            continue

        rr0_all_domains, rr1_all_domains = collect_all_real_roots_for_coeffs_in_intervals(
            coeffs=coeffs,
            intervals=intervals,
            imag_tol=imag_tol,
            merge_tol=merge_tol,
            in_eps=in_eps,
        )
        if rr0_all_domains.size == 0 or rr1_all_domains.size == 0:
            skipped_no_roots += 1
            continue

        choice_idx = choose_interval_index(intervals, rr0_all=rr0_all_domains, prefer=choose_policy)
        if choice_idx is None:
            skipped_no_roots += 1
            continue

        x_min, x_max = intervals[int(choice_idx)]

        if roots_scope == "chosen_domain":
            rr0_use, rr1_use = collect_all_real_roots_for_coeffs_in_intervals(
                coeffs=coeffs,
                intervals=[(x_min, x_max)],
                imag_tol=imag_tol,
                merge_tol=merge_tol,
                in_eps=in_eps,
            )
        else:
            rr0_use, rr1_use = rr0_all_domains, rr1_all_domains

        r0_single = pick_nearest_zero_in_list(rr0_use)
        r1_single = pick_nearest_zero_in_list(rr1_use)
        if r0_single is None or r1_single is None:
            skipped_no_roots += 1
            continue

        dom = np.full((max_domains_keep, 2), np.nan, dtype=np.float32)
        for i, (a, b) in enumerate(intervals[:max_domains_keep]):
            dom[i, 0] = np.float32(a)
            dom[i, 1] = np.float32(b)

        r0_pad, r0_cnt = pad_roots(rr0_use, max_roots=max_roots)
        r1_pad, r1_cnt = pad_roots(rr1_use, max_roots=max_roots)

        coeffs_list.append(coeffs.astype(np.float32))
        fid_list.append(fid)
        func_expr_list.append(func_expr)

        x_min_list.append(float(x_min))
        x_max_list.append(float(x_max))

        role_id_list.append(rid)
        role_name_list.append(rname)

        domain_count_list.append(len(intervals))
        domains_list.append(dom)
        domain_choice_idx_list.append(int(choice_idx))

        root0_all_list.append(r0_pad)
        root1_all_list.append(r1_pad)
        root0_count_list.append(int(r0_cnt))
        root1_count_list.append(int(r1_cnt))

        r0_single_list.append([float(r0_single)])
        r1_single_list.append([float(r1_single)])

        if len(coeffs_list) % 200 == 0:
            print(f"[INFO] collected {len(coeffs_list)} / {n_total} (tries={tries})")

    if len(coeffs_list) < n_total:
        print(f"[WARN] generated only {len(coeffs_list)} / {n_total} (tries={tries}, max_tries={max_tries})")

    print(f"[STATS] skipped_no_domain={skipped_no_domain}")
    print(f"[STATS] skipped_no_roots={skipped_no_roots}")

    coeffs_arr = np.stack(coeffs_list, axis=0)
    fid_arr = np.array(fid_list, dtype=np.int32)
    func_expr_arr = np.array(func_expr_list, dtype=object)

    x_min_arr = np.array(x_min_list, dtype=np.float32)
    x_max_arr = np.array(x_max_list, dtype=np.float32)

    role_id_arr = np.array(role_id_list, dtype=np.int32)
    role_name_arr = np.array(role_name_list, dtype="<U32")

    domain_count_arr = np.array(domain_count_list, dtype=np.int32)
    domains_arr = np.stack(domains_list, axis=0).astype(np.float32)
    domain_choice_idx_arr = np.array(domain_choice_idx_list, dtype=np.int32)

    root0_all_arr = np.stack(root0_all_list, axis=0).astype(np.float32)
    root1_all_arr = np.stack(root1_all_list, axis=0).astype(np.float32)
    root0_count_arr = np.array(root0_count_list, dtype=np.int32)
    root1_count_arr = np.array(root1_count_list, dtype=np.int32)

    r0_arr = np.array(r0_single_list, dtype=np.float32)  # (N,1)
    r1_arr = np.array(r1_single_list, dtype=np.float32)  # (N,1)

    return (
        coeffs_arr,
        r0_arr,
        r1_arr,
        fid_arr,
        func_expr_arr,
        x_min_arr,
        x_max_arr,
        role_id_arr,
        role_name_arr,
        domain_count_arr,
        domains_arr,
        domain_choice_idx_arr,
        root0_all_arr,
        root1_all_arr,
        root0_count_arr,
        root1_count_arr,
    )


# ============================================================
# main (YAML-driven)
# ============================================================

def main() -> None:
    # Resolve repo root robustly even if executed from notebooks/
    repo = find_repo_root(Path(__file__).resolve())
    os.chdir(repo)

    default_cfg = "configs/make_test_dataset.yaml"
    cfg_path = Path(os.environ.get("CFG_PATH", default_cfg))

    if not cfg_path.is_absolute():
        cfg_path = (repo / cfg_path).resolve()

    cfg = load_yaml(cfg_path)

    degree = int(get(cfg, "dataset.degree", 25))
    n_total = int(get(cfg, "dataset.n_total", 10000))
    seed = int(get(cfg, "dataset.seed", 0))

    x_min = float(get(cfg, "dataset.x_min", -1000.0))
    x_max = float(get(cfg, "dataset.x_max",  1000.0))
    if x_min >= x_max:
        raise ValueError("dataset.x_min must be < dataset.x_max")

    imag_tol = float(get(cfg, "dataset.imag_tol", 1e-8))
    merge_tol = float(get(cfg, "dataset.merge_tol", 1e-6))
    in_eps = float(get(cfg, "dataset.in_eps", 0.0))

    domain_scan_n = int(get(cfg, "domains.scan_n", 1601))
    y_abs_max_true = float(get(cfg, "domains.y_abs_max_true", 200.0))
    domain_gap_eps = float(get(cfg, "domains.gap_eps", 1e-6))
    domain_min_width = float(get(cfg, "domains.min_width", 0.05))
    max_domains_keep = int(get(cfg, "domains.max_domains_keep", 6))

    roots_scope = str(get(cfg, "roots.scope", "all_domains")).strip()
    if roots_scope not in ("all_domains", "chosen_domain"):
        raise ValueError("roots.scope must be 'all_domains' or 'chosen_domain'")

    max_roots = int(get(cfg, "roots.max_roots", 25))
    choose_policy = str(get(cfg, "roots.choose_policy", "most_roots_then_width")).strip()
    if choose_policy not in ("most_roots_then_width", "first"):
        raise ValueError("roots.choose_policy must be 'most_roots_then_width' or 'first'")

    out_rel = str(get(cfg, "output.npz_path", "data/test/taylor_test_physchem_v3_allroots_10000.npz"))
    out_path = (repo / out_rel).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional override via env (still relative to repo if not absolute)
    env_out = os.environ.get("OUT_NPZ", "").strip()
    if env_out:
        p = Path(env_out)
        out_path = p if p.is_absolute() else (repo / p).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[CFG] {cfg_path}")
    print(f"[REPO] {repo}")
    print(f"[OUT] {out_path}")

    print("[INFO] templates:")
    for i, name in enumerate(PHYSICS_TEMPLATES):
        rid = role_id_of(name)
        print(f"  {i:2d}: {name:18s} role={ROLE_NAMES[rid]}")

    (
        coeffs, r0, r1, fid, func_expr,
        x_min_arr, x_max_arr,
        role_id_arr, role_name_arr,
        domain_count_arr, domains_arr, domain_choice_idx_arr,
        root0_all, root1_all, root0_count, root1_count
    ) = generate_physics_test_dataset_v3_allroots(
        n_total=n_total,
        degree=degree,
        seed=seed,
        cli_x_min=x_min,
        cli_x_max=x_max,
        imag_tol=imag_tol,
        merge_tol=merge_tol,
        in_eps=in_eps,
        max_tries_factor=120,

        domain_scan_n=domain_scan_n,
        y_abs_max_true=y_abs_max_true,
        domain_gap_eps=domain_gap_eps,
        domain_min_width=domain_min_width,
        max_domains_keep=max_domains_keep,

        roots_scope=roots_scope,
        max_roots=max_roots,
        choose_policy=choose_policy,
    )

    N = coeffs.shape[0]
    degree_arr = np.full((N,), degree, dtype=np.int32)

    np.savez_compressed(
        out_path,
        coeffs=coeffs,
        root0=r0,                 # (N,1) compat
        root1=r1,                 # (N,1) compat
        root0_all=root0_all,      # (N,R)
        root1_all=root1_all,      # (N,R)
        root0_count=root0_count,  # (N,)
        root1_count=root1_count,  # (N,)
        func_id=fid,
        degree=degree_arr,
        func_expr=func_expr,
        x_min=x_min_arr,
        x_max=x_max_arr,
        x_role_id=role_id_arr,
        x_role_name=role_name_arr,
        domain_count=domain_count_arr,
        domains=domains_arr,
        domain_choice_idx=domain_choice_idx_arr,
    )

    print(f"[SAVE] {out_path} (N={N})")
    print("[DONE] test dataset generated (v3_allroots).")


if __name__ == "__main__":
    main()
