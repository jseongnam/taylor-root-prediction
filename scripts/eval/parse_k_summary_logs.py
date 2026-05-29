#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
import csv
from pathlib import Path
from collections import defaultdict

# =========================
# 설정
# =========================
LOG_DIR = Path("/home/seokjun/taylor-root-prediction/results/logs")
OUT_DIR = Path("/home/seokjun/taylor-root-prediction/results/log_parsed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 예: eval_k_sweep_deg15.log, eval_k_sweep_deg20.log ...
LOG_GLOB = "eval_k_sweep_deg*.log"

# degree 추출용
DEGREE_RE = re.compile(r"deg(\d+)", re.IGNORECASE)

# K summary block 시작
K_RE = re.compile(r"-+\s*\[K=(\d+)\]\s*SUMMARY\s*-+", re.IGNORECASE)

# 각 method 줄 파싱
LINE_RE = re.compile(
    r"""
    ^\[(?P<method>[^\]]+)\]\s+
    ok@1\.0e-10\s*=\s*(?P<ok_pct>[0-9.]+)%\s*\|\s*
    \|f\|\s*mean=(?P<f_mean>[0-9.eE+\-]+|nan)\s+
    p90=(?P<f_p90>[0-9.eE+\-]+|nan)\s+
    p99=(?P<f_p99>[0-9.eE+\-]+|nan)\s*\|\s*
    time\(ms\)\s*mean=(?P<t_mean>[0-9.eE+\-]+|nan)\s+
    std=(?P<t_std>[0-9.eE+\-]+|nan)\s+
    p50=(?P<t_p50>[0-9.eE+\-]+|nan)\s+
    p90=(?P<t_p90>[0-9.eE+\-]+|nan)\s+
    p99=(?P<t_p99>[0-9.eE+\-]+|nan)
    """,
    re.VERBOSE,
)


def to_float(x: str):
    x = x.strip().lower()
    if x == "nan":
        return float("nan")
    return float(x)


def extract_degree_from_filename(path: Path) -> int:
    m = DEGREE_RE.search(path.stem)
    if not m:
        raise ValueError(f"degree를 파일명에서 찾을 수 없습니다: {path.name}")
    return int(m.group(1))


def parse_log_file(path: Path):
    degree = extract_degree_from_filename(path)
    rows = []

    current_k = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()

            mk = K_RE.search(line)
            if mk:
                current_k = int(mk.group(1))
                continue

            ml = LINE_RE.search(line)
            if ml and current_k is not None:
                gd = ml.groupdict()
                rows.append(
                    {
                        "degree": degree,
                        "k": current_k,
                        "method": gd["method"].strip(),
                        "ok_pct": to_float(gd["ok_pct"]),
                        "f_mean": to_float(gd["f_mean"]),
                        "f_p90": to_float(gd["f_p90"]),
                        "f_p99": to_float(gd["f_p99"]),
                        "t_mean_ms": to_float(gd["t_mean"]),
                        "t_std_ms": to_float(gd["t_std"]),
                        "t_p50_ms": to_float(gd["t_p50"]),
                        "t_p90_ms": to_float(gd["t_p90"]),
                        "t_p99_ms": to_float(gd["t_p99"]),
                        "source_log": path.name,
                    }
                )

    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_pivot(rows: list[dict], value_key: str):
    """
    행: degree, method
    열: K
    값: value_key
    """
    ks = sorted({r["k"] for r in rows})
    grouped = defaultdict(dict)

    for r in rows:
        grouped[(r["degree"], r["method"])][r["k"]] = r[value_key]

    out_rows = []
    for (degree, method), kv in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        row = {"degree": degree, "method": method}
        for k in ks:
            row[f"K{k}"] = kv.get(k, "")
        out_rows.append(row)

    return out_rows, ["degree", "method"] + [f"K{k}" for k in ks]


def main():
    log_files = sorted(LOG_DIR.glob(LOG_GLOB))
    if not log_files:
        raise FileNotFoundError(f"로그 파일을 찾지 못했습니다: {LOG_DIR / LOG_GLOB}")

    all_rows = []
    for p in log_files:
        rows = parse_log_file(p)
        if rows:
            all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("파싱된 summary row가 없습니다. 로그 형식을 다시 확인하세요.")

    # 1) raw long-form csv
    raw_fields = [
        "degree",
        "k",
        "method",
        "ok_pct",
        "f_mean",
        "f_p90",
        "f_p99",
        "t_mean_ms",
        "t_std_ms",
        "t_p50_ms",
        "t_p90_ms",
        "t_p99_ms",
        "source_log",
    ]
    raw_csv = OUT_DIR / "k_summary_all_rows.csv"
    write_csv(raw_csv, all_rows, raw_fields)

    # 2) ok_pct pivot
    ok_rows, ok_fields = make_pivot(all_rows, "ok_pct")
    ok_csv = OUT_DIR / "k_summary_ok_pct_pivot.csv"
    write_csv(ok_csv, ok_rows, ok_fields)

    # 3) residual mean pivot
    fmean_rows, fmean_fields = make_pivot(all_rows, "f_mean")
    fmean_csv = OUT_DIR / "k_summary_f_mean_pivot.csv"
    write_csv(fmean_csv, fmean_rows, fmean_fields)

    # 4) time mean pivot
    tmean_rows, tmean_fields = make_pivot(all_rows, "t_mean_ms")
    tmean_csv = OUT_DIR / "k_summary_time_mean_ms_pivot.csv"
    write_csv(tmean_csv, tmean_rows, tmean_fields)

    # 5) degree-method별 best K (ok_pct 최대)
    best_rows = []
    grouped = defaultdict(list)
    for r in all_rows:
        grouped[(r["degree"], r["method"])].append(r)

    for (degree, method), items in sorted(grouped.items()):
        items_sorted = sorted(
            items,
            key=lambda x: (
                -(x["ok_pct"]),
                x["f_mean"] if str(x["f_mean"]) != "nan" else float("inf"),
                x["t_mean_ms"] if str(x["t_mean_ms"]) != "nan" else float("inf"),
            ),
        )
        best = items_sorted[0]
        best_rows.append(
            {
                "degree": degree,
                "method": method,
                "best_k": best["k"],
                "best_ok_pct": best["ok_pct"],
                "best_f_mean": best["f_mean"],
                "best_f_p90": best["f_p90"],
                "best_f_p99": best["f_p99"],
                "best_t_mean_ms": best["t_mean_ms"],
                "source_log": best["source_log"],
            }
        )

    best_csv = OUT_DIR / "k_summary_best_k_by_degree_method.csv"
    write_csv(
        best_csv,
        best_rows,
        [
            "degree",
            "method",
            "best_k",
            "best_ok_pct",
            "best_f_mean",
            "best_f_p90",
            "best_f_p99",
            "best_t_mean_ms",
            "source_log",
        ],
    )

    print("[DONE]")
    print("raw      :", raw_csv)
    print("ok pivot :", ok_csv)
    print("f pivot  :", fmean_csv)
    print("t pivot  :", tmean_csv)
    print("best k   :", best_csv)


if __name__ == "__main__":
    main()