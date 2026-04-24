#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run degree sweep for transformer interval model + ANN/MLP/LSTM root regressors.

Example:
  python run_taylor_degree_sweep.py \
      --degrees 5 10 15 20 25 30 35 40 45 50 \
      --device cuda \
      --interval-script /path/model_degree.py \
      --ann-script /path/ann_degree.py \
      --mlp-script /path/mlp_degree.py \
      --lstm-script /path/lstm_degree.py

By default this script only sets environment variables.
Each training script still reads its own YAML, but TAYLOR_ORDER overrides
the effective order and the default NPZ paths.
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

def run_one(script: str, env: dict):
    cmd = [sys.executable, script]
    print("[RUN]", " ".join(cmd))
    print("      TAYLOR_ORDER=", env.get("TAYLOR_ORDER"))
    subprocess.run(cmd, check=True, env=env)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degrees", type=int, nargs="+", default=[5,10,15,20,25,30,35,40,45,50])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--interval-script", type=str, default="/home/seokjun/taylor-root-prediction/revision/model_degree_patched.py")
    ap.add_argument("--ann-script", type=str, default="/home/seokjun/taylor-root-prediction/revision/ann_degree_patched.py")
    ap.add_argument("--mlp-script", type=str, default="/home/seokjun/taylor-root-prediction/revision/mlp_degree_patched.py")
    ap.add_argument("--lstm-script", type=str, default="/home/seokjun/taylor-root-prediction/revision/lstm_degree_patched.py")
    ap.add_argument("--run-interval", action="store_true")
    ap.add_argument("--run-ann", action="store_true")
    ap.add_argument("--run-mlp", action="store_true")
    ap.add_argument("--run-lstm", action="store_true")
    ap.add_argument("--interval-data-dir", type=str, default="/home/seokjun/taylor-root-prediction/taylor_data_physchem_v4_deg5")
    ap.add_argument("--taylor-data-root", type=str, default="/home/seokjun/taylor-root-prediction/revision/taylor_data_physchem_v4_interval_deg5")
    args = ap.parse_args()

    run_any = args.run_interval or args.run_ann or args.run_mlp or args.run_lstm
    if not run_any:
        args.run_interval = args.run_ann = args.run_mlp = args.run_lstm = True

    for degree in args.degrees:
        env = os.environ.copy()
        env["DEVICE"] = args.device
        env["TAYLOR_ORDER"] = str(degree)
        env["DATA_DIR_INTERVAL"] = args.interval_data_dir
        env["DATA_DIR_TAYLOR"] = f"{args.taylor_data_root}/taylor_data_physchem_v4_deg{degree}"

        if args.run_interval:
            env_i = env.copy()
            env_i["OUT_DIR"] = f"results/transformer_interval/deg{degree}"
            run_one(args.interval_script, env_i)

        if args.run_ann:
            env_a = env.copy()
            env_a["OUT_DIR"] = f"results/taylor_nn/ann/deg{degree}"
            run_one(args.ann_script, env_a)

        if args.run_mlp:
            env_m = env.copy()
            env_m["OUT_DIR"] = f"results/taylor_nn/mlp/deg{degree}"
            run_one(args.mlp_script, env_m)

        if args.run_lstm:
            env_l = env.copy()
            env_l["OUT_DIR"] = f"results/taylor_nn/lstm/deg{degree}"
            run_one(args.lstm_script, env_l)

if __name__ == "__main__":
    main()
