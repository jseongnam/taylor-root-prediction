# Taylor Root Prediction (IEEE Access / Reproducibility Package)

This repository provides a reproducible implementation for our Taylor-polynomial-based root prediction framework:
- **Transformer interval predictor** (AST-tokenized expression → top-k interval centers)
- **Taylor-root regressors** (shifted Taylor coefficients → predicted root offset)
  - ANN (MDPI-style shallow FNN)
  - LSTM baseline
  - Anchored MLP (Expectation residual over anchors)
- **Baseline solver** (scan + bracket + bisection + post-check)
- **Evaluation**: K-sweep, winner analysis, threshold sweep, fail concentration by function/template, plots

> ✅ All paths are **relative to the repository root** so that reviewers can run after `git clone` without editing absolute paths.

1. Quick Start (Reviewer)

1) Create environment & install dependencies

Option A — using requirements.txt
```bash
pip install -r requirements.txt
```
Option B — minimal install
```bash
pip install numpy torch tqdm pyyaml matplotlib pillow
# optional (only if you enable symbolic solvers)
pip install sympy requests
```

GPU is optional. If CUDA is unavailable, everything runs on CPU (slower).

2. Recommended: Run the Jupyter Notebook Demo

We provide a reviewer notebook that can:

(optional) generate small NPZ datasets

(optional) train models

evaluate all models + baseline

show plots and failure distributions

Open:

notebooks/reviewer_demo_notebook_v3.ipynb

Run from top to bottom.

3. Repository Structure
configs/
  taylor_root_ann.yaml
  taylor_root_lstm.yaml
  taylor_root_mlp.yaml
  transformer_interval.yaml
  eval_k_sweep.yaml

models/
  taylor_nn/
    ann.py
    lstm.py
    mlp.py
  transformer/
    model.py

scripts/
  data/
    generate_dataset_physchem_v4.py
    generate_interval_dataset_physchem_v4.py
  eval/
    evaluate_k_sweep.py
data/                       # (generated; not included by default)
  taylor_data_physchem_v4_deg25/
    taylor_deg25_{train,val,test}.npz
  taylor_data_physchem_v4_interval/
    taylor_deg25_{train,val,test}.npz
  
results/                    # outputs (generated)

4. Dataset Generation (NPZ)

The repository does not ship large datasets by default.
Use the generation scripts to create NPZ files under data/.

A) Root regression dataset (ANN/LSTM/MLP)

Expected outputs:

data/taylor_data_physchem_v4_deg25/taylor_deg25_train.npz

data/taylor_data_physchem_v4_deg25/taylor_deg25_val.npz

data/taylor_data_physchem_v4_deg25/taylor_deg25_test.npz

Run (example):
```bash
python scripts/data/generate_dataset_physchem_v4.py \
  --degree 25 \
  --n-total 20000 \
  --seed 42 \
  --out-dir data/taylor_data_physchem_v4_deg25 \
  --save-expr-str 1
```
B) Transformer interval dataset

Expected outputs:

data/taylor_data_physchem_v4_interval/taylor_deg25_train.npz

data/taylor_data_physchem_v4_interval/taylor_deg25_val.npz

data/taylor_data_physchem_v4_interval/taylor_deg25_test.npz

Run (example):
```bash
python scripts/data/generate_interval_dataset_physchem_v4.py \
  --degree 25 \
  --n-total 20000 \
  --seed 42 \
  --out-dir data/taylor_data_physchem_v4_interval \
  --save-expr-str 1
```

Notes:

Generation scripts can be expensive if n-total is large (e.g., 1,000,000).

For reviewer testing, we recommend starting with small sizes (20k–100k).

5. Training (YAML-driven, no CLI args required)

All training scripts read:

config yaml via TAYLOR_CFG

dataset paths via TRAIN_NPZ, VAL_NPZ, TEST_NPZ

output directory via OUT_DIR

device via DEVICE

ANN
```bash
PYTHONPATH=. \
TAYLOR_CFG=configs/taylor_root_ann.yaml \
TRAIN_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_train.npz \
VAL_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_val.npz \
TEST_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_test.npz \
OUT_DIR=results/taylor_nn/ann \
DEVICE=cuda \
python models/taylor_nn/ann.py
```
LSTM
```bash
PYTHONPATH=. \
TAYLOR_CFG=configs/taylor_root_lstm.yaml \
TRAIN_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_train.npz \
VAL_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_val.npz \
TEST_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_test.npz \
OUT_DIR=results/taylor_nn/lstm \
DEVICE=cuda \
python models/taylor_nn/lstm.py
```
Anchored MLP
```bash
PYTHONPATH=. \
TAYLOR_CFG=configs/taylor_root_mlp.yaml \
TRAIN_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_train.npz \
VAL_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_val.npz \
TEST_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_test.npz \
OUT_DIR=results/taylor_nn/mlp \
DEVICE=cuda \
python models/taylor_nn/mlp.py
```
Transformer interval predictor
```bash
PYTHONPATH=. \
CFG_PATH=configs/transformer_interval.yaml \
TRAIN_NPZ=data/taylor_data_physchem_v4_interval/taylor_deg25_train.npz \
VAL_NPZ=data/taylor_data_physchem_v4_interval/taylor_deg25_val.npz \
TEST_NPZ=data/taylor_data_physchem_v4_interval/taylor_deg25_test.npz \
OUT_DIR=results/transformer_interval \
DEVICE=cuda \
MODE=train \
python models/transformer/model.py
```
6. Evaluation (K-sweep + Baseline + Plots + Failure Distribution)

Evaluation is YAML-driven. Configure configs/eval_k_sweep.yaml and run:
```bash
PYTHONPATH=. \
EVAL_CFG=configs/eval_k_sweep.yaml \
OUTDIR=results/runs_k_sweep_viz \
DEVICE=cuda \
python evaluation/evaluate_k_sweep.py
```
Recommended flags via YAML (reviewer-friendly)

threshold sweep

winner analysis

fail concentration by func_id

residual histogram

func_id boxplot

The evaluation script generates:

summary logs

*.png plots (histograms, boxplots, edge cases)

fail_by_funcid_*.csv/json if enabled

7. Reproducing “Failure concentration” (functions that all methods fail)

In configs/eval_k_sweep.yaml:

reports.report_fail_funcid: true

reports.report_fail_mode: baseline or all

baseline: where baseline fails (ok@thr) per func_id

all: where all methods fail (winner_ok == none)

Outputs:

results/.../fail_by_funcid_{baseline|all}_K*_thr*.csv/json

8. Notes for Reviewers

All experiments can run on CPU, but training is slower.

Large-scale datasets (n-total=1,000,000) may require substantial time and disk.

For quick verification, start with smaller dataset sizes (e.g., 20k) and fewer epochs.

If you see No module named src, run with PYTHONPATH=. as in the examples above.

9. Citation

If you use this code, please cite our paper:

(Provide your IEEE Access citation here)
