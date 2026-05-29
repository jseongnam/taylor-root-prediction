# Taylor Root Prediction

Official implementation of the IEEE Access paper:

**A Root Prediction System for Single-Variable Equations with Existing Taylor Polynomials**  
Seokjun Jeong and Yoosoo Oh  
IEEE Access, Early Access, 2026  
DOI: [10.1109/ACCESS.2026.3697368](https://doi.org/10.1109/ACCESS.2026.3697368)

This repository provides a reproducible implementation of a neural-numerical root prediction framework that combines:

- Transformer-based interval localization
- 25th-order shifted Maclaurin/Taylor representations
- coefficient-based neural root regression
- multi-candidate root prediction
- residual, domain, and derivative-stability validation
- baseline numerical solver comparison
- K-sweep and failure-concentration analysis

The goal of this work is not to blindly replace numerical validation with neural prediction.  
Instead, the framework reformulates nonlinear root finding as a structured prediction problem while preserving numerical reliability through strict residual-based post-checks.

---

## Research Motivation

Classical root-finding methods such as Newton-Raphson, bisection, and secant methods are powerful and widely used, but their practical behavior can be sensitive to initialization, derivative behavior, ill-conditioning, and basin-of-attraction structure.

This work explores a neural-numerical alternative. Instead of manually selecting a single initial point, the proposed system first localizes candidate root-containing intervals using a Transformer, constructs local Taylor representations around the selected centers, predicts multiple candidate roots, and finally validates the candidates using strict numerical checks.

The central principle of this repository is:

> Neural models should assist numerical solving through localization, prediction, or correction, while residual-based validation preserves numerical reliability.

---

## Method Overview

The proposed framework consists of four main stages.

### 1. Transformer-Based Interval Localization

The input equation is tokenized as a symbolic sequence.  
A Transformer encoder processes the expression and predicts top-k candidate interval centers that are likely to contain real roots.

In the final operating point used in the paper, the system uses:

```text
top-k = 25 candidate intervals
```


---
### 2. Shifted Local Taylor Representation

For each predicted candidate center, the variable is shifted so that the function can be approximated locally by a Maclaurin expansion.

The shifted function is expanded up to the 25th order:
```text
N = 25
```
This produces a compact coefficient vector that represents local function behavior near the candidate center.


---
### 3. Coefficient-Based Multi-Candidate Root Regression

The Taylor coefficient vector is passed to neural regression models that predict multiple candidate roots.

Implemented regression backbones include:

- ANN
- LSTM
- MLP / anchored MLP variant

The regressors are treated as alternative backbones inside the same root-prediction framework.

---
### 4. Residual-Based Selection and Validation

Predicted candidate roots are evaluated using the original function.

A candidate is accepted only if it satisfies the numerical validation protocol:
```text
|f(r)| < 1e-10
```
Additional checks include:

- domain constraint validation
- derivative-based stability checking
- residual-based candidate ranking
- post-check against the original equation

## Key Contributions

This repository implements the following research components:

- Reformulation of single-variable nonlinear root finding as a structured neural prediction problem.
- Transformer-based interval localization for identifying candidate root-containing regions.
- Local Taylor/Maclaurin coefficient representation after variable shifting.
- Coefficient-based neural regression for multi-candidate root prediction.
- Strict residual-based validation instead of direct trust in raw neural outputs.
- Baseline solver comparison using scan, bracket, bisection, and post-check.
- K-sweep evaluation and operating-point analysis.
- Failure analysis by function family and template type.

## Main Results Summary

The paper evaluates the framework on a benchmark of 10,000 test equations spanning 12 nonlinear function families.

Main findings:

| Component               | Summary                                                                                        |
| ----------------------- | ---------------------------------------------------------------------------------------------- |
| Taylor truncation order | 25th-order local Taylor expansion provides a stable runtime-performance trade-off              |
| Interval candidates     | top-25 candidate intervals provide a stable operating point                                    |
| Validation threshold    | strict residual threshold of `1e-10` is used                                                   |
| Residual quality        | validated roots reach residuals on the order of `1e-12` to `1e-11`                             |
| Main limitation         | failure cases are often governed by representation-level limits of truncated Taylor expansions |

## Repository Structure

```text
taylor-root-prediction/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── taylor_root_ann.yaml
│   ├── taylor_root_lstm.yaml
│   ├── taylor_root_mlp.yaml
│   ├── transformer_interval.yaml
│   └── eval_k_sweep.yaml
│
├── models/
│   ├── taylor_nn/
│   │   ├── ann.py
│   │   ├── lstm.py
│   │   └── mlp.py
│   └── transformer/
│       └── model.py
│
├── scripts/
│   ├── data/
│   │   ├── generate_dataset_physchem_v4.py
│   │   └── generate_interval_dataset_physchem_v4.py
│   └── eval/
│       └── evaluate_k_sweep.py
│
├── notebooks/
│   └── reviewer_demo_notebook_v3.ipynb
│
├── results/
│   └── generated outputs
│
└── src/
    └── utility modules
```

Note: The notebook is provided as a reviewer-friendly demo.
The core implementation is organized as Python scripts and YAML configuration files under models/, scripts/, and configs/.

## Quick Start

1. Clone the Repository

```text
git clone https://github.com/jseongnam/taylor-root-prediction.git
cd taylor-root-prediction
```

2. Create Environment

Using requirements.txt:
```text
pip install -r requirements.txt
```
Minimal installation:
```text
pip install numpy torch tqdm pyyaml matplotlib pillow
```
Optional packages:
```text
pip install sympy requests
```
GPU is optional.
If CUDA is unavailable, the scripts can run on CPU, although training and large-scale evaluation will be slower.

## Reviewer Demo Notebook

A reviewer-friendly notebook is provided:
```text
notebooks/reviewer_demo_notebook_v3.ipynb
```
The notebook can be used to:

- generate small NPZ datasets
- train small-scale models
- evaluate neural regressors and baseline solvers
- visualize residual distributions
- inspect failure cases

Open the notebook and run from top to bottom.

## Dataset Generation

Large-scale datasets are not included in this repository due to file size limitations.
Use the provided generation scripts to create NPZ files under data/.

A. Root Regression Dataset
Expected outputs:
```text
data/taylor_data_physchem_v4_deg25/
├── taylor_deg25_train.npz
├── taylor_deg25_val.npz
└── taylor_deg25_test.npz
```

Run:

```text
python scripts/data/generate_dataset_physchem_v4.py \
  --degree 25 \
  --n-total 20000 \
  --seed 42 \
  --out-dir data/taylor_data_physchem_v4_deg25 \
  --save-expr-str 1
```

For large-scale experiments, increase --n-total.
Example:
```text
python scripts/data/generate_dataset_physchem_v4.py \
  --degree 25 \
  --n-total 1000000 \
  --seed 42 \
  --out-dir data/taylor_data_physchem_v4_deg25 \
  --save-expr-str 1
```
B. Transformer Interval Dataset
Expected outputs:
```text
data/taylor_data_physchem_v4_interval/
├── taylor_deg25_train.npz
├── taylor_deg25_val.npz
└── taylor_deg25_test.npz
```
Run:
```text
python scripts/data/generate_interval_dataset_physchem_v4.py \
  --degree 25 \
  --n-total 20000 \
  --seed 42 \
  --out-dir data/taylor_data_physchem_v4_interval \
  --save-expr-str 1
```
Notes:

- Dataset generation can be expensive for large n-total.
- For a quick sanity check, start with 20k to 100k.
- For paper-scale experiments, use the full setting described in the manuscript.

## Training

All training scripts are configured through YAML files and environment variables.

Common environment variables:

```text
TAYLOR_CFG  : path to YAML configuration file
TRAIN_NPZ   : training NPZ path
VAL_NPZ     : validation NPZ path
TEST_NPZ    : test NPZ path
OUT_DIR     : output directory
DEVICE      : cuda or cpu
```

ANN
```text
PYTHONPATH=. \
TAYLOR_CFG=configs/taylor_root_ann.yaml \
TRAIN_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_train.npz \
VAL_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_val.npz \
TEST_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_test.npz \
OUT_DIR=results/taylor_nn/ann \
DEVICE=cuda \
python models/taylor_nn/ann.py
```

CPU version:

```text
PYTHONPATH=. \
TAYLOR_CFG=configs/taylor_root_ann.yaml \
TRAIN_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_train.npz \
VAL_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_val.npz \
TEST_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_test.npz \
OUT_DIR=results/taylor_nn/ann_cpu \
DEVICE=cpu \
python models/taylor_nn/ann.py
```

LSTM
```text
PYTHONPATH=. \
TAYLOR_CFG=configs/taylor_root_lstm.yaml \
TRAIN_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_train.npz \
VAL_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_val.npz \
TEST_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_test.npz \
OUT_DIR=results/taylor_nn/lstm \
DEVICE=cuda \
python models/taylor_nn/lstm.py
```

MLP / Anchored MLP
```text
PYTHONPATH=. \
TAYLOR_CFG=configs/taylor_root_mlp.yaml \
TRAIN_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_train.npz \
VAL_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_val.npz \
TEST_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_test.npz \
OUT_DIR=results/taylor_nn/mlp \
DEVICE=cuda \
python models/taylor_nn/mlp.py
```

## Evaluation

The main evaluation script performs K-sweep analysis, baseline comparison, neural model comparison, and residual validation.

Example:

```text
PYTHONPATH=. \
EVAL_CFG=configs/eval_k_sweep.yaml \
OUTDIR=results/runs_k_sweep \
DEVICE=cuda \
python scripts/eval/evaluate_k_sweep.py
```

CPU version:

```
PYTHONPATH=. \
EVAL_CFG=configs/eval_k_sweep.yaml \
OUTDIR=results/runs_k_sweep_cpu \
DEVICE=cpu \
python scripts/eval/evaluate_k_sweep.py
```

Evaluation outputs may include:

```text
results/
├── runs_k_sweep/
│   ├── summary.csv
│   ├── residual_statistics.csv
│   ├── success_by_function_family.csv
│   ├── failure_cases.csv
│   └── plots/
```

## Reproducing the Main Paper Results

The full paper-scale experiment may require large datasets and trained checkpoints.

Recommended reproduction order:

```text
1. Generate root-regression dataset
2. Generate interval-localization dataset
3. Train Taylor-root regressors
4. Train or load Transformer interval predictor
5. Run K-sweep evaluation
6. Run baseline solver comparison
7. Analyze residual and failure distributions
```

Suggested command flow:

```text
# 1. Generate root-regression dataset
python scripts/data/generate_dataset_physchem_v4.py \
  --degree 25 \
  --n-total 1000000 \
  --seed 42 \
  --out-dir data/taylor_data_physchem_v4_deg25 \
  --save-expr-str 1

# 2. Generate interval dataset
python scripts/data/generate_interval_dataset_physchem_v4.py \
  --degree 25 \
  --n-total 1000000 \
  --seed 42 \
  --out-dir data/taylor_data_physchem_v4_interval \
  --save-expr-str 1

# 3. Train ANN
PYTHONPATH=. \
TAYLOR_CFG=configs/taylor_root_ann.yaml \
TRAIN_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_train.npz \
VAL_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_val.npz \
TEST_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_test.npz \
OUT_DIR=results/taylor_nn/ann \
DEVICE=cuda \
python models/taylor_nn/ann.py

# 4. Run evaluation
PYTHONPATH=. \
EVAL_CFG=configs/eval_k_sweep.yaml \
OUTDIR=results/runs_k_sweep \
DEVICE=cuda \
python scripts/eval/evaluate_k_sweep.py
```

## Data and Checkpoints

Large datasets and model checkpoints are not included by default.

Recommended local structure:

```text
data/
├── taylor_data_physchem_v4_deg25/
│   ├── taylor_deg25_train.npz
│   ├── taylor_deg25_val.npz
│   └── taylor_deg25_test.npz
│
└── taylor_data_physchem_v4_interval/
    ├── taylor_deg25_train.npz
    ├── taylor_deg25_val.npz
    └── taylor_deg25_test.npz
```

Recommended checkpoint structure:

```text
checkpoints/
├── interval_transformer_best.pt
├── root_regressor_ann_deg25.pt
├── root_regressor_lstm_deg25.pt
└── root_regressor_mlp_deg25.pt
```
If you use your own dataset or checkpoints, update the YAML files under configs/.

## Failure Analysis

The repository includes utilities and outputs for failure-case analysis.

Failure analysis focuses on cases where the truncated local Taylor representation is insufficient, including:

- nearby singularities
- boundary-sensitive functions
- steep exponential growth
oscillatory mismatch
- local approximation failure outside the effective convergence region

Representative failure montage files may be provided in:

```text
results/figures/
results/pdf/
```

## Related Research Direction

This repository corresponds to our IEEE Access work on Taylor-based root prediction.

A related follow-up research direction explores baseline-aware neural correction and Newton refinement for nonlinear pipe-flow equations. In that framework, neural models act as warm-start accelerators rather than solver replacements. The corrected initializer is refined by Newton iteration to preserve final numerical accuracy.

Together, these works follow the same research principle:

```text
Neural models should improve numerical initialization, prediction, or correction, while classical residual validation or Newton refinement preserves numerical reliability.
```

## Citation

If you use this repository or find it useful for your research, please cite:

```text
@article{jeong2026root,
  author  = {Seokjun Jeong and Yoosoo Oh},
  title   = {A Root Prediction System for Single-Variable Equations with Existing Taylor Polynomials},
  journal = {IEEE Access},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3697368}
}
```

## License

This project is released under the MIT License.
See:

```text
LICENSE
```

## Contact

Seokjun Jeong
Email: wjdtjrwns1109@gmail.com
GitHub: jseongnam
