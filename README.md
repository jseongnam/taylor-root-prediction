# Taylor-based Root Prediction

This repository provides the official implementation of the paper:

**"Taylor-based Root Prediction for Efficient Initialization of Nonlinear Solvers"**  
Seokjun Jeong, et al.  

---

## 📌 Overview

Finding accurate initial guesses for roots of nonlinear equations is a critical step for iterative solvers such as Newton–Raphson methods.  
This work proposes a **Taylor expansion–based root prediction method**, which estimates the location of a nearby root by leveraging local derivative information around a reference point.

The key idea is to use truncated Taylor series expansions to predict root displacement, reducing the number of iterations and improving convergence stability compared to conventional initialization strategies.

This repository contains:
- Root prediction algorithms based on Taylor expansion
- Baseline methods for comparison
- Scripts to reproduce all experiments reported in the paper

---

## 🧠 Method Summary

Given a nonlinear function \( f(x) \), we approximate its behavior around a reference point \( x_0 \) using a Taylor expansion:

\[
f(x) \approx f(x_0) + f'(x_0)(x - x_0) + \frac{1}{2}f''(x_0)(x - x_0)^2 + \cdots
\]

By solving the truncated approximation, we obtain a **predicted root location** that serves as an efficient initialization for iterative solvers.

Different truncation orders and derivative usage strategies are evaluated in the paper.

## 📁 Repository Structure

taylor-root-prediction/
├─ README.md
├─ requirements.txt
├─ src/
│ ├─ taylor_predictor.py
│ ├─ baselines.py
│ └─ solver.py
├─ scripts/
│ ├─ reproduce_main_results.sh
│ └─ run_single_experiment.py
├─ notebooks/
│ └─ analysis.ipynb
├─ data/
│ └─ README.md
└─ results/

---

## ⚙️ Environment Setup

```bash
conda create -n taylor-root python=3.9
conda activate taylor-root
pip install -r requirements.txt