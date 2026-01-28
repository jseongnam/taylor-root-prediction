#!/usr/bin/env python3
# mdpi_fnn_root_npz.py
"""
MDPI (2021) "A Neural Network-Based Approach for Approximating Arbitrary Roots of Polynomials"
스타일의 Shallow FNN으로 NPZ(root0/root1/root2) 회귀 테스트.

논문 포인트 반영:
- Shallow FNN: input -> hidden(10, tanh) -> linear output
- min-max normalization to [-1, 1] (train 기준)
- (원 논문은 LMA/PSO지만) 여기서는 Adam + (optional) LBFGS로 대체

지원:
- NPZ keys: coeffs, root0, root1, root2, ...
- targets: root0 단일 or root0,root1,root2 다중 출력

예)
python mdpi_fnn_root_npz.py \
  --train_npz /home/seokjun/math_12_3/.../taylor_deg_25_train.npz \
  --val_npz   /home/seokjun/math_12_3/.../taylor_deg_25_val.npz \
  --test_npz  /home/seokjun/math_12_3/.../taylor_deg_25_test.npz \
  --targets root0 \
  --device cuda

또는 (3개 동시)
  --targets root0,root1,root2
"""

from __future__ import annotations
import argparse
import os
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 1234) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_targets(s: str) -> List[str]:
    s = s.strip()
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s]


def resolve_triplet_path(p: str) -> Tuple[str, str, str]:
    """
    사용자가 한 번에 'train,val,test' 같은 문자열을 줬을 때 최대한 복원.
    - 만약 p가 디렉토리면 그 안에서 taylor_deg_*_{train,val,test}.npz를 찾는 건 위험하니(패턴 다양)
      여기서는 "명시 경로 3개"를 권장.
    - 그래도 사용자 편의로 p에 'train,val,test'가 들어가면 치환해 줌.
    """
    if "train,val,test" in p:
        train = p.replace("train,val,test", "train")
        val   = p.replace("train,val,test", "val")
        test  = p.replace("train,val,test", "test")
        return train, val, test
    return p, "", ""


def np_minmax_chunked(arr: np.ndarray, chunk: int = 200_000) -> Tuple[np.ndarray, np.ndarray]:
    """
    큰 배열도 버티도록 chunk 단위로 min/max 계산.
    arr: (N, D)
    """
    n = arr.shape[0]
    mn = None
    mx = None
    for i in range(0, n, chunk):
        sl = arr[i:i+chunk]
        sl_mn = np.min(sl, axis=0)
        sl_mx = np.max(sl, axis=0)
        if mn is None:
            mn = sl_mn
            mx = sl_mx
        else:
            mn = np.minimum(mn, sl_mn)
            mx = np.maximum(mx, sl_mx)
    return mn.astype(np.float32), mx.astype(np.float32)


def minmax_to_minus1_1(x: np.ndarray, mn: np.ndarray, mx: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    den = np.maximum(mx - mn, eps)
    return (2.0 * (x - mn) / den - 1.0).astype(np.float32)


def inv_minmax_from_minus1_1(x_scaled: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    # x_scaled in [-1,1] -> original
    return ((x_scaled + 1.0) * 0.5 * (mx - mn) + mn).astype(np.float32)


# -------------------------
# Dataset
# -------------------------
class NPZRootDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        targets: List[str],
        x_key: str = "coeffs",
        mmap: bool = True,
    ):
        self.npz_path = npz_path
        self.targets = targets
        self.x_key = x_key

        if not os.path.exists(npz_path):
            raise FileNotFoundError(npz_path)

        self.z = np.load(npz_path, mmap_mode="r" if mmap else None)
        self.keys = list(self.z.keys())

        if x_key not in self.z:
            raise KeyError(f"NPZ must contain key '{x_key}'. keys={self.keys}")

        X = self.z[x_key]
        # 허용: (N,D), (N,D,1), (N,1,D) 등
        X = np.array(X)
        if X.ndim == 3:
            X = np.squeeze(X)
        if X.ndim != 2:
            raise ValueError(f"Unsupported coeffs shape={X.shape}. Expect 2D (N,D).")

        ys = []
        for t in targets:
            if t not in self.z:
                raise KeyError(f"NPZ must contain target '{t}'. keys={self.keys}")
            y = np.array(self.z[t])
            y = np.squeeze(y)
            if y.ndim == 1:
                y = y[:, None]
            if y.ndim != 2 or y.shape[1] != 1:
                raise ValueError(f"Target '{t}' shape={y.shape} not supported. Expect (N,) or (N,1).")
            ys.append(y.astype(np.float32))

        Y = np.concatenate(ys, axis=1)  # (N, K)

        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"N mismatch: X={X.shape}, Y={Y.shape}")

        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])


# -------------------------
# Model (MDPI-style shallow FNN)
# -------------------------
class ShallowFNN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.Tanh()  # tansig
        self.fc2 = nn.Linear(hidden, out_dim)  # linear output

        # init (안 해도 되지만 안정성 위해)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# -------------------------
# Train/Eval
# -------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             y_mn: np.ndarray, y_mx: np.ndarray) -> Dict[str, float]:
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)  # scaled space
        # unscale to original space for metrics
        pred_np = pred.detach().cpu().numpy()
        yb_np = yb.detach().cpu().numpy()

        pred_org = inv_minmax_from_minus1_1(pred_np, y_mn, y_mx)
        y_org    = inv_minmax_from_minus1_1(yb_np,  y_mn, y_mx)

        diff = pred_org - y_org
        mse_sum += float(np.sum(diff * diff))
        mae_sum += float(np.sum(np.abs(diff)))
        n += diff.size

    return {
        "mse": mse_sum / max(n, 1),
        "mae": mae_sum / max(n, 1),
    }


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)

    targets = parse_targets(args.targets)

    train_ds_raw = NPZRootDataset(args.train_npz, targets=targets, x_key="coeffs", mmap=True)
    val_ds_raw   = NPZRootDataset(args.val_npz,   targets=targets, x_key="coeffs", mmap=True)
    test_ds_raw  = NPZRootDataset(args.test_npz,  targets=targets, x_key="coeffs", mmap=True)

    # train 기준 min/max (논문 스타일)
    x_mn, x_mx = np_minmax_chunked(train_ds_raw.X, chunk=args.minmax_chunk)
    y_mn, y_mx = np_minmax_chunked(train_ds_raw.Y, chunk=args.minmax_chunk)

    # scale to [-1,1]
    train_X = minmax_to_minus1_1(train_ds_raw.X, x_mn, x_mx)
    train_Y = minmax_to_minus1_1(train_ds_raw.Y, y_mn, y_mx)
    val_X   = minmax_to_minus1_1(val_ds_raw.X,   x_mn, x_mx)
    val_Y   = minmax_to_minus1_1(val_ds_raw.Y,   y_mn, y_mx)
    test_X  = minmax_to_minus1_1(test_ds_raw.X,  x_mn, x_mx)
    test_Y  = minmax_to_minus1_1(test_ds_raw.Y,  y_mn, y_mx)

    # scaled datasets
    class _ArrDS(Dataset):
        def __init__(self, X, Y):
            self.X = X
            self.Y = Y
        def __len__(self): return self.X.shape[0]
        def __getitem__(self, i):
            return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])

    train_ds = _ArrDS(train_X, train_Y)
    val_ds   = _ArrDS(val_X,   val_Y)
    test_ds  = _ArrDS(test_X,  test_Y)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    in_dim = train_X.shape[1]
    out_dim = train_Y.shape[1]

    model = ShallowFNN(in_dim=in_dim, out_dim=out_dim, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best = float("inf")
    patience = 0

    print(f"[DATA] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"[SHAPE] X_dim={in_dim} Y_dim={out_dim} targets={targets}")
    print(f"[NPZ keys] train keys={train_ds_raw.keys}")

    # save scaler
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    scaler_path = args.ckpt + ".scaler.json"
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump({
            "x_min": x_mn.tolist(),
            "x_max": x_mx.tolist(),
            "y_min": y_mn.tolist(),
            "y_max": y_mx.tolist(),
            "targets": targets,
        }, f, ensure_ascii=False)

    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for xb, yb in train_ld:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            running += float(loss.item())

        if ep % args.eval_every == 0:
            val_metrics = evaluate(model, val_ld, device, y_mn=y_mn, y_mx=y_mx)
            val_mse = val_metrics["mse"]
            print(f"[ep={ep:5d}] train_loss={running/max(len(train_ld),1):.6g}  "
                  f"val_mse={val_mse:.6g} val_mae={val_metrics['mae']:.6g}")

            if val_mse < best:
                best = val_mse
                patience = 0
                torch.save({
                    "model": model.state_dict(),
                    "in_dim": in_dim,
                    "out_dim": out_dim,
                    "hidden": args.hidden,
                    "targets": targets,
                    "scaler_json": scaler_path,
                }, args.ckpt)
                print(f"  -> save best: {args.ckpt}")
            else:
                patience += 1
                if patience >= args.early_stop:
                    print("Early stop.")
                    break

    # optional LBFGS refine (근사-LMA 느낌)
    if args.lbfgs_iters > 0:
        print(f"[LBFGS] refining with iters={args.lbfgs_iters}, subset={args.lbfgs_subset}")
        model.train()
        # subset pick
        n_sub = min(args.lbfgs_subset, train_X.shape[0]) if args.lbfgs_subset > 0 else train_X.shape[0]
        xb = torch.from_numpy(train_X[:n_sub]).to(device)
        yb = torch.from_numpy(train_Y[:n_sub]).to(device)
        opt2 = torch.optim.LBFGS(model.parameters(), max_iter=args.lbfgs_iters, line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if torch.isfinite(loss):
                loss.backward()
            return loss

        try:
            opt2.step(closure)
        except RuntimeError as e:
            print(f"[LBFGS] RuntimeError: {e}")

    # load best and test
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    test_metrics = evaluate(model, test_ld, device, y_mn=y_mn, y_mx=y_mx)
    print(f"[TEST] mse={test_metrics['mse']:.6g} mae={test_metrics['mae']:.6g}")
    print(f"[DONE] scaler saved: {scaler_path}")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--train_npz", type=str, default="/home/seokjun/math_12_3/taylor_data_physchem_v4_deg25/taylor_deg25_train.npz")
    p.add_argument("--val_npz",   type=str, default="/home/seokjun/math_12_3/taylor_data_physchem_v4_deg25/taylor_deg25_val.npz")
    p.add_argument("--test_npz",  type=str, default="/home/seokjun/math_12_3/taylor_data_physchem_v4_deg25/taylor_deg25_test.npz")

    p.add_argument("--targets", type=str, default="root0",
                   help="e.g., root0  or  root0,root1,root2")

    # MDPI-style: hidden=10, tanh
    p.add_argument("--hidden", type=int, default=10)

    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=6000)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--early_stop", type=int, default=300)

    # minmax chunk
    p.add_argument("--minmax_chunk", type=int, default=200_000)

    # optional LBFGS refine
    p.add_argument("--lbfgs_iters", type=int, default=0,
                   help="0이면 비활성. (예: 50~200)")
    p.add_argument("--lbfgs_subset", type=int, default=50_000,
                   help="LBFGS에 쓸 train 샘플 수(0이면 전체).")

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--ckpt", type=str, default="mdpi_fnn_best.pt")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
