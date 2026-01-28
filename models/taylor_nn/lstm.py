#!/usr/bin/env python3
# lstm_model_ieee.py
"""
LSTM baseline for your Taylor NPZ dataset.

NPZ keys (confirmed):
  ['coeffs', 'root0', 'root1', 'root2', 'func_id', 'degree',
   'template_str', 'norm_scale', 'expr_str']

We will train:
  input  : coeffs  -> treated as a sequence (N, T, 1)
  target : root0 (default)

Options:
  --target_root {root0,root1,root2}
  --use_extra_roots   (append root1/root2 to the coeff sequence as extra "tokens")

Run:
  python lstm_model_ieee.py --device cuda
"""

from __future__ import annotations
import argparse
import os
import random
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Dataset (your NPZ format)
# -------------------------
class NPZRootDataset(Dataset):
    """
    Loads NPZ with keys:
      coeffs: (N, T)  or (N, T, 1)   (float)
      root0/root1/root2: (N,) or (N,1)
      (other keys ignored)

    Output:
      X: (T, 1) or (T+K, 1) if use_extra_roots
      y: (1,)
    """

    def __init__(
        self,
        npz_path: str,
        target_root: str = "root0",
        use_extra_roots: bool = False,
        x_clip: Optional[float] = None,
    ):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ not found: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        keys = list(data.keys())

        if "coeffs" not in data:
            raise KeyError(f"NPZ must contain key 'coeffs'. keys={keys}")
        if target_root not in data:
            raise KeyError(f"NPZ must contain key '{target_root}'. keys={keys}")

        coeffs = data["coeffs"]  # (N,T) or (N,T,1)
        y = data[target_root]    # (N,) or (N,1)

        # coeffs shape normalize -> (N,T)
        if coeffs.ndim == 3 and coeffs.shape[-1] == 1:
            coeffs = coeffs[:, :, 0]
        elif coeffs.ndim != 2:
            raise ValueError(f"Expected coeffs shape (N,T) or (N,T,1), got {coeffs.shape}")

        # y shape normalize -> (N,1)
        if y.ndim == 1:
            y = y[:, None]
        elif y.ndim == 2 and y.shape[1] == 1:
            pass
        else:
            raise ValueError(f"Expected {target_root} shape (N,) or (N,1), got {y.shape}")

        # Optional: append extra roots as additional sequence tokens
        # (This mimics your chain-feature idea; for pure IEEE LSTM baseline keep it off.)
        if use_extra_roots:
            # if root1/root2 exist, append them (N,2) at end of coeffs time axis
            extras = []
            for rk in ["root1", "root2"]:
                if rk in data:
                    r = data[rk]
                    if r.ndim == 1:
                        r = r[:, None]
                    extras.append(r.astype(np.float32))
            if len(extras) > 0:
                extra_mat = np.concatenate(extras, axis=1)  # (N, K)
                coeffs = np.concatenate([coeffs, extra_mat], axis=1)  # (N, T+K)

        coeffs = coeffs.astype(np.float32)
        y = y.astype(np.float32)

        if x_clip is not None and np.isfinite(x_clip) and x_clip > 0:
            y = np.clip(y, -x_clip, x_clip)

        # Final tensor shapes:
        # X: (N, T, 1), y: (N, 1)
        X = coeffs[:, :, None]

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

        self.keys = keys
        self.target_root = target_root
        self.use_extra_roots = use_extra_roots

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# -------------------------
# Model
# -------------------------
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1)
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]          # (B, hidden)
        return self.head(h_last)  # (B, 1)


# -------------------------
# Eval
# -------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, tol: float = 1e-2) -> Dict[str, float]:
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    n = 0
    hit = 0

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(X)
        diff = pred - y

        mse_sum += float((diff.pow(2)).sum().item())
        mae_sum += float(diff.abs().sum().item())
        n += y.numel()
        hit += int((diff.abs() < tol).sum().item())

    denom = max(n, 1)
    return {
        "mse": mse_sum / denom,
        "mae": mae_sum / denom,
        "hit@tol": hit / denom,
    }


# -------------------------
# Train
# -------------------------
def build_paths(data_dir: str, degree: int) -> Tuple[str, str, str]:
    train_npz = os.path.join(data_dir, f"taylor_deg{degree}_train.npz")
    val_npz   = os.path.join(data_dir, f"taylor_deg{degree}_val.npz")
    test_npz  = os.path.join(data_dir, f"taylor_deg{degree}_test.npz")
    return train_npz, val_npz, test_npz


def train(args) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)

    train_npz, val_npz, test_npz = build_paths(args.data_dir, args.degree)

    train_ds = NPZRootDataset(
        train_npz,
        target_root=args.target_root,
        use_extra_roots=args.use_extra_roots,
        x_clip=args.y_clip,
    )
    val_ds = NPZRootDataset(
        val_npz,
        target_root=args.target_root,
        use_extra_roots=args.use_extra_roots,
        x_clip=args.y_clip,
    )
    test_ds = NPZRootDataset(
        test_npz,
        target_root=args.target_root,
        use_extra_roots=args.use_extra_roots,
        x_clip=args.y_clip,
    )

    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_ld = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print("[DATA]")
    print("  train:", train_npz, "N=", len(train_ds))
    print("  val  :", val_npz,   "N=", len(val_ds))
    print("  test :", test_npz,  "N=", len(test_ds))
    print("  keys :", train_ds.keys)
    print("  target_root:", args.target_root, "use_extra_roots:", args.use_extra_roots)
    X0, y0 = train_ds[0]
    print("[SHAPE] X:", tuple(X0.shape), "y:", tuple(y0.shape))

    model = LSTMRootRegressor(hidden=args.hidden, num_layers=args.layers, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    patience = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for X, y in train_ld:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            pred = model(X)
            loss = loss_fn(pred, y)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            running += float(loss.item())

        if ep % args.eval_every == 0:
            val_metrics = evaluate(model, val_ld, device, tol=args.tol)
            test_metrics = evaluate(model, test_ld, device, tol=args.tol)

            print(
                f"[ep={ep:5d}] "
                f"train_loss={running/max(len(train_ld),1):.6g} | "
                f"val_mse={val_metrics['mse']:.6g} val_mae={val_metrics['mae']:.6g} val_hit={val_metrics['hit@tol']:.4f} | "
                f"test_mse={test_metrics['mse']:.6g} test_mae={test_metrics['mae']:.6g} test_hit={test_metrics['hit@tol']:.4f}"
            )

            if val_metrics["mse"] < best_val:
                best_val = val_metrics["mse"]
                patience = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": vars(args),
                        "best_val_mse": best_val,
                    },
                    args.ckpt,
                )
                print(f"  -> save best (val) to {args.ckpt}")
            else:
                patience += 1
                if patience >= args.early_stop:
                    print("Early stop.")
                    break

    print("Done.")


def build_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data_dir",
        type=str,
        default="/home/seokjun/math_12_3/taylor_data_physchem_v4_deg25",
        help="Directory containing taylor_deg_{degree}_{train,val,test}.npz",
    )
    p.add_argument("--degree", type=int, default=25)

    # dataset -> label
    p.add_argument("--target_root", type=str, default="root0", choices=["root0", "root1", "root2"])
    p.add_argument("--use_extra_roots", action="store_true",
                   help="Append root1/root2 as extra sequence tokens to the coeff sequence.")
    p.add_argument("--y_clip", type=float, default=1e6, help="Clip target y to [-y_clip, y_clip].")

    # model
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)

    # train
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=6000)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--tol", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--early_stop", type=int, default=6000)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--ckpt", type=str, default="lstm_root_best_npz.pt")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
