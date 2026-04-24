#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/taylor_nn/ann.py

✅ YAML(configs/taylor_root_ann.yaml) 기반으로 작동하는 ANN Taylor Root Regressor.
- add_argument/argparse 없이 동작
- 입력: Taylor coefficients (order=25 -> 길이 26)
- 출력: num_roots(=25)개의 root 후보 (B, S)
- Loss: min_residual
    각 샘플 i에서 예측 root 후보 r_{i,s}들에 대해
    Taylor polynomial P_i(r_{i,s})의 |값|을 계산하고,
    min_s |P_i(r_{i,s})| 의 평균을 최소화.

실행:
  python models/taylor_nn/ann.py

환경변수로 override 가능:
  TAYLOR_CFG=/path/config.yaml
  TRAIN_NPZ=/path/train.npz
  VAL_NPZ=/path/val.npz
  TEST_NPZ=/path/test.npz
  OUT_DIR=runs/taylor_root_ann
  DEVICE=cuda (또는 cpu)

추가(선택) 환경변수:
  TAYLOR_ORDER=25
  DATA_DIR_TAYLOR=data/taylor_data_physchem_v4_deg25
    -> TRAIN_NPZ / VAL_NPZ / TEST_NPZ를 직접 주지 않으면
       {DATA_DIR_TAYLOR}/taylor_deg{TAYLOR_ORDER}_{split}.npz 를 기본값으로 사용

NPZ coefficient key 우선순위:
  coeffs > taylor_coefficients > coefficients
"""

from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import yaml  # PyYAML
except Exception as e:
    raise ImportError("PyYAML이 필요합니다. `pip install pyyaml`") from e


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Config dataclasses
# -------------------------
@dataclass
class ModelCfg:
    type: str = "taylor_root_regressor"
    backbone: str = "ann"
    input_feature: str = "taylor_coefficients"
    order: int = 25
    num_roots: int = 25


@dataclass
class ArchCfg:
    hidden_dim: int = 25
    layers: Any = "auto"   # "auto" | int | [int,...]
    activation: str = "tanh"
    dropout: float = 0.0
    bounded_output: bool = False
    root_range: float = 10.0


@dataclass
class TrainCfg:
    batch_size: int = 2048
    epochs: int = 1000
    learning_rate: float = 3e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    eval_every: int = 1
    num_workers: int = 0
    seed: int = 1234
    early_stop: int = 0  # 0이면 비활성


@dataclass
class LossCfg:
    type: str = "min_residual"
    root_clip: float = 0.0           # 0이면 비활성
    root_l2_weight: float = 0.0      # 0이면 비활성
    diversity_weight: float = 0.0    # 0이면 비활성
    diversity_margin: float = 1e-2


@dataclass
class FullCfg:
    model: ModelCfg
    architecture: ArchCfg
    training: TrainCfg
    loss: LossCfg


def _get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_config(yaml_path: str) -> FullCfg:
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    m = ModelCfg(
        type=_get(raw, "model.type", "taylor_root_regressor"),
        backbone=_get(raw, "model.backbone", "ann"),
        input_feature=_get(raw, "model.input.feature", "taylor_coefficients"),
        order=int(_get(raw, "model.input.order", 25)),
        num_roots=int(_get(raw, "model.output.num_roots", 25)),
    )

    a = ArchCfg(
        hidden_dim=int(_get(raw, "architecture.hidden_dim", 25)),
        layers=_get(raw, "architecture.layers", "auto"),
        activation=str(_get(raw, "architecture.activation", "tanh")),
        dropout=float(_get(raw, "architecture.dropout", 0.0)),
        bounded_output=bool(_get(raw, "architecture.bounded_output", False)),
        root_range=float(_get(raw, "architecture.root_range", 10.0)),
    )

    t = TrainCfg(
        batch_size=int(_get(raw, "training.batch_size", 2048)),
        epochs=int(_get(raw, "training.epochs", 1000)),
        learning_rate=float(_get(raw, "training.learning_rate", 3e-5)),
        weight_decay=float(_get(raw, "training.weight_decay", 0.0)),
        grad_clip=float(_get(raw, "training.grad_clip", 1.0)),
        eval_every=int(_get(raw, "training.eval_every", 1)),
        num_workers=int(_get(raw, "training.num_workers", 0)),
        seed=int(_get(raw, "training.seed", 1234)),
        early_stop=int(_get(raw, "training.early_stop", 0)),
    )

    l = LossCfg(
        type=str(_get(raw, "loss.type", "min_residual")),
        root_clip=float(_get(raw, "loss.root_clip", 0.0)),
        root_l2_weight=float(_get(raw, "loss.root_l2_weight", 0.0)),
        diversity_weight=float(_get(raw, "loss.diversity_weight", 0.0)),
        diversity_margin=float(_get(raw, "loss.diversity_margin", 1e-2)),
    )

    # env override: degree sweep 대응
    order_env = os.environ.get("TAYLOR_ORDER", "").strip()
    if order_env:
        order = int(order_env)
        m.order = order

    return FullCfg(model=m, architecture=a, training=t, loss=l)


# -------------------------
# Dataset (coeff only)
# -------------------------
def _pick_coeff_key(keys: List[str]) -> str:
    cand = ["coeffs", "taylor_coefficients", "coefficients"]
    for c in cand:
        if c in keys:
            return c
    raise KeyError(f"NPZ에 계수 키가 없습니다. 기대 키: {cand}, 실제 키: {keys}")


class TaylorCoeffDataset(Dataset):
    """
    NPZ에서 Taylor 계수만 읽는다. (label 없음)
    order=25면 D=26 (a0..a25)
    """
    def __init__(self, npz_path: str, order: int):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(npz_path)

        z = np.load(npz_path, mmap_mode="r", allow_pickle=True)
        keys = list(z.keys())
        ck = _pick_coeff_key(keys)

        X = np.array(z[ck])
        if X.ndim == 3:
            X = np.squeeze(X)
        if X.ndim != 2:
            raise ValueError(f"coeffs shape={X.shape} 지원 불가. (N,D)만 지원")

        if X.shape[1] == order:
            X = np.concatenate([np.zeros((X.shape[0], 1), dtype=X.dtype), X], axis=1)
        elif X.shape[1] != order + 1:
            raise ValueError(f"order={order}인데 coeff dim={X.shape[1]} 입니다. 기대: {order+1} (또는 {order})")

        self.X = X.astype(np.float32)
        self.keys = keys
        self.coeff_key = ck

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx])


# -------------------------
# Normalization
# -------------------------
def np_minmax_chunked(arr: np.ndarray, chunk: int = 200_000) -> Tuple[np.ndarray, np.ndarray]:
    n = arr.shape[0]
    mn = None
    mx = None
    for i in range(0, n, chunk):
        sl = arr[i:i+chunk]
        sl_mn = np.min(sl, axis=0)
        sl_mx = np.max(sl, axis=0)
        if mn is None:
            mn, mx = sl_mn, sl_mx
        else:
            mn = np.minimum(mn, sl_mn)
            mx = np.maximum(mx, sl_mx)
    return mn.astype(np.float32), mx.astype(np.float32)


def minmax_to_minus1_1(x: np.ndarray, mn: np.ndarray, mx: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    den = np.maximum(mx - mn, eps)
    return (2.0 * (x - mn) / den - 1.0).astype(np.float32)


# -------------------------
# Polynomial eval (Horner)
# -------------------------
def poly_eval_horner(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    coeffs: (B, D) with D=order+1
    x:      (B, S)
    return: (B, S) P(x)
    """
    y = coeffs[:, -1].unsqueeze(1).expand_as(x)
    for k in range(coeffs.shape[1] - 2, -1, -1):
        y = y * x + coeffs[:, k].unsqueeze(1)
    return y


# -------------------------
# Model (ANN)
# -------------------------
def get_activation(name: str) -> nn.Module:
    n = name.lower()
    if n == "tanh":
        return nn.Tanh()
    if n == "relu":
        return nn.ReLU(inplace=True)
    if n == "gelu":
        return nn.GELU()
    if n in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def resolve_layers_auto(in_dim: int, hidden_dim: int) -> List[int]:
    if in_dim <= 64:
        n_hidden = 3
    elif in_dim <= 256:
        n_hidden = 4
    else:
        n_hidden = 5
    return [hidden_dim] * n_hidden


class ANNRootRegressor(nn.Module):
    """
    입력: (B, D)
    출력: (B, S=num_roots)
    """
    def __init__(self, in_dim: int, num_roots: int, arch: ArchCfg):
        super().__init__()
        self.arch = arch
        self.num_roots = num_roots

        if isinstance(arch.layers, str) and arch.layers.lower() == "auto":
            hlist = resolve_layers_auto(in_dim, arch.hidden_dim)
        elif isinstance(arch.layers, int):
            hlist = [arch.hidden_dim] * int(arch.layers)
        elif isinstance(arch.layers, (list, tuple)):
            hlist = [int(x) for x in arch.layers]
        else:
            raise ValueError(f"Unsupported architecture.layers={arch.layers}")

        act = get_activation(arch.activation)

        layers: List[nn.Module] = []
        prev = in_dim
        for h in hlist:
            layers.append(nn.Linear(prev, h))
            layers.append(act)
            if arch.dropout > 0:
                layers.append(nn.Dropout(p=arch.dropout))
            prev = h

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(prev, num_roots)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        roots = self.head(h)
        if self.arch.bounded_output:
            roots = torch.tanh(roots) * float(self.arch.root_range)
        return roots


# -------------------------
# Loss
# -------------------------
def min_residual_loss(coeffs: torch.Tensor, roots: torch.Tensor, loss_cfg: LossCfg) -> torch.Tensor:
    """
    coeffs: (B, D)
    roots:  (B, S)
    """
    x = roots
    if loss_cfg.root_clip and loss_cfg.root_clip > 0:
        x = torch.clamp(x, -loss_cfg.root_clip, loss_cfg.root_clip)

    p = poly_eval_horner(coeffs, x)              # (B,S)
    residual = torch.abs(p)                      # (B,S)
    min_res = torch.min(residual, dim=1).values  # (B,)
    loss = min_res.mean()

    if loss_cfg.root_l2_weight and loss_cfg.root_l2_weight > 0:
        loss = loss + float(loss_cfg.root_l2_weight) * (roots ** 2).mean()

    if loss_cfg.diversity_weight and loss_cfg.diversity_weight > 0:
        r = roots
        diff = torch.abs(r.unsqueeze(2) - r.unsqueeze(1))  # (B,S,S)
        mask = 1.0 - torch.eye(r.shape[1], device=r.device).unsqueeze(0)
        diff = diff * mask
        margin = float(loss_cfg.diversity_margin)
        penalty = torch.relu(margin - diff) * mask
        loss = loss + float(loss_cfg.diversity_weight) * penalty.mean()

    return loss


# -------------------------
# Eval
# -------------------------
@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device, loss_cfg: LossCfg) -> float:
    model.eval()
    total = 0.0
    n = 0
    for xb in loader:
        xb = xb.to(device, non_blocking=True)
        roots = model(xb)
        loss = min_residual_loss(xb, roots, loss_cfg)
        total += float(loss.item()) * xb.shape[0]
        n += xb.shape[0]
    return total / max(n, 1)


# -------------------------
# Train (YAML)
# -------------------------
def train_from_yaml(
    cfg_path: str,
    train_npz: str,
    val_npz: str,
    test_npz: Optional[str],
    out_dir: str,
    device_str: str,
) -> None:
    cfg = load_config(cfg_path)

    if cfg.model.backbone.lower() != "ann":
        raise ValueError(f"이 ann.py는 backbone=ann만 처리합니다. 현재: {cfg.model.backbone}")
    if cfg.loss.type.lower() != "min_residual":
        raise ValueError(f"이 구현은 loss=min_residual만 처리합니다. 현재: {cfg.loss.type}")

    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg.training.seed)
    device = torch.device(device_str)

    train_ds_raw = TaylorCoeffDataset(train_npz, order=cfg.model.order)
    val_ds_raw   = TaylorCoeffDataset(val_npz,   order=cfg.model.order)
    test_ds_raw  = TaylorCoeffDataset(test_npz,  order=cfg.model.order) if test_npz else None

    # normalize using train
    x_mn, x_mx = np_minmax_chunked(train_ds_raw.X)
    train_X = minmax_to_minus1_1(train_ds_raw.X, x_mn, x_mx)
    val_X   = minmax_to_minus1_1(val_ds_raw.X,   x_mn, x_mx)
    test_X  = minmax_to_minus1_1(test_ds_raw.X,  x_mn, x_mx) if test_ds_raw else None

    class _DS(Dataset):
        def __init__(self, X2d: np.ndarray):
            self.X2d = X2d
        def __len__(self): return self.X2d.shape[0]
        def __getitem__(self, i: int):
            return torch.from_numpy(self.X2d[i])

    train_ds = _DS(train_X)
    val_ds   = _DS(val_X)
    test_ds  = _DS(test_X) if test_X is not None else None

    train_ld = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_ld = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == "cuda"),
    ) if test_ds is not None else None

    in_dim = train_X.shape[1]
    model = ANNRootRegressor(in_dim=in_dim, num_roots=cfg.model.num_roots, arch=cfg.architecture).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    ckpt_path = os.path.join(out_dir, "best.pt")
    scaler_path = os.path.join(out_dir, "scaler.json")
    cfg_dump_path = os.path.join(out_dir, "config_resolved.json")

    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump({"x_min": x_mn.tolist(), "x_max": x_mx.tolist()}, f, ensure_ascii=False, indent=2)

    with open(cfg_dump_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": cfg.model.__dict__,
            "architecture": cfg.architecture.__dict__,
            "training": cfg.training.__dict__,
            "loss": cfg.loss.__dict__,
        }, f, ensure_ascii=False, indent=2)

    print(f"[CONFIG] {cfg_path}")
    print(f"[DATA] train={len(train_ds)} val={len(val_ds)} test={(len(test_ds) if test_ds else 0)}")
    print(f"[NPZ] train coeff_key={train_ds_raw.coeff_key}, keys={train_ds_raw.keys}")
    print(f"[SHAPE] coeff_dim(D)={in_dim}, num_roots={cfg.model.num_roots}")
    print(f"[MODEL] hidden_dim={cfg.architecture.hidden_dim}, layers={cfg.architecture.layers}, act={cfg.architecture.activation}")
    print(f"[OUT] {out_dir}")

    best_val = float("inf")
    patience = 0

    for ep in range(1, cfg.training.epochs + 1):
        model.train()
        running = 0.0
        steps = 0

        for xb in train_ld:
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            roots = model(xb)
            loss = min_residual_loss(xb, roots, cfg.loss)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            if cfg.training.grad_clip and cfg.training.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            opt.step()

            running += float(loss.item())
            steps += 1

        if ep % max(cfg.training.eval_every, 1) == 0:
            tr_loss = running / max(steps, 1)
            val_loss = eval_epoch(model, val_ld, device, cfg.loss)
            print(f"[ep={ep:4d}] train_loss={tr_loss:.6g}  val_loss={val_loss:.6g}")

            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "in_dim": in_dim,
                        "num_roots": cfg.model.num_roots,
                        "arch": cfg.architecture.__dict__,
                        "loss": cfg.loss.__dict__,
                        "scaler_json": scaler_path,
                        "config_json": cfg_dump_path,
                        "best_val": best_val,
                    },
                    ckpt_path,
                )
                print(f"  -> save best: {ckpt_path}")
            else:
                if cfg.training.early_stop and cfg.training.early_stop > 0:
                    patience += 1
                    if patience >= cfg.training.early_stop:
                        print("[EARLY STOP]")
                        break

    if test_ld is not None and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["state_dict"])
        test_loss = eval_epoch(model, test_ld, device, cfg.loss)
        print(f"[TEST] loss(min_residual)={test_loss:.6g}")

    print("[DONE]")


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, default)).strip())
    except Exception:
        return int(default)


def _default_taylor_data_dir(degree: int) -> str:
    return f"data/taylor_data_physchem_v4_deg{degree}"


def _default_taylor_npz(data_dir: str, degree: int, split: str) -> str:
    return f"{data_dir}/taylor_deg{degree}_{split}.npz"


# -------------------------
# Entrypoint (No argparse)
# -------------------------
def main() -> None:
    import os
    import torch
    from src.path_utils import find_repo_root, resolve_repo_path, resolve_device

    repo = find_repo_root(__file__)

    degree      = _env_int("TAYLOR_ORDER", 25)
    default_cfg = "configs/taylor_root_ann.yaml"
    data_dir    = os.environ.get("DATA_DIR_TAYLOR", _default_taylor_data_dir(degree))
    default_train = _default_taylor_npz(data_dir, degree, "train")
    default_val   = _default_taylor_npz(data_dir, degree, "val")
    default_test  = _default_taylor_npz(data_dir, degree, "test")
    default_out   = f"results/taylor_nn/ann/deg{degree}"
    default_dev   = "auto"

    cfg_path   = os.environ.get("TAYLOR_CFG", default_cfg)
    train_npz  = os.environ.get("TRAIN_NPZ",  default_train)
    val_npz    = os.environ.get("VAL_NPZ",    default_val)
    test_npz   = os.environ.get("TEST_NPZ",   default_test)
    out_dir    = os.environ.get("OUT_DIR",    default_out)
    device_str = os.environ.get("DEVICE",     default_dev)

    cfg_path_p  = resolve_repo_path(cfg_path, repo)
    train_npz_p = resolve_repo_path(train_npz, repo)
    val_npz_p   = resolve_repo_path(val_npz, repo)
    test_npz_p  = resolve_repo_path(test_npz, repo)  # 빈 문자열이면 None 처리됨
    out_dir_p   = resolve_repo_path(out_dir, repo)

    device_str = resolve_device(device_str)

    train_from_yaml(
        cfg_path=str(cfg_path_p),
        train_npz=str(train_npz_p),
        val_npz=str(val_npz_p),
        test_npz=(str(test_npz_p) if test_npz_p is not None else None),
        out_dir=str(out_dir_p),
        device_str=device_str,
    )


if __name__ == "__main__":
    main()
