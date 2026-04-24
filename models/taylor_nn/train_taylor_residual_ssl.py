#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/taylor_nn/train_taylor_residual_ssl.py

Residual-only unlabeled training for Taylor root prediction.
- YAML(configs/taylor_root_residual_ssl.yaml) 기반
- argparse 없이 환경변수로 override 가능
- 입력: Taylor coefficients only (label/root GT 불필요)
- 출력: num_roots 개의 root candidates
- Loss: residual-only (min_residual / softmin_residual)
    각 샘플 i에 대해 예측 후보 roots r_{i,s}를 만들고,
    Taylor polynomial P_i(r_{i,s}) 의 residual을 최소화한다.
- 선택 옵션:
    * coefficient noise augmentation
    * prediction consistency loss (원본 vs noisy coeff)
    * root L2 regularization
    * diversity regularization
    * softmin residual (min보다 조금 더 부드러운 surrogate)

실행 예:
  TAYLOR_CFG=configs/taylor_root_residual_ssl.yaml \
  TRAIN_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_train.npz \
  VAL_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_val.npz \
  TEST_NPZ=data/taylor_data_physchem_v4_deg25/taylor_deg25_test.npz \
  OUT_DIR=results/taylor_nn/residual_ssl \
  DEVICE=cuda \
  python models/taylor_nn/train_taylor_residual_ssl.py

환경변수:
  TAYLOR_CFG, TRAIN_NPZ, VAL_NPZ, TEST_NPZ, OUT_DIR, DEVICE

NPZ coefficient key 우선순위:
  coeffs > taylor_coefficients > coefficients
"""

from __future__ import annotations

import os
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import yaml
except Exception as e:
    raise ImportError("PyYAML이 필요합니다. `pip install pyyaml`") from e

# -------------------------
# Repo/path helpers (standalone; no src import required)
# -------------------------
from pathlib import Path

def find_repo_root(start_file: str) -> Path:
    start = Path(start_file).resolve()
    cur = start.parent
    for _ in range(12):
        if (cur / ".git").exists() or (cur / "configs").exists():
            return cur
        if cur.parent == cur:
                break
        cur = cur.parent
    return start.parent

def resolve_repo_path(p: str, repo_root: Path):
    s = str(p).strip()
    if not s:
        return None
    pp = Path(s)
    if pp.is_absolute():
        return pp
    return (repo_root / pp).resolve()

def resolve_device(device_str: str) -> str:
    s = str(device_str).strip().lower()
    if s in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if s.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return s



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
    layers: Any = "auto"
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
    early_stop: int = 0
    amp: bool = False


@dataclass
class LossCfg:
    type: str = "min_residual"     # min_residual | softmin_residual
    root_clip: float = 0.0
    root_l2_weight: float = 0.0
    diversity_weight: float = 0.0
    diversity_margin: float = 1e-2
    softmin_tau: float = 0.05
    coeff_noise_std: float = 0.0
    coeff_noise_prob: float = 1.0
    consistency_weight: float = 0.0
    consistency_mode: str = "roots"  # roots | residual
    residual_transform: str = "abs"  # abs | logabs | sq
    min_root_separation: float = 0.0


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
        amp=bool(_get(raw, "training.amp", False)),
    )

    l = LossCfg(
        type=str(_get(raw, "loss.type", "min_residual")),
        root_clip=float(_get(raw, "loss.root_clip", 0.0)),
        root_l2_weight=float(_get(raw, "loss.root_l2_weight", 0.0)),
        diversity_weight=float(_get(raw, "loss.diversity_weight", 0.0)),
        diversity_margin=float(_get(raw, "loss.diversity_margin", 1e-2)),
        softmin_tau=float(_get(raw, "loss.softmin_tau", 0.05)),
        coeff_noise_std=float(_get(raw, "loss.coeff_noise_std", 0.0)),
        coeff_noise_prob=float(_get(raw, "loss.coeff_noise_prob", 1.0)),
        consistency_weight=float(_get(raw, "loss.consistency_weight", 0.0)),
        consistency_mode=str(_get(raw, "loss.consistency_mode", "roots")),
        residual_transform=str(_get(raw, "loss.residual_transform", "abs")),
        min_root_separation=float(_get(raw, "loss.min_root_separation", 0.0)),
    )

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


def torch_minmax_to_minus1_1(x: torch.Tensor, mn: torch.Tensor, mx: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    den = torch.clamp(mx - mn, min=eps)
    return 2.0 * (x - mn) / den - 1.0


def torch_inv_minmax_from_minus1_1(x_scaled: torch.Tensor, mn: torch.Tensor, mx: torch.Tensor) -> torch.Tensor:
    return (x_scaled + 1.0) * 0.5 * (mx - mn) + mn


# -------------------------
# Polynomial eval (Horner)
# -------------------------
def poly_eval_horner(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    coeffs: (B, D)
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

        layers: List[nn.Module] = []
        prev = in_dim
        for h in hlist:
            layers.append(nn.Linear(prev, h))
            layers.append(get_activation(arch.activation))
            if arch.dropout > 0:
                layers.append(nn.Dropout(p=arch.dropout))
            prev = h

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(prev, num_roots)

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
# Loss helpers
# -------------------------
def transform_residual(p: torch.Tensor, mode: str) -> torch.Tensor:
    mode = str(mode).lower()
    ap = torch.abs(p)
    if mode == "abs":
        return ap
    if mode == "logabs":
        return torch.log1p(ap)
    if mode == "sq":
        return p * p
    raise ValueError(f"Unsupported residual_transform: {mode}")


def pairwise_diversity_penalty(roots: torch.Tensor, margin: float) -> torch.Tensor:
    diff = torch.abs(roots.unsqueeze(2) - roots.unsqueeze(1))
    mask = 1.0 - torch.eye(roots.shape[1], device=roots.device).unsqueeze(0)
    penalty = torch.relu(float(margin) - diff) * mask
    return penalty.mean()


def min_separation_sort_penalty(roots: torch.Tensor, min_sep: float) -> torch.Tensor:
    if min_sep <= 0:
        return roots.new_tensor(0.0)
    r_sorted, _ = torch.sort(roots, dim=1)
    gaps = r_sorted[:, 1:] - r_sorted[:, :-1]
    return torch.relu(float(min_sep) - gaps).mean()


def residual_core_loss(coeffs: torch.Tensor, roots: torch.Tensor, loss_cfg: LossCfg) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    x = roots
    if loss_cfg.root_clip and loss_cfg.root_clip > 0:
        x = torch.clamp(x, -loss_cfg.root_clip, loss_cfg.root_clip)

    p = poly_eval_horner(coeffs, x)
    residual = transform_residual(p, loss_cfg.residual_transform)

    loss_type = str(loss_cfg.type).lower()
    if loss_type == "min_residual":
        main = torch.min(residual, dim=1).values.mean()
    elif loss_type == "softmin_residual":
        tau = max(float(loss_cfg.softmin_tau), 1e-8)
        main = (-tau * torch.logsumexp(-residual / tau, dim=1)).mean()
    else:
        raise ValueError(f"Unsupported loss.type={loss_cfg.type}")

    reg_root_l2 = roots.new_tensor(0.0)
    if loss_cfg.root_l2_weight and loss_cfg.root_l2_weight > 0:
        reg_root_l2 = (roots ** 2).mean() * float(loss_cfg.root_l2_weight)

    reg_div = roots.new_tensor(0.0)
    if loss_cfg.diversity_weight and loss_cfg.diversity_weight > 0:
        reg_div = pairwise_diversity_penalty(roots, float(loss_cfg.diversity_margin)) * float(loss_cfg.diversity_weight)

    reg_sep = roots.new_tensor(0.0)
    if loss_cfg.min_root_separation and loss_cfg.min_root_separation > 0:
        reg_sep = min_separation_sort_penalty(roots, float(loss_cfg.min_root_separation))

    total = main + reg_root_l2 + reg_div + reg_sep
    aux = {
        "main": main.detach(),
        "root_l2": reg_root_l2.detach(),
        "div": reg_div.detach(),
        "sep": reg_sep.detach(),
        "residual_mean": residual.mean().detach(),
        "residual_min_mean": torch.min(residual, dim=1).values.mean().detach(),
    }
    return total, aux


def maybe_add_coeff_noise(x: torch.Tensor, std: float, prob: float) -> torch.Tensor:
    if std <= 0:
        return x
    if prob < 1.0:
        mask = (torch.rand((x.shape[0], 1), device=x.device) < prob).float()
    else:
        mask = 1.0
    noise = torch.randn_like(x) * float(std)
    return x + noise * mask


def consistency_loss(
    model: nn.Module,
    xb: torch.Tensor,
    loss_cfg: LossCfg,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if loss_cfg.consistency_weight <= 0 or loss_cfg.coeff_noise_std <= 0:
        z = xb.new_tensor(0.0)
        return z, {"consistency": z.detach()}

    xb_noisy = maybe_add_coeff_noise(xb, loss_cfg.coeff_noise_std, loss_cfg.coeff_noise_prob)
    roots_clean = model(xb)
    roots_noisy = model(xb_noisy)

    mode = str(loss_cfg.consistency_mode).lower()
    if mode == "roots":
        cons = ((roots_clean - roots_noisy) ** 2).mean()
    elif mode == "residual":
        p1 = transform_residual(poly_eval_horner(xb, roots_clean), loss_cfg.residual_transform)
        p2 = transform_residual(poly_eval_horner(xb, roots_noisy), loss_cfg.residual_transform)
        cons = ((p1 - p2) ** 2).mean()
    else:
        raise ValueError(f"Unsupported consistency_mode={loss_cfg.consistency_mode}")

    cons = cons * float(loss_cfg.consistency_weight)
    return cons, {"consistency": cons.detach()}


# -------------------------
# Eval
# -------------------------
@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device, loss_cfg: LossCfg) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_main = 0.0
    total_res_mean = 0.0
    total_res_min = 0.0
    total_n = 0

    for xb in loader:
        xb = xb.to(device, non_blocking=True)
        roots = model(xb)
        loss, aux = residual_core_loss(xb, roots, loss_cfg)

        bsz = xb.shape[0]
        total_loss += float(loss.item()) * bsz
        total_main += float(aux["main"].item()) * bsz
        total_res_mean += float(aux["residual_mean"].item()) * bsz
        total_res_min += float(aux["residual_min_mean"].item()) * bsz
        total_n += bsz

    denom = max(total_n, 1)
    return {
        "loss": total_loss / denom,
        "main": total_main / denom,
        "residual_mean": total_res_mean / denom,
        "residual_min_mean": total_res_min / denom,
    }


# -------------------------
# Train
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
        raise ValueError(f"이 구현은 backbone=ann만 처리합니다. 현재: {cfg.model.backbone}")

    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg.training.seed)
    device = torch.device(device_str)

    train_ds_raw = TaylorCoeffDataset(train_npz, order=cfg.model.order)
    val_ds_raw = TaylorCoeffDataset(val_npz, order=cfg.model.order)
    test_ds_raw = TaylorCoeffDataset(test_npz, order=cfg.model.order) if test_npz else None

    x_mn, x_mx = np_minmax_chunked(train_ds_raw.X)
    train_X = minmax_to_minus1_1(train_ds_raw.X, x_mn, x_mx)
    val_X = minmax_to_minus1_1(val_ds_raw.X, x_mn, x_mx)
    test_X = minmax_to_minus1_1(test_ds_raw.X, x_mn, x_mx) if test_ds_raw else None

    class _DS(Dataset):
        def __init__(self, X2d: np.ndarray):
            self.X2d = X2d
        def __len__(self):
            return self.X2d.shape[0]
        def __getitem__(self, i: int):
            return torch.from_numpy(self.X2d[i])

    train_ds = _DS(train_X)
    val_ds = _DS(val_X)
    test_ds = _DS(test_X) if test_X is not None else None

    pin = (device.type == "cuda")
    train_ld = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.training.num_workers,
        pin_memory=pin,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.training.num_workers,
        pin_memory=pin,
    )
    test_ld = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.training.num_workers,
        pin_memory=pin,
    ) if test_ds is not None else None

    in_dim = train_X.shape[1]
    model = ANNRootRegressor(in_dim=in_dim, num_roots=cfg.model.num_roots, arch=cfg.architecture).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    use_amp = bool(cfg.training.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_path = os.path.join(out_dir, "best.pt")
    scaler_path = os.path.join(out_dir, "scaler.json")
    cfg_dump_path = os.path.join(out_dir, "config_resolved.json")
    history_path = os.path.join(out_dir, "history.json")

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
    print(f"[LOSS] type={cfg.loss.type}, residual_transform={cfg.loss.residual_transform}, coeff_noise_std={cfg.loss.coeff_noise_std}, consistency_weight={cfg.loss.consistency_weight}")
    print(f"[OUT] {out_dir}")

    best_val = float("inf")
    patience = 0
    history: List[Dict[str, Any]] = []

    for ep in range(1, cfg.training.epochs + 1):
        model.train()
        run_total = 0.0
        run_main = 0.0
        run_cons = 0.0
        run_rmin = 0.0
        steps = 0

        for xb in train_ld:
            xb = xb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.autocast(device_type=("cuda" if device.type == "cuda" else "cpu"), enabled=use_amp):
                roots = model(xb)
                loss_main, aux_main = residual_core_loss(xb, roots, cfg.loss)
                loss_cons, _ = consistency_loss(model, xb, cfg.loss)
                loss = loss_main + loss_cons

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            if cfg.training.grad_clip and cfg.training.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(opt)
            scaler.update()

            run_total += float(loss.item())
            run_main += float(loss_main.item())
            run_cons += float(loss_cons.item())
            run_rmin += float(aux_main["residual_min_mean"].item())
            steps += 1

        if ep % max(cfg.training.eval_every, 1) == 0:
            train_log = {
                "train_total": run_total / max(steps, 1),
                "train_main": run_main / max(steps, 1),
                "train_cons": run_cons / max(steps, 1),
                "train_rmin": run_rmin / max(steps, 1),
            }
            val_log = eval_epoch(model, val_ld, device, cfg.loss)
            row = {
                "epoch": ep,
                **train_log,
                "val_loss": val_log["loss"],
                "val_main": val_log["main"],
                "val_residual_mean": val_log["residual_mean"],
                "val_residual_min_mean": val_log["residual_min_mean"],
            }
            history.append(row)

            print(
                f"[ep={ep:4d}] "
                f"train_total={row['train_total']:.6g} "
                f"train_main={row['train_main']:.6g} "
                f"train_cons={row['train_cons']:.6g} "
                f"train_rmin={row['train_rmin']:.6g}  "
                f"val_loss={row['val_loss']:.6g} "
                f"val_rmin={row['val_residual_min_mean']:.6g}"
            )

            if val_log["loss"] < best_val:
                best_val = val_log["loss"]
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
                        "train_mode": "residual_only_unlabeled",
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

            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

    if test_ld is not None and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["state_dict"])
        test_log = eval_epoch(model, test_ld, device, cfg.loss)
        print(
            f"[TEST] loss={test_log['loss']:.6g} "
            f"main={test_log['main']:.6g} "
            f"residual_mean={test_log['residual_mean']:.6g} "
            f"residual_min_mean={test_log['residual_min_mean']:.6g}"
        )

    print("[DONE]")


# -------------------------
# Entrypoint (No argparse)
# -------------------------
def main() -> None:
    repo = find_repo_root(__file__)

    default_cfg = "configs/taylor_root_residual_ssl.yaml"
    default_train = "data/taylor_data_physchem_v4_deg25/taylor_deg25_train.npz"
    default_val = "data/taylor_data_physchem_v4_deg25/taylor_deg25_val.npz"
    default_test = "data/taylor_data_physchem_v4_deg25/taylor_deg25_test.npz"
    default_out = "results/taylor_nn/residual_ssl"
    default_dev = "auto"

    cfg_path = os.environ.get("TAYLOR_CFG", default_cfg)
    train_npz = os.environ.get("TRAIN_NPZ", default_train)
    val_npz = os.environ.get("VAL_NPZ", default_val)
    test_npz = os.environ.get("TEST_NPZ", default_test)
    out_dir = os.environ.get("OUT_DIR", default_out)
    device_str = os.environ.get("DEVICE", default_dev)

    cfg_path_p = resolve_repo_path(cfg_path, repo)
    train_npz_p = resolve_repo_path(train_npz, repo)
    val_npz_p = resolve_repo_path(val_npz, repo)
    test_npz_p = resolve_repo_path(test_npz, repo)
    out_dir_p = resolve_repo_path(out_dir, repo)

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