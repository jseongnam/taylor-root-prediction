#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stable MLP anchored root trainer for Taylor coefficient datasets.

Fixes over the original mlp.py:
1) network input uses min-max normalized coefficients, while loss/eval use RAW coefficients
2) polynomial residual computation uses float64 for stability
3) default residual_mode is nres_mse (safer than fx_mse on raw coefficients)
4) invalid anchors are masked and their probability mass is re-normalized instead of forcing 1e6 saturation
5) anchor range can be auto-inferred from root0 labels when available
6) supports TAYLOR_ORDER / DATA_DIR_TAYLOR degree sweep
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    import yaml
except Exception as e:
    raise ImportError("PyYAML이 필요합니다. `pip install pyyaml`") from e


def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class ModelCfg:
    type: str = "taylor_root_regressor"
    backbone: str = "mlp"
    input_feature: str = "taylor_coefficients"
    order: int = 25
    dimension: int = 26
    num_roots: int = 25


@dataclass
class ArchCfg:
    hidden_dim: int = 25
    layers: Any = "auto"
    activation: str = "relu"
    dropout: float = 0.0


@dataclass
class TrainCfg:
    dataset_size: int = 0
    batch_size: int = 2048
    epochs: int = 1000
    learning_rate: float = 3e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    eval_every: int = 1
    num_workers: int = 0
    seed: int = 1234
    early_stop: int = 0


@dataclass
class LossCfg:
    type: str = "min_residual"
    residual_mode: str = "nres_mse"
    invalid_penalty: float = 1e6
    root_clip: float = 0.0
    entropy_weight: float = 0.0
    entropy_target: float = 0.0
    diversity_weight: float = 0.0
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
        backbone=_get(raw, "model.backbone", "mlp"),
        input_feature=_get(raw, "model.input.feature", "taylor_coefficients"),
        order=int(_get(raw, "model.input.order", 25)),
        dimension=int(_get(raw, "model.input.dimension", 26)),
        num_roots=int(_get(raw, "model.output.num_roots", 25)),
    )
    a = ArchCfg(
        hidden_dim=int(_get(raw, "architecture.hidden_dim", 25)),
        layers=_get(raw, "architecture.layers", "auto"),
        activation=str(_get(raw, "architecture.activation", "relu")),
        dropout=float(_get(raw, "architecture.dropout", 0.0)),
    )
    t = TrainCfg(
        dataset_size=int(_get(raw, "training.dataset_size", 0)),
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
        residual_mode=str(_get(raw, "loss.residual_mode", "nres_mse")),
        invalid_penalty=float(_get(raw, "loss.invalid_penalty", 1e6)),
        root_clip=float(_get(raw, "loss.root_clip", 0.0)),
        entropy_weight=float(_get(raw, "loss.entropy_weight", 0.0)),
        entropy_target=float(_get(raw, "loss.entropy_target", 0.0)),
        diversity_weight=float(_get(raw, "loss.diversity_weight", 0.0)),
        diversity_margin=float(_get(raw, "loss.diversity_margin", 1e-2)),
    )

    order_env = os.environ.get("TAYLOR_ORDER", "").strip()
    if order_env:
        order = int(order_env)
        m.order = order
        m.dimension = order + 1

    return FullCfg(model=m, architecture=a, training=t, loss=l)


def _pick_coeff_key(keys: List[str]) -> str:
    for c in ["coeffs", "taylor_coefficients", "coefficients"]:
        if c in keys:
            return c
    raise KeyError(f"NPZ에 계수 키가 없습니다. 실제 키: {keys}")


class RootDataset(Dataset):
    def __init__(self, npz_path: str, expect_dim: int):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(npz_path)
        z = np.load(npz_path, allow_pickle=True, mmap_mode="r")
        keys = list(z.keys())
        ck = _pick_coeff_key(keys)
        X = np.array(z[ck])
        if X.ndim == 3:
            X = np.squeeze(X)
        if X.ndim != 2:
            raise ValueError(f"coeffs shape={X.shape} 지원 불가. (N,D)만 지원")
        if X.shape[1] != expect_dim:
            if X.shape[1] == expect_dim - 1:
                X = np.concatenate([np.zeros((X.shape[0], 1), dtype=X.dtype), X], axis=1)
            else:
                raise ValueError(f"coeff dim mismatch: got {X.shape[1]}, expect {expect_dim}")
        self.X = X.astype(np.float32)
        self.y = None
        if "root0" in z:
            y = np.array(z["root0"]).squeeze()
            if y.ndim == 1:
                y = y[:, None]
            if y.ndim == 2 and y.shape[1] == 1:
                self.y = y.astype(np.float32)
        self.has_y = self.y is not None
        self.keys = keys
        self.coeff_key = ck

    def __len__(self):
        return self.X.shape[0]


def np_minmax_chunked(arr: np.ndarray, chunk: int = 200_000) -> Tuple[np.ndarray, np.ndarray]:
    n = arr.shape[0]
    mn = None
    mx = None
    for i in range(0, n, chunk):
        sl = arr[i:i + chunk]
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


class _MixedDataset(Dataset):
    def __init__(self, x_in: np.ndarray, x_raw: np.ndarray, y: Optional[np.ndarray]):
        self.x_in = x_in.astype(np.float32)
        self.x_raw = x_raw.astype(np.float32)
        self.y = None if y is None else y.astype(np.float32)

    def __len__(self):
        return self.x_in.shape[0]

    def __getitem__(self, idx):
        xin = torch.from_numpy(self.x_in[idx])
        xraw = torch.from_numpy(self.x_raw[idx])
        if self.y is not None:
            y = torch.from_numpy(self.y[idx])
            return xin, xraw, y
        return xin, xraw


def poly_eval_and_norm_torch(coeffs: torch.Tensor, x: torch.Tensor, eps: float = 1e-15):
    coeffs64 = coeffs.to(torch.float64)
    x64 = x.to(torch.float64)
    x_abs = torch.abs(x64)
    fx = coeffs64[:, -1].unsqueeze(1).expand_as(x64)
    denom = torch.abs(coeffs64[:, -1]).unsqueeze(1).expand_as(x64)
    for k in range(coeffs64.size(1) - 2, -1, -1):
        fx = fx * x64 + coeffs64[:, k].unsqueeze(1)
        denom = denom * x_abs + torch.abs(coeffs64[:, k]).unsqueeze(1)
    return fx, denom + eps


def get_activation(name: str) -> nn.Module:
    n = name.lower()
    if n == "relu":
        return nn.ReLU(inplace=True)
    if n == "tanh":
        return nn.Tanh()
    if n == "gelu":
        return nn.GELU()
    if n in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def resolve_layers_auto(in_dim: int, hidden_dim: int) -> List[int]:
    if in_dim <= 64:
        return [hidden_dim, hidden_dim]
    if in_dim <= 256:
        return [hidden_dim, hidden_dim, hidden_dim]
    return [hidden_dim] * 4


class MLPAnchoredRoot(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, layers: Any, activation: str, dropout: float,
                 m: int = 25, anchor_range: float = 10.0, temperature: float = 1.0):
        super().__init__()
        self.temperature = float(temperature)
        if isinstance(layers, str) and layers.lower() == "auto":
            hlist = resolve_layers_auto(in_dim, hidden_dim)
        elif isinstance(layers, int):
            hlist = [hidden_dim] * int(layers)
        elif isinstance(layers, (list, tuple)):
            hlist = [int(x) for x in layers]
        else:
            raise ValueError(f"Unsupported architecture.layers={layers}")

        act = get_activation(activation)
        mods: List[nn.Module] = []
        prev = in_dim
        for h in hlist:
            mods.append(nn.Linear(prev, h))
            mods.append(act)
            if dropout and dropout > 0:
                mods.append(nn.Dropout(p=float(dropout)))
            prev = h
        mods.append(nn.Linear(prev, m))
        self.net = nn.Sequential(*mods)
        anchors = torch.linspace(-float(anchor_range), float(anchor_range), steps=m).view(1, m)
        self.register_buffer("anchors", anchors)
        for mm in self.modules():
            if isinstance(mm, nn.Linear):
                nn.init.xavier_uniform_(mm.weight)
                nn.init.zeros_(mm.bias)

    def forward(self, x: torch.Tensor, return_all: bool = False):
        logits = self.net(x)
        T = max(self.temperature, 1e-6)
        w = torch.softmax(logits / T, dim=1)
        x_mix = (w * self.anchors).sum(dim=1, keepdim=True)
        if return_all:
            return x_mix, logits, w
        return x_mix


def residual_sq(fx: torch.Tensor, denom: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "fx_mse":
        return fx * fx
    if mode == "nres_mse":
        r = fx / denom
        return r * r
    if mode == "fx_l1":
        return torch.abs(fx)
    if mode == "nres_l1":
        return torch.abs(fx / denom)
    raise ValueError(f"Unknown residual_mode: {mode}")


def expectation_residual_loss(coeffs_raw: torch.Tensor, w: torch.Tensor, anchors: torch.Tensor, cfg: LossCfg) -> torch.Tensor:
    x_cand = anchors.expand(coeffs_raw.size(0), -1)
    if cfg.root_clip and cfg.root_clip > 0:
        x_cand = torch.clamp(x_cand, -cfg.root_clip, cfg.root_clip)
    fx_cand, denom_cand = poly_eval_and_norm_torch(coeffs_raw, x_cand)
    r = residual_sq(fx_cand, denom_cand, cfg.residual_mode)
    finite = torch.isfinite(r)

    if finite.any():
        w_masked = w * finite.to(w.dtype)
        z = w_masked.sum(dim=1, keepdim=True)
        if (z <= 1e-12).any():
            r_safe = torch.where(finite, r, torch.full_like(r, float(cfg.invalid_penalty)))
            loss = (w * r_safe).sum(dim=1).mean()
        else:
            w_safe = w_masked / torch.clamp(z, min=1e-12)
            r_safe = torch.where(finite, r, torch.zeros_like(r))
            loss = (w_safe * r_safe).sum(dim=1).mean()
    else:
        loss = torch.tensor(float(cfg.invalid_penalty), device=coeffs_raw.device, dtype=torch.float64)

    loss = torch.nan_to_num(loss, nan=float(cfg.invalid_penalty), posinf=float(cfg.invalid_penalty), neginf=float(cfg.invalid_penalty))

    if cfg.entropy_weight and cfg.entropy_weight > 0:
        ent = -(w * torch.log(torch.clamp(w, min=1e-12))).sum(dim=1).mean()
        if cfg.entropy_target and cfg.entropy_target > 0:
            target = torch.tensor(cfg.entropy_target, device=w.device, dtype=w.dtype)
            loss = loss + float(cfg.entropy_weight) * (torch.relu(target - ent) ** 2)
        else:
            loss = loss - float(cfg.entropy_weight) * ent

    if cfg.diversity_weight and cfg.diversity_weight > 0:
        peak = torch.max(w, dim=1).values.mean()
        loss = loss + float(cfg.diversity_weight) * torch.relu(peak - (1.0 / w.shape[1]))

    return loss.to(torch.float32)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device, cfg: LossCfg) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    fx_sum = 0.0
    nres_sum = 0.0
    cnt = 0
    x_mse_sum = 0.0
    x_cnt = 0
    for batch in loader:
        if len(batch) == 3:
            xin, xraw, yb = batch
            yb = yb.to(device, non_blocking=True)
        else:
            xin, xraw = batch
            yb = None
        xin = xin.to(device, non_blocking=True)
        xraw = xraw.to(device, non_blocking=True)
        _, logits, w = model(xin, return_all=True)
        loss = expectation_residual_loss(xraw, w, model.anchors, cfg)
        bs = xin.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
        idx = torch.argmax(w, dim=1)
        x_cand = model.anchors.expand(bs, -1)
        x_hat = x_cand[torch.arange(bs, device=device), idx].unsqueeze(1)
        if cfg.root_clip and cfg.root_clip > 0:
            x_hat = torch.clamp(x_hat, -cfg.root_clip, cfg.root_clip)
        fx_hat, denom_hat = poly_eval_and_norm_torch(xraw, x_hat)
        fx_hat = fx_hat.squeeze(1)
        denom_hat = denom_hat.squeeze(1)
        nres_hat = fx_hat / denom_hat
        m = torch.isfinite(fx_hat) & torch.isfinite(nres_hat)
        if m.any():
            fx_sum += float(torch.abs(fx_hat[m]).mean().item())
            nres_sum += float(torch.abs(nres_hat[m]).mean().item())
            cnt += 1
        if yb is not None:
            x_mse_sum += float(((x_hat - yb) ** 2).mean().item())
            x_cnt += 1
    out = {
        "loss": total_loss / max(1, n),
        "mean_abs_fx_top1": fx_sum / max(1, cnt),
        "mean_abs_nres_top1": nres_sum / max(1, cnt),
    }
    if x_cnt > 0:
        out["x_mse_top1"] = x_mse_sum / x_cnt
    return out


def infer_anchor_range(train_ds_raw: RootDataset, fallback: float = 6.0) -> float:
    if train_ds_raw.has_y and train_ds_raw.y is not None:
        y = np.asarray(train_ds_raw.y).reshape(-1)
        y = y[np.isfinite(y)]
        if y.size > 0:
            q = float(np.quantile(np.abs(y), 0.995))
            return max(2.0, min(10.0, q * 1.2))
    return fallback


def train_from_yaml(cfg_path: str, train_npz: str, val_npz: str, test_npz: Optional[str], out_dir: str, device_str: str) -> None:
    cfg = load_config(cfg_path)
    if cfg.model.backbone.lower() != "mlp":
        raise ValueError(f"이 mlp.py는 backbone=mlp만 처리합니다. 현재: {cfg.model.backbone}")
    if cfg.loss.type.lower() != "min_residual":
        raise ValueError(f"이 구현은 loss=min_residual만 처리합니다. 현재: {cfg.loss.type}")

    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg.training.seed)
    device = torch.device(device_str)

    train_ds_raw = RootDataset(train_npz, expect_dim=cfg.model.dimension)
    val_ds_raw = RootDataset(val_npz, expect_dim=cfg.model.dimension)
    test_ds_raw = RootDataset(test_npz, expect_dim=cfg.model.dimension) if test_npz else None

    x_mn, x_mx = np_minmax_chunked(train_ds_raw.X)
    train_X_in = minmax_to_minus1_1(train_ds_raw.X, x_mn, x_mx)
    val_X_in = minmax_to_minus1_1(val_ds_raw.X, x_mn, x_mx)
    test_X_in = minmax_to_minus1_1(test_ds_raw.X, x_mn, x_mx) if test_ds_raw else None

    train_ds = _MixedDataset(train_X_in, train_ds_raw.X, train_ds_raw.y)
    val_ds = _MixedDataset(val_X_in, val_ds_raw.X, val_ds_raw.y)
    test_ds = _MixedDataset(test_X_in, test_ds_raw.X, test_ds_raw.y) if test_ds_raw else None

    train_ld = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, drop_last=False,
                          num_workers=cfg.training.num_workers, pin_memory=(device.type == "cuda"))
    val_ld = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, drop_last=False,
                        num_workers=cfg.training.num_workers, pin_memory=(device.type == "cuda"))
    test_ld = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, drop_last=False,
                         num_workers=cfg.training.num_workers, pin_memory=(device.type == "cuda")) if test_ds else None

    anchor_range_env = os.environ.get("ANCHOR_RANGE", "").strip()
    anchor_range = float(anchor_range_env) if anchor_range_env else infer_anchor_range(train_ds_raw, fallback=6.0)
    temperature = float(os.environ.get("TEMPERATURE", "1.0"))
    invalid_penalty = float(os.environ.get("INVALID_PENALTY", str(cfg.loss.invalid_penalty)))
    root_clip = float(os.environ.get("ROOT_CLIP", str(cfg.loss.root_clip or anchor_range)))
    cfg.loss.invalid_penalty = invalid_penalty
    cfg.loss.root_clip = root_clip

    model = MLPAnchoredRoot(
        in_dim=cfg.model.dimension,
        hidden_dim=cfg.architecture.hidden_dim,
        layers=cfg.architecture.layers,
        activation=cfg.architecture.activation,
        dropout=cfg.architecture.dropout,
        m=cfg.model.num_roots,
        anchor_range=anchor_range,
        temperature=temperature,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)

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
            "runtime": {"anchor_range": anchor_range, "temperature": temperature, "device": device_str},
        }, f, ensure_ascii=False, indent=2)

    print(f"[CONFIG] {cfg_path}")
    print(f"[DATA] train={len(train_ds_raw)} val={len(val_ds_raw)} test={(len(test_ds_raw) if test_ds_raw else 0)}")
    print(f"[NPZ] train coeff_key={train_ds_raw.coeff_key}, keys={train_ds_raw.keys}")
    print(f"[NPZ] has_y(train/val/test)={train_ds_raw.has_y}/{val_ds_raw.has_y}/{(test_ds_raw.has_y if test_ds_raw else False)}")
    print(f"[MODEL] in_dim={cfg.model.dimension}, hidden_dim={cfg.architecture.hidden_dim}, layers={cfg.architecture.layers}, out(m)={cfg.model.num_roots}")
    print(f"[ANCHOR] range=[-{anchor_range},{anchor_range}], temperature={temperature}, root_clip={root_clip}")
    print(f"[LOSS] residual_mode={cfg.loss.residual_mode}, invalid_penalty={cfg.loss.invalid_penalty}")
    print(f"[OUT] {out_dir}")

    best_val = float("inf")
    patience = 0
    t0 = time.perf_counter()

    for ep in range(1, cfg.training.epochs + 1):
        model.train()
        run_loss = 0.0
        steps = 0
        for batch in train_ld:
            if len(batch) == 3:
                xin, xraw, _ = batch
            else:
                xin, xraw = batch
            xin = xin.to(device, non_blocking=True)
            xraw = xraw.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            _, logits, w = model(xin, return_all=True)
            loss = expectation_residual_loss(xraw, w, model.anchors, cfg.loss)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if cfg.training.grad_clip and cfg.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            opt.step()
            run_loss += float(loss.item())
            steps += 1

        if ep % max(cfg.training.eval_every, 1) == 0:
            val_metrics = eval_epoch(model, val_ld, device, cfg.loss)
            tr_loss = run_loss / max(1, steps)
            val_loss = val_metrics["loss"]
            msg = f"[ep={ep:4d}] train_loss={tr_loss:.6g}  val_loss={val_loss:.6g}  fx={val_metrics['mean_abs_fx_top1']:.3e}  nres={val_metrics['mean_abs_nres_top1']:.3e}"
            if "x_mse_top1" in val_metrics:
                msg += f"  x_mse={val_metrics['x_mse_top1']:.3e}"
            print(msg)
            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                torch.save({
                    "state_dict": model.state_dict(),
                    "best_val": best_val,
                    "config_json": cfg_dump_path,
                    "scaler_json": scaler_path,
                    "anchors": model.anchors.detach().cpu(),
                }, ckpt_path)
                print(f"  -> save best: {ckpt_path}")
            else:
                if cfg.training.early_stop and cfg.training.early_stop > 0:
                    patience += 1
                    if patience >= cfg.training.early_stop:
                        print("[EARLY STOP]")
                        break

    print(f"[TIME] total={time.perf_counter() - t0:.2f}s")
    if test_ld is not None and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["state_dict"])
        test_metrics = eval_epoch(model, test_ld, device, cfg.loss)
        print(f"[TEST] loss={test_metrics['loss']:.6g}  fx={test_metrics['mean_abs_fx_top1']:.3e}  nres={test_metrics['mean_abs_nres_top1']:.3e}" +
              (f"  x_mse={test_metrics['x_mse_top1']:.3e}" if "x_mse_top1" in test_metrics else ""))
    print("[DONE]")


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, default)).strip())
    except Exception:
        return int(default)


def _default_taylor_data_dir(degree: int) -> str:
    return f"data/taylor_data_physchem_v4_deg{degree}"


def _candidate_npz_paths(data_dir: str, degree: int, split: str):
    return [
        f"{data_dir}/taylor_deg{degree}_{split}.npz",
        f"{data_dir}/{split}.npz",
        f"{data_dir}/{split}_deg{degree}.npz",
        f"{data_dir}/deg{degree}_{split}.npz",
        f"{data_dir}/taylor_{split}_deg{degree}.npz",
        f"{data_dir}/{degree}_{split}.npz",
    ]


def _default_taylor_npz(data_dir: str, degree: int, split: str) -> str:
    for cand in _candidate_npz_paths(data_dir, degree, split):
        if Path(cand).exists():
            return cand
    p = Path(data_dir)
    if p.exists():
        matches = sorted(p.glob(f"*{split}*.npz"))
        matches_deg = [m for m in matches if f"deg{degree}" in m.name or str(degree) in m.name]
        if len(matches_deg) == 1:
            return str(matches_deg[0])
        if len(matches) == 1:
            return str(matches[0])
    return _candidate_npz_paths(data_dir, degree, split)[0]


def _fallback_find_repo_root(start_file: str) -> Path:
    p = Path(start_file).resolve().parent
    for cand in [p, *p.parents]:
        if (cand / "configs").exists() or (cand / ".git").exists():
            return cand
    return p


def _fallback_resolve_repo_path(path_str: str, repo: Path):
    if path_str is None:
        return None
    s = str(path_str).strip()
    if s == "":
        return None
    p = Path(s)
    return p if p.is_absolute() else (repo / p)


def _fallback_resolve_device(device_str: str) -> str:
    s = str(device_str).strip().lower()
    if s in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return s


def _load_path_utils(start_file: str):
    try:
        from src.path_utils import find_repo_root, resolve_repo_path, resolve_device
        return find_repo_root, resolve_repo_path, resolve_device
    except Exception:
        return _fallback_find_repo_root, _fallback_resolve_repo_path, _fallback_resolve_device


def main() -> None:
    find_repo_root, resolve_repo_path, resolve_device = _load_path_utils(__file__)
    repo = find_repo_root(__file__)
    degree = _env_int("TAYLOR_ORDER", 25)
    default_cfg = "configs/taylor_root_mlp.yaml"
    data_dir = os.environ.get("DATA_DIR_TAYLOR", _default_taylor_data_dir(degree))
    default_train = _default_taylor_npz(data_dir, degree, "train")
    default_val = _default_taylor_npz(data_dir, degree, "val")
    default_test = _default_taylor_npz(data_dir, degree, "test")
    default_out = f"results/taylor_nn/mlp/deg{degree}"
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
