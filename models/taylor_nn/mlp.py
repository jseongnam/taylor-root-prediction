#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/taylor_nn/mlp.py

✅ YAML(configs/taylor_root_mlp.yaml) 기반으로 작동하는 MLP Anchored Root 모델.
- argparse/add_argument 없이 동작
- 출력: logits (B,25) -> softmax -> 기대 잔차 기반 loss (미분 가능)
- metric: Top-1 anchor (argmax w) 기준

YAML 예:
model:
  backbone: mlp
  input: { feature: taylor_coefficients, order: 25, dimension: 26 }
  output: { num_roots: 25 }
architecture:
  hidden_dim: 25
  layers: auto
training:
  batch_size: 2048
  epochs: 1000
  learning_rate: 3e-5
loss:
  type: min_residual

환경변수 override:
  TAYLOR_CFG=/path/config.yaml
  TRAIN_NPZ=/path/train.npz
  VAL_NPZ=/path/val.npz
  TEST_NPZ=/path/test.npz
  OUT_DIR=runs/taylor_root_mlp
  DEVICE=cuda|cpu

추가(선택) 환경변수:
  ANCHOR_RANGE=10.0         # anchors in [-R, R] (기본: 10.0)
  TEMPERATURE=1.0
  INVALID_PENALTY=1e6
  ROOT_CLIP=0               # 0이면 off, >0이면 Top-1 선택/metric에서 clamp
"""

from __future__ import annotations

import os
import json
import time
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
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
    backbone: str = "mlp"
    input_feature: str = "taylor_coefficients"
    order: int = 25
    dimension: int = 26
    num_roots: int = 25


@dataclass
class ArchCfg:
    hidden_dim: int = 25
    layers: Any = "auto"      # auto | int | [int,...]
    activation: str = "relu"  # 기존 코드 ReLU 유지
    dropout: float = 0.0


@dataclass
class TrainCfg:
    dataset_size: int = 0     # 있으면 로그용
    batch_size: int = 2048
    epochs: int = 1000
    learning_rate: float = 3e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    eval_every: int = 1
    num_workers: int = 0
    seed: int = 1234
    early_stop: int = 0       # 0이면 off


@dataclass
class LossCfg:
    type: str = "min_residual"
    # 기대 잔차 계산 방식 (YAML에 없어도 기본으로 안전하게 사용)
    residual_mode: str = "fx_mse"     # fx_mse | nres_mse | fx_l1 | nres_l1
    invalid_penalty: float = 1e6
    root_clip: float = 0.0           # 0이면 off (metric에서만 사용)
    # collapse 방지(선택)
    entropy_weight: float = 0.0      # w의 엔트로피를 키워서 collapse 완화
    entropy_target: float = 0.0      # (선택) 목표 엔트로피
    diversity_weight: float = 0.0    # 간단한 pairwise penalty (optional)
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
        residual_mode=str(_get(raw, "loss.residual_mode", "fx_mse")),
        invalid_penalty=float(_get(raw, "loss.invalid_penalty", 1e6)),
        root_clip=float(_get(raw, "loss.root_clip", 0.0)),
        entropy_weight=float(_get(raw, "loss.entropy_weight", 0.0)),
        entropy_target=float(_get(raw, "loss.entropy_target", 0.0)),
        diversity_weight=float(_get(raw, "loss.diversity_weight", 0.0)),
        diversity_margin=float(_get(raw, "loss.diversity_margin", 1e-2)),
    )
    return FullCfg(model=m, architecture=a, training=t, loss=l)


# -------------------------
# NPZ helpers
# -------------------------
def _pick_coeff_key(keys: List[str]) -> str:
    cand = ["coeffs", "taylor_coefficients", "coefficients"]
    for c in cand:
        if c in keys:
            return c
    raise KeyError(f"NPZ에 계수 키가 없습니다. 기대 키: {cand}, 실제 키: {keys}")


class RootDataset(Dataset):
    """
    coeffs만 필수.
    root0 있으면 y로 같이 반환(참고 metric용).
    """
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
            # 일부 데이터가 D=25(상수항 누락)일 수 있으니 보정
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

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        if self.has_y:
            y = torch.from_numpy(self.y[idx])
            return x, y
        return x


# -------------------------
# Torch polynomial eval (Horner) + denom
# -------------------------
def poly_eval_and_norm_torch(coeffs: torch.Tensor, x: torch.Tensor, eps: float = 1e-15):
    """
    coeffs: (B, D) ascending a0..a_deg
    x: (B,M) anchors
    return:
      fx    : P(x)             shape (B,M)
      denom : sum|a_k||x|^k    shape (B,M)
    """
    x_abs = torch.abs(x)
    a = coeffs

    fx = a[:, -1].unsqueeze(1).expand_as(x)
    denom = torch.abs(a[:, -1]).unsqueeze(1).expand_as(x)

    for k in range(a.size(1) - 2, -1, -1):
        fx = fx * x + a[:, k].unsqueeze(1)
        denom = denom * x_abs + torch.abs(a[:, k]).unsqueeze(1)

    return fx, denom + eps


# -------------------------
# Model: logits -> softmax -> expectation residual
# -------------------------
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
    # dimension(26) 기준이면 2~3층이 무난
    if in_dim <= 64:
        return [hidden_dim, hidden_dim]
    if in_dim <= 256:
        return [hidden_dim, hidden_dim, hidden_dim]
    return [hidden_dim] * 4


class MLPAnchoredRoot(nn.Module):
    """
    출력: logits (B, m=25)
    w = softmax(logits/T)
    x_mix = Σ w_j * anchor_j  (참고용)
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        layers: Any,
        activation: str,
        dropout: float,
        m: int = 25,
        anchor_range: float = 10.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        # assert m == 25, "요구사항: output layer 차원은 25로 고정."
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.m = m
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
        mods.append(nn.Linear(prev, m))  # logits

        self.net = nn.Sequential(*mods)

        anchors = torch.linspace(-float(anchor_range), float(anchor_range), steps=m).view(1, m)  # (1,25)
        self.register_buffer("anchors", anchors)

        # init
        for mm in self.modules():
            if isinstance(mm, nn.Linear):
                nn.init.xavier_uniform_(mm.weight)
                nn.init.zeros_(mm.bias)

    def forward(self, x: torch.Tensor, return_all: bool = False):
        logits = self.net(x)  # (B,25)
        T = max(self.temperature, 1e-6)
        w = torch.softmax(logits / T, dim=1)  # (B,25)
        x_mix = (w * self.anchors).sum(dim=1, keepdim=True)  # (B,1)
        if return_all:
            return x_mix, logits, w
        return x_mix


# -------------------------
# Loss + metrics
# -------------------------
def residual_sq(fx: torch.Tensor, denom: torch.Tensor, mode: str) -> torch.Tensor:
    # fx, denom: (B,25)
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


def expectation_residual_loss(coeffs: torch.Tensor, w: torch.Tensor, anchors: torch.Tensor, cfg: LossCfg) -> torch.Tensor:
    """
    coeffs: (B,D)
    w:      (B,25)
    anchors:(1,25) or (B,25)
    """
    x_cand = anchors.expand(coeffs.size(0), -1)  # (B,25)
    fx_cand, denom_cand = poly_eval_and_norm_torch(coeffs, x_cand)
    r = residual_sq(fx_cand, denom_cand, cfg.residual_mode)  # (B,25)

    finite = torch.isfinite(r)
    r_safe = torch.where(finite, r, torch.full_like(r, float(cfg.invalid_penalty)))

    # 기대값: Σ w_j * r_j
    loss = (w * r_safe).sum(dim=1).mean()
    loss = torch.nan_to_num(loss, nan=float(cfg.invalid_penalty), posinf=float(cfg.invalid_penalty), neginf=float(cfg.invalid_penalty))

    # (선택) 엔트로피 정규화: collapse 완화
    if cfg.entropy_weight and cfg.entropy_weight > 0:
        # entropy = -Σ w log w
        ent = -(w * torch.log(torch.clamp(w, min=1e-12))).sum(dim=1).mean()
        if cfg.entropy_target and cfg.entropy_target > 0:
            loss = loss + float(cfg.entropy_weight) * (torch.relu(cfg.entropy_target - ent) ** 2)
        else:
            # 그냥 엔트로피 크게: -ent를 빼면 엔트로피가 커짐
            loss = loss - float(cfg.entropy_weight) * ent

    # (선택) diversity penalty (약하게)
    if cfg.diversity_weight and cfg.diversity_weight > 0:
        # 기대 root들이 특정 anchor 하나에 몰리는 걸 완화: w의 peak 억제
        # peak_pen = mean(max(w) - 1/25)
        peak = torch.max(w, dim=1).values.mean()
        loss = loss + float(cfg.diversity_weight) * torch.relu(peak - (1.0 / 25.0))

    return loss


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device, cfg: LossCfg) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n = 0

    # metric(Top-1 anchor)
    fx_sum = 0.0
    nres_sum = 0.0
    cnt = 0

    # x-mse (y 있을 때만)
    x_mse_sum = 0.0
    x_cnt = 0

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            xb, yb = batch
            yb = yb.to(device, non_blocking=True)
        else:
            xb = batch
            yb = None

        xb = xb.to(device, non_blocking=True)

        x_mix, logits, w = model(xb, return_all=True)
        loss = expectation_residual_loss(xb, w, model.anchors, cfg)

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        n += bs

        # Top-1 anchor metric
        idx = torch.argmax(w, dim=1)  # (B,)
        x_cand = model.anchors.expand(bs, -1)
        x_hat = x_cand[torch.arange(bs, device=device), idx].unsqueeze(1)  # (B,1)

        if cfg.root_clip and cfg.root_clip > 0:
            x_hat = torch.clamp(x_hat, -cfg.root_clip, cfg.root_clip)

        fx_hat, denom_hat = poly_eval_and_norm_torch(xb, x_hat)
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
        "mean_abs_fx_top1": (fx_sum / max(1, cnt)),
        "mean_abs_nres_top1": (nres_sum / max(1, cnt)),
    }
    if x_cnt > 0:
        out["x_mse_top1"] = x_mse_sum / x_cnt
    return out


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

    if cfg.model.backbone.lower() != "mlp":
        raise ValueError(f"이 mlp.py는 backbone=mlp만 처리합니다. 현재: {cfg.model.backbone}")
    if cfg.loss.type.lower() != "min_residual":
        raise ValueError(f"이 구현은 loss=min_residual만 처리합니다. 현재: {cfg.loss.type}")

    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg.training.seed)
    device = torch.device(device_str)

    # env overrides (선택)
    anchor_range = float(os.environ.get("ANCHOR_RANGE", "10.0"))
    temperature = float(os.environ.get("TEMPERATURE", "1.0"))
    invalid_penalty = float(os.environ.get("INVALID_PENALTY", str(cfg.loss.invalid_penalty)))
    root_clip = float(os.environ.get("ROOT_CLIP", str(cfg.loss.root_clip)))

    cfg.loss.invalid_penalty = invalid_penalty
    cfg.loss.root_clip = root_clip

    # dataset
    train_ds_raw = RootDataset(train_npz, expect_dim=cfg.model.dimension)
    val_ds_raw   = RootDataset(val_npz,   expect_dim=cfg.model.dimension)
    test_ds_raw  = RootDataset(test_npz,  expect_dim=cfg.model.dimension) if test_npz else None

    train_ld = DataLoader(
        train_ds_raw,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_ld = DataLoader(
        val_ds_raw,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_ld = DataLoader(
        test_ds_raw,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == "cuda"),
    ) if test_ds_raw else None

    in_dim = cfg.model.dimension

    model = MLPAnchoredRoot(
        in_dim=in_dim,
        hidden_dim=cfg.architecture.hidden_dim,
        layers=cfg.architecture.layers,
        activation=cfg.architecture.activation,
        dropout=cfg.architecture.dropout,
        m=cfg.model.num_roots,
        anchor_range=anchor_range,
        temperature=temperature,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    ckpt_path = os.path.join(out_dir, "best.pt")
    cfg_dump_path = os.path.join(out_dir, "config_resolved.json")

    with open(cfg_dump_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": cfg.model.__dict__,
            "architecture": cfg.architecture.__dict__,
            "training": cfg.training.__dict__,
            "loss": cfg.loss.__dict__,
            "runtime": {
                "anchor_range": anchor_range,
                "temperature": temperature,
                "device": device_str,
            },
        }, f, ensure_ascii=False, indent=2)

    print(f"[CONFIG] {cfg_path}")
    print(f"[DATA] train={len(train_ds_raw)} val={len(val_ds_raw)} test={(len(test_ds_raw) if test_ds_raw else 0)}")
    print(f"[NPZ] train coeff_key={train_ds_raw.coeff_key}, keys={train_ds_raw.keys}")
    print(f"[NPZ] has_y(train/val/test)={train_ds_raw.has_y}/{val_ds_raw.has_y}/{(test_ds_raw.has_y if test_ds_raw else False)}")
    print(f"[MODEL] in_dim={in_dim}, hidden_dim={cfg.architecture.hidden_dim}, layers={cfg.architecture.layers}, out(m)={cfg.model.num_roots}")
    print(f"[ANCHOR] range=[-{anchor_range},{anchor_range}], temperature={temperature}")
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
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                xb, _ = batch
            else:
                xb = batch
            xb = xb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            x_mix, logits, w = model(xb, return_all=True)
            loss = expectation_residual_loss(xb, w, model.anchors, cfg.loss)

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
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "best_val": best_val,
                        "config_json": cfg_dump_path,
                        "anchors": model.anchors.detach().cpu(),
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

    t1 = time.perf_counter()
    print(f"[TIME] total={t1-t0:.2f}s")

    # test
    if test_ld is not None and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["state_dict"])
        test_metrics = eval_epoch(model, test_ld, device, cfg.loss)
        print(f"[TEST] loss={test_metrics['loss']:.6g}  fx={test_metrics['mean_abs_fx_top1']:.3e}  nres={test_metrics['mean_abs_nres_top1']:.3e}" +
              (f"  x_mse={test_metrics['x_mse_top1']:.3e}" if "x_mse_top1" in test_metrics else ""))

    print("[DONE]")


# -------------------------
# Entrypoint (No argparse)
# -------------------------
def main() -> None:
    import os
    import torch
    from src.path_utils import find_repo_root, resolve_repo_path, resolve_device

    repo = find_repo_root(__file__)

    default_cfg   = "configs/taylor_root_mlp.yaml"
    default_train = "data/taylor_data_physchem_v4_deg25/taylor_deg25_train.npz"
    default_val   = "data/taylor_data_physchem_v4_deg25/taylor_deg25_val.npz"
    default_test  = "data/taylor_data_physchem_v4_deg25/taylor_deg25_test.npz"
    default_out   = "results/taylor_nn/mlp"
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
    test_npz_p  = resolve_repo_path(test_npz, repo)
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

