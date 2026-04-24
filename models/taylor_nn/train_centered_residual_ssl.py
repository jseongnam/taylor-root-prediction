#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/taylor_nn/train_centered_residual_ssl.py

Residual-only local-offset training aligned with the AST-center evaluation pipeline.
"""

from __future__ import annotations

import os
import re
import ast
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import yaml
except Exception as e:
    raise ImportError("PyYAML is required. pip install pyyaml") from e


def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def _get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


@dataclass
class ModelCfg:
    order: int = 25
    num_roots: int = 1


@dataclass
class ArchCfg:
    hidden_dim: int = 256
    layers: Any = "auto"
    activation: str = "tanh"
    dropout: float = 0.0
    bounded_output: bool = True
    root_range: float = 10.0


@dataclass
class TrainingCfg:
    batch_size: int = 1024
    epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    eval_every: int = 1
    num_workers: int = 0
    seed: int = 1234
    early_stop: int = 20
    topk_train: int = 5
    topk_eval: int = 10
    train_reduce: str = "softmin"
    temperature: float = 5.0


@dataclass
class LossCfg:
    type: str = "softmin_residual"
    residual_transform: str = "logabs"
    root_clip: float = 0.0
    root_l2_weight: float = 0.0
    diversity_weight: float = 0.0
    diversity_margin: float = 1e-2


@dataclass
class ASTCfg:
    batch_size: int = 4096
    cache_dir: str = "center_cache"


@dataclass
class FullCfg:
    model: ModelCfg
    architecture: ArchCfg
    training: TrainingCfg
    loss: LossCfg
    ast: ASTCfg


def load_config(yaml_path: str) -> FullCfg:
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return FullCfg(
        model=ModelCfg(
            order=int(_get(raw, "model.input.order", 25)),
            num_roots=int(_get(raw, "model.output.num_roots", 1)),
        ),
        architecture=ArchCfg(
            hidden_dim=int(_get(raw, "architecture.hidden_dim", 256)),
            layers=_get(raw, "architecture.layers", "auto"),
            activation=str(_get(raw, "architecture.activation", "tanh")),
            dropout=float(_get(raw, "architecture.dropout", 0.0)),
            bounded_output=bool(_get(raw, "architecture.bounded_output", True)),
            root_range=float(_get(raw, "architecture.root_range", 10.0)),
        ),
        training=TrainingCfg(
            batch_size=int(_get(raw, "training.batch_size", 1024)),
            epochs=int(_get(raw, "training.epochs", 200)),
            learning_rate=float(_get(raw, "training.learning_rate", 1e-4)),
            weight_decay=float(_get(raw, "training.weight_decay", 0.0)),
            grad_clip=float(_get(raw, "training.grad_clip", 1.0)),
            eval_every=int(_get(raw, "training.eval_every", 1)),
            num_workers=int(_get(raw, "training.num_workers", 0)),
            seed=int(_get(raw, "training.seed", 1234)),
            early_stop=int(_get(raw, "training.early_stop", 20)),
            topk_train=int(_get(raw, "training.topk_train", 5)),
            topk_eval=int(_get(raw, "training.topk_eval", 10)),
            train_reduce=str(_get(raw, "training.train_reduce", "softmin")),
            temperature=float(_get(raw, "training.temperature", 5.0)),
        ),
        loss=LossCfg(
            type=str(_get(raw, "loss.type", "softmin_residual")),
            residual_transform=str(_get(raw, "loss.residual_transform", "logabs")),
            root_clip=float(_get(raw, "loss.root_clip", 0.0)),
            root_l2_weight=float(_get(raw, "loss.root_l2_weight", 0.0)),
            diversity_weight=float(_get(raw, "loss.diversity_weight", 0.0)),
            diversity_margin=float(_get(raw, "loss.diversity_margin", 1e-2)),
        ),
        ast=ASTCfg(
            batch_size=int(_get(raw, "ast.batch_size", 4096)),
            cache_dir=str(_get(raw, "ast.cache_dir", "center_cache")),
        ),
    )


def _pick_coeff_key(keys: List[str]) -> str:
    cand = ["coeffs", "taylor_coefficients", "coefficients"]
    for c in cand:
        if c in keys:
            return c
    raise KeyError(f"NPZ coeff key not found. expected={cand}, actual={keys}")


def _pick_expr_key(keys: List[str]) -> str:
    cand = ["expr_str", "func_expr", "expr", "expression", "equation"]
    for c in cand:
        if c in keys:
            return c
    raise KeyError(f"NPZ expression key not found. expected={cand}, actual={keys}")


class TaylorExprDataset(Dataset):
    def __init__(self, npz_path: str, order: int):
        z = np.load(npz_path, mmap_mode="r", allow_pickle=True)
        keys = list(z.keys())
        ck = _pick_coeff_key(keys)
        ek = _pick_expr_key(keys)
        X = np.array(z[ck])
        if X.ndim == 3:
            X = np.squeeze(X)
        if X.ndim != 2:
            raise ValueError(f"Unsupported coeff shape={X.shape}")
        if X.shape[1] == order:
            X = np.concatenate([np.zeros((X.shape[0], 1), dtype=X.dtype), X], axis=1)
        elif X.shape[1] != order + 1:
            raise ValueError(f"order={order}, coeff dim={X.shape[1]}, expected {order+1} (or {order})")
        expr = np.array(z[ek]).astype(object)
        self.coeffs = X.astype(np.float32)
        self.expr = expr
        self.coeff_key = ck
        self.expr_key = ek
        self.keys = keys

    def __len__(self):
        return self.coeffs.shape[0]

    def __getitem__(self, idx: int):
        return self.coeffs[idx], str(self.expr[idx])


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


def poly_eval_horner(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    y = coeffs[:, -1].unsqueeze(1).expand_as(x)
    for k in range(coeffs.shape[1] - 2, -1, -1):
        y = y * x + coeffs[:, k].unsqueeze(1)
    return y


def poly_shift_to_z_np(coeffs_asc: np.ndarray, c: float) -> np.ndarray:
    c = float(c)
    a = np.asarray(coeffs_asc, dtype=np.float64)
    n = a.shape[0] - 1
    b = np.zeros((n + 1,), dtype=np.float64)
    cp = np.ones((n + 1,), dtype=np.float64)
    for t in range(1, n + 1):
        cp[t] = cp[t - 1] * c
    for i in range(0, n + 1):
        ai = float(a[i])
        if ai == 0.0:
            continue
        for k in range(0, i + 1):
            b[k] += ai * math.comb(i, k) * cp[i - k]
    return b.astype(np.float32)


_ALLOWED_FUNCS_AST = {"sin","cos","tan","tanh","sinh","cosh","exp","log","log10","sqrt","abs","ln"}
_BASE_TOKENS = ["<PAD>","<UNK>","<CLS>","x","NUM","+","-","*","/","**","neg","pos"]
_FUNC_TOKENS = sorted(list({"sin","cos","tan","tanh","sinh","cosh","exp","log","log10","sqrt","abs"}))
VOCAB = _BASE_TOKENS + _FUNC_TOKENS
STOI = {t: i for i, t in enumerate(VOCAB)}
PAD_ID = STOI["<PAD>"]
UNK_ID = STOI["<UNK>"]
NUM_ID = STOI["NUM"]


def sanitize_expr_for_ast(raw: str) -> str:
    s = str(raw).strip()
    if "= 0" in s:
        s = s.split("= 0")[0].strip()
    elif "=0" in s:
        s = s.split("=0")[0].strip()
    s = re.sub(r"\s*\([^()]*\)\s*$", "", s).strip()
    s = s.replace("^", "**")
    s = re.sub(r"\bnp\.", "", s)
    s = re.sub(r"\bln\s*\(", "log(", s)
    return s


def _tok_id(tok: str) -> int:
    return STOI.get(tok, UNK_ID)


def ast_to_prefix(node):
    tokens = []
    nums = []
    def emit(tok, num=0.0):
        tokens.append(tok)
        nums.append(float(num))
    def visit(n):
        if isinstance(n, ast.Expression):
            return visit(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)) and np.isfinite(float(n.value)):
                emit("NUM", float(n.value)); return
            emit("<UNK>", 0.0); return
        if isinstance(n, ast.Name):
            emit("x", 0.0) if n.id == "x" else emit("<UNK>", 0.0); return
        if isinstance(n, ast.UnaryOp):
            emit("neg",0.0) if isinstance(n.op, ast.USub) else emit("pos",0.0) if isinstance(n.op, ast.UAdd) else emit("<UNK>",0.0)
            visit(n.operand); return
        if isinstance(n, ast.BinOp):
            if isinstance(n.op, ast.Add): emit("+",0.0)
            elif isinstance(n.op, ast.Sub): emit("-",0.0)
            elif isinstance(n.op, ast.Mult): emit("*",0.0)
            elif isinstance(n.op, ast.Div): emit("/",0.0)
            elif isinstance(n.op, ast.Pow): emit("**",0.0)
            else: emit("<UNK>",0.0)
            visit(n.left); visit(n.right); return
        if isinstance(n, ast.Call):
            fname = n.func.id if isinstance(n.func, ast.Name) else n.func.attr if isinstance(n.func, ast.Attribute) else None
            if fname == "ln": fname = "log"
            emit(fname,0.0) if fname in _ALLOWED_FUNCS_AST else emit("<UNK>",0.0)
            if len(n.args) >= 1: visit(n.args[0])
            else: emit("<UNK>",0.0)
            return
        emit("<UNK>",0.0)
    visit(node)
    return tokens, nums


def encode_prefix(tokens, nums, max_len: int):
    toks = ["<CLS>"] + tokens
    nvs = [0.0] + nums
    if len(toks) > max_len:
        toks, nvs = toks[:max_len], nvs[:max_len]
    ids = np.array([_tok_id(t) for t in toks], dtype=np.int64)
    numvals = np.array(nvs, dtype=np.float32)
    attn = np.ones((len(ids),), dtype=np.bool_)
    if len(ids) < max_len:
        pad_n = max_len - len(ids)
        ids = np.concatenate([ids, np.full((pad_n,), PAD_ID, dtype=np.int64)], axis=0)
        numvals = np.concatenate([numvals, np.zeros((pad_n,), dtype=np.float32)], axis=0)
        attn = np.concatenate([attn, np.zeros((pad_n,), dtype=np.bool_)], axis=0)
    return ids, numvals, attn


class ExprASTOnlyDataset(Dataset):
    def __init__(self, expr_arr, max_len: int, sanitize: bool = True):
        self.expr = expr_arr
        self.max_len = int(max_len)
        self.sanitize = bool(sanitize)
    def __len__(self): return len(self.expr)
    def __getitem__(self, idx):
        e = str(self.expr[idx])
        if self.sanitize:
            e = sanitize_expr_for_ast(e)
        try:
            node = ast.parse(e, mode="eval")
            toks, nums = ast_to_prefix(node)
        except Exception:
            toks, nums = ["<UNK>"], [0.0]
        ids, numvals, attn = encode_prefix(toks, nums, self.max_len)
        return torch.from_numpy(ids), torch.from_numpy(numvals), torch.from_numpy(attn.astype(np.uint8))


class ASTPrefixTransformerTopK(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, num_candidates: int, d_model: int = 256, nhead: int = 8, num_layers: int = 4):
        super().__init__()
        self.K = int(num_candidates)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.num_mlp = nn.Sequential(nn.Linear(1, d_model), nn.Tanh(), nn.Linear(d_model, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.y_head = nn.Linear(d_model, self.K)
    def forward(self, ids, numvals, attn_u8):
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        is_num = (ids == NUM_ID).unsqueeze(-1)
        x = x + self.num_mlp(numvals.unsqueeze(-1)) * is_num
        key_padding_mask = (attn_u8 == 0)
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.y_head(h[:, 0, :])


def load_ast_topk_model(ckpt_path: Path, device: torch.device):
    obj = torch.load(ckpt_path, map_location=device)
    cfg = obj.get("config", {})
    K = int(cfg.get("num_candidates", cfg.get("K", 10)))
    max_len = int(cfg.get("max_len", 128))
    d_model = int(cfg.get("d_model", 256))
    nhead = int(cfg.get("nhead", 8))
    num_layers = int(cfg.get("num_layers", 4))
    scale = float(cfg.get("scale", 1.0))
    sanitize_inputs = bool(cfg.get("sanitize_inputs", True))
    model = ASTPrefixTransformerTopK(len(VOCAB), max_len, K, d_model, nhead, num_layers).to(device)
    model.load_state_dict(obj["model_state"])
    model.eval()
    return model, cfg, scale, sanitize_inputs


@torch.no_grad()
def predict_centers_from_expr(expr_arr: np.ndarray, ast_model: nn.Module, max_len: int, scale: float, sanitize_inputs: bool, device: torch.device, batch_size: int) -> np.ndarray:
    ds = ExprASTOnlyDataset(expr_arr, max_len=max_len, sanitize=sanitize_inputs)
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    outs = []
    for ids, numvals, attn in ld:
        ids = ids.to(device, non_blocking=True)
        numvals = numvals.to(device, non_blocking=True)
        attn = attn.to(device, non_blocking=True)
        y = ast_model(ids, numvals, attn)
        outs.append((y.detach().cpu().numpy().astype(np.float32) * float(scale)))
    return np.concatenate(outs, axis=0)


def load_or_make_center_cache(split_name: str, expr_arr: np.ndarray, ast_model: nn.Module, ast_cfg: dict, ast_scale: float, ast_sanitize: bool, device: torch.device, cache_dir: Path, batch_size: int) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    K_all = int(ast_cfg.get("num_candidates", ast_cfg.get("K", 10)))
    cache_path = cache_dir / f"{split_name}_top{K_all}.npy"
    if cache_path.exists():
        centers = np.load(cache_path)
        if centers.ndim == 2 and centers.shape[1] == K_all and centers.shape[0] == len(expr_arr):
            print(f"[CACHE] load {cache_path}")
            return centers.astype(np.float32)
    centers = predict_centers_from_expr(expr_arr, ast_model, int(ast_cfg.get("max_len", 128)), ast_scale, ast_sanitize, device, batch_size)
    np.save(cache_path, centers.astype(np.float32))
    print(f"[CACHE] save {cache_path}")
    return centers.astype(np.float32)


def get_activation(name: str) -> nn.Module:
    n = name.lower()
    if n == "tanh": return nn.Tanh()
    if n == "relu": return nn.ReLU(inplace=True)
    if n == "gelu": return nn.GELU()
    if n in ("silu", "swish"): return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def resolve_layers_auto(in_dim: int, hidden_dim: int) -> List[int]:
    return [hidden_dim] * (3 if in_dim <= 64 else 4 if in_dim <= 256 else 5)


class LocalOffsetRegressor(nn.Module):
    def __init__(self, in_dim: int, num_roots: int, arch: ArchCfg):
        super().__init__()
        self.arch = arch
        if isinstance(arch.layers, str) and arch.layers.lower() == "auto":
            hlist = resolve_layers_auto(in_dim, arch.hidden_dim)
        elif isinstance(arch.layers, int):
            hlist = [arch.hidden_dim] * int(arch.layers)
        elif isinstance(arch.layers, (list, tuple)):
            hlist = [int(x) for x in arch.layers]
        else:
            raise ValueError(f"Unsupported architecture.layers={arch.layers}")
        layers = []
        prev = in_dim
        for h in hlist:
            layers += [nn.Linear(prev, h), get_activation(arch.activation)]
            if arch.dropout > 0:
                layers.append(nn.Dropout(arch.dropout))
            prev = h
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(prev, int(num_roots))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.head(self.backbone(x))
        if self.arch.bounded_output:
            z = torch.tanh(z) * float(self.arch.root_range)
        return z


def transform_residual(abs_p: torch.Tensor, mode: str) -> torch.Tensor:
    m = str(mode).lower()
    if m == "abs": return abs_p
    if m == "logabs": return torch.log1p(abs_p)
    if m == "square": return abs_p * abs_p
    raise ValueError(f"Unsupported residual_transform={mode}")


def candidate_penalties(local_coeff_raw: torch.Tensor, z: torch.Tensor, loss_cfg: LossCfg) -> torch.Tensor:
    x = torch.clamp(z, -loss_cfg.root_clip, loss_cfg.root_clip) if loss_cfg.root_clip and loss_cfg.root_clip > 0 else z
    return transform_residual(torch.abs(poly_eval_horner(local_coeff_raw, x)), loss_cfg.residual_transform)


def residual_objective(local_coeff_raw: torch.Tensor, z: torch.Tensor, loss_cfg: LossCfg, reduce_mode: str = "softmin", temperature: float = 5.0) -> torch.Tensor:
    pen = candidate_penalties(local_coeff_raw, z, loss_cfg)
    if str(reduce_mode).lower() == "min":
        loss = torch.min(pen, dim=1).values.mean()
    else:
        tau = max(float(temperature), 1e-6)
        w = torch.softmax(-tau * pen, dim=1)
        loss = (w * pen).sum(dim=1).mean()
    if loss_cfg.root_l2_weight and loss_cfg.root_l2_weight > 0:
        loss = loss + float(loss_cfg.root_l2_weight) * (z ** 2).mean()
    if loss_cfg.diversity_weight and loss_cfg.diversity_weight > 0 and z.shape[1] > 1:
        diff = torch.abs(z.unsqueeze(2) - z.unsqueeze(1))
        mask = 1.0 - torch.eye(z.shape[1], device=z.device).unsqueeze(0)
        penalty = torch.relu(float(loss_cfg.diversity_margin) - diff) * mask
        loss = loss + float(loss_cfg.diversity_weight) * penalty.mean()
    return loss


@torch.no_grad()
def best_abs_residual(local_coeff_raw: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return torch.min(torch.abs(poly_eval_horner(local_coeff_raw, z)), dim=1).values


def build_local_bank(coeffs_raw: np.ndarray, centers: np.ndarray, topk: int) -> np.ndarray:
    N, D = coeffs_raw.shape
    out = np.zeros((N, int(topk), D), dtype=np.float32)
    for i in range(N):
        for k in range(int(topk)):
            out[i, k] = poly_shift_to_z_np(coeffs_raw[i], float(centers[i, k]))
    return out


class LocalBankTrainDataset(Dataset):
    def __init__(self, x_in: np.ndarray, coeff_raw: np.ndarray):
        self.x_in = x_in.astype(np.float32)
        self.coeff_raw = coeff_raw.astype(np.float32)
    def __len__(self): return self.x_in.shape[0]
    def __getitem__(self, idx):
        return torch.from_numpy(self.x_in[idx]), torch.from_numpy(self.coeff_raw[idx])


class LocalBankEvalDataset(Dataset):
    def __init__(self, x_in_bank: np.ndarray, coeff_raw_bank: np.ndarray):
        self.x_in_bank = x_in_bank.astype(np.float32)
        self.coeff_raw_bank = coeff_raw_bank.astype(np.float32)
    def __len__(self): return self.x_in_bank.shape[0]
    def __getitem__(self, idx):
        return torch.from_numpy(self.x_in_bank[idx]), torch.from_numpy(self.coeff_raw_bank[idx])


@torch.no_grad()
def eval_topk(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    best_list = []
    for x_in_bank, coeff_raw_bank in loader:
        x_in_bank = x_in_bank.to(device, non_blocking=True)
        coeff_raw_bank = coeff_raw_bank.to(device, non_blocking=True)
        B, K, D = x_in_bank.shape
        z = model(x_in_bank.reshape(B * K, D))
        per_center = best_abs_residual(coeff_raw_bank.reshape(B * K, D), z).reshape(B, K)
        best_list.append(torch.min(per_center, dim=1).values.detach().cpu())
    best_all = torch.cat(best_list, dim=0).numpy().astype(np.float64)
    return float(best_all.mean()), float(np.median(best_all))


@torch.no_grad()
def eval_trainstyle(model: nn.Module, loader: DataLoader, device: torch.device, cfg: FullCfg) -> float:
    model.eval()
    total = 0.0
    n = 0
    for x_in, coeff_raw in loader:
        x_in = x_in.to(device, non_blocking=True)
        coeff_raw = coeff_raw.to(device, non_blocking=True)
        loss = residual_objective(coeff_raw, model(x_in), cfg.loss, cfg.training.train_reduce, cfg.training.temperature)
        total += float(loss.item()) * x_in.shape[0]
        n += x_in.shape[0]
    return total / max(n, 1)


def train_from_yaml(cfg_path: str, ast_ckpt: str, train_npz: str, val_npz: str, test_npz: Optional[str], out_dir: str, device_str: str) -> None:
    cfg = load_config(cfg_path)
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg.training.seed)
    device = torch.device(device_str)

    train_raw = TaylorExprDataset(train_npz, order=cfg.model.order)
    val_raw = TaylorExprDataset(val_npz, order=cfg.model.order)
    test_raw = TaylorExprDataset(test_npz, order=cfg.model.order) if test_npz else None

    print(f"[DATA] train={len(train_raw)} val={len(val_raw)} test={(len(test_raw) if test_raw else 0)}")
    print(f"[NPZ] train coeff_key={train_raw.coeff_key}, expr_key={train_raw.expr_key}, keys={train_raw.keys}")
    print(f"[DEBUG] first expr: {str(train_raw.expr[0])[:200]}")

    ast_model, ast_cfg, ast_scale, ast_sanitize = load_ast_topk_model(Path(ast_ckpt), device=device)
    K_all = int(ast_cfg.get("num_candidates", ast_cfg.get("K", 10)))
    print(f"[AST] ckpt={ast_ckpt}")
    print(f"[AST] K_all={K_all} max_len={int(ast_cfg.get('max_len',128))} scale={ast_scale} sanitize={ast_sanitize}")
    print(f"[DEBUG] topk_train={cfg.training.topk_train}")
    print(f"[DEBUG] topk_eval={cfg.training.topk_eval}")
    if cfg.training.topk_train > K_all:
        raise ValueError(f"topk_train={cfg.training.topk_train} > AST K_all={K_all}")
    if cfg.training.topk_eval > K_all:
        raise ValueError(f"topk_eval={cfg.training.topk_eval} > AST K_all={K_all}")

    cache_dir = Path(out_dir) / cfg.ast.cache_dir
    train_centers = load_or_make_center_cache("train", train_raw.expr, ast_model, ast_cfg, ast_scale, ast_sanitize, device, cache_dir, cfg.ast.batch_size)
    val_centers = load_or_make_center_cache("val", val_raw.expr, ast_model, ast_cfg, ast_scale, ast_sanitize, device, cache_dir, cfg.ast.batch_size)
    test_centers = load_or_make_center_cache("test", test_raw.expr, ast_model, ast_cfg, ast_scale, ast_sanitize, device, cache_dir, cfg.ast.batch_size) if test_raw is not None else None

    print(f"[DEBUG] train_centers.shape={train_centers.shape}")
    print(f"[DEBUG] val_centers.shape={val_centers.shape}")
    if test_centers is not None: print(f"[DEBUG] test_centers.shape={test_centers.shape}")

    train_local_bank_raw = build_local_bank(train_raw.coeffs, train_centers, cfg.training.topk_train)
    val_local_bank_raw = build_local_bank(val_raw.coeffs, val_centers, cfg.training.topk_eval)
    test_local_bank_raw = build_local_bank(test_raw.coeffs, test_centers, cfg.training.topk_eval) if test_raw is not None else None

    train_local_2d_raw = train_local_bank_raw.reshape(-1, train_local_bank_raw.shape[-1])
    x_mn, x_mx = np_minmax_chunked(train_local_2d_raw)
    train_local_2d_in = minmax_to_minus1_1(train_local_2d_raw, x_mn, x_mx)
    val_local_bank_in = minmax_to_minus1_1(val_local_bank_raw.reshape(-1, val_local_bank_raw.shape[-1]), x_mn, x_mx).reshape(val_local_bank_raw.shape)
    test_local_bank_in = minmax_to_minus1_1(test_local_bank_raw.reshape(-1, test_local_bank_raw.shape[-1]), x_mn, x_mx).reshape(test_local_bank_raw.shape) if test_local_bank_raw is not None else None

    train_ld = DataLoader(LocalBankTrainDataset(train_local_2d_in, train_local_2d_raw), batch_size=cfg.training.batch_size, shuffle=True, drop_last=True, num_workers=cfg.training.num_workers, pin_memory=(device.type == "cuda"))
    val_ld = DataLoader(LocalBankEvalDataset(val_local_bank_in, val_local_bank_raw), batch_size=max(1, cfg.training.batch_size // max(1, cfg.training.topk_eval)), shuffle=False, drop_last=False, num_workers=cfg.training.num_workers, pin_memory=(device.type == "cuda"))
    test_ld = DataLoader(LocalBankEvalDataset(test_local_bank_in, test_local_bank_raw), batch_size=max(1, cfg.training.batch_size // max(1, cfg.training.topk_eval)), shuffle=False, drop_last=False, num_workers=cfg.training.num_workers, pin_memory=(device.type == "cuda")) if test_local_bank_raw is not None else None

    model = LocalOffsetRegressor(train_local_2d_in.shape[1], cfg.model.num_roots, cfg.architecture).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)

    ckpt_path = os.path.join(out_dir, "best.pt")
    scaler_path = os.path.join(out_dir, "scaler.json")
    cfg_dump_path = os.path.join(out_dir, "config_resolved.json")
    hist_path = os.path.join(out_dir, "history.json")

    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump({"x_min": x_mn.tolist(), "x_max": x_mx.tolist()}, f, ensure_ascii=False, indent=2)
    with open(cfg_dump_path, "w", encoding="utf-8") as f:
        json.dump({"model": cfg.model.__dict__, "architecture": cfg.architecture.__dict__, "training": cfg.training.__dict__, "loss": cfg.loss.__dict__, "ast": cfg.ast.__dict__, "ast_model_config": ast_cfg}, f, ensure_ascii=False, indent=2)

    print(f"[MODEL] in_dim={train_local_2d_in.shape[1]}, num_roots={cfg.model.num_roots}")
    print(f"[OUT] {out_dir}")

    best_val = float("inf")
    patience = 0
    history = []

    for ep in range(1, cfg.training.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        for x_in, coeff_raw in train_ld:
            x_in = x_in.to(device, non_blocking=True)
            coeff_raw = coeff_raw.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = residual_objective(coeff_raw, model(x_in), cfg.loss, cfg.training.train_reduce, cfg.training.temperature)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if cfg.training.grad_clip and cfg.training.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            opt.step()
            running += float(loss.item())
            steps += 1
        if ep % max(cfg.training.eval_every, 1) == 0:
            train_loss = running / max(steps, 1)
            train_trainstyle = eval_trainstyle(model, train_ld, device, cfg)
            val_mean_abs, val_med_abs = eval_topk(model, val_ld, device)
            row = {"epoch": ep, "train_loss": train_loss, "train_trainstyle": train_trainstyle, "val_mean_best_abs": val_mean_abs, "val_median_best_abs": val_med_abs}
            history.append(row)
            print(f"[ep={ep:4d}] train_loss={train_loss:.6g}  train_trainstyle={train_trainstyle:.6g}  val_mean_best_abs={val_mean_abs:.6g}  val_median_best_abs={val_med_abs:.6g}")
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            if val_mean_abs < best_val:
                best_val = val_mean_abs
                patience = 0
                torch.save({"state_dict": model.state_dict(), "in_dim": int(train_local_2d_in.shape[1]), "num_roots": int(cfg.model.num_roots), "arch": cfg.architecture.__dict__, "loss": cfg.loss.__dict__, "training": cfg.training.__dict__, "scaler_json": scaler_path, "config_json": cfg_dump_path, "best_val_mean_abs": best_val, "mode": "local_offset_centered_residual_ssl"}, ckpt_path)
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
        test_mean_abs, test_med_abs = eval_topk(model, test_ld, device)
        print(f"[TEST] mean_best_abs={test_mean_abs:.6g} median_best_abs={test_med_abs:.6g}")
    print("[DONE]")


def main() -> None:
    repo = find_repo_root(__file__)
    cfg_path = os.environ.get("TAYLOR_CFG", "configs/taylor_root_centered_residual_ssl.yaml")
    ast_ckpt = os.environ.get("AST_CKPT", "expr_center_ast_best_10.pt")
    train_npz = os.environ.get("TRAIN_NPZ", "data/taylor_data_physchem_v4_deg25/taylor_deg25_train.npz")
    val_npz = os.environ.get("VAL_NPZ", "data/taylor_data_physchem_v4_deg25/taylor_deg25_val.npz")
    test_npz = os.environ.get("TEST_NPZ", "data/taylor_data_physchem_v4_deg25/taylor_deg25_test.npz")
    out_dir = os.environ.get("OUT_DIR", "results/taylor_nn/centered_residual_ssl")
    device_str = resolve_device(os.environ.get("DEVICE", "auto"))

    cfg_path_p = resolve_repo_path(cfg_path, repo)
    ast_ckpt_p = resolve_repo_path(ast_ckpt, repo)
    train_npz_p = resolve_repo_path(train_npz, repo)
    val_npz_p = resolve_repo_path(val_npz, repo)
    test_npz_p = resolve_repo_path(test_npz, repo)
    out_dir_p = resolve_repo_path(out_dir, repo)

    train_from_yaml(str(cfg_path_p), str(ast_ckpt_p), str(train_npz_p), str(val_npz_p), (str(test_npz_p) if test_npz_p is not None and test_npz_p.exists() else None), str(out_dir_p), device_str)


if __name__ == "__main__":
    main()