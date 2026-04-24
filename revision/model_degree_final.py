#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models/transformer/model.py

✅ configs/transformer_interval.yaml 기반으로 동작하는 Transformer Interval Center Predictor
- argparse/add_argument 없이 실행 가능
- 입력: expr_str -> AST prefix 토큰 시퀀스(tokenized_ast)
- 출력: top_k(=25) interval center 후보 (연속값)
- loss.type = max_absolute_error:
    dist_k = min_j |c_k - r_j|
    loss_i = max_k dist_k
    loss = mean_i loss_i

환경변수 override:
  CFG_PATH=configs/transformer_interval.yaml
  TRAIN_NPZ=/path/train.npz
  VAL_NPZ=/path/val.npz
  TEST_NPZ=/path/test.npz
  OUT_DIR=runs/transformer_interval
  DEVICE=cuda|cpu
  MODE=train|eval     (기본 train)

추가(선택) 환경변수:
  TAYLOR_ORDER=25
  DATA_DIR_INTERVAL=data/taylor_data_physchem_v4_interval
    -> TRAIN_NPZ / VAL_NPZ / TEST_NPZ를 직접 주지 않으면
       {DATA_DIR_INTERVAL}/taylor_deg{TAYLOR_ORDER}_{split}.npz 를 기본값으로 사용
"""

from __future__ import annotations

import os
import re
import ast
import math
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import yaml  # PyYAML
except Exception as e:
    raise ImportError("PyYAML이 필요합니다. `pip install pyyaml`") from e


# =========================
# Seed
# =========================
def set_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Config
# =========================
@dataclass
class ModelCfg:
    type: str
    representation: str
    vocab: List[str]
    layers: int
    heads: int
    hidden_dim: int
    max_sequence_length: str
    output_type: str
    top_k: int


@dataclass
class TrainCfg:
    dataset_size: int
    batch_size: int
    epochs: int
    learning_rate: float


@dataclass
class LossCfg:
    type: str
    description: str


@dataclass
class FullCfg:
    model: ModelCfg
    training: TrainCfg
    loss: LossCfg


def _get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_config(cfg_path: str) -> FullCfg:
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    vocab = list(_get(raw, "model.input.vocabulary", []))
    if not vocab:
        raise ValueError("YAML에 model.input.vocabulary 가 비어있습니다.")

    m = ModelCfg(
        type=str(_get(raw, "model.type", "transformer_interval_predictor")),
        representation=str(_get(raw, "model.input.representation", "tokenized_ast")),
        vocab=vocab,
        layers=int(_get(raw, "model.architecture.layers", 4)),
        heads=int(_get(raw, "model.architecture.heads", 8)),
        hidden_dim=int(_get(raw, "model.architecture.hidden_dim", 256)),
        max_sequence_length=str(_get(raw, "model.architecture.max_sequence_length", "256")),
        output_type=str(_get(raw, "model.output.type", "interval_center")),
        top_k=int(_get(raw, "model.output.top_k", 25)),
    )

    t = TrainCfg(
        dataset_size=int(_get(raw, "training.dataset_size", 0)),
        batch_size=int(_get(raw, "training.batch_size", 256)),
        epochs=int(_get(raw, "training.epochs", 20)),
        learning_rate=float(_get(raw, "training.learning_rate", 3e-4)),
    )

    l = LossCfg(
        type=str(_get(raw, "loss.type", "max_absolute_error")),
        description=str(_get(raw, "loss.description", "")),
    )

    return FullCfg(model=m, training=t, loss=l)


# =========================
# Expr sanitize + AST -> Prefix tokens
# =========================
_ALLOWED_FUNCS = {
    "sin", "cos", "tan", "tanh", "sinh", "cosh",
    "exp", "log", "log10", "sqrt", "abs", "ln"
}

def sanitize_expr_for_ast(raw: str) -> str:
    s = str(raw).strip()
    if "= 0" in s:
        s = s.split("= 0")[0].strip()
    elif "=0" in s:
        s = s.split("=0")[0].strip()

    # 뒤에 "(...)" 같은 suffix 제거(원본 코드 유지)
    s = re.sub(r"\s*\([^()]*\)\s*$", "", s).strip()
    s = s.replace("^", "**")
    s = re.sub(r"\bnp\.", "", s)
    s = re.sub(r"\bln\s*\(", "log(", s)  # ln -> log
    return s


def ast_to_prefix_tokens(node: ast.AST, vocab_set: set) -> Tuple[List[str], List[float]]:
    """
    YAML vocab 기준으로 토큰을 생성.
    숫자는 YAML에서 QM 토큰을 사용한다고 가정.
    """
    tokens: List[str] = []
    nums: List[float] = []

    def emit(tok: str, num: float = 0.0):
        # vocab에 없으면 UNK로 매핑되지만, 토큰 자체는 유지해도 됨.
        tokens.append(tok)
        nums.append(float(num))

    def visit(n):
        if isinstance(n, ast.Expression):
            return visit(n.body)

        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)) and np.isfinite(float(n.value)):
                emit("QM", float(n.value))  # number token
                return
            emit("UNK", 0.0)
            return

        if isinstance(n, ast.Name):
            if n.id == "x":
                emit("x", 0.0)
            else:
                emit("UNK", 0.0)
            return

        if isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.USub):
                emit("neg", 0.0)
            elif isinstance(n.op, ast.UAdd):
                emit("pos", 0.0)
            else:
                emit("UNK", 0.0)
            visit(n.operand)
            return

        if isinstance(n, ast.BinOp):
            if isinstance(n.op, ast.Add):
                emit("+", 0.0)
            elif isinstance(n.op, ast.Sub):
                emit("-", 0.0)
            elif isinstance(n.op, ast.Mult):
                emit("*", 0.0)
            elif isinstance(n.op, ast.Pow):
                emit("**", 0.0)
            else:
                # YAML vocab에 "/"가 없으므로 Div 등은 UNK 처리
                emit("UNK", 0.0)
            visit(n.left)
            visit(n.right)
            return

        if isinstance(n, ast.Call):
            fname = None
            if isinstance(n.func, ast.Name):
                fname = n.func.id
            elif isinstance(n.func, ast.Attribute):
                fname = n.func.attr

            if fname == "ln":
                fname = "log"
            if fname in _ALLOWED_FUNCS:
                emit(fname, 0.0)
            else:
                emit("UNK", 0.0)

            if len(n.args) >= 1:
                visit(n.args[0])
            else:
                emit("UNK", 0.0)
            return

        emit("UNK", 0.0)

    visit(node)
    return tokens, nums


def encode_prefix(tokens: List[str], nums: List[float], max_len: int,
                  stoi: Dict[str, int], pad_id: int, unk_id: int, cls_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ids: (L,)
    numvals: (L,)
    attn: (L,)  (1=valid, 0=pad)
    """
    toks = ["CLS"] + tokens
    nvs = [0.0] + nums

    if len(toks) > max_len:
        toks = toks[:max_len]
        nvs = nvs[:max_len]

    ids = np.array([stoi.get(t, unk_id) for t in toks], dtype=np.int64)
    numvals = np.array(nvs, dtype=np.float32)
    attn = np.ones((len(ids),), dtype=np.uint8)

    if len(ids) < max_len:
        pad_n = max_len - len(ids)
        ids = np.concatenate([ids, np.full((pad_n,), pad_id, dtype=np.int64)], axis=0)
        numvals = np.concatenate([numvals, np.zeros((pad_n,), dtype=np.float32)], axis=0)
        attn = np.concatenate([attn, np.zeros((pad_n,), dtype=np.uint8)], axis=0)

    return ids, numvals, attn


# =========================
# Dataset
# =========================
class ExprCenterASTDataset(Dataset):
    def __init__(self, expr_arr, roots_arr, max_len: int,
                 stoi: Dict[str, int], pad_id: int, unk_id: int, cls_id: int,
                 sanitize: bool = True):
        self.expr = expr_arr
        self.roots = roots_arr.astype(np.float64)
        self.max_len = int(max_len)
        self.stoi = stoi
        self.pad_id = int(pad_id)
        self.unk_id = int(unk_id)
        self.cls_id = int(cls_id)
        self.do_sanitize = bool(sanitize)

    def __len__(self):
        return self.roots.shape[0]

    def __getitem__(self, idx):
        e = str(self.expr[idx])
        if self.do_sanitize:
            e = sanitize_expr_for_ast(e)

        try:
            node = ast.parse(e, mode="eval")
            toks, nums = ast_to_prefix_tokens(node, vocab_set=set(self.stoi.keys()))
        except Exception:
            toks, nums = ["UNK"], [0.0]

        ids, numvals, attn = encode_prefix(
            toks, nums, self.max_len,
            stoi=self.stoi, pad_id=self.pad_id, unk_id=self.unk_id, cls_id=self.cls_id
        )

        r = self.roots[idx]
        mask = np.isfinite(r)

        return (
            torch.from_numpy(ids),                    # (L,) int64
            torch.from_numpy(numvals),                # (L,) float32
            torch.from_numpy(attn),                   # (L,) uint8
            torch.from_numpy(r),                      # (Kroot,) float64
            torch.from_numpy(mask.astype(np.uint8)),  # (Kroot,) uint8
        )


# =========================
# Model
# =========================
class ASTPrefixTransformerTopK(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, top_k: int,
                 d_model: int = 256, nhead: int = 8, num_layers: int = 4,
                 num_token: str = "QM", stoi: Optional[Dict[str, int]] = None):
        super().__init__()
        self.max_len = int(max_len)
        self.d_model = int(d_model)
        self.K = int(top_k)

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.num_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.y_head = nn.Linear(d_model, self.K)

        self.num_id = None
        if stoi is not None and num_token in stoi:
            self.num_id = int(stoi[num_token])

    def forward(self, ids: torch.Tensor, numvals: torch.Tensor, attn_u8: torch.Tensor):
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)

        x = self.tok_emb(ids) + self.pos_emb(pos)

        if self.num_id is not None:
            is_num = (ids == self.num_id).unsqueeze(-1)  # (B,L,1)
            num_embed = self.num_mlp(numvals.unsqueeze(-1))  # (B,L,D)
            x = x + num_embed * is_num

        key_padding_mask = (attn_u8 == 0)  # True=ignore
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)

        cls = h[:, 0, :]     # (B,D)
        y = self.y_head(cls) # (B,K)
        return y


# =========================
# Dist / Loss / Metrics
# =========================
def min_dist_candidates_to_roots(cands: torch.Tensor, roots: torch.Tensor, mask: torch.Tensor):
    """
    cands: (B,Kcand) float64
    roots: (B,Kroot) float64
    mask:  (B,Kroot) bool

    return:
      min_over_k: (B,) float64  = min_k min_j |c_k - r_j|
      max_over_k: (B,) float64  = max_k min_j |c_k - r_j|
      best_k:     (B,) int64    = argmin over candidates
    """
    diff = torch.abs(cands.unsqueeze(-1) - roots.unsqueeze(1))  # (B,Kcand,Kroot)
    inf = torch.tensor(float("inf"), device=diff.device, dtype=diff.dtype)
    diff = torch.where(mask.unsqueeze(1), diff, inf)

    # per candidate -> nearest root distance
    min_over_roots, _ = torch.min(diff, dim=-1)  # (B,Kcand)

    min_over_k, best_k = torch.min(min_over_roots, dim=1)      # (B,)
    max_over_k, _      = torch.max(min_over_roots, dim=1)      # (B,)
    return min_over_k, max_over_k, best_k


def loss_max_absolute_error(cands: torch.Tensor, roots: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    YAML 정의에 맞춘 loss:
      dist_k = min_j |c_k - r_j|
      loss_i = max_k dist_k
    """
    _, max_over_k, _ = min_dist_candidates_to_roots(cands, roots, mask)
    return torch.mean(max_over_k)


@torch.no_grad()
def eval_metrics(model, loader, device, scale: float):
    model.eval()
    n = 0
    mins = []
    maxs = []

    for ids, numvals, attn_u8, roots, mask_u8 in loader:
        ids = ids.to(device)
        numvals = numvals.to(device)
        attn_u8 = attn_u8.to(device)

        roots = roots.to(device)
        mask = (mask_u8.to(device) > 0)

        y = model(ids, numvals, attn_u8)                # (B,K) float32
        cands = (float(scale) * torch.sinh(y.double())).double()

        min_over_k, max_over_k, _ = min_dist_candidates_to_roots(cands, roots, mask)

        m1 = min_over_k.detach().cpu().numpy()
        m2 = max_over_k.detach().cpu().numpy()
        m1 = m1[np.isfinite(m1)]
        m2 = m2[np.isfinite(m2)]
        if m1.size: mins.append(m1)
        if m2.size: maxs.append(m2)
        n += int(min_over_k.numel())

    if n == 0 or len(mins) == 0:
        return {"min_mae": float("nan"), "min_p90": float("nan"), "min_p99": float("nan"),
                "max_mae": float("nan"), "max_p90": float("nan"), "max_p99": float("nan"),
                "n": int(n)}

    mins = np.concatenate(mins, axis=0)
    maxs = np.concatenate(maxs, axis=0)
    return {
        "min_mae": float(mins.mean()),
        "min_p90": float(np.percentile(mins, 90.0)),
        "min_p99": float(np.percentile(mins, 99.0)),
        "max_mae": float(maxs.mean()),
        "max_p90": float(np.percentile(maxs, 90.0)),
        "max_p99": float(np.percentile(maxs, 99.0)),
        "n": int(n),
    }


# =========================
# Data utils
# =========================
def load_npz_expr_roots(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    data = np.load(p, allow_pickle=True)
    if "expr_str" not in data:
        raise KeyError(f"{p}에 expr_str 키가 없습니다. keys={list(data.keys())}")
    if "roots" not in data:
        raise KeyError(f"{p}에 roots 키가 없습니다. keys={list(data.keys())}")
    return data["expr_str"], data["roots"]


def compute_scale_from_train_roots(roots_train: np.ndarray) -> float:
    r = np.asarray(roots_train, dtype=np.float64).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 1.0
    scale = float(np.percentile(np.abs(r), 99.0))
    return max(scale, 1.0)


def estimate_len_from_train(expr_arr: np.ndarray, stoi: Dict[str, int], rule: str, sanitize: bool = True) -> int:
    """
    rule:
      - "auto_top_1_percent_mean_x1_1" : (top 1% token lengths) mean * 1.1, ceil
      - 숫자 문자열이면 그대로
    """
    if rule.isdigit():
        return int(rule)

    if rule != "auto_top_1_percent_mean_x1_1":
        # fallback: p99*1.1 비슷하게
        rule = "auto_top_1_percent_mean_x1_1"

    lens = []
    for e in expr_arr:
        s = str(e)
        if sanitize:
            s = sanitize_expr_for_ast(s)
        try:
            node = ast.parse(s, mode="eval")
            toks, nums = ast_to_prefix_tokens(node, vocab_set=set(stoi.keys()))
            L = 1 + len(toks)  # +CLS
        except Exception:
            L = 2
        lens.append(L)

    lens = np.asarray(lens, dtype=np.float64)
    if lens.size == 0:
        return 64

    # top 1% mean
    q = np.percentile(lens, 99.0)
    top = lens[lens >= q]
    if top.size == 0:
        top = lens
    max_len = int(math.ceil(float(top.mean()) * 1.1))
    max_len = max(16, min(max_len, 2048))
    return max_len


# =========================
# Train / Eval (no argparse)
# =========================
def train_from_yaml(
    cfg_path: str,
    train_npz: str,
    val_npz: str,
    test_npz: str,
    out_dir: str,
    device_str: str,
) -> None:
    cfg = load_config(cfg_path)

    if cfg.loss.type != "max_absolute_error":
        raise ValueError(f"현재 구현은 loss.type=max_absolute_error만 지원. got={cfg.loss.type}")

    # vocab build
    vocab = cfg.model.vocab
    stoi = {t: i for i, t in enumerate(vocab)}
    if "PAD" not in stoi or "UNK" not in stoi or "CLS" not in stoi:
        raise ValueError("YAML vocab에 PAD/UNK/CLS 토큰이 반드시 포함돼야 합니다.")
    pad_id = stoi["PAD"]
    unk_id = stoi["UNK"]
    cls_id = stoi["CLS"]

    os.makedirs(out_dir, exist_ok=True)
    set_seed(1234)
    device = torch.device(device_str)

    expr_tr, roots_tr = load_npz_expr_roots(train_npz)
    expr_va, roots_va = load_npz_expr_roots(val_npz)
    expr_te, roots_te = load_npz_expr_roots(test_npz)

    # max_len auto
    max_len_rule = cfg.model.max_sequence_length
    max_len = estimate_len_from_train(expr_tr, stoi, max_len_rule, sanitize=True)

    # scale
    scale = compute_scale_from_train_roots(roots_tr)

    ds_tr = ExprCenterASTDataset(expr_tr, roots_tr, max_len=max_len, stoi=stoi, pad_id=pad_id, unk_id=unk_id, cls_id=cls_id, sanitize=True)
    ds_va = ExprCenterASTDataset(expr_va, roots_va, max_len=max_len, stoi=stoi, pad_id=pad_id, unk_id=unk_id, cls_id=cls_id, sanitize=True)
    ds_te = ExprCenterASTDataset(expr_te, roots_te, max_len=max_len, stoi=stoi, pad_id=pad_id, unk_id=unk_id, cls_id=cls_id, sanitize=True)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.training.batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0)

    model = ASTPrefixTransformerTopK(
        vocab_size=len(vocab),
        max_len=max_len,
        top_k=cfg.model.top_k,
        d_model=cfg.model.hidden_dim,
        nhead=cfg.model.heads,
        num_layers=cfg.model.layers,
        num_token="QM",
        stoi=stoi,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)

    ckpt_path = Path(out_dir) / "best.pt"
    meta_path = Path(out_dir) / "meta.json"

    # save meta/config resolved
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "cfg_path": cfg_path,
            "train_npz": train_npz,
            "val_npz": val_npz,
            "test_npz": test_npz,
            "resolved": {
                "vocab": vocab,
                "vocab_size": len(vocab),
                "max_len_rule": max_len_rule,
                "max_len": int(max_len),
                "top_k": int(cfg.model.top_k),
                "layers": int(cfg.model.layers),
                "heads": int(cfg.model.heads),
                "hidden_dim": int(cfg.model.hidden_dim),
                "scale": float(scale),
                "loss_type": cfg.loss.type,
            }
        }, f, ensure_ascii=False, indent=2)

    print(f"[CFG]  {cfg_path}")
    print(f"[DATA] train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)}")
    print(f"[VOCAB] size={len(vocab)}  max_len={max_len} (rule={max_len_rule})  top_k={cfg.model.top_k}")
    print(f"[ARCH] layers={cfg.model.layers} heads={cfg.model.heads} hidden_dim={cfg.model.hidden_dim}")
    print(f"[SCALE] asinh/sinh scale={scale:.6g}")
    print(f"[OUT]  {out_dir}")

    best_val = float("inf")

    for ep in range(1, cfg.training.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_sum = 0

        pbar = tqdm(dl_tr, desc=f"train ep{ep}/{cfg.training.epochs}", ncols=120)
        for ids, numvals, attn_u8, roots, mask_u8 in pbar:
            ids = ids.to(device)
            numvals = numvals.to(device)
            attn_u8 = attn_u8.to(device)

            roots = roots.to(device)                 # (B,Kroot) float64
            mask = (mask_u8.to(device) > 0)          # bool

            opt.zero_grad(set_to_none=True)

            y = model(ids, numvals, attn_u8)         # (B,Kcand) float32
            cands = (float(scale) * torch.sinh(y.double())).double()

            loss = loss_max_absolute_error(cands, roots, mask)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = ids.size(0)
            loss_sum += float(loss.item()) * bs
            n_sum += bs
            pbar.set_postfix(loss=f"{loss_sum/max(1,n_sum):.6f}")

        va = eval_metrics(model, dl_va, device, scale=scale)
        val_loss_proxy = va["max_mae"]  # max_abs_error 관점에서 max_mae를 proxy로 사용
        print(
            f"[VAL] ep={ep:03d} "
            f"min_mae={va['min_mae']:.6g} min_p90={va['min_p90']:.6g} "
            f"max_mae={va['max_mae']:.6g} max_p90={va['max_p90']:.6g} n={va['n']}"
        )

        # best 저장 기준: max_mae가 낮을수록 좋음
        if np.isfinite(val_loss_proxy) and val_loss_proxy < best_val:
            best_val = float(val_loss_proxy)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "scale": float(scale),
                    "max_len": int(max_len),
                    "vocab": vocab,
                    "stoi": stoi,
                    "best_val_max_mae": float(best_val),
                    "cfg_path": cfg_path,
                },
                ckpt_path
            )
            print(f"[SAVE] best -> {ckpt_path} (best_val_max_mae={best_val:.6g})")

    # test with best
    if ckpt_path.exists():
        obj = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(obj["model_state"])
        scale_best = float(obj["scale"])
        te = eval_metrics(model, dl_te, device, scale=scale_best)
        print(
            f"[TEST] "
            f"min_mae={te['min_mae']:.6g} min_p90={te['min_p90']:.6g} min_p99={te['min_p99']:.6g} | "
            f"max_mae={te['max_mae']:.6g} max_p90={te['max_p90']:.6g} max_p99={te['max_p99']:.6g} | "
            f"n={te['n']}"
        )
    print("[DONE]")


@torch.no_grad()
def eval_only(cfg_path: str, test_npz: str, ckpt_path: str, device_str: str) -> None:
    cfg = load_config(cfg_path)

    obj = torch.load(ckpt_path, map_location="cpu")
    vocab = obj["vocab"]
    stoi = obj["stoi"]
    pad_id = stoi["PAD"]
    unk_id = stoi["UNK"]
    cls_id = stoi["CLS"]
    max_len = int(obj["max_len"])
    scale = float(obj["scale"])

    device = torch.device(device_str)

    expr_te, roots_te = load_npz_expr_roots(test_npz)
    ds_te = ExprCenterASTDataset(expr_te, roots_te, max_len=max_len, stoi=stoi, pad_id=pad_id, unk_id=unk_id, cls_id=cls_id, sanitize=True)
    dl_te = DataLoader(ds_te, batch_size=cfg.training.batch_size, shuffle=False, num_workers=0)

    model = ASTPrefixTransformerTopK(
        vocab_size=len(vocab),
        max_len=max_len,
        top_k=cfg.model.top_k,
        d_model=cfg.model.hidden_dim,
        nhead=cfg.model.heads,
        num_layers=cfg.model.layers,
        num_token="QM",
        stoi=stoi,
    ).to(device)
    model.load_state_dict(obj["model_state"])

    te = eval_metrics(model, dl_te, device, scale=scale)
    print(
        f"[EVAL-TEST] "
        f"min_mae={te['min_mae']:.6g} min_p90={te['min_p90']:.6g} min_p99={te['min_p99']:.6g} | "
        f"max_mae={te['max_mae']:.6g} max_p90={te['max_p90']:.6g} max_p99={te['max_p99']:.6g} | "
        f"n={te['n']}"
    )


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, default)).strip())
    except Exception:
        return int(default)


def _candidate_interval_npz_paths(data_dir: str, degree: int, split: str):
    return [
        f"{data_dir}/taylor_deg{degree}_{split}.npz",
        f"{data_dir}/{split}.npz",
        f"{data_dir}/{split}_deg{degree}.npz",
        f"{data_dir}/deg{degree}_{split}.npz",
        f"{data_dir}/taylor_{split}_deg{degree}.npz",
        f"{data_dir}/{degree}_{split}.npz",
    ]


def _default_interval_npz(data_dir: str, degree: int, split: str) -> str:
    for cand in _candidate_interval_npz_paths(data_dir, degree, split):
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
    return _candidate_interval_npz_paths(data_dir, degree, split)[0]


# -------------------------
# Fallback path utils (used when src.path_utils is unavailable)
# -------------------------
def _fallback_find_repo_root(start_file: str) -> Path:
    p = Path(start_file).resolve().parent
    for cand in [p, *p.parents]:
        if (cand / 'configs').exists() or (cand / '.git').exists():
            return cand
    return p


def _fallback_resolve_repo_path(path_str: str, repo: Path):
    if path_str is None:
        return None
    s = str(path_str).strip()
    if s == '':
        return None
    p = Path(s)
    return p if p.is_absolute() else (repo / p)


def _fallback_resolve_device(device_str: str) -> str:
    s = str(device_str).strip().lower()
    if s in ('', 'auto'):
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return s


def _load_path_utils(start_file: str):
    try:
        from src.path_utils import find_repo_root, resolve_repo_path, resolve_device
        return find_repo_root, resolve_repo_path, resolve_device
    except Exception:
        return _fallback_find_repo_root, _fallback_resolve_repo_path, _fallback_resolve_device


def main():
    import os
    import torch
    find_repo_root, resolve_repo_path, resolve_device = _load_path_utils(__file__)

    repo = find_repo_root(__file__)

    # defaults (repo-root 기준 상대경로)
    degree     = _env_int("TAYLOR_ORDER", 25)
    data_dir   = os.environ.get("DATA_DIR_INTERVAL", "data/taylor_data_physchem_v4_interval")
    cfg_path   = os.environ.get("CFG_PATH", "configs/transformer_interval.yaml")
    device_str = os.environ.get("DEVICE", "auto")
    out_dir    = os.environ.get("OUT_DIR", f"results/transformer_interval/deg{degree}")
    mode       = os.environ.get("MODE", "train").strip().lower()

    train_npz = os.environ.get("TRAIN_NPZ", _default_interval_npz(data_dir, degree, "train"))
    val_npz   = os.environ.get("VAL_NPZ",   _default_interval_npz(data_dir, degree, "val"))
    test_npz  = os.environ.get("TEST_NPZ",  _default_interval_npz(data_dir, degree, "test"))

    cfg_path_p  = resolve_repo_path(cfg_path, repo)
    out_dir_p   = resolve_repo_path(out_dir, repo)
    train_npz_p = resolve_repo_path(train_npz, repo)
    val_npz_p   = resolve_repo_path(val_npz, repo)
    test_npz_p  = resolve_repo_path(test_npz, repo)

    device_str = resolve_device(device_str)

    if mode == "train":
        train_from_yaml(
            cfg_path=str(cfg_path_p),
            train_npz=str(train_npz_p),
            val_npz=str(val_npz_p),
            test_npz=(str(test_npz_p) if test_npz_p is not None else None),
            out_dir=str(out_dir_p),
            device_str=device_str,
        )
    else:
        ckpt_path = (out_dir_p / "best.pt")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
        eval_only(
            cfg_path=str(cfg_path_p),
            test_npz=str(test_npz_p),
            ckpt_path=str(ckpt_path),
            device_str=device_str,
        )


if __name__ == "__main__":
    main()
