#!/usr/bin/env python3
"""
Stage0 full training script (coeffs only) + Taylor-polynomial residual loss/metrics
(anchors=25 fixed)

요구사항 반영(핵심):
  (A) 모델 출력은 25차원 logits
      - w = softmax(logits / T)
      - (참고용) x_mix = Σ w_j * anchor_j  (연속 예측 1개)

  (B) 학습 loss는 "기대 잔차(Expectation residual)"로 계산 (미분 가능)
      - fx_mse:   loss = mean_i Σ_j w_ij * (P(anchor_j)^2)
      - nres_mse: loss = mean_i Σ_j w_ij * (P(anchor_j)/denom(anchor_j))^2
      - NaN/Inf 항목은 큰 값(패널티)으로 치환하여 스킵 대신 벌점화

  (C) 평가지표(metric/score)는 "Top-1 선택(anchor argmax w)" 기준
      - j* = argmax_j w_j
      - x_hat = anchor[j*]
      - fx_hat = P(x_hat), nres_hat = P(x_hat)/denom(x_hat)

추가 안전장치:
  - anchors 범위(out_scale) 제한
  - 필요시 x_clip
  - grad clipping
  - loss nan_to_num 방어(최종 안전)
"""

import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =========================
# Dataset
# =========================

class RootDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================
# Model: logits (B,25) -> softmax weights -> x_mix (B,1)
# =========================

class MLPAnchoredRoot(nn.Module):
    """
    출력: logits (B, m=25)
    w = softmax(logits/T)
    x_mix = Σ w_j * anchor_j  -> (B,1)
    """
    def __init__(self, in_dim, hidden_dim=256, m: int = 25, out_scale: float = 10.0, temperature: float = 1.0):
        super().__init__()
        assert m == 25, "요구사항: output layer 차원은 25로 고정입니다."
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.m = m
        self.out_scale = float(out_scale)
        self.temperature = float(temperature)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, m),  # logits
        )

        anchors = torch.linspace(-self.out_scale, self.out_scale, steps=self.m).view(1, self.m)  # (1,25)
        self.register_buffer("anchors", anchors)

    def forward(self, x, return_logits: bool = False):
        logits = self.net(x)  # (B,25)
        T = max(self.temperature, 1e-6)
        w = torch.softmax(logits / T, dim=1)  # (B,25)
        x_mix = (w * self.anchors).sum(dim=1, keepdim=True)  # (B,1)
        if return_logits:
            return x_mix, logits, w
        return x_mix


# =========================
# Data load
# =========================

def load_split(data_dir, degree, split):
    path = Path(data_dir) / f"taylor_deg{degree}_{split}.npz"
    data = np.load(path, allow_pickle=True)
    coeffs = data["coeffs"].astype(np.float32)
    root0  = data["root0"].astype(np.float32)
    return coeffs, root0


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# Torch polynomial eval (P(x))
# =========================

def poly_eval_and_norm_torch(coeffs: torch.Tensor, x: torch.Tensor, eps: float = 1e-15):
    """
    coeffs: (B, D) ascending a0..a_deg
    x: (B,1) or (B,M)  (broadcast)
    return:
      fx    : P(x)             shape (B,1) or (B,M)
      denom : sum|a_k||x|^k    shape (B,1) or (B,M)
    """
    x_abs = torch.abs(x)
    a = coeffs

    fx = a[:, -1].unsqueeze(-1)      # (B,1)
    denom = torch.abs(a[:, -1]).unsqueeze(-1)

    for k in range(a.size(1) - 2, -1, -1):
        fx = fx * x + a[:, k].unsqueeze(-1)
        denom = denom * x_abs + torch.abs(a[:, k]).unsqueeze(-1)

    return fx, denom + eps


# =========================
# Loss 선택
# =========================

def build_loss_mode(loss_type: str):
    if loss_type == "fx_mse":
        return "fx_mse"
    elif loss_type == "nres_mse":
        return "nres_mse"
    elif loss_type == "mse":
        return "root_mse"
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


# =========================
# Root error stats (참고용)
# =========================

def compute_relative_error_stats(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    abs_err = np.abs(y_pred - y_true)
    denom_true = np.maximum(np.abs(y_true), eps)
    rel_err = abs_err / denom_true

    denom_sym = np.abs(y_pred) + np.abs(y_true)
    sym_err = 2.0 * abs_err / (denom_sym + eps)

    def stats_from_vec(v: np.ndarray):
        v = v.astype(float).reshape(-1)
        return {
            "mean": float(v.mean()),
            "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90.0)),
            "p99": float(np.percentile(v, 99.0)),
            "max": float(v.max()),
        }

    return stats_from_vec(abs_err), stats_from_vec(rel_err), stats_from_vec(sym_err)


# =========================
# Training routine
# =========================

def train_single_stage(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    hidden_dim: int,
    loss_type: str,
    device: str,
    epochs: int,
    batch_size: int,
    patience: int,
    stage_name: str = "stage0",
    best_ckpt_path: str | None = None,
    out_scale: float = 10.0,
    temperature: float = 1.0,
    x_clip: float | None = None,
    grad_clip: float = 1.0,
    fx_baseline_x: float = 0.0,
    eps_r2: float = 1e-12,
    invalid_penalty: float = 1e6,
):
    device_t = torch.device(device)

    in_dim = X_train.shape[1]
    model = MLPAnchoredRoot(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        m=25,
        out_scale=float(out_scale),
        temperature=float(temperature),
    ).to(device_t)

    # x_clip 기본값: anchor range
    if x_clip is None:
        x_clip = float(out_scale)

    loss_mode = build_loss_mode(loss_type)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_ds = RootDataset(X_train, y_train)
    val_ds   = RootDataset(X_val,   y_val)
    test_ds  = RootDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    # x-space R2(참고용)
    y_train_all = y_train.reshape(-1, 1)
    y_val_all   = y_val.reshape(-1, 1)
    train_mean = float(y_train_all.mean())
    val_mean   = float(y_val_all.mean())
    ss_tot_train = float(((y_train_all - train_mean) ** 2).sum())
    ss_tot_val   = float(((y_val_all - val_mean) ** 2).sum())

    # early stopping 기준: loss_mode가 residual이면 "낮을수록 좋음"
    best_score = float("inf")
    best_state = None
    no_improve = 0

    best_ckpt_path_obj = None
    if best_ckpt_path is not None:
        best_ckpt_path_obj = Path(best_ckpt_path)
        best_ckpt_path_obj.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{stage_name}] Best model will be saved to: {best_ckpt_path_obj}")

    # 고정 anchors 준비 (B 확장은 매 배치에서)
    anchors_1m = model.anchors  # (1,25)

    def residual_sq_from_fx(fx, denom):
        if loss_mode == "fx_mse":
            r2 = fx * fx
        elif loss_mode == "nres_mse":
            r = fx / denom
            r2 = r * r
        else:
            raise RuntimeError("residual_sq_from_fx called in non-residual mode")
        return r2

    t_start = time.perf_counter()
    epochs_ran = 0
    pbar = tqdm(range(1, epochs + 1), desc=f"{stage_name} epochs", total=epochs, ncols=140)

    for ep in pbar:
        epochs_ran = ep

        # ----------------
        # train
        # ----------------
        model.train()
        train_loss_sum = 0.0
        n_train = 0

        sse_x_train = 0.0

        # Top-1(anchor chosen by model) 기반 metric 누적
        sse_fx_train = 0.0
        ss0_fx_train = 0.0
        sse_nres_train = 0.0
        ss0_nres_train = 0.0
        cnt_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device_t)  # (B,D)
            yb = yb.to(device_t)  # (B,1)

            opt.zero_grad(set_to_none=True)

            x_mix, logits, w = model(xb, return_logits=True)  # x_mix: (B,1), w:(B,25)

            # 후보 x들: anchors를 batch로 확장 (B,25)
            x_cand = anchors_1m.expand(xb.size(0), -1)
            # 안전 clamp
            x_cand = torch.clamp(x_cand, -x_clip, x_clip)

            fx_cand, denom_cand = poly_eval_and_norm_torch(xb, x_cand)  # (B,25)
            r2 = residual_sq_from_fx(fx_cand, denom_cand)              # (B,25)

            # NaN/Inf -> 패널티로 치환 (스킵 대신 벌점)
            finite = torch.isfinite(r2)
            r2_safe = torch.where(finite, r2, torch.full_like(r2, float(invalid_penalty)))

            if loss_mode in ("fx_mse", "nres_mse"):
                # ✅ 기대 잔차 loss: Σ w_j * r2_j
                loss = (w * r2_safe).sum(dim=1).mean()
            else:
                # 참고용 x-space mse
                x_pred = torch.clamp(x_mix, -x_clip, x_clip)
                loss = torch.mean((x_pred - yb) ** 2)

            # 최종 안전 (희귀 NaN 방어)
            loss = torch.nan_to_num(loss, nan=invalid_penalty, posinf=invalid_penalty, neginf=invalid_penalty)

            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            bs = xb.size(0)
            train_loss_sum += float(loss.item()) * bs
            n_train += bs

            # x-space (참고): 우리는 Top-1 선택을 예측값으로 간주해서 x-error 계산
            with torch.no_grad():
                # Top-1 anchor 선택: argmax w
                idx = torch.argmax(w, dim=1)  # (B,)
                x_hat = x_cand[torch.arange(bs, device=device_t), idx].unsqueeze(1)  # (B,1)

                sse_x_train += ((x_hat - yb) ** 2).sum().item()

                # chosen residual
                fx_hat, denom_hat = poly_eval_and_norm_torch(xb, x_hat)  # (B,1)
                fx_hat = fx_hat.squeeze(1)
                denom_hat = denom_hat.squeeze(1)

                # baseline at x0
                x0 = torch.full_like(x_hat, float(fx_baseline_x))
                x0 = torch.clamp(x0, -x_clip, x_clip)
                fx0, denom0 = poly_eval_and_norm_torch(xb, x0)
                fx0 = fx0.squeeze(1)
                denom0 = denom0.squeeze(1)

                fx2 = fx_hat * fx_hat
                fx02 = fx0 * fx0

                nres2 = (fx_hat / denom_hat) ** 2
                nres02 = (fx0 / denom0) ** 2

                m = torch.isfinite(fx2) & torch.isfinite(fx02) & torch.isfinite(nres2) & torch.isfinite(nres02)
                if m.any():
                    sse_fx_train += fx2[m].sum().item()
                    ss0_fx_train += fx02[m].sum().item()
                    sse_nres_train += nres2[m].sum().item()
                    ss0_nres_train += nres02[m].sum().item()
                    cnt_train += int(m.sum().item())

        train_loss = train_loss_sum / max(1, n_train)
        train_x_R2 = 1.0 - sse_x_train / (ss_tot_train + 1e-12)

        train_fx_R2 = float("nan")
        train_nres_R2 = float("nan")
        if cnt_train > 0 and ss0_fx_train > eps_r2:
            train_fx_R2 = 1.0 - sse_fx_train / (ss0_fx_train + eps_r2)
        if cnt_train > 0 and ss0_nres_train > eps_r2:
            train_nres_R2 = 1.0 - sse_nres_train / (ss0_nres_train + eps_r2)

        # ----------------
        # val
        # ----------------
        model.eval()
        val_loss_sum = 0.0
        n_val = 0

        sse_x_val = 0.0

        sse_fx_val = 0.0
        ss0_fx_val = 0.0
        sse_nres_val = 0.0
        ss0_nres_val = 0.0
        cnt_val = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device_t)
                yb = yb.to(device_t)

                x_mix, logits, w = model(xb, return_logits=True)

                bs = xb.size(0)
                x_cand = anchors_1m.expand(bs, -1)
                x_cand = torch.clamp(x_cand, -x_clip, x_clip)

                fx_cand, denom_cand = poly_eval_and_norm_torch(xb, x_cand)
                r2 = residual_sq_from_fx(fx_cand, denom_cand)

                finite = torch.isfinite(r2)
                r2_safe = torch.where(finite, r2, torch.full_like(r2, float(invalid_penalty)))

                if loss_mode in ("fx_mse", "nres_mse"):
                    loss = (w * r2_safe).sum(dim=1).mean()
                else:
                    x_pred = torch.clamp(x_mix, -x_clip, x_clip)
                    loss = torch.mean((x_pred - yb) ** 2)

                loss = torch.nan_to_num(loss, nan=invalid_penalty, posinf=invalid_penalty, neginf=invalid_penalty)

                val_loss_sum += float(loss.item()) * bs
                n_val += bs

                # Top-1 선택
                idx = torch.argmax(w, dim=1)
                x_hat = x_cand[torch.arange(bs, device=device_t), idx].unsqueeze(1)

                sse_x_val += ((x_hat - yb) ** 2).sum().item()

                fx_hat, denom_hat = poly_eval_and_norm_torch(xb, x_hat)
                fx_hat = fx_hat.squeeze(1)
                denom_hat = denom_hat.squeeze(1)

                x0 = torch.full_like(x_hat, float(fx_baseline_x))
                x0 = torch.clamp(x0, -x_clip, x_clip)
                fx0, denom0 = poly_eval_and_norm_torch(xb, x0)
                fx0 = fx0.squeeze(1)
                denom0 = denom0.squeeze(1)

                fx2 = fx_hat * fx_hat
                fx02 = fx0 * fx0
                nres2 = (fx_hat / denom_hat) ** 2
                nres02 = (fx0 / denom0) ** 2

                m = torch.isfinite(fx2) & torch.isfinite(fx02) & torch.isfinite(nres2) & torch.isfinite(nres02)
                if m.any():
                    sse_fx_val += fx2[m].sum().item()
                    ss0_fx_val += fx02[m].sum().item()
                    sse_nres_val += nres2[m].sum().item()
                    ss0_nres_val += nres02[m].sum().item()
                    cnt_val += int(m.sum().item())

        val_loss = val_loss_sum / max(1, n_val)
        val_x_R2 = 1.0 - sse_x_val / (ss_tot_val + 1e-12)

        val_fx_R2 = float("nan")
        val_nres_R2 = float("nan")
        if cnt_val > 0 and ss0_fx_val > eps_r2:
            val_fx_R2 = 1.0 - sse_fx_val / (ss0_fx_val + eps_r2)
        if cnt_val > 0 and ss0_nres_val > eps_r2:
            val_nres_R2 = 1.0 - sse_nres_val / (ss0_nres_val + eps_r2)

        # early score: residual loss가 낮을수록 좋음
        early_score = val_loss

        pbar.set_postfix(
            train_loss=f"{train_loss:.3e}",
            val_loss=f"{val_loss:.3e}",
            xR2=f"{val_x_R2:.3f}",
            fxR2=("nan" if np.isnan(val_fx_R2) else f"{val_fx_R2:.3f}"),
            nR2=("nan" if np.isnan(val_nres_R2) else f"{val_nres_R2:.3f}"),
            finite=str(cnt_val),
        )

        print(
            f"[{stage_name}] Epoch {ep:04d}: "
            f"train_loss={train_loss:.6e}, val_loss={val_loss:.6e}, "
            f"train_xR2={train_x_R2:.6f}, val_xR2={val_x_R2:.6f}, "
            f"train_fxR2={train_fx_R2}, val_fxR2={val_fx_R2}, "
            f"train_nresR2={train_nres_R2}, val_nresR2={val_nres_R2}, "
            f"finite_cnt(train/val)={cnt_train}/{cnt_val}, "
            f"early_score(val_loss)={early_score:.6e}"
        )

        # ----------------
        # early stopping + best save
        # ----------------
        improved = (early_score < best_score) and np.isfinite(early_score)

        if patience > 0:
            if improved:
                best_score = early_score
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                no_improve = 0

                if best_ckpt_path_obj is not None:
                    torch.save(
                        {
                            "epoch": ep,
                            "model_state": best_state,
                            "best_score": float(best_score),
                            "metrics": {
                                "train_loss": float(train_loss),
                                "val_loss": float(val_loss),
                                "train_xR2": float(train_x_R2),
                                "val_xR2": float(val_x_R2),
                                "train_fxR2": (None if np.isnan(train_fx_R2) else float(train_fx_R2)),
                                "val_fxR2": (None if np.isnan(val_fx_R2) else float(val_fx_R2)),
                                "train_nresR2": (None if np.isnan(train_nres_R2) else float(train_nres_R2)),
                                "val_nresR2": (None if np.isnan(val_nres_R2) else float(val_nres_R2)),
                                "finite_cnt_train": int(cnt_train),
                                "finite_cnt_val": int(cnt_val),
                            },
                            "config": {
                                "hidden_dim": hidden_dim,
                                "loss_type": loss_type,
                                "loss_mode": loss_mode,
                                "m_fixed": 25,
                                "out_scale": float(out_scale),
                                "temperature": float(temperature),
                                "x_clip": float(x_clip),
                                "grad_clip": float(grad_clip),
                                "fx_baseline_x": float(fx_baseline_x),
                                "invalid_penalty": float(invalid_penalty),
                                "train_loss_def": "E_w[ residual^2(anchor) ] (with penalty for non-finite)",
                                "metric_def": "Top-1 anchor (argmax w) residual / R2 vs baseline x0",
                            },
                        },
                        best_ckpt_path_obj,
                    )
                    print(f"[{stage_name}] ★ New best saved at epoch {ep} (val_loss={best_score:.6e}) → {best_ckpt_path_obj}")
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"[{stage_name}] Early stopping at epoch {ep} (patience={patience}, best_val_loss={best_score:.6e})")
                break

    t_end = time.perf_counter()
    elapsed = t_end - t_start
    per_epoch = elapsed / max(1, epochs_ran)
    print(f"[{stage_name}] Training time: {elapsed:.2f}s (epochs_ran={epochs_ran}, per_epoch≈{per_epoch:.4f}s)")

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device_t)

    # ----------------
    # test
    # ----------------
    model.eval()
    y_test_all = y_test.reshape(-1, 1)
    test_mean = float(y_test_all.mean())
    ss_tot_test = float(((y_test_all - test_mean) ** 2).sum())

    sse_x_test = 0.0
    n_test = 0
    preds_list = []

    sse_fx_test = 0.0
    ss0_fx_test = 0.0
    sse_nres_test = 0.0
    ss0_nres_test = 0.0
    cnt_test = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device_t)
            yb = yb.to(device_t)

            x_mix, logits, w = model(xb, return_logits=True)
            bs = xb.size(0)

            x_cand = anchors_1m.expand(bs, -1)
            x_cand = torch.clamp(x_cand, -x_clip, x_clip)

            idx = torch.argmax(w, dim=1)
            x_hat = x_cand[torch.arange(bs, device=device_t), idx].unsqueeze(1)  # (B,1)

            preds_list.append(x_hat.cpu().numpy())

            sse_x_test += ((x_hat - yb) ** 2).sum().item()
            n_test += bs

            fx_hat, denom_hat = poly_eval_and_norm_torch(xb, x_hat)
            fx_hat = fx_hat.squeeze(1)
            denom_hat = denom_hat.squeeze(1)

            x0 = torch.full_like(x_hat, float(fx_baseline_x))
            x0 = torch.clamp(x0, -x_clip, x_clip)
            fx0, denom0 = poly_eval_and_norm_torch(xb, x0)
            fx0 = fx0.squeeze(1)
            denom0 = denom0.squeeze(1)

            fx2 = fx_hat * fx_hat
            fx02 = fx0 * fx0
            nres2 = (fx_hat / denom_hat) ** 2
            nres02 = (fx0 / denom0) ** 2

            m = torch.isfinite(fx2) & torch.isfinite(fx02) & torch.isfinite(nres2) & torch.isfinite(nres02)
            if m.any():
                sse_fx_test += fx2[m].sum().item()
                ss0_fx_test += fx02[m].sum().item()
                sse_nres_test += nres2[m].sum().item()
                ss0_nres_test += nres02[m].sum().item()
                cnt_test += int(m.sum().item())

    y_pred_all = np.vstack(preds_list) if len(preds_list) > 0 else np.zeros((0, 1), dtype=np.float32)

    test_mse_x = sse_x_test / max(1, n_test)
    test_x_R2  = 1.0 - sse_x_test / (ss_tot_test + 1e-12)

    test_fx_R2 = float("nan")
    test_nres_R2 = float("nan")
    if cnt_test > 0 and ss0_fx_test > eps_r2:
        test_fx_R2 = 1.0 - sse_fx_test / (ss0_fx_test + eps_r2)
    if cnt_test > 0 and ss0_nres_test > eps_r2:
        test_nres_R2 = 1.0 - sse_nres_test / (ss0_nres_test + eps_r2)

    print(f"\n[{stage_name}] Test (x-space) MSE = {test_mse_x:.6e}, x_R2 = {test_x_R2:.6f}")
    print(f"[{stage_name}] Test (fx-based)  fx_R2 = {test_fx_R2}  (finite_cnt={cnt_test}, baseline x={fx_baseline_x})")
    print(f"[{stage_name}] Test (nres-based) nres_R2 = {test_nres_R2} (finite_cnt={cnt_test}, baseline x={fx_baseline_x})")

    if y_pred_all.shape[0] > 0:
        abs_stats, rel_stats, sym_stats = compute_relative_error_stats(
            y_true=y_test_all[:y_pred_all.shape[0]],
            y_pred=y_pred_all,
            eps=1e-8,
        )
        print("\n[Root error stats on Test set] (x-space, Top-1 anchor)")
        print(f"  |r_pred - r_true| mean={abs_stats['mean']:.3e}, p90={abs_stats['p90']:.3e}, p99={abs_stats['p99']:.3e}, max={abs_stats['max']:.3e}")
        print(f"  rel(true)       mean={rel_stats['mean']:.3e}, p90={rel_stats['p90']:.3e}, p99={rel_stats['p99']:.3e}, max={rel_stats['max']:.3e}")
        print(f"  sym(SMAPE)      mean={sym_stats['mean']:.3e}, p90={sym_stats['p90']:.3e}, p99={sym_stats['p99']:.3e}, max={sym_stats['max']:.3e}")

    return model, test_mse_x, test_x_R2, test_fx_R2, test_nres_R2


def main():
    parser = argparse.ArgumentParser(description="Stage0 training: coeffs -> root0, Expectation residual loss over 25 anchors, metric=Top-1 anchor")
    parser.add_argument("--data-dir", type=str, default="./taylor_data_generalize_1000000_composed_degree_30")
    parser.add_argument("--degree", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=26)

    parser.add_argument("--loss-type", type=str, default="fx_mse", choices=["fx_mse", "nres_mse", "mse"])
    parser.add_argument("--save_prefix", type=str, default="v4_expect25")

    parser.add_argument("--fx-baseline-x", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--x-clip", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--invalid-penalty", type=float, default=1e6, help="NaN/Inf residual 항목에 부여할 패널티 값")

    args = parser.parse_args()

    print("========== Stage0 training (coeffs -> root0) with EXPECTATION residual over 25 anchors ==========")
    print(f"Using device   : {args.device}")
    print(f"DATA_DIR       : {args.data_dir}")
    print(f"DEGREE         : {args.degree}")
    print(f"EPOCHS         : {args.epochs}")
    print(f"PATIENCE       : {args.patience}")
    print(f"HIDDEN_DIM     : {args.hidden_dim}")
    print(f"LOSS_TYPE      : {args.loss_type}")
    print(f"BATCH_SIZE     : {args.batch_size}")
    print(f"FX_BASELINE_X  : {args.fx_baseline_x}")
    print(f"TEMP           : {args.temperature}")
    print(f"X_CLIP         : {args.x_clip}")
    print(f"GRAD_CLIP      : {args.grad_clip}")
    print(f"INVALID_PENALTY: {args.invalid_penalty}")
    print("===============================================================================================\n")

    set_seed(1234)

    # data
    X_train, y_train = load_split(args.data_dir, args.degree, "train")
    X_val,   y_val   = load_split(args.data_dir, args.degree, "val")
    X_test,  y_test  = load_split(args.data_dir, args.degree, "test")

    # anchor range(out_scale): label root0 기반으로 잡되, 너무 크면 수치 폭발 위험
    max_abs_root0_train = float(np.abs(y_train).max())
    margin = 1.5
    out_scale = margin * max_abs_root0_train

    # 안전 상한(원하면 조절): 데이터가 [-3,3] 근처면 3~5 정도가 보통 더 안정적
    # out_scale = min(out_scale, 5.0)

    print(f"Train root0 max |r| = {max_abs_root0_train:.6f}")
    print(f"Use anchor range out_scale = {out_scale:.6f} -> anchors in [-out_scale, +out_scale]")

    out_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = out_dir / (
        f"{args.save_prefix}_deg{args.degree}_mlp_hd{args.hidden_dim}_best_{args.loss_type}.pt"
    )

    model, test_mse_x, test_x_R2, test_fx_R2, test_nres_R2 = train_single_stage(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        hidden_dim=args.hidden_dim,
        loss_type=args.loss_type,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        stage_name=f"stage0_root0_{args.loss_type}_expect25",
        best_ckpt_path=str(best_ckpt_path),
        out_scale=float(out_scale),
        temperature=float(args.temperature),
        x_clip=args.x_clip,
        grad_clip=args.grad_clip,
        fx_baseline_x=args.fx_baseline_x,
        invalid_penalty=float(args.invalid_penalty),
    )

    final_ckpt_path = out_dir / (
        f"{args.save_prefix}_deg{args.degree}_mlp_hd{args.hidden_dim}_final_{args.loss_type}.pt"
    )
    torch.save(model.state_dict(), final_ckpt_path)

    print(f"\n[Saved] BEST snapshot -> {best_ckpt_path}")
    print(f"[Saved] FINAL state_dict -> {final_ckpt_path}")
    print(f"[Result] x_MSE={test_mse_x:.6e}, x_R2={test_x_R2:.6f}, fx_R2={test_fx_R2}, nres_R2={test_nres_R2}")
    print("\n[Done] Stage0 training finished.")


if __name__ == "__main__":
    main()
