"""
expr_center_ast_regression_len2_topk_es.py

- 입력: expr_str (수식 문자열)
- 인코딩: Python AST 파싱 -> Prefix 토큰 시퀀스
- 출력: K개의 구간(길이 2) 중심 후보 c_k (회귀)
- 정답 판정(hit): min_{k} min_{j} |c_k - r_j| <= half_width

Loss (multi-root + multi-candidate):
  dist_i = min_k min_j |c_{i,k} - r_{i,j}|
  margin_loss = mean( relu(dist_i - half_width)^2 )

+ (선택) tie-break: 가장 가까운 후보를 nearest root로 약하게 끌어주는 MSE 항

데이터셋:
  {data_dir}/taylor_deg25_train.npz
  {data_dir}/taylor_deg25_val.npz
  {data_dir}/taylor_deg25_test.npz

각 npz 키(필수):
  - expr_str: object array (N,)
  - roots:    float32 array (N,Kroot) with NaN padding

핵심 옵션:
  --auto-max-len : train set 토큰 길이의 p99 * 1.1로 max_len 자동 설정
  --num-candidates : 출력 후보 개수 K
  --patience : early stopping patience
"""

import argparse
import math
import re
import ast
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =========================
# sanitize to Python-expr-friendly for AST
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

    s = re.sub(r"\s*\([^()]*\)\s*$", "", s).strip()
    s = s.replace("^", "**")
    s = re.sub(r"\bnp\.", "", s)
    s = re.sub(r"\bln\s*\(", "log(", s)  # ln -> log
    return s


# =========================
# AST -> Prefix tokens
# =========================

_BASE_TOKENS = [
    "<PAD>", "<UNK>", "<CLS>",
    "x", "NUM",
    "+", "-", "*", "/", "**",
    "neg", "pos",
]
_FUNC_TOKENS = sorted(list({
    "sin","cos","tan","tanh","sinh","cosh","exp","log","log10","sqrt","abs"
}))

VOCAB = _BASE_TOKENS + _FUNC_TOKENS
STOI = {t: i for i, t in enumerate(VOCAB)}
PAD_ID = STOI["<PAD>"]
UNK_ID = STOI["<UNK>"]
CLS_ID = STOI["<CLS>"]
NUM_ID = STOI["NUM"]

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
                emit("NUM", float(n.value))
                return
            emit("<UNK>", 0.0)
            return

        if isinstance(n, ast.Name):
            if n.id == "x":
                emit("x", 0.0)
            else:
                emit("<UNK>", 0.0)
            return

        if isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.USub):
                emit("neg", 0.0)
            elif isinstance(n.op, ast.UAdd):
                emit("pos", 0.0)
            else:
                emit("<UNK>", 0.0)
            visit(n.operand)
            return

        if isinstance(n, ast.BinOp):
            if isinstance(n.op, ast.Add):
                emit("+", 0.0)
            elif isinstance(n.op, ast.Sub):
                emit("-", 0.0)
            elif isinstance(n.op, ast.Mult):
                emit("*", 0.0)
            elif isinstance(n.op, ast.Div):
                emit("/", 0.0)
            elif isinstance(n.op, ast.Pow):
                emit("**", 0.0)
            else:
                emit("<UNK>", 0.0)
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
                fname = "log" if fname == "ln" else fname
                emit(fname, 0.0)
            else:
                emit("<UNK>", 0.0)

            if len(n.args) >= 1:
                visit(n.args[0])
            else:
                emit("<UNK>", 0.0)
            return

        emit("<UNK>", 0.0)

    visit(node)
    return tokens, nums

def encode_prefix(tokens, nums, max_len: int):
    toks = ["<CLS>"] + tokens
    nvs = [0.0] + nums

    if len(toks) > max_len:
        toks = toks[:max_len]
        nvs = nvs[:max_len]

    ids = np.array([_tok_id(t) for t in toks], dtype=np.int64)
    numvals = np.array(nvs, dtype=np.float32)

    attn = np.ones((len(ids),), dtype=np.bool_)
    if len(ids) < max_len:
        pad_n = max_len - len(ids)
        ids = np.concatenate([ids, np.full((pad_n,), PAD_ID, dtype=np.int64)], axis=0)
        numvals = np.concatenate([numvals, np.zeros((pad_n,), dtype=np.float32)], axis=0)
        attn = np.concatenate([attn, np.zeros((pad_n,), dtype=np.bool_)], axis=0)

    return ids, numvals, attn


# =========================
# Dataset
# =========================

class ExprCenterASTDataset(Dataset):
    def __init__(self, expr_arr, roots_arr, max_len: int, sanitize: bool = True):
        self.expr = expr_arr
        self.roots = roots_arr.astype(np.float64)
        self.max_len = int(max_len)
        self.do_sanitize = bool(sanitize)

    def __len__(self):
        return self.roots.shape[0]

    def __getitem__(self, idx):
        e = str(self.expr[idx])
        if self.do_sanitize:
            e = sanitize_expr_for_ast(e)

        try:
            node = ast.parse(e, mode="eval")
            toks, nums = ast_to_prefix(node)
        except Exception:
            toks, nums = ["<UNK>"], [0.0]

        ids, numvals, attn = encode_prefix(toks, nums, self.max_len)

        r = self.roots[idx]
        mask = np.isfinite(r)

        return (
            torch.from_numpy(ids),                         # (L,) int64
            torch.from_numpy(numvals),                     # (L,) float32
            torch.from_numpy(attn.astype(np.uint8)),       # (L,) uint8
            torch.from_numpy(r),                           # (Kroot,) float64
            torch.from_numpy(mask.astype(np.uint8)),       # (Kroot,) uint8
        )


# =========================
# Model
# =========================

class ASTPrefixTransformerTopK(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, num_candidates: int,
                 d_model: int = 256, nhead: int = 8, num_layers: int = 4):
        super().__init__()
        self.max_len = int(max_len)
        self.d_model = int(d_model)
        self.K = int(num_candidates)

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.num_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # cls -> K개의 y 출력
        self.y_head = nn.Linear(d_model, self.K)

    def forward(self, ids: torch.Tensor, numvals: torch.Tensor, attn_u8: torch.Tensor):
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, L)

        x = self.tok_emb(ids) + self.pos_emb(pos)

        is_num = (ids == NUM_ID).unsqueeze(-1)  # (B,L,1)
        num_embed = self.num_mlp(numvals.unsqueeze(-1))  # (B,L,D)
        x = x + num_embed * is_num

        key_padding_mask = (attn_u8 == 0)  # True=ignore
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)

        cls = h[:, 0, :]            # (B,D)
        y = self.y_head(cls)        # (B,K)
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
      best_dist: (B,) float64  = min_k min_j |c_k - r_j|
      best_k:    (B,) int64    = argmin over candidates
      nearest_r: (B,) float64  = nearest root for that best candidate
    """
    # (B,Kcand,Kroot)
    diff = torch.abs(cands.unsqueeze(-1) - roots.unsqueeze(1))

    inf = torch.tensor(float("inf"), device=diff.device, dtype=diff.dtype)
    diff = torch.where(mask.unsqueeze(1), diff, inf)

    # min over roots -> (B,Kcand)
    min_over_roots, argmin_root = torch.min(diff, dim=-1)

    # min over candidates -> (B,)
    best_dist, best_k = torch.min(min_over_roots, dim=1)

    # nearest root index for chosen candidate
    chosen_root_idx = argmin_root.gather(1, best_k.unsqueeze(1)).squeeze(1)  # (B,)
    nearest = roots.gather(1, chosen_root_idx.unsqueeze(1)).squeeze(1)

    return best_dist, best_k, nearest

def margin_interval_loss_multiroot_topk(cands: torch.Tensor, roots: torch.Tensor, mask: torch.Tensor, half_width: float):
    best_dist, _, _ = min_dist_candidates_to_roots(cands, roots, mask)
    viol = torch.relu(best_dist - float(half_width))
    return torch.mean(viol * viol)

@torch.no_grad()
def eval_metrics(model, loader, device, scale: float, half_width: float):
    model.eval()
    hit = 0
    n = 0
    all_err = []

    for ids, numvals, attn_u8, roots, mask_u8 in loader:
        ids = ids.to(device)
        numvals = numvals.to(device)
        attn_u8 = attn_u8.to(device)

        roots = roots.to(device)
        mask = (mask_u8.to(device) > 0)

        y = model(ids, numvals, attn_u8)                      # (B,K) float32
        cands = (float(scale) * torch.sinh(y.double())).double()  # (B,K) float64

        best_dist, _, _ = min_dist_candidates_to_roots(cands, roots, mask)

        ok = (best_dist <= float(half_width))
        hit += int(ok.sum().item())
        n += int(best_dist.numel())

        bd = best_dist.detach().cpu().numpy()
        bd = bd[np.isfinite(bd)]
        if bd.size:
            all_err.append(bd)

    if n == 0 or len(all_err) == 0:
        return {"hit": float("nan"), "mae": float("nan"), "p90": float("nan"), "p99": float("nan"), "n": int(n)}

    all_err = np.concatenate(all_err, axis=0)
    return {
        "hit": hit / max(1, n),
        "mae": float(all_err.mean()),
        "p90": float(np.percentile(all_err, 90.0)),
        "p99": float(np.percentile(all_err, 99.0)),
        "n": int(n),
    }


# =========================
# Load data
# =========================

def load_split(data_dir: Path, split: str):
    path = data_dir / f"taylor_deg25_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    data = np.load(path, allow_pickle=True)
    expr = data["expr_str"]
    roots = data["roots"]
    return expr, roots

def compute_scale_from_train_roots(roots_train: np.ndarray):
    r = np.asarray(roots_train, dtype=np.float64).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 1.0
    scale = float(np.percentile(np.abs(r), 99.0))
    return max(scale, 1.0)

def estimate_max_len_p99(expr_arr, sanitize: bool, mul: float = 1.1, p: float = 99.0, cap: int = 24):
    lens = []
    for e in expr_arr:
        s = str(e)
        if sanitize:
            s = sanitize_expr_for_ast(s)
        try:
            node = ast.parse(s, mode="eval")
            toks, nums = ast_to_prefix(node)
            L = 1 + len(toks)  # +CLS
        except Exception:
            L = 2
        lens.append(L)

    lens = np.asarray(lens, dtype=np.float64)
    p99 = float(np.percentile(lens, p))
    max_len = int(math.ceil(p99 * float(mul)))
    max_len = max(16, min(max_len, int(cap)))
    return max_len, p99


# =========================
# Train with EarlyStopping
# =========================

def train(
    data_dir: Path,
    ckpt_path: Path,
    num_candidates: int,
    auto_max_len: bool,
    max_len: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    batch_size: int,
    epochs: int,
    lr: float,
    device: str,
    half_width: float,
    tie_mse_weight: float,
    sanitize_inputs: bool,
    patience: int,
    min_delta: float,
    metric: str,  # "hit" or "mae"
):
    device = torch.device(device)

    expr_tr, roots_tr = load_split(data_dir, "train")
    expr_va, roots_va = load_split(data_dir, "val")
    expr_te, roots_te = load_split(data_dir, "test")

    if auto_max_len:
        autoL, p99 = estimate_max_len_p99(expr_tr, sanitize=sanitize_inputs, mul=1.1, p=99.0, cap=2048)
        print(f"[MAX_LEN] p99_len={p99:.1f} -> ceil(p99*1.1)={autoL}")
        max_len = autoL

    scale = compute_scale_from_train_roots(roots_tr)
    print(f"[SCALE] asinh/sinh scale = {scale:.6g} (p99(|roots_train|))")
    print(f"[VOCAB] size={len(VOCAB)}  max_len={max_len}  K_candidates={num_candidates}")

    ds_tr = ExprCenterASTDataset(expr_tr, roots_tr, max_len=max_len, sanitize=sanitize_inputs)
    ds_va = ExprCenterASTDataset(expr_va, roots_va, max_len=max_len, sanitize=sanitize_inputs)
    ds_te = ExprCenterASTDataset(expr_te, roots_te, max_len=max_len, sanitize=sanitize_inputs)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ASTPrefixTransformerTopK(
        vocab_size=len(VOCAB),
        max_len=max_len,
        num_candidates=num_candidates,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # early stopping state
    best_score = None
    bad = 0

    def is_better(new, best):
        if best is None:
            return True
        if metric == "hit":
            return new > (best + min_delta)
        else:
            # mae: smaller is better
            return new < (best - min_delta)

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n_sum = 0

        pbar = tqdm(dl_tr, desc=f"train ep{ep}/{epochs}", ncols=120)
        for ids, numvals, attn_u8, roots, mask_u8 in pbar:
            ids = ids.to(device)
            numvals = numvals.to(device)
            attn_u8 = attn_u8.to(device)

            roots = roots.to(device)                 # (B,Kroot) float64
            mask = (mask_u8.to(device) > 0)          # bool

            opt.zero_grad(set_to_none=True)
            y = model(ids, numvals, attn_u8)         # (B,Kcand) float32
            cands = (scale * torch.sinh(y.double())).double()

            loss_margin = margin_interval_loss_multiroot_topk(
                cands, roots, mask, half_width=half_width
            )

            if tie_mse_weight > 0:
                best_dist, best_k, nearest = min_dist_candidates_to_roots(cands, roots, mask)
                chosen_c = cands.gather(1, best_k.unsqueeze(1)).squeeze(1)
                loss_tie = torch.mean((chosen_c - nearest) ** 2)
                loss = loss_margin + float(tie_mse_weight) * loss_tie
            else:
                loss = loss_margin

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = ids.size(0)
            loss_sum += float(loss.item()) * bs
            n_sum += bs
            pbar.set_postfix(loss=f"{loss_sum/max(1,n_sum):.6f}")

        va = eval_metrics(model, dl_va, device, scale=scale, half_width=half_width)
        score = va["hit"] if metric == "hit" else va["mae"]
        print(f"[VAL] ep={ep:03d}  hit={va['hit']:.4f}  mae={va['mae']:.4f}  p90={va['p90']:.4f}  n={va['n']}  score({metric})={score:.6g}")

        if is_better(score, best_score):
            best_score = score
            bad = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "vocab": VOCAB,
                        "max_len": int(max_len),
                        "num_candidates": int(num_candidates),
                        "d_model": int(d_model),
                        "nhead": int(nhead),
                        "num_layers": int(num_layers),
                        "scale": float(scale),
                        "half_width": float(half_width),
                        "tie_mse_weight": float(tie_mse_weight),
                        "sanitize_inputs": bool(sanitize_inputs),
                        "earlystop": {
                            "patience": int(patience),
                            "min_delta": float(min_delta),
                            "metric": str(metric),
                        }
                    },
                    "best_score": float(best_score),
                },
                ckpt_path
            )
            print(f"[SAVE] new best -> {ckpt_path} (best_{metric}={best_score:.6g})")
        else:
            bad += 1
            if bad >= patience:
                print(f"[EARLY STOP] no improvement for {patience} epochs. stop at ep={ep}.")
                break

    # test with best
    obj = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(obj["model_state"])
    scale_best = float(obj["config"]["scale"])
    te = eval_metrics(model, dl_te, device, scale=scale_best, half_width=float(obj["config"]["half_width"]))
    print(f"[TEST] hit={te['hit']:.4f}  mae={te['mae']:.4f}  p90={te['p90']:.4f}  p99={te['p99']:.4f}  n={te['n']}")


@torch.no_grad()
def run_eval(data_dir: Path, ckpt_path: Path, batch_size: int, device: str):
    device = torch.device(device)
    obj = torch.load(ckpt_path, map_location=device)
    cfg = obj["config"]

    max_len = int(cfg["max_len"])
    scale = float(cfg["scale"])
    half_width = float(cfg.get("half_width", 1.0))
    sanitize_inputs = bool(cfg.get("sanitize_inputs", True))
    num_candidates = int(cfg.get("num_candidates", 1))

    expr_te, roots_te = load_split(data_dir, "test")
    ds_te = ExprCenterASTDataset(expr_te, roots_te, max_len=max_len, sanitize=sanitize_inputs)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ASTPrefixTransformerTopK(
        vocab_size=len(VOCAB),
        max_len=max_len,
        num_candidates=num_candidates,
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        num_layers=int(cfg["num_layers"]),
    ).to(device)
    model.load_state_dict(obj["model_state"])

    te = eval_metrics(model, dl_te, device, scale=scale, half_width=half_width)
    print(f"[EVAL-TEST] hit={te['hit']:.4f}  mae={te['mae']:.4f}  p90={te['p90']:.4f}  p99={te['p99']:.4f}  n={te['n']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    p.add_argument("--data-dir", type=str, default="./taylor_data_physchem_v4_interval")
    p.add_argument("--ckpt", type=str, default="./expr_center_ast_best_50.pt")

    p.add_argument("--num-candidates", type=int, default=50, help="출력 interval 후보 개수 K")
    p.add_argument("--auto-max-len", action="store_true", help="train split 토큰길이 p99*1.1로 max_len 자동 설정")
    p.add_argument("--max-len", type=int, default=25)

    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--half-width", type=float, default=1e-6, help="interval half width (길이2면 1)")
    p.add_argument("--tie-mse-weight", type=float, default=0.0)

    p.add_argument("--no-sanitize", action="store_true")

    # early stopping
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--metric", type=str, default="hit", choices=["hit", "mae"])

    args = p.parse_args()
    data_dir = Path(args.data_dir)
    ckpt_path = Path(args.ckpt)

    sanitize_inputs = (not args.no_sanitize)

    if args.mode == "train":
        train(
            data_dir=data_dir,
            ckpt_path=ckpt_path,
            num_candidates=int(args.num_candidates),
            auto_max_len=bool(args.auto_max_len),
            max_len=int(args.max_len),
            d_model=int(args.d_model),
            nhead=int(args.nhead),
            num_layers=int(args.num_layers),
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            lr=float(args.lr),
            device=str(args.device),
            half_width=float(args.half_width),
            tie_mse_weight=float(args.tie_mse_weight),
            sanitize_inputs=bool(sanitize_inputs),
            patience=int(args.patience),
            min_delta=float(args.min_delta),
            metric=str(args.metric),
        )
    else:
        run_eval(
            data_dir=data_dir,
            ckpt_path=ckpt_path,
            batch_size=max(256, int(args.batch_size)),
            device=str(args.device),
        )

if __name__ == "__main__":
    main()
