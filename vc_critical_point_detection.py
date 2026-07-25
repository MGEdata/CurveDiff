import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union

# --- plotting (headless safe) ---
matplotlib.use("Agg")


# ============================================================
# Behavior classes
# ============================================================
BEHAVIOR_NAME_TO_CODE = {
    "active_dissolution": 0,
    "active_passive": 1,
    "passive_with_breakdown": 2,
    "cyclic_pitting": 3,
    "transpassive": 4,
    "ambiguous": 5,
}
BEHAVIOR_CODE_TO_NAME = {v: k for k, v in BEHAVIOR_NAME_TO_CODE.items()}

STEEL_HINT_FROM_BEHAVIOR = {
    "active_dissolution": "ordinary/non-passivating steel-like",
    "active_passive": "stainless/passivating steel-like",
    "passive_with_breakdown": "pitting-sensitive stainless-steel-like",
    "cyclic_pitting": "cyclic pitting-sensitive stainless-steel-like",
    "transpassive": "high-potential passivating alloy-like",
    "ambiguous": "ambiguous",
}

# ============================================================
# Parsing: final_point string -> (strain, stress) arrays
# ============================================================
_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")

def parse_polarization_points(v: Union[str, float, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a polarization-curve string into:
        voltage array x
        current-density array y

    Expected line format examples:
        '0.10 -1e-4\\n0.12 -8e-5\\n...'
        '0.10, -1e-4\\n0.12, -8e-5\\n...'

    Returns:
        x: np.ndarray
        y: np.ndarray
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    text = str(v).strip()
    if not text:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    xs, ys = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        nums = _NUM_RE.findall(line.replace(",", " "))
        if len(nums) < 2:
            continue

        xs.append(float(nums[0]))  # voltage
        ys.append(float(nums[1]))  # current density

    if len(xs) == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def recommend_polarization_normalization_ranges(
    df_train: pd.DataFrame,
    *,
    points_col: str = "final_point",
    q_low: float = 0.005,
    q_high: float = 0.995,
    margin: float = 0.02,
    clamp_voltage_min_zero: bool = False,
    clamp_current_min_zero: bool = False,
):
    """
    Recommended (voltage_min, voltage_max, current_min, current_max) from TRAIN set only.

    Uses per-curve extrema:
        voltage_min from quantile of per-curve min voltage
        voltage_max from quantile of per-curve max voltage
        current_min from quantile of per-curve min current density
        current_max from quantile of per-curve max current density

    Notes:
    - For corrosion polarization curves, clamp_voltage_min_zero=False is usually correct.
    - For signed current density, clamp_current_min_zero=False is usually correct.
    """

    voltage_min_list, voltage_max_list = [], []
    current_min_list, current_max_list = [], []

    for v in df_train[points_col].values:
        xs, ys = parse_polarization_points(v)
        if xs.size == 0 or ys.size == 0:
            continue

        x = xs[np.isfinite(xs)]
        y = ys[np.isfinite(ys)]

        y = np.abs(y)
        if x.size == 0 or y.size == 0:
            continue

        voltage_min_list.append(float(x.min()))
        voltage_max_list.append(float(x.max()))
        current_min_list.append(float(y.min()))
        current_max_list.append(float(y.max()))

    if len(voltage_min_list) == 0 or len(current_min_list) == 0:
        raise ValueError(f"No valid curves in df_train['{points_col}'].")

    voltage_min_arr = np.array(voltage_min_list, dtype=float)
    voltage_max_arr = np.array(voltage_max_list, dtype=float)
    current_min_arr = np.array(current_min_list, dtype=float)
    current_max_arr = np.array(current_max_list, dtype=float)

    # ---- voltage bounds ----
    v_min_raw = float(np.quantile(voltage_min_arr, q_low))
    v_max_raw = float(np.quantile(voltage_max_arr, q_high))
    v_span = max(v_max_raw - v_min_raw, 1e-12)

    voltage_min = 0.0 if clamp_voltage_min_zero else (v_min_raw - margin * v_span)
    voltage_max = v_max_raw + margin * v_span

    # ---- current-density bounds ----
    c_min_raw = float(np.quantile(current_min_arr, q_low))
    c_max_raw = float(np.quantile(current_max_arr, q_high))
    c_span = max(c_max_raw - c_min_raw, 1e-12)

    current_min = c_min_raw - margin * c_span
    current_max = c_max_raw + margin * c_span

    if clamp_current_min_zero:
        current_min = max(0.0, current_min)

    # ---- nice rounding ----
    # voltage: usually round to 0.001 V
    voltage_min = float(np.floor(voltage_min * 1000.0) / 1000.0)
    voltage_max = float(np.ceil(voltage_max * 1000.0) / 1000.0)

    # current density:
    # choose adaptive rounding based on magnitude
    c_abs_max = max(abs(current_min), abs(current_max), 1e-12)

    if c_abs_max >= 1.0:
        # large current density
        current_min = float((current_min * 1000.0) / 1000.0)
        current_max = float((current_max * 1000.0) / 1000.0)
    elif c_abs_max >= 1e-3:
        current_min = float((current_min * 1e6) / 1e6)
        current_max = float((current_max * 1e6) / 1e6)
    else:
        # very small current density
        current_min = float((current_min * 1e9) / 1e9)
        current_max = float((current_max * 1e9) / 1e9)

    return voltage_min, voltage_max, current_min, current_max


def parse_final_point_xy(s: Union[str, float, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse 'final_point' string like:
        '0 0\n0.142... 117...\n...'

    Robust to extra spaces / tabs / commas.

    Returns:
        strain: (N,)
        stress: (N,)
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    text = str(s).strip()
    if not text:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    xs, ys = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Extract the first two numbers on the line (works with "x y", "x, y", etc.)
        nums = _NUM_RE.findall(line.replace(",", " "))
        if len(nums) < 2:
            continue
        x = float(nums[0])
        y = float(nums[1])
        xs.append(x)
        ys.append(y)

    if len(xs) == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def load_batch_from_final_point(
    df: pd.DataFrame,
    *,
    col: str = "final_point",
    row_indices: Optional[List[int]] = None,
    L: int = 128,
    pad_mode: str = "edge",  # "edge" or "nan"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build batch tensors (B,L) from df[col] strings.

    Returns:
        strain_t: (B,L) float32
        stress_t: (B,L) float32
        valid_n: (B,)  int64  (number of parsed points per row, clamped to [0,L])
    """
    if row_indices is None:
        row_indices = list(range(len(df)))

    B = len(row_indices)
    strain = np.full((B, L), np.nan, dtype=np.float64)
    stress = np.full((B, L), np.nan, dtype=np.float64)
    valid_n = np.zeros((B,), dtype=np.int64)

    for bi, ridx in enumerate(row_indices):
        s = df.iloc[ridx][col]
        x, y = parse_final_point_xy(s)
        n = int(min(len(x), len(y), L))
        valid_n[bi] = n
        if n <= 0:
            continue
        strain[bi, :n] = x[:n]
        stress[bi, :n] = y[:n]

        # pad remaining points (if any)
        if n < L:
            if pad_mode == "edge":
                strain[bi, n:] = strain[bi, n - 1]
                stress[bi, n:] = stress[bi, n - 1]
            else:
                # keep NaN
                pass

    strain_t = torch.tensor(strain, dtype=torch.float32)
    stress_t = torch.tensor(stress, dtype=torch.float32)
    valid_n_t = torch.tensor(valid_n, dtype=torch.long)
    return strain_t, stress_t, valid_n_t


# ============================================================
# Device helper
# ============================================================
def _resolve_device(
    device: Optional[Union[str, torch.device]],
    *refs: torch.Tensor,
) -> torch.device:
    """
    Resolve target device.

    Rules:
    - if device is given, use it
    - if device is None, use the first tensor's device
    - if CUDA is requested but unavailable, raise an error
    """
    if device is None:
        for ref in refs:
            if isinstance(ref, torch.Tensor):
                return ref.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' was requested, but CUDA is not available.")
    return dev


# ============================================================
# Faster helpers
# ============================================================
def _finite_prefix_length(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Count contiguous finite points from the start.
    Vectorized version.
    """
    m = torch.isfinite(x) & torch.isfinite(y)
    if m.ndim != 2:
        raise ValueError("x,y must be (B,L)")

    _, L = m.shape
    bad = ~m
    has_bad = bad.any(dim=1)
    first_bad = bad.to(torch.int64).argmax(dim=1)
    return torch.where(has_bad, first_bad, torch.full_like(first_bad, L))


def _moving_average_1d(y: torch.Tensor, win: int) -> torch.Tensor:
    y = y.view(-1)
    n = y.numel()
    if n < 3 or win <= 1:
        return y.clone()

    w = int(win)
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w < 3:
        return y.clone()

    pad = w // 2
    y_pad = F.pad(y.view(1, 1, -1), (pad, pad), mode="replicate")
    kernel = torch.ones((1, 1, w), device=y.device, dtype=y.dtype) / float(w)
    out = F.conv1d(y_pad, kernel)[0, 0]
    return out


def _safe_gradient(y: torch.Tensor, x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    dy/dx with simple central-difference-like approximation.
    """
    n = y.numel()
    g = torch.zeros_like(y)
    if n <= 1:
        return g

    dx = x[1:] - x[:-1]
    dx = torch.where(dx.abs() < eps, torch.full_like(dx, eps), dx)
    s = (y[1:] - y[:-1]) / dx

    g[0] = s[0]
    g[-1] = s[-1]
    if n > 2:
        g[1:-1] = 0.5 * (s[:-1] + s[1:])
    return g


def _find_reversal_index_1d(x1d: torch.Tensor, min_points: int = 5) -> int:
    """
    Find first robust reversal point in acquisition order.
    Returns raw index of reversal start, or -1 if not found.
    """
    n1 = int(x1d.numel())
    if n1 < max(6, 2 * min_points + 1):
        return -1

    dx = x1d[1:] - x1d[:-1]
    nz = torch.where(dx.abs() > 1e-12)[0]
    if nz.numel() < min_points:
        return -1

    first = dx[nz[:min(int(nz.numel()), min_points)]]
    init_sign = torch.sign(first.median())
    if float(init_sign.item()) == 0.0:
        nonzero_first = first[first != 0]
        if nonzero_first.numel() == 0:
            return -1
        init_sign = torch.sign(nonzero_first[0])

    candidates = torch.where((dx * init_sign) < 0)[0]
    if candidates.numel() == 0:
        return -1

    for c in candidates:
        c = int(c.item())
        if c < min_points - 1:
            continue
        hi = min(len(dx), c + min_points)
        seg = dx[c:hi]
        if seg.numel() < max(2, min_points // 2):
            continue
        if int(((seg * init_sign) < 0).sum().item()) >= max(2, min_points // 2):
            rev = c + 1
            if (n1 - rev) >= max(2, min_points // 2):
                return rev
    return -1


def _find_plateau_center_idx(
    z1d: torch.Tensor,
    start: int,
    end: int,
    tol: float,
    min_run: int,
) -> int:
    """
    Find a passive plateau representative point from a low-current plateau.
    Returns center index of the longest low-current run.
    """
    if end < start:
        return -1

    seg = z1d[start:end + 1]
    if seg.numel() == 0:
        return -1

    z_min = torch.min(seg)
    mask = (seg <= (z_min + float(tol)))

    best_s = -1
    best_e = -1
    i = 0
    m = int(mask.numel())

    while i < m:
        if bool(mask[i].item()):
            j = i
            while j + 1 < m and bool(mask[j + 1].item()):
                j += 1
            if (j - i + 1) >= int(min_run):
                if best_s < 0 or (j - i) > (best_e - best_s):
                    best_s, best_e = i, j
            i = j + 1
        else:
            i += 1

    if best_s >= 0:
        return start + (best_s + best_e) // 2

    return start + int(torch.argmin(seg).item())


def _apply_local_keypoint_updates_(
    weights_row: torch.Tensor,
    key_mask_row: torch.Tensor,
    neighbor_mask_row: torch.Tensor,
    idx: int,
    *,
    n: int,
    neighbor_count: int,
    alpha_neighbor: float,
    key_boost: float = 0.0,
) -> None:
    """
    In-place local update around one keypoint.
    Avoids allocating full-length masks/bump tensors every time.
    """
    if idx < 0 or idx >= n:
        return

    key_mask_row[idx] = True
    if key_boost != 0.0:
        weights_row[idx] += float(key_boost)

    if neighbor_count <= 0:
        return

    lo = max(0, idx - neighbor_count)
    hi = min(n - 1, idx + neighbor_count)
    js = torch.arange(lo, hi + 1, device=weights_row.device)
    keep = js != idx
    if not torch.any(keep):
        return

    neighbor_mask_row[js[keep]] = True

    if alpha_neighbor > 0.0:
        d = (js - idx).abs().to(weights_row.dtype)
        w = (1.0 - d / float(neighbor_count + 1)).clamp(min=0.0)
        weights_row[js[keep]] += float(alpha_neighbor) * w[keep]


def _topk_curvature_inflections(
    d2: torch.Tensor,
    lo: int,
    hi: int,
    *,
    topk: int = 5,
    min_spacing: int = 4,
    min_rel_strength: float = 0.18,
    forbidden: Optional[List[int]] = None,
) -> Tuple[List[int], List[float]]:
    """
    Pick the strongest local curvature-critical points from |d2|.
    """
    n = int(d2.numel())
    lo = max(1, lo)
    hi = min(n - 2, hi)
    if hi < lo or topk <= 0:
        return [], []

    abs_d2 = d2.abs()
    region = abs_d2[lo:hi + 1]
    if region.numel() == 0:
        return [], []

    max_strength = float(region.max().item())
    if max_strength <= 0.0:
        return [], []

    threshold = float(min_rel_strength) * max_strength

    center = abs_d2[lo:hi + 1]
    left = abs_d2[lo - 1:hi]
    right = abs_d2[lo + 1:hi + 2]

    local_max_mask = (center >= left) & (center >= right) & (center >= threshold)
    cand_rel = torch.nonzero(local_max_mask, as_tuple=False).squeeze(1)

    if cand_rel.numel() > 0:
        cand_idx = lo + cand_rel
        cand_val = center[cand_rel]
    else:
        k = min(int(region.numel()), max(3, int(topk) * 2))
        cand_val, cand_rel = torch.topk(region, k=k, largest=True, sorted=True)
        cand_idx = lo + cand_rel

    if forbidden is not None and len(forbidden) > 0 and cand_idx.numel() > 0:
        forbidden_t = torch.tensor(
            [int(v) for v in forbidden],
            device=d2.device,
            dtype=torch.long,
        )
        keep = (cand_idx[:, None] - forbidden_t[None, :]).abs().amin(dim=1) > int(min_spacing)
        cand_idx = cand_idx[keep]
        cand_val = cand_val[keep]

    if cand_idx.numel() == 0:
        return [], []

    order = torch.argsort(cand_val, descending=True)
    cand_idx = cand_idx[order]
    cand_val = cand_val[order]

    selected_idx: List[int] = []
    selected_strength: List[float] = []

    for idx_t, val_t in zip(cand_idx, cand_val):
        idx = int(idx_t.item())
        strength = float(val_t.item())
        if all(abs(idx - sidx) > int(min_spacing) for sidx in selected_idx):
            selected_idx.append(idx)
            selected_strength.append(strength)
            if len(selected_idx) >= int(topk):
                break

    return selected_idx, selected_strength


def _pick_inflection_idx_from_curvature_fast(
    d1: torch.Tensor,
    d2: torch.Tensor,
    lo: int,
    hi: int,
    *,
    anchor: Optional[int] = None,
    prefer: Optional[str] = None,
) -> int:
    """
    Same logic as before, but vectorized.
    """
    n = int(d2.numel())
    lo = max(1, lo)
    hi = min(n - 2, hi)
    if hi < lo:
        return -1

    a = d2[lo:hi]
    b = d2[lo + 1:hi + 1]

    crossed = (
        (a == 0.0) |
        (b == 0.0) |
        ((a < 0.0) & (b > 0.0)) |
        ((a > 0.0) & (b < 0.0))
    )

    if torch.any(crossed):
        base = torch.arange(lo, hi, device=d2.device)
        cand = torch.where(a.abs() <= b.abs(), base, base + 1)
        cand = cand[crossed]

        left = torch.clamp(cand - 1, min=lo)
        right = torch.clamp(cand + 1, max=hi)

        d2_l = d2[left]
        d2_c = d2[cand]
        d2_r = d2[right]

        curv_amp = torch.maximum(torch.maximum(d2_l.abs(), d2_c.abs()), d2_r.abs())
        slope_jump = (d1[right] - d1[left]).abs()

        score = 1.0 * curv_amp + 0.8 * slope_jump

        if prefer == "negative":
            neg_amp = torch.maximum(
                torch.maximum(torch.relu(-d2_l), torch.relu(-d2_c)),
                torch.relu(-d2_r),
            )
            score = score + 0.6 * neg_amp
        elif prefer == "positive":
            pos_amp = torch.maximum(
                torch.maximum(torch.relu(d2_l), torch.relu(d2_c)),
                torch.relu(d2_r),
            )
            score = score + 0.6 * pos_amp

        if anchor is not None:
            span = max(1, hi - lo + 1)
            dist = (cand - int(anchor)).abs().to(score.dtype) / float(span)
            score = score + 0.35 * (1.0 - dist)

        return int(cand[torch.argmax(score)].item())

    seg = d2[lo:hi + 1]
    if seg.numel() == 0:
        return -1

    if prefer == "negative":
        return lo + int(torch.argmin(seg).item())
    elif prefer == "positive":
        return lo + int(torch.argmax(seg).item())
    else:
        return lo + int(torch.argmax(seg.abs()).item())


# ============================================================
# Batched numeric helpers
# ============================================================
def _moving_average_masked_batch(
    y: torch.Tensor,
    valid_n: torch.Tensor,
    win: int,
) -> torch.Tensor:
    """
    Batched moving average with replicate-like boundary handling.
    """
    if y.ndim != 2:
        raise ValueError("y must be (B,L)")

    B, L = y.shape
    out = y.clone()

    if L < 3 or win <= 1:
        return out

    w = int(win)
    if w % 2 == 0:
        w += 1
    if w < 3:
        return out

    pad = w // 2
    device = y.device

    long_rows = valid_n >= w
    if torch.any(long_rows):
        row_idx = torch.where(long_rows)[0]
        yy = y[row_idx]
        nn = valid_n[row_idx]

        pos = torch.arange(L, device=device, dtype=torch.long)
        offs = torch.arange(-pad, pad + 1, device=device, dtype=torch.long)

        idx = pos.view(1, L, 1) + offs.view(1, 1, w)
        idx = idx.expand(len(row_idx), -1, -1)

        max_idx = (nn.view(-1, 1, 1) - 1).clamp(min=0)
        idx = idx.clamp(min=0)
        idx = torch.minimum(idx, max_idx)

        gathered = yy.gather(1, idx.reshape(len(row_idx), -1)).view(len(row_idx), L, w)
        out[row_idx] = gathered.mean(dim=-1)

    short_rows = torch.where(~long_rows)[0]
    for r in short_rows.tolist():
        n = int(valid_n[r].item())
        if n <= 0:
            continue
        out[r, :n] = _moving_average_1d(y[r, :n], win)

    return out


def _safe_gradient_batch(
    y: torch.Tensor,
    x: torch.Tensor,
    valid_n: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Batched version of _safe_gradient for padded variable-length sequences.
    """
    if y.ndim != 2 or x.ndim != 2:
        raise ValueError("x,y must be (B,L)")

    B, L = y.shape
    g = torch.zeros_like(y)
    if L <= 1:
        return g

    dx = x[:, 1:] - x[:, :-1]
    dx = torch.where(dx.abs() < eps, torch.full_like(dx, eps), dx)
    s = (y[:, 1:] - y[:, :-1]) / dx

    pair_pos = torch.arange(L - 1, device=y.device).view(1, -1)
    pair_valid = pair_pos < (valid_n.view(-1, 1) - 1)
    s = torch.where(pair_valid, s, torch.zeros_like(s))

    rows = torch.where(valid_n > 1)[0]
    if rows.numel() > 0:
        g[rows, 0] = s[rows, 0]
        last_pos = valid_n[rows] - 1
        last_slope = valid_n[rows] - 2
        g[rows, last_pos] = s[rows, last_slope]

    if L > 2:
        interior = 0.5 * (s[:, :-1] + s[:, 1:])
        interior_pos = torch.arange(1, L - 1, device=y.device).view(1, -1)
        interior_valid = interior_pos < (valid_n.view(-1, 1) - 1)
        g[:, 1:-1] = torch.where(interior_valid, interior, torch.zeros_like(interior))

    return g


def _prepare_forward_branches_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    valid_n: torch.Tensor,
    *,
    reversal_buffer: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build padded forward branches for the whole batch.
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x,y must be (B,L)")

    B, L = x.shape
    device = x.device
    dtype = x.dtype

    reversal_idx = torch.full((B,), -1, device=device, dtype=torch.long)
    forward_n = torch.zeros((B,), device=device, dtype=torch.long)

    for b in range(B):
        n = int(valid_n[b].item())
        if n <= 0:
            continue

        xb = x[b, :n]
        rev = _find_reversal_index_1d(xb, min_points=int(max(3, reversal_buffer)))
        reversal_idx[b] = int(rev)

        if rev >= 0:
            f_end = rev
        else:
            f_end = n - 1
        forward_n[b] = int(f_end + 1)

    max_nf = int(forward_n.max().item()) if B > 0 else 0
    if max_nf <= 0:
        xf = torch.empty((B, 0), device=device, dtype=dtype)
        yf = torch.empty((B, 0), device=device, dtype=dtype)
        map_f = torch.empty((B, 0), device=device, dtype=torch.long)
        return xf, yf, map_f, forward_n, reversal_idx

    xf_raw = torch.full((B, max_nf), float("inf"), device=device, dtype=dtype)
    yf_raw = torch.zeros((B, max_nf), device=device, dtype=dtype)
    map_f_raw = torch.full((B, max_nf), -1, device=device, dtype=torch.long)

    for b in range(B):
        nf = int(forward_n[b].item())
        if nf <= 0:
            continue
        xf_raw[b, :nf] = x[b, :nf]
        yf_raw[b, :nf] = y[b, :nf]
        map_f_raw[b, :nf] = torch.arange(nf, device=device, dtype=torch.long)

    order = torch.argsort(xf_raw, dim=1)
    xf_sorted = xf_raw.gather(1, order)
    yf_sorted = yf_raw.gather(1, order)
    map_f_sorted = map_f_raw.gather(1, order)

    return xf_sorted, yf_sorted, map_f_sorted, forward_n, reversal_idx


# ============================================================
# Main function with device control
# ============================================================
# @torch.inference_mode()
# def detect_keypoints_polarization_physical_batch_batched(
#     potential: torch.Tensor,      # (B,L), unnormalized
#     current: torch.Tensor,        # (B,L), SIGNED current density, unnormalized
#     *,
#     alpha_key: float = 5.0,
#     alpha_neighbor: float = 10.0,
#     neighbor_count: int = 3,
#     smooth_win: int = 5,
#     passivation_drop_thresh: float = 0.30,
#     breakdown_rise_thresh: float = 0.50,
#     include_end_point: bool = True,
#     device: Optional[Union[str, torch.device]] = None,
# ) -> Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, List[str]]]]:
#     """
#     Batched heavy-numeric version with explicit device control.

#     device:
#         None    -> keep the input tensor device
#         "cpu"   -> force CPU
#         "cuda"  -> force GPU
#         "cuda:0", "cuda:1", ... also supported
#     """

#     run_device = _resolve_device(device, potential, current)

#     smooth_win_current = smooth_win
#     smooth_win_log = smooth_win

#     passive_min_separation = 4
#     passive_plateau_tol = 0.10
#     passive_plateau_min_points = 3

#     breakdown_confirm_points = 2

#     repassivation_tol = 0.20
#     reversal_buffer = 5

#     min_passivation_score = 2.8
#     min_breakdown_score = 2.0
#     min_repassivation_score = 1.5
#     min_transpassive_score = 1.8

#     max_critical_inflections = 5
#     critical_inflection_min_spacing = max(3, smooth_win)
#     critical_inflection_min_rel_strength = 0.18

#     log_eps = 1e-20

#     def _run_length_near_min(z_seg: torch.Tensor, tol: float) -> Tuple[int, int, int]:
#         if z_seg.numel() == 0:
#             return -1, -1, 0

#         z_min = torch.min(z_seg)
#         mask = (z_seg <= (z_min + float(tol)))

#         best_s = -1
#         best_e = -1
#         i = 0
#         m = int(mask.numel())

#         while i < m:
#             if bool(mask[i].item()):
#                 j = i
#                 while j + 1 < m and bool(mask[j + 1].item()):
#                     j += 1
#                 if best_s < 0 or (j - i) > (best_e - best_s):
#                     best_s, best_e = i, j
#                 i = j + 1
#             else:
#                 i += 1

#         if best_s < 0:
#             return -1, -1, 0
#         return best_s, best_e, (best_e - best_s + 1)

#     if potential.ndim != 2 or current.ndim != 2:
#         raise ValueError("potential and current must be (B,L) tensors")

#     x = potential.to(device=run_device)
#     y = current.to(device=run_device)

#     device_obj = x.device
#     dtype = current.dtype
#     B, L = current.shape

#     valid_n = _finite_prefix_length(x, y)

#     weights = torch.ones((B, L), device=device_obj, dtype=dtype)
#     key_mask = torch.zeros((B, L), device=device_obj, dtype=torch.bool)
#     neighbor_mask = torch.zeros((B, L), device=device_obj, dtype=torch.bool)

#     start_idx = torch.full((B,), -1, device=device_obj, dtype=torch.long)
#     corr_idx = torch.full((B,), -1, device=device_obj, dtype=torch.long)

#     passivation_onset_idx = torch.full((B,), -1, device=device_obj, dtype=torch.long)
#     passivation_inflection_idx = torch.full((B,), -1, device=device_obj, dtype=torch.long)

#     active_peak_idx = torch.full((B,), -1, device=device_obj, dtype=torch.long)
#     passive_idx = torch.full((B,), -1, device=device_obj, dtype=torch.long)

#     breakdown_idx = torch.full((B,), -1, device=device_obj, dtype=torch.long)
#     breakdown_inflection_idx = torch.full((B,), -1, device=device_obj, dtype=torch.long)

#     repassivation_idx = torch.full((B,), -1, device=device_obj, dtype=torch.long)
#     reversal_idx_out = torch.full((B,), -1, device=device_obj, dtype=torch.long)
#     end_idx = torch.full((B,), -1, device=device_obj, dtype=torch.long)

#     critical_inflection_idx = torch.full(
#         (B, max_critical_inflections), -1, device=device_obj, dtype=torch.long
#     )
#     critical_inflection_strength = torch.zeros(
#         (B, max_critical_inflections), device=device_obj, dtype=dtype
#     )

#     behavior_code = torch.full(
#         (B,), BEHAVIOR_NAME_TO_CODE["ambiguous"], device=device_obj, dtype=torch.long
#     )
#     behavior_name_list: List[str] = []
#     steel_hint_list: List[str] = []

#     ecorr = torch.full((B,), float("nan"), device=device_obj, dtype=dtype)
#     current_at_corr_signed = torch.full((B,), float("nan"), device=device_obj, dtype=dtype)
#     icorr = torch.full((B,), float("nan"), device=device_obj, dtype=dtype)

#     e_pass_onset = torch.full((B,), float("nan"), device=device_obj, dtype=dtype)
#     e_crit = torch.full((B,), float("nan"), device=device_obj, dtype=dtype)
#     i_crit = torch.full((B,), float("nan"), device=device_obj, dtype=dtype)
#     e_pass = torch.full((B,), float("nan"), device=device_obj, dtype=dtype)
#     i_pass = torch.full((B,), float("nan"), device=device_obj, dtype=dtype)
#     e_pit = torch.full((B,), float("nan"), device=device_obj, dtype=dtype)
#     e_rp = torch.full((B,), float("nan"), device=device_obj, dtype=dtype)

#     passivation_conf = torch.zeros((B,), device=device_obj, dtype=dtype)
#     breakdown_conf = torch.zeros((B,), device=device_obj, dtype=dtype)

#     # -------------------------------------------------------
#     # Batched forward-branch preparation
#     # -------------------------------------------------------
#     xf_pad, yf_pad, map_f_pad, forward_n, reversal_idx = _prepare_forward_branches_batch(
#         x,
#         y,
#         valid_n,
#         reversal_buffer=reversal_buffer,
#     )
#     reversal_idx_out.copy_(reversal_idx)

#     if xf_pad.shape[1] > 0:
#         ys_pad = _moving_average_masked_batch(yf_pad, forward_n, smooth_win_current)
#         zf_pad = torch.log10(torch.abs(yf_pad).clamp_min(log_eps))
#         zfs_pad = _moving_average_masked_batch(zf_pad, forward_n, smooth_win_log)
#         dzdEf_pad = _safe_gradient_batch(zfs_pad, xf_pad, forward_n)
#         d2zdE2f_pad = _safe_gradient_batch(dzdEf_pad, xf_pad, forward_n)
#     else:
#         ys_pad = yf_pad.clone()
#         zfs_pad = yf_pad.clone()
#         dzdEf_pad = yf_pad.clone()
#         d2zdE2f_pad = yf_pad.clone()

#     # -------------------------------------------------------
#     # Per-curve decision logic
#     # -------------------------------------------------------
#     for b in range(B):
#         n = int(valid_n[b].item())
#         if n < 3:
#             behavior_name_list.append("ambiguous")
#             steel_hint_list.append(STEEL_HINT_FROM_BEHAVIOR["ambiguous"])
#             if n > 0:
#                 start_idx[b] = 0
#                 end_idx[b] = n - 1
#             if n < L:
#                 weights[b, n:] = 0.0
#             continue

#         nf = int(forward_n[b].item())
#         if nf < 3:
#             behavior_name_list.append("ambiguous")
#             steel_hint_list.append(STEEL_HINT_FROM_BEHAVIOR["ambiguous"])
#             if n > 0:
#                 start_idx[b] = 0
#                 end_idx[b] = n - 1
#             if n < L:
#                 weights[b, n:] = 0.0
#             continue

#         xb = x[b, :n]
#         yb = y[b, :n]

#         xf = xf_pad[b, :nf]
#         yf = yf_pad[b, :nf]
#         map_f = map_f_pad[b, :nf]

#         ys = ys_pad[b, :nf]
#         zfs = zfs_pad[b, :nf]
#         dzdEf = dzdEf_pad[b, :nf]
#         d2zdE2f = d2zdE2f_pad[b, :nf]

#         weights_b = weights[b, :n]
#         key_mask_b = key_mask[b, :n]
#         neighbor_mask_b = neighbor_mask[b, :n]

#         i_start = 0
#         i_end = n - 1
#         start_idx[b] = i_start
#         end_idx[b] = i_end

#         rev = int(reversal_idx[b].item())

#         x_span = float((xf[-1] - xf[0]).abs().item()) if nf >= 2 else 1.0
#         x_span = max(x_span, 1e-12)

#         s0 = torch.sign(ys[:-1])
#         s1 = torch.sign(ys[1:])
#         cross_np = torch.where((ys[:-1] <= 0) & (ys[1:] >= 0))[0]
#         cross_any = torch.where(s0 * s1 <= 0)[0]

#         if cross_np.numel() > 0:
#             j = int(cross_np[0].item())
#             i_corr_ref = j if abs(float(ys[j].item())) <= abs(float(ys[j + 1].item())) else (j + 1)
#         elif cross_any.numel() > 0:
#             j = int(cross_any[0].item())
#             i_corr_ref = j if abs(float(ys[j].item())) <= abs(float(ys[j + 1].item())) else (j + 1)
#         else:
#             i_corr_ref = int(torch.argmin(torch.abs(ys)).item())

#         i_corr_local = int(torch.argmin(yf).item())
#         i_corr_raw = int(map_f[i_corr_local].item())

#         corr_idx[b] = i_corr_raw
#         ecorr[b] = xf[i_corr_local]
#         current_at_corr_signed[b] = yf[i_corr_local]
#         icorr[b] = yf[i_corr_local]

#         has_passivation = False
#         has_breakdown = False
#         has_repassivation_capability = False
#         has_repassivation_evidence = False
#         has_transpassive_rise = False

#         passivation_score = 0.0
#         breakdown_score_local = 0.0
#         repassivation_score = 0.0
#         transpassive_score = 0.0

#         i_peak_local = -1
#         i_pass_local = -1
#         i_break_local = -1
#         i_pass_infl_local = -1
#         i_onset_local = -1
#         i_break_infl_local = -1

#         if i_corr_ref < nf - max(6, passive_min_separation + 2):
#             lo = i_corr_ref + 1
#             z_an = zfs[lo:]

#             rel_s, rel_e, run_len = _run_length_near_min(z_an, tol=float(passive_plateau_tol))
#             if rel_s >= 0:
#                 cand_pass_start = lo + rel_s
#                 cand_pass_end = lo + rel_e
#                 cand_pass = (cand_pass_start + cand_pass_end) // 2

#                 if (cand_pass - lo) >= int(passive_min_separation):
#                     rel_peak = int(torch.argmax(zfs[lo:cand_pass + 1]).item())
#                     cand_peak = lo + rel_peak

#                     drop = float((zfs[cand_peak] - zfs[cand_pass]).item())
#                     pre_rise = float((zfs[cand_peak] - zfs[lo]).item())

#                     plateau_seg = zfs[cand_pass_start:cand_pass_end + 1]
#                     plateau_std = float(torch.std(plateau_seg, unbiased=False).item()) if plateau_seg.numel() > 1 else 0.0
#                     plateau_slope = float(torch.mean(torch.abs(dzdEf[cand_pass_start:cand_pass_end + 1])).item()) if plateau_seg.numel() > 0 else 0.0
#                     separation = cand_pass - cand_peak

#                     passivation_score += min(max(drop / max(passivation_drop_thresh, 1e-6), 0.0), 2.0)
#                     if run_len >= int(passive_plateau_min_points):
#                         passivation_score += 0.9
#                     if plateau_std <= float(passive_plateau_tol):
#                         passivation_score += 0.7
#                     elif plateau_std <= float(passive_plateau_tol) * 1.5:
#                         passivation_score += 0.35
#                     if separation >= 2:
#                         passivation_score += 0.5
#                     if pre_rise > 0.05:
#                         passivation_score += 0.3
#                     if plateau_slope <= 0.15:
#                         passivation_score += 0.4

#                     if (drop >= float(passivation_drop_thresh)) and (passivation_score >= float(min_passivation_score)):
#                         has_passivation = True
#                         i_peak_local = cand_peak

#                         i_pass_local = _find_plateau_center_idx(
#                             zfs,
#                             start=cand_pass_start,
#                             end=cand_pass_end,
#                             tol=float(passive_plateau_tol),
#                             min_run=int(max(1, passive_plateau_min_points)),
#                         )
#                         if i_pass_local < 0:
#                             i_pass_local = cand_pass

#                         passivation_conf[b] = zfs[i_peak_local] - zfs[i_pass_local]

#                         if i_peak_local > (i_corr_ref + 1):
#                             i_pass_infl_local = _pick_inflection_idx_from_curvature_fast(
#                                 dzdEf,
#                                 d2zdE2f,
#                                 i_corr_ref + 1,
#                                 i_peak_local,
#                                 anchor=max(i_corr_ref + 1, (i_corr_ref + i_peak_local) // 2),
#                                 prefer="negative",
#                             )
#                             i_onset_local = i_pass_infl_local

#                         if i_onset_local < 0:
#                             i_onset_local = i_peak_local
#                             i_pass_infl_local = i_peak_local

#         if has_passivation and (i_pass_local >= 0) and (i_pass_local < nf - 2):
#             bl = max(0, i_pass_local - 1)
#             br = min(nf, i_pass_local + 2)
#             baseline = float(torch.median(zfs[bl:br]).item())
#             confirm = int(max(1, breakdown_confirm_points))

#             first_break_candidate = -1
#             for j in range(i_pass_local + 1, nf):
#                 rise = float((zfs[j] - baseline).item())
#                 if rise < float(breakdown_rise_thresh):
#                     continue

#                 lo_chk = max(i_pass_local + 1, j - confirm + 1)
#                 if bool(torch.all(dzdEf[lo_chk:j + 1] > 0.0).item()):
#                     first_break_candidate = j
#                     break

#             if first_break_candidate >= 0:
#                 i_break_infl_local = _pick_inflection_idx_from_curvature_fast(
#                     dzdEf,
#                     d2zdE2f,
#                     max(i_pass_local + 1, first_break_candidate - max(2, smooth_win)),
#                     min(nf - 2, first_break_candidate + max(2, smooth_win)),
#                     anchor=first_break_candidate,
#                     prefer="positive",
#                 )

#                 if i_break_infl_local >= 0:
#                     if float((zfs[i_break_infl_local] - baseline).item()) >= 0.5 * float(breakdown_rise_thresh):
#                         i_break_local = i_break_infl_local
#                     else:
#                         i_break_local = first_break_candidate
#                 else:
#                     i_break_local = first_break_candidate

#                 rise = float((zfs[i_break_local] - baseline).item())
#                 late_frac = float((xf[i_break_local] - xf[0]).item()) / x_span
#                 tail_rise = float((zfs[-1] - baseline).item())

#                 score = 0.0
#                 score += min(rise / max(float(breakdown_rise_thresh), 1e-6), 2.0)
#                 if late_frac > 0.55:
#                     score += 0.4
#                 if tail_rise > 0.0:
#                     score += 0.3
#                 if i_break_local - i_pass_local >= 2:
#                     score += 0.3

#                 breakdown_score_local = score
#                 if breakdown_score_local >= float(min_breakdown_score):
#                     has_breakdown = True
#                     breakdown_conf[b] = zfs[i_break_local] - baseline
#                 else:
#                     i_break_local = -1
#                     i_break_infl_local = -1

#         if has_breakdown and (i_break_local >= 0):
#             baseline_tp = float(zfs[i_pass_local].item()) if i_pass_local >= 0 else float(zfs[i_break_local].item())
#             rise_end = float((zfs[-1] - baseline_tp).item())
#             pos_frac = float((xf[i_break_local] - xf[0]).item()) / x_span

#             if pos_frac >= 0.75:
#                 transpassive_score += 0.9
#             if rise_end >= float(breakdown_rise_thresh):
#                 transpassive_score += 0.7
#             if float(dzdEf[-1].item()) > 0.0:
#                 transpassive_score += 0.4

#         if (rev >= 0) and has_breakdown and (i_pass_local >= 0):
#             has_repassivation_capability = True

#             xr = xb[rev:]
#             yr = yb[rev:]
#             if xr.numel() >= 3:
#                 zr = torch.log10(torch.abs(yr).clamp_min(log_eps))
#                 zrs = _moving_average_1d(zr, int(smooth_win_log))

#                 baseline_r = float(zfs[i_pass_local].item())
#                 thr_r = baseline_r + float(repassivation_tol)

#                 jr0 = int(max(1, reversal_buffer // 2))
#                 hit = torch.where(zrs[jr0:] <= thr_r)[0]
#                 if hit.numel() > 0:
#                     jr = jr0 + int(hit[0].item())
#                     rep_raw = rev + jr
#                     repassivation_idx[b] = rep_raw
#                     e_rp[b] = xb[rep_raw]

#                     has_repassivation_evidence = True
#                     repassivation_score += 1.2

#                     z_min_rev = float(torch.min(zrs).item())
#                     if z_min_rev <= baseline_r + 0.5 * float(repassivation_tol):
#                         repassivation_score += 0.6

#         if rev >= 0 and has_breakdown and has_repassivation_evidence and (repassivation_score >= float(min_repassivation_score)):
#             behavior_name = "cyclic_pitting"
#         elif has_passivation and has_breakdown and (transpassive_score >= float(min_transpassive_score)) and (not has_repassivation_evidence):
#             behavior_name = "transpassive"
#             has_transpassive_rise = True
#         elif has_passivation and has_breakdown:
#             behavior_name = "passive_with_breakdown"
#         elif has_passivation:
#             behavior_name = "active_passive"
#         else:
#             behavior_name = "active_dissolution"

#         behavior_name_list.append(behavior_name)
#         steel_hint_list.append(STEEL_HINT_FROM_BEHAVIOR[behavior_name])
#         behavior_code[b] = int(BEHAVIOR_NAME_TO_CODE[behavior_name])

#         if i_onset_local >= 0:
#             passivation_onset_idx[b] = int(map_f[i_onset_local].item())
#             e_pass_onset[b] = xf[i_onset_local]

#         if i_pass_infl_local >= 0:
#             passivation_inflection_idx[b] = int(map_f[i_pass_infl_local].item())

#         if i_peak_local >= 0:
#             active_peak_idx[b] = int(map_f[i_peak_local].item())
#             e_crit[b] = xf[i_peak_local]
#             i_crit[b] = torch.abs(yf[i_peak_local])

#         if i_pass_local >= 0:
#             passive_idx[b] = int(map_f[i_pass_local].item())
#             e_pass[b] = xf[i_pass_local]
#             i_pass[b] = torch.abs(yf[i_pass_local])

#         if i_break_local >= 0:
#             breakdown_idx[b] = int(map_f[i_break_local].item())
#             e_pit[b] = xf[i_break_local]

#         if i_break_infl_local >= 0:
#             breakdown_inflection_idx[b] = int(map_f[i_break_infl_local].item())
#         elif i_break_local >= 0:
#             breakdown_inflection_idx[b] = int(map_f[i_break_local].item())

#         crit_local_idx, crit_strength = _topk_curvature_inflections(
#             d2zdE2f,
#             1,
#             nf - 2,
#             topk=max_critical_inflections,
#             min_spacing=critical_inflection_min_spacing,
#             min_rel_strength=critical_inflection_min_rel_strength,
#             forbidden=[i_corr_local],
#         )

#         max_strength = max(crit_strength) if len(crit_strength) > 0 else 0.0
#         for j, (iloc, sval) in enumerate(zip(crit_local_idx, crit_strength)):
#             raw_idx = int(map_f[iloc].item())
#             critical_inflection_idx[b, j] = raw_idx
#             critical_inflection_strength[b, j] = float(sval)

#             strength_norm = float(sval / max_strength) if max_strength > 0 else 0.0

#             _apply_local_keypoint_updates_(
#                 weights_b,
#                 key_mask_b,
#                 neighbor_mask_b,
#                 raw_idx,
#                 n=n,
#                 neighbor_count=int(neighbor_count),
#                 alpha_neighbor=float(alpha_neighbor) * strength_norm,
#                 key_boost=float(alpha_key) * strength_norm,
#             )

#         key_ids = [i_start, i_corr_raw]

#         p_on = int(passivation_onset_idx[b].item())
#         p_inf = int(passivation_inflection_idx[b].item())
#         a_pk = int(active_peak_idx[b].item())
#         p_ps = int(passive_idx[b].item())
#         b_pk = int(breakdown_idx[b].item())
#         b_inf = int(breakdown_inflection_idx[b].item())
#         r_ps = int(repassivation_idx[b].item())

#         if p_on >= 0:
#             key_ids.append(p_on)
#         if p_inf >= 0:
#             key_ids.append(p_inf)
#         if a_pk >= 0:
#             key_ids.append(a_pk)
#         if p_ps >= 0:
#             key_ids.append(p_ps)
#         if b_pk >= 0:
#             key_ids.append(b_pk)
#         if b_inf >= 0:
#             key_ids.append(b_inf)
#         if r_ps >= 0:
#             key_ids.append(r_ps)
#         if include_end_point:
#             key_ids.append(i_end)

#         key_ids = sorted(set(kk for kk in key_ids if 0 <= kk < n))

#         for kk in key_ids:
#             _apply_local_keypoint_updates_(
#                 weights_b,
#                 key_mask_b,
#                 neighbor_mask_b,
#                 kk,
#                 n=n,
#                 neighbor_count=int(neighbor_count),
#                 alpha_neighbor=float(alpha_neighbor),
#                 key_boost=0.0,
#             )

#         weights_b[key_mask_b] += float(alpha_key)
#         neighbor_mask_b[key_mask_b] = False

#         if n < L:
#             weights[b, n:] = 0.0
#             key_mask[b, n:] = False
#             neighbor_mask[b, n:] = False

#     aux = {
#         "valid_n": valid_n,
#         "behavior_code": behavior_code,
#         "behavior_name": behavior_name_list,
#         "steel_hint": steel_hint_list,
#         "reversal_idx": reversal_idx_out,

#         "start_idx": start_idx,
#         "corr_idx": corr_idx,

#         "passivation_onset_idx": passivation_onset_idx,
#         "passivation_inflection_idx": passivation_inflection_idx,

#         "active_peak_idx": active_peak_idx,
#         "passive_idx": passive_idx,

#         "breakdown_idx": breakdown_idx,
#         "breakdown_inflection_idx": breakdown_inflection_idx,

#         "repassivation_idx": repassivation_idx,
#         "end_idx": end_idx,

#         "critical_inflection_idx": critical_inflection_idx,
#         "critical_inflection_strength": critical_inflection_strength,

#         "ecorr": ecorr,
#         "current_at_corr_signed": current_at_corr_signed,
#         "icorr": icorr,

#         "e_pass_onset": e_pass_onset,
#         "e_crit": e_crit,
#         "i_crit": i_crit,
#         "e_pass": e_pass,
#         "i_pass": i_pass,
#         "e_pit": e_pit,
#         "e_rp": e_rp,

#         "passivation_conf": passivation_conf,
#         "breakdown_conf": breakdown_conf,

#         "key_mask": key_mask,
#         "neighbor_mask": neighbor_mask,
#         "weights": weights,
#     }
#     return weights, aux


@torch.inference_mode()
def detect_keypoints_polarization_physical_batch_batched(
    potential: torch.Tensor,      # (B,L), unnormalized
    current: torch.Tensor,        # (B,L), SIGNED current density, unnormalized
    *,
    alpha_key: float = 5.0,
    alpha_neighbor: float = 10.0,
    neighbor_count: int = 3,
    smooth_win: int = 5,
    passivation_drop_thresh: float = 0.30,
    breakdown_rise_thresh: float = 0.50,
    include_end_point: bool = True,
    device: Optional[Union[str, torch.device]] = None,

    # ---- new critical-inflection controls ----
    critical_topk: int = 8,
    critical_inflection_min_spacing: Optional[int] = None,
    critical_inflection_min_rel_strength: float = 0.10,
    critical_include_zero_crossings: bool = True,
    critical_include_slope_jump: bool = True,
    critical_zero_cross_weight: float = 0.90,
    critical_slope_jump_weight: float = 0.75,
) -> Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, List[str]]]]:
    """
    Batched heavy-numeric version with explicit device control.

    Updated:
    - more sensitive critical-inflection detection
    - combines |d2| peaks, d2 zero-crossings, and strong slope-jump points
    - still keeps passivation / breakdown / repassivation logic mostly unchanged

    device:
        None    -> keep the input tensor device
        "cpu"   -> force CPU
        "cuda"  -> force GPU
        "cuda:0", "cuda:1", ... also supported
    """

    run_device = _resolve_device(device, potential, current)

    smooth_win_current = smooth_win
    smooth_win_log = smooth_win

    passive_min_separation = 4
    passive_plateau_tol = 0.10
    passive_plateau_min_points = 3

    breakdown_confirm_points = 2

    repassivation_tol = 0.20
    reversal_buffer = 5

    min_passivation_score = 2.8
    min_breakdown_score = 2.0
    min_repassivation_score = 1.5
    min_transpassive_score = 1.8

    log_eps = 1e-20

    if critical_inflection_min_spacing is None:
        critical_inflection_min_spacing = max(2, smooth_win // 2 + 1)

    def _run_length_near_min(z_seg: torch.Tensor, tol: float) -> Tuple[int, int, int]:
        if z_seg.numel() == 0:
            return -1, -1, 0

        z_min = torch.min(z_seg)
        mask = (z_seg <= (z_min + float(tol)))

        best_s = -1
        best_e = -1
        i = 0
        m = int(mask.numel())

        while i < m:
            if bool(mask[i].item()):
                j = i
                while j + 1 < m and bool(mask[j + 1].item()):
                    j += 1
                if best_s < 0 or (j - i) > (best_e - best_s):
                    best_s, best_e = i, j
                i = j + 1
            else:
                i += 1

        if best_s < 0:
            return -1, -1, 0
        return best_s, best_e, (best_e - best_s + 1)

    def _collect_critical_inflections_richer(
        d1: torch.Tensor,
        d2: torch.Tensor,
        lo: int,
        hi: int,
        *,
        topk: int,
        min_spacing: int,
        min_rel_strength: float,
        forbidden: Optional[List[int]] = None,
        include_zero_crossings: bool = True,
        include_slope_jump: bool = True,
        zero_cross_weight: float = 0.90,
        slope_jump_weight: float = 0.75,
    ) -> Tuple[List[int], List[float]]:
        """
        Richer critical-inflection miner:
        1) strong local maxima of |d2|
        2) d2 zero-crossings, scored by local curvature + slope jump
        3) strong local maxima of |Δd1|

        This is more permissive than the old pure-|d2|-peak rule and helps catch
        broader / weaker inflections that were being missed.
        """
        n = int(d2.numel())
        lo = max(1, lo)
        hi = min(n - 2, hi)
        if hi < lo or topk <= 0:
            return [], []

        abs_d2 = d2.abs()
        region = abs_d2[lo:hi + 1]
        if region.numel() == 0:
            return [], []

        max_strength = float(region.max().item())
        if max_strength <= 0.0:
            max_strength = 1e-12

        threshold = float(min_rel_strength) * max_strength
        forbidden = [] if forbidden is None else [int(v) for v in forbidden]

        candidate_scores: Dict[int, float] = {}

        def _try_add(idx: int, score: float) -> None:
            if idx < lo or idx > hi:
                return
            if any(abs(idx - f) <= int(min_spacing) for f in forbidden):
                return
            old = candidate_scores.get(int(idx), None)
            if old is None or score > old:
                candidate_scores[int(idx)] = float(score)

        # ---------------------------------------------------
        # A) local maxima of |d2|
        # ---------------------------------------------------
        center = abs_d2[lo:hi + 1]
        left = abs_d2[lo - 1:hi]
        right = abs_d2[lo + 1:hi + 2]
        local_max_mask = (center >= left) & (center >= right) & (center >= threshold)
        peak_rel = torch.nonzero(local_max_mask, as_tuple=False).squeeze(1)

        for rel in peak_rel.tolist():
            idx = lo + int(rel)
            score = float(abs_d2[idx].item())
            _try_add(idx, score)

        # fallback if too few pure curvature peaks
        if len(candidate_scores) < max(2, topk // 2):
            k = min(int(region.numel()), max(4, topk * 3))
            vals, rels = torch.topk(region, k=k, largest=True, sorted=True)
            for val_t, rel_t in zip(vals, rels):
                idx = lo + int(rel_t.item())
                score = float(val_t.item())
                if score >= threshold * 0.75:
                    _try_add(idx, score)

        # ---------------------------------------------------
        # B) zero-crossings of d2
        # ---------------------------------------------------
        if include_zero_crossings and hi > lo:
            a = d2[lo:hi]
            b = d2[lo + 1:hi + 1]
            crossed = (
                (a == 0.0) |
                (b == 0.0) |
                ((a < 0.0) & (b > 0.0)) |
                ((a > 0.0) & (b < 0.0))
            )

            if torch.any(crossed):
                base = torch.arange(lo, hi, device=d2.device)
                cand = torch.where(a.abs() <= b.abs(), base, base + 1)
                cand = cand[crossed]

                for idx_t in cand:
                    idx = int(idx_t.item())
                    left_i = max(lo, idx - 1)
                    right_i = min(hi, idx + 1)

                    curv_amp = float(torch.max(abs_d2[left_i:right_i + 1]).item())
                    slope_jump = float(torch.abs(d1[right_i] - d1[left_i]).item())
                    score = float(zero_cross_weight) * (curv_amp + 0.8 * slope_jump)

                    if score >= threshold * 0.60:
                        _try_add(idx, score)

        # ---------------------------------------------------
        # C) large slope-jump points from |Δd1|
        # ---------------------------------------------------
        if include_slope_jump and hi > lo:
            d1_jump = torch.abs(d1[1:] - d1[:-1])  # length n-1
            lo_j = lo
            hi_j = min(hi, int(d1_jump.numel()) - 1)
            if hi_j >= lo_j:
                jump_seg = d1_jump[lo_j:hi_j + 1]
                if jump_seg.numel() > 0:
                    jump_max = float(jump_seg.max().item())
                    if jump_max > 0.0:
                        jump_thr = max(threshold * 0.50, 0.10 * jump_max)

                        jc = jump_seg
                        jl = d1_jump[lo_j - 1:hi_j] if lo_j - 1 >= 0 else torch.full_like(jc, -float("inf"))
                        jr = d1_jump[lo_j + 1:hi_j + 2] if (hi_j + 1) < int(d1_jump.numel()) else torch.cat(
                            [d1_jump[lo_j + 1:hi_j + 1], torch.tensor([-float("inf")], device=d1.device, dtype=d1.dtype)]
                        )

                        if jr.numel() != jc.numel():
                            jr2 = torch.full_like(jc, -float("inf"))
                            rr = d1_jump[lo_j + 1:hi_j + 2]
                            jr2[:rr.numel()] = rr
                            jr = jr2

                        if jl.numel() != jc.numel():
                            jl2 = torch.full_like(jc, -float("inf"))
                            ll = d1_jump[max(0, lo_j - 1):hi_j]
                            jl2[-ll.numel():] = ll
                            jl = jl2

                        jump_local = (jc >= jl) & (jc >= jr) & (jc >= jump_thr)
                        jump_rel = torch.nonzero(jump_local, as_tuple=False).squeeze(1)

                        for rel in jump_rel.tolist():
                            idx = lo_j + int(rel)
                            score = float(slope_jump_weight) * (
                                float(d1_jump[idx].item()) + 0.5 * float(abs_d2[min(idx, hi)].item())
                            )
                            _try_add(idx, score)

        if len(candidate_scores) == 0:
            return [], []

        ordered = sorted(candidate_scores.items(), key=lambda t: t[1], reverse=True)

        selected_idx: List[int] = []
        selected_strength: List[float] = []

        for idx, score in ordered:
            if any(abs(idx - sidx) <= int(min_spacing) for sidx in selected_idx):
                continue
            selected_idx.append(int(idx))
            selected_strength.append(float(score))
            if len(selected_idx) >= int(topk):
                break

        return selected_idx, selected_strength

    if potential.ndim != 2 or current.ndim != 2:
        raise ValueError("potential and current must be (B,L) tensors")

    x = potential.to(device=run_device)
    y = current.to(device=run_device)

    device_obj = x.device
    dtype = x.dtype
    B, L = x.shape

    valid_n = _finite_prefix_length(x, y)

    weights = torch.ones((B, L), device=device_obj, dtype=dtype)
    key_mask = torch.zeros((B, L), device=device_obj, dtype=torch.bool)
    neighbor_mask = torch.zeros((B, L), device=device_obj, dtype=torch.bool)

    start_idx = torch.full((B,), -1, device=device_obj, dtype=torch.long)
    corr_idx = torch.full((B,), -1, device=device_obj, dtype=torch.long)

    passivation_onset_idx = torch.full((B,), -1, device=device_obj, dtype=torch.long)
    passivation_inflection_idx = torch.full((B,), -1, device=device_obj, dtype=torch.long)

    active_peak_idx = torch.full((B,), -1, device=device_obj, dtype=torch.long)
    passive_idx = torch.full((B,), -1, device=device_obj, dtype=torch.long)

    breakdown_idx = torch.full((B,), -1, device=device_obj, dtype=torch.long)
    breakdown_inflection_idx = torch.full((B,), -1, device=device_obj, dtype=torch.long)

    repassivation_idx = torch.full((B,), -1, device=device_obj, dtype=torch.long)
    reversal_idx_out = torch.full((B,), -1, device=device_obj, dtype=torch.long)
    end_idx = torch.full((B,), -1, device=device_obj, dtype=torch.long)

    critical_inflection_idx = torch.full(
        (B, critical_topk), -1, device=device_obj, dtype=torch.long
    )
    critical_inflection_strength = torch.zeros(
        (B, critical_topk), device=device_obj, dtype=dtype
    )

    behavior_code = torch.full(
        (B,), BEHAVIOR_NAME_TO_CODE["ambiguous"], device=device_obj, dtype=torch.long
    )
    behavior_name_list: List[str] = []
    steel_hint_list: List[str] = []

    ecorr = torch.full((B,), float("nan"), device=device_obj, dtype=dtype)
    current_at_corr_signed = torch.full((B,), float("nan"), device=device_obj, dtype=dtype)
    icorr = torch.full((B,), float("nan"), device=device_obj, dtype=dtype)

    e_pass_onset = torch.full((B,), float("nan"), device=device_obj, dtype=dtype)
    e_crit = torch.full((B,), float("nan"), device=device_obj, dtype=dtype)
    i_crit = torch.full((B,), float("nan"), device=device_obj, dtype=dtype)
    e_pass = torch.full((B,), float("nan"), device=device_obj, dtype=dtype)
    i_pass = torch.full((B,), float("nan"), device=device_obj, dtype=dtype)
    e_pit = torch.full((B,), float("nan"), device=device_obj, dtype=dtype)
    e_rp = torch.full((B,), float("nan"), device=device_obj, dtype=dtype)

    passivation_conf = torch.zeros((B,), device=device_obj, dtype=dtype)
    breakdown_conf = torch.zeros((B,), device=device_obj, dtype=dtype)

    # -------------------------------------------------------
    # Batched forward-branch preparation
    # -------------------------------------------------------
    xf_pad, yf_pad, map_f_pad, forward_n, reversal_idx = _prepare_forward_branches_batch(
        x,
        y,
        valid_n,
        reversal_buffer=reversal_buffer,
    )
    reversal_idx_out.copy_(reversal_idx)

    if xf_pad.shape[1] > 0:
        ys_pad = _moving_average_masked_batch(yf_pad, forward_n, smooth_win_current)
        zf_pad = torch.log10(torch.abs(yf_pad).clamp_min(log_eps))
        zfs_pad = _moving_average_masked_batch(zf_pad, forward_n, smooth_win_log)
        dzdEf_pad = _safe_gradient_batch(zfs_pad, xf_pad, forward_n)
        d2zdE2f_pad = _safe_gradient_batch(dzdEf_pad, xf_pad, forward_n)
    else:
        ys_pad = yf_pad.clone()
        zfs_pad = yf_pad.clone()
        dzdEf_pad = yf_pad.clone()
        d2zdE2f_pad = yf_pad.clone()

    # -------------------------------------------------------
    # Per-curve decision logic
    # -------------------------------------------------------
    for b in range(B):
        n = int(valid_n[b].item())
        if n < 3:
            behavior_name_list.append("ambiguous")
            steel_hint_list.append(STEEL_HINT_FROM_BEHAVIOR["ambiguous"])
            if n > 0:
                start_idx[b] = 0
                end_idx[b] = n - 1
            if n < L:
                weights[b, n:] = 0.0
            continue

        nf = int(forward_n[b].item())
        if nf < 3:
            behavior_name_list.append("ambiguous")
            steel_hint_list.append(STEEL_HINT_FROM_BEHAVIOR["ambiguous"])
            if n > 0:
                start_idx[b] = 0
                end_idx[b] = n - 1
            if n < L:
                weights[b, n:] = 0.0
            continue

        xb = x[b, :n]
        yb = y[b, :n]

        xf = xf_pad[b, :nf]
        yf = yf_pad[b, :nf]
        map_f = map_f_pad[b, :nf]

        ys = ys_pad[b, :nf]
        zfs = zfs_pad[b, :nf]
        dzdEf = dzdEf_pad[b, :nf]
        d2zdE2f = d2zdE2f_pad[b, :nf]

        weights_b = weights[b, :n]
        key_mask_b = key_mask[b, :n]
        neighbor_mask_b = neighbor_mask[b, :n]

        i_start = 0
        i_end = n - 1
        start_idx[b] = i_start
        end_idx[b] = i_end

        rev = int(reversal_idx[b].item())

        x_span = float((xf[-1] - xf[0]).abs().item()) if nf >= 2 else 1.0
        x_span = max(x_span, 1e-12)

        s0 = torch.sign(ys[:-1])
        s1 = torch.sign(ys[1:])
        cross_np = torch.where((ys[:-1] <= 0) & (ys[1:] >= 0))[0]
        cross_any = torch.where(s0 * s1 <= 0)[0]

        if cross_np.numel() > 0:
            j = int(cross_np[0].item())
            i_corr_ref = j if abs(float(ys[j].item())) <= abs(float(ys[j + 1].item())) else (j + 1)
        elif cross_any.numel() > 0:
            j = int(cross_any[0].item())
            i_corr_ref = j if abs(float(ys[j].item())) <= abs(float(ys[j + 1].item())) else (j + 1)
        else:
            i_corr_ref = int(torch.argmin(torch.abs(ys)).item())

        i_corr_local = int(torch.argmin(yf).item())
        i_corr_raw = int(map_f[i_corr_local].item())

        corr_idx[b] = i_corr_raw
        ecorr[b] = xf[i_corr_local]
        current_at_corr_signed[b] = yf[i_corr_local]
        icorr[b] = yf[i_corr_local]

        has_passivation = False
        has_breakdown = False
        has_repassivation_capability = False
        has_repassivation_evidence = False
        has_transpassive_rise = False

        passivation_score = 0.0
        breakdown_score_local = 0.0
        repassivation_score = 0.0
        transpassive_score = 0.0

        i_peak_local = -1
        i_pass_local = -1
        i_break_local = -1
        i_pass_infl_local = -1
        i_onset_local = -1
        i_break_infl_local = -1

        if i_corr_ref < nf - max(6, passive_min_separation + 2):
            lo = i_corr_ref + 1
            z_an = zfs[lo:]

            rel_s, rel_e, run_len = _run_length_near_min(z_an, tol=float(passive_plateau_tol))
            if rel_s >= 0:
                cand_pass_start = lo + rel_s
                cand_pass_end = lo + rel_e
                cand_pass = (cand_pass_start + cand_pass_end) // 2

                if (cand_pass - lo) >= int(passive_min_separation):
                    rel_peak = int(torch.argmax(zfs[lo:cand_pass + 1]).item())
                    cand_peak = lo + rel_peak

                    drop = float((zfs[cand_peak] - zfs[cand_pass]).item())
                    pre_rise = float((zfs[cand_peak] - zfs[lo]).item())

                    plateau_seg = zfs[cand_pass_start:cand_pass_end + 1]
                    plateau_std = float(torch.std(plateau_seg, unbiased=False).item()) if plateau_seg.numel() > 1 else 0.0
                    plateau_slope = float(torch.mean(torch.abs(dzdEf[cand_pass_start:cand_pass_end + 1])).item()) if plateau_seg.numel() > 0 else 0.0
                    separation = cand_pass - cand_peak

                    passivation_score += min(max(drop / max(passivation_drop_thresh, 1e-6), 0.0), 2.0)
                    if run_len >= int(passive_plateau_min_points):
                        passivation_score += 0.9
                    if plateau_std <= float(passive_plateau_tol):
                        passivation_score += 0.7
                    elif plateau_std <= float(passive_plateau_tol) * 1.5:
                        passivation_score += 0.35
                    if separation >= 2:
                        passivation_score += 0.5
                    if pre_rise > 0.05:
                        passivation_score += 0.3
                    if plateau_slope <= 0.15:
                        passivation_score += 0.4

                    if (drop >= float(passivation_drop_thresh)) and (passivation_score >= float(min_passivation_score)):
                        has_passivation = True
                        i_peak_local = cand_peak

                        i_pass_local = _find_plateau_center_idx(
                            zfs,
                            start=cand_pass_start,
                            end=cand_pass_end,
                            tol=float(passive_plateau_tol),
                            min_run=int(max(1, passive_plateau_min_points)),
                        )
                        if i_pass_local < 0:
                            i_pass_local = cand_pass

                        passivation_conf[b] = zfs[i_peak_local] - zfs[i_pass_local]

                        if i_peak_local > (i_corr_ref + 1):
                            i_pass_infl_local = _pick_inflection_idx_from_curvature_fast(
                                dzdEf,
                                d2zdE2f,
                                i_corr_ref + 1,
                                i_peak_local,
                                anchor=max(i_corr_ref + 1, (i_corr_ref + i_peak_local) // 2),
                                prefer="negative",
                            )
                            i_onset_local = i_pass_infl_local

                        if i_onset_local < 0:
                            i_onset_local = i_peak_local
                            i_pass_infl_local = i_peak_local

        if has_passivation and (i_pass_local >= 0) and (i_pass_local < nf - 2):
            bl = max(0, i_pass_local - 1)
            br = min(nf, i_pass_local + 2)
            baseline = float(torch.median(zfs[bl:br]).item())
            confirm = int(max(1, breakdown_confirm_points))

            first_break_candidate = -1
            for j in range(i_pass_local + 1, nf):
                rise = float((zfs[j] - baseline).item())
                if rise < float(breakdown_rise_thresh):
                    continue

                lo_chk = max(i_pass_local + 1, j - confirm + 1)
                if bool(torch.all(dzdEf[lo_chk:j + 1] > 0.0).item()):
                    first_break_candidate = j
                    break

            if first_break_candidate >= 0:
                i_break_infl_local = _pick_inflection_idx_from_curvature_fast(
                    dzdEf,
                    d2zdE2f,
                    max(i_pass_local + 1, first_break_candidate - max(2, smooth_win)),
                    min(nf - 2, first_break_candidate + max(2, smooth_win)),
                    anchor=first_break_candidate,
                    prefer="positive",
                )

                if i_break_infl_local >= 0:
                    if float((zfs[i_break_infl_local] - baseline).item()) >= 0.5 * float(breakdown_rise_thresh):
                        i_break_local = i_break_infl_local
                    else:
                        i_break_local = first_break_candidate
                else:
                    i_break_local = first_break_candidate

                rise = float((zfs[i_break_local] - baseline).item())
                late_frac = float((xf[i_break_local] - xf[0]).item()) / x_span
                tail_rise = float((zfs[-1] - baseline).item())

                score = 0.0
                score += min(rise / max(float(breakdown_rise_thresh), 1e-6), 2.0)
                if late_frac > 0.55:
                    score += 0.4
                if tail_rise > 0.0:
                    score += 0.3
                if i_break_local - i_pass_local >= 2:
                    score += 0.3

                breakdown_score_local = score
                if breakdown_score_local >= float(min_breakdown_score):
                    has_breakdown = True
                    breakdown_conf[b] = zfs[i_break_local] - baseline
                else:
                    i_break_local = -1
                    i_break_infl_local = -1

        if has_breakdown and (i_break_local >= 0):
            baseline_tp = float(zfs[i_pass_local].item()) if i_pass_local >= 0 else float(zfs[i_break_local].item())
            rise_end = float((zfs[-1] - baseline_tp).item())
            pos_frac = float((xf[i_break_local] - xf[0]).item()) / x_span

            if pos_frac >= 0.75:
                transpassive_score += 0.9
            if rise_end >= float(breakdown_rise_thresh):
                transpassive_score += 0.7
            if float(dzdEf[-1].item()) > 0.0:
                transpassive_score += 0.4

        if (rev >= 0) and has_breakdown and (i_pass_local >= 0):
            has_repassivation_capability = True

            xr = xb[rev:]
            yr = yb[rev:]
            if xr.numel() >= 3:
                zr = torch.log10(torch.abs(yr).clamp_min(log_eps))
                zrs = _moving_average_1d(zr, int(smooth_win_log))

                baseline_r = float(zfs[i_pass_local].item())
                thr_r = baseline_r + float(repassivation_tol)

                jr0 = int(max(1, reversal_buffer // 2))
                hit = torch.where(zrs[jr0:] <= thr_r)[0]
                if hit.numel() > 0:
                    jr = jr0 + int(hit[0].item())
                    rep_raw = rev + jr
                    repassivation_idx[b] = rep_raw
                    e_rp[b] = xb[rep_raw]

                    has_repassivation_evidence = True
                    repassivation_score += 1.2

                    z_min_rev = float(torch.min(zrs).item())
                    if z_min_rev <= baseline_r + 0.5 * float(repassivation_tol):
                        repassivation_score += 0.6

        if rev >= 0 and has_breakdown and has_repassivation_evidence and (repassivation_score >= float(min_repassivation_score)):
            behavior_name = "cyclic_pitting"
        elif has_passivation and has_breakdown and (transpassive_score >= float(min_transpassive_score)) and (not has_repassivation_evidence):
            behavior_name = "transpassive"
            has_transpassive_rise = True
        elif has_passivation and has_breakdown:
            behavior_name = "passive_with_breakdown"
        elif has_passivation:
            behavior_name = "active_passive"
        else:
            behavior_name = "active_dissolution"

        behavior_name_list.append(behavior_name)
        steel_hint_list.append(STEEL_HINT_FROM_BEHAVIOR[behavior_name])
        behavior_code[b] = int(BEHAVIOR_NAME_TO_CODE[behavior_name])

        if i_onset_local >= 0:
            passivation_onset_idx[b] = int(map_f[i_onset_local].item())
            e_pass_onset[b] = xf[i_onset_local]

        if i_pass_infl_local >= 0:
            passivation_inflection_idx[b] = int(map_f[i_pass_infl_local].item())

        if i_peak_local >= 0:
            active_peak_idx[b] = int(map_f[i_peak_local].item())
            e_crit[b] = xf[i_peak_local]
            i_crit[b] = torch.abs(yf[i_peak_local])

        if i_pass_local >= 0:
            passive_idx[b] = int(map_f[i_pass_local].item())
            e_pass[b] = xf[i_pass_local]
            i_pass[b] = torch.abs(yf[i_pass_local])

        if i_break_local >= 0:
            breakdown_idx[b] = int(map_f[i_break_local].item())
            e_pit[b] = xf[i_break_local]

        if i_break_infl_local >= 0:
            breakdown_inflection_idx[b] = int(map_f[i_break_infl_local].item())
        elif i_break_local >= 0:
            breakdown_inflection_idx[b] = int(map_f[i_break_local].item())

        # ---------------------------------------------------
        # richer critical inflection detection
        # ---------------------------------------------------
        crit_local_idx, crit_strength = _collect_critical_inflections_richer(
            dzdEf,
            d2zdE2f,
            1,
            nf - 2,
            topk=critical_topk,
            min_spacing=int(critical_inflection_min_spacing),
            min_rel_strength=float(critical_inflection_min_rel_strength),
            forbidden=[i_corr_local],
            include_zero_crossings=bool(critical_include_zero_crossings),
            include_slope_jump=bool(critical_include_slope_jump),
            zero_cross_weight=float(critical_zero_cross_weight),
            slope_jump_weight=float(critical_slope_jump_weight),
        )

        max_strength = max(crit_strength) if len(crit_strength) > 0 else 0.0
        for j, (iloc, sval) in enumerate(zip(crit_local_idx, crit_strength)):
            raw_idx = int(map_f[iloc].item())
            critical_inflection_idx[b, j] = raw_idx
            critical_inflection_strength[b, j] = float(sval)

            strength_norm = float(sval / max_strength) if max_strength > 0 else 0.0

            _apply_local_keypoint_updates_(
                weights_b,
                key_mask_b,
                neighbor_mask_b,
                raw_idx,
                n=n,
                neighbor_count=int(neighbor_count),
                alpha_neighbor=float(alpha_neighbor) * strength_norm,
                key_boost=float(alpha_key) * strength_norm,
            )

        key_ids = [i_start, i_corr_raw]

        p_on = int(passivation_onset_idx[b].item())
        p_inf = int(passivation_inflection_idx[b].item())
        a_pk = int(active_peak_idx[b].item())
        p_ps = int(passive_idx[b].item())
        b_pk = int(breakdown_idx[b].item())
        b_inf = int(breakdown_inflection_idx[b].item())
        r_ps = int(repassivation_idx[b].item())

        if p_on >= 0:
            key_ids.append(p_on)
        if p_inf >= 0:
            key_ids.append(p_inf)
        if a_pk >= 0:
            key_ids.append(a_pk)
        if p_ps >= 0:
            key_ids.append(p_ps)
        if b_pk >= 0:
            key_ids.append(b_pk)
        if b_inf >= 0:
            key_ids.append(b_inf)
        if r_ps >= 0:
            key_ids.append(r_ps)
        if include_end_point:
            key_ids.append(i_end)

        key_ids = sorted(set(kk for kk in key_ids if 0 <= kk < n))

        for kk in key_ids:
            _apply_local_keypoint_updates_(
                weights_b,
                key_mask_b,
                neighbor_mask_b,
                kk,
                n=n,
                neighbor_count=int(neighbor_count),
                alpha_neighbor=float(alpha_neighbor),
                key_boost=0.0,
            )

        weights_b[key_mask_b] += float(alpha_key)
        neighbor_mask_b[key_mask_b] = False

        if n < L:
            weights[b, n:] = 0.0
            key_mask[b, n:] = False
            neighbor_mask[b, n:] = False

    aux = {
        "valid_n": valid_n,
        "behavior_code": behavior_code,
        "behavior_name": behavior_name_list,
        "steel_hint": steel_hint_list,
        "reversal_idx": reversal_idx_out,

        "start_idx": start_idx,
        "corr_idx": corr_idx,

        "passivation_onset_idx": passivation_onset_idx,
        "passivation_inflection_idx": passivation_inflection_idx,

        "active_peak_idx": active_peak_idx,
        "passive_idx": passive_idx,

        "breakdown_idx": breakdown_idx,
        "breakdown_inflection_idx": breakdown_inflection_idx,

        "repassivation_idx": repassivation_idx,
        "end_idx": end_idx,

        "critical_inflection_idx": critical_inflection_idx,
        "critical_inflection_strength": critical_inflection_strength,

        "ecorr": ecorr,
        "current_at_corr_signed": current_at_corr_signed,
        "icorr": icorr,

        "e_pass_onset": e_pass_onset,
        "e_crit": e_crit,
        "i_crit": i_crit,
        "e_pass": e_pass,
        "i_pass": i_pass,
        "e_pit": e_pit,
        "e_rp": e_rp,

        "passivation_conf": passivation_conf,
        "breakdown_conf": breakdown_conf,

        "key_mask": key_mask,
        "neighbor_mask": neighbor_mask,
        "weights": weights,
    }
    return weights, aux


def plot_polarization_with_keypoints(
    potential_1d: np.ndarray,
    current_1d: np.ndarray,
    *,
    key_indices: Dict[str, int],
    neighbor_indices: Optional[np.ndarray] = None,
    extra_inflection_indices: Optional[np.ndarray] = None,
    extra_inflection_strengths: Optional[np.ndarray] = None,
    save_path: Union[str, Path] = "polarization_keypoints.png",
    title: Optional[str] = None,
    use_symlog: bool = True,
    show_labels: bool = False,
    neighbor_alpha: float = 0.30,
    neighbor_size: int = 18,
    keypoint_size_scale: float = 1.0,
    show_second_derivative: bool = True,
    deriv_smooth_win: int = 5,
    deriv_log_eps: float = 1e-20,
):
    """
    Plot polarization curve with named keypoints and ranked curvature-critical inflection points.

    extra_inflection_indices:
        indices of additional critical inflections (same indexing as x/y)

    extra_inflection_strengths:
        absolute |d²(log10|i|)/dE²| strengths corresponding to extra_inflection_indices
        used to scale marker size
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(potential_1d, dtype=np.float64).reshape(-1)
    y = np.asarray(current_1d, dtype=np.float64).reshape(-1)
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]
    n = len(x)

    if n == 0:
        raise ValueError("No finite data points available for plotting.")

    def _moving_average_np(arr: np.ndarray, win: int) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        if arr.size < 3 or win <= 1:
            return arr.copy()

        w = int(win)
        if w % 2 == 0:
            w += 1
        w = min(w, arr.size if arr.size % 2 == 1 else max(1, arr.size - 1))
        if w < 3:
            return arr.copy()

        pad = w // 2
        arr_pad = np.pad(arr, (pad, pad), mode="edge")
        kernel = np.ones(w, dtype=np.float64) / float(w)
        return np.convolve(arr_pad, kernel, mode="valid")

    def _safe_gradient_np(v: np.ndarray, xx: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        xx = np.asarray(xx, dtype=np.float64).reshape(-1)
        m = min(len(v), len(xx))
        v = v[:m]
        xx = xx[:m]

        if m <= 1:
            return np.zeros_like(v)

        g = np.zeros_like(v)
        dx = xx[1:] - xx[:-1]
        dx = np.where(np.abs(dx) < eps, eps, dx)
        s = (v[1:] - v[:-1]) / dx

        g[0] = s[0]
        g[-1] = s[-1]
        if m > 2:
            g[1:-1] = 0.5 * (s[:-1] + s[1:])
        return g

    # derivative diagnostics
    log_abs_y = np.log10(np.clip(np.abs(y), deriv_log_eps, None))
    log_abs_y_s = _moving_average_np(log_abs_y, deriv_smooth_win)
    d1 = _safe_gradient_np(log_abs_y_s, x)
    d2 = _safe_gradient_np(d1, x)

    if show_second_derivative:
        fig, (ax, ax2) = plt.subplots(
            2, 1,
            figsize=(7.8, 7.0),
            dpi=200,
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08}
        )
    else:
        fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=200)
        ax2 = None

    ax.set_facecolor("white")
    if ax2 is not None:
        ax2.set_facecolor("white")

    # top panel
    ax.plot(
        x, y,
        linewidth=2.2,
        color="#1f4e79",
        label="polarization curve",
        zorder=2,
    )

    if np.any(y < 0) and np.any(y > 0):
        ax.axhline(0.0, color="0.35", linewidth=0.9, linestyle="--", alpha=0.7, zorder=1)

    if neighbor_indices is not None and len(neighbor_indices) > 0:
        nb = np.asarray(neighbor_indices, dtype=int)
        nb = nb[(0 <= nb) & (nb < n)]
        if nb.size > 0:
            ax.scatter(
                x[nb], y[nb],
                s=neighbor_size,
                marker="o",
                color="r",
                alpha=float(neighbor_alpha),
                edgecolors="none",
                label="adjacent weighted",
                zorder=3,
            )

    key_style = {
        "start":                  ("s", int(70 * keypoint_size_scale),  "#2ca02c", "start"),
        "corr":                   ("X", int(95 * keypoint_size_scale),  "#ff7f0e", "Ecorr / icorr"),

        "passivation_onset":      ("<", int(88 * keypoint_size_scale),  "#bcbd22", "passivation onset"),
        "passivation_inflection": ("8", int(82 * keypoint_size_scale),  "#6b8e23", "passivation inflection"),

        "active_peak":            ("P", int(98 * keypoint_size_scale),  "#d62728", "active peak / icrit"),
        "passive":                ("D", int(88 * keypoint_size_scale),  "#9467bd", "passive / ipass"),

        "breakdown":              ("*", int(125 * keypoint_size_scale), "#8c564b", "breakdown / Epit"),
        "breakdown_inflection":   ("H", int(88 * keypoint_size_scale),  "#a0522d", "breakdown inflection"),

        "repassivation":          (">", int(95 * keypoint_size_scale),  "#e377c2", "repassivation / Erp"),
        "end":                    ("h", int(85 * keypoint_size_scale),  "#7f7f7f", "end"),
    }

    for k, idx in key_indices.items():
        if idx is None:
            continue
        try:
            idx = int(idx)
        except Exception:
            continue
        if not (0 <= idx < n):
            continue

        mk, ss, cc, lbl = key_style.get(k, ("o", int(80 * keypoint_size_scale), "#111111", str(k)))

        ax.scatter(
            [x[idx]], [y[idx]],
            s=ss,
            marker=mk,
            color=cc,
            edgecolors="white",
            linewidths=0.9,
            label=lbl,
            zorder=5,
        )

        if show_labels:
            ax.annotate(
                lbl,
                xy=(x[idx], y[idx]),
                xytext=(6, 8),
                textcoords="offset points",
                fontsize=9,
                color=cc,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=cc, alpha=0.9),
                zorder=6,
            )

    # extra ranked inflections
    if extra_inflection_indices is not None:
        ext_idx = np.asarray(extra_inflection_indices, dtype=int).reshape(-1)
        ext_idx = ext_idx[(0 <= ext_idx) & (ext_idx < n)]

        if extra_inflection_strengths is not None:
            ext_strength = np.asarray(extra_inflection_strengths, dtype=np.float64).reshape(-1)
            ext_strength = ext_strength[:len(extra_inflection_indices)]
            ext_strength = ext_strength[(0 <= np.asarray(extra_inflection_indices, dtype=int).reshape(-1)) &
                                        (np.asarray(extra_inflection_indices, dtype=int).reshape(-1) < n)]
        else:
            ext_strength = np.ones_like(ext_idx, dtype=np.float64)

        if len(ext_idx) > 0:
            max_s = float(np.max(ext_strength)) if np.max(ext_strength) > 0 else 1.0
            for rank, (idx, sval) in enumerate(zip(ext_idx, ext_strength), start=1):
                s_norm = float(sval / max_s) if max_s > 0 else 0.0
                size = int((55 + 85 * s_norm) * keypoint_size_scale)
                alpha = 0.55 + 0.40 * s_norm

                ax.scatter(
                    [x[idx]], [y[idx]],
                    s=size,
                    marker="o",
                    color="#000000",
                    edgecolors="white",
                    linewidths=0.8,
                    alpha=alpha,
                    label="critical inflection" if rank == 1 else None,
                    zorder=4,
                )

                if show_labels:
                    ax.annotate(
                        f"CI{rank}",
                        xy=(x[idx], y[idx]),
                        xytext=(5, -10),
                        textcoords="offset points",
                        fontsize=8,
                        color="#000000",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#000000", alpha=0.9),
                        zorder=6,
                    )

    finite_y = y[np.isfinite(y)]
    finite_abs = np.abs(finite_y)
    finite_abs = finite_abs[finite_abs > 0]

    if use_symlog and finite_abs.size > 0:
        if np.any(finite_y < 0) and np.any(finite_y > 0):
            linthresh = max(np.percentile(finite_abs, 10), 1e-10)
            ax.set_yscale("symlog", linthresh=linthresh)
        elif np.all(finite_y > 0):
            ax.set_yscale("log")
        elif np.all(finite_y < 0):
            linthresh = max(np.percentile(finite_abs, 10), 1e-10)
            ax.set_yscale("symlog", linthresh=linthresh)

    finite_x = x[np.isfinite(x)]
    if finite_x.size > 0:
        x_min = float(np.nanmin(finite_x))
        x_max = float(np.nanmax(finite_x))
        if x_max > x_min:
            x_pad = 0.03 * (x_max - x_min)
            ax.set_xlim(x_min - x_pad, x_max + x_pad)

    if finite_y.size > 0:
        if ax.get_yscale() == "log":
            pos_y = finite_y[finite_y > 0]
            if pos_y.size > 0:
                y_min = float(np.nanmin(pos_y))
                y_max = float(np.nanmax(pos_y))
                ax.set_ylim(max(y_min * 0.8, 1e-20), y_max * 1.2)
        else:
            y_min = float(np.nanmin(finite_y))
            y_max = float(np.nanmax(finite_y))
            if y_max <= y_min:
                pad = max(1.0, abs(y_max) * 0.1 + 1e-12)
            else:
                pad = 0.10 * (y_max - y_min)
            ax.set_ylim(y_min - pad, y_max + pad)

    ax.set_ylabel("Current density (signed, unnormalized)")
    ax.grid(True, alpha=0.22, linewidth=0.8)

    if title:
        ax.set_title(title)

    # bottom panel: second derivative
    if ax2 is not None:
        ax2.plot(
            x, d2,
            linewidth=1.8,
            color="#444444",
            label=r"$d^2(\log_{10}|i|)/dE^2$",
            zorder=2,
        )
        ax2.axhline(0.0, color="0.35", linewidth=0.9, linestyle="--", alpha=0.7, zorder=1)

        inflection_keys = [
            "passivation_inflection",
            "breakdown_inflection",
        ]

        for k in inflection_keys:
            idx = key_indices.get(k, None)
            if idx is None:
                continue
            try:
                idx = int(idx)
            except Exception:
                continue
            if not (0 <= idx < n):
                continue

            mk, ss, cc, lbl = key_style.get(k, ("o", int(80 * keypoint_size_scale), "#111111", str(k)))
            ax2.scatter(
                [x[idx]], [d2[idx]],
                s=ss,
                marker=mk,
                color=cc,
                edgecolors="white",
                linewidths=0.8,
                label=lbl,
                zorder=5,
            )

        if extra_inflection_indices is not None:
            ext_idx = np.asarray(extra_inflection_indices, dtype=int).reshape(-1)
            ext_idx = ext_idx[(0 <= ext_idx) & (ext_idx < n)]

            if extra_inflection_strengths is not None:
                ext_strength = np.asarray(extra_inflection_strengths, dtype=np.float64).reshape(-1)
                ext_strength = ext_strength[:len(extra_inflection_indices)]
                ext_strength = ext_strength[(0 <= np.asarray(extra_inflection_indices, dtype=int).reshape(-1)) &
                                            (np.asarray(extra_inflection_indices, dtype=int).reshape(-1) < n)]
            else:
                ext_strength = np.ones_like(ext_idx, dtype=np.float64)

            if len(ext_idx) > 0:
                max_s = float(np.max(ext_strength)) if np.max(ext_strength) > 0 else 1.0
                for rank, (idx, sval) in enumerate(zip(ext_idx, ext_strength), start=1):
                    s_norm = float(sval / max_s) if max_s > 0 else 0.0
                    size = int((50 + 80 * s_norm) * keypoint_size_scale)
                    alpha = 0.55 + 0.40 * s_norm

                    ax2.scatter(
                        [x[idx]], [d2[idx]],
                        s=size,
                        marker="o",
                        color="#000000",
                        edgecolors="white",
                        linewidths=0.8,
                        alpha=alpha,
                        label="critical inflection" if rank == 1 else None,
                        zorder=4,
                    )

                    if show_labels:
                        ax2.annotate(
                            f"CI{rank}",
                            xy=(x[idx], d2[idx]),
                            xytext=(5, -8),
                            textcoords="offset points",
                            fontsize=8,
                            color="#000000",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#000000", alpha=0.9),
                            zorder=6,
                        )

        finite_d2 = d2[np.isfinite(d2)]
        if finite_d2.size > 0:
            d2_min = float(np.nanmin(finite_d2))
            d2_max = float(np.nanmax(finite_d2))
            if d2_max <= d2_min:
                pad2 = max(1e-6, abs(d2_max) * 0.1 + 1e-12)
            else:
                pad2 = 0.12 * (d2_max - d2_min)
            ax2.set_ylim(d2_min - pad2, d2_max + pad2)

        ax2.set_xlabel("Potential / Voltage (unnormalized)")
        ax2.set_ylabel(r"$d^2(\log_{10}|i|)/dE^2$")
        ax2.grid(True, alpha=0.22, linewidth=0.8)

        handles2, labels2 = ax2.get_legend_handles_labels()
        uniq2 = {}
        for h, lab in zip(handles2, labels2):
            if lab not in uniq2:
                uniq2[lab] = h
        if len(uniq2) > 0:
            ax2.legend(
                list(uniq2.values()),
                list(uniq2.keys()),
                fontsize=8,
                frameon=True,
                framealpha=0.95,
                loc="upper right",
            )
    else:
        ax.set_xlabel("Potential / Voltage (unnormalized)")

    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, lab in zip(handles, labels):
        if lab not in uniq:
            uniq[lab] = h

    ax.legend(
        list(uniq.values()),
        list(uniq.keys()),
        fontsize=9,
        frameon=True,
        framealpha=0.95,
        loc="lower right",
    )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)



if __name__ == "__main__":
    xlsx_path: str = "./../../dataset/polarization_data/train_data/traindata_postprocess.xlsx"
    col_name: str = "uniform_point"
    row_indices = [i for i in range(1, 320)]
    row_to_plot = [i for i in range(1, 320, 10)]
    L = 256

    pol_df = pd.read_excel(xlsx_path)
    print(pol_df.columns)
    print(pol_df.head())

    ################  find max and mimum value of traindata   #########################
    voltage_min, voltage_max, current_min, current_max = recommend_polarization_normalization_ranges(
        pol_df,
        points_col="uniform_point",   # or "final_point"
        q_low=0.005,
        q_high=0.995,
        margin=1e-13,
        clamp_voltage_min_zero=False,
        clamp_current_min_zero=False,
    )

    print("VOLTAGE_MIN =", voltage_min)
    print("VOLTAGE_MAX =", voltage_max)
    print("CURRENT_MIN =", current_min)
    print("CURRENT_MAX =", current_max)
    ##########################################################################

    # Build batch tensors
    potential_t, current_t, valid_n = load_batch_from_final_point(
        pol_df,
        col=col_name,
        row_indices=row_indices,
        L=L,
    )

    start_time = time.time()
    # Detect keypoints + weights
    weights_t, aux = detect_keypoints_polarization_physical_batch_batched(
        potential_t,
        current_t,
        device="cpu",
        critical_topk=6,
        critical_inflection_min_rel_strength=0.08,
        critical_include_zero_crossings=True,
        critical_include_slope_jump=True,
    )
    print("[OK] weights shape:", tuple(weights_t.shape))
    print(f"[INFO] Keypoint detection completed in {time.time() - start_time:.2f} seconds.")

    out_dir = Path("./physical_constraints_results_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for ridx in row_to_plot:
        if ridx not in row_indices:
            raise ValueError(
                f"row_to_plot contains {ridx}, but it is not in row_indices={row_indices}."
            )

        x_np, y_np = parse_final_point_xy(pol_df.iloc[ridx][col_name])

        if len(x_np) == 0 or len(y_np) == 0:
            print(f"[WARN] row={ridx} has no valid parsed points. Skipped.")
            continue

        bi = row_indices.index(ridx)
        n = int(aux["valid_n"][bi].item())

        x_np = x_np[:n]
        y_np = y_np[:n]

        key_idx = {
            "start": int(aux["start_idx"][bi].item()),
            "corr": int(aux["corr_idx"][bi].item()),
            "passivation_onset": int(aux["passivation_onset_idx"][bi].item()),
            "passivation_inflection": int(aux["passivation_inflection_idx"][bi].item()),
            "active_peak": int(aux["active_peak_idx"][bi].item()),
            "passive": int(aux["passive_idx"][bi].item()),
            "breakdown": int(aux["breakdown_idx"][bi].item()),
            "breakdown_inflection": int(aux["breakdown_inflection_idx"][bi].item()),
            "repassivation": int(aux["repassivation_idx"][bi].item()),
            "end": int(aux["end_idx"][bi].item()),
        }
        key_idx = {k: v for k, v in key_idx.items() if 0 <= v < n}

        nb_idx = torch.where(aux["neighbor_mask"][bi, :n])[0].cpu().numpy().astype(int)

        crit_idx = aux["critical_inflection_idx"][bi].detach().cpu().numpy().astype(int)
        crit_strength = aux["critical_inflection_strength"][bi].detach().cpu().numpy()

        valid_crit = (crit_idx >= 0) & (crit_idx < n) & np.isfinite(crit_strength) & (crit_strength > 0)
        crit_idx = crit_idx[valid_crit]
        crit_strength = crit_strength[valid_crit]

        behavior_name = aux["behavior_name"][bi]
        steel_hint = aux["steel_hint"][bi]

        ecorr_val = float(aux["ecorr"][bi].item())
        icorr_val = float(aux["icorr"][bi].item())
        e_pass_onset_val = float(aux["e_pass_onset"][bi].item())
        e_crit_val = float(aux["e_crit"][bi].item())
        i_crit_val = float(aux["i_crit"][bi].item())
        e_pass_val = float(aux["e_pass"][bi].item())
        i_pass_val = float(aux["i_pass"][bi].item())
        e_pit_val = float(aux["e_pit"][bi].item())
        e_rp_val = float(aux["e_rp"][bi].item())

        title = (
            f"row={ridx} | class={behavior_name} | hint={steel_hint}\n"
            f"Ecorr={ecorr_val:.4g}, icorr={icorr_val:.4g}, "
            f"Epit={e_pit_val:.4g}, Erp={e_rp_val:.4g}, "
            f"N_critInf={len(crit_idx)}"
        )

        out_fig = out_dir / f"row_{ridx:04d}_polarization_keypoints.png"
        plot_polarization_with_keypoints(
            x_np,
            y_np,
            key_indices=key_idx,
            neighbor_indices=nb_idx,
            extra_inflection_indices=crit_idx,
            extra_inflection_strengths=crit_strength,
            save_path=out_fig,
            title=title,
            use_symlog=True,
            show_labels=False,
            show_second_derivative=True,
            deriv_smooth_win=5,
            deriv_log_eps=1e-20,
        )

        print(f"[OK] Saved plot: {out_fig}")
        print(
            f"[INFO] row={ridx} | class={behavior_name} | steel_hint={steel_hint} | "
            f"Ecorr={ecorr_val:.4g} | icorr={icorr_val:.4g} | "
            f"E_pass_onset={e_pass_onset_val:.4g} | "
            f"E_crit={e_crit_val:.4g} | i_crit={i_crit_val:.4g} | "
            f"E_pass={e_pass_val:.4g} | i_pass={i_pass_val:.4g} | "
            f"Epit={e_pit_val:.4g} | Erp={e_rp_val:.4g} | "
            f"critical_inflections={crit_idx.tolist()}"
        )

        row_summary = {
            "row_index": ridx,
            "behavior_name": behavior_name,
            "steel_hint": steel_hint,
            "Ecorr": ecorr_val,
            "icorr": icorr_val,
            "E_pass_onset": e_pass_onset_val,
            "E_crit": e_crit_val,
            "i_crit": i_crit_val,
            "E_pass": e_pass_val,
            "i_pass": i_pass_val,
            "Epit": e_pit_val,
            "Erp": e_rp_val,
            "start_idx": key_idx.get("start", -1),
            "corr_idx": key_idx.get("corr", -1),
            "passivation_onset_idx": key_idx.get("passivation_onset", -1),
            "passivation_inflection_idx": key_idx.get("passivation_inflection", -1),
            "active_peak_idx": key_idx.get("active_peak", -1),
            "passive_idx": key_idx.get("passive", -1),
            "breakdown_idx": key_idx.get("breakdown", -1),
            "breakdown_inflection_idx": key_idx.get("breakdown_inflection", -1),
            "repassivation_idx": key_idx.get("repassivation", -1),
            "end_idx": key_idx.get("end", -1),
        }

        # save top-K ranked critical inflections into separate columns
        max_k = aux["critical_inflection_idx"].shape[1]
        for k in range(max_k):
            row_summary[f"critical_inflection_{k+1}_idx"] = int(aux["critical_inflection_idx"][bi, k].item())
            row_summary[f"critical_inflection_{k+1}_strength"] = float(aux["critical_inflection_strength"][bi, k].item())

        summary_rows.append(row_summary)

    if len(summary_rows) > 0:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = out_dir / "polarization_keypoint_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[OK] Saved summary CSV: {summary_path}")