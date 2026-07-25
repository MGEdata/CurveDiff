# =========================
# Standard library
# =========================
import math
import time
import traceback
import warnings
from copy import deepcopy
from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict, Optional, Tuple, Union

# =========================
# PyTorch
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

# =========================
# Scientific computing
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

# =========================
# ML metrics
# =========================
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# Distance / similarity
# =========================
from fastdtw import fastdtw
from dtaidistance import dtw
# from hausdorff import hausdorff_distance
# import similaritymeasures  # Fréchet distance

# =========================
# Utilities
# =========================
from tqdm.auto import tqdm  # auto-detect environment

# =========================
# Warnings
# =========================
warnings.filterwarnings("ignore")


def combine_and_evaluate_curves_1d(
    true_curves,
    generated_curves,
    *,
    x_coords=None,              # None | (L,) | (B,L) | (B,1,L) | (B,L,1)
    ignore_prefix_k: int = 0,   # ignore first k points in metrics
    dtw_space: str = "2d",      # "1d" or "2d"
    xy_scale=None,              # None or (sx, sy) for 2D distances
    r2_sst_eps: float = 1e-6,  # skip curves with SST below this in R2 aggregation
):
    """
    Evaluate stress–strain curves y(x).

    R2 aggregation (recommended):
        - mean_r2 is computed as variance-weighted/global R2:
            1 - sum(SSE_i) / sum(SST_i)   over curves with SST_i > r2_sst_eps
        - also returns median_r2, IQR, and negative-rate diagnostics.

    Other metrics:
        - mse/mae: stress-only (1D)
        - dtw: 1D or 2D (controlled by dtw_space)
        - frechet: always 2D in (x,y)
    """
    import numpy as np
    import warnings
    from sklearn.metrics import r2_score
    from fastdtw import fastdtw
    import similaritymeasures

    def _to_numpy(a):
        if hasattr(a, "detach") and hasattr(a, "cpu"):
            a = a.detach().cpu().numpy()
        return np.asarray(a)

    def _to_2d_curves(arr, name: str) -> np.ndarray:
        arr = _to_numpy(arr)
        if arr.ndim == 3:
            if arr.shape[1] == 1:      # (B,1,L)
                arr = arr[:, 0, :]
            elif arr.shape[2] == 1:    # (B,L,1)
                arr = arr[:, :, 0]
            else:
                raise ValueError(f"{name} must be (B,L) or (B,1,L) or (B,L,1), got {arr.shape}")
        elif arr.ndim == 2:
            pass
        elif arr.ndim == 1:
            arr = arr[None, :]
        else:
            raise ValueError(f"{name} must be 1D/2D/3D, got {arr.shape}")
        return arr.astype(np.float64, copy=False)

    def _to_x_grids(x_coords, B: int, L: int):
        if x_coords is None:
            return np.linspace(0.0, 1.0, L, dtype=np.float64), None

        x = _to_numpy(x_coords).astype(np.float64)

        if x.ndim == 3:
            if x.shape[1] == 1:
                x = x[:, 0, :]
            elif x.shape[2] == 1:
                x = x[:, :, 0]

        if x.ndim == 1:
            if x.shape[0] != L:
                raise ValueError(f"x_coords must have length L={L}, got {x.shape}")
            return x, None

        if x.ndim == 2:
            if x.shape != (B, L):
                raise ValueError(f"x_coords must be (B,L)=({B},{L}), got {x.shape}")
            return None, x

        raise ValueError(f"x_coords must be None, (L,) or (B,L), got {x.shape}")

    def _slice_x(x_global, x_per_curve, k: int, L_eval: int):
        if x_global is not None:
            xg = x_global[k:]
            if xg.shape[0] != L_eval:
                raise ValueError(f"x_global mismatch after slicing: expected {L_eval}, got {xg.shape[0]}")
            return xg, None
        if x_per_curve is not None:
            xp = x_per_curve[:, k:]
            if xp.shape[1] != L_eval:
                raise ValueError(f"x_per_curve mismatch after slicing: expected (B,{L_eval}), got {xp.shape}")
            return None, xp
        return None, None

    def _get_xi(i: int, x_global, x_per_curve, L_eval: int):
        if x_per_curve is not None:
            xi = x_per_curve[i]
        elif x_global is not None:
            xi = x_global
        else:
            xi = np.linspace(0.0, 1.0, L_eval, dtype=np.float64)

        if xi is None or xi.shape[0] != L_eval or (not np.isfinite(xi).all()):
            xi = np.linspace(0.0, 1.0, L_eval, dtype=np.float64)
        return xi.astype(np.float64, copy=False)

    def _xy_points(xi, yi):
        yi = np.asarray(yi, dtype=np.float64)
        xi = np.asarray(xi, dtype=np.float64)

        if xy_scale is None:
            return np.column_stack([xi, yi])

        sx, sy = xy_scale
        sx = float(sx) if sx not in (None, 0) else 1.0
        sy = float(sy) if sy not in (None, 0) else 1.0
        return np.column_stack([xi / sx, yi / sy])

    def _euclid(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        d = a - b
        return float(np.sqrt(np.dot(d, d)))

    # --- curves ---
    y_true = _to_2d_curves(true_curves, "true_curves")
    y_pred = _to_2d_curves(generated_curves, "generated_curves")

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Curves must match after squeezing. Got {y_true.shape} and {y_pred.shape}")

    B, L = y_true.shape

    # --- ignore prefix ---
    k = int(max(0, ignore_prefix_k))
    if k >= L:
        raise ValueError(f"ignore_prefix_k={k} >= L={L}")

    y_true_eval = y_true[:, k:]
    y_pred_eval = y_pred[:, k:]
    L_eval = L - k

    # --- x grids ---
    x_global, x_per_curve = _to_x_grids(x_coords, B=B, L=L)
    x_global, x_per_curve = _slice_x(x_global, x_per_curve, k=k, L_eval=L_eval)

    dtw_space = str(dtw_space).lower().strip()
    if dtw_space not in ("1d", "2d"):
        raise ValueError(f"dtw_space must be '1d' or '2d', got {dtw_space!r}")

    # --- per-curve metrics ---
    all_r2, all_dtw, all_frechet = [], [], []
    all_mse, all_mae = [], []

    # for weighted/global R2
    sse_sum = 0.0
    sst_sum = 0.0
    n_r2_valid = 0

    for i in range(B):
        t = y_true_eval[i]
        g = y_pred_eval[i]

        if (not np.isfinite(t).all()) or (not np.isfinite(g).all()):
            all_r2.append(np.nan); all_dtw.append(np.nan); all_frechet.append(np.nan)
            all_mse.append(np.nan); all_mae.append(np.nan)
            continue

        diff = t - g
        mse_i = float(np.mean(diff**2))
        mae_i = float(np.mean(np.abs(diff)))
        all_mse.append(mse_i)
        all_mae.append(mae_i)

        # --- per-curve R2 (still useful for distribution stats) ---
        try:
            r2_i = float(r2_score(t, g))
        except Exception:
            r2_i = np.nan
        all_r2.append(r2_i)

        # --- weighted/global R2 pieces (use per-curve mean) ---
        t_mean = float(np.mean(t))
        sse_i = float(np.sum((t - g) ** 2))
        sst_i = float(np.sum((t - t_mean) ** 2))
        if np.isfinite(sse_i) and np.isfinite(sst_i) and (sst_i > float(r2_sst_eps)):
            sse_sum += sse_i
            sst_sum += sst_i
            n_r2_valid += 1

        # x for this curve
        xi = _get_xi(i, x_global, x_per_curve, L_eval=L_eval)

        # --- DTW ---
        if dtw_space == "1d":
            try:
                dtw_dist, _ = fastdtw(t, g, dist=lambda a, b: abs(float(a) - float(b)))
                all_dtw.append(float(dtw_dist))
            except Exception:
                all_dtw.append(np.nan)
        else:
            try:
                pts_t = _xy_points(xi, t)
                pts_g = _xy_points(xi, g)
                dtw_dist, _ = fastdtw(pts_t, pts_g, dist=_euclid)
                all_dtw.append(float(dtw_dist))
            except Exception:
                all_dtw.append(np.nan)

        # --- Fréchet (2D) ---
        try:
            pts_t = _xy_points(xi, t)
            pts_g = _xy_points(xi, g)
            all_frechet.append(float(similaritymeasures.frechet_dist(pts_t, pts_g)))
        except Exception:
            all_frechet.append(np.nan)

    # --- aggregates (nan-safe) ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # recommended "mean" R2:
        mean_r2_weighted = float("nan") if sst_sum <= 0 else float(1.0 - (sse_sum / sst_sum))

        # robust R2 distribution summaries
        r2_arr = np.asarray(all_r2, dtype=np.float64)
        r2_arr = r2_arr[np.isfinite(r2_arr)]
        median_r2 = float(np.nan) if r2_arr.size == 0 else float(np.median(r2_arr))
        if r2_arr.size == 0:
            r2_q25, r2_q75 = float("nan"), float("nan")
            r2_negative_frac = float("nan")
        else:
            r2_q25, r2_q75 = [float(x) for x in np.percentile(r2_arr, [25, 75])]
            r2_negative_frac = float(np.mean(r2_arr < 0.0))

        mean_dtw = float(np.nanmean(all_dtw))
        mean_frechet = float(np.nanmean(all_frechet))
        mean_mse = float(np.nanmean(all_mse))
        mean_mae = float(np.nanmean(all_mae))

    # --- overall (flatten stress only) ---
    t_flat = y_true_eval.reshape(-1)
    g_flat = y_pred_eval.reshape(-1)
    diff_flat = t_flat - g_flat
    overall_mse = float(np.mean(diff_flat**2))
    overall_mae = float(np.mean(np.abs(diff_flat)))
    try:
        overall_r2 = float(r2_score(t_flat, g_flat))
    except Exception:
        overall_r2 = np.nan

    def safe_round(v, d):
        if v is None:
            return float("nan")
        if isinstance(v, (np.floating, np.integer)):
            v = float(v)
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return float("nan")
        return round(v, d)

    metrics = {
        # overall (stress-only; uses global mean over all points)
        "mse": safe_round(overall_mse, 6),
        "mae": safe_round(overall_mae, 6),
        "r2": safe_round(overall_r2, 8),

        # recommended dataset-level "mean R2" for curves:
        "mean_r2": safe_round(mean_r2_weighted, 8),  # <- weighted/global per-curve-mean R2
        "n_r2_valid": int(n_r2_valid),

        # robust R2 stats (strongly recommended to report together)
        "median_r2": safe_round(median_r2, 8),
        "r2_q25": safe_round(r2_q25, 8),
        "r2_q75": safe_round(r2_q75, 8),
        "r2_negative_frac": safe_round(r2_negative_frac, 6),

        # shape metrics
        "dtw": safe_round(mean_dtw if np.isfinite(mean_dtw) else float("inf"), 6),
        "frechet": safe_round(mean_frechet if np.isfinite(mean_frechet) else float("inf"), 6),

        # extra error summaries
        "mean_mse": safe_round(mean_mse, 6),
        "mean_mae": safe_round(mean_mae, 6),

        # bookkeeping
        "dtw_space": dtw_space,
        "xy_scale": xy_scale,
        "r2_sst_eps": float(r2_sst_eps),
    }

    return metrics



# --- Example Usage (remains the same) ---
if __name__ == "__main__":
    n_curves = 900
    n_points = 256

    # --- 2D Example ---
    print("--- Evaluating 2D Curves ---")
    t = np.linspace(0, 2 * np.pi, n_points)
    # --- 1D Example ---
    print("\n--- Evaluating 1D Curves ---")
    true_curves_1d = np.zeros((n_curves, n_points))
    generated_curves_1d = np.zeros((n_curves, n_points))
    for i in range(n_curves):
         amp, freq = np.random.rand() * 10 + 1, np.random.rand() + 0.1
         phase = np.random.rand() * np.pi
         true_curves_1d[i, :] = amp * np.exp(-t * freq) * np.sin(5 * freq * t + phase)
         generated_curves_1d[i, :] = true_curves_1d[i, :] + np.random.randn(n_points) * 0.5 + (np.random.rand()-0.5)*2

    print(f"True Curves Shape: {true_curves_1d.shape}")
    print(f"Generated Curves Shape: {generated_curves_1d.shape}")
    time_start = time.time()
    metrics_1d = combine_and_evaluate_curves_1d(true_curves_1d, generated_curves_1d)
    time_end = time.time()
    print("1D Metrics:", metrics_1d)
    print(f"Time taken: {time_end - time_start:.3f} seconds")