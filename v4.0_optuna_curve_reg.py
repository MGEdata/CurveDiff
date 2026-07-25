############################ new features ##########################################
# - add experiment data validation
# - add bash script for running the code
# --- Built-in Modules ---
import argparse
import ast
import csv
import gc
import math
import os
import random
import re
import shutil
import statistics
import time
import warnings
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

# --- Scientific Computing and Data Processing ---
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from optuna.trial import TrialState
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

# --- Project-Specific Modules ---
from curve_mertics import combine_and_evaluate_curves_1d
from CurveUNetConditional_v6 import CurveUNetConditional_v6
from DiffusionModel_v6 import DiffusionModel_v6
from curve_mertics import EMA
from vc_critical_point_detection import detect_keypoints_polarization_physical_batch_batched


torch.set_printoptions(threshold=float("inf"))
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
torch.backends.cudnn.enabled = False


def clear_gpu_memory():
    """
    Clears the GPU memory cache in PyTorch and runs garbage collection.

    This is useful for freeing up memory that is no longer referenced but may still
    be cached by PyTorch, especially in environments like Jupyter notebooks.
    """
    try:
        print("Clearing GPU memory...")
        # Run Python's garbage collector
        gc.collect()

        # Empty the PyTorch CUDA cache
        torch.cuda.empty_cache()

        print("GPU memory cleared successfully. ✅")

    except Exception as e:
        print(f"An error occurred while clearing GPU memory: {e}")


def delete_files(path):
    """
    Deletes the entire folder and its contents if it's a folder,
    or deletes the file if it's a file, if the path exists.

    Args:
        path (str): Path to the file or folder to be deleted.

    Returns:
        None
    """
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Folder '{path}' has been deleted.")
        elif os.path.isfile(path):
            os.remove(path)
            print(f"File '{path}' has been deleted.")
    else:
        print(f"Path '{path}' does not exist.")


def print_model_parameters(model: nn.Module):
    """
    Calculates and prints the total number of parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to analyze.
    """
    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Format the number for readability (e.g., 1.2 M, 256.5 K)
    if total_params > 1_000_000:
        formatted_params = f"{total_params / 1_000_000:.2f} M"
    elif total_params > 1_000:
        formatted_params = f"{total_params / 1_000:.2f} K"
    else:
        formatted_params = f"{total_params}"

    print(f"✅ Model Parameters: {formatted_params} (Total: {total_params:,})")


def calculate_sigma_data(train_loader):
    """
    Calculates the standard deviation of the training data.
    """
    print("Calculating sigma_data...")

    num_elements = 0
    mean_sum = 0.0

    ### REVISED LOOP ###
    # Unpack all four items from the loader and assign the last one to 'targets'.
    for _, _, _, targets in train_loader:
        targets = targets.float()
        mean_sum += targets.sum()
        num_elements += targets.numel()

    if num_elements == 0:
        print("Warning: DataLoader is empty. Returning sigma_data of 1.0")
        return 1.0

    mean = mean_sum / num_elements

    var_sum = 0.0
    ### REVISED LOOP ###
    for _, _, _, targets in train_loader:
        targets = targets.float()
        var_sum += ((targets - mean) ** 2).sum()

    variance = var_sum / num_elements
    sigma_data = torch.sqrt(variance)

    print(f"Calculation complete. sigma_data = {sigma_data.item():.4f}")
    return sigma_data.item()


# Placeholder for user-defined boolean type for argparse
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Creates a learning rate schedule with a linear warmup followed by a cosine decay.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def normalize_voltage(
    voltage: torch.Tensor,
    v_min: float = -1.5,
    v_max: float = 1.5,
    eps: float = 1e-30
) -> torch.Tensor:
    """
    Linearly normalize voltage to [-1, 1].

    Args:
        voltage: Voltage tensor.
        v_min: Minimum voltage in dataset.
        v_max: Maximum voltage in dataset.
        eps: Small value to avoid division by zero.

    Returns:
        Normalized voltage tensor in [-1, 1].
    """
    x = torch.as_tensor(voltage, dtype=torch.float32)

    x_min = torch.tensor(v_min, dtype=x.dtype, device=x.device)
    x_max = torch.tensor(v_max, dtype=x.dtype, device=x.device)

    denom = (x_max - x_min).clamp_min(eps)
    x01 = (x - x_min) / denom
    x01 = x01.clamp(0.0, 1.0)
    print(f"\n x01[0, 0:20]: {x01[0, 0:20]}")
    x_norm = x01 * 2.0 - 1.0
    return x_norm


def denormalize_voltage(
    voltage_norm: torch.Tensor,
    v_min: float = -1.5,
    v_max: float = 1.5
) -> torch.Tensor:
    """
    Inverse of normalize_voltage:
        voltage_norm in [-1, 1] -> voltage in [v_min, v_max]
    """
    x_norm = torch.as_tensor(voltage_norm, dtype=torch.float32)

    x_min = torch.tensor(v_min, dtype=x_norm.dtype, device=x_norm.device)
    x_max = torch.tensor(v_max, dtype=x_norm.dtype, device=x_norm.device)

    x01 = (x_norm + 1.0) / 2.0
    x = x01 * (x_max - x_min) + x_min
    return x


def normalize_current(
    current: torch.Tensor,
    lc_min: float = 6.438105273675044e-11,
    lc_max: float = 0.4914291525600493,
    eps: float = 1e-30
) -> torch.Tensor:
    """
    Normalize current/current density to [-1, 1] using -log10 scale internally.

    Pipeline:
        current -> -log10(current) -> linear map to [-1, 1]

    Args:
        current: Positive current/current-density tensor.
                If your data are polarization magnitudes, use |i| or current density magnitude.
        lc_min: Minimum current/current-density value in original scale.
        lc_max: Maximum current/current-density value in original scale.
        eps: Small value to avoid log10(0).

    Returns:
        Normalized tensor in [-1, 1].
    """
    y = torch.as_tensor(current, dtype=torch.float32).clamp_min(eps)
    neg_log_y = -torch.log10(y)

    lc_min_t = torch.tensor(lc_min, dtype=neg_log_y.dtype, device=neg_log_y.device).clamp_min(eps)
    lc_max_t = torch.tensor(lc_max, dtype=neg_log_y.dtype, device=neg_log_y.device).clamp_min(eps)

    # Convert original current bounds to -log10(current) bounds
    y_min = -torch.log10(lc_max_t)  # transformed minimum
    y_max = -torch.log10(lc_min_t)  # transformed maximum

    denom = (y_max - y_min).clamp_min(1e-13)
    y01 = (neg_log_y - y_min) / denom
    y01 = y01.clamp(0.0, 1.0)
    y_norm = y01 * 2.0 - 1.0
    return y_norm


def denormalize_current(
    current_norm: torch.Tensor,
    lc_min: float = 6.438105273675044e-11,
    lc_max: float = 0.4914291525600493,
    eps: float = 1e-30
) -> torch.Tensor:
    """
    Inverse of normalize_current:
        normalized [-1, 1] -> -log10(current) -> current

    Args:
        current_norm: Normalized tensor in [-1, 1].
        lc_min: Minimum current/current-density value in original scale.
        lc_max: Maximum current/current-density value in original scale.
        eps: Small value to avoid log10(0).

    Returns:
        Current/current density in original scale.
    """
    y_norm = torch.as_tensor(current_norm, dtype=torch.float32)

    lc_min_t = torch.tensor(lc_min, dtype=y_norm.dtype, device=y_norm.device).clamp_min(eps)
    lc_max_t = torch.tensor(lc_max, dtype=y_norm.dtype, device=y_norm.device).clamp_min(eps)

    # Convert original current bounds to -log10(current) bounds
    y_min = -torch.log10(lc_max_t)  # transformed minimum
    y_max = -torch.log10(lc_min_t)  # transformed maximum

    y01 = (y_norm + 1.0) / 2.0
    neg_log_y = y01 * (y_max - y_min) + y_min
    y = torch.pow(10.0, -neg_log_y)
    return y


def load_and_prepare_data(data_path, prefix, save_path, device):
    """
    Load, preprocess, normalize, and concatenate condition embeddings.

    Args:
        data_path (str): Directory containing .pt files.
        prefix (str): File prefix, e.g. '', 'exp_', etc.
        save_path (str): Path to save concatenated input embedding CSV.
        device: Torch device.

    Returns:
        current_df (pd.DataFrame): Normalized current target DataFrame.
        voltage_norm (torch.Tensor): Normalized voltage tensor.
        input_embed (torch.Tensor): Concatenated condition embedding tensor.
    """
    # -------- Load tensors --------
    mat_embed = torch.load(os.path.join(data_path, f"{prefix}text_embed.pt"), map_location="cpu")
    ele_embed = torch.load(os.path.join(data_path, f"{prefix}ele_embed.pt"), map_location="cpu")
    voltage = torch.load(os.path.join(data_path, f"{prefix}voltage_embed.pt"), map_location="cpu")
    current = torch.load(os.path.join(data_path, f"{prefix}current_embed.pt"), map_location="cpu")

    print("Min:", voltage.min().item())
    print("Max:", voltage.max().item())

    # -------- Normalize --------
    voltage_norm = normalize_voltage(
        voltage,
        v_min=VOLTAGE_MIN,
        v_max=VOLTAGE_MAX
    )

    current_norm = normalize_current(
        current,
        lc_min=CURRENT_MIN,
        lc_max=CURRENT_MAX
    )

    # -------- Check sample count consistency --------
    n_samples = mat_embed.size(0)
    for name, tensor in {
        "ele_embed": ele_embed,
        "voltage_norm": voltage_norm,
        "current_norm": current_norm,
    }.items():
        if tensor.size(0) != n_samples:
            raise ValueError(
                f"Sample count mismatch: mat_embed has {n_samples}, "
                f"but {name} has {tensor.size(0)}"
            )

    # -------- Build model input --------
    input_embed = torch.cat([mat_embed, ele_embed, voltage_norm, current_norm], dim=1).to(device)

    # -------- Save target/current CSV --------
    current_raw_csv = os.path.join(data_path, f"{prefix}current_target_raw.csv")
    current_norm_csv = os.path.join(data_path, f"{prefix}current_target_norm.csv")

    pd.DataFrame(current.detach().cpu().numpy()).to_csv(current_raw_csv, index=False)
    pd.DataFrame(current_norm.detach().cpu().numpy()).to_csv(current_norm_csv, index=False)

    # -------- Save concatenated input embedding CSV --------
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    pd.DataFrame(input_embed.detach().cpu().numpy()).to_csv(save_path, index=False)
    print(f"Input embedding saved to: {save_path}")
    print(f"Raw current target saved to: {current_raw_csv}")
    print(f"Normalized current target saved to: {current_norm_csv}")

    # -------- Create return DataFrame --------
    current_df = pd.DataFrame(current_norm.detach().cpu().numpy())

    # -------- Logging --------
    print(f"{prefix}mat_embed.shape: {mat_embed.shape}")
    print(f"{prefix}ele_embed.shape: {ele_embed.shape}")
    print(f"{prefix}voltage_norm.shape: {voltage_norm.shape}")
    print(f"{prefix}current_norm.shape: {current_norm.shape}")
    print(f"{prefix}input_embed.shape: {input_embed.shape}")

    return current, voltage, input_embed


def split_train_val(df, train_size=0.8, random_state=123):
    """
    Splits a Pandas DataFrame into training and validation datasets based on the 'id' column,
    shuffling unique IDs 100 times before splitting.

    Args:
        df (pd.DataFrame): The input DataFrame with 'index', 'id', and 'values' columns.
        train_size (float): The proportion of the dataset to include in the training split (default: 0.8).
        random_state (int): Random seed for reproducibility (default: None).

    Returns:
        tuple: Two lists containing the indices for training and validation sets.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Extract unique IDs
    unique_ids = [i for i in range(df.shape[0])]

    # Shuffle unique IDs 100 times
    for _ in range(100):
        np.random.shuffle(unique_ids)

    # Split shuffled unique IDs into training and validation sets
    train_ids, val_ids = train_test_split(
        unique_ids, train_size=train_size, random_state=random_state
    )

    return train_ids, val_ids


class MultimodalDataset(Dataset):
    def __init__(self, input_embed):
        """
        Assumes input_embed columns are:
        [mat_embed (2304), ele_embed (768), voltage (256), target (256)]
        """
        mat_end = 2304
        ele_end = mat_end + 768
        voltage_end = ele_end + 256
        target_end = voltage_end + 256

        self.mat_data = input_embed[:, :mat_end]
        self.ele_data = input_embed[:, mat_end:ele_end]
        self.voltage_data = input_embed[:, ele_end:voltage_end]
        self.targets = input_embed[:, voltage_end:target_end]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            self.mat_data[idx],
            self.ele_data[idx],
            self.voltage_data[idx],
            self.targets[idx]
        )


def train_and_evaluate(
    ema_handler,
    curve_model,
    diffusion,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    scheduler,
    num_epochs,
    run_id,
    dt_string,
    cond_drop_prob=0.1,
    *,
    use_amp: bool = False,
    grad_clip_norm: float = 0.0,
    scheduler_step_per_batch: bool = True,

    # performance
    compile_model: bool = False,
    compile_kwargs={"mode": "reduce-overhead", "fullgraph": False},

    # physical constraints
    use_physical_constraints: bool = True,
    physical_loss_weight: float = 0.10,
    phys_warmup_epochs: int = 10,
    phys_ramp_epochs: int = 30,
    phys_kp_beta_norm: float = 0.05,
    phys_sigma_gate: float = 0.25,
    phys_ignore_prefix_k: int = 0,
    phys_use_point_weights: bool = True,
    phys_use_kp_loss: bool = True,

    # polarization-curve keypoint settings
    kp_alpha_key: float = 5.0,
    kp_alpha_neighbor_ratio: float = 0.4,
    kp_neighbor_count: int = 3,
    pol_smooth_win: int = 5,
    pol_passivation_drop_thresh: float = 0.30,
    pol_breakdown_rise_thresh: float = 0.50,
    pol_critical_inflection_min_rel_strength: float = 0.08,
    pol_critical_topk: int = 10,
    pol_critical_inflection_min_spacing: int = 5,

    # deterministic validation / test
    deterministic_eval: bool = True,
    eval_seed_val: int = 123,
    eval_seed_test: int = 42,
) -> Tuple[float, float, float]:
    """
    Train and evaluate an EDM/VE diffusion model for 1D polarization-curve generation.

    Main features
    -------------
    1. EDM training with log-normal sigma sampling.
    2. CFG-style condition dropout during training.
       IMPORTANT: only the non-voltage branches are dropped; voltage is always kept,
       because voltage is the coordinate scaffold of the target curve.
    3. Optional physics-aware supervision from polarization-curve keypoints.
    4. Deterministic validation/test by fixing RNG state during evaluation only.

    Returns
    -------
    avg_train_loss, avg_val_loss, avg_test_loss : Tuple[float, float, float]
    """

    # ------------------------------------------------------------------
    # Paths and logging
    # ------------------------------------------------------------------
    log_dir = f"{base_path}/runs/Reg_{dt_string}"
    save_dir = f"{base_path}/reg_model_saved"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    writer = SummaryWriter(log_dir, flush_secs=20)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"Starting training run: {run_id}")
    print(f"TensorBoard log dir: {log_dir}")
    print(f"Model save dir: {save_dir}")
    print("Saving policy: save the best validation model within the final training window")

    # ------------------------------------------------------------------
    # EDM constants
    # ------------------------------------------------------------------
    sigma_data = float(diffusion.sigma_data)
    sigma_data_sq = sigma_data ** 2
    model_in_channels = int(getattr(curve_model, "in_channels", 1))

    # Optional compilation
    if bool(compile_model) and hasattr(torch, "compile"):
        try:
            ck = compile_kwargs or {}
            curve_model = torch.compile(curve_model, **ck)
            print(f"[OK] torch.compile enabled. kwargs={ck}")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {repr(e)}. Continue without compilation.")

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _make_condition(
        mat_input: torch.Tensor,
        ele_input: torch.Tensor,
        voltage_input: torch.Tensor,
        *,
        drop_prob: float = 0.0,
        drop_non_voltage: bool = False,
    ) -> torch.Tensor:
        """
        Build the fused condition tensor.

        For CFG training:
        - drop material/text-related branches with probability drop_prob
        - always keep voltage_input, because it is the x-axis scaffold
        """
        if drop_non_voltage and drop_prob > 0:
            bsz = mat_input.shape[0]
            keep_mask = (torch.rand(bsz, device=mat_input.device) > drop_prob).unsqueeze(1)

            mat_input = torch.where(keep_mask, mat_input, torch.zeros_like(mat_input))
            ele_input = torch.where(keep_mask, ele_input, torch.zeros_like(ele_input))
            # voltage_input is intentionally kept

        return torch.cat([mat_input, ele_input, voltage_input], dim=1)

    def _make_unconditional_condition(
        mat_input: torch.Tensor,
        ele_input: torch.Tensor,
        voltage_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Explicit unconditional condition:
        - zero non-voltage branches
        - keep voltage branch
        """
        return torch.cat(
            [
                torch.zeros_like(mat_input),
                torch.zeros_like(ele_input),
                voltage_input,
            ],
            dim=1,
        )

    def _ensure_target_shape(target: torch.Tensor) -> torch.Tensor:
        """Convert (B, L) -> (B, 1, L) when model expects one input channel."""
        if target.dim() == 2 and model_in_channels == 1:
            return target.unsqueeze(1)
        return target

    def _is_plateau_scheduler(s) -> bool:
        """True if scheduler is ReduceLROnPlateau-like."""
        return s is not None and ("reducelronplateau" in s.__class__.__name__.lower())

    def _gather_2d(v2d: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Gather one value per sample from a (B, L) tensor using sample-wise indices."""
        idx = idx.clamp(min=0, max=v2d.shape[1] - 1).long()
        return v2d.gather(1, idx.view(-1, 1)).view(-1)

    def _normalize_point_weights(w: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Normalize per-sample point weights so their mean is 1."""
        if w is None:
            return None
        if w.dim() == 2:
            w = w.unsqueeze(1)  # (B, 1, L)
        mean_w = w.mean(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        return w / mean_w

    def _lambda_phys(epoch: int) -> float:
        """
        Physics-loss schedule:
        - 0 during warmup
        - then linearly ramp to physical_loss_weight
        """
        if not use_physical_constraints:
            return 0.0

        lam_max = float(physical_loss_weight)
        warmup = int(max(0, phys_warmup_epochs))
        ramp = int(max(1, phys_ramp_epochs))

        if epoch < warmup:
            return 0.0

        t = (epoch - warmup + 1) / float(ramp)
        t = max(0.0, min(1.0, t))
        return lam_max * t

    def _save_rng_state():
        """Save RNG state so deterministic evaluation does not affect training randomness."""
        return {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def _restore_rng_state(state):
        """Restore RNG state after deterministic evaluation."""
        torch.set_rng_state(state["cpu"])
        if torch.cuda.is_available() and state["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["cuda"])

    # ------------------------------------------------------------------
    # EDM forward and loss
    # ------------------------------------------------------------------
    def _edm_forward_parts(
        *,
        xt: torch.Tensor,
        sigmas: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One EDM forward pass.

        Returns
        -------
        D_x_pred : denoised prediction
        loss_weight : EDM weight
        """
        bsz = xt.shape[0]
        sigmas_r = sigmas.view(bsz, *((1,) * (xt.dim() - 1)))
        sigma_sq = sigmas_r ** 2

        # EDM preconditioning
        c_skip = sigma_data_sq / (sigma_sq + sigma_data_sq)
        c_out = sigmas_r * sigma_data / torch.sqrt(sigma_sq + sigma_data_sq)
        c_in = 1.0 / torch.sqrt(sigma_sq + sigma_data_sq)
        c_noise = (torch.log(sigmas.clamp_min(1e-12)) / 4.0).view(-1)

        model_in = c_in * xt
        F_x_pred = curve_model(model_in, c_noise, condition)

        if xt.dim() == 3 and F_x_pred.dim() == 2:
            F_x_pred = F_x_pred.unsqueeze(1)

        D_x_pred = c_skip * xt + c_out * F_x_pred
        loss_weight = (sigma_sq + sigma_data_sq) / ((sigmas_r * sigma_data) ** 2).clamp(min=1e-8)
        return D_x_pred, loss_weight

    def _edm_weighted_mse(
        *,
        D_x_pred: torch.Tensor,
        x0: torch.Tensor,
        loss_weight: torch.Tensor,
        point_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        EDM weighted MSE, optionally reweighted at important curve locations.
        """
        w = loss_weight
        if point_weights is not None:
            pw = _normalize_point_weights(point_weights).to(device=D_x_pred.device, dtype=D_x_pred.dtype)
            if pw.dim() == 2 and D_x_pred.dim() == 3:
                pw = pw.unsqueeze(1)
            w = w * pw
        return (w * (D_x_pred - x0) ** 2).mean()

    # ------------------------------------------------------------------
    # Physics-aware loss from polarization-curve keypoints
    # ------------------------------------------------------------------
    def _physical_terms_from_batch(
        *,
        voltage_norm: torch.Tensor,         # (B, L)
        current_gt_norm_2d: torch.Tensor,   # (B, L)
        current_pred_norm_2d: torch.Tensor, # (B, L)
        sigmas: torch.Tensor,               # (B,)
    ):
        """
        Build physics-aware supervision from GT polarization-curve structure.

        Returns
        -------
        phys_kp_loss : scalar
            Keypoint loss in normalized current space, sigma-gated.
        aux : dict
            Contains point_weights and diagnostics.
        """
        voltage_raw = denormalize_voltage(
            voltage_norm,
            v_min=VOLTAGE_MIN,
            v_max=VOLTAGE_MAX,
        )
        current_gt_raw = denormalize_current(
            current_gt_norm_2d,
            lc_min=CURRENT_MIN,
            lc_max=CURRENT_MAX,
        )

        alpha_neighbor = float(kp_alpha_key) * float(kp_alpha_neighbor_ratio)

        weights_t, aux = detect_keypoints_polarization_physical_batch_batched(
            voltage_raw,
            current_gt_raw,
            device="cpu",
            include_end_point=True,
            alpha_key=float(kp_alpha_key),
            alpha_neighbor=float(alpha_neighbor),
            neighbor_count=int(kp_neighbor_count),
            smooth_win=int(pol_smooth_win),
            passivation_drop_thresh=float(pol_passivation_drop_thresh),
            breakdown_rise_thresh=float(pol_breakdown_rise_thresh),
            critical_inflection_min_rel_strength=float(pol_critical_inflection_min_rel_strength),
            critical_topk=int(pol_critical_topk),
            critical_inflection_min_spacing=int(pol_critical_inflection_min_spacing),
        )

        # Optionally ignore an initial prefix region
        k0 = int(max(0, phys_ignore_prefix_k))
        if k0 > 0:
            weights_t = weights_t.clone()
            weights_t[:, :k0] = 0.0

        aux["point_weights"] = weights_t

        named_idx_list = [
            aux["corr_idx"],
            aux["passivation_onset_idx"],
            aux["active_peak_idx"],
            aux["passive_idx"],
            aux["breakdown_idx"],
            aux["repassivation_idx"],
        ]

        def _masked_keypoint_loss(
            idx_1d: torch.Tensor,
            sample_weight: Optional[torch.Tensor] = None,
        ):
            idx_1d = idx_1d.to(current_pred_norm_2d.device).long()
            if k0 > 0:
                idx_1d = idx_1d.clamp(min=k0)

            valid = (idx_1d >= 0)
            idx_safe = idx_1d.clamp(min=0, max=current_pred_norm_2d.shape[1] - 1)

            gt_v = _gather_2d(current_gt_norm_2d.to(current_pred_norm_2d.device), idx_safe)
            pr_v = _gather_2d(current_pred_norm_2d, idx_safe)

            loss_v = F.smooth_l1_loss(
                pr_v,
                gt_v,
                beta=float(phys_kp_beta_norm),
                reduction="none",
            )

            valid_f = valid.to(loss_v.dtype)
            if sample_weight is not None:
                sw = sample_weight.to(loss_v.device, dtype=loss_v.dtype)
                loss_v = loss_v * sw
                valid_f = valid_f * sw

            return loss_v, valid_f

        kp_num = torch.zeros(
            current_pred_norm_2d.shape[0],
            device=current_pred_norm_2d.device,
            dtype=current_pred_norm_2d.dtype,
        )
        kp_den = torch.zeros_like(kp_num)

        # Main named keypoints
        for idx_1d in named_idx_list:
            lv, mv = _masked_keypoint_loss(idx_1d)
            kp_num = kp_num + lv * (mv > 0).to(lv.dtype)
            kp_den = kp_den + (mv > 0).to(lv.dtype)

        # Additional ranked critical inflections
        if "critical_inflection_idx" in aux and "critical_inflection_strength" in aux:
            crit_idx = aux["critical_inflection_idx"].to(current_pred_norm_2d.device)
            crit_strength = aux["critical_inflection_strength"].to(current_pred_norm_2d.device)

            crit_max = crit_strength.max(dim=1, keepdim=True).values.clamp_min(1e-8)
            crit_strength_norm = crit_strength / crit_max

            for k in range(crit_idx.shape[1]):
                lv, mv = _masked_keypoint_loss(
                    crit_idx[:, k],
                    sample_weight=crit_strength_norm[:, k],
                )
                kp_num = kp_num + lv * (mv > 0).to(lv.dtype)
                kp_den = kp_den + mv

        kp_loss_per = kp_num / kp_den.clamp_min(1.0)

        # Apply physics loss mainly at lower sigma values
        gate = (sigmas <= float(phys_sigma_gate)).to(kp_loss_per.dtype)
        phys_kp_loss = (kp_loss_per * gate).sum() / gate.sum().clamp_min(1.0)

        aux["kp_loss_per"] = kp_loss_per.detach()
        aux["kp_gate_frac"] = gate.mean().detach() if gate.numel() else torch.tensor(0.0, device=gate.device)
        return phys_kp_loss, aux

    # ------------------------------------------------------------------
    # Shared batch-loss computation
    # ------------------------------------------------------------------
    def _compute_batch_loss(
        *,
        mat_input: torch.Tensor,
        ele_input: torch.Tensor,
        voltage_input: torch.Tensor,
        target: torch.Tensor,
        epoch: int,
        drop_condition: bool,
        need_physical_terms: bool,
    ):
        """
        Compute the full loss for one batch.
        Shared by training and evaluation to keep logic in one place.
        """
        bsz = mat_input.shape[0]

        condition = _make_condition(
            mat_input,
            ele_input,
            voltage_input,
            drop_prob=cond_drop_prob,
            drop_non_voltage=drop_condition,  # training=True, eval=False
        )

        # Optional explicit unconditional condition, useful for debugging/checking
        # that the non-voltage branches alone are nulled.
        _ = _make_unconditional_condition(mat_input, ele_input, voltage_input)

        x0 = _ensure_target_shape(target)

        sigmas, _ = diffusion.sample_sigmas(bsz, device=device, method="ve_log_uniform")
        xt, _, sigmas = diffusion.noise_curves(x0, sigmas)

        # Keep the core EDM computation in float32 for stability,
        # matching the original behavior.
        with torch.cuda.amp.autocast(enabled=False):
            D_pred, loss_w = _edm_forward_parts(
                xt=xt.float(),
                sigmas=sigmas.float(),
                condition=condition.float(),
            )

            point_w = None
            phys_kp_loss = None
            kp_gate_frac = None

            if use_physical_constraints and need_physical_terms:
                x0_2d = x0[:, 0, :] if x0.dim() == 3 else x0
                D_2d = D_pred[:, 0, :] if D_pred.dim() == 3 else D_pred

                phys_kp_loss, aux = _physical_terms_from_batch(
                    voltage_norm=voltage_input,
                    current_gt_norm_2d=x0_2d,
                    current_pred_norm_2d=D_2d,
                    sigmas=sigmas.float(),
                )

                if phys_use_point_weights:
                    point_w = aux["point_weights"]
                kp_gate_frac = float(aux["kp_gate_frac"].item()) if "kp_gate_frac" in aux else None

            loss_edm = _edm_weighted_mse(
                D_x_pred=D_pred,
                x0=x0.float(),
                loss_weight=loss_w,
                point_weights=point_w,
            )

            lam = _lambda_phys(epoch)
            loss = loss_edm
            if use_physical_constraints and phys_use_kp_loss and (phys_kp_loss is not None) and (lam > 0):
                loss = loss + lam * phys_kp_loss

        return loss, kp_gate_frac

    # ------------------------------------------------------------------
    # Evaluation on one loader
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval_loader(loader, *, use_ema: bool, epoch: int, eval_seed: Optional[int] = None) -> float:
        """
        Evaluate one loader.

        If deterministic_eval=True, the stochastic EDM validation objective is
        made reproducible by fixing RNG state during evaluation only.
        """
        if loader is None or len(loader) == 0:
            return float("inf")

        curve_model.eval()
        total = 0.0
        n = 0
        ctx = ema_handler if use_ema else nullcontext()

        rng_state = None
        if deterministic_eval and (eval_seed is not None):
            rng_state = _save_rng_state()
            torch.manual_seed(int(eval_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(eval_seed))

        try:
            with ctx:
                for mat_input, ele_input, voltage_input, target in loader:
                    mat_input = mat_input.to(device)
                    ele_input = ele_input.to(device)
                    voltage_input = voltage_input.to(device)
                    target = target.to(device)

                    loss, _ = _compute_batch_loss(
                        mat_input=mat_input,
                        ele_input=ele_input,
                        voltage_input=voltage_input,
                        target=target,
                        epoch=epoch,
                        drop_condition=False,
                        need_physical_terms=(phys_use_point_weights or phys_use_kp_loss),
                    )

                    total += float(loss.item())
                    n += 1
        finally:
            if rng_state is not None:
                _restore_rng_state(rng_state)

        return total / max(n, 1)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    avg_train_loss = float("inf")
    avg_val_loss = float("inf")
    avg_test_loss = float("inf")

    SAVE_LAST_N_EPOCHS = 30
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(num_epochs):
        epoch_start = time.time()
        lam_epoch = _lambda_phys(epoch)

        curve_model.train()
        total_train_loss = 0.0

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
            leave=False,
        )

        for batch_idx, (mat_input, ele_input, voltage_input, target) in pbar:
            mat_input = mat_input.to(device)
            ele_input = ele_input.to(device)
            voltage_input = voltage_input.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)

            loss, kp_gate_frac = _compute_batch_loss(
                mat_input=mat_input,
                ele_input=ele_input,
                voltage_input=voltage_input,
                target=target,
                epoch=epoch,
                drop_condition=True,
                need_physical_terms=(phys_use_point_weights or (phys_use_kp_loss and lam_epoch > 0)),
            )

            scaler.scale(loss).backward()

            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(curve_model.parameters(), max_norm=float(grad_clip_norm))

            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None and scheduler_step_per_batch and (not _is_plateau_scheduler(scheduler)):
                scheduler.step()

            ema_handler.update()
            total_train_loss += float(loss.item())

            postfix = {
                "loss": f"{loss.item():.4f}",
                "avg": f"{(total_train_loss / (batch_idx + 1)):.4f}",
            }
            if use_physical_constraints:
                postfix["λ"] = f"{lam_epoch:.3g}"
                if kp_gate_frac is not None:
                    postfix["kp_gate"] = f"{kp_gate_frac:.2f}"

            pbar.set_postfix(postfix)

        avg_train_loss = total_train_loss / max(len(train_loader), 1)

        # Validation/test with EMA weights
        avg_val_loss = _eval_loader(
            val_loader,
            use_ema=True,
            epoch=epoch,
            eval_seed=eval_seed_val,
        )
        avg_test_loss = _eval_loader(
            test_loader,
            use_ema=True,
            epoch=epoch,
            eval_seed=eval_seed_test,
        )

        # Scheduler step
        if scheduler is not None and _is_plateau_scheduler(scheduler):
            scheduler.step(avg_val_loss)
        elif scheduler is not None and (not scheduler_step_per_batch):
            scheduler.step()

        # Current LR
        if scheduler is not None and hasattr(scheduler, "get_last_lr"):
            current_lr = float(scheduler.get_last_lr()[0])
        else:
            current_lr = float(optimizer.param_groups[0]["lr"])

        # Console + TensorBoard logging
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | Test {avg_test_loss:.4f} | "
            f"LR {current_lr:.3e} | λ {lam_epoch:.3g} | {epoch_time:.1f}s"
        )

        writer.add_scalar(f"Reg_{dt_string}/Loss/Learning_Rate", current_lr, epoch)
        writer.add_scalar(f"Reg_{dt_string}/Loss/Lambda_Phys", lam_epoch, epoch)
        writer.add_scalars(
            f"Reg_{dt_string}/Loss/Train_Val_Loss",
            {"Train Loss": avg_train_loss, "Validation Loss": avg_val_loss},
            global_step=epoch,
        )
        writer.add_scalars(
            f"Reg_{dt_string}/Loss/Val_Test_Loss",
            {"Validation Loss": avg_val_loss, "Test Loss": avg_test_loss},
            global_step=epoch,
        )

        # Save best validation model only within the final training window
        in_last_window = epoch >= (num_epochs - SAVE_LAST_N_EPOCHS)
        if in_last_window and math.isfinite(avg_val_loss) and (avg_val_loss < best_val_loss - 1e-6):
            best_val_loss = avg_val_loss
            best_epoch = epoch

            save_path_best = f"{save_dir}/{run_id}_best_model.pt"
            if os.path.exists(save_path_best):
                delete_files(save_path_best)

            save_start_time = time.time()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": curve_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "ema_state_dict": ema_handler.state_dict(),
                },
                save_path_best,
            )
            save_duration = time.time() - save_start_time

            print(
                f"    *** [LAST-{SAVE_LAST_N_EPOCHS}] New best val loss: {best_val_loss:.6f} "
                f"at epoch {best_epoch+1}. Saved to {save_path_best} ({save_duration:.2f}s) ***"
            )

    writer.close()
    return avg_train_loss, avg_val_loss, avg_test_loss


def visualize_results(points, fig_save_path="test.png"):
    """
    Expected columns in `points`:
        points[:, 0] -> potential E (V)
        points[:, 1] -> predicted current density
        points[:, 2] -> true current density
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.linewidth": 1.2,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
    })

    # Sort data by potential
    points = points[np.argsort(points[:, 0])]
    E = points[:, 0]
    pred = points[:, 1]
    truth = points[:, 2]

    # Keep only valid positive values, then convert to log10
    mask = np.isfinite(E) & np.isfinite(pred) & np.isfinite(truth) & (pred > 0) & (truth > 0)
    E = E[mask]
    pred = np.log10(pred[mask])
    truth = np.log10(truth[mask])

    fig, axs = plt.subplots(1, 2, figsize=(7, 3.2), dpi=600)

    # Panel (a): Truth vs Prediction
    axs[0].plot(E, truth, "-", lw=1.5, color="#262626", label="Truth")
    axs[0].plot(E, pred, "--", lw=1.5, color="#D62728", label="Prediction")

    # Panel (b): Prediction Only
    axs[1].plot(E, pred, "-", lw=1.5, color="#D62728", label="Prediction")

    for i, ax in enumerate(axs):
        ax.set_xlabel("Potential voltage (V)")
        ax.set_ylabel(r"$\log_{10}(\mathrm{Current\ density}\;[\mathrm{A \cdot cm^{-2}}])$")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.grid(False)
        ax.legend(loc="best")

        ax.text(
            -0.15, 1.05,
            "a" if i == 0 else "b",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

    try:
        fig.tight_layout()
        fig.savefig(fig_save_path, bbox_inches="tight")
    except Exception as e:
        print(f"An error occurred while saving the plot: {e}")
    finally:
        plt.close(fig)


def visualize_results_all_test(points_list, fig_save_path="test_all.png"):
    """
    Plot all test samples in the same figure.

    Args:
        points_list: list of arrays, each with shape (N, 3)
            [:, 0] -> potential E (V)
            [:, 1] -> predicted current density
            [:, 2] -> true current density
        fig_save_path: output figure path
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.linewidth": 1.2,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
    })

    fig, axs = plt.subplots(1, 2, figsize=(7, 3.2), dpi=600)

    n_samples = len(points_list)
    cmap = plt.cm.get_cmap("tab20", max(n_samples, 1))

    for idx, points in enumerate(points_list):
        color = cmap(idx)

        # Sort by potential
        points = points[np.argsort(points[:, 0])]
        E = points[:, 0]
        pred = points[:, 1]
        truth = points[:, 2]

        # Keep only valid positive values, then convert to log10
        mask = (
            np.isfinite(E)
            & np.isfinite(pred)
            & np.isfinite(truth)
            & (pred > 0)
            & (truth > 0)
        )
        E = E[mask]
        pred = np.log10(pred[mask])
        truth = np.log10(truth[mask])

        if len(E) == 0:
            continue

        # Panel (a): Truth vs Prediction, same color for same sample
        axs[0].plot(
            E, truth,
            linestyle="-", lw=1.1, color=color, alpha=0.9,
            label=f"Sample {idx+1} Truth"
        )
        axs[0].plot(
            E, pred,
            linestyle="--", lw=1.1, color=color, alpha=0.9,
            label=f"Sample {idx+1} Pred"
        )

        # Panel (b): Prediction only
        axs[1].plot(
            E, pred,
            linestyle="-", lw=1.1, color=color, alpha=0.9,
            label=f"Sample {idx+1}"
        )

    for i, ax in enumerate(axs):
        ax.set_xlabel("Potential voltage (V)")
        ax.set_ylabel(r"$\log_{10}(\mathrm{Current\ density}\;[\mathrm{A \cdot cm^{-2}}])$")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.3))
        ax.grid(False)

        # Too many samples can make legend crowded
        if n_samples <= 8:
            ax.legend(loc="best", ncol=1)
        else:
            ax.legend(loc="best", ncol=2, fontsize=7)

        ax.text(
            -0.15, 1.05,
            "a" if i == 0 else "b",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

    try:
        fig.tight_layout()
        fig.savefig(fig_save_path, bbox_inches="tight")
    except Exception as e:
        print(f"An error occurred while saving the plot: {e}")
    finally:
        plt.close(fig)


TAG_MAP = {
    "date": "date",
    "seed": "rand_seed",
    "split": "split_ratio",

    "opt": "optimizer",
    "ep": "epochs",
    "bs": "batch_size",

    "T": "num_timesteps",
    "sched": "schedule_type",
    "cosS": "cosine_s",

    "ch": "model_channels",
    "attnDim": "attn_head_dim",
    "chMult": "channel_mult",
    "depths": "depths_per_level",
    "mlp": "mlp_ratio",
    "k": "kernel_size",
    "drop": "dropout_ratio",

    "warmup": "warmup_frac",
    "lr": "learning_rate",
    "wd": "weight_decay",
    "clip": "grad_clip_norm",
    "warmSteps": "num_warmup_steps",

    "sMin": "sigma_min",
    "sMax": "sigma_max",
    "sData": "sigma_data",
    "rho": "rho",

    "churn": "s_churn",
    "noise": "s_noise",
    "tminF": "s_tmin_frac",
    "tmaxF": "s_tmax_frac",
    "tmin": "s_tmin",
    "tmax": "s_tmax",

    "condDrop": "cond_drop_prob",
    "gSch": "guidance_schedule",
    "gScale": "guidance_scale",
    "gMin": "guidance_min_scale",

    "ema": "ema_decay",

    "steps": "num_sampling_steps",
    "sampler": "sampler_type",

    "vtc": "voltage_token_count",
    "vff": "voltage_fourier_features",
    "gate": "gate_max",

    # "b0": "beta_start",
    # "b1": "beta_end",

    "kpK": "kp_alpha_key",
    "kpNR": "kp_alpha_neighbor_ratio",
    "kpN": "kp_neighbor_count",
    "polSW": "pol_smooth_win",
    "polPD": "pol_passivation_drop_thresh",
    "polBR": "pol_breakdown_rise_thresh",

    # corrected names
    "critInfMinRelStr": "pol_critical_inflection_min_rel_strength",
    "critInfTopK": "pol_critical_topk",
    "critInfMinSpacing": "pol_critical_inflection_min_spacing",

    "physW": "lambda_phys_max",
    "pWU": "phys_warmup_epochs",
    "pRP": "phys_ramp_epochs",
    "kpB": "phys_kp_beta_norm",
    "sigG": "phys_sigma_gate",
}


def _auto_cast_value(value: str) -> Any:
    value = value.strip()

    if value.startswith(("(", "[", "{")) and value.endswith((")", "]", "}")):
        try:
            return ast.literal_eval(value)
        except Exception:
            return value

    if re.fullmatch(r"[+-]?\d+", value):
        try:
            return int(value)
        except Exception:
            pass

    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][+-]?\d+)?", value):
        try:
            f = float(value)
            if f.is_integer() and "." not in value and "e" not in value.lower():
                return int(f)
            return f
        except Exception:
            pass

    return value


def parse_best_hparam_log(log_str: str, use_full_names: bool = True) -> Dict[str, Any]:
    tags = sorted(TAG_MAP.keys(), key=len, reverse=True)
    tag_pattern = "|".join(re.escape(t) for t in tags)
    pattern = re.compile(rf"(^|__|_)({tag_pattern})-")

    matches = list(pattern.finditer(log_str))
    parsed = {}

    for i, m in enumerate(matches):
        short_tag = m.group(2)
        value_start = m.end()
        value_end = matches[i + 1].start() if i + 1 < len(matches) else len(log_str)
        raw_value = log_str[value_start:value_end]

        key = TAG_MAP[short_tag] if use_full_names else short_tag
        parsed[key] = _auto_cast_value(raw_value)

    return parsed


def objective(trial):

    def _fmt(x):
        # compact float formatting for filenames
        if isinstance(x, float):
            return f"{x:.8g}"
        return str(x)

    VIZ_INTERVAL = 1
    DATASET_SIZE = len(df)
    print(f"Dataset size: {DATASET_SIZE}")


    ################################ Hyperparameter Search Space ################################
    # =============================================================================
    # Fixed / semi-fixed Hyperparameters
    # =============================================================================
    SPLIT_RATIO = trial.suggest_categorical("split_ratio", [0.8])
    OPTIMIZER_NAME = trial.suggest_categorical("optimizer", ["AdamW"])
    EPOCHS = trial.suggest_categorical("epochs", [180]) # 180
    BATCH_SIZE = trial.suggest_categorical("batch_size", [32])

    # -----------------------------------------------------------------------------
    # Legacy diffusion schedule params (only matter if you still use them anywhere)
    # -----------------------------------------------------------------------------
    NUM_TIMESTEPS = trial.suggest_categorical("num_timesteps", [1000])
    SCHEDULE_TYPE = trial.suggest_categorical("schedule_type", ["cosine"])
    COSINE_S = trial.suggest_categorical("cosine_s", [0.006])

    # ========================= MODIFIED: 5-choice larger backbone preset search =========================
    BACKBONE_NAME = trial.suggest_categorical("backbone_name", ["base", "medium", "xlarge", "xlarge"])


    if BACKBONE_NAME == "base":
        MODEL_CHANNELS = 128
        CHANNEL_MULT = (1, 2, 4, 4)
        DEPTHS_PER_LEVEL = (2, 2, 2, 2)
        ATTN_HEAD_DIM = 128

    elif BACKBONE_NAME == "medium":
        MODEL_CHANNELS = 128
        CHANNEL_MULT = (1, 2, 4, 4)
        DEPTHS_PER_LEVEL = (2, 2, 2, 2)
        ATTN_HEAD_DIM = 128

    elif BACKBONE_NAME == "large":
        MODEL_CHANNELS = 192
        CHANNEL_MULT = (2, 2, 4, 4)
        DEPTHS_PER_LEVEL = (2, 2, 3, 3)
        ATTN_HEAD_DIM = 96

    elif BACKBONE_NAME == "xlarge":
        MODEL_CHANNELS = 256
        CHANNEL_MULT = (2, 2, 4, 4)
        DEPTHS_PER_LEVEL = (2, 2, 3, 3)
        ATTN_HEAD_DIM = 64

    else:
        raise ValueError(f"Unknown BACKBONE_NAME: {BACKBONE_NAME}")


    MLP_RATIO = trial.suggest_float("mlp_ratio", 2, 3.0, step=0.5)
    KERNEL_SIZE = trial.suggest_categorical("kernel_size", [3, 5])
    DROP_PROB = trial.suggest_float("dropout_ratio", 0.0, 0.1, step=0.02)

    # -----------------------------------------------------------------------------
    # Optimizer / training dynamics
    # -----------------------------------------------------------------------------
    WARMUP_FRAC = trial.suggest_float("warmup_frac", 0.02, 0.1)
    LR = trial.suggest_float("learning_rate", 5e-5, 3e-4, log=True)
    WEIGHT_DECAY = trial.suggest_float("weight_decay", 5e-5, 2e-2, log=True)
    GRAD_CLIP_NORM = trial.suggest_categorical("grad_clip_norm", [1.0, 2.0, 4.0, 6.0])

    # Warmup steps (requires DATASET_SIZE defined)
    STEPS_PER_EPOCH = int(math.ceil(DATASET_SIZE / BATCH_SIZE))
    TOTAL_STEPS = int(EPOCHS * STEPS_PER_EPOCH)

    NUM_WARMUP_STEPS = int(max(1, round(TOTAL_STEPS * WARMUP_FRAC)))
    NUM_WARMUP_STEPS = min(NUM_WARMUP_STEPS, max(1, TOTAL_STEPS - 1))

    # -----------------------------------------------------------------------------
    # EDM sigma range (important for EDM)
    # -----------------------------------------------------------------------------
    SIGMA_MIN = trial.suggest_float("sigma_min", 1e-3, 5e-3, log=True)
    SIGMA_MAX = trial.suggest_float("sigma_max", 20.0, 80.0, step=10.0)
    SIGMA_MAX = max(SIGMA_MAX, SIGMA_MIN * 1.01)
    SIGMA_DATA = trial.suggest_categorical("sigma_data", [0.3, 0.5, 0.7, 1.0])
    RHO = trial.suggest_float("rho", 4.0, 7.0, step=1.0)

    # -----------------------------------------------------------------------------
    # Stochastic sampling (EDM churn)
    # -----------------------------------------------------------------------------
    S_CHURN = trial.suggest_float("s_churn", 0.0, 2.0, step=0.5)
    S_NOISE = trial.suggest_float("s_noise", 0.9, 1.1, step=0.1)
    TMIN_FRAC = trial.suggest_categorical("s_tmin_frac", [0.0, 0.05, 0.1])
    TMAX_FRAC = trial.suggest_categorical("s_tmax_frac", [0.7, 0.85, 1.0])

    S_TMIN = SIGMA_MIN + TMIN_FRAC * (SIGMA_MAX - SIGMA_MIN)
    S_TMAX = SIGMA_MIN + TMAX_FRAC * (SIGMA_MAX - SIGMA_MIN)

    # -----------------------------------------------------------------------------
    # CFG: training vs sampling
    # -----------------------------------------------------------------------------
    COND_DROP_PROB = trial.suggest_float("cond_drop_prob", 0.04, 0.24, step=0.02)
    GUIDANCE_SCHEDULE = trial.suggest_categorical("guidance_schedule", ["cosine"])
    GUIDANCE_SCALE = trial.suggest_float("guidance_scale", 1.0, 5.0, step=0.25)
    GUIDANCE_MIN_SCALE = trial.suggest_float("guidance_min_scale", 0, 5.0, step=0.25)
    GUIDANCE_MIN_SCALE = min(GUIDANCE_MIN_SCALE, GUIDANCE_SCALE)

    # -----------------------------------------------------------------------------
    # EMA
    # -----------------------------------------------------------------------------
    EMA_DECAY = trial.suggest_categorical("ema_decay", [0.995, 0.997, 0.998, 0.999])

    # -----------------------------------------------------------------------------
    # Sampling steps / sampler choice
    # -----------------------------------------------------------------------------
    NUM_SAMPLING_STEPS = trial.suggest_categorical("num_sampling_steps", [30])
    SAMPLER_TYPE = trial.suggest_categorical("sampler_type", ["edm_heun"])

    # =============================================================================
    # Voltage tokenization / voltage-conditioning features
    # =============================================================================
    VOLTAGE_TOKEN_COUNT = trial.suggest_categorical("voltage_token_count", [2, 4, 8, 12])
    VOLTAGE_FOURIER_FEATURES = trial.suggest_categorical("voltage_fourier_features", [2, 4, 6])
    GATE_MAX = trial.suggest_categorical("gate_max", [1.5, 2.0, 2.5, 3.0])

    # =============================================================================
    # PHYSICAL CONSTRAINTS (FORCED ON) — polarization-specific
    # =============================================================================
    # -----------------------------------------------------------------------------
    # Keypoint / point-weight detector knobs
    # -----------------------------------------------------------------------------
    KP_ALPHA_KEY = trial.suggest_float("kp_alpha_key", 3.0, 12.0, step=1.0)
    KP_ALPHA_NEIGHBOR_RATIO = trial.suggest_float("kp_alpha_neighbor_ratio", 0.2, 1.0, step=0.1)
    KP_NEIGHBOR_COUNT = trial.suggest_int("kp_neighbor_count", 2, 6, step=1)

    # Smoothing for polarization keypoint / inflection detection
    POL_SMOOTH_WIN = trial.suggest_categorical("pol_smooth_win", [3, 5, 7, 9])
    POL_PASSIVATION_DROP_THRESH = trial.suggest_float("pol_passivation_drop_thresh", 0.10, 0.80, step=0.05)
    POL_BREAKDOWN_RISE_THRESH = trial.suggest_float("pol_breakdown_rise_thresh", 0.20, 1.20, step=0.05)

    # Critical inflection controls
    POL_CRITICAL_INFLECTION_MIN_REL_STRENGTH = trial.suggest_float("pol_critical_inflection_min_rel_strength", 0.03, 0.20, step=0.01)
    POL_CRITICAL_TOPK = trial.suggest_int("pol_critical_topk", 4, 8, step=1)
    POL_CRITICAL_INFLECTION_MIN_SPACING = trial.suggest_int("pol_critical_inflection_min_spacing", 1, 5, step=1)

    # -----------------------------------------------------------------------------
    # Physics loss knobs (most important)
    # -----------------------------------------------------------------------------
    PHYSICAL_LOSS_WEIGHT = trial.suggest_float("lambda_phys_max", 3e-4, 1e-2, log=True)
    PHYS_KP_BETA_NORM = trial.suggest_float("phys_kp_beta_norm", 0.01, 0.20)
    PHYS_SIGMA_GATE = trial.suggest_float("phys_sigma_gate", 0.08, 0.50)

    # -----------------------------------------------------------------------------
    # Ramp schedule
    # -----------------------------------------------------------------------------
    PHYS_WARMUP_FRAC = trial.suggest_float("phys_warmup_frac", 0.03, 0.25)
    PHYS_RAMP_FRAC = trial.suggest_float("phys_ramp_frac", 0.08, 0.50)
    PHYS_WARMUP_EPOCHS = int(round(EPOCHS * PHYS_WARMUP_FRAC))
    PHYS_RAMP_EPOCHS = max(1, int(round(EPOCHS * PHYS_RAMP_FRAC)))

    # -----------------------------------------------------------------------------
    # Useful derived constraints / guards
    # -----------------------------------------------------------------------------
    POL_SMOOTH_WIN = int(POL_SMOOTH_WIN)
    PHYS_WARMUP_EPOCHS = min(PHYS_WARMUP_EPOCHS, max(0, EPOCHS - 2))
    PHYS_RAMP_EPOCHS = min(PHYS_RAMP_EPOCHS, max(1, EPOCHS - PHYS_WARMUP_EPOCHS))


    # Data splitting
    train_indices, val_indices = split_train_val(df, train_size=SPLIT_RATIO, random_state=RAND_SEED)
    train_samples, val_samples = input_embed[train_indices, :], input_embed[val_indices, :]
    print('\n', f"Train shape: {train_samples.shape}, Validation shape: {val_samples.shape}")
    print(f"train indices: {train_indices[:10]}, val indices: {val_indices[:10]}")

    # Dataloader setup
    all_data_loader = DataLoader(MultimodalDataset(input_embed), batch_size=256, shuffle=False, drop_last=False)
    train_loader = DataLoader(MultimodalDataset(train_samples), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(MultimodalDataset(val_samples), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader = DataLoader(MultimodalDataset(test_input_embed), batch_size=len(test_input_embed), shuffle=False, drop_last=False)

    # Model initialization
    dt_string = datetime.now().strftime("_%Y.%m.%d.%H.%M.%S.%f")
    run_id = f"{dt_string}"

    saved_model_id = "__".join([  # double underscore between groups
        # -------------------------------------------------------------------------
        # 1) General / data split
        # -------------------------------------------------------------------------
        "_".join([
            f"date-{dt_string}",
            f"seed-{RAND_SEED}",
            f"split-{_fmt(SPLIT_RATIO)}",
        ]),

        # -------------------------------------------------------------------------
        # 2) Optimizer + run-length + batch
        # -------------------------------------------------------------------------
        "_".join([
            f"opt-{OPTIMIZER_NAME}",
            f"ep-{EPOCHS}",
            f"bs-{BATCH_SIZE}",
        ]),

        # -------------------------------------------------------------------------
        # 3) Legacy diffusion schedule params (only relevant if used)
        # -------------------------------------------------------------------------
        "_".join([
            f"T-{NUM_TIMESTEPS}",
            f"sched-{SCHEDULE_TYPE}",
            f"cosS-{_fmt(COSINE_S)}",
        ]),

        # -------------------------------------------------------------------------
        # 4) Network backbone params
        # -------------------------------------------------------------------------
        "_".join([
            f"ch-{MODEL_CHANNELS}",
            f"attnDim-{ATTN_HEAD_DIM}",
            f"chMult-{CHANNEL_MULT}",
            f"depths-{DEPTHS_PER_LEVEL}",
            f"mlp-{_fmt(MLP_RATIO)}",
            f"k-{KERNEL_SIZE}",
            f"drop-{_fmt(DROP_PROB)}",
        ]),

        # -------------------------------------------------------------------------
        # 5) Optimizer / training dynamics
        # -------------------------------------------------------------------------
        "_".join([
            f"warmup-{_fmt(WARMUP_FRAC)}",
            f"lr-{_fmt(LR)}",
            f"wd-{_fmt(WEIGHT_DECAY)}",
            f"clip-{_fmt(GRAD_CLIP_NORM)}",
            f"warmSteps-{NUM_WARMUP_STEPS}",
        ]),

        # -------------------------------------------------------------------------
        # 6) EDM sigma range
        # -------------------------------------------------------------------------
        "_".join([
            f"sMin-{_fmt(SIGMA_MIN)}",
            f"sMax-{_fmt(SIGMA_MAX)}",
            f"sData-{_fmt(SIGMA_DATA)}",
        ]),

        # -------------------------------------------------------------------------
        # 7) Karras schedule shape
        # -------------------------------------------------------------------------
        "_".join([
            f"rho-{_fmt(RHO)}",
        ]),

        # -------------------------------------------------------------------------
        # 8) Stochastic sampling (EDM churn)
        # -------------------------------------------------------------------------
        "_".join([
            f"churn-{_fmt(S_CHURN)}",
            f"noise-{_fmt(S_NOISE)}",
            f"tminF-{_fmt(TMIN_FRAC)}",
            f"tmaxF-{_fmt(TMAX_FRAC)}",
            f"tmin-{_fmt(S_TMIN)}",
            f"tmax-{_fmt(S_TMAX)}",
        ]),

        # -------------------------------------------------------------------------
        # 9) CFG (training vs sampling)
        # -------------------------------------------------------------------------
        "_".join([
            f"condDrop-{_fmt(COND_DROP_PROB)}",
            f"gSch-{GUIDANCE_SCHEDULE}",
            f"gScale-{_fmt(GUIDANCE_SCALE)}",
            f"gMin-{_fmt(GUIDANCE_MIN_SCALE)}",
        ]),

        # -------------------------------------------------------------------------
        # 10) EMA
        # -------------------------------------------------------------------------
        "_".join([
            f"ema-{_fmt(EMA_DECAY)}",
        ]),

        # -------------------------------------------------------------------------
        # 11) Sampling steps / sampler choice
        # -------------------------------------------------------------------------
        "_".join([
            f"steps-{NUM_SAMPLING_STEPS}",
            f"sampler-{SAMPLER_TYPE}",
        ]),

        # -------------------------------------------------------------------------
        # 12) Voltage tokenization / voltage features
        # -------------------------------------------------------------------------
        "_".join([
            f"vtc-{VOLTAGE_TOKEN_COUNT}",
            f"vff-{VOLTAGE_FOURIER_FEATURES}",
            f"gate-{_fmt(GATE_MAX)}",
        ]),

        # # -------------------------------------------------------------------------
        # # 13) (Optional) legacy beta schedule params
        # # -------------------------------------------------------------------------
        # "_".join([
        #     f"b0-{_fmt(BETA_START)}",
        #     f"b1-{_fmt(BETA_END)}",
        # ]),

        # -------------------------------------------------------------------------
        # 14) Polarization physical constraints / keypoint detector
        # -------------------------------------------------------------------------
        "_".join([
            f"kpK-{_fmt(KP_ALPHA_KEY)}",
            f"kpNR-{_fmt(KP_ALPHA_NEIGHBOR_RATIO)}",
            f"kpN-{KP_NEIGHBOR_COUNT}",
            f"polSW-{POL_SMOOTH_WIN}",
            f"polPD-{_fmt(POL_PASSIVATION_DROP_THRESH)}",
            f"polBR-{_fmt(POL_BREAKDOWN_RISE_THRESH)}",
            f"physW-{_fmt(PHYSICAL_LOSS_WEIGHT)}",   # lambda max
            f"critInfMinRelStr-{_fmt(POL_CRITICAL_INFLECTION_MIN_REL_STRENGTH)}",
            f"critInfTopK-{POL_CRITICAL_TOPK}",
            f"critInfMinSpacing-{POL_CRITICAL_INFLECTION_MIN_SPACING}",
            f"pWU-{PHYS_WARMUP_EPOCHS}",
            f"pRP-{PHYS_RAMP_EPOCHS}",
            f"kpB-{_fmt(PHYS_KP_BETA_NORM)}",
            f"sigG-{_fmt(PHYS_SIGMA_GATE)}",
        ]),
    ])

    curve_model = CurveUNetConditional_v6(

        in_channels=1,
        out_channels=1,
        dropout=DROP_PROB,

        # -------- architecture --------
        model_channels=MODEL_CHANNELS,
        channel_mult=CHANNEL_MULT,
        depths_per_level=DEPTHS_PER_LEVEL,
        attn_head_dim=ATTN_HEAD_DIM,
        mlp_ratio=MLP_RATIO,

        # -------- embeddings --------
        emb_dim=512,

        # >>> MODIFIED FOR POLARIZATION CURVE >>>
        process_emb_dim=768,
        test_cond_emb_dim=768,
        micro_emb_dim=768,
        ele_emb_dim=768,
        voltage_emb_dim=256,

        # -------- token counts for dense branches --------
        process_token_count=1,
        test_cond_token_count=1,
        micro_token_count=1,
        ele_token_count=2,

        # -------- conditioning controls --------
        use_deep_cond_cross_attn=True,
        deep_cond_context_dim=512,   # 512
        cond_fusion_heads=8,
        kernel_size=KERNEL_SIZE,

        # >>> MODIFIED FOR POLARIZATION CURVE >>>
        # -------- voltage-token conditioning --------
        voltage_token_count=VOLTAGE_TOKEN_COUNT,
        voltage_fourier_features=VOLTAGE_FOURIER_FEATURES,
        gate_max=GATE_MAX,

        # >>> MODIFIED FOR POLARIZATION CURVE >>>
        # -------- adaptive electrochemical regime modulation --------
        regime_hidden_ratio=2.0,

        # -------- EDM compat --------
        sigma_data=SIGMA_DATA,

    ).to(device)


    print_model_parameters(curve_model)

    ema_handler = EMA(curve_model, decay=EMA_DECAY, device=device)
    # criterion = nn.SmoothL1Loss()

    optimizer = getattr(optim, OPTIMIZER_NAME)(curve_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    num_training_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, NUM_WARMUP_STEPS, num_training_steps)

    for i in range(3):
        clear_gpu_memory()

    diffusion = DiffusionModel_v6(
        latent_dim=256,

        # --- legacy (kept for compatibility; only used if you sample by t_uniform / use get_sigma) ---
        num_timesteps=NUM_TIMESTEPS,
        schedule_type=SCHEDULE_TYPE,
        # beta_start=BETA_START,
        # beta_end=BETA_END,
        cosine_s=COSINE_S,

        # --- EDM/VE (actually used by your v6 training + sampler) ---
        num_sampling_steps=NUM_SAMPLING_STEPS,
        sigma_data=SIGMA_DATA,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
    )


    # Model saving path
    save_path = f"{base_path}/reg_model_saved/{run_id}_best_model.pt"
    if os.path.exists(save_path):
        os.remove(save_path)

    # Train and evaluate
    # ============================================================
    # Train and evaluate (UPDATED: physical constraints + keypoints)
    # ============================================================
    temp_avg_train_loss, temp_avg_val_loss, temp_avg_test_loss = train_and_evaluate(
        ema_handler=ema_handler,
        curve_model=curve_model,
        diffusion=diffusion,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        # criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=EPOCHS,
        run_id=run_id,
        dt_string=dt_string,
        cond_drop_prob=COND_DROP_PROB,
        grad_clip_norm=GRAD_CLIP_NORM,

        # physics knobs
        physical_loss_weight=float(PHYSICAL_LOSS_WEIGHT),
        phys_warmup_epochs=int(PHYS_WARMUP_EPOCHS),
        phys_ramp_epochs=int(PHYS_RAMP_EPOCHS),
        phys_kp_beta_norm=float(PHYS_KP_BETA_NORM),
        phys_sigma_gate=float(PHYS_SIGMA_GATE),

        # >>> MODIFIED FOR POLARIZATION CURVE >>>
        # polarization keypoint / inflection knobs
        kp_alpha_key=float(KP_ALPHA_KEY),
        kp_alpha_neighbor_ratio=float(KP_ALPHA_NEIGHBOR_RATIO),
        kp_neighbor_count=int(KP_NEIGHBOR_COUNT),

        pol_smooth_win=int(POL_SMOOTH_WIN),
        pol_passivation_drop_thresh=float(POL_PASSIVATION_DROP_THRESH),
        pol_breakdown_rise_thresh=float(POL_BREAKDOWN_RISE_THRESH),

        pol_critical_inflection_min_rel_strength=float(POL_CRITICAL_INFLECTION_MIN_REL_STRENGTH),
        pol_critical_topk=int(POL_CRITICAL_TOPK),
        pol_critical_inflection_min_spacing=int(POL_CRITICAL_INFLECTION_MIN_SPACING),
    )


    # print(f"temp_avg_train_loss: {temp_avg_train_loss}, temp_avg_val_loss: {temp_avg_val_loss}, temp_avg_test_loss: {temp_avg_test_loss}")
    if math.isnan(temp_avg_val_loss) or math.isnan(temp_avg_train_loss):
        return -100.0

    ## load saved model for evaluation and visualization
    save_path = r"E:\research\work2_research\code\curve_gen_corrosion_polarization\final_tune\reg_model_saved\_2026.07.19.17.08.24.533996_best_model.pt"
    checkpoint = torch.load(save_path, map_location=device)
    model_sd = checkpoint["model_state_dict"]

    trained_curve_model = CurveUNetConditional_v6(
        in_channels=1,
        out_channels=1,
        dropout=DROP_PROB,

        # -------- architecture --------
        model_channels=MODEL_CHANNELS,
        channel_mult=CHANNEL_MULT,
        depths_per_level=DEPTHS_PER_LEVEL,
        attn_head_dim=ATTN_HEAD_DIM,
        mlp_ratio=MLP_RATIO,

        # -------- embeddings --------
        emb_dim=512,

        process_emb_dim=768,
        test_cond_emb_dim=768,
        micro_emb_dim=768,
        ele_emb_dim=768,
        voltage_emb_dim=256,

        # -------- token counts for dense branches --------
        process_token_count=1,
        test_cond_token_count=1,
        micro_token_count=1,
        ele_token_count=2,

        # -------- conditioning controls --------
        use_deep_cond_cross_attn=True,
        deep_cond_context_dim=512,   # usually keep equal to emb_dim
        cond_fusion_heads=8,
        kernel_size=KERNEL_SIZE,

        # >>> MODIFIED FOR POLARIZATION CURVE >>>
        # -------- voltage-token conditioning --------
        voltage_token_count=VOLTAGE_TOKEN_COUNT,
        voltage_fourier_features=VOLTAGE_FOURIER_FEATURES,
        gate_max=GATE_MAX,

        # >>> MODIFIED FOR POLARIZATION CURVE >>>
        # -------- adaptive electrochemical regime modulation --------
        regime_hidden_ratio=2.0,

        # -------- EDM compat --------
        sigma_data=SIGMA_DATA,


    ).to(device)

    print_model_parameters(trained_curve_model)

    optimizer = torch.optim.AdamW(trained_curve_model.parameters())
    ema_handler = EMA(trained_curve_model, decay=EMA_DECAY, device=device) # Initialize EMA *first*
    trained_curve_model.load_state_dict(model_sd)
    if 'ema_state_dict' in checkpoint:
        ema_handler.load_state_dict(checkpoint['ema_state_dict']) # Load EMA state
    else:
        print("Warning: EMA state not found in checkpoint.")
    trained_curve_model.to(device)

    # Model evaluation
    temp_eval = [saved_model_id, dt_string, run_id]

    for i_repeat in range(100):
        optuna_optimize_objective=[]
        ######### evaluation on train and val set
        progress_bar = tqdm(enumerate(all_data_loader), total=len(all_data_loader))
        trained_curve_model.eval()
        with torch.no_grad():
            with ema_handler:
                outputs, targets = [], []
                step_tensors_list = []
                for _, (mat_input, ele_input, voltage_input, target) in progress_bar:

                    condition = torch.concat([mat_input, ele_input, voltage_input], dim=1).to(device)
                    batch_size = condition.shape[0]

                    sample_curves, train_val_step_tensor = diffusion.sample(
                        model=trained_curve_model,
                        batch_size=batch_size,
                        latent_dim=256,
                        condition=condition,
                        voltage_dim=256,

                        # ---- tracking / visualization ----
                        visualize=False,
                        visualize_interval=VIZ_INTERVAL,
                        save_path=f"{base_path}/step_vis/step_fig_{run_id}_repeat{i_repeat}/train_val_data",
                        track_indices=None,  # "all"
                        save_csv=False,
                        # use_ensemble=False,

                        final_clamp=True,
                        final_clamp_min=-1.0,
                        final_clamp_max=1.0,

                        # --- solver / steps ---
                        num_sampling_steps=NUM_SAMPLING_STEPS,
                        sampler_type=SAMPLER_TYPE,
                        rho=RHO,

                        # --- churn ---
                        s_churn=S_CHURN,
                        s_tmin=S_TMIN,
                        s_tmax=S_TMAX,
                        s_noise=S_NOISE,

                        # --- CFG during sampling (this IS classifier-free guidance at inference) ---
                        guidance_schedule=GUIDANCE_SCHEDULE,
                        guidance_scale=GUIDANCE_SCALE,
                        guidance_min_scale=GUIDANCE_MIN_SCALE,
                    )

                    outputs.append(sample_curves)
                    targets.append(target)
                    step_tensors_list.append(train_val_step_tensor)

        outputs = torch.cat(outputs, dim=0).detach().cpu()
        targets = torch.cat(targets, dim=0).detach().cpu()
        full_history_tensor = torch.cat(step_tensors_list, dim=0)
        print(f"outputs shape;, {outputs.shape}", type(outputs))
        print(f"targets shape;, {targets.shape}", type(targets))
        print(f"train_val_step_tensor shape;, {train_val_step_tensor.shape}", type(train_val_step_tensor))

        if torch.isinf(outputs).any() or torch.isnan(outputs).any():
            print("NaN/Inf detected directly from the model output, before post-processing!")
        else:
            print("Model output is clean. Proceeding to post-processing.")

        ##################################################### save final tensor as csv file for all samples
        print(f"targets[train_indices, :] shape: {targets[train_indices, :].shape}, outputs[train_indices, :] shape: {outputs[train_indices, :].shape}, voltage_embed_ts[train_indices, :] shape: {voltage_embed_ts[train_indices, :].shape}")
        train_result = combine_and_evaluate_curves_1d(
                        true_curves=targets[train_indices, :],
                        generated_curves=outputs[train_indices, :],
                        x_coords=voltage_embed_ts[train_indices, :]
                    )
        val_result = combine_and_evaluate_curves_1d(
                        true_curves=targets[val_indices, :],
                        generated_curves=outputs[val_indices, :],
                        x_coords=voltage_embed_ts[val_indices, :]
                    )

        temp_voltage_embed_ts = denormalize_voltage(voltage_embed_ts, v_min=VOLTAGE_MIN, v_max=VOLTAGE_MAX)
        outputs = denormalize_current(outputs, lc_min=CURRENT_MIN, lc_max=CURRENT_MAX)
        targets = denormalize_current(targets, lc_min=CURRENT_MIN, lc_max=CURRENT_MAX)

        pd.DataFrame(targets.cpu().numpy()).to_csv(f"{data_path}/_current_target.csv", index=False, header=False)
        # Export train data with indices as the first column
        pd.DataFrame(np.column_stack((np.array(train_indices), outputs[train_indices, :]))).to_csv(f"{base_path}/preds/{run_id}_train_pred_repeat{i_repeat}.csv", index=False)
        pd.DataFrame(np.column_stack((np.array(train_indices), targets[train_indices, :]))).to_csv(f"{base_path}/preds/{run_id}_train_target_repeat{i_repeat}.csv", index=False)
        pd.DataFrame(np.column_stack((np.array(train_indices), temp_voltage_embed_ts[train_indices, :]))).to_csv(f"{base_path}/preds/{run_id}_train_voltage_repeat{i_repeat}.csv", index=False)

        # Export val data with indices as the first column
        pd.DataFrame(np.column_stack((np.array(val_indices), outputs[val_indices, :]))).to_csv(f"{base_path}/preds/{run_id}_val_pred_repeat{i_repeat}.csv", index=False)
        pd.DataFrame(np.column_stack((np.array(val_indices), targets[val_indices, :]))).to_csv(f"{base_path}/preds/{run_id}_val_target_repeat{i_repeat}.csv", index=False)
        pd.DataFrame(np.column_stack((np.array(val_indices), temp_voltage_embed_ts[val_indices, :]))).to_csv(f"{base_path}/preds/{run_id}_val_voltage_repeat{i_repeat}.csv", index=False)
        pd.DataFrame(targets[val_indices, :]).to_csv(f"{data_path}/test_current_target.csv", index=False)
        plot_points = torch.concat([temp_voltage_embed_ts, outputs, targets], dim=1)

        ####### Evaluation on test data
        progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))
        trained_curve_model.eval()
        with torch.no_grad():
            outputs, targets = [], []
            step_tensors_list = []
            for _, (mat_input, ele_input, voltage_input, target) in progress_bar:

                condition = torch.concat([mat_input, ele_input, voltage_input], dim=1).to(device)
                # print(f"test condition shape: {condition.shape}")
                # print(f"\n test condition[0:5, 0:5]: {condition[0:5, 0:5]}")
                batch_size = condition.shape[0]
                # test_track_indices = 'all'
                sample_curves, test_step_tensor = diffusion.sample(
                    model=trained_curve_model,
                    batch_size=batch_size,
                    latent_dim=256,
                    condition=condition,
                    voltage_dim=256,

                    visualize=False,
                    visualize_interval=VIZ_INTERVAL,
                    save_path=f"{base_path}/step_vis/step_fig{run_id}_repeat{i_repeat}/test_data",
                    track_indices=None, # 'all'
                    save_csv=False,
                    # use_ensemble=False,

                    final_clamp=True,
                    final_clamp_min=-1.0,
                    final_clamp_max=1.0,

                    # --- solver / steps ---
                    num_sampling_steps=NUM_SAMPLING_STEPS,
                    sampler_type=SAMPLER_TYPE,
                    rho=RHO,

                    # --- churn ---
                    s_churn=S_CHURN,
                    s_tmin=S_TMIN,
                    s_tmax=S_TMAX,
                    s_noise=S_NOISE,

                    # --- CFG during sampling (this IS classifier-free guidance at inference) ---
                    guidance_schedule=GUIDANCE_SCHEDULE,
                    guidance_scale=GUIDANCE_SCALE,
                    guidance_min_scale=GUIDANCE_MIN_SCALE,
                )
                outputs.append(sample_curves)
                targets.append(target)
                step_tensors_list.append(test_step_tensor)

        outputs = torch.cat(outputs, dim=0).detach().cpu()
        targets = torch.cat(targets, dim=0).detach().cpu()
        full_history_tensor = torch.cat(step_tensors_list, dim=0)

        ####################################################################################
        test_result = combine_and_evaluate_curves_1d(outputs, targets, x_coords=test_voltage_embed_ts)

        test_indices = np.arange(len(outputs), dtype=int)
        outputs = denormalize_current(outputs, lc_min=CURRENT_MIN, lc_max=CURRENT_MAX)
        targets = denormalize_current(targets, lc_min=CURRENT_MIN, lc_max=CURRENT_MAX)
        test_plot_points = torch.concat([test_voltage_embed_ts, outputs, targets], dim=1)
        print(f"\n denormalize Test outputs shape;, {outputs.shape}", type(outputs))

        # pd.DataFrame(outputs).to_csv(f"{data_path}/check_test_stress_pred.csv", index=False, header=False)
        pd.DataFrame(np.column_stack((test_indices, outputs))).to_csv(f"{base_path}/preds/{run_id}_test_pred_repeat{i_repeat}.csv", index=False)
        pd.DataFrame(np.column_stack((test_indices, targets))).to_csv(f"{base_path}/preds/{run_id}_test_target_repeat{i_repeat}.csv", index=False)
        pd.DataFrame(np.column_stack((test_indices, test_voltage_embed_ts))).to_csv(f"{base_path}/preds/{run_id}_test_voltage_repeat{i_repeat}.csv", index=False)

        eval_results = temp_eval.copy()
        eval_results[1] = eval_results[1] + f"_repeat{i_repeat}"  # Append repeat number to run_id
        eval_results.extend([
            train_result['mse'], train_result['mae'], train_result['r2'],
            train_result['mean_mse'], train_result['mean_mae'], train_result['mean_r2'],
            train_result['dtw'], train_result['frechet'],

            val_result['mse'], val_result['mae'], val_result['r2'],
            val_result['mean_mse'], val_result['mean_mae'], val_result['mean_r2'],
            val_result['dtw'], val_result['frechet'],

            test_result['mse'], test_result['mae'], test_result['r2'],
            test_result['mean_mse'], test_result['mean_mae'], test_result['mean_r2'],
            test_result['dtw'], test_result['frechet'],
        ])

        # Save evaluation results
        with open(f"{base_path}/reg_model.csv", mode='a+', newline='') as csvfile:
            csv.writer(csvfile).writerow(eval_results)

        # if train_result['r2'] > val_result['r2'] and train_result['r2'] > test_result['r2'] and test_result['r2'] > 0.6:
        # if test_result['r2'] > -100:

        print(
            f"Good R2 values detected! "
            f"Train R2: {train_result['r2']}, "
            f"Val R2: {val_result['r2']}, "
            f"Test R2: {test_result['r2']}"
        )

        # Save evaluation results
        train_plot_points = plot_points[train_indices, :].detach().cpu().numpy()
        val_plot_points = plot_points[val_indices, :].detach().cpu().numpy()
        test_plot_points = test_plot_points.detach().cpu().numpy()
        print(f"val_plot_points.shape: {val_plot_points.shape}, {val_plot_points[0, :].shape}")
        print(f"test_plot_points.shape: {test_plot_points.shape}, {test_plot_points[0, :].shape}")
        print(f"train_plot_points.shape: {train_plot_points.shape}, {train_plot_points[0, :].shape}")

        # for i in range(len(train_indices)):
        for i in range(0, 600, 100):
            sub_train_plot_points = np.zeros((256, 3))
            sub_train_plot_points[:, 0] = train_plot_points[i, :256]
            sub_train_plot_points[:, 1] = train_plot_points[i, 256:512]
            sub_train_plot_points[:, 2] = train_plot_points[i, 512:768]

            fig_path = f"{base_path}/figs/{run_id}_train_{train_indices[i]}_{i_repeat}.png"
            visualize_results(sub_train_plot_points, fig_save_path=fig_path)

        # for i in range(len(val_indices)):
        for i in range(0, 10, 5):
            sub_val_plot_points = np.zeros((256, 3))
            sub_val_plot_points[:, 0] = val_plot_points[i, :256]
            sub_val_plot_points[:, 1] = val_plot_points[i, 256:512]
            sub_val_plot_points[:, 2] = val_plot_points[i, 512:768]

            fig_path = f"{base_path}/figs/{run_id}_val_{val_indices[i]}_{i_repeat}.png"
            visualize_results(sub_val_plot_points, fig_save_path=fig_path)

        for i in range(len(test_df)):
        # for i in range(0, 30, 5):
            sub_test_plot_points = np.zeros((256, 3))
            sub_test_plot_points[:, 0] = test_plot_points[i, :256]
            sub_test_plot_points[:, 1] = test_plot_points[i, 256:512]
            sub_test_plot_points[:, 2] = test_plot_points[i, 512:768]

            fig_path = f"{base_path}/figs/{run_id}_test_{test_indices[i]}_{i_repeat}.png"
            visualize_results(sub_test_plot_points, fig_save_path=fig_path)

        os.makedirs(f"{base_path}/figs", exist_ok=True)

        # all_test_points = []
        # for i in range(len(test_df)):
        #     sub_test_plot_points = np.zeros((256, 3))
        #     sub_test_plot_points[:, 0] = test_plot_points[i, :256]
        #     sub_test_plot_points[:, 1] = test_plot_points[i, 256:512]
        #     sub_test_plot_points[:, 2] = test_plot_points[i, 512:768]

        #     all_test_points.append(sub_test_plot_points)

        # fig_path = f"{base_path}/figs/{run_id}_test_all_{i_repeat}.png"
        # visualize_results_all_test(all_test_points, fig_save_path=fig_path)

        # else:
        #     print(f"Warning: Unusual R2 values detected! Train R2: {train_result['r2']}, Test R2: {test_result['r2']}")
        #     delete_files(save_path)
        optuna_optimize_objective.append(test_result['r2'])
    return statistics.mean(optuna_optimize_objective)


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Run hyperparameter optimization study for curve generation.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--root_dir", type=str, required=True, default='E:/shaohantian/research/work2_research/code/curve_gen', help="Root directory for the project")
    parser.add_argument("--data_path", type=str, required=True, default='E:/shaohantian/research/work2_research/code/curve_gen/datasets', help="Path to the datasets directory")
    parser.add_argument("--base_path", type=str, required=True, default='E:/shaohantian/research/work2_research/code/curve_gen/outputs_v3.9', help="Base path for outputs (logs, models, figures)")
    parser.add_argument("--n_trials", type=int, default=2, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=9000*60*60, help="Timeout for Optuna study in seconds")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for Optuna")
    parser.add_argument("--storage", type=str, default="sqlite:///./study_default.sqlite3", help="Optuna storage URL (e.g., sqlite:///./study.sqlite3)")
    parser.add_argument("--study_name", type=str, default="default_study", help="Optuna study name")
    parser.add_argument("--direction", type=str, default="maximize", choices=["maximize", "minimize"], help="Optuna study direction")
    parser.add_argument("--data_augmentation", default=False, help="Whether to use data augmentation for training")
    args = parser.parse_args()

    RAND_SEED = args.random_seed
    root_dir = args.root_dir
    data_path = args.data_path
    base_path = args.base_path
    data_augmentation = args.data_augmentation

    # Fix random seeds for reproducibility
    np.random.seed(RAND_SEED)
    random.seed(RAND_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"You are using '{device}' device")



    VOLTAGE_MIN = -2.1
    VOLTAGE_MAX = 2.1
    CURRENT_MIN = 7.e-11
    CURRENT_MAX = 0.5
    ####################### create output dir
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)

    child_dir = ['reg_model_saved', 'preds', 'figs', 'runs']
    for d in child_dir:
        new_dir = f"{base_path}/{d}"
        if not os.path.exists(new_dir):
            os.makedirs(new_dir, exist_ok=True)

    # log run paras and results
    outputs_cols = [
        'saved_model_id', 'create_time', 'run_id',

        # Training metrics
        'train_mse', 'train_mae', 'train_r2',
        'train_mean_mse', 'train_mean_mae', 'train_mean_r2',
        'train_dtw', 'train_frechet',

        # Validation metrics
        'val_mse', 'val_mae', 'val_r2',
        'val_mean_mse', 'val_mean_mae', 'val_mean_r2',
        'val_dtw', 'val_frechet',

        # experiment metrics
        'test_mse', 'test_mae', 'test_r2',
        'test_mean_mse', 'test_mean_mae', 'test_mean_r2',
        'test_dtw', 'test_frechet',

        # Notes
        'note'
    ]

    with open(f"{base_path}/reg_model.csv", newline='', mode='a+') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(outputs_cols)

    # ========== Train Data ==========
    df, voltage_embed_ts, input_embed = load_and_prepare_data(
        data_path=data_path,
        prefix='train_',
        save_path='./datasets/temp_embed.csv',
        device=device
    )

    test_df, test_voltage_embed_ts, test_input_embed = load_and_prepare_data(
        data_path=data_path,
        prefix='test_',
        save_path='./datasets/temp_test_embed.csv',
        device=device
    )
    print(f"test_voltage_embed_ts[0]: {test_voltage_embed_ts[0, :5]}")

    ####################### hyper-parameters otpmization
    study = optuna.create_study(
        storage=args.storage,
        study_name=args.study_name,
        direction=args.direction,
        load_if_exists=True
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        show_progress_bar=True
    )

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))