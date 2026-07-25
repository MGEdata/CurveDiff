"""EDM/VE diffusion utilities for CurveDiff.

This module contains the beta-schedule compatibility helper and the
``DiffusionModel_v6`` sampling/training helper extracted from the original
combined module. The implementation is unchanged apart from module imports.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn

def get_beta_schedule(
    num_timesteps: int,
    schedule_type: str = "linear",
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    s: float = 0.008,
) -> torch.Tensor:
    """
    Create a beta schedule.

    Important
    ---------
    In this class, the main sampling logic is EDM/VE-style and mainly uses
    sigma schedules, not beta schedules. However, these beta-related buffers
    are still kept for compatibility with older DDPM-style training or utility
    code that may still depend on them.
    """
    if schedule_type == "linear":
        return torch.linspace(beta_start, beta_end, num_timesteps)

    if schedule_type == "cosine":
        # Cosine schedule often used in diffusion literature.
        pi = torch.pi if hasattr(torch, "pi") else math.pi
        t = torch.linspace(0, num_timesteps, num_timesteps + 1)
        alphas_cumprod = torch.cos(((t / num_timesteps) + s) / (1 + s) * pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    if schedule_type == "sigmoid":
        # Smooth sigmoid transition from beta_start to beta_end.
        t = torch.linspace(-6, 6, num_timesteps)
        return torch.sigmoid(t) * (beta_end - beta_start) + beta_start

    raise NotImplementedError(f"Unknown schedule: {schedule_type}")


class DiffusionModel_v6(nn.Module):
    """
    EDM/VE diffusion helper for 1D corrosion polarization curve generation.

    Main features
    -------------
    1. VE-style forward corruption: x_t = x_0 + sigma * noise
    2. EDM preconditioned denoiser
    3. Karras sigma schedule
    4. Euler / Heun EDM sampler
    5. Classifier-free guidance (CFG)
    6. Voltage-aware unconditional branch for CFG

    Notes
    -----
    - This class is primarily EDM/VE-style.
    - DDPM alpha/beta buffers are kept only for compatibility.
    - The outer sampler uses float64, following the spirit of official EDM.
    - The model forward is explicitly cast to the model's dtype (usually float32).
    """

    def __init__(
        self,
        latent_dim: int = 128,
        num_timesteps: int = 1000,
        num_sampling_steps: int = 18,
        schedule_type: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        cosine_s: float = 0.008,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.sigma_data = float(sigma_data)

        # ------------------------------------------------------------------
        # Legacy DDPM-style buffers (compatibility only)
        # ------------------------------------------------------------------
        beta = get_beta_schedule(
            num_timesteps=num_timesteps,
            schedule_type=schedule_type,
            beta_start=beta_start,
            beta_end=beta_end,
            s=cosine_s,
        )
        alpha = 1.0 - beta
        alpha_cum = torch.cumprod(alpha, dim=0)

        self.register_buffer("alpha_cum", alpha_cum)
        self.register_buffer("sqrt_alpha_cum", torch.sqrt(alpha_cum))
        self.register_buffer("sqrt_one_minus_alpha_cum", torch.sqrt(1.0 - alpha_cum))

        # ------------------------------------------------------------------
        # EDM / VE sigma range
        # ------------------------------------------------------------------
        self.sigma_min = float(max(sigma_min, 1e-6))
        self.sigma_max = float(max(sigma_max, self.sigma_min * 1.01))

    # ======================================================================
    # Basic sigma utilities
    # ======================================================================
    def _alpha_cumprod_to_sigma(self, alpha_cumprod: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((1.0 - alpha_cumprod) / torch.clamp(alpha_cumprod, min=1e-12))

    def get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, 0, self.num_timesteps - 1).long()
        alpha_cum_t = self.alpha_cum.to(t.device)[t]
        return self._alpha_cumprod_to_sigma(alpha_cum_t)

    def _karras_sigmas(
        self,
        n: int,
        device: torch.device,
        rho: float = 7.0,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """
        Build the Karras sigma schedule used in EDM sampling.
        Returns shape: (n + 1,), with a terminal zero appended.
        """
        s_min = float(self.sigma_min if sigma_min is None else sigma_min)
        s_max = float(self.sigma_max if sigma_max is None else sigma_max)

        s_min = max(s_min, 1e-6)
        s_max = max(s_max, s_min * 1.01)

        ramp = torch.linspace(0, 1, n, device=device, dtype=dtype)
        sigmas = (s_max ** (1.0 / rho) + ramp * (s_min ** (1.0 / rho) - s_max ** (1.0 / rho))) ** rho
        return torch.cat([sigmas, torch.zeros(1, device=device, dtype=dtype)], dim=0)

    def sample_sigmas(
        self,
        n: int,
        device: torch.device,
        method: str = "edm_log_normal",
        P_mean: float = -1.2,
        P_std: float = 1.2,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample sigma values for training.
        """
        if method == "edm_log_normal":
            rnd = torch.randn(n, device=device)
            sigmas = torch.exp(rnd * P_std + P_mean)
            sigmas = sigmas.clamp(min=self.sigma_min, max=self.sigma_max)
            return sigmas, None

        if method == "ve_log_uniform":
            u = torch.rand(n, device=device)
            log_sigma = math.log(self.sigma_max) + u * (
                math.log(self.sigma_min) - math.log(self.sigma_max)
            )
            sigmas = torch.exp(log_sigma).clamp(min=self.sigma_min, max=self.sigma_max)
            return sigmas, None

        if method == "t_uniform":
            timesteps = torch.randint(0, self.num_timesteps, (n,), device=device)
            sigmas = self.get_sigma(timesteps).clamp(min=self.sigma_min, max=self.sigma_max)
            return sigmas, timesteps

        raise ValueError(f"Unknown sigma sampling method: {method}")

    def noise_curves(
        self,
        x0: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        VE-style corruption:
            x_t = x_0 + sigma * noise
        """
        noise = torch.randn_like(x0)
        sigmas_r = self._expand_batch_scalar(sigmas, x0)
        xt = x0 + sigmas_r * noise
        return xt, noise, sigmas

    # ======================================================================
    # EDM denoising
    # ======================================================================
    @staticmethod
    def _expand_batch_scalar(v: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return v.view(v.shape[0], *((1,) * (ref.dim() - 1)))

    def _make_null_condition(
        self,
        condition: Optional[torch.Tensor],
        voltage_dim: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """
        Build the unconditional condition for CFG.

        If voltage_dim is provided, the last `voltage_dim` features are kept
        and only the non-voltage part is zeroed. This is appropriate when the
        voltage branch is the coordinate scaffold of the curve.

        If voltage_dim is not provided or invalid, fall back to all zeros.
        """
        if condition is None:
            return None

        if voltage_dim is None or voltage_dim <= 0 or voltage_dim >= condition.shape[1]:
            return torch.zeros_like(condition)

        non_voltage = torch.zeros_like(condition[:, :-voltage_dim])
        voltage = condition[:, -voltage_dim:]
        return torch.cat([non_voltage, voltage], dim=1)

    def _edm_denoise(
        self,
        model: nn.Module,
        x: torch.Tensor,
        sigma: torch.Tensor,                     # shape: (B,)
        condition: Optional[torch.Tensor],
        guidance_scale: float,
        uncond_condition: Optional[torch.Tensor] = None,
        voltage_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """
        EDM preconditioned denoiser with optional classifier-free guidance.

        Important dtype policy
        ----------------------
        - x and the outer EDM solver can stay in float64
        - the actual model forward is cast to the model dtype (usually float32)
        - the model output is cast back to x.dtype before EDM reconstruction
        """
        sigma = sigma.to(device=x.device, dtype=x.dtype).view(-1)
        sigma_r = self._expand_batch_scalar(sigma, x)

        sigma2 = sigma_r ** 2
        sigma_data2 = torch.tensor(self.sigma_data ** 2, device=x.device, dtype=x.dtype)

        # EDM preconditioning coefficients
        c_skip = sigma_data2 / (sigma2 + sigma_data2)
        c_out = sigma_r * self.sigma_data / torch.sqrt(sigma2 + sigma_data2)
        c_in = 1.0 / torch.sqrt(sigma2 + sigma_data2)
        c_noise = (torch.log(torch.clamp(sigma, min=1e-12)) / 4.0).view(-1)

        def run(single_condition: Optional[torch.Tensor]) -> torch.Tensor:
            # Match the model's parameter dtype for the actual forward pass
            model_dtype = next(model.parameters()).dtype

            model_x = (c_in * x).to(dtype=model_dtype)
            model_c_noise = c_noise.to(dtype=model_dtype)
            model_sigma = sigma.to(dtype=model_dtype)

            if single_condition is not None:
                single_condition = single_condition.to(dtype=model_dtype)

            pred = model(model_x, model_c_noise, single_condition)

            if x.dim() == 3 and pred.dim() == 2:
                pred = pred.unsqueeze(1)

            pred = pred.to(dtype=x.dtype)
            return c_skip * x + c_out * pred

        # No guidance
        if condition is None or guidance_scale is None or guidance_scale <= 0.0:
            return run(condition)

        # Build unconditional branch if not explicitly provided
        if uncond_condition is None:
            uncond_condition = self._make_null_condition(
                condition,
                voltage_dim=voltage_dim,
            )

        denoised_uncond = run(uncond_condition)
        denoised_cond = run(condition)

        # Standard CFG combination
        denoised = denoised_uncond + guidance_scale * (denoised_cond - denoised_uncond)
        return denoised

    # ======================================================================
    # Output helpers
    # ======================================================================
    def _format_curve_output(self, x: torch.Tensor, *, return_2d: bool) -> torch.Tensor:
        if not return_2d:
            return x

        if x.ndim == 3 and x.shape[1] == 1:
            return x[:, 0, :]

        if x.ndim == 4 and x.shape[2] == 1:
            return x[:, :, 0, :]

        return x

    @staticmethod
    def _reduce_per_sample(t: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, t.ndim))
        return t.mean(dim=dims)

    @staticmethod
    def _apply_final_clamp(
        t: torch.Tensor,
        final_clamp: bool,
        final_clamp_min: float,
        final_clamp_max: float,
    ) -> torch.Tensor:
        if t is None or (not torch.is_tensor(t)) or t.numel() == 0:
            return t
        return t.clamp(final_clamp_min, final_clamp_max) if final_clamp else t

    def _guidance_at_step(
        self,
        i: int,
        steps_to_take: int,
        guidance_scale: float,
        guidance_schedule: str,
        guidance_min_scale: float,
    ) -> float:
        if guidance_schedule == "constant" or steps_to_take <= 1:
            return float(guidance_scale)

        r = float(i) / float(steps_to_take - 1)

        if guidance_schedule == "linear":
            return float(guidance_min_scale + (guidance_scale - guidance_min_scale) * r)

        if guidance_schedule == "cosine":
            ramp = 0.5 - 0.5 * math.cos(math.pi * r)
            return float(guidance_min_scale + (guidance_scale - guidance_min_scale) * ramp)

        return float(guidance_scale)

    # ======================================================================
    # Single-run EDM sampler
    # ======================================================================
    @torch.no_grad()
    def _sample_single_edm(
        self,
        *,
        model: nn.Module,
        batch_size: int,
        latent_dim: int,
        condition: Optional[torch.Tensor],
        steps_to_take: int,
        sampler_type: str,
        guidance_scale: float,
        guidance_schedule: str,
        guidance_min_scale: float,
        s_churn: float,
        s_tmin: float,
        s_tmax: float,
        s_noise: float,
        rho: float,
        channels: int,
        uncond_condition: Optional[torch.Tensor] = None,
        voltage_dim: Optional[int] = None,
        track_indices: Union[List[int], str, None] = None,
        visualize: bool = False,
        save_csv: bool = False,
        visualize_interval: int = 20,
        save_path: str = "outputs/diffusion_steps",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model.eval()
        device = next(model.parameters()).device

        d = int(latent_dim) if latent_dim is not None else int(self.latent_dim)

        if track_indices is None:
            indices_to_track: List[int] = []
        elif track_indices == "all":
            indices_to_track = list(range(batch_size))
        else:
            indices_to_track = list(track_indices)

        if visualize:
            os.makedirs(save_path, exist_ok=True)
        if save_csv:
            os.makedirs(os.path.join(save_path, "csv_data"), exist_ok=True)

        # Outer EDM solver in float64
        sigmas = self._karras_sigmas(
            steps_to_take,
            device=device,
            rho=rho,
            dtype=torch.float64,
        )

        x = torch.randn(batch_size, channels, d, device=device, dtype=torch.float64) * sigmas[0]
        history_list: List[torch.Tensor] = []

        for i in tqdm(range(steps_to_take), desc=f"Sampling ({sampler_type})", leave=False):
            sigma_scalar = sigmas[i]
            sigma_next_scalar = sigmas[i + 1]

            sigma = sigma_scalar.repeat(batch_size)
            sigma_next = sigma_next_scalar.repeat(batch_size)

            gs_i = self._guidance_at_step(
                i=i,
                steps_to_take=steps_to_take,
                guidance_scale=guidance_scale,
                guidance_schedule=guidance_schedule,
                guidance_min_scale=guidance_min_scale,
            )

            # --------------------------------------------------------------
            # EDM churn
            # --------------------------------------------------------------
            if s_churn > 0:
                sigma_val = float(sigma_scalar.item())
                if s_tmin <= sigma_val <= s_tmax:
                    gamma = min(s_churn / steps_to_take, math.sqrt(2.0) - 1.0)
                    sigma_hat = sigma * (1.0 + gamma)

                    eps = torch.randn_like(x) * s_noise
                    sigma_hat_r = self._expand_batch_scalar(sigma_hat, x)
                    sigma_r = self._expand_batch_scalar(sigma, x)

                    x = x + eps * torch.sqrt(torch.clamp(sigma_hat_r ** 2 - sigma_r ** 2, min=0.0))
                    sigma = sigma_hat

            # --------------------------------------------------------------
            # Denoise at current sigma
            # --------------------------------------------------------------
            denoised = self._edm_denoise(
                model=model,
                x=x,
                sigma=sigma,
                condition=condition,
                guidance_scale=gs_i,
                uncond_condition=uncond_condition,
                voltage_dim=voltage_dim,
            ).to(x.dtype)

            sigma_r = self._expand_batch_scalar(sigma, x)
            d_cur = (x - denoised) / torch.clamp(sigma_r, min=1e-12)

            dt = self._expand_batch_scalar(sigma_next - sigma, x)
            x_euler = x + dt * d_cur

            # --------------------------------------------------------------
            # Heun correction
            # --------------------------------------------------------------
            if sampler_type == "edm_heun" and float(sigma_next_scalar.item()) > 0:
                denoised_next = self._edm_denoise(
                    model=model,
                    x=x_euler,
                    sigma=sigma_next,
                    condition=condition,
                    guidance_scale=gs_i,
                    uncond_condition=uncond_condition,
                    voltage_dim=voltage_dim,
                ).to(x.dtype)

                sigma_next_r = self._expand_batch_scalar(sigma_next, x_euler)
                d_next = (x_euler - denoised_next) / torch.clamp(sigma_next_r, min=1e-12)

                x = x + dt * 0.5 * (d_cur + d_next)
            else:
                x = x_euler

            should_save = (i % visualize_interval == 0) or (i == steps_to_take - 1)
            if should_save and indices_to_track:
                history_list.append(x[indices_to_track].detach().cpu().float().clone())

                if save_csv:
                    self._save_step_data_as_csv(
                        x_current=x.float(),
                        indices_to_save=indices_to_track,
                        step_idx=i,
                        t=int(i),
                        save_path=os.path.join(save_path, "csv_data"),
                    )

                if visualize:
                    for idx in indices_to_track:
                        if idx < batch_size:
                            self._visualize_step(
                                x_current=x.float(),
                                sample_idx=idx,
                                step_idx=i,
                                total_steps=steps_to_take,
                                t=int(i),
                                save_path=save_path,
                            )

        history_tensor = torch.stack(history_list, dim=1) if history_list else torch.empty(0)
        x = x.float()

        model.train()
        return x, history_tensor

    # ======================================================================
    # Public sampling API
    # ======================================================================
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        batch_size: int,
        latent_dim: int,
        condition: Optional[torch.Tensor],
        num_sampling_steps: int = None,
        eta: float = 0.0,  # kept for compatibility; not used here
        guidance_scale: float = 3.0,
        visualize: bool = False,
        save_csv: bool = False,
        visualize_interval: int = 20,
        save_path: str = "outputs/diffusion_steps",
        track_indices: Union[List[int], str, None] = None,
        sampler_type: str = "edm_heun",
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        rho: float = 7.0,
        channels: Optional[int] = None,
        return_2d: bool = True,
        uncond_condition: Optional[torch.Tensor] = None,
        voltage_dim: Optional[int] = None,
        final_clamp: bool = True,
        final_clamp_min: float = -1.0,
        final_clamp_max: float = 1.0,
        guidance_schedule: str = "constant",
        guidance_min_scale: float = 0.0,
        sample_seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model.eval()
        device = next(model.parameters()).device

        if condition is not None:
            condition = condition.to(device)
        if uncond_condition is not None:
            uncond_condition = uncond_condition.to(device)

        valid_samplers = {"edm_heun", "edm_euler"}
        if sampler_type not in valid_samplers:
            raise ValueError(
                f"Unsupported sampler_type='{sampler_type}'. "
                f"Supported samplers are: {sorted(valid_samplers)}"
            )

        steps_to_take = self.num_sampling_steps if num_sampling_steps is None else int(num_sampling_steps)
        steps_to_take = max(1, steps_to_take)

        if channels is None:
            channels = int(getattr(model, "in_channels", 1))

        def set_seed(seed: int):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

        if sample_seed is not None:
            set_seed(int(sample_seed))

        x, history_tensor = self._sample_single_edm(
            model=model,
            batch_size=batch_size,
            latent_dim=latent_dim,
            condition=condition,
            steps_to_take=steps_to_take,
            sampler_type=sampler_type,
            guidance_scale=guidance_scale,
            guidance_schedule=guidance_schedule,
            guidance_min_scale=guidance_min_scale,
            s_churn=s_churn,
            s_tmin=s_tmin,
            s_tmax=s_tmax,
            s_noise=s_noise,
            rho=rho,
            channels=channels,
            uncond_condition=uncond_condition,
            voltage_dim=voltage_dim,
            track_indices=track_indices,
            visualize=visualize,
            save_csv=save_csv,
            visualize_interval=visualize_interval,
            save_path=save_path,
        )

        x = self._apply_final_clamp(
            x,
            final_clamp=final_clamp,
            final_clamp_min=final_clamp_min,
            final_clamp_max=final_clamp_max,
        )

        x = self._format_curve_output(x, return_2d=return_2d)
        if torch.is_tensor(history_tensor):
            history_tensor = self._format_curve_output(history_tensor, return_2d=return_2d)

        model.train()
        return x, history_tensor

    # ======================================================================
    # Visualization / CSV helpers
    # ======================================================================
    def _save_step_data_as_csv(
        self,
        x_current: torch.Tensor,
        indices_to_save: List[int],
        step_idx: int,
        t: int,
        save_path: str,
    ):
        os.makedirs(save_path, exist_ok=True)

        valid_indices = [idx for idx in indices_to_save if idx < x_current.shape[0]]
        if not valid_indices:
            return

        selected = x_current[valid_indices].detach().cpu().numpy()

        if selected.ndim == 3 and selected.shape[1] == 1:
            selected = selected[:, 0, :]

        df = pd.DataFrame(selected, columns=[f"p{i}" for i in range(selected.shape[1])])
        df.insert(0, "sample_index", valid_indices)

        filename = f"step_{(step_idx + 1):04d}_t_{t:04d}.csv"
        df.to_csv(os.path.join(save_path, filename), index=False)

    def _visualize_step(
        self,
        x_current: torch.Tensor,
        sample_idx: int,
        step_idx: int,
        total_steps: int,
        t: int,
        save_path: str,
    ):
        os.makedirs(save_path, exist_ok=True)

        sample = x_current[sample_idx].detach().cpu().numpy()

        if sample.ndim == 2 and sample.shape[0] == 1:
            sample = sample[0]

        if np.isnan(sample).any() or np.isinf(sample).any():
            return

        fig, ax = plt.subplots(figsize=(8, 3), dpi=200)
        ax.plot(sample, linewidth=2.0)
        ax.set_title(f"Sample {sample_idx} | step {step_idx + 1}/{total_steps}")
        ax.set_xlabel("Potential grid index")
        ax.set_ylabel("Pred current density (norm)")
        plt.tight_layout()

        filename = f"sample_{sample_idx}_step_{(step_idx + 1):04d}_t_{t:04d}.png"
        plt.savefig(os.path.join(save_path, filename), dpi=200, bbox_inches="tight")
        plt.close(fig)
