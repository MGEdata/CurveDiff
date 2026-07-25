
import warnings
from collections import OrderedDict
from typing import Dict, Optional, Union
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


class EMA:
    """
    Exponential Moving Average (EMA) tracker for model parameters (optionally buffers).

    Notes (important):
    - If you initialize EMA from the model weights (default), EMA is NOT "biased" in the Adam sense.
        Debiasing (divide by 1 - decay^t) is only mathematically appropriate if EMA is initialized at 0.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float,
        *,
        device: Optional[Union[torch.device, str]] = None,
        track_buffers: bool = False,
        init_strategy: str = "copy",   # "copy" (default) | "zeros"
        use_debias: bool = False,      # only meaningful when init_strategy="zeros"
    ):
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"EMA decay must be in [0,1), got {decay}")

        init_strategy = str(init_strategy).lower().strip()
        if init_strategy not in ("copy", "zeros"):
            raise ValueError(f"init_strategy must be 'copy' or 'zeros', got {init_strategy!r}")

        if use_debias and init_strategy != "zeros":
            warnings.warn(
                "use_debias=True is only mathematically correct when init_strategy='zeros'. "
                "Switching use_debias to False.",
                UserWarning,
            )
            use_debias = False

        self.model = model
        self.decay = float(decay)
        self.track_buffers = bool(track_buffers)
        self.init_strategy = init_strategy
        self.use_debias = bool(use_debias)

        self.num_updates: int = 0

        # Device to store EMA weights (None -> same device as model params when created)
        self._device = torch.device(device) if device is not None else None

        # shadow stores EMA values; backup stores originals when applying EMA temporarily
        self.shadow_params: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.shadow_buffers: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._backup_params: Dict[str, torch.Tensor] = {}
        self._backup_buffers: Dict[str, torch.Tensor] = {}

        self._init_shadow()

    def _infer_default_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            # model may have no parameters; fallback to CPU
            return torch.device("cpu")

    @torch.no_grad()
    def _init_shadow(self) -> None:
        base_device = self._device if self._device is not None else self._infer_default_device()

        # --- parameters ---
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if self.init_strategy == "copy":
                v = p.detach().clone()
            else:  # zeros
                v = torch.zeros_like(p.detach())
            self.shadow_params[name] = v.to(device=base_device, dtype=p.dtype)

        # --- buffers (optional) ---
        if self.track_buffers:
            for name, b in self.model.named_buffers():
                # buffers can be integral/bool; EMA makes sense mainly for floating buffers
                if not torch.is_floating_point(b):
                    continue
                if self.init_strategy == "copy":
                    v = b.detach().clone()
                else:
                    v = torch.zeros_like(b.detach())
                self.shadow_buffers[name] = v.to(device=base_device, dtype=b.dtype)

        # memory estimate
        total_elems = sum(t.numel() for t in self.shadow_params.values()) + sum(
            t.numel() for t in self.shadow_buffers.values()
        )
        total_bytes = sum(t.numel() * t.element_size() for t in self.shadow_params.values()) + sum(
            t.numel() * t.element_size() for t in self.shadow_buffers.values()
        )
        gb = total_bytes / (1024**3)

        print(
            f"EMA init: decay={self.decay:.6f}, init='{self.init_strategy}', "
            f"track_buffers={self.track_buffers}, use_debias={self.use_debias}. "
            f"Tracking {len(self.shadow_params)} params"
            f"{' + ' + str(len(self.shadow_buffers)) + ' buffers' if self.track_buffers else ''} "
            f"({total_elems:,} elems, ~{gb:.2f} GB) on device {base_device}."
        )

    @torch.no_grad()
    def update(self) -> None:
        """
        Call after each optimizer step.
        shadow = decay * shadow + (1 - decay) * current
        """
        self.num_updates += 1

        # Determine where EMA lives (shadow tensors device)
        ema_device = (
            next(iter(self.shadow_params.values())).device
            if len(self.shadow_params) > 0
            else (next(iter(self.shadow_buffers.values())).device if len(self.shadow_buffers) > 0 else self._infer_default_device())
        )

        # --- parameters ---
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow_params:
                continue

            cur = p.detach()
            if cur.device != ema_device:
                cur = cur.to(device=ema_device, non_blocking=True)

            shadow = self.shadow_params[name]
            # in-place EMA: shadow = decay*shadow + (1-decay)*cur
            shadow.mul_(self.decay).add_(cur, alpha=1.0 - self.decay)

        # --- buffers (optional) ---
        if self.track_buffers:
            for name, b in self.model.named_buffers():
                if name not in self.shadow_buffers:
                    continue
                if not torch.is_floating_point(b):
                    continue

                cur = b.detach()
                if cur.device != ema_device:
                    cur = cur.to(device=ema_device, non_blocking=True)

                shadow = self.shadow_buffers[name]
                shadow.mul_(self.decay).add_(cur, alpha=1.0 - self.decay)

    def _debias_factor(self) -> float:
        """
        Debiasing is only correct when init_strategy='zeros'.
        """
        if not self.use_debias:
            return 1.0
        # 1 - decay^t
        denom = 1.0 - (self.decay ** self.num_updates)
        if denom < 1e-12:
            return 1.0
        return denom

    @torch.no_grad()
    def apply(self) -> None:
        """
        Copy EMA weights into the model in-place (overwrites model params/buffers).
        Use `restore()` to revert.
        """
        if self.num_updates == 0 and self.init_strategy == "zeros":
            warnings.warn("EMA.apply() called before any updates and init_strategy='zeros'.", UserWarning)

        denom = self._debias_factor()

        # params
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow_params:
                continue
            src = self.shadow_params[name]
            if denom != 1.0:
                p.data.copy_(src / denom)
            else:
                p.data.copy_(src)

        # buffers
        if self.track_buffers:
            for name, b in self.model.named_buffers():
                if name not in self.shadow_buffers:
                    continue
                src = self.shadow_buffers[name]
                if denom != 1.0:
                    b.data.copy_(src / denom)
                else:
                    b.data.copy_(src)

    @torch.no_grad()
    def store(self) -> None:
        """Backup current model weights (to allow restore after apply)."""
        self._backup_params = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.shadow_params:
                self._backup_params[name] = p.detach().clone()

        self._backup_buffers = {}
        if self.track_buffers:
            for name, b in self.model.named_buffers():
                if name in self.shadow_buffers and torch.is_floating_point(b):
                    self._backup_buffers[name] = b.detach().clone()

    @torch.no_grad()
    def restore(self) -> None:
        """Restore weights from the last `store()` call."""
        if self._backup_params:
            for name, p in self.model.named_parameters():
                if name in self._backup_params:
                    p.data.copy_(self._backup_params[name])
            self._backup_params = {}

        if self.track_buffers and self._backup_buffers:
            for name, b in self.model.named_buffers():
                if name in self._backup_buffers:
                    b.data.copy_(self._backup_buffers[name])
            self._backup_buffers = {}

    # --- Context manager ---
    def __enter__(self):
        self.store()
        self.apply()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore()
        # don't suppress exceptions
        return False

    # --- Checkpointing ---
    def state_dict(self) -> dict:
        """
        Save EMA state. Shadows are moved to CPU for portability.
        """
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "track_buffers": self.track_buffers,
            "init_strategy": self.init_strategy,
            "use_debias": self.use_debias,
            "shadow_params": {k: v.detach().cpu() for k, v in self.shadow_params.items()},
            "shadow_buffers": {k: v.detach().cpu() for k, v in self.shadow_buffers.items()},
        }

    def load_state_dict(self, state_dict: dict, *, strict: bool = False) -> None:
        """
        Load EMA state. By default loads matching tensors; if strict=True, mismatches raise.
        """
        required = {"decay", "num_updates", "shadow_params"}
        missing = required - set(state_dict.keys())
        if missing:
            raise KeyError(f"EMA state_dict missing keys: {missing}")

        self.decay = float(state_dict["decay"])
        self.num_updates = int(state_dict["num_updates"])
        self.track_buffers = bool(state_dict.get("track_buffers", self.track_buffers))
        self.init_strategy = str(state_dict.get("init_strategy", self.init_strategy))
        self.use_debias = bool(state_dict.get("use_debias", self.use_debias))

        ema_device = self._device if self._device is not None else self._infer_default_device()

        # --- load params ---
        saved_params: Dict[str, torch.Tensor] = state_dict["shadow_params"]
        model_params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}

        self.shadow_params = OrderedDict()
        for name, mp in model_params.items():
            if name not in saved_params:
                if strict:
                    raise KeyError(f"EMA missing param in checkpoint: {name}")
                # initialize from current model
                self.shadow_params[name] = mp.detach().clone().to(device=ema_device, dtype=mp.dtype)
                continue

            sp = saved_params[name]
            if tuple(sp.shape) != tuple(mp.shape):
                if strict:
                    raise ValueError(f"EMA param shape mismatch for {name}: ckpt {sp.shape} vs model {mp.shape}")
                # fallback init
                self.shadow_params[name] = mp.detach().clone().to(device=ema_device, dtype=mp.dtype)
                continue

            self.shadow_params[name] = sp.detach().clone().to(device=ema_device, dtype=mp.dtype)

        # --- load buffers (optional) ---
        self.shadow_buffers = OrderedDict()
        saved_bufs: Dict[str, torch.Tensor] = state_dict.get("shadow_buffers", {})
        if self.track_buffers:
            model_bufs = {n: b for n, b in self.model.named_buffers() if torch.is_floating_point(b)}
            for name, mb in model_bufs.items():
                if name not in saved_bufs:
                    if strict:
                        raise KeyError(f"EMA missing buffer in checkpoint: {name}")
                    self.shadow_buffers[name] = mb.detach().clone().to(device=ema_device, dtype=mb.dtype)
                    continue
                sb = saved_bufs[name]
                if tuple(sb.shape) != tuple(mb.shape):
                    if strict:
                        raise ValueError(f"EMA buffer shape mismatch for {name}: ckpt {sb.shape} vs model {mb.shape}")
                    self.shadow_buffers[name] = mb.detach().clone().to(device=ema_device, dtype=mb.dtype)
                    continue
                self.shadow_buffers[name] = sb.detach().clone().to(device=ema_device, dtype=mb.dtype)


