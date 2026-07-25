"""Conditional one-dimensional U-Net architecture for CurveDiff.

This module contains ``CurveUNetConditional_v6`` and all supporting CuT,
attention, tokenization, fusion, sampling-resolution and modulation blocks
extracted from the original combined module. Model behavior is unchanged.
"""

from __future__ import annotations

import math
import warnings
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Calculates the total and trainable parameters of a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to analyze.

    Returns:
        Tuple[int, int]: A tuple containing:
            - total_params (int): The total number of parameters.
            - trainable_params (int): The number of trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print(f"Model: {model.__class__.__name__}")
    print(f"  - Total Parameters:     {total_params:,}")
    print(f"  - Trainable Parameters: {trainable_params:,}")
    print(f"  - Non-Trainable Params: {(total_params - trainable_params):,}")
    print("=" * 60)

    return total_params, trainable_params


# ============================================================================
# Small utilities
# ============================================================================
def get_num_groups(channels, min_groups=1, max_groups=32): # Unchanged
    if channels == 0: return min_groups
    min_groups = min(min_groups, channels) if channels > 0 else 1
    if channels < max_groups and channels >= min_groups: max_groups = channels
    elif channels < min_groups: return 1
    num = max_groups
    while num >= min_groups:
        if channels % num == 0: return num
        if max_groups & (max_groups -1) == 0 and num > min_groups : num //= 2 # check power of 2 and avoid num becoming too small
        else: num -= 1
    return max(1, min_groups if channels % min_groups == 0 else 1)

# ============================================================================
# Attention blocks
# ============================================================================
class CrossAttention1D(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads, head_dim_override=None, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim_override if head_dim_override is not None else query_dim // num_heads
        if query_dim % self.num_heads != 0 and head_dim_override is None:
            raise ValueError(f"query_dim ({query_dim}) must be divisible by num_heads ({num_heads}) if head_dim_override is not set.")
        inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5

        self.norm_query = nn.GroupNorm(get_num_groups(query_dim, min_groups=1, max_groups=query_dim if query_dim <=32 else 32), query_dim, eps=1e-6)
        self.norm_context = nn.LayerNorm(context_dim, eps=1e-6) # Context is (B, S, D)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        nn.init.zeros_(self.to_out[0].weight)
        if self.to_out[0].bias is not None: nn.init.zeros_(self.to_out[0].bias)

    def forward(self, query, context):
        # query: (B, C_query, L_query) - feature map from U-Net
        # context: (B, S_context, D_context) - conditional embedding
        B_q, C_q, L_q = query.shape

        query_norm = self.norm_query(query).permute(0, 2, 1) # (B, L_query, C_query)
        q = self.to_q(query_norm).view(B_q, L_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, L_query, Dh)

        context_norm = self.norm_context(context) # (B, S_context, D_context)
        k = self.to_k(context_norm).view(B_q, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, S_context, Dh)
        v = self.to_v(context_norm).view(B_q, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, H, S_context, Dh)

        sim = torch.einsum('b h l d, b h s d -> b h l s', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b h l s, b h s d -> b h l d', attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B_q, L_q, -1) # (B, L_query, H*Dh)

        out = self.to_out(out).permute(0, 2, 1) # (B, C_query, L_query)

        return query + out


class AttentionBlock1D(nn.Module): # Largely unchanged, self-attention
    def __init__(self, channels, num_heads=8, head_dim=None):
        super().__init__()
        self.channels = channels
        if head_dim is None:
            num_heads = num_heads if num_heads > 0 else 1
            if channels % num_heads != 0:
                warnings.warn(f"SelfAttention: Channels ({channels}) not divisible by num_heads ({num_heads}). Adjusting num_heads.")
                # Find a divisor or set to 1
                valid_heads = [h for h in range(1, num_heads + 1) if channels % h == 0]
                num_heads = valid_heads[-1] if valid_heads else 1

            self.head_dim = channels // num_heads
            self.num_heads = num_heads
        else:
            self.head_dim = head_dim if head_dim > 0 else 1
            self.num_heads = channels // self.head_dim
            if channels % self.head_dim != 0:
                raise ValueError(f"SelfAttention: Channels ({channels}) must be divisible by head_dim ({self.head_dim})")

        self.scale = self.head_dim ** -0.5
        num_groups_norm = get_num_groups(channels, max_groups=self.num_heads if self.num_heads > 0 else 1)
        self.norm = nn.GroupNorm(num_groups_norm, channels, eps=1e-6)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.proj_out.weight)
        if self.proj_out.bias is not None: nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, emb=None, cond_emb_context=None): # emb and cond_emb_context not used by this basic self-attn
        B, C, L = x.shape; h_ = self.norm(x); qkv_ = self.qkv(h_)
        qkv_ = qkv_.reshape(B, self.num_heads, self.head_dim * 3, L)
        q, k, v = torch.chunk(qkv_, 3, dim=2)
        weights = torch.einsum('b h d l, b h d m -> b h l m', q, k) * self.scale
        weights = F.softmax(weights.float(), dim=-1).type(weights.dtype)
        attn_output = torch.einsum('b h l m, b h d m -> b h l d', weights, v)
        attn_output = attn_output.permute(0, 1, 3, 2).contiguous().view(B, C, L)
        return x + self.proj_out(attn_output)

# ============================================================================
# Resolution changes
# ============================================================================
class Downsample1D(nn.Module):
    def __init__(self, channels, use_conv=True, out_channels=None, kernel_size: int = 3):
        super().__init__()
        self.out_channels = out_channels or channels
        stride = 2

        if use_conv:
            # <<< MODIFIED: Padding is now calculated dynamically from kernel_size >>>
            padding = (kernel_size - 1) // 2
            self.op = nn.Conv1d(channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            if self.out_channels != channels:
                raise ValueError("out_channels must be equal to channels for AvgPool")
            self.op = nn.AvgPool1d(kernel_size=stride, stride=stride)

    def forward(self, x):
        return self.op(x)


class Upsample1D(nn.Module):
    def __init__(self, channels, use_conv=True, out_channels=None, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        if use_conv:
            # <<< MODIFIED: Padding is now calculated dynamically from kernel_size >>>
            padding = (kernel_size - 1) // 2
            self.conv = nn.Conv1d(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        x = self.upsample(x)
        if self.use_conv:
            x = self.conv(x)
        return x


def modulate(x, scale, shift):
    # x: (B, C, L)
    # scale, shift: (B, C)
    return x * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)


# ============================================================================
# Residual / modulation blocks
# ============================================================================
class CCRBlock(nn.Module):
    """
    A revised, simplified residual block that receives modulation parameters externally.
    """
    # MODIFIED: __init__ signature no longer needs emb_channels.
    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        padding = kernel_size // 2

        # MODIFIED: The internal modulation network has been removed from this block.
        # self.modulation = nn.Sequential(...)

        self.norm1 = nn.GroupNorm(get_num_groups(channels), channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.GroupNorm(get_num_groups(channels), channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)


    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:

        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)

        h = modulate(h, scale=scale, shift=shift)
        h = F.silu(h)

        h = self.dropout(h)
        h = self.conv2(h)

        return x + h


class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal embedding for a scalar (c_noise)."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,)
        half = self.dim // 2
        device = x.device
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), half, device=device)
        )
        args = x[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


def _gate_param_from_init(init_value: float, gate_max: float = 2.0) -> torch.Tensor:
    """
    We parameterize gate = gate_max * sigmoid(p).
    This returns p such that gate(init) ~= init_value.
    """
    eps = 1e-6
    v = float(init_value)
    v = min(max(v, eps), gate_max - eps)
    p = math.log((v / gate_max) / (1.0 - (v / gate_max)))
    return torch.tensor(p, dtype=torch.float32)



class CuTBlock_v6(nn.Module):
    """
    Conditioned feature block with four stages:
        1) modulated residual convolution
        2) self-attention
        3) optional cross-attention
        4) channel MLP

    v6 stability features
    ---------------------
    - bounded residual gates: gate = gate_max * sigmoid(param)
    - LayerNorm on the conditioning embedding before modulation projection
    - optional tanh limit on FiLM scale
    - delta-only gating for conv / self-attn / cross-attn updates
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        *,
        kernel_size: int = 5,
        num_heads: int = 8,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        use_deep_cond: bool = False,
        cond_context_dim: int = None,
        num_cross_attn_heads: int = 8,
        cross_attn_head_dim: int = 64,
        dropout: float = 0.1,
        gate_max: float = 2.0,
        gate_init_conv: float = 1.0,
        gate_init_attn: float = 0.2,
        gate_init_cross: float = 0.2,
        gate_init_mlp: float = 1.0,
        film_scale_tanh: bool = True,
    ):
        super().__init__()
        self.use_deep_cond = bool(use_deep_cond)
        self.gate_max = float(gate_max)
        self.film_scale_tanh = bool(film_scale_tanh)
        self.out_channels = int(out_channels)

        # Number of FiLM parameter groups:
        # conv, self-attn, optional cross-attn, mlp
        self.num_mod_params = 8 if self.use_deep_cond else 6

        # Shared projection from the global embedding to all FiLM parameters
        self.modulation = nn.Sequential(
            nn.LayerNorm(emb_channels),
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels * self.num_mod_params),
        )

        # 1) Local convolutional refinement
        self.conv_stage = CCRBlock(
            channels=out_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # 2) Self-attention over the current feature map
        self.norm_attn = nn.LayerNorm(out_channels)
        self.attn = AttentionBlock1D(
            out_channels,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # 3) Optional cross-attention to condition tokens
        if self.use_deep_cond:
            self.norm_cross_attn = nn.LayerNorm(out_channels)
            self.deep_cross_attn = CrossAttention1D(
                query_dim=out_channels,
                context_dim=cond_context_dim,
                num_heads=num_cross_attn_heads,
                head_dim_override=cross_attn_head_dim,
                dropout=dropout,
            )

        # 4) Pointwise MLP over channels
        hidden_mlp_channels = int(out_channels * mlp_ratio)
        self.norm_mlp = nn.LayerNorm(out_channels)
        self.mlp = nn.Sequential(
            nn.Conv1d(out_channels, hidden_mlp_channels, kernel_size=1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_mlp_channels, out_channels, kernel_size=1),
        )

        # Residual projection if the channel count changes
        self.skip_connection = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        # Learnable bounded gates for each stage
        self.gate_conv_p = nn.Parameter(_gate_param_from_init(gate_init_conv, self.gate_max))
        self.gate_attn_p = nn.Parameter(_gate_param_from_init(gate_init_attn, self.gate_max))
        if self.use_deep_cond:
            self.gate_cross_p = nn.Parameter(_gate_param_from_init(gate_init_cross, self.gate_max))
        self.gate_mlp_p = nn.Parameter(_gate_param_from_init(gate_init_mlp, self.gate_max))

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _gate(self, p: torch.Tensor) -> torch.Tensor:
        """Convert an unconstrained parameter into a bounded positive gate."""
        return self.gate_max * torch.sigmoid(p)

    def _prep_film(self, scale: torch.Tensor, shift: torch.Tensor):
        """Optionally limit FiLM scale for stability; keep shift unchanged."""
        if self.film_scale_tanh:
            scale = torch.tanh(scale)
        return scale, shift

    @staticmethod
    def _apply_norm_1d(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        """
        Apply LayerNorm over channels for a tensor of shape (B, C, L).

        LayerNorm expects the normalized dimension last, so the tensor is
        temporarily transposed to (B, L, C) and then transposed back.
        """
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def _modulate_from_emb(self, emb: torch.Tensor):
        """
        Project the global conditioning embedding into FiLM parameters.

        Returns
        -------
        If cross-attention is disabled:
            (conv_film, attn_film, mlp_film)

        If cross-attention is enabled:
            (conv_film, attn_film, cross_film, mlp_film)

        Each FiLM tuple has the form:
            (scale, shift)
        """
        params = self.modulation(emb).chunk(self.num_mod_params, dim=1)

        if self.use_deep_cond:
            scale_conv, shift_conv, scale_attn, shift_attn, scale_cross, shift_cross, scale_mlp, shift_mlp = params
            return (
                self._prep_film(scale_conv, shift_conv),
                self._prep_film(scale_attn, shift_attn),
                self._prep_film(scale_cross, shift_cross),
                self._prep_film(scale_mlp, shift_mlp),
            )

        scale_conv, shift_conv, scale_attn, shift_attn, scale_mlp, shift_mlp = params
        return (
            self._prep_film(scale_conv, shift_conv),
            self._prep_film(scale_attn, shift_attn),
            self._prep_film(scale_mlp, shift_mlp),
        )

    def _apply_delta_gated_update(
        self,
        h: torch.Tensor,
        block_in: torch.Tensor,
        block_out: torch.Tensor,
        gate_param: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply delta-only residual gating.

        If a stage returns (block_in + delta), this keeps only delta and applies:
            h <- h + gate * delta
        """
        return h + self._gate(gate_param) * (block_out - block_in)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        cond_emb_context=None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, L)
            Input feature map.
        emb : (B, emb_channels)
            Global conditioning embedding used to generate FiLM parameters.
        cond_emb_context : optional
            Condition-token context used by the optional cross-attention stage.

        Returns
        -------
        h : (B, out_channels, L)
            Refined feature map.
        """
        # Base residual stream
        h = self.skip_connection(x)

        # --------------------------------------------------------------
        # Generate all FiLM parameters from the global embedding
        # --------------------------------------------------------------
        if self.use_deep_cond:
            (scale_conv, shift_conv), (scale_attn, shift_attn), (scale_cross, shift_cross), (scale_mlp, shift_mlp) = self._modulate_from_emb(emb)
        else:
            (scale_conv, shift_conv), (scale_attn, shift_attn), (scale_mlp, shift_mlp) = self._modulate_from_emb(emb)

        # --------------------------------------------------------------
        # 1) Convolution stage
        # --------------------------------------------------------------
        # conv_stage returns something like: h_in + delta
        h_in = h
        h_conv_out = self.conv_stage(h_in, scale=scale_conv, shift=shift_conv)
        h = self._apply_delta_gated_update(h, h_in, h_conv_out, self.gate_conv_p)

        # --------------------------------------------------------------
        # 2) Self-attention stage
        # --------------------------------------------------------------
        h_norm = self._apply_norm_1d(h, self.norm_attn)
        attn_in = modulate(h_norm, scale=scale_attn, shift=shift_attn)
        h_attn_out = self.attn(attn_in)  # expected form: attn_in + delta
        h = self._apply_delta_gated_update(h, attn_in, h_attn_out, self.gate_attn_p)

        # --------------------------------------------------------------
        # 3) Optional cross-attention stage
        # --------------------------------------------------------------
        if self.use_deep_cond and (cond_emb_context is not None):
            h_norm = self._apply_norm_1d(h, self.norm_cross_attn)
            cross_in = modulate(h_norm, scale=scale_cross, shift=shift_cross)
            h_cross_out = self.deep_cross_attn(cross_in, context=cond_emb_context)  # expected form: cross_in + delta
            h = self._apply_delta_gated_update(h, cross_in, h_cross_out, self.gate_cross_p)

        # --------------------------------------------------------------
        # 4) MLP stage
        # --------------------------------------------------------------
        # Here the MLP output is added directly as a residual branch.
        h_norm = self._apply_norm_1d(h, self.norm_mlp)
        h_mod = modulate(h_norm, scale=scale_mlp, shift=shift_mlp)
        h_mlp = self.mlp(h_mod)
        h = h + self._gate(self.gate_mlp_p) * h_mlp

        return h


class BlockGroup_v6(nn.Module):
    """
    Stack of CuTBlock_v6 blocks at one U-Net resolution level.

    Purpose
    -------
    This module applies several CuTBlock_v6 blocks in sequence while keeping
    the same conditioning inputs for every block in the group.

    Parameters
    ----------
    depth : int
        Number of CuTBlock_v6 blocks in the group.
    in_channels : int
        Input channel count of the first block.
    out_channels : int
        Output channel count of every block in the group.
    **cut_block_kwargs :
        Additional keyword arguments passed to each CuTBlock_v6.
    """
    def __init__(
        self,
        depth: int,
        in_channels: int,
        out_channels: int,
        **cut_block_kwargs,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        # The first block may change the channel count.
        # All later blocks keep the same output width.
        self.blocks = nn.ModuleList([
            CuTBlock_v6(
                in_channels if i == 0 else out_channels,
                out_channels,
                **cut_block_kwargs,
            )
            for i in range(depth)
        ])

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        cond_emb_context=None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, L)
            Input feature map.
        emb : (B, emb_channels)
            Global conditioning embedding shared across all blocks.
        cond_emb_context : optional
            Condition-token context used by the optional cross-attention stage.

        Returns
        -------
        x : (B, out_channels, L)
            Refined feature map after all blocks in the group.
        """
        for blk in self.blocks:
            x = blk(x, emb, cond_emb_context)
        return x

# ============================================================================
# Fusion / tokenization
# ============================================================================
class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        a, b = self.proj(x).chunk(2, dim=-1)
        return a * F.gelu(b)


class FusionBlock(nn.Module):
    """
    Cross-attn fusion block: query(=noise) attends to tokens(=cond), then GEGLU FFN.
    """
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            GEGLU(dim, dim * 4),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, q, kv):
        # q: (B,1,D), kv: (B,S,D)
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        attn_out, _ = self.attn(qn, kvn, kvn)
        q = q + attn_out
        q = q + self.ff(self.norm_ff(q))
        return q


class SequenceTokenizer1D(nn.Module):
    """
    Convert a 1D sequence of shape (B, L) into S learned tokens of shape
    (B, S, ctx_dim) using a small convolutional encoder followed by
    adaptive average pooling.

    This is more expressive than collapsing the whole sequence into a
    single token, and is therefore better suited to deep cross-attention.
    """
    def __init__(
        self,
        ctx_dim: int,
        token_count: int = 8,
        hidden: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_count = int(token_count)

        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden, ctx_dim, kernel_size=3, padding=1),
            nn.Dropout(dropout),
        )

    def forward(self, sequence_raw: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sequence_raw : torch.Tensor
            Input 1D sequence of shape (B, L).

        Returns
        -------
        torch.Tensor
            Token tensor of shape (B, S, ctx_dim), where S = token_count.
        """
        x = sequence_raw.unsqueeze(1)                    # (B, 1, L)
        x = self.net(x)                                 # (B, ctx_dim, L)
        x = F.adaptive_avg_pool1d(x, self.token_count)  # (B, ctx_dim, S)
        return x.transpose(1, 2)                        # (B, S, ctx_dim)



class DenseTokenProjector(nn.Module):
    """
    Project a dense embedding of shape (B, D) into multiple learned tokens
    of shape (B, S, ctx_dim).

    This is more expressive than collapsing a rich embedding into a single
    token, and is useful when the input branch contains semantically distinct
    information that should remain visible to cross-attention.
    """
    def __init__(
        self,
        in_dim: int,
        ctx_dim: int,
        num_tokens: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.ctx_dim = int(ctx_dim)

        hidden_dim = max(ctx_dim, in_dim // 2)

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_tokens * self.ctx_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        tokens = self.net(x)  # (B, S * ctx_dim)
        return tokens.view(x.shape[0], self.num_tokens, self.ctx_dim)  # (B, S, ctx_dim)


class CurveUNetConditional_v6(nn.Module):
    """
    Adaptive conditional 1D U-Net for corrosion polarization curve generation.

    Main updates in this version
    ----------------------------
    - split the original 2304-dimensional text branch into three explicit
        768-dimensional branches
        1) process route
        2) polarization test condition
        3) microstructure description
    - each rich text branch gets its own token projector
    - the rest of the U-Net, voltage tokenizer, voltage-map injection,
        fusion blocks, and CuTBlock backbone remain unchanged
    """
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        model_channels=64,
        channel_mult=(1, 2, 4, 4),
        depths_per_level=(2, 3, 4, 3),
        attn_head_dim=64,
        mlp_ratio=4.0,
        dropout=0.1,
        emb_dim=None,

        # -------- split conditioning dims --------
        process_emb_dim=768,
        test_cond_emb_dim=768,
        micro_emb_dim=768,
        ele_emb_dim=768,
        voltage_emb_dim=256,

        # -------- token counts for dense branches --------
        process_token_count: int = 1,
        test_cond_token_count: int = 1,
        micro_token_count: int = 1,
        ele_token_count: int = 1,

        # token/context dims
        deep_cond_context_dim=None,
        cond_fusion_heads=4,
        use_deep_cond_cross_attn=True,

        # EDM compat
        sigma_data=1.0,
        kernel_size=3,

        # voltage tokens
        voltage_token_count: int = 16,
        voltage_fourier_features: int = 8,
        gate_max: float = 3.0,

        # adaptive regime modulation
        regime_hidden_ratio: float = 2.0,
    ):
        super().__init__()
        if len(channel_mult) != len(depths_per_level):
            raise ValueError("channel_mult and depths_per_level must have the same length.")

        padding = (kernel_size - 1) // 2

        # ------------------------------------------------------------------
        # Core settings
        # ------------------------------------------------------------------
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sigma_data = float(sigma_data)
        self.kernel_size = kernel_size
        self.attn_head_dim = attn_head_dim
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.gate_max = gate_max

        if emb_dim is None:
            emb_dim = model_channels * 4
        self.emb_dim = emb_dim

        if deep_cond_context_dim is None:
            deep_cond_context_dim = emb_dim
        self.deep_cond_context_dim = deep_cond_context_dim
        self.use_deep_cond_cross_attn = use_deep_cond_cross_attn

        # ------------------------------------------------------------------
        # Condition layout
        # [process | test_condition | microstructure | ele | voltage]
        # ------------------------------------------------------------------
        self.process_emb_dim = process_emb_dim
        self.test_cond_emb_dim = test_cond_emb_dim
        self.micro_emb_dim = micro_emb_dim
        self.ele_emb_dim = ele_emb_dim
        self.voltage_emb_dim = voltage_emb_dim

        # ------------------------------------------------------------------
        # Noise embedding
        # ------------------------------------------------------------------
        self.noise_pos_emb = SinusoidalPosEmb(emb_dim)
        self.noise_mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # ------------------------------------------------------------------
        # Condition encoders
        # ------------------------------------------------------------------
        ctx_dim = self.deep_cond_context_dim

        # Separate token projectors for the three SteelBERT text branches
        self.process_token = DenseTokenProjector(
            in_dim=process_emb_dim,
            ctx_dim=ctx_dim,
            num_tokens=process_token_count,
            dropout=dropout,
        )
        self.test_cond_token = DenseTokenProjector(
            in_dim=test_cond_emb_dim,
            ctx_dim=ctx_dim,
            num_tokens=test_cond_token_count,
            dropout=dropout,
        )
        self.micro_token = DenseTokenProjector(
            in_dim=micro_emb_dim,
            ctx_dim=ctx_dim,
            num_tokens=micro_token_count,
            dropout=dropout,
        )

        # Electrochemical / auxiliary branch
        self.ele_token = DenseTokenProjector(
            in_dim=ele_emb_dim,
            ctx_dim=ctx_dim,
            num_tokens=ele_token_count,
            dropout=dropout,
        )

        # Voltage -> multiple tokens
        self.voltage_tokenizer = SequenceTokenizer1D(
            ctx_dim=ctx_dim,
            token_count=voltage_token_count,
            hidden=max(64, model_channels),
            dropout=dropout * 0.5,
        )

        # Project context tokens to emb_dim for fusion if needed
        self.ctx_to_emb = nn.Identity() if ctx_dim == emb_dim else nn.Linear(ctx_dim, emb_dim)

        if emb_dim % cond_fusion_heads != 0:
            raise ValueError(
                f"emb_dim ({emb_dim}) must be divisible by cond_fusion_heads ({cond_fusion_heads})"
            )

        # ------------------------------------------------------------------
        # Fusion blocks
        # ------------------------------------------------------------------
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(emb_dim, cond_fusion_heads, dropout),
            FusionBlock(emb_dim, cond_fusion_heads, dropout),
        ])

        # ------------------------------------------------------------------
        # Voltage coordinate-map injection
        # features = [voltage, position, dV/dpos, d2V/dpos2] + Fourier(position)
        # ------------------------------------------------------------------
        self.voltage_fourier_features = int(voltage_fourier_features)
        inj_in_dim = 4 + 2 * self.voltage_fourier_features
        self.voltage_coord_mlp = nn.Sequential(
            nn.Linear(inj_in_dim, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )
        self.voltage_inject_alpha = nn.Parameter(torch.tensor(0.15))

        # ------------------------------------------------------------------
        # Adaptive electrochemical regime modulation
        # ------------------------------------------------------------------
        regime_hidden = int(max(emb_dim, emb_dim * regime_hidden_ratio))
        self.regime_mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, regime_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(regime_hidden, emb_dim * 2),
        )
        self.regime_scale_max = 2.0

        # ------------------------------------------------------------------
        # U-Net backbone
        # ------------------------------------------------------------------
        self.init_conv = nn.Conv1d(
            in_channels,
            model_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        skips_channels = []
        current_ch = model_channels
        num_resolutions = len(channel_mult)

        # Encoder
        for level, (mult, depth) in enumerate(zip(channel_mult, depths_per_level)):
            out_ch = model_channels * mult
            self.down_blocks.append(self._make_block_group(depth, current_ch, out_ch))
            current_ch = out_ch
            skips_channels.append(current_ch)

            if level < num_resolutions - 1:
                self.down_blocks.append(
                    Downsample1D(current_ch, use_conv=True, kernel_size=kernel_size)
                )

        # Bottleneck
        bottleneck_depth = depths_per_level[-1]
        self.mid_block = self._make_block_group(bottleneck_depth, current_ch, current_ch)

        # Decoder
        for level, (mult, depth) in reversed(list(enumerate(zip(channel_mult, depths_per_level)))):
            out_ch = model_channels * mult

            if level < num_resolutions - 1:
                self.up_blocks.append(
                    Upsample1D(current_ch, use_conv=True, kernel_size=kernel_size)
                )

            in_ch_up = current_ch + skips_channels.pop()
            self.up_blocks.append(self._make_block_group(depth, in_ch_up, out_ch))
            current_ch = out_ch

        # Output head
        self.out_norm = nn.GroupNorm(get_num_groups(current_ch), current_ch)
        self.out_conv = nn.Conv1d(current_ch, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    # ----------------------------------------------------------------------
    # Builders / utilities
    # ----------------------------------------------------------------------
    def _make_block_group(self, depth: int, in_channels: int, out_channels: int) -> nn.Module:
        num_heads = max(1, out_channels // self.attn_head_dim)
        return BlockGroup_v6(
            depth=depth,
            in_channels=in_channels,
            out_channels=out_channels,
            emb_channels=self.emb_dim,
            kernel_size=self.kernel_size,
            num_heads=num_heads,
            head_dim=self.attn_head_dim,
            mlp_ratio=self.mlp_ratio,
            use_deep_cond=self.use_deep_cond_cross_attn,
            cond_context_dim=self.deep_cond_context_dim,
            num_cross_attn_heads=num_heads,
            cross_attn_head_dim=self.attn_head_dim,
            dropout=self.dropout,
            gate_max=self.gate_max,
        )

    def _split_condition(self, condition: torch.Tensor):
        """
        Split fused condition into
        [process | test_condition | microstructure | ele | voltage].
        """
        expected = (
            self.process_emb_dim
            + self.test_cond_emb_dim
            + self.micro_emb_dim
            + self.ele_emb_dim
            + self.voltage_emb_dim
        )
        if condition.shape[1] != expected:
            raise ValueError(f"condition.shape={condition.shape}, expected last dim={expected}")

        return torch.split(
            condition,
            [
                self.process_emb_dim,
                self.test_cond_emb_dim,
                self.micro_emb_dim,
                self.ele_emb_dim,
                self.voltage_emb_dim,
            ],
            dim=1,
        )

    @staticmethod
    def _resize_voltage_sequence(voltage_raw: torch.Tensor, target_len: int) -> torch.Tensor:
        voltage_x = voltage_raw.clamp(-1.0, 1.0)
        if voltage_x.shape[1] != target_len:
            voltage_x = F.interpolate(
                voltage_x.unsqueeze(1),
                size=target_len,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
        return voltage_x

    def _build_condition_tokens(
        self,
        process_raw: torch.Tensor,
        test_cond_raw: torch.Tensor,
        micro_raw: torch.Tensor,
        ele_raw: torch.Tensor,
        voltage_x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build condition tokens:
            process-route tokens
            polarization-test-condition tokens
            microstructure-description tokens
            electrochemical/auxiliary tokens
            voltage tokens
        """
        process_tok = self.process_token(process_raw)          # (B,S_p,ctx_dim)
        test_cond_tok = self.test_cond_token(test_cond_raw)   # (B,S_t,ctx_dim)
        micro_tok = self.micro_token(micro_raw)               # (B,S_m,ctx_dim)
        ele_tok = self.ele_token(ele_raw)                     # (B,S_e,ctx_dim)
        voltage_tokens = self.voltage_tokenizer(voltage_x)    # (B,S_v,ctx_dim)

        return torch.cat(
            [process_tok, test_cond_tok, micro_tok, ele_tok, voltage_tokens],
            dim=1,
        )

    def _fuse_condition(self, noise_emb: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        kv = self.ctx_to_emb(cond_tokens)   # (B,S,emb_dim)
        q = noise_emb.unsqueeze(1)          # (B,1,emb_dim)
        for blk in self.fusion_blocks:
            q = blk(q, kv)
        return q.squeeze(1)

    def _apply_regime_modulation(self, emb: torch.Tensor) -> torch.Tensor:
        regime_scale_shift = self.regime_mlp(emb)
        regime_scale, regime_shift = regime_scale_shift.chunk(2, dim=1)
        regime_scale = torch.tanh(regime_scale) * self.regime_scale_max
        return emb * (1.0 + regime_scale) + regime_shift

    def _build_voltage_map(self, voltage_x: torch.Tensor) -> torch.Tensor:
        B, L = voltage_x.shape
        device = voltage_x.device

        pos = torch.linspace(-1.0, 1.0, L, device=device).unsqueeze(0).expand(B, L)
        pos_feats = self._fourier_pos(pos)

        v_feat, dv_feat, d2v_feat = self._voltage_geometry_features(voltage_x)
        inj_in = torch.cat(
            [v_feat, pos.unsqueeze(-1), dv_feat, d2v_feat, pos_feats[..., 1:]],
            dim=-1,
        )

        return self.voltage_coord_mlp(inj_in).permute(0, 2, 1)


    def _fourier_pos(self, pos: torch.Tensor) -> torch.Tensor:
        if self.voltage_fourier_features <= 0:
            return pos.unsqueeze(-1)

        feats = [pos.unsqueeze(-1)]
        for k in range(self.voltage_fourier_features):
            freq = (2.0 ** k) * math.pi
            feats.append(torch.sin(freq * pos).unsqueeze(-1))
            feats.append(torch.cos(freq * pos).unsqueeze(-1))
        return torch.cat(feats, dim=-1)

    def _voltage_geometry_features(
        self,
        voltage_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v = voltage_x

        dv = torch.zeros_like(v)
        if v.shape[1] > 1:
            dv[:, 1:] = v[:, 1:] - v[:, :-1]
            dv[:, 0] = dv[:, 1]

        d2v = torch.zeros_like(v)
        if v.shape[1] > 2:
            d2v[:, 1:-1] = dv[:, 2:] - dv[:, 1:-1]
            d2v[:, 0] = d2v[:, 1]
            d2v[:, -1] = d2v[:, -2]

        return v.unsqueeze(-1), dv.unsqueeze(-1), d2v.unsqueeze(-1)

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(self, x, c_noise, condition):
        input_dim = x.dim()
        if input_dim == 2:
            x = x.unsqueeze(1)
        elif input_dim != 3:
            raise ValueError(f"Input must be 2D or 3D, got {x.shape}")

        B, C, L = x.shape
        device = x.device

        # 1) noise embedding
        c_noise = c_noise.to(device).float().view(-1)
        noise_emb = self.noise_mlp(self.noise_pos_emb(c_noise))  # (B, emb_dim)

        # 2) split and preprocess condition
        condition = condition.to(device).float()
        process_raw, test_cond_raw, micro_raw, ele_raw, voltage_raw = self._split_condition(condition)
        voltage_x = self._resize_voltage_sequence(voltage_raw, L)

        # 3) condition tokens and modulation embedding
        cond_tokens = self._build_condition_tokens(
            process_raw,
            test_cond_raw,
            micro_raw,
            ele_raw,
            voltage_x
        )
        cond_context = cond_tokens if self.use_deep_cond_cross_attn else None

        emb_for_modulation = self._fuse_condition(noise_emb, cond_tokens)
        emb_for_modulation = self._apply_regime_modulation(emb_for_modulation)

        # 4) voltage coordinate injection map
        voltage_map = self._build_voltage_map(voltage_x)

        # 5) U-Net backbone
        h = self.init_conv(x)
        h = h + torch.tanh(self.voltage_inject_alpha) * voltage_map

        skips = []

        # Encoder
        for blk in self.down_blocks:
            if isinstance(blk, BlockGroup_v6):
                h = blk(h, emb_for_modulation, cond_context)
                skips.append(h)
            else:
                h = blk(h)

        # Bottleneck
        h = self.mid_block(h, emb_for_modulation, cond_context)

        # Decoder
        for blk in self.up_blocks:
            if isinstance(blk, BlockGroup_v6):
                h = torch.cat([h, skips.pop()], dim=1)
                h = blk(h, emb_for_modulation, cond_context)
            else:
                h = blk(h)

        # 6) output head
        h = self.out_norm(h)
        h = F.silu(h)
        out = self.out_conv(h)

        if input_dim == 2:
            out = out.squeeze(1)
        return out



if __name__ == '__main__':
    # Test parameters
    seq_len = 256
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n--- Test Case: Refactored UNet with CuTBlocks and Flexible Depth ---")
