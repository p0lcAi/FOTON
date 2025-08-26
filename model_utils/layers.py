"""
layers.py

Building blocks used by FOTON models.

Exports:
- generate_nonlinear_layer(...): Linear + (init) + optional ConsistentDropout + activation + optional Normalization.
- ConsistentDropout(p): Dropout with a per-batch mask kept consistent across a forward pass.
- Normalization(eps): L2-normalize inputs with numerical stability.
- RandomFeedback: Random feedback matrices with optional weight mirroring and normalization.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.linalg as LA


# -----------------------------
# Factory for linear + extras
# -----------------------------

def generate_nonlinear_layer(
    in_features: int,
    out_features: int,
    bias: bool = False,
    activation_f: Optional[Union[str, nn.Module, type]] = None,
    layers_init: Optional[Union[str, Callable[[nn.Module], None]]] = None,
    p: Optional[float] = None,             # dropout prob (for ConsistentDropout)
    in_norm: bool = False,                 # append Normalization at the end
) -> nn.Sequential:
    """
    Create a linear layer followed by optional initialization, ConsistentDropout,
    activation, and optional input normalization.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to add bias in the linear layer.
        activation_f: One of:
            - str: 'ReLU'|'Tanh'|'Sigmoid' (case-insensitive)
            - nn.Module subclass (class) or instance
        layers_init: One of:
            - 'orthogonal'|'xavier_uniform'|'he_uniform'
            - callable(container_module) for custom init
        p: Dropout probability for ConsistentDropout (None/0.0 to disable).
        in_norm: If True, append a Normalization() module.

    Returns:
        nn.Sequential container.
    """
    lin = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
    container = nn.Sequential(lin)

    # --- Initialization ---
    if layers_init:
        if isinstance(layers_init, str):
            name = layers_init.lower()
            if name == "orthogonal":
                initialize_orthogonal_weight(lin.weight)
                if bias:
                    init.zeros_(lin.bias)
            elif name == "xavier_uniform":
                init.xavier_uniform_(lin.weight)
                if bias:
                    init.zeros_(lin.bias)
            elif name == "he_uniform":
                init.kaiming_uniform_(lin.weight, nonlinearity="relu")
                if bias:
                    init.zeros_(lin.bias)
            else:
                raise ValueError(f"Unknown layers_init '{layers_init}'")
        elif callable(layers_init):
            layers_init(container)
        else:
            raise ValueError("layers_init must be a str or a callable.")

    # --- Consistent Dropout ---
    if p is not None and p > 0.0:
        container = nn.Sequential(*list(container), ConsistentDropout(p=float(p)))

    # --- Activation ---
    if activation_f is not None:
        if isinstance(activation_f, str):
            act_name = activation_f.lower()
            if act_name == "relu":
                act = nn.ReLU()
            elif act_name == "tanh":
                act = nn.Tanh()
            elif act_name == "sigmoid":
                act = nn.Sigmoid()
            else:
                raise ValueError(f"Unsupported activation '{activation_f}'")
        elif isinstance(activation_f, type) and issubclass(activation_f, nn.Module):
            act = activation_f()
        elif isinstance(activation_f, nn.Module):
            act = activation_f
        else:
            raise ValueError("activation_f must be a str, nn.Module subclass, or nn.Module instance.")
        container = nn.Sequential(*list(container), act)

    # --- Optional normalization ---
    if in_norm:
        container = nn.Sequential(*list(container), Normalization())

    return container


# -----------------------------
# Consistent Dropout
# -----------------------------

class ConsistentDropout(nn.Module):
    """
    Dropout variant that samples a mask once per forward call and reuses it
    consistently across the module stack that shares this instance.

    The mask is invalidated automatically if input shape changes.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability p must be in [0,1), got {p}.")
        self.p = float(p)
        self._mask: Optional[torch.Tensor] = None
        self._mask_shape: Optional[torch.Size] = None

    def _create_mask(self, x: torch.Tensor) -> torch.Tensor:
        # Sample Bernoulli(1-p) and rescale by 1/(1-p) (inverted dropout)
        keep_prob = 1.0 - self.p
        mask = torch.bernoulli(torch.full_like(x, keep_prob))
        mask = mask / keep_prob
        return mask

    def reset_mask(self) -> None:
        self._mask = None
        self._mask_shape = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        if self._mask is None or self._mask_shape != x.shape:
            self._mask = self._create_mask(x).to(device=x.device, dtype=x.dtype)
            self._mask_shape = x.shape
        return x * self._mask


# -----------------------------
# Normalization
# -----------------------------

class Normalization(nn.Module):
    """
    Normalize each sample to unit L2 norm across all features.
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = float(eps)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Flatten per-sample to compute norm across features, then reshape
        orig_shape = X.shape
        Xf = X.view(X.shape[0], -1)
        norms = LA.norm(Xf, dim=1, keepdim=True).clamp_min(self.eps)
        Xn = Xf / norms
        return Xn.view(orig_shape)


# -----------------------------
# Random Feedback
# -----------------------------

class RandomFeedback(nn.Module):
    """
    Stack of random feedback matrices F_l used to map output errors
    back to earlier layers via matrix products.

    The effective feedback matrix is:
        F = F_0 @ F_1^T @ F_2^T @ ...   (depending on forward order and usage)
    Here we maintain a ParameterList `Fs` and provide helpers to compose them.

    Args:
        layers_dim: List of layer sizes [d0, d1, ..., dL].
        layers: Sequence of forward layers (used by weight_mirroring).
        F_init: 'uniform' or 'normal' (distribution for random init).
        F_mean_zero: If True and uniform, center around 0.
        F_std: Global scale factor for initialization.
    """

    def __init__(
        self,
        layers_dim: List[int],
        layers: Iterable[nn.Module],
        F_init: str = "uniform",
        F_mean_zero: bool = True,
        F_std: float = 0.005,
    ):
        super().__init__()

        if len(layers_dim) < 2:
            raise ValueError("layers_dim must contain at least input and output dimensions.")
        self.layers_dim = list(layers_dim)
        self.layers = list(layers)
        self.F_init = F_init
        self.F_mean_zero = bool(F_mean_zero)
        self.F_std = float(F_std)

        self.nb_layers = len(self.layers)
        self.in_dim = self.layers_dim[0]
        self.Fs = nn.ParameterList()

        if self.F_init.lower() == "uniform":
            # Base scale inspired by fan-in of input layer; then scaled by F_std
            sd = float(np.sqrt(6.0 / self.layers_dim[0]))
            for d_in, d_out in zip(self.layers_dim, self.layers_dim[1:]):
                if self.F_mean_zero:
                    base = (torch.rand(d_in, d_out) * 2 * sd - sd)
                else:
                    base = torch.rand(d_in, d_out) * sd
                F = nn.Parameter(base * self.F_std)
                self.Fs.append(F)

        elif self.F_init.lower() == "normal":
            # Base std inspired by He-like scaling; then modulated by F_std and depth
            base_std = float(np.sqrt(2.0 / self.layers_dim[0])) * self.F_std
            n = len(self.layers_dim) - 1
            el = int(np.prod(self.layers_dim[1:-1])) if len(self.layers_dim) > 2 else 1
            for d_in, d_out in zip(self.layers_dim, self.layers_dim[1:]):
                std = (base_std / (el ** 0.5)) ** (1.0 / max(1, n))
                F = nn.Parameter(torch.empty(d_in, d_out).normal_(mean=0.0, std=std))
                self.Fs.append(F)
        else:
            raise ValueError(f"Unknown F_init '{self.F_init}' (expected 'uniform' or 'normal').")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Propagate through Fs in reverse (matching many FA-style compositions)
        out = input
        for F in reversed(self.Fs):
            out = out @ F.T
        return out

    def get_F(self) -> torch.Tensor:
        """Compose all F matrices into a single matrix product."""
        F = self.Fs[0]
        for f in self.Fs[1:]:
            F = F @ f.T
        return F

    @torch.no_grad()
    def weight_mirroring(self, WM_batch_size: int, noise_amplitude: float = 0.1) -> None:
        """
        Approximate weight mirroring: correlate random inputs with layer outputs
        to update feedback matrices.

        Args:
            WM_batch_size: Batch size for the random probe.
            noise_amplitude: Amplitude of random input perturbations.
        """
        device = next(self.layers[0].parameters()).device if self.layers else "cpu"

        for l in range(self.nb_layers - 1):
            d_in = self.layers_dim[l]
            noise_x = noise_amplitude * torch.randn(WM_batch_size, d_in, device=device)
            noise_y = self.layers[l](noise_x)  # assumes callable layer
            update = (noise_x.T @ noise_y) / WM_batch_size
            self.Fs[l] -= update

    @torch.no_grad()
    def normalize_F(self) -> None:
        """
        Normalize feedback matrices so the composed matrix has target std
        (based on He-like scaling with input dim), distributed evenly across Fs.
        """
        target_std = float(np.sqrt(2.0 / self.in_dim)) * self.F_std
        current = torch.std(self.get_F())
        if current > 0:
            # Distribute scaling evenly across all Fs
            scale = (target_std / current).pow(1.0 / max(1, len(self.Fs)))
            for l in range(len(self.Fs)):
                self.Fs[l].data.mul_(scale)


# -----------------------------
# Internal deps expected by this module
# -----------------------------

def initialize_orthogonal_weight(w: torch.Tensor) -> None:
    """Orthogonal init convenience wrapper (kept here for self-containment)."""
    nn.init.orthogonal_(w)
