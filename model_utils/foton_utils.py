"""
foton_utils.py

Utility functions for FOTON training:
- Orthogonalization methods (SVD, Björck).
- Adam-style weight updates for forward-only training.
"""

import torch
import torch.nn as nn
from typing import Any, Mapping

from conv_ortho_utils.layers_utils import bjorck_orthonormalize


# -----------------------------
# Orthogonalization
# -----------------------------

@torch.no_grad()
def svd_orthogonalize(layers, nb_layers: int, layers_dim: list[int]) -> None:
    """Apply SVD-based orthogonalization to all linear layers."""
    for i in range(nb_layers - 1):
        W = layers[i][0].weight
        U, _, Vh = torch.linalg.svd(W, full_matrices=False)
        I = torch.eye(layers_dim[i + 1], layers_dim[i], device=W.device, dtype=W.dtype)
        layers[i][0].weight = nn.Parameter(U @ I @ Vh)


@torch.no_grad()
def bjork_orthogonalize_all(layers, nb_layers: int) -> None:
    """Apply Björck orthogonalization to all linear layers."""
    for i in range(nb_layers - 1):
        W = layers[i][0].weight.data
        layers[i][0].weight = nn.Parameter(
            bjorck_orthonormalize(w=W, iters=20, order=1, power_iteration_scaling=True)
        )


# -----------------------------
# Adam-like update
# -----------------------------

def adam_update(
    layer_idx: int,
    grad: torch.Tensor,
    beta1: float,
    beta2: float,
    eps: float,
    m: torch.nn.ParameterList,
    v: torch.nn.ParameterList,
    t: int,
):
    """Perform an Adam-style update step for a single layer.

    Args:
        layer_idx: Index of the layer being updated.
        grad: Gradient tensor.
        beta1, beta2: Adam hyperparameters.
        eps: Numerical stability term.
        m, v: ParameterLists tracking 1st and 2nd moments.
        t: Current timestep (1-based).

    Returns:
        Tuple (update, m, v) where update is the bias-corrected Adam update.
    """
    m[layer_idx] = beta1 * m[layer_idx] + (1 - beta1) * grad
    v[layer_idx] = beta2 * v[layer_idx] + (1 - beta2) * grad.pow(2)

    m_hat = m[layer_idx] / (1 - beta1**t)
    v_hat = v[layer_idx] / (1 - beta2**t)

    update = m_hat / (torch.sqrt(v_hat) + eps)
    return update, m, v
