"""
initializations.py

Utilities for FOTON:
- create_F_T(layers, nb_layers): build the (transposed) projection matrix used in modulated passes.
- initialize_orthogonal_weight(layer_or_tensor): orthogonal init for tensors or modules (bias -> 0).
- RAVI_XAVIER_init(layer, variant): Ravi/PEPITA-style layer initialization (he_uniform / he_normal).
"""

from __future__ import annotations

from typing import Sequence, Union
import math

import torch
import torch.nn as nn


def create_F_T(layers: Sequence[nn.Sequential], nb_layers: int) -> nn.Parameter:
    """
    Construct the projection matrix directly in transposed form (as used in the
    batch computations). This mirrors the behavior:

        F_T = W_{L-1} @ W_{L-2} @ ... @ W_0

    where each W_i is taken from layers[i][0].weight (i.e., the first module
    inside the Sequential is the linear layer).

    Args:
        layers: Sequence of layer containers (nn.Sequential) containing a linear at index 0.
        nb_layers: Total number of layers (len(layers_dim) in your models).

    Returns:
        nn.Parameter with requires_grad=False containing the product above.

    Notes:
        - Keeps the original multiplication order to avoid changing behavior.
        - Returned as a Parameter so you can assign it directly to model attributes.
          We mark it non-trainable so optimizers ignore it.
    """
    if nb_layers < 2:
        raise ValueError(f"nb_layers must be >= 2, got {nb_layers}")

    # Start from first linear weight
    W0 = layers[0][0].weight
    F_T = W0.clone()

    # Multiply remaining linear weights
    for i in range(1, nb_layers - 1):
        Wi = layers[i][0].weight
        if Wi.dim() != 2 or F_T.dim() != 2:
            raise ValueError(
                f"create_F_T expects linear weights; got shapes {tuple(Wi.shape)} and {tuple(F_T.shape)}"
            )
        if Wi.shape[1] != F_T.shape[0]:
            raise ValueError(
                f"Shape mismatch multiplying W[{i}] @ F_T: {tuple(Wi.shape)} x {tuple(F_T.shape)}"
            )
        F_T = Wi @ F_T

    # Non-trainable parameter (refreshed manually during training)
    return nn.Parameter(F_T, requires_grad=False)


# --------------------------------------------------------------------
# Layer initializations
# --------------------------------------------------------------------

def initialize_orthogonal_weight(layer_or_tensor: Union[nn.Module, torch.Tensor]) -> None:
    """
    Orthogonal initialization helper.

    - If a torch.Tensor is provided, applies nn.init.orthogonal_(tensor).
    - If a module with attribute 'weight' is provided (e.g., nn.Linear),
      initializes the weight orthogonally and zeroes the bias if present.

    Args:
        layer_or_tensor: A tensor or a module with a 'weight' attribute.

    Raises:
        ValueError: If the object has no 'weight' and is not a Tensor.
    """
    if isinstance(layer_or_tensor, torch.Tensor):
        nn.init.orthogonal_(layer_or_tensor)
        return

    if hasattr(layer_or_tensor, "weight"):
        nn.init.orthogonal_(layer_or_tensor.weight)
        if hasattr(layer_or_tensor, "bias") and layer_or_tensor.bias is not None:
            nn.init.zeros_(layer_or_tensor.bias)
        return

    raise ValueError(
        f"Cannot initialize orthogonal weight on object of type {type(layer_or_tensor)}; "
        "expects a Tensor or a module with attribute 'weight'."
    )


# --------------------------------------------------------------------
# Ravi / PEPITA-style init
# --------------------------------------------------------------------

def RAVI_XAVIER_init(layer: Union[nn.Linear, nn.Sequential], variant: str = "he_uniform") -> None:
    """
    Ravi/PEPITA-style initialization, slightly different from standard Xavier:

    - he_uniform: weight ~ U(-sqrt(6/in_features), +sqrt(6/in_features))
    - he_normal:  weight ~ N(0, sqrt(2/in_features))

    Works with either an nn.Linear or an nn.Sequential whose first module is nn.Linear.

    Args:
        layer: nn.Linear or nn.Sequential containing a Linear at index 0.
        variant: 'he_uniform' (default) or 'he_normal'.
        
    This follows the initialization used in "FORWARD LEARNING WITH TOP-DOWN FEEDBACK: EMPIRICAL AND ANALYTICAL CHARACTERIZATION" (https://openreview.net/pdf?id=My7lkRNnL9)
    """
    # Resolve potential Sequential wrapper
    if isinstance(layer, nn.Sequential):
        if len(layer) == 0 or not hasattr(layer[0], "in_features"):
            raise ValueError("Expected nn.Sequential with a Linear as the first submodule.")
        lin = layer[0]
    else:
        lin = layer
        if not hasattr(lin, "in_features"):
            raise ValueError("RAVI_XAVIER_init expects an nn.Linear or nn.Sequential[Linear, ...].")

    in_size: int = int(lin.in_features)

    if variant.lower() == "he_uniform":
        limit = math.sqrt(6.0 / in_size)
        nn.init.uniform_(lin.weight, a=-limit, b=+limit)
    elif variant.lower() == "he_normal":
        std = math.sqrt(2.0 / in_size)
        nn.init.normal_(lin.weight, mean=0.0, std=std)
    else:
        raise ValueError(f"Unknown variant '{variant}' (expected 'he_uniform' or 'he_normal').")
