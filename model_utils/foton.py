import torch
import torch.nn as nn

from .foton_utils import *
from .layers import generate_nonlinear_layer, ConsistentDropout
from .initializations import create_F_T, initialize_orthogonal_weight


class FOTON(nn.Module):
    """
    Forward-Only Training with Orthogonalization of Nonlinear networks (FOTON).

    Implements the forward-only training algorithm with modulated forward passes,
    orthogonalization of weights, and optional adaptive optimization.
    """
    def __init__(
        self,
        layers_dim: list,
        bias: bool = False,
        vision: bool = False,
        activation_f: object = None,
        layers_init: object = None,
        update_F: int = 0,            # frequency of F update (every 'update_F' steps; 0 = never)
        ortho_rate: int | None = None,
        p_dropout: float | None = None,
        error: str | None = None,
        ce_T: float = 1.0,
        L2: float = 0.0,
        batch_size: int = 1,
        optimizer: str = "SGD",
        Adam_beta_1_2: tuple = (0.9, 0.999),
        Adam_eps: float = 1e-8,
    ):
        super().__init__() 
        
        self.layers_dim = layers_dim
        self.bias = bias
        self.vision = vision
        self.activation_f = activation_f
        self.layers_init = layers_init
        self.update_F = update_F
        self.ortho_rate = ortho_rate or 1
        self.p_dropout = p_dropout
        self.error = error
        self.ce_T = ce_T
        self.L2 = L2
        self.batch_size = batch_size
        self.optimizer = optimizer.upper()
        self.Adam_beta_1_2 = Adam_beta_1_2
        self.Adam_eps = Adam_eps
        
        self.nb_layers = len(layers_dim)
        self.layers = nn.ModuleList()

        # Hidden layers
        for i in range(self.nb_layers - 2):
            layer = generate_nonlinear_layer(
                in_features=layers_dim[i],
                out_features=layers_dim[i+1],
                bias=bias,
                activation_f=activation_f,
                layers_init=layers_init,
                p=p_dropout,
            )
            self.layers.append(layer)

        # Output layer (linear)
        self.layers.append(nn.Sequential(nn.Linear(layers_dim[-2], layers_dim[-1], bias=bias)))

        # Optimizer state (for Adam)
        if self.optimizer == "ADAM":
            self.m = nn.ParameterList([
                nn.Parameter(torch.zeros_like(layer[0].weight)) for layer in self.layers
            ])
            self.v = nn.ParameterList([
                nn.Parameter(torch.zeros_like(layer[0].weight)) for layer in self.layers
            ])

        if self.vision:
            self.flatten = nn.Flatten()

        # Feedback matrix F
        self.F_T = create_F_T(layers=self.layers, nb_layers=self.nb_layers)
        self.t = 0

        # Activation storage
        self.activations = []
        for i in range(self.nb_layers - 1):
            self.layers[i].register_forward_hook(
                lambda module, input, output: self.activations.append(output)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass (storing activations)."""
        self.activations = []
        if self.vision:
            x = self.flatten(x)
        for i in range(self.nb_layers - 1):
            x = self.layers[i](x)
        return x
    
    @torch.no_grad()
    def modulated_forward(self, x, y, target, lr: float):
        """Perform a forward-only modulated update step."""
        self.t += 1
        updates = []

        target = target.float()
        if self.error == "MSE":
            e = y - target
        elif self.error == "CE":
            e = nn.Softmax(dim=1)(self.ce_T * y) - target
        else:
            raise ValueError(f"Unknown error type: {self.error}")

        if self.vision:
            x = self.flatten(x)

        h = self.activations

        # Modulated input
        x_mod = x - torch.mm(e, self.F_T)
        self.forward(x_mod)
        h_mod = self.activations

        # Update weights
        for i in range(self.nb_layers - 1):
            pre = x if i == 0 else h[i-1]
            if i < self.nb_layers - 2:
                grad = torch.mm((h[i] - h_mod[i]).T, pre) / self.batch_size
            else:
                grad = torch.mm(e.T, h[-2]) / self.batch_size

            updates.append(grad)

            if self.optimizer == "ADAM":
                grad, self.m, self.v = adam_update(
                    i, grad, *self.Adam_beta_1_2, self.Adam_eps, self.m, self.v, self.t
                )

            self.layers[i][0].weight -= lr * (grad + self.L2 * self.layers[i][0].weight)
            if self.bias:
                if i < self.nb_layers - 2:
                    db = (h[i] - h_mod[i]).T / self.batch_size
                else:
                    db = e
                self.layers[i][0].bias -= lr * db

        # Orthogonalization + F update
        if self.ortho_rate and (self.t % self.ortho_rate == 0):
            bjork_orthogonalize_all(self.layers, self.nb_layers)
        if self.update_F > 0 and (self.t % self.update_F == 0):
            self.F_T = create_F_T(layers=self.layers, nb_layers=self.nb_layers)

        self.reset_dropout_masks()
        return updates
    
    def reset_dropout_masks(self) -> None:
        """Reset masks for ConsistentDropout layers."""
        for module in self.modules():
            if isinstance(module, ConsistentDropout):
                module.reset_mask()
