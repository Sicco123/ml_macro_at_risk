"""Factor Neural Network model implementation."""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Literal
import logging

logger = logging.getLogger(__name__)

ActivationType = Literal["relu", "tanh", "gelu"]


def _make_activation(name: ActivationType) -> nn.Module:
    if name == "relu":
        return nn.ReLU()         # keep out-of-place for vmap-compat
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")

class FastNN(nn.Module):
    """
    Single shared trunk MLP + one final Linear that emits all (quantile,horizon) outputs at once.
    Output shape: (B, n_quantiles, n_horizons)
    """
    def __init__(
        self,
        input_dim: int,
        quantiles: List[float],
        forecast_horizons: List[int],
        units_per_layer: List[int],
        activation: ActivationType = "relu",
    ):
        super().__init__()
        self.quantiles = list(quantiles)
        self.horizons = list(forecast_horizons)
        self.nq = len(self.quantiles)
        self.nh = len(self.horizons)

        layers: List[nn.Module] = []
        prev = input_dim
        for u in units_per_layer:
            layers.append(nn.Linear(prev, u))
            layers.append(_make_activation(activation))
            prev = u

        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(prev, self.nq * self.nh)  # single big head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, D) -> (B, nq*nh) -> (B, nq, nh)
        z = self.trunk(x)
        out = self.head(z)
        return out.view(x.size(0), self.nq, self.nh)

    def get_last_layer_features(self, x: torch.Tensor) -> torch.Tensor:
        # Features right before the final head (like your original helper)
        return self.trunk(x)

    def get_model_size(self) -> int:
        return sum(p.numel() for p in self.parameters())

class EnsembleNN(nn.Module):
    """Ensemble of Neural Networks using vmap for efficient computation."""
    
    def __init__(
        self,
        input_dim: int,
        quantiles: List[float],
        forecast_horizons: List[int],
        units_per_layer: List[int],
        n_models: int = 5,
        activation: ActivationType = "relu"
    ):
        """Initialize Ensemble Factor Neural Network.
        
        Args:
            input_dim: Input feature dimension
            quantiles: List of quantile levels
            forecast_horizons: List of forecast horizons
            units_per_layer: Number of units in each hidden layer
            n_models: Number of models in ensemble
            activation: Activation function type
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.quantiles = quantiles
        self.forecast_horizons = forecast_horizons
        self.units_per_layer = units_per_layer
        self.n_models = n_models
        self.activation = activation
        
        # Create ensemble of models
        self.models = nn.ModuleList([
            FastNN(input_dim, quantiles, forecast_horizons, units_per_layer, activation)
            for _ in range(n_models)
        ])
        

        logger.info(f"Created EnsembleFactorNN with {n_models} models")
    

    def forward(self, x: torch.Tensor, return_ensemble: bool = True, per_model_inputs: bool = True) -> torch.Tensor:


        # run each model and collect outputs (Python loop is fine; avoid cat!)
        if per_model_inputs:
            ensemble = torch.stack([m(x[i]) for i, m in enumerate(self.models)], dim=0)  # (n_models, batch, n_q, n_h)
        else:
            ensemble = torch.stack([m(x) for m in self.models], dim=0)  # (n_models, batch, n_q, n_h)
        return ensemble if return_ensemble else ensemble.mean(dim=0)



