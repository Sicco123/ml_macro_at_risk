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
            layers.append(nn.BatchNorm1d(u))
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
    
class AR_per_country(nn.Module):
    def __init__(self,
                 intercepts, 
                 phis,
                 quantiles, 
                 horizons
                 ):
        super().__init__()
        self.intercepts = intercepts
        self.phis = phis

        phi_average = phis.mean()

        self.quantiles = quantiles
        self.horizons = horizons

        self.phi_tensors = nn.Parameter(torch.tensor([phi_average], dtype=torch.float32), requires_grad=True)
        self.intercept_tensors = nn.Parameter(torch.tensor(intercepts, dtype=torch.float32), requires_grad=True)
        
        
   
    def forward(self, x: torch.Tensor, country_codes: torch.Tensor) -> torch.Tensor:
        
        # Extract country indices once
        country_idx = country_codes[:, 0, 0]  # Shape: (batch_size,)
        
        x_expanded = x.squeeze(-1).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1)
        
        output = (self.intercept_tensors[country_idx] + 
                self.phi_tensors[0] * x_expanded)
        
        return output

class EnsembleNN(nn.Module):
    """Ensemble of Neural Networks using vmap for efficient computation."""
    
    def __init__(
        self,
        input_dim: int,
        quantiles: List[float],
        forecast_horizons: List[int],
        units_per_layer: List[int],
        n_models: int = 5,
        activation: ActivationType = "relu",
        intercepts_init = None,
        phis_init =  None,
        turn_on_neural_net: bool = True
    ):
        """Initialize Ensemble Factor Neural Network.
        
        Args:
            input_dim: Input feature dimension
            quantiles: List of quantile levels
            forecast_horizons: List of forecast horizons
            units_per_layer: Number of units in each hidden layer
            n_models: Number of models in ensemble
            activation: Activation function type
            intercepts_init: Initial intercepts for each model
            phis_init: Initial AR coefficients for each model
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.quantiles = quantiles
        self.forecast_horizons = forecast_horizons
        self.units_per_layer = units_per_layer
        self.n_models = n_models
        self.activation = activation
        self.intercepts_init = intercepts_init
        self.phis_init = phis_init

        # Create ensemble of models
        self.models = nn.ModuleList([
            FastNN(input_dim, quantiles, forecast_horizons, units_per_layer, activation)
            for _ in range(n_models)
        ])
        

        # compile models
        #self.models.compile()
           

        self.ar_models = nn.ModuleList([
            AR_per_country(intercepts_init, phis_init, quantiles, forecast_horizons)
            for _ in range(n_models)
        ])

        #self.ar_models.compile()

        self.turn_on_neural_net = turn_on_neural_net
        #logger.info(f"Created EnsembleFactorNN with {n_models} models")
    

    def forward(self, x: torch.Tensor, country_codes: torch.Tensor, return_ensemble: bool = True, per_model_inputs: bool = True) -> torch.Tensor:

        if per_model_inputs:
            # x[i] is (batch, features), need to extract first feature for AR
            if self.turn_on_neural_net:
                ensemble = torch.stack([
                    m(x[i,:,:]) + ar_m(x[i, :, 0:1], country_codes[i]) 
                    for i, (m, ar_m) in enumerate(zip(self.models, self.ar_models))
                ], dim=0)
            else:
                ensemble = torch.stack([
                    ar_m(x[i, :, 0:1], country_codes[i]) 
                    for i, ar_m in enumerate(self.ar_models)
                ], dim=0)
        else:
            if self.turn_on_neural_net:
                # x is (batch, features), extract first feature for AR  
                ensemble = torch.stack([
                    m(x[:,:]) + ar_m(x[:, 0:1], country_codes) 
                    for m, ar_m in zip(self.models, self.ar_models)
                ], dim=0)
            else:
                ensemble = torch.stack([
                    ar_m(x[:, 0:1], country_codes) 
                    for ar_m in self.ar_models
                ], dim=0)
        
        return ensemble if return_ensemble else ensemble.mean(dim=0)


