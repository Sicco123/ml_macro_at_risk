"""Factor Neural Network model implementation."""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Literal
import logging
from torch.func import stack_module_state, functional_call, vmap
import copy

logger = logging.getLogger(__name__)

ActivationType = Literal["relu", "tanh", "gelu"]


class QuantileHorizonBranch(nn.Module):
    """Individual branch for a specific (quantile, horizon) pair."""
    
    def __init__(
        self,
        input_dim: int,
        units_per_layer: List[int],
        activation: ActivationType = "relu"
    ):
        """Initialize branch.
        
        Args:
            input_dim: Input feature dimension
            units_per_layer: Number of units in each hidden layer
            activation: Activation function type
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.units_per_layer = units_per_layer
        
        # Build layers
        layers = []
        prev_dim = input_dim

        for units in units_per_layer:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(self._get_activation(activation))
            prev_dim = units
        
        # Output layer (single neuron)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)

   

    def _get_activation(self, activation: ActivationType) -> nn.Module:
        """Get activation function."""
        if activation == "relu":
            return nn.ReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

        return self.network(x)
    
    def get_last_layer_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get activations from the last hidden layer."""
        # Forward through all layers except the last one
        for layer in self.network[:-1]:
            x = layer(x)
        return x


class NeuralNetwork(nn.Module):
    """Factor Neural Network with separate branches for each (quantile, horizon) pair."""
    
    def __init__(
        self,
        input_dim: int,
        quantiles: List[float],
        forecast_horizons: List[int],
        units_per_layer: List[int],
        activation: ActivationType = "relu"
    ):
        """Initialize Factor Neural Network.
        
        Args:
            input_dim: Input feature dimension
            quantiles: List of quantile levels
            forecast_horizons: List of forecast horizons
            units_per_layer: Number of units in each hidden layer
            activation: Activation function type
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.quantiles = quantiles
        self.forecast_horizons = forecast_horizons
        self.units_per_layer = units_per_layer
        self.activation = activation
        
        # Create branches for each (quantile, horizon) pair
        self.branches = nn.ModuleDict()
        
        for q in quantiles:
            for h in forecast_horizons:
                branch_key = f"q{q:.3f}_h{h}".replace(".", "_")
                self.branches[branch_key] = QuantileHorizonBranch(
                    input_dim, units_per_layer, activation
                )
        
        logger.info(f"Created FactorNN with {len(self.branches)} branches")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, n_quantiles, n_horizons)
        """
        batch_size = x.size(0)
        n_quantiles = len(self.quantiles)
        n_horizons = len(self.forecast_horizons)
        
        # Initialize output tensor
        outputs = torch.zeros(batch_size, n_quantiles, n_horizons, device=x.device)
        
        # Forward through each branch
        for q_idx, q in enumerate(self.quantiles):
            for h_idx, h in enumerate(self.forecast_horizons):
                branch_key = f"q{q:.3f}_h{h}".replace(".", "_")
                branch_output = self.branches[branch_key](x)
                outputs[:, q_idx, h_idx] = branch_output.squeeze(-1)
        
        return outputs
    
    
    def get_model_size(self) -> int:
        """Get total number of parameters."""
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
            NeuralNetwork(input_dim, quantiles, forecast_horizons, units_per_layer, activation)
            for _ in range(n_models)
        ])
        
        # Create stateless base model for vmap
        self.base_model = copy.deepcopy(self.models[0])
        self.base_model = self.base_model.to('meta')
        
        logger.info(f"Created EnsembleFactorNN with {n_models} models")
    
    def forward(self, x: torch.Tensor, return_ensemble: bool = True) -> torch.Tensor:
        """Forward pass through ensemble.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (n_models, batch_size, input_dim)
            return_ensemble: If True, return all ensemble predictions, else return mean
            
        Returns:
            If return_ensemble=True: (n_models, batch_size, n_quantiles, n_horizons)
            If return_ensemble=False: (batch_size, n_quantiles, n_horizons)
        """
        # Stack model parameters for vmap
        params, buffers = stack_module_state(self.models)
        
        # Define function for vmap
        def fmodel(params, buffers, x):
            return functional_call(self.base_model, (params, buffers), (x,))
        
        # Check if we have different batches for each model
        if x.dim() == 3 and x.size(0) == self.n_models:
            # Different batch for each model (Option 1 from tutorial)
            ensemble_predictions = vmap(fmodel, in_dims=(0, 0, 0))(params, buffers, x)
        else:
            # Same batch for all models (Option 2 from tutorial)
            ensemble_predictions = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, x)
        
        if return_ensemble:
            return ensemble_predictions  # (n_models, batch_size, n_quantiles, n_horizons)
        else:
            return ensemble_predictions.mean(dim=0)  # (batch_size, n_quantiles, n_horizons)
    
    def get_individual_states(self) -> List[Dict]:
        """Get state dicts for individual models."""
        return [model.state_dict() for model in self.models]
    
    def load_individual_states(self, state_dicts: List[Dict]):
        """Load state dicts for individual models."""
        for model, state_dict in zip(self.models, state_dicts):
            model.load_state_dict(state_dict)
