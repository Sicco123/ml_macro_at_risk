"""Training utilities for Factor Neural Networks."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Literal
from tqdm import tqdm
import logging
import copy

from .model import EnsembleNN


logger = logging.getLogger(__name__)

OptimizerType = Literal["adam", "adamw", "sgd"]


class PinballLoss(nn.Module):
    """Pinball loss for quantile regression."""
    
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, ensemble: bool = False) -> torch.Tensor:
        """
        Pinball loss (a.k.a. quantile loss), vectorized.

        Args:
            predictions: (E, B, Q, H) if ensemble=True, else (B, Q, H)
            targets:     (B, H) or (B, Q, H)
        Returns:
            Scalar loss (mean over ensemble, batch, quantiles, horizons)
        """
        E, B, Q, H = predictions.shape

        # Expand targets: (E, B, H) -> (E, B, Q, H)

        # targets = targets.unsqueeze(2).expand(-1, -1, Q, -1)

        # Quantiles shaped to (1,1,Q,1) for broadcasting
        q = torch.as_tensor(self.quantiles, device=predictions.device, dtype=predictions.dtype).view(1, 1, Q, 1)

        # Compute error and pinball loss
        err = targets - predictions                      # (E, B, Q, H)
        loss = torch.where(err >= 0, q * err, (q - 1.0) * err)

        # Mean over all dimensions → scalar
        return loss.mean()                           # scalar


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        
    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop.
        
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


class EnsembleNNTrainer:
    """Trainer for Ensemble Neural Networks."""
    
    def __init__(
        self,
        model: 'EnsembleFactorNN',
        quantiles: List[float],
        device: Optional[torch.device] = None
    ):
        """Initialize trainer.
        
        Args:
            model: Ensemble model to train
            quantiles: List of quantile levels
            device: Device to use for training
        """
        self.model = model
        self.quantiles = quantiles
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.loss_fn = PinballLoss(quantiles)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
        # Store best model states for each ensemble member
        self.best_model_states = [None] * model.n_models

    def _get_optimizer(
        self,
        optimizer_type: OptimizerType,
        learning_rate: float, 
        penalty: float = 0.0
    ) -> optim.Optimizer:
        """Get optimizer."""
        if optimizer_type == "adam":
            return optim.Adam(self.model.parameters(), lr=learning_rate)#, weight_decay=penalty)
        elif optimizer_type == "adamw":
            return optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=penalty)
        elif optimizer_type == "sgd":
            return optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=penalty)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def train_epoch(
        self,
        train_loaders: List[DataLoader],
        optimizer: optim.Optimizer,
        l2: float = 0.0,
        verbose: bool = False
    ) -> float:
        """Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        pbar = zip(*train_loaders)

        for batches in pbar:
            features_list, targets_list = zip(*batches)
            
            # Stack features and targets from different data loaders - different batch for each model
            features = torch.stack([f.to(self.device) for f in features_list])  # (n_models, batch_size, input_dim)
            targets = torch.stack([t.to(self.device) for t in targets_list])    # (n_models, batch_size, n_quantiles, n_horizons)
            
            optimizer.zero_grad()

            # CODE HERE - Use vmap with different batches for each model (Option 1 from tutorial)
            predictions = self.model(features, return_ensemble=True)

            loss_ex_penalty = self.loss_fn(predictions, targets, ensemble=True)

            # Add L2 regularization
            if l2 > 0.0:
                l2_loss = sum(p.pow(2).sum() for p in self.model.parameters())
                loss = loss_ex_penalty + l2 * l2_loss
            else:
                loss = loss_ex_penalty

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    # def validate(self, val_loaders: List[DataLoader]) -> List[float]:
    #     """Validate model.
        
    #     Returns:
    #         List of validation losses for each ensemble member
    #     """
    #     self.model.eval()
    #     total_losses = [0.0] * self.model.n_models
    #     n_batches = 0

    #     pbar = zip(*val_loaders)
        
    #     with torch.no_grad():
    #         for batches in pbar:
    #             features_list, targets_list = zip(*batches)
                
    #             # Stack features and targets - different batch for each model
    #             features = torch.stack([f.to(self.device) for f in features_list])
    #             targets = torch.stack([t.to(self.device) for t in targets_list])
                
    #             # CODE HERE - Use vmap with different batches for each model
    #             predictions = self.model(features, return_ensemble=True)
                
              
    #             # Calculate loss for each ensemble member
    #             for e_idx in range(self.model.n_models):
    #                 pred_e = predictions[e_idx:e_idx+1]  # Keep ensemble dimension
    #                 target_e = targets[e_idx:e_idx+1]    # Use corresponding batch for each model
    #                 loss_e = self.loss_fn(pred_e, target_e, ensemble=True)
    #                 total_losses[e_idx] += loss_e.item()
                
    #             n_batches += 1
        
    #     return [total_loss / n_batches if n_batches > 0 else 0.0 for total_loss in total_losses]

    def validate(self, val_loaders: List[DataLoader]) -> List[float]:
        self.model.eval()
        total_losses = torch.zeros(self.model.n_models, device=self.device)
        n_batches = 0

        with torch.no_grad():
            for batches in zip(*val_loaders):
                features_list, targets_list = zip(*batches)
                features = torch.stack([f.to(self.device) for f in features_list])
                targets  = torch.stack([t.to(self.device) for t in targets_list])

                preds = self.model(features, return_ensemble=True)  # (E,B,Q,H)

                # reuse PinballLoss math but keep per-E means
                q = torch.as_tensor(self.quantiles, device=self.device, dtype=preds.dtype).view(1, 1, -1, 1)  # (1,1,Q,1)
                err  = targets - preds
                loss = torch.where(err >= 0, q * err, (q - 1.0) * err)  # (E,B,Q,H)
                loss_per_e = loss.mean(dim=(1,2,3))                    # (E,)
                total_losses += loss_per_e
                n_batches += 1

        return (total_losses / max(n_batches, 1)).tolist()
    
    def fit(
        self,
        train_loaders: List[DataLoader],
        ensemble_size: int = 1,
        val_loader: Optional[List[DataLoader]] = None,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        optimizer_type: OptimizerType = "adamw",
        patience: int = 10,
        verbose: int = 1, 
        l2: float = 0.0
    ) -> Dict[str, List[float]]:
        """Fit the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Maximum number of epochs
            learning_rate: Learning rate
            optimizer_type: Type of optimizer
            patience: Early stopping patience
            verbose: Verbosity level
            
        Returns:
            Training history
        """
     

        optimizer = self._get_optimizer(optimizer_type, learning_rate)
        early_stopping = EarlyStopping(patience) if val_loader is not None else None
        
        best_val_losses = np.array([np.inf] * self.model.n_models)
        
        # Initialize train losses with empty lists
        self.train_losses = []
        self.val_losses = []

        # Add tqdm 
        pbar = tqdm(range(epochs), desc="Training", disable=verbose < 1)
        for epoch in pbar:
            # Training
            train_loss = self.train_epoch(train_loaders, optimizer, l2, verbose >= 2)
            
            self.train_losses.append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_losses_epoch = self.validate(val_loader)
                self.val_losses.append(val_losses_epoch)
                avg_val_loss = np.mean(val_losses_epoch)
                pbar.set_postfix({
                    'train_loss': f"{train_loss:.4f}", 
                    'val_loss': [f"{vl:.4f}" for vl in val_losses_epoch[:5]]
                })
                # Save best model states for each ensemble member
                for e_idx in range(self.model.n_models):
                    if val_losses_epoch[e_idx] < best_val_losses[e_idx]:
                        best_val_losses[e_idx] = val_losses_epoch[e_idx]
                        if verbose >= 2:
                            logger.info(f"New best model for ensemble {e_idx} at epoch {epoch}: val_loss={val_losses_epoch[e_idx]:.6f}")
                        # CODE HERE - Store individual model state
                        self.best_model_states[e_idx] = copy.deepcopy(self.model.models[e_idx].state_dict())
                
                # Early stopping based on average validation loss
                if early_stopping and early_stopping(avg_val_loss):
                    if verbose >= 1:
                        logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Restore best model states
        for e_idx, best_state in enumerate(self.best_model_states):
            if best_state is not None:
                self.model.models[e_idx].load_state_dict(best_state)
        
        if val_loader is not None:
            logger.info(f"Restored best models with val_losses={best_val_losses}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions.
        
        Args:
            data_loader: Data loader for prediction
            
        Returns:
            Tuple of (predictions, targets) with shapes (N, Q, H) and (N, H)
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in data_loader:
                features = features.to(self.device)
            
                predictions = self.model(features, return_ensemble=False, per_model_inputs=False)
                
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        all_predictions = np.vstack(all_predictions)  # (N, Q, H)
        all_targets = np.vstack(all_targets)  # (N, H)
        return all_predictions, all_targets

