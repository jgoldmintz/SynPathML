"""
Stage 3: Neural Network Training
PyTorch neural network with three architecture options:
- BaseNN: Standard dense layers (implicit mechanism weighting)
- AttentionNN: Attention over mechanism groups (explicit, interpretable)
- GatedNN: Mutation-type conditioned gating
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from sklearn.model_selection import StratifiedKFold

from data_loader import DataLoader
from utils import (
    setup_logging,
    compute_metrics,
    compute_metrics_by_subset,
    compute_class_weights,
    expected_calibration_error,
    save_json,
    load_json,
    plot_pr_curve,
    plot_training_curves,
    plot_attention_weights,
)


def get_device(force_cpu: bool = False) -> torch.device:
    """
    Get compute device. Default GPU if available.

    Args:
        force_cpu: Force CPU even if GPU available

    Returns:
        torch.device
    """
    if force_cpu:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class MutationDataset(Dataset):
    """PyTorch Dataset for mutation data."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        is_synonymous: Optional[np.ndarray] = None
    ):
        """
        Initialize dataset.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            is_synonymous: Synonymous indicator (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

        if is_synonymous is not None:
            self.is_synonymous = torch.FloatTensor(is_synonymous)
        else:
            self.is_synonymous = torch.zeros(len(y))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx], self.is_synonymous[idx]


class BaseNN(nn.Module):
    """
    Option A: Standard dense layers with implicit mechanism weighting.

    Simple feedforward network that learns feature importance via gradient descent.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Initialize BaseNN.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.use_batch_norm = use_batch_norm

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        is_synonymous: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features (batch, n_features)
            is_synonymous: Synonymous indicator (unused in BaseNN)

        Returns:
            Logits (batch, 1)
        """
        h = self.layers(x)
        return self.output(h)


class AttentionNN(nn.Module):
    """
    Option B: Attention over mechanism groups.

    Provides interpretable attention weights showing which mechanism
    drove each prediction.
    """

    def __init__(
        self,
        mechanism_indices: Dict[str, List[int]],
        hidden_dim: int = 32,
        attention_dim: int = 16,
        dropout: float = 0.3
    ):
        """
        Initialize AttentionNN.

        Args:
            mechanism_indices: Dict mapping mechanism name to feature indices
            hidden_dim: Hidden dimension for mechanism embeddings
            attention_dim: Dimension for attention computation
            dropout: Dropout probability
        """
        super().__init__()

        self.mechanism_names = list(mechanism_indices.keys())
        self.mechanism_indices = mechanism_indices
        self.n_mechanisms = len(self.mechanism_names)

        # Per-mechanism encoders
        self.mechanism_encoders = nn.ModuleDict()
        for name, indices in mechanism_indices.items():
            input_dim = len(indices)
            self.mechanism_encoders[name] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        # Attention layers
        self.attention_key = nn.Linear(hidden_dim, attention_dim)
        self.attention_query = nn.Parameter(torch.randn(attention_dim))

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Store attention weights for interpretability
        self.last_attention_weights = None

    def forward(
        self,
        x: torch.Tensor,
        is_synonymous: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with attention over mechanisms.

        Args:
            x: Input features (batch, n_features)
            is_synonymous: Synonymous indicator (unused in AttentionNN)

        Returns:
            Logits (batch, 1)
        """
        batch_size = x.size(0)

        # Encode each mechanism
        mechanism_embeddings = []
        for name in self.mechanism_names:
            indices = self.mechanism_indices[name]
            mechanism_input = x[:, indices]
            embedding = self.mechanism_encoders[name](mechanism_input)
            mechanism_embeddings.append(embedding)

        # Stack: (batch, n_mechanisms, hidden_dim)
        mechanism_stack = torch.stack(mechanism_embeddings, dim=1)

        # Compute attention scores
        keys = self.attention_key(mechanism_stack)  # (batch, n_mechanisms, attention_dim)
        scores = torch.matmul(keys, self.attention_query)  # (batch, n_mechanisms)
        attention_weights = F.softmax(scores, dim=1)  # (batch, n_mechanisms)

        # Store for interpretability
        self.last_attention_weights = attention_weights.detach()

        # Weighted sum
        weighted = attention_weights.unsqueeze(-1) * mechanism_stack  # (batch, n_mechanisms, hidden_dim)
        context = weighted.sum(dim=1)  # (batch, hidden_dim)

        return self.fc(context)

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get last computed attention weights."""
        return self.last_attention_weights


class GatedNN(nn.Module):
    """
    Option C: Mutation-type conditioned gating.

    Uses is_synonymous to gate between regulatory and protein features.
    Learns to emphasize regulatory features for synonymous mutations
    and include protein features for nonsynonymous.
    """

    def __init__(
        self,
        input_dim: int,
        regulatory_indices: List[int],
        protein_indices: List[int],
        hidden_dim: int = 32,
        dropout: float = 0.3
    ):
        """
        Initialize GatedNN.

        Args:
            input_dim: Total number of input features
            regulatory_indices: Indices of regulatory features (splice, RNA, miRNA)
            protein_indices: Indices of protein-level features (PTM, structure, etc.)
            hidden_dim: Hidden dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.regulatory_indices = regulatory_indices
        self.protein_indices = protein_indices
        self.input_dim = input_dim

        # Gate network: takes all features + is_synonymous
        self.gate = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Regulatory pathway
        self.regulatory_encoder = nn.Sequential(
            nn.Linear(len(regulatory_indices), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Protein pathway
        self.protein_encoder = nn.Sequential(
            nn.Linear(len(protein_indices), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Output
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Store gate values for interpretability
        self.last_gate_values = None

    def forward(
        self,
        x: torch.Tensor,
        is_synonymous: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with mutation-type gating.

        Args:
            x: Input features (batch, n_features)
            is_synonymous: Synonymous indicator (batch,)

        Returns:
            Logits (batch, 1)
        """
        # Compute gate value
        gate_input = torch.cat([x, is_synonymous.unsqueeze(-1)], dim=1)
        g = self.gate(gate_input)  # (batch, 1)

        # Store for interpretability
        self.last_gate_values = g.detach()

        # Encode pathways
        regulatory_features = x[:, self.regulatory_indices]
        protein_features = x[:, self.protein_indices]

        h_reg = self.regulatory_encoder(regulatory_features)
        h_prot = self.protein_encoder(protein_features)

        # Gated combination
        h = g * h_reg + (1 - g) * h_prot

        return self.fc(h)

    def get_gate_values(self) -> Optional[torch.Tensor]:
        """Get last computed gate values."""
        return self.last_gate_values


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Down-weights easy examples, focuses on hard positives.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        pos_weight: Optional[float] = None
    ):
        """
        Initialize FocalLoss.

        Args:
            alpha: Balancing factor
            gamma: Focusing parameter (higher = more focus on hard examples)
            pos_weight: Weight for positive class
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model outputs (batch, 1)
            targets: Ground truth (batch,)

        Returns:
            Scalar loss
        """
        probs = torch.sigmoid(logits.squeeze(-1))
        targets = targets.float()

        # Binary cross entropy component
        bce = F.binary_cross_entropy_with_logits(
            logits.squeeze(-1),
            targets,
            reduction="none"
        )

        # Focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weight
        if self.pos_weight is not None:
            alpha_weight = self.pos_weight * targets + 1.0 * (1 - targets)
        else:
            alpha_weight = self.alpha

        loss = alpha_weight * focal_weight * bce

        return loss.mean()


class Trainer:
    """Neural network trainer with early stopping and calibration."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_weights: Optional[Dict[int, float]] = None,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        logger=None
    ):
        """
        Initialize Trainer.

        Args:
            model: PyTorch model
            device: Compute device
            class_weights: Class weights for loss
            use_focal_loss: Whether to use focal loss
            focal_gamma: Gamma for focal loss
            logger: Optional logger
        """
        self.model = model.to(device)
        self.device = device
        self.logger = logger

        # Loss function
        pos_weight = None
        if class_weights:
            pos_weight = class_weights[1] / class_weights[0]

        if use_focal_loss:
            self.criterion = FocalLoss(gamma=focal_gamma, pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight]) if pos_weight else None
            )

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

        # Calibration
        self.temperature = 1.0

    def train(
        self,
        train_loader: TorchDataLoader,
        val_loader: TorchDataLoader,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        early_stopping_patience: int = 10,
        min_delta: float = 1e-4
    ) -> Dict:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum epochs
            lr: Learning rate
            weight_decay: L2 regularization
            early_stopping_patience: Epochs without improvement before stopping
            min_delta: Minimum improvement threshold

        Returns:
            Training history dictionary
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        best_val_metric = -np.inf
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []

            for X_batch, y_batch, syn_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                syn_batch = syn_batch.to(self.device)

                optimizer.zero_grad()

                logits = self.model(X_batch, syn_batch)
                loss = self.criterion(logits, y_batch)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(X_batch)
                train_preds.extend(torch.sigmoid(logits).cpu().detach().numpy().flatten())
                train_labels.extend(y_batch.cpu().numpy())

            train_loss /= len(train_loader.dataset)
            train_preds = np.array(train_preds)
            train_labels = np.array(train_labels)

            train_metrics = compute_metrics(
                train_labels,
                (train_preds >= 0.5).astype(int),
                train_preds
            )

            # Validation
            val_loss, val_preds, val_labels = self._evaluate_epoch(val_loader)
            val_metrics = compute_metrics(
                val_labels,
                (val_preds >= 0.5).astype(int),
                val_preds
            )

            # Update scheduler
            scheduler.step(val_metrics["pr_auc"])

            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append(train_metrics["pr_auc"])
            self.val_metrics.append(val_metrics["pr_auc"])

            # Early stopping
            if val_metrics["pr_auc"] > best_val_metric + min_delta:
                best_val_metric = val_metrics["pr_auc"]
                patience_counter = 0
                best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1

            if self.logger and epoch % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, PR-AUC: {train_metrics['pr_auc']:.4f} | "
                    f"Val Loss: {val_loss:.4f}, PR-AUC: {val_metrics['pr_auc']:.4f}"
                )

            if patience_counter >= early_stopping_patience:
                if self.logger:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "best_epoch": len(self.val_metrics) - patience_counter,
            "best_val_pr_auc": best_val_metric
        }

    def _evaluate_epoch(
        self,
        data_loader: TorchDataLoader
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Evaluate model on a data loader."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch, syn_batch in data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                syn_batch = syn_batch.to(self.device)

                logits = self.model(X_batch, syn_batch)
                loss = self.criterion(logits, y_batch)

                total_loss += loss.item() * len(X_batch)
                all_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                all_labels.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(data_loader.dataset)
        return avg_loss, np.array(all_preds), np.array(all_labels)

    def evaluate(
        self,
        test_loader: TorchDataLoader,
        is_synonymous_test: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Evaluate model on test set.

        Args:
            test_loader: Test data loader
            is_synonymous_test: Synonymous indicators for stratified metrics

        Returns:
            Dictionary with metrics
        """
        _, preds, labels = self._evaluate_epoch(test_loader)

        # Apply temperature scaling
        preds_calibrated = self._apply_temperature(preds)

        results = {
            "predictions": preds_calibrated,
            "labels": labels,
        }

        # Overall metrics
        metrics = compute_metrics(
            labels,
            (preds_calibrated >= 0.5).astype(int),
            preds_calibrated
        )
        results["metrics"] = metrics

        # Stratified metrics
        if is_synonymous_test is not None:
            stratified = compute_metrics_by_subset(
                labels,
                preds_calibrated,
                is_synonymous_test
            )
            results["stratified_metrics"] = stratified

        # Calibration
        results["ece"] = expected_calibration_error(labels, preds_calibrated)

        return results

    def calibrate(
        self,
        val_loader: TorchDataLoader,
        method: str = "temperature"
    ) -> float:
        """
        Calibrate model predictions.

        Args:
            val_loader: Validation data loader
            method: Calibration method ('temperature' or 'platt')

        Returns:
            Optimal temperature/scale
        """
        _, preds, labels = self._evaluate_epoch(val_loader)

        if method == "temperature":
            # Grid search for optimal temperature
            best_ece = float("inf")
            best_temp = 1.0

            for temp in np.linspace(0.1, 3.0, 30):
                calibrated = self._apply_temperature(preds, temp)
                ece = expected_calibration_error(labels, calibrated)
                if ece < best_ece:
                    best_ece = ece
                    best_temp = temp

            self.temperature = best_temp

            if self.logger:
                self.logger.info(f"Calibration temperature: {best_temp:.3f}, ECE: {best_ece:.4f}")

            return best_temp

        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def _apply_temperature(
        self,
        probs: np.ndarray,
        temperature: Optional[float] = None
    ) -> np.ndarray:
        """Apply temperature scaling to probabilities."""
        if temperature is None:
            temperature = self.temperature

        # Convert to logits, scale, convert back
        eps = 1e-7
        logits = np.log(probs + eps) - np.log(1 - probs + eps)
        scaled_logits = logits / temperature
        return 1 / (1 + np.exp(-scaled_logits))

    def get_attention_weights(
        self,
        data_loader: TorchDataLoader
    ) -> Optional[np.ndarray]:
        """Get attention weights for AttentionNN."""
        if not isinstance(self.model, AttentionNN):
            return None

        self.model.eval()
        all_weights = []

        with torch.no_grad():
            for X_batch, _, syn_batch in data_loader:
                X_batch = X_batch.to(self.device)
                syn_batch = syn_batch.to(self.device)

                _ = self.model(X_batch, syn_batch)
                weights = self.model.get_attention_weights()
                if weights is not None:
                    all_weights.append(weights.cpu().numpy())

        if all_weights:
            return np.concatenate(all_weights, axis=0)
        return None

    def save_model(self, path: str) -> None:
        """Save model state dict."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "temperature": self.temperature,
        }, path)

    def load_model(self, path: str) -> None:
        """Load model state dict."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.temperature = checkpoint.get("temperature", 1.0)


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    is_synonymous: np.ndarray,
    model_class: str,
    model_kwargs: Dict,
    device: torch.device,
    k: int = 10,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    random_state: int = 42,
    logger=None
) -> Dict:
    """
    Run k-fold cross-validation.

    Args:
        X: Feature matrix
        y: Labels
        is_synonymous: Synonymous indicators
        model_class: 'base', 'attention', or 'gated'
        model_kwargs: Keyword arguments for model constructor
        device: Compute device
        k: Number of folds
        epochs: Max epochs per fold
        batch_size: Batch size
        lr: Learning rate
        random_state: Random seed
        logger: Optional logger

    Returns:
        Cross-validation results
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    fold_results = []
    all_preds = np.zeros(len(y))
    all_labels = y.copy()

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        if logger:
            logger.info(f"Fold {fold+1}/{k}")

        # Create datasets
        train_dataset = MutationDataset(
            X[train_idx], y[train_idx], is_synonymous[train_idx]
        )
        val_dataset = MutationDataset(
            X[val_idx], y[val_idx], is_synonymous[val_idx]
        )

        train_loader = TorchDataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = TorchDataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # Create model
        if model_class == "base":
            model = BaseNN(**model_kwargs)
        elif model_class == "attention":
            model = AttentionNN(**model_kwargs)
        elif model_class == "gated":
            model = GatedNN(**model_kwargs)
        else:
            raise ValueError(f"Unknown model class: {model_class}")

        # Compute class weights
        class_weights = compute_class_weights(y[train_idx])

        # Train
        trainer = Trainer(model, device, class_weights=class_weights, logger=None)
        history = trainer.train(
            train_loader, val_loader,
            epochs=epochs, lr=lr
        )

        # Calibrate
        trainer.calibrate(val_loader)

        # Evaluate
        results = trainer.evaluate(val_loader, is_synonymous[val_idx])

        fold_results.append({
            "fold": fold,
            "best_epoch": history["best_epoch"],
            "best_val_pr_auc": history["best_val_pr_auc"],
            "metrics": results["metrics"],
            "ece": results["ece"]
        })

        # Store predictions
        all_preds[val_idx] = results["predictions"]

    # Aggregate results
    pr_aucs = [r["metrics"]["pr_auc"] for r in fold_results]
    mccs = [r["metrics"]["mcc"] for r in fold_results]
    eces = [r["ece"] for r in fold_results]

    cv_results = {
        "fold_results": fold_results,
        "pr_auc_mean": float(np.mean(pr_aucs)),
        "pr_auc_std": float(np.std(pr_aucs)),
        "mcc_mean": float(np.mean(mccs)),
        "mcc_std": float(np.std(mccs)),
        "ece_mean": float(np.mean(eces)),
        "ece_std": float(np.std(eces)),
        "all_predictions": all_preds,
        "all_labels": all_labels
    }

    # Stratified metrics on aggregated predictions
    cv_results["stratified_metrics"] = compute_metrics_by_subset(
        all_labels, all_preds, is_synonymous
    )

    return cv_results


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Neural Network Training"
    )

    # Input
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input TSV file or prepared data directory"
    )
    parser.add_argument(
        "--stage2-results",
        type=str,
        help="Path to Stage 2 results directory (for recommended features and mechanism groups)"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results"
    )

    # Architecture
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["base", "attention", "gated"],
        default="base",
        help="Neural network architecture"
    )

    # Training options
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[64, 32],
        help="Hidden layer dimensions"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=10,
        help="Number of cross-validation folds (0 for no CV)"
    )

    # Device
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if GPU available"
    )

    # Data options
    parser.add_argument(
        "--label-column",
        type=str,
        default="PLACEHOLDER_LABEL_COLUMN",
        help="Name of label column"
    )
    parser.add_argument(
        "--is-synonymous-column",
        type=str,
        default="PLACEHOLDER_IS_SYN_COLUMN",
        help="Name of is_synonymous column"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Setup
    logger = setup_logging(args.output_dir, "stage3_neural_network")
    device = get_device(force_cpu=args.cpu)
    logger.info(f"Using device: {device}")

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    # Load data
    input_path = Path(args.input)
    if input_path.is_dir():
        X, y, is_synonymous, feature_names = DataLoader.load_prepared_data(input_path)
    else:
        config = {
            "label_column": args.label_column,
            "is_synonymous_column": args.is_synonymous_column,
        }
        loader = DataLoader(source_type="tsv", config=config, logger=logger)
        loader.load_from_tsv(args.input)
        X, y, is_synonymous, feature_names = loader.prepare_features()

    logger.info(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Positive: {y.sum()} ({100*y.mean():.2f}%)")

    # Load Stage 2 results if available
    mechanism_indices = None
    recommended_features = None

    if args.stage2_results:
        stage2_path = Path(args.stage2_results)

        mech_file = stage2_path / "mechanism_indices.json"
        if mech_file.exists():
            mechanism_indices = load_json(mech_file)
            logger.info(f"Loaded mechanism indices: {list(mechanism_indices.keys())}")

        rec_file = stage2_path / "recommended_features.json"
        if rec_file.exists():
            recommended_features = load_json(rec_file).get("recommended_features", [])
            logger.info(f"Loaded {len(recommended_features)} recommended features")

    # Build model kwargs
    if args.architecture == "base":
        model_kwargs = {
            "input_dim": X.shape[1],
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout
        }

    elif args.architecture == "attention":
        if mechanism_indices is None:
            # Create default mechanism groups based on feature naming
            mechanism_indices = {}
            prefixes = {
                "splicing": ["spliceai_", "genesplicer_"],
                "rna": ["rnafold_"],
                "mirna": ["miranda_"],
                "ptm": ["netnglyc_", "netphos_"],
                "other": []
            }

            for mech, prefs in prefixes.items():
                indices = []
                for i, f in enumerate(feature_names):
                    f_lower = f.lower()
                    if any(f_lower.startswith(p) for p in prefs):
                        indices.append(i)
                if indices:
                    mechanism_indices[mech] = indices

            # Catch remaining as "other"
            assigned = set()
            for indices in mechanism_indices.values():
                assigned.update(indices)
            other = [i for i in range(len(feature_names)) if i not in assigned]
            if other:
                mechanism_indices["other"] = other

            logger.info(f"Created default mechanism groups: {list(mechanism_indices.keys())}")

        model_kwargs = {
            "mechanism_indices": mechanism_indices,
            "hidden_dim": args.hidden_dims[0] if args.hidden_dims else 32,
            "dropout": args.dropout
        }

    elif args.architecture == "gated":
        # Identify regulatory and protein indices
        regulatory_prefixes = ["spliceai_", "genesplicer_", "rnafold_", "miranda_"]
        protein_prefixes = ["netnglyc_", "netphos_", "netsurfp_", "netmhc_", "evmutation_"]

        regulatory_indices = []
        protein_indices = []

        for i, f in enumerate(feature_names):
            f_lower = f.lower()
            if any(f_lower.startswith(p) for p in regulatory_prefixes):
                regulatory_indices.append(i)
            elif any(f_lower.startswith(p) for p in protein_prefixes):
                protein_indices.append(i)
            else:
                # Assign to regulatory by default
                regulatory_indices.append(i)

        logger.info(f"Regulatory features: {len(regulatory_indices)}")
        logger.info(f"Protein features: {len(protein_indices)}")

        model_kwargs = {
            "input_dim": X.shape[1],
            "regulatory_indices": regulatory_indices,
            "protein_indices": protein_indices,
            "hidden_dim": args.hidden_dims[0] if args.hidden_dims else 32,
            "dropout": args.dropout
        }

    logger.info(f"Architecture: {args.architecture}")
    logger.info(f"Model kwargs: {model_kwargs}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cross-validation or single train/test
    if args.cv_folds > 0:
        logger.info(f"Running {args.cv_folds}-fold cross-validation...")

        cv_results = run_cv(
            X, y, is_synonymous,
            model_class=args.architecture,
            model_kwargs=model_kwargs,
            device=device,
            k=args.cv_folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            random_state=args.random_state,
            logger=logger
        )

        logger.info("=" * 50)
        logger.info("Cross-validation Results:")
        logger.info(f"PR-AUC: {cv_results['pr_auc_mean']:.4f} +/- {cv_results['pr_auc_std']:.4f}")
        logger.info(f"MCC: {cv_results['mcc_mean']:.4f} +/- {cv_results['mcc_std']:.4f}")
        logger.info(f"ECE: {cv_results['ece_mean']:.4f} +/- {cv_results['ece_std']:.4f}")

        if "synonymous" in cv_results["stratified_metrics"]:
            syn_metrics = cv_results["stratified_metrics"]["synonymous"]
            logger.info(f"Synonymous PR-AUC: {syn_metrics['pr_auc']:.4f}")

        # Save CV results
        cv_results_save = {k: v for k, v in cv_results.items() if k not in ["all_predictions", "all_labels"]}
        save_json(cv_results_save, output_dir / "cv_results.json")

        # Save predictions
        np.save(output_dir / "cv_predictions.npy", cv_results["all_predictions"])
        np.save(output_dir / "cv_labels.npy", cv_results["all_labels"])

        # Plot PR curve
        plot_pr_curve(
            cv_results["all_labels"],
            cv_results["all_predictions"],
            save_path=str(output_dir / "pr_curve_cv.png"),
            title=f"PR Curve ({args.architecture}, {args.cv_folds}-fold CV)"
        )

    else:
        # Single train/val/test split
        logger.info("Training with single train/val/test split...")

        # Split data
        from sklearn.model_selection import train_test_split

        X_trainval, X_test, y_trainval, y_test, syn_trainval, syn_test = train_test_split(
            X, y, is_synonymous,
            test_size=0.2,
            random_state=args.random_state,
            stratify=y
        )

        X_train, X_val, y_train, y_val, syn_train, syn_val = train_test_split(
            X_trainval, y_trainval, syn_trainval,
            test_size=0.125,
            random_state=args.random_state,
            stratify=y_trainval
        )

        # Create data loaders
        train_dataset = MutationDataset(X_train, y_train, syn_train)
        val_dataset = MutationDataset(X_val, y_val, syn_val)
        test_dataset = MutationDataset(X_test, y_test, syn_test)

        train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = TorchDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Create model
        if args.architecture == "base":
            model = BaseNN(**model_kwargs)
        elif args.architecture == "attention":
            model = AttentionNN(**model_kwargs)
        elif args.architecture == "gated":
            model = GatedNN(**model_kwargs)

        # Train
        class_weights = compute_class_weights(y_train)
        trainer = Trainer(model, device, class_weights=class_weights, logger=logger)

        history = trainer.train(
            train_loader, val_loader,
            epochs=args.epochs,
            lr=args.lr
        )

        # Calibrate
        trainer.calibrate(val_loader)

        # Evaluate
        results = trainer.evaluate(test_loader, syn_test)

        logger.info("=" * 50)
        logger.info("Test Results:")
        logger.info(f"PR-AUC: {results['metrics']['pr_auc']:.4f}")
        logger.info(f"MCC: {results['metrics']['mcc']:.4f}")
        logger.info(f"ECE: {results['ece']:.4f}")

        if "synonymous" in results.get("stratified_metrics", {}):
            syn_metrics = results["stratified_metrics"]["synonymous"]
            logger.info(f"Synonymous PR-AUC: {syn_metrics['pr_auc']:.4f}")

        # Save model
        trainer.save_model(str(output_dir / "model.pt"))

        # Save results
        results_save = {k: v for k, v in results.items() if k not in ["predictions", "labels"]}
        save_json(results_save, output_dir / "test_results.json")

        # Save predictions
        np.save(output_dir / "test_predictions.npy", results["predictions"])
        np.save(output_dir / "test_labels.npy", results["labels"])

        # Plot training curves
        plot_training_curves(
            history["train_losses"],
            history["val_losses"],
            history["train_metrics"],
            history["val_metrics"],
            save_path=str(output_dir / "training_curves.png")
        )

        # Plot PR curve
        plot_pr_curve(
            results["labels"],
            results["predictions"],
            save_path=str(output_dir / "pr_curve_test.png"),
            title=f"PR Curve ({args.architecture})"
        )

        # Attention weights for AttentionNN
        if args.architecture == "attention":
            weights = trainer.get_attention_weights(test_loader)
            if weights is not None:
                mechanism_names = list(mechanism_indices.keys())
                plot_attention_weights(
                    weights,
                    mechanism_names,
                    save_path=str(output_dir / "attention_weights.png")
                )
                np.save(output_dir / "attention_weights.npy", weights)

    logger.info("\nStage 3 complete")


if __name__ == "__main__":
    main()
