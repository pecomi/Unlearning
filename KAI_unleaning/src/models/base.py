"""
Abstract base class for models.
All model implementations must inherit from BaseModel.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseModel(ABC):
    """
    Abstract base class for all models in the KAI Unlearning framework.

    Provides a unified interface for different model types (CNN, Regression, LLM).
    """

    def __init__(self, num_classes: int, **kwargs):
        """Initialize base model."""
        self.num_classes = num_classes
        self.model: Optional[nn.Module] = None

    @abstractmethod
    def create_model(self) -> nn.Module:
        """Create and return the PyTorch model."""
        pass


    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def save_checkpoint(self, checkpoint_path: str, epoch: int,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       metrics: Optional[dict] = None) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "epoch": epoch,
            "num_classes": self.num_classes,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if metrics is not None:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, checkpoint_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Make model callable."""
        return self.forward(x)

    def eval(self):
        """Set model to evaluation mode."""
        return self.model.eval()

    def train(self, mode: bool = True):
        """Set model to training mode."""
        return self.model.train(mode)

    def to(self, device):
        """Move model to device."""
        self.model = self.model.to(device)
        return self

    def parameters(self):
        """Return model parameters."""
        return self.model.parameters()
