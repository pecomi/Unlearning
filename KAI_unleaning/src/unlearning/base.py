"""
Abstract base class for unlearning algorithms.
All unlearning implementations must inherit from BaseUnlearning.
"""

from abc import ABC, abstractmethod


class BaseUnlearning(ABC):
    """
    Abstract base class for all unlearning algorithms.

    Provides a unified interface for different unlearning methods:
    - Fisher-based (SSD, SCRUB, etc.)
    - Influence-based
    - Gradient-ascent based
    - Retraining
    - etc.
    """

    def __init__(self, model, **kwargs):
        """
        Initialize unlearning algorithm.

        Args:
            model: BaseModel instance to apply unlearning to
            **kwargs: Algorithm-specific parameters
        """
        self.model = model

    @abstractmethod
    def unlearn(self, forget_loader, train_loader, **kwargs) -> dict:
        """
        Execute unlearning algorithm.
        """
        pass

    @abstractmethod
    def save_unlearned_model(self, save_path: str) -> None:
        """
        Save the unlearned model.
        """
        pass
