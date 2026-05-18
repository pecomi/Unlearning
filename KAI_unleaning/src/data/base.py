"""
Abstract base class for datasets.
All dataset implementations must inherit from BaseDataset.
"""

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class BaseDataset(ABC):
    """
    Abstract base class for all datasets in the KAI Unlearning framework.

    This class defines the interface that all dataset implementations must follow,
    ensuring consistency across different data sources (CIFAR, TOFU, MUSE, etc.).
    """

    def __init__(self, root: str = "./data", download: bool = True):
        """
        Initialize dataset.

        Args:
            root: Root directory for dataset storage
            download: Whether to download dataset if not present
        """
        self.root = root
        self.download = download
        self.train_dataset = None
        self.test_dataset = None

    @abstractmethod
    def load(self) -> None:
        """
        Load underlying train/test datasets onto self.train_dataset and
        self.test_dataset. Must be called before create_splits().
        """
        pass

    @abstractmethod
    def create_splits(self, forget_ratio: float, val_ratio: float,
                     batch_size: int, num_workers: int, seed: int) -> dict:
        """
        Create train/forget/retain/val splits for unlearning experiments.

        Args:
            forget_ratio: Fraction of training data to forget
            val_ratio: Fraction of training data for validation
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes
            seed: Random seed for reproducibility

        Returns:
            Dictionary containing:
                - train_loader: Full training data
                - forget_loader: Data to be forgotten
                - retain_loader: Data to retain
                - val_loader: Validation data
                - test_loader: Test data
        """
        pass

    def _create_dataloader(self, dataset, batch_size: int, shuffle: bool = True,
                           num_workers: int = 0, worker_init_fn=None,
                           pin_memory: bool = True) -> DataLoader:
        """Helper method to create a DataLoader.

        Args:
            dataset: PyTorch Dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            worker_init_fn: Worker initialization function
            pin_memory: Whether to use pinned memory (CUDA only)
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
        )
