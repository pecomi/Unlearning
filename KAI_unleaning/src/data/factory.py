"""
Dataset Factory for KAI Unlearning.
Allows plugin-like dataset registration and instantiation.
"""

from typing import Dict, Type

from .base import BaseDataset
from .cifar import CIFAR10Dataset, CIFAR100Dataset


class DatasetFactory:
    """
    데이터셋 팩토리 클래스
    """

    _datasets: Dict[str, Type[BaseDataset]] = {}

    @classmethod
    def register(cls, name: str, dataset_class: Type[BaseDataset]) -> None:
        """
        Register a new dataset class.

        Args:
            name: Name identifier for the dataset
            dataset_class: Dataset class to register

        Example:
            DatasetFactory.register("tofu", TOFUDataset)
        """
        cls._datasets[name] = dataset_class

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseDataset:
        if name not in cls._datasets:
            available = ", ".join(cls._datasets.keys())
            raise ValueError(
                f"Unknown dataset '{name}'. Available datasets: {available}"
            )
        return cls._datasets[name](**kwargs)


# Register built-in datasets
DatasetFactory.register("cifar10", CIFAR10Dataset)
DatasetFactory.register("cifar100", CIFAR100Dataset)
