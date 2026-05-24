"""
Unlearning Algorithm Factory for KAI Unlearning.
Allows plugin-like unlearning algorithm registration and instantiation.
"""

from typing import Dict, Type

from .base import BaseUnlearning
from .ekfac_influence import EKFACInfluenceUnlearning
from .ssd import SSDUnlearning


class UnlearningFactory:
    """
    언러닝 알고리즘에 대한 팩토리 클래스
    """

    _algorithms: Dict[str, Type[BaseUnlearning]] = {}

    @classmethod
    def register(cls, name: str, algorithm_class: Type[BaseUnlearning]) -> None:
        """
        Register a new unlearning algorithm.

        Args:
            name: Name identifier for the algorithm
            algorithm_class: Algorithm class to register

        Example:
            UnlearningFactory.register("scrub", SCRUBUnlearning)
        """
        cls._algorithms[name] = algorithm_class

    @classmethod
    def create(cls, name: str, model, **kwargs) -> BaseUnlearning:
        if name not in cls._algorithms:
            available = ", ".join(cls._algorithms.keys())
            raise ValueError(
                f"Unknown unlearning algorithm '{name}'. Available algorithms: {available}"
            )
        return cls._algorithms[name](model=model, **kwargs)


# Register built-in algorithms
UnlearningFactory.register("ssd", SSDUnlearning)
UnlearningFactory.register("ekfac_influence", EKFACInfluenceUnlearning)
