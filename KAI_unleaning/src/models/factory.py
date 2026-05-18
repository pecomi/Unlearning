"""
Model Factory for KAI Unlearning.
Allows plugin-like model registration and instantiation.
"""

from typing import Dict, Type

from .base import BaseModel
from .resnet import ResNet18


class ModelFactory:
    """
    모델 팩토리 클래스
    """

    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]) -> None:
        """
        Register a new model class.

        Args:
            name: Name identifier for the model
            model_class: Model class to register

        Example:
            ModelFactory.register("llm", LargeLanguageModel)
        """
        cls._models[name] = model_class

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModel:
        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise ValueError(
                f"Unknown model '{name}'. Available models: {available}"
            )
        return cls._models[name](**kwargs)


# Register built-in models
ModelFactory.register("resnet18", ResNet18)
# ModelFactory.register("resnet34", ResNet34)
