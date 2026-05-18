"""
ResNet model implementations for KAI Unlearning.
"""

import torch.nn as nn 
import torchvision

from .base import BaseModel


class ResNet18(BaseModel):
    """
    ResNet-18 implementation with Fisher Information Matrix computation.
    """

    name = "resnet18"

    def __init__(self, num_classes: int = 10, **kwargs):
        super().__init__(num_classes, **kwargs)

    def create_model(self) -> nn.Module:
        self.model = torchvision.models.resnet18(weights=None)

        return self.model
