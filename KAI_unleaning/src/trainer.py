"""
Trainer module for KAI Unlearning.

Handles training, evaluation, and unlearning operations with
comprehensive logging and checkpoint management.
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.logger import get_logger


class Trainer:
    """
    Unified trainer for model training and unlearning experiments.

    Features:
    - Standard model training with learning rate scheduling
    - Unlearning algorithm execution
    - Comprehensive evaluation metrics
    - WandB integration
    - Checkpoint management
    """

    def __init__(self, model, config: dict, logger=None):
        """
        Initialize trainer.

        Args:
            model: BaseModel instance
            config: Configuration dictionary
            logger: Logger instance
        """
        self.model = model
        self.config = config
        self.logger = logger or get_logger()

        self.device = config.get("device", "cuda")
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize WandB
        wandb_config = config.get("wandb", {})
        self.logger.init_wandb(
            config=config,
            project=wandb_config.get("project", "kai-unlearning"),
            entity=wandb_config.get("entity"),
            run_name=wandb_config.get("run_name"),
            mode=wandb_config.get("mode", "online")
        )

    def train(self, train_loader, val_loader, epochs: int,
              optimizer_config: dict, scheduler_config: dict) -> nn.Module:
        """
        Train the model.

        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            epochs: Number of training epochs
            optimizer_config: Optimizer configuration
            scheduler_config: Learning rate scheduler configuration

        Returns:
            Trained model
        """
        self.logger.info("[bold green]Starting Training[/bold green]")
        self.logger.info(f"  Epochs: {epochs}")
        self.logger.info(f"  Device: {self.device}")

        # Setup optimizer
        optimizer = self._create_optimizer(optimizer_config)

        # Setup scheduler
        scheduler = self._create_scheduler(optimizer, scheduler_config, epochs)

        # Setup loss function
        criterion = nn.CrossEntropyLoss()
        self.model.model.train()

        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(
                train_loader, optimizer, criterion
            )

            # Learning rate scheduling
            scheduler.step()

            val_loss, val_acc = self._validate(val_loader, criterion)

            # Log metrics
            metrics = {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            self.logger.log_metrics(metrics, step=epoch, prefix="train/")
        self.logger.success("Training completed!")
        return self.model.model

    def evaluate(self, test_loader, criterion: Optional[nn.Module] = None) -> tuple:
        """
        Evaluate the model on test set.

        Args:
            test_loader: DataLoader for test set
            criterion: Optional loss function (defaults to CrossEntropyLoss)

        Returns:
            Tuple of (test_loss, test_accuracy)
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.logger.info("[cyan]Evaluating on Test Set...[/cyan]")
        test_loss, test_acc = self._validate(test_loader, criterion)

        self.logger.info(f"  Test Loss: {test_loss:.4f}")
        self.logger.info(f"  Test Accuracy: {test_acc:.4f}")

        return test_loss, test_acc

    def evaluate_with_metrics(self, test_loader, metrics_calculator, prefix: str = "test/") -> Dict[str, float]:
        """
        Evaluate the model with detailed metrics.

        Args:
            test_loader: DataLoader for test set
            metrics_calculator: MetricsCalculator instance
            prefix: Prefix for metric names

        Returns:
            Dictionary of metrics
        """
        self.logger.info("[cyan]Computing detailed metrics on Test Set...[/cyan]")
        metrics = metrics_calculator.compute_classification_metrics(
            self.model.model, test_loader
        )

        for key, value in metrics.items():
            self.logger.info(f"  {prefix}{key}: {value:.4f}")

        return metrics

    def _train_epoch(self, dataloader, optimizer, criterion) -> tuple:
        """Train for one epoch."""
        self.model.model.train()
        self.model.model.to(self.device)

        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            outputs = self.model.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def _validate(self, dataloader, criterion) -> tuple:
        """Validate the model."""
        self.model.model.eval()
        self.model.model.to(self.device)

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model.model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += inputs.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def _create_optimizer(self, config: dict) -> optim.Optimizer:
        """Create optimizer from configuration."""
        name = config.get("name", "sgd").lower()
        lr = config.get("learning_rate", 0.1)
        momentum = config.get("momentum", 0.9)
        weight_decay = config.get("weight_decay", 0.0005)

        if name == "sgd":
            return optim.SGD(
                self.model.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif name == "adam":
            return optim.Adam(
                self.model.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif name == "adamw":
            return optim.AdamW(
                self.model.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {name}")

    def _create_scheduler(self, optimizer, config: dict, epochs: int):
        """Create learning rate scheduler from configuration."""
        name = config.get("name", "cosine").lower()
        warmup_epochs = config.get("warmup_epochs", 0)

        if name == "cosine":
            return CosineAnnealingLR(
                optimizer,
                T_max=epochs - warmup_epochs,
                eta_min=1e-6
            )
        elif name == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=epochs // 3,
                gamma=0.1
            )
        elif name == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {name}")

    def save_model(self, path: str, epoch: int = 0) -> None:
        """Save model checkpoint."""
        self.model.save_checkpoint(path, epoch=epoch)

    def load_model(self, path: str) -> None:
        """Load model checkpoint."""
        self.model.load_checkpoint(path)

    def finish(self) -> None:
        """Finish training session."""
        self.logger.finish_wandb()
