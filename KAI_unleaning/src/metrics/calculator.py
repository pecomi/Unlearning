"""
Evaluation metrics calculator for Machine Unlearning.

Computes both Model Utility and Forget Quality metrics as specified:
1. Model Utility (Retain Set)
   - Probability: P(y|x) - likelihood of correct prediction
   - Accuracy: Classification accuracy
   - Truth Ratio: P(True) / P(Wrong)

2. Forget Quality (Forget Set)
   - Probability: P(y|x) for forget set (should be low)
   - Accuracy: Classification accuracy on forget set (should be near random)
   - Truth Ratio: Should be close to 1 (no discrimination)

3. Gold Standard Comparison (Unlearned vs Retrained)
   - Parameter distance between models
   - Output similarity on test data
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.visualizer import UnlearningVisualizer, CIFAR10_CLASSES, CIFAR100_CLASSES


def _get_class_names(dataset_name: str) -> list:
    """Get class names for a given dataset."""
    class_names_map = {
        "cifar10": CIFAR10_CLASSES,
        "cifar100": CIFAR100_CLASSES,
    }
    return class_names_map.get(dataset_name, CIFAR10_CLASSES)


def _get_model_parameters(model: nn.Module) -> torch.Tensor:
    """Get flattened model parameters."""
    return torch.cat([p.data.flatten() for p in model.parameters() if p.requires_grad])


class MetricsCalculator:
    """
    Calculator for unlearning evaluation metrics.

    Supports both classification tasks (CIFAR) and language tasks.
    """

    def __init__(self, device: str = "cuda", enable_viz: bool = True, dataset_name: str = "cifar10"):
        """
        Initialize metrics calculator.

        Args:
            device: Device to compute on
            enable_viz: Whether to create visualizations and log to wandb
            dataset_name: Name of dataset (cifar10, cifar100) for class names
        """
        self.device = device
        self.enable_viz = enable_viz
        class_names = _get_class_names(dataset_name)
        self.visualizer = UnlearningVisualizer(class_names=class_names) if enable_viz else None

    def compute_per_class_accuracy(self, model, dataloader,
                                   num_classes: int) -> Dict[int, float]:
        """
        Compute per-class accuracy on a dataloader.

        Returns a dict mapping class index -> accuracy. Classes absent from
        the loader (or with no samples) are reported as float('nan') so the
        caller can distinguish "not seen" from "seen and got 0%".
        """
        model.eval()
        model.to(self.device)

        correct = torch.zeros(num_classes, dtype=torch.long, device=self.device)
        total = torch.zeros(num_classes, dtype=torch.long, device=self.device)

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                preds = model(inputs).argmax(dim=1)
                hits = (preds == targets).long()

                total.scatter_add_(0, targets, torch.ones_like(targets))
                correct.scatter_add_(0, targets, hits)

        per_class = {}
        for c in range(num_classes):
            t = total[c].item()
            per_class[c] = (correct[c].item() / t) if t > 0 else float("nan")
        return per_class

    def compute_classification_metrics(self, model, dataloader) -> Dict[str, float]:
        """
        Compute classification metrics for a given dataloader.

        Args:
            model: PyTorch model
            dataloader: DataLoader to evaluate on

        Returns:
            Dictionary containing:
                - accuracy: Classification accuracy
                - loss: Average cross-entropy loss
                - confidence: Average confidence on correct predictions
                - entropy: Average prediction entropy
        """
        model.eval()
        model.to(self.device)

        criterion = nn.CrossEntropyLoss(reduction="none")

        total_correct = 0
        total_samples = 0
        total_loss = 0
        confidences = []
        entropies = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                batch_size = inputs.size(0)

                # Forward pass
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)

                # Compute metrics
                losses = criterion(outputs, targets)
                total_loss += losses.sum().item()

                predictions = outputs.argmax(dim=1)
                correct = (predictions == targets).sum().item()
                total_correct += correct
                total_samples += batch_size

                # Confidence on correct predictions
                max_probs, _ = probs.max(dim=1)
                correct_mask = (predictions == targets).cpu()
                confidences.extend(max_probs[correct_mask].cpu().tolist())

                # Entropy (uncertainty)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
                entropies.extend(entropy.cpu().tolist())

        accuracy = total_correct / total_samples if total_samples > 0 else 0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0

        return {
            "accuracy": accuracy,
            "loss": avg_loss,
            "confidence": avg_confidence,
            "entropy": avg_entropy,
        }

    def compute_model_utility(self, model, retain_loader, test_loader=None,
                             forget_loader=None) -> Dict[str, float]:
        """
        Compute Model Utility metrics.

        Measures how well the model performs on data it should remember.

        Args:
            model: PyTorch model
            retain_loader: DataLoader for retain set
            test_loader: Optional DataLoader for test set
            forget_loader: Optional DataLoader for forget set comparison

        Returns:
            Dictionary containing utility metrics
        """
        retain_metrics = self.compute_classification_metrics(model, retain_loader)
        truth_ratio = self._compute_truth_ratio(model, retain_loader)

        metrics = {
            "retain_accuracy": retain_metrics["accuracy"],
            "retain_loss": retain_metrics["loss"],
            "retain_confidence": retain_metrics["confidence"],
            "truth_ratio_retain": truth_ratio,
        }
        if test_loader is not None:
            test_metrics = self.compute_classification_metrics(model, test_loader)
            metrics.update({
                "test_accuracy": test_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_confidence": test_metrics["confidence"],
            })

        return metrics

    def compute_forget_quality(self, model, forget_loader) -> Dict[str, float]:
        """
        Compute Forget Quality metrics.

        Measures how well the model has forgotten the target data.

        Args:
            model: PyTorch model
            forget_loader: DataLoader for forget set

        Returns:
            Dictionary containing forget quality metrics
        """
        forget_metrics = self.compute_classification_metrics(model, forget_loader)

        # Compute truth ratio on forget set (should be close to 1)
        truth_ratio = self._compute_truth_ratio(model, forget_loader)

        # Compute "forget score" - lower accuracy on forget set is better
        # We want accuracy close to random chance
        num_classes = model.num_classes if hasattr(model, "num_classes") else 10
        random_accuracy = 1.0 / num_classes

        forget_score = 1.0 - abs(forget_metrics["accuracy"] - random_accuracy)

        return {
            "forget_accuracy": forget_metrics["accuracy"],
            "forget_loss": forget_metrics["loss"],
            "forget_confidence": forget_metrics["confidence"],
            "forget_entropy": forget_metrics["entropy"],
            "truth_ratio_forget": truth_ratio,
            "forget_score": forget_score,
        }

    def _compute_truth_ratio(self, model, dataloader) -> float:
        """
        Compute Truth Ratio: P(True Answer) / P(Wrong Answer).

        A high ratio (>1) indicates the model confidently answers correctly.
        A ratio close to 1 indicates the model makes random predictions.

        Args:
            model: PyTorch model
            dataloader: DataLoader to evaluate on

        Returns:
            Truth ratio value
        """
        model.eval()
        model.to(self.device)

        total_true_prob = 0
        total_wrong_prob = 0
        num_samples = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                batch_size = inputs.size(0)

                # Forward pass
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)

                # Get probabilities
                for i in range(batch_size):
                    target = targets[i]
                    prob_vector = probs[i]

                    # Probability of correct class
                    true_prob = prob_vector[target].item()

                    # Probability of wrong classes (average)
                    wrong_indices = [j for j in range(len(prob_vector)) if j != target]
                    wrong_prob = prob_vector[wrong_indices].mean().item()

                    total_true_prob += true_prob
                    total_wrong_prob += wrong_prob
                    num_samples += 1

        avg_true = total_true_prob / num_samples if num_samples > 0 else 0
        avg_wrong = total_wrong_prob / num_samples if num_samples > 0 else 0

        # Avoid division by zero
        if avg_wrong < 1e-10:
            return float("inf") if avg_true > 1e-10 else 1.0

        truth_ratio = avg_true / avg_wrong
        return truth_ratio

    def compute_all_metrics(self, model, retain_loader, forget_loader,
                           test_loader=None, logger=None) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Args:
            model: PyTorch model
            retain_loader: DataLoader for retain set
            forget_loader: DataLoader for forget set
            test_loader: Optional DataLoader for test set
            logger: Optional logger instance

        Returns:
            Dictionary containing all metrics
        """
        if logger:
            logger.info("[cyan]Computing Model Utility metrics...[/cyan]")

        utility_metrics = self.compute_model_utility(
            model, retain_loader, test_loader, forget_loader
        )

        if logger:
            logger.info("[cyan]Computing Forget Quality metrics...[/cyan]")

        forget_metrics = self.compute_forget_quality(model, forget_loader)

        class_names = self.visualizer.class_names if self.visualizer else None
        per_class_metrics = {}
        per_class_acc = {}
        if test_loader is not None:
            num_classes = model.num_classes if hasattr(model, "num_classes") else 10
            per_class_acc = self.compute_per_class_accuracy(model, test_loader, num_classes)
            for c, acc in per_class_acc.items():
                label = class_names[c] if class_names and c < len(class_names) else str(c)
                per_class_metrics[f"test_acc_class_{c}_{label}"] = acc

        all_metrics = {**utility_metrics, **forget_metrics, **per_class_metrics}

        if self.visualizer:
            fig = self.visualizer.create_accuracy_comparison_plot(all_metrics)
            self.visualizer.log_figure_to_wandb(fig, "evaluation/accuracy_comparison")
            self.visualizer.close_all_figures()

            num_classes = model.num_classes if hasattr(model, "num_classes") else 10
            fig = self.visualizer.create_forget_quality_plot(
                all_metrics["forget_accuracy"],
                1.0 / num_classes
            )
            self.visualizer.log_figure_to_wandb(fig, "evaluation/forget_quality")
            self.visualizer.close_all_figures()

            if test_loader is not None:
                fig = self.visualizer.create_per_class_accuracy_plot(
                    model, test_loader, self.device, title="Test Set Per-Class Accuracy"
                )
                self.visualizer.log_figure_to_wandb(fig, "evaluation/test_per_class_accuracy")
                self.visualizer.close_all_figures()

                fig = self.visualizer.create_confusion_matrix_plot(
                    model, test_loader, self.device, title="Test Set Confusion Matrix"
                )
                self.visualizer.log_figure_to_wandb(fig, "evaluation/test_confusion_matrix")
                self.visualizer.close_all_figures()

        if logger:
            logger.print("\n[bold]=== Evaluation Summary ===[/bold]")
            logger.print(f"[green]Model Utility:[/green]")
            logger.print(f"  Retain Accuracy: {all_metrics['retain_accuracy']:.4f}")
            if "test_accuracy" in all_metrics:
                logger.print(f"  Test Accuracy:   {all_metrics['test_accuracy']:.4f}")
            logger.print(f"  Truth Ratio:     {all_metrics['truth_ratio_retain']:.4f}")
            logger.print(f"[red]Forget Quality:[/red]")
            logger.print(f"  Forget Accuracy: {all_metrics['forget_accuracy']:.4f}")
            logger.print(f"  Forget Score:   {all_metrics['forget_score']:.4f}")
            logger.print(f"  Truth Ratio:    {all_metrics['truth_ratio_forget']:.4f}")
            if per_class_acc:
                logger.print(f"[cyan]Per-Class Test Accuracy:[/cyan]")
                for c, acc in per_class_acc.items():
                    label = class_names[c] if class_names and c < len(class_names) else str(c)
                    acc_str = f"{acc:.4f}" if acc == acc else "n/a"
                    logger.print(f"  class {c:>2} ({label}): {acc_str}")

        return all_metrics

    def compute_model_distance(self, model1, model2, test_loader=None) -> Dict[str, float]:
        """
        Compute distance metrics between two models.

        Args:
            model1: First PyTorch model (e.g., unlearned model)
            model2: Second PyTorch model (e.g., retrained model)
            test_loader: Optional DataLoader for output similarity comparison

        Returns:
            Dictionary containing distance metrics
        """
        model1.eval()
        model2.eval()
        model1.to(self.device)
        model2.to(self.device)

        # Get model parameters
        params1 = _get_model_parameters(model1)
        params2 = _get_model_parameters(model2)

        # L2 distance between parameters
        l2_distance = torch.norm(params1 - params2, p=2).item()

        # Cosine similarity between parameters
        cosine_sim = F.cosine_similarity(params1.unsqueeze(0), params2.unsqueeze(0)).item()

        # Relative L2 distance (normalized by norm of model2)
        l2_relative = l2_distance / (torch.norm(params2, p=2).item() + 1e-10)

        metrics = {
            "param_l2_distance": l2_distance,
            "param_l2_relative": l2_relative,
            "param_cosine_similarity": cosine_sim,
        }

        # Output similarity on test data
        if test_loader is not None:
            output_sim = self._compute_output_similarity(model1, model2, test_loader)
            metrics["output_kl_divergence"] = output_sim["kl_divergence"]
            metrics["output_cosine_similarity"] = output_sim["cosine_similarity"]

        return metrics

    def _compute_output_similarity(self, model1, model2, dataloader) -> Dict[str, float]:
        """
        Compute output similarity between two models on given data.

        Args:
            model1: First PyTorch model
            model2: Second PyTorch model
            dataloader: DataLoader for comparison

        Returns:
            Dictionary containing similarity metrics
        """
        model1.eval()
        model2.eval()
        model1.to(self.device)
        model2.to(self.device)

        total_kl = 0
        total_cosine = 0
        num_batches = 0

        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)

                # Get outputs from both models
                outputs1 = model1(inputs)
                outputs2 = model2(inputs)

                # Softmax probabilities
                probs1 = F.softmax(outputs1, dim=1)
                probs2 = F.softmax(outputs2, dim=1)

                # KL divergence (symmetric)
                kl_12 = F.kl_div(
                    torch.log(probs1 + 1e-10),
                    probs2,
                    reduction='batchmean'
                )
                kl_21 = F.kl_div(
                    torch.log(probs2 + 1e-10),
                    probs1,
                    reduction='batchmean'
                )
                kl_divergence = (kl_12 + kl_21).item() / 2

                # Cosine similarity between flattened outputs
                cosine_sim = F.cosine_similarity(
                    outputs1.flatten(start_dim=1),
                    outputs2.flatten(start_dim=1),
                    dim=1
                ).mean().item()

                total_kl += kl_divergence
                total_cosine += cosine_sim
                num_batches += 1

        return {
            "kl_divergence": total_kl / num_batches if num_batches > 0 else 0,
            "cosine_similarity": total_cosine / num_batches if num_batches > 0 else 0,
        }

    def compare_with_gold_standard(
        self,
        unlearned_model,
        retrained_model,
        retain_loader,
        forget_loader,
        test_loader=None,
        retain_val_loader=None,
        forget_val_loader=None,
        logger=None
    ) -> Dict[str, float]:
        """
        Compare unlearned model with gold standard (retrained) model.

        Args:
            unlearned_model: Model after unlearning
            retrained_model: Model trained from scratch on retain data
            retain_loader: DataLoader for retain set
            forget_loader: DataLoader for forget set
            test_loader: Optional DataLoader for test set
            retain_val_loader: Optional DataLoader for retain validation set
            forget_val_loader: Optional DataLoader for forget validation set
            logger: Optional logger instance

        Returns:
            Dictionary containing all comparison metrics
        """
        if logger:
            logger.info("[cyan]Evaluating Unlearned Model...[/cyan]")

        unlearned_metrics = self.compute_all_metrics(
            unlearned_model, retain_loader, forget_loader, test_loader
        )

        if logger:
            logger.info("[cyan]Evaluating Retrained (Gold Standard) Model...[/cyan]")

        retrained_metrics = self.compute_all_metrics(
            retrained_model, retain_loader, forget_loader, test_loader
        )

        if logger:
            logger.info("[cyan]Computing Model Distance...[/cyan]")

        distance_metrics = self.compute_model_distance(
            unlearned_model, retrained_model, test_loader
        )

        unlearned_val_metrics = {}
        retrained_val_metrics = {}
        val_gap_metrics = {}
        validation_loaders = {
            "retain_val": retain_val_loader,
            "forget_val": forget_val_loader,
        }

        for loader_name, dataloader in validation_loaders.items():
            if dataloader is None or len(dataloader.dataset) == 0:
                continue

            if logger:
                logger.info(f"[cyan]Evaluating {loader_name} against Gold Standard...[/cyan]")

            unlearned_eval = self.compute_classification_metrics(unlearned_model, dataloader)
            retrained_eval = self.compute_classification_metrics(retrained_model, dataloader)

            for metric_name, value in unlearned_eval.items():
                unlearned_val_metrics[f"{loader_name}_{metric_name}"] = value
            for metric_name, value in retrained_eval.items():
                retrained_val_metrics[f"{loader_name}_{metric_name}"] = value

            val_gap_metrics[f"{loader_name}_accuracy_gap"] = abs(
                unlearned_eval["accuracy"] - retrained_eval["accuracy"]
            )
            val_gap_metrics[f"{loader_name}_loss_gap"] = abs(
                unlearned_eval["loss"] - retrained_eval["loss"]
            )

        # Compute accuracy gaps
        retain_acc_gap = abs(unlearned_metrics["retain_accuracy"] - retrained_metrics["retain_accuracy"])
        forget_acc_gap = abs(unlearned_metrics["forget_accuracy"] - retrained_metrics["forget_accuracy"])
        test_acc_gap = None
        if "test_accuracy" in unlearned_metrics and "test_accuracy" in retrained_metrics:
            test_acc_gap = abs(unlearned_metrics["test_accuracy"] - retrained_metrics["test_accuracy"])

        # Combine all metrics
        all_metrics = {
            **{f"unlearned_{k}": v for k, v in unlearned_metrics.items()},
            **{f"retrained_{k}": v for k, v in retrained_metrics.items()},
            **{f"unlearned_{k}": v for k, v in unlearned_val_metrics.items()},
            **{f"retrained_{k}": v for k, v in retrained_val_metrics.items()},
            **distance_metrics,
            "retain_accuracy_gap": retain_acc_gap,
            "forget_accuracy_gap": forget_acc_gap,
            **val_gap_metrics,
        }
        if test_acc_gap is not None:
            all_metrics["test_accuracy_gap"] = test_acc_gap

        # Create visualizations
        if self.visualizer:
            # Model comparison plot
            fig = self.visualizer.create_model_comparison_plot(
                unlearned_metrics, retrained_metrics
            )
            self.visualizer.log_figure_to_wandb(fig, "comparison/model_accuracy_comparison")
            self.visualizer.close_all_figures()

            # Model distance plot
            fig = self.visualizer.create_model_distance_plot(distance_metrics)
            self.visualizer.log_figure_to_wandb(fig, "comparison/model_distance")
            self.visualizer.close_all_figures()

            if unlearned_val_metrics and retrained_val_metrics:
                fig = self.visualizer.create_validation_comparison_plot(
                    unlearned_val_metrics,
                    retrained_val_metrics
                )
                self.visualizer.log_figure_to_wandb(fig, "comparison/validation_model_comparison")
                self.visualizer.close_all_figures()

            if test_loader is not None:
                # Per-class accuracy comparison for both models
                fig = self.visualizer.create_per_class_accuracy_plot(
                    unlearned_model, test_loader, self.device,
                    title="Unlearned Model - Per-Class Accuracy"
                )
                self.visualizer.log_figure_to_wandb(fig, "comparison/unlearned_per_class_accuracy")
                self.visualizer.close_all_figures()

                fig = self.visualizer.create_per_class_accuracy_plot(
                    retrained_model, test_loader, self.device,
                    title="Retrained Model (Gold Standard) - Per-Class Accuracy"
                )
                self.visualizer.log_figure_to_wandb(fig, "comparison/retrained_per_class_accuracy")
                self.visualizer.close_all_figures()

        # Log comparison summary
        if logger:
            logger.print("\n[bold]=== Gold Standard Comparison ===[/bold]")
            logger.print(f"[yellow]Model Distance:[/yellow]")
            logger.print(f"  Param L2 Distance:   {distance_metrics['param_l2_distance']:.6f}")
            logger.print(f"  Param L2 Relative:   {distance_metrics['param_l2_relative']:.6f}")
            logger.print(f"  Param Cosine Sim:    {distance_metrics['param_cosine_similarity']:.6f}")
            if "output_kl_divergence" in distance_metrics:
                logger.print(f"  Output KL Div:      {distance_metrics['output_kl_divergence']:.6f}")
                logger.print(f"  Output Cosine Sim:  {distance_metrics['output_cosine_similarity']:.6f}")
            logger.print(f"[yellow]Accuracy Gap (Unlearned vs Retrained):[/yellow]")
            logger.print(f"  Retain Acc Gap:      {retain_acc_gap:.4f}")
            logger.print(f"  Forget Acc Gap:      {forget_acc_gap:.4f}")
            if test_acc_gap is not None:
                logger.print(f"  Test Acc Gap:        {test_acc_gap:.4f}")
            if val_gap_metrics:
                logger.print(f"[yellow]Validation Gap (Unlearned vs Retrained):[/yellow]")
                if "retain_val_accuracy_gap" in val_gap_metrics:
                    logger.print(f"  Retain Val Acc Gap:  {val_gap_metrics['retain_val_accuracy_gap']:.4f}")
                    logger.print(f"  Retain Val Loss Gap: {val_gap_metrics['retain_val_loss_gap']:.4f}")
                if "forget_val_accuracy_gap" in val_gap_metrics:
                    logger.print(f"  Forget Val Acc Gap:  {val_gap_metrics['forget_val_accuracy_gap']:.4f}")
                    logger.print(f"  Forget Val Loss Gap: {val_gap_metrics['forget_val_loss_gap']:.4f}")

        return all_metrics
