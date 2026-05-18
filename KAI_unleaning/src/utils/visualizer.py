"""
시각화를 위한 유틸
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix


# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# CIFAR-100 fine label class names (100 classes)
CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]


class UnlearningVisualizer:

    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or CIFAR10_CLASSES
        plt.style.use("seaborn-v0_8-darkgrid")


    def create_accuracy_comparison_plot(
        self,
        metrics: Dict[str, float],
        title: str = "Accuracy Comparison"
    ) -> Figure:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract accuracy metrics
        accuracy_keys = [k for k in metrics.keys() if "accuracy" in k]
        accuracy_values = [metrics[k] for k in accuracy_keys]

        # Clean up labels
        labels = [k.replace("_", " ").title() for k in accuracy_keys]

        colors = ["#2ecc71" if "retain" in k or "test" in k else "#e74c3c" for k in accuracy_keys]

        bars = ax.bar(labels, accuracy_values, color=colors, alpha=0.7, edgecolor="black")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

        ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()

        return fig


    def create_model_comparison_plot(
        self,
        unlearned_metrics: Dict[str, float],
        retrained_metrics: Dict[str, float],
        title: str = "Unlearned vs Retrained (Gold Standard)"
    ) -> Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        metrics_to_compare = [
            ("retain_accuracy", "Retain Accuracy"),
            ("forget_accuracy", "Forget Accuracy"),
            ("test_accuracy", "Test Accuracy")
        ]

        for idx, (metric_key, metric_name) in enumerate(metrics_to_compare):
            ax = axes[idx]

            unlearned_val = unlearned_metrics.get(metric_key, 0)
            retrained_val = retrained_metrics.get(metric_key, 0)

            x = ["Unlearned", "Retrained"]
            y = [unlearned_val, retrained_val]

            bars = ax.bar(x, y, color=["#3498db", "#f39c12"], alpha=0.7, edgecolor="black")

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold"
                )

            ax.set_ylabel("Accuracy", fontsize=11, fontweight="bold")
            ax.set_title(metric_name, fontsize=12, fontweight="bold")
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        return fig


    def create_per_class_accuracy_plot(
        self,
        model,
        dataloader,
        device: str,
        class_names: Optional[List[str]] = None,
        title: str = "Per-Class Accuracy"
    ) -> Figure:
        class_names = class_names or self.class_names
        num_classes = len(class_names)

        model.eval()
        model.to(device)

        correct_per_class = torch.zeros(num_classes)
        total_per_class = torch.zeros(num_classes)

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                predictions = outputs.argmax(dim=1)

                for i in range(num_classes):
                    mask = (targets == i)
                    correct_per_class[i] += (predictions[mask] == targets[mask]).sum().item()
                    total_per_class[i] += mask.sum().item()

        accuracies = [
            correct_per_class[i].item() / total_per_class[i].item() if total_per_class[i] > 0 else 0
            for i in range(num_classes)
        ]

        # Adjust figure size for large datasets
        fig_size = (12, 6) if num_classes <= 30 else (30, 8)
        fig, ax = plt.subplots(figsize=fig_size)

        colors = plt.cm.viridis(np.linspace(0, 1, num_classes))
        bars = ax.bar(class_names, accuracies, color=colors, alpha=0.7, edgecolor="black")

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )

        ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(y=1.0 / num_classes, color="red", linestyle="--", alpha=0.5, label="Random Chance")

        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()

        return fig


    def create_confusion_matrix_plot(
        self,
        model,
        dataloader,
        device: str,
        class_names: Optional[List[str]] = None,
        title: str = "Confusion Matrix"
    ) -> Figure:
        class_names = class_names or self.class_names
        num_classes = len(class_names)

        model.eval()
        model.to(device)

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                predictions = outputs.argmax(dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        cm = confusion_matrix(all_targets, all_predictions, labels=range(num_classes))

        # Adjust figure size for large datasets
        fig_size = (10, 8) if num_classes <= 30 else (20, 16)
        fig, ax = plt.subplots(figsize=fig_size)

        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(num_classes),
            yticks=np.arange(num_classes),
            xticklabels=class_names,
            yticklabels=class_names,
            title=title,
            ylabel="True Label",
            xlabel="Predicted Label"
        )

        # Add text annotations (skip for large datasets like CIFAR-100 for readability)
        if num_classes <= 30:
            thresh = cm.max() / 2.0
            for i in range(num_classes):
                for j in range(num_classes):
                    ax.text(
                        j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=10
                    )

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        return fig


    def create_model_distance_plot(
        self,
        distance_metrics: Dict[str, float],
        title: str = "Model Distance Metrics"
    ) -> Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # L2 Distance
        ax1 = axes[0]
        l2_dist = distance_metrics.get("param_l2_distance", 0)
        l2_rel = distance_metrics.get("param_l2_relative", 0)

        metrics = ["L2 Distance", "Relative L2"]
        values = [l2_dist, l2_rel]

        bars = ax1.bar(metrics, values, color=["#e74c3c", "#3498db"], alpha=0.7, edgecolor="black")

        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold"
            )

        ax1.set_ylabel("Distance", fontsize=11, fontweight="bold")
        ax1.set_title("Parameter Distance", fontsize=12, fontweight="bold")
        ax1.grid(axis="y", alpha=0.3)

        # Similarity metrics
        ax2 = axes[1]
        cos_sim = distance_metrics.get("param_cosine_similarity", 0)
        out_cos = distance_metrics.get("output_cosine_similarity", cos_sim)  # fallback

        similarity_metrics = ["Param Cosine", "Output Cosine"]
        similarity_values = [cos_sim, out_cos]

        bars = ax2.bar(similarity_metrics, similarity_values, color=["#9b59b6", "#1abc9c"], alpha=0.7, edgecolor="black")

        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold"
            )

        ax2.set_ylabel("Similarity", fontsize=11, fontweight="bold")
        ax2.set_title("Similarity Metrics", fontsize=12, fontweight="bold")
        ax2.set_ylim(0, 1.0)
        ax2.grid(axis="y", alpha=0.3)

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        return fig


    def create_forget_quality_plot(
        self,
        forget_accuracy: float,
        expected_random: float,
        title: str = "Forget Quality Assessment"
    ) -> Figure:
        fig, ax = plt.subplots(figsize=(8, 6))

        categories = ["Forget Accuracy", "Random Baseline"]
        values = [forget_accuracy, expected_random]

        colors = ["#e74c3c" if forget_accuracy > expected_random else "#2ecc71", "#95a5a6"]
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor="black")

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold"
            )

        ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylim(0, max(1.0, forget_accuracy * 1.2))
        ax.grid(axis="y", alpha=0.3)

        # Add annotation
        diff = abs(forget_accuracy - expected_random)
        status = "Good!" if forget_accuracy <= expected_random * 1.1 else "Needs Improvement"
        ax.annotate(
            f"Gap: {diff:.4f}\n{status}",
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.3)
        )

        plt.tight_layout()

        return fig
    

    def log_figure_to_wandb(self, fig: Figure, name: str) -> None:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({name: wandb.Image(fig)})
        except ImportError:
            pass
        

    def close_all_figures(self) -> None:
        plt.close("all")
