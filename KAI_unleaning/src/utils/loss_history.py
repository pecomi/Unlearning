"""CSV and plot output for fixed forget/retain evaluation histories."""

import csv
from pathlib import Path
from typing import Dict, List, Tuple


HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "forget_loss",
    "retain_loss",
    "forget_accuracy",
    "retain_accuracy",
)


def save_split_loss_history(
    history: List[Dict[str, float]],
    output_dir: Path,
    run_name: str,
    loss_scale: str = "log",
) -> Tuple[Path, Path]:
    """Save full-set forget/retain metrics as CSV and a two-panel PNG."""
    if not history:
        raise ValueError("loss history must contain at least one row")
    if loss_scale not in {"linear", "log", "symlog"}:
        raise ValueError(
            "loss_scale must be one of 'linear', 'log', or 'symlog' "
            f"(got {loss_scale!r})"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"{run_name}_loss_history.csv"
    curve_path = output_dir / f"{run_name}_loss_curve.png"

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(
        epochs,
        [row["forget_loss"] for row in history],
        label="forget",
        color="#d62728",
    )
    axes[0].plot(
        epochs,
        [row["retain_loss"] for row in history],
        label="retain",
        color="#1f77b4",
    )
    axes[0].set(
        title="Mean cross-entropy loss",
        xlabel="training epoch",
        ylabel="loss",
    )
    if loss_scale == "symlog":
        axes[0].set_yscale("symlog", linthresh=1e-2)
    else:
        axes[0].set_yscale(loss_scale)
    axes[0].legend()

    axes[1].plot(
        epochs,
        [row["forget_accuracy"] for row in history],
        label="forget",
        color="#d62728",
    )
    axes[1].plot(
        epochs,
        [row["retain_accuracy"] for row in history],
        label="retain",
        color="#1f77b4",
    )
    axes[1].set(
        title="Classification accuracy",
        xlabel="training epoch",
        ylabel="accuracy",
        ylim=(0.0, 1.0),
    )
    axes[1].legend()

    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle(f"{run_name.capitalize()} full-set evaluation")
    figure.tight_layout()
    figure.savefig(curve_path, dpi=180)
    plt.close(figure)

    return history_path, curve_path
