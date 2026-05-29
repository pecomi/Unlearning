import argparse
import csv
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


def parse_float_list(value: str):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def format_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def load_last_eval(result_path: Path) -> dict:
    import json

    with result_path.open("r", encoding="utf-8") as f:
        result = json.load(f)
    history = result.get("eval_history", [])
    if not history:
        return {}
    return history[-1]


def write_summary_csv(rows, output_path: Path) -> None:
    fieldnames = [
        "step_size",
        "damping",
        "forget_val_accuracy",
        "forget_val_loss",
        "retain_val_accuracy",
        "retain_val_loss",
        "update_norm",
        "raw_update_norm",
        "clip_coef",
        "checkpoint_dir",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_summary(rows, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for damping in sorted({row["damping"] for row in rows}):
        group = sorted(
            [row for row in rows if row["damping"] == damping],
            key=lambda row: row["step_size"],
        )
        steps = [row["step_size"] for row in group]
        forget_acc = [row.get("forget_val_accuracy") for row in group]
        retain_acc = [row.get("retain_val_accuracy") for row in group]
        forget_loss = [row.get("forget_val_loss") for row in group]

        axes[0].plot(steps, forget_acc, marker="o", label=f"forget acc, damping={damping:g}")
        axes[0].plot(steps, retain_acc, marker="x", linestyle="--", label=f"retain acc, damping={damping:g}")
        axes[1].plot(steps, forget_loss, marker="o", label=f"damping={damping:g}")

    axes[0].set_title("Validation Accuracy by EKFAC Hyperparameters")
    axes[0].set_xlabel("step_size")
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Forget Validation Loss by EKFAC Hyperparameters")
    axes[1].set_xlabel("step_size")
    axes[1].set_ylabel("forget_val_loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run an EKFAC step_size/damping sweep.")
    parser.add_argument("--config", default="config_ekfac_label_based.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--step-sizes", default="0.003,0.005,0.007,0.01")
    parser.add_argument("--dampings", default="0.03,0.05,0.1")
    parser.add_argument("--output-dir", default="./runs/ekfac_hparam_sweep")
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    generated_config_dir = output_dir / "configs"
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    step_sizes = parse_float_list(args.step_sizes)
    dampings = parse_float_list(args.dampings)
    rows = []

    for step_size in step_sizes:
        for damping in dampings:
            tag = f"step{format_float(step_size)}_damp{format_float(damping)}"
            run_dir = output_dir / tag
            cfg = OmegaConf.load(base_config_path)

            OmegaConf.update(cfg, "unlearning.step_size", step_size, merge=True)
            OmegaConf.update(cfg, "unlearning.damping", damping, merge=True)
            OmegaConf.update(cfg, "checkpoint_dir", str(run_dir), merge=True)
            OmegaConf.update(cfg, "evaluation.use_test_during_unlearn", False, merge=True)

            run_name_base = OmegaConf.select(cfg, "wandb.run_name_base", default="EKFACInfluence")
            OmegaConf.update(cfg, "wandb.run_name_base", f"{run_name_base}-{tag}", merge=True)
            if args.wandb_mode is not None:
                OmegaConf.update(cfg, "wandb.mode", args.wandb_mode, merge=True)

            generated_config = generated_config_dir / f"{tag}.yaml"
            OmegaConf.save(cfg, generated_config)

            command = [
                sys.executable,
                "src/main.py",
                "--config",
                str(generated_config),
                "--mode",
                "unlearn",
            ]
            if args.checkpoint is not None:
                command.extend(["--checkpoint", args.checkpoint])

            print(f"\n=== Running {tag} ===")
            print(" ".join(command))
            if not args.dry_run:
                subprocess.run(command, check=True)

                result_path = run_dir / "unlearning_result.json"
                eval_metrics = load_last_eval(result_path)
                update_stats = {}
                result_step = {}
                if result_path.exists():
                    import json

                    with result_path.open("r", encoding="utf-8") as f:
                        result = json.load(f)
                    result_step = result.get("update_stats", {})
                    update_stats = {
                        "update_norm": result_step.get("update_norm"),
                        "raw_update_norm": result_step.get("raw_update_norm"),
                        "clip_coef": result_step.get("clip_coef"),
                    }

                rows.append({
                    "step_size": step_size,
                    "damping": damping,
                    "checkpoint_dir": str(run_dir),
                    **eval_metrics,
                    **update_stats,
                })

    if rows:
        summary_csv = output_dir / "summary.csv"
        summary_png = output_dir / "summary.png"
        write_summary_csv(rows, summary_csv)
        plot_summary(rows, summary_png)
        print(f"\nSaved summary CSV: {summary_csv}")
        print(f"Saved summary plot: {summary_png}")
    else:
        print("\nDry run complete. Generated configs only.")


if __name__ == "__main__":
    main()
