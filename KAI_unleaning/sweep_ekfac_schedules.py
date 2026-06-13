import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


def parse_float_list(value: str):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_schedule_list(value: str):
    schedules = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        schedules.append(parse_float_list(item.replace(":", ",")))
    return schedules


def format_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def format_schedule(schedule) -> str:
    return "sched" + "_".join(format_float(value) for value in schedule)


def load_result(result_path: Path) -> dict:
    with result_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def last_eval_metrics(result: dict) -> dict:
    history = result.get("eval_history", [])
    return history[-1] if history else {}


def final_update_stats(result: dict) -> dict:
    update_stats = result.get("update_stats", {})
    return {
        "update_norm": update_stats.get("update_norm"),
        "raw_update_norm": update_stats.get("raw_update_norm"),
        "nonfinite_update_tensors": update_stats.get("nonfinite_update_tensors"),
    }


def step_update_norms(result: dict) -> dict:
    values = {}
    for step in result.get("step_results", []):
        step_id = step.get("step")
        if step_id is None:
            continue
        update_stats = step.get("update_stats", {})
        values[f"step{step_id}_effective_step_size"] = step.get("effective_step_size")
        values[f"step{step_id}_update_norm"] = update_stats.get("update_norm")
        values[f"step{step_id}_raw_update_norm"] = update_stats.get("raw_update_norm")
    return values


def write_summary_csv(rows, output_path: Path) -> None:
    dynamic_fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key.startswith("step")
        }
    )
    fieldnames = [
        "tag",
        "damping",
        "step_size_schedule",
        "num_unlearn_steps",
        "forget_val_accuracy",
        "forget_val_loss",
        "retain_val_accuracy",
        "retain_val_loss",
        "update_norm",
        "raw_update_norm",
        "nonfinite_update_tensors",
        "checkpoint_dir",
        *dynamic_fields,
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_summary(rows, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["tag"] for row in rows]
    forget_acc = [row.get("forget_val_accuracy") for row in rows]
    retain_acc = [row.get("retain_val_accuracy") for row in rows]
    forget_loss = [row.get("forget_val_loss") for row in rows]
    retain_loss = [row.get("retain_val_loss") for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(rows) * 1.1), 8), sharex=True)

    axes[0].plot(labels, forget_acc, marker="o", label="forget val acc")
    axes[0].plot(labels, retain_acc, marker="x", linestyle="--", label="retain val acc")
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(labels, forget_loss, marker="o", label="forget val loss")
    axes[1].plot(labels, retain_loss, marker="x", linestyle="--", label="retain val loss")
    axes[1].set_ylabel("loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[1].tick_params(axis="x", rotation=45)
    fig.suptitle("EKFAC Step-Size Schedule Sweep")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run an EKFAC step-size schedule sweep.")
    parser.add_argument("--config", default="config_ekfac_label_based.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--schedules",
        default="0.005,0.003;0.005,0.005;0.005,0.007;0.007,0.0035",
        help="Semicolon-separated schedules. Example: '0.005,0.003;0.005,0.005'",
    )
    parser.add_argument("--dampings", default="0.1")
    parser.add_argument("--output-dir", default="./runs/ekfac_schedule_sweep")
    parser.add_argument("--fisher-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    generated_config_dir = output_dir / "configs"
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    schedules = parse_schedule_list(args.schedules)
    dampings = parse_float_list(args.dampings)
    rows = []

    for damping in dampings:
        for schedule in schedules:
            num_steps = len(schedule)
            tag = f"damp{format_float(damping)}_{format_schedule(schedule)}"
            run_dir = output_dir / tag
            cfg = OmegaConf.load(base_config_path)

            OmegaConf.update(cfg, "unlearning.step_size", schedule[0], merge=True)
            OmegaConf.update(cfg, "unlearning.damping", damping, merge=True)
            OmegaConf.update(cfg, "unlearning.num_unlearn_steps", num_steps, merge=True)
            OmegaConf.update(cfg, "unlearning.step_size_schedule", schedule, merge=True)
            OmegaConf.update(cfg, "unlearning.recompute_curvature_each_step", True, merge=True)
            OmegaConf.update(cfg, "checkpoint_dir", str(run_dir), merge=True)
            OmegaConf.update(cfg, "evaluation.use_test_during_unlearn", False, merge=True)

            if args.fisher_batch_size is not None:
                OmegaConf.update(cfg, "unlearning.fisher_batch_size", args.fisher_batch_size, merge=True)
            if args.num_workers is not None:
                OmegaConf.update(cfg, "num_workers", args.num_workers, merge=True)

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
            if args.dry_run:
                continue

            subprocess.run(command, check=True)

            result_path = run_dir / "unlearning_result.json"
            result = load_result(result_path)
            row = {
                "tag": tag,
                "damping": damping,
                "step_size_schedule": ",".join(f"{value:g}" for value in schedule),
                "num_unlearn_steps": num_steps,
                "checkpoint_dir": str(run_dir),
                **last_eval_metrics(result),
                **final_update_stats(result),
                **step_update_norms(result),
            }
            rows.append(row)

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
