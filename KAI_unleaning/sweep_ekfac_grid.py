import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


def parse_float_list(value: str):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def format_value(value) -> str:
    return str(value).replace(".", "p").replace("-", "m").replace("/", "_")


def load_result(result_path: Path) -> dict:
    with result_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def last_eval_metrics(result: dict) -> dict:
    history = result.get("eval_history", [])
    return history[-1] if history else {}


def final_update_stats(result: dict) -> dict:
    update_stats = result.get("update_stats", {})
    forget_stats = result.get("forget_stats", {})
    return {
        "forget_batches": forget_stats.get("batches"),
        "forget_samples": forget_stats.get("samples"),
        "forget_grad_norm": forget_stats.get("grad_norm"),
        "update_norm": update_stats.get("update_norm"),
        "raw_update_norm": update_stats.get("raw_update_norm"),
        "nonfinite_update_tensors": update_stats.get("nonfinite_update_tensors"),
        "updates_applied": update_stats.get("updates_applied", 1),
    }


def write_summary_csv(rows, output_path: Path) -> None:
    metric_fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key.endswith("_accuracy") or key.endswith("_loss") or key.endswith("_samples")
        }
    )
    fieldnames = [
        "tag",
        "step_size",
        "damping",
        "forget_update_mode",
        "fisher_batch_size",
        "num_unlearn_steps",
        "checkpoint_dir",
        "forget_batches",
        "forget_samples",
        "forget_grad_norm",
        "update_norm",
        "raw_update_norm",
        "nonfinite_update_tensors",
        "updates_applied",
        *metric_fields,
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="Run a grid search over EKFAC unlearning settings.")
    parser.add_argument("--config", default="config_ekfac_label_based.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--step-sizes", default="0.004,0.007")
    parser.add_argument("--dampings", default="0.05")
    parser.add_argument("--forget-update-modes", default="full_batch,minibatch")
    parser.add_argument("--fisher-batch-sizes", default="128")
    parser.add_argument("--num-unlearn-steps", default="1")
    parser.add_argument("--max-curvature-batches", type=int, default=None)
    parser.add_argument("--max-forget-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", default="./runs/ekfac_grid_search")
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    generated_config_dir = output_dir / "configs"
    generated_config_dir.mkdir(parents=True, exist_ok=True)

    step_sizes = parse_float_list(args.step_sizes)
    dampings = parse_float_list(args.dampings)
    forget_update_modes = parse_str_list(args.forget_update_modes)
    fisher_batch_sizes = parse_int_list(args.fisher_batch_sizes)
    num_unlearn_steps_values = parse_int_list(args.num_unlearn_steps)

    rows = []
    grid = itertools.product(
        step_sizes,
        dampings,
        forget_update_modes,
        fisher_batch_sizes,
        num_unlearn_steps_values,
    )

    for step_size, damping, forget_update_mode, fisher_batch_size, num_unlearn_steps in grid:
        if forget_update_mode not in {"full_batch", "minibatch"}:
            raise ValueError(f"Unsupported forget_update_mode: {forget_update_mode}")

        tag = (
            f"step{format_value(f'{step_size:g}')}"
            f"_damp{format_value(f'{damping:g}')}"
            f"_{forget_update_mode}"
            f"_fb{fisher_batch_size}"
            f"_n{num_unlearn_steps}"
        )
        run_dir = output_dir / tag
        cfg = OmegaConf.load(base_config_path)

        OmegaConf.update(cfg, "unlearning.step_size", step_size, merge=True)
        OmegaConf.update(cfg, "unlearning.damping", damping, merge=True)
        OmegaConf.update(cfg, "unlearning.forget_update_mode", forget_update_mode, merge=True)
        OmegaConf.update(cfg, "unlearning.fisher_batch_size", fisher_batch_size, merge=True)
        OmegaConf.update(cfg, "unlearning.num_unlearn_steps", num_unlearn_steps, merge=True)
        OmegaConf.update(
            cfg,
            "unlearning.step_size_schedule",
            [step_size] * num_unlearn_steps,
            merge=True,
        )
        OmegaConf.update(cfg, "checkpoint_dir", str(run_dir), merge=True)

        if args.max_curvature_batches is not None:
            OmegaConf.update(cfg, "unlearning.max_curvature_batches", args.max_curvature_batches, merge=True)
        if args.max_forget_batches is not None:
            OmegaConf.update(cfg, "unlearning.max_forget_batches", args.max_forget_batches, merge=True)
        if args.max_eval_batches is not None:
            OmegaConf.update(cfg, "unlearning.max_eval_batches", args.max_eval_batches, merge=True)
        if args.num_workers is not None:
            OmegaConf.update(cfg, "num_workers", args.num_workers, merge=True)
        if args.wandb_mode is not None:
            OmegaConf.update(cfg, "wandb.mode", args.wandb_mode, merge=True)

        run_name_base = OmegaConf.select(cfg, "wandb.run_name_base", default="EKFACInfluence")
        OmegaConf.update(cfg, "wandb.run_name_base", f"{run_name_base}-{tag}", merge=True)

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
        rows.append({
            "tag": tag,
            "step_size": step_size,
            "damping": damping,
            "forget_update_mode": forget_update_mode,
            "fisher_batch_size": fisher_batch_size,
            "num_unlearn_steps": num_unlearn_steps,
            "checkpoint_dir": str(run_dir),
            **last_eval_metrics(result),
            **final_update_stats(result),
        })

    if rows:
        summary_csv = output_dir / "summary.csv"
        write_summary_csv(rows, summary_csv)
        print(f"\nSaved summary CSV: {summary_csv}")
    else:
        print("\nDry run complete. Generated configs only.")


if __name__ == "__main__":
    main()
