import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from data.factory import DatasetFactory
from metrics.calculator import MetricsCalculator
from models.factory import ModelFactory
from unlearning.factory import UnlearningFactory
from trainer import Trainer
from utils.logger import get_logger
from utils.config import load_config
from utils.seed import set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="KAI Unlearning Research Framework")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "unlearn", "retrain", "full", "compare"],
        default="full",
        help="Execution mode (retrain: train from scratch on retain data only)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint"
    )
    return parser.parse_args()


def main():

    args = parse_args()

    # config 파일 로드 및 설정
    config = load_config(args.config)
    cfg_dict = config.to_dict()

    device = config.get("device", "auto")
    cuda_visible_devices = config.get("cuda_visible_devices", None)
    if device == "cuda" and cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    seed = config.get("seed", 42)
    set_seed(seed, deterministic=True, benchmark=False, device=device)

    # 로깅
    logger = get_logger(level=config.get("logging.level", "INFO"))

    logger.print("[bold cyan][Configuration][/bold cyan]\n")
    logger.info(f"Execution mode: [yellow]{args.mode.upper()}[/yellow]")
    logger.info(f"Random seed: {seed}")

    if device == "cuda":
        if cuda_visible_devices is not None:
            logger.info(f"CUDA_VISIBLE_DEVICES: [yellow]{cuda_visible_devices}[/yellow]")
        logger.info(f"Device: [green]{device}[/green] ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})")
    elif device == "mps":
        logger.info(f"Device: [green]{device}[/green] (Apple Silicon GPU)")
    else:
        logger.info(f"Device: [green]{device}[/green]")

    # 데이터셋 설정
    data_config = config.get("data", {})
    evaluation_config = config.get("evaluation", {})
    training_config = config.get("training", {})
    dataset_name = data_config.get("name", "cifar10")
    data_dir = config.get("data_dir", "./data")

    logger.info(f"[cyan]Loading dataset:[/cyan] {dataset_name}")

    # 데이터셋 팩토리를 통해 config에 명시된 데이터셋 로드
    dataset = DatasetFactory.create(
        dataset_name,
        root=data_dir,
        download=True
    )

    # 데이터셋 로드
    dataset.load()

    # cuda only 옵션
    use_pin_memory = config.get("pin_memory", False) and device == "cuda"

    # Create splits
    splits = dataset.create_splits(
        forget_ratio=data_config.get("forget_ratio", 0.1),
        val_ratio=data_config.get("val_ratio", 0.1),
        batch_size=training_config.get("batch_size", 128),
        eval_batch_size=evaluation_config.get("batch_size", training_config.get("batch_size", 128)),
        num_workers=config.get("num_workers", 4),
        seed=seed,
        pin_memory=use_pin_memory,
        split_strategy=data_config.get("split_strategy", "random"),
        forget_labels=data_config.get("forget_labels", None)
    )

    train_loader = splits["train_loader"]
    forget_loader = splits["forget_loader"]
    forget_val_loader = splits.get("forget_val_loader")
    retain_loader = splits["retain_loader"]
    retain_val_loader = splits.get("retain_val_loader")
    val_loader = splits["val_loader"]
    test_loader = splits["test_loader"]

    evaluation_protocol = evaluation_config.get("protocol", "tuning")
    if evaluation_protocol not in {"tuning", "final"}:
        raise ValueError(
            "evaluation.protocol must be either 'tuning' or 'final' "
            f"(got {evaluation_protocol!r})."
        )
    if evaluation_protocol == "final" and data_config.get("val_ratio", 0.0) != 0:
        raise ValueError(
            "Final protocol uses the full training split and requires data.val_ratio: 0. "
            "Use a separate final config instead of reusing a tuning config."
        )

    logger.info(f"  Train samples: {len(train_loader.dataset)}")
    logger.info(f"  Forget samples: {len(forget_loader.dataset)}")
    if forget_val_loader is not None:
        logger.info(f"  Forget val samples: {len(forget_val_loader.dataset)}")
    logger.info(f"  Retain samples: {len(retain_loader.dataset)}")
    if retain_val_loader is not None:
        logger.info(f"  Retain val samples: {len(retain_val_loader.dataset)}")
    logger.info(f"  Validation samples: {len(val_loader.dataset)}")
    logger.info(f"  Test samples: {len(test_loader.dataset)}")
    logger.info(f"  Evaluation protocol: {evaluation_protocol}")


    # 모델 설정
    model_config = config.get("model", {})
    model_name = model_config.get("name", "resnet18")
    num_classes = model_config.get("num_classes", 10)
    logger.info(f"\n[cyan]Loading model:[/cyan] {model_name}")

    # 데이터셋 팩토리를 통해 config에 명시된 데이터셋 로드
    model = ModelFactory.create(
        model_name,
        num_classes=num_classes
    )
    model.create_model()

    # WandB 설정
    wandb_config = config.get("wandb", {})
    run_name_base = wandb_config.get("run_name_base", wandb_config.get("run_name", "unlearning-experiment"))
    mode_suffix = {
        "train": "train",
        "unlearn": "unlearn",
        "retrain": "retrain",
        "full": "full",
        "compare": "compare"
    }
    suffix = mode_suffix.get(args.mode, args.mode)
    cfg_dict["wandb"]["run_name"] = f"{run_name_base}-{suffix}"

    # 모델 학습을 위한 트레이너 객체 생성
    trainer = Trainer(model, cfg_dict, logger)

    # 체크포인트 저장 디렉토리
    run_checkpoint_dir = Path(config.get("checkpoint_dir", "./checkpoints"))
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train 모드
    if args.mode in ["train", "full"]:
        logger.print("\n[bold green]=== Training Phase ===[/bold green]")
        logger.info(f"Checkpoints will be saved to: {run_checkpoint_dir}")

        model.model = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=training_config.get("epochs", 100),
            optimizer_config=training_config.get("optimizer", {}),
            scheduler_config=training_config.get("scheduler", {})
        )

        # 체크포인트 저장
        checkpoint_path = run_checkpoint_dir / "trained_model.pth"
        model.save_checkpoint(
            str(checkpoint_path),
            epoch=training_config.get("epochs", 100)
        )
        logger.success(f"Saved trained model to: {checkpoint_path}")

        if evaluation_protocol == "final":
            logger.print("\n[cyan]▶ Evaluating Trained Model on Test Set[/cyan]")
            test_loss, test_acc = trainer.evaluate(test_loader)

            logger.log_metrics({
                "test_accuracy": test_acc,
                "test_loss": test_loss
            }, step=training_config.get("epochs", 100), prefix="train/")
        else:
            logger.info(
                "Skipping test-set evaluation during train mode. "
                "Use a separate final config for test reporting."
            )


    # Unlearn 모드
    if args.mode in ["unlearn", "full"]:
        logger.print("\n[bold yellow]=== Unlearning Phase ===[/bold yellow]")

        # pretrain 모델 로드
        if args.mode == "unlearn":
            checkpoint = args.checkpoint
            if checkpoint is None:
                checkpoint = config.get("baseline_checkpoint",
                                       str(Path(config.get("checkpoint_dir", "./checkpoints")) / "trained_model.pth"))
            logger.info(f"Loading model from: {checkpoint}")
            model.load_checkpoint(checkpoint)
        else:
            logger.info(f"Using model from training phase")

        # unlearning 관련 설정 로드
        unlearning_config = config.get("unlearning", {})
        unlearning_name = unlearning_config.get("name", "ssd")
        fisher_batch_size = unlearning_config.get("fisher_batch_size", 8)

        logger.info(f"[cyan]Unlearning Configuration:[/cyan]")
        logger.info(f"  • Method: {unlearning_name}")
        if unlearning_name == "ssd":
            logger.info(f"  • Dampening constant (λ): {unlearning_config.get('dampening_constant', 1.0)}")
            logger.info(f"  • Selection weighting (α): {unlearning_config.get('selection_weighting', 10.0)}")
        elif unlearning_name == "ekfac_influence":
            logger.info(f"  • Correction strength (gamma): {unlearning_config.get('correction_strength', 1.0)}")
            logger.info(f"  • Damping mode: {unlearning_config.get('damping_mode', 'absolute')}")
            logger.info(f"  • Absolute damping: {unlearning_config.get('damping', 0.05)}")
            logger.info(f"  • Relative damping ratio: {unlearning_config.get('damping_ratio', 0.1)}")
            logger.info(f"  • Damping floor: {unlearning_config.get('damping_floor', 1e-8)}")
            logger.info(f"  • Update unsupported parameters: {unlearning_config.get('update_unsupported_params', False)}")
            logger.info(f"  • Unlearn steps: {unlearning_config.get('num_unlearn_steps', 1)}")
            logger.info(f"  • Recompute curvature each step: {unlearning_config.get('recompute_curvature_each_step', False)}")
            logger.info(f"  • Eval interval: {unlearning_config.get('unlearn_eval_interval', 1)}")
            logger.info(f"  • Max eval batches: {unlearning_config.get('max_eval_batches', None)}")
            logger.info(f"  • Forget update mode: {unlearning_config.get('forget_update_mode', 'full_batch')}")
        logger.info(f"  • Fisher batch size: {fisher_batch_size}\n")

        # unlearning 팩토리를 통해 config에 명시된 언러닝 방식 로드
        unlearning = UnlearningFactory.create(
            unlearning_name,
            model=model,
            dampening_constant=unlearning_config.get("dampening_constant", 1.0),
            selection_weighting=unlearning_config.get("selection_weighting", 10.0),
            correction_strength=unlearning_config.get("correction_strength", None),
            step_size=unlearning_config.get("step_size", None),
            damping_mode=unlearning_config.get("damping_mode", "absolute"),
            damping=unlearning_config.get("damping", 0.05),
            damping_ratio=unlearning_config.get("damping_ratio", 0.1),
            damping_floor=unlearning_config.get("damping_floor", 1e-8),
            regularization_curvature=unlearning_config.get(
                "regularization_curvature",
                config.get("training", {}).get("optimizer", {}).get("weight_decay", 0.0),
            ),
            update_unsupported_params=unlearning_config.get("update_unsupported_params", False),
            num_unlearn_steps=unlearning_config.get("num_unlearn_steps", 1),
            recompute_curvature_each_step=unlearning_config.get("recompute_curvature_each_step", False),
            unlearn_eval_interval=unlearning_config.get("unlearn_eval_interval", 1),
            max_eval_batches=unlearning_config.get("max_eval_batches", None),
            max_curvature_batches=unlearning_config.get("max_curvature_batches", None),
            max_forget_batches=unlearning_config.get("max_forget_batches", None),
            forget_update_mode=unlearning_config.get("forget_update_mode", "full_batch"),
            device=config.get("device", "cuda")
        )

        # The post-deletion objective is defined on the retain set, so its
        # curvature and the |Df|/|Dr| removal scale must use retain data.
        unlearn_train_dataset = retain_loader.dataset
        unlearn_forget_dataset = forget_loader.dataset
        use_full_split_for_unlearn = evaluation_protocol == "final"
        if use_full_split_for_unlearn:
            if retain_val_loader is not None and len(retain_val_loader.dataset) > 0:
                unlearn_train_dataset = ConcatDataset(
                    [retain_loader.dataset, retain_val_loader.dataset]
                )
                logger.info(
                    f"Using retain train + retain validation for final unlearning curvature: "
                    f"{len(unlearn_train_dataset)} samples"
                )

            if forget_val_loader is not None and len(forget_val_loader.dataset) > 0:
                unlearn_forget_dataset = ConcatDataset([
                    forget_loader.dataset,
                    forget_val_loader.dataset,
                ])
                logger.info(
                    f"Using forget train + forget validation for final forget gradient: "
                    f"{len(unlearn_forget_dataset)} samples"
                )

        # batch를 위한 DataLoader 생성
        fisher_train_loader = DataLoader(
            unlearn_train_dataset,
            batch_size=fisher_batch_size,
            shuffle=False,
            num_workers=config.get("num_workers", 4),
            pin_memory=use_pin_memory,
        )
        fisher_forget_loader = DataLoader(
            unlearn_forget_dataset,
            batch_size=fisher_batch_size,
            shuffle=False,
            num_workers=config.get("num_workers", 4),
            pin_memory=use_pin_memory,
        )

        unlearn_eval_loaders = {}
        if use_full_split_for_unlearn:
            logger.info(
                "Skipping validation-set monitoring during final unlearn protocol because "
                "validation data may be included in the final training data."
            )
        elif data_config.get("split_strategy", "random") == "label_based":
            unlearn_eval_loaders = {
                "retain_val": retain_val_loader,
                "forget_val": forget_val_loader,
                "val": val_loader,
            }
        else:
            unlearn_eval_loaders = {
                "val": val_loader,
            }

        result = unlearning.unlearn(
            forget_loader=fisher_forget_loader,
            train_loader=fisher_train_loader,
            eval_loaders=unlearn_eval_loaders,
            logger=logger
        )

        # 체크포인트 저장
        checkpoint_path = run_checkpoint_dir / "unlearned_model.pth"
        logger.info(f"\n[cyan]Saving unlearned model...[/cyan]")
        unlearning.save_unlearned_model(str(checkpoint_path))
        logger.success(f"  ✓ Saved to: {checkpoint_path}")

        result_path = run_checkpoint_dir / "unlearning_result.json"
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.success(f"  ✓ Saved unlearning result to: {result_path}")

        if evaluation_protocol == "final":
            logger.print("\n[cyan]▶ Evaluating Unlearned Model on Test Set[/cyan]")
            test_loss, test_acc = trainer.evaluate(test_loader)

            logger.print("\n[cyan]▶ Computing Unlearning Metrics[/cyan]")
            metrics_calculator = MetricsCalculator(
                device=config.get("device", "cuda"),
                enable_viz=config.get("evaluation.enable_visualizations", True),
                dataset_name=dataset_name
            )
            unlearn_metrics = metrics_calculator.compute_all_metrics(
                model=model.model,
                retain_loader=retain_loader,
                forget_loader=forget_loader,
                test_loader=test_loader,
                logger=logger
            )
            logger.log_metrics(unlearn_metrics, step=0, prefix="unlearn/")
        else:
            logger.info(
                "Skipping test-set evaluation during unlearn mode. "
                "Use validation metrics for tuning, then run a separate final config for test reporting."
            )


    # Retrain 모드
    if args.mode == "retrain":
        logger.print("\n[bold magenta]=== Retrain Phase (Gold Standard) ===[/bold magenta]")
        logger.info("[yellow]Training model from scratch on RETAIN data only[/yellow]")
        logger.info(f"Checkpoints will be saved to: {run_checkpoint_dir}")

        # 새로운 모델 생성
        retrained_model = ModelFactory.create(
            model_name,
            num_classes=num_classes
        )
        retrained_model.create_model()

        # retrain을 위한 트레이너 객체 생성
        retrain_trainer = Trainer(retrained_model, cfg_dict, logger)

        retrain_train_loader = retain_loader
        if evaluation_protocol == "final" and retain_val_loader is not None and len(retain_val_loader.dataset) > 0:
            retrain_dataset = ConcatDataset([retain_loader.dataset, retain_val_loader.dataset])
            retrain_train_loader = DataLoader(
                retrain_dataset,
                batch_size=training_config.get("batch_size", 128),
                shuffle=True,
                num_workers=config.get("num_workers", 4),
                pin_memory=use_pin_memory,
            )
            logger.info(
                f"Training final retrained model on retain train + retain val: "
                f"{len(retrain_train_loader.dataset)} samples"
            )

        # retrain 실행
        retrained_model.model = retrain_trainer.train(
            train_loader=retrain_train_loader,
            val_loader=val_loader,
            epochs=training_config.get("epochs", 100),
            optimizer_config=training_config.get("optimizer", {}),
            scheduler_config=training_config.get("scheduler", {})
        )

        # 체크포인트 저장
        checkpoint_path = run_checkpoint_dir / "retrained_model.pth"
        retrain_trainer.save_model(str(checkpoint_path))
        logger.success(f"Saved retrained model to: {checkpoint_path}")

        logger.print("\n[cyan]▶ Evaluating Retrained Model on Validation Sets[/cyan]")
        metrics_calculator = MetricsCalculator(
            device=config.get("device", "cuda"),
            enable_viz=config.get("evaluation.enable_visualizations", True),
            dataset_name=dataset_name
        )
        retrain_val_metrics = {}

        val_metrics = metrics_calculator.compute_classification_metrics(
            retrained_model.model,
            val_loader
        )
        retrain_val_metrics.update({
            f"val_{key}": value
            for key, value in val_metrics.items()
        })

        if retain_val_loader is not None and len(retain_val_loader.dataset) > 0:
            retain_val_metrics = metrics_calculator.compute_classification_metrics(
                retrained_model.model,
                retain_val_loader
            )
            retrain_val_metrics.update({
                f"retain_val_{key}": value
                for key, value in retain_val_metrics.items()
            })

        if forget_val_loader is not None and len(forget_val_loader.dataset) > 0:
            forget_val_metrics = metrics_calculator.compute_classification_metrics(
                retrained_model.model,
                forget_val_loader
            )
            retrain_val_metrics.update({
                f"forget_val_{key}": value
                for key, value in forget_val_metrics.items()
            })

        logger.log_metrics(
            retrain_val_metrics,
            step=training_config.get("epochs", 100),
            prefix="retrain_val/"
        )

        if evaluation_protocol == "final":
            logger.print("\n[cyan]▶ Evaluating Retrained Model on Test Set[/cyan]")
            test_loss, test_acc = retrain_trainer.evaluate(test_loader)

            logger.log_metrics({
                "test_accuracy": test_acc,
                "test_loss": test_loss
            }, step=training_config.get("epochs", 100), prefix="retrain/")
        else:
            logger.info(
                "Skipping test-set evaluation during retrain mode. "
                "Use validation metrics for tuning, then run a separate final config for test reporting."
            )

        retrain_trainer.finish()


    # Compare phase (Unlearned vs Retrained)
    if args.mode == "compare":
        logger.print("\n[bold magenta]=== Compare (Unlearned vs Retrained) ===[/bold magenta]")

        checkpoint_dir = Path(config.get("checkpoint_dir", "./checkpoints"))

        # unlearned 모델 로드
        unlearned_checkpoint = args.checkpoint or str(checkpoint_dir / "unlearned_model.pth")
        if not Path(unlearned_checkpoint).exists():
            logger.error(f"Unlearned model not found at: {unlearned_checkpoint}")
            logger.error("Please run with --mode unlearn first.")
            return

        logger.info(f"Loading unlearned model from: {unlearned_checkpoint}")
        model.load_checkpoint(unlearned_checkpoint)

        # retrained 모델 로드
        retrained_checkpoint = str(checkpoint_dir / "retrained_model.pth")
        if not Path(retrained_checkpoint).exists():
            logger.error(f"Retrained model not found at: {retrained_checkpoint}")
            logger.error("Please run with --mode retrain first to create the gold standard model.")
            return

        logger.info(f"Loading retrained model from: {retrained_checkpoint}")

        retrained_model = ModelFactory.create(
            model_name,
            num_classes=num_classes
        )
        retrained_model.create_model()
        retrained_model.load_checkpoint(retrained_checkpoint)


        # 모델 비교를 위한 메트릭 계산
        metrics_calculator = MetricsCalculator(
            device=config.get("device", "cuda"),
            enable_viz=config.get("evaluation.enable_visualizations", True),
            dataset_name=dataset_name
        )

        use_test_during_compare = evaluation_protocol == "final"
        compare_test_loader = test_loader if use_test_during_compare else None
        if not use_test_during_compare:
            logger.info(
                "Skipping test-set evaluation during compare mode. "
                "Use evaluation.protocol=final in a separate final config for final reporting."
            )

        all_metrics = metrics_calculator.compare_with_gold_standard(
            unlearned_model=model.model,
            retrained_model=retrained_model.model,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            test_loader=compare_test_loader,
            val_loader=val_loader,
            retain_val_loader=retain_val_loader,
            forget_val_loader=forget_val_loader,
            logger=logger
        )

        logger.log_metrics(all_metrics, step=0, prefix="compare/")

    # 정리
    trainer.finish()
    logger.success("\n[bold green]Experiment completed successfully![/bold green]\n")



if __name__ == "__main__":
    main()
