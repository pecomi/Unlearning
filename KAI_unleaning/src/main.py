import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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
        choices=["train", "unlearn", "retrain", "evaluate", "full", "compare"],
        default="full",
        help="Execution mode (retrain: train from scratch on retain data only)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["pretrain", "unlearned", "retrained"],
        default=None,
        help="Model type for evaluation mode (pretrain: evaluate pretrained model, unlearned: evaluate unlearned model, retrained: evaluate retrained model)"
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
        batch_size=config.get("training.batch_size", 128),
        num_workers=config.get("num_workers", 4),
        seed=seed,
        pin_memory=use_pin_memory,
        split_strategy=data_config.get("split_strategy", "random"),
        forget_labels=data_config.get("forget_labels", None)
    )

    train_loader = splits["train_loader"]
    forget_loader = splits["forget_loader"]
    retain_loader = splits["retain_loader"]
    val_loader = splits["val_loader"] # validation set은 현재 사용하지 않아 코드는 사용치 않고, config에서 val_ratio를 0으로 명시함.
    test_loader = splits["test_loader"]

    logger.info(f"  Train samples: {len(train_loader.dataset)}")
    logger.info(f"  Forget samples: {len(forget_loader.dataset)}")
    logger.info(f"  Retain samples: {len(retain_loader.dataset)}")
    logger.info(f"  Test samples: {len(test_loader.dataset)}")


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
        "evaluate": "evaluate",
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

        training_config = config.get("training", {})
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

        # Test set 평가
        logger.print("\n[cyan]▶ Evaluating Trained Model on Test Set[/cyan]")
        test_loss, test_acc = trainer.evaluate(test_loader)

        logger.log_metrics({
            "train/test_accuracy": test_acc,
            "train/test_loss": test_loss
        }, step=training_config.get("epochs", 100), prefix="train/")


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
        logger.info(f"  • Dampening constant (λ): {unlearning_config.get('dampening_constant', 1.0)}")
        logger.info(f"  • Selection weighting (α): {unlearning_config.get('selection_weighting', 10.0)}")
        logger.info(f"  • Fisher batch size: {fisher_batch_size}\n")

        # unlearning 팩토리를 통해 config에 명시된 언러닝 방식 로드
        unlearning = UnlearningFactory.create(
            unlearning_name,
            model=model,
            dampening_constant=unlearning_config.get("dampening_constant", 1.0),
            selection_weighting=unlearning_config.get("selection_weighting", 10.0),
            device=config.get("device", "cuda")
        )

        # batch를 위한 DataLoader 생성
        fisher_train_loader = DataLoader(
            train_loader.dataset,
            batch_size=fisher_batch_size,
            shuffle=False,
            num_workers=config.get("num_workers", 4),
            pin_memory=use_pin_memory,
        )
        fisher_forget_loader = DataLoader(
            forget_loader.dataset,
            batch_size=fisher_batch_size,
            shuffle=False,
            num_workers=config.get("num_workers", 4),
            pin_memory=use_pin_memory,
        )

        result = unlearning.unlearn(
            forget_loader=fisher_forget_loader,
            train_loader=fisher_train_loader,
            logger=logger
        )

        # 체크포인트 저장
        checkpoint_path = run_checkpoint_dir / "unlearned_model.pth"
        logger.info(f"\n[cyan]Saving unlearned model...[/cyan]")
        unlearning.save_unlearned_model(str(checkpoint_path))
        logger.success(f"  ✓ Saved to: {checkpoint_path}")

        # Test set 평가
        logger.print("\n[cyan]▶ Evaluating Unlearned Model on Test Set[/cyan]")
        test_loss, test_acc = trainer.evaluate(test_loader)

        logger.log_metrics({
            "unlearn/test_accuracy": test_acc,
            "unlearn/test_loss": test_loss
        }, step=0, prefix="unlearn/")


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

        training_config = config.get("training", {})

        # retrain 실행
        retrained_model.model = retrain_trainer.train(
            train_loader=retain_loader,
            val_loader=val_loader,
            epochs=training_config.get("epochs", 100),
            optimizer_config=training_config.get("optimizer", {}),
            scheduler_config=training_config.get("scheduler", {})
        )

        # 체크포인트 저장
        checkpoint_path = run_checkpoint_dir / "retrained_model.pth"
        retrain_trainer.save_model(str(checkpoint_path))
        logger.success(f"Saved retrained model to: {checkpoint_path}")

        # Test set 평가
        logger.print("\n[cyan]▶ Evaluating Retrained Model on Test Set[/cyan]")
        test_loss, test_acc = retrain_trainer.evaluate(test_loader)

        logger.log_metrics({
            "retrain/test_accuracy": test_acc,
            "retrain/test_loss": test_loss
        }, step=training_config.get("epochs", 100), prefix="retrain/")

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
            dataset_name=dataset_name
        )

        all_metrics = metrics_calculator.compare_with_gold_standard(
            unlearned_model=model.model,
            retrained_model=retrained_model.model,
            retain_loader=retain_loader,
            forget_loader=forget_loader,
            test_loader=test_loader,
            logger=logger
        )

        logger.log_metrics(all_metrics, step=0, prefix="compare/")

    # 정리
    trainer.finish()
    logger.success("\n[bold green]Experiment completed successfully![/bold green]\n")



if __name__ == "__main__":
    main()
