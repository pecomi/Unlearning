"""
logger 출력 관련 유틸
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn


class Logger:
    _instance: Optional["Logger"] = None
    _logger: logging.Logger
    _console: Console
    _wandb_run = None

    def __new__(cls, name: str = "KAI_Unlearning", level: str = "INFO") -> "Logger":
        """Singleton"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._setup(name, level)
        return cls._instance

    def _setup(self, name: str, level: str) -> None:
        self._console = Console(width=120)

        # Create logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.upper()))
        self._logger.handlers.clear()

        rich_handler = RichHandler(
            console=self._console,
            rich_tracebacks=True,
            show_path=False,
            show_time=True,
            markup=True,
            log_time_format="%Y-%m-%d %H:%M:%S",
        )
        rich_handler.setFormatter(
            logging.Formatter("[bold cyan]%(name)s[/bold cyan] - %(message)s")
        )
        self._logger.addHandler(rich_handler)

        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "experiment.log")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self._logger.addHandler(file_handler)

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    def debug(self, msg: str) -> None:
        self._logger.debug(msg)

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(f"[yellow]WARNING[/yellow]: {msg}")

    def error(self, msg: str) -> None:
        self._logger.error(f"[red]ERROR[/red]: {msg}")

    def critical(self, msg: str) -> None:
        self._logger.critical(f"[bold red]CRITICAL[/bold red]: {msg}")

    def success(self, msg: str) -> None:
        self._logger.info(f"[bold green]SUCCESS[/bold green]: {msg}")

    def print(self, msg: str) -> None:
        self._console.print(msg)

    def progress_bar(self, total: int, description: str = "") -> Progress:
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self._console,
        )

    def log_metrics(self, metrics: dict, step: int, prefix: str = "") -> None:
        msg_parts = []
        for key, value in metrics.items():
            msg_parts.append(f"[cyan]{prefix}{key}[/cyan]: {value:.4f}")

        self.info(f"Step {step} - " + " | ".join(msg_parts))

        # WandB logging
        if self._wandb_run is not None:
            prefixed_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
            self._wandb_run.log(prefixed_metrics, step=step)


    def init_wandb(self, config: dict, project: str, entity: str = None,
                   run_name: str = None, mode: str = "online") -> None:
        try:
            import wandb
        except ImportError:
            self.warning("wandb not installed. Skipping WandB integration.")
            return

        if mode == "disabled":
            self.info("WandB disabled. Skipping initialization.")
            return

        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            mode=mode,
            config=config,
        )
        self._wandb_run = wandb
        self.success("WandB initialized successfully.")


    def finish_wandb(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None
            self.info("WandB run finished.")

    def __repr__(self) -> str:
        return f"Logger(name={self._logger.name}, level={self._logger.level})"


# Global logger instance
_global_logger: Optional[Logger] = None


def get_logger(name: str = "KAI_Unlearning", level: str = "INFO") -> Logger:
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger(name, level)
    return _global_logger
