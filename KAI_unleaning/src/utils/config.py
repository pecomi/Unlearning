"""
config.yaml 파일을 로드하여 Config 클래스 인스턴스를 생성
"""
from pathlib import Path
from typing import Any, Optional

import torch
from omegaconf import DictConfig, OmegaConf


def get_best_device(device_str: Optional[str] = None) -> str:
    """
    CUDA > MPS > CPU 우선순위 기반에서 적합한 것 자동 반환
    """
    if device_str and device_str != "auto":
        device_lower = device_str.lower()
        if device_lower in ["cuda", "mps", "cpu"]:
            return device_lower
        raise ValueError(f"Invalid device: {device_str}. Use 'auto', 'cuda', 'mps', or 'cpu'")

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class Config:

    _instance: Optional["Config"] = None
    _config: DictConfig

    def __new__(cls, config_path: Optional[str] = None) -> "Config":
        """Singleton"""
        if cls._instance is None:
            if config_path is None:
                config_path = Path(__file__).parent.parent.parent / "config.yaml"
            cls._instance = super().__new__(cls)
            cls._instance._load(config_path)
        return cls._instance

    def _load(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        cfg = OmegaConf.load(config_path)

        device = cfg.get("device", "auto")
        if device == "auto":
            cfg.device = get_best_device()

        self._config = cfg

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return OmegaConf.select(self._config, key, default=default)
        except Exception:
            return default

    def set(self, key: str, value: Any) -> None:
        OmegaConf.update(self._config, key, value)

    def to_dict(self) -> dict:
        return OmegaConf.to_container(self._config)

    @property
    def config(self) -> DictConfig:
        return self._config

    def __repr__(self) -> str:
        return f"Config({OmegaConf.to_yaml(self._config)})"


def load_config(config_path: Optional[str] = None) -> Config:
    return Config(config_path)
