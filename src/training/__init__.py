"""
Training loop, differential optimizer setup, and multi-process trainer.
"""
from src.training.optimizer import build_differential_optimizer_groups
from src.training.trainer import train_worker

__all__ = ["build_differential_optimizer_groups", "train_worker"]
