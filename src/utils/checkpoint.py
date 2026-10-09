import os
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    step: int,
    checkpoint_dir: str,
    rank: int = 0,
    filename: str = "latest_moe_checkpoint.pt",
) -> Optional[str]:
    """
    Saves training state to disk. Only rank 0 persists to avoid concurrent write collisions.
    
    Args:
        model: PyTorch model.
        optimizer: PyTorch optimizer.
        epoch: Current epoch index.
        step: Global step counter.
        checkpoint_dir: Directory where checkpoint is stored.
        rank: Worker rank.
        filename: Checkpoint filename.
        
    Returns:
        Path to checkpoint if saved, else None.
    """
    if rank != 0:
        return None

    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, filename)

    # Use a temporary file and atomic rename to protect against mid-write crashes/disconnects
    temp_path = f"{save_path}.tmp"

    checkpoint_data = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(checkpoint_data, temp_path)
    os.replace(temp_path, save_path)
    print(f"[Rank {rank}] Safely saved checkpoint to {save_path} (epoch={epoch}, step={step})")
    return save_path


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[Optimizer],
    checkpoint_dir: str,
    device: torch.device,
    filename: str = "latest_moe_checkpoint.pt",
) -> Tuple[int, int]:
    """
    Loads latest checkpoint if available.
    
    Args:
        model: PyTorch model to restore weights into.
        optimizer: PyTorch optimizer to restore state into (optional).
        checkpoint_dir: Directory containing checkpoints.
        device: Target device for loaded tensors.
        filename: Target checkpoint filename.

    Returns:
        (start_epoch, global_step) tuple.
    """
    ckpt_path = os.path.join(checkpoint_dir, filename)
    if os.path.exists(ckpt_path):
        print(f"Discovered existing checkpoint at {ckpt_path}. Restoring state...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint.get("epoch", 0)
        step = checkpoint.get("step", 0)
        print(f"Resumed from epoch={epoch}, step={step}")
        return epoch, step
    else:
        print(f"No checkpoint found at {ckpt_path}. Starting clean from scratch.")
        return 0, 0
