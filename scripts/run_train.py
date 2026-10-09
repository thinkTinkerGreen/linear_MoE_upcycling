#!/usr/bin/env python3
"""
Entrypoint script for launching dual-process training.
"""
import argparse
import torch.multiprocessing as mp

from src.utils.config import load_config
from src.training.trainer import train_worker


def main():
    parser = argparse.ArgumentParser(description="Launch parallel MoE training.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of parallel processes (default: 2).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    num_workers = args.num_workers

    print(f"Launching {num_workers} parallel workers using start method 'spawn'...")
    mp.set_start_method("spawn", force=True)

    processes = []
    for rank in range(num_workers):
        p = mp.Process(target=train_worker, args=(rank, config))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All worker processes completed.")


if __name__ == "__main__":
    main()
