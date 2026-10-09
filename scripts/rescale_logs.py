#!/usr/bin/env python3
"""
Post-processing utility to normalize training logs.
Divides logged metrics by log_interval (20) to reflect the true per-token cross-entropy loss,
and plots the training convergence curves.
"""
import os
import argparse
import pandas as pd


def rescale_and_clean_logs(log_dir: str, factor: float = 20.0):
    """
    Reads all training CSV logs in log_dir, rescales lm_loss, aux_loss,
    and total_loss by the scale factor (20), and saves clean normalized copies.
    """
    log_files = [f for f in os.listdir(log_dir) if f.startswith("training_log_worker_") and f.endswith(".csv")]
    if not log_files:
        print(f"No worker CSV logs found in {log_dir}")
        return

    print(f"Normalizing {len(log_files)} training logs by factor of {factor}...")

    for fname in log_files:
        raw_path = os.path.join(log_dir, fname)
        df = pd.read_csv(raw_path)

        if "lm_loss" not in df.columns:
            continue

        # Check if already normalized (if first loss is around ~2-5, skip)
        if df["lm_loss"].iloc[0] < 10.0:
            print(f"Log {fname} appears already normalized (first loss={df['lm_loss'].iloc[0]:.2f}). Skipping.")
            continue

        # Normalize metrics to true per-token averages
        df["lm_loss"] = (df["lm_loss"] / factor).round(4)
        df["aux_loss"] = (df["aux_loss"] / factor).round(4)
        df["total_loss"] = (df["total_loss"] / factor).round(4)

        # Overwrite with clean normalized values
        df.to_csv(raw_path, index=False)
        print(f"Successfully normalized {fname}:")
        print(f"   Initial LM Loss: {df['lm_loss'].iloc[0]:.4f} -> Final LM Loss: {df['lm_loss'].iloc[-1]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize training logs to true per-token loss.")
    parser.add_argument("--log_dir", type=str, default="/content/drive/MyDrive/linear_MoE_upcycling")
    parser.add_argument("--factor", type=float, default=20.0)
    args = parser.parse_args()

    rescale_and_clean_logs(args.log_dir, args.factor)
