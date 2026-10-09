#!/usr/bin/env python3
"""
Save standalone fine-tuned MoE model weights and config for easy sharing and deployment.
"""
import argparse
import os
import torch
from transformers import AutoTokenizer

from src.models.upcycling import upcycle_smollm2_to_moe
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config


def export_moe_model(config_path: str, output_dir: str = None):
    config = load_config(config_path)
    model_cfg = config["model"]
    data_cfg = config["data"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_model_id = model_cfg["base_model_id"]
    shared_dir = data_cfg["shared_dir"]
    if output_dir is None:
        output_dir = os.path.join(shared_dir, "final_moe_model")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Exporting model to: {output_dir}")

    # 1. Save Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.save_pretrained(output_dir)
    print("Tokenizer saved.")

    # 2. Build MoE model and load trained checkpoint
    model = upcycle_smollm2_to_moe(
        model_id=base_model_id,
        num_experts=model_cfg["num_experts"],
        top_k=model_cfg["top_k"],
        device=str(device),
    )
    load_checkpoint(model, None, shared_dir, device)
    model.eval()

    # 3. Save weights
    weights_path = os.path.join(output_dir, "moe_model_weights.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"Model state dict saved to {weights_path}")

    # 4. Save metadata config
    metadata_path = os.path.join(output_dir, "moe_config.json")
    import json
    with open(metadata_path, "w") as f:
        json.dump({
            "base_model": base_model_id,
            "architecture": "SparseMoE-Top2",
            "num_experts": model_cfg["num_experts"],
            "top_k": model_cfg["top_k"],
            "total_layers": len(model.model.layers),
            "torch_dtype": str(next(model.parameters()).dtype)
        }, f, indent=2)

    print(f"Export complete! All files saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export final trained MoE model.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    export_moe_model(args.config, args.output_dir)
