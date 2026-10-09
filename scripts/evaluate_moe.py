#!/usr/bin/env python3
"""
Interactive text generation and expert routing inspection script.
"""
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.models.upcycling import upcycle_smollm2_to_moe
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate / inspect upcycled MoE model.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--prompt", type=str, default="What are the main kharif crops grown in Maharashtra?")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    data_cfg = config["data"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = upcycle_smollm2_to_moe(
        model_id=model_cfg["base_model_id"],
        num_experts=model_cfg["num_experts"],
        top_k=model_cfg["top_k"],
        device=str(device),
    )

    # Load checkpoint if exists
    load_checkpoint(model, None, data_cfg["shared_dir"], device)
    model.eval()

    # Format into ChatML
    chat_prompt = f"<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(device)

    print("\n--- Generating Response ---")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    response = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    print(response)


if __name__ == "__main__":
    main()
