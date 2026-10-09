import json
import os
import random
from typing import List, Dict
from datasets import load_dataset


def build_and_shard_dataset(
    output_dir: str = "/content/drive/MyDrive/linear_MoE_upcycling",
    num_shards: int = 2,
    seed: int = 42,
) -> List[str]:
    """
    Curates multi-domain Indian contextual and nutritional datasets into ChatML
    format and divides into equal shards for parallel processing.
    
    Args:
        output_dir: Target directory where shards are written.
        num_shards: Number of parallel shards (default: 2).
        seed: Random seed for shuffling.

    Returns:
        List of generated shard file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    formatted_data: List[Dict[str, str]] = []

    print("Fetching Domain 1: Regional Geography & Farming in India (KisanVaani/agriculture-qa-english-only)...")
    try:
        ds_agri = load_dataset("KisanVaani/agriculture-qa-english-only", split="train")
        for item in ds_agri:
            q = item.get("question") or item.get("Question")
            a = item.get("answers") or item.get("Answer")
            if q and a:
                formatted_data.append({
                    "text": f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
                })
        print(f"Loaded {len(ds_agri)} Agriculture QA samples.")
    except Exception as e:
        print(f"Skipping Agriculture QA: {e}")

    print("Fetching Domain 2: Nutritional Facts (tatsu-lab/alpaca filtered)...")
    try:
        ds_nutri = load_dataset("tatsu-lab/alpaca", split="train")
        count_nutri = 0
        for item in ds_nutri:
            inst = item.get("instruction", "").lower()
            if any(k in inst for k in ["nutrition", "calories", "diet"]):
                formatted_data.append({
                    "text": f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
                })
                count_nutri += 1
        print(f"Loaded {count_nutri} Nutrition samples.")
    except Exception as e:
        print(f"Skipping Alpaca Nutrition: {e}")

    print("Fetching Domains 3 & 4: Urban Lifestyles / Occupation & Income in India (yahma/alpaca-cleaned filtered)...")
    try:
        ds_india = load_dataset("yahma/alpaca-cleaned", split="train")
        count_india = 0
        keywords = ["india", "mumbai", "delhi", "bengaluru", "farming", "rupees", "income"]
        for item in ds_india:
            inst = item.get("instruction", "").lower()
            if any(kw in inst for kw in keywords):
                formatted_data.append({
                    "text": f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
                })
                count_india += 1
        print(f"Loaded {count_india} India Context samples.")
    except Exception as e:
        print(f"Skipping India Context: {e}")

    # Deduplicate and shuffle
    unique_texts = list({d["text"] for d in formatted_data})
    random.seed(seed)
    random.shuffle(unique_texts)
    total_samples = len(unique_texts)
    print(f"Total unique curated samples: {total_samples}")

    shard_size = total_samples // num_shards
    shard_paths = []

    for i in range(num_shards):
        start_idx = i * shard_size
        end_idx = (i + 1) * shard_size if i < num_shards - 1 else total_samples
        shard_subset = [{"text": t} for t in unique_texts[start_idx:end_idx]]
        
        shard_path = os.path.join(output_dir, f"shard_{i}.json")
        with open(shard_path, "w", encoding="utf-8") as f:
            json.dump(shard_subset, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(shard_subset)} samples to {shard_path}")
        shard_paths.append(shard_path)

    return shard_paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Curate and shard dataset for MoE training")
    parser.add_argument("--output_dir", type=str, default="/content/drive/MyDrive/linear_MoE_upcycling")
    parser.add_argument("--num_shards", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_and_shard_dataset(args.output_dir, args.num_shards, args.seed)
