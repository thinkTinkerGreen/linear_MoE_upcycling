import json
from typing import Dict
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class ShardedChatMLDataset(Dataset):
    """
    Dataset loader for sharded ChatML JSON files.
    Applies tokenizer formatting and computes next-token prediction causal labels.
    """

    def __init__(
        self,
        shard_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ):
        self.shard_path = shard_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(shard_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        text = sample["text"]

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Mask padding tokens (-100 ignored by CrossEntropyLoss)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
