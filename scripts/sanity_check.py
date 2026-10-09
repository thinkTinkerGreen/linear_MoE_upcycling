import sys
import os
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, AutoModelForCausalLM

from src.models.upcycling import upcycle_smollm2_to_moe
from src.training.optimizer import build_differential_optimizer_groups


def run_sanity_check():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Pre-Flight Sanity Check on {device} ===")

    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    print("1. Loading Tokenizer and Base Model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("2. Upcycling to 4 Experts (Top-2)...")
    model = upcycle_smollm2_to_moe(model_id, num_experts=4, top_k=2, device=str(device))
    model.train()

    print("3. Initializing Differential Optimizer...")
    param_groups = build_differential_optimizer_groups(model)
    optimizer = torch.optim.AdamW(param_groups)

    print("4. Executing 5 Simulated Training Steps (with gradient accumulation)...")
    grad_accum_steps = 4
    optimizer.zero_grad(set_to_none=True)

    dummy_prompts = [
        "<|im_start|>user\nWhat crops grow best in black soil in Maharashtra?<|im_end|>\n<|im_start|>assistant\nCotton and soybean thrive in black soil.<|im_end|>",
        "<|im_start|>user\nWhat is the protein content in 100g of paneer?<|im_end|>\n<|im_start|>assistant\n100 grams of paneer provides around 18 grams of protein.<|im_end|>",
        "<|im_start|>user\nHow do farmers in Punjab manage winter wheat harvest?<|im_end|>\n<|im_start|>assistant\nFarmers use combine harvesters and manage stubble.<|im_end|>",
        "<|im_start|>user\nWhat are the average household income trends in Bengaluru?<|im_end|>\n<|im_start|>assistant\nAverage household income has risen with tech jobs.<|im_end|>",
        "<|im_start|>user\nWhat dietary fiber sources are common in traditional Indian meals?<|im_end|>\n<|im_start|>assistant\nWhole pulses, millets, and vegetables are rich in fiber.<|im_end|>",
    ]

    for step_idx, prompt in enumerate(dummy_prompts):
        encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        lm_loss = outputs.loss

        # Collect auxiliary load balancing loss
        aux_loss = torch.tensor(0.0, device=device)
        for layer in model.model.layers:
            if hasattr(layer.mlp, "aux_loss"):
                aux_loss += layer.mlp.aux_loss.to(device)

        batch_loss = lm_loss + 0.01 * aux_loss

        # Assertions
        assert not torch.isnan(lm_loss), f"Sanity check FAILED: LM loss is NaN at step {step_idx}"
        assert not torch.isinf(lm_loss), f"Sanity check FAILED: LM loss is Inf at step {step_idx}"
        assert not torch.isnan(aux_loss), f"Sanity check FAILED: Aux loss is NaN at step {step_idx}"
        assert lm_loss.item() > 0.0, f"Sanity check FAILED: LM loss is <= 0 at step {step_idx}"

        total_loss = batch_loss / grad_accum_steps
        total_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        assert not torch.isnan(grad_norm), f"Sanity check FAILED: Grad norm is NaN at step {step_idx}"
        assert not torch.isinf(grad_norm), f"Sanity check FAILED: Grad norm is Inf at step {step_idx}"

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        print(f"   [Step {step_idx + 1}/5] OK | LM Loss: {lm_loss.item():.4f} | Aux Loss: {aux_loss.item():.4f} | Grad Norm: {grad_norm.item():.4f}")

    print("\n5. Checking Weight Tensors for NaNs...")
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            raise AssertionError(f"Sanity check FAILED: Parameter {name} contains NaN values!")

    print("=================================================================")
    print("ALL SANITY CHECKS PASSED: Model, routing, backprop, and optimizer are fully verified!")
    print("=================================================================\n")
    return True


if __name__ == "__main__":
    try:
        run_sanity_check()
        sys.exit(0)
    except Exception as e:
        print(f"\nSANITY CHECK ERROR: {e}")
        sys.exit(1)
