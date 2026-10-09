import os
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.models.upcycling import upcycle_smollm2_to_moe
from src.data.dataset import ShardedChatMLDataset
from src.training.optimizer import build_differential_optimizer_groups
from src.utils.checkpoint import save_checkpoint, load_checkpoint


def train_worker(rank: int, config: Dict[str, Any]):
    """
    Worker process function for dual-process training.
    """
    # Prevent CUDA fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Worker {rank}] Starting execution on {device}")

    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    opt_cfg = config["optimizer"]

    # 1. Tokenizer
    base_model_id = model_cfg["base_model_id"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Sharded Dataset & DataLoader
    shard_file = os.path.join(data_cfg["shared_dir"], f"shard_{rank}.json")
    if not os.path.exists(shard_file):
        raise FileNotFoundError(f"[Worker {rank}] Shard file not found at {shard_file}. Run curate.py first!")

    dataset = ShardedChatMLDataset(
        shard_path=shard_file,
        tokenizer=tokenizer,
        max_length=data_cfg["max_seq_length"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["local_batch_size"],
        shuffle=True,
        drop_last=True,
    )

    # 3. Model Upcycling
    model = upcycle_smollm2_to_moe(
        model_id=base_model_id,
        num_experts=model_cfg["num_experts"],
        top_k=model_cfg["top_k"],
        device=str(device),
    )

    # 4. Optimizer Setup
    param_groups = build_differential_optimizer_groups(
        model=model,
        lr_base=float(opt_cfg["lr_base"]),
        lr_experts=float(opt_cfg["lr_experts"]),
        lr_router=float(opt_cfg["lr_router"]),
        weight_decay=float(opt_cfg["weight_decay"]),
    )
    optimizer = torch.optim.AdamW(param_groups)

    # 5. Fault-Tolerant Checkpoint Resumption
    shared_dir = data_cfg["shared_dir"]
    start_epoch, global_step = load_checkpoint(model, optimizer, shared_dir, device)

    # 6. Training Loop Setup
    model.train()

    num_epochs = train_cfg["num_epochs"]
    grad_accum_steps = train_cfg["grad_accum_steps"]
    aux_loss_coef = float(train_cfg["aux_loss_coef"])
    max_grad_norm = float(train_cfg["max_grad_norm"])
    log_interval = train_cfg["log_interval"]
    save_interval = train_cfg["save_interval"]

    log_file_path = os.path.join(shared_dir, f"training_log_worker_{rank}.csv")
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write("timestamp,epoch,step,lm_loss,aux_loss,total_loss\n")

    print(f"[Worker {rank}] Ready to train for {num_epochs} epochs (Starting epoch {start_epoch}, step {global_step}).")
    print(f"[Worker {rank}] Logging metrics to {log_file_path}")

    for epoch in range(start_epoch, num_epochs):
        optimizer.zero_grad(set_to_none=True)
        running_lm_loss = 0.0
        running_aux_loss = 0.0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            lm_loss = outputs.loss

            # Sum auxiliary load-balancing loss across all decoder MoE layers
            aux_loss = torch.tensor(0.0, device=device)
            for layer in model.model.layers:
                if hasattr(layer.mlp, "aux_loss"):
                    aux_loss += layer.mlp.aux_loss.to(device)

            batch_total_loss = lm_loss + aux_loss_coef * aux_loss

            if torch.isnan(batch_total_loss) or torch.isinf(batch_total_loss):
                print(f"[Worker {rank}] Warning: NaN/Inf detected in loss at step {step}. Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            total_loss = batch_total_loss / grad_accum_steps
            total_loss.backward()
            running_lm_loss += lm_loss.item()
            running_aux_loss += aux_loss.item()

            if (step + 1) % grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"[Worker {rank}] Warning: NaN/Inf grad norm detected. Skipping optimizer step.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % log_interval == 0:
                    total_accum_batches = log_interval * grad_accum_steps
                    avg_lm = running_lm_loss / total_accum_batches
                    avg_aux = running_aux_loss / total_accum_batches
                    avg_total = avg_lm + aux_loss_coef * avg_aux
                    import time
                    print(
                        f"[Worker {rank}] Ep {epoch+1}/{num_epochs} | "
                        f"Step {global_step} | "
                        f"LM Loss: {avg_lm:.4f} | "
                        f"Aux Loss: {avg_aux:.4f} | "
                        f"Total: {avg_total:.4f}"
                    )
                    # Persist to CSV in Google Drive
                    with open(log_file_path, "a", encoding="utf-8") as f:
                        f.write(f"{time.time()},{epoch+1},{global_step},{avg_lm:.5f},{avg_aux:.5f},{avg_total:.5f}\n")

                    running_lm_loss = 0.0
                    running_aux_loss = 0.0

                if global_step % save_interval == 0:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        step=global_step,
                        checkpoint_dir=shared_dir,
                        rank=rank,
                    )

        # End of epoch checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            step=global_step,
            checkpoint_dir=shared_dir,
            rank=rank,
        )

    print(f"[Worker {rank}] Training successfully finished.")
