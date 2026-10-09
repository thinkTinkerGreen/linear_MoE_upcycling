import torch
from transformers import AutoModelForCausalLM
from src.models.moe_layer import SparseMoEBlock


def upcycle_smollm2_to_moe(
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    num_experts: int = 4,
    top_k: int = 2,
    torch_dtype: torch.dtype = None,
    device: str = "cpu",
):
    """
    Converts pre-trained SmolLM2 dense MLP layers into SparseMoEBlock modules.
    """
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)

    print(f"Loading dense base model from {model_id} (dtype={torch_dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=None,
    )

    num_layers = len(model.model.layers)
    print(f"Upcycling {num_layers} decoder layers to MoE (Experts={num_experts}, Top-{top_k})...")

    for idx, layer in enumerate(model.model.layers):
        original_mlp = layer.mlp
        layer.mlp = SparseMoEBlock(
            original_mlp=original_mlp,
            num_experts=num_experts,
            top_k=top_k,
        ).to(torch_dtype)

    # Enable gradient checkpointing to slash activation memory by ~60%
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")

    model.to(device)
    print("Upcycling completed successfully.")
    return model
