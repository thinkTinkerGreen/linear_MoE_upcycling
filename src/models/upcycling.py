import torch
from transformers import AutoModelForCausalLM
from src.models.moe_layer import SparseMoEBlock


def upcycle_smollm2_to_moe(
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    num_experts: int = 4,
    top_k: int = 2,
    torch_dtype: torch.dtype = torch.float32,
    device: str = "cpu",
):
    """
    Converts pre-trained SmolLM2 dense MLP layers into SparseMoEBlock modules.
    
    Args:
        model_id: HuggingFace hub model ID or local path.
        num_experts: Number of experts per layer (default: 4).
        top_k: Top-k experts to route per token (default: 2).
        torch_dtype: PyTorch dtype for weights.
        device: Device to place the model on.

    Returns:
        upcycled_model: HuggingFace PreTrainedModel with SparseMoE blocks.
    """
    print(f"Loading dense base model from {model_id}...")
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
        )

    model.to(device)
    print("Upcycling completed successfully.")
    return model
