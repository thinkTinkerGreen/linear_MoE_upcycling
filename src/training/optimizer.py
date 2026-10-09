from typing import List, Dict, Any
import torch.nn as nn


def build_differential_optimizer_groups(
    model: nn.Module,
    lr_base: float = 1e-6,
    lr_experts: float = 2e-5,
    lr_router: float = 1e-4,
    weight_decay: float = 0.01,
) -> List[Dict[str, Any]]:
    """
    Groups model parameters into differential learning rate buckets:
    1. Base weights (embeddings, attentions, norms): lr_base
    2. Cloned expert MLP weights: lr_experts
    3. New parametric routers: lr_router

    Args:
        model: Upcycled MoE model.
        lr_base: Learning rate for pre-trained base parameters.
        lr_experts: Learning rate for cloned expert weights.
        lr_router: Learning rate for newly initialized router projections.
        weight_decay: Weight decay factor.

    Returns:
        List of parameter group dictionaries compatible with torch.optim.Optimizer.
    """
    base_params = []
    expert_params = []
    router_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ".mlp.gate." in name:
            router_params.append(param)
        elif ".mlp.experts." in name:
            expert_params.append(param)
        else:
            base_params.append(param)

    print(
        f"Optimizer parameter groups initialized:\n"
        f"  - Base parameters:   {len(base_params)} tensors (lr={lr_base})\n"
        f"  - Expert parameters: {len(expert_params)} tensors (lr={lr_experts})\n"
        f"  - Router parameters: {len(router_params)} tensors (lr={lr_router})"
    )

    groups = []
    if base_params:
        groups.append({"params": base_params, "lr": lr_base, "weight_decay": weight_decay, "name": "base"})
    groups.append({"params": expert_params, "lr": lr_experts, "weight_decay": weight_decay, "name": "experts"})
    groups.append({"params": router_params, "lr": lr_router, "weight_decay": 0.0, "name": "router"})

    return groups
