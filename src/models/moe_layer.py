import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseMoEBlock(nn.Module):
    """
    Replaces a dense LlamaMLP block with a Sparse Top-k MoE.
    
    Attributes:
        num_experts (int): Total number of parallel experts (default: 4).
        top_k (int): Number of experts activated per token (default: 2).
        gate (nn.Linear): Trainable parametric router (hidden_dim -> num_experts).
        experts (nn.ModuleList): Expert MLP submodules deep-copied from base MLP.
        aux_loss (torch.Tensor): Load balancing auxiliary loss.
    """

    def __init__(
        self,
        original_mlp: nn.Module,
        num_experts: int = 4,
        top_k: int = 2,
        router_init_std: float = 0.02,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = original_mlp.gate_proj.in_features

        # 1. Parametric linear router
        self.gate = nn.Linear(self.hidden_dim, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=router_init_std)

        # 2. Cloned experts deep-copied from original MLP
        self.experts = nn.ModuleList([copy.deepcopy(original_mlp) for _ in range(num_experts)])

        # Stores auxiliary loss computed during each forward pass
        self.register_buffer("aux_loss", torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Sparse MoE Block.
        
        Args:
            x (torch.Tensor): Shape [batch_size, seq_len, hidden_dim]
            
        Returns:
            torch.Tensor: Aggregated output [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # [num_tokens, hidden_dim]
        num_tokens = x_flat.size(0)

        # Router logits and gating probabilities
        router_logits = self.gate(x_flat)  # [num_tokens, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # Select Top-K experts per token
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # Renormalize top-k probabilities to sum to 1
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Auxiliary Load-Balancing Loss (Switch Transformer / GShard style)
        # P: average probability assigned to expert i across all tokens
        # f: fraction of tokens dispatched to expert i (based on top-1 choice)
        P = router_probs.mean(dim=0)
        top1_indices = top_k_indices[:, 0]
        f = torch.bincount(top1_indices, minlength=self.num_experts).float() / num_tokens
        self.aux_loss = self.num_experts * torch.sum(f * P)

        # Dispatch tokens to selected experts and sum weighted outputs
        final_output = torch.zeros_like(x_flat)

        for expert_idx in range(self.num_experts):
            # Mask where expert_idx was selected across any of the top_k positions
            expert_mask = (top_k_indices == expert_idx)
            token_indices, k_positions = torch.where(expert_mask)

            if token_indices.numel() > 0:
                selected_tokens = x_flat[token_indices]
                expert_out = self.experts[expert_idx](selected_tokens)
                weights = top_k_weights[token_indices, k_positions].unsqueeze(-1)
                final_output.index_add_(0, token_indices, (expert_out * weights).to(final_output.dtype))

        return final_output.view(batch_size, seq_len, hidden_dim)
