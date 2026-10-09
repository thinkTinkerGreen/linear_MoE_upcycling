import pytest
import torch
import torch.nn as nn
from src.models.moe_layer import SparseMoEBlock


class DummyMLP(nn.Module):
    """Dummy MLP mimicking LlamaMLP projections."""
    def __init__(self, hidden_dim=64, intermediate_dim=128):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def test_moe_output_shape_and_initialization():
    batch_size = 2
    seq_len = 16
    hidden_dim = 64
    num_experts = 4
    top_k = 2

    dummy_mlp = DummyMLP(hidden_dim=hidden_dim)
    moe_block = SparseMoEBlock(dummy_mlp, num_experts=num_experts, top_k=top_k)

    assert len(moe_block.experts) == num_experts
    assert moe_block.gate.out_features == num_experts
    assert moe_block.gate.in_features == hidden_dim

    # Forward pass check
    x = torch.randn(batch_size, seq_len, hidden_dim)
    out = moe_block(x)

    assert out.shape == (batch_size, seq_len, hidden_dim)
    assert not torch.isnan(out).any()
    assert moe_block.aux_loss.item() > 0.0


def test_moe_backward_pass():
    batch_size = 2
    seq_len = 8
    hidden_dim = 64
    dummy_mlp = DummyMLP(hidden_dim=hidden_dim)
    moe_block = SparseMoEBlock(dummy_mlp, num_experts=4, top_k=2)

    x = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
    out = moe_block(x)
    loss = out.sum() + 0.01 * moe_block.aux_loss
    loss.backward()

    # Check gradients
    assert moe_block.gate.weight.grad is not None
    assert x.grad is not None
