"""
Sparse Mixture of Experts (MoE) Architecture for SmolLM2.
"""
from src.models.moe_layer import SparseMoEBlock
from src.models.upcycling import upcycle_smollm2_to_moe

__all__ = ["SparseMoEBlock", "upcycle_smollm2_to_moe"]
