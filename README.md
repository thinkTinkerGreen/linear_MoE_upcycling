# Linear MoE Upcycling: SmolLM2-135M to Sparse Mixture-of-Experts

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A structured, production-ready framework for **upcycling dense language models into Sparse Mixture-of-Experts (MoE)**, optimized for fault-tolerant training within resource-constrained environments (e.g., Google Colab T4 / free-tier GPU instances).

---

## 🌟 Key Architecture & Highlights

* **Base Model:** [`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct).
* **Sparse MoE Configuration:**
  * **4 Distinct Experts** cloned from the pre-trained dense MLP SwiGLU blocks (`gate_proj`, `up_proj`, `down_proj`).
  * **Top-2 Token Routing:** Dynamic token dispatch with parametric linear router.
  * **Switch / GShard Auxiliary Balancing Loss:** Prevents expert collapse by penalizing non-uniform token assignments.
* **Differential Learning Rates:**
  * Base network (Self-Attention, Embeddings, Norms): `1e-6`
  * Cloned Experts: `2e-5`
  * New Parametric Routers: `1e-4`
* **Multi-Domain Contextual Dataset:**
  * Domain 1: Regional Geography & Farming in India (`gandharv/Agri_QA`)
  * Domain 2: Nutritional Facts (`tatsu-lab/alpaca` filtered)
  * Domains 3 & 4: Urban Lifestyles, Occupations & Income in India (`yahma/alpaca-cleaned` filtered)
  * ChatML formatting (`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>`).
* **Fault-Tolerant Parallel Training:**
  * Dual-process (`torch.multiprocessing`) on a shared GPU via independent data shards.
  * Atomic checkpoints saved directly to shared Google Drive with seamless auto-resumption across disconnected sessions.

---

## 📁 Repository Structure

```text
linear_MoE_upcycling/
├── configs/
│   └── default_config.yaml     # Hyperparameters, paths, and optimizer settings
├── src/
│   ├── models/
│   │   ├── moe_layer.py        # SparseMoEBlock implementation with Top-k & aux loss
│   │   └── upcycling.py        # Dense-to-MoE layer replacement logic
│   ├── data/
│   │   ├── curate.py           # Multi-domain dataset fetch, deduplicate, shard
│   │   └── dataset.py          # PyTorch ShardedChatMLDataset
│   ├── training/
│   │   ├── optimizer.py        # Differential parameter grouping
│   │   └── trainer.py          # Worker training loop with mixed precision
│   └── utils/
│       ├── checkpoint.py       # Atomic checkpoint save & load
│       └── config.py           # YAML configuration loader
├── scripts/
│   ├── run_train.py            # Multiprocessing launch entrypoint
│   └── evaluate_moe.py         # Generation and inference script
├── tests/
│   └── test_moe.py             # Unit tests for MoE routing & backprop
├── notebooks/                  # Colab execution notebooks
├── pyproject.toml              # Build specifications
├── requirements.txt            # Package dependencies
└── README.md
```

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/<your-username>/linear_MoE_upcycling.git
cd linear_MoE_upcycling

pip install -r requirements.txt
# or with uv:
uv pip install -e .
```

### 2. Run Unit Tests

Verify the MoE forward pass, routing, and gradient backward propagation:

```bash
pytest tests/
```

### 3. Step-by-Step Training Workflow

#### A. Curate & Shard the Dataset
```bash
python -m src.data.curate \
  --output_dir /content/drive/MyDrive/SmolLM2_MoE_Shared \
  --num_shards 2
```

#### B. Launch Dual-Process Training
```bash
python scripts/run_train.py --config configs/default_config.yaml
```

#### C. Evaluate & Generate Responses
```bash
python scripts/evaluate_moe.py \
  --config configs/default_config.yaml \
  --prompt "What are the main kharif crops in Maharashtra?"
```

---

## 🛡️ Fault Tolerance & Session Recovery
All checkpoints are saved atomically (`.pt.tmp` -> `.pt`) to prevent corruption if a Colab session disconnects. When re-running on a new instance or fresh account, `scripts/run_train.py` automatically detects existing checkpoints in the shared Google Drive folder and picks up at the exact `epoch` and `step`.
