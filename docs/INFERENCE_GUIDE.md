# Guide: Future Deployment & Inference for Sparse MoE

This guide details how to run the fine-tuned Sparse MoE model on both **Google Colab (GPU)** and **Local / OCI Instances (CPU)** without re-running training.

---

## 1. Architecture Overview
- **Base Backbone:** `HuggingFaceTB/SmolLM2-135M-Instruct`
- **Sparse MoE Structure:** 30 Decoder layers, each containing:
  - **4 Cloned Experts** (deep-copied from SwiGLU MLP blocks).
  - **Parametric Router** projecting `hidden_dim -> 4`.
  - **Top-2 Dynamic Gating** (only 2 experts activate per token).
- **Model Checkpoint:** Saved in Google Drive at `/content/drive/MyDrive/linear_MoE_upcycling/latest_moe_checkpoint.pt`.

---

## 2. Running Inference in Google Colab (GPU)

1. Open your Colab notebook and mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Clone/pull the repository:
   ```bash
   !git clone https://github.com/thinktinkergreen/linear_MoE_upcycling.git /content/linear_MoE_upcycling
   %cd /content/linear_MoE_upcycling
   !pip install -q -r requirements.txt
   ```

3. Run interactive inference with any custom prompt:
   ```bash
   !python scripts/evaluate_moe.py \
     --config configs/default_config.yaml \
     --prompt "What are the main kharif crops grown in Maharashtra and what soil do they need?" \
     --max_new_tokens 150
   ```

---

## 3. Running Inference on Local / OCI Instance (CPU)

The model is lightweight (~135M dense active parameters, ~350M total parameters) and runs quickly on standard CPU instances.

### Setup (One-Time)
```bash
cd /home/opc/projects/linear_MoE_upcycling
source .venv/bin/activate
pip install -r requirements.txt
```

### Running Evaluation / Prompting
If you copied `latest_moe_checkpoint.pt` to your local machine (e.g. into `/home/opc/projects/linear_MoE_upcycling/checkpoints/`):

1. Update `shared_dir` in `configs/default_config.yaml`:
   ```yaml
   data:
     shared_dir: "checkpoints"
   ```
2. Run:
   ```bash
   python scripts/evaluate_moe.py \
     --config configs/default_config.yaml \
     --prompt "What are the key nutritional benefits and calorie density of moong dal?"
   ```

---

## 4. Programmatic Python API

To integrate the trained model into your own Python scripts:

```python
import torch
from transformers import AutoTokenizer
from src.models.upcycling import upcycle_smollm2_to_moe
from src.utils.checkpoint import load_checkpoint

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. Build Upcycled MoE Architecture
model = upcycle_smollm2_to_moe(model_id, num_experts=4, top_k=2, device=device)

# 3. Load Trained Weights from Checkpoint Directory
load_checkpoint(model, None, "/path/to/checkpoint_dir", torch.device(device))
model.eval()

# 4. Generate
prompt = "<|im_start|>user\nWhat is crop rotation in farming?<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```
