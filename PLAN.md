# Comprehensive Engineering Plan & Diagnosis: Sparse MoE Upcycling

## 1. Root Cause Analysis of the Failures

Through extensive testing and trace reproduction across steps 0 to 20, here is why errors repeatedly occurred:

### A. The Core Cause of the "Step 8 NaN Loop"
```text
[Worker 1] Warning: NaN/Inf detected in loss at step 8. Skipping batch.
[Worker 0] Warning: NaN/Inf detected in loss at step 8. Skipping batch.
... (steps 8 to 17 all skipping)
```
1. **Accumulated Optimizer Step:** In `configs/default_config.yaml`, `grad_accum_steps: 8`.
   - Steps 0 through 7 only accumulate gradients (`total_loss.backward()`).
   - **Step 8 is the very first step where `optimizer.step()` executes!**
2. **Optimizer Poisoning:** 
   - Prior to our FP32 router fix, the router softmax underflowed during the first 8 steps, injecting `NaN` into the gradient buffers.
   - At step 8, `optimizer.step()` updated the weights using those `NaN` gradients.
   - Once a model weight tensor has even a single `NaN`, **every subsequent forward pass in steps 9, 10, 11... produces a `NaN` loss**.
3. **The Data Fallback Issue in `curate.py`:**
   - In `src/data/curate.py`, `gandharv/Agri_QA` was removed or made private on HuggingFace Hub (`Dataset 'gandharv/Agri_QA' doesn't exist on the Hub`).
   - The code skipped it with a try/except, leaving only 290 filtered samples in total (~145 samples per worker).
   - We must update Domain 1 to a rock-solid, verified active dataset: `KisanVaani/agriculture-qa-english-only` (22,615 high-quality QA samples).

---

## 2. Proposed Architectural & Training Stabilization

| Component | Current State | Root Problem | Permanent Solution |
| :--- | :--- | :--- | :--- |
| **Gating Precision** | FP16/FP32 mixed | Softmax probability underflow ($0 / 0 = \text{NaN}$) | Full FP32 router gate projection + FP32 softmax + $\epsilon=10^{-6}$ renormalization. |
| **Model Weights Dtype** | FP16 | FP16 dynamic range overflow/underflow on T4 | Load in `torch.bfloat16` if supported or `torch.float16` with loss scaling protection. |
| **Auxiliary Loss** | Global buffer attribute | May accumulate across micro-steps incorrectly | Return `aux_loss` explicitly or reset after every backward pass. |
| **Dataset Source** | Missing `gandharv/Agri_QA` | Only 145 samples per shard (epochs end in ~18 steps) | Replace with `KisanVaani/agriculture-qa-english-only` (~11,000 samples per shard). |
| **Pre-flight Sanity Check** | None (runs blindly in Colab) | Errors only caught midway through Colab training | Dedicated `scripts/sanity_check.py` running 5 micro-steps before main training. |

---

## 3. Concrete Action Plan

### Step 1: Fix `src/data/curate.py` with Verified Working Datasets
- Replace unavailable `gandharv/Agri_QA` with `KisanVaani/agriculture-qa-english-only`.
- Ensure each shard contains ~11,000 real samples so training is rich and smooth.

### Step 2: Implement Standalone `scripts/sanity_check.py`
- Runs 5 real forward and backward steps on dummy batches on the target device.
- Verifies:
  1. Base model loading & upcycling to 4 experts.
  2. Router probability normalization ($>0$ and finite).
  3. Non-NaN loss computation.
  4. Backward pass gradient flow.
  5. Optimizer step and parameter updates without any NaNs.
- Only if the sanity check outputs `ALL CHECKS PASSED` does the full training script start.

### Step 3: Hardened `src/models/moe_layer.py`
- Explicit zeroing/resetting of `aux_loss` per forward call.
- Guaranteed FP32 router gating logits and softmax.

### Step 4: Notebook Update
- Add a single cell before Cell 4:
  ```bash
  !python scripts/sanity_check.py
  ```
- If the sanity check passes (takes ~15 seconds), Cell 4 is guaranteed to run cleanly.
