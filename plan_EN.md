# Plan: TopK Reduction with Per-Layer Compensation for Qwen3-30B-A3B

## Context

Goal: Reduce TopK from 8 to 4 experts per token in Qwen3-30B-A3B-Instruct-2507 to speed up inference. Dropping experts loses information, so we add a lightweight per-channel compensation at each MoE layer's residual connection:

```
# Original (TopK=8):
output = residual + moe_output

# Compensated (TopK=4):
output = A * moe_output + B * residual + bias
```

A, B, bias are per-channel vectors (shape [2048]), trained per-layer using calibration data. All 48 layers are MoE (decoder_sparse_step=1, mlp_only_layers=[]).

## Model Specs
- hidden_size: 2048, num_experts: 128, moe_intermediate_size: 768
- num_experts_per_tok: 8 → 4, num_hidden_layers: 48
- norm_topk_prob: true, hidden_act: silu

## Principles

- **Do not modify existing repository code**: Existing files such as `train.py`, `get_mobe.py`, `models/modeling_qwen3_mobe.py` are for reference only. All new functionality is implemented through newly created files.
- **Transformers version**: The server must have `transformers==4.51.3` installed (consistent with `requirements.txt`). Qwen3-30B-A3B-Instruct-2507 requires this version or higher. If the server version differs, use whichever version can actually load the model and update `topk_requirements.txt` accordingly.

---

## Step 1: `collect_calibration_data.py` (new file, ~150 lines)

Run full model (TopK=8) on calibration data, hook each decoder layer to save per-layer tensors.

Uses standard HF model (`AutoModelForCausalLM`), not the MoBE variant, since we need the original TopK=8 outputs.

**Hooks on each of the 48 MoE layers:**
- `post_attention_layernorm` pre-hook → capture `residual` (layernorm input = hidden_states after attention residual add)
- `post_attention_layernorm` forward hook → capture `moe_input` (layernorm output, i.e., input to MoE)
- `mlp` forward hook → capture `moe_output_topk8` (MoE output, before residual add)
  - **Note**: Standard HF Qwen3MoE's SparseMoeBlock forward returns `(hidden_states, router_logits)` tuple. The hook must unpack and take `output[0]`.

**Output per layer:**
- `layer_{i}_residual.pt` — [total_tokens, 2048]
- `layer_{i}_moe_input.pt` — [total_tokens, 2048]
- `layer_{i}_moe_output_topk8.pt` — [total_tokens, 2048]
- `metadata.json` — model_path, num_samples, seq_len, moe_layer_indices

**Args:** `--model_path`, `--save_dir`, `--dataset` (default `allenai/c4`), `--dataset_config` (default `en`), `--num_samples` (default 32), `--seq_len` (default 2048), `--batch_size` (number of sequences fed to the model per forward pass)

**Storage:** 32 samples × 2048 tokens = 65,536 tokens. Each tensor ~256MB (bf16). 3 tensors × 48 layers ≈ 36GB total.

**Key patterns:** Use `datasets` library. Concatenate text, tokenize, split into fixed-length chunks. Process batch-by-batch with `torch.no_grad()`, append to CPU lists, gc cleanup after each batch.

---

## Step 2: `calibrate_topk.py` (new file, ~250 lines)

Per-layer training of compensation parameters. Reference `train.py` training loop pattern (reference only, do not modify).

**For each layer:**
1. Load saved `residual`, `moe_input`, `moe_output_topk8` from disk
2. Load that layer's MoE weights from model safetensors (~1.2GB per layer: router + 128 experts × gate/up/down_proj)
3. Compute `moe_output_topk4` from `moe_input`:
   - Router forward → softmax → topk(4) → norm_topk_prob
   - Expert loop with one_hot mask + index_add
   - **Standard MoE expert forward** (not MoBE): `down_proj(silu(gate_proj(x)) * up_proj(x))`
   - Reference the standard HF transformers Qwen3MoE SparseMoeBlock.forward logic (not the MoBE version with B_gate/W_gate)
4. Train `CompensationModel(A, B, bias)`:
   - Init: A=1, B=1, bias=0
   - Target: `residual + moe_output_topk8`
   - Predict: `A * moe_output_topk4 + B * residual + bias`
   - Loss: MSE (float32)
   - Optimizer: Adam, lr=0.01
   - Checkpoint best every 200 epochs
5. Save `layer_{i}_compensation.pth` (dict with A, B, bias tensors)
6. gc cleanup, move to next layer

**Args:** `--model_path`, `--data_dir`, `--save_path`, `--num_epochs` (default 5000), `--learning_rate` (default 0.01), `--batch_size` (default 4096 tokens for training mini-batch size; with 65,536 total tokens, full-batch is also feasible), `--start_layer`, `--end_layer`

**Output:** `topk_compensation/layer_{i}_compensation.pth` + `training_log.json` (per-layer best MSE for deciding which layers are easy/hard to compensate)

---

## Step 3: `models/modeling_qwen3_topk.py` (new file)

Copy `models/modeling_qwen3_mobe.py` and rename the copy to `modeling_qwen3_topk.py`. Make the following modifications on the copy:

**3a. `Qwen3MoeDecoderLayer.__init__`:**
Add compensation parameters, controlled by a `topk_compensation` field in config:
```python
# Read from config whether compensation is enabled
self.has_topk_compensation = getattr(config, 'topk_compensation', False)
if self.has_topk_compensation:
    self.topk_comp_A = nn.Parameter(torch.ones(config.hidden_size), requires_grad=False)
    self.topk_comp_B = nn.Parameter(torch.ones(config.hidden_size), requires_grad=False)
    self.topk_comp_bias = nn.Parameter(torch.zeros(config.hidden_size), requires_grad=False)
```

> **Why use a config field instead of a plain bool attribute**: `save_pretrained` only saves `nn.Parameter` / `buffer` to state_dict; a plain bool attribute is not persisted. By setting `topk_compensation=True` in config, it is saved alongside `config.json`, allowing `__init__` to correctly recreate compensation parameters on load.

**3b. `Qwen3MoeDecoderLayer.forward`:**
Replace `hidden_states = residual + hidden_states` (MoE residual add) with:
```python
if self.has_topk_compensation:
    hidden_states = self.topk_comp_A * hidden_states + self.topk_comp_B * residual + self.topk_comp_bias
else:
    hidden_states = residual + hidden_states
```

TopK is already parameterized via `self.top_k = config.num_experts_per_tok`, so setting `config.num_experts_per_tok = 4` handles the routing change.

---

## Step 4: `assemble_topk4_model.py` (new file, ~80 lines)

Assemble the final model. Reference `get_mobe.py` pattern (reference only, do not modify).

1. Load model config, set `num_experts_per_tok = 4` and `topk_compensation = True`
2. Load model using `models/modeling_qwen3_topk.py` with modified config (`__init__` will automatically create compensation parameters)
3. For each layer, load `layer_{i}_compensation.pth`, assign trained A/B/bias values to the corresponding `nn.Parameter`
4. Save complete model with `model.save_pretrained()` + `config.save_pretrained()` + `tokenizer.save_pretrained()`

> Compensation parameters are automatically included in state_dict as `nn.Parameter`. `topk_compensation=True` and `num_experts_per_tok=4` in config are saved to `config.json`, enabling full self-recovery on load.

---

## Files Overview

| File | Action | Notes |
|------|--------|-------|
| `collect_calibration_data.py` | **Create** | Calibration data collection script |
| `calibrate_topk.py` | **Create** | Compensation parameter training script |
| `models/modeling_qwen3_topk.py` | **Create** | Copied from modeling_qwen3_mobe.py with compensation logic added |
| `assemble_topk4_model.py` | **Create** | Model assembly script |
| `topk_requirements.txt` | **Create** | Dependencies for TopK experiment |

**Files NOT modified (reference only):**
- `train.py` — Training loop pattern reference
- `get_mobe.py` — Model assembly pattern reference
- `models/modeling_qwen3_mobe.py` — MoE forward logic reference (note: MoBE version uses B_gate/W_gate, different from standard MoE)
- `requirements.txt` — Original dependencies, untouched
- `CLAUDE.md` — Original documentation, untouched

## Dependencies (`topk_requirements.txt`)

```
accelerate==1.7.0
datasets>=3.0.0
safetensors==0.5.3
torch==2.7.0
tqdm==4.67.1
transformers==4.51.3
```

Before running on server: `pip install -r topk_requirements.txt`

## Key Implementation Notes

1. **Standard MoE vs MoBE**: Both `collect_calibration_data.py` and `calibrate_topk.py` are based on **standard HF Qwen3MoE** (`AutoModelForCausalLM`), not the MoBE variant. The MoBE version uses B_gate/B_up/W_gate/W_up parameters, which differs from the standard expert forward `down_proj(silu(gate_proj(x)) * up_proj(x))`.

2. **MoE forward returns tuple**: Standard HF Qwen3's SparseMoeBlock.forward returns `(final_hidden_states, router_logits)`. Both hooks and manual computation must unpack correctly.

3. **Compensation parameter persistence**: Controlled by `config.topk_compensation = True`, which determines whether `__init__` creates compensation `nn.Parameter`s. This ensures `save_pretrained` / `from_pretrained` correctly saves and restores:
   - `nn.Parameter` → state_dict → safetensors files
   - `config.topk_compensation` → config.json

4. **Import path**: `assemble_topk4_model.py` imports via `from models.modeling_qwen3_topk import Qwen3TopKForCausalLM` (or similar class name). The `models/` directory needs an `__init__.py` (create an empty one if it doesn't exist).

## Verification

1. Run `collect_calibration_data.py` on a small sample (2-4 sequences) to verify hooks capture correct shapes
2. Run `calibrate_topk.py` on a single layer to verify training converges (MSE should decrease)
3. Check `training_log.json` to compare per-layer difficulty
4. Run `assemble_topk4_model.py` and do a simple generation test to verify the model produces coherent output
