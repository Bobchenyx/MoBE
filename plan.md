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

- **不修改仓库原有代码**：原有的 `train.py`、`get_mobe.py`、`models/modeling_qwen3_mobe.py` 等文件仅作参考，所有新功能通过新建文件实现。
- **Transformers 版本**：服务器上需确认安装 `transformers==4.51.3`（与 `requirements.txt` 一致）。Qwen3-30B-A3B-Instruct-2507 需要此版本或更高版本支持。若服务器版本不同，以实际能加载模型的版本为准，并更新 `topk_requirements.txt`。

---

## Step 1: `collect_calibration_data.py` (新文件, ~150 lines)

Run full model (TopK=8) on calibration data, hook each decoder layer to save per-layer tensors.

使用标准 HF 模型 (`AutoModelForCausalLM`)，不使用 MoBE 变体，因为需要的是 TopK=8 原始输出。

**Hooks on each of the 48 MoE layers:**
- `post_attention_layernorm` pre-hook → capture `residual`（layernorm 的输入 = attention residual add 之后的 hidden_states）
- `post_attention_layernorm` forward hook → capture `moe_input`（layernorm 的输出，即 MoE 的输入）
- `mlp` forward hook → capture `moe_output_topk8`（MoE 的输出，residual add 之前）
  - **注意**：标准 HF Qwen3MoE 的 SparseMoeBlock forward 返回 `(hidden_states, router_logits)` tuple，hook 中需要解包取 `output[0]`

**Output per layer:**
- `layer_{i}_residual.pt` — [total_tokens, 2048]
- `layer_{i}_moe_input.pt` — [total_tokens, 2048]
- `layer_{i}_moe_output_topk8.pt` — [total_tokens, 2048]
- `metadata.json` — model_path, num_samples, seq_len, moe_layer_indices

**Args:** `--model_path`, `--save_dir`, `--dataset` (default `allenai/c4`), `--dataset_config` (default `en`), `--num_samples` (default 32), `--seq_len` (default 2048), `--batch_size`（每次喂给模型的 sequence 数）

**Storage:** 32 samples × 2048 tokens = 65,536 tokens. Each tensor ~256MB (bf16). 3 tensors × 48 layers ≈ 36GB total.

**Key patterns:** Use `datasets` library. Concatenate text, tokenize, split into fixed-length chunks. Process batch-by-batch with `torch.no_grad()`, append to CPU lists, gc cleanup after each batch.

---

## Step 2: `calibrate_topk.py` (新文件, ~250 lines)

Per-layer training of compensation parameters。参考 `train.py` 的训练循环模式（仅参考，不修改）。

**For each layer:**
1. Load saved `residual`, `moe_input`, `moe_output_topk8` from disk
2. Load that layer's MoE weights from model safetensors (~1.2GB per layer: router + 128 experts × gate/up/down_proj)
3. Compute `moe_output_topk4` from `moe_input`:
   - Router forward → softmax → topk(4) → norm_topk_prob
   - Expert loop with one_hot mask + index_add
   - **标准 MoE expert forward**（不是 MoBE）: `down_proj(silu(gate_proj(x)) * up_proj(x))`
   - 参考标准 HF transformers 中 Qwen3MoE 的 SparseMoeBlock.forward 逻辑（不是 MoBE 版本的 B_gate/W_gate）
4. Train `CompensationModel(A, B, bias)`:
   - Init: A=1, B=1, bias=0
   - Target: `residual + moe_output_topk8`
   - Predict: `A * moe_output_topk4 + B * residual + bias`
   - Loss: MSE (float32)
   - Optimizer: Adam, lr=0.01
   - Checkpoint best every 200 epochs
5. Save `layer_{i}_compensation.pth` (dict with A, B, bias tensors)
6. gc cleanup, move to next layer

**Args:** `--model_path`, `--data_dir`, `--save_path`, `--num_epochs` (default 5000), `--learning_rate` (default 0.01), `--batch_size` (default 4096 tokens，训练 mini-batch 大小；65536 tokens 总量也可以 full-batch), `--start_layer`, `--end_layer`

**Output:** `topk_compensation/layer_{i}_compensation.pth` + `training_log.json` (per-layer best MSE for deciding which layers are easy/hard to compensate)

---

## Step 3: `models/modeling_qwen3_topk.py` (新文件)

从 `models/modeling_qwen3_mobe.py` 复制一份，命名为 `modeling_qwen3_topk.py`，在副本上做以下修改：

**3a. `Qwen3MoeDecoderLayer.__init__`:**
添加补偿参数，根据 config 中的 `topk_compensation` 字段决定是否启用：
```python
# 从 config 读取是否启用补偿
self.has_topk_compensation = getattr(config, 'topk_compensation', False)
if self.has_topk_compensation:
    self.topk_comp_A = nn.Parameter(torch.ones(config.hidden_size), requires_grad=False)
    self.topk_comp_B = nn.Parameter(torch.ones(config.hidden_size), requires_grad=False)
    self.topk_comp_bias = nn.Parameter(torch.zeros(config.hidden_size), requires_grad=False)
```

> **为什么用 config 字段而非 bool 属性**：`save_pretrained` 只保存 `nn.Parameter` / `buffer` 到 state_dict，普通 bool 属性不会被持久化。通过在 config 中设置 `topk_compensation=True`，config 会随 `config.json` 一起保存，加载时 `__init__` 能正确重建补偿参数。

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

## Step 4: `assemble_topk4_model.py` (新文件, ~80 lines)

Assemble the final model。参考 `get_mobe.py` 的模式（仅参考，不修改）。

1. Load model config, set `num_experts_per_tok = 4` and `topk_compensation = True`
2. Load model using `models/modeling_qwen3_topk.py` with modified config（此时 `__init__` 会自动创建补偿参数）
3. For each layer, load `layer_{i}_compensation.pth`, 将训练好的 A/B/bias 值赋给对应的 `nn.Parameter`
4. Save complete model with `model.save_pretrained()` + `config.save_pretrained()` + `tokenizer.save_pretrained()`

> 补偿参数会作为 `nn.Parameter` 自动进入 state_dict，config 中的 `topk_compensation=True` 和 `num_experts_per_tok=4` 会保存到 `config.json`，加载时完全自恢复。

---

## Files Overview

| File | Action | Notes |
|------|--------|-------|
| `collect_calibration_data.py` | **新建** | 数据采集脚本 |
| `calibrate_topk.py` | **新建** | 补偿参数训练脚本 |
| `models/modeling_qwen3_topk.py` | **新建** | 从 modeling_qwen3_mobe.py 复制+修改，加入补偿逻辑 |
| `assemble_topk4_model.py` | **新建** | 模型组装脚本 |
| `topk_requirements.txt` | **新建** | TopK 实验的依赖 |

**不修改的文件（仅作参考）：**
- `train.py` — 训练循环模式参考
- `get_mobe.py` — 模型组装模式参考
- `models/modeling_qwen3_mobe.py` — MoE forward 逻辑参考（注意：MoBE 版本使用 B_gate/W_gate，与标准 MoE 不同）
- `requirements.txt` — 原有依赖，不动
- `CLAUDE.md` — 原有文档，不动

## Dependencies (`topk_requirements.txt`)

```
accelerate==1.7.0
datasets>=3.0.0
safetensors==0.5.3
torch==2.7.0
tqdm==4.67.1
transformers==4.51.3
```

服务器执行前先确认：`pip install -r topk_requirements.txt`

## Key Implementation Notes

1. **标准 MoE vs MoBE**：collect_calibration_data.py 和 calibrate_topk.py 都基于**标准 HF Qwen3MoE**（AutoModelForCausalLM），不是 MoBE 变体。MoBE 版本使用 B_gate/B_up/W_gate/W_up 参数，与标准 expert forward `down_proj(silu(gate_proj(x)) * up_proj(x))` 不同。

2. **MoE forward 返回 tuple**：标准 HF Qwen3 的 SparseMoeBlock.forward 返回 `(final_hidden_states, router_logits)`。在 hook 和手动计算中都需要正确解包。

3. **补偿参数持久化**：通过 `config.topk_compensation = True` 控制 `__init__` 中是否创建补偿 `nn.Parameter`。这确保 `save_pretrained` / `from_pretrained` 能正确保存和恢复：
   - `nn.Parameter` → state_dict → safetensors 文件
   - `config.topk_compensation` → config.json

4. **import 路径**：`assemble_topk4_model.py` 中通过 `from models.modeling_qwen3_topk import Qwen3TopKForCausalLM`（或类似名字）来 import 新模型类。`models/` 目录下需要有 `__init__.py`（如果没有的话需要创建一个空文件）。

## Verification

1. Run `collect_calibration_data.py` on a small sample (2-4 sequences) to verify hooks capture correct shapes
2. Run `calibrate_topk.py` on a single layer to verify training converges (MSE should decrease)
3. Check `training_log.json` to compare per-layer difficulty
4. Run `assemble_topk4_model.py` and do a simple generation test to verify the model produces coherent output
