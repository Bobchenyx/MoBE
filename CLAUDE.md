# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MoBE (Mixture-of-Basis-Experts) is a research implementation for compressing MoE-based LLMs via weight matrix factorization: **W = A × activation(Σ αᵢBᵢ)**, where A is expert-specific and B matrices are shared bases weighted by learned coefficients. This achieves 24-30% parameter reduction with ~1-2% accuracy drop. Paper: https://arxiv.org/abs/2508.05257

## Supported Models

- **Qwen3-MoE** (235B)

## Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train MoBE decomposition (standard)
```bash
python train.py \
  --index_path <path_to_index_json> \
  --base_dir <model_shard_dir> \
  --save_path <output_dir> \
  --num_B <num_basis_matrices> \
  --truncation <max_rows_per_basis> \
  --start_layer <start> --end_layer <end> \
  --num_epochs 10000 --learning_rate 0.07
```

### Train MoBE decomposition (grouped, for models with many experts)
```bash
python train_group.py \
  --index_path <path_to_index_json> \
  --base_dir <model_shard_dir> \
  --save_path <output_dir> \
  --num_B <num_basis_matrices> \
  --truncation <max_rows_per_basis> \
  --num_groups <num_expert_groups> \
  --start_layer <start> --end_layer <end>
```

### Generate compressed model (native MoBE format)
```bash
python get_mobe.py --base_model <base_model> --mobe_dir <trained_params> --save_dir <output>
```

### Generate HuggingFace-compatible model (reconstructed MoE)
```bash
python get_hf_model.py --base_model <base_model> --mobe_dir <trained_params> --save_dir <output>
```

## Architecture

### Two-stage pipeline
1. **Training** (`train.py` / `train_group.py`): Load expert weights from safetensors shards → SVD initialization → Adam optimization of A, B, W matrices → save best-loss state as `.pth` and `.safetensors` files.
2. **Model generation**: Either `get_mobe.py` (native MoBE layers, max compression) or `get_hf_model.py` (standard MoE reconstruction, compatible with vLLM/SGLang/HuggingFace).

### Key classes
- `MoBE` (in `train.py`/`train_group.py`): `nn.Module` with learnable A, B, W parameters. Forward: softmax(W) → weighted sum of B → activation → A @ result.
- `models/modeling_*_mobe.py`: Full model implementations with custom `MoBEMLP` layers replacing standard expert projections. Each replaces `gate_proj`/`up_proj` weights with `A` matrices + shared `B_gate`/`B_up` bases.
- `models/configuration_*.py`: Config schemas adding `num_B` and `activation` fields for MoBE.

### Training output files (per layer `i`)
- `model_layers_{i}_mlp_gate_proj_WAB.pth` — trained A, B, W parameters
- `model_layers_{i}_mlp_gate_proj_weight.safetensors` — reconstructed weights
- For grouped training: `*_group{g}_WAB.pth`

### train_group.py vs train.py
`train_group.py` divides experts into groups and trains MoBE independently per group. Use for models with many experts where single-batch training causes OOM.

## Key Dependencies

PyTorch, HuggingFace Transformers, Accelerate, SafeTensors. See `requirements.txt` for pinned versions.

## Notes

- No automated tests — verification is done via benchmark accuracy evaluation.
- Memory-sensitive: code uses explicit `gc.collect()` and `torch.cuda.empty_cache()` between layers.
- Model type detection in `get_mobe.py` is name-based (checks for "Qwen3" in the model path). New models can be added as `elif` branches.
