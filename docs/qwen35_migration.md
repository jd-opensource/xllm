# Qwen3.5 Model Migration from vLLM to xLLM

## Overview

This document describes the migration of the Qwen3.5 model architecture from the vLLM project to the xLLM project. The migration includes the complete model architecture, configuration handling, and integration with xLLM's execution framework.

## Key Features of Qwen3.5

Qwen3.5 is a hybrid architecture model that combines:

1. **GatedDeltaNet (GDN) Linear Attention**: Efficient linear attention mechanism for most layers
2. **Full Attention**: Standard multi-head attention for selected layers
3. **MoE Support**: Qwen3.5-MoE variant with sparse mixture-of-experts blocks

## Migration Components

### 1. Configuration Parameters (model_args.h)

Added the following Qwen3.5-specific parameters to `ModelArgs`:

```cpp
// GatedDeltaNet (GDN) linear attention parameters
PROPERTY(int32_t, linear_conv_kernel_dim) = 4;
PROPERTY(int32_t, linear_key_head_dim) = 128;
PROPERTY(int32_t, linear_value_head_dim) = 128;
PROPERTY(int32_t, linear_num_key_heads) = 16;
PROPERTY(int32_t, linear_num_value_heads) = 32;
PROPERTY(std::vector<std::string>, layer_types) = {};
PROPERTY(int32_t, full_attention_interval) = 4;
```

### 2. GatedDeltaNet Layer (gated_delta_net.h/cpp)

The GDN layer implements the linear attention mechanism specific to Qwen3.5:

- **Input Projections**: `in_proj_qkvz` and `in_proj_ba` for query/key/value and beta/alpha projections
- **Conv1D**: 1D convolution for temporal modeling
- **State Management**: `A_log` and `dt_bias` parameters for state-space modeling
- **Output Projection**: Row-parallel linear layer for final output

### 3. Qwen3.5 Decoder Layer (qwen35_decoder_layer.h/cpp)

The decoder layer supports hybrid attention:

- Dynamically selects between `Qwen2Attention` (full attention) and `GatedDeltaNet` (linear attention)
- Layer type determined by `layer_types` configuration or `full_attention_interval`
- Standard RMSNorm and MLP components

### 4. Qwen3.5 MoE Decoder Layer (qwen35_moe_decoder_layer.h/cpp)

Extended decoder layer with MoE support:

- Integrates `FusedMoE` for sparse expert routing
- Supports `mlp_only_layers` configuration
- Compatible with expert parallelism (EP)

### 5. Model Classes (qwen35.h, qwen35_moe.h)

Two model variants:

- **QWen35ForCausalLM**: Standard dense model
- **QWen35MoeForCausalLM**: Mixture-of-experts variant

Both registered with the model registry using `REGISTER_CAUSAL_MODEL` macro.

## Key Adaptations for xLLM

### 1. Architecture Patterns

| vLLM Pattern | xLLM Adaptation |
|--------------|-----------------|
| `nn.Module` classes | `torch::nn::Module` (LibTorch) |
| `VllmConfig` | `ModelContext` |
| `ParallelConfig` | `ParallelArgs` |
| `QuantizationConfig` | `QuantArgs` |
| Weight loading via `load_weights` | `load_state_dict` with `StateDict` |

### 2. Attention Mechanism

- Full attention uses existing `Qwen2Attention` implementation
- Linear attention (GDN) implemented as new `GatedDeltaNet` module
- Both integrated into unified decoder layer interface

### 3. Parallelism Support

- Tensor parallelism via `ColumnParallelLinear` and `RowParallelLinear`
- Expert parallelism via `FusedMoE` with EP group support
- Pipeline parallelism through layer-wise construction

### 4. Memory Management

- KV cache managed through xLLM's `KVCache` class
- State dict loading optimized for safetensors format
- Weight tying support for `embed_tokens` and `lm_head`

## File Structure

```
xllm/
├── core/
│   ├── framework/model/
│   │   └── model_args.h          # Added GDN parameters
│   └── layers/
│       ├── common/
│       │   ├── gated_delta_net.h     # GDN layer header
│       │   ├── gated_delta_net.cpp   # GDN implementation
│       │   └── tests/
│       │       └── qwen35_decoder_layer_tests.cpp
│       ├── qwen35_decoder_layer.h
│       ├── qwen35_decoder_layer.cpp
│       ├── qwen35_moe_decoder_layer.h
│       └── qwen35_moe_decoder_layer.cpp
└── models/
    ├── llm/
    │   ├── qwen35.h              # Qwen3.5 model
    │   └── qwen35_moe.h          # Qwen3.5-MoE model
    └── models.h                  # Updated includes
```

## Configuration Example

```json
{
  "model_type": "qwen35",
  "hidden_size": 4096,
  "intermediate_size": 12288,
  "num_hidden_layers": 32,
  "num_attention_heads": 16,
  "num_key_value_heads": 4,
  "linear_conv_kernel_dim": 4,
  "linear_key_head_dim": 128,
  "linear_value_head_dim": 128,
  "linear_num_key_heads": 16,
  "linear_num_value_heads": 32,
  "full_attention_interval": 4
}
```

## Testing

Unit tests verify:

1. Basic forward pass with full attention layers
2. Forward pass with linear attention layers
3. Output shape correctness
4. NaN/Inf detection in outputs

Run tests with:
```bash
./qwen35_decoder_layer_tests
```

## Known Limitations

1. GDN kernel optimization pending - current implementation uses placeholder logic
2. Full numerical consistency with vLLM requires custom CUDA kernels
3. NPU-specific optimizations not yet implemented

## Future Work

1. Implement optimized GDN attention kernel
2. Add FP8 quantization support for GDN layers
3. Implement state caching for linear attention
4. Add NPU-specific implementations
