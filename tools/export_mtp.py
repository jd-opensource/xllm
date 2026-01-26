"""
Export MTP layer for multiple model types (DeepSeek-V3, DeepSeek-V3.2, DeepSeek-R1, GLM4.5, GLM4.7 etc.).
The exported model can be used for speculative decoding.

Usage:
    # DeepSeek V3
    python3 export_mtp.py --input-dir /path/to/DeepSeek-V3 --output-dir /path/to/DeepSeek-V3-mtp

    # DeepSeek V3.2
    python3 export_mtp.py --input-dir /path/to/DeepSeek-V3.2 --output-dir /path/to/DeepSeek-V3.2-mtp

    # DeepSeek R1
    python3 export_mtp.py --input-dir /path/to/DeepSeek-R1 --output-dir /path/to/DeepSeek-R1-mtp

    # GLM4 MoE
    python3 export_mtp.py --input-dir /path/to/GLM-4.5-Air --output-dir /path/to/GLM-4.5-Air-mtp
"""
# adapted from https://github.com/sgl-project/sglang/blob/main/scripts/export_deepseek_nextn.py
import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig


def detect_model_type(config):
    """Detect model type from config."""
    model_type = getattr(config, "model_type", "").lower()
    architectures = getattr(config, "architectures", [])
    
    # Check for DeepSeek models
    # Note: DeepSeek V3, V3.2, and R1 may all have model_type="deepseek_v3" in config
    # V3.2 can be distinguished by index_head_dim, index_n_heads, index_topk fields
    if "deepseek" in model_type or any("deepseek" in arch.lower() for arch in architectures):
        # Check for V3.2 specific fields (index_head_dim, index_n_heads, index_topk)
        if hasattr(config, "index_head_dim") or hasattr(config, "index_n_heads") or hasattr(config, "index_topk"):
            # V3.2 has these fields, use deepseek_v32 for MTP export
            return "deepseek_v32"
        else:
            # V3 or R1 (both use deepseek_v3 for MTP export)
            return "deepseek_v3"
    
    # Check for GLM4
    if "glm4" in model_type.lower() or any("glm4" in arch.lower() for arch in architectures):
        # Check if it's MoE variant
        if hasattr(config, "n_routed_experts") and getattr(config, "n_routed_experts", 0) > 0:
            return "glm4_moe"
        else:
            return "glm4"
    
    # Fallback: try to infer from model_type
    if model_type:
        return model_type
    
    raise ValueError(f"Unable to detect model type from config. model_type={model_type}, architectures={architectures}")


def get_mtp_layer_id(config, model_type):
    """Get MTP layer ID based on model type."""
    if not hasattr(config, "num_hidden_layers"):
        raise ValueError("'num_hidden_layers' not found in model config.")
    
    # For DeepSeek V3/V3.2/R1 and GLM4, MTP layer is the last layer
    if model_type in ["deepseek_v3", "deepseek_v32", "glm4_moe"]:
        return config.num_hidden_layers
    
    raise ValueError(f"Unsupported model type for MTP export: {model_type}")


def get_mtp_model_type(model_type):
    """Get the MTP model type name for the output config."""
    mapping = {
        "deepseek_v3": "deepseek_v3_mtp",  # Used for V3 and R1
        "deepseek_v32": "deepseek_v32_mtp",  # Used for V3.2
        "glm4_moe": "glm4_moe_mtp",
    }
    return mapping.get(model_type, f"{model_type}_mtp")


def get_mtp_architecture(model_type):
    """Get the architecture name for the output config."""
    mapping = {
        "deepseek_v3": "DeepseekMTPForCausalLM",  # Used for V3 and R1
        "deepseek_v32": "DeepseekV32MtpForCausalLM",  # Used for V3.2
        "glm4_moe": "Glm4MoeMtpForCausalLM",
    }
    return mapping.get(model_type, "MtpForCausalLM")


def update_and_save_config(config, output_dir, model_type):
    """Update and save config for MTP model."""
    new_config = config.to_dict()
    mtp_model_type = get_mtp_model_type(model_type)
    mtp_architecture = get_mtp_architecture(model_type)
    
    # Common updates for all models
    updates = {
        "num_hidden_layers": 1,
        "architectures": [mtp_architecture],
        "model_type": mtp_model_type,
        "quantization_config": "",
    }
    
    # Model-specific updates
    if model_type == "deepseek_v3":  # Used for V3 and R1
        updates["first_k_dense_replace"] = 0
    elif model_type == "deepseek_v32":  # Used for V3.2
        updates["first_k_dense_replace"] = 0
    
    new_config.update(updates)
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(new_config, f, indent=2, ensure_ascii=False, sort_keys=True)


def copy_non_safetensors_files(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        src_file_path = os.path.join(input_dir, filename)
        if os.path.isfile(src_file_path) and not filename.endswith(".safetensors"):
            dst_file_path = os.path.join(output_dir, filename)
            shutil.copy2(src_file_path, dst_file_path)
    print(f"All non-safetensors files have been copied to {output_dir}")

def block_dequant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
    block_size: list[int],
) -> torch.Tensor:
    """This function conducts block-wise dequantization.
    The inputs are block-wise quantization tensor `x_q_block`,
    block-wise quantization scale and the block size.
    The outputs are dequantized tensor.
    """
    block_n, block_k = block_size[0], block_size[1]
    n, k = x_q_block.shape
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    assert n_tiles == x_s.shape[0]
    assert k_tiles == x_s.shape[1]

    x_dq_block = x_q_block.to(torch.float32)

    for i in range(k_tiles):
        for j in range(n_tiles):
            x_dq_block[
                j * block_n:min((j + 1) * block_n, n),
                i * block_k:min((i + 1) * block_k, k),
            ] *= x_s[j][i]

    return x_dq_block.to(torch.bfloat16)

def export_mtp_layer_parameters(input_dir, output_dir, mtp_layer_id, model_type):
    """Export MTP layer parameters for the specified model type."""
    prefix = f"model.layers.{mtp_layer_id}"
    output_path = os.path.join(output_dir, "mtp_layer_parameters.safetensors")
    params = {}
    
    for filename in os.listdir(input_dir):
        if not filename.endswith(".safetensors"):
            continue

        file_path = os.path.join(input_dir, filename)
        print(f"Processing: {filename}")

        try:
            with safe_open(file_path, framework="pt") as f:
                matching_keys = [k for k in f.keys() if k.startswith(prefix)]

                if not matching_keys:
                    print(f"  No parameters starting with '{prefix}' found")
                    continue

                for key in matching_keys:
                    # Handle special keys that should be at model level
                    if any(special in key for special in ["embed_tokens", "shared_head", "enorm", "hnorm", "eh_proj"]):
                        new_key = key.replace(prefix, "model")
                    else:
                        # Map to layer 0 for MTP model
                        new_key = key.replace(prefix, "model.layers.0")
                    params[new_key] = f.get_tensor(key)

        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")

    if params:
        new_params = {}
        for key, w_tensor in params.items():
            # Handle block-wise quantization for DeepSeek models (V3, V3.2, R1)
            if "weight_scale_inv" in key and model_type in ["deepseek_v3", "deepseek_v32"]:
                weight_scale = w_tensor
                weight_key = key.replace("weight_scale_inv", "weight")
                if weight_key in params:
                    weight = params[weight_key]
                    weight = block_dequant(weight, weight_scale, [128, 128])
                    new_params[weight_key] = weight
            elif key not in new_params:
                new_params[key] = params[key]
        params = new_params
        print(f"Saving {len(params)} parameters to {output_path}")
        save_file(params, output_path)
    else:
        print("No matching parameters found.")
        raise ValueError(f"No MTP layer parameters found at layer {mtp_layer_id}")

    # Update safetensors index
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    print(f"Updating safetensors index to {index_path}")
    index_data = {"weight_map": {}}
    for key in params:
        index_data["weight_map"][key] = "mtp_layer_parameters.safetensors"
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=4)

    print("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export MTP layer parameters for multiple model types (DeepSeek-V3, DeepSeek-V3.2, DeepSeek-R1, GLM4, etc.)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input HuggingFace model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output MTP model directory.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Model type (deepseek_v3, deepseek_v32, glm4_moe). If not specified, will auto-detect. Note: DeepSeek V3 and R1 use 'deepseek_v3', V3.2 uses 'deepseek_v32'.",
    )
    args = parser.parse_args()

    # Load config
    config = AutoConfig.from_pretrained(args.input_dir, trust_remote_code=True)
    
    # Detect or use specified model type
    if args.model_type:
        model_type = args.model_type.lower()
    else:
        model_type = detect_model_type(config)
    
    print(f"Detected model type: {model_type}")
    
    # Verify MTP support
    if not hasattr(config, "num_nextn_predict_layers"):
        raise ValueError("Model does not have 'num_nextn_predict_layers' attribute. This model may not support MTP.")
    if config.num_nextn_predict_layers != 1:
        raise ValueError(f"Only 1 MTP layer is supported, but found {config.num_nextn_predict_layers}.")
    
    # Get MTP layer ID
    mtp_layer_id = get_mtp_layer_id(config, model_type)
    print(f"MTP layer ID: {mtp_layer_id}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Copy non-safetensors files
    copy_non_safetensors_files(args.input_dir, args.output_dir)
    
    # Update and save config
    update_and_save_config(config, args.output_dir, model_type)
    
    # Export MTP layer parameters
    export_mtp_layer_parameters(args.input_dir, args.output_dir, mtp_layer_id, model_type)
    
    print(f"\nMTP model exported successfully to: {args.output_dir}")
