# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Kernel definitions for xLLM MLU backend.

Add new kernels by adding @kernel decorator below.
"""

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

try:
    import torch_mlu_ops

    mlu_ops_path = torch_mlu_ops.__path__[0]
except ImportError:
    raise ImportError("MLU kernels are not available. Please install torch-mlu-ops.")


@dataclass
class KernelSignature:
    """Represents a single kernel signature variant."""

    name: str  # Descriptive name for this variant (e.g., "H=16")
    params: str  # The signature string


@dataclass
class KernelConfig:
    """Represents the configuration for a single kernel."""

    device_kernel_name: str
    kernel_file: str
    signatures: List[KernelSignature]
    kernel_name: Optional[str] = None  # Defaults to tmo_<device_kernel_name>
    archs: List[str] = field(default_factory=lambda: ["mtp_592", "mtp_613"])

    @property
    def full_kernel_name(self) -> str:
        """Get the full kernel name with tmo_ prefix."""
        if self.kernel_name:
            return self.kernel_name
        return f"tmo_{self.device_kernel_name}"


# Default architectures for all kernels
DEFAULT_ARCHS = ["mtp_592", "mtp_613"]

# Global registry for kernel configurations
_REGISTERED_KERNELS: Dict[str, KernelConfig] = {}


def kernel(
    device_kernel_name: str, kernel_file: str, **kwargs: object
) -> Callable[[Callable[[], List[KernelSignature]]], KernelConfig]:
    """
    Decorator to register a kernel configuration.

    Example usage:
        @kernel("my_kernel", "path/to/kernel.py")
        def my_kernel_config():
            return [
                KernelSignature("H=16", "signature_string_1"),
                KernelSignature("H=48", "signature_string_2"),
            ]
    """

    def decorator(func: Callable[[], List[KernelSignature]]) -> KernelConfig:
        signatures = func()
        config = KernelConfig(
            device_kernel_name=device_kernel_name,
            kernel_file=kernel_file,
            signatures=signatures,
            **kwargs,
        )
        _REGISTERED_KERNELS[device_kernel_name] = config
        return config

    return decorator


def get_kernel_configs() -> Dict[str, KernelConfig]:
    """Get all registered kernel configurations."""
    return _REGISTERED_KERNELS


# ==================== Kernel Registrations ====================
# Add new kernels by adding @kernel decorator below


@kernel(
    "causal_conv1d_fwd_vllm_kernel",
    os.path.join(mlu_ops_path, "triton/conv/kernels.py"),
    kernel_name="tmo_causal_conv1d_fwd_vllm_kernel",
)
def causal_conv1d_fn_config() -> List[KernelSignature]:
    """Causal conv1d kernel configurations."""
    base_sig = "*bf16, *bf16, *bf16, *bf16, *i32, *u1, *i32, *i32, *i32, *i32, *i32, *i32, *i32, *bf16, {dim}, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, 0, -1, 0, 4, 1, 0, 1, 4, 8, 256"
    return [
        KernelSignature(f"dim={dim}", base_sig.format(dim=dim))
        for dim in [
            384,
            512,
            640,
            768,
            1024,
            1280,
            1536,
            2048,
            2560,
            3072,
            4096,
            5120,
            6144,
            8192,
            10240,
            12288,
        ]
    ]


@kernel(
    "fused_recurrent_gated_delta_rule",
    os.path.join(mlu_ops_path, "triton/fla/fused_recurrent_fn.py"),
    kernel_name="tmo_fused_recurrent_gated_delta_rule_fwd_kernel",
)
def fused_recurrent_gated_delta_rule_config() -> List[KernelSignature]:
    """Fused recurrent gated delta rule kernel."""
    return [
        KernelSignature(
            "BK=128, BV=8",
            "*bf16, *bf16, *bf16, *fp32, *bf16, *bf16, *fp32, *fp32, *i32, *i32, *i32, fp32, i32, i32, i32, i32, i32, i32, i32, 128, 8, i32, i32, i32, i32, 1, 1, 0, 1, 1, 1, 0, 0",
        )
    ]


@kernel(
    "fused_recurrent_gated_delta_rule_packed_decode",
    os.path.join(
        os.path.dirname(__file__),
        "triton_kernel",
        "fused_recurrent_gated_delta_rule_packed_decode.py",
    ),
    kernel_name="tmo_fused_recurrent_gated_delta_rule_packed_decode_kernel",
)
def fused_recurrent_gated_delta_rule_packed_decode_config() -> List[KernelSignature]:
    """Fused recurrent gated delta rule packed decode kernel."""
    return [
        KernelSignature(
            "BK=128, BV=128",
            "*bf16, *bf16, *bf16, *bf16, *bf16, *bf16, *fp32, *fp32, *i32, fp32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, 128, 128, 20.0, 1",
        )
    ]


@kernel(
    "chunk_local_cumsum",
    os.path.join(mlu_ops_path, "triton/fla/cumsum.py"),
    kernel_name="tmo_chunk_local_cumsum_scalar_kernel",
)
def chunk_local_cumsum_config() -> List[KernelSignature]:
    """Chunk local cumsum scalar kernel."""
    base_sig = "*fp32, *fp32, *i32, *i32, i32, i32, {H}, 64, i32, i32, 0, 1, 0"
    v_heads = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    return [KernelSignature(f"H={H}", base_sig.format(H=H)) for H in v_heads]


@kernel(
    "chunk_scaled_dot_kkt_fwd",
    os.path.join(mlu_ops_path, "triton/fla/chunk_scaled_dot_kkt.py"),
    kernel_name="tmo_chunk_scaled_dot_kkt_fwd_kernel",
)
def chunk_scaled_dot_kkt_fwd_config() -> List[KernelSignature]:
    """Chunk scaled dot product KKT forward kernel."""
    base_sig = "*bf16, *bf16, *fp32, *fp32, *i32, *i32, i32, i32, {H}, {Hg}, 128, 64, 128, i32, i32, 1, 1"
    k_heads = [1, 2, 4, 8, 16, 32]
    v_heads = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    return [
        KernelSignature(f"H={H},Hg={Hg}", base_sig.format(H=H, Hg=Hg))
        for Hg in k_heads
        for H in v_heads
    ]


@kernel(
    "recompute_w_u_fwd",
    os.path.join(mlu_ops_path, "triton/fla/wy_fast.py"),
    kernel_name="tmo_recompute_w_u_fwd_kernel",
)
def recompute_w_u_fwd_config() -> List[KernelSignature]:
    """Recompute W^T U forward kernel."""
    base_sig = "*bf16, *bf16, *bf16, *bf16, *bf16, *bf16, *fp32, *i32, *i32, i32, i32, {H}, {Hg}, 128, 128, 64, 128, 128, i32, i32, 1"
    k_heads = [1, 2, 4, 8, 16, 32]
    v_heads = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    return [
        KernelSignature(f"H={H},Hg={Hg}", base_sig.format(H=H, Hg=Hg))
        for Hg in k_heads
        for H in v_heads
    ]


@kernel(
    "chunk_gated_delta_rule_fwd_h",
    os.path.join(mlu_ops_path, "triton/fla/chunk_delta_h.py"),
    kernel_name="tmo_chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
)
def chunk_gated_delta_rule_fwd_h_config() -> List[KernelSignature]:
    """Chunk gated delta rule forward H kernel."""
    base_sig = "*bf16, *bf16, *bf16, *bf16, *fp32, *fp32, *bf16, *fp32, *fp32, *i32, *i32, i32, i32, {H}, {Hg}, 128, 128, 64, 128, 64, 1, 0, 1, 1, 1, 1"
    k_heads = [1, 2, 4, 8, 16, 32]
    v_heads = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    return [
        KernelSignature(f"H={H},Hg={Hg}", base_sig.format(H=H, Hg=Hg))
        for Hg in k_heads
        for H in v_heads
    ]


@kernel(
    "chunk_fwd_o",
    os.path.join(mlu_ops_path, "triton/fla/chunk_o.py"),
    kernel_name="tmo_chunk_fwd_kernel_o",
)
def chunk_fwd_o_config() -> List[KernelSignature]:
    """Chunk forward output kernel."""
    base_sig = "*bf16, *bf16, *bf16, *bf16, *fp32, *bf16, *i32, *i32, fp32, i32, i32, {H}, {Hg}, 128, 128, 64, 128, 128, i32, i32, 1, 1"
    k_heads = [1, 2, 4, 8, 16, 32]
    v_heads = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    return [
        KernelSignature(f"H={H},Hg={Hg}", base_sig.format(H=H, Hg=Hg))
        for Hg in k_heads
        for H in v_heads
    ]


@kernel(
    "causal_conv1d_update_decode",
    os.path.join(
        os.path.dirname(__file__), "triton_kernel", "causal_conv1d_update_decode.py"
    ),
    kernel_name="tmo_causal_conv1d_update_decode_kernel",
)
def causal_conv1d_update_decode_config() -> List[KernelSignature]:
    # Pointer types: x(bf16), weight(bf16), bias(bf16/nullptr), conv_state(bf16),
    # conv_state_indices(i32), num_accepted_tokens(i32/nullptr), query_start_loc(i32/nullptr),
    # block_idx_last_scheduled_token(i32/nullptr), initial_state_idx(i32/nullptr), out(bf16)
    ptrs = "*bf16, *bf16, *bf16, *bf16, *i32, *i32, *i32, *i32, *i32, *bf16"
    # Runtime ints: batch, num_cache_lines
    runtime_dims = "i32, i32"
    # Constexpr dims: dim(varies), seqlen=1, state_len=3
    constexpr_dims = "{dim}, 1, 3"
    # Runtime strides: stride_x_seq, stride_x_dim, stride_x_token, stride_w_dim,
    # stride_w_width, stride_istate_seq, stride_istate_dim, stride_istate_tok,
    # stride_state_indices, stride_o_seq, stride_o_dim, stride_o_token, pad_slot_id
    runtime_strides = "i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32"
    # Constexpr meta: HAS_BIAS=0, KERNEL_WIDTH=4, SILU_ACTIVATION=1, IS_VARLEN=0,
    # IS_APC_ENABLED=0, IS_SPEC_DECODING=0, NP2_STATELEN=4, USE_PAD_SLOT=1, BD=8, BW=4
    constexpr_meta = "0, 4, 1, 0, 0, 0, 4, 1, 8, 4"
    base_sig = ", ".join(
        [ptrs, runtime_dims, constexpr_dims, runtime_strides, constexpr_meta]
    )
    return [
        KernelSignature(f"dim={dim}", base_sig.format(dim=dim))
        for dim in [
            384,
            512,
            640,
            768,
            1024,
            1280,
            1536,
            2048,
            2560,
            3072,
            4096,
            5120,
            6144,
            8192,
            10240,
            12288,
        ]
    ]


@kernel(
    "fused_gdn_gating",
    os.path.join(os.path.dirname(__file__), "triton_kernel", "fused_gdn_gating.py"),
    kernel_name="tmo_fused_gdn_gating_kernel",
)
def fused_gdn_gating_config() -> List[KernelSignature]:
    # Pointer types: g(fp32), beta_output(bf16), A_log(bf16), a(bf16), b(bf16), dt_bias(bf16)
    ptrs = "*fp32, *bf16, *bf16, *bf16, *bf16, *bf16"
    # Runtime int before constexpr block: seq_len
    runtime_dims_before = "i32"
    # Constexpr dims: NUM_HEADS(varies), beta=1.0, threshold=20.0, BLK_HEADS=8, core_num(varies)
    constexpr_dims = "{NUM_HEADS}, 1.0, 20.0, 8, {core_num}"
    # Runtime int after constexpr block: batch
    runtime_dims_after = "i32"
    base_sig = ", ".join(
        [ptrs, runtime_dims_before, constexpr_dims, runtime_dims_after]
    )
    num_heads_list = [4, 8, 12, 16, 24, 32, 48, 64]
    core_num_list = [32, 64]
    return [
        KernelSignature(
            f"NUM_HEADS={nh},core_num={cn}", base_sig.format(NUM_HEADS=nh, core_num=cn)
        )
        for cn in core_num_list
        for nh in num_heads_list
    ]
