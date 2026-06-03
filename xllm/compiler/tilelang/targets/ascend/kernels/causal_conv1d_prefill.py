#!/usr/bin/env python
import tilelang
import tilelang.language as T
import torch

from ....common.spec import DispatchField, TilelangKernel, register_kernel

RING_SLOTS = 5
OUT_SLOTS = 2
DIM_PER_CORE = 2048
TOKEN_BLOCK_SIZE = 256
MAX_TOKEN_BLOCKS = 32

symbol_cache_lines = T.symbolic("num_cache_lines")
symbol_state_len = T.symbolic("state_len")

pass_configs_config = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

_prefill_kernel_cache = {}


def build_causal_conv1d_prefill_kernel(
    width: int,
    dim_chunks: int,
    num_batches: int,
    token_block_size: int = TOKEN_BLOCK_SIZE,
    max_token_blocks: int = MAX_TOKEN_BLOCKS,
    dim_per_core: int = DIM_PER_CORE,
    dtype_str: str = "float16",
    has_silu: bool = True,
) -> torch.nn.Module:
    hist_len = width - 1
    symbol_dim = T.symbolic("dim")
    symbol_total_len = T.symbolic("total_len")
    total_tasks = num_batches * max_token_blocks * dim_chunks

    @T.prim_func
    def causal_conv1d_prefill(
        x: T.Tensor((symbol_total_len, symbol_dim), dtype_str),
        weight: T.Tensor((width, symbol_dim), dtype_str),
        conv_state: T.Tensor((symbol_cache_lines, symbol_state_len, symbol_dim), dtype_str),
        cu_seqlens: T.Tensor((num_batches + 1,), "int32"),
        init_indices: T.Tensor((num_batches,), "int32"),
        current_indices: T.Tensor((num_batches,), "int32"),
        initial_state_mode: T.Tensor((num_batches,), "int32"),
        bias: T.Tensor((symbol_dim,), dtype_str),
        y: T.Tensor((symbol_total_len, symbol_dim), dtype_str),
    ):
        with T.Kernel(total_tasks, is_npu=True) as (cid, vid):
            tasks_per_batch = max_token_blocks * dim_chunks
            batch_id = cid // tasks_per_batch
            remainder = cid % tasks_per_batch
            token_block_id = remainder // dim_chunks
            dim_chunk_id = remainder % dim_chunks
            d_offset = dim_chunk_id * dim_per_core

            seq_start = cu_seqlens[batch_id]
            seq_end = cu_seqlens[batch_id + 1]
            seqlen = seq_end - seq_start

            token_start = token_block_id * token_block_size

            if token_start < seqlen:
                token_end_raw = token_start + token_block_size
                token_end = T.if_then_else(token_end_raw < seqlen, token_end_raw, seqlen)
                block_len = token_end - token_start

                last_tb_id = T.if_then_else(seqlen > 0, (seqlen - 1) // token_block_size, 0)
                is_last_buf = T.alloc_ub((1,), "int32")
                T.tile.fill(is_last_buf, (dim_chunk_id == dim_chunks - 1))
                is_last_tb = T.alloc_ub((1,), "int32")
                T.tile.fill(is_last_tb, (token_block_id == last_tb_id))

                read_cache_line = init_indices[batch_id]
                write_cache_line = current_indices[batch_id]
                has_initial = initial_state_mode[batch_id]

                ring_buf = T.alloc_ub((RING_SLOTS, dim_per_core), dtype_str)
                out_buf = T.alloc_ub((OUT_SLOTS, dim_per_core), dtype_str)
                w0 = T.alloc_ub((dim_per_core,), dtype_str)
                w1 = T.alloc_ub((dim_per_core,), dtype_str)
                w2 = T.alloc_ub((dim_per_core,), dtype_str)
                w3 = T.alloc_ub((dim_per_core,), dtype_str)
                tmp = T.alloc_ub((dim_per_core,), dtype_str)
                bias_ub = T.alloc_ub((dim_per_core,), dtype_str)
                initial_hist_save = T.alloc_ub((2, dim_per_core), dtype_str)

                T.copy(weight[0, d_offset], w0)
                T.copy(weight[1, d_offset], w1)
                T.copy(weight[2, d_offset], w2)
                T.copy(weight[3, d_offset], w3)
                T.copy(bias[d_offset], bias_ub)
                T.barrier_all()

                T.tile.fill(ring_buf[0, :], 0.0)
                T.tile.fill(ring_buf[1, :], 0.0)
                T.tile.fill(ring_buf[2, :], 0.0)

                if has_initial != 0:
                    if hist_len >= 1 and symbol_state_len > 0:
                        T.copy(conv_state[read_cache_line, 0, d_offset], ring_buf[0, :])
                    if hist_len >= 2 and symbol_state_len > 1:
                        T.copy(conv_state[read_cache_line, 1, d_offset], ring_buf[1, :])
                    if hist_len >= 3 and symbol_state_len > 2:
                        T.copy(conv_state[read_cache_line, 2, d_offset], ring_buf[2, :])

                T.barrier_all()

                if is_last_buf[0] != 0 and is_last_tb[0] != 0 and seqlen > 0:
                    if block_len < 3:
                        T.copy(ring_buf[2, :], initial_hist_save[0, :])
                        if block_len <= 1:
                            T.copy(ring_buf[1, :], initial_hist_save[1, :])

                T.copy(x[seq_start + token_start, d_offset], ring_buf[3, :])
                T.set_flag("mte2", "v", 0)

                for t in T.serial(token_block_size):
                    if t < block_len:
                        slot_curr = (t + 3) % 5
                        out_slot = t % 2

                        T.wait_flag("mte2", "v", 0)

                        T.tile.mul(tmp, w0, ring_buf[slot_curr, :])
                        T.tile.mul_add_dst(tmp, w1, ring_buf[(slot_curr + 4) % 5, :])
                        T.tile.mul_add_dst(tmp, w2, ring_buf[(slot_curr + 3) % 5, :])
                        T.tile.mul_add_dst(tmp, w3, ring_buf[(slot_curr + 2) % 5, :])

                        if t + 3 < block_len:
                            prefetch_slot = (t + 3) % 5
                            T.copy(x[seq_start + token_start + t + 3, d_offset], ring_buf[prefetch_slot, :])
                            T.set_flag("mte2", "v", 0)

                        if t >= 2:
                            T.wait_flag("mte3", "v", out_slot)

                        T.tile.add(tmp, tmp, bias_ub)
                        if has_silu:
                            T.tile.silu(out_buf[out_slot, :], tmp)
                        else:
                            T.tile.add(out_buf[out_slot, :], tmp, bias_ub)

                        T.set_flag("v", "mte3", out_slot)
                        T.wait_flag("v", "mte3", out_slot)
                        T.copy(out_buf[out_slot, :], y[seq_start + token_start + t, d_offset])
                        if t + 2 < block_len:
                            T.set_flag("mte3", "v", out_slot)

                if is_last_buf[0] != 0 and is_last_tb[0] != 0 and seqlen > 0:
                    if hist_len >= 1 and symbol_state_len > 0:
                        if block_len >= 3:
                            T.copy(ring_buf[(block_len + 2) % 5, :], conv_state[write_cache_line, 2, d_offset])
                        else:
                            T.copy(initial_hist_save[0, :], conv_state[write_cache_line, 2, d_offset])
                    if hist_len >= 2 and symbol_state_len > 1:
                        if block_len >= 2:
                            T.copy(ring_buf[(block_len + 1) % 5, :], conv_state[write_cache_line, 1, d_offset])
                        else:
                            T.copy(initial_hist_save[1, :], conv_state[write_cache_line, 1, d_offset])
                    if hist_len >= 3 and symbol_state_len > 2:
                        T.copy(ring_buf[block_len % 5, :], conv_state[write_cache_line, 0, d_offset])

    return causal_conv1d_prefill


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs_config)
def _build_prefill_kernel_jit(
    width: int,
    dim_chunks: int,
    num_batches: int,
    token_block_size: int = TOKEN_BLOCK_SIZE,
    max_token_blocks: int = MAX_TOKEN_BLOCKS,
    dim_per_core: int = DIM_PER_CORE,
    dtype_str: str = "float16",
    has_silu: bool = True,
) -> torch.nn.Module:
    return build_causal_conv1d_prefill_kernel(
        width=width,
        dim_chunks=dim_chunks,
        num_batches=num_batches,
        token_block_size=token_block_size,
        max_token_blocks=max_token_blocks,
        dim_per_core=dim_per_core,
        dtype_str=dtype_str,
        has_silu=has_silu,
    )


@register_kernel
class CausalConv1dPrefillKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("num_batches", "int32"),
        DispatchField("dim", "int32"),
        DispatchField("width", "int32"),
        DispatchField("has_silu", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": "bs1_d2048_w4_silu1_f16",
            "num_batches": 1,
            "dim": 2048,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs1_d4096_w4_silu1_f16",
            "num_batches": 1,
            "dim": 4096,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs1_d8192_w4_silu1_f16",
            "num_batches": 1,
            "dim": 8192,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs2_d2048_w4_silu1_f16",
            "num_batches": 2,
            "dim": 2048,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs2_d4096_w4_silu1_f16",
            "num_batches": 2,
            "dim": 4096,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
        {
            "variant_key": "bs2_d8192_w4_silu1_f16",
            "num_batches": 2,
            "dim": 8192,
            "width": 4,
            "has_silu": 1,
            "dtype": "float16",
        },
    ]

    @staticmethod
    def generate_source(
        num_batches: int,
        dim: int,
        width: int,
        has_silu: int,
        dtype: str,
    ) -> str:
        if dtype not in ("float16", "bfloat16"):
            raise ValueError(
                f"CausalConv1D Prefill TileLang kernel only supports dtype=float16/bfloat16, "
                f"got {dtype}"
            )
        dim_chunks = (dim + DIM_PER_CORE - 1) // DIM_PER_CORE
        tilelang.disable_cache()
        tilelang_kernel = build_causal_conv1d_prefill_kernel(
            width=width,
            dim_chunks=dim_chunks,
            num_batches=num_batches,
            token_block_size=TOKEN_BLOCK_SIZE,
            max_token_blocks=MAX_TOKEN_BLOCKS,
            dim_per_core=DIM_PER_CORE,
            dtype_str=dtype,
            has_silu=bool(has_silu),
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3,
            config={
                "tl.ascend_auto_cv_combine": True,
                "tl.ascend_auto_sync": False,
                "tl.ascend_memory_planning": True,
            },
        ):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source
