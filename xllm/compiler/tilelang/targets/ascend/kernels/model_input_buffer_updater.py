#!/usr/bin/env python3

"""
Unified Python-side TileLang design for ACL graph input-buffer update.

This file intentionally keeps only one kernel design. The old split between a
"default" runtime-wired ABI and a separate "v2" prototype has been removed.
The current file is the Python-side source of truth; C++ wrapper/runtime can be
rewritten later against this ABI.

What this operator updates
==========================
1. token-length group
   - tokens
   - positions
   - new_cache_slots

2. batch-length group
   - q_seq_lens
   - kv_seq_lens
   - optional q_cu_seq_lens
   - optional linear_state_indices

3. 2D row-strided group
   - block_tables

4. optional embedding group
   - input_embedding

How input_embedding works
=========================
- with_input_embedding = 0:
  the embedding group is compiled out and the wrapper can pass null pointers

- with_input_embedding = 1:
  input_embedding is treated as a flat storage buffer plus three runtime
  scalars:
  - src_input_embedding_stride
  - dst_input_embedding_stride
  - actual_hidden_size

  This keeps hidden size out of the specialization space while still matching
  the real row stride of runtime 2D tensors.

How positions work
==================
- with_mrope = 0:
  positions are treated as a 1D prefix of length actual_num_tokens

- with_mrope = 1:
  positions are treated as a flattened [3, actual_num_tokens] buffer, i.e.
  length = 3 * actual_num_tokens

Zero-fill semantics
===================
For token-length tensors there are three logical regions:

- [0, actual_num_tokens)
  valid data copied from src to dst

- [actual_num_tokens, padded_num_tokens)
  padding region; only dst_tokens and dst_new_cache_slots are zero-filled
  because:
  - tokens padded tail must be zeroed to avoid invalid token ids being
    consumed by embedding lookup / index_select
  - new_cache_slots padded tail must be zeroed to avoid reshape_and_cache
    writing to illegal cache addresses

- [padded_num_tokens, +inf)
  untouched

Implementation style
====================
This kernel is currently intended for ACL decode-only metadata update where
num_tokens is usually small, so the implementation favors simple runtime-length
scalar logic.

The current design assumes decode semantics:
- actual_num_tokens == actual_batch_size
- the padded logical row extent is padded_num_tokens

The outer work-sharing loop therefore uses continuous row-range partitioning on
the padded decode batch rather than separate token / position / batch loops.
Tensors that do not require zero-fill naturally do nothing once they leave
their valid range.
"""

import argparse
from itertools import product
from pathlib import Path

import tilelang
import tilelang.language as T

from compiler.tilelang.common.spec import (
    DispatchField,
    TilelangKernel,
    register_kernel,
)

from .utils import DEFAULT_ASCEND_PASS_CONFIGS, detect_vec_core_num

DEFAULT_DTYPE = "int32"
DEFAULT_EMBEDDING_DTYPE = "bfloat16"
NO_INPUT_EMBEDDING_DTYPE = "float32"
VEC_NUM = 2
COMPILE_MAX_TOKENS = 16384
COMPILE_MAX_BATCH = 1024
COMPILE_MAX_BLOCK_TABLE_LEN = 8192
COMPILE_MAX_HIDDEN_SIZE = 16384
MROPE_COMPONENTS = 3
BLOCK_TABLE_BULK_SIZE = 32

# Python-side ref-checks do not need to use the full compile limits.
REF_CHECK_COMPILE_MAX_TOKENS = min(COMPILE_MAX_TOKENS, 512)
REF_CHECK_COMPILE_MAX_BATCH = min(COMPILE_MAX_BATCH, 128)
REF_CHECK_COMPILE_MAX_BLOCK_TABLE_LEN = min(COMPILE_MAX_BLOCK_TABLE_LEN, 512)
REF_CHECK_COMPILE_MAX_HIDDEN_SIZE = min(COMPILE_MAX_HIDDEN_SIZE, 256)


def _validate_flag(name: str, value: int) -> bool:
    if value not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1, got {value}")
    return bool(value)


def _validate_embedding_dtype(dtype: str) -> None:
    if dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError(
            "embedding_dtype must be one of: float16, bfloat16, float32; "
            f"got {dtype}"
        )


def build_model_input_buffer_updater_kernel(
    *,
    compile_max_tokens: int,
    compile_max_batch: int,
    compile_max_block_table_len: int,
    compile_hidden_size: int,
    vec_core_num: int,
    embedding_dtype: str = DEFAULT_EMBEDDING_DTYPE,
    with_mrope: int = 0,
    with_input_embedding: int = 0,
    with_linear_state_indices: int = 0,
    with_q_cu_seq_lens: int = 0,
):
    if compile_max_tokens <= 0:
        raise ValueError(
            f"compile_max_tokens({compile_max_tokens}) must be > 0"
        )
    if compile_max_batch <= 0:
        raise ValueError(f"compile_max_batch({compile_max_batch}) must be > 0")
    if compile_max_block_table_len <= 0:
        raise ValueError(
            "compile_max_block_table_len"
            f"({compile_max_block_table_len}) must be > 0"
        )
    if compile_hidden_size <= 0:
        raise ValueError(
            f"compile_hidden_size({compile_hidden_size}) must be > 0"
        )
    compile_input_embedding_elems = compile_max_tokens * compile_hidden_size
    if vec_core_num <= 0:
        raise ValueError(f"vec_core_num({vec_core_num}) must be > 0")
    if vec_core_num % VEC_NUM != 0:
        raise ValueError(
            f"vec_core_num({vec_core_num}) must be divisible by VEC_NUM({VEC_NUM})"
        )
    _validate_embedding_dtype(embedding_dtype)

    use_mrope = _validate_flag("with_mrope", with_mrope)
    use_input_embedding = _validate_flag(
        "with_input_embedding", with_input_embedding
    )
    use_linear_state_indices = _validate_flag(
        "with_linear_state_indices", with_linear_state_indices
    )
    use_q_cu_seq_lens = _validate_flag(
        "with_q_cu_seq_lens", with_q_cu_seq_lens
    )

    task_num = vec_core_num
    m_num = vec_core_num // VEC_NUM
    compile_position_elems = compile_max_tokens * MROPE_COMPONENTS
    compile_block_table_elems = compile_max_batch * compile_max_block_table_len

    @T.prim_func
    def model_input_buffer_updater_kernel(
        src_tokens: T.Tensor((compile_max_tokens,), DEFAULT_DTYPE),
        src_positions: T.Tensor((compile_position_elems,), DEFAULT_DTYPE),
        src_new_cache_slots: T.Tensor((compile_max_tokens,), DEFAULT_DTYPE),
        src_q_seq_lens: T.Tensor((compile_max_batch,), DEFAULT_DTYPE),
        src_kv_seq_lens: T.Tensor((compile_max_batch,), DEFAULT_DTYPE),
        src_q_cu_seq_lens: T.Tensor((compile_max_batch,), DEFAULT_DTYPE),
        src_linear_state_indices: T.Tensor((compile_max_batch,), DEFAULT_DTYPE),
        src_block_tables: T.Tensor(
            (compile_block_table_elems,), DEFAULT_DTYPE
        ),
        src_input_embedding: T.Tensor(
            (compile_input_embedding_elems,), embedding_dtype
        ),
        dst_tokens: T.Tensor((compile_max_tokens,), DEFAULT_DTYPE),
        dst_positions: T.Tensor((compile_position_elems,), DEFAULT_DTYPE),
        dst_new_cache_slots: T.Tensor((compile_max_tokens,), DEFAULT_DTYPE),
        dst_q_seq_lens: T.Tensor((compile_max_batch,), DEFAULT_DTYPE),
        dst_kv_seq_lens: T.Tensor((compile_max_batch,), DEFAULT_DTYPE),
        dst_q_cu_seq_lens: T.Tensor((compile_max_batch,), DEFAULT_DTYPE),
        dst_linear_state_indices: T.Tensor((compile_max_batch,), DEFAULT_DTYPE),
        dst_block_tables: T.Tensor(
            (compile_block_table_elems,), DEFAULT_DTYPE
        ),
        dst_input_embedding: T.Tensor(
            (compile_input_embedding_elems,), embedding_dtype
        ),
        actual_num_tokens: T.int32,
        padded_num_tokens: T.int32,
        actual_batch_size: T.int32,
        src_block_table_stride: T.int32,
        dst_block_table_stride: T.int32,
        actual_block_table_len: T.int32,
        src_positions_stride: T.int32,
        dst_positions_stride: T.int32,
        src_input_embedding_stride: T.int32,
        dst_input_embedding_stride: T.int32,
        actual_hidden_size: T.int32,
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            task_id = cid * VEC_NUM + vid
            block_m = (padded_num_tokens + task_num - 1) // task_num
            row_start = task_id * block_m
            rows_left = T.if_then_else(
                padded_num_tokens > row_start,
                padded_num_tokens - row_start,
                0,
            )
            num_rows_per_vec = T.if_then_else(
                rows_left < block_m,
                rows_left,
                block_m,
            )

            with T.Scope("V"):
                i32_scalar_ub = T.alloc_ub((1,), DEFAULT_DTYPE)
                i32_zero_ub = T.alloc_ub((1,), DEFAULT_DTYPE)
                i32_block_table_bulk_ub = T.alloc_shared(
                    (BLOCK_TABLE_BULK_SIZE,), DEFAULT_DTYPE
                )
                emb_scalar_ub = T.alloc_ub((1,), embedding_dtype)
                i32_zero_ub[0] = 0
                for row_local in T.serial(num_rows_per_vec):
                    row = row_start + row_local

                    # Group 1: tokens / positions / new_cache_slots
                    with T.If(row < actual_num_tokens):
                        with T.Then():
                            T.copy(src_tokens[row : row + 1], i32_scalar_ub[0:1])
                            T.copy(i32_scalar_ub[0:1], dst_tokens[row : row + 1])
                            T.copy(
                                src_new_cache_slots[row : row + 1],
                                i32_scalar_ub[0:1],
                            )
                            T.copy(
                                i32_scalar_ub[0:1],
                                dst_new_cache_slots[row : row + 1],
                            )
                            if use_mrope:
                                for axis in T.serial(MROPE_COMPONENTS):
                                    src_pos_idx = axis * src_positions_stride + row
                                    dst_pos_idx = axis * dst_positions_stride + row
                                    T.copy(
                                        src_positions[src_pos_idx : src_pos_idx + 1],
                                        i32_scalar_ub[0:1],
                                    )
                                    T.copy(
                                        i32_scalar_ub[0:1],
                                        dst_positions[dst_pos_idx : dst_pos_idx + 1],
                                    )
                            else:
                                T.copy(
                                    src_positions[row : row + 1],
                                    i32_scalar_ub[0:1],
                                )
                                T.copy(
                                    i32_scalar_ub[0:1],
                                    dst_positions[row : row + 1],
                                )
                        with T.Else():
                            T.copy(i32_zero_ub[0:1], dst_tokens[row : row + 1])
                            T.copy(
                                i32_zero_ub[0:1],
                                dst_new_cache_slots[row : row + 1],
                            )

                    # Group 2: batch metadata + block tables
                    with T.If(row < actual_batch_size):
                        with T.Then():
                            T.copy(
                                src_q_seq_lens[row : row + 1],
                                i32_scalar_ub[0:1],
                            )
                            T.copy(
                                i32_scalar_ub[0:1],
                                dst_q_seq_lens[row : row + 1],
                            )
                            T.copy(
                                src_kv_seq_lens[row : row + 1],
                                i32_scalar_ub[0:1],
                            )
                            T.copy(
                                i32_scalar_ub[0:1],
                                dst_kv_seq_lens[row : row + 1],
                            )
                            if use_q_cu_seq_lens:
                                T.copy(
                                    src_q_cu_seq_lens[row : row + 1],
                                    i32_scalar_ub[0:1],
                                )
                                T.copy(
                                    i32_scalar_ub[0:1],
                                    dst_q_cu_seq_lens[row : row + 1],
                                )
                            if use_linear_state_indices:
                                T.copy(
                                    src_linear_state_indices[row : row + 1],
                                    i32_scalar_ub[0:1],
                                )
                                T.copy(
                                    i32_scalar_ub[0:1],
                                    dst_linear_state_indices[row : row + 1],
                                )

                            src_row_offset = row * src_block_table_stride
                            dst_row_offset = row * dst_block_table_stride
                            block_table_bulk_chunks = (
                                actual_block_table_len
                                + BLOCK_TABLE_BULK_SIZE
                                - 1
                            ) // BLOCK_TABLE_BULK_SIZE
                            for chunk_idx in T.serial(block_table_bulk_chunks):
                                col_start = chunk_idx * BLOCK_TABLE_BULK_SIZE
                                col_end = col_start + BLOCK_TABLE_BULK_SIZE
                                with T.If(col_end <= actual_block_table_len):
                                    with T.Then():
                                        T.copy(
                                            src_block_tables[
                                                src_row_offset
                                                + col_start : src_row_offset
                                                + col_end
                                            ],
                                            i32_block_table_bulk_ub[
                                                0:BLOCK_TABLE_BULK_SIZE
                                            ],
                                        )
                                        T.copy(
                                            i32_block_table_bulk_ub[
                                                0:BLOCK_TABLE_BULK_SIZE
                                            ],
                                            dst_block_tables[
                                                dst_row_offset
                                                + col_start : dst_row_offset
                                                + col_end
                                            ],
                                        )
                                    with T.Else():
                                        with T.If(col_start < actual_block_table_len):
                                            with T.Then():
                                                for elem in T.serial(
                                                    BLOCK_TABLE_BULK_SIZE
                                                ):
                                                    col = col_start + elem
                                                    with T.If(
                                                        col < actual_block_table_len
                                                    ):
                                                        with T.Then():
                                                            T.copy(
                                                                src_block_tables[
                                                                    src_row_offset
                                                                    + col : src_row_offset
                                                                    + col
                                                                    + 1
                                                                ],
                                                                i32_scalar_ub[0:1],
                                                            )
                                                            T.copy(
                                                                i32_scalar_ub[0:1],
                                                                dst_block_tables[
                                                                    dst_row_offset
                                                                    + col : dst_row_offset
                                                                    + col
                                                                    + 1
                                                                ],
                                                            )

                    # Group 3: input embedding
                    if use_input_embedding:
                        with T.If(row < actual_num_tokens):
                            with T.Then():
                                src_emb_row_offset = row * src_input_embedding_stride
                                dst_emb_row_offset = row * dst_input_embedding_stride
                                for col in T.serial(actual_hidden_size):
                                    T.copy(
                                        src_input_embedding[
                                            src_emb_row_offset
                                            + col : src_emb_row_offset
                                            + col
                                            + 1
                                        ],
                                        emb_scalar_ub[0:1],
                                    )
                                    T.copy(
                                        emb_scalar_ub[0:1],
                                        dst_input_embedding[
                                            dst_emb_row_offset
                                            + col : dst_emb_row_offset
                                            + col
                                            + 1
                                        ],
                                    )

    return model_input_buffer_updater_kernel


@tilelang.jit(pass_configs=DEFAULT_ASCEND_PASS_CONFIGS)
def model_input_buffer_updater_kernel_jit(
    compile_max_tokens: int,
    compile_max_batch: int,
    compile_max_block_table_len: int,
    compile_hidden_size: int,
    vec_core_num: int,
    embedding_dtype: str,
    with_mrope: int,
    with_input_embedding: int,
    with_linear_state_indices: int,
    with_q_cu_seq_lens: int,
):
    return build_model_input_buffer_updater_kernel(
        compile_max_tokens=compile_max_tokens,
        compile_max_batch=compile_max_batch,
        compile_max_block_table_len=compile_max_block_table_len,
        compile_hidden_size=compile_hidden_size,
        vec_core_num=vec_core_num,
        embedding_dtype=embedding_dtype,
        with_mrope=with_mrope,
        with_input_embedding=with_input_embedding,
        with_linear_state_indices=with_linear_state_indices,
        with_q_cu_seq_lens=with_q_cu_seq_lens,
    )


def generate_source(
    *,
    compile_hidden_size: int,
    embedding_dtype: str,
    with_mrope: int,
    with_input_embedding: int,
    with_linear_state_indices: int,
    with_q_cu_seq_lens: int,
) -> str:
    tilelang.disable_cache()
    tilelang_kernel = build_model_input_buffer_updater_kernel(
        compile_max_tokens=COMPILE_MAX_TOKENS,
        compile_max_batch=COMPILE_MAX_BATCH,
        compile_max_block_table_len=COMPILE_MAX_BLOCK_TABLE_LEN,
        compile_hidden_size=compile_hidden_size,
        vec_core_num=detect_vec_core_num(),
        embedding_dtype=embedding_dtype,
        with_mrope=with_mrope,
        with_input_embedding=with_input_embedding,
        with_linear_state_indices=with_linear_state_indices,
        with_q_cu_seq_lens=with_q_cu_seq_lens,
    )
    with tilelang.tvm.transform.PassContext(
        opt_level=3, config=DEFAULT_ASCEND_PASS_CONFIGS
    ):
        kernel = tilelang.engine.lower(tilelang_kernel)
    return kernel.kernel_source


def _assert_tensor_equal(actual, expected, name: str) -> None:
    import torch

    if not torch.equal(actual.cpu(), expected.cpu()):
        raise AssertionError(
            f"{name} mismatch:\nactual={actual.cpu()}\nexpected={expected.cpu()}"
        )


def _apply_block_table_ref_update(
    *,
    src_block_tables,
    dst_block_tables,
    actual_batch_size: int,
    src_block_table_stride: int,
    dst_block_table_stride: int,
    actual_block_table_len: int,
) -> None:
    for row in range(actual_batch_size):
        src_row_offset = row * src_block_table_stride
        dst_row_offset = row * dst_block_table_stride
        dst_block_tables[
            dst_row_offset : dst_row_offset + actual_block_table_len
        ].copy_(
            src_block_tables[
                src_row_offset : src_row_offset + actual_block_table_len
            ],
            non_blocking=True,
        )


def _make_ref_case(
    *,
    device,
    compile_max_tokens: int,
    compile_max_batch: int,
    compile_max_block_table_len: int,
    actual_num_tokens: int,
    padded_num_tokens: int,
    actual_batch_size: int,
    actual_block_table_len: int,
    dst_block_table_stride: int,
    compile_hidden_size: int,
    actual_hidden_size: int,
    embedding_dtype: str,
    with_mrope: int,
    with_input_embedding: int,
    with_q_cu_seq_lens: int,
    with_linear_state_indices: int,
    seed: int,
):
    import torch

    torch.manual_seed(seed)
    int_kwargs = {"device": device, "dtype": torch.int32}
    emb_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[embedding_dtype]
    emb_kwargs = {"device": device, "dtype": emb_dtype}

    src_block_table_stride = actual_block_table_len
    compile_position_elems = compile_max_tokens * MROPE_COMPONENTS
    compile_block_table_elems = compile_max_batch * compile_max_block_table_len
    compile_input_embedding_elems = compile_max_tokens * compile_hidden_size
    src_positions_stride = actual_num_tokens if with_mrope else 0
    dst_positions_stride = compile_max_tokens if with_mrope else 0

    src_tokens = torch.randint(0, 32000, (compile_max_tokens,), **int_kwargs)
    src_positions = torch.randint(
        0, 32000, (compile_position_elems,), **int_kwargs
    )
    src_new_cache_slots = torch.randint(
        0, 2048, (compile_max_tokens,), **int_kwargs
    )
    src_q_seq_lens = torch.randint(1, 3, (compile_max_batch,), **int_kwargs)
    src_kv_seq_lens = torch.randint(1, 4096, (compile_max_batch,), **int_kwargs)
    src_q_cu_seq_lens = torch.randint(
        1, 4096, (compile_max_batch,), **int_kwargs
    )
    src_linear_state_indices = torch.randint(
        0, 4096, (compile_max_batch,), **int_kwargs
    )
    src_block_tables = torch.randint(
        0, 4096, (compile_block_table_elems,), **int_kwargs
    )
    src_input_embedding = torch.full(
        (compile_input_embedding_elems,), -1, **emb_kwargs
    )
    src_input_embedding_stride = actual_hidden_size if with_input_embedding else 0
    dst_input_embedding_stride = actual_hidden_size if with_input_embedding else 0
    if with_input_embedding:
        src_input_embedding[
            : compile_max_tokens * actual_hidden_size
        ] = torch.randn(
            (compile_max_tokens * actual_hidden_size,), **emb_kwargs
        )

    dst_tokens = torch.full((compile_max_tokens,), -1, **int_kwargs)
    dst_positions = torch.full((compile_position_elems,), -1, **int_kwargs)
    dst_new_cache_slots = torch.full((compile_max_tokens,), -1, **int_kwargs)
    dst_q_seq_lens = torch.full((compile_max_batch,), -1, **int_kwargs)
    dst_kv_seq_lens = torch.full((compile_max_batch,), -1, **int_kwargs)
    dst_q_cu_seq_lens = torch.full((compile_max_batch,), -1, **int_kwargs)
    dst_linear_state_indices = torch.full(
        (compile_max_batch,), -1, **int_kwargs
    )
    dst_block_tables = torch.full(
        (compile_block_table_elems,), -1, **int_kwargs
    )
    dst_input_embedding = torch.full(
        (compile_input_embedding_elems,), -1, **emb_kwargs
    )

    ref = {
        "dst_tokens": dst_tokens.clone(),
        "dst_positions": dst_positions.clone(),
        "dst_new_cache_slots": dst_new_cache_slots.clone(),
        "dst_q_seq_lens": dst_q_seq_lens.clone(),
        "dst_kv_seq_lens": dst_kv_seq_lens.clone(),
        "dst_q_cu_seq_lens": dst_q_cu_seq_lens.clone(),
        "dst_linear_state_indices": dst_linear_state_indices.clone(),
        "dst_block_tables": dst_block_tables.clone(),
        "dst_input_embedding": dst_input_embedding.clone(),
    }

    ref["dst_tokens"][:actual_num_tokens].copy_(
        src_tokens[:actual_num_tokens], non_blocking=True
    )
    ref["dst_new_cache_slots"][:actual_num_tokens].copy_(
        src_new_cache_slots[:actual_num_tokens], non_blocking=True
    )

    if with_mrope:
        for axis in range(MROPE_COMPONENTS):
            src_start = axis * src_positions_stride
            dst_start = axis * dst_positions_stride
            ref["dst_positions"][dst_start : dst_start + actual_num_tokens].copy_(
                src_positions[src_start : src_start + actual_num_tokens],
                non_blocking=True,
            )
    else:
        ref["dst_positions"][:actual_num_tokens].copy_(
            src_positions[:actual_num_tokens], non_blocking=True
        )

    ref["dst_q_seq_lens"][:actual_batch_size].copy_(
        src_q_seq_lens[:actual_batch_size], non_blocking=True
    )
    ref["dst_kv_seq_lens"][:actual_batch_size].copy_(
        src_kv_seq_lens[:actual_batch_size], non_blocking=True
    )
    if with_q_cu_seq_lens:
        ref["dst_q_cu_seq_lens"][:actual_batch_size].copy_(
            src_q_cu_seq_lens[:actual_batch_size], non_blocking=True
        )
    if with_linear_state_indices:
        ref["dst_linear_state_indices"][:actual_batch_size].copy_(
            src_linear_state_indices[:actual_batch_size], non_blocking=True
        )

    if padded_num_tokens > actual_num_tokens:
        ref["dst_tokens"][actual_num_tokens:padded_num_tokens].fill_(0)
        ref["dst_new_cache_slots"][actual_num_tokens:padded_num_tokens].fill_(0)

    _apply_block_table_ref_update(
        src_block_tables=src_block_tables,
        dst_block_tables=ref["dst_block_tables"],
        actual_batch_size=actual_batch_size,
        src_block_table_stride=src_block_table_stride,
        dst_block_table_stride=dst_block_table_stride,
        actual_block_table_len=actual_block_table_len,
    )

    if with_input_embedding:
        ref["dst_input_embedding"][
            : actual_num_tokens * actual_hidden_size
        ].view(actual_num_tokens, actual_hidden_size).copy_(
            src_input_embedding[: actual_num_tokens * actual_hidden_size].view(
                actual_num_tokens, actual_hidden_size
            ),
            non_blocking=True,
        )

    return {
        "src_tokens": src_tokens,
        "src_positions": src_positions,
        "src_new_cache_slots": src_new_cache_slots,
        "src_q_seq_lens": src_q_seq_lens,
        "src_kv_seq_lens": src_kv_seq_lens,
        "src_q_cu_seq_lens": src_q_cu_seq_lens,
        "src_linear_state_indices": src_linear_state_indices,
        "src_block_tables": src_block_tables,
        "src_input_embedding": src_input_embedding,
        "dst_tokens": dst_tokens,
        "dst_positions": dst_positions,
        "dst_new_cache_slots": dst_new_cache_slots,
        "dst_q_seq_lens": dst_q_seq_lens,
        "dst_kv_seq_lens": dst_kv_seq_lens,
        "dst_q_cu_seq_lens": dst_q_cu_seq_lens,
        "dst_linear_state_indices": dst_linear_state_indices,
        "dst_block_tables": dst_block_tables,
        "dst_input_embedding": dst_input_embedding,
        "ref": ref,
        "src_block_table_stride": src_block_table_stride,
        "dst_block_table_stride": dst_block_table_stride,
        "actual_num_tokens": actual_num_tokens,
        "padded_num_tokens": padded_num_tokens,
        "actual_batch_size": actual_batch_size,
        "actual_block_table_len": actual_block_table_len,
        "src_positions_stride": src_positions_stride,
        "dst_positions_stride": dst_positions_stride,
        "src_input_embedding_stride": src_input_embedding_stride,
        "dst_input_embedding_stride": dst_input_embedding_stride,
        "actual_hidden_size": actual_hidden_size,
    }


def _run_ref_check(
    *,
    device_index: int,
    compile_hidden_size: int,
    embedding_dtype: str,
) -> None:
    import torch

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available")
    torch.npu.set_device(device_index)
    device = torch.device(f"npu:{device_index}")
    vec_core_num = detect_vec_core_num()
    compile_max_tokens = REF_CHECK_COMPILE_MAX_TOKENS
    compile_max_batch = REF_CHECK_COMPILE_MAX_BATCH
    compile_max_block_table_len = REF_CHECK_COMPILE_MAX_BLOCK_TABLE_LEN
    compile_hidden_size = max(
        min(compile_hidden_size, REF_CHECK_COMPILE_MAX_HIDDEN_SIZE), 128
    )

    cases = [
        {
            "name": "plain",
            "actual_num_tokens": 4,
            "padded_num_tokens": 8,
            "actual_batch_size": 3,
            "actual_block_table_len": 5,
            "dst_block_table_stride": 16,
            "with_mrope": 0,
            "with_input_embedding": 0,
            "with_q_cu_seq_lens": 0,
            "with_linear_state_indices": 0,
            "actual_hidden_size": 0,
            "seed": 11,
        },
        {
            "name": "mrope",
            "actual_num_tokens": 5,
            "padded_num_tokens": 8,
            "actual_batch_size": 4,
            "actual_block_table_len": 7,
            "dst_block_table_stride": 16,
            "with_mrope": 1,
            "with_input_embedding": 0,
            "with_q_cu_seq_lens": 1,
            "with_linear_state_indices": 0,
            "actual_hidden_size": 0,
            "seed": 12,
        },
        {
            "name": "embedding",
            "actual_num_tokens": 6,
            "padded_num_tokens": 8,
            "actual_batch_size": 5,
            "actual_block_table_len": 9,
            "dst_block_table_stride": 16,
            "with_mrope": 0,
            "with_input_embedding": 1,
            "with_q_cu_seq_lens": 0,
            "with_linear_state_indices": 1,
            "actual_hidden_size": 96,
            "seed": 13,
        },
        {
            "name": "all",
            "actual_num_tokens": 9,
            "padded_num_tokens": 16,
            "actual_batch_size": 8,
            "actual_block_table_len": 13,
            "dst_block_table_stride": 32,
            "with_mrope": 1,
            "with_input_embedding": 1,
            "with_q_cu_seq_lens": 1,
            "with_linear_state_indices": 1,
            "actual_hidden_size": 128,
            "seed": 14,
        },
    ]

    for case in cases:
        case_args = {k: v for k, v in case.items() if k != "name"}
        kernel = model_input_buffer_updater_kernel_jit(
            compile_max_tokens=compile_max_tokens,
            compile_max_batch=compile_max_batch,
            compile_max_block_table_len=compile_max_block_table_len,
            compile_hidden_size=compile_hidden_size,
            vec_core_num=vec_core_num,
            embedding_dtype=embedding_dtype,
            with_mrope=case["with_mrope"],
            with_input_embedding=case["with_input_embedding"],
            with_linear_state_indices=case["with_linear_state_indices"],
            with_q_cu_seq_lens=case["with_q_cu_seq_lens"],
        )
        tensors = _make_ref_case(
            device=device,
            compile_max_tokens=compile_max_tokens,
            compile_max_batch=compile_max_batch,
            compile_max_block_table_len=compile_max_block_table_len,
            compile_hidden_size=compile_hidden_size,
            embedding_dtype=embedding_dtype,
            **case_args,
        )
        kernel(
            tensors["src_tokens"],
            tensors["src_positions"],
            tensors["src_new_cache_slots"],
            tensors["src_q_seq_lens"],
            tensors["src_kv_seq_lens"],
            tensors["src_q_cu_seq_lens"],
            tensors["src_linear_state_indices"],
            tensors["src_block_tables"],
            tensors["src_input_embedding"],
            tensors["dst_tokens"],
            tensors["dst_positions"],
            tensors["dst_new_cache_slots"],
            tensors["dst_q_seq_lens"],
            tensors["dst_kv_seq_lens"],
            tensors["dst_q_cu_seq_lens"],
            tensors["dst_linear_state_indices"],
            tensors["dst_block_tables"],
            tensors["dst_input_embedding"],
            tensors["actual_num_tokens"],
            tensors["padded_num_tokens"],
            tensors["actual_batch_size"],
            tensors["src_block_table_stride"],
            tensors["dst_block_table_stride"],
            tensors["actual_block_table_len"],
            tensors["src_positions_stride"],
            tensors["dst_positions_stride"],
            tensors["src_input_embedding_stride"],
            tensors["dst_input_embedding_stride"],
            tensors["actual_hidden_size"],
        )
        torch.npu.synchronize()

        _assert_tensor_equal(
            tensors["dst_tokens"],
            tensors["ref"]["dst_tokens"],
            case["name"] + ":tokens",
        )
        _assert_tensor_equal(
            tensors["dst_positions"],
            tensors["ref"]["dst_positions"],
            case["name"] + ":positions",
        )
        _assert_tensor_equal(
            tensors["dst_new_cache_slots"],
            tensors["ref"]["dst_new_cache_slots"],
            case["name"] + ":new_cache_slots",
        )
        _assert_tensor_equal(
            tensors["dst_q_seq_lens"],
            tensors["ref"]["dst_q_seq_lens"],
            case["name"] + ":q_seq_lens",
        )
        _assert_tensor_equal(
            tensors["dst_kv_seq_lens"],
            tensors["ref"]["dst_kv_seq_lens"],
            case["name"] + ":kv_seq_lens",
        )
        if case["with_q_cu_seq_lens"]:
            _assert_tensor_equal(
                tensors["dst_q_cu_seq_lens"],
                tensors["ref"]["dst_q_cu_seq_lens"],
                case["name"] + ":q_cu_seq_lens",
            )
        if case["with_linear_state_indices"]:
            _assert_tensor_equal(
                tensors["dst_linear_state_indices"],
                tensors["ref"]["dst_linear_state_indices"],
                case["name"] + ":linear_state_indices",
            )
        _assert_tensor_equal(
            tensors["dst_block_tables"],
            tensors["ref"]["dst_block_tables"],
            case["name"] + ":block_tables",
        )
        if case["with_input_embedding"]:
            _assert_tensor_equal(
                tensors["dst_input_embedding"],
                tensors["ref"]["dst_input_embedding"],
                case["name"] + ":input_embedding",
            )

    print("[INFO] model_input_buffer_updater ref-check passed")


def _embedding_dtype_variant_token(embedding_dtype: str) -> str:
    return {
        "float16": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
    }[embedding_dtype]


def _registered_specializations() -> list[dict[str, object]]:
    specializations: list[dict[str, object]] = []
    for with_input_embedding in (0, 1):
        embedding_dtypes = (
            (NO_INPUT_EMBEDDING_DTYPE,)
            if with_input_embedding == 0
            else ("float16", "bfloat16", "float32")
        )
        for (
            embedding_dtype,
            with_mrope,
            with_linear_state_indices,
            with_q_cu_seq_lens,
        ) in product(
            embedding_dtypes,
            (0, 1),
            (0, 1),
            (0, 1),
        ):
            variant_key = (
                "emb"
                f"{_embedding_dtype_variant_token(embedding_dtype)}"
                f"_mrope{with_mrope}"
                f"_iemb{with_input_embedding}"
                f"_lsi{with_linear_state_indices}"
                f"_qcu{with_q_cu_seq_lens}"
            )
            specializations.append(
                {
                    "variant_key": variant_key,
                    "embedding_dtype": embedding_dtype,
                    "with_mrope": with_mrope,
                    "with_input_embedding": with_input_embedding,
                    "with_linear_state_indices": with_linear_state_indices,
                    "with_q_cu_seq_lens": with_q_cu_seq_lens,
                }
            )
    return specializations


@register_kernel
class ModelInputBufferUpdater(TilelangKernel):
    KERNEL_NAME = "model_input_buffer_updater"
    DISPATCH_SCHEMA = [
        DispatchField("embedding_dtype", "dtype"),
        DispatchField("with_mrope", "int32"),
        DispatchField("with_input_embedding", "int32"),
        DispatchField("with_linear_state_indices", "int32"),
        DispatchField("with_q_cu_seq_lens", "int32"),
    ]
    SPECIALIZATIONS = _registered_specializations()

    @staticmethod
    def generate_source(
        *,
        embedding_dtype: str,
        with_mrope: int,
        with_input_embedding: int,
        with_linear_state_indices: int,
        with_q_cu_seq_lens: int,
    ) -> str:
        return generate_source(
            compile_hidden_size=COMPILE_MAX_HIDDEN_SIZE,
            embedding_dtype=embedding_dtype,
            with_mrope=with_mrope,
            with_input_embedding=with_input_embedding,
            with_linear_state_indices=with_linear_state_indices,
            with_q_cu_seq_lens=with_q_cu_seq_lens,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate TileLang AscendC source for model_input_buffer_updater."
        )
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output AscendC .cpp file. Required unless --run-ref-check is set.",
    )
    parser.add_argument(
        "--run-ref-check",
        action="store_true",
        help="Run Python-side torch reference checks instead of generating source.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="NPU device index used by --run-ref-check.",
    )
    parser.add_argument(
        "--compile-hidden-size",
        type=int,
        default=4096,
        help="Compile-time hidden size used by input_embedding group.",
    )
    parser.add_argument(
        "--embedding-dtype",
        default=DEFAULT_EMBEDDING_DTYPE,
        choices=["float16", "bfloat16", "float32"],
        help="Embedding dtype used by input_embedding group.",
    )
    parser.add_argument(
        "--with-mrope",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable mRoPE positions group.",
    )
    parser.add_argument(
        "--with-input-embedding",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable input_embedding group.",
    )
    parser.add_argument(
        "--with-linear-state-indices",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable linear_state_indices group.",
    )
    parser.add_argument(
        "--with-q-cu-seq-lens",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable q_cu_seq_lens group.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_ref_check:
        if args.device_index < 0:
            raise ValueError("--device-index must be >= 0")
        _run_ref_check(
            device_index=args.device_index,
            compile_hidden_size=args.compile_hidden_size,
            embedding_dtype=args.embedding_dtype,
        )
        return

    if not args.output:
        raise ValueError("--output is required unless --run-ref-check is set")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source = generate_source(
        compile_hidden_size=args.compile_hidden_size,
        embedding_dtype=args.embedding_dtype,
        with_mrope=args.with_mrope,
        with_input_embedding=args.with_input_embedding,
        with_linear_state_indices=args.with_linear_state_indices,
        with_q_cu_seq_lens=args.with_q_cu_seq_lens,
    )
    output.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
