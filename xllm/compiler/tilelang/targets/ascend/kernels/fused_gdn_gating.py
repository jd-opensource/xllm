#!/usr/bin/env python3

import argparse
from pathlib import Path

import tilelang
import tilelang.language as T

from compiler.tilelang.targets.ascend.kernels.utils import (
    DEFAULT_ASCEND_PASS_CONFIGS,
    detect_vec_core_num,
)
from compiler.tilelang.common.spec import (
    DispatchField,
    TilelangKernel,
    register_kernel,
)

DEFAULT_NUM_HEADS = 32
DEFAULT_DTYPE = "bf16"
DEFAULT_MAX_BATCH = 4096
REF_CHECK_NUM_BATCHES = 16
VEC_NUM = 2
VECTOR_BYTES_PER_ITER = 256
SUPPORTED_NUM_HEADS = (16, 32, 48, 64, 128)
MIN_FUSED_TASK_ELEMS_PER_TASK = 64


def _dtype_size_in_bytes(dtype: str) -> int:
    sizes = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
    }
    if dtype not in sizes:
        raise ValueError(f"Unsupported dtype for vector alignment: {dtype}")
    return sizes[dtype]


def _align_count_to_vector_bytes(count: int, dtype: str) -> int:
    elem_bytes = _dtype_size_in_bytes(dtype)
    elems_per_iter = VECTOR_BYTES_PER_ITER // elem_bytes
    return ((count + elems_per_iter - 1) // elems_per_iter) * elems_per_iter


def build_fused_gdn_gating_kernel(
    *,
    num_heads: int,
    vec_core_num: int,
    compile_max_batch: int,
):
    if num_heads not in SUPPORTED_NUM_HEADS:
        raise ValueError(
            f"fused_gdn_gating only supports num_heads in {SUPPORTED_NUM_HEADS}, got {num_heads}"
        )
    if vec_core_num <= 0:
        raise ValueError(f"vec_core_num({vec_core_num}) must be > 0")
    if vec_core_num % VEC_NUM != 0:
        raise ValueError(
            f"vec_core_num({vec_core_num}) must be divisible by VEC_NUM({VEC_NUM})"
        )
    if compile_max_batch <= 0:
        raise ValueError(
            f"compile_max_batch({compile_max_batch}) must be > 0"
        )

    task_num = vec_core_num
    m_num = vec_core_num // VEC_NUM
    acc_dtype = "float32"
    input_dtype = "bfloat16"
    mask_dtype = "uint8"
    ub_tensor_dim = _align_count_to_vector_bytes(num_heads, acc_dtype)
    compare_select_mask_bytes = ub_tensor_dim // 8
    fused_group_rows = ub_tensor_dim // num_heads if num_heads < ub_tensor_dim else 1
    fused_task_threshold_elems = task_num * MIN_FUSED_TASK_ELEMS_PER_TASK
    enable_small_head_fusion = num_heads < ub_tensor_dim

    @T.macro
    def init_head_constants(
        A_log: T.Buffer,
        dt_bias: T.Buffer,
        A_log_ub: T.Buffer,
        dt_bias_ub: T.Buffer,
        neg_exp_A_ub: T.Buffer,
    ) -> None:
        if enable_small_head_fusion:
            for repeat_idx in T.serial(fused_group_rows):
                start = repeat_idx * num_heads
                stop = start + num_heads
                T.copy(A_log[0], A_log_ub[0, start:stop])
                T.copy(dt_bias[0], dt_bias_ub[0, start:stop])
        else:
            T.copy(A_log[0], A_log_ub[0, :num_heads])
            T.copy(dt_bias[0], dt_bias_ub[0, :num_heads])
        T.tile.exp(neg_exp_A_ub, A_log_ub)
        T.tile.mul(neg_exp_A_ub, neg_exp_A_ub, -1.0)

    @T.macro
    def run_row_pipeline(
        row,
        io_count,
        softplus_beta,
        softplus_threshold,
        a: T.Buffer,
        b: T.Buffer,
        g_out: T.Buffer,
        beta_out: T.Buffer,
        dt_bias_ub: T.Buffer,
        neg_exp_A_ub: T.Buffer,
        a_half_ub: T.Buffer,
        b_half_ub: T.Buffer,
        a_fp32_ub: T.Buffer,
        b_fp32_ub: T.Buffer,
        x_ub: T.Buffer,
        beta_x_ub: T.Buffer,
        softplus_abs_ub: T.Buffer,
        softplus_neg_abs_ub: T.Buffer,
        softplus_exp_ub: T.Buffer,
        softplus_log_ub: T.Buffer,
        softplus_x_ub: T.Buffer,
        beta_fp32_ub: T.Buffer,
        g_ub: T.Buffer,
        beta_half_ub: T.Buffer,
        sigmoid_tmp_ub: T.Buffer,
        softplus_cmp_mask_ub: T.Buffer,
    ) -> None:
        T.copy(a[row, 0], a_half_ub[0, :io_count])
        T.copy(b[row, 0], b_half_ub[0, :io_count])

        T.tile.cast(a_fp32_ub, a_half_ub, "CAST_NONE", io_count)
        T.tile.cast(b_fp32_ub, b_half_ub, "CAST_NONE", io_count)

        T.tile.add(x_ub, a_fp32_ub, dt_bias_ub)
        T.tile.mul(beta_x_ub, x_ub, softplus_beta)
        T.tile.abs(softplus_abs_ub, beta_x_ub)
        T.tile.mul(softplus_neg_abs_ub, softplus_abs_ub, -1.0)
        T.tile.exp(softplus_exp_ub, softplus_neg_abs_ub)
        T.tile.add(softplus_exp_ub, softplus_exp_ub, 1.0)
        T.tile.ln(softplus_log_ub, softplus_exp_ub)

        # Ascend compare/select consumes one 256B vector chunk per iteration.
        # For float32 this is 64 elements, so num_heads < 64 must still use
        # UB tensors aligned to the full 256B chunk.
        T.tile.compare(
            softplus_cmp_mask_ub,
            beta_x_ub,
            softplus_threshold,
            "GT",
        )
        # softplus(x) = log(1 + exp(-abs(beta_x))) / beta
        #             + 0.5 * (beta_x + abs(beta_x)) / beta
        T.tile.mul(softplus_x_ub, beta_x_ub, 0.5 / softplus_beta)
        T.tile.axpy(softplus_x_ub, softplus_abs_ub, 0.5 / softplus_beta)
        T.tile.axpy(softplus_x_ub, softplus_log_ub, 1.0 / softplus_beta)
        T.tile.select(
            softplus_x_ub,
            softplus_cmp_mask_ub,
            x_ub,
            softplus_x_ub,
            "VSEL_TENSOR_TENSOR_MODE",
        )

        T.tile.sigmoid(beta_fp32_ub, b_fp32_ub, sigmoid_tmp_ub)
        T.tile.mul(g_ub, neg_exp_A_ub, softplus_x_ub)
        T.tile.cast(beta_half_ub, beta_fp32_ub, "CAST_RINT", io_count)

        T.copy(g_ub[0, :io_count], g_out[row, 0])
        T.copy(beta_half_ub[0, :io_count], beta_out[row, 0])

    @T.prim_func
    def fused_gdn_gating_kernel(
        A_log: T.Tensor((num_heads,), acc_dtype),
        a: T.Tensor((compile_max_batch, num_heads), input_dtype),
        b: T.Tensor((compile_max_batch, num_heads), input_dtype),
        dt_bias: T.Tensor((num_heads,), acc_dtype),
        g_out: T.Tensor((compile_max_batch, num_heads), acc_dtype),
        beta_out: T.Tensor((compile_max_batch, num_heads), input_dtype),
        num_batches: T.int32,
        softplus_beta: T.float32,
        softplus_threshold: T.float32,
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            task_id = cid * VEC_NUM + vid
            block_m = (num_batches + task_num - 1) // task_num
            row_start = task_id * block_m
            rows_left = T.if_then_else(
                num_batches > row_start, num_batches - row_start, 0
            )
            num_rows_per_vec = T.if_then_else(
                rows_left < block_m,
                rows_left,
                block_m,
            )

            with T.Scope("V"):
                A_log_ub = T.alloc_shared((1, ub_tensor_dim), acc_dtype)
                neg_exp_A_ub = T.alloc_shared((1, ub_tensor_dim), acc_dtype)
                dt_bias_ub = T.alloc_shared((1, ub_tensor_dim), acc_dtype)
                a_half_ub = T.alloc_shared((1, ub_tensor_dim), input_dtype)
                b_half_ub = T.alloc_shared((1, ub_tensor_dim), input_dtype)
                a_fp32_ub = T.alloc_shared((1, ub_tensor_dim), acc_dtype)
                b_fp32_ub = T.alloc_shared((1, ub_tensor_dim), acc_dtype)
                x_ub = T.alloc_shared((1, ub_tensor_dim), acc_dtype)
                beta_x_ub = T.alloc_shared((1, ub_tensor_dim), acc_dtype)
                softplus_abs_ub = T.alloc_shared((1, ub_tensor_dim), acc_dtype)
                softplus_neg_abs_ub = T.alloc_shared((1, ub_tensor_dim), acc_dtype)
                softplus_exp_ub = T.alloc_shared((1, ub_tensor_dim), acc_dtype)
                softplus_log_ub = T.alloc_shared((1, ub_tensor_dim), acc_dtype)
                softplus_x_ub = T.alloc_shared((1, ub_tensor_dim), acc_dtype)
                beta_fp32_ub = T.alloc_shared((1, ub_tensor_dim), acc_dtype)
                g_ub = T.alloc_shared((1, ub_tensor_dim), acc_dtype)
                beta_half_ub = T.alloc_shared((1, ub_tensor_dim), input_dtype)
                sigmoid_tmp_ub = T.alloc_ub((1, ub_tensor_dim), mask_dtype)
                softplus_cmp_mask_ub = T.alloc_ub(
                    (1, compare_select_mask_bytes), mask_dtype
                )

                init_head_constants(
                    A_log,
                    dt_bias,
                    A_log_ub,
                    dt_bias_ub,
                    neg_exp_A_ub,
                )

                if enable_small_head_fusion:
                    total_flat_elems = num_batches * num_heads
                    if total_flat_elems >= fused_task_threshold_elems:
                        fused_full_batches = (
                            num_batches // fused_group_rows
                        ) * fused_group_rows
                        fused_group_count = fused_full_batches // fused_group_rows
                        fused_block_groups = (
                            fused_group_count + task_num - 1
                        ) // task_num
                        group_start = task_id * fused_block_groups
                        groups_left = T.if_then_else(
                            fused_group_count > group_start,
                            fused_group_count - group_start,
                            0,
                        )
                        num_groups_per_vec = T.if_then_else(
                            groups_left < fused_block_groups,
                            groups_left,
                            fused_block_groups,
                        )

                        for group_local in T.serial(num_groups_per_vec):
                            group = group_start + group_local
                            row = group * fused_group_rows
                            run_row_pipeline(
                                row,
                                ub_tensor_dim,
                                softplus_beta,
                                softplus_threshold,
                                a,
                                b,
                                g_out,
                                beta_out,
                                dt_bias_ub,
                                neg_exp_A_ub,
                                a_half_ub,
                                b_half_ub,
                                a_fp32_ub,
                                b_fp32_ub,
                                x_ub,
                                beta_x_ub,
                                softplus_abs_ub,
                                softplus_neg_abs_ub,
                                softplus_exp_ub,
                                softplus_log_ub,
                                softplus_x_ub,
                                beta_fp32_ub,
                                g_ub,
                                beta_half_ub,
                                sigmoid_tmp_ub,
                                softplus_cmp_mask_ub,
                            )

                        tail_batches = num_batches - fused_full_batches
                        if task_id < tail_batches:
                            row = fused_full_batches + task_id
                            run_row_pipeline(
                                row,
                                num_heads,
                                softplus_beta,
                                softplus_threshold,
                                a,
                                b,
                                g_out,
                                beta_out,
                                dt_bias_ub,
                                neg_exp_A_ub,
                                a_half_ub,
                                b_half_ub,
                                a_fp32_ub,
                                b_fp32_ub,
                                x_ub,
                                beta_x_ub,
                                softplus_abs_ub,
                                softplus_neg_abs_ub,
                                softplus_exp_ub,
                                softplus_log_ub,
                                softplus_x_ub,
                                beta_fp32_ub,
                                g_ub,
                                beta_half_ub,
                                sigmoid_tmp_ub,
                                softplus_cmp_mask_ub,
                            )
                    else:
                        for row_local in T.serial(num_rows_per_vec):
                            row = row_start + row_local
                            run_row_pipeline(
                                row,
                                num_heads,
                                softplus_beta,
                                softplus_threshold,
                                a,
                                b,
                                g_out,
                                beta_out,
                                dt_bias_ub,
                                neg_exp_A_ub,
                                a_half_ub,
                                b_half_ub,
                                a_fp32_ub,
                                b_fp32_ub,
                                x_ub,
                                beta_x_ub,
                                softplus_abs_ub,
                                softplus_neg_abs_ub,
                                softplus_exp_ub,
                                softplus_log_ub,
                                softplus_x_ub,
                                beta_fp32_ub,
                                g_ub,
                                beta_half_ub,
                                sigmoid_tmp_ub,
                                softplus_cmp_mask_ub,
                            )
                else:
                    for row_local in T.serial(num_rows_per_vec):
                        row = row_start + row_local
                        run_row_pipeline(
                            row,
                            num_heads,
                            softplus_beta,
                            softplus_threshold,
                            a,
                            b,
                            g_out,
                            beta_out,
                            dt_bias_ub,
                            neg_exp_A_ub,
                            a_half_ub,
                            b_half_ub,
                            a_fp32_ub,
                            b_fp32_ub,
                            x_ub,
                            beta_x_ub,
                            softplus_abs_ub,
                            softplus_neg_abs_ub,
                            softplus_exp_ub,
                            softplus_log_ub,
                            softplus_x_ub,
                            beta_fp32_ub,
                            g_ub,
                            beta_half_ub,
                            sigmoid_tmp_ub,
                            softplus_cmp_mask_ub,
                        )

    return fused_gdn_gating_kernel


@tilelang.jit(pass_configs=DEFAULT_ASCEND_PASS_CONFIGS)
def fused_gdn_gating_kernel_jit(
    num_heads: int,
    vec_core_num: int,
    compile_max_batch: int,
):
    return build_fused_gdn_gating_kernel(
        num_heads=num_heads,
        vec_core_num=vec_core_num,
        compile_max_batch=compile_max_batch,
    )


@register_kernel
class FusedGdnGatingKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("num_heads", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": "nh16_bf16",
            "num_heads": 16,
            "dtype": DEFAULT_DTYPE,
        },
        {
            "variant_key": "nh32_bf16",
            "num_heads": 32,
            "dtype": DEFAULT_DTYPE,
        },
        {
            "variant_key": "nh48_bf16",
            "num_heads": 48,
            "dtype": DEFAULT_DTYPE,
        },
        {
            "variant_key": "nh64_bf16",
            "num_heads": 64,
            "dtype": DEFAULT_DTYPE,
        },
        {
            "variant_key": "nh128_bf16",
            "num_heads": 128,
            "dtype": DEFAULT_DTYPE,
        },
    ]

    @staticmethod
    def generate_source(num_heads: int, dtype: str) -> str:
        if dtype != DEFAULT_DTYPE:
            raise ValueError(
                f"fused_gdn_gating only supports dtype={DEFAULT_DTYPE}, got {dtype}"
            )
        tilelang.disable_cache()
        vec_core_num = detect_vec_core_num()
        tilelang_kernel = build_fused_gdn_gating_kernel(
            num_heads=num_heads,
            vec_core_num=vec_core_num,
            compile_max_batch=DEFAULT_MAX_BATCH,
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3, config=DEFAULT_ASCEND_PASS_CONFIGS
        ):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source


def _torch_fused_gdn_gating(
    A_log: "torch.Tensor",
    a: "torch.Tensor",
    b: "torch.Tensor",
    dt_bias: "torch.Tensor",
    softplus_beta: float,
    softplus_threshold: float,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch
    import torch.nn.functional as F

    softplus_out = F.softplus(
        a.to(torch.float32) + dt_bias,
        beta=softplus_beta,
        threshold=softplus_threshold,
    )
    g_ref = -A_log.exp() * softplus_out
    beta_ref = torch.sigmoid(b.to(torch.float32)).to(torch.bfloat16)
    return g_ref, beta_ref


def _run_ref_check(
    *,
    num_batches: int,
    num_heads: int,
    vec_core_num: int,
    compile_max_batch: int,
    softplus_beta: float,
    softplus_threshold: float,
) -> None:
    import torch

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        print("[WARN] Skip fused_gdn_gating reference check: NPU is not available")
        return

    if num_batches <= 0:
        raise ValueError(f"num_batches({num_batches}) must be > 0")
    if num_batches > compile_max_batch:
        raise ValueError(
            f"num_batches({num_batches}) must be <= compile_max_batch({compile_max_batch})"
        )

    torch.manual_seed(42)
    device = torch.device("npu")

    A_log = torch.randn((num_heads,), device=device, dtype=torch.float32)
    a = torch.randn((num_batches, num_heads), device=device, dtype=torch.bfloat16)
    b = torch.randn((num_batches, num_heads), device=device, dtype=torch.bfloat16)
    dt_bias = torch.randn((num_heads,), device=device, dtype=torch.float32)
    g_out = torch.empty((num_batches, num_heads), device=device, dtype=torch.float32)
    beta_out = torch.empty(
        (num_batches, num_heads), device=device, dtype=torch.bfloat16
    )

    kernel = fused_gdn_gating_kernel_jit(
        num_heads=num_heads,
        vec_core_num=vec_core_num,
        compile_max_batch=num_batches,
    )
    kernel(
        A_log,
        a,
        b,
        dt_bias,
        g_out,
        beta_out,
        num_batches,
        softplus_beta,
        softplus_threshold,
    )
    torch.npu.synchronize()

    g_ref, beta_ref = _torch_fused_gdn_gating(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
    )
    torch.testing.assert_close(g_out, g_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        beta_out.to(torch.float32),
        beta_ref.to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    print(f"[INFO] fused_gdn_gating output matches torch reference for num_heads={num_heads}")


def _run_ref_suite(
    *,
    num_batches: int,
    vec_core_num: int,
    compile_max_batch: int,
    softplus_beta: float,
    softplus_threshold: float,
    ref_num_heads_list: list[int],
) -> None:
    for num_heads in ref_num_heads_list:
        _run_ref_check(
            num_batches=num_batches,
            num_heads=num_heads,
            vec_core_num=vec_core_num,
            compile_max_batch=compile_max_batch,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TileLang AscendC source for fused_gdn_gating AOT kernel."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE)
    parser.add_argument(
        "--skip-ref-check",
        action="store_true",
        help="Skip runtime torch-reference check.",
    )
    parser.add_argument(
        "--ref-num-batches",
        type=int,
        default=REF_CHECK_NUM_BATCHES,
        help="Batch size used by the optional torch-reference check.",
    )
    parser.add_argument(
        "--softplus-beta",
        type=float,
        default=1.0,
        help="Softplus beta used by the optional torch-reference check.",
    )
    parser.add_argument(
        "--softplus-threshold",
        type=float,
        default=20.0,
        help="Softplus threshold used by the optional torch-reference check.",
    )
    parser.add_argument(
        "--ref-num-heads-list",
        type=int,
        nargs="+",
        default=list(SUPPORTED_NUM_HEADS),
        help="Head counts covered by the optional bf16 torch-reference test suite.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = FusedGdnGatingKernel.generate_source(
        num_heads=args.num_heads, dtype=args.dtype
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="utf-8")

    if not args.skip_ref_check:
        _run_ref_suite(
            num_batches=args.ref_num_batches,
            vec_core_num=detect_vec_core_num(),
            compile_max_batch=DEFAULT_MAX_BATCH,
            softplus_beta=args.softplus_beta,
            softplus_threshold=args.softplus_threshold,
            ref_num_heads_list=args.ref_num_heads_list,
        )


if __name__ == "__main__":
    main()
