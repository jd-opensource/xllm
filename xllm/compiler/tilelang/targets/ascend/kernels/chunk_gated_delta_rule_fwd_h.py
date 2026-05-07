#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Any

import tilelang
import tilelang.language as T

import torch

from .utils import (
    DEFAULT_ASCEND_PASS_CONFIGS,
    detect_vec_core_num,
)
from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEFAULT_H = 8
DEFAULT_HG = 4
DEFAULT_K = 128
DEFAULT_V = 128
DEFAULT_BT = 64
DEFAULT_USE_G = 1
DEFAULT_DTYPE = "float16"
DEFAULT_ACCUM_DTYPE = "float32"

SECONDARY_H = 4
SECONDARY_HG = 2
SECONDARY_K = 64
SECONDARY_V = 64
SECONDARY_BT = 32

VEC_NUM = 2

MIN_COMPILE_N = 16
MIN_COMPILE_NT_MAX = 64
MIN_COMPILE_T_TOTAL = 4096

PASS_CONFIGS_NO_AUTO_SYNC = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}


def _prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunk_offsets = []
    offset = 0
    cu_seqlens_np = cu_seqlens.cpu().numpy()
    for i in range(len(cu_seqlens_np) - 1):
        T_len = int(cu_seqlens_np[i + 1] - cu_seqlens_np[i])
        NT = (T_len + chunk_size - 1) // chunk_size
        chunk_offsets.append(offset)
        offset += NT
    return torch.tensor(chunk_offsets, dtype=torch.int32, device=cu_seqlens.device)


def build_chunk_gated_delta_rule_fwd_h_kernel(
    H: int,
    Hg: int,
    K: int,
    V: int,
    BT: int,
    USE_G: int,
    dtype: str,
    accum_dtype: str,
    compile_N: int,
    compile_T_total: int,
    compile_NT_max: int,
):
    V_half = V // 2
    USE_G_BOOL = bool(USE_G)

    SEM_WH_C2V = 0
    SEM_VNEW_V2C = 2
    SEM_HUPD_C2V = 4
    SEM_H_V2C = 6

    @T.prim_func
    def chunk_gated_delta_rule_fwd_h_kernel(
        h: T.Tensor([compile_N, compile_NT_max, H, K, V], dtype),
        k: T.Tensor([compile_T_total, Hg, K], dtype),
        v: T.Tensor([compile_T_total, H, V], dtype),
        w: T.Tensor([compile_T_total, H, K], dtype),
        g: T.Tensor([H, compile_T_total], accum_dtype),
        v_new: T.Tensor([compile_T_total, H, V], dtype),
        h0: T.Tensor([compile_N, H, K, V], dtype),
        ht: T.Tensor([compile_N, H, K, V], dtype),
        cu_seqlens: T.Tensor([compile_N + 1], "int32"),
        ws_wh: T.Tensor([compile_N, H, 2, BT, V_half], accum_dtype),
        ws_vnew: T.Tensor([compile_N, H, 2, BT, V_half], dtype),
        ws_hupd: T.Tensor([compile_N, H, 2, K, V_half], accum_dtype),
        ws_h: T.Tensor([compile_N, H, 2, K, V_half], dtype),
        N: T.int32,
        T_total: T.int32,
        NT_max: T.int32,
        STORE_FINAL_STATE: T.int32,
        SAVE_NEW_VALUE: T.int32,
    ):
        with T.Kernel(compile_N * H, is_npu=True) as (cid, vid):
            i_n = cid // H
            i_h = cid % H

            hg_ratio = H // Hg
            k_head = i_h // hg_ratio

            h_state_ub = T.alloc_ub([2, K // 2, V_half], dtype)
            h_state_ub_float = T.alloc_ub([2, K // 2, V_half], accum_dtype)
            hupd_ub_float = T.alloc_ub([2, K // 2, V_half], accum_dtype)
            wh_ub_float = T.alloc_ub([2, BT // 2, V_half], accum_dtype)

            v_chunk_ub = T.alloc_ub([2, 2, BT // 2, V_half], dtype)
            v_chunk_ub_float = T.alloc_ub([2, BT // 2, V_half], accum_dtype)

            g_chunk_ub = T.alloc_ub([2, BT // 2], accum_dtype)
            g_last_scalar = T.alloc_ub([1], accum_dtype)
            g_exp_ub = T.alloc_ub([BT // 2], accum_dtype)
            g_exp_ub_broc = T.alloc_ub([BT // 2, V_half], accum_dtype)

            k_chunk_l1 = T.alloc_L1([2, BT, K], dtype)
            w_chunk_l1 = T.alloc_L1([2, BT, K], dtype)
            h_state_l1 = T.alloc_L1([2, K, V_half], dtype)
            wh_frag = T.alloc_L0C([2, BT, V_half], accum_dtype)
            v_new_l1 = T.alloc_L1([2, BT, V_half], dtype)
            hupd_frag = T.alloc_L0C([2, K, V_half], accum_dtype)

            with T.Scope("C"):
                bos = cu_seqlens[i_n]
                eos = cu_seqlens[i_n + 1]
                T_len = eos - bos
                NT_i = T.ceildiv(T_len, BT)

                actual_len = T.if_then_else(T_len < BT, T_len, BT)
                T.copy(w[bos : bos + actual_len, i_h, :], w_chunk_l1[0, :, :])
                T.copy(k[bos : bos + actual_len, k_head, :], k_chunk_l1[0, :, :])
                T.set_flag("mte2", "m", 0)

                for i in T.serial(NT_i):
                    pid = i % 2
                    next_pid = (i + 1) % 2
                    chunk_start_next = bos + (i + 1) * BT

                    chunk_len = T.if_then_else(i * BT + BT > T_len, T_len - i * BT, BT)

                    if i + 1 < NT_i:
                        next_len = T.if_then_else((i + 1) * BT + BT > T_len, T_len - (i + 1) * BT, BT)
                        T.copy(w[chunk_start_next : chunk_start_next + next_len, i_h, :], w_chunk_l1[next_pid, :, :])
                        T.copy(k[chunk_start_next : chunk_start_next + next_len, k_head, :], k_chunk_l1[next_pid, :, :])
                        T.set_flag("mte2", "m", next_pid)

                    T.wait_flag("mte2", "m", pid)
                    for j in T.serial(2):
                        T.wait_cross_flag(SEM_H_V2C + j)
                        T.copy(ws_h[i_n, i_h, j, :, :], h_state_l1[j, :, :])
                        T.set_flag("mte2", "m", 2)
                        T.wait_flag("mte2", "m", 2)
                        T.gemm_v0(w_chunk_l1[pid, :, :], h_state_l1[j, :, :], wh_frag[j, :, :], init=True)
                        T.set_flag("m", "fix", 3)
                        T.wait_flag("m", "fix", 3)
                        T.copy(wh_frag[j, :, :], ws_wh[i_n, i_h, j, :, :])
                        T.set_cross_flag("FIX", SEM_WH_C2V + j)

                    for j in T.serial(2):
                        T.wait_cross_flag(SEM_VNEW_V2C + j)
                        T.copy(ws_vnew[i_n, i_h, j, :chunk_len, :], v_new_l1[j, :, :])
                        T.set_flag("mte2", "m", 4)
                        T.wait_flag("mte2", "m", 4)
                        T.gemm_v0(k_chunk_l1[pid, :, :], v_new_l1[j, :, :], hupd_frag[j, :, :], transpose_A=True, init=True)
                        T.set_flag("m", "fix", 5)
                        T.wait_flag("m", "fix", 5)
                        T.copy(hupd_frag[j, :, :], ws_hupd[i_n, i_h, j, :, :])
                        T.set_cross_flag("FIX", SEM_HUPD_C2V + j)

            with T.Scope("V"):
                bos = cu_seqlens[i_n]
                eos = cu_seqlens[i_n + 1]
                T_len = eos - bos
                NT_i = T.ceildiv(T_len, BT)

                for j in T.serial(2):
                    T.copy(h0[i_n, i_h, K // 2 * vid : K // 2 * vid + K // 2, j * V_half : (j + 1) * V_half], h_state_ub[j, :, :])

                chunk_len = T.if_then_else(T_len < BT, T_len, BT)
                vec_chunk_len = T.if_then_else(vid == 0, T.min(BT // 2, chunk_len), T.max(chunk_len - BT // 2, 0))
                vec_start_in_chunk = T.if_then_else(vid == 0, 0, BT // 2)
                vec_global_start = bos + vec_start_in_chunk

                for j in T.serial(2):
                    T.copy(
                        v[vec_global_start : vec_global_start + vec_chunk_len, i_h, j * V_half : (j + 1) * V_half],
                        v_chunk_ub[0, j, :, :],
                    )
                if USE_G_BOOL:
                    T.copy(g[i_h, vec_global_start : vec_global_start + vec_chunk_len], g_chunk_ub[0, :])
                T.set_flag("mte2", "v", 2)

                for i in T.serial(NT_i):
                    pid = i % 2
                    next_pid = (i + 1) % 2
                    v_flag_pid = pid + 2
                    v_flag_next = next_pid + 2
                    g_start = bos + i * BT
                    g_start_next = bos + (i + 1) * BT

                    chunk_len = T.if_then_else(i * BT + BT > T_len, T_len - i * BT, BT)
                    vec_chunk_len = T.if_then_else(vid == 0, T.min(BT // 2, chunk_len), T.max(chunk_len - BT // 2, 0))
                    vec_start_in_chunk = T.if_then_else(vid == 0, 0, BT // 2)

                    if i + 1 < NT_i:
                        next_chunk_len = T.if_then_else((i + 1) * BT + BT > T_len, T_len - (i + 1) * BT, BT)
                        next_vec_start_in_chunk = T.if_then_else(vid == 0, 0, BT // 2)
                        next_vec_chunk_len = T.if_then_else(vid == 0, T.min(BT // 2, next_chunk_len), T.max(next_chunk_len - BT // 2, 0))
                        next_vec_global_start = g_start_next + next_vec_start_in_chunk

                        for j in T.serial(2):
                            T.copy(
                                v[next_vec_global_start : next_vec_global_start + next_vec_chunk_len, i_h, j * V_half : (j + 1) * V_half],
                                v_chunk_ub[next_pid, j, :, :],
                            )
                        if USE_G_BOOL:
                            T.copy(g[i_h, next_vec_global_start : next_vec_global_start + next_vec_chunk_len], g_chunk_ub[next_pid, :])
                        T.set_flag("mte2", "v", v_flag_next)

                    for j in T.serial(2):
                        T.copy(h_state_ub[j, :, :], ws_h[i_n, i_h, j, K // 2 * vid : K // 2 * vid + K // 2, :])
                        T.set_cross_flag("MTE3", SEM_H_V2C + j)

                    for j in T.serial(2):
                        T.copy(h_state_ub[j, :, :], h[i_n, i, i_h, K // 2 * vid : K // 2 * vid + K // 2, j * V_half : (j + 1) * V_half])

                    T.wait_flag("mte2", "v", v_flag_pid)
                    if USE_G_BOOL:
                        g_last = T.if_then_else(i * BT + BT <= T_len, g[i_h, g_start + BT - 1], g[i_h, g_start + T_len - i * BT - 1])

                        T.tile.fill(g_exp_ub, g_last)
                        T.set_flag("mte2", "v", 4)
                        T.wait_flag("mte2", "v", 4)
                        T.tile.sub(g_exp_ub, g_exp_ub, g_chunk_ub[pid, :])
                        T.tile.exp(g_exp_ub, g_exp_ub)
                        T.tile.broadcast(g_exp_ub_broc, g_exp_ub, axis=1)

                        T.tile.fill(g_last_scalar, g_last)
                        T.tile.exp(g_last_scalar, g_last_scalar)

                    for j in T.serial(2):
                        T.copy(v_chunk_ub[pid, j, :, :], v_chunk_ub_float[j, :, :])

                        T.wait_cross_flag(SEM_WH_C2V + j)
                        T.copy(ws_wh[i_n, i_h, j, vec_start_in_chunk : vec_start_in_chunk + BT // 2, :], wh_ub_float[j, :, :])
                        T.set_flag("mte2", "v", 5)
                        T.wait_flag("mte2", "v", 5)
                        T.tile.sub(v_chunk_ub_float[j, :, :], v_chunk_ub_float[j, :, :], wh_ub_float[j, :, :])

                        if SAVE_NEW_VALUE:
                            T.copy(v_chunk_ub_float[j, :, :], v_chunk_ub[pid, j, :, :])
                            T.set_flag("v", "mte3", 6)
                            T.wait_flag("v", "mte3", 6)
                            T.copy(
                                v_chunk_ub[pid, j, :vec_chunk_len, :],
                                v_new[
                                    g_start + vec_start_in_chunk : g_start + vec_start_in_chunk + vec_chunk_len,
                                    i_h,
                                    j * V_half : j * V_half + V_half,
                                ],
                            )

                        if USE_G_BOOL:
                            T.tile.mul(v_chunk_ub_float[j, :, :], v_chunk_ub_float[j, :, :], g_exp_ub_broc)
                            T.copy(h_state_ub[j, :, :], h_state_ub_float[j, :, :])
                            T.tile.mul(h_state_ub_float[j, :, :], h_state_ub_float[j, :, :], g_last_scalar[0])
                        else:
                            T.copy(h_state_ub[j, :, :], h_state_ub_float[j, :, :])

                        T.set_flag("mte3", "v", 7)
                        T.wait_flag("mte3", "v", 7)
                        T.copy(v_chunk_ub_float[j, :, :], v_chunk_ub[pid, j, :, :])
                        T.set_flag("v", "mte3", 8)
                        T.wait_flag("v", "mte3", 8)
                        T.copy(v_chunk_ub[pid, j, :, :], ws_vnew[i_n, i_h, j, vec_start_in_chunk : vec_start_in_chunk + BT // 2, :])
                        T.set_cross_flag("MTE3", SEM_VNEW_V2C + j)

                    for j in T.serial(2):
                        T.wait_cross_flag(SEM_HUPD_C2V + j)
                        T.copy(ws_hupd[i_n, i_h, j, K // 2 * vid : K // 2 * vid + K // 2, :], hupd_ub_float[j, :, :])
                        T.set_flag("mte2", "v", 9)
                        T.wait_flag("mte2", "v", 9)
                        T.tile.add(h_state_ub_float[j, :, :], h_state_ub_float[j, :, :], hupd_ub_float[j, :, :])
                        T.copy(h_state_ub_float[j, :, :], h_state_ub[j, :, :])

                    T.set_flag("v", "mte3", 10)
                    T.wait_flag("v", "mte3", 10)

                if STORE_FINAL_STATE:
                    for j in T.serial(2):
                        T.copy(h_state_ub[j, :, :], ht[i_n, i_h, K // 2 * vid : K // 2 * vid + K // 2, j * V_half : (j + 1) * V_half])

    return chunk_gated_delta_rule_fwd_h_kernel


@register_kernel
class ChunkGatedDeltaRuleFwdHKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("H", "int32"),
        DispatchField("Hg", "int32"),
        DispatchField("K", "int32"),
        DispatchField("V", "int32"),
        DispatchField("BT", "int32"),
        DispatchField("USE_G", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": "h8_hg4_k128_v128_bt64_useg1_fp16",
            "H": DEFAULT_H,
            "Hg": DEFAULT_HG,
            "K": DEFAULT_K,
            "V": DEFAULT_V,
            "BT": DEFAULT_BT,
            "USE_G": DEFAULT_USE_G,
            "dtype": DEFAULT_DTYPE,
        },
        {
            "variant_key": "h4_hg2_k64_v64_bt32_useg1_fp16",
            "H": SECONDARY_H,
            "Hg": SECONDARY_HG,
            "K": SECONDARY_K,
            "V": SECONDARY_V,
            "BT": SECONDARY_BT,
            "USE_G": DEFAULT_USE_G,
            "dtype": DEFAULT_DTYPE,
        },
        {
            "variant_key": "h8_hg4_k128_v128_bt64_useg0_fp16",
            "H": DEFAULT_H,
            "Hg": DEFAULT_HG,
            "K": DEFAULT_K,
            "V": DEFAULT_V,
            "BT": DEFAULT_BT,
            "USE_G": 0,
            "dtype": DEFAULT_DTYPE,
        },
    ]

    @staticmethod
    def generate_source(
        H: int,
        Hg: int,
        K: int,
        V: int,
        BT: int,
        USE_G: int,
        dtype: str,
    ) -> str:
        tilelang.disable_cache()
        compile_N = MIN_COMPILE_N
        compile_T_total = MIN_COMPILE_T_TOTAL
        compile_NT_max = MIN_COMPILE_NT_MAX
        tilelang_kernel = build_chunk_gated_delta_rule_fwd_h_kernel(
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            BT=BT,
            USE_G=USE_G,
            dtype=dtype,
            accum_dtype=DEFAULT_ACCUM_DTYPE,
            compile_N=compile_N,
            compile_T_total=compile_T_total,
            compile_NT_max=compile_NT_max,
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3, config=PASS_CONFIGS_NO_AUTO_SYNC
        ):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source


def ref_chunk_gated_delta_rule(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    BT = chunk_size

    k = k.float().squeeze(0)
    w = w.float().squeeze(0)
    u = u.float().squeeze(0)
    g = g.float().squeeze(0) if g is not None else None
    initial_state = initial_state.float().squeeze(0) if initial_state is not None else None

    T_total, Hg, K = k.shape
    _, H, V = u.shape
    N = len(cu_seqlens) - 1

    NT_total = sum([(int(cu_seqlens[i + 1]) - int(cu_seqlens[i]) + BT - 1) // BT for i in range(N)])

    h = torch.zeros(NT_total, H, K, V, dtype=torch.float32, device=k.device)
    v_new = torch.zeros(T_total, H, V, dtype=torch.float32, device=k.device)
    final_state = torch.zeros(N, H, K, V, dtype=torch.float32, device=k.device) if output_final_state else None

    chunk_offset = 0
    for i_n in range(N):
        bos, eos = int(cu_seqlens[i_n]), int(cu_seqlens[i_n + 1])
        T_len = eos - bos
        NT = (T_len + BT - 1) // BT

        for i_h in range(H):
            h_state = (
                initial_state[i_n, i_h].clone() if initial_state is not None else torch.zeros(K, V, dtype=torch.float32, device=k.device)
            )
            k_head = i_h // (H // Hg)

            for i_t in range(NT):
                t_start = i_t * BT
                t_end = min((i_t + 1) * BT, T_len)

                h[chunk_offset + i_t, i_h] = h_state
                k_chunk, w_chunk, v_chunk = (
                    k[bos + t_start : bos + t_end, k_head, :],
                    w[bos + t_start : bos + t_end, i_h, :],
                    u[bos + t_start : bos + t_end, i_h, :],
                )

                v_n = v_chunk - torch.matmul(w_chunk, h_state)
                v_new[bos + t_start : bos + t_end, i_h, :] = v_n

                if g is not None:
                    g_chunk = g[bos + t_start : bos + t_end, i_h]
                    g_last = g_chunk[-1].item()
                    v_n = v_n * torch.exp(g_last - g_chunk)[:, None]
                    h_state = h_state * torch.exp(torch.tensor(g_last, device=k.device))

                h_state = h_state + torch.matmul(k_chunk.transpose(-1, -2), v_n)

            if output_final_state:
                final_state[i_n, i_h] = h_state
        chunk_offset += NT

    return h.half().unsqueeze(0), v_new.half().unsqueeze(0), final_state.half() if final_state is not None else None


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    BT = chunk_size
    USE_G = g is not None

    k_flat = k.squeeze(0)
    w_flat = w.squeeze(0)
    u_flat = u.squeeze(0)
    g_flat = g.squeeze(0) if g is not None else None

    T_total, Hg, K = k_flat.shape
    _, H, V = u_flat.shape
    N = len(cu_seqlens) - 1

    if chunk_offsets is None:
        chunk_offsets = _prepare_chunk_offsets(cu_seqlens, BT)

    cu_seqlens_np = cu_seqlens.cpu().numpy()
    NT_max = 0
    NT_total = 0
    for i in range(N):
        T_len = int(cu_seqlens_np[i + 1] - cu_seqlens_np[i])
        NT = (T_len + BT - 1) // BT
        NT_max = max(NT_max, NT)
        NT_total += NT

    if USE_G:
        g_c_t = g_flat.float().transpose(0, 1).contiguous()
    else:
        g_c_t = torch.empty((H, T_total), dtype=torch.float32, device=k.device)

    v_new_flat = torch.empty((T_total, H, V), dtype=torch.float16, device=k.device)

    h_out = torch.zeros((N, NT_max, H, K, V), dtype=torch.float16, device=k.device)
    h0 = torch.zeros((N, H, K, V), dtype=torch.float16, device=k.device)
    if initial_state is not None:
        h0.copy_(initial_state.squeeze(0))

    ht = torch.zeros((N, H, K, V), dtype=torch.float16, device=k.device)

    V_half = V // 2
    ws_wh = torch.zeros((N, H, 2, BT, V_half), dtype=torch.float32, device=k.device)
    ws_vnew = torch.zeros((N, H, 2, BT, V_half), dtype=torch.float16, device=k.device)
    ws_hupd = torch.zeros((N, H, 2, K, V_half), dtype=torch.float32, device=k.device)
    ws_h = torch.zeros((N, H, 2, K, V_half), dtype=torch.float16, device=k.device)

    @tilelang.jit(
    # Kernel parameters 0-8 are input/output tensors, 9-12 are workspace tensors
    # that TileLang manages internally for double buffering and cross-core sync
    workspace_idx=[9, 10, 11, 12],
    pass_configs=PASS_CONFIGS_NO_AUTO_SYNC
)
    def jit_kernel(
        N,
        H,
        T_total,
        Hg,
        K,
        V,
        NT_max,
        BT,
        USE_G,
        dtype,
        accum_dtype,
    ):
        return build_chunk_gated_delta_rule_fwd_h_kernel(
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            BT=BT,
            USE_G=USE_G,
            dtype=dtype,
            accum_dtype=accum_dtype,
            compile_N=max(N, MIN_COMPILE_N),
            compile_T_total=max(T_total, MIN_COMPILE_T_TOTAL),
            compile_NT_max=max(NT_max, MIN_COMPILE_NT_MAX),
        )

    ker = jit_kernel(
        N,
        H,
        T_total,
        Hg,
        K,
        V,
        NT_max,
        BT=BT,
        USE_G=1 if USE_G else 0,
        dtype="float16",
        accum_dtype="float32",
    )
    ker(
        h_out,
        k_flat,
        u_flat,
        w_flat,
        g_c_t,
        v_new_flat,
        h0,
        ht,
        cu_seqlens.to(torch.int32),
        ws_wh,
        ws_vnew,
        ws_hupd,
        ws_h,
        N,
        T_total,
        NT_max,
        1 if output_final_state else 0,
        1 if save_new_value else 0,
    )

    v_new_ret = v_new_flat.unsqueeze(0)

    h_ret = torch.zeros((NT_total, H, K, V), dtype=torch.float16, device=k.device)
    for i in range(N):
        NT_i = (int(cu_seqlens_np[i + 1]) - int(cu_seqlens_np[i]) + BT - 1) // BT
        offset = int(chunk_offsets[i].item())
        h_ret[offset : offset + NT_i] = h_out[i, :NT_i]
    h_ret = h_ret.unsqueeze(0)

    ht_ret = ht if output_final_state else None

    return h_ret, v_new_ret, ht_ret


def test_chunk_gated_delta_rule(seqlens, H, Hg, K, V, use_g=True, use_initial_state=True):
    print(f"Testing Varlen seqlens={seqlens}, H={H}, Hg={Hg}, K={K}, V={V}, use_g={use_g}, use_initial_state={use_initial_state}")
    torch.manual_seed(41)

    T_total = sum(seqlens)
    N = len(seqlens)
    cu_seqlens = torch.tensor([0] + [sum(seqlens[: i + 1]) for i in range(len(seqlens))], dtype=torch.int32).npu()

    torch.manual_seed(41)
    k = torch.rand(1, T_total, Hg, K, dtype=torch.float16).npu() * 0.01
    w = torch.rand(1, T_total, H, K, dtype=torch.float16).npu() * 0.01
    u = torch.rand(1, T_total, H, V, dtype=torch.float16).npu() * 0.01
    g = torch.rand(1, T_total, H, dtype=torch.float32).npu() * -1.0 if use_g else None
    initial_state = torch.rand(1, N, H, K, V, dtype=torch.float16).npu() * 0.01 if use_initial_state else None

    torch.npu.synchronize()

    h, v_new, ht = chunk_gated_delta_rule_fwd_h(k, w, u, g, initial_state=initial_state, output_final_state=True, cu_seqlens=cu_seqlens)
    torch.npu.synchronize()
    ref_h, ref_v_new, ref_ht = ref_chunk_gated_delta_rule(
        k.cpu(),
        w.cpu(),
        u.cpu(),
        g.cpu() if g is not None else None,
        initial_state=initial_state.cpu() if initial_state is not None else None,
        output_final_state=True,
        cu_seqlens=cu_seqlens.cpu(),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(h.cpu(), ref_h.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(v_new.cpu(), ref_v_new.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ht.cpu(), ref_ht.cpu(), rtol=1e-5, atol=1e-5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TileLang AscendC source for chunk_gated_delta_rule_fwd_h AOT kernel.")
    parser.add_argument("--output", required=True, help="Output AscendC .cpp file")
    parser.add_argument("--H", type=int, default=DEFAULT_H)
    parser.add_argument("--Hg", type=int, default=DEFAULT_HG)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--V", type=int, default=DEFAULT_V)
    parser.add_argument("--BT", type=int, default=DEFAULT_BT)
    parser.add_argument("--use-g", type=int, default=DEFAULT_USE_G, choices=[0, 1])
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--skip-ref-check", action="store_true", help="Skip runtime torch-reference check.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        ChunkGatedDeltaRuleFwdHKernel.generate_source(
            H=args.H,
            Hg=args.Hg,
            K=args.K,
            V=args.V,
            BT=args.BT,
            USE_G=args.use_g,
            dtype=args.dtype,
        ),
        encoding="utf-8",
    )

    if not args.skip_ref_check:
        test_chunk_gated_delta_rule(
            seqlens=[2048],
            H=args.H,
            Hg=args.Hg,
            K=args.K,
            V=args.V,
            use_g=bool(args.use_g),
            use_initial_state=True,
        )


if __name__ == "__main__":
    main()