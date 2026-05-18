import argparse
from pathlib import Path

import tilelang
from tilelang import language as T
import torch

from ....common.spec import DispatchField, TilelangKernel, register_kernel

tilelang.cache.clear_cache()

CHUNK_SIZE = 64
INPUT_SCALE = 0.01
GATE_SCALE = 0.002

COMPILE_BT = 64
DEFAULT_DTYPE = "bf16"
DEFAULT_ACCUM_DTYPE = "float32"

N_SPECIALIZATION_MIN = 1
N_SPECIALIZATION_MAX = 32
N_SPECIALIZATION_STEP = 1
N_SPECIALIZATIONS = tuple(
    range(N_SPECIALIZATION_MIN, N_SPECIALIZATION_MAX + 1, N_SPECIALIZATION_STEP)
)

_AOT_PASS_CONFIGS = {
    "tl.ascend_auto_sync": False,
    "tl.ascend_auto_cv_combine": False,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_memory_planning": False,
}

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
}


# ==========================================
# 1. Helper Functions
# ==========================================
def _prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def _prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    lens = _prepare_lens(cu_seqlens)
    nt_per_seq = (lens + chunk_size - 1) // chunk_size
    return torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=cu_seqlens.device),
            nt_per_seq,
        ]
    ).cumsum(-1)


# ==========================================
# 2. AOT Build Function
# ==========================================
def _build_chunk_gated_delta_rule_fwd_h_kernel(
    N: int,
    H: int,
    Hg: int,
    K: int,
    V: int,
    dtype: str = DEFAULT_DTYPE,
    accum_dtype: str = DEFAULT_ACCUM_DTYPE,
    bt: int = COMPILE_BT,
    use_g: bool = True,
    store_final_state: bool = True,
    save_new_value: bool = True,
):
    V_half = V // 2
    input_dtype = "bfloat16"

    NT_sym = T.symbolic("nt")
    T_total_sym = T.symbolic("total_t")

    SEM_WH_C2V = 0
    SEM_VNEW_V2C = 2
    SEM_HUPD_C2V = 4
    SEM_H_V2C = 6

    @T.prim_func
    def main(
        h: T.Tensor([N, NT_sym, H, K, V], input_dtype),
        k: T.Tensor([T_total_sym, Hg, K], input_dtype),
        v: T.Tensor([T_total_sym, H, V], input_dtype),
        w: T.Tensor([T_total_sym, H, K], input_dtype),
        g: T.Tensor([T_total_sym, H], accum_dtype),
        v_new: T.Tensor([T_total_sym, H, V], input_dtype),
        h0: T.Tensor([N, H, K, V], input_dtype),
        ht: T.Tensor([N, H, K, V], accum_dtype),
        cu_seqlens: T.Tensor([N + 1], "int32"),
        ws_wh: T.Tensor([N, H, 2, bt, V_half], accum_dtype),
        ws_vnew: T.Tensor([N, H, 2, bt, V_half], input_dtype),
        ws_hupd: T.Tensor([N, H, 2, K, V_half], accum_dtype),
        ws_h: T.Tensor([N, H, 2, K, V_half], input_dtype),
    ):
        with T.Kernel(N * H, is_npu=True) as (cid, vid):
            i_n = cid // H
            i_h = cid % H

            hg_ratio = H // Hg
            k_head = i_h // hg_ratio

            h_state_ub = T.alloc_ub([2, K // 2, V_half], input_dtype)
            h_state_ub_float = T.alloc_ub([2, K // 2, V_half], accum_dtype)
            hupd_ub_float = T.alloc_ub([2, K // 2, V_half], accum_dtype)
            wh_ub_float = T.alloc_ub([2, bt // 2, V_half], accum_dtype)

            v_chunk_ub = T.alloc_ub([2, 2, bt // 2, V_half], input_dtype)
            v_chunk_ub_float = T.alloc_ub([2, bt // 2, V_half], accum_dtype)

            g_chunk_ub = T.alloc_ub([2, bt // 2], accum_dtype)
            g_last_scalar = T.alloc_ub([1], accum_dtype)
            g_exp_ub = T.alloc_ub([bt // 2], accum_dtype)
            g_exp_ub_broc = T.alloc_ub([bt // 2, V_half], accum_dtype)

            k_chunk_l1 = T.alloc_L1([2, bt, K], input_dtype)
            w_chunk_l1 = T.alloc_L1([2, bt, K], input_dtype)
            h_state_l1 = T.alloc_L1([2, K, V_half], input_dtype)
            wh_frag = T.alloc_L0C([2, bt, V_half], accum_dtype)
            v_new_l1 = T.alloc_L1([2, bt, V_half], input_dtype)
            hupd_frag = T.alloc_L0C([2, K, V_half], accum_dtype)

            with T.Scope("C"):
                bos = cu_seqlens[i_n]
                eos = cu_seqlens[i_n + 1]
                T_len = eos - bos
                NT_i = T.ceildiv(T_len, bt)

                actual_len = T.if_then_else(T_len < bt, T_len, bt)
                T.copy(w[bos : bos + actual_len, i_h, :], w_chunk_l1[0, :, :])
                T.copy(k[bos : bos + actual_len, k_head, :], k_chunk_l1[0, :, :])
                T.set_flag("mte2", "m", 0)

                for i in T.serial(NT_i):
                    pid = i % 2
                    next_pid = (i + 1) % 2
                    chunk_start_next = bos + (i + 1) * bt

                    chunk_len = T.if_then_else(i * bt + bt > T_len, T_len - i * bt, bt)

                    if i + 1 < NT_i:
                        next_len = T.if_then_else(
                            (i + 1) * bt + bt > T_len, T_len - (i + 1) * bt, bt
                        )
                        T.copy(
                            w[chunk_start_next : chunk_start_next + next_len, i_h, :],
                            w_chunk_l1[next_pid, :, :],
                        )
                        T.copy(
                            k[
                                chunk_start_next : chunk_start_next + next_len,
                                k_head,
                                :,
                            ],
                            k_chunk_l1[next_pid, :, :],
                        )
                        T.set_flag("mte2", "m", next_pid)

                    # w @ h
                    T.wait_flag("mte2", "m", pid)
                    for j in T.serial(2):
                        T.wait_cross_flag(SEM_H_V2C + j)
                        T.copy(ws_h[i_n, i_h, j, :, :], h_state_l1[j, :, :])
                        T.set_flag("mte2", "m", 2)
                        T.wait_flag("mte2", "m", 2)
                        T.gemm_v0(
                            w_chunk_l1[pid, :, :],
                            h_state_l1[j, :, :],
                            wh_frag[j, :, :],
                            init=True,
                        )
                        T.set_flag("m", "fix", 3)
                        T.wait_flag("m", "fix", 3)
                        T.copy(wh_frag[j, :, :], ws_wh[i_n, i_h, j, :, :])
                        T.set_cross_flag("FIX", SEM_WH_C2V + j)

                    # k @ v_new
                    for j in T.serial(2):
                        T.wait_cross_flag(SEM_VNEW_V2C + j)
                        T.copy(ws_vnew[i_n, i_h, j, :chunk_len, :], v_new_l1[j, :, :])
                        T.set_flag("mte2", "m", 4)
                        T.wait_flag("mte2", "m", 4)
                        T.gemm_v0(
                            k_chunk_l1[pid, :, :],
                            v_new_l1[j, :, :],
                            hupd_frag[j, :, :],
                            transpose_A=True,
                            init=True,
                        )
                        T.set_flag("m", "fix", 5)
                        T.wait_flag("m", "fix", 5)
                        T.copy(hupd_frag[j, :, :], ws_hupd[i_n, i_h, j, :, :])
                        T.set_cross_flag("FIX", SEM_HUPD_C2V + j)

            with T.Scope("V"):
                bos = cu_seqlens[i_n]
                eos = cu_seqlens[i_n + 1]
                T_len = eos - bos
                NT_i = T.ceildiv(T_len, bt)

                for j in T.serial(2):
                    T.copy(
                        h0[
                            i_n,
                            i_h,
                            K // 2 * vid : K // 2 * vid + K // 2,
                            j * V_half : (j + 1) * V_half,
                        ],
                        h_state_ub[j, :, :],
                    )
                    T.set_flag("mte2", "v", 12)
                    T.wait_flag("mte2", "v", 12)
                    T.copy(h_state_ub[j, :, :], h_state_ub_float[j, :, :])

                chunk_len = T.if_then_else(T_len < bt, T_len, bt)
                vec_chunk_len = T.if_then_else(
                    vid == 0, T.min(bt // 2, chunk_len), T.max(chunk_len - bt // 2, 0)
                )
                vec_start_in_chunk = T.if_then_else(vid == 0, 0, bt // 2)
                vec_global_start = bos + vec_start_in_chunk

                for j in T.serial(2):
                    T.copy(
                        v[
                            vec_global_start : vec_global_start + vec_chunk_len,
                            i_h,
                            j * V_half : (j + 1) * V_half,
                        ],
                        v_chunk_ub[0, j, :, :],
                    )
                if use_g:
                    T.copy(
                        g[vec_global_start : vec_global_start + vec_chunk_len, i_h],
                        g_chunk_ub[0, :],
                    )
                T.set_flag("mte2", "v", 2)

                for i in T.serial(NT_i):
                    pid = i % 2
                    next_pid = (i + 1) % 2
                    v_flag_pid = pid + 2
                    v_flag_next = next_pid + 2
                    g_start = bos + i * bt
                    g_start_next = bos + (i + 1) * bt

                    chunk_len = T.if_then_else(i * bt + bt > T_len, T_len - i * bt, bt)
                    vec_chunk_len = T.if_then_else(
                        vid == 0,
                        T.min(bt // 2, chunk_len),
                        T.max(chunk_len - bt // 2, 0),
                    )
                    vec_start_in_chunk = T.if_then_else(vid == 0, 0, bt // 2)

                    # v[t+1], g[t+1]
                    if i + 1 < NT_i:
                        next_chunk_len = T.if_then_else(
                            (i + 1) * bt + bt > T_len, T_len - (i + 1) * bt, bt
                        )
                        next_vec_start_in_chunk = T.if_then_else(vid == 0, 0, bt // 2)
                        next_vec_chunk_len = T.if_then_else(
                            vid == 0,
                            T.min(bt // 2, next_chunk_len),
                            T.max(next_chunk_len - bt // 2, 0),
                        )
                        next_vec_global_start = g_start_next + next_vec_start_in_chunk

                        for j in T.serial(2):
                            T.copy(
                                v[
                                    next_vec_global_start : next_vec_global_start
                                    + next_vec_chunk_len,
                                    i_h,
                                    j * V_half : (j + 1) * V_half,
                                ],
                                v_chunk_ub[next_pid, j, :, :],
                            )
                        if use_g:
                            T.copy(
                                g[
                                    next_vec_global_start : next_vec_global_start
                                    + next_vec_chunk_len,
                                    i_h,
                                ],
                                g_chunk_ub[next_pid, :],
                            )
                        T.set_flag("mte2", "v", v_flag_next)

                    # h to cube
                    for j in T.serial(2):
                        T.copy(h_state_ub_float[j, :, :], h_state_ub[j, :, :])
                        T.set_flag("v", "mte3", 11)
                        T.wait_flag("v", "mte3", 11)
                        T.copy(
                            h_state_ub[j, :, :],
                            ws_h[i_n, i_h, j, K // 2 * vid : K // 2 * vid + K // 2, :],
                        )
                        T.set_cross_flag("MTE3", SEM_H_V2C + j)
                        # save h[t]
                        T.copy(
                            h_state_ub[j, :, :],
                            h[
                                i_n,
                                i,
                                i_h,
                                K // 2 * vid : K // 2 * vid + K // 2,
                                j * V_half : (j + 1) * V_half,
                            ],
                        )

                    T.wait_flag("mte2", "v", v_flag_pid)
                    # prepare gating
                    if use_g:
                        g_last = T.if_then_else(
                            i * bt + bt <= T_len,
                            g[g_start + bt - 1, i_h],
                            g[g_start + T_len - i * bt - 1, i_h],
                        )

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

                        # v_new = v - w @ h
                        T.wait_cross_flag(SEM_WH_C2V + j)
                        T.copy(
                            ws_wh[
                                i_n,
                                i_h,
                                j,
                                vec_start_in_chunk : vec_start_in_chunk + bt // 2,
                                :,
                            ],
                            wh_ub_float[j, :, :],
                        )
                        T.set_flag("mte2", "v", 5)
                        T.wait_flag("mte2", "v", 5)
                        T.tile.sub(
                            v_chunk_ub_float[j, :, :],
                            v_chunk_ub_float[j, :, :],
                            wh_ub_float[j, :, :],
                        )

                        if save_new_value:
                            T.copy(v_chunk_ub_float[j, :, :], v_chunk_ub[pid, j, :, :])
                            T.set_flag("v", "mte3", 6)
                            T.wait_flag("v", "mte3", 6)
                            T.copy(
                                v_chunk_ub[pid, j, :vec_chunk_len, :],
                                v_new[
                                    g_start + vec_start_in_chunk : g_start
                                    + vec_start_in_chunk
                                    + vec_chunk_len,
                                    i_h,
                                    j * V_half : j * V_half + V_half,
                                ],
                            )

                        if use_g:
                            # v_new *= exp(g_last - g)
                            T.tile.mul(
                                v_chunk_ub_float[j, :, :],
                                v_chunk_ub_float[j, :, :],
                                g_exp_ub_broc,
                            )
                            # h *= exp(g_last)
                            T.tile.mul(
                                h_state_ub_float[j, :, :],
                                h_state_ub_float[j, :, :],
                                g_last_scalar[0],
                            )

                        T.set_flag("mte3", "v", 7)
                        T.wait_flag("mte3", "v", 7)
                        T.copy(v_chunk_ub_float[j, :, :], v_chunk_ub[pid, j, :, :])
                        T.set_flag("v", "mte3", 8)
                        T.wait_flag("v", "mte3", 8)
                        T.copy(
                            v_chunk_ub[pid, j, :, :],
                            ws_vnew[
                                i_n,
                                i_h,
                                j,
                                vec_start_in_chunk : vec_start_in_chunk + bt // 2,
                                :,
                            ],
                        )
                        T.set_cross_flag("MTE3", SEM_VNEW_V2C + j)

                    for j in T.serial(2):
                        # h += k @ v_new
                        T.wait_cross_flag(SEM_HUPD_C2V + j)
                        T.copy(
                            ws_hupd[
                                i_n, i_h, j, K // 2 * vid : K // 2 * vid + K // 2, :
                            ],
                            hupd_ub_float[j, :, :],
                        )
                        T.set_flag("mte2", "v", 9)
                        T.wait_flag("mte2", "v", 9)
                        T.tile.add(
                            h_state_ub_float[j, :, :],
                            h_state_ub_float[j, :, :],
                            hupd_ub_float[j, :, :],
                        )

                    T.set_flag("v", "mte3", 10)
                    T.wait_flag("v", "mte3", 10)

                if store_final_state:
                    for j in T.serial(2):
                        T.copy(
                            h_state_ub_float[j, :, :],
                            ht[
                                i_n,
                                i_h,
                                K // 2 * vid : K // 2 * vid + K // 2,
                                j * V_half : (j + 1) * V_half,
                            ],
                        )

    return main


@register_kernel
class ChunkGatedDeltaRuleFwdHKernel(TilelangKernel):
    KERNEL_NAME = "chunk_gated_delta_rule_fwd_h"
    DISPATCH_SCHEMA = [
        DispatchField("N", "int32"),
        DispatchField("H", "int32"),
        DispatchField("Hg", "int32"),
        DispatchField("K", "int32"),
        DispatchField("V", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": f"n{n}_h{hv}_hg{hg}_k{k}_v{v}_bf16",
            "N": n,
            "H": hv,
            "Hg": hg,
            "K": k,
            "V": v,
            "dtype": DEFAULT_DTYPE,
        }
        for n in N_SPECIALIZATIONS
        for hv, hg, k, v in [
            (16, 1, 128, 128),
            (8, 1, 128, 128),
            (32, 16, 128, 128),
        ]
    ]

    @staticmethod
    def generate_source(N: int, H: int, Hg: int, K: int, V: int, dtype: str) -> str:
        if dtype != DEFAULT_DTYPE:
            raise ValueError(
                f"chunk_gated_delta_rule_fwd_h only supports dtype={DEFAULT_DTYPE}, got {dtype}"
            )
        tilelang.disable_cache()
        tilelang_kernel = _build_chunk_gated_delta_rule_fwd_h_kernel(
            N=N,
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            dtype=dtype,
            bt=COMPILE_BT,
        )
        with tilelang.tvm.transform.PassContext(opt_level=3, config=_AOT_PASS_CONFIGS):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source


@tilelang.jit(workspace_idx=[9, 10, 11, 12], pass_configs=pass_configs)
def chunk_gated_delta_rule_fwd_kernel_jit(
    N: int,
    H: int,
    T_total: int,
    Hg: int,
    K: int,
    V: int,
    NT_max: int,
    BT: int = 64,
    USE_G: bool = True,
    STORE_FINAL_STATE: bool = True,
    SAVE_NEW_VALUE: bool = True,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    return _build_chunk_gated_delta_rule_fwd_h_kernel(
        N=N,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        dtype=dtype,
        accum_dtype=accum_dtype,
        bt=BT,
        use_g=USE_G,
        store_final_state=STORE_FINAL_STATE,
        save_new_value=SAVE_NEW_VALUE,
    )


# ==========================================
# 4. Python Wrapper Layer (JIT)
# ==========================================
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
    IS_VARLEN = cu_seqlens is None

    # --- 1. Derive cu_seqlens, flatten inputs, extract shapes ---
    if IS_VARLEN:
        B, T, Hg, K = k.shape
        H = u.shape[-2]
        V = u.shape[-1]
        N = B
        cu_seqlens = torch.tensor(
            [i * T for i in range(B + 1)], dtype=torch.int32, device=k.device
        )
        k_flat = k.reshape(B * T, Hg, K)
        w_flat = w.reshape(B * T, H, K)
        u_flat = u.reshape(B * T, H, V)
        g_flat = g.reshape(B * T, H) if USE_G else None
        T_total = B * T
        NT_max = (T + BT - 1) // BT
    else:
        k_flat = k.squeeze(0)
        w_flat = w.squeeze(0)
        u_flat = u.squeeze(0)
        g_flat = g.squeeze(0) if USE_G else None
        T_total, Hg, K = k_flat.shape
        _, H, V = u_flat.shape
        N = len(cu_seqlens) - 1
        if chunk_offsets is None:
            chunk_offsets = _prepare_chunk_offsets(cu_seqlens, BT)
        NT_per_seq = chunk_offsets[1:] - chunk_offsets[:-1]
        NT_max = int(NT_per_seq.max().item())

    # --- 2. Common allocations ---
    g_c_t = (
        g_flat.float().contiguous()
        if USE_G
        else torch.empty((T_total, H), dtype=torch.float32, device=k.device)
    )
    v_new_flat = torch.empty((T_total, H, V), dtype=k.dtype, device=k.device)
    h_out = torch.empty((N, NT_max, H, K, V), dtype=k.dtype, device=k.device)
    h0 = torch.zeros((N, H, K, V), dtype=k.dtype, device=k.device)
    if initial_state is not None:
        h0.copy_(initial_state if IS_VARLEN else initial_state.squeeze(0))
    ht = torch.zeros((N, H, K, V), dtype=torch.float32, device=k.device)

    # --- 3. Kernel invocation ---
    ker = chunk_gated_delta_rule_fwd_kernel_jit(
        N,
        H,
        T_total,
        Hg,
        K,
        V,
        NT_max,
        BT=BT,
        USE_G=USE_G,
        STORE_FINAL_STATE=output_final_state,
        SAVE_NEW_VALUE=save_new_value,
        dtype=str(k.dtype).split(".")[-1],
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
    )

    # --- 4. Format outputs ---
    ht_ret = ht if output_final_state else None
    if IS_VARLEN:
        return h_out, v_new_flat.reshape(B, T, H, V), ht_ret

    chunk_offsets_cpu = chunk_offsets.cpu().numpy()
    slices = [
        h_out[i, : int(chunk_offsets_cpu[i + 1] - chunk_offsets_cpu[i])]
        for i in range(N)
    ]
    h_ret = torch.cat(slices, dim=0).unsqueeze(0)
    return h_ret, v_new_flat.unsqueeze(0), ht_ret


# ==========================================
# 5. Golden Reference
# ==========================================
def _ref_chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    chunk_size: int = CHUNK_SIZE,
    use_exp2: bool = False,
):
    B, T = k.shape[0], k.shape[1]
    Hg = k.shape[2]
    K = k.shape[3]
    H = w.shape[2]
    V = u.shape[3]
    NT = (T + chunk_size - 1) // chunk_size
    hg_ratio = H // Hg

    k_f = k.float()
    w_f = w.float()
    u_f = u.float()
    g_f = g.float() if g is not None else None

    h = torch.empty((B, NT, H, K, V), dtype=k.dtype)
    v_new = torch.empty_like(u)
    final_state = (
        torch.empty((B, H, K, V), dtype=torch.float32) if output_final_state else None
    )

    for b in range(B):
        for head in range(H):
            k_head = head // hg_ratio
            if initial_state is None:
                state = torch.zeros((K, V), dtype=torch.float32)
            else:
                state = initial_state[b, head].float().clone()

            for chunk_id, start in enumerate(range(0, T, chunk_size)):
                end = min(start + chunk_size, T)
                h[b, chunk_id, head] = state.to(k.dtype)

                residual = u_f[b, start:end, head] - w_f[b, start:end, head] @ state
                if g_f is not None:
                    g_chunk = g_f[b, start:end, head]
                    last_g = g_chunk[-1]
                    if use_exp2:
                        v_chunk = residual
                        state_v = residual * torch.exp2(last_g - g_chunk)[:, None]
                        state = state * torch.exp2(last_g)
                    else:
                        v_chunk = residual
                        state_v = residual * (last_g - g_chunk).exp()[:, None]
                        state = state * last_g.exp()
                else:
                    v_chunk = residual
                    state_v = residual

                v_new[b, start:end, head] = v_chunk.to(u.dtype)
                state = state + k_f[b, start:end, k_head].transpose(-1, -2) @ state_v

            if output_final_state:
                final_state[b, head] = state

    return h, v_new, final_state


# ==========================================
# 6. Test Functions
# ==========================================
def _chunk_local_cumsum_cpu(g: torch.Tensor, chunk_size: int) -> torch.Tensor:
    out = torch.empty_like(g, dtype=torch.float32)
    for start in range(0, g.shape[1], chunk_size):
        end = min(start + chunk_size, g.shape[1])
        out[:, start:end] = g[:, start:end].float().cumsum(dim=1)
    return out


def test_chunk_gated_delta_rule_batch(
    B: int,
    T: int,
    H: int,
    Hg: int,
    K: int,
    V: int,
    use_g: bool = True,
    use_initial_state: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    print(
        f"Testing Batch B={B}, T={T}, H={H}, Hg={Hg}, K={K}, V={V}, use_g={use_g}, use_initial_state={use_initial_state}"
    )
    torch.manual_seed(41)

    k = torch.randn(B, T, Hg, K, dtype=dtype).npu() * INPUT_SCALE
    w = torch.randn(B, T, H, K, dtype=dtype).npu() * INPUT_SCALE
    u = torch.randn(B, T, H, V, dtype=dtype).npu() * INPUT_SCALE
    g = (
        _chunk_local_cumsum_cpu(
            torch.randn((B, T, H), dtype=torch.float32) * GATE_SCALE, CHUNK_SIZE
        ).npu()
        if use_g
        else None
    )
    initial_state = (
        torch.randn(B, H, K, V, dtype=dtype).npu() * INPUT_SCALE
        if use_initial_state
        else None
    )

    h, v_new, ht = chunk_gated_delta_rule_fwd_h(
        k, w, u, g, initial_state=initial_state, output_final_state=True
    )
    torch.npu.synchronize()

    ref_h, ref_v_new, ref_ht = _ref_chunk_gated_delta_rule_fwd_h(
        k.cpu(),
        w.cpu(),
        u.cpu(),
        g.cpu() if g is not None else None,
        initial_state=initial_state.cpu() if initial_state is not None else None,
        output_final_state=True,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(h.cpu(), ref_h.cpu(), rtol=5e-4, atol=1e-2)
    torch.testing.assert_close(v_new.cpu(), ref_v_new.cpu(), rtol=5e-4, atol=1e-2)
    torch.testing.assert_close(ht.cpu(), ref_ht.cpu(), rtol=5e-4, atol=1e-2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test chunk gated delta rule (supports batch and varlen modes)"
    )
    parser.add_argument(
        "--aot", action="store_true", help="Run AOT source generation mode"
    )
    parser.add_argument(
        "--use_g",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to use gating (True/False)",
    )
    parser.add_argument(
        "--use_initial_state",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to use initial state (True/False)",
    )
    parser.add_argument("--B", type=int, default=1, help="Batch size for batch mode")
    parser.add_argument(
        "--T", type=int, default=2048, help="Sequence length for batch mode"
    )
    parser.add_argument("--H", type=int, default=8, help="Number of heads")
    parser.add_argument(
        "--Hg", type=int, default=8, help="Number of grouped heads (must be <= H)"
    )
    parser.add_argument("--K", type=int, default=128, help="Key dimension")
    parser.add_argument("--V", type=int, default=128, help="Value dimension")
    parser.add_argument("--output", help="Output AscendC .cpp file (for --aot mode)")
    args = parser.parse_args()

    if args.aot:
        output = Path(args.output).resolve() if args.output else None
        if output is None:
            print("Error: --output is required for AOT mode")
            exit(1)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            ChunkGatedDeltaRuleFwdHKernel.generate_source(
                N=args.B,
                H=args.H,
                Hg=args.Hg,
                K=args.K,
                V=args.V,
                dtype=DEFAULT_DTYPE,
            ),
            encoding="utf-8",
        )
    else:
        print("=" * 60)
        test_chunk_gated_delta_rule_batch(
            B=args.B,
            T=args.T,
            H=args.H,
            Hg=args.Hg,
            K=args.K,
            V=args.V,
            use_g=args.use_g,
            use_initial_state=args.use_initial_state,
        )
        print("Kernel Output Match!")
