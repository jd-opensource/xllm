// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

// Address Space for AMDGCN
// https://llvm.org/docs/AMDGPUUsage.html#address-space

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"

#include <stdint.h>

namespace ck_tile {

// TODO: we have "memory" clobber here because this inline asm is used for async copy
CK_TILE_DEVICE void m0_set_with_memory(index_t v)
{
    asm volatile("s_mov_b32 m0, %0" : : "s"(v) : "memory");
}

// NOTE: this is an immediate value
CK_TILE_DEVICE void m0_inc_with_memory(index_t v)
{
    asm volatile("s_add_u32 m0, %0, m0" : : "n"(v) : "memory");
}

template <typename T>
CK_TILE_DEVICE T warp_shuffle_up(const T& v_local, uint32_t lane_delta)
{
#if 0
    return  __shfl_up(v_local, lane_delta);
#elif 1
    static_assert(sizeof(T) == sizeof(int32_t), "wrong!");

    const uint32_t wrap_around_lane_delta = warpSize - lane_delta;

    const int32_t v_remote_tmp = __builtin_amdgcn_ds_bpermute(
        (__lane_id() << 2) + (wrap_around_lane_delta << 2), bit_cast<int32_t>(v_local));

    return bit_cast<T>(v_remote_tmp);
#endif
}

template <typename T>
CK_TILE_DEVICE T warp_shuffle_down(const T& v_local, uint32_t lane_delta)
{
#if 0
    return  __shfl_down(v_local, lane_delta);
#elif 1
    static_assert(sizeof(T) == sizeof(int32_t), "wrong!");

    const int32_t v_remote_tmp = __builtin_amdgcn_ds_bpermute(
        (__lane_id() << 2) + (lane_delta << 2), bit_cast<int32_t>(v_local));

    return bit_cast<T>(v_remote_tmp);
#endif
}

template <typename T>
CK_TILE_DEVICE T warp_shuffle(const T& v_local, uint32_t src_lane)
{
#if 0
    return  __shfl(v_local, src_lane);
#elif 1
    if constexpr(sizeof(int32_t) > sizeof(T))
    {
        union packet
        {
            int32_t x;
            T v;
        };
        packet p;
        p.v = v_local;
        packet p_remote;
        p_remote.x = __builtin_amdgcn_ds_bpermute(src_lane << 2, bit_cast<int32_t>(p));

        return p_remote.v;
    }
    else if constexpr(sizeof(int32_t) == sizeof(T))
    {
        const int32_t v_remote_tmp =
            __builtin_amdgcn_ds_bpermute(src_lane << 2, bit_cast<int32_t>(v_local));

        return bit_cast<T>(v_remote_tmp);
    }
    else
    {
        static_assert(sizeof(T) % sizeof(int32_t) == 0, "wrong!");
        constexpr index_t elm = sizeof(T) / sizeof(int32_t);
        using vector_type     = thread_buffer<int32_t, elm>;
        auto vs               = bit_cast<vector_type>(v_local);
        auto vs_remote        = vector_type{};
        static_for<0, elm, 1>{}([&](auto i_e) {
            int32_t tmp = __builtin_amdgcn_ds_bpermute(src_lane << 2, bit_cast<int32_t>(vs[i_e]));
            vs_remote(i_e) = tmp;
        });
        return bit_cast<T>(vs_remote);
    }
#endif
}

} // namespace ck_tile
