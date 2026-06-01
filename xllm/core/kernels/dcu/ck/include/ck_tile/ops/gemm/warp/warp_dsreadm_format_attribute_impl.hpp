// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/container/thread_buffer.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/utility/functional.hpp"

namespace ck_tile {

template <typename RetType,
          index_t offset,
          index_t element,
          index_t row,
          index_t col,
          index_t alt,
          bool use_m0      = false,
          bool skip_mov_m0 = false>
CK_TILE_DEVICE void hcu_ds_read_matrix_format_asm_impl(uintptr_t smem,
                                                       RetType& ret,
                                                       number<offset>,
                                                       number<element>,
                                                       number<row>,
                                                       number<col>,
                                                       number<alt>,
                                                       bool_constant<use_m0>      = {},
                                                       bool_constant<skip_mov_m0> = {})
{
#if defined(__gfx938__) || defined(__gfx92a__) || defined(__gfx946__)
    const auto soffset = __builtin_amdgcn_readfirstlane(smem);

    if constexpr(CK_TILE_DS_FORMAT_FORCE_M0 || use_m0)
    {
        if constexpr(!skip_mov_m0)
        {
            asm volatile(
                "s_mov_b32 m0, %1\n\t"
                "s_nop 0\n\t"
                "ds_read_matrix_format %0, m0 offset:%2 element:%3 row:%4 col:%5 alt:%6\n\t"
                : "=v"(ret)
                : "s"(soffset), "n"(offset), "n"(element), "n"(row), "n"(col), "n"(alt)
                : "memory");
        }
        else
        {
            detail::swallow{soffset};

            asm volatile(
                "ds_read_matrix_format %0, m0 offset:%1 element:%2 row:%3 col:%4 alt:%5\n\t"
                : "=v"(ret)
                : "n"(offset), "n"(element), "n"(row), "n"(col), "n"(alt)
                : "memory");
        }
    }
    else
    {
        asm volatile("ds_read_matrix_format %0, %1 offset:%2 element:%3 row:%4 col:%5 alt:%6\n\t"
                     : "=v"(ret)
                     : "s"(soffset), "n"(offset), "n"(element), "n"(row), "n"(col), "n"(alt)
                     : "memory");
    }
#else
    detail::swallow{smem, ret};
#endif
}

template <typename RetType,
          index_t offset,
          index_t element,
          index_t row,
          index_t col,
          index_t alt,
          bool use_m0      = false,
          bool skip_mov_m0 = false>
CK_TILE_DEVICE void hcu_ds_read_matrix_trans_format_asm_impl(uintptr_t smem,
                                                             RetType& ret,
                                                             number<offset>,
                                                             number<element>,
                                                             number<row>,
                                                             number<col>,
                                                             number<alt>,
                                                             bool_constant<use_m0>      = {},
                                                             bool_constant<skip_mov_m0> = {})
{
#if defined(__gfx938__) || defined(__gfx92a__) || defined(__gfx946__)
    const auto soffset = __builtin_amdgcn_readfirstlane(smem);

    if constexpr(CK_TILE_DS_FORMAT_FORCE_M0 || use_m0)
    {
        if constexpr(!skip_mov_m0)
        {
            asm volatile(
                "s_mov_b32 m0, %1\n\t"
                "s_nop 0\n\t"
                "ds_read_matrix_trans_format %0, m0 offset:%2 element:%3 row:%4 col:%5 alt:%6\n\t"
                : "=v"(ret)
                : "s"(soffset), "n"(offset), "n"(element), "n"(row), "n"(col), "n"(alt)
                : "memory");
        }
        else
        {
            detail::swallow{soffset};

            asm volatile(
                "ds_read_matrix_trans_format %0, m0 offset:%1 element:%2 row:%3 col:%4 alt:%5\n\t"
                : "=v"(ret)
                : "n"(offset), "n"(element), "n"(row), "n"(col), "n"(alt)
                : "memory");
        }
    }
    else
    {
        asm volatile(
            "ds_read_matrix_trans_format %0, %1 offset:%2 element:%3 row:%4 col:%5 alt:%6\n\t"
            : "=v"(ret)
            : "s"(soffset), "n"(offset), "n"(element), "n"(row), "n"(col), "n"(alt)
            : "memory");
    }
#else
    detail::swallow{smem, ret};
#endif
}

struct WarpDsreadmFormatAttributeImpl_M32x16_B16
{
    // tile read size
    static constexpr index_t kMN = 32;
    static constexpr index_t kK  = 16;

    // lane store layout
    static constexpr index_t kMNStoreLane = 16;
    static constexpr index_t kKStoreLane  = 4;

    // lane store slice
    static constexpr index_t kMN0StorePerLane = 2;
    static constexpr index_t kMN1StorePerLane = 1;
    static constexpr index_t kKStorePerLane   = 4;

    // for static check
    static constexpr index_t kMNInterleave = 1;

    // vector length for returned data
    static constexpr index_t kVectorLength = kMN0StorePerLane * kMN1StorePerLane * kKStorePerLane;

    template <typename T, index_t offset, bool use_m0 = false, bool skip_mov_m0 = false>
    CK_TILE_DEVICE void operator()(CK_TILE_LDS_ADDR T* smem,
                                   ext_vector_t<T, kVectorLength>& ret,
                                   number<offset>,
                                   bool_constant<use_m0>      = {},
                                   bool_constant<skip_mov_m0> = {}) const
    {
        hcu_ds_read_matrix_format_asm_impl(reinterpret_cast<uintptr_t>(smem),
                                           ret,
                                           number<offset>{},
                                           number<0x2>{}, // element:0x2
                                           number<0x2>{}, // row:0x2
                                           number<0x1>{}, // col:0x1
                                           number<0x0>{}, // alt:0x0
                                           bool_constant<use_m0>{},
                                           bool_constant<skip_mov_m0>{});
    }
};

struct WarpDsreadmFormatAttributeImpl_M32x16_B16_ALT2
{
    // tile read size
    static constexpr index_t kMN = 32;
    static constexpr index_t kK  = 16;

    // lane store layout
    static constexpr index_t kMNStoreLane = 16;
    static constexpr index_t kKStoreLane  = 4;

    // lane store slice
    static constexpr index_t kMN0StorePerLane = 1;
    static constexpr index_t kMN1StorePerLane = 2;
    static constexpr index_t kKStorePerLane   = 4;

    // for static check
    static constexpr index_t kMNInterleave = 2;

    // vector length for returned data
    static constexpr index_t kVectorLength = kMN0StorePerLane * kMN1StorePerLane * kKStorePerLane;

    template <typename T, index_t offset, bool use_m0 = false, bool skip_mov_m0 = false>
    CK_TILE_DEVICE void operator()(CK_TILE_LDS_ADDR T* smem,
                                   ext_vector_t<T, kVectorLength>& ret,
                                   number<offset>,
                                   bool_constant<use_m0>      = {},
                                   bool_constant<skip_mov_m0> = {}) const
    {
        hcu_ds_read_matrix_format_asm_impl(reinterpret_cast<uintptr_t>(smem),
                                           ret,
                                           number<offset>{},
                                           number<0x2>{}, // element:0x2
                                           number<0x2>{}, // row:0x2
                                           number<0x1>{}, // col:0x1
                                           number<0x1>{}, // alt:0x1
                                           bool_constant<use_m0>{},
                                           bool_constant<skip_mov_m0>{});
    }
};

struct WarpDsreadmFormatAttributeImpl_MT32x16_B16
{
    // tile read size
    static constexpr index_t kMN = 16;
    static constexpr index_t kK  = 32;

    // lane store layout
    static constexpr index_t kMNStoreLane = 16;
    static constexpr index_t kKStoreLane  = 4;

    // lane store slice
    static constexpr index_t kMN0StorePerLane = 1;
    static constexpr index_t kMN1StorePerLane = 1;
    static constexpr index_t kKStorePerLane   = 8;

    // for static check
    static constexpr index_t kMNInterleave = 1;

    // vector length for returned data
    static constexpr index_t kVectorLength = kMN0StorePerLane * kMN1StorePerLane * kKStorePerLane;

    template <typename T, index_t offset, bool use_m0 = false, bool skip_mov_m0 = false>
    CK_TILE_DEVICE void operator()(CK_TILE_LDS_ADDR T* smem,
                                   ext_vector_t<T, kVectorLength>& ret,
                                   number<offset>,
                                   bool_constant<use_m0>      = {},
                                   bool_constant<skip_mov_m0> = {}) const
    {
        hcu_ds_read_matrix_trans_format_asm_impl(reinterpret_cast<uintptr_t>(smem),
                                                 ret,
                                                 number<offset>{},
                                                 number<0x2>{}, // element:0x2
                                                 number<0x2>{}, // row:0x2
                                                 number<0x1>{}, // col:0x1
                                                 number<0x0>{}, // alt:0x0
                                                 bool_constant<use_m0>{},
                                                 bool_constant<skip_mov_m0>{});
    }
};

struct WarpDsreadmFormatAttributeImpl_MT16x32_B16
{
    // tile read size
    static constexpr index_t kMN = 32;
    static constexpr index_t kK  = 16;

    // lane store layout
    static constexpr index_t kMNStoreLane = 16;
    static constexpr index_t kKStoreLane  = 4;

    // lane store slice
    static constexpr index_t kMN0StorePerLane = 2;
    static constexpr index_t kMN1StorePerLane = 1;
    static constexpr index_t kKStorePerLane   = 4;

    // for static check
    static constexpr index_t kMNInterleave = 1;

    // vector length for returned data
    static constexpr index_t kVectorLength = kMN0StorePerLane * kMN1StorePerLane * kKStorePerLane;

    template <typename T, index_t offset, bool use_m0 = false, bool skip_mov_m0 = false>
    CK_TILE_DEVICE void operator()(CK_TILE_LDS_ADDR T* smem,
                                   ext_vector_t<T, kVectorLength>& ret,
                                   number<offset>,
                                   bool_constant<use_m0>      = {},
                                   bool_constant<skip_mov_m0> = {}) const
    {
        hcu_ds_read_matrix_trans_format_asm_impl(reinterpret_cast<uintptr_t>(smem),
                                                 ret,
                                                 number<offset>{},
                                                 number<0x2>{}, // element:0x2
                                                 number<0x1>{}, // row:0x1
                                                 number<0x2>{}, // col:0x2
                                                 number<0x0>{}, // alt:0x0
                                                 bool_constant<use_m0>{},
                                                 bool_constant<skip_mov_m0>{});
    }
};

struct WarpDsreadmFormatAttributeImpl_MT16x32_B16_ALT2
{
    // tile read size
    static constexpr index_t kMN = 32;
    static constexpr index_t kK  = 16;

    // lane store layout
    static constexpr index_t kMNStoreLane = 16;
    static constexpr index_t kKStoreLane  = 4;

    // lane store slice
    static constexpr index_t kMN0StorePerLane = 1;
    static constexpr index_t kMN1StorePerLane = 2;
    static constexpr index_t kKStorePerLane   = 4;

    // for static check
    static constexpr index_t kMNInterleave = 2;

    // vector length for returned data
    static constexpr index_t kVectorLength = kMN0StorePerLane * kMN1StorePerLane * kKStorePerLane;

    template <typename T, index_t offset, bool use_m0 = false, bool skip_mov_m0 = false>
    CK_TILE_DEVICE void operator()(T* smem,
                                   ext_vector_t<T, kVectorLength>& ret,
                                   number<offset>,
                                   bool_constant<use_m0>      = {},
                                   bool_constant<skip_mov_m0> = {}) const
    {
        hcu_ds_read_matrix_trans_format_asm_impl(reinterpret_cast<uintptr_t>(smem),
                                                 ret,
                                                 number<offset>{},
                                                 number<0x2>{}, // element:0x2
                                                 number<0x1>{}, // row:0x1
                                                 number<0x2>{}, // col:0x2
                                                 number<0x1>{}, // alt:0x1
                                                 bool_constant<use_m0>{},
                                                 bool_constant<skip_mov_m0>{});
    }
};

} // namespace ck_tile
