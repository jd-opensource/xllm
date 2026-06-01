// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/arch/amd_buffer_addressing.hpp"

namespace ck_tile {

template <typename T, index_t N>
CK_TILE_DEVICE void hcu_async_buffer_load_asm_impl(uintptr_t m0,
                                                   const int32x4_t& buffer_res,
                                                   index_t voffset,
                                                   index_t ioffset)
{
    static constexpr index_t bytes = sizeof(T) * N;

    static_assert(bytes == 4 || bytes == 8 || bytes == 16, "wrong! not implemented vector size");

    const auto wave_m0 = __builtin_amdgcn_readfirstlane(m0);

    if constexpr(bytes == 4)
    {
        asm volatile("s_mov_b32 m0, %0; \n\t"
                     "buffer_load_dword %1, %2, 0 offen offset:%3 lds;\n\t" ::"s"(wave_m0),
                     "v"(voffset),
                     "s"(buffer_res),
                     "n"(ioffset)
                     : "memory");
    }
    else if constexpr(bytes == 8)
    {
        asm volatile("s_mov_b32 m0, %0; \n\t"
                     "buffer_load_dwordx2 %1, %2, 0 offen offset:%3 lds;\n\t" ::"s"(wave_m0),
                     "v"(voffset),
                     "s"(buffer_res),
                     "n"(ioffset)
                     : "memory");
    }
    else
    {
        asm volatile("s_mov_b32 m0, %0; \n\t"
                     "buffer_load_dwordx4 %1, %2, 0 offen offset:%3 lds;\n\t" ::"s"(wave_m0),
                     "v"(voffset),
                     "s"(buffer_res),
                     "n"(ioffset)
                     : "memory");
    }
}

template <typename T, index_t N>
CK_TILE_DEVICE void hcu_async_struct_buffer_load_asm_impl(
    uintptr_t m0, const int32x4_t& buffer_res, index_t vindex, index_t voffset, index_t ioffset)
{
    static constexpr index_t bytes = sizeof(T) * N;

    static_assert(bytes == 4 || bytes == 8 || bytes == 16, "wrong! not implemented vector size");

    const auto wave_m0 = __builtin_amdgcn_readfirstlane(m0);

    int32x2_t offsets = {vindex, voffset};

    if constexpr(bytes == 4)
    {
        asm volatile("s_mov_b32 m0, %0; \n\t"
                     "buffer_load_dword %1, %2, 0 idxen offen offset:%3 lds;\n\t" ::"s"(wave_m0),
                     "v"(offsets),
                     "s"(buffer_res),
                     "n"(ioffset)
                     : "memory");
    }
    else if constexpr(bytes == 8)
    {
        asm volatile("s_mov_b32 m0, %0; \n\t"
                     "buffer_load_dwordx2 %1, %2, 0 idxen offen offset:%3 lds;\n\t" ::"s"(wave_m0),
                     "v"(offsets),
                     "s"(buffer_res),
                     "n"(ioffset)
                     : "memory");
    }
    else
    {
        asm volatile("s_mov_b32 m0, %0; \n\t"
                     "buffer_load_dwordx4 %1, %2, 0 idxen offen offset:%3 lds;\n\t" ::"s"(wave_m0),
                     "v"(offsets),
                     "s"(buffer_res),
                     "n"(ioffset)
                     : "memory");
    }
}

// global -> lds raw buffer load (offen)
template <typename T, index_t N, bool oob_conditional_check = true>
CK_TILE_DEVICE void hcu_async_buffer_load_asm(CK_TILE_LDS_ADDR T* smem,
                                              const int32x4_t& buffer_res,
                                              index_t voffset,
                                              index_t ioffset,
                                              bool is_valid_element,
                                              bool_constant<oob_conditional_check> = {})
{
    auto const m0 = reinterpret_cast<uintptr_t>(smem);

    if constexpr(oob_conditional_check)
    {
        hcu_async_buffer_load_asm_impl<T, N>(
            m0, buffer_res, is_valid_element ? voffset : buffer_res[2], ioffset);
    }
    else
    {
        hcu_async_buffer_load_asm_impl<T, N>(m0, buffer_res, voffset, ioffset);
    }
}

template <typename T, index_t N, bool oob_conditional_check = true>
CK_TILE_DEVICE void hcu_async_buffer_load_asm(CK_TILE_LDS_ADDR T* smem,
                                              const int32x4_t& buffer_res,
                                              index_t voffset,
                                              index_t ioffset,
                                              index_t lds_wrap_offset,
                                              bool is_valid_element,
                                              bool_constant<oob_conditional_check> = {})
{
    auto const m0 =
        reinterpret_cast<uintptr_t>(smem) | (static_cast<uintptr_t>(lds_wrap_offset) << 16);

    if constexpr(oob_conditional_check)
    {
        hcu_async_buffer_load_asm_impl<T, N>(
            m0, buffer_res, is_valid_element ? voffset : buffer_res[2], ioffset);
    }
    else
    {
        hcu_async_buffer_load_asm_impl<T, N>(m0, buffer_res, voffset, ioffset);
    }
}

// global -> lds struct buffer load (idxen + offen)
template <typename T, index_t N, bool oob_conditional_check = true>
CK_TILE_DEVICE void hcu_async_struct_buffer_load_asm(CK_TILE_LDS_ADDR T* smem,
                                                     const int32x4_t& buffer_res,
                                                     index_t vindex,
                                                     index_t voffset,
                                                     index_t ioffset,
                                                     bool is_valid_element,
                                                     bool_constant<oob_conditional_check> = {})
{
    auto const m0 = reinterpret_cast<uintptr_t>(smem);

    if constexpr(oob_conditional_check)
    {
        hcu_async_struct_buffer_load_asm_impl<T, N>(
            m0, buffer_res, is_valid_element ? vindex : buffer_res[2], voffset, ioffset);
    }
    else
    {
        hcu_async_struct_buffer_load_asm_impl<T, N>(m0, buffer_res, vindex, voffset, ioffset);
    }
}

template <typename T, index_t N, bool oob_conditional_check = true>
CK_TILE_DEVICE void hcu_async_struct_buffer_load_asm(CK_TILE_LDS_ADDR T* smem,
                                                     const int32x4_t& buffer_res,
                                                     index_t vindex,
                                                     index_t voffset,
                                                     index_t ioffset,
                                                     index_t lds_wrap_offset,
                                                     bool is_valid_element,
                                                     bool_constant<oob_conditional_check> = {})
{
    auto const m0 =
        reinterpret_cast<uintptr_t>(smem) | (static_cast<uintptr_t>(lds_wrap_offset) << 16);

    if constexpr(oob_conditional_check)
    {
        hcu_async_struct_buffer_load_asm_impl<T, N>(
            m0, buffer_res, is_valid_element ? vindex : buffer_res[2], voffset, ioffset);
    }
    else
    {
        hcu_async_struct_buffer_load_asm_impl<T, N>(m0, buffer_res, vindex, voffset, ioffset);
    }
}

namespace detail {

// clang-format off
template <index_t Bytes> struct hcu_buffer_load_trait;

template <> struct hcu_buffer_load_trait<1> { using payload_t = uint8_t; };
template <> struct hcu_buffer_load_trait<2> { using payload_t = uint16_t; };
template <> struct hcu_buffer_load_trait<4> { using payload_t = float; };
template <> struct hcu_buffer_load_trait<8> { using payload_t = fp32x2_t; };
template <> struct hcu_buffer_load_trait<16> { using payload_t = fp32x4_t; };
// clang-format on

} // namespace detail

template <typename T, index_t N>
CK_TILE_DEVICE thread_buffer<T, N>
hcu_buffer_load_asm_impl(const int32x4_t& buffer_res, index_t voffset, index_t ioffset)
{
    static constexpr index_t bytes = sizeof(T) * N;

    static_assert(bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8 || bytes == 16,
                  "wrong! not implemented vector size");

    using rtn_type = thread_buffer<T, N>;

    using payload_t = typename detail::hcu_buffer_load_trait<bytes>::payload_t;

    payload_t tmp;

    if constexpr(bytes == 1)
    {
        asm volatile("buffer_load_ubyte %0, %1, %2, 0 offen offset:%3;\n\t"
                     : "=v"(tmp)
                     : "v"(voffset), "s"(buffer_res), "n"(ioffset)
                     : "memory");
        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(bytes == 2)
    {
        asm volatile("buffer_load_ushort %0, %1, %2, 0 offen offset:%3;\n\t"
                     : "=v"(tmp)
                     : "v"(voffset), "s"(buffer_res), "n"(ioffset)
                     : "memory");
        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(bytes == 4)
    {
        asm volatile("buffer_load_dword %0, %1, %2, 0 offen offset:%3;\n\t"
                     : "=v"(tmp)
                     : "v"(voffset), "s"(buffer_res), "n"(ioffset)
                     : "memory");
        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(bytes == 8)
    {
        asm volatile("buffer_load_dwordx2 %0, %1, %2, 0 offen offset:%3;\n\t"
                     : "=v"(tmp)
                     : "v"(voffset), "s"(buffer_res), "n"(ioffset)
                     : "memory");
        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(bytes == 16)
    {
        asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen offset:%3;\n\t"
                     : "=v"(tmp)
                     : "v"(voffset), "s"(buffer_res), "n"(ioffset)
                     : "memory");
        return bit_cast<rtn_type>(tmp);
    }
}

template <typename T, index_t N>
CK_TILE_DEVICE thread_buffer<T, N> hcu_struct_buffer_load_asm_impl(const int32x4_t& buffer_res,
                                                                   index_t vindex,
                                                                   index_t voffset,
                                                                   index_t ioffset)
{
    static constexpr index_t bytes = sizeof(T) * N;

    static_assert(bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8 || bytes == 16,
                  "wrong! not implemented vector size");

    int32x2_t offsets = {vindex, voffset};

    using rtn_type = thread_buffer<T, N>;

    using payload_t = typename detail::hcu_buffer_load_trait<bytes>::payload_t;

    payload_t tmp;

    if constexpr(bytes == 1)
    {
        asm volatile("buffer_load_ubyte %0, %1, %2, 0 idxen offen offset:%3;\n\t"
                     : "=v"(tmp)
                     : "v"(offsets), "s"(buffer_res), "n"(ioffset)
                     : "memory");
        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(bytes == 2)
    {
        asm volatile("buffer_load_ushort %0, %1, %2, 0 idxen offen offset:%3;\n\t"
                     : "=v"(tmp)
                     : "v"(offsets), "s"(buffer_res), "n"(ioffset)
                     : "memory");
        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(bytes == 4)
    {
        asm volatile("buffer_load_dword %0, %1, %2, 0 idxen offen offset:%3;\n\t"
                     : "=v"(tmp)
                     : "v"(offsets), "s"(buffer_res), "n"(ioffset)
                     : "memory");
        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(bytes == 8)
    {
        asm volatile("buffer_load_dwordx2 %0, %1, %2, 0 idxen offen offset:%3;\n\t"
                     : "=v"(tmp)
                     : "v"(offsets), "s"(buffer_res), "n"(ioffset)
                     : "memory");
        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(bytes == 16)
    {
        asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 idxen offen offset:%3;\n\t"
                     : "=v"(tmp)
                     : "v"(offsets), "s"(buffer_res), "n"(ioffset)
                     : "memory");
        return bit_cast<rtn_type>(tmp);
    }
}

// global -> vgpr raw buffer load (offen)
template <typename T, index_t N, bool oob_conditional_check = true>
CK_TILE_DEVICE auto hcu_buffer_load_asm(const int32x4_t& buffer_res,
                                        index_t voffset,
                                        index_t ioffset,
                                        bool is_valid_element,
                                        bool_constant<oob_conditional_check> = {})
{
    if constexpr(oob_conditional_check)
    {
        return hcu_buffer_load_asm_impl<T, N>(
            buffer_res, is_valid_element ? voffset : buffer_res[2], ioffset);
    }
    else
    {
        return hcu_buffer_load_asm_impl<T, N>(buffer_res, voffset, ioffset);
    }
}

// global -> lds struct buffer load (idxen + offen)
template <typename T, index_t N, bool oob_conditional_check = true>
CK_TILE_DEVICE auto hcu_struct_buffer_load_asm(const int32x4_t& buffer_res,
                                               index_t vindex,
                                               index_t voffset,
                                               index_t ioffset,
                                               bool is_valid_element,
                                               bool_constant<oob_conditional_check> = {})
{
    if constexpr(oob_conditional_check)
    {
        return hcu_struct_buffer_load_asm_impl<T, N>(
            buffer_res, is_valid_element ? vindex : buffer_res[2], voffset, ioffset);
    }
    else
    {
        return hcu_struct_buffer_load_asm_impl<T, N>(buffer_res, vindex, voffset, ioffset);
    }
}

} // namespace ck_tile
