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

// 128 bit SGPRs to supply buffer resource in buffer instructions
// https://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/testdocbook.html#vector-memory-buffer-instructions
struct __attribute__((packed)) buffer_resource
{
    CK_TILE_HOST_DEVICE buffer_resource(const void* ptr_, uint32_t range_, uint32_t config_)
        : ptr(ptr_), range(range_), config(config_)
    {
    }

    union
    {
        const void* ptr;
        uint64_t const_stride;
    };
    uint32_t range;
    uint32_t config;
};

CK_TILE_DEVICE int32x4_t make_wave_buffer_resource(const void* ptr, uint32_t size = 0xffffffff)
{
    buffer_resource res{ptr, size, CK_TILE_BUFFER_RESOURCE_3RD_DWORD};
    int32x4_t r = __builtin_bit_cast(int32x4_t, res);
    r.x         = __builtin_amdgcn_readfirstlane(r.x);
    r.y         = __builtin_amdgcn_readfirstlane(r.y);
    r.z         = __builtin_amdgcn_readfirstlane(r.z);
    r.w         = __builtin_amdgcn_readfirstlane(r.w);
    return r;
}

CK_TILE_DEVICE int32x4_t make_wave_buffer_resource(const void* ptr,
                                                   uint32_t size, // in bytes
                                                   uint32_t const_stride)
{
    const uint32_t strided_elem_space_size = __builtin_amdgcn_readfirstlane(size / const_stride);

    buffer_resource res{ptr, strided_elem_space_size, CK_TILE_BUFFER_RESOURCE_3RD_DWORD};

    // enable cache swizzle
    res.const_stride |= uint64_t(const_stride) << 48;
    res.const_stride |= uint64_t(1) << 62;

    int32x4_t r = __builtin_bit_cast(int32x4_t, res);
    r.x         = __builtin_amdgcn_readfirstlane(r.x);
    r.y         = __builtin_amdgcn_readfirstlane(r.y);
    r.z         = __builtin_amdgcn_readfirstlane(r.z);
    r.w         = __builtin_amdgcn_readfirstlane(r.w);
    return r;
}

namespace impl {
// below type indicate the data type used for buffer load inline asm
// clang-format off
template<index_t N, typename T> struct buffer_load_trait;

template<typename T> struct buffer_load_trait<16, T> { using payload_t = fp32x4_t; };
template<typename T> struct buffer_load_trait<8 , T> { using payload_t = fp32x2_t; };
template<typename T> struct buffer_load_trait<4 , T> { using payload_t = float; };
template<typename T> struct buffer_load_trait<2 , T> { using payload_t = float; };
template<typename T> struct buffer_load_trait<1 , T> { using payload_t = float; };

#if CK_TILE_BUFFER_LOAD_RAW_BF16_WA
template<> struct buffer_load_trait<16, thread_buffer<bf16_t, 8>> { using payload_t = bf16x8_t; };
template<> struct buffer_load_trait<8 , thread_buffer<bf16_t, 4>> { using payload_t = bf16x4_t; };
template<> struct buffer_load_trait<4 , thread_buffer<bf16_t, 2>> { using payload_t = bf16x2_t; };
#endif
// clang-format on
} // namespace impl

// TODO: glc/slc/...
template <index_t bytes, bool pre_nop = false>
struct buffer_load;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
// TODO: strict aliasing rule seems fail when reinterpret_cast between vector type
// (exp_vector_type(xxx))
template <bool pre_nop>
struct buffer_load<16, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/       = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 16);
        using mbuf_t = typename impl::buffer_load_trait<16, T>::payload_t;
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "buffer_load_dwordx4 %0, %1, %2, 0 offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
        else
            asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
    }
};

template <bool pre_nop>
struct buffer_load<8, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/       = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 8);
        using mbuf_t = typename impl::buffer_load_trait<8, T>::payload_t;
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "buffer_load_dwordx2 %0, %1, %2, 0 offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
        else
            asm volatile("buffer_load_dwordx2 %0, %1, %2, 0 offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
    }
};

template <bool pre_nop>
struct buffer_load<4, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/       = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 4);
        using mbuf_t = typename impl::buffer_load_trait<4, T>::payload_t;
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "buffer_load_dword %0, %1, %2, 0 offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
        else
            asm volatile("buffer_load_dword %0, %1, %2, 0 offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
    }
};

template <bool pre_nop>
struct buffer_load<2, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/       = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 4); // subdword is buggy, use dword buf and convert manually
        using mbuf_t = typename impl::buffer_load_trait<2, T>::payload_t;
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "buffer_load_ushort %0, %1, %2, 0 offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
        else
            asm volatile("buffer_load_ushort %0, %1, %2, 0 offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
    }
};

template <bool pre_nop>
struct buffer_load<1, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/       = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 4);
        using mbuf_t = typename impl::buffer_load_trait<1, T>::payload_t;
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "buffer_load_ubyte %0, %1, %2, 0 offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
        else
            asm volatile("buffer_load_ubyte %0, %1, %2, 0 offen offset:%3"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset)
                         : "memory");
    }
};

template <index_t bytes, bool pre_nop = false>
struct buffer_load_if;

template <bool pre_nop>
struct buffer_load_if<16, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag           = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 16);
        auto saved_exec = __builtin_amdgcn_read_exec();
        using mbuf_t    = typename impl::buffer_load_trait<16, T>::payload_t;
        static_assert(sizeof(mbuf_t) == sizeof(T));
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "v_cmpx_le_u32 exec, 1, %4\n"
                         "buffer_load_dwordx4 %0, %1, %2, 0 offen offset:%3\n"
                         "s_mov_b64 exec %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
        else
            asm volatile("v_cmpx_le_u32 exec, 1, %4\n"
                         "buffer_load_dwordx4 %0, %1, %2, 0 offen offset:%3\n"
                         "s_mov_b64 exec %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
    }
};

template <bool pre_nop>
struct buffer_load_if<8, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag           = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 8);
        auto saved_exec = __builtin_amdgcn_read_exec();
        using mbuf_t    = typename impl::buffer_load_trait<8, T>::payload_t;
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "v_cmpx_le_u32 exec, 1, %4\n"
                         "buffer_load_dwordx2 %0, %1, %2, 0 offen offset:%3\n"
                         "s_mov_b64 exec %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
        else
            asm volatile("v_cmpx_le_u32 exec, 1, %4\n"
                         "buffer_load_dwordx2 %0, %1, %2, 0 offen offset:%3\n"
                         "s_mov_b64 exec %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
    }
};

template <bool pre_nop>
struct buffer_load_if<4, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag           = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 4);
        auto saved_exec = __builtin_amdgcn_read_exec();
        using mbuf_t    = typename impl::buffer_load_trait<4, T>::payload_t;
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "v_cmpx_le_u32 exec, 1, %4\n"
                         "buffer_load_dword %0, %1, %2, 0 offen offset:%3\n"
                         "s_mov_b64 exec %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
        else
            asm volatile("v_cmpx_le_u32 exec, 1, %4\n"
                         "buffer_load_dword %0, %1, %2, 0 offen offset:%3\n"
                         "s_mov_b64 exec %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
    }
};

template <bool pre_nop>
struct buffer_load_if<2, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag           = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 4);
        auto saved_exec = __builtin_amdgcn_read_exec();
        using mbuf_t    = typename impl::buffer_load_trait<2, T>::payload_t;
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "v_cmpx_le_u32 exec, 1, %4\n"
                         "buffer_load_ushort %0, %1, %2, 0 offen offset:%3\n"
                         "s_mov_b64 exec %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
        else
            asm volatile("v_cmpx_le_u32 exec, 1, %4\n"
                         "buffer_load_ushort %0, %1, %2, 0 offen offset:%3\n"
                         "s_mov_b64 exec %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
    }
};

template <bool pre_nop>
struct buffer_load_if<1, pre_nop>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag           = 0,
                                   bool_constant<pre_nop> = {})
    {
        static_assert(sizeof(T) == 4);
        auto saved_exec = __builtin_amdgcn_read_exec();
        using mbuf_t    = typename impl::buffer_load_trait<1, T>::payload_t;
        if constexpr(pre_nop)
            asm volatile("s_nop 4\n"
                         "v_cmpx_le_u32 exec, 1, %4\n"
                         "buffer_load_ubyte %0, %1, %2, 0 offen offset:%3\n"
                         "s_mov_b64 exec %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
        else
            asm volatile("v_cmpx_le_u32 exec, 1, %4\n"
                         "buffer_load_ubyte %0, %1, %2, 0 offen offset:%3\n"
                         "s_mov_b64 exec %5"
                         : "+v"(reinterpret_cast<mbuf_t&>(value))
                         : "v"(v_offset), "s"(res), "n"(i_offset), "v"(flag), "s"(saved_exec)
                         : "memory");
    }
};
#pragma clang diagnostic pop // "-Wundefined-reinterpret-cast"
template <index_t bytes>
struct buffer_store;

template <>
struct buffer_store<16>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 1)
    {
        static_assert(sizeof(T) == 16);
        using mbuf_t = fp32x4_t;
        asm volatile("buffer_store_dwordx4 %0, %1, %2, 0 offen offset:%3"
                     :
                     : "v"(bit_cast<mbuf_t>(value)), "v"(v_offset), "s"(res), "n"(i_offset)
                     : "memory");
    }
};

template <>
struct buffer_store<8>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 1)
    {
        static_assert(sizeof(T) == 8);
        using mbuf_t = fp32x2_t;
        asm volatile("buffer_store_dwordx2 %0, %1, %2, 0 offen offset:%3"
                     :
                     : "v"(bit_cast<mbuf_t>(value)), "v"(v_offset), "s"(res), "n"(i_offset)
                     : "memory");
    }
};

template <>
struct buffer_store<4>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 1)
    {
        static_assert(sizeof(T) == 4);
        using mbuf_t = float;
        asm volatile("buffer_store_dword %0, %1, %2, 0 offen offset:%3"
                     :
                     : "v"(bit_cast<mbuf_t>(value)), "v"(v_offset), "s"(res), "n"(i_offset)
                     : "memory");
    }
};

template <>
struct buffer_store<2>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 1)
    {
        static_assert(sizeof(T) == 2);
        using mbuf_t = short;
        asm volatile("buffer_store_short %0, %1, %2, 0 offen offset:%3"
                     :
                     : "v"(bit_cast<mbuf_t>(value)), "v"(v_offset), "s"(res), "n"(i_offset)
                     : "memory");
    }
};

template <>
struct buffer_store<1>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t /*flag*/ = 1)
    {
        static_assert(sizeof(T) == 4);
        using mbuf_t = float;
        asm volatile("buffer_store_byte %0, %1, %2, 0 offen offset:%3"
                     :
                     : "v"(bit_cast<mbuf_t>(value)), "v"(v_offset), "s"(res), "n"(i_offset)
                     : "memory");
    }
};

template <index_t bytes>
struct buffer_store_if;

template <>
struct buffer_store_if<16>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        static_assert(sizeof(T) == 16);
        auto save_exec = __builtin_amdgcn_read_exec();
        using mbuf_t   = fp32x4_t;
        asm volatile("v_cmpx_le_u32 exec, 1, %4\n"
                     "buffer_store_dwordx4 %0, %1, %2, 0 offen offset:%3\n"
                     "s_mov_b64 exec %5"
                     :
                     : "v"(bit_cast<mbuf_t>(value)),
                       "v"(v_offset),
                       "s"(res),
                       "n"(i_offset),
                       "v"(flag),
                       "s"(save_exec)
                     : "memory");
    }
};

template <>
struct buffer_store_if<8>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        static_assert(sizeof(T) == 8);
        auto save_exec = __builtin_amdgcn_read_exec();
        // TODO: ugly. rocm-6.0/6.1 seems neet bit_cast to same base type to avoid scratch
        using mbuf_t = ext_vector_t<typename T::value_type, T::size()>;
        asm volatile("v_cmpx_le_u32 exec, 1, %4\n"
                     "buffer_store_dwordx2 %0, %1, %2, 0 offen offset:%3\n"
                     "s_mov_b64 exec %5"
                     :
                     : "v"(bit_cast<mbuf_t>(value)),
                       "v"(v_offset),
                       "s"(res),
                       "n"(i_offset),
                       "v"(flag),
                       "s"(save_exec)
                     : "memory");
    }
};

template <>
struct buffer_store_if<4>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        static_assert(sizeof(T) == 4);
        auto save_exec = __builtin_amdgcn_read_exec();
        using mbuf_t   = float;
        asm volatile("v_cmpx_le_u32 exec, 1, %4\n"
                     "buffer_store_dword %0, %1, %2, 0 offen offset:%3\n"
                     "s_mov_b64 exec %5"
                     :
                     : "v"(bit_cast<mbuf_t>(value)),
                       "v"(v_offset),
                       "s"(res),
                       "n"(i_offset),
                       "v"(flag),
                       "s"(save_exec)
                     : "memory");
    }
};

template <>
struct buffer_store_if<2>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        static_assert(sizeof(T) == 2);
        auto save_exec = __builtin_amdgcn_read_exec();
        using mbuf_t   = short;
        asm volatile("v_cmpx_le_u32 exec, 1, %4\n"
                     "buffer_store_short %0, %1, %2, 0 offen offset:%3\n"
                     "s_mov_b64 exec %5"
                     :
                     : "v"(bit_cast<mbuf_t>(value)),
                       "v"(v_offset),
                       "s"(res),
                       "n"(i_offset),
                       "v"(flag),
                       "s"(save_exec)
                     : "memory");
    }
};

template <>
struct buffer_store_if<1>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(const T& value,
                                   int32x4_t res /*buffer resource*/,
                                   index_t v_offset,
                                   index_t /*s_offset*/,
                                   index_t i_offset /*max 0xFFF*/,
                                   index_t flag = 1)
    {
        static_assert(sizeof(T) == 4);
        auto save_exec = __builtin_amdgcn_read_exec();
        using mbuf_t   = float;
        asm volatile("v_cmpx_le_u32 exec, 1, %4\n"
                     "buffer_store_byte %0, %1, %2, 0 offen offset:%3\n"
                     "s_mov_b64 exec %5"
                     :
                     : "v"(bit_cast<mbuf_t>(value)),
                       "v"(v_offset),
                       "s"(res),
                       "n"(i_offset),
                       "v"(flag),
                       "s"(save_exec)
                     : "memory");
    }
};

CK_TILE_DEVICE void buffer_load_fence(index_t cnt = 0)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
}

CK_TILE_DEVICE void smem_fence(index_t cnt = 0)
{
    asm volatile("s_waitcnt lgkmcnt(%0)" ::"n"(cnt) : "memory");
}

namespace impl {
// below type indicate the data type used for buffer load inline asm
// clang-format off
template<index_t N, typename T> struct smem_load_trait;

template<typename T> struct smem_load_trait<16, T> { using payload_t = fp32x4_t; };
template<typename T> struct smem_load_trait<8 , T> { using payload_t = fp32x2_t; };
template<typename T> struct smem_load_trait<4 , T> { using payload_t = float; };
template<typename T> struct smem_load_trait<2 , T> { using payload_t = float; };
template<typename T> struct smem_load_trait<1 , T> { using payload_t = float; };

// clang-format on
} // namespace impl

// NOTE: smem load/store no need pre_nop to make sure dependency by sw, happy :)
template <index_t>
struct smem_load;

template <>
struct smem_load<16>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value, index_t v_offset, index_t i_offset)
    {
        static_assert(sizeof(T) == 16);
        using mbuf_t = typename impl::smem_load_trait<16, T>::payload_t;
#if CK_TILE_WORKAROUND_AVOID_V_BFI_B32_INSTS == 1
        mbuf_t tmp;
        asm volatile("ds_read_b128 %0, %1 offset:%2"
                     : "=v"(tmp) // ! direct write
                     : "v"(v_offset), "n"(i_offset)
                     : "memory");

        value = bit_cast<T>(tmp);
#else
        asm volatile("ds_read_b128 %0, %1 offset:%2"
                     : "=v"(reinterpret_cast<mbuf_t&>(value)) // ! direct write
                     : "v"(v_offset), "n"(i_offset)
                     : "memory");
#endif
    }
};

template <>
struct smem_load<8>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value, index_t v_offset, index_t i_offset)
    {
        static_assert(sizeof(T) == 8);
        using mbuf_t = typename impl::smem_load_trait<8, T>::payload_t;
#if CK_TILE_WORKAROUND_AVOID_V_BFI_B32_INSTS == 1
        mbuf_t tmp;
        asm volatile("ds_read_b64 %0, %1 offset:%2"
                     : "=v"(tmp) // ! direct write
                     : "v"(v_offset), "n"(i_offset)
                     : "memory");
        value = bit_cast<T>(tmp);
#else
        asm volatile("ds_read_b64 %0, %1 offset:%2"
                     : "=v"(reinterpret_cast<mbuf_t&>(value)) // ! direct write
                     : "v"(v_offset), "n"(i_offset)
                     : "memory");
#endif
    }
};

template <>
struct smem_load<4>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value, index_t v_offset, index_t i_offset)
    {
        static_assert(sizeof(T) == 4);
        using mbuf_t = typename impl::smem_load_trait<4, T>::payload_t;
#if CK_TILE_WORKAROUND_AVOID_V_BFI_B32_INSTS == 1
        mbuf_t tmp;
        asm volatile("ds_read_b32 %0, %1 offset:%2"
                     : "=v"(tmp) // ! direct write
                     : "v"(v_offset), "n"(i_offset)
                     : "memory");
        value = bit_cast<T>(tmp);
#else
        asm volatile("ds_read_b32 %0, %1 offset:%2"
                     : "=v"(reinterpret_cast<mbuf_t&>(value)) // ! direct write
                     : "v"(v_offset), "n"(i_offset)
                     : "memory");
#endif
    }
};

template <>
struct smem_load<2>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value, index_t v_offset, index_t i_offset)
    {
        static_assert(sizeof(T) == 4); // subdword is buggy, use dword buf and convert manually
        using mbuf_t = typename impl::smem_load_trait<1, T>::payload_t;
#if CK_TILE_WORKAROUND_AVOID_V_BFI_B32_INSTS == 1
        mbuf_t tmp;
        asm volatile("ds_read_u16 %0, %1 offset:%2"
                     : "=v"(tmp) // ! direct write
                     : "v"(v_offset), "n"(i_offset)
                     : "memory");
        value = bit_cast<T>(tmp);
#else
        asm volatile("ds_read_u16 %0, %1 offset:%2"
                     : "=v"(reinterpret_cast<mbuf_t&>(value)) // ! direct write
                     : "v"(v_offset), "n"(i_offset)
                     : "memory");
#endif
    }
};

template <>
struct smem_load<1>
{
    template <typename T>
    CK_TILE_DEVICE void operator()(T& value, index_t v_offset, index_t i_offset)
    {
        static_assert(sizeof(T) == 4);
        using mbuf_t = typename impl::smem_load_trait<1, T>::payload_t;
#if CK_TILE_WORKAROUND_AVOID_V_BFI_B32_INSTS == 1
        mbuf_t tmp;
        asm volatile("ds_read_u8 %0, %1 offset:%2"
                     : "=v"(tmp) // ! direct write
                     : "v"(v_offset), "n"(i_offset)
                     : "memory");
        value = bit_cast<T>(tmp);
#else
        asm volatile("ds_read_u8 %0, %1 offset:%2"
                     : "=v"(reinterpret_cast<mbuf_t&>(value)) // ! direct write
                     : "v"(v_offset), "n"(i_offset)
                     : "memory");
#endif
    }
};

// clang-format off
namespace impl{

// can't use "+v" since there could be potential extra move(read/write)
// use "v" can help remove such duplicated moves
// besides, fake this as "memory" operation to force later valu after this fence
// TODO: may have scratch (because this is memory?)
//       need to reduce extra move inside compiler
template<index_t N>
CK_TILE_DEVICE void insert_dummy_dep_per_dword(array<float, N>& b)
{
    constexpr auto kSize = remove_cvref_t<decltype(b)>::size(); 
    static_for<0, kSize, 1>{}([&](auto i){
        asm volatile(" " : : "v"(b.get(number<i>{})) : "memory");
    });
}
#if 1
// below specialization just merge size() of dwords into single section
template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<2>(array<float, 2>& b)
{
    asm volatile(" " : : "v"(b.get(number<0>{})), "v"(b.get(number<1>{})) : "memory");
}

template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<3>(array<float, 3>& b)
{
    asm volatile(" " : : "v"(b.get(number<0>{})), "v"(b.get(number<1>{})), "v"(b.get(number<2>{})) : "memory");
}

template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<4>(array<float, 4>& b)
{
    asm volatile(" " : : "v"(b.get(number<0>{})), "v"(b.get(number<1>{})), "v"(b.get(number<2>{})), "v"(b.get(number<3>{})) : "memory");
}

template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<8>(array<float, 8>& b)
{
    asm volatile(" " : : "v"(b.get(number<0>{})), "v"(b.get(number<1>{})), "v"(b.get(number<2>{})), "v"(b.get(number<3>{})),
                         "v"(b.get(number<4>{})), "v"(b.get(number<5>{})), "v"(b.get(number<6>{})), "v"(b.get(number<7>{})) : "memory");
}

template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<16>(array<float, 16>& b)
{
    asm volatile(" " : : "v"(b.get(number<0>{})), "v"(b.get(number<1>{})), "v"(b.get(number<2>{})), "v"(b.get(number<3>{})),
                         "v"(b.get(number<4>{})), "v"(b.get(number<5>{})), "v"(b.get(number<6>{})), "v"(b.get(number<7>{})),
                         "v"(b.get(number<8>{})), "v"(b.get(number<9>{})), "v"(b.get(number<10>{})), "v"(b.get(number<11>{})),
                         "v"(b.get(number<12>{})), "v"(b.get(number<13>{})), "v"(b.get(number<14>{})), "v"(b.get(number<15>{})) : "memory");
}

template<>
CK_TILE_DEVICE void insert_dummy_dep_per_dword<32>(array<float, 32>& b)
{
    asm volatile(" " : : "v"(b.get(number<0>{})), "v"(b.get(number<1>{})), "v"(b.get(number<2>{})), "v"(b.get(number<3>{})),
                         "v"(b.get(number<4>{})), "v"(b.get(number<5>{})), "v"(b.get(number<6>{})), "v"(b.get(number<7>{})),
                         "v"(b.get(number<8>{})), "v"(b.get(number<9>{})), "v"(b.get(number<10>{})), "v"(b.get(number<11>{})),
                         "v"(b.get(number<12>{})), "v"(b.get(number<13>{})), "v"(b.get(number<14>{})), "v"(b.get(number<15>{})),
                         "v"(b.get(number<16>{})), "v"(b.get(number<17>{})), "v"(b.get(number<18>{})), "v"(b.get(number<19>{})),
                         "v"(b.get(number<20>{})), "v"(b.get(number<21>{})), "v"(b.get(number<22>{})), "v"(b.get(number<23>{})),
                         "v"(b.get(number<24>{})), "v"(b.get(number<25>{})), "v"(b.get(number<26>{})), "v"(b.get(number<27>{})),
                         "v"(b.get(number<28>{})), "v"(b.get(number<29>{})), "v"(b.get(number<30>{})), "v"(b.get(number<31>{})) : "memory");
}
#endif
CK_TILE_DEVICE void insert_dummy_dep() {}

template<typename T>
CK_TILE_DEVICE void insert_dummy_dep(T & buffer)
{
    // TODO: indeed we expect T to be multiple of dword. subdword is always buggy
    using da_type = array<float, (sizeof(T) + 3) / 4>;
    auto & dummy = reinterpret_cast<da_type&>(buffer);
    insert_dummy_dep_per_dword(dummy);
}

template<typename Tx, typename... Ty>
CK_TILE_DEVICE void insert_dummy_dep(Tx& bx, Ty&... by)
{
    insert_dummy_dep(bx);
    insert_dummy_dep(by...);
}
}
// clang-format on
template <typename... T>
CK_TILE_DEVICE void buffer_load_fence(index_t cnt = 0, T&... o)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
    impl::insert_dummy_dep(o...);
}

CK_TILE_DEVICE void buffer_store_fence(index_t cnt = 0)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
}

// buffer load i8
CK_TILE_DEVICE_EXTERN int8_t
llvm_amdgcn_raw_buffer_load_i8(int32x4_t srsrc,
                               index_t voffset,
                               index_t soffset,
                               index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i8");

CK_TILE_DEVICE_EXTERN int8x2_t
llvm_amdgcn_raw_buffer_load_i8x2(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2i8");

CK_TILE_DEVICE_EXTERN int8x4_t
llvm_amdgcn_raw_buffer_load_i8x4(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i8");

// buffer load i16
CK_TILE_DEVICE_EXTERN int16_t
llvm_amdgcn_raw_buffer_load_i16(int32x4_t srsrc,
                                index_t voffset,
                                index_t soffset,
                                index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i16");

CK_TILE_DEVICE_EXTERN int16x2_t
llvm_amdgcn_raw_buffer_load_i16x2(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2i16");

CK_TILE_DEVICE_EXTERN int16x4_t
llvm_amdgcn_raw_buffer_load_i16x4(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i16");

// buffer load i32
CK_TILE_DEVICE_EXTERN int32_t
llvm_amdgcn_raw_buffer_load_i32(int32x4_t srsrc,
                                index_t voffset,
                                index_t soffset,
                                index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i32");

CK_TILE_DEVICE_EXTERN int32x2_t
llvm_amdgcn_raw_buffer_load_i32x2(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2i32");

CK_TILE_DEVICE_EXTERN int32x4_t
llvm_amdgcn_raw_buffer_load_i32x4(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

// buffer load fp16
CK_TILE_DEVICE_EXTERN _Float16
llvm_amdgcn_raw_buffer_load_fp16(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f16");

CK_TILE_DEVICE_EXTERN fp16x2_t
llvm_amdgcn_raw_buffer_load_fp16x2(int32x4_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2f16");

CK_TILE_DEVICE_EXTERN fp16x4_t
llvm_amdgcn_raw_buffer_load_fp16x4(int32x4_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4f16");

// buffer load fp32
CK_TILE_DEVICE_EXTERN float
llvm_amdgcn_raw_buffer_load_fp32(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f32");

CK_TILE_DEVICE_EXTERN fp32x2_t
llvm_amdgcn_raw_buffer_load_fp32x2(int32x4_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2f32");

CK_TILE_DEVICE_EXTERN fp32x4_t
llvm_amdgcn_raw_buffer_load_fp32x4(int32x4_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4f32");

// buffer store i8
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i8(int8_t vdata,
                                int32x4_t rsrc,
                                index_t voffset,
                                index_t soffset,
                                index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.i8");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i8x2(int8x2_t vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2i8");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i8x4(int8x4_t vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i8");

// buffer store i16
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i16(int16_t vdata,
                                 int32x4_t rsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.i16");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i16x2(int16x2_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2i16");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i16x4(int16x4_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i16");

// buffer store i32
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i32(int32_t vdata,
                                 int32x4_t rsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.i32");

// buffer store ui16
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_ui16(uint16_t vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.i16");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_ui16x2(uint16x2_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2i16");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_ui16x4(uint16x4_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i16");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i32x2(int32x2_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2i32");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_i32x4(int32x4_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i32");

// buffer store fp16
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_fp16(_Float16 vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.f16");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_fp16x2(fp16x2_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2f16");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_fp16x4(fp16x4_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4f16");

// buffer store fp32
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_fp32(float vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.f32");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_fp32x2(fp32x2_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2f32");

CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_store_fp32x4(fp32x4_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4f32");

// buffer atomic-add fp16
CK_TILE_DEVICE_EXTERN fp16x2_t llvm_amdgcn_raw_buffer_atomic_add_fp16x2(
    fp16x2_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fadd.v2f16");

// buffer atomic-add i32
CK_TILE_DEVICE_EXTERN int32_t llvm_amdgcn_raw_buffer_atomic_add_i32(
    int32_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.add.i32");

// buffer atomic-add fp32
CK_TILE_DEVICE_EXTERN float llvm_amdgcn_raw_buffer_atomic_add_fp32(
    float vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fadd.f32");

// buffer atomic-max fp64
CK_TILE_DEVICE_EXTERN double
llvm_amdgcn_raw_buffer_atomic_max_fp64(double vdata,
                                       int32x4_t rsrc, // dst_wave_buffer_resource
                                       int voffset,    // dst_thread_addr_offset
                                       int soffset,    // dst_wave_addr_offset
                                       int glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fmax.f64");

// Direct loads from global to LDS.
CK_TILE_DEVICE_EXTERN void
llvm_amdgcn_raw_buffer_load_lds(int32x4_t rsrc,
                                __attribute__((address_space(3))) uint32_t* lds_ptr,
                                index_t size,
                                index_t voffset,
                                index_t soffset,
                                index_t offset,
                                index_t aux) __asm("llvm.amdgcn.raw.buffer.load.lds");

template <bool pre_nop = false>
CK_TILE_DEVICE void async_buffer_load_dword_v(void* smem,
                                              int32x4_t rsrc,
                                              index_t voffset,
                                              index_t /*soffset*/,
                                              index_t ioffset /*max 0xFFF*/,
                                              index_t /*flag*/       = 0,
                                              bool_constant<pre_nop> = {})
{
    if constexpr(pre_nop)
        asm volatile("s_nop 4\n"
                     "buffer_load_dword %1, %2, 0 offen offset:%3 lds"
                     : "=r"(smem) /*dummy dependency for smem*/
                     : "v"(voffset), "s"(rsrc), "n"(ioffset)
                     : "memory");
    else
        asm volatile("buffer_load_dword %1, %2, 0 offen offset:%3 lds"
                     : "=r"(smem) /*dummy dependency for smem*/
                     : "v"(voffset), "s"(rsrc), "n"(ioffset)
                     : "memory");
}

CK_TILE_DEVICE void async_buffer_load_fence(index_t cnt = 0)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
}

// memory coherency bit for buffer store/load instruction
// check ISA manual for each GFX target
// e.g. for
// https://www.amd.com/system/files/TechDocs/instinct-mi200-cdna2-instruction-set-architecture.pdf,
// page 67~68
enum struct amd_buffer_coherence_enum
{
    coherence_default = 0, // default value
    glc               = 1,
    slc               = 2,
    glc_slc           = 3,
};

template <index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default>
CK_TILE_DEVICE thread_buffer<int8_t, N>
amd_buffer_load_impl_with_bytes(int32x4_t src_wave_buffer_resource,
                                index_t src_thread_addr_offset,
                                index_t src_wave_addr_offset)
{
    static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32 || N == 64,
                  "wrong! not implemented");

    using rtn_type = thread_buffer<int8_t, N>;

    if constexpr(N == 1)
    {
        return bit_cast<rtn_type>(llvm_amdgcn_raw_buffer_load_i8(src_wave_buffer_resource,
                                                                 src_thread_addr_offset,
                                                                 src_wave_addr_offset,
                                                                 static_cast<index_t>(coherence)));
    }
    else if constexpr(N == 2)
    {

        int16_t tmp = llvm_amdgcn_raw_buffer_load_i16(src_wave_buffer_resource,
                                                      src_thread_addr_offset,
                                                      src_wave_addr_offset,
                                                      static_cast<index_t>(coherence));

        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(N == 4)
    {
        int32_t tmp = llvm_amdgcn_raw_buffer_load_i32(src_wave_buffer_resource,
                                                      src_thread_addr_offset,
                                                      src_wave_addr_offset,
                                                      static_cast<index_t>(coherence));

        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(N == 8)
    {
        int32x2_t tmp = llvm_amdgcn_raw_buffer_load_i32x2(src_wave_buffer_resource,
                                                          src_thread_addr_offset,
                                                          src_wave_addr_offset,
                                                          static_cast<index_t>(coherence));

        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(N == 16)
    {
        int32x4_t tmp = llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                                          src_thread_addr_offset,
                                                          src_wave_addr_offset,
                                                          static_cast<index_t>(coherence));
        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(N == 32)
    {
        int32x4_t tmp0 = llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                                           src_thread_addr_offset,
                                                           src_wave_addr_offset,
                                                           static_cast<index_t>(coherence));
        int32x4_t tmp1 =
            llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                              src_thread_addr_offset,
                                              src_wave_addr_offset + 4 * sizeof(int32_t),
                                              static_cast<index_t>(coherence));
        thread_buffer<int32_t, 8> tmp;

        tmp.template get_as<int32x4_t>()(number<0>{}) = tmp0;
        tmp.template get_as<int32x4_t>()(number<1>{}) = tmp1;

        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(N == 64)
    {
        int32x4_t tmp0 = llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                                           src_thread_addr_offset,
                                                           src_wave_addr_offset,
                                                           static_cast<index_t>(coherence));
        int32x4_t tmp1 =
            llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                              src_thread_addr_offset,
                                              src_wave_addr_offset + 4 * sizeof(int32_t),
                                              static_cast<index_t>(coherence));
        int32x4_t tmp2 =
            llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                              src_thread_addr_offset,
                                              src_wave_addr_offset + 8 * sizeof(int32_t),
                                              static_cast<index_t>(coherence));
        int32x4_t tmp3 =
            llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                              src_thread_addr_offset,
                                              src_wave_addr_offset + 12 * sizeof(int32_t),
                                              static_cast<index_t>(coherence));

        thread_buffer<int32_t, 16> tmp;

        tmp.template get_as<int32x4_t>()(number<0>{}) = tmp0;
        tmp.template get_as<int32x4_t>()(number<1>{}) = tmp1;
        tmp.template get_as<int32x4_t>()(number<2>{}) = tmp2;
        tmp.template get_as<int32x4_t>()(number<3>{}) = tmp3;

        return bit_cast<rtn_type>(tmp);
    }
}

#ifndef BUFFER_LOAD_USE_INLINEASM
#define BUFFER_LOAD_USE_INLINEASM 0
#endif

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default>
CK_TILE_DEVICE thread_buffer<T, N> amd_buffer_load_impl(int32x4_t src_wave_buffer_resource,
                                                        index_t src_thread_addr_offset,
                                                        index_t src_wave_addr_offset)
{
    static_assert(
        (std::is_same<T, double>::value && (N == 1 || N == 2 || N == 4 || N == 8)) ||
            (std::is_same<T, float>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, fp16_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32)) ||
            (std::is_same<T, bf16_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, int32_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, fp8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, bf8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, int8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, pk_int4_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32 || N == 64)),
        "wrong! not implemented");

    using rtn_type = thread_buffer<T, N>;

    if constexpr(std::is_same<T, float>::value) // fp32
    {
        if constexpr(N == 1)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_fp32(src_wave_buffer_resource,
                                                 src_thread_addr_offset,
                                                 src_wave_addr_offset,
                                                 static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 2)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_fp32x2(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 4)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 8)
        {
            thread_buffer<float, 8> tmp;

            tmp.template get_as<fp32x4_t>()(number<0>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<1>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 4 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            return tmp;
        }
        else if constexpr(N == 16)
        {
            thread_buffer<float, 16> tmp;

            tmp.template get_as<fp32x4_t>()(number<0>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<1>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 4 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<2>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 8 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<3>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 12 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            return tmp;
        }
    }
    else if constexpr(std::is_same<T, fp16_t>::value) // fp16
    {
        if constexpr(N == 1)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_fp16(src_wave_buffer_resource,
                                                 src_thread_addr_offset,
                                                 src_wave_addr_offset,
                                                 static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 2)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_fp16x2(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 4)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_fp16x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 8)
        {
            // use fp32 load to mimic fp16 load
            fp32x4_t tmp = llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                              src_thread_addr_offset,
                                                              src_wave_addr_offset,
                                                              static_cast<index_t>(coherence));

            return bit_cast<rtn_type>(tmp);
        }
        else if constexpr(N == 16)
        {
            thread_buffer<fp16_t, 16> tmp;
            // thread_buffer<float, 8> tmp;

            tmp.template get_as<fp32x4_t>()(number<0>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<1>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 4 * sizeof(float),
                                                   static_cast<index_t>(coherence));
            return tmp;
            // return bit_cast<rtn_type>(tmp);
        }
        else if constexpr(N == 32)
        {
            thread_buffer<fp16_t, 32> tmp;
            // thread_buffer<float, 16> tmp;

            tmp.template get_as<fp32x4_t>()(number<0>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset,
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<1>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 4 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<2>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 8 * sizeof(float),
                                                   static_cast<index_t>(coherence));

            tmp.template get_as<fp32x4_t>()(number<3>{}) =
                llvm_amdgcn_raw_buffer_load_fp32x4(src_wave_buffer_resource,
                                                   src_thread_addr_offset,
                                                   src_wave_addr_offset + 12 * sizeof(float),
                                                   static_cast<index_t>(coherence));
            return tmp;
            // return bit_cast<rtn_type>(tmp);
        }
    }
    else if constexpr(std::is_same<T, bf16_t>::value) // bf16
    {
        if constexpr(N == 1)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_i16(src_wave_buffer_resource,
                                                src_thread_addr_offset,
                                                src_wave_addr_offset,
                                                static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 2)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_i16x2(src_wave_buffer_resource,
                                                  src_thread_addr_offset,
                                                  src_wave_addr_offset,
                                                  static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 4)
        {
            return bit_cast<rtn_type>(
                llvm_amdgcn_raw_buffer_load_i16x4(src_wave_buffer_resource,
                                                  src_thread_addr_offset,
                                                  src_wave_addr_offset,
                                                  static_cast<index_t>(coherence)));
        }
        else if constexpr(N == 8)
        {
            int32x4_t tmp = llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                                              src_thread_addr_offset,
                                                              src_wave_addr_offset,
                                                              static_cast<index_t>(coherence));

            return bit_cast<rtn_type>(tmp);
        }
    }
    else // other datatype
    {
        auto raw_data = amd_buffer_load_impl_with_bytes<sizeof(T) * N, coherence>(
            src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset);

        return bit_cast<rtn_type>(raw_data);
    }
}

template <typename T, index_t N>
CK_TILE_DEVICE thread_buffer<T, N>
amd_buffer_load_impl_inline_asm(int32x4_t src_wave_buffer_resource,
                                index_t src_thread_addr_offset,
                                index_t /* src_wave_addr_offset */)
{
    constexpr index_t bytes = sizeof(T) * N;
    static_assert(bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8 || bytes == 16,
                  "wrong! not supported by buffer_load instruction");

    using rtn_type = thread_buffer<T, N>;

    if constexpr(bytes == 1)
    {
        uint8_t tmp;
        asm volatile("buffer_load_ubyte %0, %1, %2, 0 offen;\n\t"
                     : "=v"(tmp)
                     : "v"(src_thread_addr_offset), "s"(src_wave_buffer_resource)
                     : "memory");
        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(bytes == 2)
    {
        uint16_t tmp;
        asm volatile("buffer_load_ushort %0, %1, %2, 0 offen;\n\t"
                     : "=v"(tmp)
                     : "v"(src_thread_addr_offset), "s"(src_wave_buffer_resource)
                     : "memory");
        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(bytes == 4)
    {
        float tmp;
        asm volatile("buffer_load_dword %0, %1, %2, 0 offen;\n\t"
                     : "=v"(tmp)
                     : "v"(src_thread_addr_offset), "s"(src_wave_buffer_resource)
                     : "memory");
        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(bytes == 8)
    {
        fp32x2_t tmp;
        asm volatile("buffer_load_dwordx2 %0, %1, %2, 0 offen;\n\t"
                     : "=v"(tmp)
                     : "v"(src_thread_addr_offset), "s"(src_wave_buffer_resource)
                     : "memory");
        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(bytes == 16)
    {
        fp32x4_t tmp;
        asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 offen;\n\t"
                     : "=v"(tmp)
                     : "v"(src_thread_addr_offset), "s"(src_wave_buffer_resource)
                     : "memory");
        return bit_cast<rtn_type>(tmp);
    }
}

template <typename T, index_t N>
CK_TILE_DEVICE thread_buffer<T, N>
amd_buffer_load_impl_inline_asm(int32x4_t src_wave_buffer_resource, int32x2_t offset)
{
    constexpr index_t bytes = sizeof(T) * N;
    static_assert(bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8 || bytes == 16,
                  "wrong! not supported by buffer_load instruction");

    using rtn_type = thread_buffer<T, N>;

    if constexpr(bytes == 1)
    {
        uint8_t tmp;
        asm volatile("buffer_load_ubyte %0, %1, %2, 0 idxen offen;\n\t"
                     : "=v"(tmp)
                     : "v"(offset), "s"(src_wave_buffer_resource)
                     : "memory");
        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(bytes == 2)
    {
        uint16_t tmp;
        asm volatile("buffer_load_ushort %0, %1, %2, 0 idxen offen;\n\t"
                     : "=v"(tmp)
                     : "v"(offset), "s"(src_wave_buffer_resource)
                     : "memory");
        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(bytes == 4)
    {
        float tmp;
        asm volatile("buffer_load_dword %0, %1, %2, 0 idxen offen;\n\t"
                     : "=v"(tmp)
                     : "v"(offset), "s"(src_wave_buffer_resource)
                     : "memory");
        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(bytes == 8)
    {
        fp32x2_t tmp;
        asm volatile("buffer_load_dwordx2 %0, %1, %2, 0 idxen offen;\n\t"
                     : "=v"(tmp)
                     : "v"(offset), "s"(src_wave_buffer_resource)
                     : "memory");
        return bit_cast<rtn_type>(tmp);
    }
    else if constexpr(bytes == 16)
    {
        fp32x4_t tmp;
        asm volatile("buffer_load_dwordx4 %0, %1, %2, 0 idxen offen;\n\t"
                     : "=v"(tmp)
                     : "v"(offset), "s"(src_wave_buffer_resource)
                     : "memory");
        return bit_cast<rtn_type>(tmp);
    }
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true,
          bool pre_nop                        = false>
CK_TILE_DEVICE void amd_buffer_load_raw_impl(thread_buffer<T, N>& dst,
                                             int32x4_t src_wave_buffer_resource,
                                             index_t src_thread_addr_offset,
                                             index_t src_wave_addr_offset,
                                             index_t src_linear_addr_offset,
                                             index_t flag           = 0,
                                             bool_constant<pre_nop> = {})
{
    constexpr index_t bytes = sizeof(T) * N;
    static_assert(bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8 || bytes == 16,
                  "wrong! not supported by buffer_load instruction");

    using type = thread_buffer<T, N>;
    if constexpr(oob_conditional_check)
    {
        buffer_load_if<sizeof(type), pre_nop>{}(dst,
                                                src_wave_buffer_resource,
                                                src_thread_addr_offset,
                                                src_wave_addr_offset,
                                                src_linear_addr_offset,
                                                flag,
                                                bool_constant<pre_nop>{});
    }
    else
    {
        buffer_load<sizeof(type), pre_nop>{}(dst,
                                             src_wave_buffer_resource,
                                             src_thread_addr_offset,
                                             src_wave_addr_offset,
                                             src_linear_addr_offset,
                                             flag,
                                             bool_constant<pre_nop>{});
    }
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool pre_nop                        = false>
CK_TILE_DEVICE void amd_async_buffer_load_impl(T* smem,
                                               int32x4_t src_wave_buffer_resource,
                                               index_t src_thread_addr_offset,
                                               index_t src_wave_addr_offset,
                                               index_t src_immediate_addr_offset = 0,
                                               bool_constant<pre_nop>            = {})
{
    static_assert(sizeof(T) * N == 4, "wrong! not implemented vector size");

    async_buffer_load_dword_v(smem,
                              src_wave_buffer_resource,
                              src_thread_addr_offset,
                              src_wave_addr_offset,
                              src_immediate_addr_offset,
                              0,
                              bool_constant<pre_nop>{});
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE void amd_async_buffer_load(CK_TILE_LDS_ADDR T* smem,
                                          int32x4_t src_wave_buffer_resource,
                                          index_t src_thread_addr_offset,
                                          index_t src_wave_addr_offset,
                                          index_t src_immediate_addr_offset    = 0,
                                          index_t flag                         = 0,
                                          bool_constant<oob_conditional_check> = {})
{
    static constexpr index_t bytes = sizeof(T) * N;

    static_assert(bytes == 4 || bytes == 8 || bytes == 16, "wrong! not implemented vector size");

    index_t v_offset = src_thread_addr_offset;

    if constexpr(oob_conditional_check)
    {
        v_offset = flag ? v_offset : src_wave_buffer_resource[2];
    }

    llvm_amdgcn_raw_buffer_load_lds(src_wave_buffer_resource,
                                    smem,
                                    bytes,
                                    v_offset,
                                    src_wave_addr_offset,
                                    src_immediate_addr_offset,
                                    static_cast<index_t>(coherence));
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE void amd_async_buffer_load_inline_asm(CK_TILE_LDS_ADDR T* smem,
                                                     int32x4_t src_wave_buffer_resource,
                                                     index_t src_thread_addr_offset,
                                                     index_t /*src_wave_addr_offset*/,
                                                     index_t src_immediate_addr_offset    = 0,
                                                     index_t flag                         = 0,
                                                     bool_constant<oob_conditional_check> = {})
{
    static constexpr index_t bytes = sizeof(T) * N;

    static_assert(bytes == 4 || bytes == 8 || bytes == 16, "wrong! not implemented vector size");

    auto const wave_smem_ptr = __builtin_amdgcn_readfirstlane((reinterpret_cast<uintptr_t>(smem)));

#if defined(__gfx938__) || defined(__gfx946__) || defined(__gfx936__)
    if constexpr(oob_conditional_check)
    {
        index_t v_offset = flag ? src_thread_addr_offset : src_wave_buffer_resource[2];

        if constexpr(bytes == 4)
        {
            asm volatile(
                "s_mov_b32 m0, %0; \n\t"
                "buffer_load_dword %1, %2, 0 offen offset:%3 lds;\n\t" ::"s"(wave_smem_ptr),
                "v"(v_offset),
                "s"(src_wave_buffer_resource),
                "n"(src_immediate_addr_offset)
                : "memory");
        }
        else if constexpr(bytes == 8)
        {
            asm volatile(
                "s_mov_b32 m0, %0; \n\t"
                "buffer_load_dwordx2 %1, %2, 0 offen offset:%3 lds;\n\t" ::"s"(wave_smem_ptr),
                "v"(v_offset),
                "s"(src_wave_buffer_resource),
                "n"(src_immediate_addr_offset)
                : "memory");
        }
        else
        {
            asm volatile(
                "s_mov_b32 m0, %0; \n\t"
                "buffer_load_dwordx4 %1, %2, 0 offen offset:%3 lds;\n\t" ::"s"(wave_smem_ptr),
                "v"(v_offset),
                "s"(src_wave_buffer_resource),
                "n"(src_immediate_addr_offset)
                : "memory");
        }
    }
    else
    {
        if constexpr(bytes == 4)
        {
            asm volatile(
                "s_mov_b32 m0, %0; \n\t"
                "buffer_load_dword %1, %2, 0 offen offset:%3 lds;\n\t" ::"s"(wave_smem_ptr),
                "v"(src_thread_addr_offset),
                "s"(src_wave_buffer_resource),
                "n"(src_immediate_addr_offset)
                : "memory");
        }
        else if constexpr(bytes == 8)
        {
            asm volatile(
                "s_mov_b32 m0, %0; \n\t"
                "buffer_load_dwordx2 %1, %2, 0 offen offset:%3 lds;\n\t" ::"s"(wave_smem_ptr),
                "v"(src_thread_addr_offset),
                "s"(src_wave_buffer_resource),
                "n"(src_immediate_addr_offset)
                : "memory");
        }
        else
        {
            asm volatile(
                "s_mov_b32 m0, %0; \n\t"
                "buffer_load_dwordx4 %1, %2, 0 offen offset:%3 lds;\n\t" ::"s"(wave_smem_ptr),
                "v"(src_thread_addr_offset),
                "s"(src_wave_buffer_resource),
                "n"(src_immediate_addr_offset)
                : "memory");
        }
    }
#else
    (void)wave_smem_ptr;
    (void)src_thread_addr_offset;
    (void)src_immediate_addr_offset;
    (void)flag;
    (void)src_wave_buffer_resource;
    (void)oob_conditional_check;
#endif
}

template <index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default>
CK_TILE_DEVICE void amd_buffer_store_impl_with_bytes(const thread_buffer<int8_t, N> src_thread_data,
                                                     int32x4_t dst_wave_buffer_resource,
                                                     index_t dst_thread_addr_offset,
                                                     index_t dst_wave_addr_offset)
{
    static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32 || N == 64,
                  "wrong! not implemented");

    if constexpr(N == 1)
    {
        llvm_amdgcn_raw_buffer_store_i8(bit_cast<int8_t>(src_thread_data),
                                        dst_wave_buffer_resource,
                                        dst_thread_addr_offset,
                                        dst_wave_addr_offset,
                                        static_cast<index_t>(coherence));
    }
    else if constexpr(N == 2)
    {

        llvm_amdgcn_raw_buffer_store_i16(bit_cast<int16_t>(src_thread_data),
                                         dst_wave_buffer_resource,
                                         dst_thread_addr_offset,
                                         dst_wave_addr_offset,
                                         static_cast<index_t>(coherence));
    }
    else if constexpr(N == 4)
    {
        llvm_amdgcn_raw_buffer_store_i32(bit_cast<int32_t>(src_thread_data),
                                         dst_wave_buffer_resource,
                                         dst_thread_addr_offset,
                                         dst_wave_addr_offset,
                                         static_cast<index_t>(coherence));
    }
    else if constexpr(N == 8)
    {
        llvm_amdgcn_raw_buffer_store_i32x2(bit_cast<int32x2_t>(src_thread_data),
                                           dst_wave_buffer_resource,
                                           dst_thread_addr_offset,
                                           dst_wave_addr_offset,
                                           static_cast<index_t>(coherence));
    }
    else if constexpr(N == 16)
    {
        llvm_amdgcn_raw_buffer_store_i32x4(bit_cast<int32x4_t>(src_thread_data),
                                           dst_wave_buffer_resource,
                                           dst_thread_addr_offset,
                                           dst_wave_addr_offset,
                                           static_cast<index_t>(coherence));
    }
    else if constexpr(N == 32)
    {
        llvm_amdgcn_raw_buffer_store_i32x4(
            src_thread_data.template get_as<int32x4_t>()[number<0>{}],
            dst_wave_buffer_resource,
            dst_thread_addr_offset,
            dst_wave_addr_offset,
            static_cast<index_t>(coherence));

        llvm_amdgcn_raw_buffer_store_i32x4(
            src_thread_data.template get_as<int32x4_t>()[number<1>{}],
            dst_wave_buffer_resource,
            dst_thread_addr_offset,
            dst_wave_addr_offset + sizeof(int32_t) * 4,
            static_cast<index_t>(coherence));
    }
    else if constexpr(N == 64)
    {
        llvm_amdgcn_raw_buffer_store_i32x4(
            src_thread_data.template get_as<int32x4_t>()[number<0>{}],
            dst_wave_buffer_resource,
            dst_thread_addr_offset,
            dst_wave_addr_offset,
            static_cast<index_t>(coherence));

        llvm_amdgcn_raw_buffer_store_i32x4(
            src_thread_data.template get_as<int32x4_t>()[number<1>{}],
            dst_wave_buffer_resource,
            dst_thread_addr_offset,
            dst_wave_addr_offset + sizeof(int32_t) * 4,
            static_cast<index_t>(coherence));

        llvm_amdgcn_raw_buffer_store_i32x4(
            src_thread_data.template get_as<int32x4_t>()[number<2>{}],
            dst_wave_buffer_resource,
            dst_thread_addr_offset,
            dst_wave_addr_offset + sizeof(int32_t) * 8,
            static_cast<index_t>(coherence));

        llvm_amdgcn_raw_buffer_store_i32x4(
            src_thread_data.template get_as<int32x4_t>()[number<3>{}],
            dst_wave_buffer_resource,
            dst_thread_addr_offset,
            dst_wave_addr_offset + sizeof(int32_t) * 12,
            static_cast<index_t>(coherence));
    }
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default>
CK_TILE_DEVICE void amd_buffer_store_impl(const thread_buffer<T, N> src_thread_data,
                                          int32x4_t dst_wave_buffer_resource,
                                          index_t dst_thread_addr_offset,
                                          index_t dst_wave_addr_offset)
{
    static_assert(
        (std::is_same<T, double>::value && (N == 1 || N == 2 || N == 4 || N == 8)) ||
            (std::is_same<T, float>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, fp16_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, bf16_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, int32_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, fp8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, bf8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, int8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, uint16_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (std::is_same<T, uint8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)),
        "wrong! not implemented");

    if constexpr(std::is_same<T, float>::value) // fp32
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_fp32(bit_cast<float>(src_thread_data),
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              dst_wave_addr_offset,
                                              static_cast<index_t>(coherence));
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_fp32x2(bit_cast<fp32x2_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_store_fp32x4(bit_cast<fp32x4_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
        }
        else if constexpr(N == 8)
        {
            llvm_amdgcn_raw_buffer_store_fp32x4(
                src_thread_data.template get_as<fp32x4_t>()[number<0>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset,
                static_cast<index_t>(coherence));
            llvm_amdgcn_raw_buffer_store_fp32x4(
                src_thread_data.template get_as<fp32x4_t>()[number<1>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + 4 * sizeof(float),
                static_cast<index_t>(coherence));
        }
    }
    else if constexpr(std::is_same<T, fp16_t>::value) // fp16
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_fp16(bit_cast<_Float16>(src_thread_data),
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              dst_wave_addr_offset,
                                              static_cast<index_t>(coherence));
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_fp16x2(bit_cast<fp16x2_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_store_fp16x4(bit_cast<fp16x4_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
        }
        else if constexpr(N == 8)
        {
#if 0
            thread_buffer<fp16_t, 8> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_store_fp16x4(tmp.template get_as<fp16x4_t>()[number<0>{}],
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));

            llvm_amdgcn_raw_buffer_store_fp16x4(tmp.template get_as<fp16x4_t>()[number<1>{}],
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset + 4 * sizeof(fp16_t),
                                                static_cast<index_t>(coherence));
#else
            llvm_amdgcn_raw_buffer_store_fp32x4(bit_cast<fp32x4_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
#endif
        }
    }
    else if constexpr(std::is_same<T, bf16_t>::value) // bf16
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_i16(bit_cast<int16_t>(src_thread_data),
                                             dst_wave_buffer_resource,
                                             dst_thread_addr_offset,
                                             dst_wave_addr_offset,
                                             static_cast<index_t>(coherence));
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_i16x2(bit_cast<int16x2_t>(src_thread_data),
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset,
                                               static_cast<index_t>(coherence));
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_store_i16x4(bit_cast<int16x4_t>(src_thread_data),
                                               dst_wave_buffer_resource,
                                               dst_thread_addr_offset,
                                               dst_wave_addr_offset,
                                               static_cast<index_t>(coherence));
        }
        else if constexpr(N == 8)
        {
            llvm_amdgcn_raw_buffer_store_i16x4(
                src_thread_data.template get_as<int16x4_t>()[number<0>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset,
                static_cast<index_t>(coherence));

            llvm_amdgcn_raw_buffer_store_i16x4(
                src_thread_data.template get_as<int16x4_t>()[number<1>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + 4 * sizeof(bf16_t),
                static_cast<index_t>(coherence));
        }
    }
    else if constexpr(std::is_same<T, uint16_t>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_store_ui16(bit_cast<uint16_t>(src_thread_data),
                                              dst_wave_buffer_resource,
                                              dst_thread_addr_offset,
                                              dst_wave_addr_offset,
                                              static_cast<index_t>(coherence));
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_store_ui16x2(bit_cast<uint16x2_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_store_ui16x4(bit_cast<uint16x4_t>(src_thread_data),
                                                dst_wave_buffer_resource,
                                                dst_thread_addr_offset,
                                                dst_wave_addr_offset,
                                                static_cast<index_t>(coherence));
        }
        else if constexpr(N == 8)
        {
            llvm_amdgcn_raw_buffer_store_ui16x4(
                src_thread_data.template get_as<uint16x4_t>()[number<0>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset,
                static_cast<index_t>(coherence));

            llvm_amdgcn_raw_buffer_store_ui16x4(
                src_thread_data.template get_as<uint16x4_t>()[number<1>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + 4 * sizeof(uint16_t),
                static_cast<index_t>(coherence));
        }
    }
    else
    {
        using r_t = thread_buffer<int8_t, sizeof(T) * N>;

        amd_buffer_store_impl_with_bytes<sizeof(T) * N, coherence>(bit_cast<r_t>(src_thread_data),
                                                                   dst_wave_buffer_resource,
                                                                   dst_thread_addr_offset,
                                                                   dst_wave_addr_offset);
    }
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE void amd_buffer_store_raw_impl(const thread_buffer<T, N>& dst_thread_data,
                                              int32x4_t dst_wave_buffer_resource,
                                              index_t dst_thread_addr_offset,
                                              index_t dst_wave_addr_offset,
                                              index_t dst_linear_addr_offset,
                                              index_t is_valid_element = 1)
{
    constexpr index_t bytes = sizeof(T) * N;
    static_assert(bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8 || bytes == 16,
                  "wrong! not supported by buffer_store instruction");

    using type = thread_buffer<T, N>;
    if constexpr(oob_conditional_check)
    {
        buffer_store_if<sizeof(type)>{}(dst_thread_data,
                                        dst_wave_buffer_resource,
                                        dst_thread_addr_offset,
                                        dst_wave_addr_offset,
                                        dst_linear_addr_offset,
                                        is_valid_element);
    }
    else
    {
        buffer_store<sizeof(type)>{}(dst_thread_data,
                                     dst_wave_buffer_resource,
                                     dst_thread_addr_offset,
                                     dst_wave_addr_offset,
                                     dst_linear_addr_offset);
    }
}

template <typename T, index_t N>
CK_TILE_DEVICE void amd_buffer_atomic_add_impl(const thread_buffer<T, N>& src_thread_data,
                                               int32x4_t dst_wave_buffer_resource,
                                               index_t dst_thread_addr_offset,
                                               index_t dst_wave_addr_offset)
{
    static_assert((std::is_same<T, float>::value && (N == 1 || N == 2 || N == 4)) ||
                      (std::is_same<T, fp16_t>::value && (N == 2 || N == 4 || N == 8)) ||
                      (std::is_same<T, int32_t>::value && (N == 1 || N == 2 || N == 4)),
                  "wrong! not implemented");

    if constexpr(std::is_same<T, float>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_atomic_add_fp32(bit_cast<float>(src_thread_data),
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_atomic_add_fp32(
                src_thread_data.template get_as<float>()[number<0>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset,
                0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(
                src_thread_data.template get_as<float>()[number<1>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + sizeof(float),
                0);
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_atomic_add_fp32(
                src_thread_data.template get_as<float>()[number<0>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset,
                0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(
                src_thread_data.template get_as<float>()[number<1>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + sizeof(float),
                0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(
                src_thread_data.template get_as<float>()[number<2>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + 2 * sizeof(float),
                0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(
                src_thread_data.template get_as<float>()[number<3>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + 3 * sizeof(float),
                0);
        }
    }
    else if constexpr(std::is_same<T, fp16_t>::value)
    {
#if defined(__gfx928__) || defined(__gfx936__)
        static_assert(false, "not support buffer_atomic_pk_add");
#else
        if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_atomic_add_fp16x2(bit_cast<fp16x2_t>(src_thread_data),
                                                     dst_wave_buffer_resource,
                                                     dst_thread_addr_offset,
                                                     dst_wave_addr_offset,
                                                     0);
        }
        else if constexpr(N == 4)
        {
            static_for<0, 2, 1>{}([&](auto i) {
                llvm_amdgcn_raw_buffer_atomic_add_fp16x2(
                    src_thread_data.template get_as<fp16x2_t>()[i],
                    dst_wave_buffer_resource,
                    dst_thread_addr_offset,
                    dst_wave_addr_offset + i * sizeof(fp16x2_t),
                    0);
            });
        }
        else if constexpr(N == 8)
        {
            static_for<0, 4, 1>{}([&](auto i) {
                llvm_amdgcn_raw_buffer_atomic_add_fp16x2(
                    src_thread_data.template get_as<fp16x2_t>()[i],
                    dst_wave_buffer_resource,
                    dst_thread_addr_offset,
                    dst_wave_addr_offset + i * sizeof(fp16x2_t),
                    0);
            });
        }
#endif
    }
    else if constexpr(std::is_same<T, int32_t>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_atomic_add_i32(bit_cast<int32_t>(src_thread_data),
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset,
                                                  0);
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_atomic_add_i32(
                src_thread_data.template get_as<int32_t>()[number<0>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset,
                0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(
                src_thread_data.template get_as<int32_t>()[number<1>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + sizeof(int32_t),
                0);
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_atomic_add_i32(
                src_thread_data.template get_as<int32_t>()[number<0>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset,
                0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(
                src_thread_data.template get_as<int32_t>()[number<1>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + sizeof(int32_t),
                0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(
                src_thread_data.template get_as<int32_t>()[number<2>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + 2 * sizeof(int32_t),
                0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(
                src_thread_data.template get_as<int32_t>()[number<3>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + 3 * sizeof(int32_t),
                0);
        }
    }
}

template <typename T, index_t N>
CK_TILE_DEVICE void amd_buffer_atomic_max_impl(const thread_buffer<T, N> src_thread_data,
                                               int32x4_t dst_wave_buffer_resource,
                                               index_t dst_thread_addr_offset,
                                               index_t dst_wave_addr_offset)
{
    static_assert((std::is_same<T, double>::value && (N == 1 || N == 2 || N == 4)),
                  "wrong! not implemented");
    if constexpr(std::is_same<T, double>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_atomic_max_fp64(bit_cast<double>(src_thread_data),
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);
        }
        else if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_atomic_max_fp64(
                src_thread_data.template get_as<double>()[number<0>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset,
                0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(
                src_thread_data.template get_as<double>()[number<1>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + sizeof(double),
                0);
        }
        else if constexpr(N == 4)
        {
            llvm_amdgcn_raw_buffer_atomic_max_fp64(
                src_thread_data.template get_as<double>()[number<0>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset,
                0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(
                src_thread_data.template get_as<double>()[number<1>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + sizeof(double),
                0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(
                src_thread_data.template get_as<double>()[number<2>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + 2 * sizeof(double),
                0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(
                src_thread_data.template get_as<double>()[number<3>{}],
                dst_wave_buffer_resource,
                dst_thread_addr_offset,
                dst_wave_addr_offset + 3 * sizeof(double),
                0);
        }
    }
}

// buffer_load requires:
//   1) p_src_wave must point to global memory space
//   2) p_src_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
//   oob_conditional_check : dynamic check if out-of-bound
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE thread_buffer<T, N>
amd_buffer_load_invalid_element_return_zero(const T* p_src_wave,
                                            index_t src_thread_element_offset,
                                            bool src_thread_element_valid,
                                            index_t src_element_space_size)
{
    const int32x4_t src_wave_buffer_resource =
        make_wave_buffer_resource(p_src_wave, src_element_space_size * sizeof(T));

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);

#if CK_TILE_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
    uint32_t src_addr_shift = [&]() {
        if constexpr(oob_conditional_check)
            return src_thread_element_valid ? 0 : 0x80000000;
        else
            return 0;
    }();
    return amd_buffer_load_impl<T, N, coherence>(
        src_wave_buffer_resource, src_addr_shift + src_thread_addr_offset, 0);
#else
    thread_buffer<T, N> tmp =
        amd_buffer_load_impl<T, N, coherence>(src_wave_buffer_resource, src_thread_addr_offset, 0);
    if constexpr(oob_conditional_check)
        return src_thread_element_valid ? tmp : thread_buffer<T, N>{numeric<T>::zero()};
    else
        return tmp;
#endif
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE thread_buffer<T, N>
amd_buffer_load_inline_asm_invalid_element_return_zero(const T* p_src_wave,
                                                       index_t src_thread_element_offset,
                                                       bool src_thread_element_valid,
                                                       index_t src_element_space_size)
{
    const int32x4_t src_wave_buffer_resource =
        make_wave_buffer_resource(p_src_wave, src_element_space_size * sizeof(T));

    if constexpr(oob_conditional_check)
    {
        index_t src_thread_addr_offset = src_thread_element_valid
                                             ? src_thread_element_offset * sizeof(T)
                                             : src_wave_buffer_resource[2];

        return amd_buffer_load_impl_inline_asm<T, N>(
            src_wave_buffer_resource, src_thread_addr_offset, 0);
    }
    else
    {
        index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);

        return amd_buffer_load_impl_inline_asm<T, N>(
            src_wave_buffer_resource, src_thread_addr_offset, 0);
    }
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE thread_buffer<T, N>
amd_buffer_load_inline_asm_invalid_element_return_zero(const T* p_src_wave,
                                                       index_t src_thread_element_offset,
                                                       bool src_thread_element_valid,
                                                       index_t src_element_space_size,
                                                       index_t const_stride)
{
    const int32x4_t src_wave_buffer_resource =
        make_wave_buffer_resource(p_src_wave, src_element_space_size * sizeof(T), const_stride);

    const index_t voffset = src_thread_element_offset * sizeof(T);

    if constexpr(oob_conditional_check)
    {
        int32x2_t offset = src_thread_element_valid
                               ? int32x2_t{voffset >> int(__builtin_log2f(const_stride)),
                                           voffset & (const_stride - 1)}
                               : int32x2_t{src_wave_buffer_resource[2], 0};

        return amd_buffer_load_impl_inline_asm<T, N>(src_wave_buffer_resource, offset);
    }
    else
    {
        return amd_buffer_load_impl_inline_asm<T, N>(
            src_wave_buffer_resource,
            int32x2_t{voffset >> int(__builtin_log2f(const_stride)), voffset & (const_stride - 1)});
    }
}

// buffer_load requires:
//   1) p_src_wave must point to global memory space
//   2) p_src_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE thread_buffer<T, N>
amd_buffer_load_invalid_element_return_customized_value(const T* p_src_wave,
                                                        index_t src_thread_element_offset,
                                                        bool src_thread_element_valid,
                                                        index_t src_element_space_size,
                                                        T customized_value)
{
    const int32x4_t src_wave_buffer_resource =
        make_wave_buffer_resource(p_src_wave, src_element_space_size * sizeof(T));

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);

    thread_buffer<T, N> tmp =
        amd_buffer_load_impl<T, N, coherence>(src_wave_buffer_resource, src_thread_addr_offset, 0);

    if constexpr(oob_conditional_check)
        return src_thread_element_valid ? tmp : thread_buffer<T, N>{customized_value};
    else
        return tmp;
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true,
          bool pre_nop                        = false>
CK_TILE_DEVICE void amd_buffer_load_raw(thread_buffer<T, N>& dst,
                                        const T* p_src_wave,
                                        index_t src_thread_element_offset,
                                        index_t src_linear_element_offset,
                                        index_t src_element_space_size,
                                        index_t is_valid_element = 0,
                                        bool_constant<pre_nop>   = {})
{
    const int32x4_t src_wave_buffer_resource =
        make_wave_buffer_resource(p_src_wave, src_element_space_size * sizeof(T));

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);
    index_t src_linear_addr_offset = src_linear_element_offset * sizeof(T);

    amd_buffer_load_raw_impl<T, N, coherence, oob_conditional_check, pre_nop>(
        dst,
        src_wave_buffer_resource,
        src_thread_addr_offset,
        0,
        src_linear_addr_offset,
        is_valid_element,
        bool_constant<pre_nop>{});
}

// This version support buffer resource as input arg
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true,
          bool pre_nop                        = false>
CK_TILE_DEVICE void amd_buffer_load_raw(thread_buffer<T, N>& dst,
                                        const int32x4_t src_wave_buffer_resource,
                                        index_t src_thread_element_offset,
                                        index_t src_linear_element_offset,
                                        index_t is_valid_element = 0,
                                        bool_constant<pre_nop>   = {})
{
    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);
    index_t src_linear_addr_offset = src_linear_element_offset * sizeof(T);

    amd_buffer_load_raw_impl<T, N, coherence, oob_conditional_check, pre_nop>(
        dst,
        src_wave_buffer_resource,
        src_thread_addr_offset,
        0,
        src_linear_addr_offset,
        is_valid_element,
        bool_constant<pre_nop>{});
}

// unfortunately async copy can not make sure invalid data is zero inside LDS
// ... unless people manually write zero to LDS at the proper address.
// so not support invalid_element check for now.
// buffer_load OOB still working.
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool pre_nop                        = false>
CK_TILE_DEVICE void amd_async_buffer_load_with_oob_raw(T* smem,
                                                       const T* p_src_wave,
                                                       index_t src_thread_element_offset,
                                                       index_t src_linear_element_offset,
                                                       index_t src_element_space_size,
                                                       bool_constant<pre_nop> = {})
{
    const int32x4_t src_wave_buffer_resource =
        make_wave_buffer_resource(p_src_wave, src_element_space_size * sizeof(T));

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);
    index_t src_linear_addr_offset = src_linear_element_offset * sizeof(T);

    amd_async_buffer_load_impl<T, N, coherence>(smem,
                                                src_wave_buffer_resource,
                                                src_thread_addr_offset,
                                                0,
                                                src_linear_addr_offset,
                                                bool_constant<pre_nop>{});
}

// This version support buffer resource as input arg
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool pre_nop                        = false>
CK_TILE_DEVICE void amd_async_buffer_load_with_oob_raw(T* smem,
                                                       const int32x4_t src_wave_buffer_resource,
                                                       index_t src_thread_element_offset,
                                                       index_t src_linear_element_offset,
                                                       bool_constant<pre_nop> = {})
{
    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);
    index_t src_linear_addr_offset = src_linear_element_offset * sizeof(T);

    amd_async_buffer_load_impl<T, N, coherence>(smem,
                                                src_wave_buffer_resource,
                                                src_thread_addr_offset,
                                                0,
                                                src_linear_addr_offset,
                                                bool_constant<pre_nop>{});
}

// This version support buffer resource as input arg
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = false>
CK_TILE_DEVICE void amd_async_buffer_load_with_oob(CK_TILE_LDS_ADDR T* smem,
                                                   const int32x4_t src_wave_buffer_resource,
                                                   index_t src_thread_element_offset,
                                                   index_t src_linear_element_offset,
                                                   bool is_valid_element,
                                                   bool_constant<oob_conditional_check> = {})
{
    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);
    index_t src_linear_addr_offset = src_linear_element_offset * sizeof(T);

    amd_async_buffer_load<T, N, coherence>(smem,
                                           src_wave_buffer_resource,
                                           src_thread_addr_offset,
                                           0,
                                           src_linear_addr_offset,
                                           is_valid_element,
                                           bool_constant<oob_conditional_check>{});
}

// This version support buffer resource as input arg
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = false>
CK_TILE_DEVICE void
amd_async_buffer_load_inline_asm_with_oob(CK_TILE_LDS_ADDR T* smem,
                                          const int32x4_t src_wave_buffer_resource,
                                          index_t src_thread_element_offset,
                                          index_t src_linear_element_offset,
                                          bool is_valid_element,
                                          bool_constant<oob_conditional_check> = {})
{
    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);
    index_t src_linear_addr_offset = src_linear_element_offset * sizeof(T);

    amd_async_buffer_load_inline_asm<T, N, coherence>(smem,
                                                      src_wave_buffer_resource,
                                                      src_thread_addr_offset,
                                                      0,
                                                      src_linear_addr_offset,
                                                      is_valid_element,
                                                      bool_constant<oob_conditional_check>{});
}

// buffer_store requires:
//   1) p_dst_wave must point to global memory
//   2) p_dst_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE void amd_buffer_store(const thread_buffer<T, N>& src_thread_data,
                                     T* p_dst_wave,
                                     const index_t dst_thread_element_offset,
                                     const bool dst_thread_element_valid,
                                     const index_t dst_element_space_size)
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size * sizeof(T));

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);

#if CK_TILE_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = [&]() {
        if constexpr(oob_conditional_check)
            return dst_thread_element_valid ? 0 : 0x80000000;
        else
            return 0;
    }();
    amd_buffer_store_impl<T, N, coherence>(
        src_thread_data, dst_wave_buffer_resource, dst_addr_shift + dst_thread_addr_offset, 0);
#else
    if constexpr(oob_conditional_check)
    {
        if(dst_thread_element_valid)
        {
            amd_buffer_store_impl<T, N, coherence>(
                src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
        }
    }
    else
    {
        amd_buffer_store_impl<T, N, coherence>(
            src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
    }
#endif
}

template <typename T,
          index_t N,
          amd_buffer_coherence_enum coherence = amd_buffer_coherence_enum::coherence_default,
          bool oob_conditional_check          = true>
CK_TILE_DEVICE void amd_buffer_store_raw(const thread_buffer<T, N>& src_thread_data,
                                         T* p_dst_wave,
                                         const index_t dst_thread_element_offset,
                                         const index_t dst_linear_element_offset,
                                         const bool dst_thread_element_valid,
                                         const index_t dst_element_space_size)
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size * sizeof(T));

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);
    index_t dst_linear_addr_offset = dst_linear_element_offset * sizeof(T);

    amd_buffer_store_raw_impl<T, N, coherence, oob_conditional_check>(src_thread_data,
                                                                      dst_wave_buffer_resource,
                                                                      dst_thread_addr_offset,
                                                                      0,
                                                                      dst_linear_addr_offset,
                                                                      dst_thread_element_valid);
}

// buffer_atomic_add requires:
//   1) p_dst_wave must point to global memory
//   2) p_dst_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t N>
CK_TILE_DEVICE void amd_buffer_atomic_add(const thread_buffer<T, N>& src_thread_data,
                                          T* p_dst_wave,
                                          const index_t dst_thread_element_offset,
                                          const bool dst_thread_element_valid,
                                          const index_t dst_element_space_size)
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size * sizeof(T));

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);

#if CK_TILE_EXPERIMENTAL_USE_BUFFER_ATOMIC_ADD_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_element_valid ? 0 : 0x80000000;

    amd_buffer_atomic_add_impl<T, N>(
        src_thread_data, dst_wave_buffer_resource, dst_addr_shift + dst_thread_addr_offset, 0);
#else
    if(dst_thread_element_valid)
    {
        amd_buffer_atomic_add_impl<T, N>(
            src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
    }
#endif
}

// buffer_atomic_max requires:
//   1) p_dst_wave must point to global memory
//   2) p_dst_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t N>
CK_TILE_DEVICE void amd_buffer_atomic_max(const thread_buffer<T, N>& src_thread_data,
                                          T* p_dst_wave,
                                          const index_t dst_thread_element_offset,
                                          const bool dst_thread_element_valid,
                                          const index_t dst_element_space_size)
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size * sizeof(T));

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);

#if CK_TILE_EXPERIMENTAL_USE_BUFFER_ATOMIC_MAX_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_element_valid ? 0 : 0x80000000;

    amd_buffer_atomic_max_impl<T, N>(
        src_thread_data, dst_wave_buffer_resource, dst_addr_shift + dst_thread_addr_offset, 0);
#else
    if(dst_thread_element_valid)
    {
        amd_buffer_atomic_max_impl<T, N>(
            src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
    }
#endif
}

template <typename T, index_t NumElemsPerThread>
CK_TILE_DEVICE void amd_direct_load_global_to_lds(const T* global_base_ptr,
                                                  const index_t global_offset,
                                                  T* lds_base_ptr,
                                                  const index_t lds_offset,
                                                  const bool is_valid,
                                                  const index_t src_element_space_size)
{
    // Direct loads require that each thread reads and writes exactly a single DWORD.
    constexpr auto dword_bytes      = 4;
    constexpr auto bytes_per_thread = sizeof(T) * NumElemsPerThread;
    static_assert(bytes_per_thread == dword_bytes);

    const uint32_t* global_ptr =
        reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(global_base_ptr));
    const int32x4_t src_resource =
        make_wave_buffer_resource(global_ptr, src_element_space_size * sizeof(T));
    const index_t global_offset_bytes = is_valid ? global_offset * sizeof(T) : 0x80000000;

#if CK_TILE_USE_AMD_LDS_DIRECT_LOAD_INLINE_ASM
    T* lds_ptr = lds_base_ptr + lds_offset;
    auto const lds_ptr_sgpr =
        __builtin_amdgcn_readfirstlane((reinterpret_cast<uintptr_t>(lds_ptr)));
    asm volatile("s_mov_b32 m0, %0; \n\t"
                 "buffer_load_dword %1, %2, 0 offen lds;\n\t" ::"s"(lds_ptr_sgpr),
                 "v"(global_offset_bytes),
                 "s"(src_resource)
                 : "memory");
#else
    // LDS pointer must be attributed with the LDS address space.
    __attribute__((address_space(3))) uint32_t* lds_ptr =
        reinterpret_cast<__attribute__((address_space(3))) uint32_t*>(
            reinterpret_cast<uintptr_t>(lds_base_ptr + lds_offset));

    llvm_amdgcn_raw_buffer_load_lds(
        src_resource, lds_ptr, sizeof(uint32_t), global_offset_bytes, 0, 0, 0);
#endif
}

} // namespace ck_tile
