#pragma once

#include "ck_tile/core/arch/amd_buffer_addressing.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/container/multi_index.hpp"

namespace ck_tile {

union mls_addr_union
{
    struct
    {
        int32_t addr_lo;
        int32_t addr_hi;
    };

    uintptr_t addr;
};

CK_TILE_DEVICE void move_mls_addr_base(int32x4_t& mls_res, const index_t addr_byte_offset)
{
    mls_addr_union addr_union{{mls_res.x, mls_res.y}};

    addr_union.addr += addr_byte_offset;
    mls_res.x = addr_union.addr_lo;
    mls_res.y = addr_union.addr_hi;
}

struct __attribute__((packed)) mls_resource
{
    const void* ptr;

    uint32_t stride;

    union
    {
        struct
        {
            uint32_t m_filter : 8;
            uint32_t nm_filter : 8;
            uint32_t cache_swizzle_enable : 1;
            uint32_t mfmt : 2;
            uint32_t reserved : 13;
        };

        uint32_t DW3_DATA;
    } DW3_CONFIG_UNION;
};

template <index_t mfmt>
CK_TILE_DEVICE int32x4_t make_mls_resource(const void* ptr,
                                           const uint32_t stride, // in elements
                                           const index_t m_filter,
                                           const index_t nm_filter,
                                           number<mfmt>) // interleave
{
    mls_resource res{ptr, 0, {{0, 0, 0, 0, 0}}};

    res.stride = stride;
    // always enable cache swizzle
    res.DW3_CONFIG_UNION.m_filter             = m_filter;
    res.DW3_CONFIG_UNION.nm_filter            = nm_filter;
    res.DW3_CONFIG_UNION.cache_swizzle_enable = 1;
    res.DW3_CONFIG_UNION.mfmt                 = mfmt;

    int32x4_t r = __builtin_bit_cast(int32x4_t, res);
    r.x         = __builtin_amdgcn_readfirstlane(r.x);
    r.y         = __builtin_amdgcn_readfirstlane(r.y);
    r.z         = __builtin_amdgcn_readfirstlane(r.z);
    r.w         = __builtin_amdgcn_readfirstlane(r.w);
    return r;
}

namespace detail {

// clang-format off
template <index_t Interleave> struct mfmt_traits;
template <> struct mfmt_traits<1> { static constexpr auto value = number<0>{}; };
template <> struct mfmt_traits<2> { static constexpr auto value = number<1>{}; };
template <> struct mfmt_traits<4> { static constexpr auto value = number<2>{}; };
template <> struct mfmt_traits<8> { static constexpr auto value = number<3>{}; };
// clang-format on

template <index_t moffset, bool r, bool t, bool bps>
CK_TILE_DEVICE void hcu_mls_32x16_b16_asm_impl(uintptr_t smem,
                                               const int32x4_t& mls_res,
                                               number<moffset>,
                                               bool_constant<r>,
                                               bool_constant<t>,
                                               bool_constant<bps>)
{
    const auto soffset = __builtin_amdgcn_readfirstlane(smem);

    if constexpr(!bps)
    {
        if constexpr(!r && !t)
        {
            asm volatile("matrix_load_32x16_b16 %0, %1, moffset:%2, lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else if constexpr(!r && t)
        {
            asm volatile("matrix_load_32x16_b16 %0, %1, moffset:%2, t, lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else if constexpr(r && !t)
        {
            asm volatile("matrix_load_32x16_b16 %0, %1, moffset:%2, r, lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else
        {
            asm volatile("matrix_load_32x16_b16 %0, %1, moffset:%2, t r, lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
    }
    else
    {
        if constexpr(!r && !t)
        {
            asm volatile("matrix_load_32x16_b16 %0, %1, moffset:%2, bps lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else if constexpr(!r && t)
        {
            asm volatile("matrix_load_32x16_b16 %0, %1, moffset:%2, t, bps lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else if constexpr(r && !t)
        {
            asm volatile("matrix_load_32x16_b16 %0, %1, moffset:%2, r, bps lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else
        {
            asm volatile("matrix_load_32x16_b16 %0, %1, moffset:%2, t r, bps lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
    }
}

template <index_t moffset, bool r, bool t, bool bps>
CK_TILE_DEVICE void hcu_mls_64x16_b16_asm_impl(uintptr_t smem,
                                               const int32x4_t& mls_res,
                                               number<moffset>,
                                               bool_constant<r>,
                                               bool_constant<t>,
                                               bool_constant<bps>)
{
    const auto soffset = __builtin_amdgcn_readfirstlane(smem);

    if constexpr(!bps)
    {
        if constexpr(!r && !t)
        {
            asm volatile("matrix_load_64x16_b16 %0, %1, moffset:%2, lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else if constexpr(!r && t)
        {
            asm volatile("matrix_load_64x16_b16 %0, %1, moffset:%2, t, lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else if constexpr(r && !t)
        {
            asm volatile("matrix_load_64x16_b16 %0, %1, moffset:%2, r, lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else
        {
            asm volatile("matrix_load_64x16_b16 %0, %1, moffset:%2, t r, lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
    }
    else
    {
        if constexpr(!r && !t)
        {
            asm volatile("matrix_load_64x16_b16 %0, %1, moffset:%2, bps lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else if constexpr(!r && t)
        {
            asm volatile("matrix_load_64x16_b16 %0, %1, moffset:%2, t, bps lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else if constexpr(r && !t)
        {
            asm volatile("matrix_load_64x16_b16 %0, %1, moffset:%2, r, bps lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else
        {
            asm volatile("matrix_load_64x16_b16 %0, %1, moffset:%2, t r, bps lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
    }
}

template <index_t moffset, bool r, bool t, bool bps>
CK_TILE_DEVICE void hcu_mls_32x32_b16_asm_impl(uintptr_t smem,
                                               const int32x4_t& mls_res,
                                               number<moffset>,
                                               bool_constant<r>,
                                               bool_constant<t>,
                                               bool_constant<bps>)
{
    {
        const auto soffset = __builtin_amdgcn_readfirstlane(smem);

        if constexpr(!bps)
        {
            if constexpr(!r && !t)
            {
                asm volatile("matrix_load_32x32_b16 %0, %1, moffset:%2, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else if constexpr(!r && t)
            {
                asm volatile("matrix_load_32x32_b16 %0, %1, moffset:%2, t, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else if constexpr(r && !t)
            {
                asm volatile("matrix_load_32x32_b16 %0, %1, moffset:%2, r, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                asm volatile("matrix_load_32x32_b16 %0, %1, moffset:%2, t r, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
        }
        else
        {
            if constexpr(!r && !t)
            {
                asm volatile("matrix_load_32x32_b16 %0, %1, moffset:%2, bps lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else if constexpr(!r && t)
            {
                asm volatile("matrix_load_32x32_b16 %0, %1, moffset:%2, t, bps lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else if constexpr(r && !t)
            {
                asm volatile("matrix_load_32x32_b16 %0, %1, moffset:%2, r, bps lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                asm volatile("matrix_load_32x32_b16 %0, %1, moffset:%2, t r, bps lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
        }
    }
}

template <index_t moffset, bool r, bool t, bool bps>
CK_TILE_DEVICE void hcu_mls_64x16_b8_asm_impl(uintptr_t smem,
                                              const int32x4_t& mls_res,
                                              number<moffset>,
                                              bool_constant<r>,
                                              bool_constant<t>,
                                              bool_constant<bps>)
{
    const auto soffset = __builtin_amdgcn_readfirstlane(smem);

    if constexpr(!bps)
    {
        if constexpr(!r && !t)
        {
            asm volatile("matrix_load_64x16_b8 %0, %1, moffset:%2, lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else if constexpr(!r && t)
        {
            asm volatile("matrix_load_64x16_b8 %0, %1, moffset:%2, t, lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else if constexpr(r && !t)
        {
            asm volatile("matrix_load_64x16_b8 %0, %1, moffset:%2, r, lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else
        {
            asm volatile("matrix_load_64x16_b8 %0, %1, moffset:%2, t r, lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
    }
    else
    {
        if constexpr(!r && !t)
        {
            asm volatile("matrix_load_64x16_b8 %0, %1, moffset:%2, bps lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else if constexpr(!r && t)
        {
            asm volatile("matrix_load_64x16_b8 %0, %1, moffset:%2, t, bps lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else if constexpr(r && !t)
        {
            asm volatile("matrix_load_64x16_b8 %0, %1, moffset:%2, r, bps lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else
        {
            asm volatile("matrix_load_64x16_b8 %0, %1, moffset:%2, t r, bps lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
    }
}

template <index_t moffset, bool r, bool t, bool bps>
CK_TILE_DEVICE void hcu_mls_128x16_b8_asm_impl(uintptr_t smem,
                                               const int32x4_t& mls_res,
                                               number<moffset>,
                                               bool_constant<r>,
                                               bool_constant<t>,
                                               bool_constant<bps>)
{
    const auto soffset = __builtin_amdgcn_readfirstlane(smem);

    if constexpr(!bps)
    {
        if constexpr(!r && !t)
        {
            asm volatile("matrix_load_128x16_b8 %0, %1, moffset:%2, lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else if constexpr(!r && t)
        {
            asm volatile("matrix_load_128x16_b8 %0, %1, moffset:%2, t, lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else if constexpr(r && !t)
        {
            asm volatile("matrix_load_128x16_b8 %0, %1, moffset:%2, r, lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else
        {
            asm volatile("matrix_load_128x16_b8 %0, %1, moffset:%2, t r, lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
    }
    else
    {
        if constexpr(!r && !t)
        {
            asm volatile("matrix_load_128x16_b8 %0, %1, moffset:%2, bps lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else if constexpr(!r && t)
        {
            asm volatile("matrix_load_128x16_b8 %0, %1, moffset:%2, t, bps lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else if constexpr(r && !t)
        {
            asm volatile("matrix_load_128x16_b8 %0, %1, moffset:%2, r, bps lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else
        {
            asm volatile("matrix_load_128x16_b8 %0, %1, moffset:%2, t r, bps lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
    }
}

template <index_t moffset, bool r, bool t, bool bps>
CK_TILE_DEVICE void hcu_mls_64x32_b8_asm_impl(uintptr_t smem,
                                              const int32x4_t& mls_res,
                                              number<moffset>,
                                              bool_constant<r>,
                                              bool_constant<t>,
                                              bool_constant<bps>)
{
    const auto soffset = __builtin_amdgcn_readfirstlane(smem);

    if constexpr(!bps)
    {
        if constexpr(!r && !t)
        {
            asm volatile("matrix_load_64x32_b8 %0, %1, moffset:%2, lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else if constexpr(!r && t)
        {
            asm volatile("matrix_load_64x32_b8 %0, %1, moffset:%2, t, lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else if constexpr(r && !t)
        {
            asm volatile("matrix_load_64x32_b8 %0, %1, moffset:%2, r, lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else
        {
            asm volatile("matrix_load_64x32_b8 %0, %1, moffset:%2, t r, lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
    }
    else
    {
        if constexpr(!r && !t)
        {
            asm volatile("matrix_load_64x32_b8 %0, %1, moffset:%2, bps lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else if constexpr(!r && t)
        {
            asm volatile("matrix_load_64x32_b8 %0, %1, moffset:%2, t, bps lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else if constexpr(r && !t)
        {
            asm volatile("matrix_load_64x32_b8 %0, %1, moffset:%2, r, bps lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
        else
        {
            asm volatile("matrix_load_64x32_b8 %0, %1, moffset:%2, t r, bps lds; \n\t"
                         :
                         : "s"(mls_res), "s"(soffset), "n"(moffset)
                         : "memory");
        }
    }
}

template <index_t dtype_bytes, index_t m_length, index_t nm_length, bool t, index_t mfmt>
struct mls_instr;

template <bool t, index_t mfmt>
struct mls_instr<2, 32, 16, t, mfmt>
{
    static constexpr index_t kLdsConsecBytes = 1024;

    template <index_t moffset, bool r, bool bps>
    CK_TILE_DEVICE void operator()(uintptr_t smem,
                                   const int32x4_t& mls_res,
                                   number<moffset>,
                                   bool_constant<r>,
                                   bool_constant<bps>) const
    {
        hcu_mls_32x16_b16_asm_impl(smem,
                                   mls_res,
                                   number<moffset>{},
                                   bool_constant<r>{},
                                   bool_constant<t>{},
                                   bool_constant<bps>{});
    }
};

template <bool t, index_t mfmt>
struct mls_instr<2, 64, 16, t, mfmt>
{
    // FIXME: confirm LDS layout of gfx92a/gfx946
    static constexpr index_t kLdsConsecBytes = 2048;

    template <index_t moffset, bool r, bool bps>
    CK_TILE_DEVICE void operator()(uintptr_t smem,
                                   const int32x4_t& mls_res,
                                   number<moffset>,
                                   bool_constant<r>,
                                   bool_constant<bps>) const
    {
        hcu_mls_64x16_b16_asm_impl(smem,
                                   mls_res,
                                   number<moffset>{},
                                   bool_constant<r>{},
                                   bool_constant<t>{},
                                   bool_constant<bps>{});
    }
};

#if defined(__gfx938__)
template <index_t mfmt>
struct mls_instr<2, 64, 16, false, mfmt>
{
    static constexpr index_t kLdsConsecBytes = 2048;

    template <index_t moffset, bool r, bool bps>
    CK_TILE_DEVICE void operator()(uintptr_t smem,
                                   const int32x4_t& mls_res,
                                   number<moffset>,
                                   bool_constant<r>,
                                   bool_constant<bps>) const
    {
        hcu_mls_64x16_b16_asm_impl(smem,
                                   mls_res,
                                   number<moffset>{},
                                   bool_constant<r>{},
                                   bool_constant<false>{},
                                   bool_constant<bps>{});
    }
};

template <>
struct mls_instr<2, 64, 16, true, 1>
{
    static constexpr index_t kLdsConsecBytes = 1024;

    template <index_t moffset, bool r, bool bps>
    CK_TILE_DEVICE void operator()(uintptr_t smem,
                                   const int32x4_t& mls_res,
                                   number<moffset>,
                                   bool_constant<r>,
                                   bool_constant<bps>) const
    {
        hcu_mls_64x16_b16_asm_impl(smem,
                                   mls_res,
                                   number<moffset>{},
                                   bool_constant<r>{},
                                   bool_constant<true>{},
                                   bool_constant<bps>{});
    }
};
#endif

template <bool t, index_t mfmt>
struct mls_instr<2, 32, 32, t, mfmt>
{
    static constexpr index_t kLdsConsecBytes = 2048;

    template <index_t moffset, bool r, bool bps>
    CK_TILE_DEVICE void operator()(uintptr_t smem,
                                   const int32x4_t& mls_res,
                                   number<moffset>,
                                   bool_constant<r>,
                                   bool_constant<bps>) const
    {
        hcu_mls_32x32_b16_asm_impl(smem,
                                   mls_res,
                                   number<moffset>{},
                                   bool_constant<r>{},
                                   bool_constant<t>{},
                                   bool_constant<bps>{});
    }
};

template <bool t, index_t mfmt>
struct mls_instr<1, 64, 16, t, mfmt>
{
    static constexpr index_t kLdsConsecBytes = 1024;

    template <index_t moffset, bool r, bool bps>
    CK_TILE_DEVICE void operator()(uintptr_t smem,
                                   const int32x4_t& mls_res,
                                   number<moffset>,
                                   bool_constant<r>,
                                   bool_constant<bps>) const
    {
        hcu_mls_64x16_b8_asm_impl(smem,
                                  mls_res,
                                  number<moffset>{},
                                  bool_constant<r>{},
                                  bool_constant<t>{},
                                  bool_constant<bps>{});
    }
};

#if defined(__gfx938__)
template <>
struct mls_instr<1, 64, 16, false, 1>
{
    static constexpr index_t kLdsConsecBytes = 512;

    template <index_t moffset, bool r, bool bps>
    CK_TILE_DEVICE void operator()(uintptr_t smem,
                                   const int32x4_t& mls_res,
                                   number<moffset>,
                                   bool_constant<r>,
                                   bool_constant<bps>) const
    {
        hcu_mls_64x16_b8_asm_impl(smem,
                                  mls_res,
                                  number<moffset>{},
                                  bool_constant<r>{},
                                  bool_constant<false>{},
                                  bool_constant<bps>{});
    }
};

template <>
struct mls_instr<1, 64, 16, false, 2>
{
    static constexpr index_t kLdsConsecBytes = 512;

    template <index_t moffset, bool r, bool bps>
    CK_TILE_DEVICE void operator()(uintptr_t smem,
                                   const int32x4_t& mls_res,
                                   number<moffset>,
                                   bool_constant<r>,
                                   bool_constant<bps>) const
    {
        hcu_mls_64x16_b8_asm_impl(smem,
                                  mls_res,
                                  number<moffset>{},
                                  bool_constant<r>{},
                                  bool_constant<false>{},
                                  bool_constant<bps>{});
    }
};

template <>
struct mls_instr<1, 64, 16, true, 2>
{
    static constexpr index_t kLdsConsecBytes = 512;

    template <index_t moffset, bool r, bool bps>
    CK_TILE_DEVICE void operator()(uintptr_t smem,
                                   const int32x4_t& mls_res,
                                   number<moffset>,
                                   bool_constant<r>,
                                   bool_constant<bps>) const
    {
        hcu_mls_64x16_b8_asm_impl(smem,
                                  mls_res,
                                  number<moffset>{},
                                  bool_constant<r>{},
                                  bool_constant<true>{},
                                  bool_constant<bps>{});
    }
};
#endif

template <bool t, index_t mfmt>
struct mls_instr<1, 128, 16, t, mfmt>
{
    static constexpr index_t kLdsConsecBytes = 2048;

    template <index_t moffset, bool r, bool bps>
    CK_TILE_DEVICE void operator()(uintptr_t smem,
                                   const int32x4_t& mls_res,
                                   number<moffset>,
                                   bool_constant<r>,
                                   bool_constant<bps>) const
    {
        hcu_mls_128x16_b8_asm_impl(smem,
                                   mls_res,
                                   number<moffset>{},
                                   bool_constant<r>{},
                                   bool_constant<t>{},
                                   bool_constant<bps>{});
    }
};

#if defined(__gfx938__)
template <>
struct mls_instr<1, 128, 16, false, 0>
{
    static constexpr index_t kLdsConsecBytes = 1024;

    template <index_t moffset, bool r, bool bps>
    CK_TILE_DEVICE void operator()(uintptr_t smem,
                                   const int32x4_t& mls_res,
                                   number<moffset>,
                                   bool_constant<r>,
                                   bool_constant<bps>) const
    {
        hcu_mls_128x16_b8_asm_impl(smem,
                                   mls_res,
                                   number<moffset>{},
                                   bool_constant<r>{},
                                   bool_constant<false>{},
                                   bool_constant<bps>{});
    }
};

template <>
struct mls_instr<1, 128, 16, false, 1>
{
    static constexpr index_t kLdsConsecBytes = 512;

    template <index_t moffset, bool r, bool bps>
    CK_TILE_DEVICE void operator()(uintptr_t smem,
                                   const int32x4_t& mls_res,
                                   number<moffset>,
                                   bool_constant<r>,
                                   bool_constant<bps>) const
    {
        hcu_mls_128x16_b8_asm_impl(smem,
                                   mls_res,
                                   number<moffset>{},
                                   bool_constant<r>{},
                                   bool_constant<false>{},
                                   bool_constant<bps>{});
    }
};

template <>
struct mls_instr<1, 128, 16, false, 2>
{
    static constexpr index_t kLdsConsecBytes = 512;

    template <index_t moffset, bool r, bool bps>
    CK_TILE_DEVICE void operator()(uintptr_t smem,
                                   const int32x4_t& mls_res,
                                   number<moffset>,
                                   bool_constant<r>,
                                   bool_constant<bps>) const
    {
        hcu_mls_128x16_b8_asm_impl(smem,
                                   mls_res,
                                   number<moffset>{},
                                   bool_constant<r>{},
                                   bool_constant<false>{},
                                   bool_constant<bps>{});
    }
};

template <>
struct mls_instr<1, 128, 16, true, 1>
{
    static constexpr index_t kLdsConsecBytes = 1024;

    template <index_t moffset, bool r, bool bps>
    CK_TILE_DEVICE void operator()(uintptr_t smem,
                                   const int32x4_t& mls_res,
                                   number<moffset>,
                                   bool_constant<r>,
                                   bool_constant<bps>) const
    {
        hcu_mls_128x16_b8_asm_impl(smem,
                                   mls_res,
                                   number<moffset>{},
                                   bool_constant<r>{},
                                   bool_constant<true>{},
                                   bool_constant<bps>{});
    }
};

template <>
struct mls_instr<1, 128, 16, true, 2>
{
    static constexpr index_t kLdsConsecBytes = 512;

    template <index_t moffset, bool r, bool bps>
    CK_TILE_DEVICE void operator()(uintptr_t smem,
                                   const int32x4_t& mls_res,
                                   number<moffset>,
                                   bool_constant<r>,
                                   bool_constant<bps>) const
    {
        hcu_mls_128x16_b8_asm_impl(smem,
                                   mls_res,
                                   number<moffset>{},
                                   bool_constant<r>{},
                                   bool_constant<true>{},
                                   bool_constant<bps>{});
    }
};
#endif
} // namespace detail

template <typename T, index_t m_length, index_t nm_length, bool t, index_t mfmt>
CK_TILE_HOST_DEVICE constexpr auto
get_lds_consec_bytes(number<m_length>, number<nm_length>, bool_constant<t>, number<mfmt>)
{
    using mls_instr = detail::mls_instr<sizeof(T), m_length, nm_length, t, mfmt>;

    return mls_instr::kLdsConsecBytes;
}

template <typename T,
          index_t m_length,  // major length
          index_t nm_length, // non-major length
          bool t,
          index_t mfmt,
          index_t moffset,
          bool r,
          bool bps>
CK_TILE_DEVICE void hcu_async_matrix_load_asm_impl(CK_TILE_LDS_ADDR T* smem,
                                                   const int32x4_t& mls_res,
                                                   number<m_length>,
                                                   number<nm_length>,
                                                   bool_constant<t>,
                                                   number<mfmt>,
                                                   number<moffset>,
                                                   bool_constant<r>,
                                                   bool_constant<bps>)
{
#if defined(__gfx938__) || defined(__gfx92a__) || defined(__gfx946__)
    using mls_instr = detail::mls_instr<sizeof(T), m_length, nm_length, t, mfmt>;

    if constexpr(t)
    {
        mls_instr{}(reinterpret_cast<uintptr_t>(smem) | CK_TILE_MATRIX_LOAD_TRANS_EXTRA_CONFIG,
                    mls_res,
                    number<moffset>{},
                    bool_constant<r>{},
                    bool_constant<CK_TILE_MATRIX_LOAD_SUPPORT_BPS & bps>{});
    }
    else
    {
        mls_instr{}(reinterpret_cast<uintptr_t>(smem),
                    mls_res,
                    number<moffset>{},
                    bool_constant<r>{},
                    bool_constant<CK_TILE_MATRIX_LOAD_SUPPORT_BPS & bps>{});
    }
#else
    detail::swallow{smem, mls_res};
#endif
}

} // namespace ck_tile
