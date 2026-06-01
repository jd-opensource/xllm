#pragma once

#include "ck_tile/core/arch/amd_buffer_addressing.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/container/multi_index.hpp"

namespace ck_tile {

struct gfx938_mls_32x16_b16
{
    static constexpr auto TileShape = sequence<32, 16>{};

    template <index_t moffset, bool r, bool use_m0 = true, bool s_nop = true>
    CK_TILE_DEVICE static void
    load(const uintptr_t smem, const int32x4_t& mls_res, number<moffset>, bool_constant<r>)
    {
#if defined(__gfx938__)
        const auto soffset = static_cast<const uint32_t>(smem);

        if constexpr(!r)
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, %1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_32x16_b16 %0, m0, moffset:%2, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_32x16_b16 %0, %1, moffset:%2, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
                else
                {
                    asm volatile("matrix_load_32x16_b16 %0, %1, moffset:%2, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
            }
        }
        else
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, $1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_32x16_b16 %0, m0, moffset:%2, r, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_32x16_b16 %0, %1, moffset:%2, r, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
                else
                {
                    asm volatile("matrix_load_32x16_b16 %0, %1, moffset:%2, r, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
            }
        }
#else
        detail::swallow{smem, mls_res};
#endif
    }
};

struct gfx938_mls_16x32_trans_b16
{
    static constexpr auto TileShape = sequence<16, 32>{};

    template <index_t moffset, bool r, bool use_m0 = true, bool s_nop = true>
    CK_TILE_DEVICE static void
    load(const uintptr_t smem, const int32x4_t& mls_res, number<moffset>, bool_constant<r>)
    {
#if defined(__gfx938__)
        const auto soffset =
            static_cast<const uint32_t>(smem | CK_TILE_MATRIX_LOAD_TRANS_EXTRA_CONFIG);

        if constexpr(!r)
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, %1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_32x16_b16 %0, m0, moffset:%2, t, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_32x16_b16 %0, %1, moffset:%2, t, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
                else
                {
                    asm volatile("matrix_load_32x16_b16 %0, %1, moffset:%2, t, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
            }
        }
        else
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, %1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_32x16_b16 %0, m0, moffset:%2, t r, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_32x16_b16 %0, %1, moffset:%2, t r, lds; \n\t"
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
        }
#else
        detail::swallow{smem, mls_res};
#endif
    }
};

struct gfx938_mls_32x32_b16
{
    static constexpr auto TileShape = sequence<32, 32>{};

    template <index_t moffset, bool r, bool use_m0 = true, bool s_nop = true>
    CK_TILE_DEVICE static void
    load(const uintptr_t smem, const int32x4_t& mls_res, number<moffset>, bool_constant<r>)
    {
#if defined(__gfx938__)
        const auto soffset = static_cast<const uint32_t>(smem);

        if constexpr(!r)
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, %1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_32x32_b16 %0, m0, moffset:%2, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_32x32_b16 %0, %1, moffset:%2, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
                else
                {
                    asm volatile("matrix_load_32x32_b16 %0, %1, moffset:%2, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
            }
        }
        else
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, %1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_32x32_b16 %0, m0, moffset:%2, r, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_32x32_b16 %0, %1, moffset:%2, r, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
                else
                {
                    asm volatile("matrix_load_32x32_b16 %0, %1, moffset:%2, r, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
            }
        }
#else
        detail::swallow{smem, mls_res};
#endif
    }
};

struct gfx938_mls_32x32_trans_b16
{
    static constexpr auto TileShape = sequence<32, 32>{};

    template <index_t moffset, bool r, bool use_m0 = true, bool s_nop = true>
    CK_TILE_DEVICE static void
    load(const uintptr_t smem, const int32x4_t& mls_res, number<moffset>, bool_constant<r>)
    {
#if defined(__gfx938__)
        const auto soffset =
            static_cast<const uint32_t>(smem | CK_TILE_MATRIX_LOAD_TRANS_EXTRA_CONFIG);

        if constexpr(!r)
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, %1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_32x32_b16 %0, m0, moffset:%2, t lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_32x32_b16 %0, %1, moffset:%2, t lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
                else
                {
                    asm volatile("matrix_load_32x32_b16 %0, %1, moffset:%2, t lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
            }
        }
        else
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, %1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_32x32_b16 %0, m0, moffset:%2, t r, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_32x32_b16 %0, %1, moffset:%2, t r, lds; \n\t"
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
        }
#else
        detail::swallow{smem, mls_res};
#endif
    }
};

struct gfx938_mls_64x16_b16
{
    static constexpr auto TileShape = sequence<64, 16>{};

    template <index_t moffset, bool r, bool use_m0 = true, bool s_nop = true>
    CK_TILE_DEVICE static void
    load(const uintptr_t smem, const int32x4_t& mls_res, number<moffset>, bool_constant<r>)
    {
#if defined(__gfx938__)
        const auto soffset = static_cast<const uint32_t>(smem);

        if constexpr(!r)
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, %1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_64x16_b16 %0, m0, moffset:%2, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_64x16_b16 %0, %1, moffset:%2, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
                else
                {
                    asm volatile("matrix_load_64x16_b16 %0, %1, moffset:%2, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
            }
        }
        else
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, %1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_64x16_b16 %0, m0, moffset:%2, r, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_64x16_b16 %0, %1, moffset:%2, r, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
                else
                {
                    asm volatile("matrix_load_64x16_b16 %0, %1, moffset:%2, r, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
            }
        }
#else
        detail::swallow{smem, mls_res};
#endif
    }
};

struct gfx938_mls_16x64_trans_b16
{
    static constexpr auto TileShape = sequence<16, 64>{};

    template <index_t moffset, bool r, bool use_m0 = true, bool s_nop = true>
    CK_TILE_DEVICE static void
    load(const uintptr_t smem, const int32x4_t& mls_res, number<moffset>, bool_constant<r>)
    {
#if defined(__gfx938__)
        const auto soffset =
            static_cast<const uint32_t>(smem | CK_TILE_MATRIX_LOAD_TRANS_EXTRA_CONFIG);

        if constexpr(!r)
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, %1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_64x16_b16 %0, m0, moffset:%2, t, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_64x16_b16 %0, %1, moffset:%2, t, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
                else
                {
                    asm volatile("matrix_load_64x16_b16 %0, %1, moffset:%2, t, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
            }
        }
        else
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, %1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_64x16_b16 %0, m0, moffset:%2, t r, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_64x16_b16 %0, %1, moffset:%2, t r, lds; \n\t"
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
        }
#else
        detail::swallow{smem, mls_res};
#endif
    }
};

struct gfx938_mls_64x16_b8
{
    static constexpr auto TileShape = sequence<64, 16>{};

    template <index_t moffset, bool r, bool use_m0 = true, bool s_nop = true>
    CK_TILE_DEVICE static void
    load(const uintptr_t smem, const int32x4_t& mls_res, number<moffset>, bool_constant<r>)
    {
#if defined(__gfx938__)
        const auto soffset = static_cast<const uint32_t>(smem);

        if constexpr(!r)
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, %1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_64x16_b8 %0, m0, moffset:%2, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_64x16_b8 %0, %1, moffset:%2, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
                else
                {
                    asm volatile("matrix_load_64x16_b8 %0, %1, moffset:%2, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
            }
        }
        else
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, %1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_64x16_b8 %0, m0, moffset:%2, r, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_64x16_b8 %0, %1, moffset:%2, r, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
                else
                {
                    asm volatile("matrix_load_64x16_b8 %0, %1, moffset:%2, r, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
            }
        }
#else
        detail::swallow{smem, mls_res};
#endif
    }
};

struct gfx938_mls_16x64_trans_b8
{
    static constexpr auto TileShape = sequence<16, 64>{};

    template <index_t moffset, bool r, bool use_m0 = true, bool s_nop = true>
    CK_TILE_DEVICE static void
    load(const uintptr_t smem, const int32x4_t& mls_res, number<moffset>, bool_constant<r>)
    {
#if defined(__gfx938__)
        const auto soffset =
            static_cast<const uint32_t>(smem | CK_TILE_MATRIX_LOAD_TRANS_EXTRA_CONFIG);

        if constexpr(!r)
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, %1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_64x16_b8 %0, m0, moffset:%2, t, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_64x16_b8 %0, %1, moffset:%2, t, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
                else
                {
                    asm volatile("matrix_load_64x16_b8 %0, %1, moffset:%2, t, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
            }
        }
        else
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, %1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_64x16_b8 %0, m0, moffset:%2, t r, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_64x16_b8 %0, %1, moffset:%2, t r, lds; \n\t"
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
        }
#else
        detail::swallow{smem, mls_res};
#endif
    }
};

struct gfx938_mls_64x32_b8
{
    static constexpr auto TileShape = sequence<64, 32>{};

    template <index_t moffset, bool r, bool use_m0 = true, bool s_nop = true>
    CK_TILE_DEVICE static void
    load(const uintptr_t smem, const int32x4_t& mls_res, number<moffset>, bool_constant<r>)
    {
#if defined(__gfx938__)
        const auto soffset = static_cast<const uint32_t>(smem);

        if constexpr(!r)
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, %1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_64x32_b8 %0, m0, moffset:%2, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_64x32_b8 %0, %1, moffset:%2, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
                else
                {
                    asm volatile("matrix_load_64x32_b8 %0, %1, moffset:%2, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
            }
        }
        else
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, %1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_64x32_b8 %0, m0, moffset:%2, r, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_64x32_b8 %0, %1, moffset:%2, r, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
                else
                {
                    asm volatile("matrix_load_64x32_b8 %0, %1, moffset:%2, r, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
            }
        }
#else
        detail::swallow{smem, mls_res};
#endif
    }
};

struct gfx938_mls_32x64_trans_b8
{
    static constexpr auto TileShape = sequence<32, 64>{};

    template <index_t moffset, bool r, bool use_m0 = true, bool s_nop = true>
    CK_TILE_DEVICE static void
    load(const uintptr_t smem, const int32x4_t& mls_res, number<moffset>, bool_constant<r>)
    {
#if defined(__gfx938__)
        const auto soffset =
            static_cast<const uint32_t>(smem | CK_TILE_MATRIX_LOAD_TRANS_EXTRA_CONFIG);

        if constexpr(!r)
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, %1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_64x32_b8 %0, m0, moffset:%2, t, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_64x32_b8 %0, %1, moffset:%2, t, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
                else
                {
                    asm volatile("matrix_load_64x32_b8 %0, %1, moffset:%2, t, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
            }
        }
        else
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, %1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_64x32_b8 %0, m0, moffset:%2, t r, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_64x32_b8 %0, %1, moffset:%2, t r, lds; \n\t"
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
        }
#else
        detail::swallow{smem, mls_res};
#endif
    }
};

struct gfx938_mls_128x16_b8
{
    static constexpr auto TileShape = sequence<128, 16>{};

    template <index_t moffset, bool r, bool use_m0 = true, bool s_nop = true>
    CK_TILE_DEVICE static void
    load(const uintptr_t smem, const int32x4_t& mls_res, number<moffset>, bool_constant<r>)
    {
#if defined(__gfx938__)
        const auto soffset = static_cast<const uint32_t>(smem);

        if constexpr(!r)
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, %1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_128x16_b8 %0, m0, moffset:%2, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_128x16_b8 %0, %1, moffset:%2, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
                else
                {
                    asm volatile("matrix_load_128x16_b8 %0, %1, moffset:%2, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
            }
        }
        else
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, %1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_128x16_b8 %0, m0, moffset:%2, r, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_128x16_b8 %0, %1, moffset:%2, r, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
                else
                {
                    asm volatile("matrix_load_128x16_b8 %0, %1, moffset:%2, r, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
            }
        }
#else
        detail::swallow{smem, mls_res};
#endif
    }
};

struct gfx938_mls_16x128_trans_b8
{
    static constexpr auto TileShape = sequence<16, 128>{};

    template <index_t moffset, bool r, bool use_m0 = true, bool s_nop = true>
    CK_TILE_DEVICE static void
    load(const uintptr_t smem, const int32x4_t& mls_res, number<moffset>, bool_constant<r>)
    {
#if defined(__gfx938__)
        const auto soffset =
            static_cast<const uint32_t>(smem | CK_TILE_MATRIX_LOAD_TRANS_EXTRA_CONFIG);

        if constexpr(!r)
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, %1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_128x16_b8 %0, m0, moffset:%2, t, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_128x16_b8 %0, %1, moffset:%2, t, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
                else
                {
                    asm volatile("matrix_load_128x16_b8 %0, %1, moffset:%2, t, lds; \n\t"
                                 :
                                 : "s"(mls_res), "s"(soffset), "n"(moffset)
                                 : "memory");
                }
            }
        }
        else
        {
            if constexpr(use_m0)
            {
                asm volatile("s_mov_b32 m0, %1\n\t"
                             "s_nop 0\n\t"
                             "matrix_load_128x16_b8 %0, m0, moffset:%2, t r, lds; \n\t"
                             :
                             : "s"(mls_res), "s"(soffset), "n"(moffset)
                             : "memory");
            }
            else
            {
                if constexpr(s_nop)
                {
                    asm volatile("s_nop 0\n\t"
                                 "matrix_load_128x16_b8 %0, %1, moffset:%2, t r, lds; \n\t"
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
        }
#else
        detail::swallow{smem, mls_res};
#endif
    }
};

} // namespace ck_tile
