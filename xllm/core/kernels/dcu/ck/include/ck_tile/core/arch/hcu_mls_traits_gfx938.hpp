#pragma once

#include "ck_tile/core/arch/amd_buffer_addressing.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/container/multi_index.hpp"
#include "ck_tile/core/arch/hcu_mls_traits.hpp"
#include "ck_tile/core/arch/hcu_mls_atom_gfx938.hpp"
#include "ck_tile/core/tensor/tensor_descriptor.hpp"

namespace ck_tile {

/*
 * since actual lds layout will always be wrapped in 16Dwords on major dim
 * lds layout below is not intend to do thread offset calculation
 * but is used for warp offset calculation
 */

template <index_t Alt>
struct mls_traits<gfx938_mls_32x16_b16, Alt>
{
    static_assert(Alt == 1 || Alt == 2, "Unsupported interleave config");

    static constexpr auto kMN = number<32>{};
    static constexpr auto kK  = number<16>{};

    static constexpr auto PackedShape = make_tuple(kK, kMN);
};

template <index_t Alt>
struct mls_traits<gfx938_mls_16x32_trans_b16, Alt>
{
    static_assert(Alt == 1 || Alt == 2, "Unsupported interleave config");

    static constexpr auto kMN = number<16>{};
    static constexpr auto kK  = number<32>{};

    static constexpr auto PackedShape = make_tuple(kMN, kK);
};

template <index_t Alt>
struct mls_traits<gfx938_mls_32x32_b16, Alt>
{
    static_assert(Alt == 1 || Alt == 2, "Unsupported interleave config");

    static constexpr auto kMN = number<32>{};
    static constexpr auto kK  = number<32>{};

    static constexpr auto PackedShape = make_tuple(kK, kMN);
};

template <index_t Alt>
struct mls_traits<gfx938_mls_32x32_trans_b16, Alt>
{
    static_assert(Alt == 1 || Alt == 2, "Unsupported interleave config");

    static constexpr auto kMN = number<32>{};
    static constexpr auto kK  = number<32>{};

    static constexpr auto PackedShape = make_tuple(kMN, kK);
};

template <index_t Alt>
struct mls_traits<gfx938_mls_64x16_b16, Alt>
{
    static_assert(Alt == 1 || Alt == 2, "Unsupported interleave config");

    static constexpr auto kMN0 = number<2>{};
    static constexpr auto kMN1 = number<32>{};
    static constexpr auto kK   = number<16>{};

    static constexpr auto PackedShape = make_tuple(kMN0, kK, kMN1);
};

template <>
struct mls_traits<gfx938_mls_16x64_trans_b16, 1>
{
    static constexpr auto kMN = number<16>{};
    static constexpr auto kK0 = number<2>{};
    static constexpr auto kK1 = number<32>{};

    static constexpr auto PackedShape = make_tuple(kK0, kMN, kK1);
};

template <>
struct mls_traits<gfx938_mls_16x64_trans_b16, 2>
{
    static constexpr auto kMN = number<16>{};
    static constexpr auto kK0 = number<2>{};
    static constexpr auto kK1 = number<32>{};

    static constexpr auto kSlots = number<2>{};

    static constexpr auto PackedShape = make_tuple(kK0, kSlots, kMN, kK1);
};

template <>
struct mls_traits<gfx938_mls_64x16_b8, 1>
{
    static constexpr auto kMN = number<64>{};
    static constexpr auto kK  = number<16>{};

    static constexpr auto PackedShape = make_tuple(kK, kMN);
};

template <>
struct mls_traits<gfx938_mls_64x16_b8, 2>
{
    static constexpr auto kMN = number<64>{};
    static constexpr auto kK0 = number<2>{};
    static constexpr auto kK1 = number<8>{};

    static constexpr auto kSlots = number<2>{};

    static constexpr auto PackedShape = make_tuple(kK0, kSlots, kK1, kMN);
};

template <>
struct mls_traits<gfx938_mls_64x16_b8, 4>
{
    static constexpr auto kMN = number<64>{};
    static constexpr auto kK0 = number<2>{};
    static constexpr auto kK1 = number<8>{};

    static constexpr auto kSlots = number<2>{};

    static constexpr auto PackedShape = make_tuple(kK0, kSlots, kK1, kMN);
};

template <index_t Alt>
struct mls_traits<gfx938_mls_16x64_trans_b8, Alt>
{
    static_assert(Alt == 1 || Alt == 2, "Unsupported interleave config");

    static constexpr auto kMN = number<16>{};
    static constexpr auto kK  = number<64>{};

    static constexpr auto PackedShape = make_tuple(kMN, kK);
};

template <>
struct mls_traits<gfx938_mls_16x64_trans_b8, 4>
{
    static constexpr auto kMN0 = number<2>{};
    static constexpr auto kMN1 = number<8>{};
    static constexpr auto kK   = number<64>{};

    static constexpr auto kSlots = number<4>{};

    static constexpr auto PackedShape = make_tuple(kMN0, kSlots, kMN1, kK);
};

template <index_t Alt>
struct mls_traits<gfx938_mls_64x32_b8, Alt>
{
    static_assert(Alt == 1 || Alt == 2 || Alt == 4, "Unsupported interleave config");

    static constexpr auto kMN = number<64>{};
    static constexpr auto kK  = number<32>{};

    static constexpr auto PackedShape = make_tuple(kK, kMN);
};

template <index_t Alt>
struct mls_traits<gfx938_mls_32x64_trans_b8, Alt>
{
    static_assert(Alt == 1 || Alt == 2, "Unsupported interleave config");

    static constexpr auto kMN = number<32>{};
    static constexpr auto kK  = number<64>{};

    static constexpr auto PackedShape = make_tuple(kMN, kK);
};

template <>
struct mls_traits<gfx938_mls_32x64_trans_b8, 4>
{
    static constexpr auto kMN0 = number<4>{};
    static constexpr auto kMN1 = number<8>{};
    static constexpr auto kK   = number<64>{};

    static constexpr auto kSlots = number<2>{};

    static constexpr auto PackedShape = make_tuple(kMN0, kSlots, kMN1, kK);
};

template <>
struct mls_traits<gfx938_mls_128x16_b8, 1>
{
    static constexpr auto kMN0 = number<2>{};
    static constexpr auto kMN1 = number<64>{};
    static constexpr auto kK   = number<16>{};

    static constexpr auto kSlots = number<2>{};

    static constexpr auto PackedShape = make_tuple(kMN0, kSlots, kK, kMN1);
};

template <>
struct mls_traits<gfx938_mls_128x16_b8, 2>
{
    static constexpr auto kMN0 = number<2>{};
    static constexpr auto kMN1 = number<64>{};
    static constexpr auto kK0  = number<2>{};
    static constexpr auto kK1  = number<8>{};

    static constexpr auto kSlots = number<2>{};

    static constexpr auto PackedShape = make_tuple(kMN0, kK0, kSlots, kK1, kMN1);
};

template <>
struct mls_traits<gfx938_mls_128x16_b8, 4>
{
    static constexpr auto kMN0 = number<2>{};
    static constexpr auto kMN1 = number<64>{};
    static constexpr auto kK0  = number<2>{};
    static constexpr auto kK1  = number<8>{};

    static constexpr auto kSlots = number<2>{};

    // PackedShape: {K0, M0/N0, Aperture, K1, M1/N1}
    static constexpr auto PackedShape = mls_traits<gfx938_mls_128x16_b8, 2>::PackedShape;
};

template <>
struct mls_traits<gfx938_mls_16x128_trans_b8, 1>
{
    static constexpr auto kMN = number<16>{};
    static constexpr auto kK0 = number<2>{};
    static constexpr auto kK1 = number<64>{};

    static constexpr auto PackedShape = make_tuple(kK0, kMN, kK1);
};

template <>
struct mls_traits<gfx938_mls_16x128_trans_b8, 2>
{
    static constexpr auto kMN = number<16>{};
    static constexpr auto kK0 = number<2>{};
    static constexpr auto kK1 = number<64>{};

    static constexpr auto kSlots = number<2>{};

    static constexpr auto PackedShape = make_tuple(kK0, kSlots, kMN, kK1);
};

template <>
struct mls_traits<gfx938_mls_16x128_trans_b8, 4>
{
    static constexpr auto kMN = number<16>{};
    static constexpr auto kK0 = number<2>{};
    static constexpr auto kK1 = number<64>{};

    static constexpr auto kSlots = number<2>{};

    static constexpr auto PackedShape = make_tuple(kK0, kSlots, kMN, kK1);
};

} // namespace ck_tile
