// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/math.hpp"

namespace ck_tile {

template <index_t BBits, index_t MBase, index_t SShift = BBits>
struct Swizzle
{
    static constexpr index_t num_bits = BBits;
    static constexpr index_t num_base = MBase;
    static constexpr index_t num_shft = SShift;

    static_assert(num_base >= 0, "MBase must be positive.");
    static_assert(num_bits >= 0, "BBits must be positive.");
    // static_assert(ck::math::abs(num_shft) >= num_bits, "abs(SShift) must be more than BBits.");

    // using 'int' type here to avoid unintentially casting to unsigned... unsure.
    static constexpr index_t bit_msk = (1 << num_bits) - 1;
    static constexpr index_t yyy_msk = bit_msk << (num_base + ck_tile::max(0, num_shft));
    static constexpr index_t zzz_msk = bit_msk << (num_base - ck_tile::min(0, num_shft));
    static constexpr index_t msk_sft = num_shft;

    static constexpr uint32_t swizzle_code = uint32_t(yyy_msk | zzz_msk);

    CK_TILE_HOST_DEVICE constexpr static auto apply(index_t offset)
    {
        return offset ^ ((offset & yyy_msk) >> msk_sft); // ZZZ ^= YYY
    }

    CK_TILE_HOST_DEVICE constexpr auto operator()(index_t offset) const { return apply(offset); }

    template <int B, int M, int S>
    CK_TILE_HOST_DEVICE constexpr auto operator==(Swizzle<B, M, S> const&) const
    {
        return B == BBits && M == MBase && S == SShift;
    }
};

template <index_t ds_read_bytes>
CK_TILE_HOST_DEVICE constexpr auto make_swizzle(number<ds_read_bytes> = {})
{
    if constexpr(ds_read_bytes == 4)
    {
        return Swizzle<5, 1, 5>{};
    }
    else if constexpr(ds_read_bytes == 8)
    {
        return Swizzle<4, 2, 4>{};
    }
    else if constexpr(ds_read_bytes == 16)
    {
        return Swizzle<3, 3, 3>{};
    }
    else
    {
        static_assert(false, "not implemented yet");
    }
}

} // namespace ck_tile
