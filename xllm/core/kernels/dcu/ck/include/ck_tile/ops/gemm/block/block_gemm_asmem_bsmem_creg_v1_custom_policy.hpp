// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Default policy for BlockGemmASmemBSmemCRegV1
// Default policy class should not be templated, put template on member functions instead
template <typename AType_,
          typename BType_,
          typename CType_,
          typename BlockWarps_,
          typename WarpGemm_>
struct BlockGemmASmemBSmemCRegV1CustomPolicy
{
    using AType = remove_cvref_t<AType_>;
    using BType = remove_cvref_t<BType_>;
    using CType = remove_cvref_t<CType_>;

    using BlockWarps = remove_cvref_t<BlockWarps_>;

    static constexpr index_t kMWarps = BlockWarps::at(number<0>{});
    static constexpr index_t kNWarps = BlockWarps::at(number<1>{});
    static constexpr index_t kKWarps = BlockWarps::at(number<2>{});

    using WarpGemm = remove_cvref_t<WarpGemm_>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return make_tuple(WarpGemm{}, kMWarps, kNWarps);
    }
};

} // namespace ck_tile
