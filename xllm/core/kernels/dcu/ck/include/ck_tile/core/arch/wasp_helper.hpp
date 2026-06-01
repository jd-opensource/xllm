// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/arch/arch.hpp"

namespace ck_tile {

template <index_t WarpsPerGroup_, index_t BlockSize_, bool WaspEnable_>
struct wasp_helper
{
    static constexpr auto WarpsPerGroup = WarpsPerGroup_;
    static constexpr auto BlockSize     = BlockSize_;
    static constexpr auto WaspEnable    = WaspEnable_;

    CK_TILE_HOST_DEVICE constexpr wasp_helper()
    {
        if constexpr(!WaspEnable)
        {
            static_assert(BlockSize == WarpsPerGroup * get_warp_size());
        }
    }

    CK_TILE_HOST_DEVICE static auto get_warp_id()
    {
        if constexpr(WaspEnable)
        {
            return get_sub_warp_id(number<WarpsPerGroup>{});
        }
        else
        {
            return ck_tile::get_warp_id();
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto get_warp_num() { return WarpsPerGroup; }
};

} // namespace ck_tile
