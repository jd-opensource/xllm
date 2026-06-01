// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/multi_index.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {
//  pre-defined indexing adaptor used for indexing(scatter/gather)

// this version cache the index inside thread register(which is also prefered in real senario)
// however it's user's responsibility that each thread only provide one indexing, which means
// move coordinate will not change on this dim
template <typename IndexingType>
struct indexing_adaptor_onshot_cached
{

    CK_TILE_HOST_DEVICE constexpr indexing_adaptor_onshot_cached() = default;
    CK_TILE_HOST_DEVICE constexpr indexing_adaptor_onshot_cached(const IndexingType& idx)
        : cached_idx_(idx)
    {
    }
    IndexingType cached_idx_;

    template <typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE constexpr void calculate_lower_index(LowIdx& idx_low,
                                                             const UpIdx& /*idx_up*/) const
    {
        static_assert(LowIdx::size() == 1 && UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(number<0>{}) = cached_idx_;
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE void update_lower_index(LowIdxDiff& idx_diff_low,
                                                const UpIdxDiff& idx_diff_up,
                                                LowIdx& /*idx_low*/,
                                                const UpIdx& /*idx_up*/) const
    {
        // TODO: nonthing changed here
        static_assert(LowIdxDiff::size() == 1 && UpIdxDiff::size() == 1 && LowIdx::size() == 1 &&
                          UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_diff_low(number<0>{}) = idx_diff_up[number<0>{}];

        // pass the diff to lower, but not changing the actually index
    }

    CK_TILE_HOST_DEVICE static constexpr bool is_known_at_compile_time()
    {
        return ck_tile::is_known_at_compile_time<IndexingType>::value;
    }
};
} // namespace ck_tile
