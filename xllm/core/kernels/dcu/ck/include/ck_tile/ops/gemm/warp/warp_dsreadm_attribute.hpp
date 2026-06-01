// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/container/thread_buffer.hpp"
#include "ck_tile/core/tensor/tile_distribution_encoding.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/utility/functional.hpp"

namespace ck_tile {

template <typename WarpDsreadmAttributeImpl,
          index_t MNIter,
          index_t MNPermmac,
          index_t MNInterleave>
struct WarpDsreadmAttribute
{
    using Impl = remove_cvref_t<WarpDsreadmAttributeImpl>;

    using RawType = typename Impl::RawType;
    using RetType = thread_buffer<typename Impl::RetType, 1>;

    static constexpr index_t MNWarpTile  = MNIter * MNPermmac * MNInterleave;
    static constexpr index_t MNWarpIssue = MNWarpTile / Impl::kMN;

    static_assert(MNWarpTile % Impl::kMN == 0,
                  "warp tile must be multiple of WarpDsreadmAttributeImpl::kMN");
    static_assert(MNInterleave == Impl::kMNInterleave,
                  "MNInterleave must be the same as WarpDsreadmAttributeImpl::kMNInterleave!");

    using WarpLoadDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MNWarpIssue, Impl::kMNLoadLane, Impl::kMNLoadPerLane>,
              sequence<Impl::kKLoadLane, Impl::kKLoadPerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 1>>,
        sequence<1, 2, 1>,
        sequence<0, 1, 2>>;

    using WarpStoreDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<MNIter, Impl::kMNStoreLane, MNInterleave>,
                                         sequence<Impl::kKStoreLane, Impl::kKStorePerLane>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<0, 1>>,
                                   sequence<1, 1, 2>,
                                   sequence<0, 2, 1>>;

    // unused, equivalent to WarpStoreDstrEncoding
    using WarpStoreDstrDetailEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<MNWarpIssue,
                                                  Impl::kMN0StorePerLane,
                                                  Impl::kMNStoreLane,
                                                  Impl::kMN1StorePerLane>,
                                         sequence<Impl::kKStoreLane, Impl::kKStorePerLane>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<0, 2>>,
                                   sequence<1, 1, 1, 2>,
                                   sequence<0, 1, 3, 1>>;

    template <typename T>
    CK_TILE_DEVICE RetType operator()(const T* lds_ptr) const
    {
        RetType ret;
        Impl{}(reinterpret_cast<const RawType*>(lds_ptr),
               ret.template get_as<typename Impl::RetType>()[number<0>{}]);
        return ret;
    }
};

} // namespace ck_tile
