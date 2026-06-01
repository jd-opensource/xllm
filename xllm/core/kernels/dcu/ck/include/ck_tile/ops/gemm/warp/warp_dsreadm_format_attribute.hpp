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

template <typename WarpDsreadmFormatAttributeImpl>
struct WarpDsreadmFormatAttribute
{
    using Impl = remove_cvref_t<WarpDsreadmFormatAttributeImpl>;

    static constexpr index_t kMN = Impl::kMN;
    static constexpr index_t kK  = Impl::kK;

    static constexpr index_t kMN0StorePerlane = Impl::kMN0StorePerLane;
    static constexpr index_t kMNStoreLane     = Impl::kMNStoreLane;
    static constexpr index_t kMN1StorePerLane = Impl::kMN1StorePerLane;
    static constexpr index_t kKStoreLane      = Impl::kKStoreLane;
    static constexpr index_t kKStorePerLane   = Impl::kKStorePerLane;

    static constexpr index_t kVectorLength = Impl::kVectorLength;

    // equivalent to A/B mmac encoding of WarpGemmAttributeMmacIterateK
    using WarpStoreDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<kMN0StorePerlane, kMNStoreLane, kMN1StorePerLane>,
                                         sequence<kKStoreLane, kKStorePerLane>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<0, 1>>,
                                   sequence<1, 1, 2>,
                                   sequence<0, 2, 1>>;

    template <typename T, index_t offset, bool use_m0 = false, bool skip_mov_m0 = false>
    CK_TILE_DEVICE auto operator()(CK_TILE_LDS_ADDR T* smem_ptr,
                                   number<offset>,
                                   bool_constant<use_m0>      = {},
                                   bool_constant<skip_mov_m0> = {}) const
    {
        using VectorType = ext_vector_t<T, kVectorLength>;
        using RetType    = thread_buffer<VectorType, 1>;

        RetType ret;

        Impl{}(smem_ptr,
               ret.template get_as<VectorType>()[number<0>{}],
               number<offset>{},
               bool_constant<use_m0>{},
               bool_constant<skip_mov_m0>{});

        return ret;
    }
};

} // namespace ck_tile
