// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/gemm/warp/warp_dsreadm_attribute.hpp"
#include "ck_tile/ops/gemm/warp/warp_dsreadm_attribute_impl.hpp"

namespace ck_tile {

namespace impl {

template <typename DataType, index_t MNIter, index_t MNPermmac, index_t MNInterleave>
struct WarpDsreadmDispatcher;

template <>
struct WarpDsreadmDispatcher<fp16_t, 1, 16, 2>
{
    using Type = WarpDsreadmAttribute<WarpDsreadmAttributeImplF16MN32K16Alt, 1, 16, 2>;
};

template <>
struct WarpDsreadmDispatcher<fp16_t, 2, 16, 1>
{
    using Type = WarpDsreadmAttribute<WarpDsreadmAttributeImplF16MN32K16, 2, 16, 1>;
};

} // namespace impl

template <typename DataType, index_t MNIter, index_t MNPermmac, index_t MNInterleave>
using WarpDsreadmDispatcher =
    typename impl::WarpDsreadmDispatcher<DataType, MNIter, MNPermmac, MNInterleave>::Type;

} // namespace ck_tile
