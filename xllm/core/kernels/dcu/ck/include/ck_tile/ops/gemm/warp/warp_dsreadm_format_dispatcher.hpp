// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/gemm/warp/warp_dsreadm_format_attribute.hpp"
#include "ck_tile/ops/gemm/warp/warp_dsreadm_format_attribute_impl.hpp"

namespace ck_tile {

namespace impl {

template <typename DataType, index_t MNIter, index_t MNPermmac, index_t MNInterleave, bool Trans>
struct WarpDsreadmFormatDispatcher;

template <>
struct WarpDsreadmFormatDispatcher<fp16_t, 2, 16, 1, false>
{
    using Type = WarpDsreadmFormatAttribute<WarpDsreadmFormatAttributeImpl_M32x16_B16>;
};

template <>
struct WarpDsreadmFormatDispatcher<fp16_t, 1, 16, 2, false>
{
    using Type = WarpDsreadmFormatAttribute<WarpDsreadmFormatAttributeImpl_M32x16_B16_ALT2>;
};

template <>
struct WarpDsreadmFormatDispatcher<fp16_t, 1, 16, 1, true>
{
    using Type = WarpDsreadmFormatAttribute<WarpDsreadmFormatAttributeImpl_MT32x16_B16>;
};

template <>
struct WarpDsreadmFormatDispatcher<fp16_t, 2, 16, 1, true>
{
    using Type = WarpDsreadmFormatAttribute<WarpDsreadmFormatAttributeImpl_MT16x32_B16>;
};

template <>
struct WarpDsreadmFormatDispatcher<fp16_t, 1, 16, 2, true>
{
    using Type = WarpDsreadmFormatAttribute<WarpDsreadmFormatAttributeImpl_MT16x32_B16_ALT2>;
};

} // namespace impl

template <typename DataType, index_t MNIter, index_t MNPermmac, index_t MNIterleave, bool Trans>
using WarpDsreadmFormatDispatcher = typename impl::
    WarpDsreadmFormatDispatcher<DataType, MNIter, MNPermmac, MNIterleave, Trans>::Type;

} // namespace ck_tile
