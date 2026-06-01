// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/gemm/warp/warp_dsreadm_format_attribute.hpp"
#include "ck_tile/ops/gemm/warp/warp_dsreadm_format_attribute_impl.hpp"

namespace ck_tile {

namespace impl {

template <index_t ElemBytes, index_t Row, index_t Col, index_t Alt, bool Trans>
struct WarpDsreadmFormatDispatcherV2;

template <>
struct WarpDsreadmFormatDispatcherV2<2, 32, 16, 1, false>
{
    using Type = WarpDsreadmFormatAttribute<WarpDsreadmFormatAttributeImpl_M32x16_B16>;
};

template <>
struct WarpDsreadmFormatDispatcherV2<2, 32, 16, 2, false>
{
    using Type = WarpDsreadmFormatAttribute<WarpDsreadmFormatAttributeImpl_M32x16_B16_ALT2>;
};

template <>
struct WarpDsreadmFormatDispatcherV2<2, 32, 16, 1, true>
{
    using Type = WarpDsreadmFormatAttribute<WarpDsreadmFormatAttributeImpl_MT32x16_B16>;
};

template <>
struct WarpDsreadmFormatDispatcherV2<2, 16, 32, 1, true>
{
    using Type = WarpDsreadmFormatAttribute<WarpDsreadmFormatAttributeImpl_MT16x32_B16>;
};

template <>
struct WarpDsreadmFormatDispatcherV2<2, 16, 32, 2, true>
{
    using Type = WarpDsreadmFormatAttribute<WarpDsreadmFormatAttributeImpl_MT16x32_B16_ALT2>;
};

} // namespace impl

template <index_t ElemBytes, index_t Row, index_t Col, index_t Alt, bool Trans>
using WarpDsreadmFormatDispatcherV2 =
    typename impl::WarpDsreadmFormatDispatcherV2<ElemBytes, Row, Col, Alt, Trans>::Type;

} // namespace ck_tile
