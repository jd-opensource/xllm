// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include <stdint.h>
#include <tuple>
#include <type_traits>
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/int8.hpp"
#include "ck_tile/core/numeric/mxfp_convert.hpp"

namespace ck_tile {

#if CK_TILE_USE_CUSTOM_DATA_TYPE
template <typename Y, typename X>
CK_TILE_HOST_DEVICE constexpr remove_cvref_t<Y> type_convert(const X& x)
{
    return static_cast<Y>(x);
}
#else
// Convert X to Y, both X and Y are non-const data types.
template <typename Y,
          typename X,
          std::enable_if_t<!(std::is_const_v<Y> || std::is_const_v<X>), bool> = false>
CK_TILE_HOST_DEVICE constexpr Y type_convert(X x)
{
    static_assert(!std::is_reference_v<Y> && !std::is_reference_v<X>);
    return static_cast<Y>(x);
}

// Convert X to Y, either X or Y is a const data type.
template <typename Y,
          typename X,
          std::enable_if_t<std::is_const_v<Y> || std::is_const_v<X>, bool> = false>
CK_TILE_HOST_DEVICE constexpr Y type_convert(X x)
{
    static_assert(!std::is_reference_v<Y> && !std::is_reference_v<X>);

    using non_const_y = std::remove_const_t<Y>;
    using non_const_x = std::remove_const_t<X>;
    return static_cast<Y>(type_convert<non_const_y, non_const_x>(x));
}

#define CK_TILE_TYPE_CONVERT(dtype_, dname_, stype_, sname_)                    \
    template <>                                                                 \
    CK_TILE_HOST_DEVICE constexpr dtype_ type_convert<dtype_, stype_>(stype_ x) \
    {                                                                           \
        return sname_##_to_##dname_(x);                                         \
    }

CK_TILE_TYPE_CONVERT(float, float, fp16_t, fp16)
CK_TILE_TYPE_CONVERT(float, float, bf16_t, bf16)
CK_TILE_TYPE_CONVERT(float, float, fp8_t, fp8)
CK_TILE_TYPE_CONVERT(float, float, bf8_t, bf8)

CK_TILE_TYPE_CONVERT(fp16_t, fp16, float, float)
CK_TILE_TYPE_CONVERT(bf16_t, bf16, float, float)
CK_TILE_TYPE_CONVERT(fp8_t, fp8, float, float)
CK_TILE_TYPE_CONVERT(bf8_t, bf8, float, float)

CK_TILE_TYPE_CONVERT(float, float, int8_t, int8)
CK_TILE_TYPE_CONVERT(int8_t, int8, float, float)
#undef CK_TILE_TYPE_CONVERT
#endif

} // namespace ck_tile

/*
#include "ck_tile/core/numeric/pk_fp4.hpp"

namespace ck_tile {

CK_TILE_TYPE_CONVERT(pk_fp4_t, pk_fp4, fp32x2_t, fp32x2)
CK_TILE_TYPE_CONVERT(fp32x2_t, fp32x2, pk_fp4_t, pk_fp4)
CK_TILE_TYPE_CONVERT(pk_fp4_t, pk_fp4, fp16x2_t, fp16x2)
CK_TILE_TYPE_CONVERT(fp16x2_t, fp16x2, pk_fp4_t, pk_fp4)
CK_TILE_TYPE_CONVERT(pk_fp4_t, pk_fp4, bf16x2_t, bf16x2)
CK_TILE_TYPE_CONVERT(bf16x2_t, bf16x2, pk_fp4_t, pk_fp4)
CK_TILE_TYPE_CONVERT(pk_fp4_t, pk_fp4, float, float)
CK_TILE_TYPE_CONVERT(pk_fp4_t, pk_fp4, bf16_t, bf16)
CK_TILE_TYPE_CONVERT(pk_fp4_t, pk_fp4, fp16_t, fp16)
#undef CK_TILE_TYPE_CONVERT
#endif

} // namespace ck_tile
*/
