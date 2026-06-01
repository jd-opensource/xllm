// SPDX-License-Identifier: MIT
// Copyright (c) 2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/elementwise/binary_element_wise_operation.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include <type_traits>

namespace ck_tile {
namespace element_wise {

struct AddAdd
{
    template <typename Y, typename X0, typename X1, typename X2>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X0& x0, const X1& x1, const X2& x2) const;

    template <>
    CK_TILE_HOST_DEVICE void operator()(ck_tile::fp16_t& y,
                                        const ck_tile::fp16_t& x0,
                                        const ck_tile::fp16_t& x1,
                                        const ck_tile::fp16_t& x2) const
    {
        y = x0 + x1 + x2;
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()(ck_tile::fp16x2_t& y,
                                        const ck_tile::fp16x2_t& x0,
                                        const ck_tile::fp16x2_t& x1,
                                        const ck_tile::fp16x2_t& x2) const
    {
        ck_tile::fp16x2_t y_tmp;
        ck_tile::element_wise::Add{}(y_tmp, x0, x1);
        ck_tile::element_wise::Add{}(y, y_tmp, x2);
    }

    template <>
    CK_TILE_HOST_DEVICE void
    operator()(float& y, const float& x0, const float& x1, const float& x2) const
    {
        y = x0 + x1 + x2;
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()(ck_tile::bf16_t& y,
                                        const ck_tile::bf16_t& x0,
                                        const ck_tile::bf16_t& x1,
                                        const ck_tile::bf16_t& x2) const
    {
        const float f0 = ck_tile::type_convert<float>(x0);
        const float f1 = ck_tile::type_convert<float>(x1);
        const float f2 = ck_tile::type_convert<float>(x2);
        y = ck_tile::type_convert<ck_tile::bf16_t>(f0 + f1 + f2);
    }

    template <>
    CK_TILE_HOST_DEVICE void
    operator()(int32_t& y, const int32_t& x0, const int32_t& x1, const int32_t& x2) const
    {
        y = x0 + x1 + x2;
    }
};

struct MultiplyMultiply
{
    template <typename E, typename C, typename D0, typename D1>
    CK_TILE_HOST_DEVICE void operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

    template <>
    CK_TILE_HOST_DEVICE void operator()(ck_tile::half_t& e,
                                        const ck_tile::half_t& c,
                                        const ck_tile::half_t& d0,
                                        const ck_tile::half_t& d1) const
    {
        const float x = ck_tile::type_convert<float>(c) * ck_tile::type_convert<float>(d0) *
                        ck_tile::type_convert<float>(d1);
        e = ck_tile::type_convert<ck_tile::half_t>(x);
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()(ck_tile::fp16x2_t& e,
                                        const ck_tile::fp16x2_t& c,
                                        const ck_tile::fp16x2_t& d0,
                                        const ck_tile::fp16x2_t& d1) const
    {
        const float c0  = ck_tile::type_convert<float>(c[0]);
        const float c1  = ck_tile::type_convert<float>(c[1]);
        const float d00 = ck_tile::type_convert<float>(d0[0]);
        const float d01 = ck_tile::type_convert<float>(d0[1]);
        const float d10 = ck_tile::type_convert<float>(d1[0]);
        const float d11 = ck_tile::type_convert<float>(d1[1]);
        e[0] = ck_tile::type_convert<ck_tile::half_t>(c0 * d00 * d10);
        e[1] = ck_tile::type_convert<ck_tile::half_t>(c1 * d01 * d11);
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()(ck_tile::bf16_t& e,
                                        const ck_tile::bf16_t& c,
                                        const ck_tile::bf16_t& d0,
                                        const ck_tile::bf16_t& d1) const
    {
        const float x = ck_tile::type_convert<float>(c) * ck_tile::type_convert<float>(d0) *
                        ck_tile::type_convert<float>(d1);
        e = ck_tile::type_convert<ck_tile::bf16_t>(x);
    }

    template <>
    CK_TILE_HOST_DEVICE void
    operator()(float& e, const float& c, const float& d0, const float& d1) const
    {
        e = c * d0 * d1;
    }

    template <>
    CK_TILE_HOST_DEVICE void
    operator()(int32_t& e, const int32_t& c, const int32_t& d0, const int32_t& d1) const
    {
        e = c * d0 * d1;
    }
};

struct AddAddRelu
{
    template <typename Y, typename X0, typename X1, typename X2>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X0& x0, const X1& x1, const X2& x2) const;

    template <>
    CK_TILE_HOST_DEVICE void operator()(ck_tile::fp16_t& y,
                                        const ck_tile::fp16_t& x0,
                                        const ck_tile::fp16_t& x1,
                                        const ck_tile::fp16_t& x2) const
    {
        ck_tile::fp16_t y_tmp;
        ck_tile::element_wise::AddAdd{}(y_tmp, x0, x1, x2);
        ck_tile::element_wise::Relu{}(y, y_tmp);
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()(ck_tile::fp16x2_t& y,
                                        const ck_tile::fp16x2_t& x0,
                                        const ck_tile::fp16x2_t& x1,
                                        const ck_tile::fp16x2_t& x2) const
    {
        ck_tile::fp16x2_t y_tmp;
        ck_tile::element_wise::AddAdd{}(y_tmp, x0, x1, x2);
        ck_tile::element_wise::Relu{}(y, y_tmp);
    }

    template <>
    CK_TILE_HOST_DEVICE void
    operator()(float& y, const float& x0, const float& x1, const float& x2) const
    {
        float y_tmp;
        ck_tile::element_wise::AddAdd{}(y_tmp, x0, x1, x2);
        ck_tile::element_wise::Relu{}(y, y_tmp);
    }
};

} // namespace element_wise
} // namespace ck_tile
