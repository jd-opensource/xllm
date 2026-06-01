// SPDX-License-Identifier: MIT
// Copyright (c) 2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include <type_traits>

namespace ck_tile {
namespace element_wise {

struct Add
{
    template <typename Y, typename X0, typename X1>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X0& x0, const X1& x1) const;

    template <>
    CK_TILE_HOST_DEVICE void
    operator()(ck_tile::fp16_t& y, const ck_tile::fp16_t& x0, const ck_tile::fp16_t& x1) const
    {
        y = x0 + x1;
    }

    template <>
    CK_TILE_HOST_DEVICE void
    operator()(ck_tile::fp16x2_t& y, const ck_tile::fp16x2_t& x0, const ck_tile::fp16x2_t& x1) const
    {
        asm volatile("v_pk_add_f16 %0 %1 %2;\n\t" : "=v"(y) : "v"(x0), "v"(x1));
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()(float& y, const float& x0, const float& x1) const
    {
        y = x0 + x1;
    }

    template <>
    CK_TILE_HOST_DEVICE void
    operator()(ck_tile::bf16_t& y, const ck_tile::bf16_t& x0, const ck_tile::bf16_t& x1) const
    {
        const float f0 = ck_tile::type_convert<float>(x0);
        const float f1 = ck_tile::type_convert<float>(x1);
        y = ck_tile::type_convert<ck_tile::bf16_t>(f0 + f1);
    }

    template <>
    CK_TILE_HOST_DEVICE void
    operator()(int32_t& y, const int32_t& x0, const int32_t& x1) const
    {
        y = x0 + x1;
    }
};

struct AddRelu
{
    template <typename Y, typename X0, typename X1>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X0& x0, const X1& x1) const;

    template <>
    CK_TILE_HOST_DEVICE void
    operator()(ck_tile::fp16_t& y, const ck_tile::fp16_t& x0, const ck_tile::fp16_t& x1) const
    {
        ck_tile::fp16_t y_tmp;
        ck_tile::element_wise::Add{}(y_tmp, x0, x1);
        ck_tile::element_wise::Relu{}(y, y_tmp);
    }

    template <>
    CK_TILE_HOST_DEVICE void
    operator()(ck_tile::fp16x2_t& y, const ck_tile::fp16x2_t& x0, const ck_tile::fp16x2_t& x1) const
    {
        ck_tile::fp16x2_t y_tmp = 0;
        ck_tile::element_wise::Add{}(y_tmp, x0, x1);
        ck_tile::element_wise::Relu{}(y, y_tmp);
    }

    template <>
    CK_TILE_HOST_DEVICE void operator()(float& y, const float& x0, const float& x1) const
    {
        float y_tmp;
        ck_tile::element_wise::Add{}(y_tmp, x0, x1);
        ck_tile::element_wise::Relu{}(y, y_tmp);
    }
};

} // namespace element_wise
} // namespace ck_tile
