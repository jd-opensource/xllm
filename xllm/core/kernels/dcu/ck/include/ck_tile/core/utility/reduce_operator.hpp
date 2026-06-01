// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"

namespace ck_tile {

namespace ReduceOp {
// y = ReduceOp(y, x);
struct Add
{
    template <typename T>
    CK_TILE_HOST_DEVICE static constexpr T GetIdentityValue()
    {
        return type_convert<T>(0.0f);
    };

    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double> ||
                                          std::is_same_v<T, int32_t> || std::is_same_v<T, int8_t>>>
    CK_TILE_HOST_DEVICE constexpr T operator()(const T& y, const T x) const
    {
        return y + x;
    }

    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, half_t> || std::is_same_v<T, bf16_t>>>
    CK_TILE_HOST_DEVICE constexpr T operator()(T& y, T x) const
    {
        float y_ = type_convert<float>(y);
        float x_ = type_convert<float>(x);

        return type_convert<T>(y_ + x_);
    }
};

struct SquareAdd
{
    template <typename T>
    CK_TILE_HOST_DEVICE static constexpr T GetIdentityValue()
    {
        return type_convert<T>(0.0f);
    };

    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double> ||
                                          std::is_same_v<T, int32_t> || std::is_same_v<T, int8_t>>>
    CK_TILE_HOST_DEVICE constexpr T operator()(const T& y, const T x) const
    {
        return y + (x * x);
    }
};

struct Max
{
    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double> ||
                                          std::is_same_v<T, int32_t> || std::is_same_v<T, int8_t>>>
    CK_TILE_HOST_DEVICE static constexpr T GetIdentityValue()
    {
        return numeric<T>::min();
    };

    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double> ||
                                          std::is_same_v<T, int32_t> || std::is_same_v<T, int8_t>>>
    CK_TILE_HOST_DEVICE constexpr T operator()(const T& y, const T x) const
    {
        return max(y, x);
    }
};

struct AbsMax
{
    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double> ||
                                          std::is_same_v<T, int32_t> || std::is_same_v<T, int8_t>>>
    CK_TILE_HOST_DEVICE static constexpr T GetIdentityValue()
    {
        return numeric<T>::min();
    };

    template <typename T,
              typename = std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double> ||
                                          std::is_same_v<T, int32_t> || std::is_same_v<T, int8_t>>>
    CK_TILE_HOST_DEVICE constexpr T operator()(const T& y, const T x) const
    {
        return max(y, abs(x));
    }
};

} // namespace ReduceOp
} // namespace ck_tile
