// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include <limits>
#include <stdint.h>

namespace ck_tile {

// this struct has the information of
// 1. limit of a certain type, simliar to std::numeric_limits
// 2. some pre-defined value, zero, one...
//
template <typename T>
struct numeric
{
    // minimum finite value, or minimum positive normalized value for float
    CK_TILE_HOST_DEVICE static constexpr T min() { return std::numeric_limits<T>::min(); }

    // minumum finite value
    CK_TILE_HOST_DEVICE static constexpr T lowest() { return std::numeric_limits<T>::lowest(); }

    // maximum finite value
    CK_TILE_HOST_DEVICE static constexpr T max() { return std::numeric_limits<T>::max(); }

    // difference between 1.0 and next value representable by float
    CK_TILE_HOST_DEVICE static constexpr T epsilon() { return std::numeric_limits<T>::epsilon(); }

    // maximum rounding error
    CK_TILE_HOST_DEVICE static constexpr T round_error()
    {
        return std::numeric_limits<T>::round_error();
    }

    // positive infinity value
    CK_TILE_HOST_DEVICE static constexpr T infinity() { return std::numeric_limits<T>::infinity(); }

    // quiet NaN
    CK_TILE_HOST_DEVICE static constexpr T quiet_NaN()
    {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // signaling NaN
    CK_TILE_HOST_DEVICE static constexpr T signaling_NaN()
    {
        return std::numeric_limits<T>::signaling_NaN();
    }

    // smallest positive subnormal value
    CK_TILE_HOST_DEVICE static constexpr T denorm_min()
    {
        return std::numeric_limits<T>::denorm_min();
    }

    CK_TILE_HOST_DEVICE static constexpr T zero() { return static_cast<T>(0); }

    CK_TILE_HOST_DEVICE static constexpr T one() { return static_cast<T>(1); }

#ifndef C_LOG2E
#define C_LOG2E 1.44269504088896340736 // log2(e)
#endif

    CK_TILE_HOST_DEVICE static constexpr T log2e()
    {
        if constexpr(std::is_same_v<T, float> || std::is_same_v<T, double>)
        {
            return static_cast<T>(C_LOG2E);
        }
        else
        {
            return 0; // TODO: integer?
        }
    }
};

template <typename T>
struct numeric_traits
{
    static constexpr int PackedSize = 1;
};

template <>
struct numeric_traits<float>
{
    static constexpr int exp            = 8;
    static constexpr int mant           = 23;
    static constexpr int bias           = 127;
    static constexpr uint32_t nan_mask  = 0x7F800000;
    static constexpr uint32_t head_mask = 0xFF800000;
    static constexpr uint32_t mant_mask = 0x7FFFFF;
    static constexpr uint32_t exp_mask  = 0xFF;
    static constexpr uint32_t abs_mask  = 0x7FFFFFFF;
    static constexpr uint32_t Inf       = 0x7F800000;
    static constexpr uint32_t NegInf    = 0xFF800000;
    static constexpr uint32_t NaN       = 0x7F800001;
    static constexpr uint32_t Neg0      = 0x80000000;
    static constexpr int PackedSize     = 1;
    using bitwise_type                  = uint32_t;
};

} // namespace ck_tile

#define CK_TILE_ARITHMETIC_USING_FLOAT(attr_, type_)                                       \
    attr_ bool operator==(const type_& x, const type_& y)                                  \
    {                                                                                      \
        return std::abs(static_cast<float>(x) - static_cast<float>(y)) <                   \
               static_cast<float>(numeric<type_>::epsilon());                              \
    }                                                                                      \
    attr_ bool operator!=(const type_& x, const type_& y) { return not operator==(x, y); } \
    attr_ bool operator<(const type_& x, const type_& y)                                   \
    {                                                                                      \
        return static_cast<float>(x) < static_cast<float>(y);                              \
    }                                                                                      \
    attr_ bool operator<=(const type_& x, const type_& y)                                  \
    {                                                                                      \
        return static_cast<float>(x) <= static_cast<float>(y);                             \
    }                                                                                      \
    attr_ bool operator>(const type_& x, const type_& y)                                   \
    {                                                                                      \
        return static_cast<float>(x) > static_cast<float>(y);                              \
    }                                                                                      \
    attr_ bool operator>=(const type_& x, const type_& y)                                  \
    {                                                                                      \
        return static_cast<float>(x) >= static_cast<float>(y);                             \
    }                                                                                      \
    attr_ type_ operator+(const type_& x, const type_& y)                                  \
    {                                                                                      \
        return type_(static_cast<float>(x) + static_cast<float>(y));                       \
    }                                                                                      \
    attr_ type_ operator-(const type_& x)                                                  \
    {                                                                                      \
        constexpr uint32_t bits = sizeof(type_) * 8;                                       \
        constexpr uint32_t mask = 1 << (bits - 1);                                         \
        type_ y                 = x;                                                       \
        y.data ^= static_cast<typename type_::raw_type>(mask);                             \
        return y;                                                                          \
    }                                                                                      \
    attr_ type_ operator-(const type_& x, const type_& y)                                  \
    {                                                                                      \
        return type_(static_cast<float>(x) - static_cast<float>(y));                       \
    }                                                                                      \
    attr_ type_ operator*(const type_& x, const type_& y)                                  \
    {                                                                                      \
        return type_(static_cast<float>(x) * static_cast<float>(y));                       \
    }                                                                                      \
    attr_ type_ operator/(const type_& x, const type_& y)                                  \
    {                                                                                      \
        return type_(static_cast<float>(x) / static_cast<float>(y));                       \
    }                                                                                      \
    attr_ type_& operator+=(type_& x, const type_& y)                                      \
    {                                                                                      \
        x = type_(static_cast<float>(x) + static_cast<float>(y));                          \
        return x;                                                                          \
    }                                                                                      \
    attr_ type_& operator-=(type_& x, const type_& y)                                      \
    {                                                                                      \
        x = type_(static_cast<float>(x) - static_cast<float>(y));                          \
        return x;                                                                          \
    }                                                                                      \
    attr_ type_& operator*=(type_& x, const type_& y)                                      \
    {                                                                                      \
        x = type_(static_cast<float>(x) * static_cast<float>(y));                          \
        return x;                                                                          \
    }                                                                                      \
    attr_ type_& operator/=(type_& x, const type_& y)                                      \
    {                                                                                      \
        x = type_(static_cast<float>(x) / static_cast<float>(y));                          \
        return x;                                                                          \
    }                                                                                      \
    attr_ type_& operator++(type_& x)                                                      \
    {                                                                                      \
        x = type_(static_cast<float>(x) + 1.f);                                            \
        return x;                                                                          \
    }                                                                                      \
    attr_ type_& operator--(type_& x)                                                      \
    {                                                                                      \
        x = type_(static_cast<float>(x) - 1.f);                                            \
        return x;                                                                          \
    }                                                                                      \
    attr_ type_ operator++(type_& x, int)                                                  \
    {                                                                                      \
        type_ y(x);                                                                        \
        x = type_(static_cast<float>(x) + 1.f);                                            \
        return y;                                                                          \
    }                                                                                      \
    attr_ type_ operator--(type_& x, int)                                                  \
    {                                                                                      \
        type_ y(x);                                                                        \
        x = type_(static_cast<float>(x) - 1.f);                                            \
        return y;                                                                          \
    }
