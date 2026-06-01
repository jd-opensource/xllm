// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/numeric/numeric.hpp"
#include <hip/hip_fp16.h>

#pragma once

namespace ck_tile {

using fp16_hip_t = _Float16; // most of hip internal function use this type
using fp16_raw_t = uint16_t;

CK_TILE_HOST_DEVICE
constexpr float fp16_to_float_hip(const fp16_hip_t& x);

CK_TILE_HOST_DEVICE
constexpr double fp16_to_double_hip(const fp16_hip_t& x);

CK_TILE_HOST_DEVICE
constexpr fp16_hip_t float_to_fp16_hip(const float& x);

CK_TILE_HOST_DEVICE
constexpr fp16_hip_t double_to_fp16_hip(const double& x);

#if CK_TILE_USE_CUSTOM_DATA_TYPE
// HIP use fp16_hip_t as interchangable data type for float16
struct alignas(2) half_t
{
    using raw_type = fp16_raw_t;
    raw_type data;

    CK_TILE_HOST_DEVICE
    static constexpr half_t bit_cast(raw_type x)
    {
        half_t y;
        y.data = x;
        return y;
    }

    CK_TILE_HOST_DEVICE
    constexpr fp16_hip_t to_fp16() const { return ck_tile::bit_cast<fp16_hip_t>(data); }

    // constructor
    constexpr half_t() : data{} {}

    // construct from HIP half
    CK_TILE_HOST_DEVICE
    explicit constexpr half_t(const fp16_hip_t& x) : data(ck_tile::bit_cast<raw_type>(x)) {}

    // construct from float
    CK_TILE_HOST_DEVICE
    explicit constexpr half_t(const float& x) : half_t(float_to_fp16_hip(x)) {}

    // construct from double
    CK_TILE_HOST_DEVICE
    explicit constexpr half_t(const double& x) : half_t(double_to_fp16_hip(x)) {}

    // construct from int
    CK_TILE_HOST_DEVICE
    explicit constexpr half_t(const int& x) : half_t(static_cast<fp16_hip_t>(__int2half_rn(x))) {}

    // construct from unsigned int
    CK_TILE_HOST_DEVICE
    explicit constexpr half_t(const unsigned int& x)
        : half_t(static_cast<fp16_hip_t>(__uint2half_rn(x)))
    {
    }

    // cast to float
    CK_TILE_HOST_DEVICE
    explicit constexpr operator float() const { return fp16_to_float_hip(to_fp16()); }

    // cast to double
    CK_TILE_HOST_DEVICE
    explicit constexpr operator double() const { return fp16_to_double_hip(to_fp16()); }

    // cast to int
    CK_TILE_HOST_DEVICE
    explicit constexpr operator int() const
    {
        return static_cast<int>(fp16_to_float_hip(to_fp16()));
    }

    CK_TILE_HOST_DEVICE
    explicit constexpr operator fp16_hip_t() const { return ck_tile::bit_cast<fp16_hip_t>(data); }

    // internal access
    CK_TILE_HOST_DEVICE
    constexpr raw_type& get() { return data; }

    CK_TILE_HOST_DEVICE
    constexpr raw_type get() const { return data; }
};

template <typename>
struct native_t;

template <>
struct native_t<half_t>
{
    using type = _Float16;
};

using fp16_t     = half_t;
using fp16_raw_t = typename half_t::raw_type;
#else
using fp16_t     = _Float16;
using half_t     = _Float16;
using fp16_raw_t = ushort;
#endif

// conversions
CK_TILE_HOST_DEVICE
constexpr float fp16_to_float_hip(const fp16_hip_t& x)
{
    // return __half2float(x);
    return static_cast<float>(x);
}

CK_TILE_HOST_DEVICE
constexpr double fp16_to_double_hip(const fp16_hip_t& x)
{
    return static_cast<double>(fp16_to_float_hip(x));
}

CK_TILE_HOST_DEVICE
constexpr fp16_hip_t float_to_fp16_hip(const float& x)
{
    // return __float2half(x);
    return static_cast<fp16_hip_t>(x);
}

CK_TILE_HOST_DEVICE
constexpr fp16_hip_t double_to_fp16_hip(const double& x)
{
    // return __float2half(x);
    return static_cast<fp16_hip_t>(x);
}

CK_TILE_HOST_DEVICE
constexpr float fp16_to_float(const half_t& x) { return static_cast<float>(x); }

CK_TILE_HOST_DEVICE
constexpr float fp16_to_double(const half_t& x) { return static_cast<float>(x); }

CK_TILE_HOST_DEVICE
constexpr half_t float_to_fp16(const float& x) { return static_cast<half_t>(x); }

CK_TILE_HOST_DEVICE
constexpr half_t double_to_fp16(const double& x) { return static_cast<half_t>(x); }

// limits
template <class T>
struct numeric;

template <>
struct numeric<half_t>
{
    // minimum finite value, or minimum positive normalized value for float
    CK_TILE_HOST_DEVICE static constexpr half_t min()
    {
        return bit_cast<half_t>(static_cast<fp16_raw_t>(0x0400));
    }

    // minumum finite value
    CK_TILE_HOST_DEVICE static constexpr half_t lowest()
    {
        return bit_cast<half_t>(static_cast<fp16_raw_t>(0xFBFF));
    }

    // maximum finite value
    CK_TILE_HOST_DEVICE static constexpr half_t max()
    {
        return bit_cast<half_t>(static_cast<fp16_raw_t>(0x7BFF));
    }

    // difference between 1.0 and next value representable by float
    CK_TILE_HOST_DEVICE static constexpr half_t epsilon()
    {
        return bit_cast<half_t>(static_cast<fp16_raw_t>(0x1800));
    }

    // maximum rounding error
    // bin :  f edcba 9876543210
    // bits:  s eeeee mmmmmmmmmm
    //        0 01110 0000000000 (0.5)
    //
    CK_TILE_HOST_DEVICE static constexpr half_t round_error()
    {
        return bit_cast<half_t>(static_cast<fp16_raw_t>(0x3800));
    }

    // positive infinity value
    CK_TILE_HOST_DEVICE static constexpr half_t infinity()
    {
        return bit_cast<half_t>(static_cast<fp16_raw_t>(0x7C00));
    }

    // quiet NaN
    CK_TILE_HOST_DEVICE static constexpr half_t quiet_NaN()
    {
        return bit_cast<half_t>(static_cast<fp16_raw_t>(0x7FFF));
    }

    // signaling NaN
    CK_TILE_HOST_DEVICE static constexpr half_t signaling_NaN()
    {
        return bit_cast<half_t>(static_cast<fp16_raw_t>(0x7FFF));
    }

    // smallest positive subnormal value
    CK_TILE_HOST_DEVICE static constexpr half_t denorm_min()
    {
        return bit_cast<half_t>(static_cast<fp16_raw_t>(0x0001));
    }

    CK_TILE_HOST_DEVICE static constexpr half_t zero()
    {
        return bit_cast<half_t>(static_cast<fp16_raw_t>(0));
    }
};

template <>
struct numeric_traits<half_t>
{
    static constexpr int exp            = 5;
    static constexpr int mant           = 10;
    static constexpr int bias           = 15;
    static constexpr uint16_t nan_mask  = 0x7C00;
    static constexpr uint16_t head_mask = 0xFC00;
    static constexpr uint16_t mant_mask = 0x3FF;
    static constexpr uint16_t exp_mask  = 0x1F;
    static constexpr uint16_t abs_mask  = 0x7FFF;
    static constexpr uint16_t Inf       = 0x7C00;
    static constexpr uint16_t NegInf    = 0xFC00;
    static constexpr uint16_t NaN       = 0x7C01;
    static constexpr uint16_t Neg0      = 0x8000;
    static constexpr int PackedSize     = 1;
    using bitwise_type                  = uint16_t;
};

#if CK_TILE_USE_CUSTOM_DATA_TYPE
// arithmetic
CK_TILE_DEVICE bool operator==(const half_t& x, const half_t& y)
{
    return __heq(x.to_fp16(), y.to_fp16());
}

CK_TILE_DEVICE
bool operator!=(const half_t& x, const half_t& y) { return __hne(x.to_fp16(), y.to_fp16()); }

CK_TILE_DEVICE
bool operator<(const half_t& x, const half_t& y) { return __hlt(x.to_fp16(), y.to_fp16()); }

CK_TILE_DEVICE
bool operator<=(const half_t& x, const half_t& y) { return __hle(x.to_fp16(), y.to_fp16()); }

CK_TILE_DEVICE
bool operator>(const half_t& x, const half_t& y) { return __hgt(x.to_fp16(), y.to_fp16()); }

CK_TILE_DEVICE
bool operator>=(const half_t& x, const half_t& y) { return __hge(x.to_fp16(), y.to_fp16()); }

#if 0
CK_TILE_DEVICE
half_t operator+(const half_t& x, const half_t& y)
{
    return half_t(__hadd(x.to_fp16(), y.to_fp16()));
}

CK_TILE_DEVICE
half_t operator-(const half_t& x) { return half_t(__hneg(x.to_fp16())); }

CK_TILE_DEVICE
half_t operator-(const half_t& x, const half_t& y)
{
    return half_t(__hsub(x.to_fp16(), y.to_fp16()));
}

CK_TILE_DEVICE
half_t operator*(const half_t& x, const half_t& y)
{
    return half_t(__hmul(x.to_fp16(), y.to_fp16()));
}

CK_TILE_DEVICE
half_t operator/(const half_t& x, const half_t& y)
{
    return half_t(__hdiv(x.to_fp16(), y.to_fp16()));
}

CK_TILE_DEVICE
half_t& operator+=(half_t& x, const half_t& y)
{
    x = half_t(__hadd(x.to_fp16(), y.to_fp16()));
    return x;
}

CK_TILE_DEVICE
half_t& operator-=(half_t& x, const half_t& y)
{
    x = half_t(__hsub(x.to_fp16(), y.to_fp16()));
    return x;
}

CK_TILE_DEVICE
half_t& operator*=(half_t& x, const half_t& y)
{
    x = half_t(__hmul(x.to_fp16(), y.to_fp16()));
    return x;
}

CK_TILE_DEVICE
half_t& operator/=(half_t& x, const half_t& y)
{
    x = half_t(__hdiv(x.to_fp16(), y.to_fp16()));
    return x;
}

CK_TILE_DEVICE
half_t& operator++(half_t& x)
{
    x = half_t(__hadd(x.to_fp16(), half_t(1.0f).to_fp16()));
    return x;
}

CK_TILE_DEVICE
half_t& operator--(half_t& x)
{
    x = half_t(__hsub(x.to_fp16(), half_t(1.0f).to_fp16()));
    return x;
}

CK_TILE_DEVICE
half_t operator++(half_t& x, int)
{
    half_t y(x);
    x = half_t(__hadd(x.to_fp16(), half_t(1.0f).to_fp16()));
    return y;
}

CK_TILE_DEVICE
half_t operator--(half_t& x, int)
{
    half_t y(x);
    x = half_t(__hsub(x.to_fp16(), half_t(1.0f).to_fp16()));
    return y;
}
#endif

#if CK_TILE_USE_CUSTOM_DATA_TYPE
CK_TILE_ARITHMETIC_USING_FLOAT(CK_TILE_HOST, half_t)
#endif

// math
CK_TILE_HOST_DEVICE
half_t abs(const half_t& x) { return bit_cast<half_t>(x.get() & 0x7fff); }

CK_TILE_HOST_DEVICE
bool isnan(const half_t& x)
{
    uint16_t xx = x.get();
    return (xx & 0x7FFF) > 0x7C00;
}

CK_TILE_DEVICE
half_t sqrt(half_t x)
{
    return static_cast<half_t>(__builtin_amdgcn_sqrtf(static_cast<float>(x)));
};

CK_TILE_DEVICE
half_t exp(half_t x) { return static_cast<half_t>(__ocml_exp_f32(static_cast<float>(x))); };

CK_TILE_DEVICE
half_t exp2(half_t x) { return static_cast<half_t>(exp2f(static_cast<float>(x))); };

CK_TILE_DEVICE
half_t log(half_t x) { return static_cast<half_t>(__logf(static_cast<float>(x))); };
#endif

using fp16x2_t = _Float16 __attribute__((ext_vector_type(2)));

CK_TILE_HOST fp16x2_t pk_add_f16(const fp16x2_t& x, const fp16x2_t& y)
{
    fp16x2_t vector_res;

    vector_res.x = x.x + y.x;
    vector_res.y = x.y + y.y;

    return vector_res;
}

CK_TILE_DEVICE fp16x2_t pk_add_f16(const fp16x2_t& x, const fp16x2_t& y)
{
    fp16x2_t c;
    asm volatile("v_pk_add_f16 %0, %1, %2" : "=v"(c) : "v"(x), "v"(y));
    return c;
}

} // namespace ck_tile
