// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include <type_traits>
#include <stdint.h>
#include <cmath>

namespace ck_tile {

template <typename Scale, Scale lhs>
struct scales_c
{
    template <typename Right>
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Right& rhs) const -> decltype(lhs * rhs)
    {
        return lhs * rhs;
    }
};

template <typename Scale>
struct scales
{
    static_assert(std::is_copy_constructible_v<Scale>);

    CK_TILE_HOST_DEVICE constexpr explicit scales(Scale lhs) : lhs_(lhs) {}

    template <typename Right>
    CK_TILE_HOST_DEVICE constexpr auto
    operator()(const Right& rhs) const -> decltype(std::declval<const Scale&>() * rhs)
    {
        return lhs_ * rhs;
    }

    private:
    Scale lhs_;
};

/// FIXME: create macro to replace '__host__ __device__' and nothing more
template <typename Scale>
__host__ __device__ scales(Scale) -> scales<Scale>;

template <typename Left = void, typename Right = Left>
struct plus
{
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs,
                                                  const Right& rhs) const -> decltype(lhs + rhs)
    {
        return lhs + rhs;
    }
};

template <>
struct plus<void, void>
{
    template <typename Left, typename Right>
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs,
                                                  const Right& rhs) const -> decltype(lhs + rhs)
    {
        return lhs + rhs;
    }
};

/// FIXME: create macro to replace '__host__ __device__' and nothing more
__host__ __device__ plus() -> plus<void, void>;

template <typename Left = void, typename Right = Left>
struct minus
{
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs,
                                                  const Right& rhs) const -> decltype(lhs - rhs)
    {
        return lhs - rhs;
    }
};

template <>
struct minus<void, void>
{
    template <typename Left, typename Right>
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs,
                                                  const Right& rhs) const -> decltype(lhs - rhs)
    {
        return lhs - rhs;
    }
};

/// FIXME: create macro to replace '__host__ __device__' and nothing more
__host__ __device__ minus() -> minus<void, void>;

template <typename Left = void, typename Right = Left>
struct multiplies
{
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs,
                                                  const Right& rhs) const -> decltype(lhs * rhs)
    {
        return lhs * rhs;
    }
};

template <>
struct multiplies<void, void>
{
    template <typename Left, typename Right>
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs,
                                                  const Right& rhs) const -> decltype(lhs * rhs)
    {
        return lhs * rhs;
    }
};

/// FIXME: create macro to replace '__host__ __device__' and nothing more
__host__ __device__ multiplies() -> multiplies<void, void>;

template <typename T>
struct maximize
{
    CK_TILE_HOST_DEVICE constexpr T operator()(T a, T b) const { return a >= b ? a : b; }
};

template <typename T>
struct minimize
{
    CK_TILE_HOST_DEVICE constexpr T operator()(T a, T b) const { return a <= b ? a : b; }
};

template <typename T>
struct integer_divide_ceiler
{
    CK_TILE_HOST_DEVICE constexpr T operator()(T a, T b) const
    {
        static_assert(std::is_same<T, index_t>{} || std::is_same<T, int>{}, "wrong type");
        return (a + b - number<1>{}) / b;
    }
};

template <typename X, typename Y>
CK_TILE_HOST_DEVICE constexpr auto integer_divide_floor(X x, Y y)
{
    return x / y;
}

template <typename X, typename Y>
CK_TILE_HOST_DEVICE constexpr auto integer_divide_ceil(X x, Y y)
{
    return (x + y - number<1>{}) / y;
}

template <typename X, typename Y>
CK_TILE_HOST_DEVICE constexpr auto integer_least_multiple(X x, Y y)
{
    return y * integer_divide_ceil(x, y);
}

template <typename T>
CK_TILE_HOST_DEVICE constexpr T max(T x)
{
    return x;
}

template <typename T>
CK_TILE_HOST constexpr T max(T x, T y)
{
    return x > y ? x : y;
}

template <typename T>
CK_TILE_DEVICE constexpr T max(T x, T y)
{
    return x > y ? x : y;
}

template <>
CK_TILE_DEVICE constexpr float max(float x, float y)
{
    return __builtin_fmaxf(x, y); // can resultin v_max3_f32
}

template <>
CK_TILE_DEVICE constexpr double max(double x, double y)
{
    return __builtin_fmax(x, y); // maybe still v_max3_f32
}

template <index_t X>
CK_TILE_HOST_DEVICE constexpr index_t max(number<X>, index_t y)
{
    return X > y ? X : y;
}

template <index_t Y>
CK_TILE_HOST_DEVICE constexpr index_t max(index_t x, number<Y>)
{
    return x > Y ? x : Y;
}

template <typename X, typename... Ys>
CK_TILE_HOST_DEVICE constexpr auto max(X x, Ys... ys)
{
    static_assert(sizeof...(Ys) > 0, "not enough argument");
    return max(x, max(ys...));
}

template <typename T>
CK_TILE_HOST_DEVICE constexpr T min(T x)
{
    return x;
}

template <typename T>
CK_TILE_HOST constexpr T min(T x, T y)
{
    return x < y ? x : y;
}

template <typename T>
CK_TILE_DEVICE constexpr T min(T x, T y)
{
    return x < y ? x : y;
}

template <>
CK_TILE_DEVICE constexpr float min(float x, float y)
{
    return __builtin_fminf(x, y);
}

template <>
CK_TILE_DEVICE constexpr double min(double x, double y)
{
    return __builtin_fmin(x, y);
}

template <index_t X>
CK_TILE_HOST_DEVICE constexpr index_t min(number<X>, index_t y)
{
    return X < y ? X : y;
}

template <index_t Y>
CK_TILE_HOST_DEVICE constexpr index_t min(index_t x, number<Y>)
{
    return x < Y ? x : Y;
}

template <typename X, typename... Ys>
CK_TILE_HOST_DEVICE constexpr auto min(X x, Ys... ys)
{
    static_assert(sizeof...(Ys) > 0, "not enough argument");
    return min(x, min(ys...));
}

template <typename T>
CK_TILE_HOST_DEVICE constexpr T clamp(const T& x, const T& lowerbound, const T& upperbound)
{
    return min(max(x, lowerbound), upperbound);
}

CK_TILE_HOST int clz(uint32_t x) { return __builtin_clz(x); }
CK_TILE_DEVICE int clz(uint32_t x) { return __clz(x); }

// greatest common divisor, aka highest common factor
CK_TILE_HOST_DEVICE constexpr index_t gcd(index_t x, index_t y)
{
    if(x < 0)
    {
        return gcd(-x, y);
    }
    else if(y < 0)
    {
        return gcd(x, -y);
    }
    else if(x == y || x == 0)
    {
        return y;
    }
    else if(y == 0)
    {
        return x;
    }
    else if(x > y)
    {
        return gcd(x % y, y);
    }
    else
    {
        return gcd(x, y % x);
    }
}

template <index_t X, index_t Y>
CK_TILE_HOST_DEVICE constexpr auto gcd(number<X>, number<Y>)
{
    constexpr auto r = gcd(X, Y);

    return number<r>{};
}

template <typename X,
          typename... Ys,
          typename std::enable_if<sizeof...(Ys) >= 2, bool>::type = false>
CK_TILE_HOST_DEVICE constexpr auto gcd(X x, Ys... ys)
{
    return gcd(x, gcd(ys...));
}

// least common multiple
template <typename X, typename Y>
CK_TILE_HOST_DEVICE constexpr auto lcm(X x, Y y)
{
    return (x * y) / gcd(x, y);
}

template <typename X,
          typename... Ys,
          typename std::enable_if<sizeof...(Ys) >= 2, bool>::type = false>
CK_TILE_HOST_DEVICE constexpr auto lcm(X x, Ys... ys)
{
    return lcm(x, lcm(ys...));
}

template <typename Left = void, typename Right = Left>
struct equal
{
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs,
                                                  const Right& rhs) const -> decltype(lhs == rhs)
    {
        return lhs == rhs;
    }
};

template <>
struct equal<void, void>
{
    template <typename Left, typename Right>
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs,
                                                  const Right& rhs) const -> decltype(lhs == rhs)
    {
        return lhs == rhs;
    }
};

/// FIXME: create macro to replace '__host__ __device__' and nothing more
__host__ __device__ equal() -> equal<void, void>;

template <>
struct equal<float, float>
{
    CK_TILE_HOST_DEVICE constexpr bool operator()(float lhs, float rhs) const
    {
        return bit_cast<uint32_t>(lhs) == bit_cast<uint32_t>(rhs);
    }
};

template <>
struct equal<double, double>
{
    CK_TILE_HOST_DEVICE constexpr bool operator()(double lhs, double rhs) const
    {
        return bit_cast<uint64_t>(lhs) == bit_cast<uint64_t>(rhs);
    }
};

template <typename Left = void, typename Right = Left>
struct less
{
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs,
                                                  const Right& rhs) const -> decltype(lhs < rhs)
    {
        return lhs < rhs;
    }
};

template <>
struct less<void, void>
{
    template <typename Left, typename Right>
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs,
                                                  const Right& rhs) const -> decltype(lhs < rhs)
    {
        return lhs < rhs;
    }
};

/// FIXME: create macro to replace '__host__ __device__' and nothing more
__host__ __device__ less() -> less<void, void>;

template <typename Left = void, typename Right = Left>
struct less_equal
{
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs,
                                                  const Right& rhs) const -> decltype(lhs <= rhs)
    {
        return lhs <= rhs;
    }
};

template <>
struct less_equal<void, void>
{
    template <typename Left, typename Right>
    CK_TILE_HOST_DEVICE constexpr auto operator()(const Left& lhs,
                                                  const Right& rhs) const -> decltype(lhs <= rhs)
    {
        return lhs <= rhs;
    }
};

/// FIXME: create macro to replace '__host__ __device__' and nothing more
__host__ __device__ less_equal() -> less_equal<void, void>;

template <>
struct less_equal<float, float>
{
    CK_TILE_HOST_DEVICE constexpr bool operator()(float lhs, float rhs) const
    {
        return lhs < rhs || bit_cast<uint32_t>(lhs) == bit_cast<uint32_t>(rhs);
    }
};

template <>
struct less_equal<double, double>
{
    CK_TILE_HOST_DEVICE constexpr bool operator()(double lhs, double rhs) const
    {
        return lhs < rhs || bit_cast<uint64_t>(lhs) == bit_cast<uint64_t>(rhs);
    }
};

CK_TILE_HOST_DEVICE constexpr int32_t next_power_of_two(int32_t x)
{
    // TODO: x need to be 2 ~ 0x7fffffff. 0, 1, or larger than 0x7fffffff will compile fail
    return 1 << (32 - clz(x - 1));
}

template <index_t X>
CK_TILE_HOST_DEVICE constexpr auto next_power_of_two()
{
    constexpr index_t y = next_power_of_two(X);
    return number<y>{};
}

template <index_t X>
CK_TILE_HOST_DEVICE constexpr auto next_power_of_two(number<X>)
{
    constexpr index_t y = next_power_of_two(X);
    return number<y>{};
}

CK_TILE_HOST_DEVICE constexpr int32_t integer_log2_floor(int32_t x)
{
    // TODO: x need to be 1 ~ 0x7fffffff
    // __builtin_clz will produce unexpected result if x is 0;
    return 31 - __builtin_clz(x);
}

CK_TILE_HOST_DEVICE constexpr bool is_power_of_two_integer(int32_t x)
{
    // TODO: x need to be 1 ~ 0x7fffffff
    return x == (1 << integer_log2_floor(x));
}

#ifndef C_LOG2E
#define C_LOG2E 1.44269504088896340736 // log2(e)
#endif

template <typename T>
struct log2e;

template <>
struct log2e<double>
{
    static constexpr double value = C_LOG2E;
};

template <>
struct log2e<float>
{
    static constexpr float value = C_LOG2E;
};

template <typename T = double>
constexpr T log2e_v = log2e<T>::value;

template <typename T = double>
constexpr T log2e_rcp_v = 1. / log2e<T>::value;

CK_TILE_DEVICE
float exp2(float x) { return exp2f(x); };

CK_TILE_HOST
float exp2(float x) { return std::exp2f(x); };

CK_TILE_DEVICE uint16_t sad_u16(uint16_t x, uint16_t y, uint16_t acc)
{
    return __builtin_amdgcn_sad_u16(x, y, acc);
}

CK_TILE_DEVICE uint32_t sad_u32(uint32_t x, uint32_t y, uint32_t acc)
{
    /// TODO: replace inline asm when intrinsic is available
    uint32_t res;
    asm volatile("v_sad_u32 %0, %1, %2, %3" : "=v"(res) : "v"(x), "v"(y), "v"(acc));
    return res;
}

CK_TILE_HOST uint32_t sad_u32(uint32_t x, uint32_t y, uint32_t acc)
{
    return (x > y ? (x - y) : (y - x)) + acc;
}

///////////////////////////////////////////////////////////////

} // namespace ck_tile
// blow function need data type pre-defined
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/type_convert.hpp"
#ifndef __HIP_DEVICE_COMPILE__
#include <cmath>
#endif

namespace ck_tile {
#if CK_TILE_WORKAROUND_SWDEV_383542
extern "C" CK_TILE_DEVICE float __ocml_native_recip_f32(float);
#endif

// math functions for the host,  some are implemented by calling C++ std functions

CK_TILE_HOST float abs(float x) { return std::abs(x); };

CK_TILE_HOST double abs(double x) { return std::abs(x); };

CK_TILE_HOST int8_t abs(int8_t x)
{
    int8_t sgn = x >> (8 - 1);

    return (x ^ sgn) - sgn;
};

CK_TILE_HOST int32_t abs(int32_t x)
{
    int32_t sgn = x >> (32 - 1);

    return (x ^ sgn) - sgn;
};

CK_TILE_HOST fp16_t abs(fp16_t x)
{
    uint16_t xx = bit_cast<uint16_t>(x);

    uint16_t abs_xx = xx & 0x7fff;

    fp16_t abs_x = bit_cast<fp16_t>(abs_xx);

    return abs_x;
};

#ifdef CK_TILE_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
CK_TILE_HOST int4_t abs(int4_t x)
{
    int4_t sgn = x >> (4 - 1);
    return (x ^ sgn) - sgn;
}
#endif

CK_TILE_HOST bool isnan(float x) { return std::isnan(x); };

CK_TILE_HOST bool isnan(double x) { return std::isnan(x); };

CK_TILE_HOST bool isnan(int8_t x)
{
    (void)x;
    return false;
};

CK_TILE_HOST bool isnan(int32_t x)
{
    (void)x;
    return false;
};

CK_TILE_HOST bool isnan(fp16_t x)
{
    uint16_t xx = bit_cast<uint16_t>(x);

    return (xx & 0x7FFF) > 0x7C00;
};

#ifdef CK_TILE_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
CK_TILE_HOST bool isnan(int4_t x)
{
    (void)x;
    return false;
};
#endif

CK_TILE_HOST fp16_t sqrt(fp16_t x)
{
    return static_cast<fp16_t>(std::sqrt(static_cast<float>(x)));
};

CK_TILE_HOST float sqrt(float x) { return std::sqrt(x); };

CK_TILE_HOST double sqrt(double x) { return std::sqrt(x); };

template <typename T>
CK_TILE_HOST T tanh(T x)
{
    return type_convert<T>(std::tanhf(type_convert<float>(x)));
};

template <>
CK_TILE_HOST float tanh<float>(float x)
{
    return std::tanhf(x);
};

template <>
CK_TILE_HOST double tanh<double>(double x)
{
    return std::tanh(x);
};

template <typename T>
CK_TILE_HOST T acos(T x)
{
    return type_convert<T>(std::acosf(type_convert<float>(x)));
};

template <>
CK_TILE_HOST float acos<float>(float x)
{
    return std::acosf(x);
};

template <>
CK_TILE_HOST double acos<double>(double x)
{
    return std::acos(x);
};

template <typename T>
CK_TILE_HOST T neg(T x)
{
    return type_convert<T>(-(type_convert<float>(x)));
};

template <>
CK_TILE_HOST float neg<float>(float x)
{
    return -x;
};

template <>
CK_TILE_HOST double neg<double>(double x)
{
    return -x;
};

template <>
CK_TILE_HOST int32_t neg<int32_t>(int32_t x)
{
    return -x;
};

template <>
CK_TILE_HOST int8_t neg<int8_t>(int8_t x)
{
    return -x;
};

template <typename T>
CK_TILE_HOST T atan(T x)
{
    return type_convert<T>(std::atanf(type_convert<float>(x)));
};

template <>
CK_TILE_HOST float atan<float>(float x)
{
    return std::atanf(x);
};

template <>
CK_TILE_HOST double atan<double>(double x)
{
    return std::atan(x);
};

template <typename T>
CK_TILE_HOST T sin(T x)
{
    return type_convert<T>(std::sinf(type_convert<float>(x)));
};

template <>
CK_TILE_HOST float sin<float>(float x)
{
    return std::sinf(x);
};

template <>
CK_TILE_HOST double sin<double>(double x)
{
    return std::sin(x);
};

template <typename T>
CK_TILE_HOST T asin(T x)
{
    return type_convert<T>(std::asinf(type_convert<float>(x)));
};

template <>
CK_TILE_HOST float asin<float>(float x)
{
    return std::asinf(x);
};

template <>
CK_TILE_HOST double asin<double>(double x)
{
    return std::asin(x);
};

template <typename T>
CK_TILE_HOST T asinh(T x)
{
    return type_convert<T>(std::asinhf(type_convert<float>(x)));
};

template <>
CK_TILE_HOST float asinh<float>(float x)
{
    return std::asinhf(x);
};

template <>
CK_TILE_HOST double asinh<double>(double x)
{
    return std::asinh(x);
};

template <typename T>
CK_TILE_HOST T cos(T x)
{
    return type_convert<T>(std::cosf(type_convert<float>(x)));
};

template <>
CK_TILE_HOST float cos<float>(float x)
{
    return std::cosf(x);
};

template <>
CK_TILE_HOST double cos<double>(double x)
{
    return std::cos(x);
};

template <typename T>
CK_TILE_HOST T acosh(T x)
{
    return type_convert<T>(std::acoshf(type_convert<float>(x)));
};

template <>
CK_TILE_HOST float acosh<float>(float x)
{
    return std::acoshf(x);
};

template <>
CK_TILE_HOST double acosh<double>(double x)
{
    return std::acosh(x);
};

template <typename T>
CK_TILE_HOST T tan(T x)
{
    return type_convert<T>(std::tanf(type_convert<float>(x)));
};

template <>
CK_TILE_HOST float tan<float>(float x)
{
    return std::tanf(x);
};

template <>
CK_TILE_HOST double tan<double>(double x)
{
    return std::tan(x);
};

template <typename T>
CK_TILE_HOST T atanh(T x)
{
    return type_convert<T>(std::atanhf(type_convert<float>(x)));
};

template <>
CK_TILE_HOST float atanh<float>(float x)
{
    return std::atanhf(x);
};

template <>
CK_TILE_HOST double atanh<double>(double x)
{
    return std::atanh(x);
};

template <typename T>
CK_TILE_HOST T sinh(T x)
{
    return type_convert<T>(std::sinhf(type_convert<float>(x)));
};

template <>
CK_TILE_HOST float sinh<float>(float x)
{
    return std::sinhf(x);
};

template <>
CK_TILE_HOST double sinh<double>(double x)
{
    return std::sinh(x);
};

template <typename T>
CK_TILE_HOST T ceil(T x)
{
    return type_convert<T>(std::ceilf(type_convert<float>(x)));
};

template <>
CK_TILE_HOST float ceil<float>(float x)
{
    return std::ceilf(x);
};

template <>
CK_TILE_HOST double ceil<double>(double x)
{
    return std::ceil(x);
};

template <typename T>
CK_TILE_HOST T cosh(T x)
{
    return type_convert<T>(std::coshf(type_convert<float>(x)));
};

template <>
CK_TILE_HOST float cosh<float>(float x)
{
    return std::coshf(x);
};

template <>
CK_TILE_HOST double cosh<double>(double x)
{
    return std::cosh(x);
};

template <typename T>
CK_TILE_HOST T floor(T x)
{
    return type_convert<T>(std::floorf(type_convert<float>(x)));
};

template <>
CK_TILE_HOST float floor<float>(float x)
{
    return std::floorf(x);
};

template <>
CK_TILE_HOST double floor<double>(double x)
{
    return std::floor(x);
};

template <typename T>
CK_TILE_HOST T rcp(T x)
{
    return type_convert<T>(1.f / type_convert<float>(x));
};

template <typename T>
CK_TILE_HOST T exp(T x)
{
    return type_convert<T>(std::expf(type_convert<float>(x)));
}

template <>
CK_TILE_HOST float exp<float>(float x)
{
    return std::expf(x);
}

template <>
CK_TILE_HOST double exp<double>(double x)
{
    return std::exp(x);
}

template <typename T>
CK_TILE_HOST T log(T x)
{
    return type_convert<T>(std::logf(type_convert<float>(x)));
}

template <>
CK_TILE_HOST float log<float>(float x)
{
    return std::logf(x);
}

template <>
CK_TILE_HOST double log<double>(double x)
{
    return std::log(x);
}

template <typename T>
CK_TILE_HOST T pow(T x, T gamma)
{
    return type_convert<T>(std::powf(type_convert<float>(x), type_convert<float>(gamma)));
}

template <>
CK_TILE_HOST float pow<float>(float x, float gamma)
{
    return std::powf(x, gamma);
}

template <>
CK_TILE_HOST double pow<double>(double x, double gamma)
{
    return std::pow(x, gamma);
}

template <typename T>
CK_TILE_HOST T expm1(T x)
{
    return type_convert<T>(std::expm1f(type_convert<float>(x)));
}

template <>
CK_TILE_HOST float expm1<float>(float x)
{
    return std::expm1f(x);
}

template <>
CK_TILE_HOST double expm1<double>(double x)
{
    return std::expm1(x);
}

// math functions for the HIP kernel,  some are implemented by calling hip builtin functions

CK_TILE_DEVICE float abs(float x)
{
    union
    {
        float f32;
        uint32_t u32;
    } y;
    y.f32 = x;
    y.u32 = y.u32 & 0x7fffffff;
    return y.f32;
};

CK_TILE_DEVICE double abs(double x) { return ::abs(x); };

CK_TILE_DEVICE int8_t abs(int8_t x)
{
    int8_t sgn = x >> (8 - 1);

    return (x ^ sgn) - sgn;
};

CK_TILE_DEVICE int32_t abs(int32_t x)
{
    int32_t sgn = x >> (32 - 1);

    return (x ^ sgn) - sgn;
};

#ifdef CK_TILE_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
CK_TILE_DEVICE int4_t abs(int4_t x)
{
    int4_t sgn = x >> (4 - 1);

    return (x ^ sgn) - sgn;
};
#endif

CK_TILE_DEVICE fp16_t abs(fp16_t x)
{
    uint16_t xx = bit_cast<uint16_t>(x);

    uint16_t abs_xx = xx & 0x7fff;

    fp16_t abs_x = bit_cast<fp16_t>(abs_xx);

    return abs_x;
};

CK_TILE_DEVICE bool isnan(float x) { return ::isnan(x); };

CK_TILE_DEVICE bool isnan(double x) { return ::isnan(x); };

CK_TILE_DEVICE bool isnan(int8_t x)
{
    (void)x;
    return false;
};

CK_TILE_DEVICE bool isnan(int32_t x)
{
    (void)x;
    return false;
};

#ifdef CK_TILE_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
CK_TILE_DEVICE bool isnan(int4_t x)
{
    (void)x;
    return false;
};
#endif

CK_TILE_DEVICE bool isnan(fp16_t x)
{
    uint16_t xx = bit_cast<uint16_t>(x);

    return (xx & 0x7FFF) > 0x7C00;
};

CK_TILE_DEVICE fp16_t sqrt(fp16_t x)
{
    return static_cast<fp16_t>(__builtin_amdgcn_sqrtf(static_cast<float>(x)));
};

CK_TILE_DEVICE float sqrt(float x) { return __builtin_amdgcn_sqrtf(x); };

CK_TILE_DEVICE double sqrt(double x) { return __builtin_amdgcn_sqrt(x); };

template <typename T>
CK_TILE_DEVICE T tanh(T x)
{
    return type_convert<T>(::tanhf(type_convert<float>(x)));
};

template <>
CK_TILE_DEVICE float tanh<float>(float x)
{
    return ::tanhf(x);
};

template <>
CK_TILE_DEVICE double tanh<double>(double x)
{
    return ::tanh(x);
};

template <typename T>
CK_TILE_DEVICE T acos(T x)
{
    return type_convert<T>(::acosf(type_convert<float>(x)));
};

template <>
CK_TILE_DEVICE float acos<float>(float x)
{
    return ::acosf(x);
};

template <>
CK_TILE_DEVICE double acos<double>(double x)
{
    return ::acos(x);
};

template <typename T>
CK_TILE_DEVICE T neg(T x)
{
    return type_convert<T>(-(type_convert<float>(x)));
};

template <>
CK_TILE_DEVICE float neg<float>(float x)
{
    return -x;
};

template <>
CK_TILE_DEVICE double neg<double>(double x)
{
    return -x;
};

template <>
CK_TILE_DEVICE int32_t neg<int32_t>(int32_t x)
{
    return -x;
};

template <>
CK_TILE_DEVICE int8_t neg<int8_t>(int8_t x)
{
    return -x;
};

template <>
CK_TILE_DEVICE fp16_t neg<fp16_t>(fp16_t x)
{
    return -x;
};

template <typename T>
CK_TILE_DEVICE T atan(T x)
{
    return type_convert<T>(::atanf(type_convert<float>(x)));
};

template <>
CK_TILE_DEVICE float atan<float>(float x)
{
    return ::atanf(x);
};

template <>
CK_TILE_DEVICE double atan<double>(double x)
{
    return ::atan(x);
};

template <typename T>
CK_TILE_DEVICE T sin(T x)
{
    return type_convert<T>(::sinf(type_convert<float>(x)));
};

template <>
CK_TILE_DEVICE float sin<float>(float x)
{
    return ::sinf(x);
};

template <>
CK_TILE_DEVICE double sin<double>(double x)
{
    return ::sin(x);
};

template <>
CK_TILE_DEVICE fp16_t sin<fp16_t>(fp16_t x)
{
    return __ocml_sin_f16(x);
};

template <typename T>
CK_TILE_DEVICE T asin(T x)
{
    return type_convert<T>(::asinf(type_convert<float>(x)));
};

template <>
CK_TILE_DEVICE float asin<float>(float x)
{
    return ::asinf(x);
};

template <>
CK_TILE_DEVICE double asin<double>(double x)
{
    return ::asin(x);
};

template <typename T>
CK_TILE_DEVICE T asinh(T x)
{
    return type_convert<T>(::asinhf(type_convert<float>(x)));
};

template <>
CK_TILE_DEVICE float asinh<float>(float x)
{
    return ::asinhf(x);
};

template <>
CK_TILE_DEVICE double asinh<double>(double x)
{
    return ::asinh(x);
};

template <typename T>
CK_TILE_DEVICE T acosh(T x)
{
    return type_convert<T>(::acoshf(type_convert<float>(x)));
};

template <>
CK_TILE_DEVICE float acosh<float>(float x)
{
    return ::acoshf(x);
};

template <>
CK_TILE_DEVICE double acosh<double>(double x)
{
    return ::acosh(x);
};

template <typename T>
CK_TILE_DEVICE T tan(T x)
{
    return type_convert<T>(::tanf(type_convert<float>(x)));
};

template <>
CK_TILE_DEVICE float tan<float>(float x)
{
    return ::tanf(x);
};

template <>
CK_TILE_DEVICE double tan<double>(double x)
{
    return ::tan(x);
};

template <typename T>
CK_TILE_DEVICE T atanh(T x)
{
    return type_convert<T>(::atanhf(type_convert<float>(x)));
};

template <>
CK_TILE_DEVICE float atanh<float>(float x)
{
    return ::atanhf(x);
};

template <>
CK_TILE_DEVICE double atanh<double>(double x)
{
    return ::atanh(x);
};

template <typename T>
CK_TILE_DEVICE T sinh(T x)
{
    return type_convert<T>(::sinhf(type_convert<float>(x)));
};

template <>
CK_TILE_DEVICE float sinh<float>(float x)
{
    return ::sinhf(x);
};

template <>
CK_TILE_DEVICE double sinh<double>(double x)
{
    return ::sinh(x);
};

template <typename T>
CK_TILE_DEVICE T ceil(T x)
{
    return type_convert<T>(::ceilf(type_convert<float>(x)));
};

template <>
CK_TILE_DEVICE float ceil<float>(float x)
{
    return ::ceilf(x);
};

template <>
CK_TILE_DEVICE double ceil<double>(double x)
{
    return ::ceil(x);
};

template <>
CK_TILE_DEVICE fp16_t ceil<fp16_t>(fp16_t x)
{
    return __ocml_ceil_f16(x);
};

template <typename T>
CK_TILE_DEVICE T cosh(T x)
{
    return type_convert<T>(::coshf(type_convert<float>(x)));
};

template <>
CK_TILE_DEVICE float cosh<float>(float x)
{
    return ::coshf(x);
};

template <>
CK_TILE_DEVICE double cosh<double>(double x)
{
    return ::cosh(x);
};

template <typename T>
CK_TILE_DEVICE T floor(T x)
{
    return type_convert<T>(::floorf(type_convert<float>(x)));
};

template <>
CK_TILE_DEVICE float floor<float>(float x)
{
    return ::floorf(x);
};

template <>
CK_TILE_DEVICE double floor<double>(double x)
{
    return ::floor(x);
};

template <>
CK_TILE_DEVICE fp16_t floor<fp16_t>(fp16_t x)
{
    return __ocml_floor_f16(x);
};

template <typename T>
CK_TILE_DEVICE T rcp(T x)
{
#if !CK_TILE_WORKAROUND_SWDEV_383542
    return __frcp_rn(x);
#else
    // return __ocml_native_recip_f32(x);
    return __builtin_amdgcn_rcpf(x);
#endif
};

template <typename T>
CK_TILE_DEVICE T exp(T x)
{
    return type_convert<T>(__ocml_exp_f32(type_convert<float>(x)));
};

template <>
CK_TILE_DEVICE fp16_t exp<fp16_t>(fp16_t x)
{
    return __ocml_exp_f16(x);
};

template <>
CK_TILE_DEVICE float exp<float>(float x)
{
    return __ocml_exp_f32(x);
};

template <>
CK_TILE_DEVICE double exp<double>(double x)
{
    return exp(x);
};

template <typename T>
CK_TILE_DEVICE T tanh_fast(T x)
{
    return type_convert<T>((exp<T>(2.0 * type_convert<float>(x)) - 1.0) /
                           (exp<T>(2.0 * type_convert<float>(x)) + 1.0));
};

template <>
CK_TILE_DEVICE float tanh_fast<float>(float x)
{
    // float a = __builtin_amdgcn_sinh(x);
    // float b = __builtin_amdgcn_cosh(x);
    // float e = a * __builtin_amdgcn_rcpf(b);
    // return e;

    float a = 2.0f * log2e_v<float> * x;
    a       = __ocml_exp_f32(a);
    a       = __builtin_amdgcn_rcpf(a + 1.0f);
    a       = 2 * a;
    a       = 1 - a;
    return a;

    // float e, r, s, t, d;
    // float a = x;
    // s = abs(a);
    // t = -log2e_v<float> * 2.0f * s;
    // e = __builtin_amdgcn_exp2f(t);
    // d = e + 1.0f;
    // r = __builtin_amdgcn_rcpf(d);
    // r = e * (-r) + r;
    // if (s < 4.997253418e-3f) r = a;
    // union fipnr {float f; unsigned int i;};
    // fipnr r_; r_.f = r;
    // fipnr a_; a_.f = a;
    // { r_.i = (r_.i|(a_.i&0x80000000)); r = r_.f; }
    // return r;
};

template <typename T>
CK_TILE_DEVICE T log(T x)
{
    return type_convert<T>(__logf(type_convert<float>(x)));
};

template <>
CK_TILE_DEVICE fp16_t log<fp16_t>(fp16_t x)
{
    return __ocml_log_f16(x);
};

template <>
CK_TILE_DEVICE float log<float>(float x)
{
    return __logf(x);
};

template <>
CK_TILE_DEVICE double log<double>(double x)
{
    return log(x);
};

template <typename T>
CK_TILE_DEVICE T pow(T x, T gamma)
{
    return type_convert<T>(powf(type_convert<float>(x), type_convert<float>(gamma)));
};

template <>
CK_TILE_DEVICE float pow<float>(float x, float gamma)
{
    return powf(x, gamma);
};

template <>
CK_TILE_DEVICE double pow<double>(double x, double gamma)
{
    return pow(x, gamma);
};

template <typename T>
CK_TILE_DEVICE T expm1(T x)
{
    return type_convert<T>(expm1f(type_convert<float>(x)));
};

template <>
CK_TILE_DEVICE float expm1<float>(float x)
{
    return expm1f(x);
};

template <>
CK_TILE_DEVICE double expm1<double>(double x)
{
    return expm1(x);
};

CK_TILE_HOST_DEVICE bool ispow2(index_t v) { return (v != 0) && ((v & (v - 1)) == 0); }

} // namespace ck_tile
