// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/numeric.hpp"
#include <stdint.h>

#pragma once

namespace ck_tile {

enum class bf16_rounding_mode
{
    standard = 0, // rtn
    truncate_with_nan,
    truncate,
    standard_asm,
    rta_asm, // round to nearest away
};

template <bf16_rounding_mode rounding =
              static_cast<bf16_rounding_mode>(CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT)>
CK_TILE_HOST_DEVICE constexpr uint16_t float_to_bf16_raw(float f, constant<rounding> = {});

template <bf16_rounding_mode rounding =
              static_cast<bf16_rounding_mode>(CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT)>
CK_TILE_HOST_DEVICE constexpr uint16_t double_to_bf16_raw(double f, constant<rounding> = {});

CK_TILE_HOST_DEVICE
constexpr float bf16_to_float_raw(uint16_t x);

CK_TILE_HOST_DEVICE
constexpr double bf16_to_double_raw(uint16_t x);

#if CK_TILE_USE_CUSTOM_DATA_TYPE
// HIP use __hip_bfloat16 as struct
struct alignas(2) bfloat16_t
{
    using raw_type = uint16_t;
    raw_type data;

    CK_TILE_HOST_DEVICE
    static constexpr bfloat16_t bit_cast(raw_type x)
    {
        bfloat16_t y;
        y.data = x;
        return y;
    }

    // constructor
    constexpr bfloat16_t() : data() {}

    // construct from float
    CK_TILE_HOST_DEVICE
    explicit constexpr bfloat16_t(const float& x) : data(float_to_bf16_raw(x)) {}

    // construct from double
    CK_TILE_HOST_DEVICE
    explicit constexpr bfloat16_t(const double& x) : data(double_to_bf16_raw(x)) {}

    // construct from int
    CK_TILE_HOST_DEVICE
    explicit constexpr bfloat16_t(const int& x) : data(float_to_bf16_raw(static_cast<float>(x))) {}

    // construct from unsigned int
    CK_TILE_HOST_DEVICE
    explicit constexpr bfloat16_t(const unsigned int& x)
        : data(float_to_bf16_raw(static_cast<float>(x)))
    {
    }

    // cast to float
    CK_TILE_HOST_DEVICE
    explicit constexpr operator float() const { return bf16_to_float_raw(data); }

    // cast to float
    CK_TILE_HOST_DEVICE
    explicit constexpr operator double() const { return bf16_to_double_raw(data); }

    // cast to int
    CK_TILE_HOST_DEVICE
    explicit constexpr operator int() const { return static_cast<int>(bf16_to_float_raw(data)); }

    // internal access
    CK_TILE_HOST_DEVICE
    constexpr raw_type& get() { return data; }

    CK_TILE_HOST_DEVICE
    constexpr raw_type get() const { return data; }
};
template <typename>
struct native_t;

template <>
struct native_t<bfloat16_t>
{
    using type = ushort;
};
using bf16_t     = bfloat16_t;
using bf16_raw_t = typename bf16_t::raw_type;
#else
using bfloat16_t = ushort;
using bf16_t     = bfloat16_t;
using bf16_raw_t = uint16_t;
#endif
// round to nearest
CK_TILE_HOST_DEVICE
constexpr uint16_t float_to_bf16_rtn_raw(float f)
{
    union
    {
        float fp32;
        uint32_t int32;
    } u = {f};
    if(~u.int32 & 0x7f800000)
    {
        // When the exponent bits are not all 1s, then the value is zero, normal,
        // or subnormal. We round the bfloat16 mantissa up by adding 0x7FFF, plus
        // 1 if the least significant bit of the bfloat16 mantissa is 1 (odd).
        // This causes the bfloat16's mantissa to be incremented by 1 if the 16
        // least significant bits of the float mantissa are greater than 0x8000,
        // or if they are equal to 0x8000 and the least significant bit of the
        // bfloat16 mantissa is 1 (odd). This causes it to be rounded to even when
        // the lower 16 bits are exactly 0x8000. If the bfloat16 mantissa already
        // has the value 0x7f, then incrementing it causes it to become 0x00 and
        // the exponent is incremented by one, which is the next higher FP value
        // to the unrounded bfloat16 value. When the bfloat16 value is subnormal
        // with an exponent of 0x00 and a mantissa of 0x7F, it may be rounded up
        // to a normal value with an exponent of 0x01 and a mantissa of 0x00.
        // When the bfloat16 value has an exponent of 0xFE and a mantissa of 0x7F,
        // incrementing it causes it to become an exponent of 0xFF and a mantissa
        // of 0x00, which is Inf, the next higher value to the unrounded value.
        u.int32 += 0x7fff + ((u.int32 >> 16) & 1); // Round to nearest, round to even
    }
    else if(u.int32 & 0xffff)
    {
        // When all of the exponent bits are 1, the value is Inf or NaN.
        // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
        // mantissa bit. Quiet NaN is indicated by the most significant mantissa
        // bit being 1. Signaling NaN is indicated by the most significant
        // mantissa bit being 0 but some other bit(s) being 1. If any of the
        // lower 16 bits of the mantissa are 1, we set the least significant bit
        // of the bfloat16 mantissa, in order to preserve signaling NaN in case
        // the bloat16's mantissa bits are all 0.
        u.int32 |= 0x10000; // Preserve signaling NaN
    }
    return uint16_t(u.int32 >> 16);
}

CK_TILE_HOST
constexpr uint16_t float_to_bf16_rtn_asm(float f) { return float_to_bf16_rtn_raw(f); }

CK_TILE_DEVICE
uint16_t float_to_bf16_rtn_asm(float f)
{
    union
    {
        float fp32;
        uint32_t int32;
    } u = {f};

    static constexpr uint32_t FP32_NAN            = 0x7fff0000;
    static constexpr uint32_t ROUND_BIAS_FOR_BF16 = 0x7fff;

    using uint32x2_t = uint32_t __attribute__((ext_vector_type(2)));
    uint32x2_t check_nan;
    uint32_t tmp;
    asm volatile("\n \
            v_cmp_u_f32 %0, %2, %2 \n \
            v_bfe_u32 %1, %2, 16, 1 \n \
            v_add3_u32 %1, %2, %1, %3 \n \
            v_cndmask_b32 %2, %1, %4, %0 \n \
            v_lshrrev_b32 %2, 16, %2 \n \
            "
                 : "=s"(check_nan), "+v"(tmp), "+v"(u.fp32)
                 : "v"(ROUND_BIAS_FOR_BF16), "v"(FP32_NAN));

    return uint16_t(u.int32);
}

// TODO: do we need this on host?
CK_TILE_HOST
uint16_t float_to_bf16_rta_asm(float f) { return float_to_bf16_rtn_raw(f); }

CK_TILE_DEVICE
uint16_t float_to_bf16_rta_asm(float f)
{
    union
    {
        float fp32;
        struct
        {
            uint16_t lo;
            uint16_t hi;
        };
    } u = {f};

    const uint32_t low_nan = 0x7fff;
    const uint32_t hi_nan  = 0x7fff0000;

    using uint32x2_t = uint32_t __attribute__((ext_vector_type(2)));
    uint32x2_t check_nan;

    asm volatile("v_cmp_u_f32 %[s_cnan], %[v_x], %[v_x] \n"
                 "v_add3_u32 %[v_x], %[v_x], %[v_blo], 1 \n"
                 "v_cndmask_b32 %[v_x], %[v_x], %[v_bhi], %[s_cnan]"
                 : [s_cnan] "+s"(check_nan), [v_x] "+v"(u.fp32)
                 : [v_blo] "v"(low_nan), [v_bhi] "v"(hi_nan));

    // Note: in above code snipet, we use hi 16 bit
    return u.hi;
}

// Truncate instead of rounding, preserving SNaN
CK_TILE_HOST_DEVICE
constexpr uint16_t float_to_bf16_truc_nan_raw(float f)
{
    union
    {
        float fp32;
        uint32_t int32;
    } u = {f};
    return uint16_t(u.int32 >> 16) | (!(~u.int32 & 0x7f800000) && (u.int32 & 0xffff));
}

// Fast truncate instead of rounding, RTZ
CK_TILE_HOST_DEVICE
constexpr uint16_t float_to_bf16_truc_raw(float f)
{
    union
    {
        float fp32;
        uint32_t int32;
    } u = {f};
    return uint16_t(u.int32 >> 16);
}

template <bf16_rounding_mode rounding>
CK_TILE_HOST_DEVICE constexpr uint16_t float_to_bf16_raw(float f, constant<rounding>)
{
    if constexpr(rounding == bf16_rounding_mode::standard)
        return float_to_bf16_rtn_raw(f);
    else if constexpr(rounding == bf16_rounding_mode::standard_asm)
        return float_to_bf16_rtn_asm(f);
    else if constexpr(rounding == bf16_rounding_mode::truncate_with_nan)
        return float_to_bf16_truc_nan_raw(f);
    else if constexpr(rounding == bf16_rounding_mode::rta_asm)
        return float_to_bf16_rta_asm(f);
    else
        return float_to_bf16_truc_raw(f);
}

template <bf16_rounding_mode rounding>
CK_TILE_HOST_DEVICE constexpr uint16_t double_to_bf16_raw(double f, constant<rounding>)
{
    return float_to_bf16_raw(static_cast<float>(f), constant<rounding>{});
}

CK_TILE_HOST_DEVICE
constexpr float bf16_to_float_raw(uint16_t x)
{
    union
    {
        uint32_t int32;
        float fp32;
    } u = {uint32_t(x) << 16};
    return u.fp32;
}

CK_TILE_HOST_DEVICE
constexpr double bf16_to_double_raw(uint16_t x)
{
    return static_cast<double>(bf16_to_float_raw(x));
}

template <bf16_rounding_mode rounding =
              static_cast<bf16_rounding_mode>(CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT)>
CK_TILE_HOST_DEVICE constexpr bfloat16_t float_to_bf16(float f, constant<rounding> = {})
{
    return bit_cast<bfloat16_t>(float_to_bf16_raw(f, constant<rounding>{}));
}

template <bf16_rounding_mode rounding =
              static_cast<bf16_rounding_mode>(CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT)>
CK_TILE_HOST_DEVICE constexpr bfloat16_t double_to_bf16(double f, constant<rounding> = {})
{
    return bit_cast<bfloat16_t>(double_to_bf16_raw(f, constant<rounding>{}));
}

CK_TILE_HOST_DEVICE
constexpr float bf16_to_float(bfloat16_t x) { return bf16_to_float_raw(bit_cast<uint16_t>(x)); }

CK_TILE_HOST_DEVICE
constexpr double bf16_to_double(bfloat16_t x) { return static_cast<double>(bf16_to_float_raw(x)); }

template <bf16_rounding_mode rounding =
              static_cast<bf16_rounding_mode>(CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT)>
CK_TILE_HOST_DEVICE bfloat16_t constexpr fp16_to_bf16(half_t f, constant<rounding> = {})
{
    return bit_cast<bfloat16_t>(float_to_bf16_raw(static_cast<float>(f), constant<rounding>{}));
}

CK_TILE_HOST_DEVICE
constexpr half_t bf16_to_fp16(bfloat16_t x) { return static_cast<fp16_t>(static_cast<float>(x)); }

template <class T>
struct numeric;

template <>
struct numeric<bfloat16_t>
{
    // minimum finite value, or minimum positive normalized value for float
    CK_TILE_HOST_DEVICE static constexpr bfloat16_t min()
    {
        return bit_cast<bfloat16_t>(static_cast<bf16_raw_t>(0x0080));
    }

    // minumum finite value
    CK_TILE_HOST_DEVICE static constexpr bfloat16_t lowest()
    {
        return bit_cast<bfloat16_t>(static_cast<bf16_raw_t>(0xff7f));
    }

    // maximum finite value
    CK_TILE_HOST_DEVICE static constexpr bfloat16_t max()
    {
        return bit_cast<bfloat16_t>(static_cast<bf16_raw_t>(0x7f7f));
    }

    // difference between 1.0 and next value representable by float
    CK_TILE_HOST_DEVICE static constexpr bfloat16_t epsilon()
    {
        return bit_cast<bfloat16_t>(static_cast<bf16_raw_t>(0x1000));
    }

    // maximum rounding error
    // maximum rounding error
    // bin :  f edcba 9876543210
    // bits:  s eeeeeeee mmmmmmm
    //        0 01111110 0000000 (0.5)
    //
    CK_TILE_HOST_DEVICE static constexpr bfloat16_t round_error()
    {
        return bit_cast<bfloat16_t>(static_cast<bf16_raw_t>(0x3f00));
    }

    // positive infinity value
    CK_TILE_HOST_DEVICE static constexpr bfloat16_t infinity()
    {
        return bit_cast<bfloat16_t>(static_cast<bf16_raw_t>(0x7f80));
    }

    // quiet NaN
    CK_TILE_HOST_DEVICE static constexpr bfloat16_t quiet_NaN()
    {
        return bit_cast<bfloat16_t>(static_cast<bf16_raw_t>(0x7FFF));
    }

    // signaling NaN
    CK_TILE_HOST_DEVICE static constexpr bfloat16_t signaling_NaN()
    {
        return bit_cast<bfloat16_t>(static_cast<bf16_raw_t>(0x7FFF));
    }

    // smallest positive subnormal value
    CK_TILE_HOST_DEVICE static constexpr bfloat16_t denorm_min()
    {
        return bit_cast<bfloat16_t>(static_cast<bf16_raw_t>(0x0001));
    }
    CK_TILE_HOST_DEVICE static constexpr bfloat16_t zero()
    {
        return bit_cast<bfloat16_t>(static_cast<bf16_raw_t>(0));
    }
};

template <>
struct numeric_traits<bfloat16_t>
{
    static constexpr int exp        = 8;
    static constexpr int mant       = 7;
    static constexpr int PackedSize = 1;
};

#if CK_TILE_USE_CUSTOM_DATA_TYPE
CK_TILE_ARITHMETIC_USING_FLOAT(CK_TILE_HOST_DEVICE, bfloat16_t)
#endif

// math
CK_TILE_HOST_DEVICE
bfloat16_t abs(const bfloat16_t& x)
{
    return bit_cast<bfloat16_t>(static_cast<bf16_raw_t>(bit_cast<bf16_raw_t>(x) & 0x7fff));
}

CK_TILE_HOST_DEVICE
bool isnan(const bfloat16_t& x)
{
    uint16_t xx = bit_cast<bf16_raw_t>(x);
    return (xx & 0x7FFF) > 0x7C00;
}

CK_TILE_DEVICE
bfloat16_t sqrt(bfloat16_t x)
{
    return static_cast<bfloat16_t>(__builtin_amdgcn_sqrtf(static_cast<float>(x)));
};

CK_TILE_DEVICE
bfloat16_t exp(bfloat16_t x)
{
    return static_cast<bfloat16_t>(__ocml_exp_f32(static_cast<float>(x)));
};

CK_TILE_DEVICE
bfloat16_t exp2(bfloat16_t x) { return static_cast<bfloat16_t>(exp2f(static_cast<float>(x))); };

CK_TILE_DEVICE
bfloat16_t log(bfloat16_t x) { return static_cast<bfloat16_t>(__logf(static_cast<float>(x))); };

} // namespace ck_tile
