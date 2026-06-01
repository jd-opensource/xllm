// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/numeric/numeric.hpp"
#include "ck_tile/core/utility/random.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/numeric.hpp"
#include <stdint.h>
#include <type_traits>

#pragma once

#if(defined(__gfx938__) || defined(__gfx94__) || defined(__gfx12__)) && __HIP_DEVICE_COMPILE__
#define CK_TILE_FP8_CVT_DEVICE 1
#else
#define CK_TILE_FP8_CVT_DEVICE 0
#endif

namespace ck_tile {

// fp8 rounding modes
// use standard for rounding to nearest, the faster one
// use stochastic for stochastic rounding, helps to avoid error accumulation
enum class fp8_rounding_mode
{
    standard = 0,
    stochastic
};

/**
 * \brief FP8 interpretation used in conversion algorithms
 */
enum class fp8_interpretation
{
    E4M3_OCP  = 0, // OCP FP8 E4M3
    E5M2_OCP  = 1, // OCP BF8 E5M2
    E4M3_FNUZ = 2, // FNUZ FP8 E4M3
    E5M2_FNUZ = 3, // FNUZ BF8 E5M2
};

/*
 *                 ______________FNUZ_________________    |   ______________OCP________________
 *                   e4m3               e5m2              |    e4m3                e5m2
 *      bias :        8                  16               |     7                   15
 *      inf  :       N/A                 N/A              |    N/A              s.11111.00
 *      Nan  :  1.0000.000           1.00000.00           | s.1111.111          s.11111.{01, 10, 11}
 *      zero :  0.0000.000           0.00000.00           | s.0000.000          s.00000.00
 * Max(norm) :  s.1111.111 (240)     s.11111.11(57344)    | s.1111.110(448)     s.11110.11(57344)
 * Max(snorm):  s.0000.111           s.00000.11           | s.0000.111          s.00000.11
 *                0.0068359375         2.288818e-05       | 0.013671875         4.57763671875e-05
 * Min(norm) :  s.0001.000           s.00001.00           | s.0001.000          s.00001.00
 *                2^-7(0.0078125)      2^-15(3.05176e-05) |   2^-6(0.015625) 2^-14(6.10352e-05)
 * Min(snorm):  s.0000.001           s.00000.01           | s.0000.001          s.00000.01
 *                2^-10(0.0009765625)  2^-17(7.62939e-06) |   2^-9(0.001953125) 2^-16(1.52588e-05)
 */

template <fp8_rounding_mode rounding = static_cast<fp8_rounding_mode>(CK_TILE_FLOAT_TO_FP8_DEFAULT)>
CK_TILE_HOST_DEVICE uint8_t float_to_fp8_raw(float, constant<rounding> = {});

template <fp8_rounding_mode rounding = static_cast<fp8_rounding_mode>(CK_TILE_FLOAT_TO_FP8_DEFAULT)>
CK_TILE_HOST_DEVICE uint8_t float_to_bf8_raw(float, constant<rounding> = {});

CK_TILE_HOST_DEVICE float fp8_to_float_raw(uint8_t);
CK_TILE_HOST_DEVICE float bf8_to_float_raw(uint8_t);

#if CK_TILE_USE_CUSTOM_DATA_TYPE
struct alignas(1) float8_e4m3_t
{
    static constexpr int exponent = 4;
    static constexpr int mantissa = 3;
#if CK_TILE_USE_OCP_FP8
    static constexpr int bias = 7; // OCP
#else
    static constexpr int bias = 8; // FNUZ
#endif
    using raw_type = uint8_t;
    raw_type data;

    CK_TILE_HOST_DEVICE
    static constexpr float8_e4m3_t bit_cast(raw_type x)
    {
        float8_e4m3_t y;
        y.data = x;
        return y;
    }

    // constructor
    constexpr float8_e4m3_t() : data() {}

    // construct from float
    CK_TILE_HOST_DEVICE
    explicit constexpr float8_e4m3_t(const float& x) : data(float_to_fp8_raw(x)) {}

    // construct from int
    CK_TILE_HOST_DEVICE
    explicit constexpr float8_e4m3_t(const int& x) : data(float_to_fp8_raw(static_cast<float>(x)))
    {
    }

    // construct from unsigned int
    CK_TILE_HOST_DEVICE
    explicit constexpr float8_e4m3_t(const unsigned int& x)
        : data(float_to_fp8_raw(static_cast<float>(x)))
    {
    }

    // cast to float
    CK_TILE_HOST_DEVICE
    explicit constexpr operator float() const { return fp8_to_float_raw(data); }

    // cast to int
    CK_TILE_HOST_DEVICE
    explicit constexpr operator int() const { return static_cast<int>(fp8_to_float_raw(data)); }

    // internal access
    CK_TILE_HOST_DEVICE
    constexpr raw_type& get() { return data; }

    CK_TILE_HOST_DEVICE
    constexpr raw_type get() const { return data; }
};
using fp8_t     = float8_e4m3_t;
using fp8_raw_t = typename fp8_t::raw_type;

struct alignas(1) float8_e5m2_t
{
    static constexpr int exponent = 5;
    static constexpr int mantissa = 2;
#if CK_TILE_USE_OCP_FP8
    static constexpr int bias = 15; // OCP
#else
    static constexpr int bias = 16; // FNUZ
#endif
    using raw_type = uint8_t;
    raw_type data;

    CK_TILE_HOST_DEVICE
    static constexpr float8_e5m2_t bit_cast(raw_type x)
    {
        float8_e5m2_t y;
        y.data = x;
        return y;
    }

    // constructor
    constexpr float8_e5m2_t() : data() {}

    // construct from float
    CK_TILE_HOST_DEVICE
    explicit constexpr float8_e5m2_t(const float& x) : data(float_to_bf8_raw(x)) {}

    // construct from int
    CK_TILE_HOST_DEVICE
    explicit constexpr float8_e5m2_t(const int& x) : data(float_to_bf8_raw(static_cast<float>(x)))
    {
    }

    // construct from unsigned int
    CK_TILE_HOST_DEVICE
    explicit constexpr float8_e5m2_t(const unsigned int& x)
        : data(float_to_bf8_raw(static_cast<float>(x)))
    {
    }

    // cast to float
    CK_TILE_HOST_DEVICE
    explicit constexpr operator float() const { return bf8_to_float_raw(data); }

    // cast to int
    CK_TILE_HOST_DEVICE
    explicit constexpr operator int() const { return static_cast<int>(bf8_to_float_raw(data)); }

    // internal access
    CK_TILE_HOST_DEVICE
    constexpr raw_type& get() { return data; }

    CK_TILE_HOST_DEVICE
    constexpr raw_type get() const { return data; }
};
using bf8_t     = float8_e5m2_t;
using bf8_raw_t = typename bf8_t::raw_type;

template <typename>
struct native_t;

template <>
struct native_t<fp8_t>
{
    using type = _BitInt(8);
};

template <>
struct native_t<bf8_t>
{
    using type = unsigned _BitInt(8);
};

#else

using fp8_t     = _BitInt(8);
using fp8_raw_t = uint8_t;
using bf8_t     = unsigned _BitInt(8);
using bf8_raw_t = uint8_t;
#endif

template <>
struct numeric_traits<fp8_t>
{
    using bitwise_type = fp8_raw_t;

    static constexpr int exp  = 4;
    static constexpr int mant = 3;
#if CK_TILE_USE_OCP_FP8
    static constexpr int bias                        = 7;
    static constexpr fp8_interpretation f8_interpret = fp8_interpretation::E4M3_OCP;
#else
    static constexpr int bias                        = 8;
    static constexpr fp8_interpretation f8_interpret = fp8_interpretation::E4M3_FNUZ;
#endif
    static constexpr uint8_t abs_mask = 0x7F;
    static constexpr int PackedSize   = 1;
};

template <>
struct numeric_traits<bf8_t>
{
    using bitwise_type = bf8_raw_t;

    static constexpr int exp  = 5;
    static constexpr int mant = 2;
#if CK_TILE_USE_OCP_FP8
    static constexpr int bias                        = 15;
    static constexpr fp8_interpretation f8_interpret = fp8_interpretation::E5M2_OCP;
#else
    static constexpr int bias                        = 16;
    static constexpr fp8_interpretation f8_interpret = fp8_interpretation::E5M2_FNUZ;
#endif
    static constexpr uint8_t abs_mask = 0x7F;
    static constexpr int PackedSize   = 1;
};

// below is sw fp8 conversion, not utilizing hw instruction
namespace impl {

template <typename SrcT, typename DstT, bool clip = true, bool stoch = false>
CK_TILE_HOST_DEVICE DstT run_cast_to_f8(SrcT src, unsigned int rng = 0)
{
    static_assert(std::is_same<DstT, fp8_t>::value || std::is_same<DstT, bf8_t>::value,
                  "DstT type must be fp8 or bf8.");

    constexpr bool is_half  = std::is_same<SrcT, half_t>::value;
    constexpr bool is_float = std::is_same<SrcT, float>::value;
    static_assert(is_half || is_float, "Only half and float can be cast to f8");

    // fp8/bf8 type exponent/mantissa layout
    constexpr int DstT_exp  = numeric_traits<DstT>::exp;  // exponent width of the destination type
    constexpr int DstT_mant = numeric_traits<DstT>::mant; // mantissa width of the destination type
    constexpr int DstT_bias = numeric_traits<DstT>::bias;
    constexpr bool is_fnuz =
        (numeric_traits<DstT>::f8_interpret == fp8_interpretation::E4M3_FNUZ) ||
        (numeric_traits<DstT>::f8_interpret == fp8_interpretation::E5M2_FNUZ);

    constexpr int SrcT_exp          = numeric_traits<SrcT>::exp;
    constexpr int SrcT_mant         = numeric_traits<SrcT>::mant;
    constexpr int bias              = numeric_traits<SrcT>::bias;
    constexpr unsigned int fInf     = numeric_traits<SrcT>::Inf;
    constexpr unsigned int abs_mask = numeric_traits<SrcT>::abs_mask;

    using SrcT_bitwise       = typename numeric_traits<SrcT>::bitwise_type;
    SrcT_bitwise src_bitwise = bit_cast<SrcT_bitwise>(src);

    unsigned int head, mantissa;
    int exponent;
    unsigned int sign;

    head     = src_bitwise & numeric_traits<SrcT>::head_mask;
    mantissa = src_bitwise & numeric_traits<SrcT>::mant_mask;
    exponent = (head >> SrcT_mant) & numeric_traits<SrcT>::exp_mask;
    sign     = head >> (SrcT_exp + SrcT_mant);

    unsigned int signed_inf = 0;
    unsigned int nan        = 0;
    if constexpr(is_fnuz)
    {
        signed_inf = clip ? ((sign << (DstT_exp + DstT_mant)) + 0x7f) : 0x80;
        nan        = 0x80;
    }
    else
    {
        if constexpr(DstT_exp == 4)
        { // e4m3
            signed_inf = (sign << (DstT_exp + DstT_mant)) + (clip ? 0x7e : 0x7f);
        }
        else
        { // e5m2
            signed_inf = (sign << (DstT_exp + DstT_mant)) + (clip ? 0x7b : 0x7c);
        }
        nan = (sign << (DstT_exp + DstT_mant)) + 0x7f;
    }
    // Max values
    unsigned int ifmax = 0;
    if constexpr(is_float)
    {
        if constexpr(DstT_exp == 5)
        {
            ifmax = 0x47600000;
        }
        else
        {
            if constexpr(is_fnuz)
            {
                ifmax = 0x43700000;
            }
            else
            {
                ifmax = 0x43E00000;
            }
        }
    }
    else if constexpr(is_half)
    {
        if constexpr(DstT_exp == 5)
        {
            ifmax = 0x7B00;
        }
        else
        {
            if constexpr(is_fnuz)
            {
                ifmax = 0x5B80;
            }
            else
            {
                ifmax = 0x5F00;
            }
        }
    }

    // Deal with inf and NaNs
    if((src_bitwise & fInf) == fInf)
    {
        return mantissa != 0 ? nan : signed_inf;
    }

    if((src_bitwise & abs_mask) > ifmax)
    {
        return signed_inf;
    }

    // First need to check if it is normal or denorm as there is a difference of
    // implicit 1 Then need to adjust the exponent to align with the F8 exponent,
    // in the meanwhile, shift The mantissa. Then for stochastic rounding, add rng
    // to mantissa and truncate. And for RNE, no need to add rng. Then probably
    // need to check whether there is carry and adjust exponent and mantissa again

    // For IEEE bias mode, the bias is 2^(k-1) -1 where k is the width of exponent
    // bits
    constexpr int f8_denormal_act_exponent = 1 - DstT_bias; // actual exponent of f8 denormal
    // act_exponent is the actual exponent of fp32/fp16 (after subtracting bias)
    // f8_exponent is the converted f8 exponent with bias encoding
    // exponent_diff is the diff between fp32/fp16 exponent and f8 exponent,
    // the difference needs to be adjusted and mantissa shifted
    int act_exponent, f8_exponent, exponent_diff;

    if(exponent == 0)
    { // fp32/fp16 is in denormal.
        /* fp32 denormal is below 2^-127 so it is usually not a concern here, we
    mostly concern fp16 here. In this case, f8 is usually in denormal. But there
    could be exceptions. fp16 denormal has exponent bias 15 while bf8 with NANOO has
    exponent bias 16. It means that there are some numbers in fp16 denormal but they
    are bf8 (NANOO) normals - smallest bf8 (NANOO) normal is 2^-15. fp16 numbers
    where exponent==0 (actual exponent -14) and highest bit of mantissa is 1 are bf8
    (NANOO) normal. In this case, the fp16 mantissa should be shift left by 1  */
        act_exponent  = exponent - bias + 1;
        exponent_diff = f8_denormal_act_exponent -
                        act_exponent; // actual exponent is exponent-bias+1 as it is denormal
    }
    else
    { // fp32/fp16 is normal with implicit 1
        act_exponent = exponent - bias;
        if(act_exponent <= f8_denormal_act_exponent)
        {
            /* This is the case where fp32/fp16 is normal but it is in f8 denormal
      range. For example fp8 nanoo mode, denormal exponent is -7, but if the fp32/fp16
      actual exponent is -7, it is actually larger due to the implicit 1,
      Therefore it needs to be adjust to -6 and mantissa shift right by 1.
      So for fp32/fp16, exponent -8 is the cut point to convert to fp8 nanoo */
            exponent_diff = f8_denormal_act_exponent - act_exponent;
        }
        else
        {                      // both fp32/fp16 and f8 are in normal range
            exponent_diff = 0; // exponent_diff=0 does not mean there is no difference
                               // for this case, act_exponent could be larger. Just
                               // that it does not need shift mantissa
        }
        mantissa += (1u << SrcT_mant); // Add the implicit 1 into mantissa
    }
    // The value is smaller than min f8 denormal and results in zero (the early exit also prevents
    // an undefined behavior of bit shifts >= type width).
    if(exponent_diff > DstT_mant)
    {
        return is_fnuz ? 0 : (sign << (DstT_exp + DstT_mant));
    }
    bool midpoint = (mantissa & ((1u << (SrcT_mant - DstT_mant + exponent_diff)) - 1)) ==
                    (1u << (SrcT_mant - DstT_mant + exponent_diff - 1));
    /* This part is a bit tricky. The judgment of whether it is a tie needs to be
  done before we shift right as shift right could rip off some residual part and
  make something not midpoint look like midpoint. For example, the fp16 number
  0x1002 (0 00100 0000000010), it is larger than midpoint, but after shift right
  by 4 bits, it would look like midpoint.
  */

    if(exponent_diff > 0)
        mantissa >>= exponent_diff;
    else if(exponent_diff == -1)
        mantissa <<= -exponent_diff;
    bool implicit_one = mantissa & (1u << SrcT_mant);
    // if there is no implicit 1, it  means the f8 is denormal and need to adjust
    // to denorm exponent
    f8_exponent =
        (act_exponent + exponent_diff) /*actual f8 exponent*/ + DstT_bias - (implicit_one ? 0 : 1);

    // Now we have the exponent and mantissa adjusted
    unsigned int drop_mask = (1u << (SrcT_mant - DstT_mant)) - 1;
    bool odd =
        mantissa &
        (1u << (SrcT_mant - DstT_mant)); // if the least significant bit that is not truncated is 1
    mantissa +=
        (stoch ? rng : (midpoint ? (odd ? mantissa : mantissa - 1u) : mantissa)) & drop_mask;

    // Now we deal with overflow
    if(f8_exponent == 0)
    {
        if((1u << SrcT_mant) & mantissa)
        {
            f8_exponent = 1; // denormal overflow to become normal, promote exponent
        }
    }
    else
    {
        if((1u << (SrcT_mant + 1)) & mantissa)
        {
            mantissa >>= 1;
            f8_exponent++;
        }
    }

    mantissa >>= (SrcT_mant - DstT_mant);

    // above range: quantize to maximum possible float of the same sign
    const int max_exp = (1 << DstT_exp) - 1;
    if(f8_exponent > max_exp)
    {
        if constexpr(clip)
        {
            mantissa    = (1 << DstT_mant) - 1;
            f8_exponent = max_exp;
        }
        else
        {
            return signed_inf;
        }
    }

    if(f8_exponent == 0 && mantissa == 0)
        return is_fnuz ? 0 : (sign << (DstT_exp + DstT_mant));
    mantissa &= (1 << DstT_mant) - 1;
    return (sign << (DstT_exp + DstT_mant)) | (f8_exponent << DstT_mant) | mantissa;
}

template <typename SrcT, typename DstT, bool clip = true>
CK_TILE_HOST_DEVICE DstT run_cast_from_f8(SrcT x)
{
    static_assert(std::is_same<SrcT, fp8_t>::value || std::is_same<SrcT, bf8_t>::value,
                  "SrcT type must be fp8 or bf8.");
    constexpr int SrcT_exp          = numeric_traits<SrcT>::exp;
    constexpr int SrcT_mant         = numeric_traits<SrcT>::mant;
    constexpr uint8_t SrcT_abs_mask = numeric_traits<SrcT>::abs_mask;
    constexpr bool is_fnuz =
        (numeric_traits<SrcT>::f8_interpret == fp8_interpretation::E4M3_FNUZ) ||
        (numeric_traits<SrcT>::f8_interpret == fp8_interpretation::E5M2_FNUZ);

    constexpr bool is_half  = std::is_same<DstT, half_t>::value;
    constexpr bool is_float = std::is_same<DstT, float>::value;
    static_assert(is_half || is_float, "DstT type must be half_t or float.");

    // destination type exponent/mantissa layout
    constexpr int DstT_exp  = numeric_traits<DstT>::exp;  // exponent width of the destination type
    constexpr int DstT_mant = numeric_traits<DstT>::mant; // mantissa width of the destination type

    constexpr DstT fInf    = bit_cast<DstT>(numeric_traits<DstT>::Inf);
    constexpr DstT fNegInf = bit_cast<DstT>(numeric_traits<DstT>::NegInf);
    constexpr DstT fNaN    = bit_cast<DstT>(numeric_traits<DstT>::NaN);
    constexpr DstT fNeg0   = bit_cast<DstT>(numeric_traits<DstT>::Neg0);

    DstT fmax{0}, fmin{0};
    // Max number in e5m2 57344
    if constexpr(is_half)
    {
        fmax = bit_cast<DstT>(static_cast<typename numeric_traits<DstT>::bitwise_type>(0x7B00));
        fmin = bit_cast<DstT>(static_cast<typename numeric_traits<DstT>::bitwise_type>(0xFB00));
    }
    else if constexpr(is_float)
    {
        fmax = bit_cast<DstT>(static_cast<typename numeric_traits<DstT>::bitwise_type>(0x47600000));
        fmin = bit_cast<DstT>(static_cast<typename numeric_traits<DstT>::bitwise_type>(0xC7600000));
    }

    if(x == 0)
    {
        return 0;
    }

    unsigned int sign     = x >> (SrcT_exp + SrcT_mant);
    unsigned int mantissa = x & ((1 << SrcT_mant) - 1);
    int exponent          = (x & SrcT_abs_mask) >> SrcT_mant;
    if constexpr(is_fnuz)
    {
        if((x & 0xff) == 0x80)
        {
            return fNaN;
        }
    }
    else
    {
        if(x == SrcT(0x80))
        {
            return fNeg0;
        }
        if constexpr(SrcT_exp == 4)
        { // e4m3
            if((x & 0x7F) == 0x7F)
            {
                return fNaN;
            }
        }
        else if((x & 0x7C) == 0x7C)
        { // e5m2
            if((x & 0x3) == 0)
            {
                if constexpr(clip)
                {
                    return sign ? fmin : fmax;
                }
                return sign ? fNegInf : fInf;
            }
            return fNaN;
        }
    }

    typename numeric_traits<DstT>::bitwise_type retval;

    if constexpr(SrcT_exp == 5 && is_half && !is_fnuz)
    {
        retval = static_cast<typename numeric_traits<DstT>::bitwise_type>(x) << 8;
        return bit_cast<DstT>(retval);
    }

    const int exp_low_cutoff =
        (1 << (DstT_exp - 1)) - (1 << (SrcT_exp - 1)) + 1 - (is_fnuz ? 1 : 0);

    // subnormal input
    if(exponent == 0)
    {
        int sh = 1 + clz(mantissa) - (32 - SrcT_mant);
        mantissa <<= sh;
        exponent += 1 - sh;
        mantissa &= ((1ull << SrcT_mant) - 1);
    }
    exponent += exp_low_cutoff - 1;
    mantissa <<= DstT_mant - SrcT_mant;

    // subnormal output (occurs when DstT is half_t, we=5, is_fnuz=true)
    if(exponent <= 0)
    {
        mantissa |= 1 << DstT_mant;
        mantissa >>= 1 - exponent;
        exponent = 0;
    }

    retval = (sign << (DstT_exp + DstT_mant)) | (exponent << DstT_mant) | mantissa;

    return bit_cast<DstT>(retval);
}

template <typename X, typename Y, bool clip, bool stoch>
CK_TILE_HOST_DEVICE Y cast_to_f8(X x, uint32_t rng)
{
    return bit_cast<Y>(run_cast_to_f8<X, Y, clip, stoch>(x, rng));
}

#if CK_TILE_FP8_CVT_DEVICE
/**
 * @brief Cast float to fp8/bf8 using device conversion instructions
 */
template <fp8_interpretation interpret, bool saturate, bool stochastic_rounding = false>
CK_TILE_DEVICE uint8_t cast_to_f8_from_f32(float v, unsigned int rng = 0)
{
    uint8_t i8data;
    union
    {
        float fval;
        unsigned int i32val;
        unsigned char i8val[4]; // NOTE: not endian independent
    } val;

    unsigned int ival = 0;
    val.fval          = v;

    if constexpr(saturate)
    {
        if constexpr(interpret == fp8_interpretation::E4M3_FNUZ)
        {
            if((val.i32val & 0x7F800000) != 0x7F800000)
            { /// propagate NAN/INF, no clipping
                val.fval = __builtin_amdgcn_fmed3f(val.fval, 240.0, -240.0);
            }
        }
        else if constexpr(interpret == fp8_interpretation::E4M3_OCP)
        { // OCP type
            if((val.i32val & 0x7F800000) != 0x7F800000)
            { /// propagate NAN/INF, no clipping
                val.fval = __builtin_amdgcn_fmed3f(val.fval, 448.0, -448.0);
            }
        }
        else
        {
            if((val.i32val & 0x7F800000) != 0x7F800000)
            { /// propagate NAN/INF, no clipping
                val.fval = __builtin_amdgcn_fmed3f(val.fval, 57344.0, -57344.0);
            }
        }
    }

    if constexpr(stochastic_rounding)
    {
        ival       = (interpret == fp8_interpretation::E4M3_FNUZ) ||
                       (interpret == fp8_interpretation::E4M3_OCP)
                         ? __builtin_hcu_cvt_sr_fp8_f32(val.fval, rng, ival, 0)
                         : __builtin_hcu_cvt_sr_bf8_f32(val.fval, rng, ival, 0); // 0 pos
        val.i32val = ival;
        i8data     = val.i8val[0]; // little endian
    }
    else
    { // RNE CVT
        ival       = (interpret == fp8_interpretation::E4M3_FNUZ) ||
                       (interpret == fp8_interpretation::E4M3_OCP)
                         ? __builtin_hcu_cvt_pk_fp8_f32(val.fval, val.fval, ival, false)
                         : __builtin_hcu_cvt_pk_bf8_f32(val.fval,
                                                     val.fval,
                                                     ival,
                                                     false); // false -> WORD0
        val.i32val = ival;
        i8data     = val.i8val[0];
    }
    return i8data;
}
#endif // CK_TILE_FP8_CVT_DEVICE

} // namespace impl

/**
 * @brief Converts a floating-point value to an 8-bit floating-point representation with stochastic
 * rounding.
 *
 * This function converts a floating-point value (float or half_t) to an 8-bit floating-point
 * representation of type fp8_t or bf8_t. The conversion process may
 * involve clipping and uses a pseudo-random number generator for the stochastic rounding.
 *
 * @tparam DstT The destination type (fp8_t or bf8_t).
 * @tparam SrcT The source type (float or half_t) to be converted.
 * @param x The floating-point value to be converted.
 * @return The 8-bit floating-point representation of the input value.
 */
template <typename SrcT, typename DstT>
CK_TILE_HOST_DEVICE typename numeric_traits<DstT>::bitwise_type float_to_fp8_sr_raw(SrcT x)
{
    constexpr bool clip = true;
    constexpr int seed  = 42;
    uint32_t rng        = prand_generator_t<SrcT, seed>{}(reinterpret_cast<uintptr_t>(&x), x);
#if CK_TILE_FP8_CVT_DEVICE
    return impl::cast_to_f8_from_f32<numeric_traits<DstT>::f8_interpret, clip, true>(x, rng);
#else
    return bit_cast<typename numeric_traits<DstT>::bitwise_type>(
        impl::cast_to_f8<SrcT, DstT, clip, true>(x, rng));
#endif
}

/**
 * @brief Converts a floating-point value to an 8-bit floating-point representation with rounding to
 * nearest even.
 *
 * This function converts a floating-point value (float or half_t) to an 8-bit floating-point
 * representation of type fp8_t or bf8_t. The conversion process may involve clipping.
 *
 * @tparam DstT The destination type (fp8_t or bf8_t).
 * @tparam SrcT The source type (float or half_t) to be converted.
 * @param x The floating-point value to be converted.
 * @return The 8-bit floating-point representation of the input value.
 */
template <typename SrcT, typename DstT>
CK_TILE_HOST_DEVICE typename numeric_traits<DstT>::bitwise_type float_to_fp8_rtn_raw(SrcT x)
{
    constexpr bool clip = true;
#if CK_TILE_FP8_CVT_DEVICE
    return impl::cast_to_f8_from_f32<numeric_traits<DstT>::f8_interpret, clip, false>(x, 0);
#else
    return bit_cast<typename numeric_traits<DstT>::bitwise_type>(
        impl::cast_to_f8<SrcT, DstT, clip, false>(x, 0));
#endif
}

template <fp8_rounding_mode rounding>
CK_TILE_HOST_DEVICE fp8_raw_t float_to_fp8_raw(float x, constant<rounding>)
{
    if constexpr(rounding == fp8_rounding_mode::standard)
    {
        return float_to_fp8_rtn_raw<float, fp8_t>(x);
    }
    else if constexpr(rounding == fp8_rounding_mode::stochastic)
    {
        return float_to_fp8_sr_raw<float, fp8_t>(x);
    }
    else
    {
        return fp8_raw_t{0};
    }
}

template <fp8_rounding_mode rounding>
CK_TILE_HOST_DEVICE bf8_raw_t float_to_bf8_raw(float x, constant<rounding>)
{
    if constexpr(rounding == fp8_rounding_mode::standard)
    {
        return float_to_fp8_rtn_raw<float, bf8_t>(x);
    }
    else if constexpr(rounding == fp8_rounding_mode::stochastic)
    {
        return float_to_fp8_sr_raw<float, bf8_t>(x);
    }
    else
    {
        return bf8_raw_t{0};
    }
}

CK_TILE_HOST_DEVICE float fp8_to_float_raw(fp8_raw_t x)
{
#if CK_TILE_FP8_CVT_DEVICE
    float fval;
    uint32_t i32val = static_cast<uint32_t>(x);
    // fval = __builtin_hcu_cvt_f32_fp8(i32val, true, 0, 0);
    // fval            = __builtin_amdgcn_cvt_f32_fp8(i32val, 0);
    asm volatile("v_cvt_f32_fp8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));
    return fval;
#else
    return impl::run_cast_from_f8<fp8_t, float>(bit_cast<fp8_t>(x));
#endif
}

CK_TILE_HOST_DEVICE float bf8_to_float_raw(bf8_raw_t x)
{
#if CK_TILE_FP8_CVT_DEVICE
    float fval;
    uint32_t i32val = static_cast<uint32_t>(x);
    fval = __builtin_hcu_cvt_f32_bf8(i32val, true, 0, 0);
    // fval            = __builtin_amdgcn_cvt_f32_bf8(i32val, 0);
    // asm volatile("v_cvt_f32_bf8 %0, %1 src0_sel:BYTE_0" : "=v"(fval) : "v"(i32val));
    return fval;
#else
    return impl::run_cast_from_f8<bf8_t, float>(bit_cast<bf8_t>(x));
#endif
}

template <fp8_rounding_mode rounding = static_cast<fp8_rounding_mode>(CK_TILE_FLOAT_TO_FP8_DEFAULT)>
CK_TILE_HOST_DEVICE fp8_t float_to_fp8(float x, constant<rounding> = {})
{
    return bit_cast<fp8_t>(float_to_fp8_raw(x, constant<rounding>{}));
}

template <fp8_rounding_mode rounding = static_cast<fp8_rounding_mode>(CK_TILE_FLOAT_TO_FP8_DEFAULT)>
CK_TILE_HOST_DEVICE bf8_t float_to_bf8(float x, constant<rounding> = {})
{
    return bit_cast<bf8_t>(float_to_bf8_raw(x, constant<rounding>{}));
}

CK_TILE_HOST_DEVICE float fp8_to_float(fp8_t x) { return fp8_to_float_raw(bit_cast<fp8_raw_t>(x)); }

CK_TILE_HOST_DEVICE float bf8_to_float(bf8_t x) { return bf8_to_float_raw(bit_cast<bf8_raw_t>(x)); }

template <class T>
struct numeric;

#if CK_TILE_USE_OCP_FP8
template <>
struct numeric<fp8_t>
{
    // minimum finite value, or minimum positive normal value
    CK_TILE_HOST_DEVICE static constexpr fp8_t min()
    {
        return bit_cast<fp8_t>(static_cast<fp8_raw_t>(0x08)); // 0b00001000 = 2^-6
    }

    // minumum finite value
    CK_TILE_HOST_DEVICE static constexpr fp8_t lowest()
    {
        return bit_cast<fp8_t>(static_cast<fp8_raw_t>(0xfe)); // 0b11111110 = -448
    }

    // maximum finite value
    CK_TILE_HOST_DEVICE static constexpr fp8_t max()
    {
        return bit_cast<fp8_t>(static_cast<fp8_raw_t>(0x7e)); // 0b01111110 = 448
    }

    // difference between 1.0 and next representable f8 value (1.125)
    // returns fp8_t(0.125)
    CK_TILE_HOST_DEVICE static constexpr fp8_t epsilon()
    {
        return bit_cast<fp8_t>(static_cast<fp8_raw_t>(0x20)); // 0.125
    }

    // rounding error (0.0625)
    // half of epsilon
    CK_TILE_HOST_DEVICE static constexpr fp8_t round_error()
    {
        return bit_cast<fp8_t>(static_cast<fp8_raw_t>(0x18)); // 0.0625
    }

    // quiet NaN
    CK_TILE_HOST_DEVICE static constexpr fp8_t quiet_NaN()
    {
        return bit_cast<fp8_t>(static_cast<fp8_raw_t>(0x7F)); // 0b01111111
    }

    // signaling NaN
    CK_TILE_HOST_DEVICE static constexpr fp8_t signaling_NaN()
    {
        return bit_cast<fp8_t>(static_cast<fp8_raw_t>(0xFF)); // 0b11111111
    }

    // smallest positive subnormal value
    CK_TILE_HOST_DEVICE static constexpr fp8_t denorm_min()
    {
        return bit_cast<fp8_t>(static_cast<fp8_raw_t>(0x01));
    }

    CK_TILE_HOST_DEVICE static constexpr fp8_t zero()
    {
        return bit_cast<fp8_t>(static_cast<fp8_raw_t>(0));
    }
};

template <>
struct numeric<bf8_t>
{
    // minimum finite value, or minimum positive normalized value for float
    CK_TILE_HOST_DEVICE static constexpr bf8_t min()
    {
        return bit_cast<bf8_t>(static_cast<bf8_raw_t>(0x04)); // 0b00000100 = 2^-14
    }

    // minumum finite value
    CK_TILE_HOST_DEVICE static constexpr bf8_t lowest()
    {
        return bit_cast<bf8_t>(static_cast<bf8_raw_t>(0xfb)); // 0b11111011 = -57344
    }

    // maximum finite value
    CK_TILE_HOST_DEVICE static constexpr bf8_t max()
    {
        return bit_cast<bf8_t>(static_cast<bf8_raw_t>(0x7b)); // 0b01111011 = 57344
    }

    // difference between 1.0 and next representable bf8 value (1.25)
    CK_TILE_HOST_DEVICE static constexpr bf8_t epsilon()
    {
        return bit_cast<bf8_t>(static_cast<bf8_raw_t>(0x34)); // 0.25
    }

    // rounding error (0.125)
    // half of epsilon
    CK_TILE_HOST_DEVICE static constexpr bf8_t round_error()
    {
        return bit_cast<bf8_t>(static_cast<bf8_raw_t>(0x30)); // 0.125
    }

    // positive infinity value
    CK_TILE_HOST_DEVICE static constexpr bf8_t infinity()
    {
        return bit_cast<bf8_t>(static_cast<bf8_raw_t>(0x7c)); // 0b01111100
    }

    // quiet NaN
    CK_TILE_HOST_DEVICE static constexpr bf8_t quiet_NaN()
    {
        return bit_cast<bf8_t>(static_cast<bf8_raw_t>(0x7F)); // 0b01111111
    }

    // signaling NaN
    CK_TILE_HOST_DEVICE static constexpr bf8_t signaling_NaN()
    {
        return bit_cast<bf8_t>(static_cast<bf8_raw_t>(0xFF));
    }

    // smallest positive subnormal value
    CK_TILE_HOST_DEVICE static constexpr bf8_t denorm_min()
    {
        return bit_cast<bf8_t>(static_cast<bf8_raw_t>(0x01));
    }

    CK_TILE_HOST_DEVICE static constexpr bf8_t zero()
    {
        return bit_cast<bf8_t>(static_cast<bf8_raw_t>(0));
    }
};
#else
template <>
struct numeric<fp8_t>
{
    // minimum finite value, or minimum positive normalized value for float
    CK_TILE_HOST_DEVICE static constexpr fp8_t min()
    {
        return bit_cast<fp8_t>(static_cast<fp8_raw_t>(0x08));
    }

    // minumum finite value
    CK_TILE_HOST_DEVICE static constexpr fp8_t lowest()
    {
        return bit_cast<fp8_t>(static_cast<fp8_raw_t>(0xff));
    }

    // maximum finite value
    CK_TILE_HOST_DEVICE static constexpr fp8_t max()
    {
        return bit_cast<fp8_t>(static_cast<fp8_raw_t>(0x7f));
    }

    // difference between 1.0 and next value representable by float
    CK_TILE_HOST_DEVICE static constexpr fp8_t epsilon()
    {
        return bit_cast<fp8_t>(static_cast<fp8_raw_t>(0x20));
    }

    // maximum rounding error
    // bin :  7 6543 210
    // bits:  s eeee mmm
    //        0 0110 000 (0.5)
    //
    CK_TILE_HOST_DEVICE static constexpr fp8_t round_error()
    {
        return bit_cast<fp8_t>(static_cast<fp8_raw_t>(0x30));
    }

    // positive infinity value
    CK_TILE_HOST_DEVICE static constexpr fp8_t infinity()
    {
        return bit_cast<fp8_t>(static_cast<fp8_raw_t>(0x80));
    }

    // quiet NaN
    CK_TILE_HOST_DEVICE static constexpr fp8_t quiet_NaN()
    {
        return bit_cast<fp8_t>(static_cast<fp8_raw_t>(0x80));
    }

    // signaling NaN
    CK_TILE_HOST_DEVICE static constexpr fp8_t signaling_NaN()
    {
        return bit_cast<fp8_t>(static_cast<fp8_raw_t>(0x80));
    }

    // smallest positive subnormal value
    CK_TILE_HOST_DEVICE static constexpr fp8_t denorm_min()
    {
        return bit_cast<fp8_t>(static_cast<fp8_raw_t>(0x01));
    }

    CK_TILE_HOST_DEVICE static constexpr fp8_t zero()
    {
        return bit_cast<fp8_t>(static_cast<fp8_raw_t>(0));
    }
};

template <>
struct numeric<bf8_t>
{
    // minimum finite value, or minimum positive normalized value for float
    CK_TILE_HOST_DEVICE static constexpr bf8_t min()
    {
        return bit_cast<bf8_t>(static_cast<bf8_raw_t>(0x04));
    }

    // minumum finite value
    CK_TILE_HOST_DEVICE static constexpr bf8_t lowest()
    {
        return bit_cast<bf8_t>(static_cast<bf8_raw_t>(0xff));
    }

    // maximum finite value
    CK_TILE_HOST_DEVICE static constexpr bf8_t max()
    {
        return bit_cast<bf8_t>(static_cast<bf8_raw_t>(0x7f));
    }

    // difference between 1.0 and next value representable by float
    CK_TILE_HOST_DEVICE static constexpr bf8_t epsilon()
    {
        return bit_cast<bf8_t>(static_cast<bf8_raw_t>(0x34));
    }

    // maximum rounding error
    // bin :  7 65432 10
    // bits:  s eeeee mm
    //        0 01110 00 (0.5)
    //
    CK_TILE_HOST_DEVICE static constexpr bf8_t round_error()
    {
        return bit_cast<bf8_t>(static_cast<bf8_raw_t>(0x38));
    }

    // positive infinity value
    CK_TILE_HOST_DEVICE static constexpr bf8_t infinity()
    {
        return bit_cast<bf8_t>(static_cast<bf8_raw_t>(0x80));
    }

    // quiet NaN
    CK_TILE_HOST_DEVICE static constexpr bf8_t quiet_NaN()
    {
        return bit_cast<bf8_t>(static_cast<bf8_raw_t>(0x80));
    }

    // signaling NaN
    CK_TILE_HOST_DEVICE static constexpr bf8_t signaling_NaN()
    {
        return bit_cast<bf8_t>(static_cast<bf8_raw_t>(0x80));
    }

    // smallest positive subnormal value
    CK_TILE_HOST_DEVICE static constexpr bf8_t denorm_min()
    {
        return bit_cast<bf8_t>(static_cast<bf8_raw_t>(0x01));
    }

    CK_TILE_HOST_DEVICE static constexpr bf8_t zero()
    {
        return bit_cast<bf8_t>(static_cast<bf8_raw_t>(0));
    }
};
#endif

#if CK_TILE_USE_CUSTOM_DATA_TYPE
CK_TILE_ARITHMETIC_USING_FLOAT(CK_TILE_HOST_DEVICE, fp8_t)
CK_TILE_ARITHMETIC_USING_FLOAT(CK_TILE_HOST_DEVICE, bf8_t)
#endif

// math
template <typename T>
CK_TILE_HOST_DEVICE T abs(const T& x)
{
    static_assert(std::is_same_v<T, fp8_t> || std::is_same_v<T, bf8_t>,
                  "Only fp8_t and bf8_t are supported");
    return bit_cast<T>(static_cast<uint8_t>(bit_cast<uint8_t>(x) & numeric_traits<T>::abs_mask));
}

CK_TILE_HOST_DEVICE
bool isnan(const fp8_t& x)
{
    uint8_t xx = bit_cast<fp8_raw_t>(x);

#if CK_TILE_USE_OCP_FP8
    return (xx & 0x7f) == 0x7f;
#else
    return xx == 0x80;
#endif
}
#if CK_TILE_USE_CUSTOM_DATA_TYPE
CK_TILE_DEVICE
fp8_t sqrt(fp8_t x) { return static_cast<fp8_t>(__builtin_amdgcn_sqrtf(static_cast<float>(x))); };

CK_TILE_DEVICE
fp8_t exp(fp8_t x) { return static_cast<fp8_t>(__ocml_exp_f32(static_cast<float>(x))); };

CK_TILE_DEVICE
fp8_t exp2(fp8_t x) { return static_cast<fp8_t>(exp2f(static_cast<float>(x))); };

CK_TILE_DEVICE
fp8_t log(fp8_t x) { return static_cast<fp8_t>(__logf(static_cast<float>(x))); };
#endif

CK_TILE_HOST_DEVICE
bool isnan(const bf8_t& x)
{
    uint8_t xx = bit_cast<bf8_raw_t>(x);

#if CK_TILE_USE_OCP_FP8
    return (xx & 0x7f) > 0x7c;
#else
    return xx == 0x80;
#endif
}

#if CK_TILE_USE_CUSTOM_DATA_TYPE
CK_TILE_DEVICE
bf8_t sqrt(bf8_t x) { return static_cast<bf8_t>(__builtin_amdgcn_sqrtf(static_cast<float>(x))); };

CK_TILE_DEVICE
bf8_t exp(bf8_t x) { return static_cast<bf8_t>(__ocml_exp_f32(static_cast<float>(x))); };

CK_TILE_DEVICE
bf8_t exp2(bf8_t x) { return static_cast<bf8_t>(exp2f(static_cast<float>(x))); };

CK_TILE_DEVICE
bf8_t log(bf8_t x) { return static_cast<bf8_t>(__logf(static_cast<float>(x))); };
#endif

} // namespace ck_tile
