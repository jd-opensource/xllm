// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

namespace ck_tile {
// modify from include/ck/utility/mxfp_utils.hpp

template <typename T>
struct numeric_utils : numeric_traits<T>
{

    using traits   = numeric_traits<T>;
    using _numeric = numeric<T>;
    using raw_type = typename T::raw_type;

    static constexpr int exp_mask = (1 << traits::exp) - 1;

    static constexpr int get_exponent(raw_type x)
    {
        // TODO: check if repeated calls are optimized.
        return (x >> traits::mant) & exp_mask;
    }
    static constexpr bool is_positive(raw_type x)
    {
        return (x >> (traits::exp + traits::mant)) == _numeric::binary_zero;
    }
    static constexpr bool is_subnormal(raw_type x)
    {
        return get_exponent(x) == _numeric::binary_zero;
    }
    // TODO: replace double with template arg?
    static constexpr double get_mantissa(raw_type x)
    {
        double mantissa = is_subnormal(x) ? 0.0f : 1.0f;
        for(uint32_t i = 0; i < traits::mant; ++i)
        {
            mantissa += std::ldexp(static_cast<float>(x & 0b1), -(traits::mant - i));
            x >>= 1;
        }
        return mantissa;
    }
};

template <typename T>
CK_TILE_HOST_DEVICE float convert_to_float(typename T::raw_type data, int scale_exp = 127)
{
    using utils                    = numeric_utils<T>;
    static constexpr int e8m0_bias = 127; // TODO: make it generic.
    float sign                     = utils::is_positive(data) ? 1.0 : -1.0;
    int exp    = (utils::is_subnormal(data) ? 1 : utils::get_exponent(data)) - utils::bias;
    float mant = utils::get_mantissa(data);

    return std::ldexp(sign * mant, exp + scale_exp - e8m0_bias);
}

template <typename T>
CK_TILE_HOST_DEVICE typename T::raw_type convert_to_type(float value)
{
    using bitwise_type = typename numeric_traits<T>::bitwise_type;

    if(std::abs(value) > float(numeric<T>::max()))
    {
        float max_value = numeric<T>::max();

        // cppcheck-suppress redundantAssignment
        uint32_t max_bitwise = bit_cast<uint32_t>(max_value);

        // cppcheck-suppress redundantAssignment
        bitwise_type sign =
            bit_cast<uint32_t>(value) >> (numeric_traits<float>::exp + numeric_traits<float>::mant);
        bitwise_type exp =
            ((max_bitwise >> numeric_traits<float>::mant) & numeric_traits<float>::exp_mask) -
            (numeric_traits<float>::bias - numeric_traits<T>::bias);
        bitwise_type mantissa =
            max_bitwise >> (numeric_traits<float>::mant - numeric_traits<T>::mant);

        uint32_t mant_prev = max_bitwise >> (numeric_traits<float>::mant - numeric_traits<T>::mant);
        mant_prev &= ((1 << numeric_traits<T>::mant) - 1);
        mant_prev--;

        mant_prev <<= (numeric_traits<float>::mant - numeric_traits<T>::mant);
        uint32_t prev_bit =
            ((max_bitwise >> numeric_traits<float>::mant) << numeric_traits<float>::mant) |
            mant_prev;

        float prev_val = bit_cast<float>(prev_bit);
        float diff     = max_value - prev_val;

        float actual_max = max_value + (diff / 2);

        if(std::abs(value) < actual_max)
        {
            return sign << ((numeric_traits<T>::exp + numeric_traits<T>::mant)) |
                   (exp << numeric_traits<T>::mant) | mantissa;
        }
        else
        {
            if constexpr(!numeric<T>::has_inf())
            {

                return (1 << (numeric_traits<T>::mant + numeric_traits<T>::exp)) - 1;
            }
            else
            {
                exp++;
                return sign << ((numeric_traits<T>::exp + numeric_traits<T>::mant)) |
                       (exp << numeric_traits<T>::mant);
            }
        }
    }
    const int mfmt = numeric_traits<float>::mant;
    uint32_t x;
    x = bit_cast<uint32_t>(value);

    uint32_t head, mantissa;
    int32_t exponent, bias;
    uint32_t sign;

    head     = x & numeric_traits<float>::head_mask;
    mantissa = x & numeric_traits<float>::mant_mask;
    exponent = (head >> numeric_traits<float>::mant) & numeric_traits<float>::exp_mask;
    sign     = head >> (numeric_traits<float>::mant + numeric_traits<float>::exp);
    bias     = numeric_traits<float>::bias;

    if(x == 0)
    {
        return 0b0;
    }

    const int mini_bias                  = numeric_traits<T>::bias;
    const int mini_denormal_act_exponent = 1 - mini_bias;

    int act_exponent, out_exponent, exponent_diff;

    bool is_subnorm = false;

    if(exponent == 0)
    {
        act_exponent  = exponent - bias + 1;
        exponent_diff = mini_denormal_act_exponent - act_exponent;
        is_subnorm    = true;
    }
    else
    {
        act_exponent = exponent - bias;
        if(act_exponent <= mini_denormal_act_exponent)
        {
            exponent_diff = mini_denormal_act_exponent - act_exponent;
            is_subnorm    = true;
        }
        else
        {
            exponent_diff = 0;
        }
        mantissa += (1UL << mfmt);
    }

    auto shift_amount = (mfmt - numeric_traits<T>::mant + exponent_diff);
    shift_amount      = (shift_amount >= 64) ? 63 : shift_amount;
    bool midpoint     = (mantissa & ((1UL << shift_amount) - 1)) == (1UL << (shift_amount - 1));

    float min_subnorm = float(numeric<T>::epsilon()) * (sign ? -1 : 1);

    if(is_subnorm && std::abs(value) < std::abs(min_subnorm))
    {
        // closer to 0
        if(std::abs(value) <= std::abs(min_subnorm - value))
            return sign << (numeric_traits<T>::exp + numeric_traits<T>::mant);
        else
            return 1 | (sign << (numeric_traits<T>::exp + numeric_traits<T>::mant));
    }

    if(exponent_diff > 0)
        mantissa >>= exponent_diff;
    else if(exponent_diff == -1)
        mantissa <<= -exponent_diff;
    bool implicit_one = mantissa & (1 << mfmt);
    out_exponent      = (act_exponent + exponent_diff) + mini_bias - (implicit_one ? 0 : 1);

    uint32_t drop_mask = (1UL << (mfmt - numeric_traits<T>::mant)) - 1;
    bool odd           = mantissa & (1UL << (mfmt - numeric_traits<T>::mant));
    mantissa += (midpoint ? (odd ? mantissa : mantissa - 1) : mantissa) & drop_mask;

    if(out_exponent == 0)
    {
        if((1UL << mfmt) & mantissa)
        {
            out_exponent = 1;
        }
    }
    else
    {
        if((1UL << (mfmt + 1)) & mantissa)
        {
            mantissa >>= 1;
            out_exponent++;
        }
    }

    mantissa >>= (mfmt - numeric_traits<T>::mant);

    if(out_exponent == 0 && mantissa == 0)
    {
        return sign << (numeric_traits<T>::exp + numeric_traits<T>::mant);
    }

    mantissa &= (1UL << numeric_traits<T>::mant) - 1;
    return (sign << (numeric_traits<T>::exp + numeric_traits<T>::mant)) |
           (out_exponent << numeric_traits<T>::mant) | mantissa;
}

} // namespace ck_tile
