// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.

#pragma once

#include <cmath>
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/mxfp_convert.hpp"

#if defined(__gfx950__)
#define CK_TILE_FP4_CVT_DEVICE 1
#else
#define CK_TILE_FP4_CVT_DEVICE 0
#endif

#define TEST_convert_with_table 0

namespace ck_tile {

using fp32_t   = float;
using fp32x2_t = float __attribute__((ext_vector_type(2)));
using fp16x2_t = _Float16 __attribute__((ext_vector_type(2)));
using bf16x2_t = bf16_raw_t __attribute__((ext_vector_type(2)));

CK_TILE_HOST_DEVICE constexpr uint8_t float_to_e2m1(float);

// TODO: Add stochastic method
struct pk_float4_e2m1_t
{
    static constexpr int exponent = 2;
    static constexpr int mantissa = 1;
    static constexpr int bias     = 1;
    // TODO: Can we merge raw_type and type?
    using raw_type = uint8_t;
    using type     = raw_type;
    raw_type data;

    CK_TILE_HOST_DEVICE constexpr pk_float4_e2m1_t() : data{type{}} {}
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    CK_TILE_HOST_DEVICE constexpr pk_float4_e2m1_t(T init) : data{static_cast<type>(init)}
    {
    }
    CK_TILE_HOST_DEVICE explicit constexpr pk_float4_e2m1_t(float init) : data{float_to_e2m1(init)}
    {
    }
    CK_TILE_HOST_DEVICE constexpr operator type() const { return data; }
    CK_TILE_HOST_DEVICE constexpr raw_type& get() { return data; }
    CK_TILE_HOST_DEVICE constexpr raw_type get() const { return data; }
    CK_TILE_HOST_DEVICE constexpr operator float() const;
    CK_TILE_HOST_DEVICE constexpr operator fp32x2_t() const;
    CK_TILE_HOST_DEVICE constexpr operator fp16_t() const;
    CK_TILE_HOST_DEVICE constexpr operator fp16x2_t() const;
    CK_TILE_HOST_DEVICE constexpr operator bf16_t() const;
    CK_TILE_HOST_DEVICE constexpr operator bf16x2_t() const;

    template <index_t I>
    CK_TILE_HOST_DEVICE constexpr raw_type unpack(number<I>) const;
    CK_TILE_HOST_DEVICE constexpr static pk_float4_e2m1_t pack(const type x0, const type x1)
    {
        return (x1 << 4) | (x0 & 0b00001111);
    }

#if TEST_convert_with_table
    static constexpr float e2m1_to_fp32_table[16] = {
        0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6};
    static constexpr fp16_t e2m1_to_fp16_table[16] = {
        bit_cast<fp16_t>(static_cast<uint16_t>(0x0000)), //  0
        bit_cast<fp16_t>(static_cast<uint16_t>(0x3800)), //  0.5
        bit_cast<fp16_t>(static_cast<uint16_t>(0x3C00)), //  1
        bit_cast<fp16_t>(static_cast<uint16_t>(0x3E00)), //  1.5
        bit_cast<fp16_t>(static_cast<uint16_t>(0x4000)), //  2
        bit_cast<fp16_t>(static_cast<uint16_t>(0x4200)), //  3
        bit_cast<fp16_t>(static_cast<uint16_t>(0x4400)), //  4
        bit_cast<fp16_t>(static_cast<uint16_t>(0x4600)), //  6
        bit_cast<fp16_t>(static_cast<uint16_t>(0x8000)), // -0
        bit_cast<fp16_t>(static_cast<uint16_t>(0xB800)), // -0.5
        bit_cast<fp16_t>(static_cast<uint16_t>(0xBC00)), // -1
        bit_cast<fp16_t>(static_cast<uint16_t>(0xBE00)), // -1.5
        bit_cast<fp16_t>(static_cast<uint16_t>(0xC000)), // -2
        bit_cast<fp16_t>(static_cast<uint16_t>(0xC200)), // -3
        bit_cast<fp16_t>(static_cast<uint16_t>(0xC400)), // -4
        bit_cast<fp16_t>(static_cast<uint16_t>(0xC600))  // -6
    };
#endif
};

using pk_fp4_t     = pk_float4_e2m1_t;
using pk_fp4_raw_t = typename pk_fp4_t::raw_type;

template <>
struct numeric_traits<pk_fp4_t>
{
    using bitwise_type = pk_fp4_raw_t;

    static constexpr int exp        = 2;
    static constexpr int mant       = 1;
    static constexpr int bias       = 1;
    static constexpr int PackedSize = 2;
};

// limits
template <class T>
struct numeric;

template <>
struct numeric<pk_fp4_t>
{
    static constexpr pk_fp4_raw_t binary_min_normal    = 0b00100010; // 1
    static constexpr pk_fp4_raw_t binary_max_normal    = 0b01110111; // 6
    static constexpr pk_fp4_raw_t binary_lowest_normal = 0b11111111; // -6
    static constexpr pk_fp4_raw_t binary_min_subnorm   = 0b00010001; // 0.5
    static constexpr pk_fp4_raw_t binary_max_subnorm   = 0b00010001; // 0.5
    static constexpr pk_fp4_raw_t binary_zero          = 0b00000000; // 0
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t min() { return binary_min_normal; }
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t max() { return binary_max_normal; }
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t lowest() { return binary_lowest_normal; }
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t epsilon() { return binary_min_subnorm; }
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t round_error() { return binary_min_subnorm; }
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t zero() { return binary_zero; }
    CK_TILE_HOST_DEVICE static constexpr fp8_t denorm_min() { return binary_min_subnorm; }

    CK_TILE_HOST_DEVICE static constexpr bool has_inf() { return false; }
    // N/A
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t infinity() { return max(); }
    // N/A
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t quiet_NaN() { return max(); }
    // N/A
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t signaling_NaN() { return max(); }
};

template <index_t I>
CK_TILE_HOST_DEVICE constexpr pk_fp4_raw_t pk_fp4_t::unpack(number<I>) const
{
    static_assert(I < 2, "Index is out of range.");
    if constexpr(I == 1)
        return (data >> 4);
    else
        return data & 0b00001111;
}
CK_TILE_ARITHMETIC_USING_FLOAT(CK_TILE_HOST_DEVICE, pk_fp4_t)
// TODO: consider replace this macro to improve performance

#if CK_TILE_FP4_CVT_DEVICE
namespace impl {

template <typename T>
CK_TILE_DEVICE T _from_f4(pk_fp4_raw_t src, float scale = 1.0f)
{
    if constexpr(std::is_same_v<T, fp32_t>)
        return fp32x2_t(__builtin_amdgcn_cvt_scalef32_pk_f32_fp4(src, scale, 0))[0];
    else if constexpr(std::is_same_v<T, fp32x2_t>)
        return __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(src, scale, 0);
    else if constexpr(std::is_same_v<T, fp16_t>)
        return fp16x2_t(__builtin_amdgcn_cvt_scalef32_pk_f16_fp4(src, scale, 0))[0];
    else if constexpr(std::is_same_v<T, fp16x2_t>)
        return __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(src, scale, 0);
    else if constexpr(std::is_same_v<T, bf16_t>)
        return bf16x2_t(__builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(src, scale, 0))[0];
    else if constexpr(std::is_same_v<T, bf16x2_t>)
        return __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(src, scale, 0);
    else
        static_assert(std::false_type::value, "Unsupported type.");
    return T{};
}
template <typename T>
CK_TILE_DEVICE pk_fp4_raw_t _to_f4(T src, float scale = 1.0f)
{
    union
    {
        uint32_t u32;
        pk_fp4_raw_t pf4[4];
    } cvt{0};
    if constexpr(std::is_same_v<T, fp32_t>)
        cvt.u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(cvt.u32, src, src, scale, 0);
    else if constexpr(std::is_same_v<T, fp32x2_t>)
        cvt.u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(cvt.u32, src[0], src[1], scale, 0);
    else if constexpr(std::is_same_v<T, fp16_t>)
        cvt.u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(cvt.u32, fp16x2_t{src, src}, scale, 0);
    else if constexpr(std::is_same_v<T, fp16x2_t>)
        cvt.u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(cvt.u32, src, scale, 0);
    else if constexpr(std::is_same_v<T, bf16_t>)
        cvt.u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(cvt.u32, bf16x2_t{src, src}, scale, 0);
    else if constexpr(std::is_same_v<T, bf16x2_t>)
        cvt.u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(cvt.u32, src, scale, 0);
    else
        static_assert(std::false_type::value, "Unsupported type.");
    return cvt.pf4[0];
}

} // namespace impl
#endif

CK_TILE_HOST_DEVICE constexpr pk_fp4_t::operator bf16_t() const
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_from_f4<bf16_t>(data);
#else
    return bf16_t{type_convert<bf16_t>(convert_to_float<pk_fp4_t>(unpack(number<0>{})))};
#endif
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t::operator bf16x2_t() const
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_from_f4<bf16x2_t>(data);
#else
    return bf16x2_t{type_convert<bf16_t>(convert_to_float<pk_fp4_t>(unpack(number<0>{}))),
                    type_convert<bf16_t>(convert_to_float<pk_fp4_t>(unpack(number<1>{})))};
#endif
}

// TODO: make float_to_e2m1 generic so that we can convert from directrly.
CK_TILE_HOST_DEVICE constexpr pk_fp4_raw_t float_to_e2m1(float x)
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_to_f4(x);
#else
    return convert_to_type<pk_fp4_t>(x);
#endif
}
CK_TILE_HOST_DEVICE constexpr fp32x2_t pk_fp4_to_fp32x2(const pk_fp4_t& x) { return fp32x2_t(x); }
CK_TILE_HOST_DEVICE constexpr fp16x2_t pk_fp4_to_fp16x2(const pk_fp4_t& x) { return fp16x2_t(x); }
CK_TILE_HOST_DEVICE constexpr bf16x2_t pk_fp4_to_bf16x2(const pk_fp4_t& x) { return bf16x2_t(x); }
CK_TILE_HOST_DEVICE constexpr pk_fp4_t float_to_pk_fp4(const float& x) { return float_to_e2m1(x); }
CK_TILE_HOST_DEVICE constexpr pk_fp4_t fp16_to_pk_fp4(const fp16_t& x)
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_to_f4(x);
#else
    return float_to_e2m1(type_convert<float>(x));
#endif
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t bf16_to_pk_fp4(const bf16_t& x)
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_to_f4(x);
#else
    return float_to_e2m1(type_convert<float>(x));
#endif
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t fp16x2_to_pk_fp4(const fp16x2_t& x)
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_to_f4(x);
#else
    return pk_fp4_t::pack(float_to_e2m1(type_convert<float>(x[0])),
                          float_to_e2m1(type_convert<float>(x[1])));
#endif
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t bf16x2_to_pk_fp4(const bf16x2_t& x)
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_to_f4(x);
#else
    return pk_fp4_t::pack(float_to_e2m1(type_convert<float>(x[0])),
                          float_to_e2m1(type_convert<float>(x[1])));
#endif
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t fp32x2_to_pk_fp4(const fp32x2_t& x)
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_to_f4(x);
#else
    return pk_fp4_t::pack(float_to_e2m1(x[0]), float_to_e2m1(x[1]));
#endif
}

#if TEST_convert_with_table == 0
CK_TILE_HOST_DEVICE constexpr pk_fp4_t::operator float() const
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_from_f4<fp32_t>(data);
#else
    return convert_to_float<pk_fp4_t>(unpack(number<0>{}));
#endif
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t::operator fp32x2_t() const
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_from_f4<fp32x2_t>(data);
#else
    return fp32x2_t{convert_to_float<pk_fp4_t>(unpack(number<0>{})),
                    convert_to_float<pk_fp4_t>(unpack(number<1>{}))};
#endif
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t::operator fp16_t() const
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_from_f4<fp16_t>(data);
#else
    return fp16_t{type_convert<fp16_t>(convert_to_float<pk_fp4_t>(unpack(number<0>{})))};
#endif
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t::operator fp16x2_t() const
{
#if CK_TILE_FP4_CVT_DEVICE
    return impl::_from_f4<fp16x2_t>(data);
#else
    return fp16x2_t{type_convert<fp16_t>(convert_to_float<pk_fp4_t>(unpack(number<0>{}))),
                    type_convert<fp16_t>(convert_to_float<pk_fp4_t>(unpack(number<1>{})))};
#endif
}
#else
CK_TILE_HOST_DEVICE constexpr pk_fp4_t::operator float() const
{
    return e2m1_to_fp32_table[data & 0xf];
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t::operator fp32x2_t() const
{
    return fp32x2_t{e2m1_to_fp32_table[data & 0xf], e2m1_to_fp32_table[data >> 4]};
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t::operator fp16_t() const
{
    return e2m1_to_fp16_table[data & 0xf];
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t::operator fp16x2_t() const
{
    return fp16x2_t{e2m1_to_fp16_table[data & 0xf], e2m1_to_fp16_table[data >> 4]};
}
#endif

} // namespace ck_tile
