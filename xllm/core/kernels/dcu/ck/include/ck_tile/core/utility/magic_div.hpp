// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include <stdint.h>

namespace ck_tile {

// magic number division
// Caution:
//   1. For uint32_t as dividend: magic number division implementation being used would produce
//   correct result if the dividend is uint32_t and its value is within 31-bit value range.
//   2. For int32_t as dividendd: magic number division for int32_t dividened has not been
//   implemented, the int32_t dividend would be bit-wise interpreted as uint32_t and magic number
//   division implementation for uint32_t is then used. Therefore, dividend value need to be
//   non-negative.
// TODO:
//   1. Implement magic number divison for int32_t
//   2. Implement magic number divison for unit32_t with 32-bit value range
struct magic_division32_bit_range
{
    // uint32_t
    CK_TILE_HOST_DEVICE static constexpr auto calculate_magic_numbers(uint32_t divisor)
    {
        // WARNING: magic division is only valid for division inside this range.
        // assert(divisor >= 1 && divisor <= INT32_MAX)

        uint32_t shift_u32 = 0;

        while((1U << shift_u32) < divisor)
        {
            shift_u32++;
        };

        uint64_t tmp_u64        = ((1UL << shift_u32) - divisor) << 32;
        uint32_t multiplier_u32 = tmp_u64 / divisor + 1;

        return make_tuple(multiplier_u32, shift_u32);
    }

    template <auto Divisor, typename = std::enable_if_t<(0 < Divisor)>>
    CK_TILE_HOST_DEVICE static constexpr auto calculate_magic_numbers(constant<Divisor>)
    {
        constexpr auto tmp = calculate_magic_numbers(uint32_t{Divisor});

        constexpr uint32_t multiplier = tmp[number<0>{}];
        constexpr uint32_t shift      = tmp[number<1>{}];

        return make_tuple(constant<multiplier>{}, constant<shift>{});
    }

    // magic division for uint32_t
    CK_TILE_DEVICE static constexpr uint32_t
    do_magic_division(uint32_t dividend, uint32_t multiplier, uint32_t shift)
    {
        if(__builtin_is_constant_evaluated())
        {
            uint32_t tmp = (static_cast<uint64_t>(dividend) * multiplier) >> 32;
            return (tmp + dividend) >> shift;
        }
        else
        {
            uint32_t tmp = __umulhi(dividend, multiplier);
            return (tmp + dividend) >> shift;
        }
    }

    CK_TILE_HOST static constexpr uint32_t
    do_magic_division(uint32_t dividend, uint32_t multiplier, uint32_t shift)
    {
        uint32_t tmp = (static_cast<uint64_t>(dividend) * multiplier) >> 32;
        return (tmp + dividend) >> shift;
    }

    // magic division for int32_t
    // HACK: use dividend_i32 as if it's uint32_t, dividend_i32 need to be
    // non-negative for result to be correct
    // TODO: figure out how to do magic number divison for int32_t as dividended
    CK_TILE_DEVICE static constexpr int32_t
    do_magic_division(int32_t dividend_i32, uint32_t multiplier, uint32_t shift)
    {
        if(__builtin_is_constant_evaluated())
        {
            uint32_t dividend_u32 = bit_cast<uint32_t>(dividend_i32);
            uint32_t tmp          = (static_cast<uint64_t>(dividend_u32) * multiplier) >> 32;
            return (tmp + dividend_u32) >> shift;
        }
        else
        {
            uint32_t dividend_u32 = bit_cast<uint32_t>(dividend_i32);
            uint32_t tmp          = __umulhi(dividend_u32, multiplier);
            return (tmp + dividend_u32) >> shift;
        }
    }

    CK_TILE_HOST static constexpr int32_t
    do_magic_division(int32_t dividend_i32, uint32_t multiplier, uint32_t shift)
    {
        uint32_t dividend_u32 = bit_cast<uint32_t>(dividend_i32);
        uint32_t tmp          = (static_cast<uint64_t>(dividend_u32) * multiplier) >> 32;
        return (tmp + dividend_u32) >> shift;
    }
};

// magic number division
// This version on works for divisor and dividended between [0, 1 << 16]
struct magic_division16_bit_range
{
    // uint32_t
    CK_TILE_HOST_DEVICE static constexpr auto calculate_magic_numbers(uint32_t divisor)
    {
        // WARNING: magic division is only valid for division inside this range.
        // assert(divisor >= 1 && divisor <= (1U << 16));

        uint32_t shift_u32 = 0;

        while((1U << shift_u32) < divisor)
        {
            shift_u32++;
        };

        uint32_t one            = 1;
        uint32_t multiplier_u32 = ((one << 16) * ((one << shift_u32) - divisor)) / divisor + 1;

        return make_tuple(multiplier_u32, shift_u32);
    }

    // integral_constant<uint32_t, .>
    template <auto Divisor>
    CK_TILE_HOST_DEVICE static constexpr auto calculate_magic_numbers(constant<Divisor>)
    {
        constexpr auto tmp = calculate_magic_numbers(uint32_t{Divisor});

        constexpr uint32_t multiplier = tmp[number<0>{}];
        constexpr uint32_t shift      = tmp[number<1>{}];

        return make_tuple(constant<multiplier>{}, constant<shift>{});
    }

    // magic division for uint32_t
    CK_TILE_DEVICE static constexpr uint32_t
    do_magic_division(uint32_t dividend, uint32_t multiplier, uint32_t shift)
    {
        uint32_t tmp = (dividend * multiplier) >> 16;
        return (tmp + dividend) >> shift;
    }

    CK_TILE_HOST static constexpr uint32_t
    do_magic_division(uint32_t dividend, uint32_t multiplier, uint32_t shift)
    {
        uint32_t tmp = (dividend * multiplier) >> 16;
        return (tmp + dividend) >> shift;
    }

    // magic division for int32_t
    // HACK: use dividend_i32 as if it's uint32_t, dividend_i32 need to be
    // non-negative for result to be correct
    // TODO: figure out how to do magic number divison for int32_t as dividended
    CK_TILE_DEVICE static constexpr int32_t
    do_magic_division(int32_t dividend_i32, uint32_t multiplier, uint32_t shift)
    {
        uint32_t dividend_u32 = bit_cast<uint32_t>(dividend_i32);
        uint32_t tmp          = (dividend_u32 * multiplier) >> 16;
        return (tmp + dividend_u32) >> shift;
    }

    CK_TILE_HOST static constexpr int32_t
    do_magic_division(int32_t dividend_i32, uint32_t multiplier, uint32_t shift)
    {
        uint32_t dividend_u32 = bit_cast<uint32_t>(dividend_i32);
        uint32_t tmp          = (dividend_u32 * multiplier) >> 16;
        return (tmp + dividend_u32) >> shift;
    }
};

// use 32bit version
using magic_division = magic_division32_bit_range;

struct mdiv
{
    // 1 dword -> 3 dword storage
    uint32_t divisor;
    uint32_t multiplier;
    uint32_t shift; // TODO: 8 bit is enough

    // prefer construct on host
    CK_TILE_HOST_DEVICE mdiv(uint32_t divisor_) : divisor(divisor_)
    {
        auto tmp = magic_division::calculate_magic_numbers(divisor_);

        multiplier = tmp[number<0>{}];
        shift      = tmp[number<1>{}];
    }

    CK_TILE_HOST_DEVICE mdiv() : divisor(0), multiplier(0), shift(0) {}

    CK_TILE_HOST_DEVICE void update(uint32_t divisor_)
    {
        divisor  = divisor_;
        auto tmp = magic_division::calculate_magic_numbers(divisor_);

        multiplier = tmp[number<0>{}];
        shift      = tmp[number<1>{}];
    }

    CK_TILE_HOST_DEVICE uint32_t div(uint32_t dividend_) const
    {
        return magic_division::do_magic_division(dividend_, multiplier, shift);
    }

    CK_TILE_HOST_DEVICE void
    divmod(uint32_t dividend_, uint32_t& quotient_, uint32_t& remainder_) const
    {
        quotient_  = div(dividend_);
        remainder_ = dividend_ - (quotient_ * divisor);
    }

    CK_TILE_HOST_DEVICE uint32_t get() const { return divisor; }
};

struct mdiv2
{
    // 1 dword -> 2 dword storage, divisor need compute from runtime
    uint32_t multiplier;
    uint32_t shift; // TODO: 8 bit is enough

    // prefer construct on host
    CK_TILE_HOST_DEVICE mdiv2(uint32_t divisor_)
    {
        auto tmp = magic_division::calculate_magic_numbers(divisor_);

        multiplier = tmp[number<0>{}];
        shift      = tmp[number<1>{}];
    }

    CK_TILE_HOST_DEVICE mdiv2() : multiplier(0), shift(0) {}

    CK_TILE_HOST_DEVICE uint32_t div(uint32_t dividend_) const
    {
        return magic_division::do_magic_division(dividend_, multiplier, shift);
    }

    CK_TILE_HOST_DEVICE void
    divmod(uint32_t dividend_, uint32_t divisor_, uint32_t& quotient_, uint32_t& remainder_) const
    {
        quotient_  = div(dividend_);
        remainder_ = dividend_ - (quotient_ * divisor_);
    }
};

} // namespace ck_tile
