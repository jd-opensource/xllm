// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include <stdint.h>
#include <tuple>
#include <type_traits>

namespace ck_tile {

// return 0 if data is not fp16 or fp32
template <typename T, uint32_t seed_>
struct prand_generator_t
{
    CK_TILE_HOST_DEVICE uint32_t operator()(int, T, uint32_t = seed_) { return 0; }
};

// version for fp32
template <uint32_t seed_>
struct prand_generator_t<float, seed_>
{
    CK_TILE_HOST_DEVICE uint32_t operator()(int id, float val, uint32_t seed = seed_)
    {
        uint32_t x         = *(reinterpret_cast<uint32_t*>(&val));
        uint32_t drop_bits = uint32_t(x) & 0xFFFFu;
        drop_bits ^= x >> 16;
        drop_bits = ((drop_bits & 31) << 11) | (drop_bits >> 5);
        drop_bits *= 0x7000149;
        // NOTE: If id is in 64 bit, we are only using lower 32 bit.
        //       So, it can have an effect of using same id for multiple elements when the id is
        //       very large!
        uint32_t rng = (drop_bits ^ 0x13371337 ^ (id * 229791) ^ seed);
        return rng;
    }
};

// version for fp16
template <uint32_t seed_>
struct prand_generator_t<half_t, seed_>
{
    CK_TILE_HOST_DEVICE uint32_t operator()(int id, half_t val, uint32_t seed = seed_)
    {
        uint16_t x         = *(reinterpret_cast<uint16_t*>(&val));
        uint32_t drop_bits = uint32_t(x) & 0xFFFFu;
        drop_bits          = ((drop_bits & 31) << 11) | (drop_bits >> 5);
        drop_bits *= 0x7000149;
        // NOTE: If id is in 64 bit, we are only using lower 32 bit.
        //       So, it can have an effect of using same id for multiple elements when the id is
        //       very large!
        uint32_t rng = (drop_bits ^ 0x13371337 ^ (id * 229791) ^ seed);
        return rng;
    }
};

} // namespace ck_tile
