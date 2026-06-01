// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"

namespace ck_tile {

// Reference: https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/philox.cuh
class philox
{
    public:
    CK_TILE_HOST_DEVICE philox(unsigned long long seed_, unsigned long long offset_)
        : seed(reinterpret_cast<const uint2&>(seed_))
    {

        ull2* tmp = reinterpret_cast<ull2*>(&counter);
        tmp->x    = offset_;
    }

    CK_TILE_HOST_DEVICE uint4 get_philox_4x32(const unsigned long long subsequence) const
    {

        uint4 counter_ = counter;
        ull2* tmp      = reinterpret_cast<ull2*>(&counter_);
        tmp->y         = subsequence;

        uint2 key_ = seed;
// 7-round philox
#pragma unroll
        for(int i = 0; i < 6; i++)
        {
            counter_ = philox_single_round(counter_, key_);
            key_.x += kPhilox10A;
            key_.y += kPhilox10B;
        }
        uint4 output = philox_single_round(counter_, key_);
        return output;
    }

    CK_TILE_HOST_DEVICE void get_random_16x8(uint8_t* out,
                                             const unsigned long long subsequence) const
    {
        uint4 tmp_ph;
        tmp_ph = get_philox_4x32(subsequence);

        uint32_t* out_tmp = reinterpret_cast<uint32_t*>(&out[0]);

        out_tmp[0] = tmp_ph.x;
        out_tmp[1] = tmp_ph.y;
        out_tmp[2] = tmp_ph.z;
        out_tmp[3] = tmp_ph.w;
    }

    CK_TILE_HOST_DEVICE void get_random_8x8(uint8_t* out,
                                            const unsigned long long subsequence,
                                            const index_t start_idx) const
    {
        uint4 tmp_ph;
        tmp_ph = get_philox_4x32(subsequence);

        uint32x4_t tmp;
        tmp[0]            = tmp_ph.x;
        tmp[1]            = tmp_ph.y;
        tmp[2]            = tmp_ph.z;
        tmp[3]            = tmp_ph.w;
        uint32_t* out_tmp = reinterpret_cast<uint32_t*>(&out[0]);
        out_tmp[0]        = tmp[start_idx];
        out_tmp[1]        = tmp[start_idx + 2];
    }

    CK_TILE_HOST_DEVICE void get_random_4x8(uint8_t* out,
                                            const unsigned long long subsequence,
                                            const index_t start_idx) const
    {
        uint4 tmp_ph;
        tmp_ph = get_philox_4x32(subsequence);

        uint32x4_t tmp;
        tmp[0]            = tmp_ph.x;
        tmp[1]            = tmp_ph.y;
        tmp[2]            = tmp_ph.z;
        tmp[3]            = tmp_ph.w;
        uint32_t* out_tmp = reinterpret_cast<uint32_t*>(&out[0]);
        out_tmp[0]        = tmp[start_idx];
    }

    private:
    struct ull2
    {
        uint64_t x;
        uint64_t y;
    };
    uint4 counter;
    const uint2 seed;

    CK_TILE_HOST_DEVICE uint2 mulhilo32(const unsigned int a, const unsigned int b) const
    {
        uint2* res;
        unsigned long long tmp;
        tmp = static_cast<unsigned long long>(a) * b;
        res = reinterpret_cast<uint2*>(&tmp);
        return *res;
    }

    CK_TILE_HOST_DEVICE uint4 philox_single_round(const uint4 ctr, const uint2 key) const
    {

        uint2 res0 = mulhilo32(kPhiloxSA, ctr.x);
        uint2 res1 = mulhilo32(kPhiloxSB, ctr.z);
        uint4 ret  = {res1.y ^ ctr.y ^ key.x, res1.x, res0.y ^ ctr.w ^ key.y, res0.x};
        return ret;
    }

    static const unsigned long kPhilox10A = 0x9E3779B9;
    static const unsigned long kPhilox10B = 0xBB67AE85;
    static const unsigned long kPhiloxSA  = 0xD2511F53;
    static const unsigned long kPhiloxSB  = 0xCD9E8D57;
};

} // namespace ck_tile
