// SPDX-License-Identifier: MIT
// Copyright (c) 2025, , Inc. All rights reserved.

#pragma once

// #include <hip/hip_runtime_api.h>

#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/host/stream_config.hpp"
#include "ck_tile/host/hip_check_error.hpp"

namespace ck_tile {

static inline index_t get_available_compute_units(const stream_config& s)
{
    (void)s;
    constexpr static uint32_t MAX_MASK_DWORDS = 64;

    // assume at most 64*32 = 2048 CUs
    uint32_t cu_mask[MAX_MASK_DWORDS]{};

    auto count_set_bits = [](uint32_t dword) {
        index_t count = 0;
        while(dword != 0)
        {
            if(dword & 0x1)
            {
                count++;
            }
            dword = dword >> 1;
        }
        return count;
    };

    HIP_CHECK_ERROR(hipExtStreamGetCUMask(s.stream_id_, MAX_MASK_DWORDS, &cu_mask[0]));

    index_t num_cu = 0;
    for(uint32_t i = 0; i < MAX_MASK_DWORDS; i++)
    {
        num_cu += count_set_bits(cu_mask[i]);
    }

    return num_cu;
};

} // namespace ck_tile
