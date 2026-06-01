// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/container/thread_buffer.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/utility/functional.hpp"

namespace ck_tile {

struct WarpDsreadmAttributeImplF16MN32K16
{
    using RawType = fp16_t;
    using RetType = ext_vector_t<fp16_t, 8>;

    // dsreadm inst size
    static constexpr index_t kMN = 32;
    static constexpr index_t kK  = 16;

    // lane load layout
    static constexpr index_t kMNLoadLane = 4;
    static constexpr index_t kKLoadLane  = 16;

    // lane load slice
    static constexpr index_t kMNLoadPerLane = 8;
    static constexpr index_t kKLoadPerLane  = 1;

    // lane store layout
    static constexpr index_t kMNStoreLane = 16;
    static constexpr index_t kKStoreLane  = 4;

    // lane store slice
    static constexpr index_t kMN0StorePerLane = 2;
    static constexpr index_t kMN1StorePerLane = 1;
    static constexpr index_t kKStorePerLane   = 4;

    // for static check
    static constexpr index_t kMNInterleave = 1;

    CK_TILE_DEVICE void operator()(const RawType* lds_ptr, RetType& ret) const
    {
#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__) || defined(__gfx92a__) || \
    defined(__gfx946__)
        ret = __builtin_hcu_ds_read_m32x16_f16(lds_ptr);
#else
        detail::swallow{lds_ptr, ret};
#endif
    }
};

struct WarpDsreadmAttributeImplF16MN32K16Alt
{
    using RawType = fp16_t;
    using RetType = ext_vector_t<fp16_t, 8>;

    // dsreadm inst size
    static constexpr index_t kMN = 32;
    static constexpr index_t kK  = 16;

    // lane load layout
    static constexpr index_t kMNLoadLane = 4;
    static constexpr index_t kKLoadLane  = 16;

    // lane load slice
    static constexpr index_t kMNLoadPerLane = 8;
    static constexpr index_t kKLoadPerLane  = 1;

    // lane store layout
    static constexpr index_t kMNStoreLane = 16;
    static constexpr index_t kKStoreLane  = 4;

    // lane store slice
    static constexpr index_t kMN0StorePerLane = 1;
    static constexpr index_t kMN1StorePerLane = 2;
    static constexpr index_t kKStorePerLane   = 4;

    // for static check
    static constexpr index_t kMNInterleave = 2;

    CK_TILE_DEVICE void operator()(const RawType* lds_ptr, RetType& ret) const
    {
#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__) || defined(__gfx92a__) || \
    defined(__gfx946__)
        ret = __builtin_hcu_ds_read_m32x16_f16_alt(lds_ptr);
#else
        detail::swallow{lds_ptr, ret};
#endif
    }
};

// TODO: add more

} // namespace ck_tile
