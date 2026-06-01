// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/utility/functional.hpp"

namespace ck_tile {

namespace detail {

struct ebarrier_slot
{
    union
    {
        uint32_t dword;
        struct
        {
            uint32_t red_data : 10;
            uint32_t red_op : 3;
            uint32_t pending_arrival_count : 4;
            uint32_t expected_arrival_count : 5;
            uint32_t valid : 1;
            uint32_t unused : 9;
        };
    };
};

CK_TILE_DEVICE ebarrier_slot make_ebarrier_slot(uint32_t data) { return ebarrier_slot{data}; }

} // namespace detail

struct hcu_ebarrier
{
    CK_TILE_DEVICE hcu_ebarrier() : bar_id_(0), wv_cnt_(0) {}

    CK_TILE_DEVICE hcu_ebarrier(const index_t bar_id, const index_t wv_cnt)
        : bar_id_(bar_id), wv_cnt_(wv_cnt)
    {
    }

    CK_TILE_DEVICE hcu_ebarrier& operator=(const hcu_ebarrier& ebar)
    {
        bar_id_ = ebar.bar_id_;
        wv_cnt_ = ebar.wv_cnt_;
        return *this;
    }

    CK_TILE_DEVICE void arrive() const
    {
#if CK_TILE_DEVICE_EBARRIER_SUPPORT
        asm volatile("s_ebarrier_arrive %0, %1" ::"s"(bar_id_), "s"(wv_cnt_));
#else
        asm volatile("s_barrier");
#endif
    }

    CK_TILE_DEVICE void sync() const
    {
#if CK_TILE_DEVICE_EBARRIER_SUPPORT
        asm volatile("s_ebarrier_sync %0, %1" ::"s"(bar_id_), "s"(wv_cnt_));
#else
        asm volatile("s_barrier");
#endif
    }

    CK_TILE_DEVICE auto slot_rd() const
    {
        uint32_t slot = 0;
#if CK_TILE_DEVICE_EBARRIER_SUPPORT
        asm volatile("s_ebarrier_slot_rd %0, %1" : "=s"(slot) : "s"(bar_id_));
#endif
        return slot;
    }

    CK_TILE_DEVICE auto reduce_and(const index_t src) const
    {
        index_t result = 0;
#if CK_TILE_DEVICE_EBARRIER_REDUCE_OP_SUPPORT
        asm volatile("s_ebarrier_and %0, %1, %2" : "=s"(result) : "s"(bar_id_), "s"(src));
#else
        detail::swallow{src};
#endif
        return result;
    }

    CK_TILE_DEVICE auto reduce_or(const index_t src) const
    {
        index_t result = 0;
#if CK_TILE_DEVICE_EBARRIER_REDUCE_OP_SUPPORT
        asm volatile("s_ebarrier_or %0, %1, %2" : "=s"(result) : "s"(bar_id_), "s"(src));
#else
        detail::swallow{src};
#endif
        return result;
    }

    CK_TILE_DEVICE auto reduce_popc(const index_t src) const
    {
        index_t result = 0;
#if CK_TILE_DEVICE_EBARRIER_REDUCE_OP_SUPPORT
        asm volatile("s_ebarrier_popc %0, %1, %2" : "=s"(result) : "s"(bar_id_), "s"(src));
#else
        detail::swallow{src};
#endif
        return result;
    }

    index_t bar_id_;
    index_t wv_cnt_;
};

} // namespace ck_tile
