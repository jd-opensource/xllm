// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/hcu_ebarrier.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/vector_type.hpp"
#include "ck_tile/core/utility/functional.hpp"

namespace ck_tile {

namespace detail {

struct abarrier_slot
{
    union
    {
        int32x2_t dwords;
        struct
        {
            uint64_t transaction_count : 19;
            uint64_t pending_arrival_count : 11;
            uint64_t exptedted_arrival_count : 11;
            uint64_t phase : 1;
            uint64_t valid : 1;
            uint64_t unused : 21;
        };
    };
};

CK_TILE_DEVICE abarrier_slot make_abarrier_slot(int32x2_t sdst) { return abarrier_slot{sdst}; }

} // namespace detail

struct hcu_abarrier
{
    CK_TILE_DEVICE hcu_abarrier() : bar_id_(0), wv_cnt_(0) {}

    CK_TILE_DEVICE hcu_abarrier(const index_t bar_id, const index_t wv_cnt)
        : bar_id_(bar_id), wv_cnt_(wv_cnt)
    {
    }

    CK_TILE_DEVICE hcu_abarrier& operator=(const hcu_abarrier& abar)
    {
        bar_id_ = abar.bar_id_;
        wv_cnt_ = abar.wv_cnt_;
        return *this;
    }

    CK_TILE_DEVICE void init() const
    {
#ifdef CK_TILE_DEVICE_ABARRIER_SUPPORT
        asm volatile("s_abarrier_init %0, %1" ::"s"(bar_id_), "s"(wv_cnt_));
#endif
    }

    CK_TILE_DEVICE void init(const hcu_ebarrier& ebar) const
    {
        init();
        ebar.sync();
    }

    CK_TILE_DEVICE void init(const index_t ebar_id, const index_t ebar_wv_cnt) const
    {
        hcu_ebarrier ebar(ebar_id, ebar_wv_cnt);
        init(ebar);
    }

    CK_TILE_DEVICE void inv() const
    {
#ifdef CK_TILE_DEVICE_ABARRIER_SUPPORT
        asm volatile("s_abarrier_inv %0" ::"s"(bar_id_));
#endif
    }

    CK_TILE_DEVICE void track() const
    {
#ifdef CK_TILE_DEVICE_ABARRIER_SUPPORT
        asm volatile("s_abarrier_seq %0" ::"s"(bar_id_));
#endif
    }

    CK_TILE_DEVICE auto slot_rd() const
    {
        int32x2_t abar_obj = {0, 0};
#ifdef CK_TILE_DEVICE_ABARRIER_SUPPORT
        asm volatile("s_abarrier_slot_rd %0, %1" : "=s"(abar_obj) : "s"(bar_id_));
#endif
    }

    CK_TILE_DEVICE auto arrive() const
    {
        index_t state = 0;
#ifdef CK_TILE_DEVICE_ABARRIER_SUPPORT
        asm volatile("s_abarrier_arrive %0, %1, %2" : "=s"(state) : "s"(bar_id_), "s"(1));
#endif
        return state;
    }

    CK_TILE_DEVICE auto arrive_nocomplete() const
    {
        index_t state = 0;
#ifdef CK_TILE_DEVICE_ABARRIER_SUPPORT
        asm volatile("s_abarrier_arrive_nocomplete %0, %1, %2"
                     : "=s"(state)
                     : "s"(bar_id_), "s"(1));
#endif
        return state;
    }

    CK_TILE_DEVICE auto arrive_drop() const
    {
        index_t state = 0;
#ifdef CK_TILE_DEVICE_ABARRIER_SUPPORT
        asm volatile("s_abarrier_arrive_drop %0, %1, %2" : "=s"(state) : "s"(bar_id_), "s"(1));
#endif
        return state;
    }

    CK_TILE_DEVICE auto arrive_drop_nocomplete() const
    {
        index_t state = 0;
#ifdef CK_TILE_DEVICE_ABARRIER_SUPPORT
        asm volatile("s_abarrier_arrive_drop_nocomplete %0, %1, %2"
                     : "=s"(state)
                     : "s"(bar_id_), "s"(1));
#endif
        return state;
    }

    CK_TILE_DEVICE auto try_wait(const index_t state) const
    {
        index_t completion = 0;
#ifdef CK_TILE_DEVICE_ABARRIER_SUPPORT
        asm volatile("s_abarrier_try_wait %0, %1, %2"
                     : "=s"(completion)
                     : "s"(bar_id_), "s"(state));
#else
        asm volatile("s_waitcnt vmcnt(0)\n\t"
                     "s_barrier");
        completion = 1;
#endif
        return completion;
    }

    CK_TILE_DEVICE auto test_wait(const index_t state) const
    {
        index_t completion = 0;
#ifdef CK_TILE_DEVICE_ABARRIER_SUPPORT
        asm volatile("s_abarrier_test_wait %0, %1, %2"
                     : "=s"(completion)
                     : "s"(bar_id_), "s"(state));
#else
        asm volatile("s_waitcnt vmcnt(0)\n\t"
                     "s_barrier");
        completion = 1;
#endif
        return completion;
    }

    CK_TILE_DEVICE void expect_tx(const index_t txcnt) const
    {
#ifdef CK_TILE_DEVICE_ABARRIER_SUPPORT
        asm volatile("s_abarrier_expect_tx %0, %1" ::"s"(bar_id_), "s"(txcnt));
#else
        detail::swallow{txcnt};
#endif
    }

    CK_TILE_DEVICE void complete_tx(const index_t txcnt) const
    {
#ifdef CK_TILE_DEVICE_ABARRIER_SUPPORT
        asm volatile("s_abarrier_complete_tx %0, %1" ::"s"(bar_id_), "s"(txcnt));
#else
        detail::swallow{txcnt};
#endif
    }

    index_t bar_id_;
    index_t wv_cnt_;
};

} // namespace ck_tile
