// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"

namespace ck_tile {

template <typename Context, index_t Start = 0, index_t Step = 1>
struct static_counter
{
    public:
    template <typename Unique>
    static constexpr index_t next()
    {
        return next<Unique>(0) * Step + Start;
    }

    template <unsigned long long>
    static constexpr index_t next()
    {
        struct Unique
        {
        };
        return next<Unique>(0) * Step + Start;
    }

    template <typename Unique>
    static constexpr index_t current()
    {
        return current<Unique>(0) * Step + Start;
    }

    template <unsigned long long>
    static constexpr index_t current()
    {
        struct Unique
        {
        };
        return current<Unique>(0) * Step + Start;
    }

    private:
    template <index_t I>
    struct slot
    {
        _Pragma("GCC diagnostic push");
        _Pragma("GCC diagnostic ignored \"-Wundefined-internal\"");
        friend constexpr bool slot_allocated(slot<I>);
        _Pragma("GCC diagnostic pop");
    };

    template <index_t I>
    struct allocate_slot
    {
        friend constexpr bool slot_allocated(slot<I>) { return true; }
        enum
        {
            value = I
        };
    };

    // If slot_allocated(slot<I>) has NOT been defined, then SFINAE will keep this function out of
    // the overload set...
    template <typename Unique, index_t I = 0, bool = slot_allocated(slot<I>())>
    static constexpr index_t next(index_t)
    {
        return next<Unique, I + 1>(0);
    }

    // ...And this function will be used, instead, which will define slot_allocated(slot<I>) via
    // allocate_slot<I>.
    template <typename Unique, index_t I = 0>
    static constexpr index_t next(double)
    {
        return allocate_slot<I>::value;
    }

    // If slot_allocated(slot<I>) has NOT been defined, then SFINAE will keep this function out of
    // the overload set...
    template <typename Unique, index_t I = Start, bool = slot_allocated(slot<I>())>
    static constexpr index_t current(index_t)
    {
        return current<Unique, I + 1>(0);
    }

    // ...And this function will be used, instead, which will return the current counter, or assert
    // in case next() hasn't been called yet.
    template <typename Unique, index_t I = Start>
    static constexpr index_t current(double)
    {
        static_assert(I != 0, "You must invoke next() first");

        return I - 1;
    }
};

namespace impl {
template <int I>
struct static_counter_uniq_;
}

#define MAKE_SC() \
    ck_tile::static_counter<ck_tile::impl::static_counter_uniq_<__COUNTER__>> {}
#define MAKE_SC_WITH(start_, step_) \
    ck_tile::static_counter<ck_tile::impl::static_counter_uniq_<__COUNTER__>, start_, step_> {}
#define NEXT_SC(c_) c_.next<__COUNTER__>()
#define NEXT_SCI(c_, static_i_) c_.next<__COUNTER__ + static_i_>()

// Usage:
// constexpr auto c = MAKE_SC()
// NEXT_SC(c)    // -> constexpr 0
// NEXT_SC(c)    // -> constexpr 1
// NEXT_SC(c)    // -> constexpr 2
} // namespace ck_tile
