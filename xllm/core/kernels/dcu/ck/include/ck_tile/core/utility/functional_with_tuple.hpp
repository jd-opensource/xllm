// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

// This file should not be included inside tuple.hpp!

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include <stdint.h>
#include <utility>

namespace ck_tile {

namespace detail {

// RemainLengths: sequence<...>
// Orders: sequence<...>
template <class RemainLengths, class RamainUnpacks, class Orders>
struct static_uford_impl
{
    CK_TILE_HOST_DEVICE constexpr static_uford_impl()
    {
        static_assert(RemainLengths::size() > 0, "wrong! should not get here");
        static_assert(RamainUnpacks::size() > 0, "wrong! should not get here");
    }

    template <class F, class CurrentUnpackIds>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f, CurrentUnpackIds) const
    {
        constexpr index_t pack_len = RamainUnpacks::front();
        static_for<0, RemainLengths::front(), pack_len>{}([=](auto I) {
            constexpr auto new_pack = generate_tuple(
                [&](auto idx_) {
                    constexpr auto i_new_pack = number<I + idx_ % pack_len>{};
                    constexpr auto i_pre_pack = number<idx_ / pack_len>{};
                    return CurrentUnpackIds{}.at(i_pre_pack).push_back(i_new_pack);
                },
                number<CurrentUnpackIds::size() * pack_len>{});

            static_uford_impl<decltype(RemainLengths::pop_front()),
                              decltype(RamainUnpacks::pop_front()),
                              Orders>{}(f, new_pack);
        });
    }
};

template <class Orders>
struct static_uford_impl<sequence<>, sequence<>, Orders>
{
    template <class F, class PackedId>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f, PackedId) const
    {
        constexpr auto origin_packs = transform_tuples(
            [](auto pack_) { return decltype(pack_)::reorder_old_to_new(Orders{}); }, PackedId{});
        unpack(f, origin_packs);
    }
};

template <class RemainLengths, class RamainUnpacks, class Orders>
struct static_uford_one_shot_impl
{
    template <class F, class CurrentUnpackIds, index_t current_acc>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f, CurrentUnpackIds, number<current_acc>) const
    {
        constexpr auto r_lens_stride =
            reverse_exclusive_scan_sequence(RemainLengths{}, multiplies{}, number<1>{});
        constexpr auto r_upks_stride =
            reverse_exclusive_scan_sequence(RamainUnpacks{}, multiplies{}, number<1>{});

        constexpr index_t current_stride = r_lens_stride.front() / r_upks_stride.front();
        constexpr index_t pack_len       = RamainUnpacks::front();
        constexpr index_t current_idx    = (current_acc / current_stride) * pack_len;

        constexpr auto new_pack = generate_tuple(
            [&](auto idx_) {
                constexpr auto i_new_pack = number<current_idx + idx_ % pack_len>{};
                constexpr auto i_pre_pack = number<idx_ / pack_len>{};
                return CurrentUnpackIds{}.at(i_pre_pack).push_back(i_new_pack);
            },
            number<CurrentUnpackIds::size() * pack_len>{});

        static_uford_one_shot_impl<decltype(RemainLengths::pop_front()),
                                   decltype(RamainUnpacks::pop_front()),
                                   Orders>{}(f, new_pack, number<current_acc % current_stride>{});
    }
};

template <class Orders>
struct static_uford_one_shot_impl<sequence<>, sequence<>, Orders>
{
    template <class F, class PackedId, index_t current_acc>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f, PackedId, number<current_acc>) const
    {
        constexpr auto origin_packs = transform_tuples(
            [](auto pack_) { return decltype(pack_)::reorder_old_to_new(Orders{}); }, PackedId{});
        unpack(f, origin_packs);
    }
};

} // namespace detail

// TODO: we may unify static_ford/static_uford in the future
//
// loop over nd space(sequence) with packs
// you must make sure the function passed in has same number of argument
//
// e.g.
// Lengths=seq<2, 3, 4>, Unpacks=<1, 1, 2>
// static_uford<Lengths, Unpacks>{}([&](auto i_0, auto i_1){}); // require 2 args(packs)
//
// loop #0, i_0=seq<0, 0, 0>, i_1=<0, 0, 1>
// loop #1, i_0=seq<0, 0, 2>, i_1=<0, 0, 3>
// loop #2, i_0=seq<0, 1, 0>, i_1=<0, 1, 1>
// loop #3, i_0=seq<0, 1, 2>, i_1=<0, 1, 3>
// loop #4, i_0=seq<0, 2, 0>, i_1=<0, 2, 1>
// loop #5, i_0=seq<0, 2, 2>, i_1=<0, 2, 3>
// loop #6, i_0=seq<1, 0, 0>, i_1=<1, 0, 1>
// ...
template <class Lengths,
          class Unpacks = typename uniform_sequence_gen<Lengths::size(), 1>::type,
          class Orders  = typename arithmetic_sequence_gen<0, Lengths::size(), 1>::type>
struct static_uford
{
    static constexpr index_t num_packs = reduce_on_sequence(Unpacks{}, multiplies{}, number<1>{});

    CK_TILE_HOST_DEVICE constexpr static_uford()
    {
        static_assert(Lengths::size() > 0, "wrong! Lengths is empty");
        static_assert(Lengths::size() == Unpacks::size(), "wrong! inconsistent size");
        static_assert(Lengths::size() == Orders::size(), "wrong! inconsistent size");
        static_for<0, Lengths::size(), 1>{}(
            [&](auto i) { static_assert(Lengths{}.at(i) % Unpacks{}.at(i) == 0); });
    }

    CK_TILE_HOST_DEVICE static constexpr index_t get_num_of_access()
    {
        using L_ = decltype(Lengths{} / Unpacks{});

        return reduce_on_sequence(L_{}, multiplies{}, number<1>{});
    }

    // F signature: F(sequence<...> multi_id...)
    // multi_id is the unordered multi-index
    template <class F>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f) const
    {
        constexpr auto ordered_lengths = Lengths::reorder_new_to_old(Orders{});
        constexpr auto ordered_unpacks = Unpacks::reorder_new_to_old(Orders{});
        detail::static_uford_impl<decltype(ordered_lengths), decltype(ordered_unpacks), Orders>{}(
            f, make_tuple(sequence<>{}));
    }

    // this version is friendly for issue function one by one
    template <class F, index_t i_access>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f, number<i_access>) const
    {
        static_assert(i_access < get_num_of_access());
        constexpr auto ordered_lengths = Lengths::reorder_new_to_old(Orders{});
        constexpr auto ordered_unpacks = Unpacks::reorder_new_to_old(Orders{});
        detail::static_uford_one_shot_impl<decltype(ordered_lengths),
                                           decltype(ordered_unpacks),
                                           Orders>{}(
            f, make_tuple(sequence<>{}), number<i_access>{});
    }
};

} // namespace ck_tile
