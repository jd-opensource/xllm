// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/functional_with_tuple.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

// sweep over a span of a distribted tile and apply lambda function F
template <typename TileDistributedSpan_, // tile_distributed_span<...>
          typename F                     // signature: F(tile_distributed_index<...>)
          >
CK_TILE_DEVICE void sweep_tile_span(TileDistributedSpan_, const F& f)
{
    using DstrSpan = remove_cvref_t<TileDistributedSpan_>;

    static_ford<typename DstrSpan::Impl>{}([&](auto dstr_idx_impl) {
        constexpr auto dstr_idx = detail::make_tile_distributed_index(dstr_idx_impl);

        f(dstr_idx);
    });
}

// unpacked span, this version support span with unpack(multi-arg) functor
//
template <
    typename TileDistributedSpan_, // tile_distributed_span<...>
    typename F,                    // signature: F(tile_distributed_index<...>)
    typename Unpacks = typename uniform_sequence_gen<TileDistributedSpan_::Impl::size(), 1>::type>
CK_TILE_DEVICE void sweep_tile_uspan(TileDistributedSpan_, const F& f, Unpacks = {})
{
    using DstrSpan = remove_cvref_t<TileDistributedSpan_>;

    static_uford<typename DstrSpan::Impl, Unpacks>{}(
        [&](auto... dstr_idx_impl) { f(detail::make_tile_distributed_index(dstr_idx_impl)...); });
}

namespace impl {

template <typename, typename, typename>
struct sweep_tile_impl;

template <typename DistributedTensor, typename UnpacksPerXDim, index_t I, index_t... Is>
struct sweep_tile_impl<DistributedTensor, UnpacksPerXDim, sequence<I, Is...>>
{
    CK_TILE_HOST_DEVICE constexpr auto get_y_unpacks() const
    {
        constexpr auto spans     = DistributedTensor::get_distributed_spans();
        constexpr auto y_lengths = typename decltype(spans[number<I>{}])::Impl{};
        constexpr auto x_unpacks = number<UnpacksPerXDim{}.at(number<I>{})>{};
        constexpr auto y_unpacks = get_y_unpacks_from_x_unpacks(y_lengths, x_unpacks);
        return y_unpacks;
    }
    CK_TILE_HOST_DEVICE constexpr index_t get_num_of_access() const
    {
        constexpr auto spans = DistributedTensor::get_distributed_spans();
        constexpr auto u =
            static_uford<typename decltype(spans[number<I>{}])::Impl, decltype(get_y_unpacks())>{};
        return u.get_num_of_access() *
               sweep_tile_impl<DistributedTensor, UnpacksPerXDim, sequence<Is...>>{}
                   .get_num_of_access();
    }
    template <typename F, typename SpanIdx>
    CK_TILE_HOST_DEVICE constexpr void operator()(const F& f, const SpanIdx& span_idx) const
    {
        constexpr auto spans = DistributedTensor::get_distributed_spans();

        sweep_tile_uspan(
            spans[number<I>{}],
            [&](auto... i_idx) {
                const auto next_span_idx = embed_tuples(
                    [&](auto si) { return make_tuple(concat_tuple(si, make_tuple(i_idx))...); },
                    span_idx);
                sweep_tile_impl<DistributedTensor, UnpacksPerXDim, sequence<Is...>>{}(
                    f, next_span_idx);
            },
            get_y_unpacks());
    }
    template <typename F, typename SpanIdx, index_t i_access>
    CK_TILE_HOST_DEVICE constexpr void
    operator()(const F& f, const SpanIdx& span_idx, number<i_access>) const
    {
        constexpr auto spans = DistributedTensor::get_distributed_spans();
        constexpr auto u =
            static_uford<typename decltype(spans[number<I>{}])::Impl, decltype(get_y_unpacks())>{};
        constexpr auto access_stride =
            sweep_tile_impl<DistributedTensor, UnpacksPerXDim, sequence<Is...>>{}
                .get_num_of_access();
        constexpr auto curr_i_access = number<i_access / access_stride>{};
        constexpr auto next_i_access = number<i_access % access_stride>{};
        u(
            [&](auto... i_idx) {
                const auto next_span_idx = embed_tuples(
                    [&](auto si) {
                        return make_tuple(concat_tuple(
                            si, make_tuple(detail::make_tile_distributed_index(i_idx)))...);
                    },
                    span_idx);
                sweep_tile_impl<DistributedTensor, UnpacksPerXDim, sequence<Is...>>{}(
                    f, next_span_idx, next_i_access);
            },
            curr_i_access);
    }
};

template <typename DistributedTensor, typename UnpacksPerXDim>
struct sweep_tile_impl<DistributedTensor, UnpacksPerXDim, sequence<>>
{
    CK_TILE_HOST_DEVICE constexpr index_t get_num_of_access() const { return 1; }
    template <typename F, typename SpanIdx>
    CK_TILE_HOST_DEVICE constexpr void operator()(const F& f, const SpanIdx& span_idx) const
    {
        unpack(f, span_idx);
    }
    template <typename F, typename SpanIdx, index_t i_access>
    CK_TILE_HOST_DEVICE constexpr void
    operator()(const F& f, const SpanIdx& span_idx, number<i_access>) const
    {
        unpack(f, span_idx);
    }
};

template <typename, typename, typename>
struct sweep_tile_impl_0;

// TODO: support empty tuple to remove this "entry-point" like function
template <typename DistributedTensor, typename UnpacksPerXDim, index_t I, index_t... Is>
struct sweep_tile_impl_0<DistributedTensor, UnpacksPerXDim, sequence<I, Is...>>
{
    CK_TILE_HOST_DEVICE constexpr auto get_y_unpacks() const
    {
        constexpr auto spans     = DistributedTensor::get_distributed_spans();
        constexpr auto y_lengths = typename decltype(spans[number<I>{}])::Impl{};
        constexpr auto x_unpacks = number<UnpacksPerXDim{}.at(number<I>{})>{};
        constexpr auto y_unpacks = get_y_unpacks_from_x_unpacks(y_lengths, x_unpacks);
        return y_unpacks;
    }
    CK_TILE_HOST_DEVICE constexpr index_t get_num_of_access() const
    {
        constexpr auto spans = DistributedTensor::get_distributed_spans();
        constexpr auto u =
            static_uford<typename decltype(spans[number<I>{}])::Impl, decltype(get_y_unpacks())>{};
        return u.get_num_of_access() *
               sweep_tile_impl<DistributedTensor, UnpacksPerXDim, sequence<Is...>>{}
                   .get_num_of_access();
    }
    template <typename F>
    CK_TILE_HOST_DEVICE constexpr void operator()(const F& f) const
    {
        constexpr auto spans = DistributedTensor::get_distributed_spans();
        sweep_tile_uspan(
            spans[number<I>{}],
            [&](auto... i_idx) {
                constexpr auto next_span_idx = make_tuple(make_tuple(i_idx)...);
                sweep_tile_impl<DistributedTensor, UnpacksPerXDim, sequence<Is...>>{}(
                    f, next_span_idx);
            },
            get_y_unpacks());
    }
    template <typename F, index_t i_access>
    CK_TILE_HOST_DEVICE constexpr void operator()(const F& f, number<i_access>) const
    {
        constexpr auto spans = DistributedTensor::get_distributed_spans();
        constexpr auto u =
            static_uford<typename decltype(spans[number<I>{}])::Impl, decltype(get_y_unpacks())>{};
        constexpr auto access_stride =
            sweep_tile_impl<DistributedTensor, UnpacksPerXDim, sequence<Is...>>{}
                .get_num_of_access();
        constexpr auto curr_i_access = number<i_access / access_stride>{};
        constexpr auto next_i_access = number<i_access % access_stride>{};
        u(
            [&](auto... i_idx) {
                constexpr auto next_span_idx =
                    make_tuple(make_tuple(detail::make_tile_distributed_index(i_idx))...);
                sweep_tile_impl<DistributedTensor, UnpacksPerXDim, sequence<Is...>>{}(
                    f, next_span_idx, next_i_access);
            },
            curr_i_access);
    }
};

} // namespace impl

/*
 * Enhanced sweep-tile utility, can control unpacks along each X-dim
 * the lambda function argument is the distributed-idx, which can directly
 * plugged into the distributed tensor as setter/getter
 *
 * e.g. below function, y with the type DistributedTensor, r is row scale
 *
 * // sweep tile 1 by 1
 * sweep_tile<DistributedTensor>([&](auto idx) {
 *     constexpr auto row_id = make_tuple(idx[number<0>{}]);
 *     y(idx)                = y(idx) * r(row_id);
 * });
 *
 * // sweep tile with 2 pixel from last dim each function call
 * sweep_tile<DistributedTensor>(
 *     [&](auto idx_0, auto idx_1) {
 *         constexpr auto row_id = make_tuple(idx_0[number<0>{}]);
 *         y(idx_0)              = y(idx_0) * r(row_id);
 *         y(idx_1)              = y(idx_1) * r(row_id);
 *     },
 *     sequence<1, 2>{});
 *
 * // sweep tile with 2x2 pixel each function call
 * sweep_tile<DistributedTensor>(
 *     [&](auto idx_00, auto idx_01, auto idx_10, auto idx_11) {
 *         constexpr auto row_id0 = make_tuple(idx_00[number<0>{}]);
 *         constexpr auto row_id1 = make_tuple(idx_10[number<0>{}]);
 *         y(idx_00)              = y(idx_00) * r(row_id0);
 *         y(idx_01)              = y(idx_01) * r(row_id0);
 *         y(idx_10)              = y(idx_10) * r(row_id1);
 *         y(idx_11)              = y(idx_11) * r(row_id1);
 *     },
 *     sequence<2, 2>{});
 *
 * TODO: do we need constexpr? lambda function could be non-constexpr
 */
template <typename DistributedTensor,
          typename F,
          typename UnpacksPerXDim =
              typename uniform_sequence_gen<DistributedTensor::get_num_of_dimension(), 1>::type>
CK_TILE_HOST_DEVICE constexpr void sweep_tile(const F& f, UnpacksPerXDim = {})
{
    constexpr auto spans = DistributedTensor::get_distributed_spans();

    impl::sweep_tile_impl_0<DistributedTensor,
                            UnpacksPerXDim,
                            typename arithmetic_sequence_gen<0, spans.size(), 1>::type>{}(f);
}

template <typename DistributedTensor,
          typename F,
          typename UnpacksPerXDim =
              typename uniform_sequence_gen<DistributedTensor::get_num_of_dimension(), 1>::type>
CK_TILE_HOST_DEVICE constexpr void
sweep_tile(const DistributedTensor&, const F& f, UnpacksPerXDim = {})
{
    sweep_tile<DistributedTensor, F, UnpacksPerXDim>(f, UnpacksPerXDim{});
}

/*
 * construct a sweep tile instance, which support issue the lambda one by one
 * Note that this struct will hold the lambda functor, but will not hold the distributed tensor
 * the functionality is the same as sweep_tile()
 */
template <typename DistributedTensor_,
          typename F_,
          typename UnpacksPerXDim_ =
              typename uniform_sequence_gen<DistributedTensor_::get_num_of_dimension(), 1>::type>
struct tile_sweeper
{
    using DistributedTensor = remove_cvref_t<DistributedTensor_>;
    using F                 = remove_cvref_t<F_>;
    using UnpacksPerXDim    = remove_cvref_t<UnpacksPerXDim_>;

    CK_TILE_HOST_DEVICE tile_sweeper(const F& f_, UnpacksPerXDim = {}) : f(f_) {}
    CK_TILE_HOST_DEVICE tile_sweeper(const DistributedTensor&, const F& f_, UnpacksPerXDim = {})
        : f(f_)
    {
    }
    CK_TILE_HOST_DEVICE static constexpr index_t get_num_of_access()
    {
        constexpr auto spans = DistributedTensor::get_distributed_spans();
        constexpr auto tmp =
            impl::sweep_tile_impl_0<DistributedTensor,
                                    UnpacksPerXDim,
                                    typename arithmetic_sequence_gen<0, spans.size(), 1>::type>{};
        return tmp.get_num_of_access();
    }

    CK_TILE_HOST_DEVICE void operator()() const
    {
        sweep_tile<DistributedTensor>(f, UnpacksPerXDim{});
    }

    template <index_t i_access>
    CK_TILE_HOST_DEVICE void operator()(number<i_access>) const
    {
        constexpr auto spans = DistributedTensor::get_distributed_spans();

        impl::sweep_tile_impl_0<DistributedTensor,
                                UnpacksPerXDim,
                                typename arithmetic_sequence_gen<0, spans.size(), 1>::type>{}(
            f, number<i_access>{});
    }
    F f;
};

// partial deduction is not allowed
// template <typename T, typename F, typename U>
// CK_TILE_HOST_DEVICE_EXTERN tile_sweeper(const F&, U = {})->tile_sweeper<T, F, U>;

// deduction guide
template <typename T,
          typename F,
          typename U = typename uniform_sequence_gen<T::get_num_of_dimension(), 1>::type>
CK_TILE_HOST_DEVICE_EXTERN tile_sweeper(const T&, const F&, U = {})->tile_sweeper<T, F, U>;

} // namespace ck_tile
