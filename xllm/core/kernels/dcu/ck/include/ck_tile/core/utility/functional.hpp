// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include <stdint.h>
#include <utility>

namespace ck_tile {

namespace detail {

struct swallow
{
    template <typename... Ts>
    CK_TILE_HOST_DEVICE constexpr swallow(Ts&&...)
    {
    }
};

template <class>
struct static_for_impl;

template <index_t... Is>
struct static_for_impl<sequence<Is...>>
{
    template <class F>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f) const
    {
        swallow{(f(number<Is>{}), 0)...};
    }
};

} // namespace detail

// F signature: F(number<Iter>)
template <index_t NBegin, index_t NEnd, index_t Increment>
struct static_for
{
    CK_TILE_HOST_DEVICE constexpr static_for()
    {
        static_assert(Increment != 0 && (NEnd - NBegin) % Increment == 0,
                      "Wrong! should satisfy (NEnd - NBegin) % Increment == 0");
        static_assert((Increment > 0 && NBegin <= NEnd) || (Increment < 0 && NBegin >= NEnd),
                      "wrongs! should (Increment > 0 && NBegin <= NEnd) || (Increment < 0 && "
                      "NBegin >= NEnd)");
    }

    template <class F>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f) const
    {
        detail::static_for_impl<typename arithmetic_sequence_gen<NBegin, NEnd, Increment>::type>{}(
            f);
    }
};

namespace detail {

template <typename T, T... Is>
struct applier
{
    template <typename F>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f) const
    {
        // tweak -fbracket-depth if compilation fails. Clang default limit is 256
        (f(number<Is>{}), ...);
    }
};

template <int32_t Size> // == sizeof...(Is)
using make_applier = __make_integer_seq<applier, index_t, Size>;

} // namespace detail

template <index_t N>
struct static_for<0, N, 1> : detail::make_applier<N>
{
    using detail::make_applier<N>::operator();
};

struct identity
{
    template <typename T>
    CK_TILE_HOST_DEVICE constexpr T&& operator()(T&& arg) const noexcept
    {
        return std::forward<T>(arg);
    }
};

namespace detail {

// RemainLengths: sequence<...>
// Orders: sequence<...>
template <class RemainLengths, class Orders>
struct static_ford_impl
{
    CK_TILE_HOST_DEVICE constexpr static_ford_impl()
    {
        static_assert(RemainLengths::size() > 0, "wrong! should not get here");
    }

    // F signature: F(sequence<...>)
    // CurrentOrderedId: sequence<...>
    template <class F, class CurrentOrderedId>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f, CurrentOrderedId) const
    {
        static_for<0, RemainLengths::front(), 1>{}([=](auto I) {
            static_ford_impl<decltype(RemainLengths::pop_front()), Orders>{}(
                f, CurrentOrderedId::push_back(I));
        });
    }
};

template <class Orders>
struct static_ford_impl<sequence<>, Orders>
{
    // F signature: F(sequence<...>)
    // OrderedId: sequence<...>
    template <class F, class OrderedId>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f, OrderedId) const
    {
        // retrive unordered Id
        f(OrderedId::reorder_old_to_new(Orders{}));
    }
};

} // namespace detail

// Lengths is sequence<...>, it is the length of each dimension for
// N-dimensional loop
// Orders is sequence<...>, it is the order of dimension in which static_ford
// will loop over each
// dimension
template <class Lengths,
          class Orders = typename arithmetic_sequence_gen<0, Lengths::size(), 1>::type>
struct static_ford
{
    CK_TILE_HOST_DEVICE constexpr static_ford()
    {
        static_assert(Lengths::size() > 0, "wrong! Lengths is empty");
        static_assert(Lengths::size() == Orders::size(), "wrong! inconsistent size");
    }

    // F signature: F(sequence<...> multi_id)
    // multi_id is the unordered multi-index
    template <class F>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f) const
    {
        constexpr auto ordered_lengths = Lengths::reorder_new_to_old(Orders{});
        detail::static_ford_impl<decltype(ordered_lengths), Orders>{}(f, sequence<>{});
    }
};

namespace detail {

template <typename Indices>
struct unpack_impl;

template <index_t... Is>
struct unpack_impl<sequence<Is...>>
{
    template <typename F, typename X>
    CK_TILE_HOST_DEVICE constexpr auto operator()(F&& f, X&& x) const
    {
#if 0
        return std::forward<F>(f)(std::forward<X>(x).at(number<Is>{})...);
#else
        return std::forward<F>(f)(std::forward<X>(x).template at<Is>()...);
#endif
    }
};

template <typename Seq0, typename Seq1>
struct unpack2_impl;

// TODO: remove this, after properly implementing unpack that takes any number of containers
template <index_t... Is, index_t... Js>
struct unpack2_impl<sequence<Is...>, sequence<Js...>>
{
    template <typename F, typename X, typename Y>
    CK_TILE_HOST_DEVICE constexpr auto operator()(F&& f, X&& x, Y&& y) const
    {
#if 0
        return std::forward<F>(f)(std::forward<X>(x).at(number<Is>{})...,
                                  std::forward<Y>(y).at(number<Js>{})...);
#else
        return std::forward<F>(f)(std::forward<X>(x).template at<Is>()...,
                                  std::forward<Y>(y).template at<Js>()...);
#endif
    }
};

} // namespace detail

template <typename F, typename X>
CK_TILE_HOST_DEVICE constexpr auto unpack(F&& f, X&& x)
{
    using X_ = remove_reference_t<X>;
    return detail::unpack_impl<typename arithmetic_sequence_gen<0, X_::size(), 1>::type>{}(
        std::forward<F>(f), std::forward<X>(x));
}

// TODO: properly implement unpack that takes any number of containers
template <typename F, typename X, typename Y>
CK_TILE_HOST_DEVICE constexpr auto unpack2(F&& f, X&& x, Y&& y)
{
    using X_ = remove_reference_t<X>;
    using Y_ = remove_reference_t<Y>;
    return detail::unpack2_impl<typename arithmetic_sequence_gen<0, X_::size(), 1>::type,
                                typename arithmetic_sequence_gen<0, Y_::size(), 1>::type>{}(
        std::forward<F>(f), std::forward<X>(x), std::forward<Y>(y));
}

// z = predicate ? x : y
template <bool predicate, typename X, typename Y>
constexpr auto conditional_expr(X&& x, Y&& y)
{
    if constexpr(predicate)
    {
        return std::forward<X>(x);
    }
    else
    {
        return std::forward<Y>(y);
    }
}

} // namespace ck_tile
