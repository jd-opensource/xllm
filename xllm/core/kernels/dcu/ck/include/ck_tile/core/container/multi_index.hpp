// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/utility/functional.hpp"

namespace ck_tile {

// Don't use tihs directly. This is for old CK's internal usage,
// in the future always use array instead
template <index_t N>
using multi_index = array<index_t, N>;

template <typename... Xs>
CK_TILE_HOST_DEVICE constexpr auto make_multi_index(Xs&&... xs)
{
    return make_array<index_t>(index_t{xs}...);
}

template <index_t NSize>
CK_TILE_HOST_DEVICE constexpr auto make_zero_multi_index()
{
    return unpack([](auto... xs) { return make_multi_index(xs...); },
                  typename uniform_sequence_gen<NSize, 0>::type{});
}

template <typename T>
CK_TILE_HOST_DEVICE constexpr auto to_multi_index(const T& x)
{
    return unpack([](auto... ys) { return make_multi_index(ys...); }, x);
}

template <index_t NSize, typename X>
CK_TILE_HOST_DEVICE constexpr auto operator+=(multi_index<NSize>& y, const X& x)
{
    static_assert(X::size() == NSize, "wrong! size not the same");
    static_for<0, NSize, 1>{}([&](auto i) { y[i] += x[i]; });
    return y;
}

template <index_t NSize, typename X>
CK_TILE_HOST_DEVICE constexpr auto operator-=(multi_index<NSize>& y, const X& x)
{
    static_assert(X::size() == NSize, "wrong! size not the same");
    static_for<0, NSize, 1>{}([&](auto i) { y[i] -= x[i]; });
    return y;
}

template <index_t NSize, typename T>
CK_TILE_HOST_DEVICE constexpr auto operator+(const multi_index<NSize>& a, const T& b)
{
    using type = multi_index<NSize>;
    static_assert(T::size() == NSize, "wrong! size not the same");
    type r;
    static_for<0, NSize, 1>{}([&](auto i) { r[i] = a[i] + b[i]; });
    return r;
}

template <index_t NSize, typename T>
CK_TILE_HOST_DEVICE constexpr auto operator-(const multi_index<NSize>& a, const T& b)
{
    using type = multi_index<NSize>;
    static_assert(T::size() == NSize, "wrong! size not the same");
    type r;
    static_for<0, NSize, 1>{}([&](auto i) { r[i] = a[i] - b[i]; });
    return r;
}

template <index_t NSize, typename T>
CK_TILE_HOST_DEVICE constexpr auto operator*(const multi_index<NSize>& a, const T& b)
{
    using type = multi_index<NSize>;
    static_assert(T::size() == NSize, "wrong! size not the same");
    type r;
    static_for<0, NSize, 1>{}([&](auto i) { r[i] = a[i] * b[i]; });
    return r;
}

// multi_index = index_t * multi_index
template <index_t NSize>
CK_TILE_HOST_DEVICE constexpr auto operator*(index_t a, const multi_index<NSize>& x)
{
    multi_index<NSize> r;
    static_for<0, NSize, 1>{}([&](auto i) { r[i] = a * x[i]; });
    return r;
}

// multi_index = multi_index * index_t
template <index_t NSize>
CK_TILE_HOST_DEVICE constexpr auto operator*(const multi_index<NSize>& x, index_t a)
{
    return a * x;
}

} // namespace ck_tile
