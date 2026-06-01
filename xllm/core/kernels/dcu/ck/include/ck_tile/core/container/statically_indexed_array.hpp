// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/numeric/integer.hpp"

namespace ck_tile {

#if CK_TILE_STATICALLY_INDEXED_ARRAY_DEFAULT == CK_TILE_STATICALLY_INDEXED_ARRAY_USE_TUPLE

template <typename T, index_t N>
using statically_indexed_array = tuple_array<T, N>;

#else

// consider mark this struct as deprecated
template <typename T, index_t N>
using statically_indexed_array = array<T, N>;

#endif

// consider always use ck_tile::array for this purpose
#if 0
template <typename X, typename... Xs>
CK_TILE_HOST_DEVICE constexpr auto make_statically_indexed_array(const X& x, const Xs&... xs)
{
    return statically_indexed_array<X, sizeof...(Xs) + 1>(x, static_cast<X>(xs)...);
}

// make empty statically_indexed_array
template <typename X>
CK_TILE_HOST_DEVICE constexpr auto make_statically_indexed_array()
{
    return statically_indexed_array<X, 0>();
}
#endif
} // namespace ck_tile
