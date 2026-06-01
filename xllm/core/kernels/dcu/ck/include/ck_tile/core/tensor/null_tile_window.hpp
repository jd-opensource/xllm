// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/tensor/tensor_view.hpp"

namespace ck_tile {

// placeholder type if we want to opt-out a tile window parameter
template <typename WindowLengths_>
struct null_tile_window
{
    using BottomTensorView = null_tensor_view;
    using WindowLengths    = remove_cvref_t<WindowLengths_>;

    using BottomTensorIndex = array<index_t, WindowLengths::size()>;

    CK_TILE_DEVICE constexpr null_tile_window() = default;

    CK_TILE_DEVICE constexpr null_tile_window(const WindowLengths& window_lengths)
        : window_lengths_{window_lengths}
    {
    }

    CK_TILE_DEVICE constexpr auto get_window_lengths() const { return window_lengths_; }

    CK_TILE_DEVICE constexpr auto get_bottom_tensor_view() const { return null_tensor_view{}; }

    CK_TILE_DEVICE constexpr auto get_window_origin() const { return BottomTensorIndex{}; }

    CK_TILE_DEVICE void init_raw() {}

    WindowLengths window_lengths_;
};

// utility to check if this is a Null Tile Window
namespace impl {
template <typename>
struct is_null_tile_window : public std::false_type
{
};

template <typename T>
struct is_null_tile_window<null_tile_window<T>> : public std::true_type
{
};
} // namespace impl

template <typename T>
CK_TILE_DEVICE constexpr auto is_null_tile_window(const T&)
{
    return impl::is_null_tile_window<remove_cvref_t<T>>::value;
}

template <typename WindowLengths>
CK_TILE_DEVICE constexpr auto make_null_tile_window(const WindowLengths& window_lengths)
{
    static_assert(ck_tile::is_known_at_compile_time<WindowLengths>::value,
                  "wrong! lengths should be static");

    return null_tile_window<remove_cvref_t<WindowLengths>>{window_lengths};
}

template <typename WindowLengths, typename... Ts>
CK_TILE_DEVICE constexpr auto make_tile_window(null_tensor_view,
                                               const WindowLengths& window_lengths,
                                               const multi_index<WindowLengths::size()>& /*origin*/,
                                               Ts&&...)
{
    static_assert(ck_tile::is_known_at_compile_time<WindowLengths>::value,
                  "wrong! lengths should be static");

    return null_tile_window<remove_cvref_t<WindowLengths>>{window_lengths};
}

template <typename WindowLengths, typename StaticTileDistribution>
CK_TILE_DEVICE constexpr auto make_tile_window(const null_tile_window<WindowLengths>& t,
                                               const StaticTileDistribution&)
{
    return t;
}

template <typename WindowLengths>
CK_TILE_DEVICE void
move_tile_window(null_tile_window<WindowLengths>&,
                 const typename null_tile_window<WindowLengths>::BottomTensorIndex&)
{
}

} // namespace ck_tile
