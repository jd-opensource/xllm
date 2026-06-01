// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include <iterator>
#include <type_traits>
#include <utility>

// ranges implementation are not intented to be used by user
// TODO: do we need this?
namespace ck_tile {

template <typename T>
using iter_value_t = typename std::iterator_traits<remove_cvref_t<T>>::value_type;

template <typename T>
using iter_reference_t = decltype(*std::declval<T&>());

template <typename T>
using iter_difference_t = typename std::iterator_traits<remove_cvref_t<T>>::difference_type;

namespace ranges {
template <typename R>
using iterator_t = decltype(std::begin(std::declval<R&>()));

template <typename R>
using sentinel_t = decltype(std::end(std::declval<R&>()));

template <typename R>
using range_size_t = decltype(std::size(std::declval<R&>()));

template <typename R>
using range_difference_t = ck_tile::iter_difference_t<ranges::iterator_t<R>>;

template <typename R>
using range_value_t = iter_value_t<ranges::iterator_t<R>>;

template <typename R>
using range_reference_t = iter_reference_t<ranges::iterator_t<R>>;

template <typename T, typename = void>
struct is_range : std::false_type
{
};

template <typename T>
struct is_range<
    T,
    std::void_t<decltype(std::begin(std::declval<T&>())), decltype(std::end(std::declval<T&>()))>>
    : std::true_type
{
};

template <typename T>
inline constexpr bool is_range_v = is_range<T>::value;

template <typename T, typename = void>
struct is_sized_range : std::false_type
{
};

template <typename T>
struct is_sized_range<T, std::void_t<decltype(std::size(std::declval<T&>()))>>
    : std::bool_constant<is_range_v<T>>
{
};
} // namespace ranges
} // namespace ck_tile
