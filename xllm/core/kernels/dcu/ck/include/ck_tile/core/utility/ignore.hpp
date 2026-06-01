// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

// https://en.cppreference.com/w/cpp/utility/tuple/ignore

namespace ck_tile {

namespace detail {
struct ignore_t
{
    template <typename T>
    constexpr void operator=(T&&) const noexcept
    {
    }
};
} // namespace detail

inline constexpr detail::ignore_t ignore;

} // namespace ck_tile
