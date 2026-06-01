// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"

namespace ck_tile {

template <typename Y, typename X>
CK_TILE_HOST_DEVICE constexpr Y bit_cast(const X& x)
{
    static_assert(__has_builtin(__builtin_bit_cast), "");
    static_assert(sizeof(X) == sizeof(Y), "Do not support cast between different size of type");

    return __builtin_bit_cast(Y, x);
}

} // namespace ck_tile
