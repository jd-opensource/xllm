// SPDX-License-Identifier: MIT
// Copyright (c) 2025, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"

namespace ck_tile {

/// Declare a ck_tile::print() interface that gets specialized in each header file for types that
/// can be printed.
template <typename T>
CK_TILE_HOST_DEVICE void print(const T&)
{
    static_assert(sizeof(T) == 0,
                  "No print implementation available for this type. Please specialize "
                  "ck_tile::print for your type.");
}

/// Specialization for int
template <>
CK_TILE_HOST_DEVICE void print(const int& value)
{
    printf("%d", value);
}

/// Specialization for float
template <>
CK_TILE_HOST_DEVICE void print(const float& value)
{
    printf("%f", value);
}

/// Specialization for double
template <>
CK_TILE_HOST_DEVICE void print(const double& value)
{
    printf("%f", value);
}

/// Specialization for long
template <>
CK_TILE_HOST_DEVICE void print(const long& value)
{
    printf("%ld", value);
}

/// Specialization for unsigned int
template <>
CK_TILE_HOST_DEVICE void print(const unsigned int& value)
{
    printf("%u", value);
}

/// Specialization for char
template <>
CK_TILE_HOST_DEVICE void print(const char& value)
{
    printf("%c", value);
}

/// Specialization for array
template <typename T, size_t N>
CK_TILE_HOST_DEVICE void print(const T (&value)[N])
{
    printf("[");
    for(size_t i = 0; i < N; ++i)
    {
        if(i > 0)
            printf(", ");
        print(value[i]); // Recursively call print for each element
    }
    printf("]");
}

} // namespace ck_tile
