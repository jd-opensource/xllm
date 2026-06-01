// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once
#include "ck_tile/core/container/sequence.hpp"
// TODO: use c++20 nontype template with struct to implement this

#if 1
// clang happen to support this feature (__cpp_generic_lambdas >= 201707) in c++17 mode
#define TO_SEQUENCE(a, n)                                                                    \
    _Pragma("clang diagnostic push") _Pragma(                                                \
        "clang diagnostic ignored \"-Wc++20-extensions\"")[a]<ck_tile::index_t... IDX_IDX_>( \
        ck_tile::sequence<IDX_IDX_...>)                                                      \
    {                                                                                        \
        return ck_tile::sequence<a.at(ck_tile::number<IDX_IDX_>{})...>{};                    \
    }                                                                                        \
    (ck_tile::make_index_sequence<n>{});                                                     \
    _Pragma("clang diagnostic pop")

#else
// Macro function
// convert constexpr array to sequence, both a/n need to be constexpr (can't be a rvalue like 2)
#define TO_SEQUENCE(a, n)                                                                     \
    [a, n] {                                                                                  \
        static_assert(a.size() >= n, "wrong! out of bound");                                  \
        static_assert(n <= 10, "not implemented");                                            \
        if constexpr(n == 0)                                                                  \
        {                                                                                     \
            return ck_tile::sequence<>{};                                                     \
        }                                                                                     \
        else if constexpr(n == 1)                                                             \
        {                                                                                     \
            return ck_tile::sequence<a[0]>{};                                                 \
        }                                                                                     \
        else if constexpr(n == 2)                                                             \
        {                                                                                     \
            return ck_tile::sequence<a[0], a[1]>{};                                           \
        }                                                                                     \
        else if constexpr(n == 3)                                                             \
        {                                                                                     \
            return ck_tile::sequence<a[0], a[1], a[2]>{};                                     \
        }                                                                                     \
        else if constexpr(n == 4)                                                             \
        {                                                                                     \
            return ck_tile::sequence<a[0], a[1], a[2], a[3]>{};                               \
        }                                                                                     \
        else if constexpr(n == 5)                                                             \
        {                                                                                     \
            return ck_tile::sequence<a[0], a[1], a[2], a[3], a[4]>{};                         \
        }                                                                                     \
        else if constexpr(n == 6)                                                             \
        {                                                                                     \
            return ck_tile::sequence<a[0], a[1], a[2], a[3], a[4], a[5]>{};                   \
        }                                                                                     \
        else if constexpr(n == 7)                                                             \
        {                                                                                     \
            return ck_tile::sequence<a[0], a[1], a[2], a[3], a[4], a[5], a[6]>{};             \
        }                                                                                     \
        else if constexpr(n == 8)                                                             \
        {                                                                                     \
            return ck_tile::sequence<a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]>{};       \
        }                                                                                     \
        else if constexpr(n == 9)                                                             \
        {                                                                                     \
            return ck_tile::sequence<a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]>{}; \
        }                                                                                     \
        else if constexpr(n == 10)                                                            \
        {                                                                                     \
            return ck_tile::                                                                  \
                sequence<a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]>{};       \
        }                                                                                     \
    }()
#endif
