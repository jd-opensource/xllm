// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include <cstddef>

namespace ck_tile {

// TODO: this structure is not intented to be used by user
template <index_t MaxSize>
struct meta_data_buffer
{
    CK_TILE_HOST_DEVICE constexpr meta_data_buffer() : buffer_{}, size_{0} {}

    template <typename X, typename... Xs>
    CK_TILE_HOST_DEVICE constexpr meta_data_buffer(const X& x, const Xs&... xs)
        : buffer_{}, size_{0}
    {
        push(x, xs...);
    }

    template <typename T>
    CK_TILE_HOST_DEVICE constexpr void push(const T& data)
    {
        if constexpr(!std::is_empty_v<T>)
        {
            constexpr index_t size = sizeof(T);

            auto tmp = bit_cast<array<std::byte, size>>(data);

            for(int i = 0; i < size; i++)
            {
                buffer_(size_) = tmp[i];

                size_++;
            }
        }
    }

    template <typename X, typename... Xs>
    CK_TILE_HOST_DEVICE constexpr void push(const X& x, const Xs&... xs)
    {
        push(x);
        push(xs...);
    }

    template <typename T>
    CK_TILE_HOST_DEVICE constexpr T pop(index_t& pos) const
    {
        T data;

        if constexpr(!std::is_empty_v<T>)
        {
            constexpr index_t size = sizeof(T);

            array<std::byte, size> tmp;

            for(int i = 0; i < size; i++)
            {
                tmp(i) = buffer_[pos];

                pos++;
            }

            data = bit_cast<T>(tmp);
        }

        return data;
    }

    template <typename T>
    CK_TILE_HOST_DEVICE constexpr T get(index_t pos) const
    {
        constexpr index_t size = sizeof(T);

        array<std::byte, size> tmp;

        for(int i = 0; i < size; i++)
        {
            tmp(i) = buffer_[pos];

            pos++;
        }

        auto data = bit_cast<T>(tmp);

        return data;
    }

    //
    array<std::byte, MaxSize> buffer_;
    index_t size_ = 0;
};

} // namespace ck_tile
