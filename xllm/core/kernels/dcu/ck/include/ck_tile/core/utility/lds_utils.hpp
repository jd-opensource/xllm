// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

struct lds_utils
{
    template <typename T,
              index_t NumBuffers,
              typename WindowLengths,
              typename StaticTileDistribution,
              typename... Ts>
    CK_TILE_HOST_DEVICE static auto
    AllocateLdsWindows(void* lds_ptr,
                       index_t num_elemes_per_buffer,
                       index_t start_elem_offset,
                       index_t alignment,
                       const WindowLengths& window_lengths,
                       const multi_index<WindowLengths::size()>& origin,
                       const tensor_descriptor<Ts...>& desc,
                       const StaticTileDistribution& dstr)
    {
        T* lds_start = static_cast<T*>(lds_ptr) + start_elem_offset;
        const index_t single_buffer_offset =
            ck_tile::integer_least_multiple(num_elemes_per_buffer, alignment);
        return ck_tile::generate_tuple(
            [&](auto i) {
                const index_t local_offset = i * single_buffer_offset;
                auto view =
                    make_tensor_view<address_space_enum::lds>(lds_start + local_offset, desc);
                return make_tile_window(view, window_lengths, origin, dstr);
            },
            number<NumBuffers>{});
    }

    template <typename T, index_t NumBuffers, typename WindowLengths, typename... Ts>
    CK_TILE_HOST_DEVICE static auto
    AllocateLdsWindows(void* lds_ptr,
                       index_t num_elemes_per_buffer,
                       index_t start_elem_offset,
                       index_t alignment,
                       const WindowLengths& window_lengths,
                       const multi_index<WindowLengths::size()>& origin,
                       const tensor_descriptor<Ts...>& desc)
    {
        T* lds_start = static_cast<T*>(lds_ptr) + start_elem_offset;
        const index_t single_buffer_offset =
            ck_tile::integer_least_multiple(num_elemes_per_buffer, alignment);
        return ck_tile::generate_tuple(
            [&](auto i) {
                const index_t local_offset = i * single_buffer_offset;
                auto view =
                    make_tensor_view<address_space_enum::lds>(lds_start + local_offset, desc);
                return make_tile_window(view, window_lengths, origin);
            },
            number<NumBuffers>{});
    }
};

} // namespace ck_tile
