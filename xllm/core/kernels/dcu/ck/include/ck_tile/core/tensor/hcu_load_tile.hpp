// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/tensor/tile_window_conv2d_fwd_activation_async.hpp"
#include "ck_tile/core/tensor/tile_window_conv2d_fwd_filter_async.hpp"
#include "ck_tile/core/tensor/tile_window_convnd_fwd_lds.hpp"

namespace ck_tile {

template <typename T, typename Window, index_t i_access = -1, bool oob_conditional_check = true>
CK_TILE_DEVICE void async_load_tile_asm(T* smem_ptr,
                                        const Window& tile_window,
                                        number<i_access>                     = {},
                                        bool_constant<oob_conditional_check> = {})
{
    tile_window.async_load_asm(smem_ptr,
                               number<i_access>{},
                               bool_constant<false>{},
                               bool_constant<oob_conditional_check>{});
}

template <typename T, typename Window, index_t i_access = -1, bool oob_conditional_check = true>
CK_TILE_DEVICE void async_load_tile_wrapped_asm(T* smem_ptr,
                                                const Window& tile_window,
                                                number<i_access>                     = {},
                                                bool_constant<oob_conditional_check> = {})
{
    tile_window.async_load_asm(smem_ptr,
                               number<i_access>{},
                               bool_constant<true>{},
                               bool_constant<oob_conditional_check>{});
}

template <typename LdsWindow,
          typename TileWindow,
          index_t samp_idx,
          index_t i_access           = -1,
          bool oob_conditional_check = true>
CK_TILE_DEVICE void async_load_tile_wrapped_asm(LdsWindow& lds_window,
                                                const TileWindow& tile_window,
                                                number<samp_idx>,
                                                number<i_access>                     = {},
                                                bool_constant<oob_conditional_check> = {})
{
    tile_window.async_load_asm(lds_window.get_buffer_ptr(),
                               number<samp_idx>{},
                               number<i_access>{},
                               bool_constant<true>{},
                               bool_constant<oob_conditional_check>{});
}

template <typename T, typename Window, index_t i_access = -1, bool oob_conditional_check = true>
CK_TILE_DEVICE void load_tile_asm(T& tile,
                                  const Window& tile_window,
                                  number<i_access>                     = {},
                                  bool_constant<oob_conditional_check> = {})
{
    tile_window.load_asm(tile, number<i_access>{}, bool_constant<oob_conditional_check>{});
}

template <typename Window, index_t i_access = -1, bool oob_conditional_check = true>
CK_TILE_DEVICE auto load_tile_asm(const Window& tile_window,
                                  number<i_access>                     = {},
                                  bool_constant<oob_conditional_check> = {})
{
    return tile_window.load_asm(number<i_access>{}, bool_constant<oob_conditional_check>{});
}

template <typename Windows, index_t i_access = -1, bool oob_conditional_check = true>
CK_TILE_DEVICE auto load_tiles_asm(const Windows& tile_windows,
                                   number<i_access>                     = {},
                                   bool_constant<oob_conditional_check> = {})
{
    return generate_tuple(
        [&](auto i) {
            return load_tile_asm(
                tile_windows[i], number<i_access>{}, bool_constant<oob_conditional_check>{});
        },
        number<Windows::size()>{});
}

template <typename LdsWindow,
          typename TileWindow,
          index_t samp_idx,
          index_t samp_num = 0,
          bool use_m0      = false>
CK_TILE_DEVICE void async_tls_load_tile_asm(LdsWindow& lds_window,
                                            TileWindow& tile_window,
                                            number<samp_idx>,
                                            number<samp_num>      = {},
                                            bool_constant<use_m0> = {})
{
    tile_window.async_tls_load_asm(lds_window.get_buffer_ptr(),
                                   number<samp_idx>{},
                                   number<samp_num>{},
                                   bool_constant<use_m0>{});
}

template <typename LdsWindow, typename TileWindow, bool last_load = false>
CK_TILE_DEVICE void async_mls_load_tile_asm(LdsWindow& lds_window,
                                            TileWindow& tile_window,
                                            bool_constant<last_load> = {})
{
    tile_window.async_mls_load_asm(
        lds_window.get_buffer_ptr(), bool_constant<false>{}, bool_constant<last_load>{});
}

template <typename LdsWindow, typename TileWindow, bool last_load = false>
CK_TILE_DEVICE void async_mls_bps_load_tile_asm(LdsWindow& lds_window,
                                                TileWindow& tile_window,
                                                bool_constant<last_load> = {})
{
    tile_window.async_mls_load_asm(
        lds_window.get_buffer_ptr(), bool_constant<true>{}, bool_constant<last_load>{});
}

template <typename Window, index_t next_samp_idx>
CK_TILE_DEVICE void test_and_advance_tile_window(Window& window, number<next_samp_idx>)
{
    window.test_and_advance(number<next_samp_idx>{});
}

template <typename Window>
CK_TILE_DEVICE void advance_tile_window(Window& window)
{
    window.advance();
}

template <typename Windows>
CK_TILE_DEVICE void advance_tile_windows(Windows& windows)
{
    static_for<0, Windows::size(), 1>{}([&](auto i) { advance_tile_window(windows(i)); });
}

template <typename Window, index_t samp_idx>
CK_TILE_DEVICE void advance_tile_window(Window& window, number<samp_idx>)
{
    window.advance(number<samp_idx>{});
}

} // namespace ck_tile
