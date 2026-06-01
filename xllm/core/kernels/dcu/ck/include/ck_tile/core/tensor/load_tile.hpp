// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"
#include "ck_tile/core/tensor/tile_window_conv2d_fwd_activation.hpp"
#include "ck_tile/core/tensor/tile_window_conv2d_fwd_filter.hpp"
#include "ck_tile/core/tensor/tile_window_lds.hpp"
#include "ck_tile/core/tensor/tile_window_linear.hpp"
#include "ck_tile/core/tensor/null_tile_window.hpp"
#include "ck_tile/core/tensor/null_tensor.hpp"

namespace ck_tile {

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          bool oob_conditional_check = true,
          bool swizzle               = false>
CK_TILE_DEVICE auto load_tile(const tile_window_with_static_distribution<BottomTensorView_,
                                                                         WindowLengths_,
                                                                         TileDistribution_,
                                                                         NumCoord>& tile_window,
                              bool_constant<oob_conditional_check> = {},
                              bool_constant<swizzle>               = {})
{
    return tile_window.load(
        number<-1>{}, bool_constant<oob_conditional_check>{}, bool_constant<swizzle>{});
}

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          typename LinearBottomDims_,
          bool oob_conditional_check = true>
CK_TILE_DEVICE auto load_tile(const tile_window_linear<BottomTensorView_,
                                                       WindowLengths_,
                                                       TileDistribution_,
                                                       LinearBottomDims_>& tile_window,
                              bool_constant<oob_conditional_check> = {})
{
    return tile_window.load(number<-1>{}, bool_constant<oob_conditional_check>{});
}

template <typename DistributedTensor_,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          bool oob_conditional_check = true,
          bool swizzle               = false>
CK_TILE_DEVICE auto load_tile(DistributedTensor_& dst_tile,
                              const tile_window_with_static_distribution<BottomTensorView_,
                                                                         WindowLengths_,
                                                                         TileDistribution_,
                                                                         NumCoord>& tile_window,
                              bool_constant<oob_conditional_check> = {},
                              bool_constant<swizzle>               = {})
{
    return tile_window.load(
        dst_tile, bool_constant<oob_conditional_check>{}, bool_constant<swizzle>{});
}

template <typename TileWindow, bool oob_conditional_check = true>
CK_TILE_DEVICE auto load_tile_by_inline_asm(const TileWindow& tile_window,
                                            bool_constant<oob_conditional_check> = {})
{
    return tile_window.load_by_inline_asm(number<-1>{}, bool_constant<oob_conditional_check>{});
}

template <typename DistributedTensor, typename TileWindow, bool oob_conditional_check = true>
CK_TILE_DEVICE void load_tile_by_inline_asm(DistributedTensor& dst_tile,
                                            const TileWindow& tile_window,
                                            bool_constant<oob_conditional_check> = {})
{
    tile_window.load_by_inline_asm(dst_tile, bool_constant<oob_conditional_check>{});
}

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename FilterWindow_,
          typename TileDistribution_,
          typename Layout_,
          index_t i_access           = -1,
          bool oob_conditional_check = true>
CK_TILE_DEVICE auto
load_tile_by_inline_asm(const tile_window_conv2d_fwd_activation<BottomTensorView_,
                                                                WindowLengths_,
                                                                FilterWindow_,
                                                                TileDistribution_,
                                                                Layout_>& tile_window,
                        number<i_access>                     = {},
                        bool_constant<oob_conditional_check> = {})
{
    return tile_window.load_by_inline_asm(number<i_access>{},
                                          bool_constant<oob_conditional_check>{});
}

template <typename DistributedTensor_,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename FilterWindow_,
          typename TileDistribution_,
          typename Layout_,
          index_t i_access           = -1,
          bool oob_conditional_check = true>
CK_TILE_DEVICE void
load_tile_by_inline_asm(DistributedTensor_& dst_tile,
                        const tile_window_conv2d_fwd_activation<BottomTensorView_,
                                                                WindowLengths_,
                                                                FilterWindow_,
                                                                TileDistribution_,
                                                                Layout_>& tile_window,
                        number<i_access>                     = {},
                        bool_constant<oob_conditional_check> = {})
{
    tile_window.load_by_inline_asm(
        dst_tile, number<i_access>{}, bool_constant<oob_conditional_check>{});
}

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename FilterWindow_,
          typename TileDistribution_,
          typename Layout_,
          index_t i_access           = -1,
          bool oob_conditional_check = true>
CK_TILE_DEVICE auto
load_tile_by_inline_asm(const tile_window_conv2d_fwd_filter<BottomTensorView_,
                                                            WindowLengths_,
                                                            FilterWindow_,
                                                            TileDistribution_,
                                                            Layout_>& tile_window,
                        number<i_access>                     = {},
                        bool_constant<oob_conditional_check> = {})
{
    return tile_window.load_by_inline_asm(number<i_access>{},
                                          bool_constant<oob_conditional_check>{});
}

template <typename DistributedTensor_,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename FilterWindow_,
          typename TileDistribution_,
          typename Layout_,
          index_t i_access           = -1,
          bool oob_conditional_check = true>
CK_TILE_DEVICE auto
load_tile_by_inline_asm(DistributedTensor_& dst_tile,
                        const tile_window_conv2d_fwd_filter<BottomTensorView_,
                                                            WindowLengths_,
                                                            FilterWindow_,
                                                            TileDistribution_,
                                                            Layout_>& tile_window,
                        number<i_access>                     = {},
                        bool_constant<oob_conditional_check> = {})
{
    return tile_window.load_by_inline_asm(
        dst_tile, number<i_access>{}, bool_constant<oob_conditional_check>{});
}

template <typename DsreadmInstr,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord>
CK_TILE_DEVICE auto
load_tile_by_dsreadm(const DsreadmInstr& instr,
                     const tile_window_with_static_distribution<BottomTensorView_,
                                                                WindowLengths_,
                                                                TileDistribution_,
                                                                NumCoord>& tile_window)
{
    // limitations
    static_assert(BottomTensorView_::buffer_view::get_address_space() == address_space_enum::lds,
                  "tile_window must be lds window!");
    static_assert(TileDistribution_::NDimP == 1, "tile_window must be warp tile!");

    auto tmp = tile_window.load_by_dsreadm(instr);
    return make_static_distributed_tensor<typename BottomTensorView_::DataType>(
        make_static_tile_distribution(typename DsreadmInstr::WarpStoreDstrEncoding{}),
        tmp.get_thread_buffer());
}

/**
 * @brief Loads a tile of data using inline assembly.
 *
 * @note Bare in mind that loading data this way, you have to manually initialize your
 *       thread buffer and synchronize load afterwards in order to make sure it's done before
 *       using loaded data from registers
 *       @see `tile_window_with_static_distribution::init_raw()` and `buffer_view.hpp`
 *       @see  `buffer_load_fence()`
 */
template <typename T,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          bool oob_conditional_check = true,
          bool pre_nop               = false,
          bool swizzle               = false>
CK_TILE_DEVICE auto load_tile_raw(T& tile,
                                  const tile_window_with_static_distribution<BottomTensorView_,
                                                                             WindowLengths_,
                                                                             TileDistribution_,
                                                                             NumCoord>& tile_window,
                                  bool_constant<oob_conditional_check> = {},
                                  bool_constant<pre_nop>               = {},
                                  bool_constant<swizzle>               = {})
{
    tile_window.load_raw(tile,
                         number<-1>{},
                         bool_constant<oob_conditional_check>{},
                         bool_constant<pre_nop>{},
                         bool_constant<swizzle>{});
}

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          bool oob_conditional_check = true,
          bool pre_nop               = false,
          bool swizzle               = false>
CK_TILE_DEVICE auto load_tile_raw(const tile_window_with_static_distribution<BottomTensorView_,
                                                                             WindowLengths_,
                                                                             TileDistribution_,
                                                                             NumCoord>& tile_window,
                                  bool_constant<oob_conditional_check> = {},
                                  bool_constant<pre_nop>               = {},
                                  bool_constant<swizzle>               = {})
{
    return tile_window.load_raw(number<-1>{},
                                bool_constant<oob_conditional_check>{},
                                bool_constant<pre_nop>{},
                                bool_constant<swizzle>{});
}

template <typename T,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_>
CK_TILE_DEVICE void load_tile_raw(
    T& tile,
    const tile_window_lds<BottomTensorView_, WindowLengths_, TileDistribution_>& tile_window)
{
    tile_window.load_raw(tile);
}

template <typename BottomTensorView_, typename WindowLengths_, typename TileDistribution_>
CK_TILE_DEVICE auto load_tile_raw(
    const tile_window_lds<BottomTensorView_, WindowLengths_, TileDistribution_>& tile_window)
{
    return tile_window.load_raw();
}

template <typename T,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          typename LinearBottomDims_,
          bool oob_conditional_check = true,
          bool pre_nop               = false>
CK_TILE_DEVICE auto load_tile_raw(T& tile,
                                  const tile_window_linear<BottomTensorView_,
                                                           WindowLengths_,
                                                           TileDistribution_,
                                                           LinearBottomDims_>& tile_window,
                                  bool_constant<oob_conditional_check> = {},
                                  bool_constant<pre_nop>               = {})
{
    tile_window.load_raw(
        tile, number<-1>{}, bool_constant<oob_conditional_check>{}, bool_constant<pre_nop>{});
}

template <typename LdsTileWindow_,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          bool oob_conditional_check = true,
          bool pre_nop               = false>
CK_TILE_DEVICE auto
async_load_tile_raw(LdsTileWindow_&& lds_tile,
                    const tile_window_with_static_distribution<BottomTensorView_,
                                                               WindowLengths_,
                                                               TileDistribution_,
                                                               NumCoord>& tile_window,
                    bool_constant<oob_conditional_check> = {},
                    bool_constant<pre_nop>               = {})
{
    return tile_window.async_load_raw(
        lds_tile, number<-1>{}, bool_constant<oob_conditional_check>{}, bool_constant<pre_nop>{});
}

template <typename LdsTileWindow_,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          typename LinearBottomDims_,
          bool oob_conditional_check = true,
          bool pre_nop               = false>
CK_TILE_DEVICE auto async_load_tile_raw(LdsTileWindow_&& lds_tile,
                                        const tile_window_linear<BottomTensorView_,
                                                                 WindowLengths_,
                                                                 TileDistribution_,
                                                                 LinearBottomDims_>& tile_window,
                                        bool_constant<oob_conditional_check> = {},
                                        bool_constant<pre_nop>               = {})
{
    return tile_window.async_load_raw(
        lds_tile, number<-1>{}, bool_constant<oob_conditional_check>{}, bool_constant<pre_nop>{});
}

template <typename LdsTileWindow_,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          bool oob_conditional_check = true,
          bool pre_nop               = false>
CK_TILE_DEVICE void
async_load_tile_by_inline_asm(LdsTileWindow_&& lds_tile,
                              const tile_window_with_static_distribution<BottomTensorView_,
                                                                         WindowLengths_,
                                                                         TileDistribution_,
                                                                         NumCoord>& tile_window,
                              bool_constant<oob_conditional_check> = {},
                              bool_constant<pre_nop>               = {})
{
    tile_window.async_load_by_inline_asm(
        lds_tile, number<-1>{}, bool_constant<oob_conditional_check>{});
}

CK_TILE_DEVICE auto async_load_fence(index_t cnt = 0)
{
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
}

template <typename WindowLengths>
CK_TILE_DEVICE auto load_tile(const null_tile_window<WindowLengths>&)
{
    return null_tensor{};
}

template <typename T, typename WindowLengths>
CK_TILE_DEVICE auto load_tile_raw(T& /*null_tile*/, const null_tile_window<WindowLengths>&)
{
}

} // namespace ck_tile
