// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/tensor/tile_window.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

template <typename BottomTensorView_,
          typename WindowLengths_,
          index_t... SliceBegins,
          index_t... SliceEnds>
CK_TILE_DEVICE constexpr auto
get_slice_tile(const tile_window_with_static_lengths<BottomTensorView_, WindowLengths_>& tile,
               sequence<SliceBegins...> slice_begins,
               sequence<SliceEnds...> slice_ends)
{
    using TileWindow = tile_window_with_static_lengths<BottomTensorView_, WindowLengths_>;
    // NOTE: This API will override the origin of the tile window!
    static_assert(sizeof...(SliceBegins) == sizeof...(SliceEnds));
    static_assert(sizeof...(SliceBegins) == TileWindow::get_num_of_dimension());

    constexpr auto slice_lengths = slice_ends - slice_begins;

    return make_tile_window(tile.get_bottom_tensor_view(),
                            sequence_to_tuple_of_number(slice_lengths),
                            to_multi_index(slice_begins));
}

template <typename DataType_,
          typename StaticTileDistribution_,
          index_t... SliceBegins,
          index_t... SliceEnds>
CK_TILE_DEVICE constexpr auto
get_slice_tile(const static_distributed_tensor<DataType_, StaticTileDistribution_>& tile,
               sequence<SliceBegins...> slice_begins,
               sequence<SliceEnds...> slice_ends)
{
    using DataType     = remove_cvref_t<DataType_>;
    using Distribution = remove_cvref_t<StaticTileDistribution_>;

    constexpr auto sliced_dstr_yidx_ylen =
        detail::slice_distribution_from_x(Distribution{}, slice_begins, slice_ends);

    constexpr auto sliced_dstr      = sliced_dstr_yidx_ylen.template at<0>();
    constexpr auto sliced_y_origins = sliced_dstr_yidx_ylen.template at<1>();
    constexpr auto sliced_y_lengths = sliced_dstr_yidx_ylen.template at<2>();

    auto sliced_tensor = make_static_distributed_tensor<DataType>(sliced_dstr);

    sliced_tensor.get_thread_buffer() =
        tile.get_y_sliced_thread_data(sliced_y_origins, sliced_y_lengths);

    return sliced_tensor;
}

template <typename DstDataType_,
          typename DstStaticTileDistribution_,
          typename SrcDataType_,
          typename SrcStaticTileDistribution_,
          index_t... SliceBegins,
          index_t... SliceEnds>
CK_TILE_DEVICE constexpr auto
set_slice_tile(static_distributed_tensor<DstDataType_, DstStaticTileDistribution_>& dst_tile,
               const static_distributed_tensor<SrcDataType_, SrcStaticTileDistribution_>& src_tile,
               sequence<SliceBegins...> slice_begins,
               sequence<SliceEnds...> slice_ends)
{
    using DstDistribution = remove_cvref_t<DstStaticTileDistribution_>;

    constexpr auto sliced_dstr_yidx_ylen =
        detail::slice_distribution_from_x(DstDistribution{}, slice_begins, slice_ends);

    constexpr auto sliced_dstr      = sliced_dstr_yidx_ylen.template at<0>();
    constexpr auto sliced_y_origins = sliced_dstr_yidx_ylen.template at<1>();
    constexpr auto sliced_y_lengths = sliced_dstr_yidx_ylen.template at<2>();

    static_assert(std::is_same_v<decltype(sliced_dstr), DstDistribution>, "wrong!");

    dst_tile.SetSlicedThreadData(sliced_y_origins, sliced_y_lengths, src_tile.get_thread_buffer());
}

} // namespace ck_tile
