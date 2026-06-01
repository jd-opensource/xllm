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

namespace ck_tile {

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          typename DataType_>
CK_TILE_DEVICE void
update_tile(tile_window_with_static_lengths<BottomTensorView_, WindowLengths_>& tile_window_tmp,
            const static_distributed_tensor<DataType_, TileDistribution_>& dstr_tensor)
{
    using DataType = remove_cvref_t<typename BottomTensorView_::DataType>;
    using TileDstr = remove_cvref_t<TileDistribution_>;

    static_assert(std::is_same_v<remove_cvref_t<DataType_>, DataType>, "wrong!");

    constexpr auto tile_dstr = TileDstr{};

    auto tile_window = make_tile_window(tile_window_tmp.get_bottom_tensor_view(),
                                        tile_window_tmp.get_window_lengths(),
                                        tile_window_tmp.get_window_origin(),
                                        tile_dstr);

    tile_window.update(dstr_tensor);
}

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          typename DataType_>
CK_TILE_DEVICE void
update_tile(tile_window_with_static_distribution<BottomTensorView_,
                                                 WindowLengths_,
                                                 TileDistribution_,
                                                 NumCoord>& tile_window,
            const static_distributed_tensor<DataType_, TileDistribution_>& dstr_tensor)
{
    tile_window.update(dstr_tensor);
}

} // namespace ck_tile
