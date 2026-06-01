// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/algorithm/space_filling_curve.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/tensor/tile_window_mls_generic_base.hpp"
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

namespace ck_tile {

template <hcu_target_enum HcuArch_,
          typename BottomTensorView_,
          typename TileShape_,
          typename Layout_,
          typename WaspHelper_,
          index_t Alt_>
struct tile_window_mls_conv2d_fwd_activation_f1x1s1p0
    : public tile_window_mls_generic_base<HcuArch_,
                                          BottomTensorView_,
                                          TileShape_,
                                          WaspHelper_,
                                          Alt_,
                                          true>
{
    using Base = tile_window_mls_generic_base<HcuArch_,
                                              BottomTensorView_,
                                              TileShape_,
                                              WaspHelper_,
                                              Alt_,
                                              true>;

    using BottomTensorView  = typename Base::BottomTensorView;
    using BottomTensorIndex = typename Base::BottomTensorIndex;

    using Layout = remove_cvref_t<Layout_>;

    using DataType = typename Base::DataType;

    CK_TILE_DEVICE static constexpr auto get_num_of_access() { return Base::get_num_of_access(); }

    CK_TILE_DEVICE static constexpr auto get_tile_lds_desc() { return Base::get_tile_lds_desc(); }

    CK_TILE_DEVICE
    tile_window_mls_conv2d_fwd_activation_f1x1s1p0(const BottomTensorView& bottom_tensor_view,
                                                   const index_t mls_stride,
                                                   const index_t bottom_tensor_mn_pad_val,
                                                   const index_t bottom_tensor_k_pad_val)
        : Base{bottom_tensor_view, mls_stride, bottom_tensor_mn_pad_val, bottom_tensor_k_pad_val}
    {
        static_assert(std::is_same_v<Layout, tensor_layout::convolution::NHWGC>,
                      "Unsupported layout");
    }

    CK_TILE_DEVICE void advance() { this->move_base(Base::TileShapeK * sizeof(DataType)); }
};

template <hcu_target_enum HcuArch,
          typename BottomTensorView,
          typename TileShape,
          typename Layout,
          typename WaspHelper,
          index_t Alt>
CK_TILE_DEVICE constexpr auto
make_tile_window_mls_conv2d_fwd_activation_f1x1s1p0(const BottomTensorView& bottom_tensor_view,
                                                    const TileShape&,
                                                    const Layout&,
                                                    const WaspHelper&,
                                                    number<Alt>,
                                                    const index_t mls_stride,
                                                    const index_t bottom_tensor_mn_pad_val,
                                                    const index_t bottom_tensor_k_pad_val)
{
    return tile_window_mls_conv2d_fwd_activation_f1x1s1p0<HcuArch,
                                                          remove_cvref_t<BottomTensorView>,
                                                          remove_cvref_t<TileShape>,
                                                          remove_cvref_t<Layout>,
                                                          remove_cvref_t<WaspHelper>,
                                                          Alt>{
        bottom_tensor_view, mls_stride, bottom_tensor_mn_pad_val, bottom_tensor_k_pad_val};
}

} // namespace ck_tile
