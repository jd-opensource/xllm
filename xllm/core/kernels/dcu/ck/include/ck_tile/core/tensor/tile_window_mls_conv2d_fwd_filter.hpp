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
          typename FilterSize_,
          typename Layout_,
          typename WaspHelper_,
          index_t Alt_>
struct tile_window_mls_conv2d_fwd_filter : public tile_window_mls_generic_base<HcuArch_,
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

    static constexpr auto FilterRS = container_reduce(FilterSize_{}, multiplies{}, number<1>{});

    CK_TILE_DEVICE static constexpr auto get_num_of_access() { return Base::get_num_of_access(); }

    CK_TILE_DEVICE static constexpr auto get_tile_lds_desc() { return Base::get_tile_lds_desc(); }

    CK_TILE_DEVICE tile_window_mls_conv2d_fwd_filter(const BottomTensorView& bottom_tensor_view,
                                                     const index_t mls_stride,
                                                     const index_t bottom_tensor_mn_pad_val,
                                                     const index_t bottom_tensor_k_pad_val)
        : Base{bottom_tensor_view, mls_stride, bottom_tensor_mn_pad_val, bottom_tensor_k_pad_val}
    {
        // compute filter inc value
        if constexpr(std::is_same_v<Layout, tensor_layout::convolution::GKYXC>)
        {
            const auto C = this->bottom_tensor_view_.get_tensor_descriptor()
                               .get_transforms()[number<0>{}]
                               .get_upper_lengths()[number<3>{}];

            filter_inc_rs_ = __builtin_amdgcn_readfirstlane(C);
            filter_inc_c_  = __builtin_amdgcn_readfirstlane(Base::TileShapeK - (FilterRS - 1) * C);
        }
        else
        {
            static_assert(false, "Unsupported layout");
        }
    }

    template <index_t next_samp_idx>
    CK_TILE_DEVICE void advance(number<next_samp_idx>)
    {
        if constexpr(next_samp_idx % FilterRS == 0)
        {
            this->move_base(filter_inc_c_ * sizeof(DataType));
        }
        else
        {
            this->move_base(filter_inc_rs_ * sizeof(DataType));
        }
    }

    index_t filter_inc_rs_, filter_inc_c_;
};

template <hcu_target_enum HcuArch_,
          typename BottomTensorView_,
          typename TileShape_,
          typename Layout_,
          typename WaspHelper_,
          index_t Alt_>
struct tile_window_mls_conv2d_fwd_filter<HcuArch_,
                                         BottomTensorView_,
                                         TileShape_,
                                         sequence<1, 1>,
                                         Layout_,
                                         WaspHelper_,
                                         Alt_>
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

    CK_TILE_DEVICE tile_window_mls_conv2d_fwd_filter(const BottomTensorView& bottom_tensor_view,
                                                     const index_t mls_stride,
                                                     const index_t bottom_tensor_mn_pad_val,
                                                     const index_t bottom_tensor_k_pad_val)
        : Base{bottom_tensor_view, mls_stride, bottom_tensor_mn_pad_val, bottom_tensor_k_pad_val}
    {
        // compute filter inc value
        if constexpr(std::is_same_v<Layout, tensor_layout::convolution::GKYXC>) {}
        else
        {
            static_assert(false, "Unsupported layout");
        }
    }

    CK_TILE_DEVICE void advance() { this->move_base(Base::TileShapeK * sizeof(DataType)); }
};

template <hcu_target_enum HcuArch,
          typename BottomTensorView,
          typename TileShape,
          typename FilterSize,
          typename Layout,
          typename WaspHelper,
          index_t Alt>
CK_TILE_DEVICE constexpr auto
make_tile_window_mls_conv2d_fwd_filter(const BottomTensorView& bottom_tensor_view,
                                       const TileShape&,
                                       const FilterSize&,
                                       const Layout&,
                                       const WaspHelper&,
                                       number<Alt>,
                                       const index_t mls_stride,
                                       const index_t bottom_tensor_mn_pad_val,
                                       const index_t bottom_tensor_k_pad_val)
{
    return tile_window_mls_conv2d_fwd_filter<HcuArch,
                                             remove_cvref_t<BottomTensorView>,
                                             remove_cvref_t<TileShape>,
                                             remove_cvref_t<FilterSize>,
                                             remove_cvref_t<Layout>,
                                             remove_cvref_t<WaspHelper>,
                                             Alt>{
        bottom_tensor_view, mls_stride, bottom_tensor_mn_pad_val, bottom_tensor_k_pad_val};
}

} // namespace ck_tile
