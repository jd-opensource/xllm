// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/algorithm/space_filling_curve.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/arch/hcu_tensor_addressing.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/ops/conv/utility/conv_fwd_spec_v2.hpp"

namespace ck_tile {

template <typename BottomTensorView_,
          typename OutputTensorView_,
          typename BlockWindowLengths_,
          typename TlsWindowLenghts_,
          ConvFwdSpecEnum Spec_,
          typename Layout_,
          typename WaspHelper_>
struct tile_window_conv2d_fwd_tls
{
    using BottomTensorView   = remove_cvref_t<BottomTensorView_>;
    using OutputTensorView   = remove_cvref_t<OutputTensorView_>;
    using BlockWindowLengths = remove_cvref_t<BlockWindowLengths_>;
    using TlsWindowLengths   = remove_cvref_t<TlsWindowLenghts_>;
    using Layout             = remove_cvref_t<Layout_>;
    using WaspHelper         = remove_cvref_t<WaspHelper_>;

    using BottomTensorDesc = remove_cvref_t<typename BottomTensorView::TensorDesc>;
    using DataType         = remove_cvref_t<typename BottomTensorView::DataType>;

    static constexpr auto Spec                = Spec_;
    static constexpr index_t NDimBottomTensor = BottomTensorDesc::get_num_of_dimension();

    using BottomTensorIndex = array<index_t, NDimBottomTensor>;

    using ConvFwdSpecDetail = detail::ConvFwdSpecDetail<Spec>;

    static constexpr auto FilterLengths = ConvFwdSpecDetail::FilterLengths;
    static constexpr auto Strides       = ConvFwdSpecDetail::Strides;
    static constexpr auto Dilations     = ConvFwdSpecDetail::Dilations;
    static constexpr auto Pads          = ConvFwdSpecDetail::Pads;

    static_assert(ck_tile::is_known_at_compile_time<BlockWindowLengths>::value,
                  "wrong! lengths should be static");

    static_assert(ck_tile::is_known_at_compile_time<TlsWindowLengths>::value,
                  "wrong! lengths should be static");

    static_assert((BlockWindowLengths::size() == NDimBottomTensor) &&
                  (BlockWindowLengths::size() == TlsWindowLengths::size()));

    static constexpr auto MNPerTls = TlsWindowLengths::at(number<0>{});
    static constexpr auto KPerTls  = TlsWindowLengths::at(number<1>{});

    static constexpr auto MNPerBlock = BlockWindowLengths::at(number<0>{});
    static constexpr auto KPerBlock  = BlockWindowLengths::at(number<1>{});

    static constexpr auto warp_num = WaspHelper::get_warp_num();

    // immed offset calculation
    static constexpr uint16_t warp_offset_per_issue = MNPerTls * warp_num;

    static constexpr index_t lds_elem_offset_per_warp  = MNPerTls * KPerTls;
    static constexpr index_t lds_elem_offset_per_issue = lds_elem_offset_per_warp * warp_num;

    // limitation
    static_assert((KPerTls == KPerBlock) && (MNPerBlock % (MNPerTls * warp_num) == 0));

    static constexpr auto NumTlsIssue = MNPerBlock / (MNPerTls * warp_num);

    static constexpr index_t FilterUnroll =
        container_reduce(ConvFwdSpecDetail::FilterLengths, multiplies{}, number<1>{});

    struct tls_res_params
    {
        static constexpr auto data_type = detail::tls_resource::data_type_traits<DataType>::value;
        static constexpr auto is_filter = detail::tls_resource::is_filter_traits<Layout>::value;
    };

    struct tls_sampler_params
    {
        static constexpr auto channels_per_pixel =
            detail::tls_sampler::channels_per_pixel_traits<sizeof(DataType), KPerTls>::value;
        static constexpr auto pixel_per_column =
            detail::tls_sampler::pixel_per_column_traits<MNPerTls>::value;
        static constexpr auto element_stride =
            detail::tls_sampler::element_stride_traits<remove_cvref_t<decltype(Strides)>>::value;
        static constexpr auto dilation_rate =
            detail::tls_sampler::dilation_rate_traits<remove_cvref_t<decltype(Dilations)>>::value;
        static constexpr auto padding_num =
            detail::tls_sampler::padding_num_traits<remove_cvref_t<decltype(Pads)>>::value;
        static constexpr auto filter_size =
            detail::tls_sampler::filter_size_traits<remove_cvref_t<decltype(FilterLengths)>>::value;
        static constexpr auto interleave = detail::tls_sampler::interleave_traits<Layout>::value;
        static constexpr auto oob_fill =
            detail::tls_sampler::oob_fill_traits<remove_cvref_t<decltype(Pads)>>::value;
    };

    CK_TILE_DEVICE static constexpr auto get_num_of_access() { return NumTlsIssue; }

    CK_TILE_DEVICE tile_window_conv2d_fwd_tls(const BottomTensorView& bottom_tensor_view,
                                              const OutputTensorView& output_tensor_view,
                                              const BottomTensorIndex& block_window_origin)
    {
        // upper lengths of trans[0] is original tensor
        tls_resource_ = make_tls_resource(bottom_tensor_view.get_buffer_view().p_data_,
                                          bottom_tensor_view.get_tensor_descriptor()
                                              .get_transforms()[number<0>{}]
                                              .get_upper_lengths(),
                                          tls_res_params::data_type,
                                          tls_res_params::is_filter);

        const auto bottom_tensor_block_coord =
            make_tensor_coordinate(bottom_tensor_view.get_tensor_descriptor(), block_window_origin);

        const auto warp_id = WaspHelper::get_warp_id();

        // calculate lds warp offset per issue
        static_for<0, NumTlsIssue, 1>{}([&](auto i_issue) {
            tls_issue_warp_lds_offset_(i_issue) =
                warp_id * lds_elem_offset_per_warp + i_issue * lds_elem_offset_per_issue;
        });

        if(tls_res_params::is_filter)
        {
            auto warp_coord_k =
                bottom_tensor_block_coord.get_index()[number<0>{}] + warp_id * MNPerTls;

            tls_sampler_ = make_tls_sampler(warp_coord_k,
                                            0,
                                            0,
                                            tls_sampler_params::channels_per_pixel,
                                            tls_sampler_params::pixel_per_column,
                                            tls_sampler_params::element_stride,
                                            tls_sampler_params::dilation_rate,
                                            tls_sampler_params::padding_num,
                                            tls_sampler_params::filter_size,
                                            number<0>{},
                                            tls_sampler_params::interleave,
                                            tls_sampler_params::oob_fill);
        }
        else
        {
            // TODO: consider interleaved layout?
            const auto N = output_tensor_view.get_tensor_descriptor()
                               .get_transforms()[number<0>{}]
                               .get_upper_lengths()[number<0>{}];
            const auto P = output_tensor_view.get_tensor_descriptor()
                               .get_transforms()[number<0>{}]
                               .get_upper_lengths()[number<1>{}];
            const auto Q = output_tensor_view.get_tensor_descriptor()
                               .get_transforms()[number<0>{}]
                               .get_upper_lengths()[number<2>{}];

            const auto to_tensor_coord_adaptor = make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(N, P, Q))),
                make_tuple(sequence<0, 1, 2>{}),
                make_tuple(sequence<0>{}));

            auto warp_coord_npq =
                bottom_tensor_block_coord.get_index()[number<0>{}] + warp_id * MNPerTls;

            const auto tensor_coord =
                to_tensor_coord_adaptor.calculate_bottom_index(make_multi_index(warp_coord_npq));

            tls_sampler_ = make_tls_sampler(tensor_coord[0],
                                            tensor_coord[1],
                                            tensor_coord[2],
                                            tls_sampler_params::channels_per_pixel,
                                            tls_sampler_params::pixel_per_column,
                                            tls_sampler_params::element_stride,
                                            tls_sampler_params::dilation_rate,
                                            tls_sampler_params::padding_num,
                                            tls_sampler_params::filter_size,
                                            number<0>{},
                                            tls_sampler_params::interleave,
                                            tls_sampler_params::oob_fill);
        }
    }

    template <typename T, index_t samp_idx, index_t samp_num = 0, bool use_m0 = false>
    CK_TILE_DEVICE void async_tls_load_asm(CK_TILE_LDS_ADDR T* smem,
                                           number<samp_idx>,
                                           number<samp_num>      = {},
                                           bool_constant<use_m0> = {})
    {
        static_assert(samp_idx < FilterUnroll, "Invalid samp_idx for tls instruction");

        static_for<0, NumTlsIssue, 1>{}([&](auto i_issue) {
            // immed offset calculation
            constexpr uint16_t warp_immed_offset = warp_offset_per_issue * i_issue;
            constexpr uint8_t warp_immed_offset0 = warp_immed_offset & 0xff;
            constexpr uint8_t warp_immed_offset1 = warp_immed_offset >> 8;

            hcu_async_tls_asm(smem + tls_issue_warp_lds_offset_[i_issue],
                              tls_resource_,
                              tls_sampler_,
                              number<warp_immed_offset0>{},
                              number<warp_immed_offset1>{},
                              number<samp_num>{},
                              number<samp_idx>{},
                              bool_constant<use_m0>{});
        });
    }

    template <index_t next_samp_idx>
    CK_TILE_DEVICE void test_and_advance(number<next_samp_idx>)
    {
        static_assert(next_samp_idx <= FilterUnroll, "Invalid next_samp_idx");

        // advance coffset to next 64B
        if constexpr(number<next_samp_idx>{} % FilterUnroll == 0)
        {
            const uint32_t coffset = ((tls_sampler_.y & 0x7ff0000) >> 16) + 1;

            // update coffset in tls_sampler
            tls_sampler_.y = (tls_sampler_.y & 0xf800ffff) | (coffset << 16);
        }
    }

    int32x4_t tls_resource_;
    int32x4_t tls_sampler_;

    array<index_t, NumTlsIssue> tls_issue_warp_lds_offset_;
};

template <ConvFwdSpecEnum Spec,
          typename BottomTensorView_,
          typename OutputTensorView_,
          typename BlockWindowLengths_,
          typename TlsWindowLenghts_,
          typename Layout_,
          typename WaspHelper_>
CK_TILE_DEVICE constexpr auto make_tile_window_conv_fwd_tls(
    const BottomTensorView_& bottom_tensor_view,
    const OutputTensorView_& output_tensor_view,
    const BlockWindowLengths_&,
    const TlsWindowLenghts_&,
    const multi_index<BottomTensorView_::get_num_of_dimension()>& window_origin,
    const Layout_&,
    const WaspHelper_&)
{
    return tile_window_conv2d_fwd_tls<remove_cvref_t<BottomTensorView_>,
                                      remove_cvref_t<OutputTensorView_>,
                                      remove_cvref_t<BlockWindowLengths_>,
                                      remove_cvref_t<TlsWindowLenghts_>,
                                      Spec,
                                      remove_cvref_t<Layout_>,
                                      remove_cvref_t<WaspHelper_>>{
        bottom_tensor_view, output_tensor_view, window_origin};
}

} // namespace ck_tile
