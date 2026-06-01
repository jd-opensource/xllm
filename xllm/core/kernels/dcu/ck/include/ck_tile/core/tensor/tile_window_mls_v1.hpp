// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/algorithm/space_filling_curve.hpp"
#include "ck_tile/core/arch/hcu_matrix_addressing.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

namespace ck_tile {

/*
BottomTensorView_ refers to global tensor view which is always performed as input tensor
OutputTensorview_ refers to lds tensor view which is always performed as output tensor
BlockWindowLengths_ refers to block window lengths
MlsWindowLengths_ refers to 1xmls window lengths
Layout_ refers to tensor_layout::gemm::RowMajor or tensor_layout::gemm::ColumnMajor
Interleave_ refers to mmac interleave
*/
template <typename BottomTensorView_,
          typename BlockWindowLengths_,
          typename MlsWindowLengths_,
          typename Layout_,
          index_t Interleave_,
          typename WaspHelper_>
struct tile_window_mls_v1
{
    using BottomTensorView   = remove_cvref_t<BottomTensorView_>;
    using BlockWindowLengths = remove_cvref_t<BlockWindowLengths_>;
    using MlsWindowLengths   = remove_cvref_t<MlsWindowLengths_>;
    using WaspHelper         = remove_cvref_t<WaspHelper_>;
    using Layout             = remove_cvref_t<Layout_>;

    // must be gemm layout
    static_assert(std::is_same_v<Layout, tensor_layout::gemm::ColumnMajor> ||
                  std::is_same_v<Layout, tensor_layout::gemm::RowMajor>);

    using BottomTensorDesc = remove_cvref_t<typename BottomTensorView::TensorDesc>;
    using DataType         = remove_cvref_t<typename BottomTensorView::DataType>;

    static constexpr index_t NDimBottomTensor = BottomTensorDesc::get_num_of_dimension();

    static_assert(ck_tile::is_known_at_compile_time<BlockWindowLengths>::value,
                  "Wrong! BlockWindowLengths must be known at compile time");
    static_assert(ck_tile::is_known_at_compile_time<MlsWindowLengths>::value,
                  "Wrong! MlsWindowLengths must be known at compile time");
    static_assert((BlockWindowLengths::size() == NDimBottomTensor) &&
                      (MlsWindowLengths::size() == BlockWindowLengths::size()),
                  "Wrong! BlockWindowLengths must have same dimension as MlsWindowLengths");

    static constexpr auto MNPerBlock = BlockWindowLengths::at(number<0>{});
    static constexpr auto KPerBlock  = BlockWindowLengths::at(number<1>{});
    static constexpr auto MNPerMls   = MlsWindowLengths::at(number<0>{});
    static constexpr auto KPerMls    = MlsWindowLengths::at(number<1>{});
    static constexpr auto warp_num   = WaspHelper::get_warp_num();

    static constexpr auto NumMlsMNIssuePerWG = MNPerBlock / MNPerMls;
    static constexpr auto NumMlsKIssuePerWG  = KPerBlock / KPerMls;

    // force mls k issue per warp to be 1 in this version
    static constexpr auto NumMlsKIssuePerWarp  = 1;
    static constexpr auto WarpClusterK         = NumMlsKIssuePerWG / NumMlsKIssuePerWarp;
    static constexpr auto WarpClusterMN        = warp_num / WarpClusterK;
    static constexpr auto NumMlsIssueMNPerWarp = NumMlsMNIssuePerWG / WarpClusterMN;

    static_assert(warp_num % WarpClusterK == 0,
                  "Wrong! warp_num must be divisible by WarpClusterK");

    static constexpr auto NumMlsIssuePerWarp = NumMlsIssueMNPerWarp * NumMlsKIssuePerWarp;

    static constexpr auto MNOffsetPerIssue = MNPerMls * WarpClusterMN;

    // mls params
    static constexpr auto mfmt = detail::mfmt_traits<Interleave_>::value;
    static constexpr auto t =
        bool_constant<std::is_same_v<Layout, tensor_layout::gemm::RowMajor>>{};
    static constexpr auto m_length  = t ? number<KPerMls>{} : number<MNPerMls>{};
    static constexpr auto nm_length = t ? number<MNPerMls>{} : number<KPerMls>{};

    static constexpr auto lds_consec_bytes =
        get_lds_consec_bytes<DataType>(m_length, nm_length, t, mfmt);
    // FIXME: ensure warp group writes consecutive bytes to LDS per issue
    static constexpr auto lds_offset_per_issue = lds_consec_bytes * warp_num;

    using BottomTensorIndex = array<index_t, NDimBottomTensor>;

    CK_TILE_DEVICE static constexpr auto get_num_of_access() { return NumMlsIssueMNPerWarp; }

    CK_TILE_DEVICE tile_window_mls_v1(const BottomTensorView& bottom_tensor_view,
                                      const BottomTensorIndex& block_window_origin,
                                      const index_t ld_raw,
                                      const index_t k_remainder)
    {
        const auto warp_id    = WaspHelper::get_warp_id();
        const auto warp_id_k  = warp_id % WarpClusterK;
        const auto warp_id_mn = warp_id / WarpClusterK;

        // assume raw desc is in [M/N, K] layout
        const auto raw_length_mn = bottom_tensor_view.get_tensor_descriptor()
                                       .get_transforms()[number<0>{}]
                                       .get_upper_lengths()[number<0>{}];

        // compute padding length
        static_for<0, NumMlsIssueMNPerWarp, 1>{}([&](auto i_issue) {
            const auto warp_offset_mn = warp_id_mn * MNPerMls + i_issue * MNOffsetPerIssue;
            const auto warp_offset_k  = warp_id_k * KPerMls;

            const auto bottom_tensor_warp_coord = make_tensor_coordinate(
                bottom_tensor_view.get_tensor_descriptor(),
                block_window_origin + multi_index<2>{warp_offset_mn, warp_offset_k});

            const auto warp_coord_mn = bottom_tensor_warp_coord.get_index()[number<0>{}];

            // precompute k padding for last loop
            gemm_k_pad_ = warp_offset_k + KPerMls > KPerBlock - k_remainder
                              ? __builtin_amdgcn_readfirstlane(ck_tile::min(
                                    KPerMls, warp_offset_k + KPerMls - (KPerBlock - k_remainder)))
                              : 0;

            // precompute lds offset in elem per issue
            mls_issue_warp_lds_offset_(i_issue) =
                (warp_id * lds_consec_bytes + i_issue * lds_consec_bytes * warp_num) /
                sizeof(DataType);

            // precompute m/n padding of current issue
            const index_t gemm_mn_pad =
                warp_coord_mn + MNPerMls > raw_length_mn
                    ? __builtin_amdgcn_readfirstlane(
                          ck_tile::min(MNPerMls, warp_coord_mn + MNPerMls - raw_length_mn))
                    : 0;

            if constexpr(t)
            {
                mls_resource_(i_issue) =
                    make_mls_resource(bottom_tensor_view.get_buffer_view().p_data_ +
                                          bottom_tensor_warp_coord.get_offset(),
                                      ld_raw,
                                      0,
                                      gemm_mn_pad,
                                      mfmt);

                if constexpr(i_issue == 0)
                {
                    const_addr_byte_offset_ =
                        __builtin_amdgcn_readfirstlane(KPerBlock * sizeof(DataType));
                }
            }
            else
            {
                mls_resource_(i_issue) =
                    make_mls_resource(bottom_tensor_view.get_buffer_view().p_data_ +
                                          bottom_tensor_warp_coord.get_offset(),
                                      ld_raw,
                                      gemm_mn_pad,
                                      0,
                                      mfmt);

                if constexpr(i_issue == 0)
                {
                    const_addr_byte_offset_ = __builtin_amdgcn_readfirstlane(
                        KPerBlock * raw_length_mn * sizeof(DataType));
                }
            }
        });
    }

    template <typename T, bool tail_load = false, bool bps = false>
    CK_TILE_DEVICE void async_mls_load_asm(CK_TILE_LDS_ADDR T* smem,
                                           bool_constant<tail_load> = {},
                                           bool_constant<bps>       = {})
    {
        static_for<0, NumMlsIssueMNPerWarp, 1>{}([&](auto i_issue) {
            constexpr auto moffset = number<i_issue * MNOffsetPerIssue>{};

            if constexpr(t)
            {
                // apply gemm_k_pad for tail load
                if constexpr(tail_load)
                {
                    mls_resource_(i_issue).w |= gemm_k_pad_;
                }
            }
            else
            {
                // apply gemm_k_pad for tail load
                if constexpr(tail_load)
                {
                    mls_resource_(i_issue).w |= (gemm_k_pad_ << 8);
                }
            }

            hcu_async_matrix_load_asm_impl(smem + mls_issue_warp_lds_offset_[i_issue],
                                           mls_resource_[i_issue],
                                           m_length,
                                           nm_length,
                                           t,
                                           mfmt,
                                           moffset,
                                           bool_constant<false>{},
                                           bool_constant<bps>{});
        });
    }

    CK_TILE_DEVICE void advance()
    {
        static_for<0, NumMlsIssueMNPerWarp, 1>{}([&](auto i_issue) {
            move_mls_addr_base(mls_resource_(i_issue), const_addr_byte_offset_);
        });
    }

    // this version use multiple mls resource to reduce salu instruction for gemm padding
    // if sgpr resource is critical, we should implement another version with single mls resource
    array<int32x4_t, NumMlsIssueMNPerWarp> mls_resource_;
    array<index_t, NumMlsIssueMNPerWarp> mls_issue_warp_lds_offset_;
    index_t gemm_k_pad_;
    index_t const_addr_byte_offset_;
};

template <typename BottomTensorView_,
          typename BlockWindowLengths_,
          typename MlsWindowLengths_,
          typename Layout_,
          index_t Interleave_,
          typename WaspHelper_>
CK_TILE_DEVICE constexpr auto
make_tile_window_mls(const BottomTensorView_& bottom_tensor_view,
                     const BlockWindowLengths_&,
                     const MlsWindowLengths_&,
                     const multi_index<BottomTensorView_::get_num_of_dimension()>& window_origin,
                     const Layout_&,
                     number<Interleave_>,
                     const WaspHelper_&,
                     const index_t ld_raw,
                     const index_t k_remainder)
{
    return tile_window_mls_v1<remove_cvref_t<BottomTensorView_>,
                              remove_cvref_t<BlockWindowLengths_>,
                              remove_cvref_t<MlsWindowLengths_>,
                              remove_cvref_t<Layout_>,
                              Interleave_,
                              remove_cvref_t<WaspHelper_>>{
        bottom_tensor_view, window_origin, ld_raw, k_remainder};
}

} // namespace ck_tile
