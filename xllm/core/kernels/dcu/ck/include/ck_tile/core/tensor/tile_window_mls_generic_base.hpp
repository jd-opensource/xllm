// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/algorithm/space_filling_curve.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/tensor/tile_window_mls_traits_gfx938.hpp"
#include "ck_tile/core/tensor/tile_window_mls_traits_gfx946.hpp"
#include "ck_tile/host/device_prop.hpp"

namespace ck_tile {

/*
 * tile_window_mls_generic_base is a generic base class for implementing specific mls load task
 * this class assume each warp issue mls on matrix MN and K dim, for each mls issue on MN dim,
 * an independent mls resource will be created, since mls filter on MN dim is persistant.
 * and each mls issue on K dim is presented by moffset of mls instruction.
 * lds offset for each mls issue is computed only once.
 * call set_window_origin(*) will only recompute mls resource for each issue.
 * call get_tile_lds_desc() to get lds descriptor for warp offset calculation
 * call get_ds_format_inst() to get compatible ds_read_matrix_* instruction for given window
 */
template <hcu_target_enum HcuArch_,
          typename BottomTensorView_,
          typename TileShape_,
          typename WaspHelper_,
          index_t Alt_,
          bool Trans_>
struct tile_window_mls_generic_base
{
    using BottomTensorView = remove_cvref_t<BottomTensorView_>;
    using TileShape        = remove_cvref_t<TileShape_>;
    using WaspHelper       = remove_cvref_t<WaspHelper_>;

    static constexpr auto Alt     = Alt_;
    static constexpr auto Trans   = Trans_;
    static constexpr auto HcuArch = HcuArch_;

    using BottomTensorDesc = remove_cvref_t<typename BottomTensorView::TensorDesc>;

    static constexpr index_t NDimBottomTensor = BottomTensorDesc::get_num_of_dimension();

    using BottomTensorIndex = array<index_t, NDimBottomTensor>;
    using BottomTensorCoord =
        decltype(make_tensor_coordinate(BottomTensorDesc{}, BottomTensorIndex{}));

    static_assert(NDimBottomTensor, "Must be a matrix desc");

    using DataType = remove_cvref_t<typename BottomTensorView::DataType>;

    using TileTraits = tile_window_mls_traits<TileShape, sizeof(DataType), Alt, true, HcuArch>;
    using Detail     = typename TileTraits::Detail;
    using MlsAtom    = typename Detail::MlsAtom;

    // tile info
    static constexpr auto TileShapeMN = TileShape::at(number<0>{});
    static constexpr auto TileShapeK  = TileShape::at(number<1>{});

    static constexpr auto WarpCluster          = Detail::WarpCluster;
    static constexpr auto TileLoadWarpPerIssue = Detail::TileLoadWarpPerIssue;
    static constexpr auto TileLoadWGPerIssue   = Detail::TileLoadWGPerIssue;

    static constexpr auto TileLoadWarpPerIssueMN = TileLoadWarpPerIssue.at(number<0>{});
    static constexpr auto TileLoadWarpPerIssueK  = TileLoadWarpPerIssue.at(number<1>{});

    static constexpr auto TileLoadWGPerIssueMN = TileLoadWGPerIssue.at(number<0>{});
    static constexpr auto TileLoadWGPerIssueK  = TileLoadWGPerIssue.at(number<1>{});

    // warp access info
    using SFC_WarpAccess = typename Detail::SFC_WarpAccess;

    static constexpr auto NumWarpAccess   = SFC_WarpAccess::get_num_of_access();
    static constexpr auto NumWarpAccessMN = SFC_WarpAccess::access_lengths.at(number<0>{});
    static constexpr auto NumWarpAccessK  = SFC_WarpAccess::access_lengths.at(number<1>{});

    CK_TILE_DEVICE static constexpr auto get_num_of_access() { return NumWarpAccess; }

    CK_TILE_DEVICE static constexpr auto get_tile_lds_desc() { return Detail::make_lds_desc(); }

    CK_TILE_DEVICE
    tile_window_mls_generic_base(const BottomTensorView& bottom_tensor_view,
                                 const index_t mls_stride,
                                 const index_t bottom_tensor_mn_pad_val,
                                 const index_t bottom_tensor_k_pad_val)
        : bottom_tensor_view_(bottom_tensor_view),
          mls_stride_(mls_stride),
          bottom_tensor_mn_pad_val_(bottom_tensor_mn_pad_val),
          bottom_tensor_k_pad_val_(bottom_tensor_k_pad_val)
    {
        init();
    }

    CK_TILE_DEVICE auto get_warp_cluster_idx()
    {
        constexpr auto warp_cluster_to_id_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(WarpCluster)),
            make_tuple(typename arithmetic_sequence_gen<0, WarpCluster.size(), 1>::type{}),
            make_tuple(sequence<0>{}));

        return warp_cluster_to_id_adaptor.calculate_bottom_index(
            make_multi_index(WaspHelper::get_warp_id()));
    }

    CK_TILE_DEVICE void init()
    {
        constexpr auto tile_lds_desc = get_tile_lds_desc();

        const auto warp_cluster_idx = get_warp_cluster_idx();

        static_for<0, NumWarpAccess, 1>{}([&](auto i) {
            constexpr auto access_idx = SFC_WarpAccess::get_index(i);

            const auto tile_warp_coord = generate_tuple(
                [&](auto ii) {
                    return warp_cluster_idx[ii] * TileLoadWarpPerIssue.at(ii) +
                           access_idx[ii] * TileLoadWGPerIssue.at(ii);
                },
                number<2>{});

            mls_lds_offset_(i) = __builtin_amdgcn_readfirstlane(
                tile_lds_desc.calculate_offset(to_multi_index(tile_warp_coord)));
        });
    }

    CK_TILE_DEVICE void init(const BottomTensorIndex& block_window_origin)
    {
        const auto warp_cluster_idx = get_warp_cluster_idx();

        // get padded length of matrix mn by top dim length
        const auto mn_length = bottom_tensor_view_.get_tensor_descriptor().get_length(number<0>{});
        const auto mn_length_raw = mn_length - bottom_tensor_mn_pad_val_;

        //  each issue on mn dim refer to indepedent mls resource
        static_for<0, NumWarpAccessMN, 1>{}([&](auto i) {
            constexpr auto access_idx = make_tuple(i, number<0>{});

            // tile warp coordination of MN
            const auto tile_warp_coord = generate_tuple(
                [&](auto ii) {
                    return warp_cluster_idx[ii] * TileLoadWarpPerIssue.at(ii) +
                           access_idx[ii] * TileLoadWGPerIssue.at(ii);
                },
                number<2>{});

            const auto bottom_tensor_warp_coord =
                make_tensor_coordinate(bottom_tensor_view_.get_tensor_descriptor(),
                                       block_window_origin + to_multi_index(tile_warp_coord));

            const index_t bottom_tensor_warp_coord_mn =
                bottom_tensor_warp_coord.get_index()[number<0>{}];

            const index_t mls_mn_filter =
                bottom_tensor_warp_coord_mn + TileLoadWarpPerIssueMN > mn_length_raw
                    ? __builtin_amdgcn_readfirstlane(ck_tile::min(
                          TileLoadWarpPerIssueMN,
                          bottom_tensor_warp_coord_mn + TileLoadWarpPerIssueMN - mn_length_raw))
                    : 0;

            constexpr auto mfmt = detail::mfmt_traits<Alt>::value;

            if constexpr(Trans)
            {
                mls_res_(i) = make_mls_resource(bottom_tensor_view_.get_buffer_view().p_data_ +
                                                    bottom_tensor_warp_coord.get_offset(),
                                                mls_stride_,
                                                0,
                                                mls_mn_filter,
                                                mfmt);
            }
            else
            {
                mls_res_(i) = make_mls_resource(bottom_tensor_view_.get_buffer_view().p_data_ +
                                                    bottom_tensor_warp_coord.get_offset(),
                                                mls_stride_,
                                                mls_mn_filter,
                                                0,
                                                mfmt);
            }
        });

        // save mls filter on k dim
        static_for<0, NumWarpAccessK, 1>{}([&](auto i) {
            constexpr auto access_idx = make_tuple(number<0>{}, i);

            const auto tile_warp_coord = generate_tuple(
                [&](auto ii) {
                    return warp_cluster_idx[ii] * TileLoadWarpPerIssue.at(ii) +
                           access_idx[ii] * TileLoadWGPerIssue.at(ii);
                },
                number<2>{});
            const auto tile_warp_coord_k = tile_warp_coord[number<1>{}];

            mls_k_filter_(i) =
                tile_warp_coord_k + TileLoadWarpPerIssueK > TileShapeK - bottom_tensor_k_pad_val_
                    ? __builtin_amdgcn_readfirstlane(
                          ck_tile::min(TileLoadWarpPerIssueK,
                                       tile_warp_coord_k + TileLoadWarpPerIssueK -
                                           (TileShapeK - bottom_tensor_k_pad_val_)))
                    : 0;
        });
    }

    CK_TILE_DEVICE void set_window_origin(const BottomTensorIndex& block_window_origin)
    {
        // update mls resource
        init(block_window_origin);
    }

    template <typename T, bool bps = false, bool last_load = false>
    CK_TILE_DEVICE void
    async_mls_load_asm(CK_TILE_LDS_ADDR T* smem, bool_constant<bps> = {}, bool_constant<last_load> = {})
    {
        static_for<0, NumWarpAccess, 1>{}([&](auto i) {
            constexpr auto access_idx    = SFC_WarpAccess::get_index(i);
            constexpr auto access_idx_mn = access_idx[number<0>{}];
            constexpr auto access_idx_k  = access_idx[number<1>{}];

            // moffset is always on matrix k dim
            constexpr auto moffset = number<access_idx_k * TileLoadWGPerIssue.at(number<1>{})>{};

            if constexpr(Trans)
            {
                // apply mls filter on k dim for last load
                if constexpr(last_load && (access_idx_mn == 0))
                {
                    mls_res_(access_idx_mn).w |= static_cast<uint32_t>(mls_k_filter_[access_idx_k]);
                }
            }
            else
            {
                // apply mls filter on k dim for last load
                if constexpr(last_load)
                {
                    mls_res_(access_idx_mn).w |=
                        (static_cast<uint32_t>(mls_k_filter_[access_idx_k]) << 8);
                }
            }

            if constexpr(HcuArch == hcu_target_enum::gfx938)
            {
                // issue mls, L1 bps is not supported
                MlsAtom::load(reinterpret_cast<uintptr_t>(smem + mls_lds_offset_[i]),
                              mls_res_[access_idx_mn],
                              moffset,
                              bool_constant<true>{});
            }
            else if constexpr(HcuArch == hcu_target_enum::gfx946)
            {
                // issue mls with L1 bps control
                MlsAtom::load(reinterpret_cast<uintptr_t>(smem + mls_lds_offset_[i]),
                              mls_res_[access_idx_mn],
                              moffset,
                              bool_constant<true>{},
                              bool_constant<bps>{});
            }
        });
    }

    CK_TILE_DEVICE void move_base(const index_t base_byte_offset)
    {
        static_for<0, NumWarpAccessMN, 1>{}(
            [&](auto i) { move_mls_addr_base(mls_res_(i), base_byte_offset); });
    }

    BottomTensorView bottom_tensor_view_;
    array<int32x4_t, NumWarpAccessMN> mls_res_;
    array<uint8_t, NumWarpAccessK> mls_k_filter_;

    index_t mls_stride_;
    index_t bottom_tensor_mn_pad_val_;
    index_t bottom_tensor_k_pad_val_;

    array<index_t, NumWarpAccess> mls_lds_offset_;
};

} // namespace ck_tile
