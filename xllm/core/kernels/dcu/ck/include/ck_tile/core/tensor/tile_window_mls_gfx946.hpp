// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/algorithm/space_filling_curve.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/arch/hcu_mls_traits_gfx946.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

namespace ck_tile {

/*
 * this header provides details of tile window
 * constructed by mls instruction on gfx946
 * make_lds_desc() provides logical lds descriptor
 * for warp offset calculation
 */

template <index_t Alt>
struct tile_window_mls_gfx946_256x64_trans_b16
{
    using MlsAtom   = gfx946_mls_16x64_trans_b16;
    using MlsTraits = mls_traits<MlsAtom, Alt>;

    static constexpr auto WarpCluster = sequence<4, 1>{};

    static constexpr auto TileShape            = sequence<256, 64>{};
    static constexpr auto TileLoadWarpPerIssue = MlsAtom::TileShape;
    static constexpr auto TileLoadWGPerIssue   = WarpCluster * TileLoadWarpPerIssue;

    static constexpr auto WarpMlsIssueSeq = TileShape / TileLoadWGPerIssue;

    using SFC_WarpAccess =
        space_filling_curve<decltype(WarpMlsIssueSeq), sequence<0, 1>, sequence<1, 1>, false>;

    CK_TILE_DEVICE static constexpr auto make_lds_desc()
    {
        constexpr auto tile_load_issue_mn       = number<16>{};
        constexpr auto tile_load_issue_mn_outer = tile_load_issue_mn / MlsTraits::kSlots;

        constexpr auto lds_desc_raw = make_naive_tensor_descriptor_packed(
            concat_tuple(make_tuple(tile_load_issue_mn_outer), MlsTraits::PackedShape));

        constexpr auto lds_desc_mn_k = transform_tensor_descriptor(
            lds_desc_raw,
            make_tuple(make_merge_transform(
                           make_tuple(tile_load_issue_mn_outer, MlsTraits::kSlots, MlsTraits::kMN)),
                       make_merge_transform(make_tuple(MlsTraits::kK0, MlsTraits::kK1))),
            make_tuple(sequence<0, 2, 3>{}, sequence<1, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return lds_desc_mn_k;
    }
};

template <index_t Alt>
struct tile_window_mls_gfx946_128x64_trans_b16
{
    using MlsAtom   = gfx946_mls_16x64_trans_b16;
    using MlsTraits = mls_traits<MlsAtom, Alt>;

    static constexpr auto WarpCluster = sequence<4, 1>{};

    static constexpr auto TileShape            = sequence<128, 64>{};
    static constexpr auto TileLoadWarpPerIssue = MlsAtom::TileShape;
    static constexpr auto TileLoadWGPerIssue   = WarpCluster * TileLoadWarpPerIssue;

    static constexpr auto WarpMlsIssueSeq = TileShape / TileLoadWGPerIssue;

    using SFC_WarpAccess =
        space_filling_curve<decltype(WarpMlsIssueSeq), sequence<0, 1>, sequence<1, 1>, false>;

    CK_TILE_DEVICE static constexpr auto make_lds_desc()
    {

        constexpr auto tile_load_issue_mn       = number<8>{};
        constexpr auto tile_load_issue_mn_outer = tile_load_issue_mn / MlsTraits::kSlots;

        constexpr auto lds_desc_raw = make_naive_tensor_descriptor_packed(
            concat_tuple(make_tuple(tile_load_issue_mn_outer), MlsTraits::PackedShape));

        constexpr auto lds_desc_mn_k = transform_tensor_descriptor(
            lds_desc_raw,
            make_tuple(make_merge_transform(
                           make_tuple(tile_load_issue_mn_outer, MlsTraits::kSlots, MlsTraits::kMN)),
                       make_merge_transform(make_tuple(MlsTraits::kK0, MlsTraits::kK1))),
            make_tuple(sequence<0, 2, 3>{}, sequence<1, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return lds_desc_mn_k;
    }
};

template <index_t Alt>
struct tile_window_mls_gfx946_64x64_trans_b16
{
    using MlsAtom   = gfx946_mls_16x64_trans_b16;
    using MlsTraits = mls_traits<MlsAtom, Alt>;

    static constexpr auto WarpCluster = sequence<4, 1>{};

    static constexpr auto TileShape            = sequence<64, 64>{};
    static constexpr auto TileLoadWarpPerIssue = MlsAtom::TileShape;
    static constexpr auto TileLoadWGPerIssue   = WarpCluster * TileLoadWarpPerIssue;

    static constexpr auto WarpMlsIssueSeq = TileShape / TileLoadWGPerIssue;

    using SFC_WarpAccess =
        space_filling_curve<decltype(WarpMlsIssueSeq), sequence<0, 1>, sequence<1, 1>, false>;

    CK_TILE_DEVICE static constexpr auto make_lds_desc()
    {

        constexpr auto tile_load_issue_mn       = number<4>{};
        constexpr auto tile_load_issue_mn_outer = tile_load_issue_mn / MlsTraits::kSlots;

        constexpr auto lds_desc_raw = make_naive_tensor_descriptor_packed(
            concat_tuple(make_tuple(tile_load_issue_mn_outer), MlsTraits::PackedShape));

        constexpr auto lds_desc_mn_k = transform_tensor_descriptor(
            lds_desc_raw,
            make_tuple(make_merge_transform(
                           make_tuple(tile_load_issue_mn_outer, MlsTraits::kSlots, MlsTraits::kMN)),
                       make_merge_transform(make_tuple(MlsTraits::kK0, MlsTraits::kK1))),
            make_tuple(sequence<0, 2, 3>{}, sequence<1, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return lds_desc_mn_k;
    }
};

template <index_t Alt>
struct tile_window_mls_gfx946_256x32_trans_b16
{
    using MlsAtom   = gfx946_mls_32x32_trans_b16;
    using MlsTraits = mls_traits<MlsAtom, Alt>;

    static constexpr auto WarpCluster = sequence<4, 1>{};

    static constexpr auto TileShape            = sequence<256, 32>{};
    static constexpr auto TileLoadWarpPerIssue = MlsAtom::TileShape;
    static constexpr auto TileLoadWGPerIssue   = WarpCluster * TileLoadWarpPerIssue;

    static constexpr auto WarpMlsIssueSeq = TileShape / TileLoadWGPerIssue;

    using SFC_WarpAccess =
        space_filling_curve<decltype(WarpMlsIssueSeq), sequence<0, 1>, sequence<1, 1>, false>;

    CK_TILE_DEVICE static constexpr auto make_lds_desc()
    {
        constexpr auto tile_load_issue_mn = number<8>{};

        constexpr auto lds_desc_raw = make_naive_tensor_descriptor_packed(
            concat_tuple(make_tuple(tile_load_issue_mn), MlsTraits::PackedShape));

        constexpr auto lds_desc_mn_k = transform_tensor_descriptor(
            lds_desc_raw,
            make_tuple(make_merge_transform(make_tuple(tile_load_issue_mn, MlsTraits::kMN)),
                       make_pass_through_transform(MlsTraits::kK)),
            make_tuple(sequence<0, 1>{}, sequence<2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return lds_desc_mn_k;
    }
};

template <index_t Alt>
struct tile_window_mls_gfx946_128x32_trans_b16
{
    using MlsAtom   = gfx946_mls_32x32_trans_b16;
    using MlsTraits = mls_traits<MlsAtom, Alt>;

    static constexpr auto WarpCluster = sequence<4, 1>{};

    static constexpr auto TileShape            = sequence<128, 32>{};
    static constexpr auto TileLoadWarpPerIssue = MlsAtom::TileShape;
    static constexpr auto TileLoadWGPerIssue   = WarpCluster * TileLoadWarpPerIssue;

    static constexpr auto WarpMlsIssueSeq = TileShape / TileLoadWGPerIssue;

    static constexpr auto WarpMlsIssueCnt =
        container_reduce(WarpMlsIssueSeq, multiplies{}, number<1>{});

    using SFC_WarpAccess =
        space_filling_curve<decltype(WarpMlsIssueSeq), sequence<0, 1>, sequence<1, 1>, false>;

    CK_TILE_DEVICE static constexpr auto make_lds_desc()
    {
        constexpr auto tile_load_issue_mn = number<4>{};

        constexpr auto lds_desc_raw = make_naive_tensor_descriptor_packed(
            concat_tuple(make_tuple(tile_load_issue_mn), MlsTraits::PackedShape));

        constexpr auto lds_desc_mn_k = transform_tensor_descriptor(
            lds_desc_raw,
            make_tuple(make_merge_transform(make_tuple(tile_load_issue_mn, MlsTraits::kMN)),
                       make_pass_through_transform(MlsTraits::kK)),
            make_tuple(sequence<0, 1>{}, sequence<2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return lds_desc_mn_k;
    }
};

template <index_t Alt>
struct tile_window_mls_gfx946_64x32_trans_b16
{
    using MlsAtom   = gfx946_mls_16x32_trans_b16;
    using MlsTraits = mls_traits<MlsAtom, Alt>;

    static constexpr auto WarpCluster = sequence<4, 1>{};

    static constexpr auto TileShape            = sequence<64, 32>{};
    static constexpr auto TileLoadWarpPerIssue = MlsAtom::TileShape;
    static constexpr auto TileLoadWGPerIssue   = WarpCluster * TileLoadWarpPerIssue;

    static constexpr auto WarpMlsIssueSeq = TileShape / TileLoadWGPerIssue;

    static constexpr auto WarpMlsIssueCnt =
        container_reduce(WarpMlsIssueSeq, multiplies{}, number<1>{});

    using SFC_WarpAccess =
        space_filling_curve<decltype(WarpMlsIssueSeq), sequence<0, 1>, sequence<1, 1>, false>;

    CK_TILE_DEVICE static constexpr auto make_lds_desc()
    {
        constexpr auto tile_load_issue_mn = number<4>{};

        constexpr auto lds_desc_raw = make_naive_tensor_descriptor_packed(
            concat_tuple(make_tuple(tile_load_issue_mn), MlsTraits::PackedShape));

        constexpr auto lds_desc_mn_k = transform_tensor_descriptor(
            lds_desc_raw,
            make_tuple(make_merge_transform(make_tuple(tile_load_issue_mn, MlsTraits::kMN)),
                       make_pass_through_transform(MlsTraits::kK)),
            make_tuple(sequence<0, 1>{}, sequence<2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return lds_desc_mn_k;
    }
};

// TODO: add more

} // namespace ck_tile
