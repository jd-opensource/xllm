// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/reduce.hpp"

namespace ck_tile {

template <bool kPadM_,
          bool kPadN_,
          bool UseSmoothInputScale_,
          bool UseRawStore_ = true,
          bool UseMax3_     = false>
struct DynamicQuantEpilogueTraits
{
    static constexpr bool kPadM               = kPadM_;
    static constexpr bool kPadN               = kPadN_;
    static constexpr bool UseSmoothInputScale = UseSmoothInputScale_;
    static constexpr bool UseRawStore         = UseRawStore_;
    static constexpr bool UseMax3             = UseMax3_;
};

// this epilogue just store out a M*N matrix, row major
template <typename AccDataType_,
          typename SmoothScaleDataType_,
          typename YScaleDataType_,
          typename ODataType_,
          typename BlockShape_,
          typename Traits_>
struct DynamicQuantEpilogueProblem
{
    using AccDataType         = remove_cvref_t<AccDataType_>;
    using SmoothScaleDataType = remove_cvref_t<SmoothScaleDataType_>;
    using YScaleDataType      = remove_cvref_t<YScaleDataType_>;
    using ODataType           = remove_cvref_t<ODataType_>;
    using BlockShape          = remove_cvref_t<BlockShape_>; // can consum generic 2d shape
    using Traits              = remove_cvref_t<Traits_>;
};

// TODO: we should put descriptor creation function into policy
template <typename Problem_, typename Policy_ = void>
struct DynamicQuantEpilogue
{
    using Problem                     = remove_cvref_t<Problem_>;
    using AccDataType                 = remove_cvref_t<typename Problem::AccDataType>;
    using SmoothScaleDataType         = remove_cvref_t<typename Problem::SmoothScaleDataType>;
    using YScaleDataType              = remove_cvref_t<typename Problem::YScaleDataType>;
    using ODataType                   = remove_cvref_t<typename Problem::ODataType>;
    using BlockShape                  = remove_cvref_t<typename Problem::BlockShape>;
    static constexpr bool kPadM       = Problem::Traits::kPadM;
    static constexpr bool kPadN       = Problem::Traits::kPadN;
    static constexpr bool UseRawStore = Problem::Traits::UseRawStore;
    static constexpr bool UseMax3     = Problem::Traits::UseMax3;

    CK_TILE_HOST_DEVICE static constexpr auto GetBlockReduce2d()
    {
        using P_ = BlockReduce2dProblem<AccDataType, AccDataType, BlockShape>;
        return BlockReduce2d<P_>{};
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetBlockReduce2dSync()
    {
        using P_ = BlockReduce2dProblem<AccDataType, AccDataType, BlockShape>;
        return BlockReduce2dSync<P_>{};
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetBlockReduce2dCrossWarpSync()
    {
        using P_ = BlockReduce2dProblem<AccDataType, AccDataType, BlockShape>;
        return BlockReduce2dCrossWarpSync<P_>{};
    }

    CK_TILE_DEVICE static constexpr auto MakeSmoothInputScaleTileDistribution()
    {
        using S = BlockShape;
#if 0
        // don't remove this
        // Note that if we set encoding purposely like this, you will result in compile fail
        // TODO: sm_scale create local-scratch to accept arbitrary acc input (with same length)
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<S::Repeat_M, S::WarpPerBlock_M, S::ThreadPerWarp_M>,
                tuple<sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::Vector_N>>,
                tuple<sequence<0, 1>, sequence<0, 1>>,
                tuple<sequence<1, 1>, sequence<2, 2>>,
                sequence<0, 1, 1>,
                sequence<0, 0, 3>>{});
#else
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<S::WarpPerBlock_M, S::ThreadPerWarp_M>,
                tuple<sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::Vector_N>>,
                tuple<sequence<0, 1>, sequence<0, 1>>,
                tuple<sequence<0, 1>, sequence<1, 2>>,
                sequence<1, 1>,
                sequence<0, 3>>{});
#endif
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        auto reduce_crosswarp_sync = GetBlockReduce2dCrossWarpSync();
        return reduce_crosswarp_sync.GetSmemSize();
    }

    template <typename ODramWindowTmp, typename YScaleWindow, typename OAccTile>
    CK_TILE_DEVICE auto Impl(ODramWindowTmp& o_dram_window_tmp,
                             YScaleWindow& y_scale_window,
                             const OAccTile& o_acc_tile,
                             void* smem)
    {
        auto reduce                = GetBlockReduce2d();
        auto reduce_sync           = GetBlockReduce2dSync();
        auto reduce_crosswarp_sync = GetBlockReduce2dCrossWarpSync();

        auto o_acc_tmp = o_acc_tile;

        const auto f_absmax = [](auto acc_, auto v_0_) { return max(acc_, abs(v_0_)); };

        auto row_absmax = [&]() {
            constexpr auto y_size_per_row =
                OAccTile{}.get_tile_distribution().get_ys_to_d_descriptor().get_lengths().at(
                    number<1>{});
            if constexpr(UseMax3 && std::is_same_v<AccDataType, float> && y_size_per_row % 2 == 0)
            {
                // fast max3+abs implementation
                const auto f_max3 = [](auto acc_, auto v_0_, auto v_1_) {
                    float rtn;
                    asm volatile("v_max3_f32 %0, %1, abs(%2), abs(%3)"
                                 : "=v"(rtn)
                                 : "v"(acc_), "v"(v_0_), "v"(v_1_));
                    return rtn;
                };
                return reduce(o_acc_tmp, type_convert<AccDataType>(0), f_max3, sequence<1, 2>{});
            }
            else
            {
                return reduce(o_acc_tmp, type_convert<AccDataType>(0), f_absmax);
            }
        }();
        reduce_sync(row_absmax, f_absmax);
        reduce_crosswarp_sync(row_absmax, smem, f_absmax);

        // here y_scale is Acc TYpe, need convert to YScale type later
        auto y_scale = tile_elementwise_in(
            [&](const auto& v_) {
                return v_ / type_convert<AccDataType>(numeric<ODataType>::max());
            },
            row_absmax);

        store_tile(y_scale_window, cast_tile<YScaleDataType>(y_scale));

        sweep_tile(o_acc_tmp, [&](auto idx) {
            constexpr auto row_id = make_tuple(idx[number<0>{}]);
            o_acc_tmp(idx)        = o_acc_tmp[idx] / y_scale(row_id);
        });

        // TODO: this is ugly
        if constexpr(UseRawStore && (kPadM || kPadN))
        {
            store_tile_raw(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tmp));
            buffer_store_fence();
        }
        else
        {
            store_tile(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tmp));
        }
    }

    // TODO: this function assume store out vector size is the same as OAccTile last dimension size
    //       how do we fix this ?

    // Smooth Dynamic Quant
    template <typename ODramWindowTmp,
              typename SmoothScaleWindow,
              typename YScaleWindow,
              typename OAccTile>
    CK_TILE_DEVICE auto operator()(ODramWindowTmp& o_dram_window_tmp,
                                   const SmoothScaleWindow& sm_scale_window_,
                                   YScaleWindow& y_scale_window,
                                   const OAccTile& o_acc_tile,
                                   void* smem)
    {
        const auto sm_scale_window =
            make_tile_window(sm_scale_window_, MakeSmoothInputScaleTileDistribution());

        auto sm_scale = load_tile(sm_scale_window);

        auto o_acc_tmp = o_acc_tile;

        sweep_tile(o_acc_tmp, [&](auto idx) {
            constexpr auto j_idx = make_tuple(idx[number<1>{}]);
            const auto xs_       = type_convert<AccDataType>(sm_scale[j_idx]);
            o_acc_tmp(idx)       = o_acc_tmp(idx) * xs_;
        });

        Impl(o_dram_window_tmp, y_scale_window, o_acc_tmp, smem);
    }

    // Dynamic Quant
    template <typename ODramWindowTmp, typename YScaleWindow, typename OAccTile>
    CK_TILE_DEVICE auto operator()(ODramWindowTmp& o_dram_window_tmp,
                                   YScaleWindow& y_scale_window,
                                   const OAccTile& o_acc_tile,
                                   void* smem)
    {
        Impl(o_dram_window_tmp, y_scale_window, o_acc_tile, smem);
    }
};
} // namespace ck_tile
