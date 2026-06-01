// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.

#pragma once

#include "default_2d_epilogue.hpp"
#include "dynamic_quant_epilogue.hpp"

namespace ck_tile {

// User can reuse DynamicQuantEpilogueTraits with this epilogue
template <bool kPadM_,
          bool kPadN_,
          bool UseSmoothInputScale_,
          bool UseRawStore_ = true,
          bool UseMax3_     = false>
using Default2DAndDynamicQuantEpilogueTraits =
    DynamicQuantEpilogueTraits<kPadM_, kPadN_, UseSmoothInputScale_, UseRawStore_, UseMax3_>;

// This epilogue just store out a M*N matrix, row major
template <typename AccDataType_,
          typename SmoothScaleDataType_,
          typename YScaleDataType_,
          typename ODataType_,
          typename UnquantYDataType_,
          typename BlockShape_,
          typename Traits_>
struct Default2DAndDynamicQuantEpilogueProblem
{
    using AccDataType         = remove_cvref_t<AccDataType_>;
    using SmoothScaleDataType = remove_cvref_t<SmoothScaleDataType_>;
    using YScaleDataType      = remove_cvref_t<YScaleDataType_>;
    using ODataType           = remove_cvref_t<ODataType_>;
    using UnquantYDataType    = remove_cvref_t<UnquantYDataType_>;
    using BlockShape          = remove_cvref_t<BlockShape_>; // can consum generic 2d shape
    using Traits              = remove_cvref_t<Traits_>;
};

template <typename Problem_, typename Policy_ = void>
struct Default2DAndDynamicQuantEpilogue
{
    using Problem          = remove_cvref_t<Problem_>;
    using AccDataType      = remove_cvref_t<typename Problem::AccDataType>;
    using UnquantYDataType = remove_cvref_t<typename Problem::UnquantYDataType>;

    static constexpr bool kPadM       = Problem::Traits::kPadM;
    static constexpr bool kPadN       = Problem::Traits::kPadN;
    static constexpr bool UseRawStore = Problem::Traits::UseRawStore;

    using Default2DProblem =
        Default2DEpilogueProblem<AccDataType, UnquantYDataType, kPadM, kPadN, UseRawStore>;
    using Default2D    = Default2DEpilogue<Default2DProblem>;
    using DynamicQuant = DynamicQuantEpilogue<Problem>;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(Default2D::GetSmemSize(), DynamicQuant::GetSmemSize());
    }

    template <typename ODramWindowTmpD,
              typename ODramWindowTmpQ,
              typename SmoothScaleWindow,
              typename YScaleWindow,
              typename OAccTile>
    CK_TILE_DEVICE auto operator()(ODramWindowTmpD& o_direct_dram_window_tmp,
                                   ODramWindowTmpQ& o_quant_dram_window_tmp,
                                   const SmoothScaleWindow& sm_scale_window_,
                                   YScaleWindow& y_scale_window,
                                   const OAccTile& o_acc_tile,
                                   void* smem)
    {
        Default2D{}(o_direct_dram_window_tmp, o_acc_tile, smem);
        DynamicQuant{}(o_quant_dram_window_tmp, sm_scale_window_, y_scale_window, o_acc_tile, smem);
    }

    template <typename ODramWindowTmpD,
              typename ODramWindowTmpQ,
              typename YScaleWindow,
              typename OAccTile>
    CK_TILE_DEVICE auto operator()(ODramWindowTmpD& o_direct_dram_window_tmp,
                                   ODramWindowTmpQ& o_quant_dram_window_tmp,
                                   YScaleWindow& y_scale_window,
                                   const OAccTile& o_acc_tile,
                                   void* smem)
    {
        Default2D{}(o_direct_dram_window_tmp, o_acc_tile, smem);
        DynamicQuant{}(o_quant_dram_window_tmp, y_scale_window, o_acc_tile, smem);
    }
};

} // namespace ck_tile
