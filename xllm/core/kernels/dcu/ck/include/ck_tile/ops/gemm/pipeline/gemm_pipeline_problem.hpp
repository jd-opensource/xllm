// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          typename ComputeDataType_ = ADataType_,
          bool FixedVectorSize_     = false,
          index_t VectorSizeA_      = 1,
          index_t VectorSizeB_      = 1,
          typename AScaleDataType_  = ck_tile::null_type,
          typename BScaleDataType_  = ck_tile::null_type>
struct GemmPipelineProblemBase
{
    using Traits = remove_cvref_t<Traits_>;

    using ADataType       = remove_cvref_t<ADataType_>;
    using BDataType       = remove_cvref_t<BDataType_>;
    using CDataType       = remove_cvref_t<CDataType_>; // actually AccDataType
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using AScaleDataType  = remove_cvref_t<AScaleDataType_>;
    using BScaleDataType  = remove_cvref_t<BScaleDataType_>;

    static constexpr bool FixedVectorSize = FixedVectorSize_;

    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    using ALayout = remove_cvref_t<typename Traits::ALayout>;
    using BLayout = remove_cvref_t<typename Traits::BLayout>;
    using CLayout = remove_cvref_t<typename Traits::CLayout>;

    static constexpr bool TransposeC            = Traits::TransposeC;
    static constexpr index_t NumWaveGroups      = Traits::NumWaveGroups;
    static constexpr bool UseStructuredSparsity = Traits::UseStructuredSparsity;
    static constexpr bool UseABScale            = Traits::UseABScale;

    static constexpr index_t kBlockSize = BlockGemmShape::NumWarps * get_warp_size();

    static constexpr bool kPadM = Traits::kPadM;
    static constexpr bool kPadN = Traits::kPadN;
    static constexpr bool kPadK = Traits::kPadK;

    static constexpr bool DoubleSmemBuffer  = Traits::DoubleSmemBuffer;
    static constexpr auto Scheduler         = GemmPipelineScheduler::Default;
    static constexpr index_t VectorLoadSize = Traits::_VectorSize;              //TileGemmTraits::_VectorSize = 16

    // In the base situation, the Preshuffle setting should be false.
    static constexpr bool Preshuffle = false;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "gemm_problem", 
                      concat('x', VectorLoadSize, kBlockSize),
                      concat('x', kPadM, kPadN, kPadK),
                      Scheduler);
        // clang-format on
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentA()
    {
        constexpr index_t PackedSize =
            ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;
        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            constexpr index_t pixels_per_thread =
                BlockGemmShape::kM * BlockGemmShape::kK / kBlockSize;
            return pixels_per_thread < PackedSize * VectorLoadSize / sizeof(ADataType)
                       ? pixels_per_thread
                       : PackedSize * VectorLoadSize / sizeof(ADataType);
        }
        else
        {
            return VectorLoadSize / sizeof(ADataType);          // 8 for half.
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentB()
    {
        constexpr index_t PackedSize =
            ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;
        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t pixels_per_thread =
                BlockGemmShape::kN * BlockGemmShape::kK / kBlockSize;
            return pixels_per_thread < PackedSize * VectorLoadSize / sizeof(BDataType)
                       ? pixels_per_thread
                       : PackedSize * VectorLoadSize / sizeof(BDataType);
        }
        else
        {
            return PackedSize * VectorLoadSize / sizeof(BDataType);          // 8 for half.  16 for int8.  32 for int4.
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentC()
    {
        if constexpr(std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
        {
            constexpr index_t N1 = kBlockSize / get_warp_size();
            constexpr index_t N2 = std::min(BlockGemmShape::kN / N1, get_warp_size());
            constexpr index_t M0 = get_warp_size() / N2;
            constexpr index_t M1 = BlockGemmShape::kM / M0;

            return std::min(M1, static_cast<index_t>(VectorLoadSize / sizeof(CDataType)));
        }
        else
        {
            constexpr index_t M1 = kBlockSize / get_warp_size();
            constexpr index_t M2 = std::min(BlockGemmShape::kM / M1, get_warp_size());
            constexpr index_t N0 = get_warp_size() / M2;
            constexpr index_t N1 = BlockGemmShape::kN / N0;

            return std::min(N1, static_cast<index_t>(VectorLoadSize / sizeof(CDataType)));
        }
    }

    static constexpr index_t VectorSizeA = []() {
        if constexpr(FixedVectorSize)
        {
            return VectorSizeA_;
        }
        else if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
        {
            return kPadK ? 1 : GetAlignmentA();
        }
        else
        {
            return kPadM ? 1 : GetAlignmentA();
        }
    }();

    static constexpr index_t VectorSizeB = []() {
        if constexpr(FixedVectorSize)
        {
            return VectorSizeB_;
        }
        else if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>)
        {
            return kPadN ? 1 : GetAlignmentB();
        }
        else
        {
            return kPadK ? 1 : GetAlignmentB();
        }
    }();
    static constexpr index_t VectorSizeC = []() {
        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            return kPadN ? 1 : GetAlignmentC();
        }
        else
        {
            return kPadM ? 1 : GetAlignmentC();
        }
    }();
};

// Alias for GemmPipelineProblem
template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          typename ComputeDataType_ = ADataType_,
          bool FixedVectorSize_     = false,
          index_t VectorSizeA_      = 1,
          index_t VectorSizeB_      = 1,
          typename AScaleDataType_  = ck_tile::null_type,
          typename BScaleDataType_  = ck_tile::null_type>
using GemmPipelineProblem = GemmPipelineProblemBase<ADataType_,
                                                    BDataType_,
                                                    CDataType_,
                                                    BlockGemmShape_,
                                                    Traits_,
                                                    ComputeDataType_,
                                                    FixedVectorSize_,
                                                    VectorSizeA_,
                                                    VectorSizeB_,
                                                    AScaleDataType_,
                                                    BScaleDataType_>;

template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename TileGemmTraits_,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                 = true,
          TailNumber TailNum_              = TailNumber::Full,
          typename AScaleDataType_         = ck_tile::null_type,
          typename BScaleDataType_         = ck_tile::null_type>
struct UniversalGemmPipelineProblem : public GemmPipelineProblemBase<ADataType_,
                                                                     BDataType_,
                                                                     CDataType_,
                                                                     BlockGemmShape_,
                                                                     TileGemmTraits_,
                                                                     ADataType_,
                                                                     false,
                                                                     1,
                                                                     1,
                                                                     AScaleDataType_,
                                                                     BScaleDataType_>
{
    static constexpr auto Scheduler  = Scheduler_;
    static constexpr auto HasHotLoop = HasHotLoop_;
    static constexpr auto TailNum    = TailNum_;
};

//MMac GemmPipelineProblem
template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename TileGemmTraits_,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                 = true,
          TailNumber TailNum_              = TailNumber::Full,
          typename AScaleDataType_         = ck_tile::null_type,
          typename BScaleDataType_         = ck_tile::null_type>
struct MmacUniversalGemmPipelineProblem : public GemmPipelineProblemBase<ADataType_,
                                                                     BDataType_,
                                                                     CDataType_,
                                                                     BlockGemmShape_,
                                                                     TileGemmTraits_,
                                                                     ADataType_,
                                                                     false,
                                                                     1,
                                                                     1,
                                                                     AScaleDataType_,
                                                                     BScaleDataType_>
{
    using GemmTraits = remove_cvref_t<TileGemmTraits_>;
    
    static constexpr auto Scheduler  = Scheduler_;
    static constexpr auto HasHotLoop = HasHotLoop_;
    static constexpr auto TailNum    = TailNum_;
    static constexpr auto kMRepeat = GemmTraits::MRepeat;
    static constexpr auto kNRepeat = GemmTraits::NRepeat;
    static constexpr auto kMInterleave = GemmTraits::MInterleave;
    static constexpr auto kNInterleave = GemmTraits::NInterleave;
};

} // namespace ck_tile
