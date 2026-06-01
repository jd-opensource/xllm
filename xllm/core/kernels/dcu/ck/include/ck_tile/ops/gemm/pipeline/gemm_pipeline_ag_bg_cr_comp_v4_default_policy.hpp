// SPDX-License-Identifier: MIT
// Copyright (c) 2025, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_new.hpp"

namespace ck_tile {
// Default policy for GemmPipelineAGmemBGmemCregComputeV4, except the block gemm method, it shares
// the same vector size implementation, SmemSize, Global memory tile distiribution as the
// UniversalGemm Pipeline Policy.
// Default policy class should not be templated, put template on
// member functions instead.
struct GemmPipelineAgBgCrCompV4DefaultPolicy
    : public UniversalGemmBasePolicy<GemmPipelineAgBgCrCompV4DefaultPolicy>
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        // using AccDataType     = float;
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;    // <2, 2, 1>
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;      // <32, 32, 16>

        using WarpGemm = WarpGemmMmacDispatcher<typename Problem::ADataType,  // half
                                                typename Problem::BDataType,  // half
                                                typename Problem::CDataType,  // AccDataType     float
                                                WarpTile::at(I0),    // 32
                                                WarpTile::at(I1),    // 32
                                                WarpTile::at(I2),    // 16
                                                Problem::TransposeC, // false
                                                WarpTile::at(I0) / 16, // Mrepeat
                                                WarpTile::at(I1) / 16, // Nrepeat
                                                1, // Minterleave
                                                1, // Ninterleave
                                                false, //SwizzleA
                                                Problem::UseABScale>;   //UseABScale
        using BlockGemmPolicy = BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::ADataType,
                                                                    typename Problem::BDataType,
                                                                    typename Problem::CDataType,
                                                                    BlockWarps,
                                                                    WarpGemm>;

        return BlockGemmARegBRegCRegV1<Problem, BlockGemmPolicy>{};
    }
};
} // namespace ck_tile
