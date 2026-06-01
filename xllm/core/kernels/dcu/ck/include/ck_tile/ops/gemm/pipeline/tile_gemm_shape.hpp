// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

template <typename BlockTile_,
          typename BlockWarps_,
          typename WarpTile_,
          bool PermuteA_ = false,
          bool PermuteB_ = false>
struct TileGemmShape
{
    using BlockTile  = remove_cvref_t<BlockTile_>;
    using BlockWarps = remove_cvref_t<BlockWarps_>;
    using WarpTile   = remove_cvref_t<WarpTile_>;

    static constexpr index_t NumWarps = reduce_on_sequence(BlockWarps{}, multiplies{}, number<1>{});

    static constexpr index_t kM = BlockTile::at(number<0>{});
    static constexpr index_t kN = BlockTile::at(number<1>{});
    static constexpr index_t kK = BlockTile::at(number<2>{});

    static constexpr bool PermuteA = PermuteA_;
    static constexpr bool PermuteB = PermuteB_;

    static constexpr index_t flatNPerWarp  = BlockWarps::at(number<1>{});
    static constexpr index_t flatKPerWarp  = WarpTile::at(number<2>{}) * WarpTile::at(number<1>{});
    static constexpr index_t flatKPerBlock = flatKPerWarp * kK / WarpTile::at(number<2>{});

    static constexpr index_t KIterPerWarp = kK / WarpTile::at(number<2>{});               // 128/32=4
    static constexpr index_t KLanePerBlock = KIterPerWarp * 4; //一次block切分K方向包含的Klane数量，目前每个warpGemm klane固定为4,(T0,T16,T32,T48)

    CK_TILE_HOST static std::string GetName()
    {
        // clang-format off
        return concat('_', "tile_gemm_shape",
                      concat('x', kM, kN, kK, NumWarps),
                      concat('x', BlockWarps::at(number<0>{}), BlockWarps::at(number<1>{}), BlockWarps::at(number<2>{})),
                      concat('x', (WarpTile::at(number<0>{})), WarpTile::at(number<1>{}), WarpTile::at(number<2>{})));
        // clang-format on
    }
};

template <typename BlockTile_, 
          typename ProducerBlockWarps_, 
          typename ProducerWarpTile_,
          typename ConsumerBlockWarps_, 
          typename ConsumerWarpTile_
         >
struct TileGemmShape2Groups
{
    using BlockTile  = remove_cvref_t<BlockTile_>;
    using ProducerBlockWarps = remove_cvref_t<ProducerBlockWarps_>;
    using ProducerWarpTile   = remove_cvref_t<ProducerWarpTile_>;
    using ConsumerBlockWarps = remove_cvref_t<ConsumerBlockWarps_>;
    using ConsumerWarpTile   = remove_cvref_t<ConsumerWarpTile_>;

    static constexpr index_t NumProducerWarps = reduce_on_sequence(ProducerBlockWarps{}, multiplies{}, number<1>{});
    static constexpr index_t NumConsumerWarps = reduce_on_sequence(ConsumerBlockWarps{}, multiplies{}, number<1>{});

    static constexpr index_t kM = BlockTile::at(number<0>{});
    static constexpr index_t kN = BlockTile::at(number<1>{});
    static constexpr index_t kK = BlockTile::at(number<2>{});
};

template <typename BlockTile_, 
          typename ProducerBlockWarps_, 
          typename ProducerWarpTile_,
          typename ConsumerBlockWarps_, 
          typename ConsumerWarpTile_,
          typename Consumer2BlockWarps_,
          typename Consumer2WarpTile_
         >
struct TileGemmShape3Groups
{
    using BlockTile  = remove_cvref_t<BlockTile_>;
    using ProducerBlockWarps = remove_cvref_t<ProducerBlockWarps_>;
    using ProducerWarpTile   = remove_cvref_t<ProducerWarpTile_>;
    using ConsumerBlockWarps = remove_cvref_t<ConsumerBlockWarps_>;
    using ConsumerWarpTile   = remove_cvref_t<ConsumerWarpTile_>;
    using Consumer2BlockWarps = remove_cvref_t<Consumer2BlockWarps_>;
    using Consumer2WarpTile   = remove_cvref_t<Consumer2WarpTile_>;

    static constexpr index_t NumProducerWarps = reduce_on_sequence(ProducerBlockWarps{}, multiplies{}, number<1>{});
    static constexpr index_t NumConsumerWarps = reduce_on_sequence(ConsumerBlockWarps{}, multiplies{}, number<1>{});
    static constexpr index_t NumConsumer2Warps = reduce_on_sequence(Consumer2BlockWarps{}, multiplies{}, number<1>{});

    static constexpr index_t kM = BlockTile::at(number<0>{});
    static constexpr index_t kN = BlockTile::at(number<1>{});
    static constexpr index_t kK = BlockTile::at(number<2>{});
};


} // namespace ck_tile
