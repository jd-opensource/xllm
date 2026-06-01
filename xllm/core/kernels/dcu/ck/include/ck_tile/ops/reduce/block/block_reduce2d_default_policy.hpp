// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/reduce/block/block_reduce2d_problem.hpp"
#include "ck_tile/ops/reduce/block/block_reduce2d.hpp"

namespace ck_tile {

struct BlockReduce2dDefaultPolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeXBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::Repeat_M, S::WarpPerBlock_M, S::ThreadPerWarp_M, S::Vector_M>,
                      sequence<S::Repeat_N, S::WarpPerBlock_N, S::ThreadPerWarp_N, S::Vector_N>>,
                tuple<sequence<1, 2>, sequence<1, 2>>,
                tuple<sequence<1, 1>, sequence<2, 2>>,
                sequence<1, 1, 2, 2>,
                sequence<0, 3, 0, 3>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockReduce2d()
    {
        using P_ = BlockReduce2dProblem<typename Problem::XDataType,
                                        typename Problem::ComputeDataType,
                                        typename Problem::BlockShape>;
        return BlockReduce2d<P_>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockReduce2dSync()
    {
        using P_ = BlockReduce2dProblem<typename Problem::XDataType,
                                        typename Problem::ComputeDataType,
                                        typename Problem::BlockShape>;
        return BlockReduce2dSync<P_>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockReduce2dCrossWarpSync()
    {
        using P_ = BlockReduce2dProblem<typename Problem::XDataType,
                                        typename Problem::ComputeDataType,
                                        typename Problem::BlockShape>;
        return BlockReduce2dCrossWarpSync<P_>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        if constexpr(Problem::kNeedCrossWarpSync)
        {
            using P_ = BlockReduce2dProblem<typename Problem::XDataType,
                                            typename Problem::ComputeDataType,
                                            typename Problem::BlockShape>;

            using block_reduce2d = BlockReduce2d<P_>;
            using x_block_tile =
                decltype(make_static_distributed_tensor<typename Problem::XDataType>(
                    MakeXBlockTileDistribution<Problem>()));
            using y_block_tile = decltype(block_reduce2d::template MakeYBlockTile<x_block_tile>());

            return GetBlockReduce2dCrossWarpSync<Problem>().template GetSmemSize<y_block_tile>();
        }
        else
        {
            return 1; // zero size arrays are an extension
        }
    }
};
} // namespace ck_tile
