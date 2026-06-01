// SPDX-License-Identifier: MIT
// Copyright (c) 2025, , Inc. All rights reserved.

/**
 * @file
 * We're defining the data access pattern for a 2D window (`XPerTile` by `YPerTile`)
 for `BlockSize` threads in a thread block.
 * X dimension is considered contiguous in memory, so a single instruction can access
 several adjacent and properly aligned elements (vector); the access pattern along X tile
 dimension is parameterized only by the suggested vector size `VecSize`.
 * We can't access more than `MaxVecSize = TileElementsPerThread = TileSize / BlockSize` elements
 with a single memory access, so the actual vector size along the X dimension is
 `X0 = min(MaxVecSize, VecSize)`.
 * This leaves `X1 = XPerTile / X0` threads per tile in X dimension.
 * X1 is also the number of threads per warp in X dimension, that is,
 X dimension is not split between warps, and each warp accesses X dimension entirely,
 and there is no iteration in X dimension.
 * The tuple <X0, X1> defines the X-axis access pattern.
 This part is common between the 2D distribution patterns.

 * What's different between the different 2D distribution patterns, is the Y axis access pattern.
 * There are 3 components in this access pattern;
 * (1) number of Y-axis elements (rows) per warp for a single instruction access,
 * (2) number of warps per thread block,
 * (3) number of iterations to cover the entire Y axis.

 * The raked here represents how data is partitioned across different processing granularity.
 * It represents howe we are going to access the data in thread, warp, or blocked in contiguous
 region.
 * From below, the qualifier for 'raked' is the part of warp/thread hierarchy
 * in the split of Y tile dimension where the iteration happens,
 * meaning, the iteration can be logically inserted as a tile dimension in 3 ways,
 * (1) after thread -> thread-raked,
 * (2) between warp and thread -> warp-raked,
 * (3) before warp -> block-raked

 * *Thread raked*

 * Y0 is the number of warps, which we can get from the equation `Y0 * WarpSize == BlockSize`
 * Y1 is the number of rows accessed by a warp within a single iteration,
 compute it from the equation `Y0 * X1 == WarpSize`
 * Y2 is the number of iterations to cover the tile,
 compute it from the equation `Y0 * Y1 * Y2 == YPerTile`

 * *Warp raked*

 * Y0 is the number of warps, we can get it in the same way as for thread-raked pattern,
 `Y0 * WarpSize == BlockSize`
 * Y1 is the number of iterations to cover the tile, `Y0 * Y1 * Y2 == YPerTile`.
 Compute Y2 from the equation below
 * Y2 is the number of rows accessed by a warp in a single iteration, `Y2 * X1 == WarpSize`

 * *Block raked*

 * Y0 is the number of iterations to cover the tile, `Y0 * Y1 * Y2 == YPerTile`.
 Compute Y1 and Y2 from the equations below
 * Y1 is the number of warps, `Y1 * WarpSize == BlockSize`
 * Y2 is the number of rows accessed by a warp in a single iteration, `Y2 * X1 == WarpSize`

 * In all cases, the tuple <Y0, Y1, Y2> defines the Y-axis access pattern.

 * *Selection*
 * When we are selecting, Thread-raked is used in element-wise operation because it is the
 * Thread-major memory order.
 * Warp-raked is used in matrix multiplication because the vectorization is in warp level.
 * Block-raked is used mostly for the reduction process, where will reduce the block in global
 * atomic level.
 *
 */

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/tensor/tile_distribution_encoding.hpp"
#include "ck_tile/core/utility/print.hpp"

namespace ck_tile {

/**
 * @brief Enumeration describing static tile distribution patterns.
 *
 */
enum struct tile_distribution_pattern
{
    /**
     * @brief Thread raked pattern.
     *
     */
    thread_raked,
    /**
     * @brief Warp raked pattern.
     *
     */
    warp_raked,
    /**
     * @brief Block raked pattern - aka linear.
     *
     */
    block_raked,
};

struct tile_distribution_encoding_pattern
{
};

/**
 * @brief Class creating 2D static tile distribution with different load/store patterns.
 *
 * @note We always assume that Tile is YPerTile x XPerTile where X dim (rightmost)
 *       is contiguous and we can do vector load on this dimension.
 *
 * @tparam BlockSize    Number of threads in a workgroup.
 * @tparam YPerTile    The tile size of outer/leftmost dimension.
 * @tparam XPerTile    The tile size of inner/rightmost dimension (contiguous).
 * @tparam VecSize      The vector access size.
 * @tparam DistributionPattern The enumeration describing used access pattern.
 */
template <index_t BlockSize,
          index_t YPerTile,
          index_t XPerTile,
          index_t VecSize,
          tile_distribution_pattern DistributionPattern,
          index_t NumWaveGroups = 1>
struct tile_distribution_encoding_pattern_2d : public tile_distribution_encoding_pattern
{
};

// Thread raked
template <index_t BlockSize,
          index_t YPerTile,
          index_t XPerTile,
          index_t VecSize,
          index_t NumWaveGroups>
struct tile_distribution_encoding_pattern_2d<BlockSize,
                                             YPerTile,
                                             XPerTile,
                                             VecSize,
                                             tile_distribution_pattern::thread_raked,
                                             NumWaveGroups>
    : public tile_distribution_encoding_pattern
{

    // TODO: make pattern where below condition does not need to hold - GGemmMultiDSplitk!
    static_assert(XPerTile % VecSize == 0, "XPerTile must be a multiple of VecSize!");
    static constexpr index_t warp_size  = get_warp_size();
    static constexpr index_t num_warps  = BlockSize / get_warp_size();
    static constexpr index_t LargestVec = (XPerTile * YPerTile) / (num_warps * warp_size);  //!! (64*128)/(4*64)=32
    static constexpr index_t X1         = VecSize > LargestVec ? LargestVec : VecSize;      //!! 8
    static constexpr index_t X0         = XPerTile / X1; // # of threads in X dim           //!! 64/8=8

    // # of rows in Y dim accessed by single wavefront in one iteration
    static constexpr index_t Y1 = warp_size / X0;
    static_assert(X0 * Y1 == warp_size, "X0 * Y1 must cover whole wavefront!");

    static constexpr index_t Y0 = num_warps / NumWaveGroups;
    //  YPerWarp = YPerTile / Y0;
    //  Y2 = YPerWarp / Y1;
    static constexpr index_t Y2 = YPerTile / (Y1 * Y0); // # of iters within wavefront

    static_assert(X0 * Y1 * Y0 * NumWaveGroups == BlockSize,
                  "X0 * warp_ys * Y0 must cover whole workgroup!");
    static_assert(Y0 * Y1 * Y2 == YPerTile, "Y0, Y1, Y2 must cover whole YPerTile");

    CK_TILE_HOST_DEVICE static constexpr auto make_2d_static_tile_distribution()
    {
        if constexpr(NumWaveGroups != 1)
        {
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<Y0>,
                                           tuple<sequence<Y1, Y2>, sequence<X0, X1>>,
                                           tuple<sequence<0>, sequence<1, 2>>,
                                           tuple<sequence<0>, sequence<0, 0>>, // -> <Y0>, <Y1, X0>
                                           sequence<1, 2>,
                                           sequence<1, 1>>{}); // -> <Y2, X1>
        }
        else
        {
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<Y0, Y1, Y2>, sequence<X0, X1>>,           //!! <4, 8, 4> <8, 8>
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<0>, sequence<1, 0>>, // -> <Y0>, <Y1, X0>
                                           sequence<1, 2>,
                                           sequence<2, 1>>{}); // -> <Y2, X1>
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto make_shuffled_2d_static_tile_distribution()
    {
        if constexpr(NumWaveGroups != 1)
        {
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<Y0>,
                                           tuple<sequence<X0, X1>, sequence<Y1, Y2>>,
                                           tuple<sequence<0>, sequence<2, 1>>,
                                           tuple<sequence<0>, sequence<0, 0>>, // -> <Y0>, <Y1, X0>
                                           sequence<1, 2>,
                                           sequence<1, 1>>{}); // -> <X1, Y2>
        }
        else
        {
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<X0, X1>, sequence<Y0, Y1, Y2>>,
                                           tuple<sequence<2>, sequence<2, 1>>,
                                           tuple<sequence<0>, sequence<1, 0>>, // -> <Y0>, <Y1, X0>
                                           sequence<1, 2>,
                                           sequence<1, 2>>{}); // -> <X1, Y2>
        }
    }
};

// Warp raked
template <index_t BlockSize,
          index_t YPerTile,
          index_t XPerTile,
          index_t VecSize,
          index_t NumWaveGroups>
struct tile_distribution_encoding_pattern_2d<BlockSize,
                                             YPerTile,
                                             XPerTile,
                                             VecSize,
                                             tile_distribution_pattern::warp_raked,
                                             NumWaveGroups>
    : public tile_distribution_encoding_pattern
{

    static_assert(XPerTile % VecSize == 0, "XPerTile must be a multiple of VecSize!");
    static constexpr index_t warp_size  = get_warp_size();
    static constexpr index_t num_warps  = BlockSize / get_warp_size();
    static constexpr index_t LargestVec = (XPerTile * YPerTile) / (num_warps * warp_size);
    static constexpr index_t X1         = VecSize > LargestVec ? LargestVec : VecSize;
    static constexpr index_t X0         = XPerTile / X1; // # of threads in X dim

    static constexpr index_t Y2 = warp_size / X0; // # of rows in Y dim to cover whole wavefront
    static_assert(X0 * Y2 == warp_size, "X0 * Y2 must cover whole wavefront!");

    static constexpr index_t Y0 = num_warps;
    static_assert(X0 * Y2 * Y0 == BlockSize, "X0 * Y2 * Y1 must cover whole workgroup!");

    static constexpr index_t Y1 = YPerTile / (Y2 * Y0); // # of iters within wavefront
    static_assert(Y0 * Y1 * Y2 == YPerTile, "Y0, Y1, Y2 must cover whole YPerTile");

    CK_TILE_HOST_DEVICE static constexpr auto make_2d_static_tile_distribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<Y0, Y1, Y2>, sequence<X0, X1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<0>, sequence<2, 0>>, // -> <Y0>, <Y2, X0>
                                       sequence<1, 2>,
                                       sequence<1, 1>>{}); // -> <Y1, X1>
    }

    CK_TILE_HOST_DEVICE static constexpr auto make_shuffled_2d_static_tile_distribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<X0, X1>, sequence<Y0, Y1, Y2>>,
                                       tuple<sequence<2>, sequence<2, 1>>,
                                       tuple<sequence<0>, sequence<2, 0>>, // -> <Y0>, <Y2, X0>
                                       sequence<1, 2>,
                                       sequence<1, 1>>{}); // -> <X1, Y1>
    }
};

// Block raked
template <index_t BlockSize,
          index_t YPerTile,
          index_t XPerTile,
          index_t VecSize,
          index_t NumWaveGroups>
struct tile_distribution_encoding_pattern_2d<BlockSize,
                                             YPerTile,
                                             XPerTile,
                                             VecSize,
                                             tile_distribution_pattern::block_raked,
                                             NumWaveGroups>
    : public tile_distribution_encoding_pattern
{

    // TODO: make pattern where below condition does not need to hold - GGemmMultiDSplitk!
    static_assert(XPerTile % VecSize == 0, "XPerTile must be a multiple of VecSize!");
    static constexpr index_t warp_size  = get_warp_size();
    static constexpr index_t num_warps  = BlockSize / get_warp_size();
    static constexpr index_t LargestVec = (XPerTile * YPerTile) / (num_warps * warp_size);
    static constexpr index_t X1         = VecSize > LargestVec ? LargestVec : VecSize;
    static constexpr index_t X0         = XPerTile / X1; // # of threads in X dim
    static constexpr index_t Y2 = warp_size / X0; // # of rows in Y dim to cover whole wavefront
    static_assert(X0 * Y2 == warp_size, "X0 * Y2 must cover whole wavefront!");
    static constexpr index_t Y1 = num_warps;
    static_assert(X0 * Y2 * Y1 == BlockSize, "X0 * Y2 * Y1 must cover whole workgroup!");
    static constexpr index_t Y0 = YPerTile / (Y2 * Y1); // # of iters
    static_assert(Y0 * Y1 * Y2 == YPerTile, "Y0, Y1, Y2 must cover whole YPerTile");

    CK_TILE_HOST_DEVICE static constexpr auto make_2d_static_tile_distribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<Y0, Y1, Y2>, sequence<X0, X1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>, // -> <Y1>, <Y2, X0>
                                       sequence<1, 2>,
                                       sequence<0, 1>>{}); // -> <Y0, X1>
    }

    CK_TILE_HOST_DEVICE static constexpr auto make_shuffled_2d_static_tile_distribution()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<X0, X1>, sequence<Y0, Y1, Y2>>,
                                       tuple<sequence<2>, sequence<2, 1>>,
                                       tuple<sequence<1>, sequence<2, 0>>, // -> <Y1>, <Y2, X0>
                                       sequence<1, 2>,
                                       sequence<1, 0>>{}); // -> <X1, Y0>
    }
};

// Helper function to convert enum to string
constexpr const char* tile_distribution_pattern_to_string(tile_distribution_pattern pattern)
{
    switch(pattern)
    {
    case tile_distribution_pattern::thread_raked: return "thread_raked";
    case tile_distribution_pattern::warp_raked: return "warp_raked";
    case tile_distribution_pattern::block_raked: return "block_raked";
    default: return "unknown";
    }
}

template <index_t BlockSize,
          index_t YPerTile,
          index_t XPerTile,
          index_t VecSize,
          tile_distribution_pattern DistributionPattern,
          index_t NumWaveGroups>
CK_TILE_HOST_DEVICE void print(const tile_distribution_encoding_pattern_2d<BlockSize,
                                                                           YPerTile,
                                                                           XPerTile,
                                                                           VecSize,
                                                                           DistributionPattern,
                                                                           NumWaveGroups>&)
{
    using PatternType = tile_distribution_encoding_pattern_2d<BlockSize,
                                                              YPerTile,
                                                              XPerTile,
                                                              VecSize,
                                                              DistributionPattern,
                                                              NumWaveGroups>;

    printf("tile_distribution_encoding_pattern_2d<BlockSize:%d, YPerTile:%d, XPerTile:%d, "
           "VecSize:%d, %s>: ",
           BlockSize,
           YPerTile,
           XPerTile,
           VecSize,
           tile_distribution_pattern_to_string(DistributionPattern));
    printf("{<Y0, Y1, Y2>: <%d, %d, %d>, <X0, X1>: <%d, %d>}\n",
           PatternType::Y0,
           PatternType::Y1,
           PatternType::Y2,
           PatternType::X0,
           PatternType::X1);
}

} // namespace ck_tile
