// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

//GemmTilePartitioner is used in ck_tile/03_gemm, So keep it simple now.
template <typename BlockGemmShape_>
struct GemmTilePartitioner
{
    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    static constexpr index_t kM = BlockGemmShape::kM;
    static constexpr index_t kN = BlockGemmShape::kN;
    static constexpr index_t kK = BlockGemmShape::kK;

    CK_TILE_HOST static constexpr auto GridSize(index_t M, index_t N, index_t batch_size)
    {
        index_t GridDimX = (M + kM - 1) / kM;
        index_t GridDimY = (N + kN - 1) / kN;
        index_t GridDimZ = batch_size;
        return dim3(GridDimX, GridDimY, GridDimZ);
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetLoopNum(index_t K)
    {
        return integer_divide_ceil(K, kK);
    }

    CK_TILE_DEVICE auto operator()()
    {
        const index_t iM = __builtin_amdgcn_readfirstlane(blockIdx.x * kM);
        const index_t iN = __builtin_amdgcn_readfirstlane(blockIdx.y * kN);
        return make_tuple(iM, iN);
    }
};


// The following codes are merged from github at 2025.09.19

/**
 * @brief Class providing 2D workgroup index mapping into 2D output GEMM C-tile space.
 *
 */
template <typename BlockGemmShapeType>
struct GemmTile2DPartitioner
{
    using BlockGemmShape = remove_cvref_t<BlockGemmShapeType>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    CK_TILE_HOST_DEVICE GemmTile2DPartitioner() noexcept = delete;
    CK_TILE_HOST_DEVICE GemmTile2DPartitioner([[maybe_unused]] index_t M,
                                              [[maybe_unused]] index_t N) noexcept;

    /**
     * @brief Calculates GEMM kernel grid size.
     *
     * @param M     GEMM's M dimension.
     * @param N     GEMM's N dimension.
     * @return dim3 Structure holding grid's X,Y and Z dimensions.
     */
    CK_TILE_HOST static auto
    GridSize(index_t M, index_t N) noexcept(noexcept(MPerBlock != 0 && NPerBlock != 0)) -> dim3
    {
        const index_t GridDimX = (M + MPerBlock - 1) / MPerBlock;
        const index_t GridDimY = (N + NPerBlock - 1) / NPerBlock;
        return dim3(GridDimX, GridDimY, 1);
    }

    /**
     * @brief Calculate number of loop iterations over GEMM's K dimension.
     *
     * @param K         GEMM's K dimension.
     * @return index_t  The number of loop iterations over K dimension.
     */
    CK_TILE_HOST_DEVICE static auto GetLoopNum(index_t K) noexcept -> index_t
    {
        return integer_divide_ceil(K, KPerBlock);
    }

    /**
     * @brief The function returns 2D output tile space.
     * @param [in] blockIdx is blockIdx.x
     * @param [in] blockIdy is blockIdx.y
     * @return Returns the output tile indexes.
     */

    /**
     * @brief Calculate workgroup 2D index mapping into 2D output C-tile space.
     *
     * @param blockIdx      WGP's X index.
     * @param blockIdy      WGP's Y index.
     * @return const tuple<index_t, index_t>    Tuple containing 2D output C-tile index.
     */
    CK_TILE_DEVICE static auto
    GetOutputTileIndex(index_t blockIdx, index_t blockIdy) noexcept -> const tuple<index_t, index_t>
    {
        const index_t iM = __builtin_amdgcn_readfirstlane(blockIdx);
        const index_t iN = __builtin_amdgcn_readfirstlane(blockIdy);
        return make_tuple(iM, iN);
    }
};

/**
 * @brief Class providing 1D WGP index mapping into 2D output C-tile space.
 *
 * @tparam BlockGemmShape_  A class providing basic GEMM parameters. \link TileGemmShape
 */
template <typename BlockGemmShape_>
struct GemmTile1DPartitioner
{
    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    CK_TILE_HOST_DEVICE GemmTile1DPartitioner() noexcept = delete;

    /**
     * @brief Construct a new GemmTile1DPartitioner object.
     *
     * @param M     GEMM's M dimension.
     * @param N     GEMM's N dimension.
     */
    CK_TILE_HOST_DEVICE GemmTile1DPartitioner([[maybe_unused]] index_t M, index_t N) noexcept
    {
        N_ = N;
    }

    /**
     * @brief Calculates GEMM kernel grid size.
     *
     * @param M     GEMM's M dimension.
     * @param N     GEMM's N dimension.
     * @return dim3 Structure holding grid's X,Y and Z dimensions.
     */
    CK_TILE_HOST_DEVICE static auto
    GridSize(index_t M, index_t N) noexcept(noexcept(MPerBlock != 0 && NPerBlock != 0)) -> index_t
    {
        const index_t GridDimX = (M + MPerBlock - 1) / MPerBlock;
        const index_t GridDimY = (N + NPerBlock - 1) / NPerBlock;
        return GridDimX * GridDimY;
    }

    /**
     * @brief Calculate number of loop iterations over GEMM's K dimension.
     *
     * @param K         GEMM's K dimension.
     * @return index_t  The number of loop iterations over K dimension.
     */
    CK_TILE_HOST_DEVICE static auto GetLoopNum(index_t K) noexcept -> index_t
    {
        return integer_divide_ceil(K, KPerBlock);
    }

    /**
     * @brief Calculate workgroup 1D index mapping into 2D output C-tile space.
     *
     * @param blockIdx      WGP's index.
     * @return const tuple<index_t, index_t>    Tuple containing 2D output C-tile index.
     */
    CK_TILE_DEVICE static auto
    GetOutputTileIndex(index_t blockIdx) noexcept -> const tuple<index_t, index_t>
    {
        const index_t NBlocks = integer_divide_ceil(N_, NPerBlock);

        const index_t iM = __builtin_amdgcn_readfirstlane(blockIdx / NBlocks);
        const index_t iN = __builtin_amdgcn_readfirstlane(blockIdx - iM * NBlocks);
        return make_tuple(iM, iN);
    }

    private:
    CK_TILE_DEVICE static index_t N_;
};

/**
 * @brief `GemmTile1DPartitioner::GetOutputTileIndex`'s std::false specialization,
 * checking expression validity in-place for ill-formed.
 */
template <typename, typename = void>
struct HasFnOneArgImpl : std::false_type
{
};

/**
 * @brief `GemmTile1DPartitioner::GetOutputTileIndex`'s std::true specialization,
 * checking expression validity in-place for well-formed.
 * @note: `1` - a constant value indicating the number of parameters in the function.
 */
template <typename T>
struct HasFnOneArgImpl<T, std::void_t<decltype(std::declval<T>().GetOutputTileIndex(1))>>
    : std::true_type
{
};

/**
 * @brief Struct used to calculate offseted tile indexes.
 * @note: The struct supports the 1D-Partitioner mechanism,
 * enable-if `GetOutputTileIndex`-fn is std::true_type when `GetOutputTileIndex`-fn is well-formed,
 * otherwise std::false_type.
 */
template <typename TilePartitioner,
          typename = typename std::enable_if_t<HasFnOneArgImpl<TilePartitioner>{}>>
struct OffsettedTile1DPartitioner
{
    /**
     * @brief The function subtracts the block's start (offset) from 1D raw-indexes.
     * @param [in] block_start Workgroup offset.
     * @param [in] M           Gemm's M dimension.
     * @param [in] N           Gemm's N dimension.
     * @return Returns a `tuple` [Im, In] with shifted index.
     */
    [[nodiscard]] CK_TILE_DEVICE static auto GetOffsetedTileIndex(
        index_t block_start, index_t M, index_t N) noexcept -> const tuple<index_t, index_t>
    {
        const auto [iM, iN] = TilePartitioner{M, N}.GetOutputTileIndex(blockIdx.x - block_start);
        return make_tuple(iM, iN);
    }

    /**
     * @brief The function subtracts the block's start (offset) from a given block index.
     * @param [in] block_start Workgroup offset.
     * @param [in] M           Gemm's M dimension.
     * @param [in] N           Gemm's N dimension.
     * @param [in] block_idx   Current block index of the workgroup.
     * @return Returns a `tuple` [Im, In] with shifted index.
     */
    [[nodiscard]] CK_TILE_DEVICE static auto
    GetOffsetedTileIndex(index_t block_start, index_t M, index_t N, index_t block_idx) noexcept
        -> const tuple<index_t, index_t>
    {
        const auto [iM, iN] = TilePartitioner{M, N}.GetOutputTileIndex(block_idx - block_start);
        return make_tuple(iM, iN);
    }
};

/**
 * @brief Class mapping 1D block index into 2D output tile space.
 *
 * @note It groups spatially workgroups in order to better utilize caches.
 *       It is using grouped Rows of column-vectors WGP pattern. It's optimized
 *       for gfx94x-like multiple-die chip.
 *
 * @tparam GroupNum - The number of big groups.
 * @tparam M01      - The number of groups in M dim within spatially local WGPs,
 *
 */
template <typename BlockGemmShapeType, index_t GroupNum, index_t M01>
struct GemmSpatiallyLocalTilePartitioner
{
    using BlockGemmShape = remove_cvref_t<BlockGemmShapeType>;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;    //128
    static constexpr index_t NPerBlock = BlockGemmShape::kN;    //128
    static constexpr index_t KPerBlock = BlockGemmShape::kK;    //64
    static constexpr index_t KLanePerBlock = BlockGemmShape::KLanePerBlock;

    CK_TILE_HOST_DEVICE GemmSpatiallyLocalTilePartitioner() noexcept = delete;
    CK_TILE_HOST_DEVICE GemmSpatiallyLocalTilePartitioner(index_t M_, index_t N_) noexcept
        : M(M_), N(N_)
    {
    }

    /**
     * @brief Calculates GEMM kernel grid size.
     *
     * @param M     GEMM's M dimension.
     * @param N     GEMM's N dimension.
     * @return index_t A total number of workgroups.
     */
    CK_TILE_HOST_DEVICE static auto
    GridSize(index_t M, index_t N) noexcept(noexcept(MPerBlock != 0 && NPerBlock != 0)) -> index_t
    {
        const index_t GridDimX = integer_divide_ceil(M, MPerBlock);
        const index_t GridDimY = integer_divide_ceil(N, NPerBlock);
        return GridDimX * GridDimY;
    }

    /**
     * @brief Calculate number of loop iterations over GEMM's K dimension.
     *
     * @param K         GEMM's K dimension.
     * @return index_t  The number of loop iterations over K dimension.
     */
    CK_TILE_HOST_DEVICE static auto GetLoopNum(index_t K) noexcept -> index_t
    {
        return integer_divide_ceil(K, KPerBlock);
    }

    /**
     * @brief Calculate workgroup 1D index mapping into 2D output C-tile space.
     *
     * @param [in] block_1d_id      WGP's index.
     * @return const tuple<index_t, index_t>    Tuple containing 2D output C-tile index.
     */
    CK_TILE_DEVICE auto
    GetOutputTileIndex(index_t block_1d_id) noexcept -> const tuple<index_t, index_t>
    {
        const auto M0 = integer_divide_ceil(M, MPerBlock);
        const auto N0 = integer_divide_ceil(N, NPerBlock);

        if(M0 == 1)
        {
            return make_tuple(0, block_1d_id);
        }
        else if(N0 == 1)
        {
            return make_tuple(block_1d_id, 0);
        }
        // block_1d_id = block_1d_id % (M0 * N0); // swallow batch index
        else
        {
            const auto group_size    = integer_divide_ceil(M0 * N0, GroupNum);
            const auto big_group_num = GroupNum - (group_size * GroupNum - M0 * N0);
            const auto group_id_y    = block_1d_id / GroupNum;
            const auto group_id_x    = block_1d_id - group_id_y * GroupNum;
            const auto remap_block_1d_id =
                group_id_x <= big_group_num
                    ? group_id_x * group_size + group_id_y
                    : group_id_x * group_size + big_group_num - group_id_x + group_id_y;

            const index_t idx_M0 = remap_block_1d_id / N0;
            const index_t idx_N0 = remap_block_1d_id - idx_M0 * N0;

            const index_t M0_tmp     = M0 / M01;
            const index_t M0_mod_M01 = M0 - M0_tmp * M01;

            const auto M01_adapt = (idx_M0 < M0 - M0_mod_M01) ? M01 : M0_mod_M01;

            const index_t idx_M00          = idx_M0 / M01;
            const index_t idx_M01          = idx_M0 - idx_M00 * M01;
            const index_t idx_N0_M01_local = idx_N0 + idx_M01 * N0;

            /**
             *                        idxN0
             *
             *           |<               mtx   N                 >|
             *
             *             NPerBlock   NPerBlock   NPerBlock   NPerBlock
             *                N_0         N_1        N_2         N_3
             *       -   |-----------|-----------|-----------|-----|-----|-
             *       ^   | -   -  0  |/---->  2  |           |     |     |
             *           | |   |     /     |     |           |     |     |  M_0  MPerBlock
             *           | M   |    /|     |     |           |     |     |
             *           |-0---|---/-|-----|-----|-----------|-----|-----|-
             *           | 1   |  /  |     |     |  blockid  |     |     |
             * idxM0     | |   | /   |     V     |     5     |     |     |  M_1  MPerBlock
             *           | -   V   1 |     -  3  |           |     |     |
             *           |-----------|-----------|-----------|-----|-----|-
             *    mtx M  |           |           |           |     |     |
             *           |           |           |           |     |     |  M_2  MPerBlock
             *           |           |           |           |     |     |
             *           |-----------|-----------|-----------|-----|-----|-
             *           |           |           |           |     |     |
             *           |           |           |           |     |     |  M_3  MPerBlock
             *           |           |           |           |     |     |
             *           |-----------|-----------|-----------|-----|-----|-
             *       V   |           |           |           |     |     |
             *       -   |-----------|-----------|-----------|-----|-----|- M_4  MPerBlock
             *           |           |           |           |     |     |
             *           |-----------|-----------|-----------|-----|-----|-
             *  Example:
             *   assume:
             *      M0 = 5
             *      N0 = 4
             *      block_1d_id = 5
             *      M01 = 2
             *
             *   idx_N0 = 1
             *   idx_M0 = 1
             *   M01_adapt = 2
             *   idx_M00 = 0
             *   idx_M01 = 1
             *   idx_N0_M01_local = 5
             *   output {1, 2}
             */

            const index_t N_out           = idx_N0_M01_local / M01_adapt;
            const index_t idx_loc_mod_M01 = idx_N0_M01_local - N_out * M01_adapt;

            return make_tuple(idx_loc_mod_M01 + idx_M00 * M01, N_out);
        }
    }

    private:
    index_t M;
    index_t N;
};

/**
 * @brief Stream-K tile partitioner that dynamically balances work across workgroups
 *
 * This partitioner is responsible for mapping workgroups to tiles in the C tensor
 * for the Stream-K algorithm which decomposes the GEMM problem
 * into smaller work units and distributes them more evenly across available blocks,
 * improving load balancing especially for cases where the K dimension is large.
 *
 *  @tparam BlockGemmShapeType  A class providing basic GEMM parameters.
 *  @tparam ReductionStrategy  A class that defines the reduction strategy for the results in
 *  the C Tensor.
 *  @tparam TileSwizzleSubM  A value that defines the size of the swizzle group along the m
 *  dimension, where the swizzle group denotes consecutive tiles down a column. For instance a
 *  swizzle group of 8 denotes tiles 0, 1, ..., 7, map to tiles [0,0], [1,0], ..., [7,0] in the C
 *  tensor.
 */
template <typename BlockGemmShapeType,
          StreamKReductionStrategy ReductionStrategy = ck_tile::StreamKReductionStrategy::Atomic,
          uint32_t TileSwizzleSubM                   = 8>
struct StreamKTilePartitioner
{
    using BlockGemmShape = BlockGemmShapeType;

    static constexpr uint32_t MPerBlock = BlockGemmShape::kM;
    static constexpr uint32_t NPerBlock = BlockGemmShape::kN;
    static constexpr uint32_t KPerBlock = BlockGemmShape::kK;

    CK_TILE_HOST_DEVICE StreamKTilePartitioner() noexcept = delete;

    /**
     * @brief Construct Stream-K tile partitioner with problem dimensions
     */
    CK_TILE_HOST_DEVICE StreamKTilePartitioner(uint32_t M,
                                               uint32_t N,
                                               uint32_t K,
                                               uint32_t num_cu,
                                               uint32_t occupancy,
                                               uint32_t sk_blocks = 0xffffffff) noexcept
        : M_(M), N_(N), K_(K)
    {
        num_tile_m_ = integer_divide_ceil(M, MPerBlock);
        num_tile_n_ = integer_divide_ceil(N, NPerBlock);
        num_tile_k_ = integer_divide_ceil(K, KPerBlock);

        constexpr uint32_t min_k_iters_per_sk_block = 2;
        uint32_t num_tiles                          = num_tile_m_ * num_tile_n_;
        k_iters_per_tile                            = mdiv(num_tile_k_);

        // one cu can hold one wg at one time, from the whole cZ's point of view
        // if number of wg is same as num_cu, we call it 1 dispatch
        // if number of wg is 2x num_cu, we call it 2 dispatches.
        // one dispatch can deliver wg same as num_cu (full dispatch), or less than num_cu (partial
        // dispatch)
        //
        const uint32_t full_dispatches        = num_tiles / num_cu;
        const uint32_t full_dispatch_tiles    = full_dispatches * num_cu;
        const uint32_t partial_dispatch_tiles = num_tiles - full_dispatch_tiles;

        uint32_t sk_occupancy = occupancy;
        uint32_t dp_tiles     = full_dispatch_tiles;
        uint32_t sk_tiles     = partial_dispatch_tiles;

        if(full_dispatches < occupancy)
        {
            // in this case, we allocate all blocks as sk blocks
            // sk_occupancy = occupancy - full_dispatches;
            sk_occupancy = 1;
            dp_tiles     = full_dispatch_tiles;
            sk_tiles     = partial_dispatch_tiles;
        }
        else if((occupancy > 1) && (full_dispatches % occupancy == occupancy - 1))
        {
            // e.g. occupancy = 2, full_dispatches = 3, 5, 7 ...
            //      occupancy = 3, full_dispatches = 5, 8, 11 ...
            //      occupancy = 4, full_dispatches = 7, 11 ...
            sk_occupancy = 1; // left 1 slot for sk occupancy
            dp_tiles     = full_dispatch_tiles;
            sk_tiles     = partial_dispatch_tiles;
        }
        else
        {
            // otherwise, we reduce 1 dispatch from dp, together with partial dispatch,
            // to construct sk dispatch
            sk_occupancy = occupancy - ((full_dispatches - 1) % occupancy);
            dp_tiles     = full_dispatch_tiles - num_cu;
            sk_tiles     = partial_dispatch_tiles + num_cu;
        }

        // uint32_t dp_iters_per_block = k_iters_per_tile.get();
        uint32_t sk_total_iters = k_iters_per_tile.get() * sk_tiles;
        uint32_t dp_num_blocks  = 0;

        {
            const uint32_t min_sk_tiles = (sk_tiles >= num_cu) ? num_cu : (sk_tiles + 1);
            const uint32_t max_sk_tiles =
                (sk_tiles >= num_cu) ? num_cu * sk_occupancy
                                     : min(num_cu, sk_total_iters / min_k_iters_per_sk_block);

            // if use dp for sk-block, how many iters do we need
            const uint32_t dp_for_sk_iters = k_iters_per_tile.get();

            uint32_t best_sk_score =
                std::numeric_limits<int>::max(); // we need to find the smallest sk iters
            for(uint32_t tentative_sk_blocks = min_sk_tiles; tentative_sk_blocks < max_sk_tiles;
                tentative_sk_blocks++)
            {
                const uint32_t tentative_sk_iters_per_block =
                    (sk_total_iters + tentative_sk_blocks - 1) / tentative_sk_blocks;
                const uint32_t tentative_sk_iters = tentative_sk_iters_per_block;
                const uint32_t sk_blocks_per_tile = (tentative_sk_blocks + sk_tiles - 1) / sk_tiles;

                //       the more sk_blocks_per_tile, the worse the overhead
                uint32_t cross_sk_blocks_overhead = sk_blocks_per_tile;
                if(tentative_sk_blocks % sk_tiles != 0)
                {
                    // penalty for uneven divide
                    cross_sk_blocks_overhead +=
                        sk_blocks_per_tile * tentative_sk_iters_per_block / 50;
                }

                const uint32_t tentative_sk_score = tentative_sk_iters + cross_sk_blocks_overhead;

                if(tentative_sk_score < best_sk_score)
                {
                    best_sk_score = tentative_sk_score;
                    sk_num_blocks = tentative_sk_blocks;
                }
            }

            if(best_sk_score >= dp_for_sk_iters)
            {
                sk_num_blocks = 0;
            }

            // give a chance to control num of sk blocks
            sk_num_blocks = sk_blocks != 0xffffffff ? sk_blocks : sk_num_blocks;

            if(sk_num_blocks == 0)
            {
                sk_num_big_blocks     = 0;
                k_iters_per_big_block = 0;

                dp_num_blocks      = num_tiles; // all tile to be dp block
                dp_start_block_idx = 0;
                sk_total_iters     = 0; // clear this tiles
            }
            else
            {
                // k_iters_per_sk_block is the floor of avg each ck block loop over tiles.
                // we need to decide how many iters for each sk block
                // let m = k_iters_per_sk_block
                // some of the sk block (little) will cover m iters, some (big) will cover m+1
                // we have
                // 1) l + b = sk_blocks
                // 2) l * m + b * (m + 1) = sk_total_iters
                //      => (l + b) * m + b = sk_total_iters
                //      => sk_blocks * m + b = sk_total_iters
                //      => b = sk_total_iters - m * sk_blocks
                //      NOTE: big could be zero
                const uint32_t k_iters_per_sk_block = sk_total_iters / sk_num_blocks;
                sk_num_big_blocks     = sk_total_iters - k_iters_per_sk_block * sk_num_blocks;
                k_iters_per_big_block = k_iters_per_sk_block + 1;

                dp_num_blocks      = dp_tiles;
                dp_start_block_idx = (sk_num_blocks + num_cu - 1) / num_cu * num_cu;
            }
        }
        n_tiles                   = mdiv2(num_tile_n_);
        reduction_start_block_idx = dp_start_block_idx + dp_num_blocks;

        if constexpr(ReductionStrategy == ck_tile::StreamKReductionStrategy::Reduction)
        {
            const uint32_t upper_big    = lcm(k_iters_per_big_block, k_iters_per_tile.get());
            const uint32_t upper_little = lcm(k_iters_per_big_block - 1, k_iters_per_tile.get());
            equiv_tiles_big             = mdiv(upper_big / k_iters_per_tile.get());
            equiv_tiles_little          = mdiv(upper_little / k_iters_per_tile.get());
        }
    }

    /**
     * @brief Calculate optimal grid size for Stream-K
     */
    CK_TILE_HOST auto GridSize() const noexcept -> dim3
    {
        if constexpr(ReductionStrategy == ck_tile::StreamKReductionStrategy::Reduction)
        {
            return dim3(reduction_start_block_idx + GetSkTiles(), 1, 1);
        }
        else
            return dim3(reduction_start_block_idx, 1, 1);
    }

    /**
     * @brief Calculate number of loop iterations over K dimension for given work unit
     */
    CK_TILE_HOST_DEVICE static auto GetLoopNum(uint32_t K) noexcept -> uint32_t
    {
        return integer_divide_ceil(K, KPerBlock); // Stream-K processes one K-slice at a time
    }

    /**
     * @brief Get output tile index for standard 2D mapping (compatibility)
     */
    CK_TILE_DEVICE auto
    GetOutputTileIndex(uint32_t tile_idx) const noexcept -> tuple<uint32_t, uint32_t>
    {
        uint32_t m_tile_idx, n_tile_idx;
        n_tiles.divmod(tile_idx, num_tile_n_, m_tile_idx, n_tile_idx);

        // swizzle tile

        uint32_t tile_swizzle_sub_m_rem = num_tile_m_ % TileSwizzleSubM;

        const auto sub_m_adapt = (m_tile_idx < (num_tile_m_ - tile_swizzle_sub_m_rem))
                                     ? TileSwizzleSubM
                                     : tile_swizzle_sub_m_rem;

        uint32_t m_tile_idx_sub0, m_tile_idx_sub1;
        m_tile_idx_sub0 = m_tile_idx / TileSwizzleSubM;
        m_tile_idx_sub1 = m_tile_idx % TileSwizzleSubM;

        uint32_t tile_idx_local = n_tile_idx + m_tile_idx_sub1 * num_tile_n_;

        uint32_t m_tile_idx_with_adapt, n_tile_idx_with_adapt;

        n_tile_idx_with_adapt = tile_idx_local / sub_m_adapt;
        m_tile_idx_with_adapt = tile_idx_local % sub_m_adapt;
        return make_tuple(m_tile_idx_with_adapt + m_tile_idx_sub0 * TileSwizzleSubM,
                          n_tile_idx_with_adapt);
    }

    /**
     * @brief Get work range for a given block ID
     */
    CK_TILE_DEVICE void
    GetBlockItr(uint32_t block_idx, uint32_t& iter_start, uint32_t& iter_end) const noexcept
    {
        if(block_idx < sk_num_big_blocks)
        {
            iter_start = block_idx * k_iters_per_big_block;
            iter_end   = iter_start + k_iters_per_big_block;
        }
        else if(block_idx < sk_num_blocks)
        {
            iter_start = (sk_num_big_blocks * k_iters_per_big_block) +
                         (block_idx - sk_num_big_blocks) * (k_iters_per_big_block - 1);
            iter_end = iter_start + (k_iters_per_big_block - 1);
        }
        else if(block_idx >= dp_start_block_idx)
        {
            uint32_t sk_total_iters     = GetSkTotalIters();
            uint32_t dp_iters_per_block = k_iters_per_tile.get();
            iter_start = sk_total_iters + (block_idx - dp_start_block_idx) * dp_iters_per_block;
            iter_end   = iter_start + dp_iters_per_block;
        }
    }

    /**
     * @brief Get total number of iterations for sk tiles
     */
    CK_TILE_HOST_DEVICE uint32_t GetSkTotalIters() const noexcept
    {
        uint32_t sk_total_iters = sk_num_big_blocks * k_iters_per_big_block +
                                  (sk_num_blocks - sk_num_big_blocks) * (k_iters_per_big_block - 1);
        return sk_total_iters;
    }

    /**
     * @brief Get total number of sk tiles
     */
    CK_TILE_HOST_DEVICE uint32_t GetSkTiles() const noexcept
    {
        // tiles for sk
        uint32_t sk_total_iters = GetSkTotalIters();
        return k_iters_per_tile.div(sk_total_iters);
    }

    /**
     * @brief Get length of loop iterations for stream-k loop
     */
    CK_TILE_DEVICE uint32_t GetCurrentIterLength(uint32_t iter_start,
                                                 uint32_t iter_end,
                                                 uint32_t total_iter_length) const noexcept
    {
        uint32_t iter_length_mod, iter_length_quo /*unused*/;
        k_iters_per_tile.divmod(iter_end, iter_length_quo, iter_length_mod);
        uint32_t total_iter_length_val = static_cast<uint32_t>(total_iter_length);
        uint32_t current_iter_length =
            min(iter_length_mod == 0 ? (iter_end - iter_start) : iter_length_mod,
                total_iter_length_val);
        return current_iter_length;
    }

    /**
     * @brief Get index of tile during a specified iteration
     */
    CK_TILE_DEVICE uint32_t GetTileIdx(uint32_t iter) const noexcept
    {
        return k_iters_per_tile.div(iter);
    }

    /**
     * @brief Get index of tile during a specified iteration
     */
    CK_TILE_DEVICE void
    GetTileIdxWithOffset(uint32_t iter, uint32_t& tile_idx, uint32_t& iter_offset) const noexcept
    {
        uint32_t tile_idx_val    = static_cast<uint32_t>(tile_idx);
        uint32_t iter_offset_val = static_cast<uint32_t>(iter_offset);
        k_iters_per_tile.divmod(iter, tile_idx_val, iter_offset_val);
    }

    /**
     * @brief Calculates the buffer space needed for accumulation
     */
    CK_TILE_HOST_DEVICE uint32_t GetWorkSpaceSizeForAcc(uint32_t acc_element_bytes) const noexcept
    {
        static constexpr uint32_t alignment = 128;
        uint32_t acc_buffer_bytes =
            MPerBlock * NPerBlock * GetTotalAccBuffers() * acc_element_bytes;
        return (acc_buffer_bytes + alignment - 1) / alignment * alignment;
    }

    /**
     * @brief Calculates the buffer space needed for the semaphore
     */
    CK_TILE_HOST_DEVICE uint32_t GetWorkSpaceSizeForSemaphore() const noexcept
    {
        return GetSkTiles() * sizeof(uint32_t);
    }

    /**
     * @brief Calculates the total buffer space needed for accumulation and the semaphore
     */
    CK_TILE_HOST_DEVICE uint32_t GetWorkSpaceSize(uint32_t acc_element_bytes) const noexcept
    {
        return GetWorkSpaceSizeForAcc(acc_element_bytes) + GetWorkSpaceSizeForSemaphore();
    }

    /**
     * @brief Get location of intersection of tiles for reduction
     */
    CK_TILE_HOST_DEVICE uint32_t GetTileIntersections(uint32_t tiles_,
                                                      const mdiv& equiv_tiles_) const noexcept
    {
        uint32_t tile_idx_        = tiles_ == 0 ? 0 : (tiles_ - 1);
        uint32_t max_equiv_tiles_ = equiv_tiles_.get() - 1;
        uint32_t quo_, rem_;
        equiv_tiles_.divmod(tile_idx_, quo_, rem_);
        return quo_ * max_equiv_tiles_ + rem_;
    }

    /**
     * @brief Calculate the number of tiles needed for the number of sk blocks
     */
    CK_TILE_HOST_DEVICE uint32_t GetTilesCoverSkBlock(uint32_t num_sk_blocks_,
                                                      uint32_t iters_per_sk_block_) const noexcept
    {
        return k_iters_per_tile.div(num_sk_blocks_ * iters_per_sk_block_ + k_iters_per_tile.get() -
                                    1);
    }

    /**
     * @brief Calculate the amount of total accumulation buffers required for stream-k
     */
    CK_TILE_HOST_DEVICE uint32_t GetTotalAccBuffers() const noexcept
    {
        uint32_t tiles_cover_big_blocks =
            GetTilesCoverSkBlock(sk_num_big_blocks, k_iters_per_big_block);
        uint32_t tiles_cover_little_blocks =
            GetTilesCoverSkBlock(sk_num_blocks - sk_num_big_blocks, k_iters_per_big_block - 1);

        uint32_t total_intersec_big = GetTileIntersections(tiles_cover_big_blocks, equiv_tiles_big);
        uint32_t total_intersec_little =
            GetTileIntersections(tiles_cover_little_blocks, equiv_tiles_little);

        return sk_num_blocks + total_intersec_big + total_intersec_little;
    }

    /**
     * @brief Calculate offset based on tile index for big/little tiles
     */
    CK_TILE_DEVICE uint32_t GetAccBufferOffsetFromTile(uint32_t tile_idx_) const noexcept
    {
        uint32_t tiles_cover_big_blocks =
            GetTilesCoverSkBlock(sk_num_big_blocks, k_iters_per_big_block);
        if(tile_idx_ < tiles_cover_big_blocks)
        {
            uint32_t touched_sk_blocks =
                (tile_idx_ * k_iters_per_tile.get() + k_iters_per_big_block - 1) /
                k_iters_per_big_block;
            uint32_t current_intersec = GetTileIntersections(tile_idx_, equiv_tiles_big);
            return touched_sk_blocks + current_intersec;
        }
        else
        {
            uint32_t iters_per_little_sk_block = k_iters_per_big_block - 1;
            uint32_t tile_idx_little_reverse   = GetSkTiles() - tile_idx_;
            uint32_t touched_sk_blocks =
                (tile_idx_little_reverse * k_iters_per_tile.get() + iters_per_little_sk_block - 1) /
                iters_per_little_sk_block;
            uint32_t current_intersec =
                GetTileIntersections(tile_idx_little_reverse, equiv_tiles_little);
            return GetTotalAccBuffers() - (touched_sk_blocks + current_intersec);
        }
    }

    /**
     * @brief Calculate offset based on block_idx index for big/little streamk blocks
     */
    CK_TILE_DEVICE uint32_t GetAccBufferOffsetFromBlock(uint32_t block_idx_) const noexcept
    {
        uint32_t iters_per_big_sk_block    = k_iters_per_big_block;
        uint32_t iters_per_little_sk_block = k_iters_per_big_block - 1;
        if(block_idx_ < sk_num_big_blocks)
        {
            uint32_t touched_tiles    = k_iters_per_tile.div(block_idx_ * iters_per_big_sk_block +
                                                          k_iters_per_tile.get() - 1);
            uint32_t current_intersec = GetTileIntersections(touched_tiles, equiv_tiles_big);
            return block_idx_ + current_intersec;
        }
        else
        {
            uint32_t block_idx_little_reverse = sk_num_blocks - block_idx_;
            uint32_t touched_tiles            = k_iters_per_tile.div(
                block_idx_little_reverse * iters_per_little_sk_block + k_iters_per_tile.get() - 1);
            uint32_t current_intersec = GetTileIntersections(touched_tiles, equiv_tiles_little);
            return GetTotalAccBuffers() - (block_idx_little_reverse + current_intersec);
        }
    }

    // Getters for problem dimensions
    CK_TILE_HOST_DEVICE uint32_t GetNumTileM() const noexcept { return num_tile_m_; }
    CK_TILE_HOST_DEVICE uint32_t GetNumTileN() const noexcept { return num_tile_n_; }
    CK_TILE_HOST_DEVICE uint32_t GetNumTileK() const noexcept { return num_tile_k_; }

    uint32_t sk_num_blocks;
    uint32_t sk_num_big_blocks;
    uint32_t dp_start_block_idx;
    uint32_t reduction_start_block_idx;
    uint32_t k_iters_per_big_block;
    mdiv2 n_tiles;
    mdiv k_iters_per_tile;
    mdiv equiv_tiles_big;    // for reduction
    mdiv equiv_tiles_little; // for reduction

    private:
    uint32_t M_, N_, K_;
    uint32_t num_tile_m_, num_tile_n_, num_tile_k_;
};

} // namespace ck_tile
