// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/mmac_gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp"

namespace ck_tile {

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
template <typename Problem, typename Policy = MmacGemmPipelineAGmemBGmemCRegV1DefaultPolicy>
struct MmacGemmPipelineAGmemBGmemCRegV1
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using ALayout = remove_cvref_t<typename Problem::ALayout>;
    using BLayout = remove_cvref_t<typename Problem::BLayout>;
    using CLayout = remove_cvref_t<typename Problem::CLayout>;

    static constexpr index_t BlockSize = Problem::kBlockSize;

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

    static constexpr index_t VectorSizeA = Problem::VectorSizeA;
    static constexpr index_t VectorSizeB = Problem::VectorSizeB;
    static constexpr index_t VectorSizeC = Problem::VectorSizeC;

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        //Here align LDS to 16 bytes
        return integer_divide_ceil(
                   sizeof(ADataType) *
                       Policy::template MakeALdsBlockDescriptor<Problem>().get_element_space_size(),
                   16) *
                   16 +
               sizeof(BDataType) *
                   Policy::template MakeBLdsBlockDescriptor<Problem>().get_element_space_size();
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction>
    CK_TILE_HOST_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                        const AElementFunction& a_element_func,
                                        const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                        const BElementFunction& b_element_func,
                                        index_t num_loop,
                                        void* p_smem) const
    {
        static_assert(
            std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                std::is_same_v<BDataType, remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kNPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        // A tile in LDS
        ADataType* p_a_lds = static_cast<ADataType*>(p_smem);

        constexpr auto a_lds_block_desc = Policy::template MakeALdsBlockDescriptor<Problem>();

        auto a_lds_block = make_tensor_view<address_space_enum::lds>(p_a_lds, a_lds_block_desc);

        constexpr index_t a_lds_block_space_size_aligned =
            integer_divide_ceil(sizeof(ADataType) * a_lds_block_desc.get_element_space_size(), 16) *
            16;

        // B tile in LDS
        BDataType* p_b_lds = static_cast<BDataType*>(
            static_cast<void*>(static_cast<char*>(p_smem) + a_lds_block_space_size_aligned));

        constexpr auto b_lds_block_desc = Policy::template MakeBLdsBlockDescriptor<Problem>();

        auto b_lds_block = make_tensor_view<address_space_enum::lds>(p_b_lds, b_lds_block_desc);

        // A DRAM tile window for load
        auto a_copy_dram_window =
            make_tile_window(a_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                             a_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeADramTileDistribution<Problem>());

        // A LDS tile window for store
        auto a_copy_lds_window = make_tile_window(
            a_lds_block, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // B DRAM tile window for load
        auto b_copy_dram_window =
            make_tile_window(b_dram_block_window_tmp.get_bottom_tensor_view(),
                             make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
                             b_dram_block_window_tmp.get_window_origin(),
                             Policy::template MakeBDramTileDistribution<Problem>());

        // B LDS tile window for store
        auto b_copy_lds_window = make_tile_window(
            b_lds_block, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // A LDS tile for block GEMM
        auto a_lds_gemm_window = make_tile_window(
            a_lds_block, make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // B LDS tile for block GEMM
        auto b_lds_gemm_window = make_tile_window(
            b_lds_block, make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}), {0, 0});

        // Block GEMM
        constexpr auto block_gemm = Policy::template GetBlockGemm<Problem>();

        // Acc register tile
        auto c_block_tile = decltype(block_gemm(a_lds_gemm_window, b_lds_gemm_window)){};

        // prefetch
        // global read 0
        auto a_block_tile = load_tile(a_copy_dram_window);
        auto b_block_tile = load_tile(b_copy_dram_window);

        {
            // move to 1
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});
            move_tile_window(b_copy_dram_window, {0, kKPerBlock});

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // LDS write 0
            // if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>)
            // {
            //     auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
            //         Policy::template MakeShuffledARegBlockDescriptor<Problem>());
            //     shuffle_tile(a_shuffle_tmp, a_block_tile);
            //     const auto a_block_tile_tmp = tile_elementwise_in(a_element_func, a_shuffle_tmp);
            //     store_tile(a_copy_lds_window, a_block_tile_tmp);
            // }
            // else
            // {
                store_tile(a_copy_lds_window, tile_elementwise_in(a_element_func, a_block_tile));
            // }

            // LDS write 0
            // if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
            // {
            //     auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
            //         Policy::template MakeShuffledBRegBlockDescriptor<Problem>());
            //     shuffle_tile(b_shuffle_tmp, b_block_tile);
            //     const auto b_block_tile_tmp = tile_elementwise_in(b_element_func, b_shuffle_tmp);
            //     store_tile(b_copy_lds_window, b_block_tile_tmp);
            // }
            // else
            // {
                store_tile(b_copy_lds_window, tile_elementwise_in(b_element_func, b_block_tile));
            // }
        }

        index_t iCounter = num_loop - 1;
        while(iCounter > 0)
        {
            // global read i + 1
            a_block_tile = load_tile(a_copy_dram_window);
            b_block_tile = load_tile(b_copy_dram_window);

            block_sync_lds();

            // GEMM i
            block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);

            block_sync_lds();

            // move to i + 2
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});
            move_tile_window(b_copy_dram_window, {0, kKPerBlock});

            // LDS write i + 1
            const auto a_block_tile_tmp = tile_elementwise_in(a_element_func, a_block_tile);
            store_tile(a_copy_lds_window, a_block_tile_tmp);

            // LDS write i + 1
            // if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
            // {
            //     auto b_shuffle_tmp_loop = make_static_distributed_tensor<BDataType>(
            //         Policy::template MakeShuffledBRegBlockDescriptor<Problem>());
            //     shuffle_tile(b_shuffle_tmp_loop, b_block_tile);
            //     store_tile(b_copy_lds_window,
            //                tile_elementwise_in(b_element_func, b_shuffle_tmp_loop));
            // }
            // else
            // {
                const auto b_block_tile_tmp = tile_elementwise_in(b_element_func, b_block_tile);
                store_tile(b_copy_lds_window, b_block_tile_tmp);
            // }

            iCounter--;
        }

        // tail
        {
            block_sync_lds();

            // GEMM num_loop - 1
            block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
        }
        auto c_block_output_tile = block_gemm.MakeOuputLayout(c_block_tile);

        return c_block_output_tile;
    }

    template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        return operator()(
            a_dram_block_window_tmp,
            [](const ADataType& a) { return a; }, //elementwise function, here do nothing
            b_dram_block_window_tmp,
            [](const BDataType& b) { return b; },
            num_loop,
            p_smem);
    }
};

} // namespace ck_tile
