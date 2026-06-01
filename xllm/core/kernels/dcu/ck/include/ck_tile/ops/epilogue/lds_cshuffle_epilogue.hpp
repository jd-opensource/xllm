// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

namespace ck_tile {

template <typename AccDataType_,
          typename CDataType_,
          typename WarpIters,
          typename Warps,
          typename WarpTile,
          bool Transpose_>
struct LdsCShuffleEpilogueProblem
{
    using AccDataType = AccDataType_;
    using CDataType   = CDataType_;

    static constexpr index_t MWarpIter       = WarpIters::at(number<0>{});
    static constexpr index_t NWarpIter       = WarpIters::at(number<1>{});
    static constexpr index_t MWarps          = Warps::at(number<0>{});
    static constexpr index_t NWarps          = Warps::at(number<1>{});
    static constexpr index_t MmmacIter       = WarpTile::at(number<0>{});
    static constexpr index_t NmmacIter       = WarpTile::at(number<1>{});
    static constexpr index_t MPerMmac        = WarpTile::at(number<2>{});
    static constexpr index_t NPerMmac        = WarpTile::at(number<3>{});
    static constexpr index_t MmmacInterleave = WarpTile::at(number<4>{});
    static constexpr index_t NmmacInterleave = WarpTile::at(number<5>{});

    // contigous wave tile
    static constexpr index_t MWarpTilePerIter = MmmacIter * MPerMmac * MmmacInterleave;
    static constexpr index_t NWarpTilePerIter = NmmacIter * NPerMmac * NmmacInterleave;

    // contigous block tile
    static constexpr index_t MBlockTilePerIter = MWarps * MWarpTilePerIter;
    static constexpr index_t NBlockTilePerIter = NWarps * NWarpTilePerIter;

    static constexpr bool Transpose = Transpose_;
};

template <typename Problem_, typename WG_>
struct LdsCShuffleEpilogue
{
    using Problem = remove_cvref_t<Problem_>;

    using WG          = remove_cvref_t<WG_>;
    using CWarpDstr   = typename WG::CWarpDstr;
    using CWarpTensor = typename WG::CWarpTensor;

    using AccDataType = remove_cvref_t<typename Problem::AccDataType>;
    using CDataType   = remove_cvref_t<typename Problem::CDataType>;

    static constexpr index_t MWarpIter = Problem::MWarpIter;
    static constexpr index_t NWarpIter = Problem::NWarpIter;

    static constexpr index_t MWarps = Problem::MWarps;
    static constexpr index_t NWarps = Problem::NWarps;

    static constexpr index_t MWarpTilePerIter = Problem::MWarpTilePerIter;
    static constexpr index_t NWarpTilePerIter = Problem::NWarpTilePerIter;

    static constexpr index_t MBlockTilePerIter = Problem::MBlockTilePerIter;
    static constexpr index_t NBlockTilePerIter = Problem::NBlockTilePerIter;

    static constexpr bool Transpose = Problem::Transpose;

    template <index_t VectorLength = 1>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsBlockDescriptor()
    {
        if constexpr(Transpose)
        {
            return ck_tile::make_naive_tensor_descriptor_packed(
                make_tuple(number<MBlockTilePerIter>{}, number<NBlockTilePerIter>{}),
                number<VectorLength>{});
        }
        else
        {
            return ck_tile::make_naive_tensor_descriptor_packed(
                make_tuple(number<NBlockTilePerIter>{}, number<MBlockTilePerIter>{}),
                number<VectorLength>{});
        }
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemByteSize()
    {
        return MakeLdsBlockDescriptor().get_element_space_size() * sizeof(CDataType);
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSize()
    {
        // FIXME: use buffer_store_dwordx2 as workaround
        constexpr index_t MaxVectorStoreSize = 8;
        return MaxVectorStoreSize / sizeof(CDataType);
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeCShuffleDramDistribution()
    {
        if constexpr(Transpose)
        {
            constexpr index_t CShuffleMPerIter = MBlockTilePerIter;
            constexpr index_t CShuffleNPerIter = NBlockTilePerIter;

            constexpr index_t NumWaves  = MWarps * NWarps;
            constexpr index_t BlockSize = get_warp_size() * NumWaves;

            constexpr index_t CShuffleLaneSlice = CShuffleMPerIter * CShuffleNPerIter / BlockSize;

            constexpr index_t CShuffleNLaneSlice = GetVectorSize();
            constexpr index_t CShuffleMLaneSlice = CShuffleLaneSlice / CShuffleNLaneSlice;

            constexpr index_t CShuffleNLaneCluster = CShuffleNPerIter / CShuffleNLaneSlice;

            if constexpr(get_warp_size() % CShuffleNLaneCluster == 0)
            {
                constexpr index_t CShuffleMLaneCluster = get_warp_size() / CShuffleNLaneCluster;

                static_assert(CShuffleMLaneCluster * NumWaves <= CShuffleMPerIter);

                return make_static_tile_distribution(
                    tile_distribution_encoding<
                        sequence<1>,
                        tuple<sequence<CShuffleMLaneSlice, NumWaves, CShuffleMLaneCluster>,
                              sequence<CShuffleNLaneCluster, CShuffleNLaneSlice>>,
                        tuple<sequence<1>, sequence<1, 2>>,
                        tuple<sequence<1>, sequence<2, 0>>,
                        sequence<1, 2>,
                        sequence<0, 1>>{});
            }
            else
            {
                static_assert(false);
            }
        }
        else
        {
            static_assert(false);
        }
    }

    template <typename CDramWindow, typename AccTile, typename ElementwiseOp>
    CK_TILE_HOST_DEVICE auto operator()(CDramWindow& c_dram_window,
                                        const AccTile& acc_tile,
                                        const ElementwiseOp& element_wise_op,
                                        void* p_smem)
    {
        // default wave schedule order
        const index_t wave_id_m = get_warp_id() / NWarps;
        const index_t wave_id_n = get_warp_id() - wave_id_m * NWarps;

        constexpr auto lds_store_block_desc = MakeLdsBlockDescriptor();
        auto lds_store_view                 = make_tensor_view<address_space_enum::lds>(
            static_cast<CDataType*>(p_smem), lds_store_block_desc);

        // lds store, wavewise
        auto lds_store_window = make_tile_window(
            lds_store_view,
            make_tuple(number<MWarpTilePerIter>{}, number<NWarpTilePerIter>{}),
            {number<MWarpTilePerIter>{} * wave_id_m, number<NWarpTilePerIter>{} * wave_id_n});

        // lds load, blockwise
        constexpr auto lds_load_block_desc = MakeLdsBlockDescriptor<GetVectorSize()>();
        auto lds_load_view                 = make_tensor_view<address_space_enum::lds>(
            static_cast<CDataType*>(p_smem), lds_load_block_desc);
        auto lds_load_window =
            make_tile_window(lds_load_view,
                             make_tuple(number<MBlockTilePerIter>{}, number<NBlockTilePerIter>{}),
                             {0, 0});

        // outer loop
        using SFC =
            space_filling_curve<sequence<MWarpIter, NWarpIter>, sequence<0, 1>, sequence<1, 1>>;

        constexpr index_t num_access = SFC::get_num_of_access();

        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        static_for<0, num_access, 1>{}([&](auto i) {
            constexpr auto idx         = SFC::get_index(i);
            constexpr auto m_warp_iter = number<idx.at(number<0>{})>{};
            constexpr auto n_warp_iter = number<idx.at(number<1>{})>{};

            CWarpTensor c_warp_tensor;

            // copy one iter data
            c_warp_tensor.get_thread_buffer() = acc_tile.get_y_sliced_thread_data(
                merge_sequences(sequence<m_warp_iter, n_warp_iter>{}, c_warp_y_index_zeros),
                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

            const auto c_warp_tensor_casted = cast_tile<CDataType>(c_warp_tensor);

            // lds store
            ck_tile::block_sync_lds();
            store_tile(lds_store_window, c_warp_tensor_casted);
            ck_tile::block_sync_lds();

            // lds load
            auto c_out_tensor =
                load_tile(make_tile_window(lds_load_window, MakeCShuffleDramDistribution()));

            tile_elementwise_inout(element_wise_op, c_out_tensor, c_out_tensor);

            // global store
            store_tile(c_dram_window, c_out_tensor);

            if constexpr(i != num_access - 1)
            {
                constexpr auto step = SFC::get_forward_step(i);
                move_tile_window(c_dram_window,
                                 {step.at(number<0>{}) * MBlockTilePerIter,
                                  step.at(number<1>{}) * NBlockTilePerIter});
            }
        });
    }
};

} // namespace ck_tile
