// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/mmac_block_gemm_asmem_bsmem_creg_v1_default_policy.hpp"

namespace ck_tile {

// A is block window on shared memory
// B is block window on shared memory
// C is block distributed tensor
template <typename Problem_, typename Policy_ = MmacBlockGemmASmemBSmemCRegV1DefaultPolicy>
struct MmacBlockGemmASmemBSmemCRegV1
{
    using Problem        = remove_cvref_t<Problem_>;
    using Policy         = remove_cvref_t<Policy_>;
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    // C += A * B
    template <typename CBlockTensor, typename ABlockWindow, typename BBlockWindow>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ABlockWindow& a_block_window,
                                   const BBlockWindow& b_block_window) const
    {
        static_assert(std::is_same_v<ADataType, typename ABlockWindow::DataType> &&
                          std::is_same_v<BDataType, typename BBlockWindow::DataType> &&
                          std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                      "wrong!");

        constexpr index_t MPerBlock = ABlockWindow{}.get_window_lengths()[number<0>{}];
        constexpr index_t NPerBlock = BBlockWindow{}.get_window_lengths()[number<0>{}];
        constexpr index_t KPerBlock = ABlockWindow{}.get_window_lengths()[number<1>{}];

        static_assert(MPerBlock == BlockGemmShape::kM && NPerBlock == BlockGemmShape::kN &&
                          KPerBlock == BlockGemmShape::kK,
                      "wrong!");

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);//WG::kM come from  warp_gemm_attribute.hpp 
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = KPerBlock / WG::kK;

        constexpr index_t MPerBlockPerIter = MPerBlock / MIterPerWarp;
        constexpr index_t NPerBlockPerIter = NPerBlock / NIterPerWarp;
        constexpr index_t KPerBlockPerIter = KPerBlock / KIterPerWarp;

        //TODO: support get_sub_warp_id for WASP
        const index_t iMWarp = get_warp_id() / NWarp;
        const index_t iNWarp = get_warp_id() % NWarp;

        // construct A-warp-window
        auto a_warp_window_tmp = make_tile_window(
            a_block_window.get_bottom_tensor_view(),
            make_tuple(number<WG::kM>{}, number<WG::kK>{}),
            a_block_window.get_window_origin() + multi_index<2>{iMWarp * WG::kM, 0},
            make_static_tile_distribution(typename WG::AWarpDstrEncoding{}));

#if 0 // FIXME: using array will cause register spill
        array<array<decltype(a_warp_window_tmp), KIterPerWarp>, MIterPerWarp> a_warp_windows{
            {a_warp_window_tmp}};

        for(index_t mIter = 0; mIter < MIterPerWarp; mIter++)
        {
            for(index_t kIter = 0; kIter < KIterPerWarp; kIter++)
            {
                move_tile_window(a_warp_windows(mIter)(kIter),
                                 {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
            }
        }
#else
        statically_indexed_array<
            statically_indexed_array<decltype(a_warp_window_tmp), KIterPerWarp>,
            MIterPerWarp>
            a_warp_windows;

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                a_warp_windows(mIter)(kIter) = a_warp_window_tmp;

                move_tile_window(a_warp_windows(mIter)(kIter),
                                 {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });
#endif

        // construct B-warp-window
        auto b_warp_window_tmp = make_tile_window(
            b_block_window.get_bottom_tensor_view(),
            make_tuple(number<WG::kN>{}, number<WG::kK>{}),
            b_block_window.get_window_origin() + multi_index<2>{iNWarp * WG::kN, 0},
            make_static_tile_distribution(typename WG::BWarpDstrEncoding{}));

#if 0 // FIXME: using array will cause register spill
        array<array<decltype(b_warp_window_tmp), KIterPerWarp>, NIterPerWarp> b_warp_windows{
            {b_warp_window_tmp}};

        for(index_t nIter = 0; nIter < NIterPerWarp; nIter++)
        {
            for(index_t kIter = 0; kIter < KIterPerWarp; kIter++)
            {
                move_tile_window(b_warp_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            }
        }
#else
        statically_indexed_array<
            statically_indexed_array<decltype(b_warp_window_tmp), KIterPerWarp>,
            NIterPerWarp>
            b_warp_windows;

        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                b_warp_windows(nIter)(kIter) = b_warp_window_tmp;

                move_tile_window(b_warp_windows(nIter)(kIter),
                                 {nIter * NPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });
#endif

        using CWarpDstr   = typename WG::CWarpDstr;
        using CWarpTensor = typename WG::CWarpTensor;

        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // hot loop:
        static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                // read A warp tensor from A block window
                const auto a_warp_tensor = load_tile(a_warp_windows(mIter)(kIter));

                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    // read B warp tensor from B Block window
                    const auto b_warp_tensor = load_tile(b_warp_windows(nIter)(kIter));

                    // read C warp tensor from C block tensor
                    CWarpTensor c_warp_tensor;

                    c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                    // warp GEMM
                    WG{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                    // write C warp tensor into C block tensor
                    c_block_tensor.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        c_warp_tensor.get_thread_buffer());
                });
            });
        });
    }

    CK_TILE_DEVICE static constexpr auto MakeCBlockTile()
    {
        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t NPerBlock = BlockGemmShape::kN;

        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);

        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        //Note: Here concat Block_outer_dstr and warp_dstr, for instance M sequence<MIterPerWarp, MWarp, MRepeat, MLane, MInterleave, MPerLane>
        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});
        
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);

        auto c_block_tensor = make_static_distributed_tensor<CDataType>(c_block_dstr);
        return c_block_tensor;
    }

    // C = A * B
    template <typename ABlockTensorTmp, typename BBlockWindow>
    CK_TILE_DEVICE auto operator()(const ABlockTensorTmp& a_block_tensor_tmp,
                                   const BBlockWindow& b_block_window) const
    {
        auto c_block_tensor = MakeCBlockTile();
        operator()(c_block_tensor, a_block_tensor_tmp, b_block_window);
        return c_block_tensor;
    }

    template <typename CBlockTensor>
    CK_TILE_DEVICE auto MakeOuputLayout(const CBlockTensor& c_block_tensor) const{
        constexpr auto config = Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MPerBlock = BlockGemmShape::kM;
        constexpr index_t NPerBlock = BlockGemmShape::kN;

        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();
        constexpr index_t MIterPerWarp = MPerBlock / (MWarp * WG::kM);//WG::kM come from  warp_gemm_attribute.hpp 
        constexpr index_t NIterPerWarp = NPerBlock / (NWarp * WG::kN);

        //import from warp_gemm_impl.hpp and if you want to import from warp_gemm_attribute_mmac.hpp, you need to import WG::WarpGemmAttribute
        using CWarpDstr   = typename WG::CWarpDstr;
        using CWarpOutputDstr = typename WG::CWarpOutputDstr;
        using CWarpTensor = typename WG::CWarpTensor;
        using CWarpOutputTensor = typename WG::CWarpOutputTensor;

        //define output block tensor
        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<MIterPerWarp, MWarp>, sequence<NIterPerWarp, NWarp>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};
        constexpr auto c_block_out_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpOutputDstrEncoding{});
        constexpr auto c_output_block_dstr = make_static_tile_distribution(c_block_out_dstr_encode);
        auto c_block_output_tensor = make_static_distributed_tensor<CDataType>(c_output_block_dstr);

        //tmp buffer use
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        constexpr auto c_warp_output_y_lengths = 
            to_sequence(CWarpOutputDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_output_y_index_zeros = uniform_sequence_gen_t<CWarpOutputDstr::NDimY, 0>{};

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                CWarpTensor c_warp_tensor;
                c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                CWarpOutputTensor c_warp_output_tensor = WG{}.MakeCOutputLayout(c_warp_tensor);
                /////////////////Enable the following code and disable the above line to support N interleave ==2.///////////////
                // CWarpOutputTensor c_warp_output_tensor;
                // constexpr index_t thread_buf_size = c_warp_tensor.get_thread_buffer_size();
                // constexpr index_t Gemm0NInterleave = 2;     // 2/4 interleave for gemm0 in moe
                // static_for<0, thread_buf_size, 1>{}([&](auto i) {
                //     index_t reduce_index =
                //         (i / (Gemm0NInterleave * 4)) * (Gemm0NInterleave * 4) +
                //         (i % (Gemm0NInterleave * 4)) / Gemm0NInterleave +
                //         i % Gemm0NInterleave * 4;
                //     c_warp_output_tensor.get_thread_buffer()[i] = c_warp_tensor.get_thread_buffer()[reduce_index];
                // });
                ///////////////////////////////////////

                c_block_output_tensor.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_output_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_output_y_lengths),
                        c_warp_output_tensor.get_thread_buffer());

            });
        });

        return c_block_output_tensor;

        

    }
};

} // namespace ck_tile
