// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include <array>
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

namespace ck_tile {

// this epilogue aiming to store a matrix with different layout from the shared memory to the global
// memory.
template <typename ADataType_,
          typename BDataType_,
          typename DsDataType_,
          typename AccDataType_,
          typename ODataType_,
          typename DsLayout_,
          typename ELayout_,
          typename CDElementwise_,
          index_t kM_,
          index_t kN_,
          index_t MWave_,
          index_t NWave_,
          index_t MPerXdl_,
          index_t NPerXdl_,
          index_t KPerXdl_,
          bool isCTransposed_,
          memory_operation_enum MemoryOperation_,
          bool UseABScale_ = false,
          index_t kNumWaveGroups_ = 1,
          bool FixedVectorSize_   = false,
          index_t VectorSizeC_    = 1,
          bool TiledMMAPermuteN_  = false>
struct Moe2StageEpilogueProblem
{
    using ADataType                                        = remove_cvref_t<ADataType_>;
    using BDataType                                        = remove_cvref_t<BDataType_>;
    using AccDataType                                      = remove_cvref_t<AccDataType_>;
    using ODataType                                        = remove_cvref_t<ODataType_>;        //half
    using DsDataType                                       = remove_cvref_t<DsDataType_>;
    using DsLayout                                         = remove_cvref_t<DsLayout_>;
    using ELayout                                          = remove_cvref_t<ELayout_>;
    using CDElementwise                                    = remove_cvref_t<CDElementwise_>;
    static constexpr index_t kBlockSize                    = MWave_ * NWave_ * get_warp_size();
    static constexpr index_t kMPerBlock                    = kM_;
    static constexpr index_t kNPerBlock                    = kN_;
    static constexpr index_t MWave                         = MWave_;        //2
    static constexpr index_t NWave                         = NWave_;        //2
    static constexpr index_t MPerXdl                       = MPerXdl_;      //16
    static constexpr index_t NPerXdl                       = NPerXdl_;      //64
    static constexpr index_t KPerXdl                       = KPerXdl_;      //32
    static constexpr index_t isCTransposed                 = isCTransposed_;
    static constexpr memory_operation_enum MemoryOperation = MemoryOperation_;
    static constexpr bool UseABScale                       = UseABScale_;
    static constexpr bool FixedVectorSize                  = FixedVectorSize_;      //false
    static constexpr index_t VectorSizeC                   = VectorSizeC_;
    static constexpr bool TiledMMAPermuteN                 = TiledMMAPermuteN_;
    static constexpr index_t kNumWaveGroups                = kNumWaveGroups_;
    static constexpr index_t NumDTensor                    = DsDataType::size();

    static_assert(NumDTensor == DsLayout::size(),
                  "The size of DsDataType and DsLayout should be the same");
};

template <typename Problem_, int moe_stage_ = 2, typename Policy_ = void>
struct Moe2StageEpilogue
{
    using Problem     = remove_cvref_t<Problem_>;
    using ADataType   = remove_cvref_t<typename Problem::ADataType>;
    using BDataType   = remove_cvref_t<typename Problem::BDataType>;
    using AccDataType = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType   = remove_cvref_t<typename Problem::ODataType>;
    using DsDataType  = remove_cvref_t<typename Problem::DsDataType>;
    using DsLayout    = remove_cvref_t<typename Problem::DsLayout>;
    using ATypeToUse =
        std::conditional_t<std::is_same_v<ADataType, pk_int4_t>, BDataType, ADataType>;
    // Used for weight-only quantization kernel, B would be dequantized to the same data type as A
    using BTypeToUse =
        std::conditional_t<std::is_same_v<BDataType, pk_int4_t>, ADataType, BDataType>;
    using ELayout       = remove_cvref_t<typename Problem::ELayout>;
    using CDElementwise = remove_cvref_t<typename Problem::CDElementwise>;
    static constexpr memory_operation_enum MemoryOperation = Problem::MemoryOperation;
    static constexpr index_t kBlockSize                    = Problem::kBlockSize;
    static constexpr index_t kMPerBlock                    = Problem::kMPerBlock;
    static constexpr index_t kNPerBlock                    = Problem::kNPerBlock;
    static constexpr index_t MWave                         = Problem::MWave;        //2
    static constexpr index_t NWave                         = Problem::NWave;        //2
    static constexpr index_t MPerXdl                       = Problem::MPerXdl;      //16
    static constexpr index_t NPerXdl                       = Problem::NPerXdl;      //64
    static constexpr index_t KPerXdl                       = Problem::KPerXdl;      //16 for half
    static constexpr index_t isCTransposed                 = Problem::isCTransposed;
    static constexpr bool FixedVectorSize                  = Problem::FixedVectorSize;  //false
    static constexpr bool TiledMMAPermuteN                 = Problem::TiledMMAPermuteN;
    static constexpr index_t VectorSizeC                   = Problem::VectorSizeC;
    static constexpr index_t MPerIteration                 = MPerXdl * MWave;       //64
    static constexpr index_t NPerIteration                 = NPerXdl * NWave;       //64
    static constexpr index_t NumDTensor                    = Problem::NumDTensor;
    static constexpr index_t MRepeat                       = kMPerBlock / (MPerXdl * MWave);
    static constexpr index_t NRepeat                       = kNPerBlock / (NPerXdl * NWave);

    static_assert(NumDTensor == DsLayout::size(),
                  "The size of DsDataType and DsLayout should be the same");
    /**
     * @brief Get the vector store size for C tensor.
     *
     * @note The vector store size for output C tensor would depend on multiple factors
     *       like its data layout and warp gemm C transposition. In general it would
     *       be the number of consecutive elements in contiguous C dimension hold by
     *       single thread.
     *
     * @return The vector store size for C tensor.
     */
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeC()
    {
        if constexpr(FixedVectorSize)
        {
            return VectorSizeC;
        }
        constexpr index_t max_vector_size = 16;
        if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
        {
            return std::min(static_cast<int>(NPerIteration),
                            static_cast<int>(max_vector_size / sizeof(ODataType)));     // min(64, 16/4=4)=4
        }
        else if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::ColumnMajor>)
        {
            return std::min(static_cast<int>(MPerIteration),
                            static_cast<int>(max_vector_size / sizeof(ODataType)));
        }
        else
        {
            static_assert(false, "Unsupported ELayout!");
        }
    }

    /**
     * @brief Get the vector store size for Di tensor.
     *
     * @return The vector store size for Di tensor.
     */
    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeD(number<I> index)
    {
        constexpr index_t max_vector_size = 16;
        using DiDataType = remove_cvref_t<std::tuple_element_t<index.value, DsDataType>>;
        using DiLayout   = remove_cvref_t<std::tuple_element_t<index.value, DsLayout>>;
        if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
        {
            return std::min(static_cast<int>(NPerIteration),
                            static_cast<int>(max_vector_size / sizeof(DiDataType)));
        }
        else if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::ColumnMajor>)
        {
            return std::min(static_cast<int>(MPerIteration),
                            static_cast<int>(max_vector_size / sizeof(DiDataType)));
        }
        else
        {
            static_assert(false, "Unsupported DLayout!");
        }
        return max_vector_size / sizeof(DiDataType);
    }
    /**
     * @brief Shuffle tile configuration parameters
     *
     * @details These parameters control the number of XDL tiles processed per wave in each shuffle
     * iteration:
     * - NumMXdlPerWavePerShuffle: Number of XDL tiles in M dimension processed per wave
     * - NumNXdlPerWavePerShuffle: Number of XDL tiles in N dimension processed per wave
     */
    static constexpr auto shuffle_tile_tuple = [] {
        constexpr index_t elem_per_thread = MPerXdl * NPerXdl / get_warp_size();    //16*64/64=16

        //由于output的token是sorting后离散的，SFC访问时无法保证连续存储，因此这里要求elem_per_thread >= GetVectorSizeC()
        static_assert(elem_per_thread >= GetVectorSizeC(), "Make sure elem_per_thread >= GetVectorSizeC().");

        if constexpr(elem_per_thread >= GetVectorSizeC())   // 16 >= 4
        {
            return std::make_tuple(1, 1);
        }
        else
        {
            constexpr index_t num_xdl_shuffles = GetVectorSizeC() / elem_per_thread;
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                static_assert((kMPerBlock % (MPerXdl * MWave) == 0) &&
                                  (kMPerBlock % num_xdl_shuffles == 0),
                              "kMPerBlock must be divisible by MPerXdl*MWave and "
                              "num_xdl_shuffles for Moe2StageEpilogue");
                return std::make_tuple(min(num_xdl_shuffles, kMPerBlock / (MPerXdl * MWave)), 1);
            }
            else
            {
                static_assert((kNPerBlock % (NPerXdl * NWave) == 0) &&
                                  (kNPerBlock % num_xdl_shuffles == 0),
                              "kNPerBlock must be divisible by NPerXdl*NWave and "
                              "num_xdl_shuffles for Moe2StageEpilogue");
                return std::make_tuple(1, min(num_xdl_shuffles, kNPerBlock / (NPerXdl * NWave)));
            }
        }
    }();
    static constexpr index_t NumMXdlPerWavePerShuffle = std::get<0>(shuffle_tile_tuple);
    static constexpr index_t NumNXdlPerWavePerShuffle = std::get<1>(shuffle_tile_tuple);

    static constexpr auto MNPerIterationShuffle = [] {
        constexpr index_t m_val = MPerXdl * MWave * NumMXdlPerWavePerShuffle;   // 16 * 2 * 1 = 32
        constexpr index_t n_val = NPerXdl * NWave * NumNXdlPerWavePerShuffle;   // 64 * 2 * 1 = 128
        if constexpr(kMPerBlock % m_val != 0 || kNPerBlock % n_val != 0)
            return std::make_tuple(MPerXdl * MWave, NPerXdl * NWave);           // 64 * 2 = 128, 64 * 2 = 128
        else
            return std::make_tuple(m_val, n_val);
    }();
    static constexpr index_t MPerIterationShuffle = std::get<0>(MNPerIterationShuffle);     //32
    static constexpr index_t NPerIterationShuffle = std::get<1>(MNPerIterationShuffle);     //128

    using WG = WarpGemmMmacDispatcher<ATypeToUse,  // half
                                      BTypeToUse,  // half
                                      AccDataType, // AccDataType     float
                                      MPerXdl,     // 16
                                      NPerXdl,     // 64
                                      KPerXdl,     // 32
                                      isCTransposed, // false
                                      MPerXdl / 16, // Mrepeat
                                      NPerXdl / 16, // Nrepeat
                                      1, // Minterleave
                                      1, // Ninterleave
                                      false, //SwizzleA
                                      Problem::UseABScale>;

    using CWarpDstr         = typename WG::CWarpDstr;
    // using CWarpTensor       = typename WG::CWarpTensor;
    // using CWarpDstrEncoding = typename WG::CWarpDstrEncoding;
    using SFC               = space_filling_curve<sequence<kMPerBlock, kNPerBlock>,
                                                  sequence<0, 1>,
                                                  sequence<MPerIterationShuffle, NPerIterationShuffle>>;


    //stage1: 连续写出output，M维度不做要求
    //stage2: output为sorted_token, 离散token_id不可自动偏移写；需计算每个线程访问多少元素，可以使M维度线程不repeat
    static constexpr index_t C_VECT_SIZE = (moe_stage_ == 1) ? GetVectorSizeC() : (NPerIterationShuffle / (kBlockSize / MPerIterationShuffle));
    
    //如果dram_tile_distribution即输出不能保证内存行连续，不可使用thread_raked，如sorted后的moe stage2离散输出
    //因thread_raked会将线程分组，同组会处理连续多行，而GetTileDistributionCoord只会返回首行坐标
    //如果满足M维度无重复，thread_raked与block_raked效果相当
    static constexpr auto distribution_pattern = (moe_stage_ == 1) ? tile_distribution_pattern::thread_raked : tile_distribution_pattern::block_raked;

    using TileEncodingPattern =
        tile_distribution_encoding_pattern_2d<kBlockSize,
                                              MPerIterationShuffle,
                                              NPerIterationShuffle,
                                              C_VECT_SIZE,   
                                              distribution_pattern, 
                                              Problem::kNumWaveGroups>;               

    CK_TILE_HOST_DEVICE static constexpr auto GetTileDistribution()
    {
        return TileEncodingPattern::make_2d_static_tile_distribution();
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetTileDistributionCoord()
    {
        constexpr auto dist = GetTileDistribution();
        return dist.calculate_index();
    }


    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsBlockDescriptor()
    {
        // N is contiguous dimension
        if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
        {
            return make_naive_tensor_descriptor(
                make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
                make_tuple(number<NPerIterationShuffle>{}, number<1>{}));
        }
        // M is contiguous dimension
        else if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::ColumnMajor>)
        {
            return make_naive_tensor_descriptor(
                make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
                make_tuple(number<1>{}, number<MPerIterationShuffle>{}));
        }
        else
        {
            static_assert(false, "Unsupported ELayout!");
        }
    }

    CK_TILE_DEVICE static constexpr auto MakeLdsDistributionEncode()
    {
        constexpr auto block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<NumMXdlPerWavePerShuffle, MWave>,
                                             sequence<NumNXdlPerWavePerShuffle, NWave>>,
                                       tuple<sequence<1, 2>>,
                                       tuple<sequence<1, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};
        constexpr auto block_dstr_encoding = detail::make_embed_tile_distribution_encoding(
            block_outer_dstr_encoding, typename CWarpDstr::DstrEncode{});

        return block_dstr_encoding;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return MPerIterationShuffle * NPerIterationShuffle * sizeof(ODataType);
    }

    template <index_t iAccess, typename LdsTile, typename ScaleM, typename ScaleN>
    CK_TILE_DEVICE void
    scale_tile(LdsTile& lds_tile, ScaleM& scale_m_window, ScaleN& scale_n_window)
    {
        constexpr bool has_scale_m = !std::is_same<ScaleM, EmptyScale>::value;
        constexpr bool has_scale_n = !std::is_same<ScaleN, EmptyScale>::value;


        if constexpr(has_scale_m && has_scale_n)
        {
            const auto scale_m_tile = load_tile(scale_m_window);
            const auto scale_n_tile = load_tile(scale_n_window);
            tile_elementwise_inout(element_wise::MultiDMultiply{},
                                    lds_tile,
                                    lds_tile,
                                    scale_m_tile,
                                    scale_n_tile);
        }
        else if constexpr(has_scale_m)
        {
            const auto scale_m_tile = load_tile(scale_m_window);
            tile_elementwise_inout(element_wise::MultiDMultiply{},
                                    lds_tile,
                                    lds_tile,
                                    scale_m_tile);
        }
        else if constexpr(has_scale_n)
        {
            const auto scale_n_tile = load_tile(scale_n_window);
            tile_elementwise_inout(element_wise::MultiDMultiply{},
                                    lds_tile,
                                    lds_tile,
                                    scale_n_tile);
        }

        constexpr index_t num_access = SFC::get_num_of_access();
        if constexpr(iAccess != num_access - 1)
        {
            constexpr auto step = SFC::get_forward_step(number<iAccess>{});

            if constexpr(has_scale_m)
            {
                move_tile_window(scale_m_window,
                                    {step.at(number<0>{}), step.at(number<1>{})});
            }

            if constexpr(has_scale_n)
            {
                move_tile_window(scale_n_window,
                                    {step.at(number<0>{}), step.at(number<1>{})});
            }
        }
    }


    template <index_t iAccess, typename OAccTile, typename LdsTile>
    CK_TILE_DEVICE void slice_acc_tile(const OAccTile& o_acc_tile, LdsTile& lds_tile)
    {
        constexpr auto idx_y_start = SFC::get_index(number<iAccess>{});

        constexpr auto mIter = number<idx_y_start.at(number<0>{}) / (MPerIterationShuffle)>{};
        constexpr auto nIter = number<idx_y_start.at(number<1>{}) / (NPerIterationShuffle)>{};
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        lds_tile.get_thread_buffer() = o_acc_tile.get_y_sliced_thread_data(
            merge_sequences(
                sequence<mIter * NumMXdlPerWavePerShuffle, nIter * NumNXdlPerWavePerShuffle>{},
                c_warp_y_index_zeros),
            merge_sequences(sequence<NumMXdlPerWavePerShuffle, NumNXdlPerWavePerShuffle>{},
                            c_warp_y_lengths));
    }

    template <typename LdsTile, typename InLdsWindow>
    CK_TILE_DEVICE void cast_lds_tile(LdsTile& lds_tile, InLdsWindow& in_lds_window)
    {
        const auto c_warptile_in_tensor_casted = cast_tile<ODataType>(lds_tile);

        store_tile(in_lds_window, c_warptile_in_tensor_casted);
    }

    template <typename DramWindows, typename COutTensor>
    CK_TILE_DEVICE void apply_d_tensors(DramWindows& d_dram_windows, COutTensor& c_out_tensor)
    {
        const auto ds_tensor = generate_tuple(
            [&](auto idx) { return load_tile(d_dram_windows[idx]); }, number<NumDTensor>{});

        const auto c_ds_tiles = concat_tuple_of_reference(
            tie(c_out_tensor, c_out_tensor),
            generate_tie([&](auto idx) -> const auto& { return ds_tensor[idx]; },
                         number<NumDTensor>{}));

        tile_elementwise_inout_unpack(typename Problem::CDElementwise{}, c_ds_tiles);
    }

    template <typename OutDramWindow, typename COutTensor>
    CK_TILE_DEVICE void store_to_dram(OutDramWindow& out_dram_window,
                                      const COutTensor& c_out_tensor)
    {
        if constexpr(MemoryOperation == memory_operation_enum::set)
        {
            store_tile(out_dram_window, c_out_tensor);
        }
        else
        {
            update_tile(out_dram_window, c_out_tensor);
        }
    }

    /**
     * @brief Move both the output and D tensors windows for the next access.
     */
    template <index_t iAccess, typename OutDramWindow, typename DDramWindows>
    CK_TILE_DEVICE void move_windows(OutDramWindow& out_dram_window, DDramWindows& d_dram_windows)
    {
        constexpr index_t num_access = SFC::get_num_of_access();
        if constexpr(iAccess != num_access - 1)
        {
            constexpr auto step = SFC::get_forward_step(number<iAccess>{});

            // move the output dram window
            move_tile_window(out_dram_window, {step.at(number<0>{}), step.at(number<1>{})});

            // move windows for each of the D matrices (inputs for element-wise)
            static_for<0, NumDTensor, 1>{}([&](auto idx) {
                move_tile_window(d_dram_windows[idx], {step.at(number<0>{}), step.at(number<1>{})});
            });
        }
    }

    // TODO: Check if there would be nicer ways to overload rather than with EmptyScale or nullptr_t
    struct EmptyScale
    {
    };

    template <typename ODramWindow,
              typename OAccTile,
              typename DsDramWindows,
              typename ScaleM                         = EmptyScale,
              typename ScaleN                         = EmptyScale,
              int EnablePermuateN_                    = TiledMMAPermuteN,
              std::enable_if_t<EnablePermuateN_, int> = 0>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   void* /*p_smem*/,
                                   const ScaleM& scale_m = {},
                                   const ScaleN& scale_n = {})
    {
        (void)scale_m;
        (void)scale_n;
        (void)out_dram_window;
        (void)o_acc_tile;
        (void)ds_dram_windows;

        static_assert(false, "TiledMMAPermuteN is not supported yet in Moe2StageEpilogue");
    }

    template <index_t thread_buf_size, typename ShlfTensor>
    CK_TILE_DEVICE void shfl_tensor_regs(ShlfTensor& tensor) {
        using DataType = typename ShlfTensor::DataType;

        static_assert(thread_buf_size % 4 == 0, "thread_buf_size should be multiple of 4");
        constexpr index_t loops = thread_buf_size / 4;
        DataType val_0[4], val_1[4];

        const int group = __lane_id() / 16;
        const int lane = __lane_id() % 16;

        static_for<0, loops, 1>{}([&](int lo) {
            int loop_off = lo * 4;

            for (int i = 0; i < 4; i++) {
                val_0[i] = tensor.get_thread_buffer()[loop_off + i];
            }

            static_for<0, 4, 1>{}([&](int g) {

                val_1[0] = warp_shuffle(val_0[g], lane);
                val_1[1] = warp_shuffle(val_0[g], 16 + lane);
                val_1[2] = warp_shuffle(val_0[g], 32 + lane);
                val_1[3] = warp_shuffle(val_0[g], 48 + lane);

                if (group == g) {
                    tensor.get_thread_buffer()[loop_off + 0] = val_1[0];
                    tensor.get_thread_buffer()[loop_off + 1] = val_1[1];
                    tensor.get_thread_buffer()[loop_off + 2] = val_1[2];
                    tensor.get_thread_buffer()[loop_off + 3] = val_1[3];
                }
            });

        });
    }

    

    template <typename ODramWindow,
              typename OAccTile,
              typename DsDramWindows,
              typename ScaleM                          = EmptyScale,
              typename ScaleN                          = EmptyScale,
              int EnablePermuateN_                     = TiledMMAPermuteN,
              std::enable_if_t<!EnablePermuateN_, int> = 0>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   void* p_smem,
                                   const ScaleM& scale_m = {},
                                   const ScaleN& scale_n = {})
    {
        constexpr auto LdsTileDistr = make_static_tile_distribution(MakeLdsDistributionEncode());

        auto lds_tile = make_static_distributed_tensor<AccDataType>(LdsTileDistr);

        constexpr auto lds_block_desc = MakeLdsBlockDescriptor<Problem>();
        auto o_lds_block              = make_tensor_view<address_space_enum::lds>(
            static_cast<ODataType*>(p_smem), lds_block_desc);

        auto in_lds_window = make_tile_window(
            o_lds_block,
            make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
            {0, 0},
            LdsTileDistr);

        auto out_lds_window = make_tile_window(
            o_lds_block,
            make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
            {0, 0});

        constexpr index_t num_access = SFC::get_num_of_access();    //block包含多少个 配置的block gemm

        static_assert(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>,
                      "Currently, the CShuffle Epilogue only supports the Row Major Output layout");

        constexpr auto dram_tile_distribution = GetTileDistribution();

        static_assert(NumDTensor==0, "Currently, the CShuffle Epilogue only supports no D tensors");
        auto d_dram_windows = generate_tuple(
            [&](auto idx) {
                return make_tile_window(ds_dram_windows[idx], dram_tile_distribution);
            },
            number<NumDTensor>{});

        // constexpr bool has_scales =
        //     !std::is_same<ScaleM, EmptyScale>::value && !std::is_same<ScaleN, EmptyScale>::value;
        constexpr bool has_scale_m = !std::is_same<ScaleM, EmptyScale>::value;
        constexpr bool has_scale_n = !std::is_same<ScaleN, EmptyScale>::value;

        using LdsTileType = remove_cvref_t<decltype(lds_tile)>;

        auto scale_m_window = [&]() {
            if constexpr(has_scale_m)
            {
                return make_tile_window(scale_m,
                                        LdsTileType::get_lengths(),
                                        make_zero_multi_index<LdsTileType::get_num_of_dimension()>(),
                                        LdsTileType::get_tile_distribution());
            }
            else
            {
                return EmptyScale{};
            }
        }();
        auto scale_n_window = [&]() {
            if constexpr(has_scale_n)
            {
                return make_tile_window(scale_n,
                                        LdsTileType::get_lengths(),
                                        make_zero_multi_index<LdsTileType::get_num_of_dimension()>(),
                                        LdsTileType::get_tile_distribution());
            }
            else
            {
                return EmptyScale{};
            }
        }();

        static_for<0, num_access, 1>{}([&](auto iAccess) {
            // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
            //     printf("######### Thread[%d]   iAccess: %d\n", threadIdx.x, i);
            //     i+=1;
            // }

            block_sync_lds();
            slice_acc_tile<iAccess>(o_acc_tile, lds_tile);

            //for hg mmac && no Ninterleave, need to shuffle the tensor registers
            constexpr index_t thread_buf_size = lds_tile.get_thread_buffer_size();
            shfl_tensor_regs<thread_buf_size>(lds_tile);

            if constexpr(has_scale_m || has_scale_n)
            {
                scale_tile<iAccess>(lds_tile, scale_m_window, scale_n_window);
            }

            cast_lds_tile(lds_tile, in_lds_window);
            block_sync_lds();

            auto c_out_tensor = load_tile(make_tile_window(out_lds_window, dram_tile_distribution));

            apply_d_tensors(d_dram_windows, c_out_tensor);
            store_to_dram(out_dram_window, c_out_tensor);
            move_windows<iAccess>(out_dram_window, d_dram_windows);
        });
    }
};

} // namespace ck_tile
