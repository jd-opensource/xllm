// SPDX-License-Identifier: MIT
// Copyright (c) 2025, , Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/host.hpp"
#include "../grouped_gemm.hpp"

template <typename GemmConfig,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType>
int run_grouped_gemm_c_impl(const ck_tile_dcu_grouped_gemm_desc* descs,
                            int group_count,
                            void* workspace,
                            hipStream_t stream,
                            int warmup,
                            int repeat,
                            float* avg_ms)
{
    if(descs == nullptr || group_count <= 0 || avg_ms == nullptr)
    {
        return -1;
    }

    const int k_batch = descs[0].k_batch;
    const int num_d   = descs[0].num_d_tensors;
    if(k_batch <= 0)
    {
        return -2;
    }

    bool use_multi_d = (num_d > 0 && descs[0].d_ptrs != nullptr && descs[0].stride_Ds != nullptr);

    if(use_multi_d && num_d != 1 && num_d != 2)
    {
        return -6; // only 1 or 2 D tensors supported
    }

    ck_tile::DeviceMem owned_workspace;
    void* kargs_ptr = workspace;
    const auto stream_cfg =
        ck_tile::stream_config{stream, repeat > 0, 0, warmup, repeat > 0 ? repeat : 1};

    if(use_multi_d)
    {
        if(num_d == 1)
        {
            // Single-D bias: C = GEMM(A,B) + D0
            std::vector<ck_tile::GroupedGemmHostArgsImpl<1>> gemm_descs_vec;
            gemm_descs_vec.reserve(group_count);
            std::vector<ck_tile::GemmTransKernelArgImpl<1>> kargs;
            kargs.reserve(group_count);
            for(int i = 0; i < group_count; ++i)
            {
                const auto& arg = descs[i];
                if(arg.a_ptr == nullptr || arg.b_ptr == nullptr || arg.c_ptr == nullptr || arg.M <= 0 ||
                   arg.N <= 0 || arg.K <= 0 || arg.k_batch != k_batch || arg.num_d_tensors != num_d ||
                   arg.d_ptrs == nullptr || arg.stride_Ds == nullptr)
                {
                    return -3;
                }
                std::array<const void*, 1> ds_ptrs = {arg.d_ptrs[0]};
                std::array<ck_tile::index_t, 1> ds_strides = {arg.stride_Ds[0]};
                gemm_descs_vec.emplace_back(arg.a_ptr,
                                            arg.b_ptr,
                                            ds_ptrs,
                                            arg.c_ptr,
                                            arg.k_batch,
                                            arg.M,
                                            arg.N,
                                            arg.K,
                                            arg.stride_A,
                                            arg.stride_B,
                                            ds_strides,
                                            arg.stride_C);
                kargs.emplace_back(ck_tile::UniversalGemmKernelArgs<1, 1, 1>{{arg.a_ptr},
                                                                              {arg.b_ptr},
                                                                              ds_ptrs,
                                                                              arg.c_ptr,
                                                                              arg.M,
                                                                              arg.N,
                                                                              arg.K,
                                                                              {arg.stride_A},
                                                                              {arg.stride_B},
                                                                              ds_strides,
                                                                              arg.stride_C,
                                                                              arg.k_batch});
            }

            if(kargs_ptr == nullptr)
            {
                owned_workspace.Realloc(group_count * sizeof(ck_tile::GemmTransKernelArgImpl<1>));
                kargs_ptr = owned_workspace.GetDeviceBuffer();
            }
            HIP_CHECK_ERROR(
                hipMemcpyWithStream(kargs_ptr,
                                    kargs.data(),
                                    kargs.size() * sizeof(ck_tile::GemmTransKernelArgImpl<1>),
                                    hipMemcpyHostToDevice,
                                    stream_cfg.stream_id_));

            *avg_ms = grouped_gemm<GemmConfig,
                                   ADataType,
                                   BDataType,
                                   ck_tile::tuple<CDataType>,
                                   AccDataType,
                                   CDataType,
                                   ALayout,
                                   BLayout,
                                   ck_tile::tuple<CLayout>,
                                   CLayout,
                                   ck_tile::element_wise::Add>(
                gemm_descs_vec,
                stream_cfg,
                kargs_ptr);
        }
        else
        {
            // Two-D: C = GEMM(A,B) op D0 op D1
        std::vector<ck_tile::GroupedGemmHostArgsImpl<2>> gemm_descs_vec;
        gemm_descs_vec.reserve(group_count);
        std::vector<ck_tile::GemmTransKernelArgImpl<2>> kargs;
        kargs.reserve(group_count);
        for(int i = 0; i < group_count; ++i)
        {
            const auto& arg = descs[i];
            if(arg.a_ptr == nullptr || arg.b_ptr == nullptr || arg.c_ptr == nullptr || arg.M <= 0 ||
               arg.N <= 0 || arg.K <= 0 || arg.k_batch != k_batch || arg.num_d_tensors != num_d ||
               arg.d_ptrs == nullptr || arg.stride_Ds == nullptr)
            {
                return -3;
            }
            std::array<const void*, 2> ds_ptrs = {arg.d_ptrs[0], arg.d_ptrs[1]};
            std::array<ck_tile::index_t, 2> ds_strides = {arg.stride_Ds[0], arg.stride_Ds[1]};
            gemm_descs_vec.emplace_back(arg.a_ptr,
                                        arg.b_ptr,
                                        ds_ptrs,
                                        arg.c_ptr,
                                        arg.k_batch,
                                        arg.M,
                                        arg.N,
                                        arg.K,
                                        arg.stride_A,
                                        arg.stride_B,
                                        ds_strides,
                                        arg.stride_C);
            kargs.emplace_back(ck_tile::UniversalGemmKernelArgs<1, 1, 2>{{arg.a_ptr},
                                                                          {arg.b_ptr},
                                                                          ds_ptrs,
                                                                          arg.c_ptr,
                                                                          arg.M,
                                                                          arg.N,
                                                                          arg.K,
                                                                          {arg.stride_A},
                                                                          {arg.stride_B},
                                                                          ds_strides,
                                                                          arg.stride_C,
                                                                          arg.k_batch});
        }

        if(kargs_ptr == nullptr)
        {
            owned_workspace.Realloc(group_count * sizeof(ck_tile::GemmTransKernelArgImpl<2>));
            kargs_ptr = owned_workspace.GetDeviceBuffer();
        }
        HIP_CHECK_ERROR(
            hipMemcpyWithStream(kargs_ptr,
                                kargs.data(),
                                kargs.size() * sizeof(ck_tile::GemmTransKernelArgImpl<2>),
                                hipMemcpyHostToDevice,
                                stream_cfg.stream_id_));

        *avg_ms = grouped_gemm<GemmConfig,
                               ADataType,
                               BDataType,
                               ck_tile::tuple<CDataType, CDataType>,
                               AccDataType,
                               CDataType,
                               ALayout,
                               BLayout,
                               ck_tile::tuple<CLayout, CLayout>,
                               CLayout,
                               ck_tile::element_wise::AddAdd>(
            gemm_descs_vec,
            stream_cfg,
            kargs_ptr);
        } // end else (num_d == 2)
    }
    else
    {
        std::vector<ck_tile::GemmTransKernelArg> kargs;
        kargs.reserve(group_count);
        for(int i = 0; i < group_count; ++i)
        {
            const auto& arg = descs[i];
            if(arg.a_ptr == nullptr || arg.b_ptr == nullptr || arg.c_ptr == nullptr || arg.M <= 0 ||
               arg.N <= 0 || arg.K <= 0 || arg.k_batch != k_batch)
            {
                return -3;
            }

            kargs.emplace_back(ck_tile::UniversalGemmKernelArgs<>{{arg.a_ptr},
                                                                  {arg.b_ptr},
                                                                  {},
                                                                  arg.c_ptr,
                                                                  arg.M,
                                                                  arg.N,
                                                                  arg.K,
                                                                  {arg.stride_A},
                                                                  {arg.stride_B},
                                                                  {},
                                                                  arg.stride_C,
                                                                  arg.k_batch});
        }

        if(kargs_ptr == nullptr)
        {
            owned_workspace.Realloc(ck_tile_dcu_grouped_gemm_workspace_size(group_count));
            kargs_ptr = owned_workspace.GetDeviceBuffer();
        }
        HIP_CHECK_ERROR(hipMemcpyWithStream(kargs_ptr,
                                            kargs.data(),
                                            kargs.size() * sizeof(ck_tile::GemmTransKernelArg),
                                            hipMemcpyHostToDevice,
                                            stream_cfg.stream_id_));

        *avg_ms = grouped_gemm_tileloop<GemmConfig,
                                        ALayout,
                                        BLayout,
                                        CLayout,
                                        ADataType,
                                        BDataType,
                                        AccDataType,
                                        CDataType>(
            stream_cfg, group_count, kargs_ptr, k_batch > 1);
    }
    return 0;
}

template <typename GemmConfig, typename PrecType>
int dispatch_grouped_gemm_c_layout(const ck_tile_dcu_grouped_gemm_desc* descs,
                                   int group_count,
                                   char a_layout,
                                   char b_layout,
                                   void* workspace,
                                   hipStream_t stream,
                                   int warmup,
                                   int repeat,
                                   float* avg_ms)
{
    using Row   = ck_tile::tensor_layout::gemm::RowMajor;
    using Col   = ck_tile::tensor_layout::gemm::ColumnMajor;
    using Types = GemmTypeConfig<PrecType>;

    using ADataType   = typename Types::ADataType;
    using BDataType   = typename Types::BDataType;
    using AccDataType = typename Types::AccDataType;
    using CDataType   = typename Types::CDataType;

    if(a_layout == 'R' && b_layout == 'C')
    {
        return run_grouped_gemm_c_impl<GemmConfig,
                                       Row,
                                       Col,
                                       Row,
                                       ADataType,
                                       BDataType,
                                       AccDataType,
                                       CDataType>(descs,
                                                  group_count,
                                                  workspace,
                                                  stream,
                                                  warmup,
                                                  repeat,
                                                  avg_ms);
    }
    if(a_layout == 'R' && b_layout == 'R')
    {
        return run_grouped_gemm_c_impl<GemmConfig,
                                       Row,
                                       Row,
                                       Row,
                                       ADataType,
                                       BDataType,
                                       AccDataType,
                                       CDataType>(descs,
                                                  group_count,
                                                  workspace,
                                                  stream,
                                                  warmup,
                                                  repeat,
                                                  avg_ms);
    }
    if(a_layout == 'C' && b_layout == 'R')
    {
        return run_grouped_gemm_c_impl<GemmConfig,
                                       Col,
                                       Row,
                                       Row,
                                       ADataType,
                                       BDataType,
                                       AccDataType,
                                       CDataType>(descs,
                                                  group_count,
                                                  workspace,
                                                  stream,
                                                  warmup,
                                                  repeat,
                                                  avg_ms);
    }
    if(a_layout == 'C' && b_layout == 'C')
    {
        return run_grouped_gemm_c_impl<GemmConfig,
                                       Col,
                                       Col,
                                       Row,
                                       ADataType,
                                       BDataType,
                                       AccDataType,
                                       CDataType>(descs,
                                                  group_count,
                                                  workspace,
                                                  stream,
                                                  warmup,
                                                  repeat,
                                                  avg_ms);
    }
    return -4;
}

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
float grouped_gemm(const std::vector<ck_tile::GroupedGemmHostArgsImpl<DsDataType::size()>>&
                       gemm_descs,
                   const ck_tile::stream_config& s,
                   void* kargs_ptr)
{
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>>;

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                   GemmConfig::TileParitionerGroupNum,
                                                   GemmConfig::TileParitionerM01>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                                                 GemmConfig::kPadN,
                                                                 GemmConfig::kPadK,
                                                                 GemmConfig::DoubleSmemBuffer,
                                                                 ALayout,
                                                                 BLayout,
                                                                 CLayout,
                                                                 GemmConfig::TransposeC>;

    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, GemmUniversalTraits>;

    using BaseGemmPipeline = typename PipelineTypeTraits<
        GemmConfig::Pipeline>::template UniversalGemmPipeline<GemmPipelineProblem>;

    const ck_tile::index_t k_grain = gemm_descs[0].k_batch * GemmConfig::K_Tile;
    const ck_tile::index_t K_split = (gemm_descs[0].K + k_grain - 1) / k_grain * GemmConfig::K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);


    float ave_time{0};

    const auto Run = [&](const auto has_hot_loop_,
                         const auto tail_number_,
                         const auto memory_operation_) {
        constexpr bool has_hot_loop_v   = has_hot_loop_.value;
        constexpr auto tail_number_v    = tail_number_.value;
        constexpr auto scheduler        = GemmConfig::Scheduler;
        constexpr auto memory_operation = memory_operation_.value;

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler,
                                                                           has_hot_loop_v,
                                                                           tail_number_v>;


        using GemmPipeline = typename PipelineTypeTraits<
            GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;
        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             CLayout,
                                             CDEElementWise,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             GemmConfig::M_Warp,
                                             GemmConfig::N_Warp,
                                             GemmConfig::M_Warp_Tile,
                                             GemmConfig::N_Warp_Tile,
                                             GemmConfig::K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC,
                                             memory_operation>>;

        using Kernel = ck_tile::GroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKargs(gemm_descs);(void)kargs;
        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Kernel arguments not supported!");
        }

        constexpr dim3 blocks = Kernel::BlockSize();
        const dim3 grids  = Kernel::GridSize(gemm_descs);

        HIP_CHECK_ERROR(hipMemcpyWithStream(kargs_ptr,
                                            kargs.data(),
                                            get_workspace_size(gemm_descs),
                                            hipMemcpyHostToDevice,
                                            s.stream_id_));

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel: " << Kernel::GetName() << " with args:" << " grid: {"
                      << grids.x << ", " << grids.y << ", " << grids.z << "}" << ", blocks: {"
                      << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << std::endl;
        }

        ave_time =
            ck_tile::launch_kernel(s,
                                   ck_tile::make_kernel<blocks.x, GemmConfig::kBlockPerCu>(
                                       Kernel{},
                                       grids,
                                       blocks,
                                       0,
                                       ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                                       gemm_descs.size()));

        return ave_time;
    };

    const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {
        if(gemm_descs[0].k_batch != 1)
        {
            throw std::runtime_error("multi-D grouped_gemm does not support SplitK");
        }

        Run(has_hot_loop_,
            tail_number_,
            ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                       ck_tile::memory_operation_enum::set>{});
    };

    BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);

    return ave_time;
}

template <typename GemmConfig,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType>
float grouped_gemm_tileloop(const ck_tile::stream_config& s,
                            const ck_tile::index_t num_groups,
                            void* kargs_ptr,
                            bool splitk)
{
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::
            sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>>;
    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                   GemmConfig::TileParitionerGroupNum,
                                                   GemmConfig::TileParitionerM01>;

    using GemmUniversalTraits =
        ck_tile::PersistentTileGemmUniversalTraits<GemmConfig::kPadM,
                               GemmConfig::kPadN,
                               GemmConfig::kPadK,
                               GemmConfig::DoubleSmemBuffer,
                               ALayout,
                               BLayout,
                               CLayout,
                               false,
                               false,
                               false>;

    float ave_time{0};

    const auto Run = [&](const auto memory_operation_) {
        constexpr auto scheduler        = GemmConfig::Scheduler;
        constexpr auto memory_operation = memory_operation_.value;

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler>;

        using GemmPipeline = typename PipelineTypeTraits<
            GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;
        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             ck_tile::tuple<>,
                                             AccDataType,
                                             CDataType,
                                             ck_tile::tuple<>,
                                             CLayout,
                                             ck_tile::element_wise::PassThrough,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             GemmConfig::M_Warp,
                                             GemmConfig::N_Warp,
                                             GemmConfig::M_Warp_Tile,
                                             GemmConfig::N_Warp_Tile,
                                             GemmConfig::K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC,
                                             memory_operation>>;
        using Kernel      = ck_tile::GroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        constexpr dim3 blocks = Kernel::BlockSize();
        const dim3 grids  = Kernel::MaxOccupancyGridSize(s);

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel: " << Kernel::GetName() << " with args:" << " grid: {"
                      << grids.x << ", " << grids.y << ", " << grids.z << "}" << ", blocks: {"
                      << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << std::endl;
        }

        ave_time =
            ck_tile::launch_kernel(s,
                                   ck_tile::make_kernel<blocks.x, GemmConfig::kBlockPerCu>(
                                       Kernel{},
                                       grids,
                                       blocks,
                                       0,
                                       ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                                       num_groups));

        return ave_time;
    };

    if constexpr(std::is_same_v<CDataType, ck_tile::half_t> ||
                 std::is_same_v<CDataType, ck_tile::bf16_t>)
    {
        if(splitk)
        {
            throw std::runtime_error("fp16/bf16 grouped_gemm output does not support SplitK");
        }

        Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                       ck_tile::memory_operation_enum::set>{});
    }
    else if(!splitk)
    {
        Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                       ck_tile::memory_operation_enum::set>{});
    }
    else
    {
        Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                       ck_tile::memory_operation_enum::atomic_add>{});
    }

    return ave_time;
}

#include "../run_grouped_gemm_example.inc"

template <typename GemmConfig, typename PrecType>
int run_gemm_example_prec_type(std::string a_layout, std::string b_layout, int argc, char* argv[])
{
    using Row   = ck_tile::tensor_layout::gemm::RowMajor;
    using Col   = ck_tile::tensor_layout::gemm::ColumnMajor;
    using Types = GemmTypeConfig<PrecType>;
    using ADataType   = typename Types::ADataType;
    using BDataType   = typename Types::BDataType;
    using AccDataType = typename Types::AccDataType;
    using CDataType   = typename Types::CDataType;
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        return -1;
    }
    const bool multiple_d      = arg_parser.get_bool("multiple_d");
    const std::string md_op_str = arg_parser.get_str("multiple_d_op");
    const bool md_multiply      = (md_op_str == "multiply");
    const bool bias             = arg_parser.get_bool("bias");

    auto run_multi_d = [&](auto a_layout_tag, auto b_layout_tag, auto c_layout_tag) {
        if(md_multiply)
        {
            return run_grouped_gemm_multiple_d_example_with_layouts<GemmConfig,
                                                                    ADataType,
                                                                    BDataType,
                                                                    CDataType,
                                                                    AccDataType,
                                                                    ck_tile::element_wise::MultiplyMultiply>(
                argc, argv, a_layout_tag, b_layout_tag, c_layout_tag);
        }
        return run_grouped_gemm_multiple_d_example_with_layouts<GemmConfig,
                                                                ADataType,
                                                                BDataType,
                                                                CDataType,
                                                                AccDataType,
                                                                ck_tile::element_wise::AddAdd>(
            argc, argv, a_layout_tag, b_layout_tag, c_layout_tag);
    };

    auto run_bias = [&](auto a_layout_tag, auto b_layout_tag, auto c_layout_tag) {
        return run_grouped_gemm_bias_example_with_layouts<GemmConfig,
                                                          ADataType,
                                                          BDataType,
                                                          CDataType,
                                                          AccDataType>(
            argc, argv, a_layout_tag, b_layout_tag, c_layout_tag);
    };

#if defined(CK_TILE_GROUPED_GEMM_FAST_RC_ONLY)
    if(a_layout == "R" && b_layout == "C")
    {
        if(bias)
        {
            return run_bias(Row{}, Col{}, Row{});
        }
        if(multiple_d)
        {
            return run_multi_d(Row{}, Col{}, Row{});
        }
        return run_grouped_gemm_example_with_layouts<GemmConfig,
                                                     ADataType,
                                                     BDataType,
                                                     CDataType,
                                                     AccDataType>(argc, argv, Row{}, Col{}, Row{});
    }
    throw std::runtime_error("Fast grouped_gemm target only supports A row-major and B column-major.");
#else

    if(a_layout == "R" && b_layout == "C")
    {
        if(bias)
        {
            return run_bias(Row{}, Col{}, Row{});
        }
        if(multiple_d)
        {
            return run_multi_d(Row{}, Col{}, Row{});
        }
        return run_grouped_gemm_example_with_layouts<GemmConfig,
                                                     ADataType,
                                                     BDataType,
                                                     CDataType,
                                                     AccDataType>(argc, argv, Row{}, Col{}, Row{});
    }
    else if(a_layout == "R" && b_layout == "R")
    {
        if(bias || multiple_d)
        {
            throw std::runtime_error("bias/multiple-D currently supports A row-major and B column-major.");
        }
        return run_grouped_gemm_example_with_layouts<GemmConfig,
                                                     ADataType,
                                                     BDataType,
                                                     CDataType,
                                                     AccDataType>(argc, argv, Row{}, Row{}, Row{});
    }
    else if(a_layout == "C" && b_layout == "R")
    {
        if(bias || multiple_d)
        {
            throw std::runtime_error("bias/multiple-D currently supports A row-major and B column-major.");
        }
        return run_grouped_gemm_example_with_layouts<GemmConfig,
                                                     ADataType,
                                                     BDataType,
                                                     CDataType,
                                                     AccDataType>(argc, argv, Col{}, Row{}, Row{});
    }
    else if(a_layout == "C" && b_layout == "C")
    {
        if(bias || multiple_d)
        {
            throw std::runtime_error("bias/multiple-D currently supports A row-major and B column-major.");
        }
        return run_grouped_gemm_example_with_layouts<GemmConfig,
                                                     ADataType,
                                                     BDataType,
                                                     CDataType,
                                                     AccDataType>(argc, argv, Col{}, Col{}, Row{});
    }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A and B tensors!");
    }
#endif
}
