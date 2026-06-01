// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, , Inc. All rights reserved.

#pragma once

#include <cstddef>
#include <string>
#include <tuple>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/elementwise/element_wise_operation.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/utility/json_dump.hpp"
// #include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4.hpp"

// #include "ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_v2.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"


#define CK_TILE_PIPELINE_COMPUTE_V3 1
#define CK_TILE_PIPELINE_MEMORY 2
#define CK_TILE_PIPELINE_COMPUTE_V4 3
#define CK_TILE_PIPELINE_PRESHUFFLE_V2 4

#ifndef CK_TILE_PIPELINE_DEFAULT
#define CK_TILE_PIPELINE_DEFAULT CK_TILE_PIPELINE_COMPUTE_V3
#endif

template <typename PrecType, ck_tile::index_t M_Warp_Tile>
constexpr ck_tile::index_t get_k_warp_tile()
{
#if defined(CK_GFX950_SUPPORT)
    constexpr bool is_8bit_float =
        std::is_same_v<PrecType, ck_tile::fp8_t> || std::is_same_v<PrecType, ck_tile::bf8_t>;
    if constexpr(M_Warp_Tile == 32)
        return is_8bit_float ? 64 : 16;
    else
        return is_8bit_float ? 128 : 32;
#else
    if constexpr(M_Warp_Tile == 32)
        return 16;
    else
        return 32;
#endif
}

template <typename PrecType, ck_tile::index_t M_Warp_Tile>
constexpr ck_tile::index_t get_k_warp_tile_flatmm()
{
#if defined(CK_GFX950_SUPPORT)
    if constexpr(M_Warp_Tile == 32)
        return sizeof(PrecType) == 2 ? 16 : 64;
    else
        return sizeof(PrecType) == 2 ? 32 : 128;
#else
    if constexpr(M_Warp_Tile == 32)
        return sizeof(PrecType) == 2 ? 16 : 32;
    else
        return sizeof(PrecType) == 2 ? 32 : 64;
#endif
}

template <typename DataType>
struct GemmTypeConfig;

template <>
struct GemmTypeConfig<ck_tile::half_t>
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using CDataType   = ck_tile::half_t;
    using AccDataType = float;
};

template <>
struct GemmTypeConfig<ck_tile::bf16_t>
{
    using ADataType   = ck_tile::bf16_t;
    using BDataType   = ck_tile::bf16_t;
    using CDataType   = ck_tile::bf16_t;
    using AccDataType = float;
};

template <>
struct GemmTypeConfig<ck_tile::fp8_t>
{
    using ADataType   = ck_tile::fp8_t;
    using BDataType   = ck_tile::fp8_t;
    using AccDataType = float;
    using CDataType   = float;
};

template <>
struct GemmTypeConfig<ck_tile::bf8_t>
{
    using ADataType   = ck_tile::bf8_t;
    using BDataType   = ck_tile::bf8_t;
    using AccDataType = float;
    using CDataType   = float;
};

template <>
struct GemmTypeConfig<ck_tile::int8_t>
{
    using ADataType   = ck_tile::int8_t;
    using BDataType   = ck_tile::int8_t;
    using AccDataType = int32_t;
    using CDataType   = int32_t;
    // using CDataType   = ck_tile::half_t;
};

template <>
struct GemmTypeConfig<ck_tile::pk_int4_t>
{
    using ADataType   = ck_tile::pk_int4_t;
    using BDataType   = ck_tile::pk_int4_t;
    using AccDataType = int32_t;
    using CDataType   = int32_t;
};

struct GemmConfigBase
{
    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool PermuteA = false;
    static constexpr bool PermuteB = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                         = 1;
    // static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    // static constexpr ck_tile::index_t TileParitionerM01      = 4;
    static constexpr ck_tile::index_t TileParitionerGroupNum = 1;
    static constexpr ck_tile::index_t TileParitionerM01      = 1;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Intrawave;
    static constexpr ck_tile::index_t Pipeline      = CK_TILE_PIPELINE_COMPUTE_V3;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool Preshuffle                = false;
    static constexpr bool Persistent                = true;
    static constexpr bool DoubleSmemBuffer          = false;
};

template <typename PrecType>
struct GemmConfigComputeV3_2 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr bool DoubleSmemBuffer     = false;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V3;

    static constexpr int kBlockPerCu = 1;
};

template <typename PrecType>
struct GemmConfigComputeV4 : public GemmConfigBase
{
    // Compute V4 only support Intrawave scheduler
    // Using the ping pong reader in the lds level
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    // static constexpr ck_tile::index_t K_Warp_Tile = get_k_warp_tile<PrecType, M_Warp_Tile>();
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;

    static constexpr int kBlockPerCu = 1;
};

template <typename PrecType>
struct GemmConfigComputeV5 : public GemmConfigBase
{
    // Compute V5 only support Intrawave scheduler
    // Using the ping pong reader in the lds level
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    static constexpr ck_tile::index_t K_Warp_Tile = 64;

    static constexpr bool Persistent           = true;
    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;
    
    static constexpr int kBlockPerCu = 1;
};

template <typename PrecType>
struct GemmConfigComputeFp8N64K64 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    static constexpr ck_tile::index_t K_Warp_Tile = 64;

    static constexpr bool Persistent           = true;
    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;

    static constexpr int kBlockPerCu = 1;
};

template <typename PrecType>
struct GemmConfigComputeBf8 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 1;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr bool Persistent           = true;
    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;

    static constexpr int kBlockPerCu = 1;
};

template <typename PrecType>
struct GemmConfigComputeBf8V2 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr bool Persistent           = true;
    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;

    static constexpr int kBlockPerCu = 1;
};

template <typename PrecType>
struct GemmConfigComputeBf8K64 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    static constexpr ck_tile::index_t K_Warp_Tile = 64;

    static constexpr bool Persistent           = true;
    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;

    static constexpr int kBlockPerCu = 1;
};

template <typename PrecType>
struct GemmConfigComputeV6 : public GemmConfigBase
{
    // Compute V5 only support Intrawave scheduler
    // Using the ping pong reader in the lds level
    static constexpr ck_tile::index_t M_Tile = 32;
    static constexpr ck_tile::index_t N_Tile = 32;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr bool Persistent           = true;
    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;
    
    static constexpr int kBlockPerCu = 1;
};

template <typename PrecType>
struct GemmConfigComputeV7 : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 64;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 32;

    static constexpr bool Persistent           = true;
    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;

    static constexpr int kBlockPerCu = 1;
};

template <typename PrecType>
struct GemmConfigPreshuffleDecode : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 256 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = get_k_warp_tile_flatmm<PrecType, M_Warp_Tile>();

    static constexpr bool kPadK = true;

    static constexpr int kBlockPerCu           = 1;
    static constexpr auto Scheduler            = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_PRESHUFFLE_V2;
    static constexpr bool Preshuffle           = true;
    static constexpr bool DoubleSmemBuffer     = true;
};

template <typename PrecType>
struct GemmConfigPreshufflePrefill : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = get_k_warp_tile_flatmm<PrecType, M_Warp_Tile>();

    static constexpr int kBlockPerCu           = 2;
    static constexpr auto Scheduler            = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_PRESHUFFLE_V2;
    static constexpr bool Preshuffle           = true;
    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr bool kPadK                = true;
};

template <typename PrecType>
struct GemmConfigPreshuffleDecode_Wmma : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 32 / sizeof(PrecType);
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 256 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool kPadK = true;

    static constexpr int kBlockPerCu           = 1;
    static constexpr auto Scheduler            = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_PRESHUFFLE_V2;
    static constexpr bool Preshuffle           = true;
    static constexpr bool DoubleSmemBuffer     = true;
};

template <ck_tile::index_t PipelineId>
struct PipelineTypeTraits;

// template <>
// struct PipelineTypeTraits<CK_TILE_PIPELINE_MEMORY>
// {
//     template <typename PipelineProblem>
//     using GemmPipeline = ck_tile::GemmPipelineAgBgCrMem<PipelineProblem>;
//     template <typename PipelineProblem>
//     using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrMem<PipelineProblem>;
// };

// template <>
// struct PipelineTypeTraits<CK_TILE_PIPELINE_COMPUTE_V3>
// {
//     template <typename PipelineProblem>
//     using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>;
//     template <typename PipelineProblem>
//     using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV3<PipelineProblem>;
// };

template <>
struct PipelineTypeTraits<CK_TILE_PIPELINE_COMPUTE_V4>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV4<PipelineProblem>;
    template <typename PipelineProblem>
    using UniversalGemmPipeline = ck_tile::BaseGemmPipelineAgBgCrCompV4<PipelineProblem>;
};

template <typename PrecType>
struct GemmConfigComputeInt4 : public GemmConfigBase
{
    // Production config (WT32x64x64 MR2NR4, same shape as GemmConfigComputeV5)
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);  // 128 pk_int4 = 256 logical

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 64;
    static constexpr ck_tile::index_t K_Warp_Tile = 64;

    static constexpr bool Persistent           = true;
    static constexpr bool DoubleSmemBuffer     = true;
    static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_COMPUTE_V4;

    static constexpr int kBlockPerCu = 1;
};

// template <>
// struct PipelineTypeTraits<CK_TILE_PIPELINE_PRESHUFFLE_V2>
// {
//     template <typename PipelineProblem>
//     using GemmPipeline = ck_tile::WeightPreshufflePipelineAGmemBGmemCRegV2<PipelineProblem>;
//     template <typename PipelineProblem>
//     using UniversalGemmPipeline =
//         ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2<PipelineProblem>;
// };

using grouped_gemm_kargs = ck_tile::GroupedGemmHostArgs;

extern "C" {

enum ck_tile_dcu_grouped_gemm_dtype
{
    CK_TILE_DCU_GROUPED_GEMM_FP16 = 0,
    CK_TILE_DCU_GROUPED_GEMM_FP8  = 1,
    CK_TILE_DCU_GROUPED_GEMM_INT8 = 2,
    CK_TILE_DCU_GROUPED_GEMM_BF8  = 3,
    CK_TILE_DCU_GROUPED_GEMM_BF16 = 4,
    CK_TILE_DCU_GROUPED_GEMM_INT4 = 5
};

struct ck_tile_dcu_grouped_gemm_desc
{
    const void* a_ptr;
    const void* b_ptr;
    void* c_ptr;
    int k_batch;
    int M;
    int N;
    int K;
    int stride_A;
    int stride_B;
    int stride_C;
    int num_d_tensors;
    const void* const* d_ptrs;
    const int* stride_Ds;
};

std::size_t ck_tile_dcu_grouped_gemm_workspace_size(int group_count, int num_d_tensors = 0);

int ck_tile_dcu_grouped_gemm_run(const ck_tile_dcu_grouped_gemm_desc* descs,
                                 int group_count,
                                 int dtype,
                                 char a_layout,
                                 char b_layout,
                                 void* workspace,
                                 hipStream_t stream,
                                 int warmup,
                                 int repeat,
                                 float* avg_ms);

} // extern "C"

// Per-dtype C ABI dispatch wrappers (non-template, defined in instances/)
int grouped_gemm_c_run_fp16(const ck_tile_dcu_grouped_gemm_desc* descs,
    int group_count, char a_layout, char b_layout,
    void* workspace, hipStream_t stream,
    int warmup, int repeat, float* avg_ms);
int grouped_gemm_c_run_bf16(const ck_tile_dcu_grouped_gemm_desc* descs,
    int group_count, char a_layout, char b_layout,
    void* workspace, hipStream_t stream,
    int warmup, int repeat, float* avg_ms);
int grouped_gemm_c_run_fp8(const ck_tile_dcu_grouped_gemm_desc* descs,
    int group_count, char a_layout, char b_layout,
    void* workspace, hipStream_t stream,
    int warmup, int repeat, float* avg_ms);
int grouped_gemm_c_run_bf8(const ck_tile_dcu_grouped_gemm_desc* descs,
    int group_count, char a_layout, char b_layout,
    void* workspace, hipStream_t stream,
    int warmup, int repeat, float* avg_ms);
int grouped_gemm_c_run_int8(const ck_tile_dcu_grouped_gemm_desc* descs,
    int group_count, char a_layout, char b_layout,
    void* workspace, hipStream_t stream,
    int warmup, int repeat, float* avg_ms);
int grouped_gemm_c_run_int4(const ck_tile_dcu_grouped_gemm_desc* descs,
    int group_count, char a_layout, char b_layout,
    void* workspace, hipStream_t stream,
    int warmup, int repeat, float* avg_ms);

// Per-dtype example runner wrappers (non-template, defined in instances/)
int grouped_gemm_example_run_fp16(std::string a_layout, std::string b_layout, int argc, char* argv[]);
int grouped_gemm_example_run_bf16(std::string a_layout, std::string b_layout, int argc, char* argv[]);
int grouped_gemm_example_run_fp8(std::string a_layout, std::string b_layout, int argc, char* argv[]);
int grouped_gemm_example_run_fp8_128x64(std::string a_layout,
                                        std::string b_layout,
                                        int argc,
                                        char* argv[]);
int grouped_gemm_example_run_fp8_128x128_k32(std::string a_layout,
                                             std::string b_layout,
                                             int argc,
                                             char* argv[]);
int grouped_gemm_example_run_bf8_k64(std::string a_layout, std::string b_layout, int argc, char* argv[]);
int grouped_gemm_example_run_bf8_128x64(std::string a_layout, std::string b_layout, int argc, char* argv[]);
int grouped_gemm_example_run_bf8_128x128(std::string a_layout, std::string b_layout, int argc, char* argv[]);
int grouped_gemm_example_run_int8_32x32(std::string a_layout, std::string b_layout, int argc, char* argv[]);
int grouped_gemm_example_run_int8(std::string a_layout, std::string b_layout, int argc, char* argv[]);
int grouped_gemm_example_run_int4(std::string a_layout, std::string b_layout, int argc, char* argv[]);

inline std::pair<bool, ck_tile::ArgParser> create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("Ms", "", "M dimensions - empty by default.")
        .insert("Ns", "", "N dimensions - empty by default.")
        .insert("Ks", "", "K dimensions - empty by default.")
        .insert("stride_As", "", "Tensor A strides - it is empty by default.")
        .insert("stride_Bs", "", "Tensor B strides - it is empty by default.")
        .insert("stride_Cs", "", "Tensor C strides - it is empty by default.")
        .insert("a_layout", "R", "A tensor data layout - Row by default.")
        .insert("b_layout", "C", "B tensor data layout - Row by default.")
        .insert("c_layout", "R", "C tensor data layout - Row by default.")
        .insert("validate", "1", "0. No validation, 1. Validation on CPU.")
        .insert("prec", "int8", "data type. fp16/bf16/fp8/bf8/int8/int4")
        .insert("config", "", "optional config selector, e.g. int8_32x32/bf8_128x64/bf8_128x128")
        .insert("warmup", "10", "number of iterations before benchmark the kernel.")
        .insert("repeat", "100", "number of iterations to benchmark the kernel.")
        .insert("group_count", "8", "group count.")
        .insert("kbatch", "1", "kbatch for SplitK")
        .insert("bias", "0", "0: GEMM only, 1: GEMM plus single bias tensor (C=A*B+D)")
        .insert("multiple_d", "0", "0: GEMM only, 1: GEMM plus two D tensors")
        .insert("multiple_d_op", "add", "multi-D epilogue op: add (C+D0+D1) / multiply (C*D0*D1)")
        .insert("json", "0", "0: No Json, 1: Dump Results in Json format")
        .insert("jsonfile", "grouped_gemm.json", "json file name to dump results")
        .insert("debug", "0", "0: no debug prints, 1: print validation summaries")
        .insert("int4_const", "", "optional int4 constant initializer for A/B, e.g. 1")
        .insert("int4_const_a", "", "optional int4 constant initializer for A only")
        .insert("int4_const_b", "", "optional int4 constant initializer for B only")
        .insert("int4_b_one_k", "-1", "debug: set B to one at one logical K and zero elsewhere");

    bool result = arg_parser.parse(argc, argv);
    return std::make_pair(result, arg_parser);
}

inline std::size_t get_workspace_size(const std::vector<grouped_gemm_kargs>& gemm_descs)
{
    return gemm_descs.size() * sizeof(ck_tile::GemmTransKernelArg);
}

template <ck_tile::index_t NumDTensor>
inline std::size_t
get_workspace_size(const std::vector<ck_tile::GroupedGemmHostArgsImpl<NumDTensor>>& gemm_descs)
{
    return gemm_descs.size() * sizeof(ck_tile::GemmTransKernelArgImpl<NumDTensor>);
}

// // template <typename GemmConfig, typename T>
// // auto shuffle_b(const ck_tile::HostTensor<T>& t)
// // {
// //     assert(t.get_lengths().size() == 2);
// //     int n_ = t.get_lengths()[1];
// //     int k_ = t.get_lengths()[0];

// //     if(ck_tile::is_gfx12_supported())
// //     {
// //         // TODO: Please modify it once kABK0PerLane is changed in WmmaTraitsBase<gfx12>
// //         constexpr int divisor      = 2;
// //         constexpr int kABK0PerLane = 2;
// //         ck_tile::HostTensor<T> t_view({n_ / GemmConfig::N_Warp_Tile,
// //                                        GemmConfig::N_Warp_Tile,
// //                                        k_ / GemmConfig::K_Warp_Tile,
// //                                        divisor,
// //                                        kABK0PerLane,
// //                                        GemmConfig::K_Warp_Tile / divisor / kABK0PerLane});
// //         std::copy(t.begin(), t.end(), t_view.begin());
// //         return ck_tile::reference_permute(t_view, {0, 2, 4, 1, 3, 5});
// //     }
// //     else
// //     {
// //         int divisor = 1;
// //         if(ck_tile::is_gfx11_supported())
// //         {
// //             divisor = 1;
// //         }
// //         else
// //         {
// //             assert(is_wave32() == false);
// //             divisor = GemmConfig::N_Warp_Tile == 32 ? 2 : 4;
// //         }
// //         ck_tile::HostTensor<T> t_view({n_ / GemmConfig::N_Warp_Tile,
// //                                        GemmConfig::N_Warp_Tile,
// //                                        k_ / GemmConfig::K_Warp_Tile,
// //                                        divisor,
// //                                        GemmConfig::K_Warp_Tile / divisor});
// //         std::copy(t.begin(), t.end(), t_view.begin());
// //         return ck_tile::reference_permute(t_view, {0, 2, 3, 1, 4});
// //     }
// // }

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
          typename CDEElementWise>
float grouped_gemm(const std::vector<ck_tile::GroupedGemmHostArgsImpl<DsDataType::size()>>& gemm_descs,
                   const ck_tile::stream_config& s,
                   void* kargs_ptr);

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
                            bool splitk = false);
