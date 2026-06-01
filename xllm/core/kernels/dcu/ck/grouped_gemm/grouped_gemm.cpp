// SPDX-License-Identifier: MIT
// Copyright (c) 2025, , Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/host.hpp"
#include "grouped_gemm.hpp"

std::size_t ck_tile_dcu_grouped_gemm_workspace_size(int32_t group_count, int32_t num_d_tensors)
{
    if (group_count <= 0) {
        return 0;
    }
    if (num_d_tensors <= 0) {
        return static_cast<std::size_t>(group_count) * sizeof(ck_tile::GemmTransKernelArg);
    }
    if (num_d_tensors == 1) {
        return static_cast<std::size_t>(group_count) * sizeof(ck_tile::GemmTransKernelArgImpl<1>);
    }
    return static_cast<std::size_t>(group_count) * sizeof(ck_tile::GemmTransKernelArgImpl<2>);
}

int32_t ck_tile_dcu_grouped_gemm_run(const ck_tile_dcu_grouped_gemm_desc* descs,
                                 int32_t group_count,
                                 int32_t dtype,
                                 char a_layout,
                                 char b_layout,
                                 void* workspace,
                                 hipStream_t stream,
                                 int32_t warmup,
                                 int32_t repeat,
                                 float* avg_ms)
{
    try
    {
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_FP16)
        if(dtype == CK_TILE_DCU_GROUPED_GEMM_FP16)
        {
            return grouped_gemm_c_run_fp16(descs, group_count,
                a_layout, b_layout, workspace, stream,
                warmup, repeat, avg_ms);
        }
#endif
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_FP8)
        if(dtype == CK_TILE_DCU_GROUPED_GEMM_FP8)
        {
            return grouped_gemm_c_run_fp8(descs, group_count,
                a_layout, b_layout, workspace, stream,
                warmup, repeat, avg_ms);
        }
#endif
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_BF16)
        if(dtype == CK_TILE_DCU_GROUPED_GEMM_BF16)
        {
            return grouped_gemm_c_run_bf16(descs, group_count,
                a_layout, b_layout, workspace, stream,
                warmup, repeat, avg_ms);
        }
#endif
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_BF8)
        if(dtype == CK_TILE_DCU_GROUPED_GEMM_BF8)
        {
            return grouped_gemm_c_run_bf8(descs, group_count,
                a_layout, b_layout, workspace, stream,
                warmup, repeat, avg_ms);
        }
#endif
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_INT8)
        if(dtype == CK_TILE_DCU_GROUPED_GEMM_INT8)
        {
            return grouped_gemm_c_run_int8(descs, group_count,
                a_layout, b_layout, workspace, stream,
                warmup, repeat, avg_ms);
        }
#endif
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_INT4)
        if(dtype == CK_TILE_DCU_GROUPED_GEMM_INT4)
        {
            return grouped_gemm_c_run_int4(descs, group_count,
                a_layout, b_layout, workspace, stream,
                warmup, repeat, avg_ms);
        }
#endif
        return -5;
    }
    catch(...)
    {
        return -100;
    }
}

int run_grouped_gemm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        return -1;
    }

    const std::string a_layout  = arg_parser.get_str("a_layout");
    const std::string b_layout  = arg_parser.get_str("b_layout");
    const std::string data_type = arg_parser.get_str("prec");
    const std::string config    = arg_parser.get_str("config");

#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_FP16)
    if(data_type == "fp16") {
        return grouped_gemm_example_run_fp16(a_layout, b_layout, argc, argv);
    }
    else
#endif
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_FP8)
    if(data_type == "fp8") {
        if(config == "fp8_128x64") {
            return grouped_gemm_example_run_fp8_128x64(a_layout, b_layout, argc, argv);
        }
        if(config == "fp8_128x128_k32") {
            return grouped_gemm_example_run_fp8_128x128_k32(a_layout, b_layout, argc, argv);
        }
        return grouped_gemm_example_run_fp8(a_layout, b_layout, argc, argv);
    }
    else
#endif
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_BF16)
    if(data_type == "bf16") {
        return grouped_gemm_example_run_bf16(a_layout, b_layout, argc, argv);
    }
    else
#endif
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_BF8)
    if(data_type == "bf8") {
        if(config == "bf8_128x64")
            return grouped_gemm_example_run_bf8_128x64(a_layout, b_layout, argc, argv);
        if(config == "bf8_k64")
            return grouped_gemm_example_run_bf8_k64(a_layout, b_layout, argc, argv);
        if(config == "bf8_128x128")
            return grouped_gemm_example_run_bf8_128x128(a_layout, b_layout, argc, argv);
        return grouped_gemm_example_run_bf8_k64(a_layout, b_layout, argc, argv);
    }
    else
#endif
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_INT8)
    if(data_type == "int8") {
        if(config == "int8_32x32")
            return grouped_gemm_example_run_int8_32x32(a_layout, b_layout, argc, argv);
        return grouped_gemm_example_run_int8(a_layout, b_layout, argc, argv);
    }
    else
#endif
#if !defined(CK_TILE_GROUPED_GEMM_FAST_BUILD) || defined(CK_TILE_GROUPED_GEMM_FAST_INT4)
    if(data_type == "int4") {
        return grouped_gemm_example_run_int4(a_layout, b_layout, argc, argv);
    }
    else
#endif
    {
        throw std::runtime_error("Unsupported data type configuration.");
    }
}

int main(int argc, char* argv[]) {
    return !run_grouped_gemm_example(argc, argv);
}
