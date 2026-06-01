// SPDX-License-Identifier: MIT
// Copyright (c) 2025, , Inc. All rights reserved.
#include "grouped_gemm_impl.hpp"

int grouped_gemm_c_run_bf16(const ck_tile_dcu_grouped_gemm_desc* descs,
                            int group_count,
                            char a_layout,
                            char b_layout,
                            void* workspace,
                            hipStream_t stream,
                            int warmup,
                            int repeat,
                            float* avg_ms)
{
    return dispatch_grouped_gemm_c_layout<GemmConfigComputeV4<ck_tile::bf16_t>, ck_tile::bf16_t>(
        descs, group_count, a_layout, b_layout, workspace, stream, warmup, repeat, avg_ms);
}

int grouped_gemm_example_run_bf16(std::string a_layout, std::string b_layout, int argc, char* argv[])
{
    return run_gemm_example_prec_type<GemmConfigComputeV4<ck_tile::bf16_t>, ck_tile::bf16_t>(
        a_layout, b_layout, argc, argv);
}
