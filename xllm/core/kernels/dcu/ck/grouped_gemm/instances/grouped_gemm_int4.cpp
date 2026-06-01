// SPDX-License-Identifier: MIT
// Copyright (c) 2025, , Inc. All rights reserved.
#include "grouped_gemm_impl.hpp"

template <typename GemmConfig, typename ALayout, typename BLayout, typename CLayout>
int grouped_gemm_c_run_int4_impl(const ck_tile_dcu_grouped_gemm_desc* descs,
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
    if(k_batch <= 0)
    {
        return -2;
    }

    std::vector<ck_tile::GemmTransKernelArg> kargs;
    kargs.reserve(group_count);
    for(int i = 0; i < group_count; ++i)
    {
        const auto& arg = descs[i];
        if(arg.a_ptr == nullptr || arg.b_ptr == nullptr || arg.c_ptr == nullptr || arg.M <= 0 ||
           arg.N <= 0 || arg.K <= 0 || arg.k_batch != k_batch || arg.num_d_tensors != 0)
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

    ck_tile::DeviceMem owned_workspace;
    void* kargs_ptr = workspace;
    if(kargs_ptr == nullptr)
    {
        owned_workspace.Realloc(ck_tile_dcu_grouped_gemm_workspace_size(group_count));
        kargs_ptr = owned_workspace.GetDeviceBuffer();
    }

    const auto stream_cfg =
        ck_tile::stream_config{stream, repeat > 0, 0, warmup, repeat > 0 ? repeat : 1};
    HIP_CHECK_ERROR(hipMemcpyWithStream(kargs_ptr,
                                        kargs.data(),
                                        kargs.size() * sizeof(ck_tile::GemmTransKernelArg),
                                        hipMemcpyHostToDevice,
                                        stream_cfg.stream_id_));

    *avg_ms = grouped_gemm_tileloop<GemmConfig,
                                    ALayout,
                                    BLayout,
                                    CLayout,
                                    ck_tile::pk_int4_t,
                                    ck_tile::pk_int4_t,
                                    int32_t,
                                    int32_t>(
        stream_cfg, group_count, kargs_ptr, k_batch > 1);
    return 0;
}

int grouped_gemm_c_run_int4(const ck_tile_dcu_grouped_gemm_desc* descs,
                            int group_count,
                            char a_layout,
                            char b_layout,
                            void* workspace,
                            hipStream_t stream,
                            int warmup,
                            int repeat,
                            float* avg_ms)
{
    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;
    using Config = GemmConfigComputeInt4<ck_tile::pk_int4_t>;

    if(a_layout == 'R' && b_layout == 'C')
    {
        return grouped_gemm_c_run_int4_impl<Config, Row, Col, Row>(
            descs, group_count, workspace, stream, warmup, repeat, avg_ms);
    }
    if(a_layout == 'R' && b_layout == 'R')
    {
        return grouped_gemm_c_run_int4_impl<Config, Row, Row, Row>(
            descs, group_count, workspace, stream, warmup, repeat, avg_ms);
    }
    if(a_layout == 'C' && b_layout == 'R')
    {
        return grouped_gemm_c_run_int4_impl<Config, Col, Row, Row>(
            descs, group_count, workspace, stream, warmup, repeat, avg_ms);
    }
    if(a_layout == 'C' && b_layout == 'C')
    {
        return grouped_gemm_c_run_int4_impl<Config, Col, Col, Row>(
            descs, group_count, workspace, stream, warmup, repeat, avg_ms);
    }
    return -4;
}

int grouped_gemm_example_run_int4(std::string a_layout, std::string b_layout, int argc, char* argv[])
{
    return run_gemm_example_prec_type<GemmConfigComputeInt4<ck_tile::pk_int4_t>,
                                      ck_tile::pk_int4_t>(
        a_layout, b_layout, argc, argv);
}
