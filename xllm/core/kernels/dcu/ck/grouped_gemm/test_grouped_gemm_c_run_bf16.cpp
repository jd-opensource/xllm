// SPDX-License-Identifier: MIT
// Copyright (c) 2025, , Inc. All rights reserved.
//
// Test case for grouped_gemm_c_run_bf16() C ABI.
//
// Build (standalone, like build.sh):
//   /opt/dtk/bin/aicc \
//     -DCK_TILE_GROUPED_GEMM_FAST_BUILD \
//     -DCK_TILE_GROUPED_GEMM_FAST_BF16 \
//     -I../include \
//     -isystem /opt/dtk/llvm/lib/clang/17.0.0/include/.. \
//     -O3 -DNDEBUG -std=c++17 \
//     -xhip --offload-arch=gfx938 \
//     -o test_grouped_gemm_c_run_bf16 \
//     test_grouped_gemm_c_run_bf16.cpp \
//     instances/grouped_gemm_bf16.cpp \
//     /opt/dtk/hip/lib/libgalaxyhip.so --hip-link --offload-arch=gfx938 \
//     -L"/opt/dtk/llvm/lib/clang/17.0.0/include/../lib/linux" -lclang_rt.builtins-x86_64

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "grouped_gemm.hpp"

std::size_t ck_tile_dcu_grouped_gemm_workspace_size(int group_count, int num_d_tensors)
{
    if(group_count <= 0)
        return 0;
    if(num_d_tensors <= 0)
        return static_cast<std::size_t>(group_count) * sizeof(ck_tile::GemmTransKernelArg);
    if(num_d_tensors == 1)
        return static_cast<std::size_t>(group_count) * sizeof(ck_tile::GemmTransKernelArgImpl<1>);
    return static_cast<std::size_t>(group_count) * sizeof(ck_tile::GemmTransKernelArgImpl<2>);
}

// ---------------------------------------------------------------------------
// bf16 CPU reference: C = A * B   (A: MxK row-major, B: KxN col-major, C: MxN row-major)
// B is stored as [N][K] row-major (= col-major [K][N] view), so B(k,n) = B_storage[n*K + k].
// ---------------------------------------------------------------------------
static void cpu_reference_bf16(const std::vector<ck_tile::HostTensor<ck_tile::bf16_t>>& a_tensors,
                               const std::vector<ck_tile::HostTensor<ck_tile::bf16_t>>& b_tensors,
                               std::vector<ck_tile::HostTensor<ck_tile::bf16_t>>& c_refs)
{
    for (size_t g = 0; g < a_tensors.size(); ++g) {
        const auto& A = a_tensors[g];
        const auto& B = b_tensors[g];
        auto& C       = c_refs[g];

        int M = static_cast<int>(A.mDesc.get_lengths()[0]);
        int K = static_cast<int>(A.mDesc.get_lengths()[1]);
        int N = static_cast<int>(B.mDesc.get_lengths()[1]);

        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                    float av = static_cast<float>(A(m, k));
                    float bv = static_cast<float>(B(k, n));  // col-major view
                    acc += av * bv;
                }
                C(m, n) = static_cast<ck_tile::bf16_t>(static_cast<float>(
                    static_cast<ck_tile::bf16_t>(acc)));
            }
        }
    }
}

// ---------------------------------------------------------------------------
static bool validate(const std::vector<ck_tile::HostTensor<ck_tile::bf16_t>>& result,
                     const std::vector<ck_tile::HostTensor<ck_tile::bf16_t>>& reference,
                     float rtol = 1e-2f, float atol = 1e-2f)
{
    bool pass = true;
    for (size_t g = 0; g < result.size(); ++g) {
        const auto& R = result[g];
        const auto& T = reference[g];
        int M = static_cast<int>(R.mDesc.get_lengths()[0]);
        int N = static_cast<int>(R.mDesc.get_lengths()[1]);
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float rv = static_cast<float>(R(m, n));
                float tv = static_cast<float>(T(m, n));
                float diff = std::abs(rv - tv);
                float denom = std::max(std::abs(tv), 1e-6f);
                if (diff > atol && diff / denom > rtol) {
                    std::cerr << "  Mismatch group=" << g << " [" << m << "," << n << "] "
                              << "ref=" << tv << " result=" << rv
                              << " diff=" << diff << " rel=" << (diff / denom) << std::endl;
                    pass = false;
                }
            }
        }
    }
    return pass;
}

// ---------------------------------------------------------------------------
int main()
{
    constexpr int group_count   = 4;
    constexpr int k_batch       = 1;
    constexpr char a_layout     = 'R';
    constexpr char b_layout     = 'C';
    constexpr int warmup        = 5;
    constexpr int repeat        = 20;

    // Group problem sizes
    const int Ms[group_count] = {64, 128, 256, 512};
    const int Ns[group_count] = {128, 256, 128, 256};
    const int Ks[group_count] = {128, 256, 512, 1024};

    // Strides (default for R / C layout)
    std::vector<int> stride_As(group_count);
    std::vector<int> stride_Bs(group_count);
    std::vector<int> stride_Cs(group_count);

    // Host tensors for A, B, C (use push_back, not default-construct)
    std::vector<ck_tile::HostTensor<ck_tile::bf16_t>> a_host;
    std::vector<ck_tile::HostTensor<ck_tile::bf16_t>> b_host;
    std::vector<ck_tile::HostTensor<ck_tile::bf16_t>> c_host;
    std::vector<ck_tile::HostTensor<ck_tile::bf16_t>> c_ref;
    a_host.reserve(group_count);
    b_host.reserve(group_count);
    c_host.reserve(group_count);
    c_ref.reserve(group_count);

    // Device buffers
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_dev;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> b_dev;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> c_dev;
    a_dev.reserve(group_count);
    b_dev.reserve(group_count);
    c_dev.reserve(group_count);

    // Descriptors for C ABI
    std::vector<ck_tile_dcu_grouped_gemm_desc> descs;
    descs.reserve(group_count);

    std::cout << "=== test_grouped_gemm_c_run_bf16 ===" << std::endl;

    for (int g = 0; g < group_count; ++g) {
        int M = Ms[g];
        int N = Ns[g];
        int K = Ks[g];

        stride_As[g] = ck_tile::get_default_stride(M, K, 0, ck_tile::bool_constant<true>{});
        stride_Bs[g] = ck_tile::get_default_stride(K, N, 0, ck_tile::bool_constant<false>{});
        stride_Cs[g] = ck_tile::get_default_stride(M, N, 0, ck_tile::bool_constant<true>{});

        // Row-major host tensors (emplace_back)
        a_host.emplace_back(
            ck_tile::host_tensor_descriptor(M, K, stride_As[g], ck_tile::bool_constant<true>{}));
        b_host.emplace_back(
            ck_tile::host_tensor_descriptor(K, N, stride_Bs[g], ck_tile::bool_constant<false>{}));
        c_host.emplace_back(
            ck_tile::host_tensor_descriptor(M, N, stride_Cs[g], ck_tile::bool_constant<true>{}));

        // Fill with deterministic pattern
        ck_tile::FillUniformDistribution<ck_tile::bf16_t>{-1.f, 1.f}(a_host[g]);
        ck_tile::FillUniformDistribution<ck_tile::bf16_t>{-1.f, 1.f}(b_host[g]);
        c_host[g].SetZero();

        // Allocate device memory
        a_dev.push_back(std::make_unique<ck_tile::DeviceMem>(a_host[g]));
        b_dev.push_back(std::make_unique<ck_tile::DeviceMem>(b_host[g]));
        c_dev.push_back(std::make_unique<ck_tile::DeviceMem>(c_host[g]));
        c_dev[g]->SetZero();

        // CPU reference
        c_ref.emplace_back(
            ck_tile::host_tensor_descriptor(M, N, stride_Cs[g], ck_tile::bool_constant<true>{}));
        c_ref[g].SetZero();

        // Fill descriptor
        ck_tile_dcu_grouped_gemm_desc desc;
        desc.a_ptr         = a_dev[g]->GetDeviceBuffer();
        desc.b_ptr         = b_dev[g]->GetDeviceBuffer();
        desc.c_ptr         = c_dev[g]->GetDeviceBuffer();
        desc.k_batch       = k_batch;
        desc.M             = M;
        desc.N             = N;
        desc.K             = K;
        desc.stride_A      = stride_As[g];
        desc.stride_B      = stride_Bs[g];
        desc.stride_C      = stride_Cs[g];
        desc.num_d_tensors = 0;
        desc.d_ptrs        = nullptr;
        desc.stride_Ds     = nullptr;
        descs.push_back(desc);

        std::cout << "  Group[" << g << "] M=" << M << " N=" << N << " K=" << K
                  << " stride_A=" << stride_As[g]
                  << " stride_B=" << stride_Bs[g]
                  << " stride_C=" << stride_Cs[g] << std::endl;
    }

    // Compute CPU reference
    cpu_reference_bf16(a_host, b_host, c_ref);

    // Allocate workspace
    std::size_t ws_size = ck_tile_dcu_grouped_gemm_workspace_size(group_count, 0);
    std::cout << "  Workspace size: " << ws_size << " bytes" << std::endl;
    ck_tile::DeviceMem workspace(ws_size);

    // Launch
    float avg_ms = 0.0f;
    int ret = grouped_gemm_c_run_bf16(descs.data(),
                                      group_count,
                                      a_layout,
                                      b_layout,
                                      workspace.GetDeviceBuffer(),
                                      nullptr,  // default stream
                                      warmup,
                                      repeat,
                                      &avg_ms);

    if (ret != 0) {
        std::cerr << "grouped_gemm_c_run_bf16 returned error code: " << ret << std::endl;
        return 1;
    }

    std::cout << "  Average time: " << avg_ms << " ms" << std::endl;

    // Compute total FLOPS
    std::size_t total_flop = 0;
    for (int g = 0; g < group_count; ++g) {
        total_flop += std::size_t(2) * Ms[g] * Ns[g] * Ks[g];
    }
    float tflops = static_cast<float>(total_flop) / 1.e9f / avg_ms;
    std::cout << "  Throughput: " << tflops << " TFlops" << std::endl;

    // Copy back results
    for (int g = 0; g < group_count; ++g) {
        c_dev[g]->FromDevice(c_host[g].data());
    }

    // Validate
    bool pass = validate(c_host, c_ref);
    std::cout << "  Validation: " << (pass ? "PASS" : "FAIL") << std::endl;

    return pass ? 0 : 1;
}
