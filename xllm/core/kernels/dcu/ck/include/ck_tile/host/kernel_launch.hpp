// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/host/stream_config.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/timer.hpp"
#include <hip/hip_runtime.h>
#include <cstddef>

namespace ck_tile {
template <int MaxThreadPerBlock, int MinBlockPerCu, typename Kernel, typename... Args>
#if CK_TILE_USE_LAUNCH_BOUNDS
__launch_bounds__(MaxThreadPerBlock, MinBlockPerCu)
#endif
    __global__ void kentry(Args... args)
{
    Kernel{}(args...);
}

//
// return a anonymous functor(lambda) to be called later
// the KernelImpl should be a class without non-static data member, or let's say
// can be instantiate with "KernelImpl{}"
//
// the "static __device__ operator()(some_arg)" is the entry point of KernelImpl
//
template <int MaxThreadPerBlock = CK_TILE_MAX_THREAD_PER_BLOCK,
          int MinBlockPerCu     = CK_TILE_MIN_BLOCK_PER_CU,
          typename KernelImpl,
          typename... Args>
CK_TILE_HOST auto
make_kernel(KernelImpl /*f*/, dim3 grid_dim, dim3 block_dim, std::size_t lds_byte, Args... args)
{
    const auto kernel = kentry<MaxThreadPerBlock, MinBlockPerCu, KernelImpl, Args...>;

    return [=](const stream_config& s) {
        kernel<<<grid_dim, block_dim, lds_byte, s.stream_id_>>>(args...);
    };
}

// clang-format off
/*
 * launch_kernel()
 *
 * this is the function to launch arbitrary number of kernels with optional timer(selected by stream_config)
 * the callables should have signature as "operator()(const stream_config& s){ ... }" to call
 * 
 * the simplest way is pass in a lambda function, with "[=](const stream_config& s){ call_your_kernel_here() }"
 * as signature, for the callable (pay attention to the capture list)
 * 
 * e.g.
 *  ck_tile::launch_kernel(s,
 *                      [=](const stream_config& s){ hipMemset(ptr, 0, size) },
 *                      [=](const stream_config& s){ some_kernel<<<grids, blocks>>>(arg); }
 *                      );
 * 
 * if you use ck_tile kernel, or similiar to this style (structure with "static __device__ operator()(...){}")
 * you can pass your kernel to ck_tile::make_kernel(), which will create a anonymous functor for you,
 * then pass it to ck_tile::launch_kernel()
 * 
 * e.g.
 *  ck_tile::launch_kernel(s,
 *                      ck_tile::make_kernel<T0, B0>(kernel_0{}, grids0, blocks0, 0, kargs0),
 *                      ck_tile::make_kernel<T0, B1>(kernel_1{}, grids1, blocks1, 0, kargs1),
 *                       ...);
 **/
// clang-format on
template <typename... Callables>
CK_TILE_HOST float launch_kernel(const stream_config& s, Callables... callables)
{
    // clang-format off
    if(!s.time_kernel_) {
        (callables(s),...); HIP_CHECK_ERROR(hipGetLastError());
        return 0;
    }
    if(s.is_gpu_timer_) {
        gpu_timer timer {};

        // warmup
        for(int i = 0; i < s.cold_niters_; i++) { (callables(s),...); } HIP_CHECK_ERROR(hipGetLastError());

        timer.start(s.stream_id_);
        for(int i = 0; i < s.nrepeat_; i++) { (callables(s),...); } HIP_CHECK_ERROR(hipGetLastError());
        timer.stop(s.stream_id_);

        return timer.duration() / s.nrepeat_;
    }
    else {
        cpu_timer timer {};

        // warmup
        for(int i = 0; i < s.cold_niters_; i++) { (callables(s),...); } HIP_CHECK_ERROR(hipGetLastError());

        timer.start(s.stream_id_);
        for(int i = 0; i < s.nrepeat_; i++) { (callables(s),...); } HIP_CHECK_ERROR(hipGetLastError());
        timer.stop(s.stream_id_);

        return timer.duration() / s.nrepeat_;
    }
    // clang-format on
}

} // namespace ck_tile
