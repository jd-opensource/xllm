// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include <hip/hip_runtime.h>
#include <cstddef>
#include <chrono>

namespace ck_tile {

struct gpu_timer
{
    CK_TILE_HOST gpu_timer()
    {
        HIP_CHECK_ERROR(hipEventCreate(&start_evt));
        HIP_CHECK_ERROR(hipEventCreate(&stop_evt));
    }

    CK_TILE_HOST ~gpu_timer() noexcept(false)
    {
        HIP_CHECK_ERROR(hipEventDestroy(start_evt));
        HIP_CHECK_ERROR(hipEventDestroy(stop_evt));
    }

    CK_TILE_HOST void start(const hipStream_t& s)
    {
        HIP_CHECK_ERROR(hipStreamSynchronize(s));
        HIP_CHECK_ERROR(hipEventRecord(start_evt, s));
    }

    CK_TILE_HOST void stop(const hipStream_t& s)
    {
        HIP_CHECK_ERROR(hipEventRecord(stop_evt, s));
        HIP_CHECK_ERROR(hipEventSynchronize(stop_evt));
    }
    // return in ms
    CK_TILE_HOST float duration() const
    {
        float ms = 0;
        HIP_CHECK_ERROR(hipEventElapsedTime(&ms, start_evt, stop_evt));
        return ms;
    }

    private:
    hipEvent_t start_evt, stop_evt;
};

struct cpu_timer
{
    // torch.utils.benchmark.Timer(), there is a sync inside each timer callback
    CK_TILE_HOST void start(const hipStream_t& s)
    {
        HIP_CHECK_ERROR(hipStreamSynchronize(s));
        start_tick = std::chrono::high_resolution_clock::now();
    }
    // torch.utils.benchmark.Timer(), there is a sync inside each timer callback
    CK_TILE_HOST void stop(const hipStream_t& s)
    {
        HIP_CHECK_ERROR(hipStreamSynchronize(s));
        stop_tick = std::chrono::high_resolution_clock::now();
    }
    // return in ms
    CK_TILE_HOST float duration() const
    {
        double sec =
            std::chrono::duration_cast<std::chrono::duration<double>>(stop_tick - start_tick)
                .count();
        return static_cast<float>(sec * 1e3);
    }

    private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_tick;
    std::chrono::time_point<std::chrono::high_resolution_clock> stop_tick;
};

} // namespace ck_tile
