// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/host/hip_check_error.hpp"

#include <map>
#include <sstream>
#include <string>
#include <stdexcept>
#include <hip/hip_runtime.h>

namespace ck_tile {

enum struct hcu_target_enum
{
    unk = 0,
    gfx928,
    gfx936,
    gfx938,
    gfx92a,
    gfx946,
};

struct hcu_device_ctx
{
    private:
    hcu_device_ctx()
    {
        HIP_CHECK_ERROR(hipGetDevice(&device_));
        HIP_CHECK_ERROR(hipGetDeviceProperties(&props_, device_));
    }

    int device_;
    hipDeviceProp_t props_;

    public:
    int device() const { return device_; }
    const hipDeviceProp_t& device_prop() const { return props_; }

    static const hcu_device_ctx& get_instance()
    {
        static hcu_device_ctx ctx;

        return ctx;
    }
};

CK_TILE_HOST std::string get_device_name()
{
    static const auto& ctx = hcu_device_ctx::get_instance();

    const std::string raw_name(ctx.device_prop().gcnArchName);

    // strip sramecc/xnack
    const auto name = raw_name.substr(0, raw_name.find(':'));
    return name;
}

CK_TILE_HOST hcu_target_enum get_hcu_target_enum()
{
    static std::map<std::string, hcu_target_enum> target_map = {
        {"gfx928", hcu_target_enum::gfx928},
        {"gfx936", hcu_target_enum::gfx936},
        {"gfx938", hcu_target_enum::gfx938},
        {"gfx92a", hcu_target_enum::gfx92a},
        {"gfx946", hcu_target_enum::gfx946},
    };

    const auto device_name = get_device_name();

    auto match = target_map.find(device_name);

    if(match != target_map.end())
        return match->second;
    return hcu_target_enum::unk;
}

CK_TILE_HOST auto get_max_shared_mem_per_block()
{
    static const auto& ctx = hcu_device_ctx::get_instance();

    return ctx.device_prop().sharedMemPerBlock;
}

CK_TILE_HOST auto get_hcu_lds_capacity()
{
    if(get_hcu_target_enum() == hcu_target_enum::gfx946)
    {
        return 128 * 1024;
    }
    else
    {
        return 64 * 1024;
    }
}

CK_TILE_HOST auto get_num_cu()
{
    static const auto& ctx = hcu_device_ctx::get_instance();

    return ctx.device_prop().multiProcessorCount;
}

} // namespace ck_tile
