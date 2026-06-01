// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.

#pragma once

// Address Space for AMDGCN
// https://llvm.org/docs/AMDGPUUsage.html#address-space

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"

namespace ck_tile {

template <typename, bool>
struct safe_underlying_type;

template <typename T>
struct safe_underlying_type<T, true>
{
    using type = std::underlying_type_t<T>;
};

template <typename T>
struct safe_underlying_type<T, false>
{
    using type = void;
};

template <typename T>
using safe_underlying_type_t = typename safe_underlying_type<T, std::is_enum<T>::value>::type;

enum struct address_space_enum : std::uint16_t
{
    generic = 0,
    global,
    lds,
    sgpr,
    constant,
    vgpr
};

enum struct memory_operation_enum : std::uint16_t
{
    set = 0,
    atomic_add,
    atomic_max,
    add
};

CK_TILE_HOST_DEVICE constexpr index_t get_warp_size()
{
    // warpSize is defined by HIP
    return warpSize;
}

CK_TILE_HOST constexpr bool is_wave32()
{
    return false;   //for now, assume all devices are wave64

    // hipDeviceProp_t props{};
    // int device;
    // auto status = hipGetDevice(&device);
    // if(status != hipSuccess)
    // {
    //     return false;
    // }
    // status = hipGetDeviceProperties(&props, device);
    // if(status != hipSuccess)
    // {
    //     return false;
    // }
    // return props.major > 9;
}

CK_TILE_DEVICE index_t get_grid_size() { return gridDim.x; }

CK_TILE_DEVICE index_t get_block_size() { return blockDim.x; }

// TODO: deprecate these
CK_TILE_DEVICE index_t get_thread_local_1d_id() { return threadIdx.x; }

CK_TILE_DEVICE index_t get_thread_global_1d_id() { return  (blockIdx.z * blockDim.z + threadIdx.z) * (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y) + 
                                                           (blockIdx.y * blockDim.y + threadIdx.y) * (gridDim.x * blockDim.x) +
                                                           blockIdx.x * blockDim.x + threadIdx.x;}

CK_TILE_DEVICE index_t get_block_1d_id() { return blockIdx.x; }

// Use these instead
CK_TILE_DEVICE index_t get_lane_id() { return __lane_id(); }

CK_TILE_DEVICE index_t get_warp_id()
{
    return __builtin_amdgcn_readfirstlane(threadIdx.x / get_warp_size());
}

//wasp ids
template <index_t SubWGSize>
CK_TILE_DEVICE index_t get_sub_warp_id(number<SubWGSize>)
{
    static_assert(SubWGSize % 2 == 0, "wrong! SubWGSize must be multiples of 2!");
    return __builtin_amdgcn_readfirstlane(threadIdx.x / get_warp_size() & (SubWGSize-1));
}

template <index_t SubWGSize>
CK_TILE_DEVICE index_t get_sub_warpgroup_id(number<SubWGSize>)
{
    static_assert(SubWGSize % 2 == 0, "wrong! SubWGSize must be multiples of 2!");
    return __builtin_amdgcn_readfirstlane(threadIdx.x / get_warp_size() / SubWGSize);
}

CK_TILE_DEVICE index_t get_thread_id() { return threadIdx.x; }

CK_TILE_DEVICE index_t get_block_id() { return blockIdx.x; }

CK_TILE_DEVICE void block_sync_lds()
{
#if CK_TILE_EXPERIMENTAL_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM
    // asm volatile("\
    // s_waitcnt lgkmcnt(0) \n \
    // s_barrier \
    // " ::);

    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_s_barrier();
#else
    __syncthreads();
#endif
}

CK_TILE_DEVICE void block_sync_lds_direct_load()
{
    asm volatile("\
    s_waitcnt vmcnt(0) \n \
    s_waitcnt lgkmcnt(0) \n \
    s_barrier \
    " ::);
}

CK_TILE_DEVICE void block_sync_load_raw(index_t cnt = 0)
{
    asm volatile("s_waitcnt vmcnt(%0) \n"
                 "s_barrier"
                 :
                 : "n"(cnt)
                 : "memory");
}

CK_TILE_DEVICE void s_nop(index_t cnt = 0)
{
#if 1
    asm volatile("s_nop %0" : : "n"(cnt) :);
#else
    __builtin_amdgcn_sched_barrier(cnt);
#endif
}

#define CK_TILE_CONSTANT_ADDRESS_SPACE \
    __attribute__((address_space( \
        static_cast<safe_underlying_type_t<address_space_enum>>(address_space_enum::constant))))

template <typename T>
__device__ T* cast_pointer_to_generic_address_space(T CK_TILE_CONSTANT_ADDRESS_SPACE* p)
{
    // cast a pointer in "Constant" address space (4) to "Generic" address space (0)
    // only c-style pointer cast seems be able to be compiled
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
    return (T*)(p); // NOLINT(old-style-cast)
#pragma clang diagnostic pop
}

template <typename T>
__host__ __device__ T CK_TILE_CONSTANT_ADDRESS_SPACE* cast_pointer_to_constant_address_space(T* p)
{
    // cast a pointer in "Generic" address space (0) to "Constant" address space (4)
    // only c-style pointer cast seems be able to be compiled;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
    return (T CK_TILE_CONSTANT_ADDRESS_SPACE*)p; // NOLINT(old-style-cast)
#pragma clang diagnostic pop
}

/// Helper function to convert address space enum to string
CK_TILE_HOST_DEVICE constexpr const char* address_space_to_string(address_space_enum addr_space)
{
    switch(addr_space)
    {
    case address_space_enum::generic: return "generic";
    case address_space_enum::global: return "global";
    case address_space_enum::lds: return "lds";
    case address_space_enum::sgpr: return "sgpr";
    case address_space_enum::constant: return "constant";
    case address_space_enum::vgpr: return "vgpr";
    default: return "unknown";
    }
}

template <bool use_asm = false>
CK_TILE_DEVICE void wg_sync(bool_constant<use_asm> = {})
{
    if constexpr(use_asm)
    {
        asm volatile("s_barrier\n\t");
    }
    else
    {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }
}

template <bool use_asm = false>
CK_TILE_DEVICE void wg_sync_lds(bool_constant<use_asm> = {})
{
    if constexpr(use_asm)
    {
        asm volatile("s_waitcnt lgkmcnt(0); \n\t"
                     "s_barrier; \n\t");
    }
    else
    {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_waitcnt(0xc07f);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }
}

CK_TILE_DEVICE void promote_prio()
{
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_setprio(1);
    __builtin_amdgcn_sched_barrier(0);
}

CK_TILE_DEVICE void restore_prio()
{
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_sched_barrier(0);
}

CK_TILE_DEVICE void lds_wait()
{
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_sched_barrier(0);
}

CK_TILE_DEVICE void lds_fence()
{
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
}

CK_TILE_HOST_DEVICE constexpr index_t get_smem_capacity()
{
#if defined(__gfx950__)
    return 163840;
#else
    return 65536;
#endif
}

} // namespace ck_tile
