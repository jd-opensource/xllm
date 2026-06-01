// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || \
    defined(__gfx941__) || defined(__gfx942__)
#define __gfx9__
#endif
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#define __gfx94__
#endif
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || \
    defined(__gfx10_3_generic__)
#define __gfx103__
#endif
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx11_generic__)
#define __gfx11__
#endif
#if defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx12_generic__)
#define __gfx12__
#endif

#include "hip/hip_version.h"
#ifndef CK_TILE_DONT_USE_HIP_RUNTIME_HEADERS
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#endif

#ifdef __HIPCC__
#define CK_TILE_HOST inline __host__
#define CK_TILE_DEVICE inline __device__
#define CK_TILE_HOST_DEVICE inline __host__ __device__
#define CK_TILE_DEVICE_EXTERN __device__
#define CK_TILE_HOST_DEVICE_EXTERN __host__ __device__
#else
#define CK_TILE_HOST inline
#define CK_TILE_DEVICE inline
#define CK_TILE_HOST_DEVICE inline
#define CK_TILE_DEVICE_EXTERN
#define CK_TILE_HOST_DEVICE_EXTERN
#endif

// implementing the "memory address space" attribute
// https://llvm.org/docs/AMDGPUUsage.html#amdgpu-address-spaces-table
#ifdef __HIPCC_
#define CK_TILE_GENERIC_ADDR __attribute__((address_space(0)))
#define CK_TILE_GLOBAL_ADDR __attribute__((address_space(1)))
#define CK_TILE_LDS_ADDR __attribute__((address_space(3)))
#define CK_TILE_BUF_RES_ADDR __attribute__((address_space(8)))
#else
#define CK_TILE_GENERIC_ADDR
#define CK_TILE_GLOBAL_ADDR
#define CK_TILE_LDS_ADDR
#define CK_TILE_BUF_RES_ADDR
#endif
#ifndef CK_TILE_USE_CUSTOM_DATA_TYPE
#define CK_TILE_USE_CUSTOM_DATA_TYPE 0 // custom data type will generate extra move/bfi code
#endif

#define CK_TILE_FLOAT_TO_BFLOAT16_STANDARD 0
#define CK_TILE_FLOAT_TO_BFLOAT16_TRUNCATE_WITH_NAN 1
#define CK_TILE_FLOAT_TO_BFLOAT16_TRUNCATE 2
#define CK_TILE_FLOAT_TO_BFLOAT16_STANDARD_ASM 3
#define CK_TILE_FLOAT_TO_BFLOAT16_RTA_ASM 4

#ifndef CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT
#define CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT CK_TILE_FLOAT_TO_BFLOAT16_TRUNCATE
#endif

#define CK_TILE_FLOAT_TO_FP8_STANDARD 0
#define CK_TILE_FLOAT_TO_FP8_STOCHASTIC 1

#ifndef CK_TILE_FLOAT_TO_FP8_DEFAULT
#define CK_TILE_FLOAT_TO_FP8_DEFAULT CK_TILE_FLOAT_TO_FP8_STANDARD
#endif

// in the old rocm period, we have to use tuple array implementation to implement this
// so turn on the _USE_TUPLE if meet compiler error, otherwise _USE_ARRAY by default.
#define CK_TILE_STATICALLY_INDEXED_ARRAY_USE_ARRAY 0
#define CK_TILE_STATICALLY_INDEXED_ARRAY_USE_TUPLE 1
#ifndef CK_TILE_STATICALLY_INDEXED_ARRAY_DEFAULT
#define CK_TILE_STATICALLY_INDEXED_ARRAY_DEFAULT CK_TILE_STATICALLY_INDEXED_ARRAY_USE_TUPLE
#endif

#define CK_TILE_THREAD_BUFFER_USE_ARRAY 0
#define CK_TILE_THREAD_BUFFER_USE_TUPLE 1
#ifndef CK_TILE_THREAD_BUFFER_DEFAULT
#define CK_TILE_THREAD_BUFFER_DEFAULT CK_TILE_THREAD_BUFFER_USE_ARRAY
#endif

#ifndef CK_TILE_TUPLE_CTOR_WITH_INITIALIZER_LIST
#if CK_TILE_THREAD_BUFFER_DEFAULT == CK_TILE_THREAD_BUFFER_USE_TUPLE
// if using tuple-array as thread_buffer implementation, need to support {} brace init
// ... with similiar behavior as array
#define CK_TILE_TUPLE_CTOR_WITH_INITIALIZER_LIST 1
#else
#define CK_TILE_TUPLE_CTOR_WITH_INITIALIZER_LIST 0
#endif
#endif

#ifndef CK_TILE_USE_LAUNCH_BOUNDS
#define CK_TILE_USE_LAUNCH_BOUNDS 1
#endif

#ifndef CK_TILE_TIME_KERNEL
#define CK_TILE_TIME_KERNEL 1
#endif

#define CK_TILE_MAX_THREAD_PER_BLOCK 768
#define CK_TILE_MIN_BLOCK_PER_CU 1

#ifndef CK_TILE_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
#define CK_TILE_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK 0
#endif

#ifndef CK_TILE_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
#define CK_TILE_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK 1
#endif

#ifndef CK_TILE_EXPERIMENTAL_USE_BUFFER_ATOMIC_ADD_OOB_CHECK_OFFSET_TRICK
#define CK_TILE_EXPERIMENTAL_USE_BUFFER_ATOMIC_ADD_OOB_CHECK_OFFSET_TRICK 1
#endif

#ifndef CK_TILE_EXPERIMENTAL_USE_BUFFER_ATOMIC_MAX_OOB_CHECK_OFFSET_TRICK
#define CK_TILE_EXPERIMENTAL_USE_BUFFER_ATOMIC_MAX_OOB_CHECK_OFFSET_TRICK 1
#endif

#ifndef CK_TILE_USE_AMD_LDS_DIRECT_LOAD_INLINE_ASM
#define CK_TILE_USE_AMD_LDS_DIRECT_LOAD_INLINE_ASM 1
#endif

#ifndef CK_TILE_USE_AMD_BUFFER_LOAD
#define CK_TILE_USE_AMD_BUFFER_LOAD 1
#endif

#ifndef CK_TILE_USE_AMD_BUFFER_STORE
#define CK_TILE_USE_AMD_BUFFER_STORE 1
#endif

#ifndef CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER
#define CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER 1
#endif

#ifndef CK_TILE_USE_PK4_LAYOUT_SHUFFLE
#define CK_TILE_USE_PK4_LAYOUT_SHUFFLE 1
#endif

#ifndef CK_TILE_TENSOR_LOAD_FORCE_M0
#define CK_TILE_TENSOR_LOAD_FORCE_M0 0
#endif

#if defined(__gfx92a__) || defined(__gfx946__)
#define CK_TILE_DS_FORMAT_FORCE_M0 0
#else
#define CK_TILE_DS_FORMAT_FORCE_M0 1
#endif

#if defined(__gfx946__)
#define CK_TILE_MATRIX_LOAD_SUPPORT_BPS 1
#else
#define CK_TILE_MATRIX_LOAD_SUPPORT_BPS 0
#endif

#if defined(__gfx938__)
#define CK_TILE_MATRIX_LOAD_TRANS_EXTRA_CONFIG 0x80000000
#else
#define CK_TILE_MATRIX_LOAD_TRANS_EXTRA_CONFIG 0x00000000
#endif

#if defined(__gfx92a__) || defined(__gfx946__)
#define CK_TILE_DEVICE_EBARRIER_SUPPORT 1
#else
#define CK_TILE_DEVICE_EBARRIER_SUPPORT 0
#endif

#if defined(__gfx946__)
#define CK_TILE_DEVICE_EBARRIER_REDUCE_OP_SUPPORT 1
#else
#define CK_TILE_DEVICE_EBARRIER_REDUCE_OP_SUPPORT 0
#endif

#if defined(__gfx946__)
#define CK_TILE_DEVICE_ABARRIER_SUPPORT 1
#else
#define CK_TILE_DEVICE_ABARRIER_SUPPORT 0
#endif

#if defined(__gfx946__)
#define CK_TILE_BLOCK_MAX_LDS_SIZE 131072
#else
#define CK_TILE_BLOCK_MAX_LDS_SIZE 65536
#endif

// buffer atomic add: floating point
#ifndef __HIP_DEVICE_COMPILE__ // for host code
#define CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT 1
#elif defined(__gfx9__) || defined(__gfx928__) || defined(__gfx92a__) || defined(__gfx936__) || \
    defined(__gfx938__) // for GPU code
#define CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT 1
#else // for GPU code
#define CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT 0
#endif

#if (defined(__gfx90a__) || defined(__gfx94__)) // for GPU code
#define CK_TILE_USE_AMD_BUFFER_ATOMIC_MAX_FLOAT64 1
#else
#define CK_TILE_USE_AMD_BUFFER_ATOMIC_MAX_FLOAT64 0
#endif

#ifndef CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
#define CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS 0
#endif

#ifndef CK_TILE_WORKAROUND_SWDEV_XXXXXX_INT8_DS_WRITE_ISSUE
#define CK_TILE_WORKAROUND_SWDEV_XXXXXX_INT8_DS_WRITE_ISSUE 1
#endif

#ifndef CK_TILE_WORKAROUND_ROCM_6_1_SCRATCH_MEMORY_ISSUE
#if HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR == 1 && HIP_VERSION_PATCH >= 40091
#define CK_TILE_WORKAROUND_ROCM_6_1_SCRATCH_MEMORY_ISSUE 1
#else
#define CK_TILE_WORKAROUND_ROCM_6_1_SCRATCH_MEMORY_ISSUE 0
#endif
#endif

// workaround for ROCm 6.2 and later
#ifndef CK_TILE_WORKAROUND_ROCM_6_2_SCRATCH_MEMORY_ISSUE
#if (HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR == 2 && HIP_VERSION_PATCH >= 41133) || \
    (HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR == 3 && HIP_VERSION_PATCH >= 42131) || \
    (HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR > 3)
#define CK_TILE_WORKAROUND_ROCM_6_2_SCRATCH_MEMORY_ISSUE 1
#else
#define CK_TILE_WORKAROUND_ROCM_6_2_SCRATCH_MEMORY_ISSUE 0
#endif
#endif

#ifndef CK_TILE_DEBUG_LOG
#define CK_TILE_DEBUG_LOG 0
#endif

#ifndef __HIP_DEVICE_COMPILE__ // for host code
#define CK_TILE_BUFFER_RESOURCE_3RD_DWORD 0xffffffff
#elif defined(__gfx9__) || defined(__gfx916__) || defined(__gfx926__) || defined(__gfx928__) || \
    defined(__gfx936__) || defined(__gfx938__) || defined(__gfx92a__) ||                        \
    defined(__gfx946__) // for GPU code
#define CK_TILE_BUFFER_RESOURCE_3RD_DWORD 0x00020000
#endif

#ifndef CK_TILE_EXPERIMENTAL_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM
#define CK_TILE_EXPERIMENTAL_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM 1
#endif

#ifndef CK_TILE_USE_SUBDWORD_TILE_CAST
#define CK_TILE_USE_SUBDWORD_TILE_CAST 0
#endif

#ifndef CK_TILE_USE_PK_FP16_TILE_CAST
#define CK_TILE_USE_PK_FP16_TILE_CAST 1
#endif

// TODO: better solve this inside compiler
#ifndef CK_TILE_FMHA_FWD_FAST_EXP2
#define CK_TILE_FMHA_FWD_FAST_EXP2 0
#endif

#ifndef CK_TILE_BUFFER_LOAD_RAW_BF16_WA
#define CK_TILE_BUFFER_LOAD_RAW_BF16_WA 1
#endif

// workaround: compiler not emiting reciprocal instruction frm __frcp_rn()
#ifndef CK_TILE_WORKAROUND_SWDEV_383542
#define CK_TILE_WORKAROUND_SWDEV_383542 1
#endif

#ifndef CK_TILE_REFERENCE_MOE_SORTING_MOCK_ID
#define CK_TILE_REFERENCE_MOE_SORTING_MOCK_ID 1
#endif

#ifndef CK_TILE_REFERENCE_MOE_SORTING_MOCK_EXPERT_ID
#define CK_TILE_REFERENCE_MOE_SORTING_MOCK_EXPERT_ID 0
#endif

#ifndef __HIP_DEVICE_COMPILE__ // for host code
// #ifdef CK_TILE_USE_OCP_FP8
#define CK_TILE_USE_OCP_FP8 1   // Use OCP not FNUZ
// #else
// #define CK_TILE_USE_OCP_FP8 0
// #endif
#elif  defined(__gfx938__) || defined(__gfx950__) || defined(__gfx12__) // for GPU code
#define CK_TILE_USE_OCP_FP8 1
#else // for GPU code
#define CK_TILE_USE_OCP_FP8 0
#endif

#ifndef CK_TILE_WORKAROUND_AVOID_V_BFI_B32_INSTS
#define CK_TILE_WORKAROUND_AVOID_V_BFI_B32_INSTS 1
#endif

#ifndef CK_TILE_WA_ISSUE_2028
#define CK_TILE_WA_ISSUE_2028 0
#endif
