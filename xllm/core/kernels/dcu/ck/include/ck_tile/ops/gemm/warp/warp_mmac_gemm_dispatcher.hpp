// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm.hpp"

namespace ck_tile {

namespace impl {

template <typename AType,
          typename BType,
          typename CType,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPerWave,
          bool TransposeC,
          index_t MRepeat,
          index_t NRepeat,
          index_t MInterleave,
          index_t NInterleave,
          bool SwizzleA = false,
          bool UseABScale = false>
struct WarpGemmMmacDispatcher;

// clang-format off
// fp16
template<> struct WarpGemmMmacDispatcher<half_t, half_t, float, 32, 64, 32, false, 2, 2, 1, 4> { using Type = WarpGemmMmacF16F16F32_WT32x64x32_MR2NR2MI1NI4; };

template<> struct WarpGemmMmacDispatcher<half_t, half_t, float, 32, 64, 32, false, 2, 1, 1, 4> { using Type = WarpGemmMmacF16F16F32_WT32x64x32_MR2NR1MI1NI4; };

template<> struct WarpGemmMmacDispatcher<half_t, half_t, float, 16, 64, 32, false, 1, 1, 1, 4> { using Type = WarpGemmMmacF16F16F32_WT16x64x32_MR1NR1MI1NI4; };
template<> struct WarpGemmMmacDispatcher<half_t, half_t, float, 16, 64, 32, false, 1, 4, 1, 1> { using Type = WarpGemmMmacF16F16F32_WT16x64x32_MR1NR4MI1NI1; };
template<> struct WarpGemmMmacDispatcher<half_t, half_t, float, 32, 64, 32, false, 2, 4, 1, 1> { using Type = WarpGemmMmacF16F16F32_WT32x64x32_MR2NR4MI1NI1; };

// bf16
template<> struct WarpGemmMmacDispatcher<bf16_t, bf16_t, float, 16, 64, 32, false, 1, 4, 1, 1> { using Type = WarpGemmMmacBF16BF16F32_WT16x64x32_MR1NR4MI1NI1; };

//int8
template<> struct WarpGemmMmacDispatcher<int8_t, int8_t, int32_t, 16, 16, 32, false, 1, 1, 1, 1> { using Type = WarpGemmMmacI8I8I32_WT16x16x32_MR1NR1MI1NI1; };
template<> struct WarpGemmMmacDispatcher<int8_t, int8_t, int32_t, 32, 64, 32, false, 2, 1, 1, 4> { using Type = WarpGemmMmacI8I8I32_WT32x64x32_MR2NR1MI1NI4; };
template<> struct WarpGemmMmacDispatcher<int8_t, int8_t, int32_t, 32, 64, 64, false, 2, 4, 1, 1> { using Type = WarpGemmMmacI8I8I32_WT32x64x64_MR2NR4MI1NI1; };

template<> struct WarpGemmMmacDispatcher<int8_t, int8_t, float, 16, 16, 32, false, 1, 1, 1, 1, false, true> { using Type = WarpGemmMmacI8I8F32_WT16x16x32_MR1NR1MI1NI1; };
template<> struct WarpGemmMmacDispatcher<int8_t, int8_t, float, 16, 16, 64, false, 1, 1, 1, 1, false, true> { using Type = WarpGemmMmacI8I8F32_WT16x16x64_MR1NR1MI1NI1; };
template<> struct WarpGemmMmacDispatcher<int8_t, int8_t, float, 16, 32, 64, false, 1, 2, 1, 1, false, true> { using Type = WarpGemmMmacI8I8F32_WT16x32x64_MR1NR2MI1NI1; };
template<> struct WarpGemmMmacDispatcher<int8_t, int8_t, float, 16, 32, 128, false, 1, 2, 1, 1, false, true> { using Type = WarpGemmMmacI8I8F32_WT16x32x128_MR1NR2MI1NI1; };
template<> struct WarpGemmMmacDispatcher<int8_t, int8_t, float, 16, 64, 64, false, 1, 4, 1, 1, false, true> { using Type = WarpGemmMmacI8I8F32_WT16x64x64_MR1NR4MI1NI1; };
template<> struct WarpGemmMmacDispatcher<int8_t, int8_t, float, 16, 64, 128, false, 1, 4, 1, 1, false, true> { using Type = WarpGemmMmacI8I8F32_WT16x64x128_MR1NR4MI1NI1; };
template<> struct WarpGemmMmacDispatcher<int8_t, int8_t, float, 16, 64, 32, false, 1, 4, 1, 1, false, true> { using Type = WarpGemmMmacI8I8F32_WT16x64x32_MR1NR4MI1NI1; };
template<> struct WarpGemmMmacDispatcher<int8_t, int8_t, float, 32, 64, 32, false, 2, 4, 1, 1, false, true> { using Type = WarpGemmMmacI8I8F32_WT32x64x32_MR2NR4MI1NI1; };
template<> struct WarpGemmMmacDispatcher<int8_t, int8_t, float, 32, 64, 64, false, 2, 4, 1, 1, false, true> { using Type = WarpGemmMmacI8I8F32_WT32x64x64_MR2NR4MI1NI1; };

//MmacBlockGemmASmemBSmemCRegV1中 KPerBlockPerIter = KPerBlock / KIterPerWarp, 会按实际WarpGemmMmacI8I8I32_WT16x64x128_MR1NR1MI1NI4中‘kK’长度分割循环，所以WarpGemm中kKIter配少了，不影响正确性
template<> struct WarpGemmMmacDispatcher<int8_t, int8_t, int32_t, 16, 32, 128, false, 1, 1, 1, 2> { using Type = WarpGemmMmacI8I8I32_WT16x32x128_MR1NR1MI1NI2; };
template<> struct WarpGemmMmacDispatcher<int8_t, int8_t, int32_t, 16, 64, 128, false, 1, 1, 1, 4> { using Type = WarpGemmMmacI8I8I32_WT16x64x128_MR1NR1MI1NI4; };

// int4
template<> struct WarpGemmMmacDispatcher<pk_int4_t, pk_int4_t, int32_t, 16, 16, 64, false, 1, 1, 1, 1> { using Type = WarpGemmMmacI4I4I32_WT16x16x64_MR1NR1MI1NI1; };
template<> struct WarpGemmMmacDispatcher<pk_int4_t, pk_int4_t, int32_t, 16, 32, 64, false, 1, 2, 1, 1> { using Type = WarpGemmMmacI4I4I32_WT16x32x64_MR1NR2MI1NI1; };
template<> struct WarpGemmMmacDispatcher<pk_int4_t, pk_int4_t, int32_t, 16, 64, 64, false, 1, 4, 1, 1> { using Type = WarpGemmMmacI4I4I32_WT16x64x64_MR1NR4MI1NI1; };
template<> struct WarpGemmMmacDispatcher<pk_int4_t, pk_int4_t, int32_t, 16, 64, 64, false, 1, 2, 1, 2> { using Type = WarpGemmMmacI4I4I32_WT16x64x64_MR1NR2MI1NI2; };
template<> struct WarpGemmMmacDispatcher<pk_int4_t, pk_int4_t, int32_t, 32, 16, 64, false, 2, 1, 1, 1> { using Type = WarpGemmMmacI4I4I32_WT32x16x64_MR2NR1MI1NI1; };
template<> struct WarpGemmMmacDispatcher<pk_int4_t, pk_int4_t, int32_t, 32, 32, 64, false, 2, 2, 1, 1> { using Type = WarpGemmMmacI4I4I32_WT32x32x64_MR2NR2MI1NI1; };
template<> struct WarpGemmMmacDispatcher<pk_int4_t, pk_int4_t, int32_t, 32, 64, 64, false, 2, 4, 1, 1> { using Type = WarpGemmMmacI4I4I32_WT32x64x64_MR2NR4MI1NI1; };
template<> struct WarpGemmMmacDispatcher<pk_int4_t, pk_int4_t, int32_t, 32, 64, 64, false, 2, 2, 1, 2> { using Type = WarpGemmMmacI4I4I32_WT32x64x64_MR2NR2MI1NI2; };

// fp8 fp8
template<> struct WarpGemmMmacDispatcher<fp8_t, fp8_t, float, 16, 16, 32, false, 1, 1, 1, 1> { using Type = WarpGemmMmacfp8fp8f32_WT16x16x32_MR1NR1MI1NI1; };
template<> struct WarpGemmMmacDispatcher<fp8_t, fp8_t, float, 16, 32, 32, false, 1, 2, 1, 1> { using Type = WarpGemmMmacfp8fp8f32_WT16x32x32_MR1NR2MI1NI1; };
template<> struct WarpGemmMmacDispatcher<fp8_t, fp8_t, float, 16, 32, 64, false, 1, 2, 1, 1> { using Type = WarpGemmMmacfp8fp8f32_WT16x32x64_MR1NR2MI1NI1; };
template<> struct WarpGemmMmacDispatcher<fp8_t, fp8_t, float, 16, 64, 64, false, 1, 4, 1, 1> { using Type = WarpGemmMmacfp8fp8f32_WT16x64x64_MR1NR4MI1NI1; };
template<> struct WarpGemmMmacDispatcher<fp8_t, fp8_t, float, 32, 64, 32, false, 2, 1, 1, 4> { using Type = WarpGemmMmacfp8fp8f32_WT32x64x32_MR2NR1MI1NI4; };
template<> struct WarpGemmMmacDispatcher<fp8_t, fp8_t, float, 16, 64, 32, false, 1, 4, 1, 1> { using Type = WarpGemmMmacfp8fp8f32_WT16x64x32_MR1NR4MI1NI1; };
template<> struct WarpGemmMmacDispatcher<fp8_t, fp8_t, float, 32, 64, 32, false, 2, 4, 1, 1> { using Type = WarpGemmMmacfp8fp8f32_WT32x64x32_MR2NR4MI1NI1; };
template<> struct WarpGemmMmacDispatcher<fp8_t, fp8_t, float, 32, 64, 64, false, 2, 4, 1, 1> { using Type = WarpGemmMmacfp8fp8f32_WT32x64x64_MR2NR4MI1NI1; };

//fp8 fp8 scale
template<> struct WarpGemmMmacDispatcher<fp8_t, fp8_t, float, 16, 32, 128, false, 1, 2, 1, 1, false, true> { using Type = WarpScaleGemmMmacfp8fp8f32_WT16x32x128_MR1NR2MI1NI1; };

// fp8 bf8
template<> struct WarpGemmMmacDispatcher<fp8_t, bf8_t, float, 32, 64, 32, false, 2, 1, 1, 4> { using Type = WarpGemmMmacfp8bf8f32_WT32x64x32_MR2NR1MI1NI4; };

// bf8 fp8
template<> struct WarpGemmMmacDispatcher<bf8_t, fp8_t, float, 32, 64, 32, false, 2, 1, 1, 4> { using Type = WarpGemmMmacbf8fp8f32_WT32x64x32_MR2NR1MI1NI4; };

// bf8 bf8
template<> struct WarpGemmMmacDispatcher<bf8_t, bf8_t, float, 32, 64, 32, false, 2, 1, 1, 4> { using Type = WarpGemmMmacbf8bf8f32_WT32x64x32_MR2NR1MI1NI4; };
template<> struct WarpGemmMmacDispatcher<bf8_t, bf8_t, float, 32, 64, 32, false, 2, 4, 1, 1> { using Type = WarpGemmMmacbf8bf8f32_WT32x64x32_MR2NR4MI1NI1; };
template<> struct WarpGemmMmacDispatcher<bf8_t, bf8_t, float, 32, 64, 64, false, 2, 4, 1, 1> { using Type = WarpGemmMmacbf8bf8f32_WT32x64x64_MR2NR4MI1NI1; };

// clang-format on
} // namespace impl

template <typename AType,
          typename BType,
          typename CType,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPerWave,
          bool TransposeC,
          index_t MRepeat,
          index_t NRepeat,
          index_t MInterleave,
          index_t NInterleave,
          bool SwizzleA = false,
          bool UseABScale = false>
using WarpGemmMmacDispatcher = typename impl::WarpGemmMmacDispatcher<AType,
                                                                     BType,
                                                                     CType,
                                                                     MPerWave,
                                                                     NPerWave,
                                                                     KPerWave,
                                                                     TransposeC,
                                                                     MRepeat,
                                                                     NRepeat,
                                                                     MInterleave,
                                                                     NInterleave,
                                                                     SwizzleA,
                                                                     UseABScale>::Type;

} // namespace ck_tile
