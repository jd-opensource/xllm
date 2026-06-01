// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mmac.hpp"

namespace ck_tile {
// Note:WT refers to Wave Tile Shape, MR refers to M Repeat, NR refers to N Repeat,
// MI refers to M Interleave, NI refers to N Interleave, KIterate refers to K Iterate.

// fp16
// FIXME:NR=2 X NI=4 = 16 X 8 = 128 not 64
using WarpGemmMmacF16F16F32_WT32x64x32_MR2NR2MI1NI4 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 2, 2, 1, 4, 2>>;

using WarpGemmMmacF16F16F32_WT32x64x32_MR2NR1MI1NI4 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 2, 1, 1, 4, 2>>;

using WarpGemmMmacF16F16F32_WT32x32x16_MR2NR2MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 2, 2, 1, 1, 1>>;

using WarpGemmMmacF16F16F32_WT16x64x32_MR1NR4MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 1, 4, 1, 1, 2>>;
using WarpGemmMmacF16F16F32_WT16x64x32_MR1NR1MI1NI4 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 1, 1, 1, 4, 2>>;

using WarpGemmMmacF16F16F32_WT32x64x32_MR2NR4MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 2, 4, 1, 1, 2>>;

using WarpGemmMmacF16F16F32_WT16x32x128_MR1NR1MI1NI2 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 1, 1, 1, 2, 8>>;

using WarpGemmMmacF16F16F32_WT16x16x128_MR1NR1MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 1, 1, 1, 1, 8>>;

// bf16
using WarpGemmMmacBF16BF16F32_WT16x32x128_MR1NR1MI1NI2 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 1, 1, 2, 8>>;

using WarpGemmMmacBF16BF16F32_WT16x16x128_MR1NR1MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 1, 1, 1, 8>>;

using WarpGemmMmacBF16BF16F32_WT16x64x32_MR1NR4MI1NI1 = WarpGemmImpl<
    WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16, 1, 4, 1, 1, 2>>;

// v2 refers to KIterate not continuous in KPerLane
using WarpGemmMmacF16F16F32_WT32x16x256_MR2NR1MI1NI1_TRANSC = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC_v2<WarpGemmAttributeMmacImplF16F16F32M16N16K16TransC,
                                           2,
                                           1,
                                           1,
                                           1,
                                           16>>;

// moe matrix B loaded into none swizzled lds
using WarpGemmMmacF16F16F32_WT16x16x128_MR1NR1MI1NI1_TRANSC = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC_v2<WarpGemmAttributeMmacImplF16F16F32M16N16K16TransC,
                                           1,
                                           1,
                                           1,
                                           1,
                                           8>>;

// moe matrix B loaded into swizzled lds
using WarpGemmMmacF16F16F32_WT16x16x64_MR1NR1MI1NI1_TRANSC = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC_v3<WarpGemmAttributeMmacImplF16F16F32M16N16K16TransC,
                                           1,
                                           1,
                                           1,
                                           1,
                                           4>>; 

// moe matrix B loaded into registers directly
using WarpGemmMmacF16F16F32_WT16x32x128_MR1NR2MI1NI1_TRANSC = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImplF16F16F32M16N16K16TransC,
                                        1,
                                        2,
                                        1,
                                        1,
                                        8>>;

using WarpGemmMmacF16F16F32_WT16x32x64_MR1NR2MI1NI1_TRANSC = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImplF16F16F32M16N16K16TransC,
                                        1,
                                        2,
                                        1,
                                        1,
                                        4>>;
// bf16
using WarpGemmMmacBF16BF16F32_WT16x32x128_MR1NR2MI1NI1_TRANSC = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImplBF16BF16F32M16N16K16TransC,
                                        1,
                                        2,
                                        1,
                                        1,
                                        8>>;

using WarpGemmMmacBF16BF16F32_WT16x32x64_MR1NR2MI1NI1_TRANSC = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImplBF16BF16F32M16N16K16TransC,
                                        1,
                                        2,
                                        1,
                                        1,
                                        4>>;

// moe preshuffle matrix B loaded into registers directly
using WarpGemmMmacF16F16F32_WT16x32x128_MR1NR1MI1NI2_Preshuffle = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKShuffle<WarpGemmAttributeMmacImplF16F16F32M16N16K16, 1, 1, 1, 2, 8>>;

using WarpGemmMmacF16F16F32_WT16x32x128_MR1NR2MI1NI1_Preshuffle = WarpGemmImpl<
    WarpGemmAttributeMmacIterateKTransC_Shuffle<WarpGemmAttributeMmacImplF16F16F32M16N16K16TransC,
                                                1,
                                                2,
                                                1,
                                                1,
                                                8>>;

//int8
using WarpGemmMmacI8I8I32_WT16x16x32_MR1NR1MI1NI1 = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImplI8I8I32M16N16K32, 1, 1, 1, 1, 1>>;
using WarpGemmMmacI8I8I32_WT32x64x32_MR2NR1MI1NI4 = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImplI8I8I32M16N16K32, 2, 1, 1, 4, 1>>;
using WarpGemmMmacI8I8I32_WT32x64x64_MR2NR4MI1NI1 = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImplI8I8I32M16N16K32, 2, 4, 1, 1, 2>>;

// for MOE GEMM0
using WarpGemmMmacI8I8I32_WT16x16x64_MR1NR1MI1NI1 = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImplI8I8I32M16N16K32, 1, 1, 1, 1, 2>>;
using WarpGemmMmacI8I8I32_WT16x16x128_MR1NR1MI1NI1 = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImplI8I8I32M16N16K32, 1, 1, 1, 1, 4>>;
using WarpGemmMmacI8I8I32_WT16x32x128_MR1NR1MI1NI2 = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImplI8I8I32M16N16K32, 1, 1, 1, 2, 4>>;
using WarpGemmMmacI8I8I32_WT16x64x128_MR1NR1MI1NI4 = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImplI8I8I32M16N16K32, 1, 1, 1, 4, 4>>;
using WarpGemmMmacI8I8I32_WT16x32x128_MR1NR1MI1NI2_Preshuffle = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateKShuffle<WarpGemmAttributeMmacImplI8I8I32M16N16K32, 1, 1, 1, 2, 4>>;

// including extra scales for matrix A and B
using WarpGemmMmacI8I8F32_WT16x16x32_MR1NR1MI1NI1 = WarpInt8ScaleChannelGemmImpl<WarpGemmAttributeInt8ScaleChannelMmacIterateK<WarpGemmAttributeMmacImplI8I8F32M16N16K32Scale, 1, 1, 1, 1, 1>>;
using WarpGemmMmacI8I8F32_WT16x16x64_MR1NR1MI1NI1 = WarpInt8ScaleChannelGemmImpl<WarpGemmAttributeInt8ScaleChannelMmacIterateK<WarpGemmAttributeMmacImplI8I8F32M16N16K32Scale, 1, 1, 1, 1, 2>>;
using WarpGemmMmacI8I8F32_WT16x32x64_MR1NR2MI1NI1 = WarpInt8ScaleChannelGemmImpl<WarpGemmAttributeInt8ScaleChannelMmacIterateK<WarpGemmAttributeMmacImplI8I8F32M16N16K32Scale, 1, 2, 1, 1, 2>>;
using WarpGemmMmacI8I8F32_WT16x32x128_MR1NR2MI1NI1 = WarpInt8ScaleChannelGemmImpl<WarpGemmAttributeInt8ScaleChannelMmacIterateK<WarpGemmAttributeMmacImplI8I8F32M16N16K32Scale, 1, 2, 1, 1, 4>>;
using WarpGemmMmacI8I8F32_WT16x64x64_MR1NR4MI1NI1 = WarpInt8ScaleChannelGemmImpl<WarpGemmAttributeInt8ScaleChannelMmacIterateK<WarpGemmAttributeMmacImplI8I8F32M16N16K32Scale, 1, 4, 1, 1, 2>>;
using WarpGemmMmacI8I8F32_WT16x64x128_MR1NR4MI1NI1 = WarpInt8ScaleChannelGemmImpl<WarpGemmAttributeInt8ScaleChannelMmacIterateK<WarpGemmAttributeMmacImplI8I8F32M16N16K32Scale, 1, 4, 1, 1, 4>>;
using WarpGemmMmacI8I8F32_WT16x64x32_MR1NR4MI1NI1 = WarpInt8ScaleChannelGemmImpl<WarpGemmAttributeInt8ScaleChannelMmacIterateK<WarpGemmAttributeMmacImplI8I8F32M16N16K32Scale, 1, 4, 1, 1, 1>>;
using WarpGemmMmacI8I8F32_WT32x64x32_MR2NR4MI1NI1 = WarpInt8ScaleChannelGemmImpl<WarpGemmAttributeInt8ScaleChannelMmacIterateK<WarpGemmAttributeMmacImplI8I8F32M16N16K32Scale, 2, 4, 1, 1, 1>>;
using WarpGemmMmacI8I8F32_WT32x64x64_MR2NR4MI1NI1 = WarpInt8ScaleChannelGemmImpl<WarpGemmAttributeInt8ScaleChannelMmacIterateK<WarpGemmAttributeMmacImplI8I8F32M16N16K32Scale, 2, 4, 1, 1, 2>>;


// for MOE GEMM1
using WarpGemmMmacI8I8I32_WT16x16x64_MR1NR1MI1NI1_TRANSC = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateKTransC<WarpGemmAttributeMmacImplI8I8I32M16N16K32TransC, 1, 1, 1, 1, 2>>;
using WarpGemmMmacI8I8I32_WT16x32x64_MR1NR2MI1NI1_TRANSC = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateKTransC<WarpGemmAttributeMmacImplI8I8I32M16N16K32TransC, 1, 2, 1, 1, 2>>;
using WarpGemmMmacI8I8I32_WT16x32x128_MR1NR2MI1NI1_TRANSC = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateKTransC<WarpGemmAttributeMmacImplI8I8I32M16N16K32TransC, 1, 2, 1, 1, 4>>;
using WarpGemmMmacI8I8I32_WT16x32x256_MR1NR2MI1NI1_TRANSC = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateKTransC<WarpGemmAttributeMmacImplI8I8I32M16N16K32TransC, 1, 2, 1, 1, 8>>;
using WarpGemmMmacI8I8I32_WT16x64x128_MR1NR4MI1NI1_TRANSC = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateKTransC<WarpGemmAttributeMmacImplI8I8I32M16N16K32TransC, 1, 4, 1, 1, 4>>;
using WarpGemmMmacI8I8I32_WT16x32x128_MR1NR2MI1NI1_TRANSC_Preshuffle = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateKTransC_Shuffle<WarpGemmAttributeMmacImplI8I8I32M16N16K32TransC, 1, 2, 1, 1, 4>>;
using WarpGemmMmacI8I8I32_WT16x32x256_MR1NR2MI1NI1_TRANSC_Preshuffle = WarpInt8GemmImpl<WarpGemmAttributeInt8MmacIterateKTransC_Shuffle<WarpGemmAttributeMmacImplI8I8I32M16N16K32TransC, 1, 2, 1, 1, 8>>;

// int4
using WarpGemmMmacI4I4I32_WT16x16x64_MR1NR1MI1NI1 =
    WarpInt4GemmImpl<WarpGemmAttributeInt4MmacIterateK<
        WarpGemmAttributeMmacImplI4I4I32M16N16K64, 1, 1, 1, 1, 1>>;
using WarpGemmMmacI4I4I32_WT16x64x64_MR1NR4MI1NI1 =
    WarpInt4GemmImpl<WarpGemmAttributeInt4MmacIterateK<
        WarpGemmAttributeMmacImplI4I4I32M16N16K64, 1, 4, 1, 1, 1>>;
using WarpGemmMmacI4I4I32_WT16x64x64_MR1NR2MI1NI2 =
    WarpInt4GemmImpl<WarpGemmAttributeInt4MmacIterateK<
        WarpGemmAttributeMmacImplI4I4I32M16N16K64, 1, 2, 1, 2, 1>>;
using WarpGemmMmacI4I4I32_WT16x32x64_MR1NR2MI1NI1 =
    WarpInt4GemmImpl<WarpGemmAttributeInt4MmacIterateK<
        WarpGemmAttributeMmacImplI4I4I32M16N16K64, 1, 2, 1, 1, 1>>;
using WarpGemmMmacI4I4I32_WT32x16x64_MR2NR1MI1NI1 =
    WarpInt4GemmImpl<WarpGemmAttributeInt4MmacIterateK<
        WarpGemmAttributeMmacImplI4I4I32M16N16K64, 2, 1, 1, 1, 1>>;
using WarpGemmMmacI4I4I32_WT32x32x64_MR2NR2MI1NI1 =
    WarpInt4GemmImpl<WarpGemmAttributeInt4MmacIterateK<
        WarpGemmAttributeMmacImplI4I4I32M16N16K64, 2, 2, 1, 1, 1>>;
using WarpGemmMmacI4I4I32_WT32x64x64_MR2NR4MI1NI1 =
    WarpInt4GemmImpl<WarpGemmAttributeInt4MmacIterateK<
        WarpGemmAttributeMmacImplI4I4I32M16N16K64, 2, 4, 1, 1, 1>>;
using WarpGemmMmacI4I4I32_WT32x64x64_MR2NR2MI1NI2 =
    WarpInt4GemmImpl<WarpGemmAttributeInt4MmacIterateK<
        WarpGemmAttributeMmacImplI4I4I32M16N16K64, 2, 2, 1, 2, 1>>;

//fp8 fp8
using WarpGemmMmacfp8fp8f32_WT16x16x32_MR1NR1MI1NI1 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_fp8_fp8, 1, 1, 1, 1, 1>>;
using WarpGemmMmacfp8fp8f32_WT16x32x32_MR1NR2MI1NI1 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_fp8_fp8, 1, 2, 1, 1, 1>>;
using WarpGemmMmacfp8fp8f32_WT16x32x64_MR1NR2MI1NI1 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_fp8_fp8, 1, 2, 1, 1, 2>>;
using WarpGemmMmacfp8fp8f32_WT16x64x32_MR1NR4MI1NI1 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_fp8_fp8, 1, 4, 1, 1, 1>>;
using WarpGemmMmacfp8fp8f32_WT16x64x64_MR1NR4MI1NI1 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_fp8_fp8, 1, 4, 1, 1, 2>>;
using WarpGemmMmacfp8fp8f32_WT32x64x32_MR2NR1MI1NI4 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_fp8_fp8, 2, 1, 1, 4, 1>>;
using WarpGemmMmacfp8fp8f32_WT32x64x32_MR2NR4MI1NI1 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_fp8_fp8, 2, 4, 1, 1, 1>>;
using WarpGemmMmacfp8fp8f32_WT32x64x64_MR2NR4MI1NI1 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_fp8_fp8, 2, 4, 1, 1, 2>>;

//fp8 fp8 scale
using WarpScaleGemmMmacfp8fp8f32_WT16x32x128_MR1NR2MI1NI1 = WarpFp8Bf8ScaleChannelGemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_fp8_fp8, 1, 2, 1, 1, 4>>;

//fp8 bf8
using WarpGemmMmacfp8bf8f32_WT32x64x32_MR2NR1MI1NI4 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_fp8_bf8, 2, 1, 1, 4, 1>>;

//bf8 fp8
using WarpGemmMmacbf8fp8f32_WT32x64x32_MR2NR1MI1NI4 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_bf8_fp8, 2, 1, 1, 4, 1>>;

//bf8 bf8
using WarpGemmMmacbf8bf8f32_WT32x64x32_MR2NR1MI1NI4 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_bf8_bf8, 2, 1, 1, 4, 1>>;
using WarpGemmMmacbf8bf8f32_WT32x64x32_MR2NR4MI1NI1 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_bf8_bf8, 2, 4, 1, 1, 1>>;
using WarpGemmMmacbf8bf8f32_WT32x64x64_MR2NR4MI1NI1 = WarpFp8Bf8GemmImpl<WarpGemmAttributeFp8Bf8MmacIterateK<WarpGemmAttributeMmacImpl_f32_16x16x32_bf8_bf8, 2, 4, 1, 1, 2>>;

} // namespace ck_tile
