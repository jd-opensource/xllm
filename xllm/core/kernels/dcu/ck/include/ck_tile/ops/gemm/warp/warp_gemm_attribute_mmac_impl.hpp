// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// FP16
// 16x16x16 a/b vec contains 4 elements.
struct WarpGemmAttributeMmacImplF16F16F32M16N16K16
{
    using ADataType = fp16_t;
    using BDataType = fp16_t;
    using CDataType = float;

    using AVecType = ext_vector_t<fp16_t, 4>;
    using BVecType = ext_vector_t<fp16_t, 4>;
    using CVecType = ext_vector_t<float, 4>;

    // mmac inst size
    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    // thread layout
    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 4; // 4 element per thread vec

    // mmac C layout
    static constexpr index_t kCMLane = 16;
    static constexpr index_t kCNLane = 4;

    static constexpr index_t kCMPerLane  = 1;
    static constexpr index_t kCN0PerLane = 4; // jump layout
    static constexpr index_t kCN1PerLane = 1;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__) || defined(__gfx92a__) || \
    defined(__gfx946__)
        c_vec = __builtin_hcu_mmac_f32_16x16x16_f16(a_vec, b_vec, c_vec);
#else
        ignore = c_vec;
        ignore = a_vec;
        ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__) || defined(__gfx92a__) || \
    defined(__gfx946__)
        return bit_cast<CVecType>(
            __builtin_hcu_mmac_f32_16x16x16_f16(a_vec, b_vec, fp32x4_t{0.f}));
#else
        ignore = a_vec;
        ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

// FP16
// 16x16x16 a/b vec contains 4 elements.
// C transposed
struct WarpGemmAttributeMmacImplF16F16F32M16N16K16TransC
    : public WarpGemmAttributeMmacImplF16F16F32M16N16K16
{
    using Base = WarpGemmAttributeMmacImplF16F16F32M16N16K16;

    using ADataType = typename Base::ADataType;
    using BDataType = typename Base::BDataType;
    using CDataType = typename Base::CDataType;

    using AVecType = typename Base::AVecType;
    using BVecType = typename Base::BVecType;
    using CVecType = typename Base::CVecType;

    // mmac inst size
    static constexpr index_t kM = Base::kM;
    static constexpr index_t kN = Base::kN;
    static constexpr index_t kK = Base::kK;

    // thread layout
    static constexpr index_t kAMLane     = Base::kAMLane;
    static constexpr index_t kBNLane     = Base::kBNLane;
    static constexpr index_t kABKLane    = Base::kABKLane;
    static constexpr index_t kABKPerLane = Base::kABKPerLane; // 4 element per thread vec

    // tranposed mmac C layout
    static constexpr index_t kCMLane = Base::kCNLane;
    static constexpr index_t kCNLane = Base::kCMLane;

    static constexpr index_t kCNPerLane  = Base::kCMPerLane;
    static constexpr index_t kCM0PerLane = Base::kCN0PerLane;
    static constexpr index_t kCM1PerLane = Base::kCN1PerLane;

    // FIXME: workaournd
    static constexpr index_t kCMPerLane  = -1;
    static constexpr index_t kCN0PerLane = -1;
    static constexpr index_t kCN1PerLane = -1;

    // c_vec += b_vec * a_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__) || defined(__gfx92a__) || \
    defined(__gfx946__)
        c_vec = __builtin_hcu_mmac_f32_16x16x16_f16(b_vec, a_vec, c_vec);
#else
        detail::swallow{a_vec, b_vec, c_vec};
#endif
    }

    // c_vec = b_vec * a_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__) || defined(__gfx92a__) || \
    defined(__gfx946__)
        return bit_cast<CVecType>(
            __builtin_hcu_mmac_f32_16x16x16_f16(b_vec, a_vec, fp32x4_t{0.f}));
#else
        detail::swallow{a_vec, b_vec};
        return CVecType{0.f};
#endif
    }
};

// FP16 LIT
// LIT: Won't change Lane distribution, but change distribution in lane
struct WarpGemmAttributeMmacImplF16F16F32M16N16K16Lit
{
    using ADataType = fp16_t;
    using BDataType = fp16_t;
    using CDataType = float;

    using AVecType = ext_vector_t<fp16_t, 4>;
    using BVecType = ext_vector_t<fp16_t, 4>;
    using CVecType = ext_vector_t<float, 4>;

    // mmac inst size
    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    // thread layout
    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 4; // 4 element per thread vec

    // mmac C layout
    static constexpr index_t kCMLane = 16;
    static constexpr index_t kCNLane = 4;

    static constexpr index_t kCMPerLane  = 1;
    static constexpr index_t kCN0PerLane = 1; // jump layout
    static constexpr index_t kCN1PerLane = 4;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx938__) || defined(__gfx946__)
        static constexpr bool lit = true;
        static constexpr bool lts = false;
        c_vec = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(a_vec, b_vec, c_vec, lit, lts);
#else
        ignore = c_vec;
        ignore = a_vec;
        ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx938__) || defined(__gfx946__)
        static constexpr bool lit = true;
        static constexpr bool lts = false;

        CVecType c_vec =
            __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(a_vec, b_vec, fp32x4_t{0.f}, lit, lts);

        return c_vec;
#else
        ignore = a_vec;
        ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

// FP16 LTS
// LTS: Change C lane distribution from MN to NM
struct WarpGemmAttributeMmacImplF16F16F32M16N16K16Lts
{
    using ADataType = fp16_t;
    using BDataType = fp16_t;
    using CDataType = float;

    using AVecType = ext_vector_t<fp16_t, 4>;
    using BVecType = ext_vector_t<fp16_t, 4>;
    using CVecType = ext_vector_t<float, 4>;

    // mmac inst size
    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    // thread layout
    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 4; // 4 element per thread vec

    // mmac C layout
    static constexpr index_t kCMLane = 4;
    static constexpr index_t kCNLane = 16;

    static constexpr index_t kCM0PerLane = 4;
    static constexpr index_t kCM1PerLane = 1; // continuous
    static constexpr index_t kCNPerLane  = 1;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx938__) || defined(__gfx946__)
        static constexpr bool lit = false;
        static constexpr bool lts = true;
        c_vec = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(a_vec, b_vec, c_vec, lit, lts);
#else
        ignore = c_vec;
        ignore = a_vec;
        ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx938__) || defined(__gfx946__)
        static constexpr bool lit = false;
        static constexpr bool lts = true;

        CVecType c_vec =
            __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(a_vec, b_vec, fp32x4_t{0.f}, lit, lts);

        return c_vec;
#else
        ignore = a_vec;
        ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

// FP16 LIT LTS
// Composed of LIT and LTS
struct WarpGemmAttributeMmacImplF16F16F32M16N16K16LitLts
{
    using ADataType = fp16_t;
    using BDataType = fp16_t;
    using CDataType = float;

    using AVecType = ext_vector_t<fp16_t, 4>;
    using BVecType = ext_vector_t<fp16_t, 4>;
    using CVecType = ext_vector_t<float, 4>;

    // mmac inst size
    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    // thread layout
    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 4; // 4 element per thread vec

    // mmac C layout
    static constexpr index_t kCMLane = 4;
    static constexpr index_t kCNLane = 16;

    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 4; // continuous
    static constexpr index_t kCNPerLane  = 1;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx938__) || defined(__gfx946__)
        static constexpr bool lit = true;
        static constexpr bool lts = true;
        c_vec = __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(a_vec, b_vec, c_vec, lit, lts);
#else
        ignore = c_vec;
        ignore = a_vec;
        ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx938__) || defined(__gfx946__)
        static constexpr bool lit = true;
        static constexpr bool lts = true;
        CVecType c_vec =
            __builtin_hcu_mmac_f32_16x16x16_f16_lit_lts(a_vec, b_vec, fp32x4_t{0.f}, lit, lts);

        return c_vec;
#else
        ignore = a_vec;
        ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

// Bf16
struct WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16
{
    using ADataType = bf16_t;
    using BDataType = bf16_t;
    using CDataType = float;

    using AVecType = ext_vector_t<bf16_t, 4>;
    using BVecType = ext_vector_t<bf16_t, 4>;
    using CVecType = ext_vector_t<float, 4>;

    // mmac inst size
    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    // thread layout
    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 4; // 4 element per thread vec

    // mmac C layout
    static constexpr index_t kCMLane = 16;
    static constexpr index_t kCNLane = 4;

    static constexpr index_t kCMPerLane  = 1;
    static constexpr index_t kCNPerLane  = 4;
    static constexpr index_t kCN0PerLane = 4; // jump layout
    static constexpr index_t kCN1PerLane = 1;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__) || defined(__gfx92a__) || \
    defined(__gfx946__)
        c_vec = __builtin_hcu_mmac_f32_16x16x16_bf16(a_vec, b_vec, c_vec);
#else
        ignore = c_vec;
        ignore = a_vec;
        ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__) || defined(__gfx92a__) || \
    defined(__gfx946__)
        return bit_cast<CVecType>(
            __builtin_hcu_mmac_f32_16x16x16_bf16(a_vec, b_vec, fp32x4_t{0.f}));
#else
        ignore = a_vec;
        ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

// BF16
// C transposed
struct WarpGemmAttributeMmacImplBF16BF16F32M16N16K16TransC
    : public WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16
{
    using Base = WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16;

    using ADataType = typename Base::ADataType;
    using BDataType = typename Base::BDataType;
    using CDataType = typename Base::CDataType;

    using AVecType = typename Base::AVecType;
    using BVecType = typename Base::BVecType;
    using CVecType = typename Base::CVecType;

    // mmac inst size
    static constexpr index_t kM = Base::kM;
    static constexpr index_t kN = Base::kN;
    static constexpr index_t kK = Base::kK;

    // thread layout
    static constexpr index_t kAMLane     = Base::kAMLane;
    static constexpr index_t kBNLane     = Base::kBNLane;
    static constexpr index_t kABKLane    = Base::kABKLane;
    static constexpr index_t kABKPerLane = Base::kABKPerLane; // 4 element per thread vec

    // tranposed mmac C layout
    static constexpr index_t kCMLane = Base::kCNLane;
    static constexpr index_t kCNLane = Base::kCMLane;

    static constexpr index_t kCNPerLane  = Base::kCMPerLane;
    static constexpr index_t kCM0PerLane = Base::kCN0PerLane;
    static constexpr index_t kCM1PerLane = Base::kCN1PerLane;

    // FIXME: workaournd
    static constexpr index_t kCMPerLane  = -1;
    static constexpr index_t kCN0PerLane = -1;
    static constexpr index_t kCN1PerLane = -1;

    // c_vec += b_vec * a_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__) || defined(__gfx92a__) || \
    defined(__gfx946__)
        c_vec = __builtin_hcu_mmac_f32_16x16x16_bf16(b_vec, a_vec, c_vec);
#else
        detail::swallow{a_vec, b_vec, c_vec};
#endif
    }

    // c_vec = b_vec * a_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__) || defined(__gfx92a__) || \
    defined(__gfx946__)
        return bit_cast<CVecType>(
            __builtin_hcu_mmac_f32_16x16x16_bf16(b_vec, a_vec, fp32x4_t{0.f}));
#else
        detail::swallow{a_vec, b_vec};
        return CVecType{0.f};
#endif
    }
};

// BF16 LIT
// LIT: Won't change Lane distribution, but change distribution in lane
struct WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16Lit
{
    using ADataType = fp16_t;
    using BDataType = fp16_t;
    using CDataType = float;

    using AVecType = ext_vector_t<fp16_t, 4>;
    using BVecType = ext_vector_t<fp16_t, 4>;
    using CVecType = ext_vector_t<float, 4>;

    // mmac inst size
    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    // thread layout
    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 4; // 4 element per thread vec

    // mmac C layout
    static constexpr index_t kCMLane = 16;
    static constexpr index_t kCNLane = 4;

    static constexpr index_t kCMPerLane = 1;
    static constexpr index_t kCNPerLane = 4; // continuouse

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx938__) || defined(__gfx946__)
        static constexpr bool lit = true;
        static constexpr bool lts = false;
        c_vec = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a_vec, b_vec, c_vec, lit, lts);
#else
        ignore = c_vec;
        ignore = a_vec;
        ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx938__) || defined(__gfx946__)
        static constexpr bool lit = true;
        static constexpr bool lts = false;
        CVecType c_vec =
            __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a_vec, b_vec, fp32x4_t{0.f}, lit, lts);

        return c_vec;
#else
        ignore = a_vec;
        ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

// BF16 LTS
// LTS: Change C lane distribution from MN to NM
struct WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16Lts
{
    using ADataType = fp16_t;
    using BDataType = fp16_t;
    using CDataType = float;

    using AVecType = ext_vector_t<fp16_t, 4>;
    using BVecType = ext_vector_t<fp16_t, 4>;
    using CVecType = ext_vector_t<float, 4>;

    // mmac inst size
    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    // thread layout
    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 4; // 4 element per thread vec

    // mmac C layout
    static constexpr index_t kCMLane = 4;
    static constexpr index_t kCNLane = 16;

    static constexpr index_t kCMPerLane = 1;
    static constexpr index_t kCNPerLane = 4; // jump layout

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx938__) || defined(__gfx946__)
        static constexpr bool lit = false;
        static constexpr bool lts = true;
        c_vec = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a_vec, b_vec, c_vec, lit, lts);
#else
        ignore = c_vec;
        ignore = a_vec;
        ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx938__) || defined(__gfx946__)
        static constexpr bool lit = false;
        static constexpr bool lts = true;
        CVecType c_vec =
            __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a_vec, b_vec, fp32x4_t{0.f}, lit, lts);

        return c_vec;
#else
        ignore = a_vec;
        ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

// BF16 LIT LTS
// Composed of LIT and LTS
struct WarpGemmAttributeMmacImplBf16Bf16F32M16N16K16LitLts
{
    using ADataType = fp16_t;
    using BDataType = fp16_t;
    using CDataType = float;

    using AVecType = ext_vector_t<fp16_t, 4>;
    using BVecType = ext_vector_t<fp16_t, 4>;
    using CVecType = ext_vector_t<float, 4>;

    // mmac inst size
    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    // thread layout
    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 4; // 4 element per thread vec

    // mmac C layout
    static constexpr index_t kCMLane = 4;
    static constexpr index_t kCNLane = 16;

    static constexpr index_t kCMPerLane = 1;
    static constexpr index_t kCNPerLane = 4; // continuous

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx938__) || defined(__gfx946__)
        static constexpr bool lit = true;
        static constexpr bool lts = true;
        c_vec = __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a_vec, b_vec, c_vec, lit, lts);
#else
        ignore = c_vec;
        ignore = a_vec;
        ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx938__) || defined(__gfx946__)
        static constexpr bool lit = true;
        static constexpr bool lts = true;
        CVecType c_vec =
            __builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts(a_vec, b_vec, fp32x4_t{0.f}, lit, lts);
        return c_vec;
#else
        ignore = a_vec;
        ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

//int8
struct WarpGemmAttributeMmacImplI8I8I32M16N16K32
{
    using ADataType = int8_t;
    using BDataType = int8_t;
    using CDataType = int32_t;

    using AComputeDataType = int32_t;
    using BComputeDataType = int32_t;
    using CComputeDataType = int32_t;

    using AVecType = ext_vector_t<int32_t, 2>;
    using BVecType = ext_vector_t<int32_t, 2>;
    using CVecType = ext_vector_t<int32_t, 4>;

    using AInt8VecType = ext_vector_t<int8_t, 8>;
    using BInt8VecType = ext_vector_t<int8_t, 8>;
    using CInt32VecType = ext_vector_t<int32_t, 4>;
    // mmac inst size
    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 32;

    // thread layout
    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 8; // 8 element per thread vec

    // mmac C layout
    static constexpr index_t kCMLane = 16;
    static constexpr index_t kCNLane = 4;

    static constexpr index_t kCMPerLane = 1;
    static constexpr index_t kCN0PerLane = 4; // jump layout
    static constexpr index_t kCN1PerLane = 1;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
        //c_vec = __builtin_hcu_mmac_f32_16x16x16_f16(a_vec, b_vec, c_vec);
        c_vec = __builtin_hcu_mmac_i32_16x16x32_i8(a_vec, b_vec, c_vec);
#else
        ignore = c_vec;
        ignore = a_vec;
        ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
        return bit_cast<CVecType>(
            //__builtin_hcu_mmac_f32_16x16x16_f16(a_vec, b_vec, fp32x4_t{0.f}));
            __builtin_hcu_mmac_i32_16x16x32_i8(a_vec, b_vec, int32x4_t{0}));
#else
        ignore = a_vec;
        ignore = b_vec;
        return CVecType{0};
#endif
    }

};

struct WarpGemmAttributeMmacImplI8I8I32M16N16K32TransC
    : public WarpGemmAttributeMmacImplI8I8I32M16N16K32
{
    using Base = WarpGemmAttributeMmacImplI8I8I32M16N16K32;

    using ADataType = typename Base::ADataType;
    using BDataType = typename Base::BDataType;
    using CDataType = typename Base::CDataType;

    using AComputeDataType = typename Base::AComputeDataType;
    using BComputeDataType = typename Base::BComputeDataType;
    using CComputeDataType = typename Base::CComputeDataType;

    using AVecType = typename Base::AVecType;
    using BVecType = typename Base::BVecType;
    using CVecType = typename Base::CVecType;

    using AInt8VecType = typename Base::AInt8VecType;
    using BInt8VecType = typename Base::BInt8VecType;
    using CInt32VecType = typename Base::CInt32VecType;

    // mmac inst size
    static constexpr index_t kM = Base::kM;
    static constexpr index_t kN = Base::kN;
    static constexpr index_t kK = Base::kK;

    // thread layout
    static constexpr index_t kAMLane     = Base::kAMLane;
    static constexpr index_t kBNLane     = Base::kBNLane;
    static constexpr index_t kABKLane    = Base::kABKLane;
    static constexpr index_t kABKPerLane = Base::kABKPerLane; // 8 element per thread vec

    // mmac C layout
    static constexpr index_t kCMLane = Base::kCNLane;           // 4
    static constexpr index_t kCNLane = Base::kCMLane;           // 16

    static constexpr index_t kCNPerLane  = Base::kCMPerLane;    // 1
    static constexpr index_t kCM0PerLane = Base::kCN0PerLane;   // 4
    static constexpr index_t kCM1PerLane = Base::kCN1PerLane;   // 1

    static constexpr index_t kCMPerLane  = -1;
    static constexpr index_t kCN0PerLane = -1;
    static constexpr index_t kCN1PerLane = -1;

    // c_vec += b_vec * a_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
        //c_vec = __builtin_hcu_mmac_f32_16x16x16_f16(a_vec, b_vec, c_vec);
        c_vec = __builtin_hcu_mmac_i32_16x16x32_i8(b_vec, a_vec, c_vec);
#else
        ignore = c_vec;
        ignore = a_vec;
        ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
        return bit_cast<CVecType>(
            //__builtin_hcu_mmac_f32_16x16x16_f16(a_vec, b_vec, fp32x4_t{0.f}));
            __builtin_hcu_mmac_i32_16x16x32_i8(b_vec, a_vec, int32x4_t{0}));
#else
        ignore = a_vec;
        ignore = b_vec;
        return CVecType{0};
#endif
    }

};

struct WarpGemmAttributeMmacImplI8I8F32M16N16K32Scale
{
    using ADataType = int8_t;
    using BDataType = int8_t;
    using CDataType = float;

    using AScaleType = float;
    using BScaleType = float;

    using AComputeDataType = int32_t;
    using BComputeDataType = int32_t;
    using CComputeDataType = float;

    using AVecType = ext_vector_t<int32_t, 2>;
    using BVecType = ext_vector_t<int32_t, 2>;
    using CVecType = ext_vector_t<float, 4>;
    using CVecComputeType = ext_vector_t<int32_t, 4>;

    using AInt8VecType = ext_vector_t<int8_t, 8>;
    using BInt8VecType = ext_vector_t<int8_t, 8>;
    using CInt32VecType = ext_vector_t<int32_t, 4>;
    using CFloat32VecType = ext_vector_t<float, 4>;
    // mmac inst size
    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 32;

    // thread layout
    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 8; // 8 element per thread vec

    // mmac C layout
    static constexpr index_t kCMLane = 16;
    static constexpr index_t kCNLane = 4;

    static constexpr index_t kCMPerLane = 1;
    static constexpr index_t kCN0PerLane = 4; // jump layout
    static constexpr index_t kCN1PerLane = 1;

    // c_vec += a_vec * b_vec scaled by shared K-channel factors
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   const AScaleType& a_scale,
                                   const BScaleType& b_scale) const
    {
#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
        auto acc = CInt32VecType{0, 0, 0, 0};
        acc      = __builtin_hcu_mmac_i32_16x16x32_i8(a_vec, b_vec, acc);
        const float scale = a_scale * b_scale;
        c_vec.x += static_cast<float>(acc.x) * scale;
        c_vec.y += static_cast<float>(acc.y) * scale;
        c_vec.z += static_cast<float>(acc.z) * scale;
        c_vec.w += static_cast<float>(acc.w) * scale;

#else
        ignore = c_vec;
        ignore = a_vec;
        ignore = b_vec;
        ignore = a_scale;
        ignore = b_scale;
#endif
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
        return bit_cast<CVecType>(
            //__builtin_hcu_mmac_f32_16x16x16_f16(a_vec, b_vec, fp32x4_t{0.f}));
            __builtin_hcu_mmac_i32_16x16x32_i8(a_vec, b_vec, int32x4_t{0}));
#else
        ignore = a_vec;
        ignore = b_vec;
        return CVecType{0};
#endif
    }

};

// int4
struct WarpGemmAttributeMmacImplI4I4I32M16N16K64
{
    using ADataType = pk_int4_t;
    using BDataType = pk_int4_t;
    using CDataType = int32_t;

    using AComputeDataType = int32_t;
    using BComputeDataType = int32_t;
    using CComputeDataType = int32_t;

    using AVecType = ext_vector_t<int32_t, 2>;
    using BVecType = ext_vector_t<int32_t, 2>;
    using CVecType = ext_vector_t<int32_t, 4>;

    using AInt4VecType = ext_vector_t<int8_t, 8>;
    using BInt4VecType = ext_vector_t<int8_t, 8>;
    using CInt32VecType = ext_vector_t<int32_t, 4>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 64;

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 8; // 8 packed pk_int4_t values = 16 int4 values

    static constexpr index_t kCMLane = 16;
    static constexpr index_t kCNLane = 4;

    static constexpr index_t kCMPerLane  = 1;
    static constexpr index_t kCN0PerLane = 4;
    static constexpr index_t kCN1PerLane = 1;

    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx938__)
        c_vec = __builtin_hcu_mmac_i32_16x16x64_i4(a_vec, b_vec, c_vec);
#else
        ignore = c_vec;
        ignore = a_vec;
        ignore = b_vec;
#endif
    }

    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx938__)
        return bit_cast<CVecType>(
            __builtin_hcu_mmac_i32_16x16x64_i4(a_vec, b_vec, int32x4_t{0}));
#else
        ignore = a_vec;
        ignore = b_vec;
        return CVecType{0};
#endif
    }
};

// FP8
template <typename AType_, typename BType_>
struct WarpGemmAttributeMmacImpl_f32_16x16x32_f8_base
{
    using ADataType = AType_;
    using BDataType = BType_;
    using CDataType = float;

    using AScaleType = float;
    using BScaleType = float;

    using AVecType = ext_vector_t<int32_t, 2>;
    using BVecType = ext_vector_t<int32_t, 2>;

    using AFp8Bf8VecType = ext_vector_t<ADataType, 8>;
    using BFp8Bf8VecType = ext_vector_t<BDataType, 8>;
    using CVecType = ext_vector_t<CDataType, 4>;

    using AComputeDataType = float;
    using BComputeDataType = float;
    using CComputeDataType = float;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 32;

    static constexpr index_t kAMLane     = 16;
    static constexpr index_t kBNLane     = 16;
    static constexpr index_t kABKLane    = 4;
    static constexpr index_t kABKPerLane = 8;

    // mmac C layout
    static constexpr index_t kCMLane = 16;
    static constexpr index_t kCNLane = 4;

    static constexpr index_t kCMPerLane = 1;
    static constexpr index_t kCN0PerLane = 4; // jump layout
    static constexpr index_t kCN1PerLane = 1;

    // static constexpr index_t kCNPerLane = 4;


    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   const AScaleType& a_scale,
                                   const BScaleType& b_scale) const
    {
#if defined(__gfx938__) || defined(__gfx946__)
    auto acc = CVecType{0, 0, 0, 0};
    const float scale = a_scale * b_scale;

    if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, fp8_t>)
        acc = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8(a_vec, b_vec, acc);
    else if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, bf8_t>)
        acc = __builtin_hcu_mmac_f32_16x16x32_fp8_bf8(a_vec, b_vec, acc);
    else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, fp8_t>)
        acc = __builtin_hcu_mmac_f32_16x16x32_bf8_fp8(a_vec, b_vec, acc);
    else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, bf8_t>)
        acc = __builtin_hcu_mmac_f32_16x16x32_bf8_bf8(a_vec, b_vec, acc);

    c_vec.x += static_cast<CDataType>(acc.x * scale);
    c_vec.y += static_cast<CDataType>(acc.y * scale);
    c_vec.z += static_cast<CDataType>(acc.z * scale);
    c_vec.w += static_cast<CDataType>(acc.w * scale);

#else
        ignore = c_vec;
        ignore = a_vec;
        ignore = b_vec;
        ignore = a_scale;
        ignore = b_scale;
#endif
    }

    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx938__) || defined(__gfx946__)
    if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, fp8_t>)
        c_vec = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8(a_vec, b_vec, c_vec);
    else if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, bf8_t>)
        c_vec = __builtin_hcu_mmac_f32_16x16x32_fp8_bf8(a_vec, b_vec, c_vec);
    else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, fp8_t>)
        c_vec = __builtin_hcu_mmac_f32_16x16x32_bf8_fp8(a_vec, b_vec, c_vec);
    else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, bf8_t>)
        c_vec = __builtin_hcu_mmac_f32_16x16x32_bf8_bf8(a_vec, b_vec, c_vec);
#else
        ignore = c_vec;
        ignore = a_vec;
        ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
#if defined(__gfx938__) || defined(__gfx946__)
        if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, fp8_t>)
            return bit_cast<CVecType>(__builtin_hcu_mmac_f32_16x16x32_fp8_fp8(
                a_vec, b_vec, CVecType{0.f}));
        else if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, bf8_t>)
            return bit_cast<CVecType>(__builtin_hcu_mmac_f32_16x16x32_fp8_bf8(
                a_vec, b_vec, CVecType{0.f}));
        else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, fp8_t>)
            return bit_cast<CVecType>(__builtin_hcu_mmac_f32_16x16x32_bf8_fp8(
                a_vec, b_vec, CVecType{0.f}));
        else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, bf8_t>)
            return bit_cast<CVecType>(__builtin_hcu_mmac_f32_16x16x32_bf8_bf8(
                a_vec, b_vec, CVecType{0.f}));
#else
        ignore = a_vec;
        ignore = b_vec;
        return CVecType{0};
#endif
    }

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AFp8Bf8VecType& a_vec, const BFp8Bf8VecType& b_vec) const
    {
#if defined(__gfx938__) || defined(__gfx946__)
        if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, fp8_t>)
            c_vec = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8(
                bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), c_vec);
        else if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, bf8_t>)
            c_vec = __builtin_hcu_mmac_f32_16x16x32_fp8_bf8(
                bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), c_vec);
        else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, fp8_t>)
            c_vec = __builtin_hcu_mmac_f32_16x16x32_bf8_fp8(
                bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), c_vec);
        else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, bf8_t>)
            c_vec = __builtin_hcu_mmac_f32_16x16x32_bf8_bf8(
                bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), c_vec);
#else
        ignore = c_vec;
        ignore = a_vec;
        ignore = b_vec;
#endif
    }

    // c_vec = a_vec * b_vec
    CK_TILE_DEVICE CVecType operator()(const AFp8Bf8VecType& a_vec, const BFp8Bf8VecType& b_vec) const
    {
#if defined(__gfx938__) || defined(__gfx946__)
        if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, fp8_t>)
            return bit_cast<CVecType>(__builtin_hcu_mmac_f32_16x16x32_fp8_fp8(
                bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), CVecType{0.f}));
        else if constexpr(std::is_same_v<ADataType, fp8_t> && std::is_same_v<BDataType, bf8_t>)
            return bit_cast<CVecType>(__builtin_hcu_mmac_16x16x32_fp8_bf8(
                bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), CVecType{0.f}));
        else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, fp8_t>)
            return bit_cast<CVecType>(__builtin_hcu_mmac_16x16x32_bf8_fp8(
                bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), CVecType{0.f}));
        else if constexpr(std::is_same_v<ADataType, bf8_t> && std::is_same_v<BDataType, bf8_t>)
            return bit_cast<CVecType>(__builtin_hcu_mmac_16x16x32_bf8_bf8(
                bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), CVecType{0.f}));
#else
        ignore = a_vec;
        ignore = b_vec;
        return CVecType{0.f};
#endif
    }
};

using WarpGemmAttributeMmacImpl_f32_16x16x32_fp8_fp8 =
    WarpGemmAttributeMmacImpl_f32_16x16x32_f8_base<fp8_t, fp8_t>;
using WarpGemmAttributeMmacImpl_f32_16x16x32_fp8_bf8 =
    WarpGemmAttributeMmacImpl_f32_16x16x32_f8_base<fp8_t, bf8_t>;
using WarpGemmAttributeMmacImpl_f32_16x16x32_bf8_fp8 =
    WarpGemmAttributeMmacImpl_f32_16x16x32_f8_base<bf8_t, fp8_t>;
using WarpGemmAttributeMmacImpl_f32_16x16x32_bf8_bf8 =
    WarpGemmAttributeMmacImpl_f32_16x16x32_f8_base<bf8_t, bf8_t>;

} // namespace ck_tile
