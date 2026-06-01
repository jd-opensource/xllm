// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
namespace ck_tile {

template <typename WarpGemmAttribute_>
struct WarpGemmImpl
{
    using WarpGemmAttribute = remove_cvref_t<WarpGemmAttribute_>;

    static constexpr index_t kM = WarpGemmAttribute::kM;
    static constexpr index_t kN = WarpGemmAttribute::kN;
    static constexpr index_t kK = WarpGemmAttribute::kK;
    static constexpr index_t kKPerThread = WarpGemmAttribute::kKPerThread;

    using ADataType = typename WarpGemmAttribute::ADataType;
    using BDataType = typename WarpGemmAttribute::BDataType;
    using CDataType = typename WarpGemmAttribute::CDataType;

    using AWarpDstrEncoding = typename WarpGemmAttribute::AWarpDstrEncoding;
    using BWarpDstrEncoding = typename WarpGemmAttribute::BWarpDstrEncoding;
    using CWarpDstrEncoding = typename WarpGemmAttribute::CWarpDstrEncoding;
    using CWarpOutputDstrEncoding = typename WarpGemmAttribute::CWarpOutputDstrEncoding;

    using AWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(AWarpDstrEncoding{}))>;
    using BWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(BWarpDstrEncoding{}))>;
    using CWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(CWarpDstrEncoding{}))>;
    using CWarpOutputDstr = remove_cvref_t<decltype(make_static_tile_distribution(CWarpOutputDstrEncoding{}))>;


    using AWarpTensor = static_distributed_tensor<ADataType, AWarpDstr>;
    using BWarpTensor = static_distributed_tensor<BDataType, BWarpDstr>;
    using CWarpTensor = static_distributed_tensor<CDataType, CWarpDstr>;
    using CWarpOutputTensor = static_distributed_tensor<CDataType, CWarpOutputDstr>;

    CK_TILE_DEVICE void operator()(CWarpTensor& c, const AWarpTensor& a, const BWarpTensor& b) const
    {
#if CK_TILE_WORKAROUND_AVOID_V_BFI_B32_INSTS == 1
        WarpGemmAttribute{}(
            c.get_thread_buffer().template get_as<typename WarpGemmAttribute::Impl::CVecType>(),
            a.get_thread_buffer().template get_as<typename WarpGemmAttribute::Impl::AVecType>(),
            b.get_thread_buffer().template get_as<typename WarpGemmAttribute::Impl::BVecType>());
#else
        using AVec = ext_vector_t<ADataType, AWarpTensor::get_thread_buffer_size()>;
        using BVec = ext_vector_t<BDataType, BWarpTensor::get_thread_buffer_size()>;
        using CVec = ext_vector_t<CDataType, CWarpTensor::get_thread_buffer_size()>;

        constexpr auto I0 = number<0>{};

        const auto a_vec = a.get_thread_buffer().template get_as<AVec>()[I0];
        const auto b_vec = b.get_thread_buffer().template get_as<BVec>()[I0];
        auto c_vec       = c.get_thread_buffer().template get_as<CVec>()[I0];

        // c_vec += a_vec * b_vec
        WarpGemmAttribute{}(c_vec, a_vec, b_vec);
        c.get_thread_buffer().template set_as<CVec>(I0, c_vec);
#endif
    }

    CK_TILE_DEVICE auto operator()(const AWarpTensor& a, const BWarpTensor& b) const
    {
        CWarpTensor c;

        using AVec = ext_vector_t<ADataType, AWarpTensor::get_thread_buffer_size()>;
        using BVec = ext_vector_t<BDataType, BWarpTensor::get_thread_buffer_size()>;
        using CVec = ext_vector_t<CDataType, CWarpTensor::get_thread_buffer_size()>;

        constexpr auto I0 = number<0>{};

        const auto a_vec = a.get_thread_buffer().template get_as<AVec>()[I0];
        const auto b_vec = b.get_thread_buffer().template get_as<BVec>()[I0];

        // c_vec = a_vec * b_vec
        auto c_vec = WarpGemmAttribute{}(a_vec, b_vec);

        c.get_thread_buffer().template set_as<CVec>(I0, c_vec);

        return c;
    }

    CK_TILE_DEVICE auto MakeCOutputLayout(const CWarpTensor& c) const
    {

        return WarpGemmAttribute{}.MakeCOutputLayout(c);
    }
};

template <typename WarpGemmAttribute_>
struct WarpInt8GemmImpl
{
    using WarpGemmAttribute = remove_cvref_t<WarpGemmAttribute_>;

    static constexpr index_t kM = WarpGemmAttribute::kM;
    static constexpr index_t kN = WarpGemmAttribute::kN;
    static constexpr index_t kK = WarpGemmAttribute::kK;
    static constexpr index_t kKPerThread = WarpGemmAttribute::kKPerThread;

    using ADataType = typename WarpGemmAttribute::ADataType;
    using BDataType = typename WarpGemmAttribute::BDataType;
    using CDataType = typename WarpGemmAttribute::CDataType;
    
    using AComputeDataType = typename WarpGemmAttribute::AComputeDataType;
    using BComputeDataType = typename WarpGemmAttribute::BComputeDataType;
    using CComputeDataType = typename WarpGemmAttribute::CComputeDataType;

    using AWarpDstrEncoding = typename WarpGemmAttribute::AWarpDstrEncoding;
    using BWarpDstrEncoding = typename WarpGemmAttribute::BWarpDstrEncoding;
    using CWarpDstrEncoding = typename WarpGemmAttribute::CWarpDstrEncoding;
    using CWarpOutputDstrEncoding = typename WarpGemmAttribute::CWarpOutputDstrEncoding;

    using AWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(AWarpDstrEncoding{}))>;
    using BWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(BWarpDstrEncoding{}))>;
    using CWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(CWarpDstrEncoding{}))>;
    using CWarpOutputDstr = remove_cvref_t<decltype(make_static_tile_distribution(CWarpOutputDstrEncoding{}))>;


    using AWarpTensor = static_distributed_tensor<ADataType, AWarpDstr>;
    using BWarpTensor = static_distributed_tensor<BDataType, BWarpDstr>;
    using CWarpTensor = static_distributed_tensor<CDataType, CWarpDstr>;
    using CWarpOutputTensor = static_distributed_tensor<CDataType, CWarpOutputDstr>;

    CK_TILE_DEVICE void operator()(CWarpTensor& c, const AWarpTensor& a, const BWarpTensor& b) const
    {
// #if CK_TILE_WORKAROUND_AVOID_V_BFI_B32_INSTS == 1
//         WarpGemmAttribute{}(
//             c.get_thread_buffer().template get_as<typename WarpGemmAttribute::Impl::CVecType>(),
//             a.get_thread_buffer().template get_as<typename WarpGemmAttribute::Impl::AVecType>(),
//             b.get_thread_buffer().template get_as<typename WarpGemmAttribute::Impl::BVecType>());
// #else
        // static_assert(AWarpTensor::get_thread_buffer_size()==8*1*2,"-------------------------11--------------------------");
        // static_assert(BWarpTensor::get_thread_buffer_size()==8*1*4,"-------------------------12--------------------------");
        // static_assert(CWarpTensor::get_thread_buffer_size()==8*1*4,"-------------------------13--------------------------");
        using AVec = ext_vector_t<ADataType, AWarpTensor::get_thread_buffer_size()>;
        using BVec = ext_vector_t<BDataType, BWarpTensor::get_thread_buffer_size()>;
        using CVec = ext_vector_t<CDataType, CWarpTensor::get_thread_buffer_size()>;

        constexpr auto I0 = number<0>{};

        const auto a_vec = a.get_thread_buffer().template get_as<AVec>()[I0];
        const auto b_vec = b.get_thread_buffer().template get_as<BVec>()[I0];
        auto c_vec       = c.get_thread_buffer().template get_as<CVec>()[I0];

        // c_vec += a_vec * b_vec
        WarpGemmAttribute{}(c_vec, a_vec, b_vec);

        c.get_thread_buffer().template set_as<CVec>(I0, c_vec);
// #endif
    }

    CK_TILE_DEVICE auto operator()(const AWarpTensor& a, const BWarpTensor& b) const
    {
        CWarpTensor c;

        using AVec = ext_vector_t<ADataType, AWarpTensor::get_thread_buffer_size()>;
        using BVec = ext_vector_t<BDataType, BWarpTensor::get_thread_buffer_size()>;
        using CVec = ext_vector_t<CDataType, CWarpTensor::get_thread_buffer_size()>;

        constexpr auto I0 = number<0>{};

        const auto a_vec = a.get_thread_buffer().template get_as<AVec>()[I0];
        const auto b_vec = b.get_thread_buffer().template get_as<BVec>()[I0];

        // c_vec = a_vec * b_vec
        auto c_vec = WarpGemmAttribute{}(a_vec, b_vec);

        c.get_thread_buffer().template set_as<CVec>(I0, c_vec);

        return c;
    }

    CK_TILE_DEVICE auto MakeCOutputLayout(const CWarpTensor& c) const
    {

        return WarpGemmAttribute{}.MakeCOutputLayout(c);
    }
};

template <typename WarpGemmAttribute_>
struct WarpInt4GemmImpl
{
    using WarpGemmAttribute = remove_cvref_t<WarpGemmAttribute_>;

    static constexpr index_t kM = WarpGemmAttribute::kM;
    static constexpr index_t kN = WarpGemmAttribute::kN;
    static constexpr index_t kK = WarpGemmAttribute::kK;
    static constexpr index_t kKPerThread = WarpGemmAttribute::kKPerThread;

    using ADataType = typename WarpGemmAttribute::ADataType;
    using BDataType = typename WarpGemmAttribute::BDataType;
    using CDataType = typename WarpGemmAttribute::CDataType;

    using AComputeDataType = typename WarpGemmAttribute::AComputeDataType;
    using BComputeDataType = typename WarpGemmAttribute::BComputeDataType;
    using CComputeDataType = typename WarpGemmAttribute::CComputeDataType;

    using AWarpDstrEncoding = typename WarpGemmAttribute::AWarpDstrEncoding;
    using BWarpDstrEncoding = typename WarpGemmAttribute::BWarpDstrEncoding;
    using CWarpDstrEncoding = typename WarpGemmAttribute::CWarpDstrEncoding;
    using CWarpOutputDstrEncoding = typename WarpGemmAttribute::CWarpOutputDstrEncoding;

    using AWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(AWarpDstrEncoding{}))>;
    using BWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(BWarpDstrEncoding{}))>;
    using CWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(CWarpDstrEncoding{}))>;
    using CWarpOutputDstr =
        remove_cvref_t<decltype(make_static_tile_distribution(CWarpOutputDstrEncoding{}))>;

    using AWarpTensor = static_distributed_tensor<ADataType, AWarpDstr>;
    using BWarpTensor = static_distributed_tensor<BDataType, BWarpDstr>;
    using CWarpTensor = static_distributed_tensor<CDataType, CWarpDstr>;
    using CWarpOutputTensor = static_distributed_tensor<CDataType, CWarpOutputDstr>;

    CK_TILE_DEVICE void operator()(CWarpTensor& c, const AWarpTensor& a, const BWarpTensor& b) const
    {
        using AVec = ext_vector_t<int8_t, AWarpTensor::get_thread_buffer_size()>;
        using BVec = ext_vector_t<int8_t, BWarpTensor::get_thread_buffer_size()>;
        using CVec = ext_vector_t<CDataType, CWarpTensor::get_thread_buffer_size()>;

        constexpr auto I0 = number<0>{};

        const auto a_vec = a.get_thread_buffer().template get_as<AVec>()[I0];
        const auto b_vec = b.get_thread_buffer().template get_as<BVec>()[I0];
        auto c_vec       = c.get_thread_buffer().template get_as<CVec>()[I0];

        WarpGemmAttribute{}(c_vec, a_vec, b_vec);

        c.get_thread_buffer().template set_as<CVec>(I0, c_vec);
    }

    CK_TILE_DEVICE auto MakeCOutputLayout(const CWarpTensor& c) const
    {
        return WarpGemmAttribute{}.MakeCOutputLayout(c);
    }
};

template <typename WarpGemmAttribute_>
struct WarpInt8ScaleChannelGemmImpl
{
    using WarpGemmAttribute = remove_cvref_t<WarpGemmAttribute_>;

    static constexpr index_t kM = WarpGemmAttribute::kM;
    static constexpr index_t kN = WarpGemmAttribute::kN;
    static constexpr index_t kK = WarpGemmAttribute::kK;
    static constexpr index_t kKPerThread = WarpGemmAttribute::kKPerThread;

    static constexpr index_t kKIter = WarpGemmAttribute::kKIteration;
    static constexpr index_t kABKPerLane = WarpGemmAttribute::kABKPerLane;
    static constexpr index_t kCN0PerLane = WarpGemmAttribute::kCN0PerLane;

    using ADataType = typename WarpGemmAttribute::ADataType;
    using BDataType = typename WarpGemmAttribute::BDataType;
    using CDataType = typename WarpGemmAttribute::CDataType;

    using AScaleType = typename WarpGemmAttribute::AScaleType;
    using BScaleType = typename WarpGemmAttribute::BScaleType;

    using AScaleVec = typename WarpGemmAttribute::AScaleVec;
    using BScaleVec = typename WarpGemmAttribute::BScaleVec;
    /* 
    using AComputeDataType = typename WarpGemmAttribute::AComputeDataType;
    using BComputeDataType = typename WarpGemmAttribute::BComputeDataType;
    using CComputeDataType = typename WarpGemmAttribute::CComputeDataType;
    */
    using AWarpDstrEncoding = typename WarpGemmAttribute::AWarpDstrEncoding;
    using BWarpDstrEncoding = typename WarpGemmAttribute::BWarpDstrEncoding;
    using CWarpDstrEncoding = typename WarpGemmAttribute::CWarpDstrEncoding;

    using AScaleWarpDstrEncoding = typename WarpGemmAttribute::AScaleWarpDstrEncoding;
    using BScaleWarpDstrEncoding = typename WarpGemmAttribute::BScaleWarpDstrEncoding;

    using AWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(AWarpDstrEncoding{}))>;
    using BWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(BWarpDstrEncoding{}))>;
    using CWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(CWarpDstrEncoding{}))>;

    using AScaleWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(AScaleWarpDstrEncoding{}))>;
    using BScaleWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(BScaleWarpDstrEncoding{}))>;

    using AWarpTensor = static_distributed_tensor<ADataType, AWarpDstr>;
    using BWarpTensor = static_distributed_tensor<BDataType, BWarpDstr>;
    using CWarpTensor = static_distributed_tensor<CDataType, CWarpDstr>;

    using AScaleWarpTensor = static_distributed_tensor<AScaleType, AScaleWarpDstr>;
    using BScaleWarpTensor = static_distributed_tensor<BScaleType, BScaleWarpDstr>;


    CK_TILE_DEVICE void operator()(CWarpTensor& c, const AWarpTensor& a, const BWarpTensor& b, const AScaleWarpTensor& a_scale, const BScaleWarpTensor& b_scale) const
    {
        WarpGemmAttribute{}(
            c.get_thread_buffer().template get_as<typename WarpGemmAttribute::Impl::CVecType>(),
            a.get_thread_buffer().template get_as<typename WarpGemmAttribute::Impl::AVecType>(),
            b.get_thread_buffer().template get_as<typename WarpGemmAttribute::Impl::BVecType>(),
            a_scale.get_thread_buffer().template get_as<typename WarpGemmAttribute::Impl::AScaleType>(),
            b_scale.get_thread_buffer().template get_as<typename WarpGemmAttribute::Impl::BScaleType>());
    }

    // CK_TILE_DEVICE void operator()(CWarpTensor& c, const AWarpTensor& a, const BWarpTensor& b, const AScaleVec& a_scale_vec, const BScaleVec& b_scale_vec) const
    // {
    //     using AVec = ext_vector_t<ADataType, AWarpTensor::get_thread_buffer_size()>;
    //     using BVec = ext_vector_t<BDataType, BWarpTensor::get_thread_buffer_size()>;
    //     using CVec = ext_vector_t<CDataType, CWarpTensor::get_thread_buffer_size()>;

    //     constexpr auto I0 = number<0>{};

    //     const auto a_vec = a.get_thread_buffer().template get_as<AVec>()[I0];
    //     const auto b_vec = b.get_thread_buffer().template get_as<BVec>()[I0];
    //     auto c_vec       = c.get_thread_buffer().template get_as<CVec>()[I0];

    //     // c_vec += a_vec * b_vec
    //     WarpGemmAttribute{}(c_vec, a_vec, b_vec, a_scale_vec, b_scale_vec);

    //     c.get_thread_buffer().template set_as<CVec>(I0, c_vec);
    // }

    CK_TILE_DEVICE auto operator()(const AWarpTensor& a, const BWarpTensor& b) const
    {
        CWarpTensor c;

        using AVec = ext_vector_t<ADataType, AWarpTensor::get_thread_buffer_size()>;
        using BVec = ext_vector_t<BDataType, BWarpTensor::get_thread_buffer_size()>;
        using CVec = ext_vector_t<CDataType, CWarpTensor::get_thread_buffer_size()>;

        constexpr auto I0 = number<0>{};

        const auto a_vec = a.get_thread_buffer().template get_as<AVec>()[I0];
        const auto b_vec = b.get_thread_buffer().template get_as<BVec>()[I0];

        // c_vec = a_vec * b_vec
        auto c_vec = WarpGemmAttribute{}(a_vec, b_vec);

        c.get_thread_buffer().template set_as<CVec>(I0, c_vec);

        return c;
    }

};

template <typename WarpGemmAttribute_>
struct WarpFp8Bf8GemmImpl
{
    using WarpGemmAttribute = remove_cvref_t<WarpGemmAttribute_>;

    static constexpr index_t kM = WarpGemmAttribute::kM;
    static constexpr index_t kN = WarpGemmAttribute::kN;
    static constexpr index_t kK = WarpGemmAttribute::kK;
    static constexpr index_t kKPerThread = WarpGemmAttribute::kKPerThread;

    using ADataType = typename WarpGemmAttribute::ADataType;
    using BDataType = typename WarpGemmAttribute::BDataType;
    using CDataType = typename WarpGemmAttribute::CDataType;
    
    using AComputeDataType = typename WarpGemmAttribute::AComputeDataType;
    using BComputeDataType = typename WarpGemmAttribute::BComputeDataType;
    using CComputeDataType = typename WarpGemmAttribute::CComputeDataType;

    using AWarpDstrEncoding = typename WarpGemmAttribute::AWarpDstrEncoding;
    using BWarpDstrEncoding = typename WarpGemmAttribute::BWarpDstrEncoding;
    using CWarpDstrEncoding = typename WarpGemmAttribute::CWarpDstrEncoding;
    using CWarpOutputDstrEncoding = typename WarpGemmAttribute::CWarpOutputDstrEncoding;

    using AWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(AWarpDstrEncoding{}))>;
    using BWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(BWarpDstrEncoding{}))>;
    using CWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(CWarpDstrEncoding{}))>;
    using CWarpOutputDstr = remove_cvref_t<decltype(make_static_tile_distribution(CWarpOutputDstrEncoding{}))>;


    using AWarpTensor = static_distributed_tensor<ADataType, AWarpDstr>;
    using BWarpTensor = static_distributed_tensor<BDataType, BWarpDstr>;
    using CWarpTensor = static_distributed_tensor<CDataType, CWarpDstr>;
    using CWarpOutputTensor = static_distributed_tensor<CDataType, CWarpOutputDstr>;

    CK_TILE_DEVICE void operator()(CWarpTensor& c, const AWarpTensor& a, const BWarpTensor& b) const
    {
        using AVec = ext_vector_t<ADataType, AWarpTensor::get_thread_buffer_size()>;
        using BVec = ext_vector_t<BDataType, BWarpTensor::get_thread_buffer_size()>;
        using CVec = ext_vector_t<CDataType, CWarpTensor::get_thread_buffer_size()>;

        constexpr auto I0 = number<0>{};

        const auto a_vec = a.get_thread_buffer().template get_as<AVec>()[I0];
        const auto b_vec = b.get_thread_buffer().template get_as<BVec>()[I0];
        auto c_vec       = c.get_thread_buffer().template get_as<CVec>()[I0];

        // c_vec += a_vec * b_vec
        WarpGemmAttribute{}(c_vec, a_vec, b_vec);

        c.get_thread_buffer().template set_as<CVec>(I0, c_vec);
// #endif
    }

    CK_TILE_DEVICE auto operator()(const AWarpTensor& a, const BWarpTensor& b) const
    {
        CWarpTensor c;

        using AVec = ext_vector_t<ADataType, AWarpTensor::get_thread_buffer_size()>;
        using BVec = ext_vector_t<BDataType, BWarpTensor::get_thread_buffer_size()>;
        using CVec = ext_vector_t<CDataType, CWarpTensor::get_thread_buffer_size()>;

        constexpr auto I0 = number<0>{};

        const auto a_vec = a.get_thread_buffer().template get_as<AVec>()[I0];
        const auto b_vec = b.get_thread_buffer().template get_as<BVec>()[I0];

        // c_vec = a_vec * b_vec
        auto c_vec = WarpGemmAttribute{}(a_vec, b_vec);

        c.get_thread_buffer().template set_as<CVec>(I0, c_vec);

        return c;
    }

};


template <typename WarpGemmAttribute_>
struct WarpFp8Bf8ScaleChannelGemmImpl
{
    using WarpGemmAttribute = remove_cvref_t<WarpGemmAttribute_>;

    static constexpr index_t kM = WarpGemmAttribute::kM;
    static constexpr index_t kN = WarpGemmAttribute::kN;
    static constexpr index_t kK = WarpGemmAttribute::kK;
    static constexpr index_t kKPerThread = WarpGemmAttribute::kKPerThread;

    using ADataType = typename WarpGemmAttribute::ADataType;
    using BDataType = typename WarpGemmAttribute::BDataType;
    using CDataType = typename WarpGemmAttribute::CDataType;
    
    using AComputeDataType = typename WarpGemmAttribute::AComputeDataType;
    using BComputeDataType = typename WarpGemmAttribute::BComputeDataType;
    using CComputeDataType = typename WarpGemmAttribute::CComputeDataType;

    using AWarpDstrEncoding = typename WarpGemmAttribute::AWarpDstrEncoding;
    using BWarpDstrEncoding = typename WarpGemmAttribute::BWarpDstrEncoding;
    using CWarpDstrEncoding = typename WarpGemmAttribute::CWarpDstrEncoding;
    using CWarpOutputDstrEncoding = typename WarpGemmAttribute::CWarpOutputDstrEncoding;

    using AWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(AWarpDstrEncoding{}))>;
    using BWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(BWarpDstrEncoding{}))>;
    using CWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(CWarpDstrEncoding{}))>;
    using CWarpOutputDstr = remove_cvref_t<decltype(make_static_tile_distribution(CWarpOutputDstrEncoding{}))>;


    using AWarpTensor = static_distributed_tensor<ADataType, AWarpDstr>;
    using BWarpTensor = static_distributed_tensor<BDataType, BWarpDstr>;
    using CWarpTensor = static_distributed_tensor<CDataType, CWarpDstr>;
    using CWarpOutputTensor = static_distributed_tensor<CDataType, CWarpOutputDstr>;

    using AScaleType = typename WarpGemmAttribute::AScaleType;
    using BScaleType = typename WarpGemmAttribute::BScaleType;

    using AScaleWarpDstrEncoding = typename WarpGemmAttribute::AScaleWarpDstrEncoding;
    using BScaleWarpDstrEncoding = typename WarpGemmAttribute::BScaleWarpDstrEncoding;

    using AScaleWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(AScaleWarpDstrEncoding{}))>;
    using BScaleWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(BScaleWarpDstrEncoding{}))>;

    using AScaleWarpTensor = static_distributed_tensor<AScaleType, AScaleWarpDstr>;
    using BScaleWarpTensor = static_distributed_tensor<BScaleType, BScaleWarpDstr>;


    CK_TILE_DEVICE void operator()(CWarpTensor& c, const AWarpTensor& a, const BWarpTensor& b, const AScaleWarpTensor& a_scale, const BScaleWarpTensor& b_scale) const
    {
        WarpGemmAttribute{}(
            c.get_thread_buffer().template get_as<typename WarpGemmAttribute::Impl::CVecType>(),
            a.get_thread_buffer().template get_as<typename WarpGemmAttribute::Impl::AVecType>(),
            b.get_thread_buffer().template get_as<typename WarpGemmAttribute::Impl::BVecType>(),
            a_scale.get_thread_buffer().template get_as<typename WarpGemmAttribute::Impl::AScaleType>(),
            b_scale.get_thread_buffer().template get_as<typename WarpGemmAttribute::Impl::BScaleType>());
    }

};

} // namespace ck_tile
