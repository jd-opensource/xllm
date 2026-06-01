// SPDX-License-Identifier: MIT
// Copyright (c) 2025, , Inc. All rights reserved.

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/stream_utils.hpp"
#include "ck_tile/core/utility/env.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/ops/common/utils.hpp" 

namespace ck_tile {

/// @brief The Universal GEMM kernel host arguments.
///
/// @par Overview
///      This structure is passed to @ref UniversalGemmKernel "UniversalGemmKernel" when creating
///      kernel arguments object. It contain all necessary information required to build proper
///      kernel argument and launch kernel on GPU. This structure defines the GEMM problem
///      configuration by stating all required information like M,N,K sizes and respective strides.
///      NumATensor describes the number of A tensors. The minimum number of tensors is 1(required).
///      NumBTensor describes the number of B tensors. The minimum number of tensors is 1(required).
///      NumDTensor describes the number of D tensors. The minimum number of tensors is 0(not
///      required).
template <index_t NumATensor = 1, index_t NumBTensor = 1, index_t NumDTensor = 0>
struct UniversalGemmHostArgs
{
    CK_TILE_HOST UniversalGemmHostArgs(const std::array<const void*, NumATensor>& as_ptr_,
                                       const std::array<const void*, NumBTensor>& bs_ptr_,
                                       const std::array<const void*, NumDTensor>& ds_ptr_,
                                       void* e_ptr_,
                                       index_t k_batch_,
                                       index_t M_,
                                       index_t N_,
                                       index_t K_,
                                       const std::array<index_t, NumATensor>& stride_As_,
                                       const std::array<index_t, NumBTensor>& stride_Bs_,
                                       const std::array<index_t, NumDTensor>& stride_Ds_,
                                       index_t stride_E_)
        : as_ptr(as_ptr_),
          bs_ptr(bs_ptr_),
          ds_ptr(ds_ptr_),
          e_ptr(e_ptr_),
          M(M_),
          N(N_),
          K(K_),
          stride_As(stride_As_),
          stride_Bs(stride_Bs_),
          stride_Ds(stride_Ds_),
          stride_E(stride_E_),
          k_batch(k_batch_)
    {
    }

    const std::array<const void*, NumATensor> as_ptr;
    const std::array<const void*, NumBTensor> bs_ptr;
    const std::array<const void*, NumDTensor> ds_ptr;
    union
    {
        void* e_ptr;
        void* c_ptr;
    };
    index_t M;
    index_t N;
    index_t K;
    const std::array<index_t, NumATensor> stride_As;
    const std::array<index_t, NumBTensor> stride_Bs;
    const std::array<index_t, NumDTensor> stride_Ds;
    union
    {
        index_t stride_E;
        index_t stride_C;
    };

    index_t k_batch;
};

/// @brief The GEMM kernel device arguments.
template <index_t NumATensor = 1, index_t NumBTensor = 1, index_t NumDTensor = 0>
struct UniversalGemmKernelArgs
{
    /// @brief The As input tensor's pointer to device memory.
    const std::array<const void*, NumATensor> as_ptr;
    /// @brief The Bs input tensor's pointer to device memory.
    const std::array<const void*, NumBTensor> bs_ptr;
    /// @brief The Ds input tensor's pointer to device memory.
    const std::array<const void*, NumDTensor> ds_ptr;
    /// @brief The E output tensor's pointer to device memory.
    void* e_ptr;
    /// @brief GEMM's M dimension size.
    index_t M;
    /// @brief GEMM's N dimension size.
    index_t N;
    /// @brief GEMM's K dimension size.
    index_t K;
    /// @brief The distance between consecutive elements of non-contiguous dimension
    ///        (in memory) of As tensor.
    std::array<index_t, NumATensor> stride_As;
    /// @brief The distance between consecutive elements of non-contiguous dimension
    ///        (in memory) of Bs tensor.
    std::array<index_t, NumBTensor> stride_Bs;
    /// @brief The distance between consecutive elements of non-contiguous dimension
    ///        (in memory) of Ds tensor.
    std::array<index_t, NumDTensor> stride_Ds;
    /// @brief The distance between consecutive elements of non-contiguous dimension
    ///        (in memory) of E tensor.
    index_t stride_E;
    index_t k_batch;
};

/// @brief The Universal GEMM kernel template.
///
/// @paragraph Overview Overview
///            This class provides the generic matrix multiplication kernel template. By semantic
///            division of GEMM algorithm into following parts we achieve flexible, versatile
///            and robust kernel implementation.
///
///            @li @b Prolog - The start of GEMM kernel implementation in @ref operator()
///                function call operator" which determines the work scope of each workgroup.
///            @li @b GemmPipeline - The core part @a "heart" of matrix multiplication algorithm.
///                This is the place where each workgroup is loading data from global memory and
///                carrying out dot products.
///            @li @b Epilogue - The @a "final" part of matrix multiplication implementation
///                 responsible for storing results to global memory. This is also the place where
///                 any additional operator fusion may take place.
///
///            Additionally both @ref GemmPipeline_ "GemmPipeline" and @ref EpiloguePipeline_
///            "EpiloguePipeline" are parameterized with so called @a Policy which determines all
///            internal details of those functional parts. You can think of it like both gemm and
///            epilogue pipelines provides the control-flow logic controlled by policies. Moreover
///            the policy is responsible for definition of all necessary data layouts and thread's
///            work distribution.
///
/// @tparam TilePartitioner_    The type of class providing mapping of workgroup index into the
///                             output data tile to be calculated. It determines the workgroup to
///                             data relationship (or in other words - which data would be
///                             processed and calculated by which workgroup).
/// @tparam GemmPipeline_       The type of class which provides the core part of matrix
///                             multiplication. This class should provide implementation of data
///                             loading from global memory and performing block-wise matrix
///                             multiplication. You can think of it as a work done by single
///                             workgroup point of view.
/// @tparam EpiloguePipeline_   The type of class providing the final part of matrix
///                             multiplication implementation. It is responsible for storing
///                             results calculated by @ref GemmPipeline_ "GemmPipeline" to
///                             the output E tensor in global memory.
template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct UniversalGemmKernel
{
    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;

    static constexpr bool ADataTypeIsTuple =
        is_detected<is_tuple, typename GemmPipeline::ADataType>::value;
    static constexpr bool BDataTypeIsTuple =
        is_detected<is_tuple, typename GemmPipeline::BDataType>::value;
    static constexpr bool DDataTypeIsTuple =
        is_detected<is_tuple, typename EpiloguePipeline::DsDataType>::value;
    static constexpr bool ALayoutIsTuple =
        is_detected<is_tuple, typename GemmPipeline::ALayout>::value;
    static constexpr bool BLayoutIsTuple =
        is_detected<is_tuple, typename GemmPipeline::BLayout>::value;
    static constexpr bool DLayoutIsTuple =
        is_detected<is_tuple, typename EpiloguePipeline::DsLayout>::value;
    

    using AsLayout = std::conditional_t<ALayoutIsTuple,
                                        remove_cvref_t<typename GemmPipeline::ALayout>,
                                        remove_cvref_t<tuple<typename GemmPipeline::ALayout>>>;
    using BsLayout = std::conditional_t<BLayoutIsTuple,
                                        remove_cvref_t<typename GemmPipeline::BLayout>,
                                        remove_cvref_t<tuple<typename GemmPipeline::BLayout>>>;

    using DsLayout = std::conditional_t<DLayoutIsTuple,
                                        remove_cvref_t<typename EpiloguePipeline::DsLayout>,
                                        remove_cvref_t<tuple<typename EpiloguePipeline::DsLayout>>>;

    using AsDataType = std::conditional_t<ADataTypeIsTuple,
                                          remove_cvref_t<typename GemmPipeline::ADataType>,
                                          remove_cvref_t<tuple<typename GemmPipeline::ADataType>>>;

    using BsDataType = std::conditional_t<BDataTypeIsTuple,
                                          remove_cvref_t<typename GemmPipeline::BDataType>,
                                          remove_cvref_t<tuple<typename GemmPipeline::BDataType>>>;

    using DsDataType =
        std::conditional_t<DDataTypeIsTuple,
                           remove_cvref_t<typename EpiloguePipeline::DsDataType>,
                           remove_cvref_t<tuple<typename EpiloguePipeline::DsDataType>>>;

    using ELayout   = remove_cvref_t<typename GemmPipeline::CLayout>;
    using EDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    static constexpr index_t kBlockSize = GemmPipeline::BlockSize;

    // Get the persistent kernel if the pipeline has it available
    struct has_persistent_kernel
    {
        template <typename T>
        using has_persistent_type = decltype(T::UsePersistentKernel);

        static constexpr bool value = []() {
            if constexpr(is_detected<has_persistent_type, GemmPipeline>{})
                return GemmPipeline::UsePersistentKernel;
            else
                return false;
        }();
    };
    static constexpr bool PersistentKernel = has_persistent_kernel::value;

    // Check if TilePartitioner has GetOutputOffset method with kargs and k_id
    struct has_tile_partitioner_output_offset_impl
    {
        template <typename T, typename KernelArgs>
        using has_get_output_offset_t =
            decltype(T::GetOutputOffset(std::declval<KernelArgs>(), std::declval<index_t>()));

        static constexpr bool value = []() {
            if constexpr(is_detected<has_get_output_offset_t, TilePartitioner>{})
                return true;
            else
                return false;
        }();
    };
    static constexpr bool has_tile_partitioner_output_offset =
        has_tile_partitioner_output_offset_impl::value;

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>{};

    static constexpr index_t NumATensor = AsDataType::size();
    static constexpr index_t NumBTensor = BsDataType::size();
    static constexpr index_t NumDTensor = DsDataType::size();

    using ADataType = remove_cvref_t<std::tuple_element_t<I0, AsDataType>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<I0, BsDataType>>;

    static_assert(AsLayout::size() == AsDataType::size(),
                  "The size of AsLayout and AsDataType should be the same");

    static_assert(BsLayout::size() == BsDataType::size(),
                  "The size of BsLayout and BsDataType should be the same");

    static_assert(DsLayout::size() == DsDataType::size(),
                  "The size of DsLayout and DsDataType should be the same");

    using KernelArgs =
        UniversalGemmKernelArgs<AsLayout::size(), BsLayout::size(), DsLayout::size()>;  // <1, 1, 1>

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "gemm", gemm_prec_str<ADataType, BDataType>(), GemmPipeline::GetName());
        // clang-format on
    }

    CK_TILE_HOST static constexpr auto GridSize(index_t M, index_t N, index_t KBatch)
    {
        return dim3(TilePartitioner::GridSize(M, N), 1, KBatch);
    }

    /**
     * @brief Get the maximum occupancy grid size for the persistent kernel on the current device.
     * @return The maximum occupancy grid size.
     * @note This function queries the maximum occupancy of the kernel using
     *       `hipOccupancyMaxActiveBlocksPerMultiprocessor`.
     */
    CK_TILE_HOST static auto MaxOccupancyGridSize(const stream_config& s) -> dim3
    {
        using Kernel      = UniversalGemmKernel<TilePartitioner, GemmPipeline, EpiloguePipeline>;
        const auto kernel = kentry<1, Kernel, KernelArgs>;
        int occupancy;
        hip_check_error(
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, BlockSize().x, 0));

        const int grid_size = get_available_compute_units(s) * occupancy;
        return dim3(grid_size, 1, 1);
    }

    CK_TILE_HOST static auto BlockSize()
    {
        if(ck_tile::is_wave32())
        {
            return dim3(kBlockSize / 2);
        }
        else
        {
            return dim3(kBlockSize);
        }
    }

    CK_TILE_HOST static constexpr KernelArgs
    MakeKernelArgs(const UniversalGemmHostArgs<NumATensor, NumBTensor, NumDTensor>& hostArgs)
    {
        return KernelArgs{hostArgs.as_ptr,
                          hostArgs.bs_ptr,
                          hostArgs.ds_ptr,
                          hostArgs.e_ptr,
                          hostArgs.M,
                          hostArgs.N,
                          hostArgs.K,
                          hostArgs.stride_As,
                          hostArgs.stride_Bs,
                          hostArgs.stride_Ds,
                          hostArgs.stride_E,
                          hostArgs.k_batch};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    struct SplitKBatchOffset
    {
        __device__ SplitKBatchOffset(const KernelArgs& kargs, const std::size_t k_id = blockIdx.z)
        {
            constexpr auto K1   = TilePartitioner::BlockGemmShape::WarpTile::at(number<2>{});
            const index_t K_t   = __builtin_amdgcn_readfirstlane(kargs.k_batch * K1);
            const index_t KRead = __builtin_amdgcn_readfirstlane((kargs.K + K_t - 1) / K_t * K1);

            // Convert logical K to physical (packed) units for pk_int4_t etc.
            // Tensor views, window steps, and num_loop all operate in packed element units.
            constexpr index_t PackedSize = ck_tile::numeric_traits<ADataType>::PackedSize;
            const index_t KReadPacked =
                __builtin_amdgcn_readfirstlane(KRead / PackedSize);

            static_for<0, NumATensor, 1>{}([&](auto index) {
                using AiLayout = remove_cvref_t<std::tuple_element_t<index.value, AsLayout>>;
                if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, AiLayout>)
                {
                    as_k_split_offset[index] = __builtin_amdgcn_readfirstlane(k_id * KReadPacked);
                }
                else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, AiLayout>)
                {
                    as_k_split_offset[index] =
                        __builtin_amdgcn_readfirstlane(k_id * KReadPacked * kargs.stride_As[index]);
                }
            });

            static_for<0, NumBTensor, 1>{}([&](auto index) {
                using BiLayout = remove_cvref_t<std::tuple_element_t<index.value, BsLayout>>;
                if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, BiLayout>)
                {
                    bs_k_split_offset[index] =
                        __builtin_amdgcn_readfirstlane(k_id * KReadPacked * kargs.stride_Bs[index]);
                }
                else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, BiLayout>)
                {
                    bs_k_split_offset[index] = __builtin_amdgcn_readfirstlane(k_id * KReadPacked);
                }
            });

            if(k_id < static_cast<uint32_t>(kargs.k_batch - 1))
            {
                splitted_k = __builtin_amdgcn_readfirstlane(KReadPacked);
            }
            else
            {
                splitted_k = __builtin_amdgcn_readfirstlane(
                    (kargs.K / PackedSize) - KReadPacked * (kargs.k_batch - 1));
            }
        }

        std::array<index_t, NumATensor> as_k_split_offset;
        std::array<index_t, NumBTensor> bs_k_split_offset;
        index_t splitted_k;
    };

    CK_TILE_HOST static bool IsSupportedArgument(const KernelArgs& kargs)
    {
        if constexpr(EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
                     is_any_of<EDataType, fp16_t, bf16_t>::value)
        {
            if(kargs.k_batch != 1)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("Conditions not met for Kbatch >1 !");
                }
                return false;
            }
        }

        const auto vectorSizeA = is_wave32() ? GemmPipeline::template GetVectorSizeA<true>()
                                             : GemmPipeline::template GetVectorSizeA<false>();
        bool AsTesnorIsValid   = {true};
        static_for<0, NumATensor, 1>{}([&](auto index) {
            using AiLayout = remove_cvref_t<std::tuple_element_t<index.value, AsLayout>>;
            if constexpr(std::is_same_v<AiLayout, tensor_layout::gemm::RowMajor>)
            {
                if(kargs.K % (TilePartitioner::KPerBlock * kargs.k_batch) != 0 &&
                   GemmPipeline::kPadK == false)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR(
                            "Can't support K that is not a multiple of k_batch * KPerBlock "
                            "without padding!");
                    }
                    AsTesnorIsValid = false;
                }
                if(kargs.K % vectorSizeA != 0)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("K is not a multiple of vector load size for A tensor!");
                    }
                    AsTesnorIsValid = false;
                }
            }
            else
            {
                if(kargs.M % TilePartitioner::MPerBlock != 0 && GemmPipeline::kPadM == false)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR(
                            "Can't support M that is not a multiple of MPerBlock without padding!");
                    }
                    AsTesnorIsValid = false;
                }
                if(kargs.M % vectorSizeA != 0)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("M is not a multiple of vector load size for A tensor!");
                    }
                    AsTesnorIsValid = false;
                }
            }
        });

        bool BsTesnorIsValid   = {true};
        const auto vectorSizeB = is_wave32() ? GemmPipeline::template GetVectorSizeB<true>()
                                             : GemmPipeline::template GetVectorSizeB<false>();
        static_for<0, NumBTensor, 1>{}([&](auto index) {
            using BiLayout = remove_cvref_t<std::tuple_element_t<index.value, BsLayout>>;
            if constexpr(std::is_same_v<BiLayout, tensor_layout::gemm::RowMajor>)
            {
                if(kargs.N % TilePartitioner::NPerBlock != 0 && GemmPipeline::kPadN == false)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR(
                            "Can't support N that is not a multiple of NPerBlock without padding!");
                    }
                    BsTesnorIsValid = false;
                }
                if(kargs.N % vectorSizeB != 0)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("N is not a multiple of vector load size for B tensor!");
                    }
                    BsTesnorIsValid = false;
                }
            }
            else
            {
                if(kargs.K % (TilePartitioner::KPerBlock * kargs.k_batch) != 0 &&
                   GemmPipeline::kPadK == false)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR(
                            "Can't support K that is not a multiple of k_batch * KPerBlock "
                            "without padding!");
                    }
                    BsTesnorIsValid = false;
                }
                if(kargs.K % vectorSizeB != 0)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("K is not a multiple of vector load size for B tensor!");
                    }
                    BsTesnorIsValid = false;
                }
            }
        });

        bool DTesnorIsValid = {true};
        static_for<0, NumDTensor, 1>{}([&](auto index) {
            using DiLayout = remove_cvref_t<std::tuple_element_t<index.value, DsLayout>>;
            if(std::is_same_v<DiLayout, ELayout> == false)
            {
                DTesnorIsValid = false;
            }
            if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
            {
                if(kargs.N % TilePartitioner::NPerBlock != 0 && GemmPipeline::kPadN == false)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("Can't support N for tensor D that is not a multiple of "
                                      "NPerBlock without padding!");
                    }
                    DTesnorIsValid = false;
                }
                if(kargs.N % EpiloguePipeline::GetVectorSizeD(index) != 0)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("N is not a multiple of vector load size for D tensor!");
                    }
                    DTesnorIsValid = false;
                }
            }
            else
            {
                if(kargs.M % TilePartitioner::MPerBlock != 0 && GemmPipeline::kPadM == false)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("Can't support M for tensor D that is not a multiple of "
                                      "MPerBlock without padding!");
                    }
                    DTesnorIsValid = false;
                }
                if(kargs.M % EpiloguePipeline::GetVectorSizeD(index) != 0)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("M is not a multiple of vector load size for D tensor!");
                    }
                    DTesnorIsValid = false;
                }
            }
        });

        if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.N % TilePartitioner::NPerBlock != 0 && GemmPipeline::kPadN == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support N that is not a multiple of NPerBlock without padding!");
                }
                return false;
            }
            if(kargs.N % EpiloguePipeline::GetVectorSizeC() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("N is not a multiple of vector load size for C tensor!");
                }
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::MPerBlock != 0 && GemmPipeline::kPadM == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support M that is not a multiple of MPerBlock without padding!");
                }
                return false;
            }
            if(kargs.M % EpiloguePipeline::GetVectorSizeC() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("M is not a multiple of vector load size for C tensor!");
                }
                return false;
            }
        }
        return AsTesnorIsValid && BsTesnorIsValid && DTesnorIsValid;
    }

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set>
    CK_TILE_DEVICE static auto
    MakeGemmTensorViews(const std::array<const ADataType*, NumATensor>& as_ptr,
                        const std::array<const BDataType*, NumBTensor>& bs_ptr,
                        const std::array<const void*, NumDTensor>& ds_ptr,
                        EDataType* e_ptr,
                        const KernelArgs& kargs,
                        const SplitKBatchOffset& splitk_batch_offset)
    {
        static_assert(!TilePartitioner::BlockGemmShape::PermuteA, "Not implemented!");

        const auto& as_tensor_view = generate_tuple(
            [&](auto i) {
                using AiLayout   = remove_cvref_t<std::tuple_element_t<i.value, AsLayout>>;
                using AiDataType = remove_cvref_t<std::tuple_element_t<i.value, AsDataType>>;
                if constexpr(std::is_same_v<AiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        static_cast<const AiDataType*>(as_ptr[i]),
                        make_tuple(kargs.M, splitk_batch_offset.splitted_k),
                        make_tuple(kargs.stride_As[i], 1),
                        number<GemmPipeline::GetVectorSizeA()>{},
                        number<1>{});
                }
                else
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        static_cast<const AiDataType*>(as_ptr[i]),
                        make_tuple(splitk_batch_offset.splitted_k, kargs.M),
                        make_tuple(kargs.stride_As[i], 1),
                        number<GemmPipeline::GetVectorSizeA()>{},
                        number<1>{});
                }
            },
            number<NumATensor>{});

        const auto& bs_tensor_view = generate_tuple(
            [&](auto i) {
                using BiLayout   = remove_cvref_t<std::tuple_element_t<i.value, BsLayout>>;
                using BiDataType = remove_cvref_t<std::tuple_element_t<i.value, BsDataType>>;
                if constexpr(std::is_same_v<BiLayout, tensor_layout::gemm::RowMajor>)
                {
                    if constexpr(TilePartitioner::BlockGemmShape::PermuteB)
                    {
                        constexpr index_t K1 = GemmPipeline::GetSmemPackB();
                        const index_t K0     = splitk_batch_offset.splitted_k / K1;
                        constexpr index_t VectorSizeB =
                            std::min(K1, GemmPipeline::GetVectorSizeB());
                        const auto b_k0_n_k1_desc =
                            make_naive_tensor_descriptor(make_tuple(K0, kargs.N, K1),
                                                         make_tuple(kargs.N * K1, K1, I1),
                                                         number<VectorSizeB>{},
                                                         number<1>{});
                        const auto b_n_k_desc = transform_tensor_descriptor(
                            b_k0_n_k1_desc,
                            make_tuple(make_merge_transform(make_tuple(K0, K1)),
                                       make_pass_through_transform(kargs.N)),
                            make_tuple(sequence<0, 2>{}, sequence<1>{}),
                            make_tuple(sequence<0>{}, sequence<1>{}));
                        return make_tensor_view<address_space_enum::global>(
                            static_cast<const BiDataType*>(bs_ptr[i]), b_n_k_desc);
                    }
                    else
                    {
                        return make_naive_tensor_view<address_space_enum::global>(
                            bs_ptr[i],
                            make_tuple(splitk_batch_offset.splitted_k, kargs.N),
                            make_tuple(kargs.stride_Bs[i], 1),
                            number<GemmPipeline::GetVectorSizeB()>{},
                            number<1>{});
                    }
                }
                else
                {
                    if constexpr(TilePartitioner::BlockGemmShape::PermuteB)
                    {
                        constexpr index_t K1 = GemmPipeline::GetSmemPackB();
                        const index_t K0     = splitk_batch_offset.splitted_k / K1;
                        constexpr index_t VectorSizeB =
                            std::min(K1, GemmPipeline::GetVectorSizeB());
                        const auto b_k0_n_k1_desc =
                            make_naive_tensor_descriptor(make_tuple(K0, kargs.N, K1),
                                                         make_tuple(kargs.N * K1, K1, I1),
                                                         number<VectorSizeB>{},
                                                         number<1>{});
                        const auto b_n_k_desc = transform_tensor_descriptor(
                            b_k0_n_k1_desc,
                            make_tuple(make_merge_transform(make_tuple(K0, K1)),
                                       make_pass_through_transform(kargs.N)),
                            make_tuple(sequence<0, 2>{}, sequence<1>{}),
                            make_tuple(sequence<1>{}, sequence<0>{}));
                        return make_tensor_view<address_space_enum::global>(
                            static_cast<const BiDataType*>(bs_ptr[i]), b_n_k_desc);
                    }
                    else
                    {
                        if constexpr(GemmPipeline::Preshuffle)
                        {
                            index_t kFlatK =
                                GemmPipeline::BlockGemmShape::flatKPerWarp *
                                (splitk_batch_offset.splitted_k /
                                 TilePartitioner::BlockGemmShape::WarpTile::at(number<2>{}));
                            index_t kFlatN = kargs.N * kargs.K / kFlatK;

                            return make_naive_tensor_view<address_space_enum::global>(
                                bs_ptr[i],
                                make_tuple(kFlatN, kFlatK),
                                make_tuple(kFlatK, 1),
                                number<GemmPipeline::GetVectorSizeB()>{},
                                number<1>{});
                        }
                        else
                        {
                            return make_naive_tensor_view<address_space_enum::global>(
                                bs_ptr[i],
                                make_tuple(kargs.N, splitk_batch_offset.splitted_k),
                                make_tuple(kargs.stride_Bs[i], 1),
                                number<GemmPipeline::GetVectorSizeB()>{},
                                number<1>{});
                        }
                    }
                }
            },
            number<NumBTensor>{});

        const auto& ds_tensor_view = generate_tuple(
            [&](auto i) {
                using DiLayout   = remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
                using DDataType_ = remove_cvref_t<std::tuple_element_t<i.value, DsDataType>>;
                if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        static_cast<const DDataType_*>(ds_ptr[i]),
                        make_tuple(kargs.M, kargs.N),
                        make_tuple(kargs.stride_Ds[i], 1),
                        number<EpiloguePipeline::GetVectorSizeD(i)>{},
                        number<1>{});
                }
                else
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        static_cast<const DDataType_*>(ds_ptr[i]),
                        make_tuple(kargs.N, kargs.M),
                        make_tuple(kargs.stride_Ds[i], 1),
                        number<EpiloguePipeline::GetVectorSizeD(i)>{},
                        number<1>{});
                }
            },
            number<NumDTensor>{});

        // TODO: enable vector write for C in ColMajor
        const auto& e_tensor_view = [&]() {
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    e_ptr,
                    make_tuple(kargs.M, kargs.N), // arguments not matching with flatmm.
                    make_tuple(kargs.stride_E, 1),
                    number<EpiloguePipeline::GetVectorSizeC()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    e_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(1, kargs.stride_E),
                    number<1>{},
                    number<1>{});
            }
        }();

        return make_tuple(as_tensor_view, bs_tensor_view, ds_tensor_view, e_tensor_view);
    }

    template <typename TensorView>
    CK_TILE_DEVICE static auto MakeGemmPadViews(const TensorView& views)
    {
        const auto& as_pad_view = generate_tuple(
            [&](auto i) {
                const auto& a_tensor_view = views.at(I0);
                using AiLayout            = remove_cvref_t<std::tuple_element_t<i.value, AsLayout>>;
                if constexpr(std::is_same_v<AiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return pad_tensor_view(a_tensor_view[i],
                                           make_tuple(number<TilePartitioner::MPerBlock>{},
                                                      number<TilePartitioner::KPerBlock>{}),
                                           sequence<false, GemmPipeline::kPadK>{});
                }
                else
                {
                    return pad_tensor_view(a_tensor_view[i],
                                           make_tuple(number<TilePartitioner::KPerBlock>{},
                                                      number<TilePartitioner::MPerBlock>{}),
                                           sequence<false, GemmPipeline::kPadM>{});
                }
            },
            number<NumATensor>{});

        const auto& b_flat_pad_view = views.at(I1);

        const auto& bs_pad_view = generate_tuple(
            [&](auto i) {
                const auto& b_tensor_view = views.at(I1);
                using BiLayout            = remove_cvref_t<std::tuple_element_t<i.value, BsLayout>>;
                if constexpr(std::is_same_v<BiLayout, tensor_layout::gemm::ColumnMajor>)
                {
                    return pad_tensor_view(b_tensor_view[i],
                                           make_tuple(number<TilePartitioner::NPerBlock>{},
                                                      number<TilePartitioner::KPerBlock>{}),
                                           sequence<false, GemmPipeline::kPadK>{});
                }
                else
                {
                    return pad_tensor_view(b_tensor_view[i],
                                           make_tuple(number<TilePartitioner::KPerBlock>{},
                                                      number<TilePartitioner::NPerBlock>{}),
                                           sequence<false, GemmPipeline::kPadN>{});
                }
            },
            number<NumBTensor>{});

        const auto& ds_pad_view = generate_tuple(
            [&](auto i) {
                const auto& d_tensor_view = views.at(I2);
                using DiLayout            = remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
                if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return pad_tensor_view(d_tensor_view[i],
                                           make_tuple(number<TilePartitioner::MPerBlock>{},
                                                      number<TilePartitioner::NPerBlock>{}),
                                           sequence<false, GemmPipeline::kPadN>{});
                }
                else
                {
                    return pad_tensor_view(d_tensor_view[i],
                                           make_tuple(number<TilePartitioner::NPerBlock>{},
                                                      number<TilePartitioner::MPerBlock>{}),
                                           sequence<false, GemmPipeline::kPadM>{});
                }
            },
            number<NumDTensor>{});

        // TODO vector write in for C in ColMajor
        const auto& e_pad_view = [&]() {
            const auto& e_tensor_view = views.at(I3);
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(e_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadN>{});
            }
            else
            {
                return pad_tensor_view(e_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<GemmPipeline::kPadM, false>{});
            }
        }();

        if constexpr(GemmPipeline::Preshuffle)
        {
            // For flatmm, we need to use the flat B tensor view
            return make_tuple(as_pad_view, b_flat_pad_view, ds_pad_view, e_pad_view);
        }
        else
        {
            return make_tuple(as_pad_view, bs_pad_view, ds_pad_view, e_pad_view);
        }
    }

    template <typename PadView>
    CK_TILE_DEVICE static auto
    MakeGemmTileWindows(const PadView& views, const index_t i_m, const index_t i_n)
    {
        const auto& as_pad_view = views.at(I0);
        const auto& bs_pad_view = views.at(I1);
        const auto& ds_pad_view = views.at(I2);
        const auto& e_pad_view  = views.at(I3);

        const auto& as_block_window = generate_tuple(
            [&](auto i) {
                using AiLayout = remove_cvref_t<std::tuple_element_t<i.value, AsLayout>>;
                if constexpr(std::is_same_v<AiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_tile_window(as_pad_view[i],
                                            make_tuple(number<TilePartitioner::MPerBlock>{},
                                                       number<TilePartitioner::KPerBlock>{}),
                                            {i_m, 0});
                }
                else
                {
                    return make_tile_window(as_pad_view[i],
                                            make_tuple(number<TilePartitioner::KPerBlock>{},
                                                       number<TilePartitioner::MPerBlock>{}),
                                            {0, i_m});
                }
            },
            number<NumATensor>{});

        const auto& bs_block_window = generate_tuple(
            [&](auto i) {
                using BiLayout = remove_cvref_t<std::tuple_element_t<i.value, BsLayout>>;
                if constexpr(GemmPipeline::Preshuffle)
                {
                    return make_tile_window(
                        bs_pad_view[i],
                        make_tuple(number<GemmPipeline::BlockGemmShape::flatNPerWarp>{},
                                   number<GemmPipeline::BlockGemmShape::flatKPerWarp>{}),
                        {static_cast<int>(i_n / GemmPipeline::BlockGemmShape::WarpTile::at(I1)),
                         0});
                }
                else
                {
                    if constexpr(std::is_same_v<BiLayout, tensor_layout::gemm::ColumnMajor>)
                    {
                        return make_tile_window(bs_pad_view[i],
                                                make_tuple(number<TilePartitioner::NPerBlock>{},
                                                           number<TilePartitioner::KPerBlock>{}),
                                                {i_n, 0});
                    }
                    else
                    {
                        return make_tile_window(bs_pad_view[i],
                                                make_tuple(number<TilePartitioner::KPerBlock>{},
                                                           number<TilePartitioner::NPerBlock>{}),
                                                {0, i_n});
                    }
                }
            },
            number<NumBTensor>{});

        const auto ds_block_window = generate_tuple(
            [&](auto i) {
                using DiLayout = remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
                if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_tile_window(ds_pad_view[i],
                                            make_tuple(number<TilePartitioner::MPerBlock>{},
                                                       number<TilePartitioner::NPerBlock>{}),
                                            {i_m, i_n});
                }
                else
                {
                    return make_tile_window(ds_pad_view[i],
                                            make_tuple(number<TilePartitioner::NPerBlock>{},
                                                       number<TilePartitioner::MPerBlock>{}),
                                            {i_n, i_m});
                }
            },
            number<NumDTensor>{});

        auto e_block_window = make_tile_window(
            e_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {i_m, i_n});

        return make_tuple(as_block_window, bs_block_window, ds_block_window, e_block_window);
    }

    /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @param as_ptr input As pointer
     * @param bs_ptr input Bs pointer
     * @param ds_ptr input Ds pointer
     * @param e_ptr output E pointer
     * @param smem_ptr_0 The start memory pointer of the shared memory block.
     * @param kargs GEMM kernel arguments
     * @param splitk_batch_offset splitk_batch_offset Utility structure used to calculate k batch.
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     *
     */
    template <bool UseDefaultScheduler = true>
    CK_TILE_DEVICE static void RunGemm(const std::array<const ADataType*, NumATensor>& as_ptr,
                                       const std::array<const BDataType*, NumBTensor>& bs_ptr,
                                       const std::array<const void*, NumDTensor>& ds_ptr,
                                       EDataType* e_ptr,
                                       void* smem_ptr_0,
                                       const KernelArgs& kargs,
                                       const SplitKBatchOffset& splitk_batch_offset,
                                       const index_t block_idx_m,
                                       const index_t block_idx_n)
    {
        // Create Gemm tensor views, pad views and tile windows
        const auto& gemm_tensor_views_tuple =
            MakeGemmTensorViews<EpiloguePipeline::MemoryOperation>(
                as_ptr, bs_ptr, ds_ptr, e_ptr, kargs, splitk_batch_offset);

        const auto& gemm_pad_views = MakeGemmPadViews(gemm_tensor_views_tuple);
        auto gemm_tile_windows     = MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n);

        const index_t num_loop = __builtin_amdgcn_readfirstlane(
            TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k));

        // Run GEMM cooperatively by whole workgroup.
        const auto& as_block_window = gemm_tile_windows.at(I0);
        const auto& bs_block_window = gemm_tile_windows.at(I1);
        const auto& ds_block_window = gemm_tile_windows.at(I2);

        const auto& c_block_tile =
            GemmPipeline{}(as_block_window[I0], bs_block_window[I0], num_loop, smem_ptr_0);

        if(UseDefaultScheduler || (get_warp_id() == 0))
        {
            // Run Epilogue Pipeline
            auto& c_block_window = gemm_tile_windows.at(I3);

            EpiloguePipeline{}(c_block_window, c_block_tile, ds_block_window, smem_ptr_0);
        }
    }

    /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @note RunGEMM2LDS in with two shared memory buffers using the ping pong buffer mechanism.
     *
     * @param as_ptr input As pointer
     * @param bs_ptr input Bs pointer
     * @param ds_ptr input Ds pointer
     * @param e_ptr output E pointer
     * @param smem_ptr_0 The starting pointer of 1st shared memory block.
     * @param smem_ptr_1 The starting pointer of 2nd shared memory block.
     * @param kargs GEMM kernel arguments
     * @param splitk_batch_offset Utility structure used to calculate k batch.
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     *
     */
    CK_TILE_DEVICE static void RunGemm2LDS(const std::array<const ADataType*, NumATensor>& as_ptr,
                                           const std::array<const BDataType*, NumBTensor>& bs_ptr,
                                           const std::array<const void*, NumDTensor>& ds_ptr,
                                           EDataType* e_ptr,
                                           void* __restrict__ smem_ptr_0,
                                           void* __restrict__ smem_ptr_1,
                                           const KernelArgs& kargs,
                                           const SplitKBatchOffset& splitk_batch_offset,
                                           const index_t block_idx_m,
                                           const index_t block_idx_n)
    {
        // Create Gemm tensor views, pad views and tile windows
        const auto& gemm_tensor_views_tuple =
            MakeGemmTensorViews<EpiloguePipeline::MemoryOperation>(
                as_ptr, bs_ptr, ds_ptr, e_ptr, kargs, splitk_batch_offset);

        const auto& gemm_pad_views = MakeGemmPadViews(gemm_tensor_views_tuple);
        auto gemm_tile_windows     = MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n);

        const index_t num_loop = __builtin_amdgcn_readfirstlane(
            TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k));

        // Run GEMM cooperatively by whole workgroup.
        const auto& as_block_window = gemm_tile_windows.at(I0);
        const auto& bs_block_window = gemm_tile_windows.at(I1);
        const auto& ds_block_window = gemm_tile_windows.at(I2);

        const auto& c_block_tile = GemmPipeline{}(
            as_block_window[I0], bs_block_window[I0], num_loop, smem_ptr_0, smem_ptr_1);

        // Run Epilogue Pipeline
        auto& c_block_window = gemm_tile_windows.at(I3);

        EpiloguePipeline{}(c_block_window, c_block_tile, ds_block_window, smem_ptr_0);
    }

    // Non-persistent kernel entry point
    template <bool U = !PersistentKernel, typename = std::enable_if_t<U>>
    CK_TILE_DEVICE void operator()(KernelArgs kargs) const
    {
        const auto blockId  = __builtin_amdgcn_readfirstlane(blockIdx.x);
        const auto [iM, iN] = TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(blockId);
        const index_t i_m   = __builtin_amdgcn_readfirstlane(iM * TilePartitioner::MPerBlock);
        const index_t i_n   = __builtin_amdgcn_readfirstlane(iN * TilePartitioner::NPerBlock);

        const SplitKBatchOffset splitk_batch_offset(kargs);

        // options
        std::array<const ADataType*, NumATensor> as_ptr;
        static_for<0, NumATensor, 1>{}([&](auto i) {
            as_ptr[i] = static_cast<const ADataType*>(kargs.as_ptr[i]) +
                        splitk_batch_offset.as_k_split_offset[i];
        });

        std::array<const BDataType*, NumBTensor> bs_ptr;
        static_for<0, NumBTensor, 1>{}([&](auto i) {
            bs_ptr[i] = static_cast<const BDataType*>(kargs.bs_ptr[i]) +
                        splitk_batch_offset.bs_k_split_offset[i];
        });

        // Calculate output offset from tile partitioner and apply to output pointer
        EDataType* e_ptr = static_cast<EDataType*>(kargs.e_ptr);
        if constexpr(has_tile_partitioner_output_offset)
        {
            const index_t output_offset = TilePartitioner::GetOutputOffset(kargs, blockIdx.z);
            e_ptr += output_offset;
        }

        // allocate LDS
        __shared__ char smem_ptr_0[GetSmemSize()];

        if constexpr(GemmPipeline::DoubleSmemBuffer == true)
        {
            __shared__ char smem_ptr_1[GetSmemSize()];
            if constexpr(!(EpiloguePipeline::MemoryOperation == memory_operation_enum::atomic_add &&
                           EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
                           is_any_of<EDataType, fp16_t, bf16_t>::value))
            {
                RunGemm2LDS(as_ptr,
                            bs_ptr,
                            kargs.ds_ptr,
                            e_ptr,
                            smem_ptr_0,
                            smem_ptr_1,
                            kargs,
                            splitk_batch_offset,
                            i_m,
                            i_n);
            }
        }
        else
        {
            if constexpr(!(EpiloguePipeline::MemoryOperation == memory_operation_enum::atomic_add &&
                           EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
                           is_any_of<EDataType, fp16_t, bf16_t>::value))
            {
                constexpr auto scheduler_type = (GemmPipeline::NumWaveGroups == 1);
                RunGemm<scheduler_type>(as_ptr,
                                        bs_ptr,
                                        kargs.ds_ptr,
                                        e_ptr,
                                        smem_ptr_0,
                                        kargs,
                                        splitk_batch_offset,
                                        i_m,
                                        i_n);
            }
        }
    }

    // Persistent kernel entry point
    template <bool U = PersistentKernel, typename = std::enable_if_t<U>, typename = void>
    CK_TILE_DEVICE void operator()(KernelArgs kargs) const
    {
        const auto grid_size = __builtin_amdgcn_readfirstlane(get_grid_size());
        const auto num_tiles =
            __builtin_amdgcn_readfirstlane(TilePartitioner::GridSize(kargs.M, kargs.N));
        const auto num_work = __builtin_amdgcn_readfirstlane(num_tiles * kargs.k_batch);
        auto block_id       = __builtin_amdgcn_readfirstlane(get_block_id());

        while(block_id < num_work)
        {
            // Get the tile index for this block
            const auto tile_idx = __builtin_amdgcn_readfirstlane(block_id % num_tiles);
            const auto [iM, iN] = TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(tile_idx);
            const index_t i_m   = __builtin_amdgcn_readfirstlane(iM * TilePartitioner::MPerBlock);
            const index_t i_n   = __builtin_amdgcn_readfirstlane(iN * TilePartitioner::NPerBlock);

            // Get the SplitK offset for this block
            const auto k_batch = __builtin_amdgcn_readfirstlane(block_id / num_tiles);
            const SplitKBatchOffset splitk_batch_offset(kargs, k_batch);

            std::array<const ADataType*, NumATensor> as_ptr;
            static_for<0, NumATensor, 1>{}([&](auto i) {
                as_ptr[i] = static_cast<const ADataType*>(kargs.as_ptr[i]) +
                            splitk_batch_offset.as_k_split_offset[i];
            });

            std::array<const BDataType*, NumBTensor> bs_ptr;
            static_for<0, NumBTensor, 1>{}([&](auto i) {
                bs_ptr[i] = static_cast<const BDataType*>(kargs.bs_ptr[i]) +
                            splitk_batch_offset.bs_k_split_offset[i];
            });

            // Calculate output offset from tile partitioner and apply to output pointer
            EDataType* e_ptr = static_cast<EDataType*>(kargs.e_ptr);
            if constexpr(has_tile_partitioner_output_offset)
            {
                const index_t output_offset = TilePartitioner::GetOutputOffset(kargs, k_batch);
                e_ptr += output_offset;
            }

            // allocate LDS
            __shared__ char smem_ptr_0[GetSmemSize()];
            // Run the GEMM
            if constexpr(GemmPipeline::DoubleSmemBuffer == true)
            {
                __shared__ char smem_ptr_1[GetSmemSize()];
                if constexpr(!(EpiloguePipeline::MemoryOperation ==
                                   memory_operation_enum::atomic_add &&
                               EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
                               is_any_of<EDataType, fp16_t, bf16_t>::value))
                {
                    RunGemm2LDS(as_ptr,
                                bs_ptr,
                                kargs.ds_ptr,
                                e_ptr,
                                smem_ptr_0,
                                smem_ptr_1,
                                kargs,
                                splitk_batch_offset,
                                i_m,
                                i_n);
                }
            }
            else
            {
                if constexpr(!(EpiloguePipeline::MemoryOperation ==
                                   memory_operation_enum::atomic_add &&
                               EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
                               is_any_of<EDataType, fp16_t, bf16_t>::value))
                {
                    RunGemm(as_ptr,
                            bs_ptr,
                            kargs.ds_ptr,
                            e_ptr,
                            smem_ptr_0,
                            kargs,
                            splitk_batch_offset,
                            i_m,
                            i_n);
                }
            }
            // Advance to the next work item
            block_id += grid_size;
            if(block_id >= num_work)
            {
                break;
            }
        }
    }
};
} // namespace ck_tile
