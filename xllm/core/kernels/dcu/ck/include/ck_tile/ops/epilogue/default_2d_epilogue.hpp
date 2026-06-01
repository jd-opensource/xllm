// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

namespace ck_tile {

// this epilogue just store out a M*N matrix, row major

template <typename AccDataType_,
          typename ODataType_,
          bool kPadM_,
          bool kPadN_,
          bool UseRawStore_                      = true,
          memory_operation_enum MemoryOperation_ = memory_operation_enum::set>
struct Default2DEpilogueProblem
{
    using AccDataType                                      = remove_cvref_t<AccDataType_>;
    using ODataType                                        = remove_cvref_t<ODataType_>;
    static constexpr bool kPadM                            = kPadM_;
    static constexpr bool kPadN                            = kPadN_;
    static constexpr bool UseRawStore                      = UseRawStore_;
    static constexpr memory_operation_enum MemoryOperation = MemoryOperation_;
};

template <typename ADataType_,
          typename BDataType_,
          typename AccDataType_,
          typename ODataType_,
          typename CLayout_,
          bool kPadM_,
          bool kPadN_,
          index_t kMPerXdl_,
          index_t kNPerXdl_,
          index_t kKPerXdl_,
          bool isCTransposed_,
          bool UseRawStore_                      = true,
          memory_operation_enum MemoryOperation_ = memory_operation_enum::set>
struct DefaultGemm2DEpilogueProblem : public Default2DEpilogueProblem<AccDataType_,
                                                                      ODataType_,
                                                                      kPadM_,
                                                                      kPadN_,
                                                                      UseRawStore_,
                                                                      MemoryOperation_>
{
    using ADataType                        = remove_cvref_t<ADataType_>;
    using BDataType                        = remove_cvref_t<BDataType_>;
    using CLayout                          = remove_cvref_t<CLayout_>;
    static constexpr index_t kMPerXdl      = kMPerXdl_;
    static constexpr index_t kNPerXdl      = kNPerXdl_;
    static constexpr index_t kKPerXdl      = kKPerXdl_;
    static constexpr index_t isCTransposed = isCTransposed_;
};

template <typename Problem_, typename Policy_ = void>
struct Default2DEpilogue
{
    using Problem                     = remove_cvref_t<Problem_>;
    using AccDataType                 = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType                   = remove_cvref_t<typename Problem::ODataType>;
    static constexpr bool kPadM       = Problem::kPadM;
    static constexpr bool kPadN       = Problem::kPadN;
    static constexpr bool UseRawStore = Problem::UseRawStore;
    static constexpr memory_operation_enum MemoryOperation = Problem::MemoryOperation;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    // TODO: this function assume store out vector size is the same as OAccTile last dimension size
    //       how do we fix this ?
    template <typename ODramWindowTmp, typename OAccTile>
    CK_TILE_DEVICE auto
    operator()(ODramWindowTmp& o_dram_window_tmp, const OAccTile& o_acc_tile, void* = nullptr)
    {

        // TODO: this is ugly
        if constexpr(UseRawStore && (kPadM || kPadN))
        {
            if constexpr(MemoryOperation == memory_operation_enum::set)
            {
                store_tile_raw(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
            }
            else
            {
                update_tile_raw(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
            }
            buffer_store_fence();
        }
        else
        {
            if constexpr(MemoryOperation == memory_operation_enum::set)
            {
                store_tile(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
            }
            else
            {
                update_tile(o_dram_window_tmp, cast_tile<ODataType>(o_acc_tile));
            }
        }
    }
};
} // namespace ck_tile
