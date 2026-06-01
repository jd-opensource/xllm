// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Problem Description for BlockGemm
template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          index_t kBlockSize_,
          typename BlockGemmShape_>
struct BlockGemmProblem
{
    using ADataType      = remove_cvref_t<ADataType_>;
    using BDataType      = remove_cvref_t<BDataType_>;
    using CDataType      = remove_cvref_t<CDataType_>;
    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    static constexpr index_t kBlockSize = kBlockSize_;
};

} // namespace ck_tile
