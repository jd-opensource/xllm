// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename CDataType, typename MaskingType>
CK_TILE_HOST void reference_batched_masking(HostTensor<CDataType>& c_b_m_n, const MaskingType& mask)
{
    const int M = c_b_m_n.mDesc.get_lengths()[1];
    const int N = c_b_m_n.mDesc.get_lengths()[2];

    auto f = [&](auto batch) {
        for(int n = 0; n < N; ++n)
        {
            for(int m = 0; m < M; ++m)
            {
                if(mask.IsOutOfBound(m, n))
                    c_b_m_n(batch, m, n) = -ck_tile::numeric<CDataType>::infinity();
            }
        }
    };

    make_ParallelTensorFunctor(f,
                               c_b_m_n.mDesc.get_lengths()[0])(std::thread::hardware_concurrency());
}
} // namespace ck_tile
