// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename DataType, typename RandValOutputDataType>
CK_TILE_HOST void reference_batched_dropout(HostTensor<DataType>& in_out_b_m_n,
                                            const HostTensor<RandValOutputDataType>& randval_b_m_n,
                                            const uint8_t& p_undrop_in_uint8_t,
                                            const float scale)
{
    const int N = in_out_b_m_n.mDesc.get_lengths()[2];
    auto f      = [&](auto batch, auto m) {
        for(int n = 0; n < N; ++n)
        {
            float tmp = ck_tile::type_convert<float>(in_out_b_m_n(batch, m, n)) * scale;
            in_out_b_m_n(batch, m, n) = randval_b_m_n(batch, m, n) <= p_undrop_in_uint8_t
                                                 ? ck_tile::type_convert<DataType>(tmp)
                                                 : DataType(0);
        }
    };

    make_ParallelTensorFunctor(
        f, randval_b_m_n.mDesc.get_lengths()[0], randval_b_m_n.mDesc.get_lengths()[1])(
        std::thread::hardware_concurrency());
}
} // namespace ck_tile
