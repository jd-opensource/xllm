// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename ADataType,
          typename CompDataType,
          typename BDataType,
          typename CompElementOp = ck_tile::identity>
CK_TILE_HOST void reference_batched_softmax(
    const HostTensor<ADataType>& a_b_m_n,
    HostTensor<BDataType>& b_b_m_n,
    const CompElementOp& comp_element_op                                    = {},
    std::optional<std::reference_wrapper<HostTensor<CompDataType>>> lse_b_m = std::nullopt)
{
    const int N = a_b_m_n.mDesc.get_lengths()[2];

    auto f = [&](auto batch, auto m) {
        CompDataType v_max = -ck_tile::numeric<CompDataType>::infinity();

        // max
        for(int n = 0; n < N; ++n)
        {
            const CompDataType v_a = ck_tile::type_convert<CompDataType>(a_b_m_n(batch, m, n));

            v_max = v_max < v_a ? v_a : v_max;
        }

        CompDataType v_exp_sum = 0;
        // validate v_max if all the elements within a row are -INF
        if(std::isinf(v_max) && v_max < 0)
        {
            v_max = ck_tile::type_convert<CompDataType>(0.f);
        }

        // sum
        for(int n = 0; n < N; ++n)
        {
            const CompDataType v_a = ck_tile::type_convert<CompDataType>(a_b_m_n(batch, m, n));

            v_exp_sum += ck_tile::exp(v_a - v_max);
        }

        // if sum is zero(masked), or nan/inf(other computation error), don't do divide
        CompDataType inv_sum = (v_exp_sum == 0.f ? 1.f : 1.f / v_exp_sum);

        // elementwise
        for(int n = 0; n < N; ++n)
        {
            const CompDataType v_a = ck_tile::type_convert<CompDataType>(a_b_m_n(batch, m, n));
            const CompDataType v_b = ck_tile::exp(v_a - v_max) * inv_sum;

            b_b_m_n(batch, m, n) = ck_tile::type_convert<BDataType>(comp_element_op(v_b));
        }
        // lse
        if(lse_b_m)
        {
            lse_b_m->get()(batch, m) = v_max + ck_tile::log(v_exp_sum);
        }
    };

    make_ParallelTensorFunctor(f, b_b_m_n.mDesc.get_lengths()[0], b_b_m_n.mDesc.get_lengths()[1])(
        std::thread::hardware_concurrency());
}
} // namespace ck_tile
