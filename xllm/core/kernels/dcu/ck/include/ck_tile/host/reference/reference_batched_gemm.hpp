// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename AElementOp   = ck_tile::identity,
          typename BElementOp   = ck_tile::identity,
          typename ACCElementOp = ck_tile::identity>
CK_TILE_HOST void reference_batched_gemm(const HostTensor<ADataType>& a_b_m_k,
                                         const HostTensor<BDataType>& b_b_n_k,
                                         HostTensor<CDataType>& c_b_m_n,
                                         const AElementOp& a_element_op     = {},
                                         const BElementOp& b_element_op     = {},
                                         const ACCElementOp& acc_element_op = {})
{
    const int N = b_b_n_k.mDesc.get_lengths()[1];
    const int K = b_b_n_k.mDesc.get_lengths()[2];

    auto f = [&](auto batch, auto m) {
        for(int n = 0; n < N; ++n)
        {
            AccDataType v_acc = 0;

            for(int k = 0; k < K; ++k)
            {
                ADataType v_a = a_element_op(a_b_m_k(batch, m, k));
                BDataType v_b = b_element_op(b_b_n_k(batch, n, k));

                v_acc += ck_tile::type_convert<AccDataType>(v_a) *
                         ck_tile::type_convert<AccDataType>(v_b);
            }

            c_b_m_n(batch, m, n) = ck_tile::type_convert<CDataType>(acc_element_op(v_acc));
        }
    };

    make_ParallelTensorFunctor(f, c_b_m_n.mDesc.get_lengths()[0], c_b_m_n.mDesc.get_lengths()[1])(
        std::thread::hardware_concurrency());
}
} // namespace ck_tile
