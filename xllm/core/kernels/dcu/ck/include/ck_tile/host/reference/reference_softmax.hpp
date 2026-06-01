// SPDX-License-Identifier: MIT
// Copyright (c) 2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename InputType, typename ComputeType, typename OutputType = ComputeType>
CK_TILE_HOST void
reference_softmax(const HostTensor<InputType>& x, HostTensor<OutputType>& y, index_t dim = -1)
{
    index_t rank = x.get_num_of_dimension();
    assert(rank == y.get_num_of_dimension());
    assert(dim == -1 || dim < rank);

    index_t target_dim  = dim == -1 ? (rank - 1) : dim;
    index_t softmax_len = x.get_length(target_dim);
    index_t n_parallel  = x.get_element_size() / softmax_len;
    auto x_len          = x.get_lengths();

    auto f = [&](auto i_element) {
        std::vector<size_t> coord = [&]() {
            std::vector<size_t> t_(rank, 0);
            size_t r = i_element;
            for(index_t i = rank - 1; i >= 0; i--)
            {
                if(i == target_dim)
                    continue;
                t_[i] = r % x_len[i];
                r     = r / x_len[i];
            }
            return t_;
        }();

        ComputeType v_max = -ck_tile::numeric<ComputeType>::infinity();

        // compute max
        for(auto idx = 0; idx < softmax_len; idx++)
        {
            auto c_               = coord;
            c_[target_dim]        = idx;
            const ComputeType v_x = ck_tile::type_convert<ComputeType>(x(c_));
            v_max                 = v_max < v_x ? v_x : v_max;
        }

        ComputeType v_exp_sum = static_cast<ComputeType>(0);

        // sum
        for(auto idx = 0; idx < softmax_len; idx++)
        {
            auto c_        = coord;
            c_[target_dim] = idx;

            const ComputeType v_x = ck_tile::type_convert<ComputeType>(x(c_));

            v_exp_sum += ck_tile::exp(v_x - v_max);
        }

        // elementwise
        for(auto idx = 0; idx < softmax_len; idx++)
        {
            auto c_        = coord;
            c_[target_dim] = idx;

            const ComputeType v_x = ck_tile::type_convert<ComputeType>(x(c_));

            auto out = ck_tile::exp(v_x - v_max) / v_exp_sum;

            y(c_) = ck_tile::type_convert<OutputType>(out);
        }
    };

    make_ParallelTensorFunctor(f, n_parallel)(std::thread::hardware_concurrency());
}

template <typename InputType, typename ComputeType, typename OutputType = ComputeType>
CK_TILE_HOST auto reference_softmax(const HostTensor<InputType>& x, index_t dim = -1)
{
    HostTensor<OutputType> y(x.get_lengths(), x.get_strides());

    reference_softmax<InputType, ComputeType, OutputType>(x, y, dim);

    return y;
}
} // namespace ck_tile
