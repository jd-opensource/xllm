// SPDX-License-Identifier: MIT
// Copyright (c) 2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>
#include <numeric>
#include <functional>
#include <utility>
#include <algorithm>

namespace ck_tile {

/*
    similiar to torch.topk()
    x (Tensor) – the input tensor.
    k (int) – the k in “top-k”
    dim (int, optional) – the dimension to sort along
    largest (bool, optional) – largest or smallest elements
    sorted (bool, optional) – elements in sorted order or not

    output:
    y_values
    y_indices

    https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/TopKImpl.h
*/
template <typename DataType, typename IndexType = index_t>
CK_TILE_HOST void reference_topk(const HostTensor<DataType>& x,
                                 HostTensor<DataType>& y_values,
                                 HostTensor<IndexType>& y_indices,
                                 index_t k,
                                 index_t dim  = -1,
                                 bool largest = true,
                                 bool sorted  = true)
{
    // rank must be the same
    index_t rank = x.get_num_of_dimension();
    assert(rank == y_values.get_num_of_dimension());
    assert(rank == y_indices.get_num_of_dimension());
    assert(dim == -1 || dim < rank);

    index_t topk_dim     = dim == -1 ? (rank - 1) : dim;
    index_t topk_src_len = x.get_length(topk_dim);
    auto x_len           = x.get_lengths();

    assert(k <= topk_src_len);
    assert(k == y_values.get_length(topk_dim) && k == y_indices.get_length(topk_dim));

    index_t n_parallel = x.get_element_size() / topk_src_len;

    // clang-format off
    auto f = [&](auto i_element) {
        std::vector<size_t> topk_coord = [&](){
            std::vector<size_t> t_(rank, 0);
            size_t r = i_element;
            for(index_t i = rank - 1; i >= 0; i--) {
                if(i == topk_dim)          continue; // topk dim should be zero
                t_[i] = r % x_len[i];      r = r / x_len[i];
            }
            return t_;
        }();

        using elem_t = std::pair<DataType, IndexType>;
        std::vector<elem_t> q = [&](){
            std::vector<elem_t> t_(topk_src_len);
            for(index_t i = 0; i < topk_src_len; i++) {
                auto c_ = topk_coord;  c_[topk_dim] = i;
                t_[i].first = x(c_);   t_[i].second = i;
            }
            return t_;
        }();

        // run topk
        if(largest) {
            std::nth_element(q.begin(), q.begin() + k - 1, q.end(),
            [](const elem_t& lhs, const elem_t& rhs) -> bool { return lhs.first > rhs.first; });
            if(sorted) {
                std::sort(q.begin(), q.begin() + k - 1,
                [](const elem_t& lhs, const elem_t& rhs) -> bool { return lhs.first > rhs.first; });
            }
        } else {
            std::nth_element(q.begin(), q.begin() + k - 1, q.end(),
            [](const elem_t& lhs, const elem_t& rhs) -> bool { return lhs.first < rhs.first; });
            if(sorted) {
                std::sort(q.begin(), q.begin() + k - 1,
                [](const elem_t& lhs, const elem_t& rhs) -> bool { return lhs.first < rhs.first; });
            }
        }

        // write out
        for(index_t i = 0; i < k; i++) {
            auto c_ = topk_coord;  c_[topk_dim] = i;
            y_values(c_) = q[i].first;  y_indices(c_) = q[i].second;
        }
    };
    // clang-format on

    make_ParallelTensorFunctor(f, n_parallel)(std::thread::hardware_concurrency());
}

// TODO: if using this method, the return tensor would be dense(no stride)
template <typename DataType, typename IndexType = index_t>
CK_TILE_HOST auto reference_topk(const HostTensor<DataType>& x,
                                 index_t k,
                                 index_t dim  = -1,
                                 bool largest = true,
                                 bool sorted  = true)
{
    auto lens          = x.get_lengths();
    index_t target_dim = (dim == -1) ? (lens.size() - 1) : dim;
    assert(target_dim < lens.size());
    assert(k <= lens[target_dim]);
    lens[target_dim] = k;
    HostTensor<DataType> y_values(lens);
    HostTensor<IndexType> y_indices(lens);

    reference_topk<DataType, IndexType>(x, y_values, y_indices, k, dim, largest, sorted);

    return ck_tile::make_tuple(y_values, y_indices);
}
} // namespace ck_tile
