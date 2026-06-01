// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>
#include <numeric>
#include <functional>

namespace ck_tile {

/*
    this will do permute + contiguous like functionality in pytorch
*/
template <typename DataType>
CK_TILE_HOST void
reference_permute(const HostTensor<DataType>& x, HostTensor<DataType>& y, std::vector<index_t> dims)
{
    const auto x_len = x.mDesc.get_lengths();
    const auto y_len = y.mDesc.get_lengths();
    assert(x_len.size() == y_len.size());
    index_t rank     = x_len.size();
    const auto x_elm = std::accumulate(x_len.begin(), x_len.end(), 1, std::multiplies<index_t>());
    const auto y_elm = std::accumulate(y_len.begin(), y_len.end(), 1, std::multiplies<index_t>());
    assert(x_elm == y_elm);
    (void)y_elm;

    auto f = [&](auto i_element) {
        std::vector<size_t> y_coord = [&]() {
            std::vector<size_t> tmp(rank, 0);
            size_t r = i_element;
            for(index_t i = rank - 1; i >= 0; i--)
            {
                tmp[i] = r % y_len[i];
                r      = r / y_len[i];
            }
            return tmp;
        }();

        std::vector<size_t> x_coord = [&]() {
            std::vector<size_t> tmp(rank, 0);
            for(index_t i = 0; i < rank; i++)
            {
                tmp[dims[i]] = y_coord[i];
            }
            return tmp;
        }();

        // do permute
        y(y_coord) = x(x_coord);
    };

    make_ParallelTensorFunctor(f, x_elm)(std::thread::hardware_concurrency());
}

template <typename DataType>
CK_TILE_HOST auto reference_permute(const HostTensor<DataType>& x, std::vector<index_t> perm)
{
    auto x_shape                          = x.get_lengths();
    ck_tile::index_t rank                 = perm.size();
    std::vector<ck_tile::index_t> y_shape = [&]() {
        std::vector<ck_tile::index_t> tmp(rank, 0);
        for(int i = 0; i < static_cast<int>(rank); i++)
        {
            tmp[i] = x_shape[perm[i]];
        }
        return tmp;
    }();

    HostTensor<DataType> y(y_shape);
    reference_permute(x, y, perm);
    return y;
}
} // namespace ck_tile
