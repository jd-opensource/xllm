// SPDX-License-Identifier: MIT
// Copyright (c) 2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {
// [indexing implementation-1]
// using M_a as constexpr block_size to partition all tokens into different slices
// each slice map to one expert, and one expert can have multiple slices
// e.g. num_experts = 6, topk=3, M_a = 4, input_tokens = 5
// before sort, topk_ids is : [[0, 3, 5], [2, 3, 5], [1, 3, 5], [1, 2, 3], [1, 3, 5]]
//                            tok-0      tok-1      tok-2      tok-3      tok-4
//           topk_weight is : [[a, b, c], [d, e, f], [g, h, i], [j, k, l], [m, n, o]] (some float
//           number)
//
// token_id_per_expert is : [[0], [2, 3, 4], [1, 3], [0, 1, 2, 3, 4], [], [0, 1, 2, 5]]
//  (only for reference)    exp-0  exp-1     exp-2   exp-3          exp-4  exp-5
// weight_id_per_expert is: [[a], [g, j, m], [d, k], [b, e, h, l, n], [], [c, f, i, o]]
//
// max_num_tokens_padded : topk * input_tokens + num_experts * (M_a - 1)
// max_num_tokens_padded : topk * input_tokens + num_experts * M_a - topk (updated)
// * this could be larger than actual, since actual tokens are on GPU
//
// sorted_token_ids_ptr   : [0, 6, 6, 6, 2, 3, 4, 6, 1, 3, 6, 6, 0, 1, 2, 3, 4, 6, 6, 6, 6, 6, 6, 6,
// 0, 1, 2, 5]
//                          |-  exp-0  -|-  exp-1  -|-  exp-2  -|-      exp-3          -|-  exp-4
//                          -|-  exp-5  -|
// sorted_weight_ptr      : [a, *, *, *, g, j, m, *, d, k, *, *, b, e, h, l, n, *, *, *, *, *, *, *,
// c, f, i, o]
//
// * length is max_num_tokens_padded, actual size is num_tokens_post_padded_ptr
//
// sorted_expert_ids_ptr  : [0, 1, 2, 3, 3, 4, 5]
// * length is (max_num_tokens_padded + block_size - 1) / block_size
///
// num_tokens_post_padded_ptr : [28]
// num_sorted_tiles_ptr : [7]

template <typename AccDataType, // you only need to explcitly set this one
          typename Activation,  // ck_tile::element_wise::Gelu
          typename ADataType,
          typename GDataType,
          typename DDataType,
          typename ODataType,
          typename AScaleDataType,
          typename GScaleDataType,
          typename DScaleDataType,
          typename YSmoothScaleDataType,
          typename TopkWeightDataType,
          typename IndexDataType>
void reference_fused_moe(
    const ck_tile::HostTensor<ADataType>& a_host,       // [tokens, hidden_size]
    const ck_tile::HostTensor<GDataType>& g_host,       // [experts, interme_size_0, hidden_size]
    const ck_tile::HostTensor<DDataType>& d_host,       // [experts, hidden_size, interme_size_1]
    const ck_tile::HostTensor<AScaleDataType>& sa_host, // [tokens, 1],
    const ck_tile::HostTensor<GScaleDataType>& sg_host, // [experts, 1, interme_size_0]
    const ck_tile::HostTensor<DScaleDataType>& sd_host, // [experts, 1, hidden_size],
    const ck_tile::HostTensor<YSmoothScaleDataType>& sy_host,        // [experts, 1, interme_size_0]
    ck_tile::HostTensor<ODataType>& o_host,                          // [tokens, hidden_size]
    const ck_tile::HostTensor<IndexDataType>& sorted_token_ids_host, // [max_num_tokens_padded]
    const ck_tile::HostTensor<TopkWeightDataType>& sorted_weight_host, // [max_num_tokens_padded]
    const ck_tile::HostTensor<IndexDataType>&
        sorted_expert_ids_host, // [(max_num_tokens_padded + block_size - 1) / block_size]
    const ck_tile::HostTensor<IndexDataType>& num_sorted_tiles_host, // [1]

    const ck_tile::HostTensor<IndexDataType>&
        token_ids_host, // [tokens, topk] --> ugly!!! remove in the future

    ck_tile::index_t block_m,
    ck_tile::index_t tokens,
    ck_tile::index_t experts,
    ck_tile::index_t hidden_size,
    ck_tile::index_t intermediate_size, // this size is for gate/up/down
    ck_tile::index_t topk,
    ck_tile::index_t gate_only)
{
    assert(sorted_token_ids_host.get_num_of_dimension() == 1);
    assert(sorted_weight_host.get_num_of_dimension() == 1);
    assert(sorted_expert_ids_host.get_num_of_dimension() == 1);
    assert(num_sorted_tiles_host.get_element_size() == 1);
    ck_tile::index_t num_sorted_tiles    = num_sorted_tiles_host.mData[0] / block_m;
    ck_tile::index_t intermediate_size_0 = intermediate_size * (gate_only ? 1 : 2);
    ck_tile::index_t intermediate_size_1 = intermediate_size;

    ck_tile::HostTensor<AccDataType> out_topk_tokens({tokens, topk, hidden_size});

    int max_num_tokens_padded = topk * tokens + experts * block_m - topk;
    // assert();
    auto f = [&](auto i_flatten) {
        ck_tile::index_t i_tile = i_flatten / block_m;
        if(i_tile >= num_sorted_tiles)
            return;
        ck_tile::index_t i_expert = sorted_expert_ids_host.mData[i_tile];

#if CK_TILE_REFERENCE_MOE_SORTING_MOCK_ID
        ck_tile::index_t i_token = sorted_token_ids_host.mData[i_flatten];
        ck_tile::index_t i_topk  = i_token >> 24;
        i_token &= 0xffffff;
        if(i_token >= tokens)
            return;
        (void)token_ids_host;
#elif CK_TILE_REFERENCE_MOE_SORTING_MOCK_EXPERT_ID
        auto get_topk_id = [&](ck_tile::index_t token_id_, ck_tile::index_t expert_id_) {
            for(ck_tile::index_t i_ = 0; i_ < topk; i_++)
            {
                if(token_ids_host(token_id_, i_) == expert_id_)
                    return i_;
            }
            throw std::runtime_error("not correct token/expert pair\n");
            return -1; // TODO: not correct!!
        };
        ck_tile::index_t i_token = sorted_token_ids_host.mData[i_flatten];
        i_token &= 0xffffff;
        if(i_token >= tokens)
            return;
        ck_tile::index_t i_topk = get_topk_id(i_token, i_expert);
#else
        // TODO: better remove this in the future, or modify the token_id value
        auto get_topk_id = [&](ck_tile::index_t token_id_, ck_tile::index_t expert_id_) {
            for(ck_tile::index_t i_ = 0; i_ < topk; i_++)
            {
                if(token_ids_host(token_id_, i_) == expert_id_)
                    return i_;
            }
            throw std::runtime_error("not correct token/expert pair\n");
            return -1; // TODO: not correct!!
        };
        ck_tile::index_t i_token = sorted_token_ids_host.mData[i_flatten];
        if(i_token >= tokens)
            return;
        ck_tile::index_t i_topk = get_topk_id(i_token, i_expert); // TODO: ugly
#endif
        auto weight = sorted_weight_host.mData[i_flatten];

        ck_tile::HostTensor<AccDataType> acc_0({1, intermediate_size_0});
        // first gemm
        for(ck_tile::index_t i_n = 0; i_n < intermediate_size_0; i_n++)
        {
            AccDataType acc = static_cast<AccDataType>(0);
            for(ck_tile::index_t i_k = 0; i_k < hidden_size; i_k++)
            {
                acc += type_convert<AccDataType>(a_host(i_token, i_k)) *
                       type_convert<AccDataType>(g_host(i_expert, i_n, i_k));
            }
            acc_0(0, i_n) = acc;
            // printf("ie:%2d, it:%3d, in:%d, %f\n", i_expert, i_token, i_n, acc);
        }

        ck_tile::HostTensor<AccDataType> y({1, intermediate_size_1});
        if(gate_only)
        {
            if(intermediate_size_1 != intermediate_size_0)
                throw std::runtime_error(
                    "intermediate_size not correct, 0:" + std::to_string(intermediate_size_0) +
                    ", 1:" + std::to_string(intermediate_size_1));
            for(ck_tile::index_t i_n = 0; i_n < intermediate_size_1; i_n++)
            {
                Activation{}(y(0, i_n), acc_0(0, i_n));
                // printf("ie:%2d, it:%3d, in:%d, %f\n", i_expert, i_token, i_n, y(0, i_n));
            }
        }
        else
        {
            if(intermediate_size_1 * 2 != intermediate_size_0)
                throw std::runtime_error(
                    "intermediate_size not correct, 0:" + std::to_string(intermediate_size_0) +
                    ", 1:" + std::to_string(intermediate_size_1));
            for(ck_tile::index_t i_n = 0; i_n < intermediate_size_1; i_n++)
            {
                AccDataType tmp;
                Activation{}(tmp, acc_0(0, i_n));
                y(0, i_n) = tmp * acc_0(0, i_n + intermediate_size_1); // TODO: elementwise mul
            }
        }

        // second gemm, loop along gemm-n
        ck_tile::HostTensor<AccDataType> acc_1({1, hidden_size});
        for(ck_tile::index_t i_n = 0; i_n < hidden_size; i_n++)
        {
            AccDataType acc = static_cast<AccDataType>(0);
            for(ck_tile::index_t i_k = 0; i_k < intermediate_size_1; i_k++)
            {
                acc += y(0, i_k) * type_convert<AccDataType>(d_host(i_expert, i_n, i_k));
            }
            acc_1(0, i_n) = acc * weight; // multiple weight here
        }

        for(ck_tile::index_t i_n = 0; i_n < hidden_size; i_n++)
        {
            out_topk_tokens(i_token, i_topk, i_n) = acc_1(0, i_n);
        }
    };

    // make_ParallelTensorFunctor(f, max_num_tokens_padded)(std::thread::hardware_concurrency());
    make_ParallelTensorFunctor(f, max_num_tokens_padded)(1);

    // reduce
    auto r = [&](auto i_token) {
        for(ck_tile::index_t i_n = 0; i_n < hidden_size; i_n++)
        {
            AccDataType acc = type_convert<AccDataType>(0);
            for(ck_tile::index_t i_topk = 0; i_topk < topk; i_topk++)
            {
                acc += out_topk_tokens(i_token, i_topk, i_n);
            }
            o_host(i_token, i_n) = type_convert<ODataType>(acc);
        }
    };
    make_ParallelTensorFunctor(r, tokens)(std::thread::hardware_concurrency());

    (void)num_sorted_tiles_host;
    (void)sa_host;
    (void)sg_host;
    (void)sd_host;
    (void)sy_host;
}
} // namespace ck_tile
