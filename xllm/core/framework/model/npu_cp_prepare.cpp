/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "framework/model/npu_cp_prepare.h"

#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace xllm {

    torch::Tensor generate_cp_load_balance_idx(const torch::Tensor& input_lengths) {
        TORCH_CHECK(input_lengths.dtype() == torch::kInt32, "input_lengths must be int32 tensor");
        TORCH_CHECK(input_lengths.dim() == 1, "input_lengths must be 1D tensor");

        std::vector<int> lengths_vec;
        int* lengths_ptr = input_lengths.data_ptr<int>();
        int64_t n = input_lengths.numel();
        for (int64_t i = 0; i < n; ++i) {
            lengths_vec.push_back(lengths_ptr[i]);
        }

        std::vector<int> cp_load_balance_idx_first, cp_load_balance_idx_last;
        int base = 0;
        for (int length : lengths_vec) {
            std::vector<int> length_range(length);
            std::iota(length_range.begin(), length_range.end(), base); // 生成连续整数序列
            int divider = length / 2;
            // 前半段
            cp_load_balance_idx_first.insert(
                cp_load_balance_idx_first.end(),
                length_range.begin(),
                length_range.begin() + divider);
            // 后半段
            cp_load_balance_idx_last.insert(
                cp_load_balance_idx_last.end(),
                length_range.begin() + divider,
                length_range.end());
            base += length;
        }

        // 合并前半段+后半段
        cp_load_balance_idx_first.insert(
            cp_load_balance_idx_first.end(),
            cp_load_balance_idx_last.begin(),
            cp_load_balance_idx_last.end());

        // 转换为torch张量（NPU设备）
        auto tensor = torch::tensor(cp_load_balance_idx_first, torch::dtype(torch::kInt32).device(torch::kCPU));
        return tensor;
    }

    // 生成输出结果恢复索引
    torch::Tensor generate_cp_o_recover_idx(const std::vector<int>& chunk_lengths) {
        std::vector<int> cp_o_recover_idx;
        int base = 0;
        int chunk_lengths_sum = std::accumulate(chunk_lengths.begin(), chunk_lengths.end(), 0);

        for (int chunk_len : chunk_lengths) {
            std::vector<int> length_range(chunk_len);
            std::iota(length_range.begin(), length_range.end(), base);
            // 前半段
            cp_o_recover_idx.insert(cp_o_recover_idx.end(), length_range.begin(), length_range.end());
            // 后半段（偏移chunk_lengths_sum）
            std::vector<int> last_part(length_range.size());
            std::transform(
                length_range.begin(), length_range.end(),
                last_part.begin(),
                [chunk_lengths_sum](int x) { return x + chunk_lengths_sum; });
            cp_o_recover_idx.insert(cp_o_recover_idx.end(), last_part.begin(), last_part.end());
            base += chunk_len;
        }

        return torch::tensor(cp_o_recover_idx, torch::dtype(torch::kInt32).device(torch::kCPU));
    }

    // 生成KV缓存恢复索引
    torch::Tensor generate_cp_kv_recover_idx(int cp_size, int input_ids_size, const std::vector<int>& chunk_lengths) {
        std::vector<int> cp_kv_recover_idx;
        int req_offset = 0;

        for (int req_chunk_len : chunk_lengths) {
            std::vector<std::vector<int>> gather_idx_per_chunk(cp_size * 2);
            for (int cp_rank_id = 0; cp_rank_id < cp_size; ++cp_rank_id) {
                int rank_offset = cp_rank_id * input_ids_size;
                // 前半段索引
                std::vector<int> first_part(req_chunk_len);
                std::iota(first_part.begin(), first_part.end(), rank_offset + req_offset);
                gather_idx_per_chunk[cp_rank_id] = first_part;

                std::vector<int> last_part(req_chunk_len);
                std::iota(last_part.begin(), last_part.end(), rank_offset + req_offset + req_chunk_len);
                gather_idx_per_chunk[cp_size * 2 - 1 - cp_rank_id] = last_part;
            }

            for (const auto& vec : gather_idx_per_chunk) {
                cp_kv_recover_idx.insert(cp_kv_recover_idx.end(), vec.begin(), vec.end());
            }
            req_offset += req_chunk_len * 2;
        }

        return torch::tensor(cp_kv_recover_idx, torch::dtype(torch::kInt32).device(torch::kCPU));
    }

    std::pair<torch::Tensor, torch::Tensor> compute_input_lengths_cumsum_cp(
        const torch::Tensor& input_lengths_cumsum) {
        TORCH_CHECK(input_lengths_cumsum.dtype() == torch::kInt32, "input_lengths_cumsum must be int32 tensor");
        TORCH_CHECK(input_lengths_cumsum.dim() == 1, "input_lengths_cumsum must be 1D tensor");

        int64_t n = input_lengths_cumsum.numel();
        auto input_lengths_cumsum_cp_prev = torch::zeros({n}, torch::dtype(torch::kInt32).device(torch::kCPU));
        auto input_lengths_cumsum_cp_next = torch::zeros({n}, torch::dtype(torch::kInt32).device(torch::kCPU));

        int offset = 0;
        auto cumsum_data = input_lengths_cumsum.data_ptr<int>();
        auto prev_data = input_lengths_cumsum_cp_prev.data_ptr<int>();
        auto next_data = input_lengths_cumsum_cp_next.data_ptr<int>();

        for (int64_t i = 0; i < n; ++i) {
            prev_data[i] = offset + (cumsum_data[i] - offset) / 2;
            next_data[i] = cumsum_data[i];
            offset = cumsum_data[i];
        }

        return {input_lengths_cumsum_cp_prev, input_lengths_cumsum_cp_next};
    }

    std::pair<torch::Tensor, torch::Tensor> generate_k_gather_index(
        const torch::Tensor& actual_seq_lengths_kv_cp_prev,
        const torch::Tensor& actual_seq_lengths_kv_cp_next,
        const torch::Tensor& input_lengths,
        int cp_size) {
        TORCH_CHECK(actual_seq_lengths_kv_cp_prev.dim() == 1, "actual_seq_lengths_kv_cp_prev must be 1D");
        TORCH_CHECK(actual_seq_lengths_kv_cp_next.dim() == 1, "actual_seq_lengths_kv_cp_next must be 1D");
        TORCH_CHECK(input_lengths.dim() == 1, "input_lengths must be 1D");

        std::vector<int> k_gather_index_prev, k_gather_index_next;
        int k_offset = 0;
        int64_t n = input_lengths.numel();

        auto prev_len_data = actual_seq_lengths_kv_cp_prev.data_ptr<int>();
        auto next_len_data = actual_seq_lengths_kv_cp_next.data_ptr<int>();
        auto input_len_data = input_lengths.data_ptr<int>();

        for (int64_t i = 0; i < n; ++i) {
            // 前半段索引
            std::vector<int> prev_range(prev_len_data[i]);
            std::iota(prev_range.begin(), prev_range.end(), k_offset);
            k_gather_index_prev.insert(k_gather_index_prev.end(), prev_range.begin(), prev_range.end());

            // 完整段索引
            std::vector<int> next_range(next_len_data[i]);
            std::iota(next_range.begin(), next_range.end(), k_offset);
            k_gather_index_next.insert(k_gather_index_next.end(), next_range.begin(), next_range.end());

            k_offset += input_len_data[i] * cp_size;
        }

        auto prev_tensor = torch::tensor(k_gather_index_prev, torch::dtype(torch::kInt32).device(torch::kCPU));
        auto next_tensor = torch::tensor(k_gather_index_next, torch::dtype(torch::kInt32).device(torch::kCPU));
        return {prev_tensor, next_tensor};
    }

    CpPrefillInputs prepare_cp_prefill_inputs(
        int cp_size,
        const torch::Tensor& input_ids,
        const torch::Tensor& position_ids,
        const torch::Tensor& input_lengths) {


        TORCH_CHECK(cp_size > 0, "cp_size must be positive");
        CpPrefillInputs inputs;

        std::vector<int> chunk_lengths;
        auto input_len_data = input_lengths.data_ptr<int>();
        for (int64_t i = 0; i < input_lengths.numel(); ++i) {
            chunk_lengths.push_back(input_len_data[i] / 2);
        }

        inputs.cp_load_balance_idx = generate_cp_load_balance_idx(input_lengths);

        inputs.cp_o_recover_idx = generate_cp_o_recover_idx(chunk_lengths);

        inputs.cp_kv_recover_idx = generate_cp_kv_recover_idx(cp_size, input_ids.numel(), chunk_lengths);

        auto input_lengths_cumsum = torch::cumsum(input_lengths, 0, torch::kInt32);
        auto [input_lengths_cumsum_cp_prev, input_lengths_cumsum_cp_next] =
            compute_input_lengths_cumsum_cp(input_lengths_cumsum);

        auto gather_index_prev = (input_lengths_cumsum_cp_prev - 1).to(torch::kLong);
        auto gather_index_next = (input_lengths_cumsum_cp_next - 1).to(torch::kLong);
        auto position_ids_prev = position_ids.index_select(0, gather_index_prev) + 1;
        auto position_ids_next = position_ids.index_select(0, gather_index_next) + 1;
        auto actual_seq_lengths_kv_cp_prev = position_ids_prev.to(torch::kInt32);
        auto actual_seq_lengths_kv_cp_next = position_ids_next.to(torch::kInt32);

        std::tie(inputs.k_gather_index_prev, inputs.k_gather_index_next) = generate_k_gather_index(
            actual_seq_lengths_kv_cp_prev,
            actual_seq_lengths_kv_cp_next,
            input_lengths,
            cp_size);

        auto actual_seq_lengths_kv_cp_prev_cumsum = torch::cumsum(actual_seq_lengths_kv_cp_prev, 0, torch::kInt32);
        auto actual_seq_lengths_kv_cp_next_cumsum = torch::cumsum(actual_seq_lengths_kv_cp_next, 0, torch::kInt32);
        inputs.actual_seq_lengths_key_prev = actual_seq_lengths_kv_cp_prev_cumsum;
        inputs.actual_seq_lengths_key_next = actual_seq_lengths_kv_cp_next_cumsum;

        auto input_lengths_cumsum_half = input_lengths_cumsum / 2;
        inputs.actual_seq_lengths_query_prev = input_lengths_cumsum_half;
        inputs.actual_seq_lengths_query_next = input_lengths_cumsum_half;
        return inputs;
    }


}  // namespace xllm
