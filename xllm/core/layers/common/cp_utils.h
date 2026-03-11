#pragma once

#include <torch/torch.h>
// #include <torch_npu/torch_npu.h>
#include <vector>
#include <numeric>
#include <algorithm>

// 命名空间隔离
namespace xllm {
    namespace layer {

        // 核心数据结构：存储CP预处理的所有索引和长度信息
        struct CPInputDict {
            torch::Tensor cp_load_balance_idx;
            torch::Tensor cp_o_recover_idx;
            torch::Tensor cp_kv_recover_idx;
            std::pair<torch::Tensor, torch::Tensor> k_gather_index;
            std::pair<torch::Tensor, torch::Tensor> actual_seq_lengths_key;
            std::pair<torch::Tensor, torch::Tensor> actual_seq_lengths_query;
        };

        /**
         * @brief 生成负载均衡输入索引（cp_load_balance_idx）
         * @param input_lengths 每个请求的原始长度（torch::Tensor, int32）
         * @return 负载均衡输入索引张量
         */
        torch::Tensor generate_cp_load_balance_idx(const torch::Tensor& input_lengths);

        /**
         * @brief 生成输出结果恢复索引（cp_o_recover_idx）
         * @param chunk_lengths 每个请求的分片长度（std::vector<int>）
         * @return 输出恢复索引张量
         */
        torch::Tensor generate_cp_o_recover_idx(const std::vector<int>& chunk_lengths);

        /**
         * @brief 生成KV缓存恢复索引（cp_kv_recover_idx）
         * @param cp_size CP并行卡数
         * @param input_ids_size input_ids的总长度
         * @param chunk_lengths 每个请求的分片长度（std::vector<int>）
         * @return KV恢复索引张量
         */
        torch::Tensor generate_cp_kv_recover_idx(int cp_size, int input_ids_size, const std::vector<int>& chunk_lengths);

        /**
         * @brief 计算长度累积的前后段值
         * @param input_lengths_cumsum 输入长度累积值（torch::Tensor, int32）
         * @return 前半段/完整段长度累积值的pair
         */
        std::pair<torch::Tensor, torch::Tensor> compute_input_lengths_cumsum_cp(
            const torch::Tensor& input_lengths_cumsum);

        /**
         * @brief 生成KV gather索引
         * @param actual_seq_lengths_kv_cp_prev 前半段KV实际长度（torch::Tensor, int32）
         * @param actual_seq_lengths_kv_cp_next 完整段KV实际长度（torch::Tensor, int32）
         * @param input_lengths 每个请求的原始长度（torch::Tensor, int32）
         * @param cp_size CP并行卡数
         * @return KV gather索引的pair
         */
        std::pair<torch::Tensor, torch::Tensor> generate_k_gather_index(
            const torch::Tensor& actual_seq_lengths_kv_cp_prev,
            const torch::Tensor& actual_seq_lengths_kv_cp_next,
            const torch::Tensor& input_lengths,
            int cp_size);

        /**
         * @brief 核心函数：CP预填充输入预处理
         * @param cp_size CP并行卡数
         * @param input_ids 输入序列ID（torch::Tensor, int32）
         * @param position_ids 位置编码ID（torch::Tensor, int32）
         * @param input_lengths_cumsum 输入长度累积值（torch::Tensor, int32）
         * @param input_lengths 每个请求的原始长度（torch::Tensor, int32）
         * @return CP预处理的索引和长度信息
         */
        CPInputDict prepare_cp_prefill_inputs(
            int cp_size,
            const torch::Tensor& input_ids,
            const torch::Tensor& position_ids,
            const torch::Tensor& input_lengths_cumsum,
            const torch::Tensor& input_lengths);

    } // namespace layer
} // namespace xllm

