#include "cp_utils.h"
// #include <torch/npu/guard.h> // 适配NPU（华为昇腾），CPU/GPU场景可替换为torch::DeviceGuard

namespace xllm {
    namespace layer {

        // 生成负载均衡输入索引
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

                    // 后半段索引（反向分配）
                    std::vector<int> last_part(req_chunk_len);
                    std::iota(last_part.begin(), last_part.end(), rank_offset + req_offset + req_chunk_len);
                    gather_idx_per_chunk[cp_size * 2 - 1 - cp_rank_id] = last_part;
                }

                // 展平并添加到总索引
                for (const auto& vec : gather_idx_per_chunk) {
                    cp_kv_recover_idx.insert(cp_kv_recover_idx.end(), vec.begin(), vec.end());
                }
                req_offset += req_chunk_len * 2;
            }

            return torch::tensor(cp_kv_recover_idx, torch::dtype(torch::kInt32).device(torch::kCPU));
        }

        // 计算长度累积的前后段值
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

        // 生成KV gather索引
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

        // 核心函数：CP预填充输入预处理
        CPInputDict prepare_cp_prefill_inputs(
            int cp_size,
            const torch::Tensor& input_ids,
            const torch::Tensor& position_ids,
            const torch::Tensor& input_lengths_cumsum,
            const torch::Tensor& input_lengths) {
            // 输入合法性检查
            TORCH_CHECK(cp_size > 0, "cp_size must be positive");
            // TORCH_CHECK(input_ids.device().type() == torch::kCPU, "input_ids must be on NPU");
            // TORCH_CHECK(position_ids.device().type() == torch::kCPU, "position_ids must be on NPU");
            // TORCH_CHECK(input_lengths_cumsum.device().type() == torch::kCPU, "input_lengths_cumsum must be on NPU");
            // TORCH_CHECK(input_lengths.device().type() == torch::kCPU, "input_lengths must be on NPU");

            CPInputDict cp_input_dict;

            // 1. 计算分片长度
            std::vector<int> chunk_lengths;
            auto input_len_data = input_lengths.data_ptr<int>();
            for (int64_t i = 0; i < input_lengths.numel(); ++i) {
                chunk_lengths.push_back(input_len_data[i] / 2);
            }

            // 2. 生成负载均衡输入索引
            cp_input_dict.cp_load_balance_idx = generate_cp_load_balance_idx(input_lengths);

            // 3. 生成输出结果恢复索引
            cp_input_dict.cp_o_recover_idx = generate_cp_o_recover_idx(chunk_lengths);

            // 4. 生成KV缓存恢复索引
            cp_input_dict.cp_kv_recover_idx = generate_cp_kv_recover_idx(cp_size, input_ids.numel(), chunk_lengths);

            // 5. 计算长度累积的前后段值
            auto [input_lengths_cumsum_cp_prev, input_lengths_cumsum_cp_next] =
                compute_input_lengths_cumsum_cp(input_lengths_cumsum);

            // 6. 计算KV实际长度
            auto position_ids_prev = position_ids.index({input_lengths_cumsum_cp_prev - 1}) + 1;
            auto position_ids_next = position_ids.index({input_lengths_cumsum_cp_next - 1}) + 1;
            auto actual_seq_lengths_kv_cp_prev = position_ids_prev.to(torch::kInt32);
            auto actual_seq_lengths_kv_cp_next = position_ids_next.to(torch::kInt32);

            // 7. 生成KV gather索引
            cp_input_dict.k_gather_index = generate_k_gather_index(
                actual_seq_lengths_kv_cp_prev,
                actual_seq_lengths_kv_cp_next,
                input_lengths,
                cp_size);

            // 8. 计算KV长度累积值
            auto actual_seq_lengths_kv_cp_prev_cumsum = torch::cumsum(actual_seq_lengths_kv_cp_prev, 0, torch::kInt32);
            auto actual_seq_lengths_kv_cp_next_cumsum = torch::cumsum(actual_seq_lengths_kv_cp_next, 0, torch::kInt32);
            cp_input_dict.actual_seq_lengths_key = {actual_seq_lengths_kv_cp_prev_cumsum, actual_seq_lengths_kv_cp_next_cumsum};

            // 9. 计算Query长度累积值
            auto input_lengths_cumsum_half = input_lengths_cumsum / 2;
            cp_input_dict.actual_seq_lengths_query = {input_lengths_cumsum_half, input_lengths_cumsum_half};

            return cp_input_dict;
        }

        // bool verify_vector(const std::vector<int>& actual, const std::vector<int>& expected, const std::string& test_name) {
        //     if (actual.size() != expected.size()) {
        //         std::cerr << "[FAIL] " << test_name << ": Size mismatch! Actual=" << actual.size() << ", Expected=" << expected.size() << std::endl;
        //         return false;
        //     }
        //     for (size_t i = 0; i < actual.size(); ++i) {
        //         if (actual[i] != expected[i]) {
        //             std::cerr << "[FAIL] " << test_name << ": Index " << i << " mismatch! Actual=" << actual[i] << ", Expected=" << expected[i] << std::endl;
        //             return false;
        //         }
        //     }
        //     std::cout << "[PASS] " << test_name << std::endl;
        //     return true;
        // }
        //
        // std::vector<int> tensor_to_vector(const torch::Tensor& tensor) {
        //     // 修复：CPU tensor无需再转换设备
        //     int* data_ptr = tensor.data_ptr<int>();
        //     int64_t n = tensor.numel();
        //     std::vector<int> vec(n);
        //     for (int64_t i = 0; i < n; ++i) {
        //         vec[i] = data_ptr[i];
        //     }
        //     return vec;
        // }

        // int main() {
        //     std::cout << "===== Start Testing =====" << std::endl;
        //     int pass_count = 0;
        //     int total_count = 0;
        //
        //     // 测试用例1
        //     total_count++;
        //     std::cout << "\nTest 1: generate_cp_load_balance_idx" << std::endl;
        //     auto input_lengths = torch::tensor({4, 6}, torch::dtype(torch::kInt32).device(torch::kCPU));
        //     std::vector<int> expected_load_balance = {0,1,4,5,6,2,3,7,8,9};
        //     auto result_load_balance = generate_cp_load_balance_idx(input_lengths);
        //     std::vector<int> actual_load_balance = tensor_to_vector(result_load_balance);
        //     if (verify_vector(actual_load_balance, expected_load_balance, "generate_cp_load_balance_idx")) {
        //         pass_count++;
        //     }
        //
        //     // 测试用例2
        //     total_count++;
        //     std::cout << "\nTest 2: generate_cp_o_recover_idx" << std::endl;
        //     std::vector<int> chunk_lengths = {2, 3};
        //     std::vector<int> expected_o_recover = {0,1,5,6,2,3,4,7,8,9};
        //     auto result_o_recover = generate_cp_o_recover_idx(chunk_lengths);
        //     std::vector<int> actual_o_recover = tensor_to_vector(result_o_recover);
        //     if (verify_vector(actual_o_recover, expected_o_recover, "generate_cp_o_recover_idx")) {
        //         pass_count++;
        //     }
        //
        //     // 测试用例3
        //     total_count++;
        //     std::cout << "\nTest 3: prepare_cp_prefill_inputs (full logic)" << std::endl;
        //     int cp_size = 2;
        //     auto input_ids = torch::tensor({101,102,103,104,201,202,203,204,205,206}, torch::dtype(torch::kInt32).device(torch::kCPU));
        //     auto position_ids = torch::tensor({0,1,2,3,0,1,2,3,4,5}, torch::dtype(torch::kInt32).device(torch::kCPU));
        //     auto input_lengths_cumsum = torch::tensor({4,10}, torch::dtype(torch::kInt32).device(torch::kCPU));
        //
        //     auto cp_input_dict = prepare_cp_prefill_inputs(cp_size, input_ids, position_ids, input_lengths_cumsum, input_lengths);
        //
        //     bool full_test_pass = true;
        //     auto key_prev_vec = tensor_to_vector(cp_input_dict.actual_seq_lengths_key.first);
        //     if (key_prev_vec[0] != 2 || key_prev_vec[1] != 5) {
        //         std::cerr << "[FAIL] prepare_cp_prefill_inputs: actual_seq_lengths_key.first mismatch! Actual=[" << key_prev_vec[0] << "," << key_prev_vec[1] << "], Expected=[2,5]" << std::endl;
        //         full_test_pass = false;
        //     }
        //     auto key_next_vec = tensor_to_vector(cp_input_dict.actual_seq_lengths_key.second);
        //     if (key_next_vec[0] != 4 || key_next_vec[1] != 10) {
        //         std::cerr << "[FAIL] prepare_cp_prefill_inputs: actual_seq_lengths_key.second mismatch! Actual=[" << key_next_vec[0] << "," << key_next_vec[1] << "], Expected=[4,10]" << std::endl;
        //         full_test_pass = false;
        //     }
        //     auto full_load_balance_vec = tensor_to_vector(cp_input_dict.cp_load_balance_idx);
        //     if (!verify_vector(full_load_balance_vec, expected_load_balance, "prepare_cp_prefill_inputs (load_balance_idx)")) {
        //         full_test_pass = false;
        //     }
        //     if (full_test_pass) {
        //         std::cout << "[PASS] prepare_cp_prefill_inputs (full logic)" << std::endl;
        //         pass_count++;
        //     } else {
        //         std::cerr << "[FAIL] prepare_cp_prefill_inputs (full logic)" << std::endl;
        //     }
        //
        //     std::cout << "\n===== Test Summary =====" << std::endl;
        //     std::cout << "Total Tests: " << total_count << std::endl;
        //     std::cout << "Passed Tests: " << pass_count << std::endl;
        //     std::cout << "Failed Tests: " << (total_count - pass_count) << std::endl;
        //
        //     return (total_count == pass_count) ? 0 : 1;
        // }
        //
    } // namespace layer
} // namespace xllm
//
// int main() {
//     return context_parallel_forward_input::main();
// }
