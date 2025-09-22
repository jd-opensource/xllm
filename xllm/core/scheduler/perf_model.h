#pragma once

#include <cassert>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace xllm {
namespace perf_model {

// Enumeration type definitions
enum class OpType {
  PREFILL_GEMM = 1,
  PREFILL_ATTENTION = 2,
  DECODE_GEMM = 3,
  DECODE_ATTENTION = 4
};

// Forward declarations
class PerfModel;
class Resource;

// Global function declarations
void set_perf_model(std::shared_ptr<PerfModel> model);
std::shared_ptr<PerfModel> get_perf_model();

// Resource class definition
class Resource {
 public:
  int64_t flops = 0;
  int64_t memory = 0;
  int64_t network = 0;
  double latency = 0.0;

  Resource() = default;
  Resource(int64_t flops,
           int64_t memory,
           int64_t network = 0,
           double latency = 0.0);

  Resource operator+(const Resource& other) const;
  Resource& operator+=(const Resource& other);

  friend std::ostream& operator<<(std::ostream& os, const Resource& res);
};

// PerfModel class definition
class PerfModel {
 public:
  double flop_s_gemm;
  double flop_s_attn;
  double memory_bw_byte_s_gemm;
  double memory_bw_byte_s_attn;
  double overhead_prefill_ms;
  double overhead_decode_ms;
  std::optional<double> network_bw_byte_s;

  PerfModel(double flop_s_gemm,
            double flop_s_attn,
            double memory_bw_byte_s_gemm,
            double memory_bw_byte_s_attn,
            double overhead_prefill_ms = 0.0,
            double overhead_decode_ms = 0.0,
            std::optional<double> network_bw_byte_s = std::nullopt);

  void apply_latency_roofline(Resource& resource,
                              OpType op_type = OpType::PREFILL_GEMM) const;
};

// LinearOpFlops class definition
class LinearOpFlops {
 public:
  int64_t input_dim;
  int64_t output_dim;
  int64_t dtype_byte;

  LinearOpFlops(int64_t input_dim, int64_t output_dim, int64_t dtype_byte = 2);
  Resource operator()(int64_t batch_size) const;
  int64_t size() const;
  int64_t _saturation_bs() const;
};

// MHA_OpFlops class definition
class MHA_OpFlops {
 public:
  int64_t hidden_dim;
  int64_t dtype_byte;
  int64_t q_per_kv_head;

  MHA_OpFlops(int64_t hidden_dim,
              int64_t dtype_byte = 2,
              int64_t q_per_kv_head = 1);
  Resource operator()(int64_t q_len, int64_t kv_len, bool is_decode) const;
};

// AllReduceOpFlops class definition
class AllReduceOpFlops {
 public:
  int64_t dtype_byte;
  int64_t tp;

  AllReduceOpFlops(int64_t dtype_byte = 2, int64_t tp = 1);
  Resource operator()(int64_t data_size) const;
};

// MLP_Flops class definition
class MLP_Flops {
 public:
  int64_t hidden_dim;
  int64_t intermediate_dim;
  int64_t dtype_byte;
  int64_t tp;

  LinearOpFlops up;
  LinearOpFlops gate;
  LinearOpFlops down;
  AllReduceOpFlops all_reduce;

  MLP_Flops(int64_t hidden_dim,
            int64_t intermediate_dim,
            int64_t dtype_byte = 2,
            int64_t tp = 1);
  Resource operator()(const std::vector<int>& req_lens, bool is_decode) const;
  int64_t size() const;
};

// AttentionFlops class definition
class AttentionFlops {
 public:
  int64_t hidden_dim;
  int64_t dtype_byte;
  int64_t q_per_kv_head;
  int64_t tp;

  LinearOpFlops qkv_proj;
  MHA_OpFlops mha;
  LinearOpFlops o_proj;
  AllReduceOpFlops all_reduce;

  AttentionFlops(int64_t hidden_dim,
                 int64_t dtype_byte = 2,
                 int64_t q_per_kv_head = 1,
                 int64_t tp = 1);
  Resource operator()(const std::vector<int>& q_lens,
                      const std::vector<int>& kv_lens,
                      bool is_decode) const;
  int64_t size() const;
};

// TransformerLayerFlops class definition
class TransformerLayerFlops {
 public:
  int64_t hidden_dim;
  int64_t intermediate_dim;
  int64_t dtype_byte;
  int64_t tp;

  AttentionFlops attention;
  MLP_Flops mlp;

  TransformerLayerFlops(int64_t hidden_dim,
                        int64_t intermediate_dim,
                        int64_t q_per_kv_head = 1,
                        int64_t dtype_byte = 2,
                        int64_t tp = 1);
  Resource operator()(bool is_decode, const std::vector<int>& req_lens) const;
  int64_t size() const;
};

// LLMFlops class definition
class LLMFlops {
 public:
  int64_t num_layers;
  int64_t vocab_size;
  int64_t hidden_dim;
  int64_t intermediate_dim;
  int64_t q_per_kv_head;
  int64_t dtype_byte;
  int64_t tp;

  LinearOpFlops embedding;
  std::vector<TransformerLayerFlops> transformer_layers;
  LinearOpFlops lm_head;

  LLMFlops(int64_t num_layers,
           int64_t vocab_size,
           int64_t hidden_dim,
           int64_t intermediate_dim,
           int64_t q_per_kv_head = 1,
           int64_t dtype_byte = 2,
           int64_t tp = 1);

  Resource operator()(bool is_decode, const std::vector<int>& req_lens) const;
  Resource prefill(const std::vector<int>& req_lens) const;
  Resource decode(const std::vector<int>& req_lens) const;
  int64_t size() const;
  int64_t kv_size(const std::vector<int>& req_lens) const;
  int64_t linear_saturation_bs() const;
  int64_t decode_preferred_req_len(const std::vector<int>& current_batch,
                                   int64_t target_bs,
                                   double target_slo,
                                   int64_t remain_vram_req_total_len) const;
};
}  // namespace perf_model
}  // namespace xllm