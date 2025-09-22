#include "perf_model.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>

namespace xllm {
namespace perf_model {

// Global variables
std::shared_ptr<PerfModel> _CURRENT_PERF_MODEL = nullptr;

// Global function implementations
void set_perf_model(std::shared_ptr<PerfModel> model) {
  _CURRENT_PERF_MODEL = model;
}

std::shared_ptr<PerfModel> get_perf_model() { return _CURRENT_PERF_MODEL; }

// Resource class implementation
Resource::Resource(int64_t flops,
                   int64_t memory,
                   int64_t network,
                   double latency)
    : flops(flops), memory(memory), network(network), latency(latency) {}

Resource Resource::operator+(const Resource& other) const {
  return Resource(flops + other.flops,
                  memory + other.memory,
                  network + other.network,
                  latency + other.latency);
}

Resource& Resource::operator+=(const Resource& other) {
  flops += other.flops;
  memory += other.memory;
  network += other.network;
  latency += other.latency;
  return *this;
}

std::ostream& operator<<(std::ostream& os, const Resource& res) {
  os << "[" << res.flops / 1e9 << " GFLOPS"
     << ", " << res.memory / 1e9 << " GB ("
     << (res.memory > 0 ? static_cast<double>(res.flops) / res.memory : 0.0)
     << ")"
     << ", " << res.network / 1e9 << " GB"
     << ", " << res.latency * 1e3 << " ms]";
  return os;
}

// PerfModel class implementation
PerfModel::PerfModel(double flop_s_gemm,
                     double flop_s_attn,
                     double memory_bw_byte_s_gemm,
                     double memory_bw_byte_s_attn,
                     double overhead_prefill_ms,
                     double overhead_decode_ms,
                     std::optional<double> network_bw_byte_s)
    : flop_s_gemm(flop_s_gemm),
      flop_s_attn(flop_s_attn),
      memory_bw_byte_s_gemm(memory_bw_byte_s_gemm),
      memory_bw_byte_s_attn(memory_bw_byte_s_attn),
      overhead_prefill_ms(overhead_prefill_ms),
      overhead_decode_ms(overhead_decode_ms),
      network_bw_byte_s(network_bw_byte_s) {}

void PerfModel::apply_latency_roofline(Resource& resource,
                                       OpType op_type) const {
  auto current_model = get_perf_model();
  if (!current_model) {
    return;
  }

  assert(current_model->flop_s_gemm > 0 && "flop_s_gemm must be set");
  assert(current_model->flop_s_attn > 0 && "flop_s_attn must be set");
  assert(current_model->memory_bw_byte_s_gemm > 0 &&
         "memory_bw_byte_s_gemm must be set");
  assert(current_model->memory_bw_byte_s_attn > 0 &&
         "memory_bw_byte_s_attn must be set");

  double op_flop_s =
      (op_type == OpType::PREFILL_GEMM || op_type == OpType::DECODE_GEMM)
          ? current_model->flop_s_gemm
          : current_model->flop_s_attn;

  double op_memory_bw_byte_s =
      (op_type == OpType::PREFILL_GEMM || op_type == OpType::DECODE_GEMM)
          ? current_model->memory_bw_byte_s_gemm
          : current_model->memory_bw_byte_s_attn;

  double latency_flops = resource.flops / op_flop_s;
  double memory_latency = resource.memory / op_memory_bw_byte_s;
  double network_latency =
      current_model->network_bw_byte_s
          ? resource.network / *current_model->network_bw_byte_s
          : 0.0;

  resource.latency = std::max({latency_flops, memory_latency, network_latency});
}

// LinearOpFlops class implementation
LinearOpFlops::LinearOpFlops(int64_t input_dim,
                             int64_t output_dim,
                             int64_t dtype_byte)
    : input_dim(input_dim), output_dim(output_dim), dtype_byte(dtype_byte) {}

Resource LinearOpFlops::operator()(int64_t batch_size) const {
  int64_t flops = 2 * input_dim * output_dim * batch_size;
  int64_t memory =
      dtype_byte * (batch_size * input_dim + input_dim * output_dim +
                    batch_size * output_dim);

  Resource r(flops, memory);
  auto current_model = get_perf_model();
  if (current_model) {
    OpType op_type = OpType::PREFILL_GEMM;
    current_model->apply_latency_roofline(r, op_type);
  }
  return r;
}

int64_t LinearOpFlops::size() const { return input_dim * output_dim; }

int64_t LinearOpFlops::_saturation_bs() const {
  auto model = get_perf_model();
  if (!model) {
    throw std::runtime_error("Performance model is not set");
  }

  int64_t i = 0;
  double target_ratio = model->flop_s_gemm / model->memory_bw_byte_s_gemm;
  while (i < 10000) {
    i++;
    Resource r = this->operator()(i);
    double cur_ratio = r.flops / static_cast<double>(r.memory);
    if (cur_ratio >= target_ratio) {
      break;
    }
  }
  return i;
}

// MHA_OpFlops class implementation
MHA_OpFlops::MHA_OpFlops(int64_t hidden_dim,
                         int64_t dtype_byte,
                         int64_t q_per_kv_head)
    : hidden_dim(hidden_dim),
      dtype_byte(dtype_byte),
      q_per_kv_head(q_per_kv_head) {}

Resource MHA_OpFlops::operator()(int64_t q_len,
                                 int64_t kv_len,
                                 bool is_decode) const {
  int64_t flops = 2 * hidden_dim * (q_len * kv_len + kv_len * q_len);
  int64_t memory =
      (q_len + 2 * kv_len / q_per_kv_head + q_len) * hidden_dim * dtype_byte;

  Resource r(flops, memory);
  auto current_model = get_perf_model();
  if (current_model) {
    OpType op_type =
        is_decode ? OpType::DECODE_ATTENTION : OpType::PREFILL_ATTENTION;
    current_model->apply_latency_roofline(r, op_type);
  }
  return r;
}

// AllReduceOpFlops class implementation
AllReduceOpFlops::AllReduceOpFlops(int64_t dtype_byte, int64_t tp)
    : dtype_byte(dtype_byte), tp(tp) {}

Resource AllReduceOpFlops::operator()(int64_t data_size) const {
  int64_t comm = dtype_byte * data_size * (tp - 1) / tp;
  Resource r(0, 0, comm);
  auto current_model = get_perf_model();
  if (current_model) {
    current_model->apply_latency_roofline(r);
  }
  return r;
}

// MLP_Flops class implementation
MLP_Flops::MLP_Flops(int64_t hidden_dim,
                     int64_t intermediate_dim,
                     int64_t dtype_byte,
                     int64_t tp)
    : hidden_dim(hidden_dim),
      intermediate_dim(intermediate_dim),
      dtype_byte(dtype_byte),
      tp(tp),
      up(hidden_dim, intermediate_dim / tp, dtype_byte),
      gate(hidden_dim, intermediate_dim / tp, dtype_byte),
      down(intermediate_dim / tp, hidden_dim, dtype_byte),
      all_reduce(dtype_byte, tp) {}

Resource MLP_Flops::operator()(const std::vector<int>& req_lens,
                               bool is_decode) const {
  Resource r;
  int64_t total_len = 0;
  for (int64_t len : req_lens) {
    total_len += len;
  }

  r += up(total_len);
  r += gate(total_len);
  r += down(total_len);
  r += all_reduce(total_len * hidden_dim);

  return r;
}

int64_t MLP_Flops::size() const {
  return up.size() + gate.size() + down.size();
}

// AttentionFlops class implementation
AttentionFlops::AttentionFlops(int64_t hidden_dim,
                               int64_t dtype_byte,
                               int64_t q_per_kv_head,
                               int64_t tp)
    : hidden_dim(hidden_dim),
      dtype_byte(dtype_byte),
      q_per_kv_head(q_per_kv_head),
      tp(tp),
      qkv_proj(hidden_dim,
               (hidden_dim + 2 * hidden_dim / q_per_kv_head) / tp,
               dtype_byte),
      mha(hidden_dim / tp, dtype_byte, q_per_kv_head),
      o_proj(hidden_dim / tp, hidden_dim, dtype_byte),
      all_reduce(dtype_byte, tp) {}

Resource AttentionFlops::operator()(const std::vector<int>& q_lens,
                                    const std::vector<int>& kv_lens,
                                    bool is_decode) const {
  if (q_lens.size() != kv_lens.size()) {
    throw std::runtime_error("q_lens and kv_lens must have the same length");
  }

  Resource r;
  int64_t total_q_len = 0;
  for (int64_t len : q_lens) {
    total_q_len += len;
  }

  r += qkv_proj(total_q_len);

  for (size_t i = 0; i < q_lens.size(); i++) {
    r += mha(q_lens[i], kv_lens[i], is_decode);
  }

  r += o_proj(total_q_len);
  r += all_reduce(total_q_len * hidden_dim);

  return r;
}

int64_t AttentionFlops::size() const { return qkv_proj.size() + o_proj.size(); }

// TransformerLayerFlops class implementation
TransformerLayerFlops::TransformerLayerFlops(int64_t hidden_dim,
                                             int64_t intermediate_dim,
                                             int64_t q_per_kv_head,
                                             int64_t dtype_byte,
                                             int64_t tp)
    : hidden_dim(hidden_dim),
      intermediate_dim(intermediate_dim),
      dtype_byte(dtype_byte),
      tp(tp),
      attention(hidden_dim, dtype_byte, q_per_kv_head, tp),
      mlp(hidden_dim, intermediate_dim, dtype_byte, tp) {}

Resource TransformerLayerFlops::operator()(
    bool is_decode,
    const std::vector<int>& req_lens) const {
  Resource r;

  if (!is_decode) {
    r += attention(req_lens, req_lens, is_decode);
    r += mlp(req_lens, is_decode);
  } else {
    std::vector<int> new_tokens(req_lens.size(), 1);
    r += attention(new_tokens, req_lens, is_decode);
    r += mlp(new_tokens, is_decode);
  }

  return r;
}

int64_t TransformerLayerFlops::size() const {
  return attention.size() + mlp.size();
}

// LLMFlops class implementation
LLMFlops::LLMFlops(int64_t num_layers,
                   int64_t vocab_size,
                   int64_t hidden_dim,
                   int64_t intermediate_dim,
                   int64_t q_per_kv_head,
                   int64_t dtype_byte,
                   int64_t tp)
    : num_layers(num_layers),
      vocab_size(vocab_size),
      hidden_dim(hidden_dim),
      intermediate_dim(intermediate_dim),
      q_per_kv_head(q_per_kv_head),
      dtype_byte(dtype_byte),
      tp(tp),
      embedding(vocab_size, hidden_dim, dtype_byte),
      lm_head(hidden_dim, vocab_size, dtype_byte) {
  LOG(INFO) << "Initializing LLMFlops with num_layers=" << num_layers
            << ", vocab_size=" << vocab_size << ", hidden_dim=" << hidden_dim
            << ", intermediate_dim=" << intermediate_dim
            << ", q_per_kv_head=" << q_per_kv_head
            << ", dtype_byte=" << dtype_byte << ", tp=" << tp;

  for (int64_t i = 0; i < num_layers; i++) {
    transformer_layers.emplace_back(
        hidden_dim, intermediate_dim, q_per_kv_head, dtype_byte, tp);
  }
}

Resource LLMFlops::operator()(bool is_decode,
                              const std::vector<int>& req_lens) const {
  Resource r;

  // for (const auto& layer : transformer_layers) {
  //     r += layer(is_decode, req_lens);
  // }

  Resource layer_r = transformer_layers[0](is_decode, req_lens);
  for (const auto& layer : transformer_layers) {
    r += layer_r;
  }

  r += lm_head(req_lens.size());  // only last token is needed

  return r;
}

Resource LLMFlops::prefill(const std::vector<int>& req_lens) const {
  return this->operator()(false, req_lens);
}

Resource LLMFlops::decode(const std::vector<int>& req_lens) const {
  return this->operator()(true, req_lens);
}

int64_t LLMFlops::size() const {
  int64_t total_size = embedding.size() + lm_head.size();
  for (const auto& layer : transformer_layers) {
    total_size += layer.size();
  }
  return total_size;
}

int64_t LLMFlops::kv_size(const std::vector<int>& req_lens) const {
  int64_t total_len = 0;
  for (int64_t len : req_lens) {
    total_len += len;
  }
  return dtype_byte * num_layers * total_len * hidden_dim * 2 / q_per_kv_head;
}

int64_t LLMFlops::linear_saturation_bs() const {
  auto model = get_perf_model();
  if (!model) {
    LOG(FATAL) << "Performance model is not set";
  }

  const TransformerLayerFlops& tl = transformer_layers[0];
  std::vector<LinearOpFlops> l_layers = {tl.mlp.up,
                                         tl.mlp.gate,
                                         tl.mlp.down,
                                         tl.attention.qkv_proj,
                                         tl.attention.o_proj};

  int64_t max_bs = 0;
  for (const auto& l : l_layers) {
    int64_t bs = l._saturation_bs();
    if (bs > max_bs) {
      max_bs = bs;
    }
  }

  return max_bs;
}

int64_t LLMFlops::decode_preferred_req_len(
    const std::vector<int>& current_batch,
    int64_t target_bs,
    double target_slo,
    int64_t remain_vram_req_total_len) const {
  auto model = get_perf_model();
  if (!model) {
    throw std::runtime_error("Performance model is not set");
  }

  int64_t left_bs = std::max(target_bs - current_batch.size(), 1UL);

  // binary search
  int64_t low = 1;
  int64_t high = remain_vram_req_total_len / left_bs + 1;

  while (low < high - low / 100) {
    int64_t mid = (low + high) / 2;
    std::vector<int> target_batch = current_batch;
    target_batch.insert(target_batch.end(), left_bs, mid);

    double latency = this->decode(target_batch).latency;

    if (latency > target_slo) {
      high = mid;
    } else {
      low = mid + 1;
    }
  }

  return low - 1;
}

}  // namespace perf_model
}  // namespace xllm
