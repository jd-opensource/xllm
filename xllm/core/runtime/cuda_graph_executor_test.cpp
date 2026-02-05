/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/batch/batch_forward_type.h"
#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/layers/cuda/attention.h"
#include "core/layers/cuda/flashinfer_workspace.h"
#include "core/runtime/cuda_graph_executor_impl.h"
#include "core/runtime/options.h"
#include "layers/common/attention_metadata_builder.h"

namespace xllm {
namespace {

class CudaGraphExecutorTestEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    google::InitGoogleLogging("cuda_graph_executor_test");
    google::SetStderrLogging(google::INFO);

    // Keep the test minimal and deterministic.
    FLAGS_block_size = 1;
    FLAGS_max_tokens_per_batch = 8;
    FLAGS_enable_graph_no_padding = true;

    // Seed all RNGs once per test environment.
    torch::manual_seed(0);
  }

  void TearDown() override { google::ShutdownGoogleLogging(); }
};

::testing::Environment* const test_env =
    ::testing::AddGlobalTestEnvironment(new CudaGraphExecutorTestEnvironment);

// A minimal CausalLM whose forward contains a FlashInfer batch-decode attention
// call. This model is used to compare eager vs CUDA-graph execution under the
// same inputs.
class FakeAttnCausalLM final : public CausalLM {
 public:
  FakeAttnCausalLM(const ModelArgs& args, const torch::Device& device)
      : args_(args),
        device_(device),
        options_(
            torch::TensorOptions().dtype(torch::kBFloat16).device(device)) {
    const int64_t vocab_size = std::max<int64_t>(args_.vocab_size(), 1024);
    embedding_ =
        register_module("embedding",
                        torch::nn::Embedding(torch::nn::EmbeddingOptions(
                            vocab_size, args_.hidden_size())));
    q_proj_ = register_module("q_proj",
                              torch::nn::Linear(torch::nn::LinearOptions(
                                  args_.hidden_size(), args_.hidden_size())));
    k_proj_ = register_module("k_proj",
                              torch::nn::Linear(torch::nn::LinearOptions(
                                  args_.hidden_size(), args_.hidden_size())));
    v_proj_ = register_module("v_proj",
                              torch::nn::Linear(torch::nn::LinearOptions(
                                  args_.hidden_size(), args_.hidden_size())));

    // Move modules to target device before initializing weights.
    this->to(device_);

    // Initialize projections with random bf16 weights/bias.
    q_proj_->to(torch::kBFloat16);
    k_proj_->to(torch::kBFloat16);
    v_proj_->to(torch::kBFloat16);
    q_proj_->weight.data().normal_();
    k_proj_->weight.data().normal_();
    v_proj_->weight.data().normal_();
    if (q_proj_->bias.defined()) {
      q_proj_->bias.data().normal_();
    }
    if (k_proj_->bias.defined()) {
      k_proj_->bias.data().normal_();
    }
    if (v_proj_->bias.defined()) {
      v_proj_->bias.data().normal_();
    }

    const int n_heads = args_.n_heads();
    const int head_dim = args_.head_dim();
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    attn_ = std::make_unique<layer::AttentionImpl>(n_heads,
                                                   head_dim,
                                                   scale,
                                                   /*num_kv_heads=*/n_heads,
                                                   /*sliding_window=*/-1);
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& params) override {
    (void)positions;
    CHECK(!kv_caches.empty());

    // Use the executor-provided metadata when available (CUDA graph mode).
    layer::AttentionMetadata attn_meta;
    if (params.attn_metadata) {
      attn_meta = *params.attn_metadata;
    } else {
      attn_meta = layer::AttentionMetadataBuilder::build(params);
    }
    CHECK(attn_meta.plan_info) << "attn_meta.plan_info must be set";
    attn_meta.plan_info->layer_id = 0;

    // Query/Key/Value are identical embeddings. For kv_len=1, attention output
    // equals value exactly.
    auto token_ids = tokens.to(torch::kInt64);
    auto x = embedding_->forward(token_ids).to(torch::kBFloat16);
    torch::Tensor q = q_proj_->forward(x);
    torch::Tensor k = k_proj_->forward(x);
    torch::Tensor v = v_proj_->forward(x);

    auto [out, out_lse] = attn_->forward(attn_meta, q, k, v, kv_caches[0]);
    (void)out_lse;
    return ModelOutput(out);
  }

  const torch::TensorOptions& options() const override { return options_; }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) override {
    (void)selected_idxes;
    const int64_t vocab_size = std::max<int64_t>(args_.vocab_size(), 1024);
    return torch::zeros({hidden_states.size(0), vocab_size},
                        torch::dtype(torch::kFloat32).device(device_));
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    (void)loader;
  }
  torch::Device device() const override { return device_; }
  void prepare_expert_weight(int32_t, const std::vector<int32_t>&) override {}
  void update_expert_weight(int32_t) override {}

 private:
  ModelArgs args_;
  torch::Device device_;
  torch::TensorOptions options_;
  torch::nn::Embedding embedding_{nullptr};
  torch::nn::Linear q_proj_{nullptr};
  torch::nn::Linear k_proj_{nullptr};
  torch::nn::Linear v_proj_{nullptr};
  std::unique_ptr<layer::AttentionImpl> attn_;
};

ModelInputParams MakeDecodeParams(const torch::Device& device) {
  ModelInputParams p;
  p.batch_forward_type = BatchForwardType::DECODE;
  p.num_sequences = 1;
  p.kv_max_seq_len = 10;
  p.q_max_seq_len = 1;
  p.enable_cuda_graph = false;  // executor will set metadata->enable_cuda_graph

  auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(device);
  // cumulative lengths (cu_seq_lens)
  p.q_seq_lens = torch::tensor({0, 1}, iopt);
  p.kv_seq_lens = torch::tensor({0, 10}, iopt);
  p.q_cu_seq_lens = p.q_seq_lens;

  // slot mapping for the single token -> last slot in the 10-length kv cache
  p.new_cache_slots = torch::tensor({9}, iopt);
  // block table is required by AttentionMetadataBuilder for decode path
  p.block_tables = torch::tensor({{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}, iopt);

  // FlashInfer paged-kv metadata: one page (block) per sequence.
  p.paged_kv_indptr = torch::tensor({0, 10}, iopt);
  p.paged_kv_indices = torch::tensor({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, iopt);
  p.paged_kv_last_page_len = torch::tensor({1}, iopt);

  return p;
}

std::vector<KVCache> MakeKvCaches(const torch::Device& device,
                                  int64_t num_pages,
                                  int64_t page_size,
                                  int64_t num_kv_heads,
                                  int64_t head_dim) {
  auto opt = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  auto k_cache =
      torch::randn({num_pages, page_size, num_kv_heads, head_dim}, opt);
  auto v_cache =
      torch::randn({num_pages, page_size, num_kv_heads, head_dim}, opt);
  return {KVCache(k_cache, v_cache)};
}

TEST(CudaGraphExecutorTest, BatchDecodeCaptureAndReplay) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available at runtime.";
  }

  const torch::Device device(torch::kCUDA, 0);
  xllm::layer::flashinfer::FlashinferWorkspace::get_instance().initialize(
      device);

  ModelArgs args;
  args.model_type("fake_attn");
  args.dtype("bfloat16");
  args.hidden_size(64);
  args.max_position_embeddings(16);
  args.vocab_size(2048);
  args.n_layers(1);
  args.n_heads(1);
  args.head_dim(64);
  args.n_kv_heads(1);

  runtime::Options options;
  options.block_size(1);
  options.max_seqs_per_batch(1);

  auto model = std::make_unique<FakeAttnCausalLM>(args, device);
  auto graph_exec = std::make_unique<cuda::CudaGraphExecutorImpl>(
      model.get(), args, device, options);

  auto tokens = torch::tensor(
      {1}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto positions = torch::tensor(
      {0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto params = MakeDecodeParams(device);

  // Eager baseline and CUDA graph runs share the same KVCache.
  auto kv = MakeKvCaches(device,
                         /*num_pages=*/16,
                         /*page_size=*/1,
                         /*num_kv_heads=*/1,
                         /*head_dim=*/64);
  auto eager_out =
      model->forward(tokens, positions, kv, params).hidden_states.clone();
  torch::cuda::synchronize();
  // LOG(INFO) << "eager_out: " << eager_out;
  // Graph capture (first run) + replay (second run).
  auto out1 = graph_exec->run(tokens, positions, kv, params).hidden_states;
  torch::cuda::synchronize();
  // LOG(INFO) << "out1: " << out1;
  EXPECT_TRUE(torch::allclose(out1, eager_out, /*rtol=*/1e-3, /*atol=*/1e-3))
      << "graph capture output should match eager output";

  auto out2 = graph_exec->run(tokens, positions, kv, params).hidden_states;
  torch::cuda::synchronize();
  EXPECT_TRUE(torch::allclose(out2, eager_out, /*rtol=*/1e-3, /*atol=*/1e-3))
      << "graph replay output should match eager output";
}

}  // namespace
}  // namespace xllm
