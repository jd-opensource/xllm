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

#include "vlm_executor_impl.h"

#include <glog/logging.h>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/multimodal/mm_visitor.h"
#include "platform/device.h"

namespace xllm {

VlmExecutorImpl::VlmExecutorImpl(CausalLM* model,
                                 const ModelArgs& args,
                                 const torch::Device& device,
                                 const runtime::Options& options)
    : model_(dynamic_cast<CausalVLM*>(model)),
      args_(args),
      device_(device),
      options_(options) {
  if (::xllm::ExecutionConfig::get_instance().enable_graph()) {
    llm_executor_ = ExecutorImplFactory::get_instance().create_executor_impl(
        model, args, device, options, Device::type_str());
  }
}

ForwardInput VlmExecutorImpl::prepare_inputs(Batch& batch) {
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

MMDict VlmExecutorImpl::encode(const ForwardInput& input) {
  return dynamic_cast<CausalVLM*>(model_)->encode(input);
}

ModelOutput VlmExecutorImpl::run(const ForwardInput& input,
                                 std::vector<KVCache>& kv_caches) {
  torch::NoGradGuard no_grad;
  ForwardInput model_input = input;
  auto& mm_data = model_input.multimodal.mm_data;
  EncoderInputGatherVisitor input_gather;
  mm_data.foreach (input_gather);
  CHECK(input_gather.finish(mm_data));
  mm_data.to(device_);

  auto embedding = encode(model_input);
  EncoderOutputScatterVisitor scatter(embedding);
  mm_data.foreach (scatter);
  CHECK(scatter.finish());

  EncoderEmbeddingGatherVisitor gather(device_,
                                       mm_data.type(),
                                       model_input.attention.host.kv_seq_lens,
                                       model_input.attention.host.q_seq_lens);
  mm_data.foreach (gather);
  CHECK(gather.finish(mm_data));

  model_input.embedding.input_embedding =
      model_->get_input_embeddings(model_input.token_ids, model_input);

  if (llm_executor_) {
    return llm_executor_->run(model_input, kv_caches);
  }

  return model_->forward(model_input, kv_caches);
}

}  // namespace xllm
