#pragma once

#include <folly/futures/Future.h>
#include <torch/torch.h>

#include "executor.h"
#include "forward_params.h"
#include "framework/model/causal_lm.h"
#include "framework/model/embedding_lm.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "options.h"
#include "runtime/worker_impl.h"

namespace xllm {

class LLMWorkerImpl : public WorkerImpl {
 public:
  LLMWorkerImpl(const ParallelArgs& parallel_args,
                const torch::Device& device,
                const runtime::Options& options);

  ~LLMWorkerImpl() override = default;

  // initialize model, cache manager. blocking call
  bool init_model(torch::ScalarType dtype,
                  const ModelArgs& args,
                  const QuantArgs& quant_args) override;

  std::optional<ForwardOutput> step(const ForwardInput& inputs) override;

#if defined(USE_NPU)
  hf::LlmHead get_lm_head() { return model_->get_lm_head(); };

  void set_lm_head(hf::LlmHead& head) { model_->set_lm_head(head); };

  hf::AtbWordEmbedding get_word_embedding() {
    return model_->get_word_embedding();
  };

  void set_word_embedding(hf::AtbWordEmbedding& embedding) {
    model_->set_word_embedding(embedding);
  };
#elif defined(USE_MLU)
// TODO(mlu): implement mlu get/set lm head and word embedding
#endif
};

}  // namespace xllm
