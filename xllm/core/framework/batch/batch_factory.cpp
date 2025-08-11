#include "batch_factory.h"

namespace xllm {

std::vector<Batch> BatchFactory::create_batches(
    const std::vector<Sequence*>& running_sequences,
    const std::vector<size_t>& running_sequences_budgets,
    const std::vector<Sequence*>& copy_out_sequences) {
  size_t num_prompt_tokens = 0;
  size_t num_generated_tokens = 0;
  std::vector<Batch> batches(dp_size_);
  for (size_t i = 0; i < running_sequences.size(); ++i) {
    auto* sequence = running_sequences[i];
    const size_t token_budget = running_sequences_budgets[i];

    const size_t remaining_prompt_tokens =
        sequence->num_prompt_tokens() >
                sequence->kv_state().kv_cache_tokens_num()
            ? sequence->num_prompt_tokens() -
                  sequence->kv_state().kv_cache_tokens_num()
            : 0;
    const size_t prompt_tokens =
        std::min(remaining_prompt_tokens, token_budget);
    const size_t generated_tokens = token_budget - prompt_tokens;
    num_prompt_tokens += prompt_tokens;
    num_generated_tokens += generated_tokens;

    // if dp enabled, each sequence is required to
    // dispatch to the same rank in the whole lifetime
    if (sequence->dp_rank() >= 0) {
      batches[sequence->dp_rank()].add(sequence, token_budget);
    } else {
      batches[i % dp_size_].add(sequence, token_budget);
      sequence->set_dp_rank(i % dp_size_);
    }
  }

  for (auto seq : copy_out_sequences) {
    batches[seq->dp_rank()].add_copy_out(seq);
  }

  COUNTER_ADD(num_processing_tokens_total_prompt, num_prompt_tokens);
  COUNTER_ADD(num_processing_tokens_total_generated, num_generated_tokens);

  HISTOGRAM_OBSERVE(num_prompt_tokens_per_request, num_prompt_tokens);
  HISTOGRAM_OBSERVE(num_generated_tokens_per_request, num_generated_tokens);

  return batches;
}

}  // namespace xllm
