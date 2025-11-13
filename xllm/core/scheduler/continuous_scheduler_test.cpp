#include "continuous_scheduler.h"

#include <absl/time/clock.h>
#include <gtest/gtest.h>

#include <limits>
#include <optional>

#include "chunked_prefill_scheduler.h"
#include "runtime/engine.h"
#include "util/utils.h"

namespace xllm {

namespace {
class FakeTokenizer : public Tokenizer {
 public:
  bool encode(const std::string_view& text, std::vector<int32_t>* ids) const {
    LOG(FATAL) << "Not implemented";
  }
  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const {
    LOG(FATAL) << "Not implemented";
  }
  std::optional<int32_t> token_to_id(const std::string_view& token) const {
    LOG(FATAL) << "Not implemented";
  }
  std::string id_to_token(int32_t id) const { LOG(FATAL) << "Not implemented"; }
  size_t vocab_size() const { LOG(FATAL) << "Not implemented"; }
  std::unique_ptr<Tokenizer> clone() const {
    return std::make_unique<FakeTokenizer>();
  }
};

class FakeEngine : public Engine {
 public:
  FakeEngine(int32_t num_blocks, int32_t block_size) {
    BlockManagerPool::Options opt;
    opt.num_blocks_ = num_blocks;
    opt.block_size_ = block_size;
    opt.enable_prefix_cache_ = false;  // we dont consider prefix cache here
    fake_tokenizer_ = std::make_unique<FakeTokenizer>();
    fake_block_manager_ = std::make_unique<BlockManagerPool>(opt, 1);
  }
  ForwardOutput step(std::vector<Batch>& batch) {
    LOG(FATAL) << "Not implemented";
  }
  void update_last_step_result(std::vector<Batch>& batch) {
    LOG(FATAL) << "Not implemented";
  }
  const Tokenizer* tokenizer() const { return fake_tokenizer_.get(); }
  BlockManagerPool* block_manager_pool() const {
    return fake_block_manager_.get();
  }
  const ModelArgs& model_args() const { LOG(FATAL) << "Not implemented"; }
  const TokenizerArgs& tokenizer_args() const {
    LOG(FATAL) << "Not implemented";
  }
  std::vector<int64_t> get_active_activation_memory() const {
    LOG(FATAL) << "Not implemented";
  }
  bool init() override { return true; }

 private:
  std::unique_ptr<Tokenizer> fake_tokenizer_;
  std::unique_ptr<BlockManagerPool> fake_block_manager_;
};

ContinuousScheduler::Options create_scheduler_options(
    int32_t max_tokens_per_batch,
    int32_t max_seqs_per_batch,
    int32_t num_speculative_tokens,
    int32_t max_tokens_per_chunk_for_prefill,
    int32_t dp_size,
    const std::string& priority_strategy = "FCFS",
    bool enable_profile_kv_blocks = true,
    bool enable_latency_aware_schedule = false,
    int32_t max_global_ttft_ms = std::numeric_limits<int32_t>::max(),
    int32_t max_global_tpot_ms = std::numeric_limits<int32_t>::max()) {
  ContinuousScheduler::Options opt;
  opt.num_speculative_tokens_ = num_speculative_tokens;
  opt.max_tokens_per_chunk_for_prefill_ = max_tokens_per_chunk_for_prefill;
  opt.max_tokens_per_batch_ = max_tokens_per_batch;
  opt.max_seqs_per_batch_ = max_seqs_per_batch;
  opt.dp_size_ = dp_size;
  opt.priority_strategy_ = priority_strategy;
  opt.enable_profile_kv_blocks_ = enable_profile_kv_blocks;
  opt.enable_latency_aware_schedule_ = enable_latency_aware_schedule;
  opt.max_global_ttft_ms_ = max_global_ttft_ms;
  opt.max_global_tpot_ms_ = max_global_tpot_ms;
  return opt;
}

std::vector<std::shared_ptr<Request>> generate_request(
    const std::vector<int32_t>& prompt_lens,
    const std::vector<int32_t>& max_tokens,
    std::optional<std::vector<bool>> offlines,
    std::optional<std::vector<int32_t>> priorities,
    int32_t max_context_len) {
  std::vector<std::shared_ptr<Request>> requests;
  EXPECT_TRUE(prompt_lens.size() == max_tokens.size());

  size_t batch_size = prompt_lens.size();
  std::vector<bool> offline_vec;
  std::vector<int32_t> priority_vec;
  if (offlines.has_value()) {
    offline_vec = *offlines;
  } else {
    offline_vec = std::vector<bool>(batch_size, false);
  }

  if (priorities.has_value()) {
    priority_vec = priorities.value();
  } else {
    priority_vec = std::vector<int32_t>(
        batch_size, static_cast<int32_t>(RequestPriority::NORMAL));
  }

  for (size_t i = 0; i < batch_size; ++i) {
    std::vector<int32_t> prompt_token_ids;
    prompt_token_ids.resize(prompt_lens[i]);
    RequestSamplingParam sampling_param;
    StoppingChecker stopping_checker;
    stopping_checker.set_max_generated_tokens(max_tokens[i]);
    stopping_checker.set_max_context_len(max_context_len);
    stopping_checker.set_ignore_eos(true);

    RequestState req_state("x",
                           prompt_token_ids,
                           sampling_param,
                           stopping_checker,
                           prompt_lens[i] + 30000,
                           1,
                           1,
                           false,
                           false,
                           false,
                           false,
                           false,
                           nullptr,
                           nullptr);

    auto request = std::make_shared<Request>(
        "1",
        "1",
        "1",
        std::move(req_state),
        "1",
        offline_vec[i],
        0,
        static_cast<RequestPriority>(priority_vec[i]));
    requests.emplace_back(request);
  }

  return requests;
}

// dont not consider speculative decoding.
void update_requests(std::vector<std::shared_ptr<Request>> requests) {
  for (auto req : requests) {
    for (auto& seq : req->sequences()) {
      if (seq->kv_state().kv_cache_tokens_num() == 0) {
        seq->kv_state().incr_kv_cache_tokens_num(seq->num_prompt_tokens());
      } else {
        seq->kv_state().incr_kv_cache_tokens_num(1);
      }
      Token token(1);
      seq->append_token(token);
    }
  }
}

}  // namespace

// TEST-1:
// test preempt
TEST(ContinuousSchedulerTest, OnDecodePreemptOffDecode) {
  // set max free blocks: 9, support 9*32=288 tokens
  // actually only 8 free blocks , because default 1 block is for padding
  int block_num = 9;
  int block_size = 32;
  int max_tokens_per_chunk_for_prefill = 1024;
  // set chunked max_tokens budgets 10000 per step
  ContinuousScheduler::Options opt = create_scheduler_options(
      10000, 256, 0, max_tokens_per_chunk_for_prefill, 1);
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  BlockManagerPool* block_manager_pool = engine->block_manager_pool();
  EXPECT_TRUE(scheduler != nullptr);

  std::vector<std::shared_ptr<Request>> running_requests;

  // 1. schedule two new online prefill requests
  auto requests = generate_request({127, 127},
                                   {10, 10},
                                   std::vector<bool>{true, false},
                                   std::vector<int32_t>{2, 2},
                                   30000);
  running_requests = requests;
  for (auto req : requests) {
    scheduler->add_request(req);
  }
  auto batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 2);
  update_requests(running_requests);

  batch = scheduler->prepare_batch_test();

  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 2);
  update_requests(running_requests);

  int free_blocks_before_preempt =
      util::max(block_manager_pool->num_free_blocks());
  batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 1);
  int free_blocks_after_preempt =
      util::max(block_manager_pool->num_free_blocks());
  EXPECT_TRUE(free_blocks_after_preempt > free_blocks_before_preempt);

  // check the running request is online request
  EXPECT_TRUE(scheduler->get_running_requests().size() == 1);
  EXPECT_TRUE(scheduler->get_running_requests()[0]->offline() == false);
  EXPECT_TRUE(scheduler->get_waiting_requests_num() == 1);
}

// TEST-2:
// test preempt
TEST(ContinuousSchedulerTest, OnPrefillPreemptOffDecode) {
  // set max free blocks: 9, support 9*32=288 tokens
  // actually only 8 free blocks , because default 1 block is for padding
  int block_num = 9;
  int block_size = 32;
  int max_tokens_per_chunk_for_prefill = 1024;
  // set chunked max_tokens budgets 10000 per step
  ContinuousScheduler::Options opt = create_scheduler_options(
      10000, 256, 0, max_tokens_per_chunk_for_prefill, 1);
  FLAGS_prefill_scheduling_memory_usage_threshold = 2;  // release threshold

  {
    // 1. two offline decode requests then one online prefill request preempt
    // them
    auto engine = std::make_unique<FakeEngine>(block_num, block_size);
    auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
    BlockManagerPool* block_manager_pool = engine->block_manager_pool();
    EXPECT_TRUE(scheduler != nullptr);

    std::vector<std::shared_ptr<Request>> running_requests;

    auto requests = generate_request({100, 100},
                                     {10, 10},
                                     std::vector<bool>{true, true},
                                     std::vector<int32_t>{2, 2},
                                     30000);
    running_requests = requests;
    for (auto req : requests) {
      scheduler->add_request(req);
    }
    auto batch = scheduler->prepare_batch_test();
    EXPECT_TRUE(batch.size() == 1);
    EXPECT_TRUE(batch[0].size() == 2);
    EXPECT_TRUE(util::max(block_manager_pool->num_free_blocks()) == 0);
    update_requests(running_requests);

    batch = scheduler->prepare_batch_test();
    EXPECT_TRUE(batch.size() == 1);
    EXPECT_TRUE(batch[0].size() == 2);
    EXPECT_TRUE(util::max(block_manager_pool->num_free_blocks()) == 0);
    update_requests(running_requests);

    auto new_requests = generate_request({80},
                                         {10},
                                         std::vector<bool>{false},
                                         std::vector<int32_t>{2},
                                         30000);  // use 3 blocks
    scheduler->add_request(new_requests[0]);
    batch = scheduler->prepare_batch_test();
    EXPECT_TRUE(batch.size() == 1);
    EXPECT_TRUE(batch[0].size() == 1);

    // online prefill request preempt offline decode request
    EXPECT_TRUE(scheduler->get_running_requests().size() == 1);
    EXPECT_TRUE(scheduler->get_running_requests()[0]->offline() == false);
    EXPECT_TRUE(scheduler->get_waiting_requests_num() == 1);

    // offline is evicted
    EXPECT_TRUE(util::max(block_manager_pool->num_free_blocks()) == 1);
  }

  // 2. another case: longer online prefill request arrives, but can not evict
  // offline because evicting offline is not enough
  {
    auto engine = std::make_unique<FakeEngine>(block_num, block_size);
    auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
    BlockManagerPool* block_manager_pool = engine->block_manager_pool();
    EXPECT_TRUE(scheduler != nullptr);

    std::vector<std::shared_ptr<Request>> running_requests;
    // one online, one offline
    auto requests = generate_request({100, 100},
                                     {10, 10},
                                     std::vector<bool>{true, false},
                                     std::vector<int32_t>{2, 2},
                                     30000);
    running_requests = requests;
    for (auto req : requests) {
      scheduler->add_request(req);
    }
    auto batch = scheduler->prepare_batch_test();
    EXPECT_TRUE(batch.size() == 1);
    EXPECT_TRUE(batch[0].size() == 2);
    EXPECT_TRUE(util::max(block_manager_pool->num_free_blocks()) == 0);
    update_requests(running_requests);

    auto new_requests = generate_request(
        {200}, {10}, std::vector<bool>{false}, std::vector<int32_t>{2}, 30000);
    scheduler->add_request(new_requests[0]);
    batch = scheduler->prepare_batch_test();
    // online is still waiting
    EXPECT_TRUE(batch.size() == 1);
    EXPECT_TRUE(batch[0].size() == 2);
    EXPECT_TRUE(scheduler->get_waiting_requests().size() == 1);
    EXPECT_TRUE(scheduler->get_waiting_requests()[0].get() ==
                new_requests[0].get());
  }
}

// TEST-3:
// test priority schedule
TEST(ContinuousSchedulerTest, PrioritySchedule) {
  // set max free blocks: 12
  // actually only 11 free blocks , because default 1 block is for padding
  int block_num = 12;
  int block_size = 32;
  int max_tokens_per_chunk_for_prefill = 1024;
  // set chunked max_tokens budgets 10000 per step
  ContinuousScheduler::Options opt = create_scheduler_options(
      10000, 256, 0, max_tokens_per_chunk_for_prefill, 1, "priority");
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  EXPECT_TRUE(scheduler != nullptr);

  std::vector<std::shared_ptr<Request>> running_requests;

  // 1: HIGH, 2: NORMAL, 3: LOW
  auto requests = generate_request({128, 128, 128},
                                   {10, 10, 10},
                                   std::vector<bool>{false, false, false},
                                   std::vector<int32_t>{3, 3, 2},
                                   30000);
  for (auto req : requests) {
    scheduler->add_request(req);
  }
  auto batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 2);
  EXPECT_TRUE(scheduler->get_running_requests().size() == 2);
  EXPECT_TRUE(scheduler->get_running_requests()[0]->priority() ==
              RequestPriority::NORMAL /*NORMAL*/);
  EXPECT_TRUE(scheduler->get_running_requests()[1]->priority() ==
              RequestPriority::LOW /*LOW*/);
  running_requests = scheduler->get_running_requests();
  update_requests(running_requests);

  // new HIGH priority request arrives, its prefill starts
  auto new_requests = generate_request({32},
                                       {10},
                                       std::vector<bool>{false},
                                       std::vector<int32_t>{1},
                                       30000);  // use 1 blocks
  scheduler->add_request(new_requests[0]);
  batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 1);
  EXPECT_TRUE(scheduler->get_running_requests().size() == 1);
  update_requests(new_requests);

  // only HIGH and NORMAL requests decode
  batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 2);
  EXPECT_TRUE(scheduler->get_running_requests().size() == 2);
  EXPECT_TRUE(scheduler->get_running_requests()[0]->priority() ==
              RequestPriority::HIGH /*HIGH*/);
  EXPECT_TRUE(scheduler->get_running_requests()[1]->priority() ==
              RequestPriority::NORMAL /*NORMAL*/);
}

// TEST-4:
// test latency budget
TEST(ContinuousSchedulerTest, LatencySchedule) {
  // block is enough
  int block_num = 12;
  int block_size = 32;
  int max_tokens_per_chunk_for_prefill = 1024;
  // set chunked max_tokens budgets 10000 per step
  ContinuousScheduler::Options opt =
      create_scheduler_options(10000,
                               256,
                               0,
                               max_tokens_per_chunk_for_prefill,
                               1,
                               "FCFS",
                               false,
                               true,
                               350,
                               25);
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ContinuousScheduler>(engine.get(), opt);
  EXPECT_TRUE(scheduler != nullptr);

  // mannuly created profile data for y=0.5x^2+10x
  std::vector<std::pair<int32_t, double>> created_profile_data = {
      {2, 22}, {4, 48}, {6, 78}, {8, 112}};
  auto profile_manager = scheduler->get_profile_manager();
  // fit y=0.5x^2+10x
  profile_manager->train_prefill_time_predictor(created_profile_data);

  auto requests = generate_request(
      {10, 10, 10}, {10, 10, 10}, std::nullopt, std::nullopt, 30000);
  // check if time equation fits well
  EXPECT_EQ(static_cast<int32_t>(profile_manager->predict_step_time(
                requests[0]->sequences()[0].get(), true, true)),
            150);
  EXPECT_EQ(static_cast<int32_t>(
                profile_manager->predict_step_time(2, 0, true, true)),
            22);

  std::vector<std::shared_ptr<Request>> running_requests;

  // 1. two requests enter prefill
  for (auto req : requests) {
    scheduler->add_request(req);
  }
  auto batch = scheduler->prepare_batch_test();

  EXPECT_EQ(batch.size(), 1);
  // 2*150 < ttft_slo=350 < 3 * 150, only two requests enter prefill
  EXPECT_EQ(batch[0].size(), 2);
  EXPECT_EQ(scheduler->get_running_requests().size(), 2);
  running_requests = scheduler->get_running_requests();
  update_requests(running_requests);

  // 2. one request enter prefill
  batch = scheduler->prepare_batch_test();
  EXPECT_EQ(batch.size(), 1);
  EXPECT_EQ(batch[0].size(), 1);
  EXPECT_EQ(scheduler->get_running_requests().size(), 1);
  running_requests = scheduler->get_running_requests();
  update_requests(running_requests);

  // 3. two requests start decode
  // batch = scheduler->prepare_batch_test();
  // EXPECT_TRUE(batch.size() == 1);
  // // 2*10 < tpot_slo=25 < 3 * 10, only two requests enter decode
  // EXPECT_TRUE(batch[0].size() == 2);
  // EXPECT_TRUE(scheduler->get_running_requests().size() == 2);
}

}  // namespace xllm