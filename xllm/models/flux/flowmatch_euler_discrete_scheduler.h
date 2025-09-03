#pragma once
// lym
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "models/model_registry.h"
#include "processors/input_processor.h"
#include "processors/pywarpper_image_processor.h"
namespace xllm::hf {
struct FlowMatchEulerDiscreteSchedulerOutput {
  torch::Tensor prev_sample;
  explicit FlowMatchEulerDiscreteSchedulerOutput(torch::Tensor sample)
      : prev_sample(std::move(sample)) {}
};
class FlowMatchEulerDiscreteSchedulerImpl : public torch::nn::Module {
 private:
  // 配置参数（原config结构体的成员变量）
  int num_train_timesteps_;
  float shift_;
  bool use_dynamic_shifting_;
  std::optional<float> base_shift_;
  std::optional<float> max_shift_;
  std::optional<int> base_image_seq_len_;
  std::optional<int> max_image_seq_len_;
  bool invert_sigmas_;
  std::optional<float> shift_terminal_;
  bool use_karras_sigmas_;
  bool use_exponential_sigmas_;
  bool use_beta_sigmas_;
  std::string time_shift_type_;
  bool stochastic_sampling_;

  // 状态变量
  torch::Tensor timesteps_;
  torch::Tensor sigmas_;
  float sigma_min_;
  float sigma_max_;
  std::optional<int> step_index_;
  std::optional<int> begin_index_;

  // 私有工具函数
  torch::Tensor convert_to_karras(const torch::Tensor& in_sigmas,
                                  int num_inference_steps) {
    float sigma_min = sigma_min_;
    float sigma_max = sigma_max_;
    if (in_sigmas.numel() > 0) {
      sigma_min = in_sigmas[-1].item<float>();
      sigma_max = in_sigmas[0].item<float>();
    }

    const float rho = 7.0f;
    std::vector<float> ramp(num_inference_steps);
    for (int i = 0; i < num_inference_steps; ++i) {
      ramp[i] = static_cast<float>(i) / (num_inference_steps - 1);
    }
    torch::Tensor ramp_tensor =
        torch::from_blob(ramp.data(), {num_inference_steps}, torch::kFloat32);

    float min_inv_rho = std::pow(sigma_min, 1.0f / rho);
    float max_inv_rho = std::pow(sigma_max, 1.0f / rho);
    return torch::pow(max_inv_rho + ramp_tensor * (min_inv_rho - max_inv_rho),
                      rho);
  }

  torch::Tensor convert_to_exponential(const torch::Tensor& in_sigmas,
                                       int num_inference_steps) {
    float sigma_min = sigma_min_;
    float sigma_max = sigma_max_;
    if (in_sigmas.numel() > 0) {
      sigma_min = in_sigmas[-1].item<float>();
      sigma_max = in_sigmas[0].item<float>();
    }

    std::vector<float> exp_sigmas(num_inference_steps);
    float log_sigma_max = std::log(sigma_max);
    float log_sigma_min = std::log(sigma_min);
    for (int i = 0; i < num_inference_steps; ++i) {
      float t = static_cast<float>(i) / (num_inference_steps - 1);
      exp_sigmas[i] =
          std::exp(log_sigma_max + t * (log_sigma_min - log_sigma_max));
    }
    return torch::from_blob(
               exp_sigmas.data(), {num_inference_steps}, torch::kFloat32)
        .clone();
  }

  torch::Tensor convert_to_beta(const torch::Tensor& in_sigmas,
                                int num_inference_steps,
                                float alpha = 0.6f,
                                float beta = 0.6f) {
    // 注意：实际使用需要链接scipy的beta分布实现，此处仅为框架示意
    throw std::runtime_error(
        "Beta sigmas implementation requires scipy integration");
  }

  torch::Tensor time_shift_exponential(float mu,
                                       float sigma,
                                       const torch::Tensor& t) {
    auto exp_mu = std::exp(mu);
    return exp_mu / (exp_mu + torch::pow(1.0f / t - 1.0f, sigma));
  }

  torch::Tensor time_shift_linear(float mu,
                                  float sigma,
                                  const torch::Tensor& t) {
    return mu / (mu + torch::pow(1.0f / t - 1.0f, sigma));
  }

  void init_step_index(const torch::Tensor& timestep) {
    if (!begin_index_.has_value()) {
      torch::Tensor ts = timestep.to(timesteps_.device());
      step_index_ = index_for_timestep(ts);
    } else {
      step_index_ = begin_index_.value();
    }
  }

  int index_for_timestep(const torch::Tensor& timestep,
                         const torch::Tensor& schedule_timesteps = {}) {
    torch::Tensor sched =
        schedule_timesteps.defined() ? schedule_timesteps : timesteps_;
    torch::Tensor indices = (sched == timestep).nonzero();

    int pos = indices.size(0) > 1 ? 1 : 0;
    return indices.index({pos, 0}).item<int>();
  }

 public:
  int64_t order = 1;  // 默认阶数为1
  ModelArgs args;
  int base_image_seq_len() { return base_image_seq_len_.value(); }
  int max_image_seq_len() { return max_image_seq_len_.value(); }
  float base_shift() { return base_shift_.value(); }
  float max_shift() { return max_shift_.value(); }
  FlowMatchEulerDiscreteSchedulerImpl(const Context& context)
      : args(context.get_model_args()),
        num_train_timesteps_(args.scheduler_num_train_timesteps()),
        shift_(args.scheduler_shift()),
        use_dynamic_shifting_(args.scheduler_use_dynamic_shifting()),
        base_shift_(args.scheduler_base_shift()),
        max_shift_(args.scheduler_max_shift()),
        base_image_seq_len_(args.scheduler_base_image_seq_len()),
        max_image_seq_len_(args.scheduler_max_image_seq_len()),
        invert_sigmas_(false),
        shift_terminal_(std::nullopt),
        use_karras_sigmas_(false),
        use_exponential_sigmas_(false),
        use_beta_sigmas_(false),
        time_shift_type_("exponential"),
        stochastic_sampling_(false) {
    std::vector<float> timesteps_vec(num_train_timesteps_);
    for (int i = 0; i < num_train_timesteps_; ++i) {
      timesteps_vec[i] = num_train_timesteps_ - i;
    }
    torch::Tensor timesteps = torch::from_blob(
        timesteps_vec.data(), {num_train_timesteps_}, torch::kFloat32);

    torch::Tensor sigmas = timesteps / num_train_timesteps_;
    if (!use_dynamic_shifting_) {
      sigmas = shift_ * sigmas / (1 + (shift_ - 1) * sigmas);
    }

    timesteps_ = sigmas * num_train_timesteps_;
    sigmas_ = sigmas.to(torch::kCPU);
    sigma_min_ = sigmas_.index({-1}).item<float>();
    sigma_max_ = sigmas_.index({0}).item<float>();
    step_index_ = std::nullopt;
    begin_index_ = std::nullopt;
  }
  void set_begin_index(int begin_index) { begin_index_ = begin_index; }

  void set_shift(float shift) { shift_ = shift; }
  torch::Tensor scale_noise(
      const torch::Tensor& sample,
      const torch::Tensor& timestep,
      const std::optional<torch::Tensor>& noise = std::nullopt) {
    torch::Tensor sigmas = sigmas_.to(sample.device()).to(sample.dtype());
    torch::Tensor schedule_timesteps = timesteps_.to(sample.device());
    torch::Tensor ts = timestep.to(sample.device());

    std::vector<int> step_indices;
    if (!begin_index_.has_value()) {
      for (int i = 0; i < ts.size(0); ++i) {
        step_indices.push_back(index_for_timestep(ts[i], schedule_timesteps));
      }
    } else if (step_index_.has_value()) {
      step_indices = std::vector<int>(ts.size(0), step_index_.value());
    } else {
      step_indices = std::vector<int>(ts.size(0), begin_index_.value());
    }

    torch::Tensor sigma_indices = torch::tensor(
        step_indices, torch::dtype(torch::kLong).device(sigmas.device()));
    torch::Tensor sigma = sigmas.index_select(0, sigma_indices).flatten();
    while (sigma.dim() < sample.dim()) {
      sigma.unsqueeze_(-1);
    }

    torch::Tensor noise_tensor =
        noise.has_value() ? noise.value() : torch::randn_like(sample);
    return sigma * noise_tensor + (1.0f - sigma) * sample;
  }
  torch::Tensor time_shift(float mu, float sigma, const torch::Tensor& t) {
    if (time_shift_type_ == "exponential") {
      return time_shift_exponential(mu, sigma, t);
    } else {
      return time_shift_linear(mu, sigma, t);
    }
  }

  torch::Tensor stretch_shift_to_terminal(const torch::Tensor& t) {
    if (!shift_terminal_.has_value()) {
      throw std::runtime_error("shift_terminal is not set");
    }
    torch::Tensor one_minus_z = 1.0f - t;
    float scale_factor = one_minus_z.index({-1}).item<float>() /
                         (1.0f - shift_terminal_.value());
    return 1.0f - (one_minus_z / scale_factor);
  }
  void set_timesteps(
      int num_inference_steps,
      const torch::Device& device = torch::kCPU,
      const std::optional<std::vector<float>>& sigmas = std::nullopt,
      const std::optional<float>& mu = std::nullopt,
      const std::optional<std::vector<float>>& timesteps = std::nullopt) {
    LOG(INFO) << "Setting timesteps for FluxScheduler";
    LOG(INFO) << "num_inference_steps: " << num_inference_steps;
    LOG(INFO) << "device: " << device;
    if (sigmas.has_value()) {
      LOG(INFO) << "sigmas: " << sigmas.value();
    }
    if (timesteps.has_value()) {
      LOG(INFO) << "timesteps: " << timesteps.value();
    }
    if (mu.has_value()) {
      LOG(INFO) << "mu: " << mu.value();
    }
    if (use_dynamic_shifting_ && !mu.has_value()) {
      throw std::invalid_argument(
          "mu must be provided when use_dynamic_shifting is true");
    }
    if (sigmas.has_value() && timesteps.has_value() &&
        sigmas->size() != timesteps->size()) {
      throw std::invalid_argument(
          "sigmas and timesteps must have the same length");
    }

    int num_steps = num_inference_steps;
    if (num_steps <= 0) {
      num_steps = sigmas.has_value() ? sigmas->size() : timesteps->size();
    }

    bool is_timesteps_provided = timesteps.has_value();
    torch::Tensor ts_tensor;
    torch::Tensor sigmas_tensor;

    if (is_timesteps_provided) {
      auto* timesteps_data = const_cast<float*>(timesteps->data());
      ts_tensor = torch::from_blob(timesteps_data, {num_steps}, torch::kFloat32)
                      .clone();
    }

    if (!sigmas.has_value()) {
      if (!timesteps.has_value()) {
        std::vector<float> ts_vec(num_steps);
        float start = sigma_max_ * num_train_timesteps_;
        float end = sigma_min_ * num_train_timesteps_;
        for (int i = 0; i < num_steps; ++i) {
          ts_vec[i] = start + (end - start) * i / (num_steps - 1);
        }
        ts_tensor =
            torch::from_blob(ts_vec.data(), {num_steps}, torch::kFloat32)
                .clone();
      }
      sigmas_tensor = ts_tensor / num_train_timesteps_;
    } else {
      auto* sigmas_data = const_cast<float*>(sigmas->data());
      sigmas_tensor =
          torch::from_blob(sigmas_data, {num_steps}, torch::kFloat32).clone();
    }

    if (use_dynamic_shifting_) {
      sigmas_tensor = time_shift(mu.value(), 1.0f, sigmas_tensor);
    } else {
      sigmas_tensor =
          shift_ * sigmas_tensor / (1.0f + (shift_ - 1.0f) * sigmas_tensor);
    }

    if (shift_terminal_.has_value()) {
      sigmas_tensor = stretch_shift_to_terminal(sigmas_tensor);
    }

    if (use_karras_sigmas_) {
      sigmas_tensor = convert_to_karras(sigmas_tensor, num_steps);
    } else if (use_exponential_sigmas_) {
      sigmas_tensor = convert_to_exponential(sigmas_tensor, num_steps);
    } else if (use_beta_sigmas_) {
      sigmas_tensor = convert_to_beta(sigmas_tensor, num_steps);
    }

    sigmas_tensor = sigmas_tensor.to(device).to(torch::kFloat32);
    if (!is_timesteps_provided) {
      ts_tensor = sigmas_tensor * num_train_timesteps_;
    } else {
      ts_tensor = ts_tensor.to(device).to(torch::kFloat32);
    }

    if (invert_sigmas_) {
      sigmas_tensor = 1.0f - sigmas_tensor;
      ts_tensor = sigmas_tensor * num_train_timesteps_;
      sigmas_tensor = torch::cat(
          {sigmas_tensor, torch::ones({1}, torch::kFloat32).to(device)});
    } else {
      sigmas_tensor = torch::cat(
          {sigmas_tensor, torch::zeros({1}, torch::kFloat32).to(device)});
    }

    timesteps_ = ts_tensor;
    sigmas_ = sigmas_tensor;
    step_index_ = std::nullopt;
    begin_index_ = std::nullopt;
  }
  FlowMatchEulerDiscreteSchedulerOutput step(
      const torch::Tensor& model_output,
      const torch::Tensor& timestep,
      const torch::Tensor& sample,
      float s_churn = 0.0f,
      float s_tmin = 0.0f,
      float s_tmax = std::numeric_limits<float>::infinity(),
      float s_noise = 1.0f,
      const std::optional<torch::Generator>& generator = std::nullopt,
      const std::optional<torch::Tensor>& per_token_timesteps = std::nullopt,
      bool return_dict = true) {
    if (!step_index_.has_value()) {
      init_step_index(timestep);
    }
    LOG(INFO) << "FlowMatchEulerDiscreteScheduler";
    torch::Tensor sample_float = sample.to(torch::kFloat32);
    torch::Tensor prev_sample;

    if (per_token_timesteps.has_value()) {
      torch::Tensor per_token_sigmas =
          per_token_timesteps.value() / num_train_timesteps_;
      torch::Tensor sigmas = sigmas_.unsqueeze(1).unsqueeze(1);
      torch::Tensor lower_mask =
          sigmas < (per_token_sigmas.unsqueeze(0) - 1e-6f);
      torch::Tensor lower_sigmas = lower_mask * sigmas;
      auto max_vals = lower_sigmas.max(0);
      lower_sigmas = std::get<0>(max_vals);

      torch::Tensor current_sigma = per_token_sigmas.unsqueeze(-1);
      torch::Tensor next_sigma = lower_sigmas.unsqueeze(-1);
      torch::Tensor dt = current_sigma - next_sigma;

      if (stochastic_sampling_) {
        torch::Tensor x0 = sample_float - current_sigma * model_output;
        torch::Tensor noise;
        noise = torch::randn_like(sample_float);
        prev_sample = (1.0f - next_sigma) * x0 + next_sigma * noise;
      } else {
        prev_sample = sample_float + dt * model_output;
      }
    } else {
      int sigma_idx = step_index_.value();
      torch::Tensor sigma = sigmas_[sigma_idx];
      torch::Tensor sigma_next = sigmas_[sigma_idx + 1];
      torch::Tensor dt = sigma_next - sigma;
      if (stochastic_sampling_) {
        torch::Tensor x0 = sample_float - sigma * model_output;
        torch::Tensor noise;
        noise = torch::randn_like(sample_float);
        prev_sample = (1.0f - sigma_next) * x0 + sigma_next * noise;
      } else {
        prev_sample = sample_float + dt * model_output;
      }
    }
    step_index_ = step_index_.value() + 1;
    if (!per_token_timesteps.has_value()) {
      prev_sample = prev_sample.to(model_output.dtype());
    }
    return FlowMatchEulerDiscreteSchedulerOutput(prev_sample);
  }
  std::optional<int> step_index() const { return step_index_; }
  std::optional<int> begin_index() const { return begin_index_; }
  const torch::Tensor& timesteps() const { return timesteps_; }
  const torch::Tensor& sigmas() const { return sigmas_; }
  int size() const { return num_train_timesteps_; }
  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        std::vector<KVCache>& kv_caches,
                        const ModelInputParams& input_params) {
    // 1. 测试参数（可通过input_params传入，或硬编码用于调试）
    const int num_inference_steps = 50;
    const float mu = 0.5f;              // 动态偏移参数
    const bool use_stochastic = false;  // 先测试确定性模式

    // 2. 配置调度器
    this->set_timesteps(
        num_inference_steps, tokens.device(), /*sigmas=*/std::nullopt, mu);
    this->set_begin_index(0);

    // 3. 生成测试输入（与Python端保持一致的随机种子）
    torch::manual_seed(42);  // 固定随机种子，确保结果可复现
    torch::Tensor sample = torch::randn(
        {1, 3, 32, 32}, torch::dtype(torch::kFloat32));      // 模拟样本
    torch::Tensor model_output = torch::randn_like(sample);  // 模拟模型输出
    torch::Tensor timestep = this->timesteps()[0];           // 初始时间步
    model_output = model_output.to(timestep.device())
                       .to(torch::kFloat32);  // 确保与sample同设备和dtype
    sample = sample.to(timestep.device())
                 .to(torch::kFloat32);  // 确保与timestep同设备和dtype
    // 4. 执行一步调度器计算
    auto output = this->step(model_output,
                             timestep,
                             sample,
                             0.0f,
                             0.0f,
                             std::numeric_limits<float>::infinity(),
                             1.0f,
                             /*generator=*/std::nullopt,
                             /*per_token_timesteps=*/std::nullopt);

    return output.prev_sample;
  }
};
TORCH_MODULE(FlowMatchEulerDiscreteScheduler);
REGISTER_MODEL_ARGS(flowmatcheulerdiscretedescheduler, [&] {
  LOAD_ARG_OR(model_type, "model_type", "flowmatcheulerdiscretedescheduler");
  LOAD_ARG_OR(scheduler_num_train_timesteps, "num_train_timesteps", 1000);
  LOAD_ARG_OR(scheduler_shift, "shift", 1);
  LOAD_ARG_OR(scheduler_use_dynamic_shifting, "use_dynamic_shifting", true);
  LOAD_ARG_OR(scheduler_base_shift, "base_shift", 0.5f);
  LOAD_ARG_OR(scheduler_max_shift, "max_shift", 1.15f);
  LOAD_ARG_OR(scheduler_base_image_seq_len, "base_image_seq_len", 256);
  LOAD_ARG_OR(scheduler_max_image_seq_len, "max_image_seq_len", 4096);

  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG_OR(head_dim, "head_dim", 56);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 128000);
});
}  // namespace xllm::hf