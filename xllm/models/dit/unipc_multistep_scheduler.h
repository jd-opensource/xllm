#pragma once
#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/framework/model/model_input_params.h"
#include "core/framework/state_dict/state_dict.h"
#include "models/model_registry.h"
#include "processors/input_processor.h"
#include "processors/pywarpper_image_processor.h"
namespace xllm {
struct UniPCMultistepSchedulerOutput {
  torch::Tensor prev_sample;
  explicit UniPCMultistepSchedulerOutput(torch::Tensor sample)
      : prev_sample(std::move(sample)) {}
};
class UniPCMultistepSchedulerImpl : public torch::nn::Module {
 private:
  int num_train_timesteps_;
  float beta_start_;
  float beta_end_;
  std::string beta_schedule_;
  std::optional<std::vector<float>> trained_betas_;

  int solver_order_;
  std::string prediction_type_;
  bool thresholding_;
  float dynamic_thresholding_ratio_;
  float sample_max_value_;
  bool predict_x0_;
  std::string solver_type_;
  bool lower_order_final_;
  std::vector<int> disable_corrector_;

  std::string timestep_spacing_;
  int steps_offset_;
  bool rescale_betas_zero_snr_;
  std::string final_sigmas_type_;

  bool use_karras_sigmas_;
  bool use_exponential_sigmas_;
  bool use_beta_sigmas_;
  bool use_flow_sigmas_;
  float flow_shift_;
  bool use_dynamic_shifting_;
  std::string time_shift_type_;

  torch::Tensor timesteps_;
  torch::Tensor sigmas_;
  torch::Tensor betas_;
  torch::Tensor alphas_;
  torch::Tensor alphas_cumprod_;

  std::vector<torch::Tensor> model_outputs_;
  std::vector<torch::Tensor> timestep_list_;
  torch::Tensor last_sample_;
  std::optional<int> step_index_;
  std::optional<int> begin_index_;
  int num_inference_steps_;
  int lower_order_nums_;
  int this_order_;

  torch::Tensor lambda_t_;
  torch::Tensor alpha_t_;
  torch::Tensor sigma_t_;

  torch::Tensor betas_for_alpha_bar(int num_diffusion_timesteps,
                                    float max_beta = 0.999f) {
    auto alpha_bar_fn = [](float t) {
      return std::pow(std::cos((t + 0.008f) / 1.008f * M_PI / 2.0f), 2);
    };

    std::vector<float> betas_vec;
    for (int i = 0; i < num_diffusion_timesteps; ++i) {
      float t1 = static_cast<float>(i) / num_diffusion_timesteps;
      float t2 = static_cast<float>(i + 1) / num_diffusion_timesteps;
      float beta =
          std::min(1.0f - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta);
      betas_vec.push_back(beta);
    }
    return torch::tensor(betas_vec);
  }

  torch::Tensor rescale_zero_terminal_snr(const torch::Tensor& betas) {
    torch::Tensor alphas = 1.0f - betas;
    torch::Tensor alphas_cumprod = torch::cumprod(alphas, 0);
    torch::Tensor alphas_bar_sqrt = torch::sqrt(alphas_cumprod);

    torch::Tensor alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone();
    torch::Tensor alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone();

    alphas_bar_sqrt = alphas_bar_sqrt - alphas_bar_sqrt_T;
    alphas_bar_sqrt = alphas_bar_sqrt * alphas_bar_sqrt_0 /
                      (alphas_bar_sqrt_0 - alphas_bar_sqrt_T);

    torch::Tensor alphas_bar = torch::pow(alphas_bar_sqrt, 2);
    alphas = torch::cat({alphas_bar.slice(0, 0, 1),
                         alphas_bar.slice(0, 1) / alphas_bar.slice(0, 0, -1)});
    return 1.0f - alphas;
  }

  void init_betas() {
    if (trained_betas_.has_value()) {
      betas_ = torch::tensor(trained_betas_.value());
    } else if (beta_schedule_ == "linear") {
      betas_ = torch::linspace(beta_start_, beta_end_, num_train_timesteps_);
    } else if (beta_schedule_ == "scaled_linear") {
      float start = std::sqrt(beta_start_);
      float end = std::sqrt(beta_end_);
      betas_ = torch::pow(torch::linspace(start, end, num_train_timesteps_), 2);
    } else if (beta_schedule_ == "squaredcos_cap_v2") {
      betas_ = betas_for_alpha_bar(num_train_timesteps_);
    } else {
      throw std::invalid_argument("Unknown beta_schedule: " + beta_schedule_);
    }

    if (rescale_betas_zero_snr_) {
      betas_ = rescale_zero_terminal_snr(betas_);
    }

    alphas_ = 1.0f - betas_;
    alphas_cumprod_ = torch::cumprod(alphas_, 0);

    if (rescale_betas_zero_snr_) {
      alphas_cumprod_[-1] = std::pow(2.0f, -24);
    }

    alpha_t_ = torch::sqrt(alphas_cumprod_);
    sigma_t_ = torch::sqrt(1.0 - alphas_cumprod_);
    lambda_t_ = torch::log(alpha_t_) - torch::log(sigma_t_);
    sigmas_ = torch::sqrt((1.0 - alphas_cumprod_) / alphas_cumprod_);
  }

  // private tool function
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
    // NOTE: Actual usage requires the `beta distribution` implementation from
    // scipy, this is just for illustration.
    throw std::runtime_error(
        "Beta sigmas implementation requires scipy integration");
  }

  int sigma_to_t(float sigma, const std::vector<float>& log_sigmas) {
    float log_sigma = std::log(std::max(sigma, 1e-10f));

    std::vector<float> dists;
    for (float ls : log_sigmas) {
      dists.push_back(std::abs(log_sigma - ls));
    }

    int low_idx = std::min_element(dists.begin(), dists.end()) - dists.begin();
    low_idx = std::min(low_idx, static_cast<int>(log_sigmas.size()) - 2);
    int high_idx = low_idx + 1;

    float low = log_sigmas[low_idx];
    float high = log_sigmas[high_idx];
    float w = (low - log_sigma) / (low - high);
    w = std::clamp(w, 0.0f, 1.0f);

    return static_cast<int>((1.0f - w) * low_idx + w * high_idx);
  }

  std::tuple<torch::Tensor, torch::Tensor> sigma_to_alpha_sigma_t(
      const torch::Tensor& sigma) {
    if (use_flow_sigmas_) {
      return {1.0f - sigma, sigma};
    } else {
      torch::Tensor alpha_t = 1.0f / torch::sqrt(sigma.pow(2) + 1.0f);
      torch::Tensor sigma_t = sigma * alpha_t;
      return {alpha_t, sigma_t};
    }
  }

  int index_for_timestep(const torch::Tensor& timestep,
                         const torch::Tensor& schedule_timesteps = {}) {
    torch::Tensor sched =
        schedule_timesteps.defined() ? schedule_timesteps : timesteps_;
    torch::Tensor indices = (sched == timestep).nonzero();

    if (indices.size(0) == 0) {
      return timesteps_.size(0) - 1;
    }

    int pos = indices.size(0) > 1 ? 1 : 0;
    return indices.index({pos, 0}).item<int>();
  }

  void init_step_index(const torch::Tensor& timestep) {
    if (!begin_index_.has_value()) {
      torch::Tensor ts = timestep.to(timesteps_.device());
      step_index_ = index_for_timestep(ts);
    } else {
      step_index_ = begin_index_.value();
    }
  }

  torch::Tensor threshold_sample(const torch::Tensor& sample) {
    auto dtype = sample.dtype();
    int64_t batch_size = sample.size(0);
    int64_t channels = sample.size(1);

    std::vector<int64_t> remaining_dims(sample.sizes().begin() + 2,
                                        sample.sizes().end());
    int64_t prod = 1;
    for (auto d : remaining_dims) prod *= d;

    torch::Tensor sample_float = sample.to(torch::kFloat32);
    torch::Tensor sample_flat =
        sample_float.reshape({batch_size, channels * prod});
    torch::Tensor abs_sample = sample_flat.abs();

    torch::Tensor s =
        std::get<0>(abs_sample.quantile(dynamic_thresholding_ratio_, 1));
    s = torch::clamp(s, 1, sample_max_value_);
    s = s.unsqueeze(1);

    sample_flat = torch::clamp(sample_flat, -s, s) / s;

    std::vector<int64_t> new_shape = {batch_size, channels};
    new_shape.insert(
        new_shape.end(), remaining_dims.begin(), remaining_dims.end());

    return sample_flat.reshape(new_shape).to(dtype);
  }

  torch::Tensor convert_model_output(const torch::Tensor& model_output,
                                     const torch::Tensor& sample) {
    torch::Tensor sigma = sigmas_[step_index_.value()];
    auto [alpha_t, sigma_t] = sigma_to_alpha_sigma_t(sigma);

    torch::Tensor x0_pred;
    if (predict_x0_) {
      if (prediction_type_ == "epsilon") {
        x0_pred = (sample - sigma_t * model_output) / alpha_t;
      } else if (prediction_type_ == "sample") {
        x0_pred = model_output;
      } else if (prediction_type_ == "v_prediction") {
        x0_pred = alpha_t * sample - sigma_t * model_output;
      } else if (prediction_type_ == "flow_prediction") {
        sigma_t = sigmas_[step_index_.value()];
        x0_pred = sample - sigma_t * model_output;
      } else {
        throw std::invalid_argument("Unknown prediction_type: " +
                                    prediction_type_);
      }

      if (thresholding_) {
        x0_pred = threshold_sample(x0_pred);
      }
      return x0_pred;
    } else {
      if (prediction_type_ == "epsilon") {
        return model_output;
      } else if (prediction_type_ == "sample") {
        return (sample - alpha_t * model_output) / sigma_t;
      } else if (prediction_type_ == "v_prediction") {
        return alpha_t * model_output + sigma_t * sample;
      } else {
        throw std::invalid_argument("Unknown prediction_type: " +
                                    prediction_type_);
      }
    }
  }

  torch::Tensor multistep_uni_p_bh_update(const torch::Tensor& model_output,
                                          const torch::Tensor& sample,
                                          int order) {
    torch::Tensor sigma_t = sigmas_[step_index_.value() + 1];
    torch::Tensor sigma_s0 = sigmas_[step_index_.value()];

    auto [alpha_t, sigma_t_val] = sigma_to_alpha_sigma_t(sigma_t);
    auto [alpha_s0, sigma_s0_val] = sigma_to_alpha_sigma_t(sigma_s0);

    torch::Tensor lambda_t = torch::log(alpha_t) - torch::log(sigma_t_val);
    torch::Tensor lambda_s0 = torch::log(alpha_s0) - torch::log(sigma_s0_val);
    torch::Tensor h = lambda_t - lambda_s0;

    torch::Tensor m0 = model_outputs_[model_outputs_.size() - 1];
    torch::Tensor x = sample;

    torch::Device device = sample.device();
    std::vector<float> rks_vec;
    std::vector<torch::Tensor> D1s;

    for (int i = 1; i < order; ++i) {
      int si = step_index_.value() - i;
      torch::Tensor mi = model_outputs_[model_outputs_.size() - (i + 1)];
      torch::Tensor sigma_si = sigmas_[si];
      auto [alpha_si, sigma_si_val] = sigma_to_alpha_sigma_t(sigma_si);
      torch::Tensor lambda_si = torch::log(alpha_si) - torch::log(sigma_si_val);
      float rk = ((lambda_si - lambda_s0) / h).item<float>();
      rks_vec.push_back(rk);
      D1s.push_back((mi - m0) / rk);
    }

    rks_vec.push_back(1.0f);
    torch::Tensor rks =
        torch::tensor(rks_vec, torch::TensorOptions().device(device));

    torch::Tensor hh = predict_x0_ ? -h : h;
    torch::Tensor h_phi_1 = torch::expm1(hh);
    torch::Tensor h_phi_k = h_phi_1 / hh - 1.0f;

    torch::Tensor B_h = solver_type_ == "bh1" ? hh : torch::expm1(hh);

    std::vector<torch::Tensor> R_list;
    std::vector<float> b_vec;
    int factorial_i = 1;

    for (int i = 1; i <= order; ++i) {
      R_list.push_back(torch::pow(rks, i - 1));
      b_vec.push_back((h_phi_k / B_h).item<float>() * factorial_i);
      factorial_i *= (i + 1);
      h_phi_k = h_phi_k / hh - 1.0f / factorial_i;
    }

    torch::Tensor R = torch::stack(R_list);
    torch::Tensor b =
        torch::tensor(b_vec, torch::TensorOptions().device(device));

    torch::Tensor rhos_p;
    if (!D1s.empty()) {
      torch::Tensor D1s_stacked = torch::stack(D1s, 1);
      if (order == 2) {
        rhos_p = torch::tensor(
            {0.5f},
            torch::TensorOptions().dtype(sample.dtype()).device(device));
      } else {
        rhos_p = torch::linalg_solve(R.slice(0, 0, -1).slice(1, 0, -1),
                                     b.slice(0, 0, -1))
                     .to(device)
                     .to(sample.dtype());
      }
    }

    torch::Tensor x_t_;
    if (predict_x0_) {
      x_t_ = sigma_t_val / sigma_s0_val * x - alpha_t * h_phi_1 * m0;
    } else {
      x_t_ = alpha_t / alpha_s0 * x - sigma_t_val * h_phi_1 * m0;
    }

    torch::Tensor x_t = x_t_;
    if (!D1s.empty()) {
      torch::Tensor D1s_stacked = torch::stack(D1s, 1);
      torch::Tensor pred_res =
          torch::einsum("k,bkc...->bc...", {rhos_p, D1s_stacked});
      if (predict_x0_) {
        x_t = x_t_ - alpha_t * B_h * pred_res;
      } else {
        x_t = x_t_ - sigma_t_val * B_h * pred_res;
      }
    }

    return x_t.to(sample.dtype());
  }

  torch::Tensor multistep_uni_c_bh_update(
      const torch::Tensor& this_model_output,
      const torch::Tensor& last_sample,
      const torch::Tensor& this_sample,
      int order) {
    torch::Tensor sigma_t = sigmas_[step_index_.value()];
    torch::Tensor sigma_s0 = sigmas_[step_index_.value() - 1];

    auto [alpha_t, sigma_t_val] = sigma_to_alpha_sigma_t(sigma_t);
    auto [alpha_s0, sigma_s0_val] = sigma_to_alpha_sigma_t(sigma_s0);

    torch::Tensor lambda_t = torch::log(alpha_t) - torch::log(sigma_t_val);
    torch::Tensor lambda_s0 = torch::log(alpha_s0) - torch::log(sigma_s0_val);
    torch::Tensor h = lambda_t - lambda_s0;

    torch::Tensor m0 = model_outputs_[model_outputs_.size() - 1];
    torch::Tensor x = last_sample;
    torch::Tensor model_t = this_model_output;

    torch::Device device = this_sample.device();
    std::vector<float> rks_vec;
    std::vector<torch::Tensor> D1s;

    for (int i = 1; i < order; ++i) {
      int si = step_index_.value() - (i + 1);
      torch::Tensor mi = model_outputs_[model_outputs_.size() - (i + 1)];
      torch::Tensor sigma_si = sigmas_[si];
      auto [alpha_si, sigma_si_val] = sigma_to_alpha_sigma_t(sigma_si);
      torch::Tensor lambda_si = torch::log(alpha_si) - torch::log(sigma_si_val);
      float rk = ((lambda_si - lambda_s0) / h).item<float>();
      rks_vec.push_back(rk);
      D1s.push_back((mi - m0) / rk);
    }

    rks_vec.push_back(1.0f);
    torch::Tensor rks =
        torch::tensor(rks_vec, torch::TensorOptions().device(device));

    torch::Tensor hh = predict_x0_ ? -h : h;
    torch::Tensor h_phi_1 = torch::expm1(hh);
    torch::Tensor h_phi_k = h_phi_1 / hh - 1.0f;

    torch::Tensor B_h = solver_type_ == "bh1" ? hh : torch::expm1(hh);

    std::vector<torch::Tensor> R_list;
    std::vector<float> b_vec;
    int factorial_i = 1;

    for (int i = 1; i <= order; ++i) {
      R_list.push_back(torch::pow(rks, i - 1));
      b_vec.push_back((h_phi_k / B_h).item<float>() * factorial_i);
      factorial_i *= (i + 1);
      h_phi_k = h_phi_k / hh - 1.0f / factorial_i;
    }

    torch::Tensor R = torch::stack(R_list);
    torch::Tensor b =
        torch::tensor(b_vec, torch::TensorOptions().device(device));

    torch::Tensor rhos_c;
    if (order == 1) {
      rhos_c = torch::tensor(
          {0.5f}, torch::TensorOptions().dtype(x.dtype()).device(device));
    } else {
      rhos_c = torch::linalg_solve(R, b).to(device).to(x.dtype());
    }

    torch::Tensor x_t_;
    if (predict_x0_) {
      x_t_ = sigma_t_val / sigma_s0_val * x - alpha_t * h_phi_1 * m0;
    } else {
      x_t_ = alpha_t / alpha_s0 * x - sigma_t_val * h_phi_1 * m0;
    }

    torch::Tensor corr_res = torch::zeros_like(x_t_);
    if (!D1s.empty()) {
      torch::Tensor D1s_stacked = torch::stack(D1s, 1);
      corr_res = torch::einsum("k,bkc...->bc...",
                               {rhos_c.slice(0, 0, -1), D1s_stacked});
    }

    torch::Tensor D1_t = model_t - m0;
    torch::Tensor x_t;
    if (predict_x0_) {
      x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t);
    } else {
      x_t = x_t_ - sigma_t_val * B_h * (corr_res + rhos_c[-1] * D1_t);
    }

    return x_t.to(x.dtype());
  }

 public:
  int64_t order = 1;
  ModelArgs args;

  UniPCMultistepSchedulerImpl(const ModelContext& context)
      : args(context.get_model_args()) {
    num_train_timesteps_ = args.scheduler_num_train_timesteps();
    beta_start_ = args.scheduler_beta_start();
    beta_end_ = args.scheduler_beta_end();
    beta_schedule_ = args.scheduler_beta_schedule();

    solver_order_ = args.scheduler_solver_order();
    prediction_type_ = args.scheduler_prediction_type();
    thresholding_ = args.scheduler_thresholding();
    dynamic_thresholding_ratio_ = args.scheduler_dynamic_thresholding_ratio();
    sample_max_value_ = args.scheduler_sample_max_value();
    predict_x0_ = args.scheduler_predict_x0();
    solver_type_ = args.scheduler_solver_type();
    lower_order_final_ = args.scheduler_lower_order_final();

    timestep_spacing_ = args.scheduler_timestep_spacing();
    steps_offset_ = args.scheduler_steps_offset();
    rescale_betas_zero_snr_ = args.scheduler_rescale_betas_zero_snr();
    final_sigmas_type_ = args.scheduler_final_sigmas_type();

    use_karras_sigmas_ = args.scheduler_use_karras_sigmas();
    use_exponential_sigmas_ = args.scheduler_use_exponential_sigmas();
    use_beta_sigmas_ = args.scheduler_use_beta_sigmas();
    use_flow_sigmas_ = args.scheduler_use_flow_sigmas();
    flow_shift_ = args.scheduler_flow_shift();
    use_dynamic_shifting_ = args.scheduler_use_dynamic_shifting();
    time_shift_type_ = args.scheduler_time_shift_type();
    disable_corrector_ = args.scheduler_disable_corrector();

    init_betas();

    std::vector<float> timesteps_vec(num_train_timesteps_);
    for (int i = 0; i < num_train_timesteps_; ++i) {
      timesteps_vec[i] = num_train_timesteps_ - 1 - i;
    }
    timesteps_ = torch::tensor(timesteps_vec);

    model_outputs_.resize(solver_order_, torch::Tensor());
    timestep_list_.resize(solver_order_, torch::Tensor());
    lower_order_nums_ = 0;
    last_frame_ = std::nullopt;
    step_index_ = std::nullopt;
    begin_index_ = std::nullopt;
    num_inference_steps_ = 0;
  }

  void set_begin_index(int begin_index) { begin_index_ = begin_index; }

  void set_timesteps(int num_inference_steps,
                     const torch::Device& device = torch::kCPU,
                     const std::optional<float>& mu = std::nullopt) {
    if (mu.has_value() && use_dynamic_shifting_ &&
        time_shift_type_ == "exponential") {
      flow_shift_ = std::exp(mu.value());
    }

    std::vector<int64_t> timesteps_vec;
    if (timestep_spacing_ == "linspace") {
      torch::Tensor ts =
          torch::linspace(0, num_train_timesteps_ - 1, num_inference_steps + 1);
      ts = ts.round().flip(0).slice(0, 0, -1);
      timesteps_vec = std::vector<int64_t>(ts.data_ptr<float>(),
                                           ts.data_ptr<float>() + ts.numel());
    } else if (timestep_spacing_ == "leading") {
      int step_ratio = num_train_timesteps_ / (num_inference_steps + 1);
      torch::Tensor ts = torch::arange(0, num_inference_steps + 1) * step_ratio;
      ts = (ts.round().flip(0).slice(0, 0, -1) + steps_offset_);
      timesteps_vec = std::vector<int64_t>(ts.data_ptr<float>(),
                                           ts.data_ptr<float>() + ts.numel());
    } else if (timestep_spacing_ == "trailing") {
      float step_ratio =
          static_cast<float>(num_train_timesteps_) / num_inference_steps;
      torch::Tensor ts =
          torch::arange(num_train_timesteps_, 0, -step_ratio).round() - 1;
      timesteps_vec = std::vector<int64_t>(ts.data_ptr<float>(),
                                           ts.data_ptr<float>() + ts.numel());
    }

    torch::Tensor sigmas =
        torch::sqrt((1.0f - alphas_cumprod_) / alphas_cumprod_);

    if (use_flow_sigmas_) {
      torch::Tensor alphas = torch::linspace(
          1, 1.0f / num_train_timesteps_, num_inference_steps + 1);
      sigmas = 1.0f - alphas;
      sigmas = sigmas.flip(0).slice(0, 0, -1);
      sigmas = flow_shift_ * sigmas / (1.0f + (flow_shift_ - 1.0f) * sigmas);

      std::vector<float> timesteps_float_vec(sigmas.numel());
      auto sigmas_acc = sigmas.accessor<float, 1>();
      for (int i = 0; i < sigmas.numel(); ++i) {
        timesteps_float_vec[i] = sigmas_acc[i] * num_train_timesteps_;
      }
      timesteps_ = torch::tensor(timesteps_float_vec, torch::kInt64);

      float sigma_last;
      if (final_sigmas_type_ == "sigma_min") {
        sigma_last = sigmas[-1].item<float>();
      } else if (final_sigmas_type_ == "zero") {
        sigma_last = 0.0f;
      } else {
        throw std::invalid_argument(
            "final_sigmas_type must be one of 'zero' or 'sigma_min'");
      }
      sigmas = torch::cat({sigmas, torch::tensor({sigma_last})});
    } else if (use_karras_sigmas_ || use_exponential_sigmas_ ||
               use_beta_sigmas_) {
      std::vector<float> log_sigmas_vec;
      auto sigmas_acc = sigmas.accessor<float, 1>();
      for (int i = 0; i < sigmas.numel(); ++i) {
        log_sigmas_vec.push_back(std::log(sigmas_acc[i]));
      }

      sigmas = sigmas.flip(0);
      if (use_karras_sigmas_) {
        sigmas = convert_to_karras(sigmas, num_inference_steps);
      } else if (use_beta_sigmas_) {
        sigmas = convert_to_beta(sigmas, num_inference_steps);
      } else {
        sigmas = convert_to_exponential(sigmas, num_inference_steps);
      }

      std::vector<int64_t> new_timesteps;
      auto sigmas_acc2 = sigmas.accessor<float, 1>();
      for (int i = 0; i < sigmas.numel(); ++i) {
        new_timesteps.push_back(sigma_to_t(sigmas_acc2[i], log_sigmas_vec));
      }
      timesteps_vec = new_timesteps;

      float sigma_last;
      if (final_sigmas_type_ == "sigma_min") {
        sigma_last = sigmas[-1].item<float>();
      } else if (final_sigmas_type_ == "zero") {
        sigma_last = 0.0f;
      } else {
        throw std::invalid_argument(
            "final_sigmas_type must be one of 'zero' or 'sigma_min'");
      }
      sigmas = torch::cat({sigmas, torch::tensor({sigma_last})});
      timesteps_ = torch::tensor(timesteps_vec, torch::kInt64);
    } else {
      torch::Tensor timesteps_tensor =
          torch::tensor(timesteps_vec, torch::kFloat32);
      torch::Tensor arange_tensor =
          torch::arange(0, sigmas.numel(), torch::kFloat32);
      sigmas = torch::from_blob(
          torch::numpy_t<float>(sigmas.data_ptr<float>(), {sigmas.numel()}),
          {sigmas.numel()},
          torch::kFloat32);
      sigmas = torch::nn::functional::interpolate(
                   sigmas.unsqueeze(0).unsqueeze(0),
                   torch::nn::functional::InterpolateFuncOptions()
                       .size(std::vector<int64_t>{
                           static_cast<int64_t>(timesteps_vec.size())})
                       .mode(torch::kLinear)
                       .align_corners(false))
                   .squeeze();

      float sigma_last;
      if (final_sigmas_type_ == "sigma_min") {
        sigma_last =
            torch::sqrt((1.0f - alphas_cumprod_[0]) / alphas_cumprod_[0])
                .item<float>();
      } else if (final_sigmas_type_ == "zero") {
        sigma_last = 0.0f;
      } else {
        throw std::invalid_argument(
            "final_sigmas_type must be one of 'zero' or 'sigma_min'");
      }
      sigmas = torch::cat({sigmas, torch::tensor({sigma_last})});
      timesteps_ = torch::tensor(timesteps_vec, torch::kInt64);
    }

    sigmas_ = sigmas.to(torch::kCPU);
    timesteps_ = timesteps_.to(device).to(torch::kInt64);

    num_inference_steps_ = timesteps_vec.size();
    model_outputs_ = std::vector<torch::Tensor>(solver_order_, torch::Tensor());
    timestep_list_ = std::vector<torch::Tensor>(solver_order_, torch::Tensor());
    lower_order_nums_ = 0;
    last_sample_ = torch::Tensor();
    step_index_ = std::nullopt;
    begin_index_ = std::nullopt;
  }

  torch::Tensor scale_model_input(const torch::Tensor& sample) {
    return sample;
  }

  UniPCMultistepSchedulerOutput step(const torch::Tensor& model_output,
                                     const torch::Tensor& timestep,
                                     const torch::Tensor& sample,
                                     bool return_dict = true) {
    if (!num_inference_steps_) {
      throw std::runtime_error(
          "Number of inference steps is None, run set_timesteps first");
    }

    if (!step_index_.has_value()) {
      init_step_index(timestep);
    }

    bool use_corrector =
        (step_index_.value() > 0 &&
         std::find(disable_corrector_.begin(),
                   disable_corrector_.end(),
                   step_index_.value() - 1) == disable_corrector_.end() &&
         last_sample_.defined());

    torch::Tensor model_output_convert =
        convert_model_output(model_output, sample);

    torch::Tensor current_sample = sample;
    if (use_corrector) {
      current_sample = multistep_uni_c_bh_update(
          model_output_convert, last_sample_, sample, this_order_);
    }

    for (int i = 0; i < solver_order_ - 1; ++i) {
      model_outputs_[i] = model_outputs_[i + 1];
      timestep_list_[i] = timestep_list_[i + 1];
    }

    model_outputs_[solver_order_ - 1] = model_output_convert;
    timestep_list_[solver_order_ - 1] = timestep;

    int order;
    if (lower_order_final_) {
      order =
          std::min(solver_order_,
                   static_cast<int>(timesteps_.size(0)) - step_index_.value());
    } else {
      order = solver_order_;
    }

    this_order_ = std::min(order, lower_order_nums_ + 1);

    last_sample_ = current_sample;
    torch::Tensor prev_sample =
        multistep_uni_p_bh_update(model_output, current_sample, this_order_);

    if (lower_order_nums_ < solver_order_) {
      lower_order_nums_ += 1;
    }

    step_index_ = step_index_.value() + 1;

    return UniPCMultistepSchedulerOutput(prev_sample);
  }

  torch::Tensor add_noise(const torch::Tensor& original_samples,
                          const torch::Tensor& noise,
                          const torch::Tensor& timesteps) {
    torch::Tensor sigmas =
        sigmas_.to(original_samples.device()).to(original_samples.dtype());
    torch::Tensor schedule_timesteps = timesteps_.to(original_samples.device());
    torch::Tensor ts = timesteps.to(original_samples.device());

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
        step_indices,
        torch::TensorOptions().dtype(torch::kLong).device(sigmas.device()));
    torch::Tensor sigma = sigmas.index_select(0, sigma_indices).flatten();

    while (sigma.dim() < original_samples.dim()) {
      sigma = sigma.unsqueeze(-1);
    }

    auto [alpha_t, sigma_t] = sigma_to_alpha_sigma_t(sigma);
    torch::Tensor noisy_samples = alpha_t * original_samples + sigma_t * noise;
    return noisy_samples;
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
    // 1. Test params (can be passed in via input_params, or hard-coded for
    // debugging purposes.)
    const int num_inference_steps = 50;
    const float mu = 0.5f;              // Dynamic offset parameter
    const bool use_stochastic = false;  // Test the deterministic mode first.

    // 2. config the scheduler
    this->set_timesteps(
        num_inference_steps, tokens.device(), /*sigmas=*/std::nullopt, mu);
    this->set_begin_index(0);

    // 3. Generate test inputs (with fixed random seed to ensure
    // reproducibility)
    torch::manual_seed(42);  // Fixed random seed for reproducibility
    torch::Tensor sample = torch::randn(
        {1, 3, 32, 32}, torch::dtype(torch::kFloat32));  // Mock sample
    torch::Tensor model_output =
        torch::randn_like(sample);                  // Mock model output
    torch::Tensor timestep = this->timesteps()[0];  // Initial timestep
    model_output =
        model_output.to(timestep.device())
            .to(torch::kFloat32);  // Ensure same device and dtype as sample
    sample =
        sample.to(timestep.device())
            .to(torch::kFloat32);  // Ensure same device and dtype as timestep
    // 4. Execute one step of the scheduler calculation
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

TORCH_MODULE(UniPCMultistepScheduler);
REGISTER_MODEL_ARGS(UniPCMultistepScheduler, [&] {
  LOAD_ARG_OR(scheduler_num_train_timesteps, "num_train_timesteps", 1000);
  LOAD_ARG_OR(scheduler_beta_start, "beta_start", 0.0001f);
  LOAD_ARG_OR(scheduler_beta_end, "beta_end", 0.02f);
  LOAD_ARG_OR(scheduler_beta_schedule, "beta_schedule", "linear");

  LOAD_ARG_OR(scheduler_solver_order, "solver_order", 2);
  LOAD_ARG_OR(scheduler_prediction_type, "prediction_type", "flow_prediction");
  LOAD_ARG_OR(scheduler_thresholding, "thresholding", false);
  LOAD_ARG_OR(scheduler_dynamic_thresholding_ratio,
              "dynamic_thresholding_ratio",
              0.995f);
  LOAD_ARG_OR(scheduler_sample_max_value, "sample_max_value", 1.0f);
  LOAD_ARG_OR(scheduler_predict_x0, "predict_x0", true);
  LOAD_ARG_OR(scheduler_solver_type, "solver_type", "bh2");
  LOAD_ARG_OR(scheduler_lower_order_final, "lower_order_final", true);

  LOAD_ARG_OR(scheduler_timestep_spacing, "timestep_spacing", "linspace");
  LOAD_ARG_OR(scheduler_steps_offset, "steps_offset", 0);
  LOAD_ARG_OR(
      scheduler_rescale_betas_zero_snr, "rescale_betas_zero_snr", false);
  LOAD_ARG_OR(scheduler_final_sigmas_type, "final_sigmas_type", "zero");

  LOAD_ARG_OR(scheduler_use_karras_sigmas, "use_karras_sigmas", false);
  LOAD_ARG_OR(
      scheduler_use_exponential_sigmas, "use_exponential_sigmas", false);
  LOAD_ARG_OR(scheduler_use_beta_sigmas, "use_beta_sigmas", false);
  LOAD_ARG_OR(scheduler_use_flow_sigmas, "use_flow_sigmas", true);
  LOAD_ARG_OR(scheduler_flow_shift, "flow_shift", 3.0f);
  LOAD_ARG_OR(scheduler_use_dynamic_shifting, "use_dynamic_shifting", false);
  LOAD_ARG_OR(scheduler_time_shift_type, "time_shift_type", "exponential");
  LOAD_ARG_OR(
      scheduler_disable_corrector, "disable_corrector", std::vector<int>{});
});
}  // namespace xllm
