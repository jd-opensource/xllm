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

#pragma once

#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <torch/torch.h>

#include <atomic>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model/dit_model.h"
#include "core/framework/model_context.h"
#include "core/framework/tokenizer/fast_tokenizer.h"
#include "core/framework/tokenizer/tokenizer_args.h"
#include "core/util/json_reader.h"
#include "models/model_registry.h"
#include "text_vae_cola.h"
#include "transformer_cola_dit.h"

namespace xllm {

// ---------------------------------------------------------------------------
// ColaDLMPipeline — main pipeline for Cola-DLM text diffusion inference
// ---------------------------------------------------------------------------
// Reference: cola_dlm/inference.py generate_task_repaint_inference()
//
// Implements the three-step inference algorithm:
// 1. Prefix encode: z^pre = VAE.encode(x^pre)
// 2. Block-wise latent prior transport with CFG + Euler integration
// 3. Conditional decode: x^res = VAE.decode(z)

class ColaDLMPipelineImpl : public torch::nn::Module {
 public:
  explicit ColaDLMPipelineImpl(const DiTModelContext& context) {
    options_ = context.get_tensor_options();

    // Create DiT transformer
    dit_ = register_module(
        "dit", ColaDiTTransformer(context.get_model_context("cola_dit")));

    // Create VAE
    vae_ = register_module(
        "vae", ColaTextVAEModel(context.get_model_context("cola_vae")));

    LOG(INFO) << "Initializing Cola-DLM pipeline...";
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    LOG(INFO) << "Cola-DLM pipeline loading model from "
              << loader->model_root_path();

    if (loader->has_component("cola_dit")) {
      // Component layout: separate subdirectories per component.
      auto dit_loader = loader->take_component_loader("cola_dit");
      auto vae_loader = loader->take_component_loader("cola_vae");

      dit_->load_model(std::move(dit_loader));
      dit_->to(options_.device());

      vae_->load_model(std::move(vae_loader));
      vae_->to(options_.device());

      // Load tokenizer: either from component or from tokenizer.json in root
      if (loader->has_component("tokenizer")) {
        auto tokenizer_loader = loader->take_component_loader("tokenizer");
        tokenizer_ = tokenizer_loader->tokenizer();
      } else {
        // Look for tokenizer.json in the model root directory
        const std::string root_path = loader->model_root_path();
        std::filesystem::path tokenizer_path =
            std::filesystem::path(root_path) / "tokenizer.json";
        if (!std::filesystem::exists(tokenizer_path)) {
          // Also check parent directory (for nested layouts like
          // model_root/cola_dlm/ where tokenizer.json is in model_root/)
          tokenizer_path =
              std::filesystem::path(root_path).parent_path() / "tokenizer.json";
        }
        CHECK(std::filesystem::exists(tokenizer_path))
            << "tokenizer.json not found in model root or parent: "
            << root_path;
        LOG(INFO) << "Loading tokenizer from " << tokenizer_path.string();
        TokenizerArgs tokenizer_args;
        tokenizer_args.vocab_file(tokenizer_path.string());
        tokenizer_args.tokenizer_type("fast");
        tokenizer_ = std::make_unique<FastTokenizer>(tokenizer_args);
      }
    } else {
      LOG(FATAL) << "Cola-DLM: model_index.json must define cola_dit, "
                    "cola_vae, and tokenizer components";
    }

    // Read block_size from DiT config.json
    // Try multiple possible locations for nested directory layouts
    std::vector<std::string> config_candidates = {
        loader->model_root_path() + "/cola_dit/config.json",
        loader->model_root_path() + "/cola_dlm/cola_dit/config.json",
    };
    // Also check parent directory for nested layouts
    std::filesystem::path root(loader->model_root_path());
    if (root.has_parent_path()) {
      config_candidates.push_back(root.parent_path().string() +
                                  "/cola_dit/config.json");
    }
    for (const auto& config_path : config_candidates) {
      JsonReader cfg;
      if (cfg.parse(config_path)) {
        if (auto v = cfg.value<int32_t>("block_size")) {
          block_size_ = v.value();
          LOG(INFO) << "Cola-DLM: block_size=" << block_size_ << " from "
                    << config_path;
          break;
        }
      }
    }
  }

  DiTForwardOutput forward(const DiTForwardInput& input) {
    torch::NoGradGuard no_grad;

    const auto& gp = input.generation_params;
    auto device = options_.device();

    // Generation parameters.
    //
    // Text-diffusion–specific fields (max_new_tokens, diffusion_steps, …) are
    // available via the new proto fields when the client sends them.
    // For clients that haven't yet been updated we also accept the semantically
    // equivalent image-generation fields as a fallback:
    //   num_inference_steps  →  diffusion_steps  (same concept)
    //   max_sequence_length  →  max_new_tokens   (temporary re-use)
    //
    // Priority: dedicated field > legacy fallback > built-in default.
    int64_t max_new_tokens = gp.max_new_tokens > 0 ? gp.max_new_tokens
                             : gp.max_sequence_length > 0
                                 ? gp.max_sequence_length
                                 : 256;
    int32_t diffusion_steps = gp.diffusion_steps > 0 ? gp.diffusion_steps
                              : gp.num_inference_steps > 0
                                  ? gp.num_inference_steps
                                  : 16;
    float temperature = gp.temperature;
    int32_t top_k = gp.top_k;
    float top_p = gp.top_p;
    float guidance_scale = gp.guidance_scale > 0 ? gp.guidance_scale : 7.0f;
    int64_t seed = gp.seed;

    // Note: per-request CUDA RNG randomization is handled one level up in
    // DiTWorkerImpl::step(), which seeds the CUDA generator with a unique
    // per-call value before the pipeline is invoked.  Here we only override
    // the seed when the caller has explicitly requested a specific one.

    int64_t block_size = block_size_;
    int64_t patch_size = vae_->patch_size();
    int64_t chunk = patch_size * block_size;

    // Diffusion timesteps: linspace(T=1000, 0, diffusion_steps+1)
    constexpr float T = 1000.0f;
    auto timesteps =
        torch::linspace(T, 0.0f, diffusion_steps + 1, torch::kFloat32);

    // -----------------------------------------------------------------
    // Step 1: Tokenize + block-align pad
    // -----------------------------------------------------------------
    CHECK(tokenizer_ != nullptr) << "Tokenizer not loaded";

    std::string prompt_text = input.prompts.empty() ? "" : input.prompts[0];
    LOG(INFO) << "Cola-DLM: prompt_text='" << prompt_text << "'";
    std::vector<int32_t> tokens;
    if (!tokenizer_->encode(prompt_text, &tokens, /*add_bos=*/false)) {
      LOG(WARNING) << "Cola-DLM: tokenizer encode failed for prompt: "
                   << prompt_text;
      tokens.clear();
    }
    LOG(INFO) << "Cola-DLM: tokenized " << tokens.size() << " tokens";

    if (tokens.empty()) {
      LOG(ERROR)
          << "Cola-DLM: empty tokens after encoding, returning empty output";
      DiTForwardOutput output;
      output.text_output.push_back("");
      return output;
    }

    // Pad to multiple of chunk (patch_size * block_size)
    int64_t orig_len = static_cast<int64_t>(tokens.size());
    int64_t pad_len = (chunk - orig_len % chunk) % chunk;
    // OLMo tokenizer: pad_token = 100277, EOS = 100257 (<|endoftext|>)
    constexpr int32_t kPadTokenId = 100277;
    constexpr int32_t kEosTokenId = 100257;
    for (int64_t i = 0; i < pad_len; ++i) {
      tokens.push_back(kPadTokenId);
    }

    auto input_ids =
        torch::tensor(tokens, torch::dtype(torch::kLong)).to(device);

    // Token labels: 1 = real, 3 = padding
    std::vector<int64_t> token_labels_vec(orig_len, 1);
    token_labels_vec.resize(orig_len + pad_len, 3);
    auto token_labels =
        torch::tensor(token_labels_vec, torch::dtype(torch::kLong)).to(device);

    // -----------------------------------------------------------------
    // Step 2: VAE encode
    // -----------------------------------------------------------------
    std::vector<torch::Tensor> input_ids_list = {input_ids};
    at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
    at::autocast::set_autocast_enabled(at::kCUDA, true);
    auto [latents, sample_lens] = vae_->encode(input_ids_list);
    at::autocast::set_autocast_enabled(at::kCUDA, false);
    latents = latents.to(torch::kFloat32);  // match Python .float()
    LOG(INFO) << "Cola-DLM: VAE encode done, latents shape=[" << latents.size(0)
              << ", " << latents.size(1)
              << "], mean=" << latents.mean().item<float>()
              << ", std=" << latents.std().item<float>();

    // -----------------------------------------------------------------
    // Step 3: Latent labels + prefix / first-block split
    // (matches official Python generate_task_repaint_inference)
    // -----------------------------------------------------------------
    int64_t n_patches = latents.size(0);
    int64_t latent_dim = latents.size(1);

    torch::Tensor latent_labels;
    if (patch_size > 1) {
      auto reshaped = token_labels.reshape({n_patches, patch_size});
      auto c1 = reshaped.eq(1).any(/*dim=*/-1);
      auto c2 = reshaped.eq(2).any(/*dim=*/-1);
      latent_labels = torch::full(
          {n_patches},
          3,
          torch::TensorOptions().dtype(torch::kLong).device(device));
      latent_labels.masked_fill_(c2, 2);
      latent_labels.masked_fill_(c1, 1);
    } else {
      latent_labels = token_labels;
    }

    int64_t num_real = latent_labels.eq(1).sum().item<int64_t>();

    torch::Tensor prefix_latents;
    torch::Tensor first_block_latents;
    torch::Tensor
        first_block_labels;  // latent-level labels for first gen block
    int64_t first_block_prompt_tokens = 0;

    if (num_real % block_size != 0) {
      int64_t start_idx = (num_real / block_size) * block_size;
      prefix_latents = latents.slice(0, 0, start_idx);

      if (start_idx + block_size <= n_patches) {
        first_block_latents =
            latents.slice(0, start_idx, start_idx + block_size).clone();
        first_block_labels =
            latent_labels.slice(0, start_idx, start_idx + block_size).clone();
        // Label-3 pads within the first gen block are treated as to-generate.
        first_block_labels.masked_fill_(first_block_labels.eq(3), 2);
        // Prompt token count inside the first gen block.
        int64_t token_start = start_idx * patch_size;
        int64_t token_end = std::min(token_start + block_size * patch_size,
                                     token_labels.size(0));
        first_block_prompt_tokens =
            token_labels.slice(0, token_start, token_end)
                .eq(1)
                .sum()
                .item<int64_t>();
      } else {
        prefix_latents = latents.slice(0, 0, num_real);
        first_block_latents =
            latents.slice(0, n_patches - block_size, n_patches).clone();
        first_block_labels = torch::full(
            {block_size},
            2,
            torch::TensorOptions().dtype(torch::kLong).device(device));
      }
    } else {
      prefix_latents = latents.slice(0, 0, num_real);
      first_block_latents =
          latents.slice(0, n_patches - block_size, n_patches).clone();
      first_block_labels = torch::full(
          {block_size},
          2,
          torch::TensorOptions().dtype(torch::kLong).device(device));
    }

    int64_t prefix_len = prefix_latents.size(0);

    // first_block_labels is used only to derive first_block_prompt_tokens
    // above. The actual pin logic uses slice(0, 0, first_block_prompt_tokens)
    // which is correct when all prompt latents are at the start of the block.
    (void)first_block_labels;  // suppress unused-variable warning

    int64_t max_blocks = (max_new_tokens + block_size * patch_size - 1) /
                         (block_size * patch_size);

    // Per-sample CFG scale for block 0: when prefix is empty, cond == uncond
    // so CFG just amplifies bf16 noise; fall back to scale=1.
    float cfg_scale_block0 = (prefix_len == 0) ? 1.0f : guidance_scale;
    if (prefix_len == 0) {
      LOG(INFO) << "[inference] prompt shorter than block_size=" << block_size
                << "; CFG disabled for first block (guidance_scale -> 1.0)";
    }

    LOG(INFO) << "Cola-DLM: n_patches=" << n_patches
              << ", num_real=" << num_real << ", prefix_len=" << prefix_len
              << ", block_size=" << block_size << ", patch_size=" << patch_size
              << ", max_blocks=" << max_blocks
              << ", first_block_prompt_tokens=" << first_block_prompt_tokens;

    // -----------------------------------------------------------------
    // Step 4: Enable per-layer KV caches on DiT and VAE decoder.
    // -----------------------------------------------------------------
    dit_->set_kv_cache(true);
    vae_->set_kv_cache(true);

    // -----------------------------------------------------------------
    // Step 4a: Prefix prefetch — write prefix K/V into DiT and VAE caches.
    // (Matches official Python inference.py lines 499-517.)
    // -----------------------------------------------------------------
    if (prefix_len > 0) {
      std::vector<int64_t> prefix_k_lens = {prefix_len};

      // DiT prefix prefetch at timestep=0.
      auto ts_prefix = torch::zeros(
          {prefix_len},
          torch::TensorOptions().dtype(torch::kFloat32).device(device));
      at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
      at::autocast::set_autocast_enabled(at::kCUDA, true);
      dit_->forward(prefix_latents.to(torch::kBFloat16),
                    prefix_k_lens,
                    prefix_k_lens,
                    ts_prefix,
                    /*update_kv=*/true,
                    /*use_kv_cache=*/true);
      at::autocast::set_autocast_enabled(at::kCUDA, false);

      // VAE decoder prefix prefetch.
      at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
      at::autocast::set_autocast_enabled(at::kCUDA, true);
      vae_->decode(prefix_latents,
                   prefix_k_lens,
                   prefix_k_lens,
                   /*update_kv=*/true);
      at::autocast::set_autocast_enabled(at::kCUDA, false);
    }

    // -----------------------------------------------------------------
    // Step 5: Block-wise generation loop
    // (Matches official Python inference.py lines 571-706.)
    // -----------------------------------------------------------------
    // Per-sample cumulative K length, initially = prefix_len.
    int64_t k_len_cum = prefix_len;

    std::vector<int64_t> all_token_ids;
    const int64_t sample_id = 0;

    for (int64_t block_idx = 0; block_idx < max_blocks; ++block_idx) {
      // After adding current block, total K = k_len_cum + block_size.
      k_len_cum += block_size;
      std::vector<int64_t> k_lens_cond = {k_len_cum};
      std::vector<int64_t> q_lens = {block_size};
      std::vector<int64_t> k_lens_uncond = {block_size};

      float block_cfg_scale =
          (block_idx == 0) ? cfg_scale_block0 : guidance_scale;

      LOG(INFO) << "Cola-DLM: block " << block_idx << "/" << max_blocks;

      // Draw initial noise using a per-request seeded Generator so that:
      //  (a) When seed > 0: the caller gets fully reproducible noise.
      //  (b) When seed == 0: a per-block monotonic counter ensures every
      //      request draws distinct noise even though init_model() fixed the
      //      global CUDA RNG to a constant value.
      //
      // We use a *local* at::Generator object instead of
      // torch::cuda::manual_seed because the global CUDA RNG state can be
      // pinned to a fixed value by init_model's device_.set_seed() call, and
      // manual_seed on the global generator may not affect the per-stream or
      // per-thread generator that torch::randn actually uses on the worker
      // thread.  Creating a fresh generator and passing it explicitly to randn
      // bypasses all global state. Draw initial noise with a per-block seeded
      // Generator.
      //
      // Seed formula mirrors the official Cola-DLM Python implementation
      // (COLA_INFER_PER_SAMPLE_NOISE_SEED environment variable):
      //
      //   noise_seed = base_seed + sample_id * 1_000 + block_idx * 10_000_000
      //
      // When gp.seed >= 0 (the default is 0), use this deterministic formula
      // with base_seed = gp.seed.  seed=0 is treated as a valid base seed
      // (same as the official code when COLA_INFER_PER_SAMPLE_NOISE_SEED=0),
      // producing reproducible noise across requests.
      //
      // When gp.seed < 0, fall back to a per-request monotonic counter so
      // that successive requests with no explicit seed still draw uncorrelated
      // noise (useful for interactive / streaming use where the caller cannot
      // easily supply a seed).
      //
      // A local at::Generator is used rather than torch::cuda::manual_seed so
      // that the explicit generator is passed directly to randn, bypassing the
      // global CUDA RNG state which may have been pinned by init_model's
      // device_.set_seed() call.
      at::Generator noise_gen = at::cuda::detail::createCUDAGenerator(
          static_cast<at::DeviceIndex>(device.index()));
      {
        uint64_t effective_seed;
        if (seed >= 0) {
          // Deterministic: matches official COLA_INFER_PER_SAMPLE_NOISE_SEED
          // formula: base_seed + sample_id * 1000 + block_idx * 10_000_000
          effective_seed = static_cast<uint64_t>(seed + sample_id * 1000LL +
                                                 block_idx * 10'000'000LL);
        } else {
          // Stochastic fallback: monotonic counter → uncorrelated noise per
          // request even when the global CUDA RNG is fixed.
          effective_seed =
              (++forward_counter_) * 6364136223846793005ULL +
              1442695040888963407ULL +
              static_cast<uint64_t>(block_idx) * 2862933555777941757ULL;
        }
        noise_gen.set_current_seed(effective_seed);
        LOG(INFO) << "Cola-DLM: b" << block_idx
                  << " noise_seed=" << effective_seed << " (seed=" << seed
                  << ")";
      }
      auto txt =
          torch::randn({block_size, latent_dim}, noise_gen, latents.options());
      LOG(INFO) << "Cola-DLM: block " << block_idx
                << " noise: mean=" << txt.mean().item<float>()
                << ", std=" << txt.std().item<float>();
      // Log initial noise for gen positions to compare with final denoised
      // values.
      if (block_idx == 0 && first_block_prompt_tokens > 0 &&
          first_block_prompt_tokens < block_size) {
        auto noise_cpu = txt.cpu();
        auto gen_noise = noise_cpu.slice(0, first_block_prompt_tokens);
        auto gen_flat = gen_noise.flatten();
        std::string nvals;
        for (int64_t i = 0; i < std::min(gen_flat.size(0), int64_t(8)); ++i) {
          nvals += std::to_string(gen_flat[i].item<float>()) + " ";
        }
        LOG(INFO) << "Cola-DLM: b0 INITIAL gen noise (before diffusion): ["
                  << nvals << "] mean=" << gen_noise.mean().item<float>()
                  << ", std=" << gen_noise.std().item<float>();
      }

      // --- Euler integration loop ------------------------------------------
      for (int64_t t_idx = 0; t_idx < diffusion_steps; ++t_idx) {
        float t_curr = timesteps[t_idx].item<float>();
        float t_next = timesteps[t_idx + 1].item<float>();
        float dt = (t_curr - t_next) / T;

        // Block 0 clean-guidance: pin prompt latent positions to ground truth
        // and fix their timestep to 0 throughout ALL diffusion steps.
        // flat_mask has 'first_block_prompt_tokens' True entries at positions
        // [0..first_block_prompt_tokens-1] (all prompt latents are contiguous
        // because padding comes after the prompt).
        if (block_idx == 0 && first_block_prompt_tokens > 0) {
          txt.slice(0, 0, first_block_prompt_tokens) =
              first_block_latents.slice(0, 0, first_block_prompt_tokens);
        }

        // Timestep tensor for current block only.
        auto ts_tensor = torch::full(
            {block_size},
            t_curr,
            torch::TensorOptions().dtype(torch::kFloat32).device(device));
        if (block_idx == 0 && first_block_prompt_tokens > 0) {
          // Zero timestep for prompt positions (already clean).
          ts_tensor.slice(0, 0, first_block_prompt_tokens).fill_(0.0f);
        }

        // --- Conditional pass (use_kv_cache=True) ---
        torch::Tensor drift_cond;
        {
          at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
          at::autocast::set_autocast_enabled(at::kCUDA, true);
          drift_cond = dit_->forward(txt.to(torch::kBFloat16),
                                     k_lens_cond,
                                     q_lens,
                                     ts_tensor,
                                     /*update_kv=*/false,
                                     /*use_kv_cache=*/true);
          at::autocast::set_autocast_enabled(at::kCUDA, false);
        }

        // --- Unconditional pass (no cache) ---
        torch::Tensor drift_uncond;
        {
          at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
          at::autocast::set_autocast_enabled(at::kCUDA, true);
          drift_uncond = dit_->forward(txt.to(torch::kBFloat16),
                                       k_lens_uncond,
                                       q_lens,
                                       ts_tensor,
                                       /*update_kv=*/false,
                                       /*use_kv_cache=*/false);
          at::autocast::set_autocast_enabled(at::kCUDA, false);
        }

        // CFG combination (fp32).
        auto drift =
            block_cfg_scale * (drift_cond - drift_uncond) + drift_uncond;
        txt = txt - drift * dt;

        // Re-pin prompt positions after Euler step.
        if (block_idx == 0 && first_block_prompt_tokens > 0) {
          txt.slice(0, 0, first_block_prompt_tokens) =
              first_block_latents.slice(0, 0, first_block_prompt_tokens);
        }

        if (t_idx == 0) {
          LOG(INFO) << "Cola-DLM: block " << block_idx << " step " << t_idx
                    << ": t_curr=" << t_curr << ", dt=" << dt
                    << ", drift_cond: mean=" << drift_cond.mean().item<float>()
                    << ", std=" << drift_cond.std().item<float>()
                    << ", drift: mean=" << drift.mean().item<float>()
                    << ", std=" << drift.std().item<float>()
                    << ", txt_after: mean=" << txt.mean().item<float>()
                    << ", std=" << txt.std().item<float>();
        }
      }

      LOG(INFO) << "Cola-DLM: block " << block_idx
                << " denoised: mean=" << txt.mean().item<float>()
                << ", std=" << txt.std().item<float>()
                << ", min=" << txt.min().item<float>()
                << ", max=" << txt.max().item<float>();
      // Log per-region stats for block 0: prompt positions vs generated
      // positions.
      if (block_idx == 0 && first_block_prompt_tokens > 0 &&
          first_block_prompt_tokens < block_size) {
        auto txt_cpu = txt.cpu();
        auto prompt_part = txt_cpu.slice(0, 0, first_block_prompt_tokens);
        auto gen_part = txt_cpu.slice(0, first_block_prompt_tokens);
        LOG(INFO) << "Cola-DLM: b0 prompt latents (0:"
                  << first_block_prompt_tokens
                  << "): mean=" << prompt_part.mean().item<float>()
                  << ", std=" << prompt_part.std().item<float>();
        // Also log the pad latents (positions 11-15 of first_block_latents)
        // to compare with final gen latents - if similar, diffusion had no
        // effect
        if (first_block_latents.defined() &&
            first_block_latents.size(0) > first_block_prompt_tokens) {
          auto pad_part =
              first_block_latents.cpu().slice(0, first_block_prompt_tokens);
          auto pad_flat = pad_part.flatten();
          std::string pvals;
          for (int64_t i = 0; i < std::min(pad_flat.size(0), int64_t(8)); ++i) {
            pvals += std::to_string(pad_flat[i].item<float>()) + " ";
          }
          LOG(INFO) << "Cola-DLM: b0 PAD latents (VAE-encoded pad tokens "
                    << first_block_prompt_tokens << ":" << block_size << "): ["
                    << pvals << "] mean=" << pad_part.mean().item<float>()
                    << ", std=" << pad_part.std().item<float>();
        }
        LOG(INFO) << "Cola-DLM: b0 generated latents ("
                  << first_block_prompt_tokens << ":" << block_size
                  << "): mean=" << gen_part.mean().item<float>()
                  << ", std=" << gen_part.std().item<float>();
        // Also print first few latent values to compare with official
        auto gen_flat = gen_part.flatten();
        std::string vals;
        for (int64_t i = 0; i < std::min(gen_flat.size(0), int64_t(8)); ++i) {
          vals += std::to_string(gen_flat[i].item<float>()) + " ";
        }
        LOG(INFO) << "Cola-DLM: b0 gen latents first values: [" << vals << "]";
      }

      // --- VAE decode current block (update_kv=True commits to VAE cache) ---
      at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
      at::autocast::set_autocast_enabled(at::kCUDA, true);
      auto decoded = vae_->decode(txt,
                                  k_lens_cond,
                                  q_lens,
                                  /*update_kv=*/true);
      at::autocast::set_autocast_enabled(at::kCUDA, false);

      LOG(INFO) << "Cola-DLM: VAE decode done, decoded shape=["
                << decoded.size(0) << ", " << decoded.size(1) << ", "
                << decoded.size(2) << "]";

      // decoded: (1, block_size*patch_size, vocab)
      auto block_logits = decoded.squeeze(0);  // (block_tokens, vocab)
      int64_t block_tokens = block_size * patch_size;

      // Greedy or sampling decode — matches official sample_with_strategies().
      //
      // Apply repetition_penalty first (before temperature/top_k/top_p),
      // exactly as the official sample_with_strategies() does even for greedy
      // (temperature=0) decoding.
      // repetition_penalty=1.0 is a no-op; values > 1.0 penalise reuse.
      constexpr float kRepetitionPenalty = 1.1f;
      auto logits_rep = block_logits.clone();
      if (kRepetitionPenalty != 1.0f && !all_token_ids.empty()) {
        // Build a (block_tokens, n_prev) index tensor of previously generated
        // token IDs, then gather their current logit scores, apply the penalty,
        // and scatter back.
        auto prev_ids = torch::tensor(
            std::vector<int64_t>(all_token_ids.begin(), all_token_ids.end()),
            torch::TensorOptions().dtype(torch::kLong).device(device));
        // Broadcast prev_ids to (block_tokens, n_prev)
        int64_t n_prev = prev_ids.size(0);
        auto prev_ids_exp =
            prev_ids.unsqueeze(0).expand({block_tokens, n_prev});
        // Gather logit for each prev token at every generated position
        auto scores = torch::gather(logits_rep, /*dim=*/1, prev_ids_exp);
        // Penalty: if logit < 0 → multiply by penalty; else → divide
        scores = torch::where(scores < 0,
                              scores * kRepetitionPenalty,
                              scores / kRepetitionPenalty);
        logits_rep.scatter_(/*dim=*/1, prev_ids_exp, scores);
      }

      torch::Tensor block_ids;
      if (temperature < 1e-5f) {
        block_ids = logits_rep.argmax(/*dim=*/-1);  // (block_tokens,)
      } else {
        // Apply temperature scaling in logit space (before softmax).
        // Use logits_rep (with repetition penalty already applied).
        auto logits = logits_rep / temperature;

        // top_k: keep only the top-k logits, set the rest to -inf.
        // Must be done in LOGIT space so that softmax re-normalises correctly.
        if (top_k > 0) {
          int64_t k = std::min(top_k, static_cast<int32_t>(logits.size(-1)));
          auto [topk_values, topk_indices] = torch::topk(logits, k, /*dim=*/-1);
          // threshold = k-th largest logit per row
          auto min_topk = topk_values.select(-1, k - 1).unsqueeze(-1);
          logits = torch::where(
              logits < min_topk,
              torch::full_like(logits, -std::numeric_limits<float>::infinity()),
              logits);
        }

        // top_p (nucleus sampling): keep the smallest set whose cumulative
        // softmax probability >= top_p, set the rest to -inf.
        if (top_p > 0.0f && top_p < 1.0f) {
          auto [sorted_logits, sorted_indices] =
              torch::sort(logits, /*dim=*/-1, /*descending=*/true);
          auto cumulative_probs =
              torch::softmax(sorted_logits, /*dim=*/-1).cumsum(/*dim=*/-1);
          // Remove tokens with cumulative probability above the threshold
          // (shift right so the first token above threshold is kept)
          auto remove_mask = cumulative_probs > top_p;
          remove_mask.narrow(-1, 1, remove_mask.size(-1) - 1)
              .copy_(remove_mask.narrow(-1, 0, remove_mask.size(-1) - 1));
          remove_mask.narrow(-1, 0, 1).fill_(false);
          auto indices_to_remove =
              remove_mask.scatter(-1, sorted_indices, remove_mask);
          logits = logits.masked_fill(indices_to_remove,
                                      -std::numeric_limits<float>::infinity());
        }

        // Softmax over the filtered logit distribution, then sample.
        auto probs = torch::softmax(logits, /*dim=*/-1);
        block_ids = torch::multinomial(probs, /*num_samples=*/1).squeeze(-1);
      }

      // Collect tokens.
      auto ids_cpu = block_ids.cpu().contiguous();
      std::string block_tok_str;
      for (int64_t i = 0; i < ids_cpu.size(0); ++i) {
        all_token_ids.push_back(ids_cpu[i].item<int64_t>());
        if (i < 8) {
          block_tok_str += std::to_string(ids_cpu[i].item<int64_t>()) + " ";
        }
      }
      LOG(INFO) << "Cola-DLM: block " << block_idx << " tokens: ["
                << block_tok_str << "...]";

      // For block 0: log top-3 logits for gen positions (pos 11-15)
      // to diagnose whether "Paris" (token 12366) is near the top.
      if (block_idx == 0 && first_block_prompt_tokens > 0 &&
          first_block_prompt_tokens < block_tokens) {
        auto logits_cpu = block_logits.cpu();
        for (int64_t pos = first_block_prompt_tokens; pos < block_tokens;
             ++pos) {
          auto [top3v, top3i] =
              torch::topk(logits_cpu[pos], 3, /*dim=*/-1, /*largest=*/true);
          // Also look up token 12366 ("Paris") logit specifically
          float paris_logit = logits_cpu[pos][12366].item<float>();
          LOG(INFO) << "Cola-DLM: b0 pos " << pos << " top3_ids=["
                    << top3i[0].item<int64_t>() << ","
                    << top3i[1].item<int64_t>() << ","
                    << top3i[2].item<int64_t>() << "] top3_vals=["
                    << top3v[0].item<float>() << "," << top3v[1].item<float>()
                    << "," << top3v[2].item<float>()
                    << "] paris_logit=" << paris_logit;
        }
      }

      // Stop if pad or EOS token found.
      bool has_stop = false;
      for (int64_t i = 0; i < ids_cpu.size(0); ++i) {
        int64_t tok = ids_cpu[i].item<int64_t>();
        if (tok == kPadTokenId || tok == kEosTokenId) {
          has_stop = true;
          break;
        }
      }
      if (has_stop) {
        LOG(INFO) << "Cola-DLM: stop token found at block " << block_idx;
        // --- Commit denoised block to DiT KV cache at timestep=0 ---
        // (so future blocks see correct K/V even if we're stopping; matches
        // official Python which always commits before checking stop).
        {
          auto ts_zero = torch::zeros(
              {block_size},
              torch::TensorOptions().dtype(torch::kFloat32).device(device));
          at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
          at::autocast::set_autocast_enabled(at::kCUDA, true);
          dit_->forward(txt.to(torch::kBFloat16),
                        k_lens_cond,
                        q_lens,
                        ts_zero,
                        /*update_kv=*/true,
                        /*use_kv_cache=*/true);
          at::autocast::set_autocast_enabled(at::kCUDA, false);
        }
        break;
      }

      // --- Commit denoised block to DiT KV cache at timestep=0 ---
      // (matches official Python inference.py lines 690-700)
      {
        auto ts_zero = torch::zeros(
            {block_size},
            torch::TensorOptions().dtype(torch::kFloat32).device(device));
        at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
        at::autocast::set_autocast_enabled(at::kCUDA, true);
        dit_->forward(txt.to(torch::kBFloat16),
                      k_lens_cond,
                      q_lens,
                      ts_zero,
                      /*update_kv=*/true,
                      /*use_kv_cache=*/true);
        at::autocast::set_autocast_enabled(at::kCUDA, false);
      }
    }

    // -----------------------------------------------------------------
    // Step 6: Clean up KV caches.
    // -----------------------------------------------------------------
    dit_->set_kv_cache(false);
    vae_->set_kv_cache(false);

    LOG(INFO) << "Cola-DLM: generation done, all_token_ids.size()="
              << all_token_ids.size();

    // -----------------------------------------------------------------
    // Step 7: Trim leading prompt tokens from first block, detokenize.
    // -----------------------------------------------------------------
    int64_t trim_count =
        std::max(int64_t{0},
                 std::min(first_block_prompt_tokens,
                          static_cast<int64_t>(all_token_ids.size())));

    std::vector<int32_t> output_tokens(all_token_ids.begin() + trim_count,
                                       all_token_ids.end());

    std::string output_text;
    if (tokenizer_) {
      output_text =
          tokenizer_->decode(output_tokens, /*skip_special_tokens=*/true);
    }

    LOG(INFO) << "Cola-DLM: trim_count=" << trim_count
              << ", output_tokens.size()=" << output_tokens.size()
              << ", output_text='" << output_text << "'";

    DiTForwardOutput output;
    output.text_output.push_back(output_text);
    return output;
  }

 private:
  torch::TensorOptions options_;
  ColaDiTTransformer dit_{nullptr};
  ColaTextVAEModel vae_{nullptr};
  std::unique_ptr<Tokenizer> tokenizer_;
  int64_t block_size_ = 4;
  // Monotonic per-forward counter used to derive unique block noise seeds
  // when the caller has not specified an explicit seed (seed == 0).
  std::atomic<uint64_t> forward_counter_{0};
};
TORCH_MODULE(ColaDLMPipeline);

// ---------------------------------------------------------------------------
// Register Cola-DLM pipeline
// ---------------------------------------------------------------------------

REGISTER_DIT_MODEL(cola_dlm, ColaDLMPipeline);

}  // namespace xllm
