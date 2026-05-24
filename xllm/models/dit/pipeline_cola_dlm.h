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

#include <torch/torch.h>

#include <filesystem>
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
      // Use bf16 to match Python's torch.autocast("cuda", dtype=torch.bfloat16)
      // which makes all linear layers compute in bf16.
      dit_->to(torch::kBFloat16);

      vae_->load_model(std::move(vae_loader));
      vae_->to(options_.device());
      vae_->to(torch::kBFloat16);

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

    // Generation parameters
    int64_t max_new_tokens = gp.max_new_tokens > 0 ? gp.max_new_tokens : 256;
    int32_t diffusion_steps = gp.diffusion_steps > 0 ? gp.diffusion_steps : 16;
    float temperature = gp.temperature;
    int32_t top_k = gp.top_k;
    float top_p = gp.top_p;
    float guidance_scale = gp.guidance_scale > 0 ? gp.guidance_scale : 7.0f;
    int64_t seed = gp.seed;

    int64_t block_size = block_size_;
    int64_t patch_size = vae_->patch_size();
    int64_t chunk = patch_size * block_size;

    // Diffusion timesteps: T=1000, linearly spaced
    constexpr float T = 1000.0f;
    auto timesteps =
        torch::linspace(T, 0.0f, diffusion_steps + 1, torch::kFloat32);

    // -----------------------------------------------------------------
    // Step 1: Tokenize prompt
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
      LOG(ERROR) << "Cola-DLM: empty tokens after encoding, returning empty "
                    "output";
      DiTForwardOutput output;
      output.text_output.push_back("");
      return output;
    }

    // Pad to multiple of chunk (patch_size * block_size)
    int64_t orig_len = tokens.size();
    int64_t pad_len = (chunk - orig_len % chunk) % chunk;
    // Cola-DLM uses pad_token_id = 100277 for the OLMo tokenizer
    constexpr int32_t kPadTokenId = 100277;
    for (int64_t i = 0; i < pad_len; ++i) {
      tokens.push_back(kPadTokenId);
    }

    auto input_ids =
        torch::tensor(tokens, torch::dtype(torch::kLong)).to(device);

    // Token labels: 1 for real tokens, 3 for padding
    std::vector<int64_t> token_labels_vec;
    token_labels_vec.resize(orig_len, 1);
    token_labels_vec.resize(orig_len + pad_len, 3);
    auto token_labels =
        torch::tensor(token_labels_vec, torch::dtype(torch::kLong)).to(device);

    // -----------------------------------------------------------------
    // Step 2: VAE encode
    // -----------------------------------------------------------------
    std::vector<torch::Tensor> input_ids_list = {input_ids};
    auto [latents, sample_lens] = vae_->encode(input_ids_list);
    // Match Python: latents are fp32 after encode (Python does .float())
    latents = latents.to(torch::kFloat32);
    LOG(INFO) << "Cola-DLM: VAE encode done, latents shape=[" << latents.size(0)
              << ", " << latents.size(1) << "], sample_lens=[" << sample_lens[0]
              << "]";

    // Apply scaling and shifting (already done in VAE encode, but the
    // Python code does it again with explicit shift/scale)
    // The VAE encode() already applies (z - shifting) * scaling internally.

    // -----------------------------------------------------------------
    // Step 3: Compute latent labels and prefix/first-block split
    // -----------------------------------------------------------------
    int64_t n_patches = latents.size(0);

    // Derive latent labels from token labels
    // For patch_size=1: latent_labels == token_labels
    // For patch_size>1: label 1 wins over 2 wins over 3
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

    // Count real (label 1) latents
    int64_t num_real = latent_labels.eq(1).sum().item<int64_t>();

    // Split into prefix and first-generation-block
    torch::Tensor prefix_latents;
    torch::Tensor first_block_latents;
    int64_t first_block_prompt_tokens = 0;

    if (num_real % block_size != 0) {
      // Need to fill the partial block
      int64_t prefix_fill = block_size - (num_real % block_size);
      int64_t start_idx = (num_real / block_size) * block_size;

      prefix_latents = latents.slice(0, 0, start_idx);
      first_block_latents =
          latents.slice(0, start_idx, start_idx + block_size).clone();

      // Count how many prompt tokens are in the first generation block
      int64_t token_start = start_idx * patch_size;
      int64_t token_end = std::min(token_start + block_size * patch_size,
                                   static_cast<int64_t>(token_labels.size(0)));
      first_block_prompt_tokens = token_labels.slice(0, token_start, token_end)
                                      .eq(1)
                                      .sum()
                                      .item<int64_t>();
    } else {
      prefix_latents = latents.slice(0, 0, num_real);
      // Use random placeholder for first block (will be overwritten)
      first_block_latents =
          torch::zeros({block_size, latents.size(1)}, latents.options());
    }

    int64_t prefix_len = prefix_latents.size(0);

    // -----------------------------------------------------------------
    // Step 4: Block-wise generation loop (NO KV cache for simplicity)
    // -----------------------------------------------------------------
    // Each iteration generates one block of block_size tokens.
    // The DiT runs a full forward pass each time (no KV cache).
    // We track committed latents so the conditional pass sees real values.
    int64_t max_blocks = (max_new_tokens + block_size * patch_size - 1) /
                         (block_size * patch_size);

    LOG(INFO) << "Cola-DLM: n_patches=" << n_patches
              << ", num_real=" << num_real << ", prefix_len=" << prefix_len
              << ", block_size=" << block_size << ", patch_size=" << patch_size
              << ", max_blocks=" << max_blocks
              << ", first_block_prompt_tokens=" << first_block_prompt_tokens;

    int64_t latent_dim = latents.size(1);
    std::vector<int64_t> all_token_ids;
    std::vector<torch::Tensor> committed_latents;  // denoised blocks

    // Set RNG seed for reproducibility
    if (seed > 0) {
      torch::manual_seed(seed);
    }

    for (int64_t block_idx = 0; block_idx < max_blocks; ++block_idx) {
      int64_t current_q_len = block_size;
      LOG(INFO) << "Cola-DLM: block " << block_idx << "/" << max_blocks;

      // For the conditional pass, K = prefix + all committed + current
      int64_t cond_k_len =
          prefix_len +
          static_cast<int64_t>(committed_latents.size()) * block_size +
          block_size;
      std::vector<int64_t> k_lens_cond = {cond_k_len};
      std::vector<int64_t> q_lens = {current_q_len};

      // For the unconditional pass, K = Q = current block only
      std::vector<int64_t> k_lens_uncond = {current_q_len};

      // CFG scale: use 1.0 for first block when prefix is empty,
      // since cond and uncond see the same input (no CFG effect).
      float block_cfg_scale =
          (block_idx == 0 && prefix_len == 0) ? 1.0f : guidance_scale;

      // Draw noise for this block
      auto txt = torch::randn({block_size, latent_dim}, latents.options());

      // Euler integration loop
      for (int64_t t_idx = 0; t_idx < diffusion_steps; ++t_idx) {
        float t_curr = timesteps[t_idx].item<float>();
        float t_next = timesteps[t_idx + 1].item<float>();
        float dt = (t_curr - t_next) / T;

        // Pin prompt positions: set timestep=0 and restore ground truth
        // latents at EVERY denoising step (not just the first).
        if (block_idx == 0 && first_block_prompt_tokens > 0) {
          txt.slice(0, 0, first_block_prompt_tokens) =
              first_block_latents.slice(0, 0, first_block_prompt_tokens);
        }

        auto ts_tensor = torch::full(
            {block_size},
            t_curr,
            torch::TensorOptions().dtype(torch::kFloat32).device(device));

        // Prompt positions are always clean (t=0)
        if (block_idx == 0 && first_block_prompt_tokens > 0) {
          ts_tensor.slice(0, 0, first_block_prompt_tokens).fill_(0);
        }

        // Conditional pass: attend to prefix + committed + current
        torch::Tensor drift_cond;
        {
          std::vector<torch::Tensor> parts;
          if (prefix_len > 0) {
            parts.push_back(prefix_latents.to(torch::kFloat32));
          }
          for (auto& cl : committed_latents) {
            parts.push_back(cl.to(torch::kFloat32));
          }
          parts.push_back(txt);

          auto full_txt = torch::cat(parts, /*dim=*/0);
          auto full_ts = torch::full(
              {full_txt.size(0)},
              t_curr,
              torch::TensorOptions().dtype(torch::kFloat32).device(device));

          // Set timestep=0 for prefix and committed positions (already
          // denoised). This matches Python's KV cache behavior where
          // prefix prefetch and block commit both use timestep=0.
          int64_t committed_len =
              static_cast<int64_t>(committed_latents.size()) * block_size;
          int64_t non_current_len = prefix_len + committed_len;
          if (non_current_len > 0) {
            full_ts.slice(0, 0, non_current_len).fill_(0);
          }

          // For first block, prompt tokens within current block also
          // get timestep=0 (matching Python's ts_batch[flat_mask] = 0).
          if (block_idx == 0 && first_block_prompt_tokens > 0) {
            full_ts
                .slice(0,
                       non_current_len,
                       non_current_len + first_block_prompt_tokens)
                .fill_(0);
          }

          drift_cond = dit_->forward(full_txt, k_lens_cond, q_lens, full_ts);
        }

        // Unconditional pass: attend to current block only
        torch::Tensor drift_uncond;
        {
          drift_uncond = dit_->forward(txt, k_lens_uncond, q_lens, ts_tensor);
        }

        // CFG combination
        auto drift =
            block_cfg_scale * (drift_cond - drift_uncond) + drift_uncond;
        txt = txt - drift * dt;

        // Re-pin prompt positions after Euler step
        if (block_idx == 0 && first_block_prompt_tokens > 0) {
          txt.slice(0, 0, first_block_prompt_tokens) =
              first_block_latents.slice(0, 0, first_block_prompt_tokens);
        }
      }

      // Commit this block's denoised latents
      auto txt_cpu = txt.detach().cpu();
      LOG(INFO) << "Cola-DLM: block " << block_idx
                << " denoised latent stats: mean="
                << txt_cpu.mean().item<float>()
                << ", std=" << txt_cpu.std().item<float>()
                << ", min=" << txt_cpu.min().item<float>()
                << ", max=" << txt_cpu.max().item<float>();
      committed_latents.push_back(txt.detach().clone());

      // -----------------------------------------------------------------
      // Step 5: VAE decode this block
      // -----------------------------------------------------------------
      int64_t decoder_k_len =
          prefix_len +
          static_cast<int64_t>(committed_latents.size()) * block_size;
      std::vector<int64_t> dec_k_lens = {decoder_k_len};
      std::vector<int64_t> dec_q_lens = {current_q_len};

      // Build full sequence for decoder (no KV cache)
      torch::Tensor decode_input;
      {
        std::vector<torch::Tensor> parts;
        if (prefix_len > 0) {
          parts.push_back(prefix_latents.to(torch::kFloat32));
        }
        for (auto& cl : committed_latents) {
          parts.push_back(cl.to(torch::kFloat32));
        }
        decode_input = torch::cat(parts, /*dim=*/0);
      }

      auto decoded = vae_->decode(decode_input, dec_k_lens, dec_q_lens);
      LOG(INFO) << "Cola-DLM: VAE decode done, decoded shape=["
                << decoded.size(0) << ", " << decoded.size(1) << ", "
                << decoded.size(2) << "]";

      // Take only the last block_size * patch_size tokens
      int64_t block_tokens = block_size * patch_size;
      auto block_logits = decoded.slice(1, decoded.size(1) - block_tokens);

      // Greedy decoding (temperature ≈ 0) or sampling
      torch::Tensor block_ids;
      if (temperature < 1e-5f) {
        block_ids = block_logits.argmax(/*dim=*/-1);  // (1, block_tokens)
      } else {
        auto probs = torch::softmax(block_logits / temperature, /*dim=*/-1);
        if (top_k > 0) {
          auto [topk_values, topk_indices] =
              torch::topk(probs, top_k, /*dim=*/-1);
          probs =
              torch::zeros_like(probs).scatter_(-1, topk_indices, topk_values);
        }
        block_ids = torch::multinomial(probs.reshape({-1, probs.size(-1)}),
                                       /*num_samples=*/1);
        block_ids = block_ids.reshape({1, block_tokens});
      }

      // Collect generated token IDs
      auto ids_vec = block_ids.squeeze(0).cpu().contiguous();
      std::string block_tok_str;
      for (int64_t i = 0; i < ids_vec.size(0); ++i) {
        all_token_ids.push_back(ids_vec[i].item<int64_t>());
        if (i < 8) {
          block_tok_str += std::to_string(ids_vec[i].item<int64_t>()) + " ";
        }
      }
      LOG(INFO) << "Cola-DLM: block " << block_idx << " tokens: ["
                << block_tok_str << "...]";

      // Check for pad tokens (stop condition)
      bool has_stop = false;
      for (int64_t i = 0; i < ids_vec.size(0); ++i) {
        int64_t tok = ids_vec[i].item<int64_t>();
        if (tok == kPadTokenId) {
          has_stop = true;
          break;
        }
      }
      if (has_stop) {
        LOG(INFO) << "Cola-DLM: stop token found at block " << block_idx;
        break;
      }
    }

    LOG(INFO) << "Cola-DLM: generation done, all_token_ids.size()="
              << all_token_ids.size();

    // -----------------------------------------------------------------
    // Step 6: Detokenize output
    // -----------------------------------------------------------------
    // Remove leading prompt tokens from the first block
    // When prefix_len > 0, the first block contains only generated tokens.
    // When prefix_len == 0, the first block contains prompt + generated tokens.
    int64_t trim_count =
        std::max(0L,
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
};
TORCH_MODULE(ColaDLMPipeline);

// ---------------------------------------------------------------------------
// Register Cola-DLM pipeline
// ---------------------------------------------------------------------------

REGISTER_DIT_MODEL(cola_dlm, ColaDLMPipeline);

}  // namespace xllm
