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

#include "feature_extractor.h"

namespace xllm {

FeatureExtractor::FeatureExtractor(int64_t feature_size,
                                   int64_t sampling_rate,
                                   std::optional<double> padding_value,
                                   const std::string& padding_side,
                                   bool return_attention_mask)
    : feature_size_(feature_size),
      sampling_rate_(sampling_rate),
      padding_value_(padding_value),
      padding_side_(padding_side),
      return_attention_mask_(return_attention_mask) {}

/*
Args:
   processed_features: BatchFeature
       A data class that contains input_features and masks
   input_features: vector<torch::Tensor>
       input: input_features is expected to be a vector<torch::Tensor> (tensor,
n), with tensor shape [feature_len, 1]

       output: updata input_features to a  vector<torch::Tensor> (tensor, 1),
       with tensor shape [n, pad_feature_len, 1]
*/
void FeatureExtractor::pad(BatchFeature& processed_features,
                           PaddingStrategy padding,
                           std::optional<int64_t> max_length,
                           bool truncation,
                           std::optional<int64_t> pad_to_multiple_of,
                           std::optional<bool> return_attention_mask) {
  CHECK(!model_input_names_.empty() && model_input_names_.size() > 0)
      << "model_input_names should be a vector of string that contains the "
         "input name, "
      << "but got an empty vector";

  std::string main_input_name = model_input_names_[0];
  const auto& input =
      processed_features.get<std::vector<torch::Tensor>>(main_input_name);

  CHECK(input.has_value()) << "You should supply an instance of BatchFeatures "
                              "to this method that contains "
                           << main_input_name;

  const auto& required_input = input.value();

  return_attention_mask = return_attention_mask || return_attention_mask_;

  if (required_input.empty()) {
    if (return_attention_mask) {
      processed_features.add("attention_mask", std::vector<torch::Tensor>());
    }
  }

  check_padding_strategy_(padding, max_length);

  size_t batch_size = required_input.size();

  std::vector<BatchFeature> truncated_inputs;
  const BatchFeatureDict& data = processed_features.data();
  for (size_t i = 0; i < batch_size; i++) {
    BatchFeature inputs_slice;
    for (const auto& [key, value] : data) {
      inputs_slice.add(key, std::get<std::vector<torch::Tensor>>(value)[i]);
    }

    truncate_(inputs_slice, max_length, pad_to_multiple_of, truncation);
    truncated_inputs.push_back(inputs_slice);
  }

  if (padding == PaddingStrategy::LONGEST) {
    max_length = 0;
    for (const auto& input_slice : truncated_inputs) {
      const auto& input =
          input_slice.get<torch::Tensor>(main_input_name).value();
      max_length =
          std::max(max_length.value(), static_cast<int64_t>(input.size(0)));
    }
    padding = PaddingStrategy::MAX_LENGTH;
  }

  std::map<std::string, std::vector<torch::Tensor>> batch_outputs;

  for (size_t i = 0; i < batch_size; i++) {
    pad_(truncated_inputs[i],
         padding,
         max_length,
         pad_to_multiple_of,
         return_attention_mask);

    const BatchFeatureDict& data = truncated_inputs[i].data();
    for (const auto& [key, value] : data) {
      const auto& itor = batch_outputs.find(key);
      if (itor != batch_outputs.end()) {
        itor->second.push_back(std::get<torch::Tensor>(value));
      } else {
        std::vector<torch::Tensor> value_vec{std::get<torch::Tensor>(value)};
        batch_outputs.insert({key, std::move(value_vec)});
      }
    }
  }
  for (const auto& [key, value] : batch_outputs) {
    torch::Tensor stacked = torch::stack(value, 0);
    std::vector<torch::Tensor> stack_vec{stacked};
    processed_features.update(key, stack_vec);
  }
}

void FeatureExtractor::pad_(BatchFeature& processed_features,
                            PaddingStrategy padding_strategy,
                            std::optional<int64_t> max_length,
                            std::optional<int64_t> pad_to_multiple_of,
                            std::optional<bool> return_attention_mask) {
  std::string main_input_name = model_input_names_[0];

  torch::Tensor required_input;
  if (auto res = processed_features.get<torch::Tensor>(main_input_name))
    required_input = res.value();

  if (padding_strategy == PaddingStrategy::LONGEST)
    max_length = required_input.size(0);

  if (max_length.has_value() && pad_to_multiple_of.has_value() &&
      (max_length.value() % pad_to_multiple_of.value() != 0)) {
    max_length = ((max_length.value() / pad_to_multiple_of.value()) + 1) *
                 pad_to_multiple_of.value();
  }

  bool needs_padding =
      padding_strategy != PaddingStrategy::DO_NOT_PAD &&
      static_cast<int64_t>(required_input.size(0)) < max_length.value();

  if (return_attention_mask.has_value() && return_attention_mask.value() &&
      !processed_features.has("attention_mask")) {
    auto attention_mask = torch::ones({required_input.size(0)}, torch::kInt32);
    processed_features.add("attention_mask", attention_mask);
  }

  if (needs_padding) {
    CHECK(max_length.has_value())
        << "max_length expected to be a not null value when using padding";
    long difference = max_length.value() - required_input.size(0);

    torch::Tensor attention_mask;
    if (auto res = processed_features.get<torch::Tensor>("attention_mask"))
      attention_mask = res.value();

    if (padding_side_ == "right") {
      if (return_attention_mask.has_value() && return_attention_mask.value()) {
        auto padded_mask =
            torch::constant_pad_nd(attention_mask, {0, difference}, 0);
        processed_features.update("attention_mask", padded_mask);
      }

      std::vector<int64_t> pad_shape;
      if (feature_size_ > 1) {
        pad_shape = {0, 0, 0, difference};
      } else {
        pad_shape = {0, difference};
      }

      auto padded_input = torch::constant_pad_nd(
          required_input, pad_shape, padding_value_.value());
      processed_features.update(main_input_name, padded_input);

    } else if (padding_side_ == "left") {
      if (return_attention_mask) {
        auto padded_mask =
            torch::constant_pad_nd(attention_mask, {difference, 0}, 0);
        processed_features.update("attention_mask", padded_mask);
      }

      std::vector<int64_t> pad_shape;
      if (feature_size_ > 1) {
        pad_shape = {0, 0, difference, 0};
      } else {
        pad_shape = {difference, 0};
      }

      auto padded_input = torch::constant_pad_nd(
          required_input, pad_shape, padding_value_.value());
      processed_features.update(main_input_name, padded_input);

    } else {
      CHECK(false) << "Invalid padding side: " << padding_side_;
    }
  }
}

void FeatureExtractor::truncate_(BatchFeature& processed_features,
                                 std::optional<int64_t> max_length,
                                 std::optional<int64_t> pad_to_multiple_of,
                                 std::optional<bool> truncation) {
  if (!truncation.has_value() || !truncation.value()) {
    return;
  }

  if (truncation.value() && !max_length.has_value()) {
    CHECK(false) << "When setting truncation=true, max_length must be defined";
  }

  std::string main_input_name = model_input_names_[0];

  torch::Tensor required_input;
  if (auto res = processed_features.get<torch::Tensor>(main_input_name))
    required_input = res.value();

  if (max_length.has_value() && pad_to_multiple_of.has_value() &&
      (max_length.value() % pad_to_multiple_of.value() != 0)) {
    max_length = ((max_length.value() / pad_to_multiple_of.value()) + 1) *
                 pad_to_multiple_of.value();
  }

  bool needs_truncation = required_input.size(0) > max_length.value();

  if (needs_truncation) {
    auto truncated_input = required_input.slice(0, 0, max_length.value());
    processed_features.update(main_input_name, truncated_input);

    if (processed_features.has("attention_mask")) {
      torch::Tensor attention_mask;
      if (auto res = processed_features.get<torch::Tensor>("attention_mask"))
        attention_mask = res.value();

      auto truncated_mask = attention_mask.slice(0, 0, max_length.value());
      processed_features.update("attention_mask", truncated_mask);
    }
  }
}

void FeatureExtractor::check_padding_strategy_(
    PaddingStrategy padding,
    std::optional<int64_t> max_length) {
  if (!max_length.has_value() && padding == PaddingStrategy::MAX_LENGTH) {
    CHECK(false) << "When using max_length padding, max_length must be defined";
  }

  if (padding != PaddingStrategy::DO_NOT_PAD && !padding_value_.has_value()) {
    CHECK(false) << "Padding value is not set. Please set a padding_value.";
  }
}

}  // namespace xllm
