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

#include <glog/logging.h>

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/common/message.h"
#include "core/framework/multimodal/mm_input.h"
#include "processors/audio_processor.h"
#include "processors/image_processor.h"
#include "processors/processor_util.h"
#include "processors/prompt_processor.h"
#include "processors/video_processor.h"

namespace xllm {

class JinjaChatTemplate;
class ModelArgs;
class Tokenizer;
struct TokenizerArgs;

struct PreprocessOutput {
  std::string prompt;
  std::vector<int32_t> prompt_tokens;
  MMData mm_data;
  std::string error_message;
};

class MultimodalProcessorBase {
 public:
  virtual ~MultimodalProcessorBase();

  virtual bool preprocess(const std::vector<Message>& messages,
                          const std::vector<JsonTool>& tools,
                          const nlohmann::ordered_json& chat_template_kwargs,
                          const std::string& payload,
                          PreprocessOutput& output) = 0;

  virtual bool preprocess(const std::string& prompt,
                          MMData mm_data,
                          PreprocessOutput& output) = 0;

 protected:
  MultimodalProcessorBase(const TokenizerArgs& tokenizer_args,
                          std::shared_ptr<Tokenizer> tokenizer);

  bool render_prompt(const std::vector<Message>& messages,
                     const std::vector<JsonTool>& tools,
                     const nlohmann::ordered_json& chat_template_kwargs,
                     std::string& prompt,
                     std::string& error_message) const;

  bool encode_prompt(const std::string& prompt,
                     std::vector<int32_t>& prompt_tokens,
                     std::string& error_message) const;

 private:
  std::unique_ptr<JinjaChatTemplate> chat_template_;
  std::shared_ptr<Tokenizer> tokenizer_;
};

std::unique_ptr<MultimodalProcessorBase> create_multimodal_processor(
    const ModelArgs& model_args,
    const TokenizerArgs& tokenizer_args,
    std::shared_ptr<Tokenizer> tokenizer);

class NoImageProcessor final : public ImageProcessor {
 public:
  bool process(const std::vector<torch::Tensor>& images,
               std::vector<MMDataItem>& output_items) const override {
    LOG(ERROR) << "Image processor is not configured.";
    return false;
  }

  MMDict process_embedding(const EmbeddingOutput& embedding) const override {
    LOG(ERROR) << "Image embedding processor is not configured.";
    return MMDict{};
  }
};

class NoVideoProcessor final : public VideoProcessor {
 public:
  bool process(const torch::Tensor& origin_video,
               const VideoMetadata& metadata,
               MMDataItem& output_item) const override {
    LOG(ERROR) << "Video processor is not configured.";
    return false;
  }
};

class NoAudioProcessor final : public AudioProcessor {};

template <typename PromptProcessorT,
          typename ImageProcessorT = NoImageProcessor,
          typename VideoProcessorT = NoVideoProcessor,
          typename AudioProcessorT = NoAudioProcessor>
class MultimodalProcessor final : public MultimodalProcessorBase {
 private:
  static_assert(std::is_base_of_v<PromptProcessor, PromptProcessorT>);
  static_assert(std::is_base_of_v<ImageProcessor, ImageProcessorT>);
  static_assert(std::is_base_of_v<VideoProcessor, VideoProcessorT>);
  static_assert(std::is_base_of_v<AudioProcessor, AudioProcessorT>);

 public:
  MultimodalProcessor(const ModelArgs& model_args,
                      const TokenizerArgs& tokenizer_args,
                      std::shared_ptr<Tokenizer> tokenizer);

  MultimodalProcessor(std::unique_ptr<ImageProcessorT> image_processor,
                      std::unique_ptr<VideoProcessorT> video_processor,
                      std::unique_ptr<AudioProcessorT> audio_processor,
                      std::unique_ptr<PromptProcessorT> prompt_processor,
                      const TokenizerArgs& tokenizer_args,
                      std::shared_ptr<Tokenizer> tokenizer);
  ~MultimodalProcessor() override = default;

  bool preprocess(const std::vector<Message>& messages,
                  const std::vector<JsonTool>& tools,
                  const nlohmann::ordered_json& chat_template_kwargs,
                  const std::string& payload,
                  PreprocessOutput& output) override;
  bool preprocess(const std::string& prompt,
                  MMData mm_data,
                  PreprocessOutput& output) override;

 private:
  bool process_mm_input(const MMInput& inputs, MMData& data) const;

  bool process_images(const std::vector<torch::Tensor>& images,
                      std::vector<MMDataItem>& output_items) const;

  bool process_videos(const std::vector<torch::Tensor>& videos,
                      const std::vector<VideoMetadata>& video_metadata,
                      std::vector<MMDataItem>& output_items) const;

  bool process_audios(const std::vector<torch::Tensor>& audios,
                      const std::vector<AudioMetadata>& audio_metadata,
                      std::vector<MMDataItem>& output_items) const;

  static std::unique_ptr<ImageProcessorT> create_image_processor(
      const ModelArgs& model_args) {
    if constexpr (std::is_same_v<ImageProcessorT, NoImageProcessor>) {
      return std::make_unique<NoImageProcessor>();
    } else {
      return std::make_unique<ImageProcessorT>(model_args);
    }
  }

  static std::unique_ptr<VideoProcessorT> create_video_processor(
      const ModelArgs& model_args) {
    if constexpr (std::is_same_v<VideoProcessorT, NoVideoProcessor>) {
      return std::make_unique<NoVideoProcessor>();
    } else {
      return std::make_unique<VideoProcessorT>(model_args);
    }
  }

  static std::unique_ptr<AudioProcessorT> create_audio_processor(
      const ModelArgs& model_args) {
    if constexpr (std::is_same_v<AudioProcessorT, NoAudioProcessor>) {
      return std::make_unique<NoAudioProcessor>();
    } else {
      return std::make_unique<AudioProcessorT>(model_args);
    }
  }

 private:
  std::unique_ptr<ImageProcessorT> image_processor_;
  std::unique_ptr<VideoProcessorT> video_processor_;
  std::unique_ptr<AudioProcessorT> audio_processor_;
  std::unique_ptr<PromptProcessorT> prompt_processor_;
};

template <typename PromptProcessorT,
          typename ImageProcessorT,
          typename VideoProcessorT,
          typename AudioProcessorT>
MultimodalProcessor<
    PromptProcessorT,
    ImageProcessorT,
    VideoProcessorT,
    AudioProcessorT>::MultimodalProcessor(const ModelArgs& model_args,
                                          const TokenizerArgs& tokenizer_args,
                                          std::shared_ptr<Tokenizer> tokenizer)
    : MultimodalProcessorBase(tokenizer_args, std::move(tokenizer)),
      image_processor_(create_image_processor(model_args)),
      video_processor_(create_video_processor(model_args)),
      audio_processor_(create_audio_processor(model_args)),
      prompt_processor_(std::make_unique<PromptProcessorT>(model_args)) {}

template <typename PromptProcessorT,
          typename ImageProcessorT,
          typename VideoProcessorT,
          typename AudioProcessorT>
MultimodalProcessor<PromptProcessorT,
                    ImageProcessorT,
                    VideoProcessorT,
                    AudioProcessorT>::
    MultimodalProcessor(std::unique_ptr<ImageProcessorT> image_processor,
                        std::unique_ptr<VideoProcessorT> video_processor,
                        std::unique_ptr<AudioProcessorT> audio_processor,
                        std::unique_ptr<PromptProcessorT> prompt_processor,
                        const TokenizerArgs& tokenizer_args,
                        std::shared_ptr<Tokenizer> tokenizer)
    : MultimodalProcessorBase(tokenizer_args, std::move(tokenizer)),
      image_processor_(std::move(image_processor)),
      video_processor_(std::move(video_processor)),
      audio_processor_(std::move(audio_processor)),
      prompt_processor_(std::move(prompt_processor)) {}

template <typename PromptProcessorT,
          typename ImageProcessorT,
          typename VideoProcessorT,
          typename AudioProcessorT>
bool MultimodalProcessor<
    PromptProcessorT,
    ImageProcessorT,
    VideoProcessorT,
    AudioProcessorT>::process_mm_input(const MMInput& inputs,
                                       MMData& data) const {
  std::vector<torch::Tensor> images;
  std::vector<torch::Tensor> videos;
  std::vector<torch::Tensor> audios;
  std::vector<VideoMetadata> video_metadata;
  std::vector<AudioMetadata> audio_metadata;
  std::vector<MMDict> image_embeddings;
  for (const MMInputItem& input_item : inputs) {
    if (input_item.has_type(MMType::IMAGE)) {
      if constexpr (std::is_same_v<ImageProcessorT, NoImageProcessor>) {
        LOG(ERROR)
            << "Received image input, but image modality is not supported.";
        return false;
      }
      if (image_processor_ == nullptr) {
        LOG(ERROR)
            << "Received image input, but image modality is not supported.";
        return false;
      }
      if (input_item.is_embedding()) {
        image_embeddings.push_back(
            image_processor_->process_embedding(input_item.embedding));
      } else {
        images.push_back(input_item.decode_image);
      }
    }
    if (input_item.has_type(MMType::VIDEO)) {
      if constexpr (std::is_same_v<VideoProcessorT, NoVideoProcessor>) {
        LOG(ERROR)
            << "Received video input, but video modality is not supported.";
        return false;
      }
      if (video_processor_ == nullptr) {
        LOG(ERROR)
            << "Received video input, but video modality is not supported.";
        return false;
      }
      videos.push_back(input_item.decode_video);
      video_metadata.push_back(input_item.video_meta);
    }
    if (input_item.has_type(MMType::AUDIO)) {
      if constexpr (std::is_same_v<AudioProcessorT, NoAudioProcessor>) {
        LOG(ERROR)
            << "Received audio input, but audio modality is not supported.";
        return false;
      }
      if (audio_processor_ == nullptr) {
        LOG(ERROR)
            << "Received audio input, but audio modality is not supported.";
        return false;
      }
      audios.push_back(input_item.decode_audio);
      audio_metadata.push_back(input_item.audio_meta);
    }
  }

  std::vector<MMDataItem> image_items;
  if (!images.empty() && !process_images(images, image_items)) {
    LOG(ERROR) << "process image failed.";
    return false;
  }

  std::vector<MMDataItem> video_items;
  if (!videos.empty() && !process_videos(videos, video_metadata, video_items)) {
    LOG(ERROR) << "process video failed.";
    return false;
  }

  std::vector<MMDataItem> audio_items;
  if (!audios.empty() && !process_audios(audios, audio_metadata, audio_items)) {
    LOG(ERROR) << "process audio failed.";
    return false;
  }

  size_t image_index = 0;
  size_t video_index = 0;
  size_t audio_index = 0;
  size_t embedding_index = 0;
  for (const MMInputItem& input_item : inputs) {
    if (input_item.has_type(MMType::IMAGE)) {
      if (input_item.is_embedding()) {
        MMDataItem& item = data.add(MMType::IMAGE);
        item.set_data(image_embeddings[embedding_index]);
        ++embedding_index;
      } else {
        CHECK(image_index < image_items.size());
        data.add(MMType::IMAGE, image_items[image_index]);
        ++image_index;
      }
    }
    if (input_item.has_type(MMType::VIDEO)) {
      CHECK(video_index < video_items.size());
      data.add(MMType::VIDEO, video_items[video_index]);
      ++video_index;
    }
    if (input_item.has_type(MMType::AUDIO)) {
      CHECK(audio_index < audio_items.size());
      data.add(MMType::AUDIO, audio_items[audio_index]);
      ++audio_index;
    }
  }
  return true;
}

template <typename PromptProcessorT,
          typename ImageProcessorT,
          typename VideoProcessorT,
          typename AudioProcessorT>
bool MultimodalProcessor<PromptProcessorT,
                         ImageProcessorT,
                         VideoProcessorT,
                         AudioProcessorT>::
    process_images(const std::vector<torch::Tensor>& images,
                   std::vector<MMDataItem>& output_items) const {
  output_items.clear();
  output_items.assign(images.size(), MMDataItem(MMType::NONE));
  const auto image_buckets = group_images_by_shape(images);
  for (const auto& image_bucket : image_buckets) {
    const ImageBatchBucket& bucket = image_bucket.second;
    std::vector<MMDataItem> batch_items;
    if (!image_processor_->process(bucket.images, batch_items)) {
      LOG(ERROR)
          << "Failed to process image batch. The shape(channels, height, "
             "width) is: "
          << bucket.images[0].sizes();
      return false;
    }
    CHECK(batch_items.size() == bucket.indices.size())
        << "image processor returned mismatched batch result count.";
    const size_t bucket_size = bucket.indices.size();
    for (size_t index = 0; index < bucket_size; ++index) {
      const size_t output_index = bucket.indices[index];
      output_items[output_index] = std::move(batch_items[index]);
    }
  }
  return true;
}

template <typename PromptProcessorT,
          typename ImageProcessorT,
          typename VideoProcessorT,
          typename AudioProcessorT>
bool MultimodalProcessor<PromptProcessorT,
                         ImageProcessorT,
                         VideoProcessorT,
                         AudioProcessorT>::
    process_videos(const std::vector<torch::Tensor>& videos,
                   const std::vector<VideoMetadata>& video_metadata,
                   std::vector<MMDataItem>& output_items) const {
  CHECK(videos.size() == video_metadata.size());
  output_items.clear();
  output_items.reserve(videos.size());
  const size_t video_size = videos.size();
  for (size_t index = 0; index < video_size; ++index) {
    MMDataItem item(MMType::VIDEO);
    if (!video_processor_->process(
            videos[index], video_metadata[index], item)) {
      LOG(ERROR) << "Failed to process video. The shape(num_frames, channels, "
                    "height, width) is: "
                 << videos[index].sizes();
      return false;
    }
    output_items.push_back(std::move(item));
  }
  return true;
}

template <typename PromptProcessorT,
          typename ImageProcessorT,
          typename VideoProcessorT,
          typename AudioProcessorT>
bool MultimodalProcessor<PromptProcessorT,
                         ImageProcessorT,
                         VideoProcessorT,
                         AudioProcessorT>::
    process_audios(const std::vector<torch::Tensor>& audios,
                   const std::vector<AudioMetadata>& audio_metadata,
                   std::vector<MMDataItem>& output_items) const {
  CHECK(audios.size() == audio_metadata.size());
  output_items.clear();
  output_items.reserve(audios.size());
  const size_t audio_size = audios.size();
  for (size_t index = 0; index < audio_size; ++index) {
    MMDataItem item(MMType::AUDIO);
    if (!audio_processor_->process(
            audios[index], audio_metadata[index], item)) {
      LOG(ERROR) << "Failed to process audio. The shape(num_samples) is: "
                 << audios[index].sizes();
      return false;
    }
    output_items.push_back(std::move(item));
  }
  return true;
}

template <typename PromptProcessorT,
          typename ImageProcessorT,
          typename VideoProcessorT,
          typename AudioProcessorT>
bool MultimodalProcessor<PromptProcessorT,
                         ImageProcessorT,
                         VideoProcessorT,
                         AudioProcessorT>::
    preprocess(const std::vector<Message>& messages,
               const std::vector<JsonTool>& tools,
               const nlohmann::ordered_json& chat_template_kwargs,
               const std::string& payload,
               PreprocessOutput& output) {
  static MMInputTransfer mm_input_transfer;

  MMInput mm_inputs(payload);
  MMErrCode code = mm_input_transfer.trans(messages, mm_inputs);
  if (code != MMErrCode::SUCCESS) {
    output.error_message = MMErrToString(code);
    LOG(ERROR) << output.error_message;
    return false;
  }

  if (!mm_inputs.empty() && !process_mm_input(mm_inputs, output.mm_data)) {
    output.error_message = "multimodal processor process failed.";
    LOG(ERROR) << output.error_message;
    return false;
  }
  if (!render_prompt(messages,
                     tools,
                     chat_template_kwargs,
                     output.prompt,
                     output.error_message)) {
    return false;
  }

  prompt_processor_->process(output.prompt, output.mm_data);
  if (!encode_prompt(
          output.prompt, output.prompt_tokens, output.error_message)) {
    return false;
  }
  prompt_processor_->find_mm_spans(output.prompt_tokens, output.mm_data);
  return true;
}

template <typename PromptProcessorT,
          typename ImageProcessorT,
          typename VideoProcessorT,
          typename AudioProcessorT>
bool MultimodalProcessor<PromptProcessorT,
                         ImageProcessorT,
                         VideoProcessorT,
                         AudioProcessorT>::preprocess(const std::string& prompt,
                                                      MMData mm_data,
                                                      PreprocessOutput&
                                                          output) {
  output.prompt = prompt;
  output.mm_data = std::move(mm_data);
  prompt_processor_->process(output.prompt, output.mm_data);
  if (!encode_prompt(
          output.prompt, output.prompt_tokens, output.error_message)) {
    return false;
  }
  prompt_processor_->find_mm_spans(output.prompt_tokens, output.mm_data);
  return true;
}

}  // namespace xllm
