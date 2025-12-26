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

#include "qwen2_vl_image_processor.h"

namespace xllm {

namespace {

using Size = std::pair<int, int>;
std::optional<Size> smart_resize(int height,
                                 int width,
                                 int factor = 28,
                                 int min_pixels = 56 * 56,
                                 int max_pixels = 14 * 14 * 4 * 1280) {
  if (static_cast<double>(std::max(height, width)) / std::min(height, width) >
      200) {
    LOG(ERROR) << "Absolute aspect ratio must be smaller than 200";
    return std::nullopt;
  }

  int h_bar =
      static_cast<int>(std::round(height / static_cast<double>(factor))) *
      factor;
  int w_bar =
      static_cast<int>(std::round(width / static_cast<double>(factor))) *
      factor;

  if (h_bar * w_bar > max_pixels) {
    double beta = std::sqrt((height * width) / static_cast<double>(max_pixels));
    h_bar = static_cast<int>(
                std::floor(height / beta / static_cast<double>(factor))) *
            factor;
    w_bar = static_cast<int>(
                std::floor(width / beta / static_cast<double>(factor))) *
            factor;
  } else if (h_bar * w_bar < min_pixels) {
    double beta = std::sqrt(min_pixels / static_cast<double>(height * width));
    h_bar = static_cast<int>(
                std::ceil(height * beta / static_cast<double>(factor))) *
            factor;
    w_bar = static_cast<int>(
                std::ceil(width * beta / static_cast<double>(factor))) *
            factor;
  }

  return std::make_pair(h_bar, w_bar);
}
}  // namespace

torch::Tensor Qwen2VLImageProcessor::sample_frames(
    const VideoMetadata& metadata,
    int temporal_patch_size,
    int min_frames,
    int max_frames,
    int num_frames,
    double set_fps) {
  if (set_fps > 0.0 && num_frames > 0) {
    LOG(FATAL) << "num_frames and fps are mutually exclusive arguments, please "
                  "use only one!";
  }

  double fps = set_fps;

  int total_num_frames = metadata.total_num_frames;

  if (num_frames > 0) {
    double double_num_frames =
        std::round(static_cast<double>(num_frames) / temporal_patch_size) *
        temporal_patch_size;
    num_frames = static_cast<int>(double_num_frames);
  } else if (fps > 0.0) {
    if (metadata.fps <= 0.0) {
      LOG(FATAL)
          << "Asked to sample `fps` frames per second but no video metadata "
             "was provided which is required when sampling with `fps`. ";
    }

    max_frames =
        (std::min(max_frames, total_num_frames) / temporal_patch_size) *
        temporal_patch_size;
    double double_num_frames =
        static_cast<double>(total_num_frames) / metadata.fps * fps;
    double_num_frames = std::min(
        std::min(std::max(double_num_frames, static_cast<double>(min_frames)),
                 static_cast<double>(max_frames)),
        static_cast<double>(total_num_frames));
    double_num_frames = std::floor(double_num_frames / temporal_patch_size) *
                        temporal_patch_size;

    num_frames = static_cast<int>(double_num_frames);
  }

  if (num_frames > total_num_frames) {
    LOG(FATAL) << "Video can't be sampled. The inferred num_frames="
               << num_frames << " exceeds total_num_frames=" << total_num_frames
               << ".";
  }

  if (num_frames > 0) {
    std::vector<int64_t> indices;
    indices.reserve(num_frames);
    for (int i = 0; i < num_frames; ++i) {
      int64_t k = static_cast<int64_t>(
          (static_cast<int64_t>(i) * total_num_frames) / num_frames);
      if (k >= total_num_frames) k = total_num_frames - 1;
      indices.push_back(k);
    }
    return torch::tensor(indices, torch::TensorOptions().dtype(torch::kLong));
  } else {
    return torch::arange(0,
                         static_cast<int64_t>(total_num_frames),
                         torch::TensorOptions().dtype(torch::kLong));
  }
}

Qwen2VLImageProcessor::Qwen2VLImageProcessor(const ModelArgs& args) {
  image_mean_ = args.mm_image_normalize_mean();
  image_std_ = args.mm_image_normalize_std();
  if (args.mm_image_max_pixels() && args.mm_image_min_pixels()) {
    min_pixels_ = args.mm_image_min_pixels();
    max_pixels_ = args.mm_image_max_pixels();
  } else if (args.mm_image_shortest_edge() && args.mm_image_longest_edge()) {
    min_pixels_ = args.mm_image_shortest_edge();
    max_pixels_ = args.mm_image_longest_edge();
  }
  patch_size_ = args.mm_image_patch_size();
  temporal_patch_size_ = args.mm_image_temporal_patch_size();

  merge_size_ = args.mm_image_merge_size();
  size_ = {{"longest_edge", 12845056}, {"shortest_edge", 3136}};

  // fuse image mean/std and rescale_factor
  if (do_rescale_ && do_normalize_) {
    for (auto& item : image_mean_) {
      item = item * (1.0 / rescale_factor_);
    }

    for (auto& item : image_std_) {
      item = item * (1.0 / rescale_factor_);
    }

    do_rescale_ = false;
  }
}

bool Qwen2VLImageProcessor::process(const MMInput& inputs, MMData& datas) {
  std::vector<torch::Tensor> images = inputs.get_decode_data(MMType::IMAGE);
  std::vector<torch::Tensor> videos = inputs.get_decode_data(MMType::VIDEO);
  std::vector<VideoMetadata> video_meta_list = inputs.get_video_metadata();

  if (images.empty() && (videos.empty() || video_meta_list.empty())) {
    LOG(ERROR) << "no image/video tensor found.";
    return false;
  }

  if (!images.empty()) {
    if (!this->process_images(images, datas)) {
      LOG(ERROR) << " process image failed.";
      return false;
    }
  }

  if (!videos.empty()) {
    if (!this->process_videos(videos, video_meta_list, datas)) {
      LOG(ERROR) << " process video failed.";
      return false;
    }
  }

  return true;
}

bool Qwen2VLImageProcessor::process_images(std::vector<torch::Tensor> images,
                                           MMData& mm_datas) {
  torch::Tensor pixel_values;
  torch::Tensor thw;

  for (const auto& img : images) {
    if (!this->process_image(img, pixel_values, thw)) {
      return false;
    }

    auto& item = mm_datas.add(MMType::IMAGE);
    item.set_data({{"pixel_values", pixel_values}, {"image_grid_thw", thw}});
  }

  return true;
}

bool Qwen2VLImageProcessor::process_image(torch::Tensor image,
                                          torch::Tensor& pixel_values,
                                          torch::Tensor& thw) {
  auto shape = image.sizes();

  auto resized_height = shape[1];
  auto resized_width = shape[2];

  // do_convert_rgb

  // resize
  if (do_resize_) {
    auto size = smart_resize(resized_height,
                             resized_width,
                             patch_size_ * merge_size_,
                             min_pixels_,
                             max_pixels_);
    // size_["shortest_edge"],
    // size_["longest_edge"]);
    if (!size) {
      return false;
    }

    std::tie(resized_height, resized_width) = *size;
    image =
        this->resize(image, {resized_height, resized_width}, resample_, false);
  }

  // normalize
  if (do_normalize_) {
    image = this->normalize(image, image_mean_, image_std_);
  }

  // rescale
  if (do_rescale_) {
    image = this->rescale(image, rescale_factor_);
  }

  auto patches = torch::stack({image}, 0);
  auto repeats =
      patches[-1].unsqueeze(0).repeat({temporal_patch_size_ - 1, 1, 1, 1});
  patches = torch::cat({patches, repeats}, 0);
  shape = patches.sizes();

  auto channel = shape[1];
  auto grid_t = shape[0] / temporal_patch_size_;

  auto grid_h = resized_height / patch_size_;
  auto grid_w = resized_width / patch_size_;

  patches = patches.view({grid_t,
                          temporal_patch_size_,
                          channel,
                          grid_h / merge_size_,
                          merge_size_,
                          patch_size_,
                          grid_w / merge_size_,
                          merge_size_,
                          patch_size_});

  patches = patches.permute({0, 3, 6, 4, 7, 2, 1, 5, 8});
  patches = patches.reshape(
      {grid_t * grid_h * grid_w,
       channel * temporal_patch_size_ * patch_size_ * patch_size_});

  pixel_values = patches;
  thw = torch::tensor({grid_t, grid_h, grid_w}).clone().reshape({-1, 3});

  return true;
}

bool Qwen2VLImageProcessor::process_videos(
    std::vector<torch::Tensor> videos,
    std::vector<VideoMetadata> video_meta_list,
    MMData& mm_datas) {
  torch::Tensor pixel_values;
  torch::Tensor thw;

  auto opts = torch::TensorOptions().dtype(torch::kFloat32);

  const size_t video_size = videos.size();
  for (size_t i = 0; i < video_size; ++i) {
    auto& vid = videos[i];
    auto& metadata = video_meta_list[i];
    if (!this->process_video(vid, metadata, pixel_values, thw)) {
      return false;
    }

    double fps =
        metadata.sampled_fps > 0.0 ? metadata.sampled_fps : metadata.fps;
    double seconds_per_grid = static_cast<double>(temporal_patch_size_) / fps;
    auto second_per_grid_ts = torch::tensor({seconds_per_grid}, opts);

    auto& item = mm_datas.add(MMType::VIDEO);
    item.set_data({{"pixel_values_videos", pixel_values},
                   {"video_grid_thw", thw},
                   {"second_per_grid_ts", second_per_grid_ts}});

    item.set_metadata(metadata);
  }

  return true;
}

bool Qwen2VLImageProcessor::process_video(torch::Tensor origin_video,
                                          VideoMetadata& metadata,
                                          torch::Tensor& pixel_values,
                                          torch::Tensor& thw) {
  if (origin_video.dim() != 4) {
    LOG(FATAL) << "video must be TCHW";
  }

  torch::Tensor indices;
  if (do_sample_frame_) {
    indices = this->sample_frames(metadata,
                                  temporal_patch_size_,
                                  min_frames_,
                                  max_frames_,
                                  /*num_frames=*/-1,
                                  /*set_fps=*/2.0);
  } else {
    indices = torch::arange(0,
                            static_cast<int64_t>(origin_video.size(0)),
                            torch::TensorOptions().dtype(torch::kLong));
  }
  auto video = origin_video.index_select(/*dim=*/0, indices);
  int64_t sampled_total_frames = video.size(0);

  metadata.frame_indices = indices;
  metadata.timestamps.clear();
  metadata.timestamps.reserve(static_cast<size_t>(sampled_total_frames));
  double fps_for_ts = (metadata.fps > 0.0) ? metadata.fps : 24.0;
  for (int64_t i = 0; i < sampled_total_frames; ++i) {
    int64_t frame_idx = metadata.frame_indices[i].item<int64_t>();
    metadata.timestamps.push_back(static_cast<double>(frame_idx) / fps_for_ts);
  }

  if (metadata.total_num_frames > 0 && metadata.fps > 0.0) {
    metadata.sampled_fps = double(sampled_total_frames) /
                           double(metadata.total_num_frames) * metadata.fps;
  } else {
    metadata.sampled_fps = fps_for_ts;
  }

  auto shape = video.sizes();
  auto time_len = shape[0];
  auto channel = shape[1];
  auto resized_height = shape[2];
  auto resized_width = shape[3];

  if (do_resize_) {
    auto size = smart_resize(resized_height,
                             resized_width,
                             patch_size_ * merge_size_,
                             size_["shortest_edge"],
                             size_["longest_edge"]);
    if (!size) {
      return false;
    }
    std::tie(resized_height, resized_width) = *size;
  }

  std::vector<torch::Tensor> out_frames;
  out_frames.reserve(time_len);
  // for each frame
  auto frames = video.unbind(0);
  for (auto& frame : frames) {
    // resize
    if (do_resize_)
      frame =
          this->resize(frame, {resized_height, resized_width}, resample_, true);
    // normalize
    if (do_normalize_) frame = this->normalize(frame, image_mean_, image_std_);
    // rescale
    if (do_rescale_) frame = this->rescale(frame, rescale_factor_);
    out_frames.push_back(frame);
  }

  auto out_video = torch::stack(out_frames);  // [T,C,H,W]

  auto pad_t = (temporal_patch_size_ - (time_len % temporal_patch_size_)) %
               temporal_patch_size_;
  if (pad_t > 0) {
    auto last =
        out_video.index({time_len - 1}).unsqueeze(0).repeat({pad_t, 1, 1, 1});
    out_video = torch::cat({out_video, last}, 0);
  }

  shape = out_video.sizes();
  auto grid_h = resized_height / patch_size_;
  auto grid_w = resized_width / patch_size_;
  auto grid_t = shape[0] / temporal_patch_size_;

  out_video = out_video.contiguous();

  auto patches = out_video.view({grid_t,
                                 temporal_patch_size_,
                                 channel,
                                 grid_h / merge_size_,
                                 merge_size_,
                                 patch_size_,
                                 grid_w / merge_size_,
                                 merge_size_,
                                 patch_size_});

  patches = patches.permute({0, 3, 6, 4, 7, 2, 1, 5, 8});
  patches = patches.reshape(
      {grid_t * grid_h * grid_w,
       channel * temporal_patch_size_ * patch_size_ * patch_size_});

  pixel_values = patches;
  thw = torch::tensor({grid_t, grid_h, grid_w}).clone().reshape({-1, 3});

  return true;
}

}  // namespace xllm
