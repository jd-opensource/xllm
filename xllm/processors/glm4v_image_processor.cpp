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

#include "glm4v_image_processor.h"

namespace xllm {

namespace {

using Size = std::pair<int, int>;

std::optional<Size> smart_resize(int num_frames,
                                 int height,
                                 int width,
                                 int temporal_factor,
                                 int factor = 28,
                                 int min_pixels = 56 * 56,
                                 int max_pixels = 14 * 14 * 4 * 1280) {
  if (height < factor || width < factor) {
    LOG(ERROR) << "Height or width must be larger than factor";
    return std::nullopt;
  }
  if (num_frames < temporal_factor) {
    LOG(ERROR) << "t:{num_frames} must be larger than "
                  "temporal_factor:{temporal_factor}";
    return std::nullopt;
  }

  if (static_cast<double>(std::max(height, width)) / std::min(height, width) >
      200) {
    LOG(ERROR) << "Absolute aspect ratio must be smaller than 200";
    return std::nullopt;
  }
  int t_bar = static_cast<int>(std::round(
                  num_frames / static_cast<double>(temporal_factor))) *
              temporal_factor;
  int h_bar =
      static_cast<int>(std::round(height / static_cast<double>(factor))) *
      factor;
  int w_bar =
      static_cast<int>(std::round(width / static_cast<double>(factor))) *
      factor;

  if (t_bar * h_bar * w_bar > max_pixels) {
    double beta = std::sqrt((num_frames * height * width) /
                            static_cast<double>(max_pixels));
    h_bar = static_cast<int>(
                std::floor(height / beta / static_cast<double>(factor))) *
            factor;
    w_bar = static_cast<int>(
                std::floor(width / beta / static_cast<double>(factor))) *
            factor;
  } else if (t_bar * h_bar * w_bar < min_pixels) {
    double beta = std::sqrt(min_pixels /
                            static_cast<double>(height * width * num_frames));
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

torch::Tensor Glm4VImageProcessor::sample_frames(const VideoMetadata& metadata,
                                                 int temporal_patch_size) {
  // video: [T, C, H, W]
  const int total_frames = metadata.total_num_frames;
  if (total_frames <= 0) {
    return torch::empty({0}, torch::dtype(torch::kLong));
  }

  if (metadata.fps <= 0.0) {
    LOG(FATAL) << "invalid metadata.fps <= 0";
  }

  const int max_frame_idx = total_frames - 1;

  // duration = metadata.duration or round(max_idx / fps) + 1
  double duration = metadata.duration;
  if (duration <= 0.0) {
    duration =
        std::round(static_cast<double>(max_frame_idx) / metadata.fps) + 1.0;
  }

  constexpr double DYN_FPS_30 = 3.0;
  constexpr double DYN_FPS_300 = 1.0;
  constexpr double DYN_FPS_2400 = 0.5;
  constexpr int MAX_FRAME_COUNT_DYNAMIC = 640;
  constexpr double MAX_DURATION = 2400.0;

  const double effective_duration = std::min(duration, MAX_DURATION);

  double target_fps = 0.0;
  if (effective_duration <= 30.0) {
    target_fps = DYN_FPS_30;
  } else if (effective_duration <= 300.0) {
    target_fps = DYN_FPS_300;
  } else {
    target_fps = DYN_FPS_2400;
  }

  // extract_t = int(effective_duration * target_fps * temporal_patch_size)
  int extract_t = static_cast<int>(effective_duration * target_fps *
                                   static_cast<double>(temporal_patch_size));
  extract_t = std::min(extract_t, MAX_FRAME_COUNT_DYNAMIC);

  const double duration_per_frame = 1.0 / metadata.fps;
  std::vector<double> timestamps(total_frames);
  for (int i = 0; i < total_frames; ++i) {
    timestamps[i] = static_cast<double>(i) * duration_per_frame;
  }
  const int max_second = static_cast<int>(duration);

  torch::Tensor frame_indices;

  if (total_frames < extract_t) {
    frame_indices = torch::linspace(
        0, total_frames - 1, extract_t, torch::dtype(torch::kLong));
  } else {
    std::vector<int64_t> tmp;
    tmp.reserve(static_cast<size_t>(total_frames));
    double current_second = 0.0;
    const double inv_fps =
        1.0 / (static_cast<double>(temporal_patch_size) * target_fps);

    for (int frame_index = 0; frame_index < total_frames; frame_index++) {
      if (timestamps[frame_index] >= current_second) {
        current_second += inv_fps;
        tmp.push_back(frame_index);
        if (current_second >= static_cast<double>(max_second)) {
          break;
        }
      }
    }
    frame_indices =
        torch::tensor(tmp, torch::TensorOptions().dtype(torch::kLong));
  }
  int64_t len = frame_indices.size(0);
  if (len < extract_t) {
    int64_t start, end;
    if (len == 0) {
      start = 0;
      end = std::max<int64_t>(total_frames - 1, 0);
    } else {
      start = frame_indices[0].item<int64_t>();
      end = frame_indices[len - 1].item<int64_t>();
    }
    frame_indices =
        torch::linspace(start, end, extract_t, torch::dtype(torch::kLong));
  } else if (len > extract_t) {
    frame_indices = torch::linspace(
        0, total_frames - 1, extract_t, torch::dtype(torch::kLong));
  }

  len = frame_indices.size(0);
  std::unordered_set<int64_t> seen;
  seen.reserve(static_cast<size_t>(len) * 2);
  std::vector<int64_t> uniq;
  uniq.reserve(static_cast<size_t>(len));

  for (int64_t i = 0; i < len; ++i) {
    auto idx = frame_indices[i].item<int64_t>();
    if (seen.insert(idx).second) {
      uniq.push_back(idx);
    }
  }

  if (!uniq.empty() && (uniq.size() & 1)) {
    uniq.push_back(uniq.back());
  }

  return torch::tensor(uniq, torch::TensorOptions().dtype(torch::kLong));
}

Glm4VImageProcessor::Glm4VImageProcessor(const ModelArgs& args) {
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

  video_mean_ = args.mm_video_normalize_mean();
  video_std_ = args.mm_video_normalize_std();

  video_min_pixels_ = args.mm_video_shortest_edge();
  video_max_pixels_ = args.mm_video_longest_edge();

  video_patch_size_ = args.mm_video_patch_size();
  video_temporal_patch_size_ = args.mm_video_temporal_patch_size();
  video_merge_size_ = args.mm_video_merge_size();

  size_ = {{"longest_edge", 12845056}, {"shortest_edge", 3136}};

  // fuse image mean/std and rescale_factor
  if (do_rescale_ && do_normalize_) {
    for (auto& item : image_mean_) {
      item = item * (1.0 / rescale_factor_);
    }

    for (auto& item : image_std_) {
      item = item * (1.0 / rescale_factor_);
    }

    for (auto& item : video_mean_) {
      item = item * (1.0 / rescale_factor_);
    }

    for (auto& item : video_std_) {
      item = item * (1.0 / rescale_factor_);
    }

    do_rescale_ = false;
  }
}

bool Glm4VImageProcessor::process(const MMInput& inputs, MMData& datas) {
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

bool Glm4VImageProcessor::process_images(std::vector<torch::Tensor> images,
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

bool Glm4VImageProcessor::process_image(torch::Tensor image,
                                        torch::Tensor& pixel_values,
                                        torch::Tensor& thw) {
  auto shape = image.sizes();

  auto resized_height = shape[1];
  auto resized_width = shape[2];

  // do_convert_rgb

  // resize
  if (do_resize_) {
    auto size = smart_resize(temporal_patch_size_,
                             resized_height,
                             resized_width,
                             temporal_patch_size_,
                             patch_size_ * merge_size_,
                             min_pixels_,
                             max_pixels_);
    if (!size) {
      return false;
    }

    std::tie(resized_height, resized_width) = *size;
    image =
        this->resize(image, {resized_height, resized_width}, resample_, true);
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

  auto repeats = patches[-1].unsqueeze(0).repeat(
      /*{temporal_patch_size_ - (shape[0] % temporal_patch_size_)*/ {
          temporal_patch_size_ - 1, 1, 1, 1});
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

bool Glm4VImageProcessor::process_videos(
    std::vector<torch::Tensor> videos,
    std::vector<VideoMetadata> video_meta_list,
    MMData& mm_datas) {
  torch::Tensor pixel_values;
  torch::Tensor thw;

  const size_t video_size = videos.size();
  for (size_t i = 0; i < video_size; ++i) {
    auto& vid = videos[i];
    auto& metadata = video_meta_list[i];
    if (!this->process_video(vid, metadata, pixel_values, thw)) {
      return false;
    }

    auto& item = mm_datas.add(MMType::VIDEO);
    item.set_data(
        {{"pixel_values_videos", pixel_values}, {"video_grid_thw", thw}});
    item.set_metadata(metadata);
  }

  return true;
}

bool Glm4VImageProcessor::process_video(torch::Tensor origin_video,
                                        VideoMetadata& metadata,
                                        torch::Tensor& pixel_values,
                                        torch::Tensor& thw) {
  if (origin_video.dim() != 4) {
    LOG(FATAL) << "video must be TCHW";
  }

  torch::Tensor indices;
  if (do_sample_frame_) {
    indices = this->sample_frames(metadata, video_temporal_patch_size_);
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
    auto size = smart_resize(time_len,
                             resized_height,
                             resized_width,
                             video_temporal_patch_size_,
                             video_patch_size_ * video_merge_size_,
                             video_min_pixels_,
                             video_max_pixels_);
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
    if (do_normalize_) frame = this->normalize(frame, video_mean_, video_std_);
    // rescale
    if (do_rescale_) frame = this->rescale(frame, rescale_factor_);
    out_frames.push_back(frame);
  }

  auto out_video = torch::stack(out_frames);  // [T,C,H,W]

  if (out_video.size(0) % video_temporal_patch_size_) {
    auto last = out_video.index({time_len - 1})
                    .unsqueeze(0)
                    .repeat({video_temporal_patch_size_ - 1, 1, 1, 1});
    out_video = torch::cat({out_video, last}, 0);
  }

  shape = out_video.sizes();
  auto grid_h = resized_height / video_patch_size_;
  auto grid_w = resized_width / video_patch_size_;
  auto grid_t = shape[0] / video_temporal_patch_size_;

  out_video = out_video.contiguous();

  auto patches = out_video.view({grid_t,
                                 video_temporal_patch_size_,
                                 channel,
                                 grid_h / video_merge_size_,
                                 video_merge_size_,
                                 video_patch_size_,
                                 grid_w / video_merge_size_,
                                 video_merge_size_,
                                 video_patch_size_});

  patches = patches.permute({0, 3, 6, 4, 7, 2, 1, 5, 8});
  patches = patches.reshape({grid_t * grid_h * grid_w,
                             channel * video_temporal_patch_size_ *
                                 video_patch_size_ * video_patch_size_});

  pixel_values = patches;
  thw = torch::tensor({grid_t, grid_h, grid_w}).clone().reshape({-1, 3});

  return true;
}

}  // namespace xllm
