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

#include "clip_image_processor.h"

namespace xllm {

torch::Tensor ImageProcessor::resize(const torch::Tensor& image,
                                     const std::vector<int64_t>& size,
                                     int resample,
                                     bool antialias) {
  if (image.dim() != 3) {
    LOG(FATAL) << "Input image must be a 3D tensor (C x H x W).";
  }
  auto options = torch::nn::functional::InterpolateFuncOptions()
                     .size(size)
                     .align_corners(false)
                     .antialias(antialias);
  switch (resample) {
    case 1:
      options.mode(torch::kNearest);
      break;
    case 2:
      options.mode(torch::kBilinear);
      break;
    case 3:
      options.mode(torch::kBicubic);
      break;
    default:
      LOG(FATAL) << "Invalid resample value. Must be one of 1, 2, or 3.";
  }
  return torch::nn::functional::interpolate(image.unsqueeze(0), options)
      .squeeze(0)
      .clamp(0, 255)
      .to(torch::kUInt8);
}

torch::Tensor ImageProcessor::centerCrop(const torch::Tensor& image,
                                         const std::pair<int, int>& cropSize) {
  if (image.dim() != 3) {
    LOG(FATAL)
        << "Input image must be a 3-dimensional tensor in (C, H, W) format.";
  }

  int cropHeight = cropSize.first;
  int cropWidth = cropSize.second;
  int origHeight = image.size(1);
  int origWidth = image.size(2);

  int top = (origHeight - cropHeight) / 2;
  int bottom = top + cropHeight;
  int left = (origWidth - cropWidth) / 2;
  int right = left + cropWidth;

  if (top >= 0 && bottom <= origHeight && left >= 0 && right <= origWidth) {
    return image.index({torch::indexing::Slice(),
                        torch::indexing::Slice(top, bottom),
                        torch::indexing::Slice(left, right)});
  }

  int newHeight = std::max(cropHeight, origHeight);
  int newWidth = std::max(cropWidth, origWidth);
  auto paddedImage =
      torch::zeros({image.size(0), newHeight, newWidth}, image.options());

  int topPad = (newHeight - origHeight + 1) / 2;
  int leftPad = (newWidth - origWidth + 1) / 2;

  paddedImage.index_put_({torch::indexing::Slice(),
                          torch::indexing::Slice(topPad, topPad + origHeight),
                          torch::indexing::Slice(leftPad, leftPad + origWidth)},
                         image);

  top = (newHeight - cropHeight) / 2;
  bottom = top + cropHeight;
  left = (newWidth - cropWidth) / 2;
  right = left + cropWidth;

  return paddedImage.index({torch::indexing::Slice(),
                            torch::indexing::Slice(top, bottom),
                            torch::indexing::Slice(left, right)});
}

torch::Tensor ImageProcessor::rescale(const torch::Tensor& image,
                                      double scale) {
  return image * scale;
}

torch::Tensor ImageProcessor::normalize(const torch::Tensor& image,
                                        const std::vector<double>& mean,
                                        const std::vector<double>& std) {
  if (image.dim() != 3) {
    LOG(FATAL)
        << "Input image must be a 3-dimensional tensor in (C, H, W) format.";
  }

  int numChannels = image.size(0);
  if (mean.size() != numChannels || std.size() != numChannels) {
    LOG(FATAL) << "Mean and std vectors must have the same number "
               << "of elements as the number of channels in the "
               << "image.";
  }

  auto result = image;
  if (!image.is_floating_point()) {
    result = image.to(torch::kFloat32);
  }

  auto device = image.device();
  auto options = torch::dtype(torch::kFloat32).device(device);

  auto m_tensor = torch::tensor(mean, options).reshape({-1, 1, 1});
  auto s_tensor = torch::tensor(std, options).reshape({-1, 1, 1});

  result = result.sub(m_tensor);
  return result.div_(s_tensor);
}

torch::Tensor ImageProcessor::init_frames(const VideoMetadata& metadata) {
  int total_num_frames = metadata.total_num_frames;
  int nframes_len = 32;
  if (total_num_frames <= 0) {
    return torch::empty({0}, torch::dtype(torch::kLong));
  }
  auto idx = torch::linspace(
      0, total_num_frames - 1, nframes_len, torch::dtype(torch::kLong));
  return idx;
}

torch::Tensor ImageProcessor::sample_frames(const VideoMetadata& metadata,
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

torch::Tensor ImageProcessor::GLM_sample_frames(const VideoMetadata& metadata,
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

}  // namespace xllm
