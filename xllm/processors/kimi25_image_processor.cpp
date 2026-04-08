/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
==============================================================================*/

#include "kimi25_image_processor.h"

#include "glog/logging.h"

namespace xllm {
KimiK25ImageProcessor::KimiK25ImageProcessor(const ModelArgs& args) {
  config_.patch_size =
      args.mm_km_patch_size() > 0 ? args.mm_km_patch_size() : 14;
  config_.merge_kernel_size =
      args.mm_km_merge_kernel_size() > 0 ? args.mm_km_merge_kernel_size() : 2;
  config_.temporal_merge_kernel_size =
      args.mm_km_temporal_merge_kernel_size() > 0
          ? args.mm_km_temporal_merge_kernel_size()
          : 4;

  config_.in_patch_limit =
      args.mm_km_in_patch_limit() > 0 ? args.mm_km_in_patch_limit() : 16384;
  config_.in_patch_limit_each_frame =
      args.mm_km_in_patch_limit_each_frame() > 0
          ? args.mm_km_in_patch_limit_each_frame()
          : -1;
  config_.patch_limit_on_one_side = args.mm_km_patch_limit_on_one_side() > 0
                                        ? args.mm_km_patch_limit_on_one_side()
                                        : 512;
  config_.in_patch_limit_video = args.mm_km_in_patch_limit_video() > 0
                                     ? args.mm_km_in_patch_limit_video()
                                     : -1;
  config_.fixed_output_tokens = args.mm_km_fixed_output_tokens();

  config_.sample_fps =
      args.mm_km_sample_fps() > 0 ? args.mm_km_sample_fps() : 2.0;
  config_.max_num_frames_each_video =
      args.mm_km_max_num_frames_each_video() > 0
          ? args.mm_km_max_num_frames_each_video()
          : 2;
  config_.min_frames = 4;
  config_.timestamp_mode = args.mm_km_timestamp_mode();

  if (!args.mm_km_image_mean().empty()) {
    config_.image_mean.assign(args.mm_km_image_mean().begin(),
                              args.mm_km_image_mean().end());
  } else {
    config_.image_mean = {0.5, 0.5, 0.5};
  }

  if (!args.mm_km_image_std().empty()) {
    config_.image_std.assign(args.mm_km_image_std().begin(),
                             args.mm_km_image_std().end());
  } else {
    config_.image_std = {0.5, 0.5, 0.5};
  }

  config_.do_convert_rgb = true;
  config_.do_normalize = true;
  config_.do_resize = true;

  config_.resample = 3;
}

torch::Tensor KimiK25ImageProcessor::km_normalize(
    const torch::Tensor& image,
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

  result = result / 255.0;

  auto device = image.device();
  auto options = torch::dtype(torch::kFloat32).device(device);

  auto m_tensor = torch::tensor(mean, options).reshape({-1, 1, 1});
  auto s_tensor = torch::tensor(std, options).reshape({-1, 1, 1});

  auto s_inv_tensor = 1.0 / s_tensor;

  result = result.sub(m_tensor);
  return result.mul_(s_inv_tensor);
}

bool KimiK25ImageProcessor::process(const MMInput& inputs, MMData& datas) {
  for (const auto& input_item : inputs) {
    std::vector<torch::Tensor> images, videos;
    std::vector<VideoMetadata> video_meta_list;

    if (input_item.has_type(MMType::IMAGE)) {
      images.push_back(input_item.decode_image);
    }
    if (input_item.has_type(MMType::VIDEO)) {
      videos.push_back(input_item.decode_video);
      video_meta_list.push_back(input_item.video_meta);
    }

    if (images.empty() && videos.empty()) {
      LOG(ERROR) << "No image/video input";
      return false;
    }

    if (!images.empty() && !process_images(images, datas)) {
      return false;
    }

    if (!videos.empty() && !process_videos(videos, video_meta_list, datas)) {
      return false;
    }
  }
  return true;
}

std::vector<VideoChunkMetadata> KimiK25ImageProcessor::split_video_chunks(
    const VideoMetadata& video_meta) {
  std::vector<VideoChunkMetadata> chunks;

  torch::Tensor frame_indices = sample_frames(video_meta);
  std::vector<int64_t> indices(
      frame_indices.data_ptr<int64_t>(),
      frame_indices.data_ptr<int64_t>() + frame_indices.numel());

  int chunk_size = config_.temporal_merge_kernel_size;
  int num_chunks = (indices.size() + chunk_size - 1) / chunk_size;

  for (int i = 0; i < num_chunks; ++i) {
    VideoChunkMetadata chunk;
    chunk.chunk_id = i;

    int start = i * chunk_size;
    int end = std::min((i + 1) * chunk_size, static_cast<int>(indices.size()));
    chunk.frame_indices =
        std::vector<int>(indices.begin() + start, indices.begin() + end);
    chunk.num_frames = chunk.frame_indices.size();

    double start_time =
        static_cast<double>(chunk.frame_indices[0]) / video_meta.fps;
    chunk.start_timestamp = start_time;

    chunk.timestamp_text = timestamp_as_str(start_time, config_.timestamp_mode);
    chunk.prompt = make_chunk_prompt(chunk.timestamp_text);

    chunks.push_back(chunk);
  }

  return chunks;
}

torch::Tensor KimiK25ImageProcessor::sample_frames(
    const VideoMetadata& metadata) {
  int total_frames = metadata.total_num_frames;
  double sample_fps = std::min(config_.sample_fps, metadata.fps);

  int sampled_frames = std::max(
      static_cast<int>(std::round(total_frames * sample_fps / metadata.fps)),
      1);

  torch::Tensor frame_inds_float =
      torch::linspace(0, static_cast<double>(total_frames - 1), sampled_frames);
  torch::Tensor frame_inds = frame_inds_float.round().to(torch::kLong);

  return frame_inds;
}

std::string KimiK25ImageProcessor::timestamp_as_str(
    double timestamp,
    const std::string& timestamp_mode) {
  int hours = static_cast<int>(timestamp) / 3600;
  int minutes = (static_cast<int>(timestamp) % 3600) / 60;
  int seconds = static_cast<int>(timestamp) % 60;
  int milliseconds =
      static_cast<int>((timestamp - static_cast<int>(timestamp)) * 1000);

  char buffer[32];
  if (timestamp_mode == "hh:mm:ss.fff") {
    snprintf(buffer,
             sizeof(buffer),
             "%02d:%02d:%02d.%03d",
             hours,
             minutes,
             seconds,
             milliseconds);
  } else if (timestamp_mode == "mm:ss.fff") {
    snprintf(buffer,
             sizeof(buffer),
             "%02d:%02d.%03d",
             minutes,
             seconds,
             milliseconds);
  } else if (timestamp_mode == "mm:ss") {
    snprintf(buffer, sizeof(buffer), "%02d:%02d", minutes, seconds);
  } else {
    LOG(ERROR) << "Invalid timestamp mode: " << timestamp_mode;
    return "";
  }
  return std::string(buffer);
}

std::string KimiK25ImageProcessor::make_chunk_prompt(
    const std::string& timestamp_text) {
  return timestamp_text +
         "<|media_begin|>video<|media_content|><|media_pad|><|media_end|>";
}

std::optional<NavitResizeResult> navit_resize_image(
    int width,
    int height,
    int patch_size,
    int merge_kernel_size,
    int in_patch_limit,
    int patch_limit_on_one_side,
    std::optional<int> fixed_output_tokens) {
  if (static_cast<double>(std::max(height, width)) / std::min(height, width) >
      200) {
    LOG(ERROR) << "Aspect ratio exceeds 200";
    return std::nullopt;
  }

  NavitResizeResult result;
  double s1 =
      std::sqrt(in_patch_limit /
                (std::max(1.0, static_cast<double>(width / patch_size)) *
                 std::max(1.0, static_cast<double>(height / patch_size))));
  double s2 = static_cast<double>(patch_limit_on_one_side * patch_size) / width;
  double s3 =
      static_cast<double>(patch_limit_on_one_side * patch_size) / height;
  double scale = std::min({1.0, s1, s2, s3});

  int new_w = std::max(1, static_cast<int>(width * scale));
  int new_h = std::max(1, static_cast<int>(height * scale));
  new_w = std::min(new_w, patch_limit_on_one_side * patch_size);
  new_h = std::min(new_h, patch_limit_on_one_side * patch_size);

  int factor = merge_kernel_size * patch_size;

  int pad_height = (factor - new_h % factor) % factor;
  int pad_width = (factor - new_w % factor) % factor;

  int num_tokens;
  if (fixed_output_tokens.has_value()) {
    num_tokens = fixed_output_tokens.value();
  } else {
    int token_height = (new_h + pad_height) / factor;
    int token_width = (new_w + pad_width) / factor;

    if (token_height * merge_kernel_size > patch_limit_on_one_side) {
      LOG(ERROR) << "token_height " << token_height << " * merge_kernel_size "
                 << merge_kernel_size << " > patch_limit_on_one_side "
                 << patch_limit_on_one_side;
      return std::nullopt;
    }
    if (token_width * merge_kernel_size > patch_limit_on_one_side) {
      LOG(ERROR) << "token_width " << token_width << " * merge_kernel_size "
                 << merge_kernel_size << " > patch_limit_on_one_side "
                 << patch_limit_on_one_side;
      return std::nullopt;
    }

    num_tokens = token_height * token_width;
  }

  result.sampled_nframes = 1;
  result.num_tokens = num_tokens;
  result.new_width = new_w;
  result.new_height = new_h;
  result.pad_width = pad_width;
  result.pad_height = pad_height;

  return result;
}

std::optional<NavitResizeResult> navit_resize_video(
    int width,
    int height,
    int nframes,
    double avg_fps,
    double sample_fps,
    int patch_size,
    int merge_kernel_size,
    int in_patch_limit_each_frame,
    int patch_limit_on_one_side,
    std::optional<int> in_patch_limit_total,
    std::optional<int> max_num_frames_each_video,
    std::optional<int> fixed_output_tokens_each_frame) {
  sample_fps = std::min(sample_fps, avg_fps);
  int sampled_nframes =
      std::max(static_cast<int>(std::round(nframes * sample_fps / avg_fps)), 1);
  if (max_num_frames_each_video.has_value()) {
    sampled_nframes =
        std::min(sampled_nframes, max_num_frames_each_video.value());
  }

  if (in_patch_limit_total.has_value()) {
    int per_frame_limit = static_cast<int>(std::round(
        static_cast<double>(in_patch_limit_total.value()) / sampled_nframes));
    in_patch_limit_each_frame =
        std::min(per_frame_limit, in_patch_limit_each_frame);
  }

  auto ret = navit_resize_image(width,
                                height,
                                patch_size,
                                merge_kernel_size,
                                in_patch_limit_each_frame,
                                patch_limit_on_one_side,
                                fixed_output_tokens_each_frame);

  if (ret) {
    ret->sampled_nframes = sampled_nframes;
  }
  return ret;
}

std::optional<NavitResizeResult> KimiK25ImageProcessor::navit_resize(
    int height,
    int width,
    bool is_video,
    int nframes,
    double avg_fps) {
  if (!is_video) {
    std::optional<int> fixed_output_tokens =
        config_.fixed_output_tokens >= 0
            ? std::optional<int>(config_.fixed_output_tokens)
            : std::nullopt;
    return navit_resize_image(width,
                              height,
                              config_.patch_size,
                              config_.merge_kernel_size,
                              config_.in_patch_limit,
                              config_.patch_limit_on_one_side,
                              fixed_output_tokens);
  } else {
    std::optional<int> in_patch_limit_total_opt =
        config_.in_patch_limit_video > 0
            ? std::optional<int>(config_.in_patch_limit_video)
            : std::nullopt;

    double sample_fps = std::numeric_limits<double>::infinity();
    std::optional<int> max_num_frames_each_video_opt = std::nullopt;

    int in_patch_limit_each_frame = config_.in_patch_limit_each_frame > 0
                                        ? config_.in_patch_limit_each_frame
                                        : config_.in_patch_limit;
    std::optional<int> fixed_output_tokens_opt =
        config_.fixed_output_tokens >= 0
            ? std::optional<int>(config_.fixed_output_tokens)
            : std::nullopt;

    return navit_resize_video(width,
                              height,
                              nframes,
                              avg_fps,
                              sample_fps,
                              config_.patch_size,
                              config_.merge_kernel_size,
                              in_patch_limit_each_frame,
                              config_.patch_limit_on_one_side,
                              in_patch_limit_total_opt,
                              max_num_frames_each_video_opt,
                              fixed_output_tokens_opt);
  }
}

bool KimiK25ImageProcessor::process_images(std::vector<torch::Tensor> images,
                                           MMData& mm_datas) {
  torch::Tensor pixel_values, thw;
  for (const auto& img : images) {
    if (!process_image(img, pixel_values, thw)) {
      return false;
    }

    auto& item = mm_datas.add(MMType::IMAGE);
    item.set_data({{"pixel_values", pixel_values}, {"image_grid_thw", thw}});
  }
  return true;
}

bool KimiK25ImageProcessor::process_image(torch::Tensor image,
                                          torch::Tensor& pixel_values,
                                          torch::Tensor& thw) {
  if (image.dim() != 3) {
    LOG(ERROR) << "Image must be CHW format";
    return false;
  }

  auto shape = image.sizes();
  int height = shape[1];
  int width = shape[2];

  if (config_.do_resize) {
    auto resize_result = navit_resize(height, width, false);
    if (!resize_result) return false;

    int new_h = resize_result->new_height;
    int new_w = resize_result->new_width;
    int pad_height = resize_result->pad_height;
    int pad_width = resize_result->pad_width;

    image = this->resize(image, {new_h, new_w}, config_.resample, false);

    if (pad_height > 0 || pad_width > 0) {
      std::vector<int64_t> padding = {
          0,
          pad_width,
          0,
          pad_height,
          0,
          0};  // [left, right, top, bottom, front, back]
      image = torch::nn::functional::pad(
          image,
          torch::nn::functional::PadFuncOptions(padding)
              .mode(torch::kConstant)
              .value(0));
    }

    height = new_h + pad_height;
    width = new_w + pad_width;
  }

  if (config_.do_normalize) {
    image = this->km_normalize(image, config_.image_mean, config_.image_std);
  }

  image = image.unsqueeze(0);

  auto [patches, grid_thw_tensor] = navit_patchify(image);
  pixel_values = patches;

  thw = grid_thw_tensor.unsqueeze(0);

  return true;
}

bool KimiK25ImageProcessor::process_videos(
    std::vector<torch::Tensor> videos,
    std::vector<VideoMetadata> video_meta_list,
    MMData& mm_datas) {
  const size_t video_count = videos.size();
  auto opts = torch::TensorOptions().dtype(torch::kFloat32);

  for (size_t i = 0; i < video_count; ++i) {
    auto& video = videos[i];
    auto& meta = video_meta_list[i];
    torch::Tensor pixel_values;
    torch::Tensor thw;

    std::vector<VideoChunkMetadata> chunks = split_video_chunks(meta);

    double seconds_per_grid = 0;
    std::vector<std::string> video_prompts = {};
    for (const auto& chunk : chunks) {
      torch::Tensor pixel_values_chunk;
      torch::Tensor thw_chunk;

      torch::Tensor chunk_frames = video.index_select(
          0, torch::tensor(chunk.frame_indices, torch::kLong));

      if (!process_video_chunk(
              chunk_frames, meta, pixel_values_chunk, thw_chunk)) {
        return false;
      }

      double single_seconds_per_grid =
          static_cast<double>(chunk.num_frames) / meta.fps;
      seconds_per_grid += single_seconds_per_grid;

      if (pixel_values.numel() == 0) {
        pixel_values = pixel_values_chunk;
        thw = thw_chunk;
      } else {
        pixel_values = torch::cat({pixel_values, pixel_values_chunk}, 0);
        thw = torch::cat({thw, thw_chunk}, 0);
      }
      video_prompts.push_back(chunk.prompt);
    }

    auto second_per_grid_ts = torch::tensor({seconds_per_grid}, opts);
    auto& item = mm_datas.add(MMType::VIDEO);

    item.set_data({{"pixel_values_videos", pixel_values},
                   {"video_grid_thw", thw},
                   {"second_per_grid_ts", second_per_grid_ts}});
    //{"video_prompts", video_prompts}});
    item.set_metadata(meta);
  }
  return true;
}

bool KimiK25ImageProcessor::process_video_chunk(torch::Tensor video_chunk,
                                                const VideoMetadata& video_meta,
                                                torch::Tensor& pixel_values,
                                                torch::Tensor& thw) {
  if (video_chunk.dim() != 4) {
    LOG(ERROR) << "Video chunk must be TCHW format";
    return false;
  }

  auto shape = video_chunk.sizes();
  int t = shape[0];
  int c = shape[1];
  int h = shape[2];
  int w = shape[3];

  std::vector<torch::Tensor> processed_frames;
  processed_frames.reserve(t);

  auto frames = video_chunk.unbind(0);
  for (auto& frame : frames) {
    if (config_.do_resize) {
      auto resize_result = navit_resize(h, w, true, t, 1.0);
      if (!resize_result) return false;

      int new_h = resize_result->new_height;
      int new_w = resize_result->new_width;
      int pad_height = resize_result->pad_height;
      int pad_width = resize_result->pad_width;

      frame = this->resize(frame, {new_h, new_w}, config_.resample, true);

      if (pad_height > 0 || pad_width > 0) {
        std::vector<int64_t> padding = {
            0,
            pad_width,
            0,
            pad_height,
            0,
            0};  // [left, right, top, bottom, front, back]
        frame = torch::nn::functional::pad(
            frame,
            torch::nn::functional::PadFuncOptions(padding)
                .mode(torch::kConstant)
                .value(0));
      }

      h = new_h + pad_height;
      w = new_w + pad_width;
    }

    if (config_.do_normalize) {
      frame = this->km_normalize(frame, config_.image_mean, config_.image_std);
    }

    processed_frames.push_back(frame);
  }

  torch::Tensor processed_video = torch::stack(processed_frames);

  auto [patches, grid_thw_tensor] = navit_patchify(processed_video);
  pixel_values = patches;

  thw = grid_thw_tensor.unsqueeze(0);

  return true;
}

std::pair<torch::Tensor, torch::Tensor> KimiK25ImageProcessor::navit_patchify(
    torch::Tensor pixels) {
  int t = pixels.size(0);
  int c = pixels.size(1);
  int h = pixels.size(2);
  int w = pixels.size(3);

  int patch_size = config_.patch_size;

  auto patches = pixels.permute({0, 2, 3, 1});

  patches = patches.view(
      {t, h / patch_size, patch_size, w / patch_size, patch_size, c});

  patches = patches.permute({0, 1, 3, 5, 2, 4});

  patches = patches.reshape(
      {t * (h / patch_size) * (w / patch_size), c, patch_size, patch_size});

  torch::Tensor grid_thw =
      torch::tensor({t, h / patch_size, w / patch_size}, torch::kInt64);

  return {patches, grid_thw};
}

}  // namespace xllm
