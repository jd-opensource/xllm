/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
==============================================================================*/

#pragma once

#include <tuple>
#include <unordered_map>
#include <vector>

#include "core/framework/model/model_args.h"
#include "image_processor.h"

namespace xllm {

struct VideoChunkMetadata {
  int chunk_id;
  double start_timestamp;
  int num_frames;
  std::vector<int> frame_indices;
  std::string timestamp_text;
  std::string prompt;
};

struct NavitResizeResult {
  int num_tokens = 0;
  int new_width = 0;
  int new_height = 0;
  int pad_width = 0;
  int pad_height = 0;
  int sampled_nframes = 1;
};

struct KimiK25MediaConfig {
  int patch_size = 14;
  int merge_kernel_size = 2;
  int temporal_merge_kernel_size = 4;

  int in_patch_limit = 16384;
  int in_patch_limit_each_frame = -1;
  int patch_limit_on_one_side = 512;
  int in_patch_limit_video = -1;
  int fixed_output_tokens = -1;

  double sample_fps = 2.0;
  int max_num_frames_each_video = 128;
  int min_frames = 4;
  std::string timestamp_mode = "hh:mm:ss.fff";

  std::vector<double> image_mean = {0.5, 0.5, 0.5};
  std::vector<double> image_std = {0.5, 0.5, 0.5};

  bool do_convert_rgb = true;
  bool do_normalize = true;
  bool do_resize = true;
  int resample = 3;
};

class KimiK25ImageProcessor : public ImageProcessor {
 public:
  KimiK25ImageProcessor(const ModelArgs& args);
  ~KimiK25ImageProcessor() override = default;

  bool process(const MMInput& mm_inputs, MMData& mm_datas) override;

  std::vector<VideoChunkMetadata> split_video_chunks(
      const VideoMetadata& video_meta);

 private:
  bool process_images(std::vector<torch::Tensor> images, MMData& mm_datas);
  bool process_image(torch::Tensor image,
                     torch::Tensor& pixel_values,
                     torch::Tensor& thw);

  bool process_videos(std::vector<torch::Tensor> videos,
                      std::vector<VideoMetadata> video_meta_list,
                      MMData& mm_datas);
  bool process_video_chunk(torch::Tensor video_chunk,
                           const VideoMetadata& video_meta,
                           torch::Tensor& pixel_values,
                           torch::Tensor& thw);

  torch::Tensor sample_frames(const VideoMetadata& metadata);
  std::string timestamp_as_str(double timestamp,
                               const std::string& timestamp_mode);
  std::string make_chunk_prompt(const std::string& timestamp_text);
  std::optional<NavitResizeResult> navit_resize(int height,
                                                int width,
                                                bool is_video,
                                                int nframes = 1,
                                                double avg_fps = 24.0);
  std::pair<torch::Tensor, torch::Tensor> navit_patchify(torch::Tensor pixels);
  torch::Tensor km_normalize(const torch::Tensor& image,
                             const std::vector<double>& mean,
                             const std::vector<double>& std);

 private:
  KimiK25MediaConfig config_;
};

}  // namespace xllm
