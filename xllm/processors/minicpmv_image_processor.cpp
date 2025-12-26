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

#include "minicpmv_image_processor.h"

namespace xllm {

MiniCPMVImageProcessor::MiniCPMVImageProcessor(const ModelArgs& args) {
  max_slice_nums_ = args.vision_max_slice_nums();
  scale_resolution_ = args.mm_scale_resolution();
  patch_size_ = args.mm_patch_size();
  slice_mode_ = args.mm_slice_mode();
  image_feature_size_ = args.mm_image_feature_size();
  norm_mean_ = args.mm_image_normalize_mean();
  norm_std_ = args.mm_image_normalize_std();
}

bool MiniCPMVImageProcessor::process(const MMInput& mm_inputs,
                                     MMData& mm_datas) {
  std::vector<torch::Tensor> images = mm_inputs.get_decode_data(MMType::IMAGE);
  if (images.empty()) {
    LOG(ERROR) << " image tensor not found.";
    return false;
  }

  if (!this->process_images(images, mm_datas)) {
    LOG(ERROR) << " process image failed.";
    return false;
  }

  return true;
}

bool MiniCPMVImageProcessor::process_images(std::vector<torch::Tensor> images,
                                            MMData& mm_datas) {
  std::vector<torch::Tensor> new_images;
  std::vector<torch::Tensor> tgt_sizes;

  const size_t image_size = images.size();
  new_images.reserve(image_size *
                     (size_t{1} + static_cast<size_t>(max_slice_nums_)));
  tgt_sizes.reserve(image_size *
                    (size_t{1} + static_cast<size_t>(max_slice_nums_)));

  for (const auto& image : images) {
    new_images.clear();
    tgt_sizes.clear();

    if (!this->process_image(image, new_images, tgt_sizes)) return false;

    // image shape: [C, H, W]
    const auto& image_size = image.sizes();
    int orig_w = image_size[2];
    int orig_h = image_size[1];

    auto image_sizes = torch::tensor({orig_w, orig_h}, torch::kInt64);
    auto tgt_tensor = torch::stack(tgt_sizes);

    auto& item = mm_datas.add(MMType::IMAGE);
    item.set_data({{"pixel_values", new_images},
                   {"image_sizes", image_sizes},
                   {"tgt_sizes", tgt_tensor}});
  }

  return true;
}

bool MiniCPMVImageProcessor::process_image(
    torch::Tensor image,
    std::vector<torch::Tensor>& new_images,
    std::vector<torch::Tensor>& tgt_sizes) {
  auto image_patches = get_sliced_images(image, max_slice_nums_);

  for (auto& patch : image_patches) {
    patch = patch.to(torch::kFloat32);
    patch = patch / 255.0;
    patch = this->normalize(patch, norm_mean_, norm_std_);

    const auto& one_patch_size = patch.sizes();
    int64_t tgt_h = one_patch_size[1] / patch_size_;
    int64_t tgt_w = one_patch_size[2] / patch_size_;
    tgt_sizes.emplace_back(torch::tensor({tgt_h, tgt_w}, torch::kInt64));

    patch = this->reshape_by_patch(patch);
    new_images.emplace_back(patch);
  }
  return true;
}

std::pair<int, int> MiniCPMVImageProcessor::find_best_resize(
    const std::pair<int, int>& original_size,
    int scale_resolution,
    int patch_size,
    bool allow_upscale) const {
  int width = original_size.first;
  int height = original_size.second;

  if ((width * height > scale_resolution * scale_resolution) || allow_upscale) {
    float aspect_ratio = static_cast<float>(width) / height;
    height = static_cast<int>(scale_resolution / std::sqrt(aspect_ratio));
    width = static_cast<int>(height * aspect_ratio);
  }

  int best_width = ensure_divide(width, patch_size);
  int best_height = ensure_divide(height, patch_size);

  return {best_width, best_height};
}

std::pair<int, int> MiniCPMVImageProcessor::get_sliced_grid(
    const std::pair<int, int>& original_size,
    int max_slice_nums,
    int scale_resolution,
    bool never_split) {
  int width = original_size.first;
  int height = original_size.second;

  double log_ratio = std::log(static_cast<double>(width) / height);
  double ratio = static_cast<double>(width * height) /
                 (scale_resolution * scale_resolution);
  int multiple = std::min(static_cast<int>(std::ceil(ratio)), max_slice_nums);
  if (never_split || multiple <= 1) {
    return {-1, -1};
  }

  std::vector<int> candidate_split_grids_nums;
  candidate_split_grids_nums.reserve(3);
  for (int i : {multiple - 1, multiple, multiple + 1}) {
    if (i > 1 && i <= max_slice_nums) {
      candidate_split_grids_nums.emplace_back(i);
    }
  }

  std::vector<std::pair<int, int>> candidate_grids;
  candidate_grids.reserve(2 * multiple);
  for (int split_grids_nums : candidate_split_grids_nums) {
    for (int m = 1; m <= split_grids_nums; ++m) {
      if (split_grids_nums % m == 0) {
        candidate_grids.emplace_back(m, split_grids_nums / m);
      }
    }
  }

  std::pair<int, int> best_grid = {1, 1};
  double min_error = std::numeric_limits<double>::infinity();
  for (const auto& grid : candidate_grids) {
    double error = std::abs(
        log_ratio - std::log(static_cast<double>(grid.first) / grid.second));
    if (error < min_error) {
      best_grid = grid;
      min_error = error;
    }
  }

  return best_grid;
}

std::vector<std::vector<torch::Tensor>>
MiniCPMVImageProcessor::split_to_patches(
    const torch::Tensor& image,
    const std::pair<int, int>& grid) const {
  int width = image.size(2);
  int height = image.size(1);
  int grid_x = width / grid.first;
  int grid_y = height / grid.second;
  std::vector<std::vector<torch::Tensor>> patches;
  patches.reserve(grid.second + 1);
  for (int i = 0; i < height; i += grid_y) {
    std::vector<torch::Tensor> row_patches;
    row_patches.reserve(grid.first + 1);
    for (int j = 0; j < width; j += grid_x) {
      torch::Tensor patch;
      patch = image.index({torch::indexing::Slice(),
                           torch::indexing::Slice(i, i + grid_y),
                           torch::indexing::Slice(j, j + grid_x)});
      patch = patch.clone();
      row_patches.emplace_back(patch);
    }
    if (!row_patches.empty()) patches.emplace_back(row_patches);
  }

  return patches;
}

std::pair<int, int> MiniCPMVImageProcessor::get_refine_size(
    const std::pair<int, int>& original_size,
    const std::pair<int, int>& grid,
    int scale_resolution,
    int patch_size,
    bool allow_upscale) const {
  int width = original_size.first;
  int height = original_size.second;
  int grid_x = grid.first;
  int grid_y = grid.second;

  int refine_width = ensure_divide(width, grid_x);
  int refine_height = ensure_divide(height, grid_y);

  float grid_width = static_cast<float>(refine_width) / grid_x;
  float grid_height = static_cast<float>(refine_height) / grid_y;

  auto best_grid_size = find_best_resize(
      {grid_width, grid_height}, scale_resolution, patch_size, allow_upscale);
  return {best_grid_size.first * grid_x, best_grid_size.second * grid_y};
}

std::tuple<torch::Tensor,
           std::vector<std::vector<torch::Tensor>>,
           std::pair<int, int>>
MiniCPMVImageProcessor::slice_image(const torch::Tensor& image,
                                    int max_slice_nums,
                                    int scale_resolution,
                                    int patch_size,
                                    bool never_split) {
  std::pair<int, int> original_size = {image.size(2), image.size(1)};
  torch::Tensor source_image;
  std::vector<std::vector<torch::Tensor>> patches;

  auto best_grid = get_sliced_grid(
      original_size, max_slice_nums, scale_resolution, never_split);

  if (best_grid.first == -1 && best_grid.second == -1) {
    auto best_size =
        find_best_resize(original_size, scale_resolution, patch_size, true);
    std::vector<int64_t> size = {best_size.second, best_size.first};
    source_image = resize(image, size, 3, true);
  } else {
    auto best_resize =
        find_best_resize(original_size, scale_resolution, patch_size);
    auto refine_size = get_refine_size(
        original_size, best_grid, scale_resolution, patch_size, true);

    std::vector<int64_t> best_size = {best_resize.second, best_resize.first};
    source_image = this->resize(image, best_size, 3, true);

    std::vector<int64_t> refine_sz = {refine_size.second, refine_size.first};
    torch::Tensor refine_image = this->resize(image, refine_sz, 3, true);

    patches = split_to_patches(refine_image, best_grid);
  }
  return {source_image, patches, best_grid};
}

torch::Tensor MiniCPMVImageProcessor::reshape_by_patch(
    const torch::Tensor& image) {
  if (image.dim() != 3) {
    LOG(FATAL) << "Input must be a 3D tensor with shape [C, H, W].";
  }

  auto input = image.unsqueeze(0);

  auto unfolded = torch::nn::functional::unfold(
      input,
      torch::nn::functional::UnfoldFuncOptions({patch_size_, patch_size_})
          .stride({patch_size_, patch_size_}));

  unfolded = unfolded.squeeze(0);
  auto reshaped =
      unfolded.reshape({image.size(0), patch_size_, patch_size_, -1});
  reshaped =
      reshaped.permute({0, 1, 3, 2}).reshape({image.size(0), patch_size_, -1});
  return reshaped;
}

std::vector<torch::Tensor> MiniCPMVImageProcessor::get_sliced_images(
    const torch::Tensor& image,
    int max_slice_nums) {
  std::vector<torch::Tensor> slice_images;
  // bool slice_mode = true;
  if (!slice_mode_) {
    slice_images.reserve(1);
    slice_images.emplace_back(image);
    return slice_images;
  }

  max_slice_nums = (max_slice_nums == -1) ? max_slice_nums_ : max_slice_nums;
  assert(max_slice_nums > 0);

  auto [source_image, patches, sliced_grid] =
      slice_image(image, max_slice_nums, scale_resolution_, patch_size_, false);

  auto gx = (sliced_grid.first > 0) ? sliced_grid.first : 0;
  auto gy = (sliced_grid.second > 0) ? sliced_grid.second : 0;
  auto total = 1 + static_cast<size_t>(gx) * static_cast<size_t>(gy);
  slice_images.reserve(total);
  slice_images.emplace_back(source_image);
  if (!patches.empty()) {
    for (auto& row : patches) {
      for (auto& patch : row) {
        slice_images.emplace_back(patch);
      }
    }
  }
  return slice_images;
}

}  // namespace xllm
