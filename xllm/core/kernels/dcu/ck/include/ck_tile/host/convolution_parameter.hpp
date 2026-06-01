// SPDX-License-Identifier: MIT
// Copyright (c) 2024, , Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <numeric>
#include <iterator>
#include <vector>

namespace ck_tile {
namespace conv {

struct ConvParam
{
    ConvParam(ck_tile::index_t n_dim,
              ck_tile::index_t group_count,
              ck_tile::index_t n_batch,
              ck_tile::index_t n_out_channels,
              ck_tile::index_t n_in_channels,
              const std::vector<ck_tile::index_t>& filters_len,
              const std::vector<ck_tile::index_t>& input_len,
              const std::vector<ck_tile::index_t>& strides,
              const std::vector<ck_tile::index_t>& dilations,
              const std::vector<ck_tile::index_t>& left_pads,
              const std::vector<ck_tile::index_t>& right_pads)
        : num_dim_spatial_(static_cast<ck_tile::long_index_t>(n_dim)),
          G_(static_cast<ck_tile::long_index_t>(group_count)),
          N_(static_cast<ck_tile::long_index_t>(n_batch)),
          K_(static_cast<ck_tile::long_index_t>(n_out_channels)),
          C_(static_cast<ck_tile::long_index_t>(n_in_channels)),
          filter_spatial_lengths_(num_dim_spatial_),
          input_spatial_lengths_(num_dim_spatial_),
          output_spatial_lengths_(num_dim_spatial_),
          conv_filter_strides_(num_dim_spatial_),
          conv_filter_dilations_(num_dim_spatial_),
          input_left_pads_(num_dim_spatial_),
          input_right_pads_(num_dim_spatial_)
    {
        if(static_cast<ck_tile::index_t>(filter_spatial_lengths_.size()) != num_dim_spatial_ ||
           static_cast<ck_tile::index_t>(input_spatial_lengths_.size()) != num_dim_spatial_ ||
           static_cast<ck_tile::index_t>(conv_filter_strides_.size()) != num_dim_spatial_ ||
           static_cast<ck_tile::index_t>(conv_filter_dilations_.size()) != num_dim_spatial_ ||
           static_cast<ck_tile::index_t>(input_left_pads_.size()) != num_dim_spatial_ ||
           static_cast<ck_tile::index_t>(input_right_pads_.size()) != num_dim_spatial_)
        {
            throw(std::runtime_error(
                "ConvParam::ConvParam: "
                "parameter size is different from number of declared dimensions!"));
        }

        for(ck_tile::index_t i = 0; i < num_dim_spatial_; ++i)
        {
            filter_spatial_lengths_[i] = static_cast<ck_tile::long_index_t>(filters_len[i]);
            input_spatial_lengths_[i]  = static_cast<ck_tile::long_index_t>(input_len[i]);
            conv_filter_strides_[i]    = static_cast<ck_tile::long_index_t>(strides[i]);
            conv_filter_dilations_[i]  = static_cast<ck_tile::long_index_t>(dilations[i]);
            input_left_pads_[i]        = static_cast<ck_tile::long_index_t>(left_pads[i]);
            input_right_pads_[i]       = static_cast<ck_tile::long_index_t>(right_pads[i]);

            // XEff = (X - 1) * conv_dilation_w + 1;
            // Wo = (Wi + in_left_pad_w + in_right_pad_w - XEff) / conv_stride_w + 1;
            const ck_tile::long_index_t x_eff =
                (filter_spatial_lengths_[i] - 1) * conv_filter_dilations_[i] + 1;

            output_spatial_lengths_[i] =
                (input_spatial_lengths_[i] + input_left_pads_[i] + input_right_pads_[i] - x_eff) /
                    conv_filter_strides_[i] +
                1;
        }
    }

    ConvParam(ck_tile::long_index_t n_dim,
              ck_tile::long_index_t group_count,
              ck_tile::long_index_t n_batch,
              ck_tile::long_index_t n_out_channels,
              ck_tile::long_index_t n_in_channels,
              const std::vector<ck_tile::long_index_t>& filters_len,
              const std::vector<ck_tile::long_index_t>& input_len,
              const std::vector<ck_tile::long_index_t>& strides,
              const std::vector<ck_tile::long_index_t>& dilations,
              const std::vector<ck_tile::long_index_t>& left_pads,
              const std::vector<ck_tile::long_index_t>& right_pads)
        : num_dim_spatial_(n_dim),
          G_(group_count),
          N_(n_batch),
          K_(n_out_channels),
          C_(n_in_channels),
          filter_spatial_lengths_(filters_len),
          input_spatial_lengths_(input_len),
          output_spatial_lengths_(num_dim_spatial_),
          conv_filter_strides_(strides),
          conv_filter_dilations_(dilations),
          input_left_pads_(left_pads),
          input_right_pads_(right_pads)
    {
        if(static_cast<ck_tile::index_t>(filter_spatial_lengths_.size()) != num_dim_spatial_ ||
           static_cast<ck_tile::index_t>(input_spatial_lengths_.size()) != num_dim_spatial_ ||
           static_cast<ck_tile::index_t>(conv_filter_strides_.size()) != num_dim_spatial_ ||
           static_cast<ck_tile::index_t>(conv_filter_dilations_.size()) != num_dim_spatial_ ||
           static_cast<ck_tile::index_t>(input_left_pads_.size()) != num_dim_spatial_ ||
           static_cast<ck_tile::index_t>(input_right_pads_.size()) != num_dim_spatial_)
        {
            throw(std::runtime_error(
                "ConvParam::ConvParam: "
                "parameter size is different from number of declared dimensions!"));
        }

        for(ck_tile::index_t i = 0; i < num_dim_spatial_; ++i)
        {
            // XEff = (X - 1) * conv_dilation_w + 1;
            // Wo = (Wi + in_left_pad_w + in_right_pad_w - XEff) / conv_stride_w + 1;
            const ck_tile::long_index_t x_eff =
                (filter_spatial_lengths_[i] - 1) * conv_filter_dilations_[i] + 1;

            output_spatial_lengths_[i] =
                (input_spatial_lengths_[i] + input_left_pads_[i] + input_right_pads_[i] - x_eff) /
                    conv_filter_strides_[i] +
                1;
        }
    }

    ck_tile::long_index_t num_dim_spatial_;
    ck_tile::long_index_t G_;
    ck_tile::long_index_t N_;
    ck_tile::long_index_t K_;
    ck_tile::long_index_t C_;

    std::vector<ck_tile::long_index_t> filter_spatial_lengths_;
    std::vector<ck_tile::long_index_t> input_spatial_lengths_;
    std::vector<ck_tile::long_index_t> output_spatial_lengths_;

    std::vector<ck_tile::long_index_t> conv_filter_strides_;
    std::vector<ck_tile::long_index_t> conv_filter_dilations_;

    std::vector<ck_tile::long_index_t> input_left_pads_;
    std::vector<ck_tile::long_index_t> input_right_pads_;

    std::vector<ck_tile::long_index_t> GetOutputSpatialLengths() const
    {
        return output_spatial_lengths_;
    }

    std::size_t GetFlops() const
    {
        // 2 * G * N * K * C * <output spatial lengths product> * <filter spatial lengths product>
        return static_cast<std::size_t>(2) * G_ * N_ * K_ * C_ *
               std::accumulate(std::begin(output_spatial_lengths_),
                               std::next(std::begin(output_spatial_lengths_), num_dim_spatial_),
                               1,
                               std::multiplies<>()) *
               std::accumulate(std::begin(filter_spatial_lengths_),
                               std::next(std::begin(filter_spatial_lengths_), num_dim_spatial_),
                               1,
                               std::multiplies<>());
    }

    template <typename InDataType>
    std::size_t GetInputByte() const
    {
        // sizeof(InDataType) * (G * N * C * <input spatial lengths product>) +
        return sizeof(InDataType) *
               (G_ * N_ * C_ *
                std::accumulate(std::begin(input_spatial_lengths_),
                                std::next(std::begin(input_spatial_lengths_), num_dim_spatial_),
                                1,
                                std::multiplies<>()));
    }

    template <typename WeiDataType>
    std::size_t GetWeightByte() const
    {
        // sizeof(WeiDataType) * (G * K * C * <filter spatial lengths product>) +
        return sizeof(WeiDataType) *
               (G_ * K_ * C_ *
                std::accumulate(std::begin(filter_spatial_lengths_),
                                std::next(std::begin(filter_spatial_lengths_), num_dim_spatial_),
                                1,
                                std::multiplies<>()));
    }

    template <typename OutDataType>
    std::size_t GetOutputByte() const
    {
        // sizeof(OutDataType) * (G * N * K * <output spatial lengths product>);
        return sizeof(OutDataType) * (G_ * N_ * K_ *
                                      std::accumulate(std::begin(output_spatial_lengths_),
                                                      std::end(output_spatial_lengths_),
                                                      static_cast<std::size_t>(1),
                                                      std::multiplies<std::size_t>()));
    }

    template <typename InDataType, typename WeiDataType, typename OutDataType>
    std::size_t GetByte() const
    {
        return GetInputByte<InDataType>() + GetWeightByte<WeiDataType>() +
               GetOutputByte<OutDataType>();
    }
};

CK_TILE_HOST std::string get_conv_param_parser_helper_msg()
{
    std::string msg;

    msg += "Following arguments (depending on number of spatial dims):\n"
           " Number of spatial dimensions (1=Conv1d, 2=Conv2d, 3=Conv3d)\n"
           " G, N, K, C, \n"
           " <filter spatial dimensions>, (ie Y, X for 2D)\n"
           " <input image spatial dimensions>, (ie Hi, Wi for 2D)\n"
           " <strides>, (ie Sy, Sx for 2D)\n"
           " <dilations>, (ie Dy, Dx for 2D)\n"
           " <left padding>, (ie LeftPy, LeftPx for 2D)\n"
           " <right padding>, (ie RightPy, RightPx for 2D)\n";

    return msg;
}

CK_TILE_HOST ck_tile::conv::ConvParam
parse_conv_param(int num_dim_spatial, int arg_idx, char* const argv[])
{
    const ck_tile::long_index_t G = std::stol(argv[arg_idx++]);
    const ck_tile::long_index_t N = std::stol(argv[arg_idx++]);
    const ck_tile::long_index_t K = std::stol(argv[arg_idx++]);
    const ck_tile::long_index_t C = std::stol(argv[arg_idx++]);

    std::vector<ck_tile::long_index_t> filter_spatial_lengths(num_dim_spatial);
    std::vector<ck_tile::long_index_t> input_spatial_lengths(num_dim_spatial);
    std::vector<ck_tile::long_index_t> conv_filter_strides(num_dim_spatial);
    std::vector<ck_tile::long_index_t> conv_filter_dilations(num_dim_spatial);
    std::vector<ck_tile::long_index_t> input_left_pads(num_dim_spatial);
    std::vector<ck_tile::long_index_t> input_right_pads(num_dim_spatial);

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        filter_spatial_lengths[i] = std::stol(argv[arg_idx++]);
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        input_spatial_lengths[i] = std::stol(argv[arg_idx++]);
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        conv_filter_strides[i] = std::stol(argv[arg_idx++]);
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        conv_filter_dilations[i] = std::stol(argv[arg_idx++]);
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        input_left_pads[i] = std::stol(argv[arg_idx++]);
    }

    for(int i = 0; i < num_dim_spatial; ++i)
    {
        input_right_pads[i] = std::stol(argv[arg_idx++]);
    }

    return ck_tile::conv::ConvParam{num_dim_spatial,
                                    G,
                                    N,
                                    K,
                                    C,
                                    filter_spatial_lengths,
                                    input_spatial_lengths,
                                    conv_filter_strides,
                                    conv_filter_dilations,
                                    input_left_pads,
                                    input_right_pads};
}

} // namespace conv
} // namespace ck_tile
