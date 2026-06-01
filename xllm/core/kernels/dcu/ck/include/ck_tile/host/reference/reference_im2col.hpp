// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename InDataType, typename OutDataType, index_t NDimSpatial>
CK_TILE_HOST void reference_im2col(const HostTensor<InDataType>& in_host,
                                   HostTensor<OutDataType>& out_host,
                                   const ck_tile::conv::ConvParam& conv_params)
{
    const long_index_t G = in_host.get_lengths()[0];
    const long_index_t N = in_host.get_lengths()[1];
    const long_index_t C = in_host.get_lengths()[2];

    if constexpr(NDimSpatial == 1)
    {
        const long_index_t Wo = conv_params.output_spatial_lengths_[0];
        auto func             = [&](auto g, auto n, auto wo) {
            long_index_t row    = n * Wo + wo;
            long_index_t column = 0;

            for(long_index_t x = 0; x < conv_params.filter_spatial_lengths_[0]; ++x)
            {
                auto wi = static_cast<long_index_t>(wo * conv_params.conv_filter_strides_[0]) +
                          static_cast<long_index_t>(x * conv_params.conv_filter_dilations_[0]) -
                          static_cast<long_index_t>(conv_params.input_left_pads_[0]);

                for(long_index_t c = 0; c < C; ++c)
                {
                    if(wi >= 0 && type_convert<std::size_t>(wi) < in_host.get_lengths()[3])
                    {
                        InDataType v_in          = in_host(g, n, c, wi);
                        out_host(g, row, column) = type_convert<OutDataType>(v_in);
                    }
                    column++;
                }
            }
        };

        make_ParallelTensorFunctor(func, G, N, Wo)(std::thread::hardware_concurrency());
    }
    else if constexpr(NDimSpatial == 2)
    {
        const long_index_t Ho = conv_params.output_spatial_lengths_[0];
        const long_index_t Wo = conv_params.output_spatial_lengths_[1];

        auto func = [&](auto g, auto n, auto ho, auto wo) {
            long_index_t row    = n * Ho * Wo + ho * Wo + wo;
            long_index_t column = 0;

            for(long_index_t y = 0; y < conv_params.filter_spatial_lengths_[0]; ++y)
            {
                auto hi = static_cast<long_index_t>(ho * conv_params.conv_filter_strides_[0]) +
                          static_cast<long_index_t>(y * conv_params.conv_filter_dilations_[0]) -
                          static_cast<long_index_t>(conv_params.input_left_pads_[0]);

                for(long_index_t x = 0; x < conv_params.filter_spatial_lengths_[1]; ++x)
                {
                    auto wi = static_cast<long_index_t>(wo * conv_params.conv_filter_strides_[1]) +
                              static_cast<long_index_t>(x * conv_params.conv_filter_dilations_[1]) -
                              static_cast<long_index_t>(conv_params.input_left_pads_[1]);

                    for(long_index_t c = 0; c < C; ++c)
                    {

                        if(hi >= 0 && type_convert<std::size_t>(hi) < in_host.get_lengths()[3] &&
                           wi >= 0 && type_convert<std::size_t>(wi) < in_host.get_lengths()[4])
                        {
                            InDataType v_in          = in_host(g, n, c, hi, wi);
                            out_host(g, row, column) = type_convert<OutDataType>(v_in);
                        }
                        column++;
                    }
                }
            }
        };

        make_ParallelTensorFunctor(func, G, N, Ho, Wo)(std::thread::hardware_concurrency());
    }
    else if constexpr(NDimSpatial == 3)
    {
        const long_index_t Do = conv_params.output_spatial_lengths_[0];
        const long_index_t Ho = conv_params.output_spatial_lengths_[1];
        const long_index_t Wo = conv_params.output_spatial_lengths_[2];

        auto func = [&](auto g, auto n, auto d_o, auto ho, auto wo) {
            long_index_t row    = n * Do * Ho * Wo + d_o * Ho * Wo + ho * Wo + wo;
            long_index_t column = 0;

            for(long_index_t z = 0; z < conv_params.filter_spatial_lengths_[0]; ++z)
            {
                auto di = static_cast<long_index_t>(d_o * conv_params.conv_filter_strides_[0]) +
                          static_cast<long_index_t>(z * conv_params.conv_filter_dilations_[0]) -
                          static_cast<long_index_t>(conv_params.input_left_pads_[0]);
                for(long_index_t y = 0; y < conv_params.filter_spatial_lengths_[1]; ++y)
                {
                    auto hi = static_cast<long_index_t>(ho * conv_params.conv_filter_strides_[1]) +
                              static_cast<long_index_t>(y * conv_params.conv_filter_dilations_[1]) -
                              static_cast<long_index_t>(conv_params.input_left_pads_[1]);
                    for(long_index_t x = 0; x < conv_params.filter_spatial_lengths_[2]; ++x)
                    {
                        auto wi =
                            static_cast<long_index_t>(wo * conv_params.conv_filter_strides_[2]) +
                            static_cast<long_index_t>(x * conv_params.conv_filter_dilations_[2]) -
                            static_cast<long_index_t>(conv_params.input_left_pads_[2]);
                        for(long_index_t c = 0; c < C; ++c)
                        {
                            if(di >= 0 &&
                               type_convert<std::size_t>(di) < in_host.get_lengths()[3] &&
                               hi >= 0 &&
                               type_convert<std::size_t>(hi) < in_host.get_lengths()[4] &&
                               wi >= 0 && type_convert<std::size_t>(wi) < in_host.get_lengths()[5])
                            {
                                InDataType v_in          = in_host(g, n, c, di, hi, wi);
                                out_host(g, row, column) = type_convert<OutDataType>(v_in);
                            }
                            column++;
                        }
                    }
                }
            }
        };

        make_ParallelTensorFunctor(func, G, N, Do, Ho, Wo)(std::thread::hardware_concurrency());
    }
}
} // namespace ck_tile
