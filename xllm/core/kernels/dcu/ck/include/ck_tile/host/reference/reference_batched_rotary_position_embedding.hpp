// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

#include <cassert>
#include <thread>

namespace ck_tile {

template <typename DataType, typename ComputeDataType = float>
CK_TILE_HOST void reference_batched_rotary_position_embedding(const HostTensor<DataType>& input_bsd,
                                                              const HostTensor<DataType>& cos_sd,
                                                              const HostTensor<DataType>& sin_sd,
                                                              bool interleaved,
                                                              HostTensor<DataType>& output_bsd,
                                                              bool use_1_row_sin_cos = false)
{
    assert(cos_sd.get_num_of_dimension() == 2 && sin_sd.get_num_of_dimension() == 2);
    assert(cos_sd.get_length(0) == sin_sd.get_length(0) &&
           cos_sd.get_length(1) == sin_sd.get_length(1));

    const index_t rotary_dim = cos_sd.get_length(1) * 2;
    assert(static_cast<std::size_t>(rotary_dim) <= input_bsd.get_length(2));

    output_bsd.ForEach([&](auto& self, auto i) {
        const index_t i_d = i[2];
        if(rotary_dim <= i_d)
        {
            self(i) = input_bsd(i);
            return;
        }
        assert(i_d < rotary_dim);

        const index_t i_s         = i[1];
        const index_t i_s_cos_sin = (use_1_row_sin_cos ? 0 : i_s);

        const ComputeDataType cos = type_convert<ComputeDataType>(
            interleaved ? cos_sd(i_s_cos_sin, i_d / 2)
                        : cos_sd(i_s_cos_sin, i_d % cos_sd.get_length(1)));
        const ComputeDataType sin = type_convert<ComputeDataType>(
            interleaved ? sin_sd(i_s_cos_sin, i_d / 2)
                        : sin_sd(i_s_cos_sin, i_d % sin_sd.get_length(1)));

        const ComputeDataType half_rotated_input = [&] {
            const index_t i_b = i[0];

            if(interleaved)
            {
                const bool is_even         = (i_d % 2 == 0);
                const index_t pos          = i_d + (is_even ? 1 : -1);
                const ComputeDataType sign = (is_even ? -1 : 1);
                return sign * type_convert<ComputeDataType>(input_bsd(i_b, i_s, pos));
            }
            else
            {
                const index_t half_rdim    = (rotary_dim / 2);
                const index_t pos          = (i_d + half_rdim) % rotary_dim;
                const ComputeDataType sign = (pos < half_rdim ? 1 : -1);
                return sign * type_convert<ComputeDataType>(input_bsd(i_b, i_s, pos));
            }
        }();
        ComputeDataType result =
            type_convert<ComputeDataType>(input_bsd(i)) * cos + half_rotated_input * sin;

        self(i) = type_convert<DataType>(result);
    });
}

} // namespace ck_tile
