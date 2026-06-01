// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/algorithm/space_filling_curve.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/container/thread_buffer.hpp"
#include "ck_tile/core/container/statically_indexed_array.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/tensor/tile_elementwise.hpp"
#include "ck_tile/core/utility/transpose_vectors.hpp"

namespace ck_tile {
namespace detail {

template <typename OutTensor, typename InTensor>
CK_TILE_DEVICE void shuffle_tile_impl_in_thread(OutTensor& out_tensor, const InTensor& in_tensor)
{
    constexpr auto I0 = number<0>{};

    using DataType = typename InTensor::DataType;

    constexpr auto y_in_desc  = InTensor::get_tile_distribution().get_ys_to_d_descriptor();
    constexpr auto y_out_desc = OutTensor::get_tile_distribution().get_ys_to_d_descriptor();

    // y_dim_out_to_in
    constexpr auto get_rh_major_minor_to_y = [](auto dstr_tensor) {
        using DstrEncode = typename decltype(dstr_tensor.get_tile_distribution())::DstrEncode;

        map<array<index_t, 2>, index_t> rh_major_minor_to_y_;

        static_for<0, DstrEncode::NDimY, 1>{}([&](auto i) {
            constexpr index_t rh_major = DstrEncode::ys_to_rhs_major_[i];
            constexpr index_t rh_minor = DstrEncode::ys_to_rhs_minor_[i];

            rh_major_minor_to_y_({rh_major, rh_minor}) = i;
        });

        return rh_major_minor_to_y_;
    };

    constexpr auto rh_major_minor_to_y_in  = get_rh_major_minor_to_y(InTensor{});
    constexpr auto rh_major_minor_to_y_out = get_rh_major_minor_to_y(OutTensor{});

    constexpr auto y_dim_out_to_in = [&] {
        map<index_t, index_t> y_dim_out_to_in_;

        for(const auto& [rh_major_minor, y_out] : rh_major_minor_to_y_out)
        {
            y_dim_out_to_in_(y_out) = rh_major_minor_to_y_in[rh_major_minor];
        }

        return y_dim_out_to_in_;
    }();

    //
    constexpr index_t NDimY = InTensor::get_tile_distribution().get_num_of_dimension_y();

    constexpr auto y_lengths = to_sequence(y_in_desc.get_lengths());

    // input and output vector dim in the order of input Y dims
    constexpr index_t y_dim_vec_in  = NDimY - 1;
    constexpr index_t y_dim_vec_out = y_dim_out_to_in[NDimY - 1];

    // vector lengths
    constexpr index_t vec_length_in  = y_lengths[y_dim_vec_in];
    constexpr index_t vec_length_out = y_lengths[y_dim_vec_out];

    // # of vectors
    constexpr index_t num_vec_in  = vec_length_out;
    constexpr index_t num_vec_out = vec_length_in;

    using InVec  = array<DataType, vec_length_in>;
    using OutVec = array<DataType, vec_length_out>;

    // using InVec  = typename InVec::type;
    // using OutVec = typename OutVec::type;

    // SFC
    constexpr auto scalars_per_access_arr = generate_array(
        [&](auto i) { return (i == y_dim_vec_in or i == y_dim_vec_out) ? y_lengths[i] : 1; },
        number<NDimY>{});

    constexpr auto scalars_per_access = TO_SEQUENCE(scalars_per_access_arr, NDimY);

    using SFC_Y = space_filling_curve<decltype(y_lengths),
                                      typename arithmetic_sequence_gen<0, NDimY, 1>::type,
                                      decltype(scalars_per_access)>;

    constexpr index_t num_access = SFC_Y::get_num_of_access();

    static_assert(num_access > 0, "wrong! num_access should be larger than 0");

    // in/out vectors to be transposed
    thread_buffer<InVec, num_vec_in> in_vectors;
    thread_buffer<OutVec, num_vec_out> out_vectors;

    // loop over SFC and do transpose
    static_for<0, num_access, 1>{}([&](auto iAccess) {
        // data index [y0, y1, ...] in the order of input tensor
        constexpr auto idx_y_start = SFC_Y::get_index(iAccess);

        // get input vectors
        static_for<0, num_vec_in, 1>{}([&](auto i) {
            constexpr auto idx_y_in = generate_tuple(
                [&](auto ii) {
                    return ii == y_dim_vec_out ? idx_y_start[ii] + i : idx_y_start[ii];
                },
                number<NDimY>{});

            constexpr index_t in_offset = y_in_desc.calculate_offset(idx_y_in);
            static_assert(in_offset % vec_length_in == 0);

            in_vectors(i).template get_as<InVec>()(I0) =
                in_tensor.get_thread_buffer()
                    .template get_as<InVec>()[number<in_offset / vec_length_in>{}];
        });

        // transpose
        transpose_vectors<DataType, num_vec_in, num_vec_out>{}(in_vectors, out_vectors);

        // set output vectors
        static_for<0, num_vec_out, 1>{}([&](auto i) {
            constexpr auto idx_y_out_tmp = generate_array(
                [&](auto ii) { return ii == y_dim_vec_in ? idx_y_start[ii] + i : idx_y_start[ii]; },
                number<NDimY>{});

            constexpr auto idx_y_out =
                container_reorder_given_new2old(idx_y_out_tmp, y_dim_out_to_in);

            constexpr index_t out_offset = y_out_desc.calculate_offset(idx_y_out);
            static_assert(out_offset % vec_length_out == 0);

            out_tensor.get_thread_buffer().template set_as<OutVec>(
                number<out_offset / vec_length_out>{},
                out_vectors[i].template get_as<OutVec>()[I0]);
        });
    });
}

} // namespace detail

template <typename OutTensor, typename InTensor>
CK_TILE_DEVICE void shuffle_tile(OutTensor& out, const InTensor& in)
{
    using InDataType  = typename InTensor::DataType;
    using OutDataType = typename OutTensor::DataType;

    using InDstrEncode  = typename InTensor::StaticTileDistribution::DstrEncode;
    using OutDstrEncode = typename OutTensor::StaticTileDistribution::DstrEncode;

    // type convert
    const auto in_tmp = tile_elementwise_in(type_convert<OutDataType, InDataType>, in);

    // shuffle
    if constexpr(InDstrEncode::rs_lengths_ == OutDstrEncode::rs_lengths_ &&
                 InDstrEncode::hs_lengthss_ == OutDstrEncode::hs_lengthss_ &&
                 InDstrEncode::ps_to_rhss_major_ == OutDstrEncode::ps_to_rhss_major_ &&
                 InDstrEncode::ps_to_rhss_minor_ == OutDstrEncode::ps_to_rhss_minor_ &&
                 InDstrEncode::NDimY == OutDstrEncode::NDimY)
    {
        detail::shuffle_tile_impl_in_thread(out, in_tmp);
    }
    else
    {
        static_assert(false, "The shuffle should always happen!");
    }
}

} // namespace ck_tile
