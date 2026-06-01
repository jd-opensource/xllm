// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/container/thread_buffer.hpp"

namespace ck_tile {

template <typename DataType_, typename StaticTileDistribution_>
struct static_distributed_tensor
{
    using DataType               = remove_cvref_t<DataType_>;
    using StaticTileDistribution = remove_cvref_t<StaticTileDistribution_>;

    static_assert(StaticTileDistribution::is_static(),
                  "wrong! StaticTileDistribution should be known at compile tile");

    using ThreadTensorDesc =
        remove_cvref_t<decltype(StaticTileDistribution{}.get_ys_to_d_descriptor())>;

    static constexpr index_t kThreadElementSpaceSize = ThreadTensorDesc{}.get_element_space_size();

    CK_TILE_HOST_DEVICE static constexpr auto get_num_of_dimension()
    {
        return StaticTileDistribution::get_num_of_dimension_x();
    }

    CK_TILE_HOST_DEVICE static constexpr auto get_lengths()
    {
        return StaticTileDistribution::get_lengths();
    }

    CK_TILE_HOST_DEVICE static constexpr auto get_tile_distribution()
    {
        return StaticTileDistribution{};
    }

    CK_TILE_HOST_DEVICE static constexpr auto get_distributed_spans()
    {
        return StaticTileDistribution::get_distributed_spans();
    }

    CK_TILE_HOST_DEVICE void initialize(const DataType& x) { thread_buf_.initialize(x); }

    CK_TILE_HOST_DEVICE constexpr const auto& get_thread_buffer() const { return thread_buf_; }

    CK_TILE_HOST_DEVICE constexpr auto& get_thread_buffer() { return thread_buf_; }

    CK_TILE_HOST_DEVICE static constexpr index_t get_thread_buffer_size()
    {
        return kThreadElementSpaceSize;
    }

    template <index_t... YSliceOrigins, index_t... YSliceLengths>
    CK_TILE_HOST_DEVICE auto get_y_sliced_thread_data(sequence<YSliceOrigins...>,
                                                      sequence<YSliceLengths...>) const
    {
        static_assert(sizeof...(YSliceOrigins) == StaticTileDistribution::NDimY &&
                          sizeof...(YSliceLengths) == StaticTileDistribution::NDimY,
                      "wrong!");

        constexpr auto sliced_thread_tensor_desc =
            make_naive_tensor_descriptor_packed(make_tuple(YSliceLengths...));

        thread_buffer<DataType, sliced_thread_tensor_desc.get_element_space_size()>
            sliced_thread_data;

        static_ford<sequence<YSliceLengths...>>{}([&](auto idx) {
            constexpr auto idx_ys = idx + sequence<YSliceOrigins...>{};

            sliced_thread_data(number<sliced_thread_tensor_desc.calculate_offset(idx)>{}) =
                thread_buf_[number<ThreadTensorDesc{}.calculate_offset(idx_ys)>{}];
        });

        return sliced_thread_data;
    }

    template <index_t... YSliceOrigins, index_t... YSliceLengths, typename SlicedThreadData>
    CK_TILE_HOST_DEVICE void set_y_sliced_thread_data(sequence<YSliceOrigins...>,
                                                      sequence<YSliceLengths...>,
                                                      const SlicedThreadData& sliced_thread_data)
    {
        static_assert(sizeof...(YSliceOrigins) == StaticTileDistribution::NDimY &&
                          sizeof...(YSliceLengths) == StaticTileDistribution::NDimY,
                      "wrong!");

        constexpr auto sliced_thread_tensor_desc =
            make_naive_tensor_descriptor_packed(make_tuple(YSliceLengths...));

        static_ford<sequence<YSliceLengths...>>{}([&](auto idx) {
            constexpr auto idx_ys = idx + sequence<YSliceOrigins...>{};

            thread_buf_(number<ThreadTensorDesc{}.calculate_offset(idx_ys)>{}) =
                sliced_thread_data[number<sliced_thread_tensor_desc.calculate_offset(idx)>{}];
        });
    }

    template <typename TileDistributedIndices>
    CK_TILE_HOST_DEVICE constexpr const DataType& operator[](TileDistributedIndices) const
    {
        static_assert(is_static_v<TileDistributedIndices>,
                      "wrong! Tile Distributed Indices should be static");

        constexpr auto y_idx = get_tile_distribution().get_y_indices_from_distributed_indices(
            TileDistributedIndices{});

        return thread_buf_[number<ThreadTensorDesc{}.calculate_offset(y_idx)>{}];
    }

    template <typename TileDistributedIndices>
    CK_TILE_HOST_DEVICE constexpr DataType& operator()(TileDistributedIndices)
    {
        static_assert(is_static_v<TileDistributedIndices>,
                      "wrong! Tile Distributed Indices should be static");

        constexpr auto y_idx = get_tile_distribution().get_y_indices_from_distributed_indices(
            TileDistributedIndices{});

        return thread_buf_(number<ThreadTensorDesc{}.calculate_offset(y_idx)>{});
    }

    //
    thread_buffer<DataType, kThreadElementSpaceSize> thread_buf_;
};

template <typename DataType, typename StaticTileDistribution>
CK_TILE_HOST_DEVICE constexpr auto make_static_distributed_tensor(const StaticTileDistribution&)
{
    return static_distributed_tensor<remove_cvref_t<DataType>,
                                     remove_cvref_t<StaticTileDistribution>>{};
}

template <typename DataType, typename StaticTileDistribution, typename ThreadBuffer>
CK_TILE_HOST_DEVICE constexpr auto make_static_distributed_tensor(const StaticTileDistribution&,
                                                                  ThreadBuffer&& thread_buffer_)
{
    return static_distributed_tensor<remove_cvref_t<DataType>,
                                     remove_cvref_t<StaticTileDistribution>>{thread_buffer_};
}

// get X indices from tuple of tile_distributed_index<>
template <typename StaticTileDistribution, typename DistributedIndices>
CK_TILE_HOST_DEVICE constexpr auto
get_x_indices_from_distributed_indices(StaticTileDistribution tile_distribution,
                                       DistributedIndices distributed_indices)
{
    const auto partition_index = detail::get_partition_index(tile_distribution);
    constexpr auto y_indices =
        tile_distribution.get_y_indices_from_distributed_indices(distributed_indices);

    const auto x_coord = make_tensor_adaptor_coordinate(
        tile_distribution.get_ps_ys_to_xs_adaptor(),
        container_concat(partition_index, to_array<ck_tile::index_t, y_indices.size()>(y_indices)));

    return x_coord.get_bottom_index();
}

template <typename DataType, typename StaticTileDistribution, typename XIndicesPredicate>
CK_TILE_HOST_DEVICE void
set_tile_if(static_distributed_tensor<DataType, StaticTileDistribution>& out_tensor,
            DataType value,
            XIndicesPredicate predicate)
{
    constexpr auto out_spans =
        static_distributed_tensor<DataType, StaticTileDistribution>::get_distributed_spans();
    sweep_tile_span(out_spans[number<0>{}], [&](auto idx0) {
        sweep_tile_span(out_spans[number<1>{}], [&](auto idx1) {
            constexpr auto distributed_indices = make_tuple(idx0, idx1);
            const auto x_indices = get_x_indices_from_distributed_indices(StaticTileDistribution{},
                                                                          distributed_indices);

            if(predicate(x_indices))
            {
                out_tensor(distributed_indices) = value;
            }
        });
    });
}

// this function used inside span loop over
template <typename YLengths, index_t XUnpacks>
CK_TILE_HOST_DEVICE constexpr auto get_y_unpacks_from_x_unpacks(YLengths, number<XUnpacks>)
{
    constexpr auto y_size  = reduce_on_sequence(YLengths{}, multiplies{}, number<1>{});
    constexpr auto y_packs = number<XUnpacks>{};
    static_assert(y_size % y_packs == 0);
    constexpr auto y_slice_size = y_size / y_packs;

    constexpr auto slice_info = slice_sequence(YLengths{}, number<y_slice_size>{});
    constexpr auto unpacks    = slice_info[number<1>{}];
    return unpacks;
}

} // namespace ck_tile
