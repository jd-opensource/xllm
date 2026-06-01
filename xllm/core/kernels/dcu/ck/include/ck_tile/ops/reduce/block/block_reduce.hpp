// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include <tuple>

// This file is not support cross warp reduce
namespace ck_tile {

/*
 * TODO: block_tile_reduce_sync() currently has a limitation
 * Y dim must have at least one dim not been reduced
 */
// synchronize reduce result (cross lane reduction and broadcast on replicated dimension)
template <typename AccDistributedTensor_, typename ReduceFunc, bool WithBroadcast = true>
CK_TILE_DEVICE void block_tile_reduce_sync(AccDistributedTensor_& acc_tensor,
                                           const ReduceFunc& reduce_func,
                                           bool_constant<WithBroadcast> = {})
{
    using Dstr             = typename AccDistributedTensor_::StaticTileDistribution;
    using DstrEncode       = typename Dstr::DstrEncode;
    using DstrEncodeDetail = typename DstrEncode::detail;

    constexpr index_t NDimP = Dstr::get_num_of_dimension_p();
    constexpr index_t NDimR = Dstr::get_num_of_dimension_r();

    constexpr index_t idim_p_lane = NDimP - 1;

    const auto ps_idx = detail::get_partition_index(acc_tensor.get_tile_distribution());
    const auto rs_idx = acc_tensor.get_tile_distribution().calculate_rs_index_from_ps_index(ps_idx);

    constexpr index_t thread_buf_size = AccDistributedTensor_::get_thread_buffer_size();

    // loop over thread data
    static_for<0, thread_buf_size, 1>{}([&](auto i) {
        auto v_local = acc_tensor.get_thread_buffer()[i];

        // cross-lane reduce for replication
        // only reduce on R dimension correspond to lane
        // (lane id maps to this R dimension)
        static_for<0, NDimR, 1>{}([&](auto idim_r) {
            // FIXME: nasty to use does_p_own_r_
            if constexpr(DstrEncodeDetail::does_p_own_r_[idim_p_lane][idim_r])
            {
                constexpr index_t r_length = DstrEncode::rs_lengths_[idim_r];

                constexpr index_t lid_over_rid_derivative =
                    DstrEncodeDetail::ps_over_rs_derivative_[idim_p_lane][idim_r];

                static_assert(is_power_of_two_integer(r_length),
                              "wrong! only support power of 2 reduction");

                constexpr index_t nstage = integer_log2_floor(r_length);

                // reduction sweep forward
                static_for<0, nstage, 1>{}([&](auto istage) {
                    constexpr index_t lid_delta =
                        lid_over_rid_derivative * (1 << (nstage - istage - 1));

                    // pull data from remote lane
                    const auto v_remote = warp_shuffle_down(v_local, lid_delta);

                    // reduce
                    v_local = reduce_func(v_local, v_remote);
                });
            }
        });

        if constexpr(WithBroadcast)
        {
            // cross-lane broadcast for replication
            // only broadcast on R dimension correspond to lane
            // (lane id maps to this R dimension)
            static_for<0, NDimR, 1>{}([&](auto idim_r) {
                // FIXME: nasty to use does_p_own_r_
                if constexpr(DstrEncodeDetail::does_p_own_r_[idim_p_lane][idim_r])
                {
                    const index_t r_id = rs_idx[idim_r];

                    constexpr index_t r_length = DstrEncode::rs_lengths_[idim_r];

                    constexpr index_t lid_over_rid_derivative =
                        DstrEncodeDetail::ps_over_rs_derivative_[NDimP - 1][idim_r];

                    static_assert(is_power_of_two_integer(r_length),
                                  "wrong! only support power of 2 reduction");

                    constexpr index_t nstage = integer_log2_floor(r_length);

                    // broadcast sweep backward
                    static_for<0, nstage, 1>{}([&](auto istage) {
                        // do I hold reduced data?
                        const bool do_i_hold_reduced_data = r_id < (1 << istage);

                        constexpr index_t lid_delta = lid_over_rid_derivative * (1 << istage);

                        // pull data from remote lane
                        const auto v_remote = warp_shuffle_up(v_local, lid_delta);

                        // decide whether to update local data with remote data
                        v_local = do_i_hold_reduced_data ? v_local : v_remote;
                    });
                }
            });
        }

        acc_tensor.get_thread_buffer()(i) = v_local;
    });
}

/*
 * this version is faster, using xor to do reduce, no need broadcast anymore
 * TODO: the limitation is to-be-reduced P dim can only mapping to one R dim?
 */
template <typename AccDistributedTensor_, typename ReduceFunc>
CK_TILE_DEVICE void block_tile_reduce_xor_sync(AccDistributedTensor_& acc_tensor,
                                               const ReduceFunc& reduce_func)
{
    using Dstr             = typename AccDistributedTensor_::StaticTileDistribution;
    using DstrEncode       = typename Dstr::DstrEncode;
    using DstrEncodeDetail = typename DstrEncode::detail;

    constexpr index_t NDimP = Dstr::get_num_of_dimension_p();
    constexpr index_t NDimR = Dstr::get_num_of_dimension_r();

    constexpr index_t idim_p_lane = NDimP - 1;

    constexpr index_t thread_buf_size = AccDistributedTensor_::get_thread_buffer_size();

    // loop over thread data
    static_for<0, thread_buf_size, 1>{}([&](auto i) {
        auto v_local = acc_tensor.get_thread_buffer()[i];

        // cross-lane reduce for replication
        // only reduce on R dimension correspond to lane
        // (lane id maps to this R dimension)
        static_for<0, NDimR, 1>{}([&](auto idim_r) {
            // FIXME: nasty to use does_p_own_r_
            if constexpr(DstrEncodeDetail::does_p_own_r_[idim_p_lane][idim_r])
            {
                constexpr index_t r_length = DstrEncode::rs_lengths_[idim_r];

                constexpr index_t lid_over_rid_derivative =
                    DstrEncodeDetail::ps_over_rs_derivative_[idim_p_lane][idim_r];

                static_assert(is_power_of_two_integer(r_length),
                              "wrong! only support power of 2 reduction");

                constexpr index_t nstage = integer_log2_floor(r_length);

                // reduction sweep forward
                static_for<0, nstage, 1>{}([&](auto istage) {
                    // xor
                    index_t src_lane =
                        __lane_id() ^ (number<lid_over_rid_derivative << istage.value>{}.value);

                    // pull data from remote lane
                    const auto v_remote = warp_shuffle(v_local, src_lane);

                    // reduce
                    v_local = reduce_func(v_local, v_remote);
                });
            }
        });

        acc_tensor.get_thread_buffer()(i) = v_local;
    });
}

// FIXME: this is for 2D to 1D reduce only, need to support n-D
template <typename AccDistributedTensor_,
          typename InDistributedTensor_,
          index_t... InReduceDims,
          typename ReduceFunc>
CK_TILE_DEVICE void block_tile_reduce(AccDistributedTensor_& acc_tensor,
                                      const InDistributedTensor_& in_tensor,
                                      sequence<InReduceDims...>,
                                      const ReduceFunc& reduce_func)
{
    constexpr auto I0 = number<0>{};
    constexpr auto I1 = number<1>{};

#if 0
    constexpr auto in_reduce_dims = sequence<InReduceDims...>{};

    constexpr index_t ndim_in        = InDistributedTensor_::get_num_of_dimension();
    constexpr index_t ndim_in_reduce = in_reduce_dims.size();
    constexpr index_t ndim_in_free   = ndim_in - ndim_in_reduce;

    constexpr auto in_free_dims_arr = [&] {
        array<bool, ndim_free> is_free_dims{true};

        for(index_t i = 0; i < ndim_reduce; i++)
        {
            is_free_dims(in_reduce_dims[i]) = false;
        }

        array<index_t, ndim_free> in_free_dims{-1};

        index_t cnt = 0;

        for(index_t i = 0; i < ndim_in; i++)
        {
            if(is_free_dims[i])
            {
                in_free_dims(cnt) = i;

                cnt++
            }
        }

        return is_free_dims;
    }();

    constexpr auto in_free_dims = TO_SEQUENCE(is_free_dims_arr, ndim_in_free);
#else

    constexpr auto spans = InDistributedTensor_::get_distributed_spans();

    // in-thread reduction
    // FIXME: hard coded to be 2D to 1D reduction
    sweep_tile_span(spans[I0], [&](auto dstr_idx_i0) {
        constexpr auto acc_dstr_idx = make_tuple(dstr_idx_i0);

        auto acc = acc_tensor[acc_dstr_idx];

        // FIXME
        sweep_tile_span(spans[I1], [&](auto dstr_idx_i1) {
            constexpr auto in_dstr_idx = make_tuple(dstr_idx_i0, dstr_idx_i1);

            const auto in = in_tensor[in_dstr_idx];

            acc = reduce_func(acc, in);
        });

        acc_tensor(acc_dstr_idx) = acc;
    });
#endif
}

/*
 * TODO: block_tile_reduce() currently has a limitation
 * Y dim must have at least one dim not been reduced
 */
template <typename AccDataType_,
          typename InDistributedTensor_,
          index_t... InReduceDims,
          typename ReduceFunc,
          typename InDataType_>
CK_TILE_DEVICE auto block_tile_reduce(const InDistributedTensor_& in_tensor,
                                      sequence<InReduceDims...> in_reduce_dims,
                                      const ReduceFunc& reduce_func,
                                      const InDataType_& reduce_init)
{
    using InDataType  = typename InDistributedTensor_::DataType;
    using AccDataType = remove_cvref_t<AccDataType_>;

    static_assert(std::is_same_v<InDataType, remove_cvref_t<InDataType_>>, "wrong!");

    // declare acc_tensor
    constexpr auto acc_dstr =
        make_static_tile_distribution(ck_tile::detail::make_reduce_tile_distribution_encoding(
            InDistributedTensor_::get_tile_distribution().get_static_tile_distribution_encoding(),
            sequence<InReduceDims...>{}));

    auto acc_tensor = make_static_distributed_tensor<AccDataType>(acc_dstr);

    // init acc_tensor
    tile_elementwise_inout([&](auto& acc) { acc = type_convert<AccDataType>(reduce_init); },
                           acc_tensor);

    // warp reduce
    block_tile_reduce(acc_tensor, in_tensor, in_reduce_dims, reduce_func);

    return acc_tensor;
}

// this version only support 2D->1D reduce (reduce-dim=seq<0, 1>)
// this version only support in/acc/out datatypes are the same
// this version will call thread/warp+sync in one function call
//
template <typename InDistributedTensor_>
struct BlockReduce2D
{
    using InDistributedTensor = remove_cvref_t<InDistributedTensor_>;
    using InDataType          = typename InDistributedTensor::DataType;

    CK_TILE_HOST_DEVICE BlockReduce2D(const InDistributedTensor& t_, const InDataType& reduce_init_)
        : t(t_), reduce_init(reduce_init_)
    {
    }

    CK_TILE_HOST_DEVICE constexpr auto MakeDstBlockTile() const
    {
        using ReduceDim = sequence<1>; // hard coded
        constexpr auto acc_dstr =
            make_static_tile_distribution(ck_tile::detail::make_reduce_tile_distribution_encoding(
                InDistributedTensor::get_tile_distribution()
                    .get_static_tile_distribution_encoding(),
                ReduceDim{}));

        auto dst_ = make_static_distributed_tensor<InDataType>(acc_dstr);
        // init acc_tensor
        tile_elementwise_inout([&](auto& x_) { x_ = type_convert<InDataType>(reduce_init); }, dst_);
        return dst_;
    }

    // return number of pixels each lane need to reduce
    CK_TILE_HOST_DEVICE constexpr auto get_reduce_length_y() const
    {
        constexpr auto spans = InDistributedTensor::get_distributed_spans();
    }

    // Here ReducePacksPerXDim is not the same meaning as that in static_uford/sweep_tile_uspan
    // this is number of packs along the X-dim. We need to compute the Unpacks along the Y dim
    // internally
    // For simplicity, we just support along the row dimension, ReducePacksPerXDim is always 2
    // element , and the first element is always ignored For simplicity, will always try from
    // right-to-left to find alone which Y dim to split
    template <typename ReduceFunc,
              typename ReduceSyncFunc,
              typename ReducePacksPerXDim = uniform_sequence_gen_t<2, 1>>
    CK_TILE_HOST_DEVICE auto operator()(const ReduceFunc& reduce_func,
                                        const ReduceSyncFunc& reduce_sync_func,
                                        ReducePacksPerXDim = {}) const
    {
        constexpr auto spans = InDistributedTensor::get_distributed_spans();

        constexpr auto row_y_unpacks = [&]() {
            constexpr auto row_y_lengths = typename decltype(spans[number<1>{}])::Impl{};
            constexpr auto row_y_size =
                reduce_on_sequence(row_y_lengths, multiplies{}, number<1>{});
            constexpr auto row_y_packs = ReducePacksPerXDim{}.at(number<1>{});

            static_assert(row_y_size % row_y_packs == 0);

            constexpr auto row_y_slice_size = row_y_size / row_y_packs;

            constexpr auto slice_info = slice_sequence(row_y_lengths, number<row_y_slice_size>{});
            constexpr auto unpacks    = slice_info[number<1>{}];
            return unpacks;
        }();

        auto acc_tensor = MakeDstBlockTile();

        // in-thread reduction
        // FIXME: hard coded to be 2D to 1D reduction
        sweep_tile_span(spans[number<0>{}], [&](auto dstr_idx_i0) {
            constexpr auto acc_dstr_idx = make_tuple(dstr_idx_i0);

            auto acc = acc_tensor[acc_dstr_idx];

            sweep_tile_uspan(
                spans[number<1>{}],
                [&](auto... dstr_idx_i1) {
                    acc = reduce_func(acc, t[make_tuple(dstr_idx_i0, dstr_idx_i1)]...);
                },
                row_y_unpacks);

            acc_tensor(acc_dstr_idx) = acc;
        });

        // TODO: always use xor to do cross-lane reduce
        block_tile_reduce_xor_sync(acc_tensor, reduce_sync_func);

        return acc_tensor;
    }

    template <typename ReduceFunc>
    CK_TILE_HOST_DEVICE auto operator()(const ReduceFunc& reduce_func) const
    {
        return operator()(reduce_func, reduce_func);
    }

    InDistributedTensor t;
    InDataType reduce_init;
};

// deduction guide
template <typename T>
CK_TILE_HOST_DEVICE_EXTERN BlockReduce2D(const T&, const typename T::DataType&)->BlockReduce2D<T>;

} // namespace ck_tile
