// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename Problem_, typename Policy_ = void>
struct BlockReduce2d
{
    // in-thread reduction
    using Problem         = remove_cvref_t<Problem_>;
    using XDataType       = typename Problem::XDataType;
    using ComputeDataType = typename Problem::ComputeDataType;

    CK_TILE_DEVICE constexpr BlockReduce2d() {}

    template <typename XDistributedTensor_,
              typename YDistributedTensor_,
              typename ReduceFunc,
              typename ReducePacksPerXDim = uniform_sequence_gen_t<2, 1>>
    CK_TILE_DEVICE void operator()(const XDistributedTensor_& x_tensor,
                                   YDistributedTensor_& y_tensor,
                                   const ReduceFunc& reduce_func,
                                   ReducePacksPerXDim = {})
    {
        sweep_tile<XDistributedTensor_>(
            [&](auto... idx_) {
                constexpr auto idx_0 = make_tuple(make_tuple(idx_[number<0>{}]...)[number<0>{}]);
                y_tensor(idx_0)      = reduce_func(
                    y_tensor(idx_0), ck_tile::type_convert<ComputeDataType>(x_tensor[idx_])...);
            },
            ReducePacksPerXDim{});
#if 0
        constexpr auto I0 = number<0>{};
        constexpr auto I1 = number<1>{};
        constexpr auto spans = XDistributedTensor_::get_distributed_spans();

        // FIXME: hard coded to reduce 2nd axis
        sweep_tile_span(spans[I0], [&](auto dstr_idx_i0) {
            constexpr auto y_dstr_idx = make_tuple(dstr_idx_i0);

            auto y = y_tensor[y_dstr_idx];

            sweep_tile_span(spans[I1], [&](auto dstr_idx_i1) {
                constexpr auto in_dstr_idx = make_tuple(dstr_idx_i0, dstr_idx_i1);
                const auto x = ck_tile::type_convert<ComputeDataType>(x_tensor[in_dstr_idx]);

                y = reduce_func(y, x);
            });

            y_tensor(y_dstr_idx) = y;
        });
#endif
    }

    template <typename XDistributedTensor_>
    CK_TILE_DEVICE static auto MakeYBlockTile()
    {
        static_assert(std::is_same_v<XDataType, typename XDistributedTensor_::DataType>, "wrong!");

        // FIXME: hard coded to reduce 2nd axis
        constexpr auto reduce_dims = sequence<1>{};

        constexpr auto dstr =
            make_static_tile_distribution(detail::make_reduce_tile_distribution_encoding(
                XDistributedTensor_::get_tile_distribution()
                    .get_static_tile_distribution_encoding(),
                reduce_dims));

        auto tensor = make_static_distributed_tensor<ComputeDataType>(dstr);

        return tensor;
    }

    template <typename XDistributedTensor_,
              typename ReduceFunc,
              typename ReducePacksPerXDim = uniform_sequence_gen_t<2, 1>>
    CK_TILE_DEVICE auto operator()(const XDistributedTensor_& x_tensor,
                                   const ComputeDataType& reduce_init,
                                   const ReduceFunc& reduce_func,
                                   ReducePacksPerXDim = {})
    {
        auto y_tensor = MakeYBlockTile<XDistributedTensor_>();
        set_tile(y_tensor, reduce_init);
        (*this)(x_tensor, y_tensor, reduce_func, ReducePacksPerXDim{});

        return y_tensor;
    }
};

template <typename Problem_, typename Policy_ = void>
struct BlockReduce2dSync
{
    using Problem = remove_cvref_t<Problem_>;

    template <typename YDistributedTensor_, typename ReduceFunc>
    CK_TILE_DEVICE void operator()(YDistributedTensor_& y_tensor, const ReduceFunc& reduce_func)
    {
        using Dstr             = typename YDistributedTensor_::StaticTileDistribution;
        using DstrEncode       = typename Dstr::DstrEncode;
        using DstrEncodeDetail = typename DstrEncode::detail;

        constexpr index_t NDimP = Dstr::get_num_of_dimension_p();
        constexpr index_t NDimR = Dstr::get_num_of_dimension_r();

        constexpr index_t idim_p_lane = NDimP - 1;

        // const auto ps_idx = make_array<index_t>(get_warp_id(), get_lane_id());
        // const auto rs_idx =
        //     y_tensor.get_tile_distribution().calculate_rs_index_from_ps_index(ps_idx);

        constexpr index_t thread_buf_size = YDistributedTensor_::get_thread_buffer_size();

        // loop over thread data
        static_for<0, thread_buf_size, 1>{}([&](auto i) {
            auto v_local = y_tensor.get_thread_buffer()[i];

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
                            (__lane_id()) ^
                            (number<lid_over_rid_derivative << istage.value>{}.value);

                        // pull data from remote lane
                        const auto v_remote = warp_shuffle(v_local, src_lane);

                        // reduce
                        v_local = reduce_func(v_local, v_remote);
                    });
                }
            });

            // TODO - Do we need to broadcast to other lane?
            y_tensor.get_thread_buffer()(i) = v_local;
        });
    }
};

template <typename Problem_, typename Policy_ = void>
struct BlockReduce2dCrossWarpSync
{
    using Problem    = remove_cvref_t<Problem_>;
    using BlockShape = typename Problem::BlockShape;

    template <typename YDistributedTensor_>
    CK_TILE_DEVICE static constexpr index_t GetReduceWarps()
    {
        constexpr index_t num_reduce_warps = [&]() {
            using Dstr             = typename YDistributedTensor_::StaticTileDistribution;
            using DstrEncode       = typename Dstr::DstrEncode;
            using DstrEncodeDetail = typename DstrEncode::detail;

            constexpr index_t NDimR = Dstr::get_num_of_dimension_r();

            constexpr index_t idim_p_warp = 0;

            index_t len_ = 1;
            static_for<0, NDimR, 1>{}([&](auto idim_r) {
                if constexpr(DstrEncodeDetail::does_p_own_r_[idim_p_warp][idim_r])
                {
                    constexpr index_t r_length = DstrEncode::rs_lengths_[idim_r];
                    len_ *= r_length;
                }
            });
            return len_;
        }();
        return num_reduce_warps;
    }

    // return in byte
    template <typename YDistributedTensor_>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        using DataType = typename YDistributedTensor_::DataType;
        // constexpr auto num_reduce_warps = GetReduceWarps<YDistributedTensor_>();

        constexpr index_t thread_buf_size = YDistributedTensor_::get_thread_buffer_size();

        // we need to store all data from every wave into smem
        // e.g. 2x2 reduce along N
        //     -------------> reduce N
        //    | w0 | w1 |   ___>      | w01 |
        //    | w2 | w3 |             | w23 |
        //
        //   -> store data from every wave into LDS
        //
        //
        //     -------------> reduce N
        //    | w0 | w1 | w2 | w3 |   ----->  | w0123 |
        //
        //   -> also store data from every wave into LDS
        constexpr index_t num_warps = BlockShape::BlockSize / warpSize;
        return num_warps * thread_buf_size * sizeof(DataType);
    }

    template <typename YDistributedTensor_, typename ReduceFunc>
    CK_TILE_DEVICE void
    operator()(YDistributedTensor_& y_tensor, void* smem, const ReduceFunc& reduce_func)
    {
        using DataType = typename YDistributedTensor_::DataType;

        constexpr index_t thread_buf_size = YDistributedTensor_::get_thread_buffer_size();

        DataType* smem_ptr              = reinterpret_cast<DataType*>(smem);
        const index_t lane_id           = get_lane_id();
        //TODO: support get_sub_warp_id for WASP
        const index_t warp_id           = get_warp_id();
        constexpr auto num_reduce_warps = GetReduceWarps<YDistributedTensor_>();
        constexpr index_t num_warps     = BlockShape::BlockSize / warpSize;
        const index_t smem_offset       = warp_id;

        // skip if nonthing to do
        if constexpr(num_reduce_warps == 1)
            return;

        // store into smem only for lane-0 within one warp
        if(lane_id == 0)
        {
            static_for<0, thread_buf_size, 1>{}([&](auto i) {
                smem_ptr[smem_offset + i * num_warps] = y_tensor.get_thread_buffer()[i];
            });
        }
        block_sync_lds();

        // load from smem. here we let everythread to do compute :)
        index_t local_warp_id = warp_id / num_reduce_warps;
        index_t local_smem_os = local_warp_id * num_reduce_warps;
        DataType all_scratch[thread_buf_size * num_reduce_warps];
        static_for<0, thread_buf_size, 1>{}([&](auto i_0) {
            static_for<0, num_reduce_warps, 1>{}([&](auto i_1) {
                all_scratch[i_0 * num_reduce_warps + i_1] =
                    smem_ptr[i_0 * num_warps + local_smem_os + i_1];
            });
        });
        block_sync_lds(); // TODO: we don't need sync here

        static_for<0, thread_buf_size, 1>{}([&](auto i_0) {
            // TODO: use descriptor for this
            auto v_local = all_scratch[i_0 * num_reduce_warps];

            // further reduce mean/var
            static_for<0, num_reduce_warps - 1, 1>{}([&](auto i_1_n1) {
                constexpr auto i_1      = number<i_1_n1 + 1>{};
                const DataType v_remote = all_scratch[i_0 * num_reduce_warps + i_1];

                // reduce
                v_local = reduce_func(v_local, v_remote);
            });

            y_tensor.get_thread_buffer()(i_0) = v_local;
        });
    }
};

} // namespace ck_tile
