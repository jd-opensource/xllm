// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/tensor/tile_window_with_dstr_base.hpp"
#include "ck_tile/core/container/sequence.hpp"

namespace ck_tile {

enum struct lds_layout_enum
{
    k32,
    k64,
};

template <index_t KPerBlock>
struct lds_layout_traits;

template <>
struct lds_layout_traits<32>
{
    static constexpr auto value = lds_layout_enum::k32;
};

template <>
struct lds_layout_traits<64>
{
    static constexpr auto value = lds_layout_enum::k64;
};

// Note: this tile window do not support single issue
// you need to use tile_window_linear structure for this purpose
template <typename BottomTensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          lds_layout_enum lds_layout>
struct tile_window_convnd_fwd_lds
    : public tile_window_with_dstr_base<BottomTensorView_, WindowLengths_, StaticTileDistribution_>
{
    using Base =
        tile_window_with_dstr_base<BottomTensorView_, WindowLengths_, StaticTileDistribution_>;

    using DataType = typename Base::DataType;
    using Traits   = typename Base::load_store_traits;
    using vector_t = typename Traits::vector_t;
    using SFC_Ys   = typename Traits::SFC_Ys;
    using TileDstr = typename Base::TileDstr;

    using BottomTensorIndex = typename Base::BottomTensorIndex;

    static constexpr index_t NumAccess = Traits::NumAccess;

    CK_TILE_DEVICE constexpr tile_window_convnd_fwd_lds() = default;

    CK_TILE_DEVICE constexpr tile_window_convnd_fwd_lds(
        const BottomTensorView_& bottom_tensor_view,
        const WindowLengths_& window_lengths,
        const BottomTensorIndex& window_origin,
        const StaticTileDistribution_& tile_distribution)
        : Base{bottom_tensor_view, window_lengths, window_origin, tile_distribution}
    {
        // TODO: support more dtype
        static_assert(std::is_same_v<DataType, fp16_t> && (Base::warp_bytes_per_load == 1024));
        pre_compute_offsets();
    }

    CK_TILE_DEVICE constexpr void pre_compute_offsets()
    {
        const auto window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
            this->tile_dstr_.get_ps_ys_to_xs_adaptor(),
            container_concat(detail::get_partition_index(this->tile_dstr_),
                             array<index_t, Base::NDimY>{0}));

        BottomTensorIndex bottom_tensor_thread_origin_idx_tmp =
            this->window_origin_ + window_adaptor_thread_coord_tmp.get_bottom_index();

        const auto bottom_tensor_thread_coord_tmp = make_tensor_coordinate(
            this->bottom_tensor_view_.get_tensor_descriptor(), bottom_tensor_thread_origin_idx_tmp);

        // pre-compute NumCoord (WindowAdaptorCoord, BottomTensorCoord) bundles to speed up
        // future load/store() calls (might allocate more registers)
        auto window_adaptor_thread_coord = window_adaptor_thread_coord_tmp;
        auto bottom_tensor_thread_coord  = bottom_tensor_thread_coord_tmp;

        static_for<0, NumAccess, 1>{}([&](auto i_access) {
            // tile_row, tile_col -> lds_row, lds_bank
            const index_t tile_row = bottom_tensor_thread_coord.get_hidden_index()[number<1>{}];
            const index_t tile_col = bottom_tensor_thread_coord.get_hidden_index()[number<2>{}];

            // TODO: simplify lds offset computation
            if constexpr(lds_layout == lds_layout_enum::k32)
            {
                const index_t rows_per_load = 64;
                const index_t wid           = (tile_row % 8) / 2;
                const index_t lds_row       = wid * 8 + (tile_row / 8) % 8;
                const index_t base_offset   = tile_row / rows_per_load * 4096 + wid * 1024;
                const index_t bank_offset   = (tile_row % 2) * 64 + tile_col * sizeof(DataType);
                // lds_wrap_num = wave_id * 2
                const index_t wrap_offset = wid * 32;
                const index_t lds_offset =
                    base_offset + ((lds_row % 8) * 128 + bank_offset + wrap_offset) % 1024;
                pre_computed_offsets_(i_access) = lds_offset / sizeof(DataType);
            }
            else if constexpr(lds_layout == lds_layout_enum::k64)
            {
                const index_t rows_per_load = 64;
                const index_t wid           = (tile_row % 8) / 2;
                const index_t lds_row       = wid * 8 + (tile_row / 8) % 8;
                const index_t base_offset =
                    tile_row / rows_per_load * 8192 + (tile_row % 2) * 4096 + wid * 1024;
                const index_t bank_offset = tile_col * sizeof(DataType);
                // lds_wrap_num = wave_id * 2
                const index_t wrap_offset = ((wid * 2 + (tile_row % 2) * 4) % 8) * 16;
                const index_t lds_offset =
                    base_offset + ((lds_row % 8) * 128 + bank_offset + wrap_offset) % 1024;
                pre_computed_offsets_(i_access) = lds_offset / sizeof(DataType);
            }
            else
            {
                static_assert(false, "Invalid lds_layout");
            }

            if constexpr(i_access != (NumAccess - 1))
            {
                constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(i_access);

                constexpr auto idx_diff_ps_ys = container_concat(
                    generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                    idx_diff_ys);

                this->move_window_adaptor_and_bottom_tensor_thread_coordinate(
                    window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
            }
        });
    }

    template <index_t i_access = -1, bool oob_conditional_check = false>
    CK_TILE_DEVICE auto load_asm(number<i_access>                     = {},
                                 bool_constant<oob_conditional_check> = {}) const
    {
        constexpr auto tile_dstr = TileDstr{};
        auto dst_tensor          = make_static_distributed_tensor<DataType>(tile_dstr);
        load_asm(dst_tensor, number<i_access>{}, bool_constant<oob_conditional_check>{});
        return dst_tensor;
    }

    template <typename DstTile, index_t i_access = -1, bool oob_conditional_check = false>
    CK_TILE_DEVICE void load_asm(DstTile& dst_tensor,
                                 number<i_access>                     = {},
                                 bool_constant<oob_conditional_check> = {}) const
    {
        static constexpr index_t YElementSize =
            TileDstr{}.get_ys_to_d_descriptor().get_element_space_size();
        static_assert(YElementSize % Traits::ScalarPerVector == 0);

        constexpr auto tile_dstr = TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumAccess, 1>{}([&](auto ia) {
            // data index [y0, y1, ...]
            constexpr auto idx_ys_start = SFC_Ys::get_index(ia);
            constexpr index_t d = tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys_start);
            static_assert(d % Traits::ScalarPerVector == 0);

            // TODO: add lds support in hcu tensor/buffer backend?
            this->get_bottom_tensor_view().template get_vectorized_elements_raw<vector_t>(
                dst_tensor.get_thread_buffer().template get_as<vector_t>()(d /
                                                                           Traits::ScalarPerVector),
                pre_computed_offsets_[ia]);
        });
    }

    CK_TILE_DEVICE void move(const BottomTensorIndex& step)
    {
        this->window_origin_ += step;

        pre_compute_offsets();
    }

    array<index_t, NumAccess> pre_computed_offsets_;
};

// TODO: use strategy
template <lds_layout_enum LdsLayout,
          typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_>
CK_TILE_DEVICE constexpr auto
make_tile_window_convnd_fwd_lds(const TensorView_& tensor_view,
                                const WindowLengths_& window_lengths,
                                const multi_index<TensorView_::get_num_of_dimension()>& origin,
                                const StaticTileDistribution_& tile_distribution)
{
    return tile_window_convnd_fwd_lds<remove_cvref_t<TensorView_>,
                                      remove_cvref_t<WindowLengths_>,
                                      remove_cvref_t<StaticTileDistribution_>,
                                      LdsLayout>{
        tensor_view, window_lengths, origin, tile_distribution};
}

template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          lds_layout_enum LdsLayout>
CK_TILE_DEVICE void move_tile_window(
    tile_window_convnd_fwd_lds<TensorView_, WindowLengths_, StaticTileDistribution_, LdsLayout>&
        window,
    const typename tile_window_convnd_fwd_lds<TensorView_,
                                              WindowLengths_,
                                              StaticTileDistribution_,
                                              LdsLayout>::BottomTensorIndex& step)
{
    window.move(step);
}

} // namespace ck_tile
