// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/tensor/tile_window_convnd_fwd_lds.hpp"
#include "ck_tile/core/tensor/tile_window_with_dstr_base.hpp"
#include "ck_tile/core/container/sequence.hpp"

namespace ck_tile {

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename FilterLengths_,
          typename StaticTileDistribution_,
          typename Layout_,
          index_t CStep_,
          lds_layout_enum LdsLayout>
struct tile_window_conv2d_fwd_filter_async
    : public tile_window_with_dstr_base<BottomTensorView_,
                                        WindowLengths_,
                                        StaticTileDistribution_,
                                        false>
{
    using Base = tile_window_with_dstr_base<BottomTensorView_,
                                            WindowLengths_,
                                            StaticTileDistribution_,
                                            false>;

    using DataType = typename Base::DataType;
    using Traits   = typename Base::load_store_traits;
    using vector_t = typename Traits::vector_t;
    using SFC_Ys   = typename Traits::SFC_Ys;

    using BottomTensorIndex = typename Base::BottomTensorIndex;

    static constexpr auto NumAccess = Traits::NumAccess;

    static constexpr auto RS = FilterLengths_::at(number<0>{}) * FilterLengths_::at(number<1>{});

    static constexpr index_t CStep = CStep_;

    CK_TILE_DEVICE constexpr tile_window_conv2d_fwd_filter_async() = default;

    CK_TILE_DEVICE constexpr tile_window_conv2d_fwd_filter_async(
        const BottomTensorView_& bottom_tensor_view,
        const WindowLengths_& window_lengths,
        const BottomTensorIndex& window_origin,
        const FilterLengths_&,
        const StaticTileDistribution_& tile_distribution)
        : Base{bottom_tensor_view, window_lengths, window_origin, tile_distribution},
          filter_pos_rs_{0}
    {
        pre_compute_offsets_and_oob(window_origin);
    }

    CK_TILE_DEVICE void pre_compute_offsets_and_oob(const BottomTensorIndex& window_origin)
    {
        const auto window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
            this->tile_dstr_.get_ps_ys_to_xs_adaptor(),
            container_concat(detail::get_partition_index(this->tile_dstr_),
                             array<index_t, Base::NDimY>{0}));

        BottomTensorIndex bottom_tensor_thread_origin_idx_tmp =
            window_origin + window_adaptor_thread_coord_tmp.get_bottom_index();

        const auto bottom_tensor_thread_coord_tmp = make_tensor_coordinate(
            this->bottom_tensor_view_.get_tensor_descriptor(), bottom_tensor_thread_origin_idx_tmp);

        // pre-compute NumCoord (WindowAdaptorCoord, BottomTensorCoord) bundles to speed up
        // future load/store() calls (might allocate more registers)

        auto window_adaptor_thread_coord = window_adaptor_thread_coord_tmp;
        auto bottom_tensor_thread_coord  = bottom_tensor_thread_coord_tmp;

        if constexpr(std::is_same_v<Layout_, tensor_layout::convolution::GKYXC>)
        {
            // TODO: can is_valid be computed directly by tensor desc?
            const auto K = this->bottom_tensor_view_.get_tensor_descriptor()
                               .get_transforms()[number<0>{}]
                               .get_upper_lengths()[number<0>{}];
            const auto Y = this->bottom_tensor_view_.get_tensor_descriptor()
                               .get_transforms()[number<0>{}]
                               .get_upper_lengths()[number<1>{}];
            const auto X = this->bottom_tensor_view_.get_tensor_descriptor()
                               .get_transforms()[number<0>{}]
                               .get_upper_lengths()[number<2>{}];
            const auto C = this->bottom_tensor_view_.get_tensor_descriptor()
                               .get_transforms()[number<0>{}]
                               .get_upper_lengths()[number<3>{}];

            // TODO: consider dilation and group
            offset_delta_rs_ = C;
            offset_delta_c_  = CStep - (RS - 1) * C;

            static_for<0, NumAccess, 1>{}([&](auto i_access) {
                // lds offsets, for m0
                pre_computed_lds_offsets_(i_access) =
                    Base::wg_bytes_per_load * i_access +
                    StaticTileDistribution_::_get_warp_id() * Base::warp_bytes_per_load;

                pre_computed_elem_offsets_(i_access) = bottom_tensor_thread_coord.get_offset();
                pre_computed_oob_(i_access)          = 0;

                // TODO: can is_valid be computed directly by tensor desc?
                const auto k = bottom_tensor_thread_coord.get_hidden_index()[number<1>{}];
                const auto y = bottom_tensor_thread_coord.get_hidden_index()[number<2>{}];
                const auto x = bottom_tensor_thread_coord.get_hidden_index()[number<3>{}];
                const auto c = bottom_tensor_thread_coord.get_hidden_index()[number<4>{}];

                // oob predicate computation, for padded desc
                if(k >= 0 && k < K && y >= 0 && y < Y && x >= 0 && x < X && c >= 0 && c < C)
                {
                    pre_computed_oob_(i_access) = 1;
                }

                // lds wrap num
                if constexpr(LdsLayout == lds_layout_enum::k32)
                {
                    lds_wrap_num_(i_access) = StaticTileDistribution_::_get_warp_id() * 2;
                }
                else if constexpr(LdsLayout == lds_layout_enum::k64)
                {
                    if constexpr(i_access % 2 == 0)
                    {
                        lds_wrap_num_(i_access) = StaticTileDistribution_::_get_warp_id() * 2;
                    }
                    else
                    {
                        lds_wrap_num_(i_access) =
                            (StaticTileDistribution_::_get_warp_id() * 2 + 4) % 8;
                    }
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
        else
        {
            static_assert(false, "Unsupported layout");
        }
    }

    template <typename T,
              index_t i_access           = -1,
              bool lds_wrap_mode         = false,
              bool oob_conditional_check = true>
    CK_TILE_DEVICE void async_load_asm(T* smem,
                                       number<i_access>                     = {},
                                       bool_constant<lds_wrap_mode>         = {},
                                       bool_constant<oob_conditional_check> = {}) const
    {
        auto issue = [&](auto ia) {
            // read from bottom tensor
            if constexpr(lds_wrap_mode)
            {
                this->get_bottom_tensor_view()
                    .template async_get_vectorized_elements_wrapped_asm<vector_t>(
                        reinterpret_cast<DataType*>(reinterpret_cast<uintptr_t>(smem) +
                                                    pre_computed_lds_offsets_[ia]),
                        pre_computed_elem_offsets_[ia],
                        0,
                        lds_wrap_num_[ia],
                        pre_computed_oob_[ia],
                        bool_constant<oob_conditional_check>{});
            }
            else
            {
                this->get_bottom_tensor_view().template async_get_vectorized_elements_asm<vector_t>(
                    reinterpret_cast<DataType*>(reinterpret_cast<uintptr_t>(smem) +
                                                pre_computed_lds_offsets_[ia]),
                    pre_computed_elem_offsets_[ia],
                    0,
                    pre_computed_oob_[ia],
                    bool_constant<oob_conditional_check>{});
            }
        };

        if constexpr(i_access < 0)
        {
            static_for<0, NumAccess, 1>{}([&](auto ia) { issue(ia); });
        }
        else
        {
            static_assert(i_access < NumAccess);
            issue(number<i_access>{});
        }
    }

    CK_TILE_DEVICE void advance()
    {
        // TODO: consider oob_predicate update?
        // advance rs
        ++filter_pos_rs_;
        index_t offset_delta = offset_delta_rs_;

        if(filter_pos_rs_ == RS)
        {
            // reset rs
            filter_pos_rs_ = 0;

            // advance r
            offset_delta = offset_delta_c_;
        }

        static_for<0, NumAccess, 1>{}(
            [&](auto i_access) { pre_computed_elem_offsets_(i_access) += offset_delta; });
    }

    CK_TILE_DEVICE void set_window_origin(const BottomTensorIndex& new_window_origin)
    {
        this->window_origin_ = new_window_origin;

        pre_compute_offsets_and_oob(new_window_origin);
    }

    array<index_t, NumAccess> pre_computed_elem_offsets_;

    array<uint16_t, NumAccess> pre_computed_oob_;

    array<index_t, NumAccess> pre_computed_lds_offsets_;

    array<index_t, NumAccess> lds_wrap_num_;

    index_t filter_pos_rs_;

    index_t offset_delta_c_, offset_delta_rs_;
};

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          typename Layout_,
          index_t CStep_,
          lds_layout_enum LdsLayout>
struct tile_window_conv2d_fwd_filter_async<BottomTensorView_,
                                           WindowLengths_,
                                           sequence<1, 1>,
                                           StaticTileDistribution_,
                                           Layout_,
                                           CStep_,
                                           LdsLayout>
    : public tile_window_with_dstr_base<BottomTensorView_,
                                        WindowLengths_,
                                        StaticTileDistribution_,
                                        false>
{
    using Base = tile_window_with_dstr_base<BottomTensorView_,
                                            WindowLengths_,
                                            StaticTileDistribution_,
                                            false>;

    using DataType = typename Base::DataType;
    using Traits   = typename Base::load_store_traits;
    using vector_t = typename Traits::vector_t;
    using SFC_Ys   = typename Traits::SFC_Ys;

    using BottomTensorIndex = typename Base::BottomTensorIndex;

    using FilterLengths_ = sequence<1, 1>;

    static constexpr auto NumAccess = Traits::NumAccess;

    static constexpr index_t CStep = CStep_;

    CK_TILE_DEVICE constexpr tile_window_conv2d_fwd_filter_async() = default;

    CK_TILE_DEVICE constexpr tile_window_conv2d_fwd_filter_async(
        const BottomTensorView_& bottom_tensor_view,
        const WindowLengths_& window_lengths,
        const BottomTensorIndex& window_origin,
        const FilterLengths_&,
        const StaticTileDistribution_& tile_distribution)
        : Base{bottom_tensor_view, window_lengths, window_origin, tile_distribution}
    {
        pre_compute_offsets_and_oob(window_origin);
    }

    CK_TILE_DEVICE void pre_compute_offsets_and_oob(const BottomTensorIndex& window_origin)
    {
        const auto window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
            this->tile_dstr_.get_ps_ys_to_xs_adaptor(),
            container_concat(detail::get_partition_index(this->tile_dstr_),
                             array<index_t, Base::NDimY>{0}));

        BottomTensorIndex bottom_tensor_thread_origin_idx_tmp =
            window_origin + window_adaptor_thread_coord_tmp.get_bottom_index();

        const auto bottom_tensor_thread_coord_tmp = make_tensor_coordinate(
            this->bottom_tensor_view_.get_tensor_descriptor(), bottom_tensor_thread_origin_idx_tmp);

        // pre-compute NumCoord (WindowAdaptorCoord, BottomTensorCoord) bundles to speed up
        // future load/store() calls (might allocate more registers)

        auto window_adaptor_thread_coord = window_adaptor_thread_coord_tmp;
        auto bottom_tensor_thread_coord  = bottom_tensor_thread_coord_tmp;

        if constexpr(std::is_same_v<Layout_, tensor_layout::convolution::GKYXC>)
        {
            // TODO: can is_valid be computed directly by tensor desc?
            const auto K = this->bottom_tensor_view_.get_tensor_descriptor()
                               .get_transforms()[number<0>{}]
                               .get_upper_lengths()[number<0>{}];
            const auto C = this->bottom_tensor_view_.get_tensor_descriptor()
                               .get_transforms()[number<0>{}]
                               .get_upper_lengths()[number<3>{}];

            static_for<0, NumAccess, 1>{}([&](auto i_access) {
                // lds offsets, for m0
                pre_computed_lds_offsets_(i_access) =
                    Base::wg_bytes_per_load * i_access +
                    StaticTileDistribution_::_get_warp_id() * Base::warp_bytes_per_load;

                pre_computed_elem_offsets_(i_access) = bottom_tensor_thread_coord.get_offset();
                pre_computed_oob_(i_access)          = 0;

                // TODO: can is_valid be computed directly by tensor desc?
                const auto k = bottom_tensor_thread_coord.get_hidden_index()[number<1>{}];
                const auto c = bottom_tensor_thread_coord.get_hidden_index()[number<4>{}];

                // oob predicate computation, for padded desc
                if(k >= 0 && k < K && c >= 0 && c < C)
                {
                    pre_computed_oob_(i_access) = 1;
                }

                // lds wrap num
                if constexpr(LdsLayout == lds_layout_enum::k32)
                {
                    lds_wrap_num_(i_access) = StaticTileDistribution_::_get_warp_id() * 2;
                }
                else if constexpr(LdsLayout == lds_layout_enum::k64)
                {
                    if constexpr(i_access % 2 == 0)
                    {
                        lds_wrap_num_(i_access) = StaticTileDistribution_::_get_warp_id() * 2;
                    }
                    else
                    {
                        lds_wrap_num_(i_access) =
                            (StaticTileDistribution_::_get_warp_id() * 2 + 4) % 8;
                    }
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
        else
        {
            static_assert(false, "Unsupported layout");
        }
    }

    template <typename T,
              index_t i_access           = -1,
              bool lds_wrap_mode         = false,
              bool oob_conditional_check = false>
    CK_TILE_DEVICE void async_load_asm(T* smem,
                                       number<i_access>                     = {},
                                       bool_constant<lds_wrap_mode>         = {},
                                       bool_constant<oob_conditional_check> = {}) const
    {
        auto issue = [&](auto ia) {
            const bool is_src_valid = pre_computed_oob_[ia];

            // read from bottom tensor
            if constexpr(lds_wrap_mode)
            {
                this->get_bottom_tensor_view()
                    .template async_get_vectorized_elements_wrapped_asm<vector_t>(
                        reinterpret_cast<DataType*>(reinterpret_cast<uintptr_t>(smem) +
                                                    pre_computed_lds_offsets_[ia]),
                        pre_computed_elem_offsets_[ia],
                        0,
                        lds_wrap_num_[ia],
                        is_src_valid,
                        bool_constant<oob_conditional_check>{});
            }
            else
            {
                this->get_bottom_tensor_view().template async_get_vectorized_elements_asm<vector_t>(
                    reinterpret_cast<DataType*>(reinterpret_cast<uintptr_t>(smem) +
                                                pre_computed_lds_offsets_[ia]),
                    pre_computed_elem_offsets_[ia],
                    0,
                    is_src_valid,
                    bool_constant<oob_conditional_check>{});
            }
        };

        if constexpr(i_access < 0)
        {
            static_for<0, NumAccess, 1>{}([&](auto ia) { issue(ia); });
        }
        else
        {
            static_assert(i_access < NumAccess);
            issue(number<i_access>{});
        }
    }

    CK_TILE_DEVICE void advance()
    {
        static_for<0, NumAccess, 1>{}(
            [&](auto i_access) { pre_computed_elem_offsets_(i_access) += CStep; });
    }

    CK_TILE_DEVICE void set_window_origin(const BottomTensorIndex& new_window_origin)
    {
        this->window_origin_ = new_window_origin;

        pre_compute_offsets_and_oob(new_window_origin);
    }

    array<index_t, NumAccess> pre_computed_elem_offsets_;

    array<uint16_t, NumAccess> pre_computed_oob_;

    array<index_t, NumAccess> pre_computed_lds_offsets_;

    array<index_t, NumAccess> lds_wrap_num_;
};

template <lds_layout_enum LdsLayout,
          typename TensorView_,
          typename WindowLengths_,
          typename FilterLengths_,
          typename StaticTileDistribution_,
          typename Layout_,
          index_t CStep_>
CK_TILE_DEVICE constexpr auto make_tile_window_conv_fwd_filter_async(
    const TensorView_& tensor_view,
    const WindowLengths_& window_lengths,
    const multi_index<TensorView_::get_num_of_dimension()>& window_origin,
    const FilterLengths_& filter_lengths,
    const StaticTileDistribution_& tile_dstr,
    const Layout_&,
    number<CStep_>)
{
    if constexpr(std::is_same_v<Layout_, tensor_layout::convolution::GKYXC>)
    {
        return tile_window_conv2d_fwd_filter_async<remove_cvref_t<TensorView_>,
                                                   remove_cvref_t<WindowLengths_>,
                                                   remove_cvref_t<FilterLengths_>,
                                                   remove_cvref_t<StaticTileDistribution_>,
                                                   remove_cvref_t<Layout_>,
                                                   CStep_,
                                                   LdsLayout>{
            tensor_view, window_lengths, window_origin, filter_lengths, tile_dstr};
    }
    else
    {
        static_assert(false, "Not implemented yet");
    }
}

} // namespace ck_tile
