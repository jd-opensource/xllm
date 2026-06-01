// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/arch/hcu_buffer_addressing.hpp"
#include "ck_tile/core/arch/hcu_matrix_addressing.hpp"
#include "ck_tile/core/tensor/tile_window_convnd_fwd_lds.hpp"
#include "ck_tile/core/tensor/tile_window_with_dstr_base.hpp"
#include "ck_tile/core/container/sequence.hpp"

namespace ck_tile {
template <typename BottomTensorView_,
          typename TileShape_,
          typename FilterSize_,
          typename StaticTileDistribution_,
          typename Layout_,
          lds_layout_enum LdsLayout_>
struct tile_window_conv2d_fwd_filter_async_v2
    : public tile_window_with_dstr_base<BottomTensorView_,
                                        TileShape_,
                                        StaticTileDistribution_,
                                        false>
{
    using Base =
        tile_window_with_dstr_base<BottomTensorView_, TileShape_, StaticTileDistribution_, false>;

    using DataType          = typename Base::DataType;
    using Traits            = typename Base::load_store_traits;
    using vector_t          = typename Traits::vector_t;
    using SFC_Ys            = typename Traits::SFC_Ys;
    using BottomTensorIndex = typename Base::BottomTensorIndex;

    static constexpr index_t scalar_per_t_vector =
        vector_traits<remove_cvref_t<DataType>>::vector_size;
    static constexpr index_t scalar_per_x_vector =
        vector_traits<remove_cvref_t<vector_t>>::vector_size;

    static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                  "wrong! X should contain multiple T");

    static constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

    static constexpr auto NumAccess = Traits::NumAccess;

    static constexpr auto R  = FilterSize_::at(number<0>{});
    static constexpr auto S  = FilterSize_::at(number<1>{});
    static constexpr auto RS = R * S;

    // limitation
    static_assert(RS == 1);

    static constexpr index_t TileShapeK = TileShape_::at(number<1>{});

    CK_TILE_DEVICE static constexpr auto get_num_of_access() { return NumAccess; }

    CK_TILE_DEVICE constexpr tile_window_conv2d_fwd_filter_async_v2(
        const BottomTensorView_& bottom_tensor_view,
        const TileShape_& window_lengths,
        const StaticTileDistribution_& tile_distribution)
        : Base{bottom_tensor_view, window_lengths, make_multi_index(0, 0), tile_distribution}
    {
        init();
    }

    CK_TILE_DEVICE void init()
    {
        if constexpr(std::is_same_v<Layout_, tensor_layout::convolution::GKYXC>)
        {
            const auto C = this->bottom_tensor_view_.get_tensor_descriptor()
                               .get_transforms()[number<0>{}]
                               .get_upper_lengths()[number<3>{}];

            offset_delta_rs_ = __builtin_amdgcn_readfirstlane(C);
            offset_delta_c_  = __builtin_amdgcn_readfirstlane(TileShapeK - (RS - 1) * C);

            static_for<0, NumAccess, 1>{}([&](auto i_access) {
                // lds offsets, for m0
                pre_computed_lds_offsets_(i_access) = __builtin_amdgcn_readfirstlane(
                    Base::wg_bytes_per_load * i_access +
                    StaticTileDistribution_::_get_warp_id() * Base::warp_bytes_per_load);

                // lds wrap num
                if constexpr(LdsLayout_ == lds_layout_enum::k32)
                {
                    lds_wrap_num_(i_access) =
                        __builtin_amdgcn_readfirstlane(StaticTileDistribution_::_get_warp_id() * 2);
                }
                else if constexpr(LdsLayout_ == lds_layout_enum::k64)
                {
                    if constexpr(i_access % 2 == 0)
                    {
                        lds_wrap_num_(i_access) = __builtin_amdgcn_readfirstlane(
                            StaticTileDistribution_::_get_warp_id() * 2);
                    }
                    else
                    {
                        lds_wrap_num_(i_access) = __builtin_amdgcn_readfirstlane(
                            (StaticTileDistribution_::_get_warp_id() * 2 + 4) % 8);
                    }
                }
            });
        }
        else
        {
            static_assert(false, "Unsupported layout");
        }
    }

    CK_TILE_DEVICE void init(const BottomTensorIndex& block_window_origin)
    {
        const auto window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
            this->tile_dstr_.get_ps_ys_to_xs_adaptor(),
            container_concat(detail::get_partition_index(this->tile_dstr_),
                             array<index_t, Base::NDimY>{0}));

        BottomTensorIndex bottom_tensor_thread_origin_idx_tmp =
            block_window_origin + window_adaptor_thread_coord_tmp.get_bottom_index();

        const auto bottom_tensor_thread_coord_tmp = make_tensor_coordinate(
            this->bottom_tensor_view_.get_tensor_descriptor(), bottom_tensor_thread_origin_idx_tmp);

        // pre-compute NumCoord (WindowAdaptorCoord, BottomTensorCoord) bundles to speed up
        // future load/store() calls (might allocate more registers)

        auto window_adaptor_thread_coord = window_adaptor_thread_coord_tmp;
        auto bottom_tensor_thread_coord  = bottom_tensor_thread_coord_tmp;

        if constexpr(std::is_same_v<Layout_, tensor_layout::convolution::GKYXC>)
        {
            const index_t buf_stride = this->bottom_tensor_view_.get_buffer_view().stride_;

            buf_res_ = make_wave_buffer_resource(
                this->bottom_tensor_view_.get_buffer_view().p_data_,
                this->bottom_tensor_view_.get_buffer_view().buffer_size_ * sizeof(DataType),
                buf_stride);

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

            static_for<0, NumAccess, 1>{}([&](auto i_access) {
                pre_computed_oob_(i_access) = 0;

                // precompute vindex and voffset
                index_t byte_offset = bottom_tensor_thread_coord.get_offset() * sizeof(DataType);
                pre_computed_vindex_voffset_(i_access)(number<0>{}) =
                    byte_offset >> int(__builtin_log2f(buf_stride));
                pre_computed_vindex_voffset_(i_access)(number<1>{}) =
                    byte_offset & (buf_stride - 1);

                const auto k = bottom_tensor_thread_coord.get_hidden_index()[number<1>{}];
                const auto y = bottom_tensor_thread_coord.get_hidden_index()[number<2>{}];
                const auto x = bottom_tensor_thread_coord.get_hidden_index()[number<3>{}];
                const auto c = bottom_tensor_thread_coord.get_hidden_index()[number<4>{}];

                // oob predicate computation, for padded desc
                if(k >= 0 && k < K && y >= 0 && y < Y && x >= 0 && x < X && c >= 0 && c < C)
                {
                    pre_computed_oob_(i_access) = 1;
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

    CK_TILE_DEVICE void set_window_origin(const BottomTensorIndex& block_window_origin)
    {
        init(block_window_origin);
    }

    template <typename T,
              index_t samp_idx,
              index_t i_access           = -1,
              bool lds_wrap_mode         = true,
              bool oob_conditional_check = true>
    CK_TILE_DEVICE void async_load_asm(T* smem,
                                       number<samp_idx>,
                                       number<i_access>                     = {},
                                       bool_constant<lds_wrap_mode>         = {},
                                       bool_constant<oob_conditional_check> = {}) const
    {
        static_assert(lds_wrap_mode == true);

        auto issue = [&](auto ia) {
            if constexpr(oob_conditional_check)
            {
                hcu_async_struct_buffer_load_asm<remove_cvref_t<DataType>, t_per_x>(
                    reinterpret_cast<DataType*>(reinterpret_cast<uintptr_t>(smem) +
                                                pre_computed_lds_offsets_[ia]),
                    buf_res_,
                    pre_computed_vindex_voffset_[ia][number<0>{}],
                    pre_computed_vindex_voffset_[ia][number<1>{}],
                    0,
                    lds_wrap_num_[ia],
                    pre_computed_oob_[ia],
                    bool_constant<oob_conditional_check>{});
            }
            else
            {
                hcu_async_struct_buffer_load_asm<remove_cvref_t<DataType>, t_per_x>(
                    reinterpret_cast<DataType*>(reinterpret_cast<uintptr_t>(smem) +
                                                pre_computed_lds_offsets_[ia]),
                    buf_res_,
                    pre_computed_vindex_voffset_[ia][number<0>{}],
                    pre_computed_vindex_voffset_[ia][number<1>{}],
                    0,
                    lds_wrap_num_[ia],
                    true,
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

    template <index_t next_samp_idx>
    CK_TILE_DEVICE void advance(number<next_samp_idx>)
    {
        if constexpr((next_samp_idx % RS) == 0)
        {
            move_mls_addr_base(buf_res_, offset_delta_c_ * sizeof(DataType));
        }
        else
        {
            move_mls_addr_base(buf_res_, offset_delta_rs_ * sizeof(DataType));
        }
    }

    array<array<index_t, 2>, NumAccess> pre_computed_vindex_voffset_;
    array<index_t, NumAccess> pre_computed_lds_offsets_;
    array<uint16_t, NumAccess> pre_computed_oob_;

    array<index_t, NumAccess> lds_wrap_num_;
    index_t offset_delta_c_, offset_delta_rs_;

    int32x4_t buf_res_;
};

template <lds_layout_enum LdsLayout_,
          typename TensorView_,
          typename TileShape_,
          typename FilterSize_,
          typename StaticTileDistribution_,
          typename Layout_>
CK_TILE_DEVICE constexpr auto
make_tile_window_conv_fwd_filter_async_v2(const TensorView_& tensor_view,
                                          const TileShape_& window_lengths,
                                          const FilterSize_&,
                                          const StaticTileDistribution_& tile_dstr,
                                          const Layout_&)
{
    if constexpr(std::is_same_v<Layout_, tensor_layout::convolution::GKYXC>)
    {
        return tile_window_conv2d_fwd_filter_async_v2<remove_cvref_t<TensorView_>,
                                                      remove_cvref_t<TileShape_>,
                                                      remove_cvref_t<FilterSize_>,
                                                      remove_cvref_t<StaticTileDistribution_>,
                                                      remove_cvref_t<Layout_>,
                                                      LdsLayout_>{
            tensor_view, window_lengths, tile_dstr};
    }
    else
    {
        static_assert(false, "Not implemented yet");
    }
}

} // namespace ck_tile
