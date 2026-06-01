// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/arch/utility.hpp"
#include "ck_tile/core/algorithm/space_filling_curve.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/tensor/static_distributed_tensor.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

// Note: this tile window do not support single issue
// you need to use tile_window_linear structure for this purpose
template <typename BottomTensorView_, typename WindowLengths_, typename StaticTileDistribution_>
struct tile_window_lds
{
    using BottomTensorView = remove_reference_t<BottomTensorView_>;
    using WindowLengths    = remove_cvref_t<WindowLengths_>;
    using TileDstr         = remove_cvref_t<StaticTileDistribution_>;

    using WindowAdaptor    = typename TileDstr::PsYs2XsAdaptor;
    using BottomTensorDesc = typename BottomTensorView::TensorDesc;

    using DataType = remove_cvref_t<typename BottomTensorView::DataType>;

    static constexpr index_t NDimWindowAdaptorTop = WindowAdaptor::get_num_of_top_dimension();
    static constexpr index_t NDimBottomTensor     = BottomTensorDesc::get_num_of_dimension();

    static constexpr index_t NDimP = TileDstr::get_num_of_dimension_p();
    static constexpr index_t NDimY = TileDstr::get_num_of_dimension_y();

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};

    static_assert(ck_tile::is_known_at_compile_time<WindowLengths>::value,
                  "wrong! lengths should be static");
    static_assert(TileDstr::is_static(), "wrong!");

    static_assert(NDimBottomTensor == WindowAdaptor::get_num_of_bottom_dimension(),
                  "wrong! inconsistent # of diemsnions");

    using AdaptorTopIndex   = array<index_t, NDimWindowAdaptorTop>;
    using BottomTensorIndex = array<index_t, NDimBottomTensor>;

    using WindowAdaptorCoord =
        decltype(make_tensor_adaptor_coordinate(WindowAdaptor{}, AdaptorTopIndex{}));

    using BottomTensorCoord =
        decltype(make_tensor_coordinate(BottomTensorDesc{}, BottomTensorIndex{}));

    struct load_store_traits
    {
        private:
        static constexpr auto get_vector_dim_y_scalar_per_vector()
        {
            const auto [ys_vector_lengths, ys_vector_strides] =
                tile_window_lds::get_window_adaptor_ys_safe_vector_length_strides();

            index_t VectorDimY_      = 0;
            index_t ScalarPerVector_ = 1;

            for(index_t i = 0; i < NDimY; ++i)
            {
                if(ys_vector_strides[i] == 1 && ys_vector_lengths[i] > ScalarPerVector_)
                {
                    ScalarPerVector_ = ys_vector_lengths[i];
                    VectorDimY_      = i;
                }
            }

            return make_tuple(VectorDimY_, ScalarPerVector_);
        }

        public:
        static constexpr index_t VectorDimY = get_vector_dim_y_scalar_per_vector().template at<0>();
        static constexpr index_t ScalarPerVector =
            get_vector_dim_y_scalar_per_vector().template at<1>();

        // using vector_type_t = vector_type_maker_t<DataType, ScalarPerVector>;
        // using vector_t      = typename vector_type_t::type;
        using vector_t = thread_buffer<DataType, ScalarPerVector>;

        private:
        static constexpr auto scalars_per_access_ = [] {
            constexpr auto scalars_per_access_arr = generate_array(
                [&](auto i) { return (i == VectorDimY) ? ScalarPerVector : 1; }, number<NDimY>{});

            /// TODO: add non-automatic storage argument support to macro TO_SEQUENCE()
            constexpr auto NDimY_ = NDimY;

            return TO_SEQUENCE(scalars_per_access_arr, NDimY_);
        }();

        static constexpr auto get_space_filling_curve()
        {
            constexpr auto tile_dstr = TileDstr{};

            constexpr auto thread_tensor_lengths_ys =
                to_sequence(tile_dstr.get_ys_to_d_descriptor().get_lengths());

            // FIXME: need logic to judge dim access order
            using DimAccessOrder = typename arithmetic_sequence_gen<0, NDimY, 1>::type;

            return space_filling_curve<decltype(thread_tensor_lengths_ys),
                                       DimAccessOrder,
                                       decltype(scalars_per_access_)>{};
        }

        public:
        using SFC_Ys = decltype(get_space_filling_curve());

        static constexpr index_t NumAccess = SFC_Ys::get_num_of_access();

        static_assert(0 < NumAccess, "Wrong! NumAccess should be larger than 0");
    };

    static constexpr index_t NumAccess = load_store_traits::NumAccess;

    CK_TILE_DEVICE constexpr tile_window_lds() = default;

    CK_TILE_DEVICE constexpr tile_window_lds(const BottomTensorView& bottom_tensor_view,
                                             const WindowLengths& window_lengths,
                                             const BottomTensorIndex& window_origin,
                                             const TileDstr& tile_distribution)
        : bottom_tensor_view_{bottom_tensor_view},
          window_lengths_{window_lengths},
          window_origin_{window_origin},
          tile_dstr_{tile_distribution}
    {
        pre_compute_offsets();
    }

    CK_TILE_DEVICE constexpr void pre_compute_offsets()
    {
        const auto window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
            tile_dstr_.get_ps_ys_to_xs_adaptor(),
            container_concat(detail::get_partition_index(tile_dstr_), array<index_t, NDimY>{0}));

        BottomTensorIndex bottom_tensor_thread_origin_idx_tmp =
            window_origin_ + window_adaptor_thread_coord_tmp.get_bottom_index();

        const auto bottom_tensor_thread_coord_tmp = make_tensor_coordinate(
            bottom_tensor_view_.get_tensor_descriptor(), bottom_tensor_thread_origin_idx_tmp);

        // pre-compute NumCoord (WindowAdaptorCoord, BottomTensorCoord) bundles to speed up
        // future load/store() calls (might allocate more registers)
        using Traits   = load_store_traits;
        using SFC_Ys   = typename Traits::SFC_Ys;
        using vector_t = typename Traits::vector_t;

        auto window_adaptor_thread_coord = window_adaptor_thread_coord_tmp;
        auto bottom_tensor_thread_coord  = bottom_tensor_thread_coord_tmp;

        constexpr auto swizzle_functor = make_swizzle(number<sizeof(vector_t)>{});

        static_for<0, NumAccess, 1>{}([&](auto iAccess) {
            pre_computed_offsets_(iAccess) =
                swizzle_functor(bottom_tensor_thread_coord.get_offset());

            if constexpr(iAccess != (NumAccess - 1))
            {
                constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                constexpr auto idx_diff_ps_ys = container_concat(
                    generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                    idx_diff_ys);

                move_window_adaptor_and_bottom_tensor_thread_coordinate(
                    window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
            }
        });
    }

    CK_TILE_DEVICE static constexpr index_t get_num_of_dimension() { return NDimBottomTensor; }

    CK_TILE_DEVICE static constexpr bool has_static_tile_distribution()
    {
        return TileDstr::is_static();
    }

    CK_TILE_DEVICE constexpr auto get_window_lengths() const { return window_lengths_; }

    CK_TILE_DEVICE constexpr auto get_tile_distribution() const { return tile_dstr_; }

    CK_TILE_DEVICE constexpr auto get_bottom_tensor_view() const { return bottom_tensor_view_; }

    CK_TILE_DEVICE constexpr auto get_window_origin() const { return window_origin_; }

    CK_TILE_DEVICE constexpr void
    set_bottom_tensor_view_data_ptr(typename BottomTensorView::DataType* data)
    {
        bottom_tensor_view_.buf_.p_data_ = data;
    }

    // move thread's window adaptor coordinate and bottom tensor coordinate
    // [p0, p1, ..., y0, y1, ...] ==> [x0, x1, ...] ==> [x0', x1', ...] ==> [offset]
    template <typename ATopIndex>
    CK_TILE_DEVICE void move_window_adaptor_and_bottom_tensor_thread_coordinate(
        WindowAdaptorCoord& window_adaptor_thread_coord,
        BottomTensorCoord& bottom_tensor_thread_coord,
        const ATopIndex& idx_diff_adaptor_top) const
    {
        array<index_t, NDimBottomTensor> idx_diff_adaptor_bottom;

        move_tensor_adaptor_coordinate(tile_dstr_.get_ps_ys_to_xs_adaptor(),
                                       window_adaptor_thread_coord,
                                       idx_diff_adaptor_top,
                                       idx_diff_adaptor_bottom);

        move_tensor_coordinate(bottom_tensor_view_.get_tensor_descriptor(),
                               bottom_tensor_thread_coord,
                               idx_diff_adaptor_bottom);
    }

    // return vector dimension among [y0, y1, ...]
    CK_TILE_DEVICE static constexpr auto get_window_adaptor_ys_safe_vector_length_strides()
    {
        // bottom tensor top dimension vector lengths and strides
        const auto [bottom_tensor_top_dim_vector_lengths, bottom_tensor_top_dim_vector_strides] =
            BottomTensorDesc::get_top_dimension_safe_vector_length_strides();

        // window vector lengths/strides
        const auto window_adaptor_bottom_dim_vector_lengths = bottom_tensor_top_dim_vector_lengths;
        const auto window_adaptor_bottom_dim_vector_strides = bottom_tensor_top_dim_vector_strides;

        // window adaptor [p0, p1, ..., y0, y1, ...]
        array<index_t, WindowAdaptor::get_num_of_hidden_dimension()> window_adaptor_vector_lengths{
            -1};
        array<index_t, WindowAdaptor::get_num_of_hidden_dimension()> window_adaptor_vector_strides{
            -1};

        constexpr auto window_adaptor_bottom_dims =
            WindowAdaptor::get_bottom_dimension_hidden_ids();

        set_container_subset(
            window_adaptor_vector_lengths, // array with
            window_adaptor_bottom_dims, // VGPR Distribution transformed ids（Hs Sequence number）
            window_adaptor_bottom_dim_vector_lengths); // LDS Visble lengths
        set_container_subset(window_adaptor_vector_strides,
                             window_adaptor_bottom_dims,
                             window_adaptor_bottom_dim_vector_strides);

        const auto [window_adaptor_ps_ys_vector_lengths, window_adaptor_ps_ys_vector_strides] =
            WindowAdaptor{}.get_top_dimension_safe_vector_length_strides(
                window_adaptor_vector_lengths, window_adaptor_vector_strides);

        // [y0, y1, ...]
        constexpr auto y_dims = typename arithmetic_sequence_gen<TileDstr::get_num_of_dimension_p(),
                                                                 NDimWindowAdaptorTop,
                                                                 1>::type{};

        return make_tuple(get_container_subset(window_adaptor_ps_ys_vector_lengths, y_dims),
                          get_container_subset(window_adaptor_ps_ys_vector_strides, y_dims));
    }

    CK_TILE_DEVICE constexpr auto get_num_of_access() const { return load_store_traits::NumAccess; }

    CK_TILE_DEVICE auto load_raw() const
    {
        constexpr auto tile_dstr = TileDstr{};
        auto dst_tensor          = make_static_distributed_tensor<DataType>(tile_dstr);
        load_raw(dst_tensor);
        return dst_tensor;
    }

    template <typename DstTile>
    CK_TILE_DEVICE void load_raw(DstTile& dst_tensor) const
    {
        using Traits = load_store_traits;

        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;
        static constexpr index_t YElementSize =
            TileDstr{}.get_ys_to_d_descriptor().get_element_space_size();
        static_assert(YElementSize % Traits::ScalarPerVector == 0);

        constexpr auto tile_dstr = TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumAccess, 1>{}([&](auto iAccess) {
            // data index [y0, y1, ...]
            constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);
            constexpr index_t d = tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys_start);
            static_assert(d % Traits::ScalarPerVector == 0);


            get_bottom_tensor_view().template get_vectorized_elements_raw<vector_t>(
                dst_tensor.get_thread_buffer().template get_as<vector_t>()(d /
                                                                           Traits::ScalarPerVector),
                pre_computed_offsets_[iAccess]);
        });
    }

    CK_TILE_DEVICE void move(const BottomTensorIndex& step)
    {
        window_origin_ += step;

        pre_compute_offsets();
    }

    // this is the bottom tensor view
    // [x0', x1', ...] ==> [offset]
    BottomTensorView bottom_tensor_view_;

    //
    WindowLengths window_lengths_;

    // origin ([x0', x1', ...]) of window on bottom tensor
    BottomTensorIndex window_origin_;

    // Tile tensor distribution, which contains:
    //   1. adaptor for window: [p0, p1, ..., y0, y1, ...] ==> [x0, x1, ...]
    //   2. thread descriptor for thread tensor in register: [y0, y1, ...] ==> [d]
    TileDstr tile_dstr_;

    array<index_t, NumAccess> pre_computed_offsets_;
};

// TODO: use strategy
template <typename TensorView_, typename WindowLengths_, typename StaticTileDistribution_>
CK_TILE_DEVICE constexpr auto
make_tile_window_lds(const TensorView_& tensor_view,
                     const WindowLengths_& window_lengths,
                     const multi_index<TensorView_::get_num_of_dimension()>& origin,
                     const StaticTileDistribution_& tile_distribution)
{
    return tile_window_lds<remove_cvref_t<TensorView_>,
                           remove_cvref_t<WindowLengths_>,
                           remove_cvref_t<StaticTileDistribution_>>{
        tensor_view, window_lengths, origin, tile_distribution};
}

template <typename TensorView_, typename WindowLengths_, typename StaticTileDistribution_>
CK_TILE_DEVICE void
move_tile_window(tile_window_lds<TensorView_, WindowLengths_, StaticTileDistribution_>& window,
                 const typename tile_window_lds<TensorView_,
                                                WindowLengths_,
                                                StaticTileDistribution_>::BottomTensorIndex& step)
{
    window.move(step);
}

} // namespace ck_tile
