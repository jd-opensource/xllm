// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.

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

/**
 * @brief This class provides description of tile windowed view on the device memory.
 *
 * @note This class does not provide any functions to read or modify device memory.
 *
 * @tparam BottomTensorView_    Class describing & holding device tensor memory.
 * @tparam WindowLengths_       Spatial sizes of windowed view on tensor.
 */
template <typename TileWindowType_, typename BottomTensorView_, typename WindowLengths_>
struct tile_window_base
{

    using BottomTensorView = remove_reference_t<BottomTensorView_>;
    using WindowLengths    = remove_cvref_t<WindowLengths_>;
    using BottomTensorDesc = typename BottomTensorView::TensorDesc;
    using DataType         = remove_cvref_t<typename BottomTensorView::DataType>;

    static constexpr index_t NDimBottomTensor = BottomTensorDesc::get_num_of_dimension();

    static_assert(ck_tile::is_known_at_compile_time<WindowLengths>::value,
                  "wrong! lengths should be static");

    using BottomTensorIndex = array<index_t, NDimBottomTensor>;

    CK_TILE_DEVICE constexpr auto get_window_origin() const { return window_origin_; }
    CK_TILE_DEVICE constexpr auto get_window_lengths() const { return window_lengths_; }
    CK_TILE_DEVICE constexpr auto get_bottom_tensor_view() const { return bottom_tensor_view_; }
    CK_TILE_DEVICE static constexpr index_t get_num_of_dimension() { return NDimBottomTensor; }

    CK_TILE_DEVICE void set_window_origin(const BottomTensorIndex& new_window_origin)
    {
        window_origin_ = new_window_origin;

        // Delegate to child if it implements extra logic
        static_cast<TileWindowType_*>(this)->set_window_origin_extended(new_window_origin);
    }
    // Default no-op; can be overridden in child
    CK_TILE_DEVICE void set_window_origin_extended(const BottomTensorIndex&) {}

    CK_TILE_DEVICE constexpr void
    set_bottom_tensor_view_data_ptr(typename BottomTensorView::DataType* data)
    {
        bottom_tensor_view_.buf_.p_data_ = data;
    }

    // move window-origin
    CK_TILE_DEVICE void move(const BottomTensorIndex& step)
    {
        window_origin_ += step;

        // Delegate to child if it implements extra movement logic
        static_cast<TileWindowType_*>(this)->move_extended(step);
    }

    // Default no-op; can be overridden in child
    CK_TILE_DEVICE void move_extended(const BottomTensorIndex&) {}

    // origin ([x0', x1', ...]) of window on bottom tensor
    BottomTensorIndex window_origin_;

    WindowLengths window_lengths_;

    // this is the bottom tensor view
    // [x0', x1', ...] ==> [offset]
    BottomTensorView bottom_tensor_view_;
};

template <typename TileWindowType_,
          typename BottomTensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_>
struct tile_window_with_tile_dstr_base
    : public tile_window_base<TileWindowType_, BottomTensorView_, WindowLengths_>
{
    using TileDstr       = remove_cvref_t<StaticTileDistribution_>;
    using TileWindowBase = tile_window_base<TileWindowType_, BottomTensorView_, WindowLengths_>;

    using WindowAdaptor = typename TileDstr::PsYs2XsAdaptor;

    static constexpr index_t NDimWindowAdaptorTop = WindowAdaptor::get_num_of_top_dimension();

    static constexpr index_t NDimP = TileDstr::get_num_of_dimension_p();
    static constexpr index_t NDimY = TileDstr::get_num_of_dimension_y();

    using AdaptorTopIndex = array<index_t, NDimWindowAdaptorTop>;
    // using BottomTensorIndex = array<index_t, TileWindowBase::NDimBottomTensor>;

    using WindowAdaptorCoord =
        decltype(make_tensor_adaptor_coordinate(WindowAdaptor{}, AdaptorTopIndex{}));

    using BottomTensorCoord = decltype(make_tensor_coordinate(
        typename TileWindowBase::BottomTensorDesc{}, typename TileWindowBase::BottomTensorIndex{}));

    static_assert(TileDstr::is_static(), "wrong!");
    static_assert(TileWindowBase::NDimBottomTensor == WindowAdaptor::get_num_of_bottom_dimension(),
                  "wrong! inconsistent # of diemsnions");

    CK_TILE_DEVICE constexpr auto get_tile_distribution() const { return tile_dstr_; }
    CK_TILE_HOST_DEVICE void init_raw() { this->bottom_tensor_view_.init_raw(); }

    CK_TILE_DEVICE static constexpr bool has_static_tile_distribution()
    {
        return TileDstr::is_static();
    }

    // move thread's window adaptor coordinate and bottom tensor coordinate
    // [p0, p1, ..., y0, y1, ...] ==> [x0, x1, ...] ==> [x0', x1', ...] ==> [offset]
    template <typename ATopIndex>
    CK_TILE_DEVICE void move_window_adaptor_and_bottom_tensor_thread_coordinate(
        WindowAdaptorCoord& window_adaptor_thread_coord,
        BottomTensorCoord& bottom_tensor_thread_coord,
        const ATopIndex& idx_diff_adaptor_top) const
    {
        array<index_t, TileWindowBase::NDimBottomTensor> idx_diff_adaptor_bottom;

        move_tensor_adaptor_coordinate(tile_dstr_.get_ps_ys_to_xs_adaptor(),
                                       window_adaptor_thread_coord,
                                       idx_diff_adaptor_top,
                                       idx_diff_adaptor_bottom);

        move_tensor_coordinate(this->bottom_tensor_view_.get_tensor_descriptor(),
                               bottom_tensor_thread_coord,
                               idx_diff_adaptor_bottom);
    }

    struct Traits
    {
        public:
        static constexpr index_t PackedSize =
            ck_tile::numeric_traits<remove_cvref_t<typename TileWindowBase::DataType>>::PackedSize;

        static constexpr auto get_vector_dim_y_scalar_per_vector()
        {
            const auto [ys_vector_lengths, ys_vector_strides] =
                tile_window_with_tile_dstr_base::get_window_adaptor_ys_safe_vector_length_strides();

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

        static constexpr index_t VectorDimY = get_vector_dim_y_scalar_per_vector().template at<0>();
        static constexpr index_t ScalarPerVector =
            get_vector_dim_y_scalar_per_vector().template at<1>();
        using vector_t =
            thread_buffer<typename TileWindowBase::DataType, ScalarPerVector / PackedSize>;

        static constexpr auto scalars_per_access_ = [] {
            constexpr auto scalars_per_access_arr = generate_array(
                [&](auto i) { return (i == VectorDimY) ? ScalarPerVector : 1; }, number<NDimY>{});

            /// TODO: add non-automatic storage argument support to macro TO_SEQUENCE()
            constexpr auto NDimY_ = NDimY;

            return TO_SEQUENCE(scalars_per_access_arr, NDimY_);
        }();

        static constexpr auto get_space_filling_curve()
        {
            constexpr auto thread_tensor_lengths_ys =
                to_sequence(TileDstr{}.get_ys_to_d_descriptor().get_lengths());

            // FIXME: need logic to judge dim access order
            using DimAccessOrder = typename arithmetic_sequence_gen<0, NDimY, 1>::type;

            return space_filling_curve<decltype(thread_tensor_lengths_ys),
                                       DimAccessOrder,
                                       decltype(scalars_per_access_),
                                       false /*!!! no snaked curve! */>{};
        }

        using SFC_Ys = decltype(get_space_filling_curve());

        static constexpr index_t NumAccess = SFC_Ys::get_num_of_access();

        static_assert(0 < NumAccess, "Wrong! NumAccess should be larger than 0");
    };

    // return vector dimension among [y0, y1, ...]
    CK_TILE_DEVICE static constexpr auto get_window_adaptor_ys_safe_vector_length_strides()
    {
        // bottom tensor top dimension vector lengths and strides
        const auto [bottom_tensor_top_dim_vector_lengths, bottom_tensor_top_dim_vector_strides] =
            TileWindowBase::BottomTensorDesc::get_top_dimension_safe_vector_length_strides();

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

        set_container_subset(window_adaptor_vector_lengths,
                             window_adaptor_bottom_dims,
                             window_adaptor_bottom_dim_vector_lengths);
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

    CK_TILE_DEVICE constexpr auto get_num_of_access() const { return Traits::NumAccess; }
    // Tile tensor distribution, which contains:
    //   1. adaptor for window: [p0, p1, ..., y0, y1, ...] ==> [x0, x1, ...]
    //   2. thread descriptor for thread tensor in register: [y0, y1, ...] ==> [d]
    TileDstr tile_dstr_;
};

} // namespace ck_tile
