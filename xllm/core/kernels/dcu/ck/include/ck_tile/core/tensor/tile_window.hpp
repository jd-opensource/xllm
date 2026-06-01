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
template <typename BottomTensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t NumCoord,
          bool UseHcuTensorView = false>
struct tile_window_with_static_distribution
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
    static_assert(NumCoord == 1);

    // TODO: check WindowLengths and StaticTileDistribution are consistent

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
                tile_window_with_static_distribution::
                    get_window_adaptor_ys_safe_vector_length_strides();

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
        static_assert(NumAccess % NumCoord == 0, "wrong! # of access is not divisible by NumCoord");
    };

    static constexpr index_t NumAccessPerCoord = load_store_traits::NumAccess / NumCoord;

    CK_TILE_DEVICE constexpr tile_window_with_static_distribution() = default;

    CK_TILE_DEVICE constexpr tile_window_with_static_distribution(
        const BottomTensorView& bottom_tensor_view,
        const WindowLengths& window_lengths,
        const BottomTensorIndex& window_origin,
        const TileDstr& tile_distribution)
        : bottom_tensor_view_{bottom_tensor_view},
          window_lengths_{window_lengths},
          window_origin_{window_origin},
          tile_dstr_{tile_distribution},
          pre_computed_coords_{}
    {
#if 0 // debug
      // TODO: this use more register for FA, but less register for GEMM
      // need investigation
      // only support warp-tile and block-tile
        static_assert(NDimP == 1 or NDimP == 2, "wrong!");

        WindowAdaptorCoord window_adaptor_thread_coord_tmp;

        if constexpr(NDimP == 1)
        {
            window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
                tile_distribution.get_ps_ys_to_xs_adaptor(), AdaptorTopIndex{get_lane_id(), 0});
        }
        else if constexpr(NDimP == 2)
        {
        //TODO: support get_sub_warp_id for WASP
            window_adaptor_thread_coord_tmp =
                make_tensor_adaptor_coordinate(tile_distribution.get_ps_ys_to_xs_adaptor(),
                                               AdaptorTopIndex{get_warp_id(), get_lane_id(), 0});
        }
#else
        // TODO: this use less register for FA, but more register for GEMM
        // need investigation
        // TODO: support wasp partition_index
        const auto window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
            tile_distribution.get_ps_ys_to_xs_adaptor(),
            container_concat(detail::get_partition_index(tile_distribution),
                             array<index_t, NDimY>{0}));
#endif

        BottomTensorIndex bottom_tensor_thread_origin_idx_tmp =
            window_origin + window_adaptor_thread_coord_tmp.get_bottom_index();

        const auto bottom_tensor_thread_coord_tmp = make_tensor_coordinate(
            bottom_tensor_view_.get_tensor_descriptor(), bottom_tensor_thread_origin_idx_tmp);

        // pre-compute NumCoord (WindowAdaptorCoord, BottomTensorCoord) bundles to speed up
        // future load/store() calls (might allocate more registers)
        using Traits = load_store_traits;
        using SFC_Ys = typename Traits::SFC_Ys;

        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = window_adaptor_thread_coord_tmp;
            auto bottom_tensor_thread_coord  = bottom_tensor_thread_coord_tmp;

            constexpr auto idx_diff_ys =
                SFC_Ys::get_step_between(number<0>{}, number<iCoord * NumAccessPerCoord>{});

            constexpr auto idx_diff_ps_ys = container_concat(
                generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}), idx_diff_ys);

            move_window_adaptor_and_bottom_tensor_thread_coordinate(
                window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);

            pre_computed_coords_(iCoord) =
                make_tuple(window_adaptor_thread_coord, bottom_tensor_thread_coord);
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

    CK_TILE_DEVICE auto get_buffer_ptr() const
    {
        return get_bottom_tensor_view().get_buffer_view().p_data_;
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

    CK_TILE_DEVICE static constexpr auto get_num_of_access()
    {
        return load_store_traits::NumAccess;
    }

    template <index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true,
              bool swizzle                = false>
    CK_TILE_DEVICE auto load(number<i_access_unsupport_>          = {},
                             bool_constant<oob_conditional_check> = {},
                             bool_constant<swizzle>               = {}) const
    {
        constexpr auto tile_dstr = TileDstr{};
        auto dst_tensor          = make_static_distributed_tensor<DataType>(tile_dstr);
        load(dst_tensor, bool_constant<oob_conditional_check>{}, bool_constant<swizzle>{});
        return dst_tensor;
    }

    template <typename DistributedTensor, bool oob_conditional_check = true, bool swizzle = false>
    CK_TILE_DEVICE auto load(DistributedTensor& dst_tensor,
                             bool_constant<oob_conditional_check> = {},
                             bool_constant<swizzle>               = {}) const
    {
        using Traits   = load_store_traits;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);

                // read from bottom tensor
                const vector_t vec_value =
                    get_bottom_tensor_view().template get_vectorized_elements<vector_t>(
                        bottom_tensor_thread_coord,
                        0,
                        bool_constant<oob_conditional_check>{},
                        bool_constant<swizzle>{});
#if 0
                // write into distributed tensor
                static_for<0, Traits::ScalarPerVector, 1>{}([&](auto j) {
                    constexpr auto idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<NDimY>{});

                    constexpr index_t d =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys);

                    dst_tensor.get_thread_buffer().template at<d>() =
                        vec_value.template get_as<DataType>()[j];
                });
#else
                constexpr index_t d =
                    tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys_start);
                static_assert(d % Traits::ScalarPerVector == 0);

                dst_tensor.get_thread_buffer().template get_as<vector_t>()(
                    number<d / Traits::ScalarPerVector>{}) = bit_cast<vector_t>(vec_value);
#endif
                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        idx_diff_ys);

                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    template <index_t i_access_unsupport_ = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE auto load_by_inline_asm(number<i_access_unsupport_>          = {},
                                           bool_constant<oob_conditional_check> = {}) const
    {
        constexpr auto tile_dstr = TileDstr{};
        auto dst_tensor          = make_static_distributed_tensor<DataType>(tile_dstr);
        load_by_inline_asm(dst_tensor, bool_constant<oob_conditional_check>{});
        return dst_tensor;
    }

    template <typename DistributedTensor, bool oob_conditional_check = true>
    CK_TILE_DEVICE auto load_by_inline_asm(DistributedTensor& dst_tensor,
                                           bool_constant<oob_conditional_check> = {}) const
    {
        using Traits   = load_store_traits;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);

                if constexpr(UseHcuTensorView)
                {
                    // read from bottom tensor
                    const vector_t vec_value =
                        get_bottom_tensor_view().template get_vectorized_elements_asm<vector_t>(
                            bottom_tensor_thread_coord, 0, bool_constant<oob_conditional_check>{});

                    // write into distributed tensor
                    static_for<0, Traits::ScalarPerVector, 1>{}([&](auto j) {
                        constexpr auto idx_ys = generate_tuple(
                            [&](auto jj) {
                                return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                                : idx_ys_start[jj];
                            },
                            number<NDimY>{});

                        constexpr index_t d =
                            tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys);

                        dst_tensor.get_thread_buffer().template at<d>() =
                            vec_value.template get_as<DataType>()[j];
                    });
                }
                else
                {
                    // read from bottom tensor
                    const vector_t vec_value =
                        get_bottom_tensor_view()
                            .template get_vectorized_elements_by_inline_asm<vector_t>(
                                bottom_tensor_thread_coord,
                                0,
                                bool_constant<oob_conditional_check>{});
#if 1
                    // write into distributed tensor
                    static_for<0, Traits::ScalarPerVector, 1>{}([&](auto j) {
                        constexpr auto idx_ys = generate_tuple(
                            [&](auto jj) {
                                return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                                : idx_ys_start[jj];
                            },
                            number<NDimY>{});

                        constexpr index_t d =
                            tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys);

                        dst_tensor.get_thread_buffer().template at<d>() =
                            vec_value.template get_as<DataType>()[j];
                    });
#else
                    constexpr index_t d =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys_start);
                    static_assert(d % Traits::ScalarPerVector == 0);

                    dst_tensor.get_thread_buffer().template get_as<vector_t>()(
                        number<d / Traits::ScalarPerVector>{}) = bit_cast<vector_t>(vec_value);
#endif
                }

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        idx_diff_ys);

                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    template <typename DsreadmInstr, index_t i_access_unsupport_ = -1>
    CK_TILE_DEVICE auto load_by_dsreadm(const DsreadmInstr& instr) const
    {
        constexpr auto tile_dstr = TileDstr{};
        auto dst_tensor          = make_static_distributed_tensor<DataType>(tile_dstr);
        load_by_dsreadm(instr, dst_tensor);
        return dst_tensor;
    }

    template <typename DistributedTensor, typename DsreadmInstr>
    CK_TILE_DEVICE auto load_by_dsreadm(const DsreadmInstr& instr,
                                        DistributedTensor& dst_tensor) const
    {
        using Traits   = load_store_traits;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);

                // read from bottom tensor
                const vector_t vec_value =
                    get_bottom_tensor_view().template get_vectorized_elements_by_dsreadm<vector_t>(
                        instr, bottom_tensor_thread_coord, 0);
#if 1
                // write into distributed tensor
                static_for<0, Traits::ScalarPerVector, 1>{}([&](auto j) {
                    constexpr auto idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<NDimY>{});

                    constexpr index_t d =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys);

                    dst_tensor.get_thread_buffer().template at<d>() =
                        vec_value.template get_as<DataType>()[j];
                });
#else
                constexpr index_t d =
                    tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys_start);
                static_assert(d % Traits::ScalarPerVector == 0);

                dst_tensor.get_thread_buffer().template get_as<vector_t>()(
                    number<d / Traits::ScalarPerVector>{}) = bit_cast<vector_t>(vec_value);
#endif
                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        idx_diff_ys);

                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    template <index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true,
              bool pre_nop                = false,
              bool swizzle                = false>
    CK_TILE_DEVICE auto load_raw(number<i_access_unsupport_>          = {},
                                 bool_constant<oob_conditional_check> = {},
                                 bool_constant<pre_nop>               = {},
                                 bool_constant<swizzle>               = {}) const
    {
        constexpr auto tile_dstr = TileDstr{};
        auto dst_tensor          = make_static_distributed_tensor<DataType>(tile_dstr);
        load_raw(dst_tensor,
                 number<i_access_unsupport_>{},
                 bool_constant<oob_conditional_check>{},
                 bool_constant<pre_nop>{},
                 bool_constant<swizzle>{});
        return dst_tensor;
    }

    template <typename DstTile,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true,
              bool pre_nop                = false,
              bool swizzle                = false>
    CK_TILE_DEVICE void load_raw(DstTile& dst_tensor,
                                 number<i_access_unsupport_>          = {},
                                 bool_constant<oob_conditional_check> = {},
                                 bool_constant<pre_nop>               = {},
                                 bool_constant<swizzle>               = {}) const
    {
        using Traits = load_store_traits;

        // using vector_type_t = typename Traits::vector_type_t;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;
        static constexpr index_t YElementSize =
            TileDstr{}.get_ys_to_d_descriptor().get_element_space_size();
        static_assert(YElementSize % Traits::ScalarPerVector == 0);
        using vectorized_tbuf = array<vector_t, YElementSize / Traits::ScalarPerVector>;
        // StaticBuffer<address_space_enum::vgpr,
        //                                      vector_t,
        //                                      YElementSize / Traits::ScalarPerVector,
        //                                      true>;

        constexpr auto tile_dstr = TileDstr{};

        auto& dst_vec_tbuf = reinterpret_cast<vectorized_tbuf&>(dst_tensor.get_thread_buffer());

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess  = number<iCoord * NumAccessPerCoord + iCoordAccess>{};
                constexpr auto pre_nop_ = [&]() {
                    if constexpr(pre_nop && iCoord == 0 && iCoordAccess == 0)
                        return bool_constant<true>{};
                    else
                        return bool_constant<false>{};
                }();

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);
                constexpr index_t d =
                    tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys_start);
                static_assert(d % Traits::ScalarPerVector == 0);

                get_bottom_tensor_view().template get_vectorized_elements_raw<vector_t>(
                    dst_vec_tbuf.template at<d / Traits::ScalarPerVector>(),
                    bottom_tensor_thread_coord,
                    0 /**/,
                    bool_constant<oob_conditional_check>{},
                    pre_nop_,
                    bool_constant<swizzle>{});
#if CK_TILE_WORKAROUND_ROCM_6_1_SCRATCH_MEMORY_ISSUE || \
    CK_TILE_WORKAROUND_ROCM_6_2_SCRATCH_MEMORY_ISSUE
                asm volatile(
                    ""); // this is starting from rocm-6.2, but same sympton, reuse this flag
#endif
                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        idx_diff_ys);

                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    // TODO: currently async load only implemented in inline asm
    template <typename LdsTileWindow_,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true,
              bool pre_nop                = false>
    CK_TILE_DEVICE auto async_load_raw(LdsTileWindow_&& lds_tile,
                                       number<i_access_unsupport_>          = {},
                                       bool_constant<oob_conditional_check> = {},
                                       bool_constant<pre_nop>               = {}) const
    {
        using LdsTileWindow = remove_cvref_t<LdsTileWindow_>;
        // using LdsTensorView = typename LdsTileWindow::BottomTensorView;
        using LdsDataType = typename LdsTileWindow::DataType;
        // using LdsDescriptor = typename LdsTileWindow::BottomTensorDesc;

        // issues * warps * lanes
        static_assert(LdsTileWindow::get_num_of_dimension() == 3); // TODO: hard coded

        const index_t size_per_buf =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<0>{}, number<0>{}, number<0>{})) *
            sizeof(LdsDataType);

        const index_t size_per_wave =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<0>{}, number<1>{}, number<0>{})) *
                sizeof(LdsDataType) -
            size_per_buf;

        const index_t size_per_issue =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<1>{}, number<0>{}, number<0>{})) *
                sizeof(LdsDataType) -
            size_per_buf;

        // TODO: support get_sub_warp_id for WASP
        const index_t m0_init_value = size_per_buf + size_per_wave * get_warp_id();
        m0_set_with_memory(m0_init_value); // This should be wave independent

        using Traits = load_store_traits;

        // using vector_type_t = typename Traits::vector_type_t;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        LdsDataType* smem = lds_tile.get_bottom_tensor_view().get_buffer_view().p_data_;

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess  = number<iCoord * NumAccessPerCoord + iCoordAccess>{};
                constexpr auto pre_nop_ = [&]() {
                    if constexpr(pre_nop && iCoord == 0 && iCoordAccess == 0)
                        return bool_constant<true>{};
                    else
                        return bool_constant<false>{};
                }();

                // read from bottom tensor
                get_bottom_tensor_view().template async_get_vectorized_elements_raw<vector_t>(
                    smem, bottom_tensor_thread_coord, 0, pre_nop_);

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        idx_diff_ys);

                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);

                    m0_inc_with_memory(size_per_issue);
                }
            });
        });
    }

    template <typename LdsTileWindow_,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true>
    CK_TILE_DEVICE auto async_load(LdsTileWindow_&& lds_tile,
                                   number<i_access_unsupport_>          = {},
                                   bool_constant<oob_conditional_check> = {}) const
    {
        using LdsTileWindow = remove_cvref_t<LdsTileWindow_>;
        using LdsDataType   = typename LdsTileWindow::DataType;

        // issues * warps * lanes
        static_assert(LdsTileWindow::get_num_of_dimension() == 3); // TODO: hard coded

        // TODO: LDS offset is not good for intrinsic based implementation(compiler can't figure out
        // dependency) hence avoid use offset based solution. size_per_buf should be zero (how to
        // check?)
        constexpr index_t size_per_buf =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<0>{}, number<0>{}, number<0>{}));

        constexpr index_t size_per_wave =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<0>{}, number<1>{}, number<0>{})) -
            size_per_buf;

        constexpr index_t size_per_issue =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<1>{}, number<0>{}, number<0>{})) -
            size_per_buf;

        // TODO: support get_sub_warp_id for WASP
        const index_t m0_init_value = size_per_buf + size_per_wave * get_warp_id();

        using Traits = load_store_traits;

        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        // TODO: we force CK_TILE_LDS_ADDR
        CK_TILE_LDS_ADDR LdsDataType* smem =
            lds_tile.get_bottom_tensor_view().get_buffer_view().p_data_ + m0_init_value;

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // read from bottom tensor
                get_bottom_tensor_view().template async_get_vectorized_elements<vector_t>(
                    smem, bottom_tensor_thread_coord, 0, bool_constant<oob_conditional_check>{});

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        idx_diff_ys);

                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);

                    smem += size_per_issue; // Note we manually increase the per-issue offset
                }
            });
        });
    }

    template <typename LdsTileWindow_,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true>
    CK_TILE_DEVICE auto async_load_by_inline_asm(LdsTileWindow_&& lds_tile,
                                                 number<i_access_unsupport_>          = {},
                                                 bool_constant<oob_conditional_check> = {}) const
    {
        using LdsTileWindow = remove_cvref_t<LdsTileWindow_>;
        using LdsDataType   = typename LdsTileWindow::DataType;

        // issues * warps * lanes
        static_assert(LdsTileWindow::get_num_of_dimension() == 3); // TODO: hard coded

        // FIXME:compile error with constexpr
#if 0
        // TODO: LDS offset is not good for intrinsic based implementation(compiler can't figure out
        // dependency) hence avoid use offset based solution. size_per_buf should be zero (how to
        // check?)
        constexpr index_t size_per_buf =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<0>{}, number<0>{}, number<0>{}));

        constexpr index_t size_per_wave =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<0>{}, number<1>{}, number<0>{})) -
            size_per_buf;

        constexpr index_t size_per_issue =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<1>{}, number<0>{}, number<0>{})) -
            size_per_buf;
#else
        // TODO: LDS offset is not good for intrinsic based implementation(compiler can't figure out
        // dependency) hence avoid use offset based solution. size_per_buf should be zero (how to
        // check?)
        const index_t size_per_buf =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<0>{}, number<0>{}, number<0>{}));

        const index_t size_per_wave =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<0>{}, number<1>{}, number<0>{})) -
            size_per_buf;

        const index_t size_per_issue =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<1>{}, number<0>{}, number<0>{})) -
            size_per_buf;
#endif

        // TODO: support get_sub_warp_id for WASP
        const index_t m0_init_value = size_per_buf + size_per_wave * get_warp_id();

        using Traits = load_store_traits;

        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        // TODO: we force CK_TILE_LDS_ADDR
        CK_TILE_LDS_ADDR LdsDataType* smem =
            lds_tile.get_bottom_tensor_view().get_buffer_view().p_data_ + m0_init_value;

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // read from bottom tensor
                get_bottom_tensor_view()
                    .template async_get_vectorized_elements_by_inline_asm<vector_t>(
                        smem,
                        bottom_tensor_thread_coord,
                        0,
                        bool_constant<oob_conditional_check>{});

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        idx_diff_ys);

                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);

                    smem += size_per_issue; // Note we manually increase the per-issue offset
                }
            });
        });
    }

    template <index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true,
              bool swizzle                = false>
    CK_TILE_DEVICE void store(const static_distributed_tensor<DataType, TileDstr>& dstr_tensor,
                              number<i_access_unsupport_>          = {},
                              bool_constant<oob_conditional_check> = {},
                              bool_constant<swizzle>               = {}) const
    {
        using Traits = load_store_traits;

        // using vector_type_t = typename Traits::vector_type_t;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);

                // read from distributed tensor
                // vector_type_t vec;
                vector_t vec_value;

                static_for<0, Traits::ScalarPerVector, 1>{}([&](auto j) {
                    constexpr auto idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<NDimY>{});

                    constexpr index_t d =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys);

                    vec_value.template get_as<DataType>()(j) =
                        dstr_tensor.get_thread_buffer().template at<d>();
                });

                // const vector_t vec_value = vec.template
                // get_as<vector_t>().template at<0>();

                // write into bottom tensor
                get_bottom_tensor_view().template set_vectorized_elements<vector_t>(
                    bottom_tensor_thread_coord,
                    0,
                    vec_value,
                    bool_constant<oob_conditional_check>{},
                    bool_constant<swizzle>{});

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        idx_diff_ys);

                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    template <index_t i_access_unsupport_ = -1>
    CK_TILE_DEVICE void store_raw(const static_distributed_tensor<DataType, TileDstr>& dstr_tensor,
                                  number<i_access_unsupport_> = {}) const
    {
        using Traits = load_store_traits;

        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr                    = TileDstr{};
        static constexpr bool oob_conditional_check = true;

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);

                // read from distributed tensor
                vector_t vec_value;
                static_for<0, Traits::ScalarPerVector, 1>{}([&](auto j) {
                    constexpr auto idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<NDimY>{});
                    constexpr index_t d =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys);
                    vec_value.template get_as<DataType>()(j) =
                        dstr_tensor.get_thread_buffer().template at<d>();
                });

                // write into bottom tensor
                get_bottom_tensor_view()
                    .template set_vectorized_elements_raw<vector_t, oob_conditional_check>(
                        bottom_tensor_thread_coord, 0, vec_value);

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        idx_diff_ys);

                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    template <index_t i_access_unsupport_ = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE void update(const static_distributed_tensor<DataType, TileDstr>& dstr_tensor,
                               number<i_access_unsupport_>          = {},
                               bool_constant<oob_conditional_check> = {}) const
    {
        using Traits = load_store_traits;

        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);

                // read from distributed tensor
                vector_t vec_value;

                static_for<0, Traits::ScalarPerVector, 1>{}([&](auto j) {
                    constexpr auto idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<NDimY>{});

                    constexpr index_t d =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys);

                    vec_value.template get_as<DataType>()(j) =
                        dstr_tensor.get_thread_buffer().template at<d>();
                });

                // write into bottom tensor
                get_bottom_tensor_view().template update_vectorized_elements<vector_t>(
                    bottom_tensor_thread_coord,
                    0,
                    vec_value,
                    bool_constant<oob_conditional_check>{});

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        idx_diff_ys);

                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    // move thread's botom tensor coordiante
    // [x0', x1', ... ] ==> [offset]
    // also move window-origin
    CK_TILE_DEVICE void move(const BottomTensorIndex& step)
    {
        window_origin_ += step;

        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            move_tensor_coordinate(bottom_tensor_view_.get_tensor_descriptor(),
                                   pre_computed_coords_(iCoord)(I1),
                                   step);
        });
    }

    CK_TILE_DEVICE void set_window_origin(const BottomTensorIndex& new_window_origin)
    {
        window_origin_ = new_window_origin;

#if 0 // debug
      // TODO: this use more register for FA, but less register for GEMM
      // need investigation
      // only support warp-tile and block-tile
        static_assert(NDimP == 1 or NDimP == 2, "wrong!");

        WindowAdaptorCoord window_adaptor_thread_coord_tmp;

        if constexpr(NDimP == 1)
        {
            window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
                tile_dstr_.get_ps_ys_to_xs_adaptor(), AdaptorTopIndex{get_lane_id(), 0});
        }
        else if constexpr(NDimP == 2)
        {
            //TODO: support get_sub_warp_id for WASP
            window_adaptor_thread_coord_tmp =
                make_tensor_adaptor_coordinate(tile_dstr_.get_ps_ys_to_xs_adaptor(),
                                               AdaptorTopIndex{get_warp_id(), get_lane_id(), 0});
        }
#else
        // TODO: this use less register for FA, but more register for GEMM
        // need investigation
        const auto window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
            tile_dstr_.get_ps_ys_to_xs_adaptor(),
            container_concat(detail::get_partition_index(tile_dstr_), array<index_t, NDimY>{0}));
#endif

        BottomTensorIndex bottom_tensor_thread_origin_idx_tmp =
            window_origin_ + window_adaptor_thread_coord_tmp.get_bottom_index();

        const auto bottom_tensor_thread_coord_tmp = make_tensor_coordinate(
            bottom_tensor_view_.get_tensor_descriptor(), bottom_tensor_thread_origin_idx_tmp);

        // pre-compute NumCoord (WindowAdaptorCoord, BottomTensorCoord) bundles to speed up
        // future load/store() calls (might allocate more registers)
        using Traits = load_store_traits;
        using SFC_Ys = typename Traits::SFC_Ys;

        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = window_adaptor_thread_coord_tmp;
            auto bottom_tensor_thread_coord  = bottom_tensor_thread_coord_tmp;

            constexpr auto idx_diff_ys =
                SFC_Ys::get_step_between(number<0>{}, number<iCoord * NumAccessPerCoord>{});

            constexpr auto idx_diff_ps_ys = container_concat(
                generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}), idx_diff_ys);

            move_window_adaptor_and_bottom_tensor_thread_coordinate(
                window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);

            pre_computed_coords_(iCoord) =
                make_tuple(window_adaptor_thread_coord, bottom_tensor_thread_coord);
        });
    }

    CK_TILE_HOST_DEVICE void init_raw() { bottom_tensor_view_.init_raw(); }

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

    // this contains:
    //   per-thread coordinate for window adaptor
    //   per-thread coordinate for bottom tensor
    array<tuple<WindowAdaptorCoord, BottomTensorCoord>, NumCoord> pre_computed_coords_;
};

// TODO: use strategy
template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t NumCoord      = 1,
          bool UseHcuTensorView = false>
CK_TILE_DEVICE constexpr auto
make_tile_window(const TensorView_& tensor_view,
                 const WindowLengths_& window_lengths,
                 const multi_index<TensorView_::get_num_of_dimension()>& origin,
                 const StaticTileDistribution_& tile_distribution,
                 number<NumCoord>                = {},
                 bool_constant<UseHcuTensorView> = {})
{
    return tile_window_with_static_distribution<remove_cvref_t<TensorView_>,
                                                remove_cvref_t<WindowLengths_>,
                                                remove_cvref_t<StaticTileDistribution_>,
                                                NumCoord,
                                                UseHcuTensorView>{
        tensor_view, window_lengths, origin, tile_distribution};
}

// this version can't be called in a constexpr context
template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t NumCoord = 1>
CK_TILE_DEVICE auto
make_tile_window_raw(const TensorView_& tensor_view,
                     const WindowLengths_& window_lengths,
                     const multi_index<TensorView_::get_num_of_dimension()>& origin,
                     const StaticTileDistribution_& tile_distribution,
                     number<NumCoord> = {})
{
    auto w = tile_window_with_static_distribution<remove_cvref_t<TensorView_>,
                                                  remove_cvref_t<WindowLengths_>,
                                                  remove_cvref_t<StaticTileDistribution_>,
                                                  NumCoord>{
        tensor_view, window_lengths, origin, tile_distribution};
    w.init_raw();
    return w;
}

template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t NumCoord,
          bool UseHcuTensorView>
CK_TILE_DEVICE void move_tile_window(
    tile_window_with_static_distribution<TensorView_,
                                         WindowLengths_,
                                         StaticTileDistribution_,
                                         NumCoord,
                                         UseHcuTensorView>& window,
    const typename tile_window_with_static_distribution<TensorView_,
                                                        WindowLengths_,
                                                        StaticTileDistribution_,
                                                        NumCoord,
                                                        UseHcuTensorView>::BottomTensorIndex& step)
{
    window.move(step);
}

template <typename BottomTensorView_, typename WindowLengths_>
struct tile_window_with_static_lengths
{
    using BottomTensorView = remove_reference_t<BottomTensorView_>;
    using WindowLengths    = remove_cvref_t<WindowLengths_>;
    using BottomTensorDesc = typename BottomTensorView::TensorDesc;
    using DataType         = typename BottomTensorView::DataType;

    static constexpr index_t NDimBottomTensor = BottomTensorDesc::get_num_of_dimension();

    static_assert(ck_tile::is_known_at_compile_time<WindowLengths>::value,
                  "wrong! lengths should be static");

    using BottomTensorIndex = array<index_t, NDimBottomTensor>;

    CK_TILE_DEVICE constexpr tile_window_with_static_lengths() = default;

    CK_TILE_DEVICE constexpr tile_window_with_static_lengths(
        const BottomTensorView& bottom_tensor_view,
        const WindowLengths& window_lengths,
        const BottomTensorIndex& window_origin)
        : bottom_tensor_view_{bottom_tensor_view},
          window_lengths_{window_lengths},
          window_origin_{window_origin}
    {
    }

    CK_TILE_DEVICE static constexpr index_t get_num_of_dimension() { return NDimBottomTensor; }

    CK_TILE_DEVICE constexpr auto get_window_lengths() const { return window_lengths_; }

    CK_TILE_DEVICE constexpr auto get_bottom_tensor_view() const { return bottom_tensor_view_; }

    CK_TILE_DEVICE constexpr auto get_window_origin() const { return window_origin_; }

    CK_TILE_DEVICE void set_window_origin(const BottomTensorIndex& new_window_origin)
    {
        window_origin_ = new_window_origin;
    }

    CK_TILE_DEVICE constexpr void
    set_bottom_tensor_view_data_ptr(typename BottomTensorView::DataType* data)
    {
        bottom_tensor_view_.buf_.p_data_ = data;
    }

    CK_TILE_DEVICE auto get_buffer_ptr() const
    {
        return get_bottom_tensor_view().get_buffer_view().p_data_;
    }

    // move window-origin
    CK_TILE_DEVICE void move(const BottomTensorIndex& step) { window_origin_ += step; }

    // this is the bottom tensor view
    // [x0', x1', ...] ==> [offset]
    BottomTensorView bottom_tensor_view_;

    //
    WindowLengths window_lengths_;

    // origin ([x0', x1', ...]) of window on bottom tensor
    BottomTensorIndex window_origin_;
};

template <typename TensorView_, typename WindowLengths_>
CK_TILE_DEVICE constexpr auto
make_tile_window(const TensorView_& tensor_view,
                 const WindowLengths_& window_lengths,
                 const multi_index<TensorView_::get_num_of_dimension()>& origin)
{
    static_assert(ck_tile::is_known_at_compile_time<WindowLengths_>::value,
                  "wrong! lengths should be static");

    return tile_window_with_static_lengths<remove_cvref_t<TensorView_>,
                                           remove_cvref_t<WindowLengths_>>{
        tensor_view, window_lengths, origin};
}

// duplicate tile window and replace its origin
template <typename TensorView, typename WindowLengths>
CK_TILE_DEVICE constexpr auto
make_tile_window(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                 const multi_index<TensorView::get_num_of_dimension()>& origin)
{
    return tile_window_with_static_lengths<TensorView, WindowLengths>{
        tile_window.get_bottom_tensor_view(), tile_window.get_window_lengths(), origin};
}

template <typename TensorView, typename WindowLengths, typename StaticTileDistribution>
CK_TILE_DEVICE constexpr auto
make_tile_window(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                 const multi_index<TensorView::get_num_of_dimension()>& origin,
                 const StaticTileDistribution& tile_distribution)
{
    return make_tile_window(tile_window.get_bottom_tensor_view(),
                            tile_window.get_window_lengths(),
                            origin,
                            tile_distribution);
}

template <typename TensorView, typename WindowLengths, typename StaticTileDistribution>
CK_TILE_DEVICE constexpr auto
make_tile_window(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                 const StaticTileDistribution& tile_distribution)
{
    return make_tile_window(tile_window.get_bottom_tensor_view(),
                            tile_window.get_window_lengths(),
                            tile_window.get_window_origin(),
                            tile_distribution);
}

template <typename TensorView, typename WindowLengths, typename StaticTileDistribution>
CK_TILE_DEVICE constexpr auto
make_tile_window_raw(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                     const StaticTileDistribution& tile_distribution)
{
    auto w = make_tile_window(tile_window.get_bottom_tensor_view(),
                              tile_window.get_window_lengths(),
                              tile_window.get_window_origin(),
                              tile_distribution);
    w.init_raw();
    return w;
}

template <typename TensorView_, typename WindowLengths_>
CK_TILE_DEVICE void move_tile_window(
    tile_window_with_static_lengths<TensorView_, WindowLengths_>& window,
    const typename tile_window_with_static_lengths<TensorView_, WindowLengths_>::BottomTensorIndex&
        step)
{
    window.move(step);
}

} // namespace ck_tile
