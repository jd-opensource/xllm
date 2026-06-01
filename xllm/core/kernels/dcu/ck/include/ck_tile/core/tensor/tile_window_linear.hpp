// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.

#pragma once
#include "ck_tile/core/arch/arch.hpp"
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
#include "ck_tile/core/tensor/tile_window_base.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

#define WINDOW_DISPATCH_ISSUE()                                     \
    if constexpr(i_access < 0)                                      \
    {                                                               \
        static_for<0, NumAccess, 1>{}([&](auto ia) { issue(ia); }); \
    }                                                               \
    else                                                            \
    {                                                               \
        static_assert(i_access < NumAccess);                        \
        issue(number<i_access>{});                                  \
    }

//
// This version of tile window will pre-cache offset/flags based on need
//
// LinearBottomDims_, e.g seq<0, 1> for 2d tensor, the last one is linear dim
// so last dim can use immediate offset to indexing, can save register
// TODO: if using this struct, better use load_raw()/store_raw(), can control
//       the the immediate offset on the fly
// space-filing-curve is non-snaked here!
// This struct inherits from tile_window_with_tile_dstr_base, which is an intermediary base class
// with the ultimate parent class being tile_window_base.
template <typename BottomTensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          typename LinearBottomDims_>
struct tile_window_linear
    : public tile_window_with_tile_dstr_base<tile_window_linear<BottomTensorView_,
                                                                WindowLengths_,
                                                                StaticTileDistribution_,
                                                                LinearBottomDims_>,
                                             BottomTensorView_,
                                             WindowLengths_,
                                             StaticTileDistribution_>
{
    using Base = tile_window_with_tile_dstr_base<tile_window_linear<BottomTensorView_,
                                                                    WindowLengths_,
                                                                    StaticTileDistribution_,
                                                                    LinearBottomDims_>,
                                                 BottomTensorView_,
                                                 WindowLengths_,
                                                 StaticTileDistribution_>;

    using LinearBottomDims = remove_cvref_t<LinearBottomDims_>;

    static_assert(LinearBottomDims::size() == Base::BottomTensorView::get_num_of_dimension());

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};

    struct traits
    {
        private:
        static constexpr auto get_num_non_linear_access()
        {
            constexpr auto sfc_access_lens = Base::Traits::SFC_Ys::access_lengths;
            using ys_to_rhs_major =
                typename decltype(typename Base::TileDstr{}
                                      .get_static_tile_distribution_encoding())::Ys2RHsMajor;

            constexpr auto non_linear = [&]() {
                index_t cnt = 1;
                static_for<0, Base::NDimY, 1>{}([&](auto i_dim_y) {
                    constexpr auto rhs_major    = ys_to_rhs_major{}[i_dim_y];
                    constexpr auto target_h_dim = number<rhs_major - 1>{}; // no r dim here!
                    if constexpr(LinearBottomDims{}[target_h_dim] == 0)
                    {
                        cnt *= sfc_access_lens[i_dim_y];
                    }
                });
                return cnt;
            }();

            return non_linear;
        }

        // example:
        // non_linear_access_map: sequence<0, 0, 0, 0, 1, 1, 1, 1> for 8 access, totally 2 register
        // used
        //  -> histogram : sequence<4, 4>
        //  -> prefixsum : seqneuce<0, 4, 8>
        // non_linear_access_map: sequence<0, 1, 2, 3, 4, 5, 6, 7> for 8 access, totally 8 register
        // used, will pre-cache 8
        //  -> histogram : sequence<1, 1, 1, 1, 1, 1, 1, 1>
        //  -> prefixsum : seqneuce<0, 1, 2, 3, 4, 5, 6, 7, 8>
        // non_linear_access_map: sequence<0, 0, 1, 1, 2, 2, 3, 3> for 8 access, totally 4 register
        // used, will pre-cache 4
        //  -> histogram : sequence<2, 2, 2, 2>
        //  -> prefixsum : seqneuce<0, 2, 4, 6, 8>
        static constexpr auto get_non_linear_access_map()
        {
            constexpr auto sfc_access_lens = Base::Traits::SFC_Ys::access_lengths;
            using ys_to_rhs_major =
                typename decltype(typename Base::TileDstr{}
                                      .get_static_tile_distribution_encoding())::Ys2RHsMajor;
            constexpr auto non_linear_map = [&]() {
                array<index_t, Base::Traits::NumAccess> m_{0};
                index_t cumulative_len_            = 1;
                index_t cumulative_non_linear_len_ = 1;
                static_for<0, Base::NDimY, 1>{}([&](auto i_y) {
                    constexpr auto i_dim_y = number<Base::NDimY - i_y - 1>{}; // from right to left
                    constexpr auto rhs_major     = ys_to_rhs_major{}[i_dim_y];
                    constexpr auto target_h_dim  = number<rhs_major - 1>{}; // no r dim here!
                    constexpr auto is_linear_dim = LinearBottomDims{}[target_h_dim];

                    array<index_t, Base::Traits::NumAccess> current_m_{0};
                    constexpr auto current_len_ = sfc_access_lens[i_dim_y];

                    // copy cumulative length as current pattern
                    for(auto i_ = 0; i_ < cumulative_len_; i_++)
                    {
                        current_m_(i_) = m_[i_];
                    }
                    for(auto j_ = 0; j_ < current_len_; j_++)
                    {
                        auto j_offset_ = is_linear_dim ? 0 : j_ * cumulative_non_linear_len_;
                        for(auto i_ = 0; i_ < cumulative_len_; i_++)
                        {
                            m_(j_ * cumulative_len_ + i_) = current_m_[i_] + j_offset_;
                        }
                    }
                    cumulative_len_ *= current_len_;
                    if(!is_linear_dim)
                        cumulative_non_linear_len_ *= current_len_;
                });
                return m_;
            }();

            return TO_SEQUENCE(non_linear_map, Base::Traits::NumAccess);
        }

        static constexpr auto get_non_linear_access_histogram()
        {
            constexpr auto m_ = get_non_linear_access_map();

            constexpr auto r_ =
                typename arithmetic_sequence_gen<0, get_num_non_linear_access() + 1, 1>::type{};

            constexpr auto h_ = histogram_sorted_sequence(m_, r_);

            return h_;
        }

        static constexpr auto get_non_linear_access_histogram_prefix_sum()
        {
            constexpr auto h_            = get_non_linear_access_histogram();
            constexpr auto h_prefix_sum_ = prefix_sum_sequence(h_);
            return h_prefix_sum_;
        }

        public:
        static constexpr index_t NumAccess_NonLinear = get_num_non_linear_access();
        using AccessMap_NonLinear       = decltype(get_non_linear_access_map()); // sequence
        using AccessHistogram_NonLinear = decltype(get_non_linear_access_histogram());
        using AccessPrefixSum_NonLinear = decltype(get_non_linear_access_histogram_prefix_sum());
    };

    static constexpr index_t NumAccess           = Base::Traits::NumAccess;
    static constexpr index_t NumAccess_NonLinear = traits::NumAccess_NonLinear;
    using AccessMap_NonLinear                    = typename traits::AccessMap_NonLinear;
    using AccessHistogram_NonLinear              = typename traits::AccessHistogram_NonLinear;
    using AccessPrefixSum_NonLinear              = typename traits::AccessPrefixSum_NonLinear;

    CK_TILE_DEVICE constexpr tile_window_linear() = default;

    CK_TILE_DEVICE constexpr tile_window_linear(
        const typename Base::BottomTensorView& bottom_tensor_view,
        const typename Base::WindowLengths& window_lengths,
        const typename Base::BottomTensorIndex& window_origin,
        const typename Base::TileDstr& tile_distribution)
        : cached_coords_{}, cached_window_adaptor_coords_{}, cached_flags_{}
    {
        this->bottom_tensor_view_            = bottom_tensor_view;
        this->window_lengths_                = window_lengths;
        this->window_origin_                 = window_origin;
        this->tile_dstr_                     = tile_distribution;
        auto window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
            tile_distribution.get_ps_ys_to_xs_adaptor(),
            container_concat(
                make_tuple(get_warp_id(), get_lane_id()),
                generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimY>{})));

        typename Base::BottomTensorIndex bottom_tensor_thread_origin_idx_tmp =
            window_origin + window_adaptor_thread_coord_tmp.get_bottom_index();

        auto bottom_tensor_thread_coord_tmp = make_tensor_coordinate(
            this->bottom_tensor_view_.get_tensor_descriptor(), bottom_tensor_thread_origin_idx_tmp);

        // future load/store() calls (might allocate more registers)
        using SFC_Ys = typename Base::Traits::SFC_Ys;

        static_for<0, NumAccess, 1>{}([&](auto i_access) {
            constexpr auto non_linear_id = number<AccessMap_NonLinear{}[i_access]>{};
            constexpr auto need_save_non_linear_coord =
                bool_constant<AccessPrefixSum_NonLinear{}[non_linear_id] == i_access>{};

            if constexpr(need_save_non_linear_coord)
            {
                cached_coords_(non_linear_id)                = bottom_tensor_thread_coord_tmp;
                cached_window_adaptor_coords_(non_linear_id) = window_adaptor_thread_coord_tmp;
            }

            // TODO: need pad_tensor_view to check which dim need use flag to check
            //      cached flag is independent from non-linear-coord
            //      but need be updated in move_tile, with proper dims
            cached_flags_(i_access) = coordinate_has_valid_offset_assuming_top_index_is_valid(
                this->bottom_tensor_view_.get_tensor_descriptor(), bottom_tensor_thread_coord_tmp);

            if constexpr(i_access != (NumAccess - 1))
            {
                constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(i_access); // tuple of number
                constexpr auto idx_diff_ps_ys = container_concat(
                    generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                    idx_diff_ys);

                Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                    window_adaptor_thread_coord_tmp,
                    bottom_tensor_thread_coord_tmp,
                    idx_diff_ps_ys);
            }
        });
    }

    template <index_t i_access>
    CK_TILE_DEVICE static constexpr auto get_bottom_linear_coordinate(number<i_access>)
    {
        using SFC_Ys          = typename Base::Traits::SFC_Ys;
        constexpr auto idx_ys = SFC_Ys::get_index(number<i_access>{});
        using ys_to_rhs_major =
            typename decltype(typename Base::TileDstr{}
                                  .get_static_tile_distribution_encoding())::Ys2RHsMajor;

        constexpr auto modified_idx_ys = generate_tuple(
            [&](auto i_dim_y) {
                constexpr auto rhs_major    = ys_to_rhs_major{}[i_dim_y];
                constexpr auto target_h_dim = number<rhs_major - 1>{}; // no r dim here!
                if constexpr(LinearBottomDims{}[target_h_dim] == 0)
                {
                    return number<0>{};
                }
                else
                {
                    return number<idx_ys[i_dim_y]>{};
                }
            },
            number<Base::NDimY>{});

        constexpr auto adaptor_ = typename Base::TileDstr{}.get_ps_ys_to_xs_adaptor();
        constexpr auto idx_ =
            container_concat(make_tuple(number<0>{}, number<0>{}), modified_idx_ys);

        return adaptor_.calculate_bottom_index(idx_);
    }

    template <index_t i_access>
    CK_TILE_DEVICE static constexpr index_t get_bottom_linear_offset(number<i_access>)
    {
        constexpr auto linear_coord = get_bottom_linear_coordinate(number<i_access>{});
        constexpr auto is_pure_linear_tensor =
            reduce_on_sequence(LinearBottomDims{}, multiplies{}, number<1>{});
        if constexpr(is_pure_linear_tensor)
        {
            // this case usually is a LDS window, everything is known at compile tile.
            // we directly use BottomTensorView transform to compute the offset, in case padding
            auto bottom_tensor_coord = make_tensor_coordinate(
                typename Base::BottomTensorView{}.get_tensor_descriptor(), linear_coord);
            return bottom_tensor_coord.get_offset();
        }
        else
        {
            // this case usually is a global window, where last dim can be linear
            // we hack here, that use the original TileDstr to compute the linear offset
            // ... hoping that there is no extra padding between other dims, which make sense
            // since that would introduce runtime length (so can't use linear offset)
            constexpr index_t linear_offset = [&]() {
                constexpr auto x_idx_ = linear_coord;
                constexpr auto x_len_ = typename Base::TileDstr{}.get_lengths();
                static_assert(x_idx_.size() == x_len_.size());
                constexpr index_t x_dims_ = x_idx_.size();
                index_t cu_stride_        = 1;
                index_t cu_offset_        = 0;
                static_for<0, x_dims_, 1>{}([&](auto i_) {
                    auto r_i_ = number<x_dims_ - i_ - 1>{};
                    cu_offset_ += x_idx_[r_i_] * cu_stride_;
                    cu_stride_ *= x_len_[r_i_];
                });
                return cu_offset_;
            }();
            return linear_offset;
        }
    }

    template <index_t i_access = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE auto load(number<i_access> = {}, bool_constant<oob_conditional_check> = {}) const
    {
        using vector_t = typename Base::Traits::vector_t;
        using SFC_Ys   = typename Base::Traits::SFC_Ys;

        constexpr auto tile_dstr = typename Base::TileDstr{};

        auto dst_tensor = make_static_distributed_tensor<typename Base::DataType>(tile_dstr);

        auto issue = [&](auto i_access_) {
            constexpr auto IAccess = number<i_access_>{};

            constexpr auto non_linear_id    = number<AccessMap_NonLinear{}[IAccess]>{};
            auto bottom_tensor_thread_coord = cached_coords_[non_linear_id];
            auto bottom_tensor_flag         = cached_flags_[IAccess];

            constexpr auto linear_offset = get_bottom_linear_offset(IAccess);

            // read from bottom tensor
            const vector_t vec_value =
                this->get_bottom_tensor_view().template get_vectorized_elements<vector_t>(
                    bottom_tensor_thread_coord,
                    linear_offset,
                    bottom_tensor_flag,
                    bool_constant<oob_conditional_check>{});

            // data index [y0, y1, ...]
            constexpr auto idx_diff_ys = SFC_Ys::get_index(IAccess);
            // write into distributed tensor
            static_for<0, Base::Traits::ScalarPerVector, Base::Traits::PackedSize>{}([&](auto j) {
                constexpr auto idx_ys = generate_tuple(
                    [&](auto jj) {
                        return jj == Base::Traits::VectorDimY ? (idx_diff_ys[jj] + j)
                                                              : idx_diff_ys[jj];
                    },
                    number<Base::NDimY>{});

                constexpr index_t d = tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                                      Base::Traits::PackedSize;

                dst_tensor.get_thread_buffer().template at<d>() =
                    vec_value
                        .template get_as<typename Base::DataType>()[j / Base::Traits::PackedSize];
            });
        };

        WINDOW_DISPATCH_ISSUE();

        return dst_tensor;
    }

    template <typename DstTile, index_t i_access = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE auto load(DstTile& dst_tensor,
                             number<i_access>                     = {},
                             bool_constant<oob_conditional_check> = {}) const
    {
        using vector_t = typename Base::Traits::vector_t;
        using SFC_Ys   = typename Base::Traits::SFC_Ys;

        constexpr auto tile_dstr = typename Base::TileDstr{};

        // auto dst_tensor = make_static_distributed_tensor<DataType>(tile_dstr);

        auto issue = [&](auto i_access_) {
            constexpr auto IAccess = number<i_access_>{};

            constexpr auto non_linear_id    = number<AccessMap_NonLinear{}[IAccess]>{};
            auto bottom_tensor_thread_coord = cached_coords_[non_linear_id];
            auto bottom_tensor_flag         = cached_flags_[IAccess];

            constexpr auto linear_offset = get_bottom_linear_offset(IAccess);

            // read from bottom tensor
            const vector_t vec_value =
                this->get_bottom_tensor_view().template get_vectorized_elements<vector_t>(
                    bottom_tensor_thread_coord,
                    linear_offset,
                    bottom_tensor_flag,
                    bool_constant<oob_conditional_check>{});
            // data index [y0, y1, ...]
            constexpr auto idx_diff_ys = SFC_Ys::get_index(IAccess);
            // write into distributed tensor
            static_for<0, Base::Traits::ScalarPerVector, Base::Traits::PackedSize>{}([&](auto j) {
                constexpr auto idx_ys = generate_tuple(
                    [&](auto jj) {
                        return jj == Base::Traits::VectorDimY ? (idx_diff_ys[jj] + j)
                                                              : idx_diff_ys[jj];
                    },
                    number<Base::NDimY>{});

                constexpr index_t d = tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                                      Base::Traits::PackedSize;

                dst_tensor.get_thread_buffer().template at<d>() =
                    vec_value
                        .template get_as<typename Base::DataType>()[j / Base::Traits::PackedSize];
            });
        };

        WINDOW_DISPATCH_ISSUE();

        return dst_tensor;
    }

    template <typename DstTile,
              index_t i_access           = -1,
              bool oob_conditional_check = true,
              bool pre_nop               = false>
    CK_TILE_DEVICE void load_raw(DstTile& dst_tensor,
                                 number<i_access> = {}, // negative means loop over all num_access
                                 bool_constant<oob_conditional_check> = {},
                                 bool_constant<pre_nop>               = {}) const
    {
        using vector_t = typename Base::Traits::vector_t;
        using SFC_Ys   = typename Base::Traits::SFC_Ys;
        static constexpr index_t YElementSize =
            typename Base::TileDstr{}.get_ys_to_d_descriptor().get_element_space_size();
        static_assert(YElementSize % (Base::Traits::PackedSize * Base::Traits::ScalarPerVector) ==
                      0);
        using vectorized_tbuf =
            array<vector_t,
                  YElementSize / (Base::Traits::PackedSize * Base::Traits::ScalarPerVector)>;

        constexpr auto tile_dstr = typename Base::TileDstr{};

        auto& dst_vec_tbuf = reinterpret_cast<vectorized_tbuf&>(dst_tensor.get_thread_buffer());

        auto issue = [&](auto i_access_) {
            constexpr auto IAccess  = number<i_access_>{};
            constexpr auto pre_nop_ = [&]() {
                if constexpr(pre_nop && i_access_ == 0 &&
                             Base::BottomTensorView::buffer_view::get_address_space() ==
                                 address_space_enum::global)
                    return bool_constant<true>{};
                else
                    return bool_constant<false>{};
            }();

            constexpr auto non_linear_id    = number<AccessMap_NonLinear{}[IAccess]>{};
            auto bottom_tensor_thread_coord = cached_coords_[non_linear_id];
            constexpr auto linear_offset    = get_bottom_linear_offset(IAccess);
            auto bottom_tensor_flag         = cached_flags_[IAccess];

            // data index [y0, y1, ...]
            constexpr auto idx_ys_start = SFC_Ys::get_index(IAccess);
            constexpr index_t d =
                tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys_start) /
                Base::Traits::PackedSize;
            static_assert(d % Base::Traits::ScalarPerVector == 0);

            this->get_bottom_tensor_view().template get_vectorized_elements_raw<vector_t>(
                dst_vec_tbuf.template at<d / Base::Traits::ScalarPerVector>(),
                bottom_tensor_thread_coord,
                linear_offset /**/,
                bottom_tensor_flag,
                bool_constant<oob_conditional_check>{},
                pre_nop_);
#if CK_TILE_WORKAROUND_ROCM_6_1_SCRATCH_MEMORY_ISSUE || \
    CK_TILE_WORKAROUND_ROCM_6_2_SCRATCH_MEMORY_ISSUE
            asm volatile(""); // this is starting from rocm-6.2, but same sympton, reuse this flag
#endif
        };

        WINDOW_DISPATCH_ISSUE();
    }

    // TODO: currently async load only implemented in inline asm
    template <typename LdsTileWindow_,
              index_t i_access           = -1,
              bool oob_conditional_check = true,
              bool pre_nop               = false>
    CK_TILE_DEVICE auto async_load_raw(LdsTileWindow_&& lds_tile,
                                       number<i_access>                     = {},
                                       bool_constant<oob_conditional_check> = {},
                                       bool_constant<pre_nop>               = {}) const
    {
        using LdsTileWindow = remove_cvref_t<LdsTileWindow_>;
        using LdsDataType   = typename LdsTileWindow::DataType;

        // currently we only support everything is non linear dim
        // actually it's not performant if we have linear dim(e.g. fast changing)
        static_assert(NumAccess_NonLinear == NumAccess);
        static_assert(Base::BottomTensorView::buffer_view::get_address_space() ==
                      address_space_enum::global);

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

        const index_t m0_init_value = size_per_buf + size_per_wave * get_warp_id();
        m0_set_with_memory(m0_init_value); // This should be wave independent

        using vector_t = typename Base::Traits::vector_t;

        LdsDataType* smem = lds_tile.get_bottom_tensor_view().get_buffer_view().p_data_;

        // loop over thread tensor space [y0, y1, ...]
        auto issue = [&](auto i_access_) {
            constexpr auto IAccess  = number<i_access_>{};
            constexpr auto pre_nop_ = [&]() {
                if constexpr(pre_nop && i_access_ == 0)
                    return bool_constant<true>{};
                else
                    return bool_constant<false>{};
            }();

            constexpr auto non_linear_id    = number<AccessMap_NonLinear{}[IAccess]>{};
            auto bottom_tensor_thread_coord = cached_coords_[non_linear_id];
            auto bottom_tensor_flag         = cached_flags_[IAccess]; // get this flag anyway

            // read from bottom tensor
            this->get_bottom_tensor_view().template async_get_vectorized_elements_raw<vector_t>(
                smem, bottom_tensor_thread_coord, 0, bottom_tensor_flag, pre_nop_);

            // move thread coordinate
            if constexpr(i_access_ != (NumAccess - 1))
            {
                m0_inc_with_memory(size_per_issue);
            }
        };

        WINDOW_DISPATCH_ISSUE();
    }

    template <typename LdsTileWindow_, index_t i_access = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE auto async_load(LdsTileWindow_&& lds_tile,
                                   number<i_access>                     = {},
                                   bool_constant<oob_conditional_check> = {}) const
    {
        using LdsTileWindow = remove_cvref_t<LdsTileWindow_>;
        using LdsDataType   = typename LdsTileWindow::DataType;
        using vector_t      = typename traits::vector_t;

        static_assert(NumAccess_NonLinear == NumAccess, "Unsupported configuration");
        static_assert(Base::BottomTensorView::buffer_view::get_address_space() ==
                          address_space_enum::global,
                      "Requires global memory");

        // Precompute invariant values outside the lambda
        const auto window_origin       = lds_tile.get_window_origin();
        const auto& bottom_tensor_view = lds_tile.get_bottom_tensor_view();
        const auto& tensor_descriptor  = bottom_tensor_view.get_tensor_descriptor();
        auto smem_base_ptr             = bottom_tensor_view.get_buffer_view().p_data_;

        auto issue = [&](auto i_access_) {
            constexpr auto IAccess       = number<i_access_>{};
            constexpr auto non_linear_id = number<AccessMap_NonLinear{}[IAccess]>{};

            // Use precomputed values
            auto bottom_tensor_thread_coord = cached_coords_[non_linear_id];
            auto window_adaptor_coord       = cached_window_adaptor_coords_[non_linear_id];
            auto bottom_tensor_flag         = cached_flags_[IAccess];

            auto lds_bottom_tensor_thread_idx =
                window_origin + window_adaptor_coord.get_bottom_index();
            const auto lds_coord =
                make_tensor_coordinate(tensor_descriptor, lds_bottom_tensor_thread_idx);

            CK_TILE_LDS_ADDR LdsDataType* smem = smem_base_ptr + lds_coord.get_offset();

            // Read from bottom tensor
            this->get_bottom_tensor_view().template async_get_vectorized_elements<vector_t>(
                smem,
                bottom_tensor_thread_coord,
                0,
                bottom_tensor_flag,
                bool_constant<oob_conditional_check>{});
        };

        WINDOW_DISPATCH_ISSUE();
    }

    template <typename Policy, index_t i_access_unsupport_ = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE auto load_transpose() const
    {
        constexpr auto tile_dstr = typename Base::TileDstr{};
        auto dst_tensor = make_static_distributed_tensor<typename Base::DataType>(tile_dstr);
        this->template load_transpose_linear<Policy>(
            dst_tensor, number<i_access_unsupport_>{}, bool_constant<oob_conditional_check>{});
        return dst_tensor;
    }

    template <typename Policy,
              typename DistributedTensor,
              index_t i_access           = -1,
              bool oob_conditional_check = true>
    CK_TILE_DEVICE auto load_transpose_linear(DistributedTensor& dst_tensor,
                                              number<i_access>                     = {},
                                              bool_constant<oob_conditional_check> = {}) const
    {
        using vector_t = typename traits::vector_t;
        using SFC_Ys   = typename traits::SFC_Ys;

        constexpr auto tile_dstr = typename Base::TileDstr{};

        constexpr auto group_func = Policy::group_func;

        auto issue = [&](auto i_access_) {
            constexpr auto IAccess          = number<i_access_>{};
            constexpr auto non_linear_id    = number<AccessMap_NonLinear{}[IAccess]>{};
            auto bottom_tensor_thread_coord = cached_coords_[non_linear_id];
            auto bottom_tensor_flag         = cached_flags_[IAccess];

            constexpr auto idx_ys_start = SFC_Ys::get_index(IAccess);

            // read from bottom tensor
            const vector_t vec_value =
                this->get_bottom_tensor_view().template get_transpose_vectorized_elements<vector_t>(
                    bottom_tensor_thread_coord, 0);
            // write into distributed tensor
            static_for<0, traits::ScalarPerVector, 1>{}([&](auto j) {
                constexpr auto idx_ys = generate_tuple(
                    [&](auto jj) {
                        return jj == traits::VectorDimY ? (idx_ys_start[jj] + j) : idx_ys_start[jj];
                    },
                    number<Base::NDimY>{});

                constexpr index_t linear_distributed_index =
                    tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys);
                dst_tensor.get_thread_buffer().template at<linear_distributed_index>() =
                    vec_value.template get_as<typename Base::DataType>()[j];
            });
        };
        WINDOW_DISPATCH_ISSUE();
    }

    template <index_t i_access = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE void store(const static_distributed_tensor<typename Base::DataType,
                                                              typename Base::TileDstr>& dstr_tensor,
                              number<i_access>                     = {},
                              bool_constant<oob_conditional_check> = {}) const
    {

        using vector_t = typename Base::Traits::vector_t;
        using SFC_Ys   = typename Base::Traits::SFC_Ys;

        constexpr auto tile_dstr = typename Base::TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        auto issue = [&](auto i_access_) {
            constexpr auto IAccess          = number<i_access_>{};
            constexpr auto non_linear_id    = number<AccessMap_NonLinear{}[IAccess]>{};
            auto bottom_tensor_thread_coord = cached_coords_[non_linear_id];
            constexpr auto linear_offset    = get_bottom_linear_offset(IAccess);
            auto bottom_tensor_flag         = cached_flags_[IAccess];
            // data index [y0, y1, ...]
            constexpr auto idx_ys_start = SFC_Ys::get_index(IAccess);

            // read from distributed tensor
            vector_t vec_value;

            static_for<0, Base::Traits::ScalarPerVector, Base::Traits::PackedSize>{}([&](auto j) {
                constexpr auto idx_ys = generate_tuple(
                    [&](auto jj) {
                        return jj == Base::Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                              : idx_ys_start[jj];
                    },
                    number<Base::NDimY>{});

                constexpr index_t d = tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                                      Base::Traits::PackedSize;

                vec_value.template get_as<typename Base::DataType>()(j / Base::Traits::PackedSize) =
                    dstr_tensor.get_thread_buffer().template at<d>();
            });

            // write into bottom tensor
            this->get_bottom_tensor_view().template set_vectorized_elements<vector_t>(
                bottom_tensor_thread_coord,
                linear_offset,
                bottom_tensor_flag,
                vec_value,
                bool_constant<oob_conditional_check>{});
        };

        WINDOW_DISPATCH_ISSUE();
    }

    template <index_t i_access = -1>
    CK_TILE_DEVICE void
    store_raw(const static_distributed_tensor<typename Base::DataType, typename Base::TileDstr>&
                  dstr_tensor,
              number<i_access> = {}) const
    {
        using vector_t = typename Base::Traits::vector_t;
        using SFC_Ys   = typename Base::Traits::SFC_Ys;

        constexpr auto tile_dstr                    = typename Base::TileDstr{};
        static constexpr bool oob_conditional_check = true;

        // loop over thread tensor space [y0, y1, ...]
        auto issue = [&](auto i_access_) {
            constexpr auto IAccess          = number<i_access_>{};
            constexpr auto non_linear_id    = number<AccessMap_NonLinear{}[IAccess]>{};
            auto bottom_tensor_thread_coord = cached_coords_[non_linear_id];
            constexpr auto linear_offset    = get_bottom_linear_offset(IAccess);
            auto bottom_tensor_flag         = cached_flags_[IAccess];

            // data index [y0, y1, ...]
            constexpr auto idx_ys_start = SFC_Ys::get_index(IAccess);

            // read from distributed tensor
            vector_t vec_value;
            static_for<0, Base::Traits::ScalarPerVector, Base::Traits::PackedSize>{}([&](auto j) {
                constexpr auto idx_ys = generate_tuple(
                    [&](auto jj) {
                        return jj == Base::Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                              : idx_ys_start[jj];
                    },
                    number<Base::NDimY>{});
                constexpr index_t d = tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                                      Base::Traits::PackedSize;
                vec_value.template get_as<typename Base::DataType>()(j / Base::Traits::PackedSize) =
                    dstr_tensor.get_thread_buffer().template at<d>();
            });

            // write into bottom tensor
            this->get_bottom_tensor_view()
                .template set_vectorized_elements_raw<vector_t, oob_conditional_check>(
                    bottom_tensor_thread_coord, linear_offset, bottom_tensor_flag, vec_value);
        };

        WINDOW_DISPATCH_ISSUE();
    }

    template <index_t i_access = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE void
    update(const static_distributed_tensor<typename Base::DataType, typename Base::TileDstr>&
               dstr_tensor,
           number<i_access>                     = {},
           bool_constant<oob_conditional_check> = {}) const
    {

        using vector_t = typename Base::Traits::vector_t;
        using SFC_Ys   = typename Base::Traits::SFC_Ys;

        constexpr auto tile_dstr = typename Base::TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        auto issue = [&](auto i_access_) {
            constexpr auto IAccess          = number<i_access_>{};
            constexpr auto non_linear_id    = number<AccessMap_NonLinear{}[IAccess]>{};
            auto bottom_tensor_thread_coord = cached_coords_[non_linear_id];
            constexpr auto linear_offset    = get_bottom_linear_offset(IAccess);
            auto bottom_tensor_flag         = cached_flags_[IAccess];

            // data index [y0, y1, ...]
            constexpr auto idx_ys_start = SFC_Ys::get_index(IAccess);

            // read from distributed tensor
            vector_t vec_value;

            static_for<0, Base::Traits::ScalarPerVector, Base::Traits::PackedSize>{}([&](auto j) {
                constexpr auto idx_ys = generate_tuple(
                    [&](auto jj) {
                        return jj == Base::Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                              : idx_ys_start[jj];
                    },
                    number<Base::NDimY>{});

                constexpr index_t d = tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                                      Base::Traits::PackedSize;

                vec_value.template get_as<typename Base::DataType>()(j / Base::Traits::PackedSize) =
                    dstr_tensor.get_thread_buffer().template at<d>();
            });

            // write into bottom tensor
            this->get_bottom_tensor_view().template update_vectorized_elements<vector_t>(
                bottom_tensor_thread_coord,
                linear_offset,
                bottom_tensor_flag,
                vec_value,
                bool_constant<oob_conditional_check>{});
        };

        WINDOW_DISPATCH_ISSUE();
    }

    template <index_t i_access = -1, bool oob_conditional_check = true, bool pre_nop = false>
    CK_TILE_DEVICE void
    update_raw(const static_distributed_tensor<typename Base::DataType, typename Base::TileDstr>&
                   dstr_tensor,
               number<i_access>                     = {},
               bool_constant<oob_conditional_check> = {},
               bool_constant<pre_nop>               = {}) const
    {

        using vector_t = typename Base::Traits::vector_t;
        using SFC_Ys   = typename Base::Traits::SFC_Ys;

        constexpr auto tile_dstr = typename Base::TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        auto issue = [&](auto i_access_) {
            constexpr auto IAccess          = number<i_access_>{};
            constexpr auto non_linear_id    = number<AccessMap_NonLinear{}[IAccess]>{};
            auto bottom_tensor_thread_coord = cached_coords_[non_linear_id];
            constexpr auto linear_offset    = get_bottom_linear_offset(IAccess);
            auto bottom_tensor_flag         = cached_flags_[IAccess];

            // data index [y0, y1, ...]
            constexpr auto idx_ys_start = SFC_Ys::get_index(IAccess);

            // read from distributed tensor
            vector_t vec_value;

            static_for<0, Base::Traits::ScalarPerVector, Base::Traits::PackedSize>{}([&](auto j) {
                constexpr auto idx_ys = generate_tuple(
                    [&](auto jj) {
                        return jj == Base::Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                              : idx_ys_start[jj];
                    },
                    number<Base::NDimY>{});

                constexpr index_t d = tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                                      Base::Traits::PackedSize;

                vec_value.template get_as<typename Base::DataType>()(j / Base::Traits::PackedSize) =
                    dstr_tensor.get_thread_buffer().template at<d>();
            });

            // write into bottom tensor
            this->get_bottom_tensor_view().template update_vectorized_elements_raw<vector_t>(
                bottom_tensor_thread_coord,
                linear_offset,
                bottom_tensor_flag,
                vec_value,
                bool_constant<oob_conditional_check>{},
                bool_constant<pre_nop>{});
        };

        WINDOW_DISPATCH_ISSUE();
    }
    // *_extended() functions acts like a virtual function with a default implementation exisiting
    // in the base class
    CK_TILE_DEVICE void move_extended(const typename Base::BottomTensorIndex& step)
    {
        static_for<0, NumAccess, 1>{}([&](auto i_access) {
            constexpr auto IAccess       = number<i_access>{};
            constexpr auto non_linear_id = number<AccessMap_NonLinear{}[i_access]>{};
            constexpr auto need_update_non_linear_coord =
                bool_constant<AccessPrefixSum_NonLinear{}[non_linear_id] == i_access>{};

            if constexpr(need_update_non_linear_coord)
            {
                move_tensor_coordinate(this->bottom_tensor_view_.get_tensor_descriptor(),
                                       cached_coords_(non_linear_id),
                                       step);
            }

            // move the current coord with linear_coords
            auto tmp_coords             = cached_coords_[non_linear_id];
            constexpr auto linear_coord = get_bottom_linear_coordinate(IAccess);
            move_tensor_coordinate(
                this->bottom_tensor_view_.get_tensor_descriptor(), tmp_coords, linear_coord);

            cached_flags_(IAccess) = coordinate_has_valid_offset_assuming_top_index_is_valid(
                this->bottom_tensor_view_.get_tensor_descriptor(), tmp_coords);
        });
    }

    CK_TILE_DEVICE void set_window_origin_extended(const typename Base::BottomTensorIndex&)
    {
        auto window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
            typename Base::TileDstr{}.get_ps_ys_to_xs_adaptor(),
            container_concat(
                make_tuple(get_warp_id(), get_lane_id()),
                generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimY>{})));

        typename Base::BottomTensorIndex bottom_tensor_thread_origin_idx_tmp =
            this->window_origin_ + window_adaptor_thread_coord_tmp.get_bottom_index();

        auto bottom_tensor_thread_coord_tmp = make_tensor_coordinate(
            this->bottom_tensor_view_.get_tensor_descriptor(), bottom_tensor_thread_origin_idx_tmp);

        // future load/store() calls (might allocate more registers)
        using SFC_Ys = typename Base::Traits::SFC_Ys;

        static_for<0, NumAccess, 1>{}([&](auto i_access) {
            constexpr auto non_linear_id = number<AccessMap_NonLinear{}[i_access]>{};
            constexpr auto need_save_non_linear_coord =
                bool_constant<AccessPrefixSum_NonLinear{}[non_linear_id] == i_access>{};

            if constexpr(need_save_non_linear_coord)
            {
                cached_coords_(non_linear_id)                = bottom_tensor_thread_coord_tmp;
                cached_window_adaptor_coords_(non_linear_id) = window_adaptor_thread_coord_tmp;
            }

            if constexpr(i_access != (NumAccess - 1))
            {
                constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(i_access); // tuple of number
                constexpr auto idx_diff_ps_ys = container_concat(
                    generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                    idx_diff_ys);

                Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                    window_adaptor_thread_coord_tmp,
                    bottom_tensor_thread_coord_tmp,
                    idx_diff_ps_ys);
            }
        });
    }

    // this contains:
    array<typename Base::BottomTensorCoord, traits::NumAccess_NonLinear> cached_coords_;
    array<typename Base::WindowAdaptorCoord, traits::NumAccess_NonLinear>
        cached_window_adaptor_coords_;
    array<bool, Base::Traits::NumAccess> cached_flags_;
};

#undef WINDOW_DISPATCH_ISSUE

namespace impl {
template <address_space_enum, index_t len_>
struct default_linear_bottom_dims_impl
{
    using type = typename uniform_sequence_gen<len_, 0>::type;
};

template <index_t len_>
struct default_linear_bottom_dims_impl<address_space_enum::global, len_>
{
    // global default to seq<0,0,....1>
    using type = typename sequence_merge<typename uniform_sequence_gen<len_ - 1, 0>::type,
                                         sequence<1>>::type;
};

template <index_t len_>
struct default_linear_bottom_dims_impl<address_space_enum::lds, len_>
{
    // lds default to seq<1,1.....1>
    using type = typename uniform_sequence_gen<len_, 1>::type;
};
} // namespace impl

template <typename TensorView_>
using default_linear_bottom_dims =
    typename impl::default_linear_bottom_dims_impl<TensorView_::buffer_view::get_address_space(),
                                                   TensorView_::get_num_of_dimension()>::type;

// if using this API, will create a tile_window_linear
// this structure can have the chance to use immediate value, save register
// need pass in LinearBottomDims_ properly to control which dim is linear
// so to generate a constexpr offset as linear_offset for this dim
// (and finally pass to the immediate offset of buffer/lds instruction)
//
// Note: there is no internal check for which dim is OK to use linear offset
// user must make sure by themselves
//
// e.g.
// 2d global matrix, set LinearBottomDims_=seq<0, 1>, the last dim will generate
// immediate offset if each thread has multiple issue along last dim
//
// 2d LDS buffer, set LinearBottomDims_=seq<1, 1>, then only one vgpr used as offset
// everything else is just using immediate offset.
//
template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          typename LinearBottomDims_ = default_linear_bottom_dims<TensorView_>>
CK_TILE_DEVICE constexpr auto
make_tile_window_linear(const TensorView_& tensor_view,
                        const WindowLengths_& window_lengths,
                        const multi_index<TensorView_::get_num_of_dimension()>& origin,
                        const StaticTileDistribution_& tile_distribution,
                        LinearBottomDims_ = {})
{
    static_assert(LinearBottomDims_::size() == TensorView_::get_num_of_dimension());
    return tile_window_linear<remove_cvref_t<TensorView_>,
                              remove_cvref_t<WindowLengths_>,
                              remove_cvref_t<StaticTileDistribution_>,
                              remove_cvref_t<LinearBottomDims_>>{
        tensor_view, window_lengths, origin, tile_distribution};
}

template <
    typename TileWindow_,
    typename StaticTileDistribution_,
    typename LinearBottomDims_ = default_linear_bottom_dims<typename TileWindow_::BottomTensorView>>
CK_TILE_DEVICE constexpr auto
make_tile_window_linear(const TileWindow_& tile_window,
                        const StaticTileDistribution_& tile_distribution,
                        LinearBottomDims_ = {})
{
    return make_tile_window_linear(tile_window.get_bottom_tensor_view(),
                                   tile_window.get_window_lengths(),
                                   tile_window.get_window_origin(),
                                   tile_distribution,
                                   LinearBottomDims_{});
}

// this version must not be called under a constexpr context
template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          typename LinearBottomDims_ = default_linear_bottom_dims<TensorView_>>
CK_TILE_DEVICE auto
make_tile_window_linear_raw(const TensorView_& tensor_view,
                            const WindowLengths_& window_lengths,
                            const multi_index<TensorView_::get_num_of_dimension()>& origin,
                            const StaticTileDistribution_& tile_distribution,
                            LinearBottomDims_ = {})
{
    static_assert(LinearBottomDims_::size() == TensorView_::get_num_of_dimension());
    auto w = tile_window_linear<remove_cvref_t<TensorView_>,
                                remove_cvref_t<WindowLengths_>,
                                remove_cvref_t<StaticTileDistribution_>,
                                remove_cvref_t<LinearBottomDims_>>{
        tensor_view, window_lengths, origin, tile_distribution};
    w.init_raw();
    return w;
}

template <
    typename TileWindow_,
    typename StaticTileDistribution_,
    typename LinearBottomDims_ = default_linear_bottom_dims<typename TileWindow_::BottomTensorView>>
CK_TILE_DEVICE constexpr auto
make_tile_window_linear_raw(const TileWindow_& tile_window,
                            const StaticTileDistribution_& tile_distribution,
                            LinearBottomDims_ = {})
{
    return make_tile_window_linear_raw(tile_window.get_bottom_tensor_view(),
                                       tile_window.get_window_lengths(),
                                       tile_window.get_window_origin(),
                                       tile_distribution,
                                       LinearBottomDims_{});
}

template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          typename LinearBottomDims_>
CK_TILE_DEVICE void move_tile_window(
    tile_window_linear<TensorView_, WindowLengths_, StaticTileDistribution_, LinearBottomDims_>&
        window,
    const typename tile_window_linear<TensorView_,
                                      WindowLengths_,
                                      StaticTileDistribution_,
                                      LinearBottomDims_>::BottomTensorIndex& step)
{
    window.move(step);
}

/**
 * @brief Type trait to determine if a type is a linear tile window.
 *
 * Defaults to `false_type`. Specialized to `true_type` for types that match
 * `tile_window_linear<...>`.
 *
 * @tparam T The type to check.
 */
template <typename T>
struct is_tile_window_linear : std::false_type
{
};

/**
 * @brief Specialization of `is_tile_window_linear` for `tile_window_linear`.
 *
 * Evaluates to `true_type` if the type is a `tile_window_linear` with the given template
 * parameters.
 *
 * @tparam BottomTensorView_ Bottom tensor view type of the tile window.
 * @tparam WindowLengths_ Static window lengths.
 * @tparam StaticTileDistribution_ Tile distribution policy.
 * @tparam LinearBottomDims_ Dimensions of the bottom tensor view that participate in linearization.
 */
template <typename BottomTensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          typename LinearBottomDims_>
struct is_tile_window_linear<tile_window_linear<BottomTensorView_,
                                                WindowLengths_,
                                                StaticTileDistribution_,
                                                LinearBottomDims_>> : std::true_type
{
};

/**
 * @brief Helper variable template to check if a type is a linear tile window.
 *
 * Equivalent to `is_tile_window_linear<T>::value`.
 *
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr bool is_tile_window_linear_v = is_tile_window_linear<T>::value;

} // namespace ck_tile
