// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/tensor/tensor_adaptor_coordinate.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/container/multi_index.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

template <typename RsLengths_,    // sequence<...>
          typename HsLengthss_,   // tuple<sequence<...>, ...>
          typename Ps2RHssMajor_, // tuple<sequence<...>, ...>
          typename Ps2RHssMinor_, // tuple<sequence<...>, ...>
          typename Ys2RHsMajor_,  // sequence<...>
          typename Ys2RHsMinor_>  // sequence<...>
struct tile_distribution_encoding
{
    using RsLengths    = remove_cvref_t<RsLengths_>;
    using HsLengthss   = remove_cvref_t<HsLengthss_>;
    using Ps2RHssMajor = remove_cvref_t<Ps2RHssMajor_>;
    using Ps2RHssMinor = remove_cvref_t<Ps2RHssMinor_>;
    using Ys2RHsMajor  = remove_cvref_t<Ys2RHsMajor_>;
    using Ys2RHsMinor  = remove_cvref_t<Ys2RHsMinor_>;

    static_assert(Ps2RHssMajor::size() == Ps2RHssMinor::size(), "wrong!");
    static_assert(Ys2RHsMajor::size() == Ys2RHsMinor::size(), "wrong!");

    static constexpr index_t NDimX = HsLengthss::size(); // M,K 2
    static constexpr index_t NDimP = Ps2RHssMajor::size();
    static constexpr index_t NDimY = Ys2RHsMajor::size();
    static constexpr index_t NDimR = RsLengths::size();

    // FIXME: move into detail
    static constexpr auto rs_lengths_       = RsLengths{};  // replication length
    static constexpr auto hs_lengthss_      = HsLengthss{}; // hidden dim length
    static constexpr auto ps_to_rhss_major_ = Ps2RHssMajor{};
    static constexpr auto ps_to_rhss_minor_ = Ps2RHssMinor{};
    static constexpr auto ys_to_rhs_major_  = Ys2RHsMajor{}; // <1,2>
    static constexpr auto ys_to_rhs_minor_  = Ys2RHsMinor{}; // <1,1> or <0,1>

    // redundant but useful info
    // TODO: really bad code, should be over-hauled
    struct detail
    {
        // ndim_rh_major_, ndim_span_mainor_
        static constexpr index_t ndim_rh_major_   = NDimX + 1; //
        static constexpr index_t ndim_span_major_ = NDimX;

        // ndims_rhs_minor_[ndim_rh_major_]
        // here unwind all real dim and hidden dim size. ndims_rhs_minor = [ndim_real,
        // ndim_hidden(M/N), ndim_hidden(K)] e.g [1, 3, 2]
        static constexpr auto ndims_rhs_minor_ = generate_array(
            [](auto i) {
                if constexpr(i.value == 0)
                {
                    return rs_lengths_.size();
                }
                else
                {
                    return hs_lengthss_[i - number<1>{}].size();
                }
            },
            number<ndim_rh_major_>{});

        // max_ndim_rh_minor_
        // e.g [3]
        static constexpr index_t max_ndim_rh_minor_ =
            container_reduce(ndims_rhs_minor_, maximize<index_t>{}, 0);

        // rhs_lengthss_[ndim_rh_major_][max_ndim_rh_minor_]
        // get every real_dimension and corresponding  hidden_dim
        static constexpr auto rhs_lengthss_ =
            to_array_of_array(container_concat(make_tuple(rs_lengths_), hs_lengthss_));

        // ys_lengths_
        // 2d array
        static constexpr auto ys_lengths_ = [] {
            array<index_t, NDimY> ys_lengths_tmp{-1};

            for(index_t i = 0; i < NDimY; i++)
            {
                index_t rh_major = ys_to_rhs_major_[i];
                index_t rh_minor = ys_to_rhs_minor_[i];

                ys_lengths_tmp(i) = rhs_lengthss_[rh_major][rh_minor]; // major choose which sequence, minor choose dim
            }

            return ys_lengths_tmp;
        }();

        // rhs_major_minor_to_ys_[ndim_rh_majpr_][max_ndim_rh_minor_]
        static constexpr auto rhs_major_minor_to_ys_ = [] {
            array<array<index_t, max_ndim_rh_minor_>, NDimX + 1> rhs_major_minor_to_ys_tmp{{-1}};

            static_for<0, NDimY, 1>{}([&](auto i) {
                constexpr index_t rh_major = ys_to_rhs_major_[i];
                constexpr index_t rh_minor = ys_to_rhs_minor_[i];

                rhs_major_minor_to_ys_tmp(rh_major)(rh_minor) = i;
            });

            return rhs_major_minor_to_ys_tmp;
        }();

        // ndims_span_minor_[NDimY]
        static constexpr auto ndims_span_minor_ = [] {
            array<index_t, NDimX> ndims_span_minor{0};

            for(index_t i = 0; i < NDimY; i++)
            {
                const index_t span_major = ys_to_rhs_major_[i] - 1;

                ndims_span_minor(span_major)++;
            }

            return ndims_span_minor;
        }();

        // max_ndim_span_minor_
        static constexpr index_t max_ndim_span_minor_ =
            container_reduce(ndims_span_minor_, maximize<index_t>{}, 0);

        // rhs_major_minor_to_span_minor_ [ndim_rh_major_][max_ndim_rh_minor_]
        static constexpr auto rhs_major_minor_to_span_minor_ = [] {
            array<array<index_t, max_ndim_rh_minor_>, ndim_rh_major_> rhs_major_minor_to_span_minor{
                {-1}};

            static_for<0, ndim_rh_major_, 1>{}([&](auto rh_major) {
                constexpr index_t ndim_rh_minor = ndims_rhs_minor_[rh_major];

                index_t cnt_ndim_span_minor = 0;

                static_for<0, ndim_rh_minor, 1>{}([&](auto rh_minor) {
                    constexpr index_t idim_y = rhs_major_minor_to_ys_[rh_major][rh_minor];

                    if(idim_y >= 0)
                    {
                        rhs_major_minor_to_span_minor(rh_major)(rh_minor) = cnt_ndim_span_minor;

                        cnt_ndim_span_minor++;
                    }
                });
            });

            return rhs_major_minor_to_span_minor;
        }();

        // ys_to_span_major_[NDimY]
        static constexpr auto ys_to_span_major_ =
            generate_array([](auto i) { return ys_to_rhs_major_[i] - 1; }, number<NDimY>{});

        // ys_to_span_minor_[NDimY]
        static constexpr auto ys_to_span_minor_ = generate_array(
            [](auto i) {
                return rhs_major_minor_to_span_minor_[ys_to_rhs_major_[i]][ys_to_rhs_minor_[i]];
            },
            number<NDimY>{});

        // distributed_spans_lengthss_[ndim_span_major_][max_ndim_span_minor_]
        // same as y copy but not include replicationdim
        static constexpr auto distributed_spans_lengthss_ = [] {
            array<array<index_t, max_ndim_span_minor_>, ndim_span_major_>
                distributed_spans_lengthss{{-1}};

            static_for<0, NDimY, 1>{}([&](auto i) {
                const index_t rh_major = ys_to_rhs_major_[i];
                const index_t rh_minor = ys_to_rhs_minor_[i];

                const index_t h_length = hs_lengthss_[number<rh_major - 1>{}][rh_minor];

                const index_t span_major = rh_major - 1;
                const index_t span_minor = rhs_major_minor_to_span_minor_[rh_major][rh_minor];

                distributed_spans_lengthss(span_major)(span_minor) = h_length;
            });

            return distributed_spans_lengthss;
        }();

        // ndims_distributed_spans_minor_[ndim_span_major_]
        static constexpr auto ndims_distributed_spans_minor_ = [] {
            array<index_t, ndim_span_major_> ndims_distributed_spans_minor{0};

            static_for<0, NDimY, 1>{}([&](auto i) {
                const index_t span_major = ys_to_rhs_major_[i] - 1;

                ndims_distributed_spans_minor(span_major)++;
            });

            return ndims_distributed_spans_minor;
        }();

        // does_p_own_r_[NDimP][NDimR]
        static constexpr auto does_p_own_r_ = [] {
            if constexpr(NDimR > 0)
            {
                array<array<bool, NDimR>, NDimP> does_p_own_r{{false}};

                static_for<0, NDimP, 1>{}([&](auto idim_p) {
                    constexpr index_t ndim_low = ps_to_rhss_major_[idim_p].size();

                    static_for<0, ndim_low, 1>{}([&](auto idim_low) {
                        constexpr index_t rh_major = ps_to_rhss_major_[idim_p][idim_low];
                        constexpr index_t rh_minor = ps_to_rhss_minor_[idim_p][idim_low];

                        if constexpr(rh_major == 0)
                        {
                            does_p_own_r(idim_p)(rh_minor) = true;
                        }
                    });
                });

                return does_p_own_r;
            }
            else
            {
                return array<array<bool, NDimR>, NDimP>{};
            }
        }();

        // ps_over_rs_derivative_[NDimP][NDimR]
        static constexpr auto ps_over_rs_derivative_ = [] {
            if constexpr(NDimR > 0)
            {
                array<array<index_t, NDimR>, NDimP> ps_over_rs_derivative{{0}};

                static_for<0, NDimP, 1>{}([&](auto idim_p) {
                    constexpr index_t ndim_low = ps_to_rhss_major_[idim_p].size();

                    index_t p_over_rh_derivative = 1;

                    static_for<ndim_low - 1, -1, -1>{}([&](auto idim_low) {
                        constexpr index_t rh_major = ps_to_rhss_major_[idim_p][idim_low];
                        constexpr index_t rh_minor = ps_to_rhss_minor_[idim_p][idim_low];

                        constexpr index_t rh_length = rhs_lengthss_[rh_major][rh_minor];

                        if constexpr(rh_major == 0)
                        {
                            ps_over_rs_derivative(idim_p)(rh_minor) = p_over_rh_derivative;
                        }

                        p_over_rh_derivative *= rh_length;
                    });
                });

                return ps_over_rs_derivative;
            }
            else
            {
                return array<array<index_t, NDimR>, NDimP>{};
            }
        }();

        // e.g. tuple<seq<1, 4, 32>, seq<4, 1, 4, 2, 4>> --> seq<3, 5> --> seq<0, 3, 8>
        CK_TILE_HOST_DEVICE static constexpr auto get_h_dim_lengths_prefix_sum()
        {
            // <len_d0, len_d1, ...>
            // e.g. tuple<seq<1, 4, 32>, seq<4, 1, 4, 2, 4>> --> seq<3, 5>
            constexpr auto uniformed_h_dim_lengths = generate_sequence_v2(
                [&](auto i) {
                    constexpr index_t size = HsLengthss{}[i].size();
                    return number<size>{};
                },
                number<NDimX>{});

            // <0, len_d0, len_d0+len_d1, ...>
            // e.g. seq<3, 5> --> seq<0, 3, 8>
            constexpr auto h_dim_prefix_sum = prefix_sum_sequence(uniformed_h_dim_lengths);

            return h_dim_prefix_sum;
        }

        CK_TILE_HOST_DEVICE static constexpr auto get_uniformed_idx_y_to_h()
        {
            constexpr auto all_ys_2_rhss = transform_sequences(
                [](auto major, auto minor) constexpr {
                    // <0, 0, len_d0, len_d0+len_d1, ...>
                    constexpr auto x_dim_prefix_sum = merge_sequences(
                        sequence<0>{} /*for R dims*/, get_h_dim_lengths_prefix_sum());
                    return x_dim_prefix_sum.at(major) + minor;
                },
                Ys2RHsMajor{},
                Ys2RHsMinor{});

            return all_ys_2_rhss;
        }

        // return tuple<sorted_dims, sorted_maps, sorted_prefix_sum>
        template <typename IdxSeq, typename PrefixSumSeq>
        CK_TILE_HOST_DEVICE static constexpr auto get_sorted_info(IdxSeq, PrefixSumSeq)
        {
            using sorted_idx = sequence_unique_sort<IdxSeq, less<index_t>, equal<index_t>>;

            constexpr auto sorted_dims = typename sorted_idx::type{};
            constexpr auto sorted_maps = typename sorted_idx::sorted2unsorted_map{};

            constexpr auto sorted_histogram =
                histogram_sorted_sequence(sorted_dims, PrefixSumSeq{});
            constexpr auto sorted_prefix_sum = prefix_sum_sequence(sorted_histogram);

            return make_tuple(sorted_dims, sorted_maps, sorted_prefix_sum);
        }

        CK_TILE_HOST_DEVICE static constexpr auto get_sorted_y_info()
        {
            return get_sorted_info(get_uniformed_idx_y_to_h(), get_h_dim_lengths_prefix_sum());
        }

        CK_TILE_HOST_DEVICE void print() const
        {
            printf("tile_distribution_encoding::detail{");
            //
            printf("ndim_rh_major_: ");
            print(ndim_rh_major_);
            printf(", ");
            //
            printf("ndim_span_major_: ");
            print(ndim_span_major_);
            printf(", ");
            //
            printf("ndims_rhs_minor_: ");
            print(ndims_rhs_minor_);
            printf(", ");
            //
            printf("ndim_rh_major_: ");
            print(ndim_rh_major_);
            printf(", ");
            //
            printf("max_ndim_rh_minor_: ");
            print(max_ndim_rh_minor_);
            printf(", ");
            //
            printf("rhs_lengthss_: ");
            print(rhs_lengthss_);
            printf(", ");
            //
            printf("ys_lengths_: ");
            print(ys_lengths_);
            printf(", ");
            //
            printf("rhs_major_minor_to_ys_: ");
            print(rhs_major_minor_to_ys_);
            printf(", ");
            //
            printf("ndims_span_minor_: ");
            print(ndims_span_minor_);
            printf(", ");
            //
            printf("max_ndim_span_minor_: ");
            print(max_ndim_span_minor_);
            printf(", ");
            //
            printf("ys_to_span_major_: ");
            print(ys_to_span_major_);
            printf(", ");
            //
            printf("ys_to_span_minor_: ");
            print(ys_to_span_minor_);
            printf(", ");
            //
            printf("distributed_spans_lengthss_: ");
            print(distributed_spans_lengthss_);
            printf(", ");
            //
            printf("ndims_distributed_spans_minor_: ");
            print(ndims_distributed_spans_minor_);
            printf(", ");
            //
            printf("ps_over_rs_derivative_: ");
            print(ps_over_rs_derivative_);
            //
            printf("}");
        }
    };

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("tile_distribution_encoding{");
        //
        printf("NDimX: %d, NDimP: %d, NDimY: %d, ", NDimX, NDimP, NDimY);
        //
        printf("rs_lengths_: ");
        print(rs_lengths_);
        printf(", ");
        //
        printf("hs_lengthss_: ");
        print(hs_lengthss_);
        printf(", ");
        //
        printf("ps_to_rhss_major_: ");
        print(ps_to_rhss_major_);
        printf(", ");
        //
        printf("ps_to_rhss_minor_: ");
        print(ps_to_rhss_minor_);
        printf(", ");
        //
        printf("ys_to_rhs_major_: ");
        print(ys_to_rhs_major_);
        printf(", ");
        //
        printf("ys_to_rhs_minor_: ");
        print(ys_to_rhs_minor_);
        printf(", ");
        //
        printf("detail: ");
        print(detail{});
        //
        printf("}");
    }
};

namespace detail {

template <typename OuterDstr, typename InnerDstr>
CK_TILE_HOST_DEVICE constexpr auto make_embed_tile_distribution_encoding(OuterDstr, InnerDstr)
{
    static_assert(OuterDstr::NDimX == InnerDstr::NDimX, "wrong!");

    constexpr index_t NDimHMajor = OuterDstr::NDimX;

    using RsLengths =
        sequence_merge_t<typename OuterDstr::RsLengths, typename InnerDstr::RsLengths>;

    constexpr auto hs_lengthss = generate_tuple(
        [&](auto i) {
            return merge_sequences(typename OuterDstr::HsLengthss{}[i],
                                   typename InnerDstr::HsLengthss{}[i]);
        },
        number<NDimHMajor>{});

    //
    constexpr auto rhs_major_2_ndim_outer_rhs_minor = [&]() {
        array<index_t, NDimHMajor + 1> rhs_major_2_ndim_outer_rhs_minor_;

        // R dimension
        rhs_major_2_ndim_outer_rhs_minor_(0) = OuterDstr::RsLengths::size();

        // Hs dimensions
        static_for<0, NDimHMajor, 1>{}([&](auto i) {
            rhs_major_2_ndim_outer_rhs_minor_(i + 1) = typename OuterDstr::HsLengthss{}[i].size();
        });

        return rhs_major_2_ndim_outer_rhs_minor_;
    }();

    // Ps2RHssMinor
    constexpr auto updated_inner_ps_2_rhss_minor = generate_tuple(
        [&](auto p) {
            constexpr auto inner_p_2_rhss_major = typename InnerDstr::Ps2RHssMajor{}[p];
            constexpr auto inner_p_2_rhss_minor = typename InnerDstr::Ps2RHssMinor{}[p];

            constexpr index_t ndim_tmp = inner_p_2_rhss_minor.size();

            constexpr auto updated_inner_p_2_rhss_minor = [&]() {
                array<index_t, ndim_tmp> updated_inner_p_2_rhss_minor_;

                for(index_t i = 0; i < ndim_tmp; i++)
                {
                    index_t rh_major = inner_p_2_rhss_major[i];

                    index_t ndim_outer_h_minor = rhs_major_2_ndim_outer_rhs_minor[rh_major];

                    updated_inner_p_2_rhss_minor_(i) = inner_p_2_rhss_minor[i] + ndim_outer_h_minor;
                }

                return updated_inner_p_2_rhss_minor_;
            }();

            return TO_SEQUENCE(updated_inner_p_2_rhss_minor, ndim_tmp);
        },
        number<InnerDstr::NDimP>{});

    // Ys2RHsMinor
    constexpr auto updated_inner_ys_2_rhs_minor = [&]() {
        constexpr auto inner_ys_2_rhs_major = typename InnerDstr::Ys2RHsMajor{};
        constexpr auto inner_ys_2_rhs_minor = typename InnerDstr::Ys2RHsMinor{};

        constexpr index_t ndim_tmp = inner_ys_2_rhs_minor.size();

        constexpr auto updated_inner_ys_2_rhs_minor_ = [&]() {
            array<index_t, ndim_tmp> updated_inner_ys_2_rhs_minor__;

            for(index_t i = 0; i < ndim_tmp; i++)
            {
                index_t rh_major = inner_ys_2_rhs_major[i];

                index_t ndim_outer_h_minor = rhs_major_2_ndim_outer_rhs_minor[rh_major];

                updated_inner_ys_2_rhs_minor__(i) = inner_ys_2_rhs_minor[i] + ndim_outer_h_minor;
            }

            return updated_inner_ys_2_rhs_minor__;
        }();

        return TO_SEQUENCE(updated_inner_ys_2_rhs_minor_, ndim_tmp);
    }();

    //
    constexpr auto ps_2_rhss_major =
        container_concat(typename OuterDstr::Ps2RHssMajor{}, typename InnerDstr::Ps2RHssMajor{});

    constexpr auto ps_2_rhss_minor =
        container_concat(typename OuterDstr::Ps2RHssMinor{}, updated_inner_ps_2_rhss_minor);

    //
    constexpr auto ys_2_rhs_major =
        merge_sequences(typename OuterDstr::Ys2RHsMajor{}, typename InnerDstr::Ys2RHsMajor{});

    constexpr auto ys_2_rhs_minor =
        merge_sequences(typename OuterDstr::Ys2RHsMinor{}, updated_inner_ys_2_rhs_minor);

    return tile_distribution_encoding<RsLengths,
                                      remove_cvref_t<decltype(hs_lengthss)>,
                                      remove_cvref_t<decltype(ps_2_rhss_major)>,
                                      remove_cvref_t<decltype(ps_2_rhss_minor)>,
                                      remove_cvref_t<decltype(ys_2_rhs_major)>,
                                      remove_cvref_t<decltype(ys_2_rhs_minor)>>{};
}

template <typename InDstr, index_t... InReduceDimXs>
CK_TILE_HOST_DEVICE constexpr auto
make_reduce_tile_distribution_encoding_impl(InDstr, sequence<InReduceDimXs...> reduce_dim_xs_in)
{
    constexpr auto I1 = number<1>{};

    // FIXME: increase if fail
    constexpr index_t max_ndim_r_out = 20;
    constexpr index_t max_ndim_y_out = 20;

    //
    constexpr index_t ndim_p               = InDstr::NDimP;
    constexpr index_t ndim_x_in            = InDstr::NDimX;
    constexpr index_t ndim_y_in            = InDstr::NDimY;
    constexpr index_t ndim_rh_major_in     = InDstr::NDimX + 1;
    constexpr index_t ndim_x_out           = ndim_x_in - sizeof...(InReduceDimXs);
    constexpr index_t max_ndim_rh_minor_in = InDstr::detail::max_ndim_rh_minor_;

    // ndims_ps_low
    constexpr auto ndims_ps_low = generate_array(
        [&](auto i) { return InDstr::ps_to_rhss_major_[i].size(); }, number<ndim_p>{});

    // is_rh_major_in_for_reduce
    array<bool, ndim_rh_major_in> is_rh_major_in_for_reduce{false};

    for(index_t i = 0; i < reduce_dim_xs_in.size(); i++)
    {
        index_t rh_major = reduce_dim_xs_in[i] + 1;

        is_rh_major_in_for_reduce(rh_major) = true;
    }

    // is_y_in_for_reduce
    array<bool, ndim_y_in> is_y_in_for_reduce{false};

    for(index_t i = 0; i < ndim_y_in; i++)
    {
        index_t rh_major = InDstr::ys_to_rhs_major_[i];

        if(is_rh_major_in_for_reduce[rh_major])
        {
            is_y_in_for_reduce(i) = true;
        }
    }

    // is_rh_minor_in_for_y_reduce
    array<array<bool, max_ndim_rh_minor_in>, ndim_rh_major_in> is_rh_minor_in_for_y_reduce{{false}};

    static_for<0, ndim_y_in, 1>{}([&](auto i) {
        index_t rh_major = InDstr::ys_to_rhs_major_[i];
        index_t rh_minor = InDstr::ys_to_rhs_minor_[i];

        if(is_y_in_for_reduce[i])
        {
            is_rh_minor_in_for_y_reduce(rh_major)(rh_minor) = true;
        }
    });

    // in2out_rh_major
    array<index_t, ndim_rh_major_in> in2out_rh_major{-1};
    index_t cnt_ndim_rh_major_out = 0;

    for(index_t i = 0; i < ndim_rh_major_in; i++)
    {
        if(is_rh_major_in_for_reduce[i])
        {
            in2out_rh_major(i) = 0;
        }
        else
        {
            in2out_rh_major(i) = cnt_ndim_rh_major_out;

            cnt_ndim_rh_major_out++;
        }
    }

    // rs_lengths_out, in2out_rh_minor
    array<index_t, max_ndim_r_out> rs_lengths_out{-1};
    array<array<index_t, max_ndim_rh_minor_in>, ndim_rh_major_in> in2out_rh_minor{{-1}};

    // loop over input R dim
    for(index_t i = 0; i < InDstr::rs_lengths_.size(); i++)
    {
        // rs_lengths_out
        rs_lengths_out(i) = InDstr::rs_lengths_[i];

        // in2out_rh_minor
        in2out_rh_minor(0)(i) = i;
    }

    // loop over input H Dim
    index_t cnt_ndim_r_out = InDstr::rs_lengths_.size();

    static_for<1, ndim_rh_major_in, 1>{}([&](auto rh_major_in) {
        constexpr auto h_major_in = rh_major_in - I1;

        constexpr index_t ndim_rh_minor_in = InDstr::hs_lengthss_[h_major_in].size();

        if(is_rh_major_in_for_reduce[rh_major_in])
        {
            for(index_t rh_minor_in = 0; rh_minor_in < ndim_rh_minor_in; rh_minor_in++)
            {
                if(not is_rh_minor_in_for_y_reduce[rh_major_in][rh_minor_in])
                {
                    // rs_lengths_out
                    rs_lengths_out(cnt_ndim_r_out) = InDstr::hs_lengthss_[h_major_in][rh_minor_in];

                    // in2out_rh_minor
                    in2out_rh_minor(rh_major_in)(rh_minor_in) = cnt_ndim_r_out;

                    cnt_ndim_r_out++;
                }
            }
        }
        else
        {
            for(index_t rh_minor_in = 0; rh_minor_in < ndim_rh_minor_in; rh_minor_in++)
            {
                // in2out_rh_minor
                in2out_rh_minor(rh_major_in)(rh_minor_in) = rh_minor_in;
            }
        }
    });

    // ndim_r_out
    const index_t ndim_r_out = cnt_ndim_r_out;

    // ndims_hs_minor_out, hs_lengthss_out
    array<index_t, ndim_x_out> ndims_hs_minor_out{-1};
    array<array<index_t, max_ndim_rh_minor_in>, ndim_x_out> hs_lengthss_out{{-1}};

    index_t cnt_ndim_x_out = 0;

    static_for<0, ndim_x_in, 1>{}([&](auto i) {
        if(not is_rh_major_in_for_reduce[i + I1])
        {
            // ndims_hs_minor_out
            ndims_hs_minor_out(cnt_ndim_x_out) = InDstr::hs_lengthss_[i].size();

            // hs_lengthss_out
            static_for<0, InDstr::hs_lengthss_[i].size(), 1>{}(
                [&](auto j) { hs_lengthss_out(cnt_ndim_x_out)(j) = InDstr::hs_lengthss_[i][j]; });

            cnt_ndim_x_out++;
        }
    });

    // ps_to_rhss_major_out, ps_to_rhss_minor_out
    array<array<index_t, max_ndim_rh_minor_in>, ndim_p> ps_to_rhss_major_out{{-1}};
    array<array<index_t, max_ndim_rh_minor_in>, ndim_p> ps_to_rhss_minor_out{{-1}};

    static_for<0, ndim_p, 1>{}([&](auto idim_p) {
        static_for<0, InDstr::ps_to_rhss_major_[idim_p].size(), 1>{}([&](auto idim_low) {
            index_t rh_major_in = InDstr::ps_to_rhss_major_[idim_p][idim_low];
            index_t rh_minor_in = InDstr::ps_to_rhss_minor_[idim_p][idim_low];

            ps_to_rhss_major_out(idim_p)(idim_low) = in2out_rh_major[rh_major_in];
            ps_to_rhss_minor_out(idim_p)(idim_low) = in2out_rh_minor[rh_major_in][rh_minor_in];
        });
    });

    // ys_to_rhs_major_out, ys_to_rhs_minor_out
    array<index_t, max_ndim_y_out> ys_to_rhs_major_out{-1};
    array<index_t, max_ndim_y_out> ys_to_rhs_minor_out{-1};

    index_t cnt_ndim_y_out = 0;

    static_for<0, ndim_y_in, 1>{}([&](auto i) {
        if(not is_y_in_for_reduce[i])
        {
            index_t rh_major_in = InDstr::ys_to_rhs_major_[i];
            index_t rh_minor_in = InDstr::ys_to_rhs_minor_[i];

            ys_to_rhs_major_out(cnt_ndim_y_out) = in2out_rh_major[rh_major_in];
            ys_to_rhs_minor_out(cnt_ndim_y_out) = in2out_rh_minor[rh_major_in][rh_minor_in];

            cnt_ndim_y_out++;
        }
    });

    // ndim_y_out
    const index_t ndim_y_out = cnt_ndim_y_out;

    //
    return make_tuple(ndim_x_out,
                      ndim_p,
                      ndim_y_out,
                      ndim_r_out,
                      ndims_hs_minor_out,
                      ndims_ps_low,
                      rs_lengths_out,
                      hs_lengthss_out,
                      ps_to_rhss_major_out,
                      ps_to_rhss_minor_out,
                      ys_to_rhs_major_out,
                      ys_to_rhs_minor_out);
}

template <typename InDstr, index_t... InReduceDimXs>
CK_TILE_HOST_DEVICE constexpr auto
make_reduce_tile_distribution_encoding(InDstr, sequence<InReduceDimXs...> reduce_dim_xs_in)
{
    constexpr auto impl = make_reduce_tile_distribution_encoding_impl(InDstr{}, reduce_dim_xs_in);

    constexpr index_t ndim_x             = impl.template at<0>();
    constexpr index_t ndim_p             = impl.template at<1>();
    constexpr index_t ndim_y             = impl.template at<2>();
    constexpr index_t ndim_r             = impl.template at<3>();
    constexpr auto ndims_hs_minor        = impl.template at<4>();
    constexpr auto ndims_ps_low          = impl.template at<5>();
    constexpr auto rs_lengths_impl       = impl.template at<6>();
    constexpr auto hs_lengthss_impl      = impl.template at<7>();
    constexpr auto ps_to_rhss_major_impl = impl.template at<8>();
    constexpr auto ps_to_rhss_minor_impl = impl.template at<9>();
    constexpr auto ys_to_rhs_major_impl  = impl.template at<10>();
    constexpr auto ys_to_rhs_minor_impl  = impl.template at<11>();

    constexpr auto rs_lengths  = TO_SEQUENCE(rs_lengths_impl, ndim_r);
    constexpr auto hs_lengthss = TO_TUPLE_OF_SEQUENCE(hs_lengthss_impl, ndim_x, ndims_hs_minor);
    constexpr auto ps_to_rhss_major =
        TO_TUPLE_OF_SEQUENCE(ps_to_rhss_major_impl, ndim_p, ndims_ps_low);
    constexpr auto ps_to_rhss_minor =
        TO_TUPLE_OF_SEQUENCE(ps_to_rhss_minor_impl, ndim_p, ndims_ps_low);
    constexpr auto ys_to_rhs_major = TO_SEQUENCE(ys_to_rhs_major_impl, ndim_y);
    constexpr auto ys_to_rhs_minor = TO_SEQUENCE(ys_to_rhs_minor_impl, ndim_y);

    return tile_distribution_encoding<remove_cvref_t<decltype(rs_lengths)>,
                                      remove_cvref_t<decltype(hs_lengthss)>,
                                      remove_cvref_t<decltype(ps_to_rhss_major)>,
                                      remove_cvref_t<decltype(ps_to_rhss_minor)>,
                                      remove_cvref_t<decltype(ys_to_rhs_major)>,
                                      remove_cvref_t<decltype(ys_to_rhs_minor)>>{};
}

} // namespace detail
} // namespace ck_tile
