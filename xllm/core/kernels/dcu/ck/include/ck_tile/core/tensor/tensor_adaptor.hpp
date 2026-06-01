// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/numeric/numeric.hpp"

namespace ck_tile {

// Transforms: Tuple<transforms...>
// LowerDimensionHiddenIdss : Tuple<Sequence<...>, ...>
// UpperDimensionHiddenIdss : Tuple<Sequence<...>, ...>
// BottomDimensionHiddenIds : Sequence<...>
// TopDimensionHiddenIds : Sequence<...>
template <typename Transforms,
          typename LowerDimensionHiddenIdss,
          typename UpperDimensionHiddenIdss,
          typename BottomDimensionHiddenIds,
          typename TopDimensionHiddenIds>
struct tensor_adaptor
{
    CK_TILE_HOST_DEVICE static constexpr index_t get_num_of_transform()
    {
        return Transforms::size();
    }

    CK_TILE_HOST_DEVICE constexpr const auto& get_transforms() const { return transforms_; }

    CK_TILE_HOST_DEVICE static constexpr auto get_lower_dimension_hidden_idss()
    {
        return LowerDimensionHiddenIdss{};
    }

    CK_TILE_HOST_DEVICE static constexpr auto get_upper_dimension_hidden_idss()
    {
        return UpperDimensionHiddenIdss{};
    }

    CK_TILE_HOST_DEVICE static constexpr auto get_bottom_dimension_hidden_ids()
    {
        return BottomDimensionHiddenIds{};
    }

    CK_TILE_HOST_DEVICE static constexpr auto get_top_dimension_hidden_ids()
    {
        return TopDimensionHiddenIds{};
    }

    CK_TILE_HOST_DEVICE static constexpr auto initialize_element_size(const Transforms& transforms)
    {
        const auto lengths = generate_tuple(
            [&](auto idim_top) {
                constexpr index_t idim_hidden = TopDimensionHiddenIds::at(idim_top);

                constexpr auto tmp = get_transform_and_its_upper_dimension(number<idim_hidden>{});

                constexpr index_t itran   = tmp[number<0>{}];
                constexpr index_t idim_up = tmp[number<1>{}];
                constexpr bool found      = tmp[number<2>{}];

                static_assert(found == true,
                              "wrong! not found matching transformation and upper-dimension");

                const auto length =
                    transforms[number<itran>{}].get_upper_lengths()[number<idim_up>{}];

                return length;
            },
            number<ndim_top_>{});

        // TODO: make container_reduce support tuple of number and index_t
        return container_reduce(lengths, multiplies{}, number<1>{});
    }

    template <index_t IDimHidden>
    CK_TILE_HOST_DEVICE static constexpr auto
        get_transform_and_its_upper_dimension(number<IDimHidden>)
    {
        // FIXME: length of bottom dimension is not known, since info about lower dim length are not
        // saved in transformation
        static_assert(IDimHidden >= ndim_bottom_, "wrong! not implemented");

        index_t itran_found   = 0;
        index_t idim_up_found = 0;
        bool found            = false;

        static_for<0, ntransform_, 1>{}([&](auto itran) {
            constexpr auto up_dim_ids = UpperDimensionHiddenIdss{}[itran];

            static_for<0, up_dim_ids.size(), 1>{}([&](auto idim_up) {
                if constexpr(up_dim_ids[idim_up] == IDimHidden)
                {
                    itran_found   = itran;
                    idim_up_found = idim_up;
                    found         = true;
                }
            });
        });

        return make_tuple(itran_found, idim_up_found, found);
    }

    CK_TILE_HOST_DEVICE static constexpr index_t get_num_of_bottom_dimension()
    {
        return BottomDimensionHiddenIds::size();
    }

    CK_TILE_HOST_DEVICE static constexpr index_t get_num_of_top_dimension()
    {
        return TopDimensionHiddenIds::size();
    }

    CK_TILE_HOST_DEVICE static constexpr index_t get_num_of_hidden_dimension()
    {
        constexpr auto all_low_dim_ids = unpack(
            [](auto&&... xs) constexpr { return merge_sequences(xs...); },
            LowerDimensionHiddenIdss{});

        constexpr auto all_up_dim_ids = unpack(
            [](auto&&... xs) constexpr { return merge_sequences(xs...); },
            UpperDimensionHiddenIdss{});

        constexpr auto all_dim_ids = merge_sequences(all_low_dim_ids, all_up_dim_ids);

        using unique_sort_all_dim_ids = typename sequence_unique_sort<decltype(all_dim_ids),
                                                                      less<index_t>,
                                                                      equal<index_t>>::type;

        return unique_sort_all_dim_ids::size();
    }

    constexpr static index_t ntransform_  = get_num_of_transform();
    constexpr static index_t ndim_hidden_ = get_num_of_hidden_dimension();
    constexpr static index_t ndim_bottom_ = get_num_of_bottom_dimension();
    constexpr static index_t ndim_top_    = get_num_of_top_dimension();

    using HiddenIndex = multi_index<ndim_hidden_>;
    using BottomIndex = multi_index<ndim_bottom_>;
    using TopIndex    = multi_index<ndim_top_>;

    // may be index_t or number<>
    using ElementSize = remove_cv_t<decltype(initialize_element_size(Transforms{}))>;

    public:
    CK_TILE_HOST_DEVICE constexpr tensor_adaptor() = default;

    CK_TILE_HOST_DEVICE constexpr tensor_adaptor(const Transforms& transforms)
        : transforms_{transforms}, element_size_{initialize_element_size(transforms)}
    {
        static_assert(Transforms::size() == ntransform_ &&
                          LowerDimensionHiddenIdss::size() == ntransform_ &&
                          UpperDimensionHiddenIdss::size() == ntransform_,
                      "wrong! inconsistent # of transformations");

        // TODO check dependency of dimensions is valid
    }

    CK_TILE_HOST_DEVICE constexpr auto get_element_size() const { return element_size_; }

    // FIXME: this logic is wrong when getting bottome dimension lengths
    template <index_t IDimHidden>
    CK_TILE_HOST_DEVICE constexpr auto get_hidden_dimension_length(number<IDimHidden>) const
    {
        static_assert(IDimHidden >= 0 && IDimHidden < ndim_hidden_, "wrong! out of range");

        constexpr auto tmp = get_transform_and_its_upper_dimension(number<IDimHidden>{});

        constexpr index_t itran   = tmp[number<0>{}];
        constexpr index_t idim_up = tmp[number<1>{}];
        constexpr bool found      = tmp[number<2>{}];

        static_assert(found == true,
                      "wrong! not found matching transformation and upper-dimension");

        return transforms_[number<itran>{}].get_upper_lengths()[number<idim_up>{}];
    }

    template <index_t IDimTop>
    CK_TILE_HOST_DEVICE constexpr auto get_top_dimension_length(number<IDimTop> idim_top) const
    {
        return get_hidden_dimension_length(TopDimensionHiddenIds::at(idim_top));
    }

#if 0
    // FIXME: get_hidden_dimension_length is wrong when getting bottome dimension lengths
    template <index_t IDimBottom>
    CK_TILE_HOST_DEVICE constexpr index_t
    get_bottom_dimension_length(number<IDimBottom> idim_bottom) const
    {
        return get_hidden_dimension_length(TopDimensionHiddenIds::at(idim_bottom));
    }
#endif

    CK_TILE_HOST_DEVICE constexpr auto get_top_dimension_lengths() const
    {
        return generate_tuple([&](auto i) { return get_top_dimension_length(i); },
                              number<ndim_top_>{});
    }

#if 0
    // FIXME: get_hidden_dimension_length is wrong when getting bottome dimension lengths
    CK_TILE_HOST_DEVICE constexpr auto GetBottomDimensionLengths() const
    {
        return generate_tuple([&](auto i) { return get_bottom_dimension_length(i); },
                              number<ndim_bottom_>{});
    }
#endif

    template <typename TopIdx>
    CK_TILE_HOST_DEVICE constexpr auto calculate_bottom_index(const TopIdx& idx_top) const
    {
        static_assert(TopIdx::size() == TopDimensionHiddenIds::size(),
                      "wrong! # of dimension inconsistent");

        constexpr index_t ntransform  = get_num_of_transform();
        constexpr index_t ndim_hidden = get_num_of_hidden_dimension();

        multi_index<ndim_hidden> idx_hidden;

        // initialize uppest index
        set_container_subset(idx_hidden, get_top_dimension_hidden_ids(), idx_top);

        // calculate hidden index
        static_for<ntransform, 0, -1>{}([&](auto itran_p1) {
            auto itran              = itran_p1 - number<1>{};
            const auto& tran        = get_transforms().at(itran);
            constexpr auto dims_low = get_lower_dimension_hidden_idss().at(itran);
            constexpr auto dims_up  = get_upper_dimension_hidden_idss().at(itran);

            const auto idx_up = get_container_subset(idx_hidden, dims_up);

            multi_index<dims_low.size()> idx_low;

            tran.calculate_lower_index(idx_low, idx_up);

            set_container_subset(idx_hidden, dims_low, idx_low);
        });

        return get_container_subset(idx_hidden, BottomDimensionHiddenIds{});
    }

    CK_TILE_HOST_DEVICE static constexpr bool is_static()
    {
        bool is_known = true;

        static_for<0, Transforms::size(), 1>{}([&](auto i) {
            is_known &= remove_cvref_t<decltype(Transforms{}[i])>::is_known_at_compile_time();
        });

        return is_known && ck_tile::is_known_at_compile_time<ElementSize>::value;
    }

    CK_TILE_HOST_DEVICE static constexpr bool is_known_at_compile_time() { return is_static(); }

    CK_TILE_HOST_DEVICE static constexpr auto get_top_dimension_safe_vector_length_strides(
        const array<index_t, ndim_hidden_>& guaranteed_vector_lengths,
        const array<index_t, ndim_hidden_>& guaranteed_vector_strides)
    {
        auto vector_lengths = guaranteed_vector_lengths;
        auto vector_strides = guaranteed_vector_strides;

        static_for<0, get_num_of_transform(), 1>{}([&](auto itran) {
            constexpr auto low_dims = get_lower_dimension_hidden_idss().at(itran);
            constexpr auto up_dims  = get_upper_dimension_hidden_idss().at(itran);

            const auto up_guaranteed_vector_lengths =
                get_container_subset(guaranteed_vector_lengths, up_dims);
            const auto up_guaranteed_vector_strides =
                get_container_subset(guaranteed_vector_strides, up_dims);

            // only need type of transform
            auto [up_vector_lengths, up_vector_strides] =
                Transforms{}.at(itran).calculate_upper_dimension_safe_vector_length_strides(
                    get_container_subset(vector_lengths, low_dims),
                    get_container_subset(vector_strides, low_dims));

            if constexpr(up_dims.size() > 0)
            {
                for(index_t i = 0; i < up_dims.size(); ++i)
                {
                    up_vector_lengths(i) = (up_guaranteed_vector_lengths[i] != -1)
                                               ? up_guaranteed_vector_lengths[i]
                                               : up_vector_lengths[i];

                    up_vector_strides(i) = (up_guaranteed_vector_strides[i] != -1)
                                               ? up_guaranteed_vector_strides[i]
                                               : up_vector_strides[i];
                }
            }

            set_container_subset(vector_lengths, up_dims, up_vector_lengths);
            set_container_subset(vector_strides, up_dims, up_vector_strides);
        });

        constexpr auto top_dims = TopDimensionHiddenIds{};

        return make_tuple(get_container_subset(vector_lengths, top_dims),
                          get_container_subset(vector_strides, top_dims));
    }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("tensor_adaptor{");

        //
        printf("transforms: ");
        print(transforms_);
        printf(", ");

        //
        printf("LowerDimensionHiddenIds: ");
        print(LowerDimensionHiddenIdss{});
        printf(", ");

        //
        printf("UpperDimensionHiddenIds: ");
        print(UpperDimensionHiddenIdss{});
        printf(", ");

        //
        printf("BottomDimensionHiddenIds: ");
        print(BottomDimensionHiddenIds{});
        printf(", ");

        //
        printf("TopDimensionHiddenIds: ");
        print(TopDimensionHiddenIds{});

        printf("}");
    }

    private:
    Transforms transforms_;
    ElementSize element_size_;
};

// Transforms: Tuple<transforms...>
// LowerDimensionOldTopIdss: Tuple<Sequence<...>, ...>
// UpperDimensionNewTopIdss: Tuple<Sequence<...>, ...>
template <typename Transforms, typename LowerDimensionOldTopIdss, typename UpperDimensionNewTopIdss>
CK_TILE_HOST_DEVICE constexpr auto make_single_stage_tensor_adaptor(const Transforms& transforms,
                                                                    LowerDimensionOldTopIdss,
                                                                    UpperDimensionNewTopIdss)
{
    constexpr index_t ntransform = Transforms::size();

    static_assert(LowerDimensionOldTopIdss::size() == ntransform &&
                      UpperDimensionNewTopIdss::size() == ntransform,
                  "wrong!");

    // sanity check on LowerDimensionOldTopIdss and UpperDimensionNewTopIdss
    constexpr auto all_low_dim_old_top_ids = unpack(
        [](auto&&... xs) constexpr { return merge_sequences(xs...); }, LowerDimensionOldTopIdss{});

    constexpr auto all_up_dim_new_top_ids = unpack(
        [](auto&&... xs) constexpr { return merge_sequences(xs...); }, UpperDimensionNewTopIdss{});

    static_assert(is_valid_sequence_map<decltype(all_low_dim_old_top_ids)>::value &&
                      is_valid_sequence_map<decltype(all_up_dim_new_top_ids)>::value,
                  "wrong!");

    constexpr index_t ndim_old_top = all_low_dim_old_top_ids.size();
    constexpr index_t ndim_new_top = all_up_dim_new_top_ids.size();

    // low_dim_hidden_idss
    constexpr auto low_dim_hidden_idss = LowerDimensionOldTopIdss{};

    // up_dim_hidden_idss: shift UpperDimensionNewTopIdss by ndim_bottom
    constexpr auto up_dim_hidden_idss = generate_tuple(
        [](auto itran) { return UpperDimensionNewTopIdss{}[itran] + number<ndim_old_top>{}; },
        number<ntransform>{});

    // bottom_dim_hidden_ids
    constexpr auto bottom_dim_hidden_ids =
        typename arithmetic_sequence_gen<0, ndim_old_top, 1>::type{};

    // top_dim_hidden_ids
    constexpr auto top_dim_hidden_ids =
        typename arithmetic_sequence_gen<0, ndim_new_top, 1>::type{} + number<ndim_old_top>{};

    return tensor_adaptor<remove_cvref_t<Transforms>,
                          remove_cvref_t<decltype(low_dim_hidden_idss)>,
                          remove_cvref_t<decltype(up_dim_hidden_idss)>,
                          remove_cvref_t<decltype(bottom_dim_hidden_ids)>,
                          remove_cvref_t<decltype(top_dim_hidden_ids)>>{transforms};
}

// TODO: How to fix this? It uses an struct instead of lambda because lambda
// doesn't have constructor, and to put it outside the scope where it is used
// (transform_tensor_adaptor) because template cannot be defined inside a function
// template
template <typename NewTransforms>
struct lambda_get_up_dim_num
{
    template <typename I>
    CK_TILE_HOST_DEVICE constexpr auto operator()(I) const
    {
        using Tran = remove_reference_t<decltype(NewTransforms{}.at(I{}))>;
        return number<Tran::get_num_of_upper_dimension()>{};
    }
};

template <typename OldTensorAdaptor,
          typename NewTransforms,
          typename NewLowerDimensionOldTopIdss,
          typename NewUpperDimensionNewTopIdss>
CK_TILE_HOST_DEVICE constexpr auto
transform_tensor_adaptor(const OldTensorAdaptor& old_tensor_adaptor,
                         const NewTransforms& new_transforms,
                         NewLowerDimensionOldTopIdss,
                         NewUpperDimensionNewTopIdss)
{
    // sanity check
    {
        static_assert(NewTransforms::size() == NewLowerDimensionOldTopIdss::size() &&
                          NewTransforms::size() == NewUpperDimensionNewTopIdss::size(),
                      "wrong! inconsitent number of transform");

        constexpr auto all_old_top_ids = unpack([](auto... xs) { return merge_sequences(xs...); },
                                                NewLowerDimensionOldTopIdss{});

        constexpr auto all_new_top_ids = unpack([](auto... xs) { return merge_sequences(xs...); },
                                                NewUpperDimensionNewTopIdss{});

        static_assert(is_valid_sequence_map<decltype(all_old_top_ids)>::value &&
                          is_valid_sequence_map<decltype(all_new_top_ids)>::value,
                      "wrong!");
    }

    // lower dimension's hidden idss
    // convert lower dimension top idss (tuple of sequences) to hidden idss (tuple of
    // sequences)
    constexpr auto low_dim_hidden_idss = transform_tuples(
        // convert lower dimension top ids (a sequence) to hidden ids (a sequence)
        [](auto low_dim_top_ids) constexpr {
            return transform_sequences(
                // convert lower dimension top id to hidden id
                [](auto low_dim_top_id) constexpr {
                    return OldTensorAdaptor::get_top_dimension_hidden_ids()[low_dim_top_id];
                },
                low_dim_top_ids);
        },
        NewLowerDimensionOldTopIdss{});

    constexpr index_t num_new_transform = NewTransforms::size();

    // upper dimension's hidden idss
    constexpr index_t old_hidden_dim_number = OldTensorAdaptor::get_num_of_hidden_dimension();

    constexpr auto up_dim_numbers =
        generate_sequence(lambda_get_up_dim_num<NewTransforms>{}, number<num_new_transform>{});

    constexpr auto up_dim_numbers_scan = merge_sequences(
        sequence<0>{}, inclusive_scan_sequence(up_dim_numbers, plus<index_t>{}, number<0>{}));

    constexpr auto up_dim_hidden_idss = generate_tuple(
        [ old_hidden_dim_number, up_dim_numbers_scan ](auto i) constexpr {
            return
                typename arithmetic_sequence_gen<old_hidden_dim_number + up_dim_numbers_scan[i],
                                                 old_hidden_dim_number + up_dim_numbers_scan[i + 1],
                                                 1>::type{};
        },
        number<num_new_transform>{});

    // new top dimension's hidden ids
    constexpr auto unordered_new_top_dim_hidden_ids = unpack(
        [](auto... xs) constexpr { return merge_sequences(xs...); }, up_dim_hidden_idss);

    constexpr auto new_top_dim_unordered2ordered = unpack(
        [](auto... xs) constexpr { return merge_sequences(xs...); }, NewUpperDimensionNewTopIdss{});

    constexpr auto new_top_dim_hidden_ids =
        unordered_new_top_dim_hidden_ids.reorder_old_to_new(new_top_dim_unordered2ordered);

    // put everything together
    const auto all_transforms =
        container_concat(old_tensor_adaptor.get_transforms(), new_transforms);

    constexpr auto all_low_dim_hidden_idss =
        container_concat(OldTensorAdaptor::get_lower_dimension_hidden_idss(), low_dim_hidden_idss);

    constexpr auto all_up_dim_hidden_idss =
        container_concat(OldTensorAdaptor::get_upper_dimension_hidden_idss(), up_dim_hidden_idss);

    return tensor_adaptor<
        remove_cvref_t<decltype(all_transforms)>,
        remove_cvref_t<decltype(all_low_dim_hidden_idss)>,
        remove_cvref_t<decltype(all_up_dim_hidden_idss)>,
        remove_cvref_t<decltype(OldTensorAdaptor::get_bottom_dimension_hidden_ids())>,
        remove_cvref_t<decltype(new_top_dim_hidden_ids)>>{all_transforms};
}

template <typename TensorAdaptor0, typename TensorAdaptor1>
CK_TILE_HOST_DEVICE constexpr auto chain_tensor_adaptors(const TensorAdaptor0& adaptor0,
                                                         const TensorAdaptor1& adaptor1)
{
    static_assert(TensorAdaptor0::get_num_of_top_dimension() ==
                      TensorAdaptor1::get_num_of_bottom_dimension(),
                  "wrong!");

    // all_transforms = transform0 + transform1
    const auto all_transforms =
        container_concat(adaptor0.get_transforms(), adaptor1.get_transforms());

    // shift
    constexpr index_t adaptor0_max_hidden_id = [&]() {
        index_t adaptor0_max_hidden_id_ = numeric<index_t>::min();

        static_for<0, TensorAdaptor0::get_num_of_transform(), 1>{}([&](auto itran) {
            constexpr index_t ndim_low =
                TensorAdaptor0{}.get_transforms()[itran].get_num_of_lower_dimension();

            static_for<0, ndim_low, 1>{}([&](auto idim_low) {
                adaptor0_max_hidden_id_ =
                    max(adaptor0_max_hidden_id_,
                        TensorAdaptor0::get_lower_dimension_hidden_idss()[itran][idim_low].value);
            });

            constexpr index_t ndim_up =
                TensorAdaptor0{}.get_transforms()[itran].get_num_of_upper_dimension();

            static_for<0, ndim_up, 1>{}([&](auto idim_up) {
                adaptor0_max_hidden_id_ =
                    max(adaptor0_max_hidden_id_,
                        TensorAdaptor0::get_upper_dimension_hidden_idss()[itran][idim_up].value);
            });
        });

        return adaptor0_max_hidden_id_;
    }();

    constexpr index_t adaptor1_min_hidden_id = [&]() {
        index_t adaptor1_min_hidden_id_ = numeric<index_t>::max();

        static_for<0, TensorAdaptor1::get_num_of_transform(), 1>{}([&](auto itran) {
            constexpr index_t ndim_low =
                TensorAdaptor1{}.get_transforms()[itran].get_num_of_lower_dimension();

            // get the min of all lower dimenions, but not bottom dimension (because their id will
            // be matched with top id from adaptor0)
            static_for<0, ndim_low, 1>{}([&](auto idim_low) {
                constexpr index_t low_dim_hidden_id =
                    TensorAdaptor1::get_lower_dimension_hidden_idss()[itran][idim_low].value;

                bool is_bottom_dim = false;
                static_for<0, TensorAdaptor1::get_num_of_bottom_dimension(), 1>{}([&](auto i) {
                    if constexpr(low_dim_hidden_id ==
                                 TensorAdaptor1::get_bottom_dimension_hidden_ids()[i])
                    {
                        is_bottom_dim = true;
                    }
                });

                if(!is_bottom_dim)
                {
                    adaptor1_min_hidden_id_ = min(adaptor1_min_hidden_id_, low_dim_hidden_id);
                }
            });

            constexpr index_t ndim_up =
                TensorAdaptor1{}.get_transforms()[itran].get_num_of_upper_dimension();

            // get the min of all upper dimensions
            static_for<0, ndim_up, 1>{}([&](auto idim_up) {
                adaptor1_min_hidden_id_ =
                    min(adaptor1_min_hidden_id_,
                        TensorAdaptor1::get_upper_dimension_hidden_idss()[itran][idim_up].value);
            });
        });

        return adaptor1_min_hidden_id_;
    }();

    constexpr index_t adaptor1_hidden_id_shift =
        adaptor0_max_hidden_id + 1 - adaptor1_min_hidden_id;

    constexpr index_t ndim_bottom_1 = TensorAdaptor1::get_num_of_bottom_dimension();

    // all_low_dim_hidden_idss =
    // low_dim_hidden_idss_0 + match_hidden_id_for_1(shift_hidden_id_for_1(low_dim_hiden_idss_1))
    constexpr auto low_dim_hidden_idss_1 = generate_tuple(
        // generate sequence of ids for a transform
        [&](auto itran) {
            constexpr auto ndim_low_1 =
                TensorAdaptor1::get_lower_dimension_hidden_idss()[itran].size();

            constexpr auto low_dim_hidden_ids_1 =
                TensorAdaptor1::get_lower_dimension_hidden_idss()[itran];

            // sequence in, sequence out
            constexpr auto low_dim_hidden_ids_1_mod = [&]() constexpr
            {
                auto low_dim_hidden_ids_1_mod_ = to_multi_index(low_dim_hidden_ids_1);

                // shift hidden id so every dim id is unique
                static_for<0, ndim_low_1, 1>{}([&](auto idim_low_1) {
                    low_dim_hidden_ids_1_mod_(idim_low_1) += adaptor1_hidden_id_shift;
                });

                // match hidden id
                static_for<0, ndim_low_1, 1>{}([&](auto idim_low_1) {
                    static_for<0, ndim_bottom_1, 1>{}([&](auto idim_bottom_1) {
                        // if this low dim is bottom dim, then do id matching
                        if constexpr(low_dim_hidden_ids_1[idim_low_1] ==
                                     TensorAdaptor1::get_bottom_dimension_hidden_ids()
                                         [idim_bottom_1])
                        {
                            low_dim_hidden_ids_1_mod_(idim_low_1) =
                                TensorAdaptor0::get_top_dimension_hidden_ids()[idim_bottom_1];
                        }
                    });
                });

                return low_dim_hidden_ids_1_mod_;
            }
            ();

            return generate_sequence_v2(
                [&](auto i) constexpr { return number<low_dim_hidden_ids_1_mod[i]>{}; },
                number<ndim_low_1>{});
        },
        number<TensorAdaptor1::get_num_of_transform()>{});

    constexpr auto all_low_dim_hidden_idss =
        container_concat(TensorAdaptor0::get_lower_dimension_hidden_idss(), low_dim_hidden_idss_1);

    // all_up_dim_hidden_idss =
    // up_dim_hidden_idss_0 + shift_hidden_id_for_1(up_dim_hiden_idss_1)
    constexpr auto up_dim_hidden_idss_1 = generate_tuple(
        // generate sequence of ids for a transform
        [&](auto itran) {
            constexpr auto ndim_up_1 =
                TensorAdaptor1::get_upper_dimension_hidden_idss()[itran].size();

            constexpr auto up_dim_hidden_ids_1 =
                TensorAdaptor1::get_upper_dimension_hidden_idss()[itran];

            // sequence in, constexpr tuple out
            constexpr auto up_dim_hidden_ids_1_mod = [&]() constexpr
            {
                auto up_dim_hidden_ids_1_mod_ = to_multi_index(up_dim_hidden_ids_1);

                // shift hidden id
                static_for<0, ndim_up_1, 1>{}([&](auto idim_up_1) {
                    up_dim_hidden_ids_1_mod_(idim_up_1) += adaptor1_hidden_id_shift;
                });

                return up_dim_hidden_ids_1_mod_;
            }
            ();

            // constexpr tuple to sequence
            return generate_sequence_v2(
                [&](auto i) constexpr { return number<up_dim_hidden_ids_1_mod[i]>{}; },
                number<ndim_up_1>{});
        },
        number<TensorAdaptor1::get_num_of_transform()>{});

    constexpr auto all_up_dim_hidden_idss =
        container_concat(TensorAdaptor0::get_upper_dimension_hidden_idss(), up_dim_hidden_idss_1);

    // bottom_dim_hidden_ids = bottom_dim_hidden_ids_0
    constexpr auto bottom_dim_hidden_ids = TensorAdaptor0::get_bottom_dimension_hidden_ids();

    // top_dim_hidden_ids = shift_hidden_id(top_dim_hidden_ids_1)
    constexpr auto top_dim_hidden_ids =
        TensorAdaptor1::get_top_dimension_hidden_ids() + number<adaptor1_hidden_id_shift>{};

    // put everything together
    return tensor_adaptor<remove_cvref_t<decltype(all_transforms)>,
                          remove_cvref_t<decltype(all_low_dim_hidden_idss)>,
                          remove_cvref_t<decltype(all_up_dim_hidden_idss)>,
                          remove_cvref_t<decltype(bottom_dim_hidden_ids)>,
                          remove_cvref_t<decltype(top_dim_hidden_ids)>>{all_transforms};
}

template <typename X,
          typename... Xs,
          typename std::enable_if<sizeof...(Xs) >= 2, bool>::type = false>
CK_TILE_HOST_DEVICE constexpr auto chain_tensor_adaptors(const X& x, const Xs&... xs)
{
    return chain_tensor_adaptors(x, chain_tensor_adaptors(xs...));
}

} // namespace ck_tile

// Macro function
// construct constexpr tensor_adaptor from constexpr encoding
// encoded_tensor_adaptor are Tuple of following objects:
//    1. encoded transforms (array of fixed size). Each encoded transform is a Tuple of following:
//           1.1 name (coord_transform_enum)
//           1.2 meta data for constructor of the transform
//           1.3 num of lower dimension (index_t)
//           1.4 lower dimension Ids (array of fixed size)
//           1.5 num of up dimension (index_t)
//           1.6 upper dimension Ids (array of fixed size)
//    2. num of transforms (index_t)
//    3. encoded bottom dimension Ids (array of fixed size)
//    4. num of bottom dimension (index_t)
//    5. encoded top dimension Ids (array of fixed size)
//    6. num of top dimension (index_t)
#define CONSTRUCT_TENSOR_ADAPTOR_FROM_ENCODING(encoded_tensor_adaptor)                            \
    [encoded_tensor_adaptor]() {                                                                  \
        using namespace ck_tile;                                                                  \
                                                                                                  \
        constexpr auto encoded_transforms  = encoded_tensor_adaptor.template at<0>();             \
        constexpr index_t num_transform    = encoded_tensor_adaptor.template at<1>();             \
        constexpr auto encoded_bottom_dims = encoded_tensor_adaptor.template at<2>();             \
        constexpr index_t num_bottom_dim   = encoded_tensor_adaptor.template at<3>();             \
        constexpr auto encoded_top_dims    = encoded_tensor_adaptor.template at<4>();             \
        constexpr index_t num_top_dim      = encoded_tensor_adaptor.template at<5>();             \
                                                                                                  \
        constexpr auto trans = [&encoded_transforms]() {                                          \
            return generate_tuple(                                                                \
                [&encoded_transforms](auto i) constexpr {                                         \
                    constexpr auto name        = encoded_transforms[i].template at<0>();          \
                    constexpr auto meta_data   = encoded_transforms[i].template at<1>();          \
                    constexpr auto num_low_dim = encoded_transforms[i].template at<2>();          \
                    constexpr auto num_up_dim  = encoded_transforms[i].template at<4>();          \
                                                                                                  \
                    static_assert(name == coord_transform_enum::pass_through ||                   \
                                      name == coord_transform_enum::pad ||                        \
                                      name == coord_transform_enum::embed ||                      \
                                      name == coord_transform_enum::merge ||                      \
                                      name == coord_transform_enum::unmerge ||                    \
                                      name == coord_transform_enum::replicate,                    \
                                  "");                                                            \
                                                                                                  \
                    if constexpr(name == coord_transform_enum::pass_through)                      \
                    {                                                                             \
                        index_t pos  = 0;                                                         \
                        auto low_len = meta_data.template pop<index_t>(pos);                      \
                                                                                                  \
                        return make_pass_through_transform(low_len);                              \
                    }                                                                             \
                    else if constexpr(name == coord_transform_enum::pad)                          \
                    {                                                                             \
                        index_t pos    = 0;                                                       \
                        auto low_len   = meta_data.template pop<index_t>(pos);                    \
                        auto left_pad  = meta_data.template pop<index_t>(pos);                    \
                        auto right_pad = meta_data.template pop<index_t>(pos);                    \
                                                                                                  \
                        return make_pad_transform(low_len, left_pad, right_pad);                  \
                    }                                                                             \
                    else if constexpr(name == coord_transform_enum::embed)                        \
                    {                                                                             \
                        index_t pos  = 0;                                                         \
                        auto up_lens = meta_data.template pop<array<index_t, num_up_dim>>(pos);   \
                        auto coefficients =                                                       \
                            meta_data.template pop<array<index_t, num_up_dim>>(pos);              \
                                                                                                  \
                        return make_embed_transform(up_lens, coefficients);                       \
                    }                                                                             \
                    else if constexpr(name == coord_transform_enum::merge)                        \
                    {                                                                             \
                        index_t pos   = 0;                                                        \
                        auto low_lens = meta_data.template pop<array<index_t, num_low_dim>>(pos); \
                                                                                                  \
                        return make_merge_transform(low_lens);                                    \
                    }                                                                             \
                    else if constexpr(name == coord_transform_enum::unmerge)                      \
                    {                                                                             \
                        index_t pos  = 0;                                                         \
                        auto up_lens = meta_data.template pop<array<index_t, num_up_dim>>(pos);   \
                                                                                                  \
                        return make_unmerge_transform(up_lens);                                   \
                    }                                                                             \
                    else if constexpr(name == coord_transform_enum::replicate)                    \
                    {                                                                             \
                        index_t pos  = 0;                                                         \
                        auto up_lens = meta_data.template pop<array<index_t, num_up_dim>>(pos);   \
                                                                                                  \
                        return make_replicate_transform(up_lens);                                 \
                    }                                                                             \
                },                                                                                \
                number<num_transform>{});                                                         \
        }();                                                                                      \
                                                                                                  \
        constexpr auto low_dim_idss = [&encoded_transforms, &num_transform]() {                   \
            return generate_tuple(                                                                \
                [&encoded_transforms](auto i) {                                                   \
                    constexpr auto num_low_dim = encoded_transforms[i].template at<2>();          \
                    constexpr auto low_dims    = encoded_transforms[i].template at<3>();          \
                                                                                                  \
                    return TO_SEQUENCE(low_dims, num_low_dim);                                    \
                },                                                                                \
                number<num_transform>());                                                         \
        }();                                                                                      \
                                                                                                  \
        constexpr auto up_dim_idss = [&encoded_transforms, &num_transform] {                      \
            return generate_tuple(                                                                \
                [&encoded_transforms](auto i) {                                                   \
                    constexpr auto num_up_dim = encoded_transforms[i].template at<4>();           \
                    constexpr auto up_dims    = encoded_transforms[i].template at<5>();           \
                                                                                                  \
                    return TO_SEQUENCE(up_dims, num_up_dim);                                      \
                },                                                                                \
                number<num_transform>());                                                         \
        }();                                                                                      \
                                                                                                  \
        constexpr auto bottom_dim_ids = TO_SEQUENCE(encoded_bottom_dims, num_bottom_dim);         \
        constexpr auto top_dim_ids    = TO_SEQUENCE(encoded_top_dims, num_top_dim);               \
                                                                                                  \
        return tensor_adaptor<remove_cvref_t<decltype(trans)>,                                    \
                              remove_cvref_t<decltype(low_dim_idss)>,                             \
                              remove_cvref_t<decltype(up_dim_idss)>,                              \
                              remove_cvref_t<decltype(bottom_dim_ids)>,                           \
                              remove_cvref_t<decltype(top_dim_ids)>>{trans};                      \
    }()

// Macro function
// construct static tensor_adaptor from constexpr encoding
// encoded_tensor_adaptor are Tuple of following objects:
//    1. encoded transforms (array of fixed size). Each encoded transform is a Tuple of following:
//           1.1 name (coord_transform_enum)
//           1.2 meta data for constructor of the transform
//           1.3 num of lower dimension (index_t)
//           1.4 lower dimension Ids (array of fixed size)
//           1.5 num of up dimension (index_t)
//           1.6 upper dimension Ids (array of fixed size)
//    2. num of transforms (index_t)
//    3. encoded bottom dimension Ids (array of fixed size)
//    4. num of bottom dimension (index_t)
//    5. encoded top dimension Ids (array of fixed size)
//    6. num of top dimension (index_t)
#define CONSTRUCT_STATIC_TENSOR_ADAPTOR_FROM_ENCODING(encoded_tensor_adaptor)                      \
    [encoded_tensor_adaptor]() {                                                                   \
        using namespace ck_tile;                                                                   \
                                                                                                   \
        constexpr auto encoded_transforms  = encoded_tensor_adaptor.template at<0>();              \
        constexpr index_t num_transform    = encoded_tensor_adaptor.template at<1>();              \
        constexpr auto encoded_bottom_dims = encoded_tensor_adaptor.template at<2>();              \
        constexpr index_t num_bottom_dim   = encoded_tensor_adaptor.template at<3>();              \
        constexpr auto encoded_top_dims    = encoded_tensor_adaptor.template at<4>();              \
        constexpr index_t num_top_dim      = encoded_tensor_adaptor.template at<5>();              \
                                                                                                   \
        constexpr auto trans = [&encoded_transforms]() {                                           \
            return generate_tuple(                                                                 \
                [&encoded_transforms](auto i) constexpr {                                          \
                    constexpr auto name        = encoded_transforms[i].template at<0>();           \
                    constexpr auto meta_data   = encoded_transforms[i].template at<1>();           \
                    constexpr auto num_low_dim = encoded_transforms[i].template at<2>();           \
                    constexpr auto num_up_dim  = encoded_transforms[i].template at<4>();           \
                                                                                                   \
                    static_assert(name == coord_transform_enum::pass_through ||                    \
                                      name == coord_transform_enum::pad ||                         \
                                      name == coord_transform_enum::embed ||                       \
                                      name == coord_transform_enum::merge ||                       \
                                      name == coord_transform_enum::unmerge ||                     \
                                      name == coord_transform_enum::replicate,                     \
                                  "");                                                             \
                                                                                                   \
                    if constexpr(name == coord_transform_enum::pass_through)                       \
                    {                                                                              \
                        constexpr index_t low_len = meta_data.template get<index_t>(0);            \
                                                                                                   \
                        return make_pass_through_transform(number<low_len>{});                     \
                    }                                                                              \
                    else if constexpr(name == coord_transform_enum::pad)                           \
                    {                                                                              \
                        constexpr index_t low_len = meta_data.template get<index_t>(0);            \
                                                                                                   \
                        constexpr index_t left_pad =                                               \
                            meta_data.template get<index_t>(sizeof(low_len));                      \
                                                                                                   \
                        constexpr index_t right_pad =                                              \
                            meta_data.template pop<index_t>(sizeof(low_len) + sizeof(left_pad));   \
                                                                                                   \
                        return make_pad_transform(                                                 \
                            number<low_len>{}, number<left_pad>{}, number<right_pad>{});           \
                    }                                                                              \
                    else if constexpr(name == coord_transform_enum::embed)                         \
                    {                                                                              \
                        constexpr auto up_lens =                                                   \
                            meta_data.template get<array<index_t, num_up_dim>>(0);                 \
                                                                                                   \
                        constexpr auto coefficients =                                              \
                            meta_data.template get<array<index_t, num_up_dim>>(sizeof(up_lens));   \
                                                                                                   \
                        return make_embed_transform(TO_TUPLE_OF_NUMBER(up_lens, num_up_dim),       \
                                                    TO_TUPLE_OF_NUMBER(coefficients, num_up_dim)); \
                    }                                                                              \
                    else if constexpr(name == coord_transform_enum::merge)                         \
                    {                                                                              \
                        constexpr auto low_lens =                                                  \
                            meta_data.template get<array<index_t, num_low_dim>>(0);                \
                                                                                                   \
                        return make_merge_transform(TO_TUPLE_OF_NUMBER(low_lens, num_low_dim));    \
                    }                                                                              \
                    else if constexpr(name == coord_transform_enum::unmerge)                       \
                    {                                                                              \
                        constexpr auto up_lens =                                                   \
                            meta_data.template get<array<index_t, num_up_dim>>(0);                 \
                                                                                                   \
                        return make_unmerge_transform(TO_TUPLE_OF_NUMBER(up_lens, num_up_dim));    \
                    }                                                                              \
                    else if constexpr(name == coord_transform_enum::replicate)                     \
                    {                                                                              \
                        constexpr auto up_lens =                                                   \
                            meta_data.template get<array<index_t, num_up_dim>>(0);                 \
                                                                                                   \
                        return make_replicate_transform(TO_TUPLE_OF_NUMBER(up_lens, num_up_dim));  \
                    }                                                                              \
                },                                                                                 \
                number<num_transform>{});                                                          \
        }();                                                                                       \
                                                                                                   \
        constexpr auto low_dim_idss = [&encoded_transforms]() {                                    \
            return generate_tuple(                                                                 \
                [&encoded_transforms](auto i) {                                                    \
                    constexpr auto num_low_dim = encoded_transforms[i].template at<2>();           \
                    constexpr auto low_dims    = encoded_transforms[i].template at<3>();           \
                                                                                                   \
                    return TO_SEQUENCE(low_dims, num_low_dim);                                     \
                },                                                                                 \
                number<num_transform>());                                                          \
        }();                                                                                       \
                                                                                                   \
        constexpr auto up_dim_idss = [&encoded_transforms] {                                       \
            return generate_tuple(                                                                 \
                [&encoded_transforms](auto i) {                                                    \
                    constexpr auto num_up_dim = encoded_transforms[i].template at<4>();            \
                    constexpr auto up_dims    = encoded_transforms[i].template at<5>();            \
                                                                                                   \
                    return TO_SEQUENCE(up_dims, num_up_dim);                                       \
                },                                                                                 \
                number<num_transform>());                                                          \
        }();                                                                                       \
                                                                                                   \
        constexpr auto bottom_dim_ids = TO_SEQUENCE(encoded_bottom_dims, num_bottom_dim);          \
        constexpr auto top_dim_ids    = TO_SEQUENCE(encoded_top_dims, num_top_dim);                \
                                                                                                   \
        return tensor_adaptor<remove_cvref_t<decltype(trans)>,                                     \
                              remove_cvref_t<decltype(low_dim_idss)>,                              \
                              remove_cvref_t<decltype(up_dim_idss)>,                               \
                              remove_cvref_t<decltype(bottom_dim_ids)>,                            \
                              remove_cvref_t<decltype(top_dim_ids)>>{trans};                       \
    }()
