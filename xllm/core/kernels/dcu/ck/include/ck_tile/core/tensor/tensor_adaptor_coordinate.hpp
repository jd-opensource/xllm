// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/container/multi_index.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

template <index_t NDimHidden, typename BottomDimensionHiddenIds, typename TopDimensionHiddenIds>
struct tensor_adaptor_coordinate
{
    static constexpr index_t ndim_bottom_ = BottomDimensionHiddenIds::size();
    static constexpr index_t ndim_top_    = TopDimensionHiddenIds::size();

    using HiddenIndex = multi_index<NDimHidden>;
    using BottomIndex = multi_index<ndim_bottom_>;
    using TopIndex    = multi_index<ndim_top_>;

    public:
    CK_TILE_HOST_DEVICE constexpr tensor_adaptor_coordinate() = default;

    CK_TILE_HOST_DEVICE constexpr tensor_adaptor_coordinate(const HiddenIndex& idx_hidden)
        : idx_hidden_{idx_hidden}
    {
    }

    CK_TILE_HOST_DEVICE constexpr auto get_top_index() const
    {
        return get_container_subset(idx_hidden_, TopDimensionHiddenIds{});
    }

    CK_TILE_HOST_DEVICE constexpr auto get_bottom_index() const
    {
        return get_container_subset(idx_hidden_, BottomDimensionHiddenIds{});
    }

    CK_TILE_HOST_DEVICE constexpr const auto& get_hidden_index() const { return idx_hidden_; }

    CK_TILE_HOST_DEVICE constexpr auto& get_hidden_index() { return idx_hidden_; }

    //
    HiddenIndex idx_hidden_;
};

template <typename Adaptor, typename TopIndex>
CK_TILE_HOST_DEVICE constexpr auto make_tensor_adaptor_coordinate(const Adaptor& adaptor,
                                                                  const TopIndex& idx_top)
{
    static_assert(Adaptor::get_num_of_top_dimension() == TopIndex::size(),
                  "wrong! # of dimension inconsistent");

    constexpr index_t ntransform  = Adaptor::get_num_of_transform();
    constexpr index_t ndim_hidden = Adaptor::get_num_of_hidden_dimension();
    constexpr auto bottom_dim_ids = Adaptor::get_bottom_dimension_hidden_ids();
    constexpr auto top_dim_ids    = Adaptor::get_top_dimension_hidden_ids();

    multi_index<ndim_hidden> idx_hidden;

    // initialize visible index
    set_container_subset(idx_hidden, top_dim_ids, idx_top);

    // calculate hidden index
    static_for<ntransform, 0, -1>{}([&adaptor, &idx_hidden](auto itran_p1) {
        auto itran              = itran_p1 - number<1>{};
        const auto& tran        = adaptor.get_transforms().at(itran);
        constexpr auto dims_low = Adaptor::get_lower_dimension_hidden_idss().at(itran);
        constexpr auto dims_up  = Adaptor::get_upper_dimension_hidden_idss().at(itran);

        const auto idx_up = get_container_subset(idx_hidden, dims_up);

        multi_index<dims_low.size()> idx_low;

        tran.calculate_lower_index(idx_low, idx_up);

        set_container_subset(idx_hidden, dims_low, idx_low);
    });

    return tensor_adaptor_coordinate<ndim_hidden,
                                     remove_cvref_t<decltype(bottom_dim_ids)>,
                                     remove_cvref_t<decltype(top_dim_ids)>>{idx_hidden};
}

template <bool JudgeDoTransforms = true,
          typename Adaptor,
          typename AdaptorCoord,
          typename TopIndex,
          typename BottomIndex>
CK_TILE_HOST_DEVICE constexpr void move_tensor_adaptor_coordinate(const Adaptor& adaptor,
                                                                  AdaptorCoord& coord,
                                                                  const TopIndex& idx_diff_top,
                                                                  BottomIndex& idx_diff_bottom)
{
    constexpr index_t ndim_hidden = Adaptor::get_num_of_hidden_dimension();
    constexpr index_t ndim_top    = Adaptor::get_num_of_top_dimension();
    //  constexpr index_t ndim_bottom = Adaptor::get_num_of_bottom_dimension();
    constexpr index_t ntransform = Adaptor::get_num_of_transform();

    //  static_assert(TopIndex::size() == ndim_top && BottomIndex::size() == ndim_bottom, "");

    // judge whether calculation of lower diff is needed for each transform
    // use index_t for boolean type
    auto do_transforms = make_zero_multi_index<ntransform>();

    if constexpr(JudgeDoTransforms)
    {
        auto is_non_zero_diff = make_zero_multi_index<ndim_hidden>();

        // decide do_transform by checkout non-zero index diff components
        multi_index<ndim_top> non_zero_diff_pick_top;

        static_for<0, ndim_top, 1>{}(
            [&](auto i) { non_zero_diff_pick_top(i) = (idx_diff_top[i] != 0); });

        set_container_subset(
            is_non_zero_diff, Adaptor::get_top_dimension_hidden_ids(), non_zero_diff_pick_top);

        static_for<ntransform - 1, -1, -1>{}([&](auto itran) {
            constexpr auto dims_low = Adaptor::get_lower_dimension_hidden_idss().at(itran);
            constexpr auto dims_up  = Adaptor::get_upper_dimension_hidden_idss().at(itran);

            const auto non_zero_diff_pick_up = get_container_subset(is_non_zero_diff, dims_up);

            multi_index<dims_low.size()> non_zero_diff_pick_low;

            // if any of upper index diff components is non-zero, then
            //   1) Need to do this transform
            //   2) all components of lower index diff will assume to be non-zero and need to be
            //   computed
            const bool idx_diff_up_has_non_zero = container_reduce(
                non_zero_diff_pick_up, [](auto a, auto b) constexpr { return a or b; }, false);

            do_transforms(itran) = idx_diff_up_has_non_zero;

            static_for<0, dims_low.size(), 1>{}(
                [&](auto i) { non_zero_diff_pick_low(i) = idx_diff_up_has_non_zero; });

            set_container_subset(is_non_zero_diff, dims_low, non_zero_diff_pick_low);
        });
    }
    else
    {
        static_for<ntransform - 1, -1, -1>{}([&](auto itran) { do_transforms(itran) = 1; });
    }

    // this is what needs to be calculated
    auto idx_diff_hidden = make_zero_multi_index<ndim_hidden>();

    // initialize top index diff
    set_container_subset(idx_diff_hidden, Adaptor::get_top_dimension_hidden_ids(), idx_diff_top);

    // this is what needs to be updated
    auto& idx_hidden = coord.get_hidden_index();

    // update top index
    auto idx_hidden_pick_top =
        get_container_subset(idx_hidden, Adaptor::get_top_dimension_hidden_ids());

    idx_hidden_pick_top += idx_diff_top;

    set_container_subset(idx_hidden, Adaptor::get_top_dimension_hidden_ids(), idx_hidden_pick_top);

    // update rest of hidden index
    static_for<ntransform - 1, -1, -1>{}([&](auto itran) {
        if(do_transforms[itran])
        {
            const auto& tran        = adaptor.get_transforms().at(itran);
            constexpr auto dims_low = Adaptor::get_lower_dimension_hidden_idss().at(itran);
            constexpr auto dims_up  = Adaptor::get_upper_dimension_hidden_idss().at(itran);

            const auto idx_up_new  = get_container_subset(idx_hidden, dims_up);
            auto idx_low           = get_container_subset(idx_hidden, dims_low);
            const auto idx_diff_up = get_container_subset(idx_diff_hidden, dims_up);

            multi_index<dims_low.size()> idx_diff_low;

            tran.update_lower_index(idx_diff_low, idx_diff_up, idx_low, idx_up_new);

            set_container_subset(idx_diff_hidden, dims_low, idx_diff_low);
            set_container_subset(idx_hidden, dims_low, idx_low);
        }
    });

    // set bottom index diff
    idx_diff_bottom =
        get_container_subset(idx_diff_hidden, Adaptor::get_bottom_dimension_hidden_ids());
}

template <bool JudgeDoTransforms = true, typename Adaptor, typename AdaptorCoord, typename TopIndex>
CK_TILE_HOST_DEVICE constexpr void move_tensor_adaptor_coordinate(const Adaptor& adaptor,
                                                                  AdaptorCoord& coord,
                                                                  const TopIndex& idx_diff_top)
{
    constexpr index_t ndim_bottom = Adaptor::get_num_of_bottom_dimension();

    multi_index<ndim_bottom> tmp;

    move_tensor_adaptor_coordinate<JudgeDoTransforms>(adaptor, coord, idx_diff_top, tmp);
}

template <typename Adaptor, typename AdaptorCoord>
CK_TILE_HOST_DEVICE constexpr bool
adaptor_coordinate_is_valid_assuming_top_index_is_valid(const Adaptor& adaptor,
                                                        const AdaptorCoord& coord)
{
    bool valid = true;

    constexpr index_t ntransform = Adaptor::get_num_of_transform();

    const auto& idx_hidden = coord.get_hidden_index();

    static_for<ntransform - 1, -1, -1>{}([&adaptor, &idx_hidden, &valid](auto itran) {
        const auto tran = adaptor.get_transforms().at(itran);

        // check validity, only if current transformation does not always has a valid mapping
        if constexpr(!decltype(tran)::is_valid_upper_index_always_mapped_to_valid_lower_index())
        {
            const auto idx_up = get_container_subset(
                idx_hidden, Adaptor::get_upper_dimension_hidden_idss().at(itran));

            // Comment: using valid = valid && .. will result in weird control flow in ISA
            valid &= tran.is_valid_upper_index_mapped_to_valid_lower_index(idx_up);
        }
    });

    return valid;
}

template <typename Adaptor, typename AdpatorCoord>
CK_TILE_HOST_DEVICE constexpr bool adaptor_coordinate_is_valid(const Adaptor& adaptor,
                                                               const AdpatorCoord& coord)
{
    // check top index
    const auto& idx_top = coord.get_top_index();

    bool is_top_index_valid = true;

    static_for<0, Adaptor::get_num_of_dimension(), 1>{}(
        [&is_top_index_valid, &idx_top, &adaptor](auto i) {
            is_top_index_valid =
                is_top_index_valid && (idx_top[i] >= 0 && idx_top[i] < adaptor.get_length(i));
        });

    // check other hidden index
    return is_top_index_valid &&
           adaptor_coordinate_is_valid_assuming_top_index_is_valid(adaptor, coord);
}

} // namespace ck_tile
