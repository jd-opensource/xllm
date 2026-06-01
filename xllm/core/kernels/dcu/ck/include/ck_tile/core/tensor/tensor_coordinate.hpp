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

template <index_t NDimHidden, typename TopDimensionHiddenIds>
struct tensor_coordinate
    : public tensor_adaptor_coordinate<NDimHidden, sequence<0>, TopDimensionHiddenIds>
{
    using Base = tensor_adaptor_coordinate<NDimHidden, sequence<0>, TopDimensionHiddenIds>;

    // TODO make these private
    static constexpr index_t ndim_top_ = TopDimensionHiddenIds::size();

    using HiddenIndex = multi_index<NDimHidden>;
    using TopIndex    = multi_index<ndim_top_>;

    public:
    CK_TILE_HOST_DEVICE constexpr tensor_coordinate() = default;

    CK_TILE_HOST_DEVICE constexpr tensor_coordinate(const HiddenIndex& idx_hidden)
        : Base{idx_hidden}
    {
    }

    // construct from TensorAdaptorCoordinte base class
    CK_TILE_HOST_DEVICE constexpr tensor_coordinate(const Base& adaptor_coord) : Base{adaptor_coord}
    {
    }

    CK_TILE_HOST_DEVICE constexpr auto get_index() const { return Base::get_top_index(); }

    CK_TILE_HOST_DEVICE constexpr index_t get_offset() const
    {
        return Base::get_bottom_index()[number<0>{}];
    }

    CK_TILE_HOST_DEVICE constexpr const auto& get_hidden_index() const
    {
        return Base::get_hidden_index();
    }

    CK_TILE_HOST_DEVICE auto& get_hidden_index() { return Base::get_hidden_index(); }
};

template <typename TensorDesc, typename TopIndex>
CK_TILE_HOST_DEVICE constexpr auto make_tensor_coordinate(const TensorDesc& tensor_desc,
                                                          const TopIndex& idx_top)
{
    const auto adaptor_coord = make_tensor_adaptor_coordinate(tensor_desc, idx_top);

    return tensor_coordinate<TensorDesc::get_num_of_hidden_dimension(),
                             remove_cvref_t<decltype(TensorDesc::get_top_dimension_hidden_ids())>>{
        adaptor_coord};
}

template <bool JudgeDoTransforms = true, typename TensorDesc, typename TensorCoord, typename Index>
CK_TILE_HOST_DEVICE constexpr void
move_tensor_coordinate(const TensorDesc& tensor_desc, TensorCoord& coord, const Index& coord_step)
{
    move_tensor_adaptor_coordinate(tensor_desc, coord, coord_step);
}

template <typename TensorDesc, typename TensorCoord>
CK_TILE_HOST_DEVICE constexpr bool
coordinate_has_valid_offset_assuming_top_index_is_valid(const TensorDesc& tensor_desc,
                                                        const TensorCoord& coord)
{
    return adaptor_coordinate_is_valid_assuming_top_index_is_valid(tensor_desc, coord);
}

template <typename TensorDesc, typename TensorCoord>
CK_TILE_HOST_DEVICE constexpr bool coordinate_has_valid_offset(const TensorDesc& tensor_desc,
                                                               const TensorCoord& coord)
{
    return adaptor_coordinate_is_valid(tensor_desc, coord);
}

} // namespace ck_tile
