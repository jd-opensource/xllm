// SPDX-License-Identifier: MIT
// Copyright (c) 2024, , Inc. All rights reserved.

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
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

#pragma once
namespace ck_tile {

// input a lds store tile, extract some information from it
// used to set m0 value for gfx9 serious
template <typename LdsTileWindow_>
CK_TILE_DEVICE auto get_async_store_smem_info(LdsTileWindow_&& lds_tile)
{
    using LdsTileWindow = remove_cvref_t<LdsTileWindow_>;
    using LdsDataType   = typename LdsTileWindow::DataType;

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

    return make_tuple(m0_init_value, size_per_issue);
}

} // namespace ck_tile
