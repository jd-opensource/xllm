// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/algorithm/space_filling_curve.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/ops/gemm/warp/warp_dsreadm_format_dispatcher_v2.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/host/device_prop.hpp"

namespace ck_tile {

/*
 * mls_ds_traits provides adapted ds instruction for given mls atom
 * mls atom is defined in hcu_mls_atom_*.hpp
 */
template <typename MlsAtom, index_t ElemBytes, index_t Alt>
struct mls_ds_traits;

/*
 * tile_window_mls_traits provides tile window detail for given tile shape,
 * element bytes, interleave config, transpose config and target architecture
 */
template <typename TileShape, index_t ElemBytes, index_t Alt, bool Trans, hcu_target_enum HcuArch>
struct tile_window_mls_traits;

} // namespace ck_tile
