// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/algorithm/space_filling_curve.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/tensor/tile_window_mls_traits.hpp"
#include "ck_tile/core/tensor/tile_window_mls_gfx946.hpp"
#include "ck_tile/ops/gemm/warp/warp_dsreadm_format_dispatcher_v2.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/host/device_prop.hpp"

namespace ck_tile {

/*
 * mls_ds_traits provides adapted ds instruction for given mls atom
 * mls atom is defined in hcu_mls_atom_*.hpp
 */

// TODO: confirm this
template <index_t Alt>
struct mls_ds_traits<gfx946_mls_32x16_b16, 2, Alt>
{
    using DsFormatInst = WarpDsreadmFormatDispatcherV2<2, 32, 16, Alt, false>;
};

template <>
struct mls_ds_traits<gfx946_mls_16x32_trans_b16, 2, 1>
{
    using DsFormatInst = WarpDsreadmFormatDispatcherV2<2, 32, 16, 1, true>;
};

template <>
struct mls_ds_traits<gfx946_mls_16x32_trans_b16, 2, 2>
{
    using DsFormatInst = WarpDsreadmFormatDispatcherV2<2, 16, 32, 2, true>;
};

template <index_t Alt>
struct mls_ds_traits<gfx946_mls_32x32_b16, 2, Alt>
{
    using DsFormatInst = WarpDsreadmFormatDispatcherV2<2, 32, 16, Alt, false>;
};

template <>
struct mls_ds_traits<gfx946_mls_32x32_trans_b16, 2, 1>
{
    using DsFormatInst = WarpDsreadmFormatDispatcherV2<2, 32, 16, 1, true>;
};

template <>
struct mls_ds_traits<gfx946_mls_32x32_trans_b16, 2, 2>
{
    using DsFormatInst = WarpDsreadmFormatDispatcherV2<2, 16, 32, 2, true>;
};

template <index_t Alt>
struct mls_ds_traits<gfx946_mls_64x16_b16, 2, Alt>
{
    using DsFormatInst = WarpDsreadmFormatDispatcherV2<2, 32, 16, Alt, false>;
};

template <>
struct mls_ds_traits<gfx946_mls_16x64_trans_b16, 2, 1>
{
    using DsFormatInst = WarpDsreadmFormatDispatcherV2<2, 32, 16, 1, true>;
};

template <>
struct mls_ds_traits<gfx946_mls_16x64_trans_b16, 2, 2>
{
    using DsFormatInst = WarpDsreadmFormatDispatcherV2<2, 16, 32, 2, true>;
};

/*
 * tile_window_mls_traits provides tile window detail for given tile shape,
 * element bytes, interleave config, transpose config and target architecture
 */
template <index_t Alt>
struct tile_window_mls_traits<sequence<256, 64>, 2, Alt, true, hcu_target_enum::gfx946>
{
    using Detail = tile_window_mls_gfx946_256x64_trans_b16<Alt>;
};

template <index_t Alt>
struct tile_window_mls_traits<sequence<128, 64>, 2, Alt, true, hcu_target_enum::gfx946>
{
    using Detail = tile_window_mls_gfx946_128x64_trans_b16<Alt>;
};

template <index_t Alt>
struct tile_window_mls_traits<sequence<64, 64>, 2, Alt, true, hcu_target_enum::gfx946>
{
    using Detail = tile_window_mls_gfx946_64x64_trans_b16<Alt>;
};

template <index_t Alt>
struct tile_window_mls_traits<sequence<256, 32>, 2, Alt, true, hcu_target_enum::gfx946>
{
    using Detail = tile_window_mls_gfx946_256x32_trans_b16<Alt>;
};

template <index_t Alt>
struct tile_window_mls_traits<sequence<128, 32>, 2, Alt, true, hcu_target_enum::gfx946>
{
    using Detail = tile_window_mls_gfx946_128x32_trans_b16<Alt>;
};

template <index_t Alt>
struct tile_window_mls_traits<sequence<64, 32>, 2, Alt, true, hcu_target_enum::gfx946>
{
    using Detail = tile_window_mls_gfx946_64x32_trans_b16<Alt>;
};

} // namespace ck_tile
