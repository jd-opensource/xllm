// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include <string>
#include "ck_tile/core/container/sequence.hpp"

namespace ck_tile {

enum struct ConvFwdSpecEnum
{
    F1x1_S1_D1_P0,
    F1x1_S2_D1_P0,
    F3x3_S1_D1_P1,
    F3x3_S2_D1_P1,
    F5x5_S1_D1_P2,
    F5x5_S2_D1_P2,
    F7x7_S2_D1_P3,
};

namespace detail {

template <ConvFwdSpecEnum Spec>
struct ConvFwdSpecDetail;

template <>
struct ConvFwdSpecDetail<ConvFwdSpecEnum::F1x1_S1_D1_P0>
{
    static constexpr auto NDimSpatial   = 2;
    static constexpr auto FilterLengths = sequence<1, 1>{};
    static constexpr auto Strides       = sequence<1, 1>{};
    static constexpr auto Dilations     = sequence<1, 1>{};
    static constexpr auto Pads          = sequence<0, 0, 0, 0>{};
};

template <>
struct ConvFwdSpecDetail<ConvFwdSpecEnum::F1x1_S2_D1_P0>
{
    static constexpr auto NDimSpatial   = 2;
    static constexpr auto FilterLengths = sequence<1, 1>{};
    static constexpr auto Strides       = sequence<2, 2>{};
    static constexpr auto Dilations     = sequence<1, 1>{};
    static constexpr auto Pads          = sequence<0, 0, 0, 0>{};
};

template <>
struct ConvFwdSpecDetail<ConvFwdSpecEnum::F3x3_S1_D1_P1>
{
    static constexpr auto NDimSpatial   = 2;
    static constexpr auto FilterLengths = sequence<3, 3>{};
    static constexpr auto Strides       = sequence<1, 1>{};
    static constexpr auto Dilations     = sequence<1, 1>{};
    static constexpr auto Pads          = sequence<1, 1, 1, 1>{};
};

template <>
struct ConvFwdSpecDetail<ConvFwdSpecEnum::F3x3_S2_D1_P1>
{
    static constexpr auto NDimSpatial   = 2;
    static constexpr auto FilterLengths = sequence<3, 3>{};
    static constexpr auto Strides       = sequence<2, 2>{};
    static constexpr auto Dilations     = sequence<1, 1>{};
    static constexpr auto Pads          = sequence<1, 1, 1, 1>{};
};

template <>
struct ConvFwdSpecDetail<ConvFwdSpecEnum::F5x5_S1_D1_P2>
{
    static constexpr auto NDimSpatial   = 2;
    static constexpr auto FilterLengths = sequence<5, 5>{};
    static constexpr auto Strides       = sequence<1, 1>{};
    static constexpr auto Dilations     = sequence<1, 1>{};
    static constexpr auto Pads          = sequence<2, 2, 2, 2>{};
};

template <>
struct ConvFwdSpecDetail<ConvFwdSpecEnum::F5x5_S2_D1_P2>
{
    static constexpr auto NDimSpatial   = 2;
    static constexpr auto FilterLengths = sequence<5, 5>{};
    static constexpr auto Strides       = sequence<2, 2>{};
    static constexpr auto Dilations     = sequence<1, 1>{};
    static constexpr auto Pads          = sequence<2, 2, 2, 2>{};
};

template <>
struct ConvFwdSpecDetail<ConvFwdSpecEnum::F7x7_S2_D1_P3>
{
    static constexpr auto NDimSpatial   = 2;
    static constexpr auto FilterLengths = sequence<7, 7>{};
    static constexpr auto Strides       = sequence<2, 2>{};
    static constexpr auto Dilations     = sequence<1, 1>{};
    static constexpr auto Pads          = sequence<3, 3, 3, 3>{};
};

}; // namespace detail

inline std::string GetConvFwdSpecString(const ConvFwdSpecEnum& s)
{
    switch(s)
    {
    case ConvFwdSpecEnum::F1x1_S1_D1_P0: return "f1x1s1d1p0";
    case ConvFwdSpecEnum::F1x1_S2_D1_P0: return "f1x1s2d1p0";
    case ConvFwdSpecEnum::F3x3_S1_D1_P1: return "f3x3s1d1p1";
    case ConvFwdSpecEnum::F3x3_S2_D1_P1: return "f3x3s2d1p1";
    case ConvFwdSpecEnum::F5x5_S1_D1_P2: return "f5x5s1d1p2";
    case ConvFwdSpecEnum::F5x5_S2_D1_P2: return "f5x5s2d1p2";
    case ConvFwdSpecEnum::F7x7_S2_D1_P3: return "f7x7s2d1p3";
    default: return "Unrecognized specialization !";
    }
}

template <ConvFwdSpecEnum Spec, typename ConvProblem>
bool IsApplicableConvProblem(const ConvProblem& problem)
{
    using Detail = detail::ConvFwdSpecDetail<Spec>;

    constexpr auto NDimSpatial = Detail::NDimSpatial;

    if constexpr(NDimSpatial == 2)
    {
        return (problem.weight.lengths[3] == Detail::FilterLengths[0] &&
                problem.weight.lengths[4] == Detail::FilterLengths[1] &&
                problem.conv_filter_strides[0] == Detail::Strides[0] &&
                problem.conv_filter_strides[1] == Detail::Strides[1] &&
                problem.conv_filter_dilations[0] == Detail::Dilations[0] &&
                problem.conv_filter_dilations[1] == Detail::Dilations[1] &&
                problem.input_left_pads[0] == Detail::Pads[0] &&
                problem.input_left_pads[1] == Detail::Pads[1] &&
                problem.input_right_pads[0] == Detail::Pads[2] &&
                problem.input_right_pads[1] == Detail::Pads[3]);
    }
    else
    {
        static_assert(false, "Unsupported NDimSpatial");
    }
}

} // namespace ck_tile
