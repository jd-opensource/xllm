// SPDX-License-Identifier: MIT
// Copyright (c) 2024, , Inc. All rights reserved.

#pragma once

#include <ostream>

#include "ck_tile/core.hpp"

namespace ck_tile {

enum struct GemmPipelineScheduler
{
    Default,
    Intrawave,
    Interwave,
};

enum struct TailNumber
{
    // Single / Double buffer pipeline
    Odd,
    Even,

    // Long prefetch pipeline, up to 8
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,

    // Unroll stages > Prefetch stages, number of loop is multiple of unroll stages
    Empty,
    // Unroll stages <= Prefetch stages, number of loop is multiple of unroll stages add
    // prefetchstages
    Full,
};

} // namespace ck_tile

inline std::ostream& operator<<(std::ostream& os, const ck_tile::GemmPipelineScheduler& s)
{
    switch(s)
    {
    case ck_tile::GemmPipelineScheduler::Default: os << "Default"; break;
    case ck_tile::GemmPipelineScheduler::Intrawave: os << "Intrawave"; break;
    case ck_tile::GemmPipelineScheduler::Interwave: os << "Interwave"; break;
    default: os << "";
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const ck_tile::TailNumber& s)
{
    switch(s)
    {
    case ck_tile::TailNumber::Odd: os << "Odd"; break;
    case ck_tile::TailNumber::Even: os << "Even"; break;
    case ck_tile::TailNumber::One: os << "One"; break;
    case ck_tile::TailNumber::Two: os << "Two"; break;
    case ck_tile::TailNumber::Three: os << "Three"; break;
    case ck_tile::TailNumber::Four: os << "Four"; break;
    case ck_tile::TailNumber::Five: os << "Five"; break;
    case ck_tile::TailNumber::Six: os << "Six"; break;
    case ck_tile::TailNumber::Seven: os << "Seven"; break;
    case ck_tile::TailNumber::Empty: os << "Empty"; break;
    case ck_tile::TailNumber::Full: os << "Full"; break;
    default: os << "";
    }
    return os;
}
