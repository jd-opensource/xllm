// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include <string>

namespace ck_tile {

enum struct GemmSpec
{
    // Gemm
    Default,
    MPadding,
    NPadding,
    KPadding,
    MNPadding,
    MKPadding,
    NKPadding,
    MNKPadding,
    // Gemm + Gemm
    OPadding,
    MOPadding,
    NOPadding,
    KOPadding,
    MNOPadding,
    MKOPadding,
    NKOPadding,
    MNKOPadding,
};

inline std::string getGemmSpecializationString(const GemmSpec& s)
{
    switch(s)
    {
    case GemmSpec::Default: return "Default";
    case GemmSpec::MPadding: return "MPadding";
    case GemmSpec::NPadding: return "NPadding";
    case GemmSpec::KPadding: return "KPadding";
    case GemmSpec::MNPadding: return "MNPadding";
    case GemmSpec::MKPadding: return "MKPadding";
    case GemmSpec::NKPadding: return "NKPadding";
    case GemmSpec::MNKPadding: return "MNKPadding";
    case GemmSpec::OPadding: return "OPadding";
    case GemmSpec::MOPadding: return "MOPadding";
    case GemmSpec::NOPadding: return "NOPadding";
    case GemmSpec::KOPadding: return "KOPadding";
    case GemmSpec::MNOPadding: return "MNOPadding";
    case GemmSpec::MKOPadding: return "MKOPadding";
    case GemmSpec::NKOPadding: return "NKOPadding";
    case GemmSpec::MNKOPadding: return "MNKOPadding";
    default: return "Unrecognized specialization!";
    }
}

} // namespace ck_tile
