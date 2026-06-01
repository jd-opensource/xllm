// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include <sstream>
#include <stdexcept>
#include <hip/hip_runtime.h>

namespace ck_tile {
// To be removed, which really does not tell the location of failed HIP functional call
CK_TILE_HOST void hip_check_error(hipError_t x)
{
    if(x != hipSuccess)
    {
        std::ostringstream ss;
        ss << "HIP runtime error: " << hipGetErrorString(x) << ". " << __FILE__ << ": " << __LINE__
           << "in function: " << __func__;
        throw std::runtime_error(ss.str());
    }
}
} // namespace ck_tile

#define HIP_CHECK_ERROR(retval_or_funcall)                                         \
    do                                                                             \
    {                                                                              \
        hipError_t _tmpVal = retval_or_funcall;                                    \
        if(_tmpVal != hipSuccess)                                                  \
        {                                                                          \
            std::ostringstream ostr;                                               \
            ostr << "HIP Function Failed (" << __FILE__ << "," << __LINE__ << ") " \
                 << hipGetErrorString(_tmpVal);                                    \
            throw std::runtime_error(ostr.str());                                  \
        }                                                                          \
    } while(0)
