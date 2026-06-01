// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {
template <typename XDataType, typename ScaleDataType, typename QXDataType>
CK_TILE_HOST void reference_rowwise_quantization2d(const HostTensor<XDataType>& x_m_n,
                                                   const HostTensor<ScaleDataType>& scale_m,
                                                   HostTensor<QXDataType>& qx_m_n)
{
    auto f = [&](auto m) {
        const int N = x_m_n.mDesc.get_lengths()[1];

        for(int n = 0; n < N; ++n)
        {
            auto v_x = x_m_n(m, n);
            // scale = amax / 127 for int8
            auto v_scale = type_convert<XDataType>(scale_m(m));
            auto v_qx    = v_x / v_scale;
            qx_m_n(m, n) = type_convert<QXDataType>(saturates<QXDataType>{}(v_qx));
        }
    };

    make_ParallelTensorFunctor(f,
                               scale_m.mDesc.get_lengths()[0])(std::thread::hardware_concurrency());
}

} // namespace ck_tile
