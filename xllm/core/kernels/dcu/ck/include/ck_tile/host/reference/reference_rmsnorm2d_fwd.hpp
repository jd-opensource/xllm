// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {

// Note: for simplicity, each functor only care about single M
struct reference_rmsnorm2d_default_epilogue
{
    template <typename OutDataType, typename AccDataType>
    void operator()(int m, HostTensor<OutDataType>& o, const HostTensor<AccDataType>& acc)
    {
        const int N = acc.mDesc.get_lengths()[1];
        for(int n = 0; n < N; ++n)
        {
            o(m, n) = ck_tile::type_convert<OutDataType>(acc(m, n));
        }
    }

    template <typename OutDataType, typename AccDataType>
    auto operator()(int m, const HostTensor<AccDataType>& acc)
    {
        HostTensor<OutDataType> o(acc.get_lengths(), acc.get_strides());
        operator()(m, o, acc);
        return o;
    }
};

template <typename XDataType,
          typename GammaDataType,
          typename ComputeDataType,
          typename YDataType,
          typename InvRmsDataType,
          typename UnquantYDataType,
          typename Epilogue = reference_rmsnorm2d_default_epilogue>
void reference_rmsnorm2d_fwd(const HostTensor<XDataType>& x_m_n,
                             const HostTensor<GammaDataType>& gamma_n,
                             HostTensor<YDataType>& y_m_n,
                             HostTensor<InvRmsDataType>& invRms_m,
                             HostTensor<UnquantYDataType>& unquant_y_m_n,
                             ComputeDataType epsilon,
                             Epilogue epilogue_functor = {})
{
    auto rmsnorm2d_fwd_func = [&](auto m) {
        const int N = x_m_n.mDesc.get_lengths()[1];

        ComputeDataType mean_square = 0;
        ComputeDataType divisor     = 0;

        for(int n = 0; n < N; ++n)
        {
            ComputeDataType x = ck_tile::type_convert<ComputeDataType>(x_m_n(m, n));
            mean_square += x * x;
        }

        mean_square = mean_square / N;
        divisor = ck_tile::type_convert<ComputeDataType>(1) / ck_tile::sqrt(mean_square + epsilon);

        if constexpr(!std::is_same_v<InvRmsDataType, ck_tile::null_type>)
            invRms_m(m) = ck_tile::type_convert<InvRmsDataType>(divisor);

        HostTensor<ComputeDataType> acc(x_m_n.get_lengths(), x_m_n.get_strides());
        for(int n = 0; n < N; ++n)
        {
            ComputeDataType x     = ck_tile::type_convert<ComputeDataType>(x_m_n(m, n));
            ComputeDataType gamma = ck_tile::type_convert<ComputeDataType>(gamma_n(n));
            acc(m, n)             = x * divisor * gamma;
        }

        if constexpr(!std::is_same_v<UnquantYDataType, ck_tile::null_type>)
        {
            epilogue_functor(m, unquant_y_m_n, y_m_n, acc);
        }
        else
        {
            epilogue_functor(m, y_m_n, acc);
        }
    };

    make_ParallelTensorFunctor(rmsnorm2d_fwd_func, invRms_m.mDesc.get_lengths()[0])(
        std::thread::hardware_concurrency());
}
} // namespace ck_tile
