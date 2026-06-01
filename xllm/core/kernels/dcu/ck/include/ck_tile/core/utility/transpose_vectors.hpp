// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/thread_buffer.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"
#include "ck_tile/core/utility/functional.hpp"

namespace ck_tile {

// S: scalar type (or it can be non-scalar type)
// NX: # of vector before transpose
// NY: # of vector after transpose
// we got [NX, NY] amount of S data to be transposed into [NY, NX] amount of S data
template <typename S_, index_t NX, index_t NY>
struct transpose_vectors
{
    static constexpr index_t s_per_x = NY;
    static constexpr index_t s_per_y = NX;

    using S = remove_cvref_t<S_>;

    using VX = array<S, s_per_x>;
    using VY = array<S, s_per_y>;

    CK_TILE_DEVICE void operator()(const thread_buffer<VX, NX>& vx_tuple,
                                   thread_buffer<VY, NY>& vy_tuple)
    {
        constexpr auto I1 = number<1>{};
        constexpr auto I2 = number<2>{};
        constexpr auto I3 = number<3>{};
        constexpr auto I4 = number<4>{};

        if constexpr(sizeof(S) == 2)
        {
            static_assert((NX % 2 == 0 && NY % 2 == 0), "wrong!");

            using S2 = array<S, 2>; // typename array<S, 2>::type;

            // loop over 2x2 tile and transpose data from vx_tuple into vy_tuple
            static_for<0, NY, 2>{}([&](auto iy) {
                static_for<0, NX, 2>{}([&](auto ix) {
                    // 2 16bitx2 data from vx_tuple to be transposed
                    const int32_t x_s2_0 =
                        bit_cast<int32_t>(vx_tuple[ix].template get_as<S2>()[iy / I2]);
                    const int32_t x_s2_1 =
                        bit_cast<int32_t>(vx_tuple[ix + I1].template get_as<S2>()[iy / I2]);

                    constexpr int32_t m0 = 0x05040100;
                    constexpr int32_t m1 = 0x07060302;

                    // transpose 2x2 16bit
                    // ex: v_perm_b32(0x 11 22 33 44, 0x 55 66 77 88, 0x 05 01 04 00) -> 0x33774488
                    //                   -- -- -- --     -- -- -- --      -  -  -  -
                    //             index  7  6  5  4      3  2  1  0     33 77 44 88
                    // index is reversed because of little endianness (least significant bits first)
                    const int32_t y_s2_0 = __builtin_amdgcn_perm(x_s2_1, x_s2_0, m0);
                    const int32_t y_s2_1 = __builtin_amdgcn_perm(x_s2_1, x_s2_0, m1);

                    // 2 16bitx2 data after transposed
                    vy_tuple(iy).template get_as<S2>()(ix / I2)      = bit_cast<S2>(y_s2_0);
                    vy_tuple(iy + I1).template get_as<S2>()(ix / I2) = bit_cast<S2>(y_s2_1);
                });
            });
        }
        else if constexpr(sizeof(S) == 1)
        {
            static_assert((NX % 4 == 0 && NY % 4 == 0), "wrong!");

            using S4 = array<S, 4>; // typename array<S, 4>::type;

            // loop over 4x4 tile and transpose data from vx_tuple into vy_tuple
            static_for<0, NY, 4>{}([&](auto iy) {
                static_for<0, NX, 4>{}([&](auto ix) {
                    // 4 int8x4 data from vx_tuple
                    const int32_t x_s4_0 =
                        bit_cast<int32_t>(vx_tuple[ix].template get_as<S4>()[iy / I4]);
                    const int32_t x_s4_1 =
                        bit_cast<int32_t>(vx_tuple[ix + I1].template get_as<S4>()[iy / I4]);
                    const int32_t x_s4_2 =
                        bit_cast<int32_t>(vx_tuple[ix + I2].template get_as<S4>()[iy / I4]);
                    const int32_t x_s4_3 =
                        bit_cast<int32_t>(vx_tuple[ix + I3].template get_as<S4>()[iy / I4]);

                    // transpose
                    int32_t t_s4_0, t_s4_1;
                    int32_t y_s4_0, y_s4_1, y_s4_2, y_s4_3;

                    constexpr int32_t m0 = 0x05010400;
                    constexpr int32_t m1 = 0x05040100;
                    constexpr int32_t m2 = 0x07060302;
                    constexpr int32_t m3 = 0x07030602;

                    // ex: v_perm_b32(0x 11 22 33 44, 0x 55 66 77 88, 0x 05 01 04 00) -> 0x33774488
                    //                   -- -- -- --     -- -- -- --      -  -  -  -
                    //             index  7  6  5  4      3  2  1  0     33 77 44 88
                    // index is reversed because of little endianness (least significant bits first)
                    t_s4_0 = __builtin_amdgcn_perm(x_s4_1, x_s4_0, m0);
                    t_s4_1 = __builtin_amdgcn_perm(x_s4_3, x_s4_2, m0);
                    y_s4_0 = __builtin_amdgcn_perm(t_s4_1, t_s4_0, m1);
                    y_s4_1 = __builtin_amdgcn_perm(t_s4_1, t_s4_0, m2);
                    t_s4_0 = __builtin_amdgcn_perm(x_s4_1, x_s4_0, m3);
                    t_s4_1 = __builtin_amdgcn_perm(x_s4_3, x_s4_2, m3);
                    y_s4_2 = __builtin_amdgcn_perm(t_s4_1, t_s4_0, m1);
                    y_s4_3 = __builtin_amdgcn_perm(t_s4_1, t_s4_0, m2);

                    // 4 int8x4 data from vy_tuple
                    vy_tuple(iy).template get_as<S4>()(ix / I4)      = bit_cast<S4>(y_s4_0);
                    vy_tuple(iy + I1).template get_as<S4>()(ix / I4) = bit_cast<S4>(y_s4_1);
                    vy_tuple(iy + I2).template get_as<S4>()(ix / I4) = bit_cast<S4>(y_s4_2);
                    vy_tuple(iy + I3).template get_as<S4>()(ix / I4) = bit_cast<S4>(y_s4_3);
                });
            });
        }
        else
        {
            static_assert(false, "not implemented");
        }
    }
};

} // namespace ck_tile
