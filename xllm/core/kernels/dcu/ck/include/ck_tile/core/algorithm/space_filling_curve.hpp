// SPDX-License-Identifier: MIT
// Copyright (c) 2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/multi_index.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/container/statically_indexed_array.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

template <typename TensorLengths,
          typename DimAccessOrder,
          typename ScalarsPerAccess,
          bool SnakeCurved = true> // # of scalars per access in each dimension
struct space_filling_curve
{
    static constexpr index_t TensorSize =
        reduce_on_sequence(TensorLengths{}, multiplies{}, number<1>{});
    static_assert(0 < TensorSize,
                  "space_filling_curve should be used to access a non-empty tensor");

    static constexpr index_t nDim = TensorLengths::size();

    using Index = multi_index<nDim>;

    static constexpr index_t ScalarPerVector =
        reduce_on_sequence(ScalarsPerAccess{}, multiplies{}, number<1>{});

    static constexpr auto access_lengths   = TensorLengths{} / ScalarsPerAccess{};
    static constexpr auto dim_access_order = DimAccessOrder{};
    static constexpr auto ordered_access_lengths =
        container_reorder_given_new2old(access_lengths, dim_access_order);

    static constexpr auto to_index_adaptor = make_single_stage_tensor_adaptor(
        make_tuple(make_merge_transform(ordered_access_lengths)),
        make_tuple(typename arithmetic_sequence_gen<0, nDim, 1>::type{}),
        make_tuple(sequence<0>{}));

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};

    CK_TILE_HOST_DEVICE static constexpr index_t get_num_of_access()
    {
        static_assert(TensorLengths::size() == ScalarsPerAccess::size());
        static_assert(TensorLengths{} % ScalarsPerAccess{} ==
                      typename uniform_sequence_gen<TensorLengths::size(), 0>::type{});

        return reduce_on_sequence(TensorLengths{}, multiplies{}, number<1>{}) / ScalarPerVector;
    }

    template <index_t AccessIdx1dHead, index_t AccessIdx1dTail>
    static CK_TILE_HOST_DEVICE constexpr auto get_step_between(number<AccessIdx1dHead>,
                                                               number<AccessIdx1dTail>)
    {
        static_assert(AccessIdx1dHead >= 0 && AccessIdx1dHead < get_num_of_access(),
                      "1D index out of range");
        static_assert(AccessIdx1dTail >= 0 && AccessIdx1dTail < get_num_of_access(),
                      "1D index out of range");

        constexpr auto idx_head = get_index(number<AccessIdx1dHead>{});
        constexpr auto idx_tail = get_index(number<AccessIdx1dTail>{});
        return idx_tail - idx_head;
    }

    template <index_t AccessIdx1d>
    static CK_TILE_HOST_DEVICE constexpr auto get_forward_step(number<AccessIdx1d>)
    {
        static_assert(AccessIdx1d < get_num_of_access(), "1D index should be larger than 0");
        return get_step_between(number<AccessIdx1d>{}, number<AccessIdx1d + 1>{});
    }

    template <index_t AccessIdx1d>
    static CK_TILE_HOST_DEVICE constexpr auto get_backward_step(number<AccessIdx1d>)
    {
        static_assert(AccessIdx1d > 0, "1D index should be larger than 0");

        return get_step_between(number<AccessIdx1d>{}, number<AccessIdx1d - 1>{});
    }

    // Do not use this function directly!
    // TODO: can refactor into generic lambda in the future
    template <index_t AccessIdx1d>
    static CK_TILE_HOST_DEVICE constexpr Index _get_index(number<AccessIdx1d>)
    {
#if 0
        /*
         * \todo: tensor_adaptor::calculate_bottom_index does NOT return constexpr as expected.
         */
        constexpr auto ordered_access_idx = to_index_adaptor.calculate_bottom_index(make_multi_index(number<AccessIdx1d>{}));
#else

        constexpr auto access_strides =
            container_reverse_exclusive_scan(ordered_access_lengths, multiplies{}, number<1>{});

        constexpr auto idx_1d = number<AccessIdx1d>{};
        // Given tensor strides \p access_lengths, and 1D index of space-filling-curve, compute the
        // idim-th element of multidimensional index.
        // All constexpr variables have to be captured by VALUE.
        constexpr auto compute_index = [idx_1d, access_strides](auto idim) constexpr {
            constexpr auto compute_index_impl = [idx_1d, access_strides](auto jdim) constexpr {
                auto res = idx_1d.value;
                auto id  = 0;

                static_for<0, jdim.value + 1, 1>{}([&](auto kdim) {
                    id = res / access_strides[kdim].value;
                    res -= id * access_strides[kdim].value;
                });

                return id;
            };

            constexpr auto id = compute_index_impl(idim);
            return number<id>{};
        };

        constexpr auto ordered_access_idx = generate_tuple(compute_index, number<nDim>{});
#endif
        constexpr auto forward_sweep = [&]() {
            statically_indexed_array<bool, nDim> forward_sweep_;

            forward_sweep_(I0) = true;

            static_for<1, nDim, 1>{}([&](auto idim) {
                index_t tmp = ordered_access_idx[I0];

                static_for<1, idim, 1>{}(
                    [&](auto j) { tmp = tmp * ordered_access_lengths[j] + ordered_access_idx[j]; });

                forward_sweep_(idim) = tmp % 2 == 0;
            });

            return forward_sweep_;
        }();

        // calculate multi-dim tensor index
        auto idx_md = [&]() {
            Index ordered_idx;

            static_for<0, nDim, 1>{}([&](auto idim) {
                ordered_idx(idim) =
                    !SnakeCurved || forward_sweep[idim]
                        ? ordered_access_idx[idim]
                        : ordered_access_lengths[idim] - 1 - ordered_access_idx[idim];
            });

            return container_reorder_given_old2new(ordered_idx, dim_access_order) *
                   ScalarsPerAccess{};
        }();
        return idx_md;
    }

    // FIXME: return tuple of number<>, which is compile time only variable
    template <index_t AccessIdx1d>
    static CK_TILE_HOST_DEVICE constexpr auto get_index(number<AccessIdx1d>)
    {
        constexpr auto idx = _get_index(number<AccessIdx1d>{});

        return generate_tuple([&](auto i) { return number<idx[i]>{}; }, number<nDim>{});
    }
};

} // namespace ck_tile
