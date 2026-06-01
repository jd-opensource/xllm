// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/multi_index.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/utility/magic_div.hpp"

namespace ck_tile {

enum struct coord_transform_enum
{
    undefined,
    pass_through,
    pad,
    embed,
    merge,
    unmerge,
    replicate,
    xor_t,
    offset,
    indexing,
};

template <index_t NDimLow, index_t NDimUp>
struct base_transform
{
    CK_TILE_HOST_DEVICE static constexpr auto get_type_enum()
    {
        return coord_transform_enum::undefined;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t get_num_of_lower_dimension() { return NDimLow; }

    CK_TILE_HOST_DEVICE static constexpr index_t get_num_of_upper_dimension() { return NDimUp; }

    // return safe value for vector length/stride, based on compile-time known only
    // variables
    // MUST be static function
    template <typename LowVectorLengths, typename LowVectorStrides>
    CK_TILE_HOST_DEVICE static constexpr auto
    calculate_upper_dimension_safe_vector_length_strides(const LowVectorLengths&,
                                                         const LowVectorStrides&)
    {
        if constexpr(NDimUp > 0)
        {
            array<index_t, NDimUp> up_vector_lengths{-1};
            array<index_t, NDimUp> up_vector_strides{-1};

            return make_tuple(up_vector_lengths, up_vector_strides);
        }
        else
        {
            return make_tuple(array<index_t, 0>{}, array<index_t, 0>{});
        }
    }
};

template <typename LowLength>
struct pass_through : public base_transform<1, 1>
{
    static constexpr auto type_enum = coord_transform_enum::pass_through;

    using LowerIndex = multi_index<1>;
    using UpperIndex = multi_index<1>;

    using UpLengths = decltype(make_tuple(LowLength{}));

    UpLengths up_lengths_;

    CK_TILE_HOST_DEVICE constexpr pass_through() = default;

    CK_TILE_HOST_DEVICE constexpr pass_through(const LowLength& low_length)
        : up_lengths_{make_tuple(low_length)}
    {
    }

    CK_TILE_HOST_DEVICE static constexpr auto get_type_enum()
    {
        return coord_transform_enum::pass_through;
    }

    CK_TILE_HOST_DEVICE constexpr const auto& get_upper_lengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE static constexpr void calculate_lower_index(LowIdx& idx_low,
                                                                    const UpIdx& idx_up)
    {
        static_assert(LowIdx::size() == 1 && UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(number<0>{}) = idx_up[number<0>{}];
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE static void update_lower_index(LowIdxDiff& idx_diff_low,
                                                       const UpIdxDiff& idx_diff_up,
                                                       LowIdx& idx_low,
                                                       const UpIdx&)
    {
        static_assert(LowIdxDiff::size() == 1 && UpIdxDiff::size() == 1 && LowIdx::size() == 1 &&
                          UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = number<0>{};

        idx_diff_low[I0] = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_always_mapped_to_valid_lower_index()
    {
        return true;
    }

    template <typename UpIdx>
    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_mapped_to_valid_lower_index(const UpIdx& /* idx_up */)
    {
        return true;
    }

    CK_TILE_HOST_DEVICE static constexpr bool is_known_at_compile_time()
    {
        return ck_tile::is_known_at_compile_time<UpLengths>::value;
    }

    // MUST be static function
    template <typename LowVectorLengths, typename LowVectorStrides>
    CK_TILE_HOST_DEVICE static constexpr auto
    calculate_upper_dimension_safe_vector_length_strides(const LowVectorLengths& low_vector_lengths,
                                                         const LowVectorStrides& low_vector_strides)
    {
        return make_tuple(low_vector_lengths, low_vector_strides);
    }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("pass_through{");

        //
        printf("up_lengths_:");
        print(up_lengths_);

        //
        printf("}");
    }
};

template <typename LowLength,
          typename LeftPadLength,
          typename RightPadLength,
          bool SkipIsValidCheck = false>
struct pad : public base_transform<1, 1>
{
    using LowerIndex = multi_index<1>;
    using UpperIndex = multi_index<1>;

    using UpLengths = decltype(make_tuple(LowLength{} + LeftPadLength{} + RightPadLength{}));

    UpLengths up_lengths_;
    LeftPadLength left_pad_length_;
    RightPadLength right_pad_length_;

    CK_TILE_HOST_DEVICE constexpr pad() : up_lengths_{}, left_pad_length_{}, right_pad_length_{} {}

    CK_TILE_HOST_DEVICE constexpr pad(const LowLength& low_length,
                                      const LeftPadLength& left_pad_length,
                                      const RightPadLength& right_pad_length)
        : up_lengths_{make_tuple(low_length + left_pad_length + right_pad_length)},
          left_pad_length_{left_pad_length},
          right_pad_length_{right_pad_length}
    {
    }

    CK_TILE_HOST_DEVICE constexpr const auto& get_upper_lengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE constexpr void calculate_lower_index(LowIdx& idx_low,
                                                             const UpIdx& idx_up) const
    {
        static_assert(LowIdx::size() == 1 && UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(number<0>{}) = idx_up[number<0>{}] - left_pad_length_;
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE static void update_lower_index(LowIdxDiff& idx_diff_low,
                                                       const UpIdxDiff& idx_diff_up,
                                                       LowIdx& idx_low,
                                                       const UpIdx&)
    {
        static_assert(LowIdxDiff::size() == 1 && UpIdxDiff::size() == 1 && LowIdx::size() == 1 &&
                          UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = number<0>{};

        idx_diff_low[I0] = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_always_mapped_to_valid_lower_index()
    {
        return SkipIsValidCheck;
    }

    template <typename UpIdx>
    CK_TILE_HOST_DEVICE constexpr bool
    is_valid_upper_index_mapped_to_valid_lower_index(const UpIdx& idx_up) const
    {
        return SkipIsValidCheck ||
               ((idx_up[number<0>{}] >= left_pad_length_) &&
                (idx_up[number<0>{}] < up_lengths_[number<0>{}] - right_pad_length_));
    }

    CK_TILE_HOST_DEVICE static constexpr bool is_known_at_compile_time()
    {
        return ck_tile::is_known_at_compile_time<UpLengths>::value &&
               ck_tile::is_known_at_compile_time<LeftPadLength>::value &&
               ck_tile::is_known_at_compile_time<RightPadLength>::value;
    }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("pad{");

        //
        printf("up_lengths_: ");
        print(up_lengths_);
        printf(", ");

        //
        printf("left_pad_length_: ");
        print(left_pad_length_);
        printf(", ");

        //
        printf("right_pad_length_: ");
        print(right_pad_length_);

        printf("}");
    }
};

template <typename LowLength, typename LeftPadLength, bool SkipIsValidCheck = false>
struct left_pad
{
    using LowerIndex = multi_index<1>;
    using UpperIndex = multi_index<1>;

    using UpLengths = decltype(make_tuple(LowLength{} + LeftPadLength{}));

    UpLengths up_lengths_;
    LeftPadLength left_pad_length_;

    CK_TILE_HOST_DEVICE constexpr left_pad() = default;

    CK_TILE_HOST_DEVICE constexpr left_pad(const LowLength& low_length,
                                           const LeftPadLength& left_pad_length)
        : up_lengths_{make_tuple(low_length + left_pad_length)}, left_pad_length_{left_pad_length}
    {
    }

    CK_TILE_HOST_DEVICE constexpr const auto& get_upper_lengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE constexpr void calculate_lower_index(LowIdx& idx_low,
                                                             const UpIdx& idx_up) const
    {
        static_assert(LowIdx::size() == 1 && UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(number<0>{}) = idx_up[number<0>{}] - left_pad_length_;
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE static void update_lower_index(LowIdxDiff& idx_diff_low,
                                                       const UpIdxDiff& idx_diff_up,
                                                       LowIdx& idx_low,
                                                       const UpIdx&)
    {
        static_assert(LowIdxDiff::size() == 1 && UpIdxDiff::size() == 1 && LowIdx::size() == 1 &&
                          UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = number<0>{};

        idx_diff_low[I0] = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_always_mapped_to_valid_lower_index()
    {
        return SkipIsValidCheck;
    }

    template <typename UpIdx>
    CK_TILE_HOST_DEVICE constexpr bool
    is_valid_upper_index_mapped_to_valid_lower_index(const UpIdx& idx_up) const
    {
        return SkipIsValidCheck || (idx_up[number<0>{}] >= left_pad_length_);
    }

    CK_TILE_HOST_DEVICE static constexpr bool is_known_at_compile_time()
    {
        return ck_tile::is_known_at_compile_time<UpLengths>::value &&
               ck_tile::is_known_at_compile_time<LeftPadLength>::value;
    }

    // MUST be static function
    template <typename LowVectorLengths, typename LowVectorStrides>
    CK_TILE_HOST_DEVICE static constexpr auto
    calculate_upper_dimension_safe_vector_length_strides(const LowVectorLengths& low_vector_lengths,
                                                         const LowVectorStrides& low_vector_strides)
    {
        // TODO: we allow pass through this vector length. If one need per-pixel check,
        //       should change the guaranteed vector length while creating the tensor view.
        //       It's up to runtime to check the padding length should be multiple of vector length
        return make_tuple(low_vector_lengths, low_vector_strides);
    }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("left_pad{");

        //
        printf("up_lengths_: ");
        print(up_lengths_);
        printf(", ");

        //
        printf("left_pad_length_: ");
        print(left_pad_length_);

        printf("}");
    }
};

template <typename LowLength, typename RightPadLength, bool SkipIsValidCheck = false>
struct right_pad : public base_transform<1, 1>
{
    using LowerIndex = multi_index<1>;
    using UpperIndex = multi_index<1>;

    using UpLengths = decltype(make_tuple(LowLength{} + RightPadLength{}));

    UpLengths up_lengths_;
    LowLength low_length_;
    RightPadLength right_pad_length_;

    CK_TILE_HOST_DEVICE constexpr right_pad() = default;

    CK_TILE_HOST_DEVICE constexpr right_pad(const LowLength& low_length,
                                            const RightPadLength& right_pad_length)
        : up_lengths_{make_tuple(low_length + right_pad_length)},
          low_length_{low_length},
          right_pad_length_{right_pad_length}
    {
    }

    CK_TILE_HOST_DEVICE constexpr const auto& get_upper_lengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE static constexpr void calculate_lower_index(LowIdx& idx_low,
                                                                    const UpIdx& idx_up)
    {
        static_assert(LowIdx::size() == 1 && UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(number<0>{}) = idx_up[number<0>{}];
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE static void update_lower_index(LowIdxDiff& idx_diff_low,
                                                       const UpIdxDiff& idx_diff_up,
                                                       LowIdx& idx_low,
                                                       const UpIdx&)
    {
        static_assert(LowIdxDiff::size() == 1 && UpIdxDiff::size() == 1 && LowIdx::size() == 1 &&
                          UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = number<0>{};

        idx_diff_low[I0] = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_always_mapped_to_valid_lower_index()
    {
        return SkipIsValidCheck;
    }

    template <typename UpIdx>
    CK_TILE_HOST_DEVICE constexpr bool
    is_valid_upper_index_mapped_to_valid_lower_index(const UpIdx& idx_up) const
    {
        return SkipIsValidCheck || (idx_up[number<0>{}] < low_length_);
    }

    CK_TILE_HOST_DEVICE static constexpr bool is_known_at_compile_time()
    {
        return ck_tile::is_known_at_compile_time<UpLengths>::value &&
               ck_tile::is_known_at_compile_time<LowLength>::value &&
               ck_tile::is_known_at_compile_time<RightPadLength>::value;
    }

    // MUST be static function
    template <typename LowVectorLengths, typename LowVectorStrides>
    CK_TILE_HOST_DEVICE static constexpr auto
    calculate_upper_dimension_safe_vector_length_strides(const LowVectorLengths& low_vector_lengths,
                                                         const LowVectorStrides& low_vector_strides)
    {
        // TODO: we allow pass through this vector length. If one need per-pixel check,
        //       should change the guaranteed vector length while creating the tensor view.
        //       It's up to runtime to check the padding length should be multiple of vector length
        return make_tuple(low_vector_lengths, low_vector_strides);
    }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("right_pad{");

        //
        printf("up_lengths_: ");
        print(up_lengths_);
        printf(", ");

        //
        printf("right_pad_length_: ");
        print(right_pad_length_);

        printf("}");
    }
};

// idx_low = coefficients[0, ...nDimUp-1] * idx_up[0, ...nDimUp-1]
// UpLengths and Coefficients can be either of the followings:
//   1) Tuple of index_t, which is known at run-time, or
//   2) Tuple of number, which is known at compile-time, or
//   3) Tuple of mixture of index_t and number, which is known partially at run-time and partially
//   at compile-time
template <typename UpLengths,
          typename Coefficients,
          typename std::enable_if<UpLengths::size() == Coefficients::size(), bool>::type = false>
struct embed : public base_transform<1, UpLengths::size()>
{
    static constexpr index_t NDimUp = UpLengths::size();

    using LowerIndex = multi_index<1>;
    using UpperIndex = multi_index<NDimUp>;

    UpLengths up_lengths_;
    Coefficients coefficients_;

    CK_TILE_HOST_DEVICE constexpr embed() = default;

    CK_TILE_HOST_DEVICE constexpr embed(const UpLengths& up_lengths,
                                        const Coefficients& coefficients)
        : up_lengths_{up_lengths}, coefficients_{coefficients}
    {
    }

    CK_TILE_HOST_DEVICE static constexpr auto get_type_enum()
    {
        return coord_transform_enum::embed;
    }

    CK_TILE_HOST_DEVICE constexpr const auto& get_upper_lengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE constexpr void calculate_lower_index(LowIdx& idx_low,
                                                             const UpIdx& idx_up) const
    {
        static_assert(LowIdx::size() == 1 && UpIdx::size() == NDimUp,
                      "wrong! inconsistent # of dimension");

        idx_low(number<0>{}) = 0;

        static_for<0, NDimUp, 1>{}([&idx_low, &idx_up, this](auto i) {
            idx_low(number<0>{}) += idx_up[i] * this->coefficients_[i];
        });
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE void update_lower_index(LowIdxDiff& idx_diff_low,
                                                const UpIdxDiff& idx_diff_up,
                                                LowIdx& idx_low,
                                                const UpIdx&) const
    {
        static_assert(LowIdxDiff::size() == 1 && UpIdxDiff::size() == NDimUp &&
                          LowIdx::size() == 1 && UpIdx::size() == NDimUp,
                      "wrong! inconsistent # of dimension");

        idx_diff_low(number<0>{}) = 0;

        static_for<0, NDimUp, 1>{}(
            [&](auto i) { idx_diff_low(number<0>{}) += idx_diff_up[i] * coefficients_[i]; });

        idx_low += idx_diff_low;
    }

    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_always_mapped_to_valid_lower_index()
    {
        return true;
    }

    template <typename UpIdx>
    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_mapped_to_valid_lower_index(const UpIdx& /* idx_up */)
    {
        return true;
    }

    CK_TILE_HOST_DEVICE static constexpr bool is_known_at_compile_time()
    {
        return ck_tile::is_known_at_compile_time<UpLengths>::value &&
               ck_tile::is_known_at_compile_time<Coefficients>::value;
    }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("embed{");

        //
        printf("up_lengths_: ");
        print(up_lengths_);
        printf(", ");

        //
        printf("coefficients_: ");
        print(coefficients_);

        printf("}");
    }
};

template <typename LowLengths>
struct lambda_merge_generate_MagicDivision_calculate_magic_divisor
{
    template <index_t I>
    CK_TILE_HOST_DEVICE constexpr auto operator()(number<I> i) const
    {
        return magic_division::calculate_magic_numbers(LowLengths{}[i]);
    }
};

// Implementation of "merge" transformation primitive that uses magic-number-division to do lowering
// of both multi-index and delta of multi-index
// Caution:
//   1. The magic number division implementation being used would produce correct result if the
//   dividended is uint32_t and its value is with in 31-bit value range of uint32_t.
//   2. The magic number division for int32_t dividened has not been implemented, the int32_t
//   dividend would be bit-wise interpreted as uint32_t and magic number division implementation for
//   uint32_t is then used.
//   3. For merge primitive, upper-index is the dividend.
//   4. When upper-index is uint32_t, its value need to be within 31-bit range.
//   5. When upper-index is int32_t type (when index_t is int32_t), its value need to be
//   non-negative.
template <typename LowLengths>
struct merge_v2_magic_division : public base_transform<LowLengths::size(), 1>
{
    static constexpr index_t NDimLow = LowLengths::size();

    using LowerIndex = multi_index<NDimLow>;
    using UpperIndex = multi_index<1>;

    using UpLengths =
        decltype(make_tuple(container_reduce(LowLengths{}, multiplies{}, number<1>{})));

    using LowLengthsMagicDivisor = decltype(generate_tuple(
        lambda_merge_generate_MagicDivision_calculate_magic_divisor<LowLengths>{},
        number<NDimLow>{}));

    LowLengths low_lengths_;
    LowLengthsMagicDivisor low_lengths_magic_divisor_;
    UpLengths up_lengths_;

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};

    CK_TILE_HOST_DEVICE constexpr merge_v2_magic_division() = default;

    CK_TILE_HOST_DEVICE constexpr merge_v2_magic_division(const LowLengths& low_lengths)
        : low_lengths_{low_lengths},
          low_lengths_magic_divisor_{generate_tuple(
              [&](auto i) { return magic_division::calculate_magic_numbers(low_lengths[i]); },
              number<NDimLow>{})},
          up_lengths_{make_tuple(container_reduce(low_lengths, multiplies{}, I1))}
    {
        static_assert(LowerIndex::size() == NDimLow, "wrong!");
    }

    CK_TILE_HOST_DEVICE static constexpr auto get_type_enum()
    {
        return coord_transform_enum::merge;
    }

    CK_TILE_HOST_DEVICE constexpr const auto& get_upper_lengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE constexpr void calculate_lower_index(LowIdx& idx_low,
                                                             const UpIdx& idx_up) const
    {
        static_assert(LowIdx::size() == NDimLow && UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");

        index_t tmp = idx_up[I0];

        static_for<NDimLow - 1, 0, -1>{}([&, this](auto i) {
            index_t tmp2 =
                magic_division::do_magic_division(tmp,
                                                  this->low_lengths_magic_divisor_[i][I0],
                                                  this->low_lengths_magic_divisor_[i][I1]);
            idx_low(i) = tmp - tmp2 * this->low_lengths_[i];
            tmp        = tmp2;
        });

        idx_low(number<0>{}) = tmp;
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE void update_lower_index(LowIdxDiff& idx_diff_low,
                                                const UpIdxDiff&,
                                                LowIdx& idx_low,
                                                const UpIdx& idx_up_new) const
    {
        static_assert(LowIdxDiff::size() == NDimLow && UpIdxDiff::size() == 1 &&
                          LowIdx::size() == NDimLow && UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");

        index_t tmp = idx_up_new[number<0>{}];

        static_for<NDimLow - 1, 0, -1>{}([&, this](auto i) {
            index_t tmp2 =
                magic_division::do_magic_division(tmp,
                                                  this->low_lengths_magic_divisor_[i][I0],
                                                  this->low_lengths_magic_divisor_[i][I1]);

            index_t idx_low_old = idx_low[i];

            idx_low(i) = tmp - tmp2 * this->low_lengths_[i];
            tmp        = tmp2;

            idx_diff_low(i) = idx_low[i] - idx_low_old;
        });

        idx_diff_low(number<0>{}) = tmp - idx_low(number<0>{});

        idx_low(number<0>{}) = tmp;
    }

    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_always_mapped_to_valid_lower_index()
    {
        return true;
    }

    CK_TILE_HOST_DEVICE static constexpr bool is_known_at_compile_time()
    {
        return ck_tile::is_known_at_compile_time<LowLengths>::value &&
               ck_tile::is_known_at_compile_time<LowLengthsMagicDivisor>::value &&
               ck_tile::is_known_at_compile_time<UpLengths>::value;
    }

    template <typename UpIdx>
    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_mapped_to_valid_lower_index(const UpIdx& /* idx_up */)
    {
        return true;
    }

    // MUST be static function
    template <typename LowVectorLengths, typename LowVectorStrides>
    CK_TILE_HOST_DEVICE static constexpr auto
    calculate_upper_dimension_safe_vector_length_strides(const LowVectorLengths& low_vector_lengths,
                                                         const LowVectorStrides& low_vector_strides)
    {
        array<index_t, 1> up_vector_lengths{-1};
        array<index_t, 1> up_vector_strides{-1};

        up_vector_lengths[0] = low_vector_lengths[number<NDimLow - 1>{}];
        up_vector_strides[0] = low_vector_strides[number<NDimLow - 1>{}];

        return make_tuple(up_vector_lengths, up_vector_strides);
    }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("merge_v2_magic_division{");

        //
        printf("low_lengths_ ");
        print(low_lengths_);
        printf(", ");

        //
        printf("up_lengths_ ");
        print(up_lengths_);

        printf("}");
    }
};

// Implementation of "merge" transformation primitive that uses division and mod. It is supposed to
// be used for low_lengths that are known at compile time and are power of 2, otherwise performance
// will be very bad
template <typename LowLengths>
struct merge_v3_division_mod : public base_transform<LowLengths::size(), 1>
{
    static constexpr index_t NDimLow = LowLengths::size();

    using LowerIndex = multi_index<NDimLow>;
    using UpperIndex = multi_index<1>;

    using LowLengthsScan =
        decltype(container_reverse_exclusive_scan(LowLengths{}, multiplies{}, number<1>{}));

    using UpLengths =
        decltype(make_tuple(container_reduce(LowLengths{}, multiplies{}, number<1>{})));

    LowLengths low_lengths_;
    LowLengthsScan low_lengths_scan_;
    UpLengths up_lengths_;

    CK_TILE_HOST_DEVICE constexpr merge_v3_division_mod() = default;

    CK_TILE_HOST_DEVICE constexpr merge_v3_division_mod(const LowLengths& low_lengths)
        : low_lengths_{low_lengths},
          low_lengths_scan_{
              container_reverse_exclusive_scan(low_lengths, multiplies{}, number<1>{})},
          up_lengths_{make_tuple(container_reduce(low_lengths, multiplies{}, number<1>{}))}
    {
        static_assert(LowerIndex::size() == NDimLow, "wrong!");
    }

    CK_TILE_HOST_DEVICE constexpr const auto& get_upper_lengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE constexpr void calculate_lower_index(LowIdx& idx_low,
                                                             const UpIdx& idx_up) const
    {
        static_assert(LowIdx::size() == NDimLow && UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");

        index_t tmp = idx_up[number<0>{}];

        // division and mod
        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_low(i) = tmp / this->low_lengths_scan_[i];
            tmp %= this->low_lengths_scan_[i];
        });

        idx_low(number<NDimLow - 1>{}) = tmp;
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE void update_lower_index(LowIdxDiff& idx_diff_low,
                                                const UpIdxDiff&,
                                                LowIdx& idx_low,
                                                const UpIdx& idx_up_new) const
    {
        static_assert(LowIdxDiff::size() == NDimLow && UpIdxDiff::size() == 1 &&
                          LowIdx::size() == NDimLow && UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0   = number<0>{};
        constexpr auto INm1 = number<NDimLow - 1>{};

        index_t tmp = idx_up_new[I0];

        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            const index_t tmp2 = idx_low[i];
            idx_low(i)         = tmp / this->low_lengths_scan_[i];
            idx_diff_low(i)    = idx_low[i] - tmp2;
            tmp %= this->low_lengths_scan_[i];
        });

        const index_t tmp2 = idx_low[INm1];
        idx_low(INm1)      = tmp;
        idx_diff_low(INm1) = idx_low[INm1] - tmp2;
    }

    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_always_mapped_to_valid_lower_index()
    {
        return true;
    }

    CK_TILE_HOST_DEVICE static constexpr bool is_known_at_compile_time()
    {
        return ck_tile::is_known_at_compile_time<LowLengths>::value &&
               ck_tile::is_known_at_compile_time<LowLengthsScan>::value &&
               ck_tile::is_known_at_compile_time<UpLengths>::value;
    }

    template <typename UpIdx>
    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_mapped_to_valid_lower_index(const UpIdx& /* idx_up */)
    {
        return true;
    }

    // MUST be static function
    template <typename LowVectorLengths, typename LowVectorStrides>
    CK_TILE_HOST_DEVICE static constexpr auto
    calculate_upper_dimension_safe_vector_length_strides(const LowVectorLengths& low_vector_lengths,
                                                         const LowVectorStrides& low_vector_strides)
    {
        array<index_t, 1> up_vector_lengths{-1};
        array<index_t, 1> up_vector_strides{-1};

        up_vector_lengths[0] = low_vector_lengths[number<NDimLow - 1>{}];
        up_vector_strides[0] = low_vector_strides[number<NDimLow - 1>{}];

        return make_tuple(up_vector_lengths, up_vector_strides);
    }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("Merge_v3_direct_division_mod{");

        //
        printf("low_lengths_ ");
        print(low_lengths_);
        printf(", ");

        //
        printf("low_lengths_scan_ ");
        print(low_lengths_scan_);
        printf(", ");

        //
        printf("up_lengths_ ");
        print(up_lengths_);

        printf("}");
    }
};

template <typename UpLengths, bool Use24BitIntegerCalculation>
struct unmerge : public base_transform<1, UpLengths::size()>
{
    static constexpr index_t NDimUp = UpLengths::size();

    using LowerIndex = multi_index<1>;
    using UpperIndex = multi_index<NDimUp>;

    using UpLengthsScan =
        decltype(container_reverse_exclusive_scan(UpLengths{}, multiplies{}, number<1>{}));

    UpLengths up_lengths_;
    UpLengthsScan up_lengths_scan_;

    CK_TILE_HOST_DEVICE constexpr unmerge() = default;

    CK_TILE_HOST_DEVICE constexpr unmerge(const UpLengths& up_lengths)
        : up_lengths_{up_lengths},
          up_lengths_scan_{container_reverse_exclusive_scan(up_lengths, multiplies{}, number<1>{})}
    {
    }

    CK_TILE_HOST_DEVICE static constexpr auto get_type_enum()
    {
        return coord_transform_enum::unmerge;
    }

    CK_TILE_HOST_DEVICE constexpr const auto& get_upper_lengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE constexpr void calculate_lower_index(LowIdx& idx_low,
                                                             const UpIdx& idx_up) const
    {
        if constexpr(!Use24BitIntegerCalculation)
        {
            idx_low(number<0>{}) = idx_up[number<NDimUp - 1>{}];

            static_for<0, NDimUp - 1, 1>{}(
                [&](auto i) { idx_low(number<0>{}) += idx_up[i] * up_lengths_scan_[i]; });
        }
        else
        {
            idx_low(number<0>{}) = idx_up[number<NDimUp - 1>{}];

            static_for<0, NDimUp - 1, 1>{}([&](auto i) {
                idx_low(number<0>{}) =
                    (0x00ffffff & idx_low[number<0>{}]) +
                    (0x00ffffff & idx_up[i]) * (0x00ffffff & up_lengths_scan_[i]);
            });
        }
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE void update_lower_index(LowIdxDiff& idx_diff_low,
                                                const UpIdxDiff& idx_diff_up,
                                                LowIdx& idx_low,
                                                const UpIdx&) const
    {
        calculate_lower_index(idx_diff_low, idx_diff_up);

        idx_low += idx_diff_low;
    }

    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_always_mapped_to_valid_lower_index()
    {
        return true;
    }

    template <typename UpIdx>
    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_mapped_to_valid_lower_index(const UpIdx& /* idx_up */)
    {
        return true;
    }

    CK_TILE_HOST_DEVICE static constexpr bool is_known_at_compile_time()
    {
        return ck_tile::is_known_at_compile_time<UpLengths>::value &&
               ck_tile::is_known_at_compile_time<UpLengthsScan>::value;
    }

    // MUST be static function
    template <typename LowVectorLengths, typename LowVectorStrides>
    CK_TILE_HOST_DEVICE static constexpr auto
    calculate_upper_dimension_safe_vector_length_strides(const LowVectorLengths& low_vector_lengths,
                                                         const LowVectorStrides& low_vector_strides)
    {
        array<index_t, NDimUp> up_vector_lengths{-1};
        array<index_t, NDimUp> up_vector_strides{-1};

        constexpr auto up_length_last = UpLengths{}[number<NDimUp - 1>{}];

        if constexpr(ck_tile::is_known_at_compile_time<decltype(up_length_last)>::value)
        {
            if(low_vector_lengths[0] != -1)
            {
                up_vector_lengths(NDimUp - 1) = gcd(low_vector_lengths[0], up_length_last);
            }
        }

        up_vector_strides(NDimUp - 1) = low_vector_strides[0];

        return make_tuple(up_vector_lengths, up_vector_strides);
    }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("unmerge{");

        //
        printf("up_lengths_");
        print(up_lengths_);
        printf(", ");

        //
        printf("up_lengths_scan_");
        print(up_lengths_scan_);

        printf("}");
    }
};

template <typename LowerIndex>
struct freeze : public base_transform<1, 0>
{
    LowerIndex low_idx_;

    CK_TILE_HOST_DEVICE constexpr freeze() = default;

    CK_TILE_HOST_DEVICE constexpr freeze(const LowerIndex& low_idx) : low_idx_{low_idx} {}

    CK_TILE_HOST_DEVICE static constexpr auto get_upper_lengths() { return tuple<>{}; }

    template <typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE constexpr void calculate_lower_index(LowIdx& idx_low,
                                                             const UpIdx& /* idx_up */) const
    {
        static_assert(LowIdx::size() == 1 && UpIdx::size() == 0,
                      "wrong! inconsistent # of dimension");

        idx_low(number<0>{}) = low_idx_;
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE static void update_lower_index(LowIdxDiff& idx_diff_low,
                                                       const UpIdxDiff& /* idx_diff_up */,
                                                       LowIdx& /* idx_low */,
                                                       const UpIdx& /* idx_up_new */)
    {
        idx_diff_low(number<0>{}) = 0;
    }

    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_always_mapped_to_valid_lower_index()
    {
        return true;
    }

    template <typename UpIdx>
    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_mapped_to_valid_lower_index(const UpIdx& /* idx_up */)
    {
        return true;
    }

    CK_TILE_HOST_DEVICE static constexpr bool is_known_at_compile_time()
    {
        return ck_tile::is_known_at_compile_time<LowerIndex>::value;
    }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("freeze{");

        //
        printf("low_idx_: ");
        print(low_idx_);

        printf("}");
    }
};

// insert a dangling upper dimension without lower dimension
template <typename UpperLength>
struct insert : public base_transform<0, 1>
{
    using UpLengths = decltype(make_tuple(UpperLength{}));

    UpLengths up_lengths_;

    CK_TILE_HOST_DEVICE constexpr insert() = default;

    CK_TILE_HOST_DEVICE constexpr insert(const UpperLength& up_length)
        : up_lengths_{make_tuple(up_length)}
    {
    }

    CK_TILE_HOST_DEVICE static constexpr index_t get_num_of_lower_dimension() { return 0; }

    CK_TILE_HOST_DEVICE static constexpr index_t get_num_of_upper_dimension() { return 1; }

    CK_TILE_HOST_DEVICE constexpr auto get_upper_lengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE constexpr void calculate_lower_index(LowIdx&, const UpIdx&) const
    {
        static_assert(LowIdx::size() == 0 && UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE static void
    update_lower_index(LowIdxDiff&, const UpIdxDiff&, LowIdx&, const UpIdx&)
    {
        static_assert(LowIdxDiff::size() == 0 && UpIdxDiff::size() == 1 && LowIdx::size() == 0 &&
                          UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");
    }

    CK_TILE_HOST_DEVICE static constexpr bool IsLinearTransform() { return true; }

    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_always_mapped_to_valid_lower_index()
    {
        return true;
    }

    template <typename UpIdx>
    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_mapped_to_valid_lower_index(const UpIdx& /* idx_up */)
    {
        return true;
    }

    CK_TILE_HOST_DEVICE static constexpr bool is_known_at_compile_time()
    {
        return ck_tile::is_known_at_compile_time<UpperLength>::value;
    }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("insert{");

        //
        print(up_lengths_);

        printf("}");
    }
};

// replicate the original tensor and create a higher dimensional tensor
template <typename UpLengths>
struct replicate : public base_transform<0, UpLengths::size()>
{
    static constexpr index_t NDimUp = UpLengths::size();

    CK_TILE_HOST_DEVICE constexpr replicate() = default;

    CK_TILE_HOST_DEVICE constexpr replicate(const UpLengths& up_lengths) : up_lengths_{up_lengths}
    {
    }

    CK_TILE_HOST_DEVICE constexpr auto get_upper_lengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE constexpr void calculate_lower_index(LowIdx&, const UpIdx&) const
    {
        static_assert(LowIdx::size() == 0 && UpIdx::size() == NDimUp,
                      "wrong! inconsistent # of dimension");
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE static void
    update_lower_index(LowIdxDiff&, const UpIdxDiff&, LowIdx&, const UpIdx&)
    {
        static_assert(LowIdxDiff::size() == 0 && UpIdxDiff::size() == NDimUp &&
                          LowIdx::size() == 0 && UpIdx::size() == NDimUp,
                      "wrong! inconsistent # of dimension");
    }

    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_always_mapped_to_valid_lower_index()
    {
        return true;
    }

    template <typename UpIdx>
    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_mapped_to_valid_lower_index(const UpIdx& /* idx_up */)
    {
        return true;
    }

    CK_TILE_HOST_DEVICE static constexpr bool is_known_at_compile_time()
    {
        return ck_tile::is_known_at_compile_time<UpLengths>::value;
    }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("replicate{");

        //
        printf("up_lengths_: ");
        print(up_lengths_);

        printf("}");
    }

    //
    UpLengths up_lengths_;
};

template <typename LowLength, typename SliceBegin, typename SliceEnd>
struct slice : public base_transform<1, 1>
{
    using LowerIndex = multi_index<1>;
    using UpperIndex = multi_index<1>;

    using UpLengths = decltype(make_tuple(SliceEnd{} - SliceBegin{}));

    UpLengths up_lengths_;
    SliceBegin slice_begin_;
    SliceEnd slice_end_;

    CK_TILE_HOST_DEVICE constexpr slice() = default;

    CK_TILE_HOST_DEVICE constexpr slice(const LowLength&,
                                        const SliceBegin& slice_begin,
                                        const SliceEnd& slice_end)
        : up_lengths_{make_tuple(slice_end - slice_begin)},
          slice_begin_{slice_begin},
          slice_end_{slice_end}
    {
    }

    CK_TILE_HOST_DEVICE constexpr const auto& get_upper_lengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE constexpr void calculate_lower_index(LowIdx& idx_low,
                                                             const UpIdx& idx_up) const
    {
        static_assert(LowIdx::size() == 1 && UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(number<0>{}) = idx_up[number<0>{}] + slice_begin_;
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE static void update_lower_index(LowIdxDiff& idx_diff_low,
                                                       const UpIdxDiff& idx_diff_up,
                                                       LowIdx& idx_low,
                                                       const UpIdx&)
    {
        static_assert(LowIdxDiff::size() == 1 && UpIdxDiff::size() == 1 && LowIdx::size() == 1 &&
                          UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = number<0>{};

        idx_diff_low[I0] = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_always_mapped_to_valid_lower_index()
    {
        return true;
    }

    template <typename UpIdx>
    CK_TILE_HOST_DEVICE constexpr bool
    is_valid_upper_index_mapped_to_valid_lower_index(const UpIdx&) const
    {
        return true;
    }

    CK_TILE_HOST_DEVICE static constexpr bool is_known_at_compile_time()
    {
        return ck_tile::is_known_at_compile_time<UpLengths>::value &&
               ck_tile::is_known_at_compile_time<SliceBegin>::value &&
               ck_tile::is_known_at_compile_time<SliceEnd>::value;
    }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("slice{");

        //
        printf("up_lengths_: ");
        print(up_lengths_);
        printf(", ");

        //
        printf("slice_begin_: ");
        print(slice_begin_);
        printf(", ");

        //
        printf("slice_end_: ");
        print(slice_end_);

        printf("}");
    } // namespace ck
};    // namespace ck

/*
 * \brief lower_idx = upper_idx % modulus.
 * TODO: Need an improved implementation since the modulo operation is expensive.
 */
template <typename Modulus, typename UpLength>
struct modulo : public base_transform<1, 1>
{
    using LowerIndex = multi_index<1>;
    using UpperIndex = multi_index<1>;
    using UpLengths  = decltype(make_tuple(UpLength{}));

    Modulus modulus_;
    UpLengths up_lengths_;

    CK_TILE_HOST_DEVICE constexpr modulo() = default;

    CK_TILE_HOST_DEVICE constexpr modulo(const Modulus& modulus, const UpLength& up_length)
        : modulus_{modulus}, up_lengths_{make_tuple(up_length)}
    {
    }

    CK_TILE_HOST_DEVICE constexpr const auto& get_upper_lengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE constexpr void calculate_lower_index(LowIdx& idx_low,
                                                             const UpIdx& idx_up) const
    {
        static_assert(LowIdx::size() == 1 && UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(number<0>{}) = idx_up[number<0>{}] % modulus_;
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE void update_lower_index(LowIdxDiff& idx_diff_low,
                                                const UpIdxDiff& idx_diff_up,
                                                LowIdx& idx_low,
                                                const UpIdx& up_idx) const
    {
        static_assert(LowIdxDiff::size() == 1 && UpIdxDiff::size() == 1 && LowIdx::size() == 1 &&
                          UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = number<0>{};

        const auto idx_low_old = idx_low;
        idx_low[I0]            = (up_idx[I0] + idx_diff_up[I0]) % modulus_;
        idx_diff_low[I0]       = idx_low - idx_low_old;
    }

    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_always_mapped_to_valid_lower_index()
    {
        return true;
    }

    template <typename UpIdx>
    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_mapped_to_valid_lower_index(const UpIdx& /* idx_up */)
    {
        return true;
    }

    CK_TILE_HOST_DEVICE static constexpr bool is_known_at_compile_time()
    {
        return ck_tile::is_known_at_compile_time<UpLengths>::value;
    }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("Modulus{");

        //
        printf("up_lengths_: ");
        print(up_lengths_);

        printf("}");
    }
};

// 2D XOR, NOTE: "xor" is a keyword
template <typename LowLengths>
struct xor_t : public base_transform<2, 2>
{
    static constexpr auto type_enum = coord_transform_enum::xor_t;

    using LowerIndex = multi_index<2>;
    using UpperIndex = multi_index<2>;

    using UpLengths = LowLengths;

    UpLengths up_lengths_;

    CK_TILE_HOST_DEVICE constexpr xor_t() : up_lengths_{} {}

    CK_TILE_HOST_DEVICE constexpr xor_t(const LowLengths& low_lengths) : up_lengths_{low_lengths} {}

    CK_TILE_HOST_DEVICE static constexpr auto get_type_enum()
    {
        return coord_transform_enum::xor_t;
    }

    CK_TILE_HOST_DEVICE constexpr const auto& get_upper_lengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE constexpr void calculate_lower_index(LowIdx& idx_low,
                                                             const UpIdx& idx_up) const
    {
        static_assert(LowIdx::size() == 2 && UpIdx::size() == 2,
                      "wrong! inconsistent # of dimension");

        idx_low(number<0>{}) = idx_up[number<0>{}];

        idx_low(number<1>{}) =
            idx_up[number<1>{}] ^ (idx_up[number<0>{}] % up_lengths_[number<1>{}]);
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE void update_lower_index(LowIdxDiff& idx_diff_low,
                                                const UpIdxDiff&,
                                                LowIdx& idx_low,
                                                const UpIdx& idx_up) const
    {
        static_assert(LowIdxDiff::size() == 2 && UpIdxDiff::size() == 2 && LowIdx::size() == 2 &&
                          UpIdx::size() == 2,
                      "wrong! inconsistent # of dimension");

        const auto idx_low_old = idx_low;

        calculate_lower_index(idx_low, idx_up);

        idx_diff_low = idx_low - idx_low_old;
    }

    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_always_mapped_to_valid_lower_index()
    {
        return true;
    }

    template <typename UpIdx>
    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_mapped_to_valid_lower_index(const UpIdx& /* idx_up */)
    {
        return true;
    }

    CK_TILE_HOST_DEVICE static constexpr bool is_known_at_compile_time()
    {
        return ck_tile::is_known_at_compile_time<UpLengths>::value;
    }

    // MUST be static function
    template <typename LowVectorLengths, typename LowVectorStrides>
    CK_TILE_HOST_DEVICE constexpr auto calculate_upper_dimension_safe_vector_length_strides(
        const LowVectorLengths& low_vector_lengths,
        const LowVectorStrides& low_vector_strides) const
    {
        array<index_t, 2> up_vector_lengths = low_vector_lengths;
        array<index_t, 2> up_vector_strides = low_vector_strides;

        return make_tuple(up_vector_lengths, up_vector_strides);
    }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("xor_t{");

        //
        printf("up_lengths_: ");
        print(up_lengths_);
        printf(", ");

        printf("}");
    }
};

template <typename LowLength, typename OffsetLength>
struct offset : public base_transform<1, 1>
{
    using LowerIndex = multi_index<1>;
    using UpperIndex = multi_index<1>;

    using UpLengths = decltype(make_tuple(LowLength{}));

    UpLengths up_lengths_;
    OffsetLength offset_length_;

    CK_TILE_HOST_DEVICE constexpr offset() = default;

    CK_TILE_HOST_DEVICE constexpr offset(const LowLength& low_length,
                                         const OffsetLength& offset_length)
        : up_lengths_{make_tuple(low_length)}, offset_length_{offset_length}
    {
    }

    CK_TILE_HOST_DEVICE static constexpr auto get_type_enum()
    {
        return coord_transform_enum::offset;
    }

    CK_TILE_HOST_DEVICE constexpr const auto& get_upper_lengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE constexpr void calculate_lower_index(LowIdx& idx_low,
                                                             const UpIdx& idx_up) const
    {
        static_assert(LowIdx::size() == 1 && UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(number<0>{}) = idx_up[number<0>{}] + offset_length_;
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE static void update_lower_index(LowIdxDiff& idx_diff_low,
                                                       const UpIdxDiff& idx_diff_up,
                                                       LowIdx& idx_low,
                                                       const UpIdx&)
    {
        static_assert(LowIdxDiff::size() == 1 && UpIdxDiff::size() == 1 && LowIdx::size() == 1 &&
                          UpIdx::size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = number<0>{};

        idx_diff_low[I0] = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_always_mapped_to_valid_lower_index()
    {
        return true;
    }

    template <typename UpIdx>
    CK_TILE_HOST_DEVICE constexpr bool
    is_valid_upper_index_mapped_to_valid_lower_index(const UpIdx&) const
    {
        return true;
    }

    CK_TILE_HOST_DEVICE static constexpr bool is_known_at_compile_time()
    {
        return ck_tile::is_known_at_compile_time<UpLengths>::value &&
               ck_tile::is_known_at_compile_time<OffsetLength>::value;
    }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("offset{");

        //
        printf("up_lengths_: ");
        print(up_lengths_);
        printf(", ");

        //
        printf("offset_length_: ");
        print(offset_length_);

        printf("}");
    }
};

template <typename UpLength, typename IndexingAdaptor>
struct indexing : public base_transform<1, 1>
{
    static constexpr index_t NDimUp = 1;

    using LowerIndex = multi_index<1>;
    using UpperIndex = multi_index<1>;

    using UpLengths = decltype(make_tuple(UpLength{}));
    UpLengths up_lengths_;
    IndexingAdaptor iadaptor_;

    CK_TILE_HOST_DEVICE constexpr indexing() = default;

    CK_TILE_HOST_DEVICE constexpr indexing(const UpLength& up_length,
                                           const IndexingAdaptor& iadaptor)
        : up_lengths_{make_tuple(up_length)}, iadaptor_{iadaptor}
    {
    }

    CK_TILE_HOST_DEVICE static constexpr auto get_type_enum()
    {
        return coord_transform_enum::indexing;
    }

    CK_TILE_HOST_DEVICE constexpr const auto& get_upper_lengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE constexpr void calculate_lower_index(LowIdx& idx_low,
                                                             const UpIdx& idx_up) const
    {
        static_assert(LowIdx::size() == 1 && UpIdx::size() == NDimUp,
                      "wrong! inconsistent # of dimension");
        iadaptor_.calculate_lower_index(idx_low, idx_up);
    }

    template <typename LowIdxDiff, typename UpIdxDiff, typename LowIdx, typename UpIdx>
    CK_TILE_HOST_DEVICE void update_lower_index(LowIdxDiff& idx_diff_low,
                                                const UpIdxDiff& idx_diff_up,
                                                LowIdx& idx_low,
                                                const UpIdx& idx_up) const
    {
        // TODO: nonthing changed here
        static_assert(LowIdxDiff::size() == 1 && UpIdxDiff::size() == NDimUp &&
                          LowIdx::size() == 1 && UpIdx::size() == NDimUp,
                      "wrong! inconsistent # of dimension");

        iadaptor_.update_lower_index(idx_diff_low, idx_diff_up, idx_low, idx_up);
    }

    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_always_mapped_to_valid_lower_index()
    {
        return true;
    }

    template <typename UpIdx>
    CK_TILE_HOST_DEVICE static constexpr bool
    is_valid_upper_index_mapped_to_valid_lower_index(const UpIdx& /* idx_up */)
    {
        return true;
    }

    CK_TILE_HOST_DEVICE static constexpr bool is_known_at_compile_time()
    {
        return ck_tile::is_known_at_compile_time<UpLengths>::value &&
               IndexingAdaptor::is_known_at_compile_time();
    }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("embed{");

        //
        printf("up_lengths_: ");
        print(up_lengths_);
        printf(", ");

        printf("}");
    }
};

//*******************************************************************************************************

template <typename LowLength>
CK_TILE_HOST_DEVICE constexpr auto make_pass_through_transform(const LowLength& low_length)
{
    return pass_through<LowLength>{low_length};
}

template <typename LowLength, typename LeftPad, typename RightPad, bool SkipIsValidCheck = false>
CK_TILE_HOST_DEVICE constexpr auto
make_pad_transform(const LowLength& low_length,
                   const LeftPad& left_pad,
                   const RightPad& right_pad,
                   bool_constant<SkipIsValidCheck> = bool_constant<false>{})
{
    return pad<LowLength, LeftPad, RightPad, SkipIsValidCheck>{low_length, left_pad, right_pad};
}

template <typename LowLength, typename LeftPadLength, bool SkipIsValidCheck = false>
CK_TILE_HOST_DEVICE constexpr auto
make_left_pad_transform(const LowLength& low_length,
                        const LeftPadLength& left_pad_,
                        bool_constant<SkipIsValidCheck> = bool_constant<false>{})
{
    return left_pad<LowLength, LeftPadLength, SkipIsValidCheck>{low_length, left_pad_};
}

template <typename LowLength, typename RightPadLength, bool SkipIsValidCheck = false>
CK_TILE_HOST_DEVICE constexpr auto
make_right_pad_transform(const LowLength& low_length,
                         const RightPadLength& right_pad_,
                         bool_constant<SkipIsValidCheck> = bool_constant<false>{})
{
    return right_pad<LowLength, RightPadLength, SkipIsValidCheck>{low_length, right_pad_};
}

template <typename UpLengths,
          typename Coefficients,
          typename std::enable_if<UpLengths::size() == Coefficients::size(), bool>::type = false>
CK_TILE_HOST_DEVICE constexpr auto make_embed_transform(const UpLengths& up_lengths,
                                                        const Coefficients& coefficients)
{
    return embed<UpLengths, Coefficients>{up_lengths, coefficients};
}

template <typename LowLengths>
CK_TILE_HOST_DEVICE constexpr auto
make_merge_transform_v2_magic_division(const LowLengths& low_lengths)
{
    return merge_v2_magic_division<LowLengths>{low_lengths};
}

template <typename LowLengths>
CK_TILE_HOST_DEVICE constexpr auto
make_merge_transform_v3_division_mod(const LowLengths& low_lengths)
{
    return merge_v3_division_mod<LowLengths>{low_lengths};
}

template <typename LowLengths>
CK_TILE_HOST_DEVICE constexpr auto make_merge_transform(const LowLengths& low_lengths)
{
    return make_merge_transform_v2_magic_division(low_lengths);
}

template <typename UpLengths, bool Use24BitIntegerCalculation = false>
CK_TILE_HOST_DEVICE constexpr auto
make_unmerge_transform(const UpLengths& up_lengths,
                       bool_constant<Use24BitIntegerCalculation> = bool_constant<false>{})
{
    return unmerge<UpLengths, Use24BitIntegerCalculation>{up_lengths};
}

template <typename LowerIndex>
CK_TILE_HOST_DEVICE constexpr auto make_freeze_transform(const LowerIndex& low_idx)
{
    return freeze<LowerIndex>{low_idx};
}

template <typename UpperIndex>
CK_TILE_HOST_DEVICE constexpr auto make_insert_transform(const UpperIndex& up_idx)
{
    return insert<UpperIndex>{up_idx};
}

template <typename UpLengths>
CK_TILE_HOST_DEVICE constexpr auto make_replicate_transform(const UpLengths& up_lengths)
{
    return replicate<UpLengths>{up_lengths};
}

template <typename LowLength, typename SliceBegin, typename SliceEnd>
CK_TILE_HOST_DEVICE constexpr auto make_slice_transform(const LowLength& low_length,
                                                        const SliceBegin& slice_begin,
                                                        const SliceEnd& slice_end)
{
    return slice<LowLength, SliceBegin, SliceEnd>{low_length, slice_begin, slice_end};
}

template <typename Modulus, typename UpLength>
CK_TILE_HOST_DEVICE constexpr auto make_modulo_transform(const Modulus& modulus,
                                                         const UpLength& up_length)
{
    return modulo<Modulus, UpLength>{modulus, up_length};
}

template <typename LowLengths>
CK_TILE_HOST_DEVICE constexpr auto make_xor_transform(const LowLengths& low_lengths)
{
    return xor_t<LowLengths>{low_lengths};
}

template <typename LowLength, typename OffsetLength>
CK_TILE_HOST_DEVICE constexpr auto make_offset_transform(const LowLength& low_length,
                                                         const OffsetLength& offset_length)
{
    return offset<LowLength, OffsetLength>{low_length, offset_length};
}

} // namespace ck_tile

#include "ck_tile/core/algorithm/indexing_adaptor.hpp"
namespace ck_tile {

template <typename UpLength, typename Indices>
CK_TILE_HOST_DEVICE constexpr auto make_indexing_transform(const UpLength& up_lengths,
                                                           const Indices& indices)
{
    // by default we use the simplest one
    return indexing<UpLength, indexing_adaptor_onshot_cached<remove_cvref_t<Indices>>>{
        up_lengths, indexing_adaptor_onshot_cached<remove_cvref_t<Indices>>{indices}};
}

template <typename UpLength, typename IndexingAdaptor>
CK_TILE_HOST_DEVICE constexpr auto
make_indexing_transform_with_adaptor(const UpLength& up_lengths, const IndexingAdaptor& iadaptor)
{
    return indexing<UpLength, IndexingAdaptor>{up_lengths, iadaptor};
}

} // namespace ck_tile
