// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.

#pragma once

// #include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/algorithm/space_filling_curve.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/container/thread_buffer.hpp"
#include "ck_tile/core/container/statically_indexed_array.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

constexpr int DS_READ_TR_SIZE()
{
    return 8; // Literal constant, evaluated at compile time
}

namespace util {
template <typename Suffix, typename Sequence>
struct is_sequence_suffix
{
    static constexpr bool size_check = (Suffix::size() <= Sequence::size());

    static constexpr index_t start_pos = Sequence::size() - Suffix::size();
    using extract_indices = typename arithmetic_sequence_gen<start_pos, Sequence::size(), 1>::type;

    static constexpr bool value =
        size_check && (Suffix{} == decltype(Sequence::extract(extract_indices{})){});
};

template <index_t... Xs>
struct is_sequence_suffix<sequence<>, sequence<Xs...>>
{
    static constexpr bool value = true;
};

template <typename Suffix, typename Sequence>
constexpr bool is_sequence_suffix_v = is_sequence_suffix<Suffix, Sequence>::value;

} // namespace util

// Default policy: Retains original 2D transpose behavior
template <typename DataType>
struct DefaultTranspose
{
    template <index_t LaneGroupSize>
    struct Quad16
    {
        static_assert(LaneGroupSize == 64 || LaneGroupSize == 32 || LaneGroupSize == 16,
                      "LaneGroupSize must be 64, 32, or 16");
        using InputEncoding =
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<4>, sequence<LaneGroupSize / 16, 4, 4>>,
                                       tuple<sequence<2, 1, 2>>,
                                       tuple<sequence<0, 0, 1>>,
                                       sequence<2>,
                                       sequence<2>>;

        using OutputEncoding =
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<LaneGroupSize>, sequence<4>>,
                                       tuple<sequence<1>>,
                                       tuple<sequence<0>>,
                                       sequence<2>,
                                       sequence<0>>;
    };

    template <index_t LaneGroupSize>
    struct Quad8
    {
        static_assert(LaneGroupSize == 64 || LaneGroupSize == 32 || LaneGroupSize == 16,
                      "LaneGroupSize must be 64, 32, or 16");
        using InputEncoding =
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<8>, sequence<LaneGroupSize / 16, 2, 8>>,
                                       tuple<sequence<2, 1, 2>>,
                                       tuple<sequence<0, 0, 1>>,
                                       sequence<2>,
                                       sequence<2>>;

        using OutputEncoding =
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<LaneGroupSize>, sequence<8>>,
                                       tuple<sequence<1>>,
                                       tuple<sequence<0>>,
                                       sequence<2>,
                                       sequence<0>>;
    };

    // Select based on data size
    template <index_t LaneGroupSize>
    using QuadInputEncoding = std::conditional_t<sizeof(DataType) == 2,
                                                 typename Quad16<LaneGroupSize>::InputEncoding,
                                                 typename Quad8<LaneGroupSize>::InputEncoding>;

    template <index_t LaneGroupSize>
    using QuadOutputEncoding = std::conditional_t<sizeof(DataType) == 2,
                                                  typename Quad16<LaneGroupSize>::OutputEncoding,
                                                  typename Quad8<LaneGroupSize>::OutputEncoding>;

    // Always swap last two dimensions
    static constexpr auto transpose_dims = sequence<1, 0>{};

    // Programmable: Element grouping function
    static constexpr auto group_func = [](auto idx) {
        return idx; // Identity mapping
    };

    template <typename InDstrEncode, bool ReverseDirection, index_t LaneGroupSize>
    struct ValidationTraitsImpl
    {
        using QuadEncoding             = std::conditional_t<ReverseDirection,
                                                            QuadOutputEncoding<LaneGroupSize>,
                                                            QuadInputEncoding<LaneGroupSize>>;
        static constexpr auto I0       = number<0>{};
        static constexpr auto I1       = number<1>{};
        static constexpr auto input_hs = InDstrEncode::hs_lengthss_;
        static constexpr auto quad_hs  = QuadEncoding::hs_lengthss_;
        // 1. Must be 2D tensor
        static constexpr bool dims_valid = (InDstrEncode::NDimX == 2);
        // 2. Quad pattern must be suffix of input pattern
        static constexpr bool suffix_valid_dim0 =
            util::is_sequence_suffix_v<decltype(quad_hs[I0]), decltype(input_hs[I0])>;
        static constexpr bool suffix_valid_dim1 =
            util::is_sequence_suffix_v<decltype(quad_hs[I1]), decltype(input_hs[I1])>;

        // 3. PS→RHS mapping constraints
        static constexpr auto input_ps_major = InDstrEncode::ps_to_rhss_major_;
        static constexpr auto input_ps_minor = InDstrEncode::ps_to_rhss_minor_;

        static constexpr auto quad_ps_major0 = QuadEncoding::ps_to_rhss_major_[I0];
        static constexpr auto quad_ps_minor0 = QuadEncoding::ps_to_rhss_minor_[I0];

        static constexpr auto input_ps_major_last =
            input_ps_major[number<input_ps_major.size() - 1>{}];
        static constexpr auto input_ps_minor_last =
            input_ps_minor[number<input_ps_minor.size() - 1>{}];

        using psys_offset = ck_tile::sequence<input_hs[I0].size() - quad_hs[I0].size(),
                                              input_hs[I1].size() - quad_hs[I1].size()>;
        static constexpr auto shifted_quad_ps_minor0 = generate_sequence_v2(
            [](auto i) {
                return number<quad_ps_minor0[i] + psys_offset{}[quad_ps_major0[i] - 1]>{};
            },
            number<quad_ps_minor0.size()>{});

        static constexpr bool ps_mapping_valid =
            util::is_sequence_suffix_v<decltype(quad_ps_major0), decltype(input_ps_major_last)> &&
            util::is_sequence_suffix_v<decltype(shifted_quad_ps_minor0),
                                       decltype(input_ps_minor_last)>;

        // 4. YS→RHS mapping constraints
        static constexpr auto input_ys_major = InDstrEncode::ys_to_rhs_major_;
        static constexpr auto input_ys_minor = InDstrEncode::ys_to_rhs_minor_;
        static constexpr auto quad_ys_major  = QuadEncoding::ys_to_rhs_major_;
        static constexpr auto quad_ys_minor  = QuadEncoding::ys_to_rhs_minor_;

        static_assert(quad_ys_major.size() == 1 && quad_ys_minor.size() == 1,
                      "YS->RHS mapping must be single dimension");
        static_assert(quad_ys_major.back() == 2 && quad_ys_minor.back() == quad_hs[I1].size() - 1,
                      "YS->RHS mapping must be the last dimension");
        static constexpr bool ys_mapping_valid =
            (input_ys_major.back() == 2) && (input_ys_minor.back() == input_hs[I1].size() - 1);

        static constexpr bool value = dims_valid && suffix_valid_dim0 && suffix_valid_dim1 &&
                                      ps_mapping_valid && ys_mapping_valid;
    };

    template <typename InDstrEncode, bool ReverseDirection = false>
    struct ValidationTraits
    {
        static constexpr bool value =
            ValidationTraitsImpl<InDstrEncode, ReverseDirection, 64>::value ||
            ValidationTraitsImpl<InDstrEncode, ReverseDirection, 32>::value ||
            ValidationTraitsImpl<InDstrEncode, ReverseDirection, 16>::value;
        static constexpr index_t LaneGroupSize =
            ValidationTraitsImpl<InDstrEncode, ReverseDirection, 64>::value   ? 64
            : ValidationTraitsImpl<InDstrEncode, ReverseDirection, 32>::value ? 32
            : ValidationTraitsImpl<InDstrEncode, ReverseDirection, 16>::value ? 16
                                                                              : 0;
    };
};
template <typename TileDistribution_, typename DataType_, typename Policy>
struct TransposeTileDistrChecker
{
    using InDstrEncode = typename remove_cvref_t<TileDistribution_>::DstrEncode;

    using Validator = typename Policy::template ValidationTraits<InDstrEncode>;

    static constexpr bool distr_encoding_valid = Validator::value;
};

// this is used to generate the transposed output tile distribution encoding
// based on the input tile distribution encoding
template <typename TileDistributionEncoding_,
          typename DataType_,
          typename Policy       = DefaultTranspose<DataType_>,
          bool ReverseDirection = false>
struct TransposeTileDistributionTraits
{
    using InDstrEncode                      = remove_cvref_t<TileDistributionEncoding_>;
    static constexpr auto input_hs_lengthss = InDstrEncode::hs_lengthss_;
    static constexpr index_t LaneGroupSize =
        Policy::template ValidationTraits<InDstrEncode, ReverseDirection>::LaneGroupSize;
    static_assert(Policy::template ValidationTraits<InDstrEncode, ReverseDirection>::value,
                  "The input tile distribution encoding is not valid for transpose!");

    using QuadInputEncoding  = std::conditional_t< //
        ReverseDirection,
        typename Policy::template QuadOutputEncoding<LaneGroupSize>,
        typename Policy::template QuadInputEncoding<LaneGroupSize>>;
    using QuadOutputEncoding = std::conditional_t< //
        ReverseDirection,
        typename Policy::template QuadInputEncoding<LaneGroupSize>,
        typename Policy::template QuadOutputEncoding<LaneGroupSize>>;

    static constexpr auto quad_input_hs_lengthss  = QuadInputEncoding::hs_lengthss_;
    static constexpr auto quad_output_hs_lengthss = QuadOutputEncoding::hs_lengthss_;

    static constexpr auto input_ps_to_rhss_major = InDstrEncode::ps_to_rhss_major_;
    static constexpr auto input_ps_to_rhss_minor = InDstrEncode::ps_to_rhss_minor_;
    static constexpr auto input_ys_to_rhs_major  = InDstrEncode::ys_to_rhs_major_;
    static constexpr auto input_ys_to_rhs_minor  = InDstrEncode::ys_to_rhs_minor_;

    static constexpr auto I0                            = number<0>{};
    static constexpr auto quad_input_ps_to_rhss_major0  = QuadInputEncoding::ps_to_rhss_major_[I0];
    static constexpr auto quad_input_ps_to_rhss_minor0  = QuadInputEncoding::ps_to_rhss_minor_[I0];
    static constexpr auto quad_output_ps_to_rhss_major0 = QuadOutputEncoding::ps_to_rhss_major_[I0];
    static constexpr auto quad_output_ps_to_rhss_minor0 = QuadOutputEncoding::ps_to_rhss_minor_[I0];
    static constexpr auto quad_output_ys_to_rhs_major   = QuadOutputEncoding::ys_to_rhs_major_;
    static constexpr auto quad_output_ys_to_rhs_minor   = QuadOutputEncoding::ys_to_rhs_minor_;

    static constexpr index_t dim0 = Policy::transpose_dims[0];
    static constexpr index_t dim1 = Policy::transpose_dims[1];

    static constexpr auto swap_one_and_two = [](const index_t idx) {
        return (idx == 1) ? 2 : (idx == 2) ? 1 : idx;
    };

    // for transpose load
    // remove the quad_input_hs_lengthss from the input_hs_lengthss for each dimension and reverse
    // dims and append the quad_output_hs_lengthss to the end of each dimension
    static constexpr auto outer_hs_lengthss = generate_tuple(
        [](auto i) {
            constexpr auto input_i   = input_hs_lengthss[i];
            constexpr auto outer_len = input_i.size() - quad_input_hs_lengthss[i].size();
            return typename sequence_split<decltype(input_i), outer_len>::left_type{};
        },
        number<InDstrEncode::NDimX>{});
    static constexpr auto reversed_outer_hs_lengthss = tuple_reverse(outer_hs_lengthss);
    static constexpr auto dst_out_hs_lengthss        = generate_tuple(
        [](auto i) {
            auto outer_i = reversed_outer_hs_lengthss[i];
            // append the reversed quad output hs lengths to the outer hs lengths
            return outer_i.push_back(quad_output_hs_lengthss[i]);
        },
        number<InDstrEncode::NDimX>{});

    // for PS→RHS mapping(both major and minor), we need to modify the last element (which is for
    // thread distr) of the major sequence
    static constexpr auto dst_ps_to_rhss_major = generate_tuple(
        // for major because of dst_out_hs_lengthss is reversed, this index also need to be reversed
        [](auto i) {
            if constexpr(i == input_ps_to_rhss_major.size() - 1)
            {
                constexpr auto current_size             = input_ps_to_rhss_major[i].size();
                constexpr auto reduce_size              = quad_input_ps_to_rhss_major0.size();
                constexpr auto quad_out                 = quad_output_ps_to_rhss_major0;
                constexpr auto reduced_ps_to_rhss_major = input_ps_to_rhss_major[i].extract(
                    typename arithmetic_sequence_gen<0, current_size - reduce_size, 1>::type{});
                return reduced_ps_to_rhss_major.transform(swap_one_and_two).push_back(quad_out);
            }
            else
            {
                // For all other sequences (i.e. warp), keep them unchanged
                return input_ps_to_rhss_major[i].transform(swap_one_and_two);
            }
        },
        number<input_ps_to_rhss_major.size()>{});

    static constexpr auto quad_idx_offset =
        transform_tuples([](auto x) { return number<x.size()>{}; }, reversed_outer_hs_lengthss);

    // minus 1 because RsLength is not counted
    static constexpr auto quad_output_ps_minor_offset = to_sequence(generate_tuple_for(
        [](auto x) { return quad_idx_offset[number<x - 1>{}]; }, quad_output_ps_to_rhss_major0));
    static constexpr auto quad_output_ys_minor_offset = to_sequence(generate_tuple_for(
        [](auto x) { return quad_idx_offset[number<x - 1>{}]; }, quad_output_ys_to_rhs_major));

    static constexpr auto dst_ps_to_rhss_minor = generate_tuple(
        [](auto i) {
            constexpr auto input_i = input_ps_to_rhss_minor[i];
            if constexpr(i == input_ps_to_rhss_minor.size() - 1)
            {
                constexpr auto outer_len = input_i.size() - quad_input_ps_to_rhss_minor0.size();
                constexpr auto outer_ps =
                    typename sequence_split<decltype(input_i), outer_len>::left_type{};

                return outer_ps.push_back(quad_output_ps_minor_offset +
                                          quad_output_ps_to_rhss_minor0);
            }
            else
            {
                // For all other sequences, keep them unchanged
                return input_i;
            }
        },
        number<input_ps_to_rhss_minor.size()>{});

    static constexpr auto outer_input_ys_to_rhs_major = input_ys_to_rhs_major.pop_back();

    // for major because of dst_out_hs_lengthss is reversed, this index also need to be reversed
    static constexpr auto dst_ys_to_rhs_major =
        outer_input_ys_to_rhs_major.transform(swap_one_and_two).push_back(number<2>{});

    static constexpr auto dst_ys_to_rhs_minor = input_ys_to_rhs_minor.pop_back().push_back(
        number<(quad_output_ys_minor_offset + quad_output_ys_to_rhs_minor)[I0]>{});

    using TransposedDstrEncode =
        tile_distribution_encoding<typename InDstrEncode::RsLengths,
                                   remove_cvref_t<decltype(dst_out_hs_lengthss)>,
                                   remove_cvref_t<decltype(dst_ps_to_rhss_major)>,
                                   remove_cvref_t<decltype(dst_ps_to_rhss_minor)>,
                                   remove_cvref_t<decltype(dst_ys_to_rhs_major)>,
                                   remove_cvref_t<decltype(dst_ys_to_rhs_minor)>>;
};

template <typename TileDistributionEncoding_,
          typename DataType_,
          typename Policy = DefaultTranspose<DataType_>>
using OutputTileDistributionTraits =
    TransposeTileDistributionTraits<TileDistributionEncoding_, DataType_, Policy, false>;
template <typename TileDistributionEncoding_,
          typename DataType_,
          typename Policy = DefaultTranspose<DataType_>>
using InputTileDistributionTraits =
    TransposeTileDistributionTraits<TileDistributionEncoding_, DataType_, Policy, true>;

template <typename InnerEncode,
          index_t kLeadIterPerWarp,
          index_t kSecondIterPerWarp,
          index_t kLeadNumWarps,
          index_t kSecondNumWarps>
CK_TILE_HOST_DEVICE constexpr auto InputTileDistributionEncoding()
{
    constexpr auto block_outer_dst_encoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<kSecondIterPerWarp, kSecondNumWarps>,
                                         sequence<kLeadIterPerWarp, kLeadNumWarps>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<1, 1>>,
                                   sequence<2, 1>,
                                   sequence<0, 0>>{};
    constexpr auto blk_distr_encode =
        detail::make_embed_tile_distribution_encoding(block_outer_dst_encoding, InnerEncode{});

    return blk_distr_encode;
}

/**
 * @brief transpose loads tile from a tensor and returns the resulting tensor with a new
 * (transposed) tile distribution. use SFINAE to ensure the tile distribution encoding is valid.
 *
 * This function is intended for use with statically distributed tensor tiles, where the input
 * and output tile distributions differ due to the transpose operation. It ensures that the
 * element space size and vector length remain consistent between the input and output
 * distributions.
 *
 * @tparam BottomTensorView_      The type of the bottom tensor view.
 * @tparam WindowLengths_         The type representing the window lengths.
 * @tparam TileDistribution_      The type representing the tile distribution.
 * @tparam NumCoord               The number of coordinates (dimensions).
 * @tparam Policy                 The transpose policy to use (defaults to DefaultTranspose).
 * the last is SFINAE to ensure the tile distribution encoding is valid.
 *
 * @param tile_window             The tile window with static distribution to load and transpose.
 *
 * @return A statically distributed tensor containing the transposed tile data.
 *
 * @note
 * - The function uses compile-time checks to ensure the input and output tile distributions
 *   are compatible in terms of element space size and vector length.
 * - The transpose operation is performed according to the specified Policy.
 */
template <
    typename BottomTensorView_,
    typename WindowLengths_,
    typename TileDistribution_,
    index_t NumCoord,
    typename Policy = DefaultTranspose<typename BottomTensorView_::DataType>,
    typename        = std::enable_if_t<TransposeTileDistrChecker<TileDistribution_,
                                                                 typename BottomTensorView_::DataType,
                                                                 Policy>::distr_encoding_valid,
                                       Policy>>
CK_TILE_DEVICE auto
load_tile_transpose(const tile_window_with_static_distribution<BottomTensorView_,
                                                               WindowLengths_,
                                                               TileDistribution_,
                                                               NumCoord>& tile_window)
{
    using OutTileDstrEncode = typename OutputTileDistributionTraits<
        typename TileDistribution_::DstrEncode,
        typename BottomTensorView_::DataType>::TransposedDstrEncode;
    auto out_tensor = make_static_distributed_tensor<typename BottomTensorView_::DataType>(
        make_static_tile_distribution(OutTileDstrEncode{}));
    auto trans_tensor           = tile_window.template load_transpose<Policy>();
    constexpr auto input_distr  = TileDistribution_{};
    constexpr auto output_distr = make_static_tile_distribution(OutTileDstrEncode{});

    constexpr auto y_in_desc  = input_distr.get_ys_to_d_descriptor();
    constexpr auto y_out_desc = output_distr.get_ys_to_d_descriptor();

    constexpr index_t NDimYIn  = input_distr.get_num_of_dimension_y();
    constexpr index_t NDimYOut = output_distr.get_num_of_dimension_y();

    constexpr auto y_in_lengths  = to_sequence(y_in_desc.get_lengths());
    constexpr auto y_out_lengths = to_sequence(y_out_desc.get_lengths());

    constexpr auto y_in_element_space_size  = y_in_desc.get_element_space_size();
    constexpr auto y_out_element_space_size = y_out_desc.get_element_space_size();
    static_assert(y_in_element_space_size == y_out_element_space_size,
                  "the element space size is not the same!");
    static_assert(y_in_lengths[NDimYIn - 1] == y_out_lengths[NDimYOut - 1],
                  "the vector length is not the same!");
    constexpr index_t vecLoadSize = y_in_lengths[NDimYIn - 1];
    constexpr index_t num_of_access =
        reduce_on_sequence(y_in_lengths, multiplies{}, number<1>{}) / vecLoadSize;

    using DataVec = array<typename BottomTensorView_::DataType, vecLoadSize>;
    static_for<0, num_of_access, 1>{}([&](auto iAccess) {
        out_tensor.get_thread_buffer().template set_as<DataVec>(
            number<iAccess>{},
            trans_tensor.get_thread_buffer().template get_as<DataVec>(number<iAccess>{}));
    });

    return out_tensor;
}

} // namespace ck_tile
