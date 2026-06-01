// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <type_traits>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host/ranges.hpp"

namespace ck_tile {

/** @brief Maximum number of error values to display when checking errors */
constexpr int ERROR_DETAIL_LIMIT = 5;

/** @brief 8-bit floating point type */
using F8 = ck_tile::fp8_t;
/** @brief 8-bit brain floating point type */
using BF8 = ck_tile::bf8_t;
/** @brief 16-bit floating point (half precision) type */
using F16 = ck_tile::half_t;
/** @brief 16-bit brain floating point type */
using BF16 = ck_tile::bf16_t;
/** @brief 32-bit floating point (single precision) type */
using F32 = float;
/** @brief 8-bit signed integer type */
using I8 = int8_t;
/** @brief 32-bit signed integer type */
using I32 = int32_t;

/**
 * @brief Calculate relative error threshold for numerical comparisons
 *
 * Calculates the relative error threshold based on the mantissa bits and characteristics
 * of the data types involved in the computation.
 *
 * @tparam ComputeDataType Type used for computation
 * @tparam OutDataType Type used for output
 * @tparam AccDataType Type used for accumulation (defaults to ComputeDataType)
 * @param number_of_accumulations Number of accumulation operations performed
 * @return Relative error threshold based on data type characteristics
 */
template <typename ComputeDataType, typename OutDataType, typename AccDataType = ComputeDataType>
CK_TILE_HOST double get_relative_threshold(const int number_of_accumulations = 1)
{

    static_assert(
        is_any_of<ComputeDataType, F8, BF8, F16, BF16, F32, pk_int4_t, I8, I32, int>::value,
        "Warning: Unhandled ComputeDataType for setting up the relative threshold!");

    double compute_error = 0;
    if constexpr(is_any_of<ComputeDataType, pk_int4_t, I8, I32, int>::value)
    {
        return 0;
    }
    else
    {
        compute_error = std::pow(2, -numeric_traits<ComputeDataType>::mant) * 0.5;
    }

    static_assert(is_any_of<OutDataType, F8, BF8, F16, BF16, F32, pk_int4_t, I8, I32, int>::value,
                  "Warning: Unhandled OutDataType for setting up the relative threshold!");

    double output_error = 0;
    if constexpr(is_any_of<OutDataType, pk_int4_t, I8, I32, int>::value)
    {
        return 0;
    }
    else
    {
        output_error = std::pow(2, -numeric_traits<OutDataType>::mant) * 0.5;
    }
    double midway_error = std::max(compute_error, output_error);

    static_assert(is_any_of<AccDataType, F8, BF8, F16, BF16, F32, pk_int4_t, I8, I32, int>::value,
                  "Warning: Unhandled AccDataType for setting up the relative threshold!");

    double acc_error = 0;
    if constexpr(is_any_of<AccDataType, pk_int4_t, I8, I32, int>::value)
    {
        return 0;
    }
    else
    {
        acc_error = std::pow(2, -numeric_traits<AccDataType>::mant) * 0.5 * number_of_accumulations;
    }
    return std::max(acc_error, midway_error);
}

/**
 * @brief Calculate absolute error threshold for numerical comparisons
 *
 * Calculates the absolute error threshold based on the maximum possible value and
 * the characteristics of the data types involved in the computation.
 *
 * @tparam ComputeDataType Type used for computation
 * @tparam OutDataType Type used for output
 * @tparam AccDataType Type used for accumulation (defaults to ComputeDataType)
 * @param max_possible_num Maximum possible value in the computation
 * @param number_of_accumulations Number of accumulation operations performed
 * @return Absolute error threshold based on data type characteristics and maximum value
 */
template <typename ComputeDataType, typename OutDataType, typename AccDataType = ComputeDataType>
CK_TILE_HOST double get_absolute_threshold(const double max_possible_num,
                                           const int number_of_accumulations = 1)
{

    static_assert(
        is_any_of<ComputeDataType, F8, BF8, F16, BF16, F32, pk_int4_t, I8, I32, int>::value,
        "Warning: Unhandled ComputeDataType for setting up the absolute threshold!");

    auto expo            = std::log2(std::abs(max_possible_num));
    double compute_error = 0;
    if constexpr(is_any_of<ComputeDataType, pk_int4_t, I8, I32, int>::value)
    {
        return 0;
    }
    else
    {
        compute_error = std::pow(2, expo - numeric_traits<ComputeDataType>::mant) * 0.5;
    }

    static_assert(is_any_of<OutDataType, F8, BF8, F16, BF16, F32, pk_int4_t, I8, I32, int>::value,
                  "Warning: Unhandled OutDataType for setting up the absolute threshold!");

    double output_error = 0;
    if constexpr(is_any_of<OutDataType, pk_int4_t, I8, I32, int>::value)
    {
        return 0;
    }
    else
    {
        output_error = std::pow(2, expo - numeric_traits<OutDataType>::mant) * 0.5;
    }
    double midway_error = std::max(compute_error, output_error);

    static_assert(is_any_of<AccDataType, F8, BF8, F16, BF16, F32, pk_int4_t, I8, I32, int>::value,
                  "Warning: Unhandled AccDataType for setting up the absolute threshold!");

    double acc_error = 0;
    if constexpr(is_any_of<AccDataType, pk_int4_t, I8, I32, int>::value)
    {
        return 0;
    }
    else
    {
        acc_error =
            std::pow(2, expo - numeric_traits<AccDataType>::mant) * 0.5 * number_of_accumulations;
    }
    return std::max(acc_error, midway_error);
}

/**
 * @brief Stream operator overload for vector output
 *
 * Provides a formatted string representation of a vector, useful for debugging and logging.
 *
 * @tparam T Type of vector elements
 * @param os Output stream
 * @param v Vector to output
 * @return Reference to the output stream
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    using size_type = typename std::vector<T>::size_type;

    os << "[";
    for(size_type idx = 0; idx < v.size(); ++idx)
    {
        if(0 < idx)
        {
            os << ", ";
        }
        os << v[idx];
    }
    return os << "]";
}

/**
 * @brief Check for size mismatch between output and reference ranges
 *
 * Verifies that the output and reference ranges are the same size.
 *
 * @tparam Range Type of output range
 * @tparam RefRange Type of reference range
 * @param out Output range to check
 * @param ref Reference range to check against
 * @param msg Error message to display if sizes mismatch
 * @return True if sizes mismatch, false otherwise
 */
template <typename Range, typename RefRange>
CK_TILE_HOST bool check_size_mismatch(const Range& out,
                                      const RefRange& ref,
                                      const std::string& msg = "Error: Incorrect results!")
{
    if(out.size() != ref.size())
    {
        std::cerr << msg << " out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return true;
    }
    return false;
}

/**
 * @brief Report error statistics for numerical comparisons
 *
 * Outputs statistics about numerical comparison errors including count and maximum error.
 *
 * @param err_count Number of errors found
 * @param max_err Maximum error value encountered
 * @param total_size Total number of elements compared
 */
CK_TILE_HOST void report_error_stats(int err_count, double max_err, std::size_t total_size)
{
    const float error_percent =
        static_cast<float>(err_count) / static_cast<float>(total_size) * 100.f;
    std::cerr << "max err: " << max_err;
    std::cerr << ", number of errors: " << err_count;
    std::cerr << ", " << error_percent << "% wrong values" << std::endl;
}

/**
 * @brief Check errors between floating point ranges using the specified tolerances.
 *
 * Compares two ranges of floating point values within specified relative and absolute tolerances.
 * This overload handles standard floating point types except half precision floating point.
 *
 * @tparam Range Type of output range
 * @tparam RefRange Type of reference range
 * @param out Output range to check
 * @param ref Reference range to check against
 * @param msg Error message to display if check fails
 * @param rtol Relative tolerance
 * @param atol Absolute tolerance
 * @param allow_infinity_ref Whether to allow infinity in reference values
 * @return True if check passes, false otherwise
 */
template <typename Range, typename RefRange>
typename std::enable_if<
    std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
        std::is_floating_point_v<ranges::range_value_t<Range>> &&
        !std::is_same_v<ranges::range_value_t<Range>, half_t>,
    bool>::type CK_TILE_HOST
check_err(const Range& out,
          const RefRange& ref,
          const std::string& msg  = "Error: Incorrect results!",
          double rtol             = 1e-5,
          double atol             = 3e-6,
          bool allow_infinity_ref = false)
{

    if(check_size_mismatch(out, ref, msg))
        return false;

    const auto is_infinity_error = [=](auto o, auto r) {
        const bool either_not_finite = !std::isfinite(o) || !std::isfinite(r);
        const bool both_infinite_and_same =
            std::isinf(o) && std::isinf(r) && (bit_cast<uint64_t>(o) == bit_cast<uint64_t>(r));

        return either_not_finite && !(allow_infinity_ref && both_infinite_and_same);
    };

    bool res{true};
    int err_count  = 0;
    double err     = 0;
    double max_err = std::numeric_limits<double>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const double o = *std::next(std::begin(out), i);
        const double r = *std::next(std::begin(ref), i);
        err            = std::abs(o - r);
        if(err > atol + rtol * std::abs(r) || is_infinity_error(o, r))
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < ERROR_DETAIL_LIMIT)
            {
                std::cerr << msg << std::setw(12) << std::setprecision(7) << " out[" << i
                          << "] != ref[" << i << "]: " << o << " != " << r << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        report_error_stats(err_count, max_err, ref.size());
    }
    return res;
}

/**
 * @brief Check errors between floating point ranges using the specified tolerances
 *
 * Compares two ranges of brain floating point values within specified relative and absolute
 * tolerances.
 *
 * @tparam Range Type of output range
 * @tparam RefRange Type of reference range
 * @param out Output range to check
 * @param ref Reference range to check against
 * @param msg Error message to display if check fails
 * @param rtol Relative tolerance
 * @param atol Absolute tolerance
 * @param allow_infinity_ref Whether to allow infinity in reference values
 * @return True if check passes, false otherwise
 */
template <typename Range, typename RefRange>
typename std::enable_if<
    std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
        std::is_same_v<ranges::range_value_t<Range>, bf16_t>,
    bool>::type CK_TILE_HOST
check_err(const Range& out,
          const RefRange& ref,
          const std::string& msg  = "Error: Incorrect results!",
          double rtol             = 1e-3,
          double atol             = 1e-3,
          bool allow_infinity_ref = false)
{
    if(check_size_mismatch(out, ref, msg))
        return false;

    const auto is_infinity_error = [=](auto o, auto r) {
        const bool either_not_finite = !std::isfinite(o) || !std::isfinite(r);
        const bool both_infinite_and_same =
            std::isinf(o) && std::isinf(r) && (bit_cast<uint64_t>(o) == bit_cast<uint64_t>(r));

        return either_not_finite && !(allow_infinity_ref && both_infinite_and_same);
    };

    bool res{true};
    int err_count = 0;
    double err    = 0;
    // TODO: This is a hack. We should have proper specialization for bf16_t data type.
    double max_err = std::numeric_limits<float>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const double o = type_convert<float>(*std::next(std::begin(out), i));
        const double r = type_convert<float>(*std::next(std::begin(ref), i));
        err            = std::abs(o - r);
        if(err > atol + rtol * std::abs(r) || is_infinity_error(o, r))
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < ERROR_DETAIL_LIMIT)
            {
                std::cerr << msg << std::setw(12) << std::setprecision(7) << " out[" << i
                          << "] != ref[" << i << "]: " << o << " != " << r << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        report_error_stats(err_count, max_err, ref.size());
    }
    return res;
}

/**
 * @brief Check errors between half precision floating point ranges
 *
 * Compares two ranges of half precision floating point values within specified tolerances.
 * This specialization handles the specific requirements and characteristics of half precision
 * floating point comparisons.
 *
 * @tparam Range Type of output range
 * @tparam RefRange Type of reference range
 * @param out Output range to check
 * @param ref Reference range to check against
 * @param msg Error message to display if check fails
 * @param rtol Relative tolerance
 * @param atol Absolute tolerance
 * @param allow_infinity_ref Whether to allow infinity in reference values
 * @return True if check passes, false otherwise
 */
template <typename Range, typename RefRange>
typename std::enable_if<
    std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
        std::is_same_v<ranges::range_value_t<Range>, half_t>,
    bool>::type CK_TILE_HOST
check_err(const Range& out,
          const RefRange& ref,
          const std::string& msg  = "Error: Incorrect results!",
          double rtol             = 1e-3,
          double atol             = 1e-3,
          bool allow_infinity_ref = false)
{
    if(check_size_mismatch(out, ref, msg))
        return false;

    const auto is_infinity_error = [=](auto o, auto r) {
        const bool either_not_finite = !std::isfinite(o) || !std::isfinite(r);
        const bool both_infinite_and_same =
            std::isinf(o) && std::isinf(r) && (bit_cast<uint64_t>(o) == bit_cast<uint64_t>(r));

        return either_not_finite && !(allow_infinity_ref && both_infinite_and_same);
    };

    bool res{true};
    int err_count  = 0;
    double err     = 0;
    double max_err = static_cast<double>(std::numeric_limits<ranges::range_value_t<Range>>::min());
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const double o = type_convert<float>(*std::next(std::begin(out), i));
        const double r = type_convert<float>(*std::next(std::begin(ref), i));
        err            = std::abs(o - r);
        if(err > atol + rtol * std::abs(r) || is_infinity_error(o, r))
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < ERROR_DETAIL_LIMIT)
            {
                std::cerr << msg << std::setw(12) << std::setprecision(7) << " out[" << i
                          << "] != ref[" << i << "]: " << o << " != " << r << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        report_error_stats(err_count, max_err, ref.size());
    }
    return res;
}

/**
 * @brief Check errors between integer ranges
 *
 * Compares two ranges of integer values with an absolute tolerance.
 * This specialization handles integer types and optionally int4_t when the
 * experimental bit int extension is enabled.
 *
 * @tparam Range Type of output range
 * @tparam RefRange Type of reference range
 * @param out Output range to check
 * @param ref Reference range to check against
 * @param msg Error message to display if check fails
 * @param atol Absolute tolerance
 * @return True if check passes, false otherwise
 */
template <typename Range, typename RefRange>
std::enable_if_t<(std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
                  std::is_integral_v<ranges::range_value_t<Range>> &&
                  !std::is_same_v<ranges::range_value_t<Range>, bf16_t>)
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
                     || std::is_same_v<ranges::range_value_t<Range>, int4_t>
#endif
                 ,
                 bool>
    CK_TILE_HOST check_err(const Range& out,
                           const RefRange& ref,
                           const std::string& msg = "Error: Incorrect results!",
                           double                 = 0,
                           double atol            = 0)
{
    if(check_size_mismatch(out, ref, msg))
        return false;

    bool res{true};
    int err_count   = 0;
    int64_t err     = 0;
    int64_t max_err = std::numeric_limits<int64_t>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const int64_t o = *std::next(std::begin(out), i);
        const int64_t r = *std::next(std::begin(ref), i);
        err             = std::abs(o - r);

        if(err > atol)
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < ERROR_DETAIL_LIMIT)
            {
                std::cerr << msg << " out[" << i << "] != ref[" << i << "]: " << o << " != " << r
                          << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        report_error_stats(err_count, static_cast<double>(max_err), ref.size());
    }
    return res;
}

/**
 * @brief Check errors between FP8 ranges
 *
 * Specialized comparison for 8-bit floating point values that takes into account
 * the unique characteristics and limitations of FP8 arithmetic, including
 * rounding point distances and special handling of infinity values.
 *
 * @tparam Range Type of output range
 * @tparam RefRange Type of reference range
 * @param out Output range to check
 * @param ref Reference range to check against
 * @param msg Error message to display if check fails
 * @param max_rounding_point_distance Maximum allowed distance between rounding points
 * @param atol Absolute tolerance
 * @param allow_infinity_ref Whether to allow infinity in reference values
 * @return True if check passes, false otherwise
 */
template <typename Range, typename RefRange>
std::enable_if_t<(std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
                  std::is_same_v<ranges::range_value_t<Range>, fp8_t>),
                 bool>
    CK_TILE_HOST check_err(const Range& out,
                           const RefRange& ref,
                           const std::string& msg               = "Error: Incorrect results!",
                           unsigned max_rounding_point_distance = 1,
                           double atol                          = 1e-1,
                           bool allow_infinity_ref              = false)
{
    if(check_size_mismatch(out, ref, msg))
        return false;

    const auto is_infinity_error = [=](auto o, auto r) {
        const bool either_not_finite = !std::isfinite(o) || !std::isfinite(r);
        const bool both_infinite_and_same =
            std::isinf(o) && std::isinf(r) && (bit_cast<uint64_t>(o) == bit_cast<uint64_t>(r));

        return either_not_finite && !(allow_infinity_ref && both_infinite_and_same);
    };

    static const auto get_rounding_point_distance = [](fp8_t o, fp8_t r) -> unsigned {
        static const auto get_sign_bit = [](fp8_t v) -> bool {
            return 0x80 & bit_cast<uint8_t>(v);
        };

        if(get_sign_bit(o) ^ get_sign_bit(r))
        {
            return std::numeric_limits<unsigned>::max();
        }
        else
        {
            return std::abs(bit_cast<int8_t>(o) - bit_cast<int8_t>(r));
        }
    };

    bool res{true};
    int err_count  = 0;
    double err     = 0;
    double max_err = std::numeric_limits<float>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const fp8_t o_fp8   = *std::next(std::begin(out), i);
        const fp8_t r_fp8   = *std::next(std::begin(ref), i);
        const double o_fp64 = type_convert<float>(o_fp8);
        const double r_fp64 = type_convert<float>(r_fp8);
        err                 = std::abs(o_fp64 - r_fp64);
        if(!(less_equal<double>{}(err, atol) ||
             get_rounding_point_distance(o_fp8, r_fp8) <= max_rounding_point_distance) ||
           is_infinity_error(o_fp64, r_fp64))
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < ERROR_DETAIL_LIMIT)
            {
                std::cerr << msg << std::setw(12) << std::setprecision(7) << " out[" << i
                          << "] != ref[" << i << "]: " << o_fp64 << " != " << r_fp64 << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        report_error_stats(err_count, max_err, ref.size());
    }
    return res;
}

/**
 * @brief Check errors between BF8 ranges
 *
 * Specialized comparison for 8-bit brain floating point values that considers
 * the specific numerical properties and error characteristics of the BF8 format.
 *
 * @tparam Range Type of output range
 * @tparam RefRange Type of reference range
 * @param out Output range to check
 * @param ref Reference range to check against
 * @param msg Error message to display if check fails
 * @param rtol Relative tolerance
 * @param atol Absolute tolerance
 * @param allow_infinity_ref Whether to allow infinity in reference values
 * @return True if check passes, false otherwise
 */
template <typename Range, typename RefRange>
std::enable_if_t<(std::is_same_v<ranges::range_value_t<Range>, ranges::range_value_t<RefRange>> &&
                  std::is_same_v<ranges::range_value_t<Range>, bf8_t>),
                 bool>
    CK_TILE_HOST check_err(const Range& out,
                           const RefRange& ref,
                           const std::string& msg  = "Error: Incorrect results!",
                           double rtol             = 1e-3,
                           double atol             = 1e-3,
                           bool allow_infinity_ref = false)
{
    if(check_size_mismatch(out, ref, msg))
        return false;

    const auto is_infinity_error = [=](auto o, auto r) {
        const bool either_not_finite = !std::isfinite(o) || !std::isfinite(r);
        const bool both_infinite_and_same =
            std::isinf(o) && std::isinf(r) && (bit_cast<uint64_t>(o) == bit_cast<uint64_t>(r));

        return either_not_finite && !(allow_infinity_ref && both_infinite_and_same);
    };

    bool res{true};
    int err_count  = 0;
    double err     = 0;
    double max_err = std::numeric_limits<float>::min();
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        const double o = type_convert<float>(*std::next(std::begin(out), i));
        const double r = type_convert<float>(*std::next(std::begin(ref), i));
        err            = std::abs(o - r);
        if(err > atol + rtol * std::abs(r) || is_infinity_error(o, r))
        {
            max_err = err > max_err ? err : max_err;
            err_count++;
            if(err_count < ERROR_DETAIL_LIMIT)
            {
                std::cerr << msg << std::setw(12) << std::setprecision(7) << " out[" << i
                          << "] != ref[" << i << "]: " << o << " != " << r << std::endl;
            }
            res = false;
        }
    }
    if(!res)
    {
        report_error_stats(err_count, max_err, ref.size());
    }
    return res;
}

} // namespace ck_tile
