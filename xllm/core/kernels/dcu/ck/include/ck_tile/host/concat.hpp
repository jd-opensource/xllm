// SPDX-License-Identifier: MIT
// Copyright (c) 2025, , Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"

namespace ck_tile {

template <typename T>
struct IsCharArray : std::false_type
{
};

template <std::size_t N>
struct IsCharArray<char[N]> : std::true_type
{
};

template <std::size_t N>
struct IsCharArray<const char[N]> : std::true_type
{
};

template <std::size_t N>
struct IsCharArray<char (&)[N]> : std::true_type
{
};

template <std::size_t N>
struct IsCharArray<const char (&)[N]> : std::true_type
{
};

template <typename... Ts>
inline constexpr bool AllConvertibleToStringView =
    ((std::is_convertible_v<Ts, std::string_view> || IsCharArray<Ts>::value ||
      std::is_same_v<Ts, char>) &&
     ...);

template <typename... Ts>
[[nodiscard]] auto
concat(const Ts&... xs) -> std::enable_if_t<!AllConvertibleToStringView<Ts...>, std::string>
{
    using ::operator<<;
    thread_local std::ostringstream oss;
    oss.str("");

    (oss << ... << xs);
    return oss.str();
}

template <std::size_t N>
[[nodiscard]] constexpr inline std::size_t getSize(char (&)[N]) noexcept
{
    return N;
}

template <std::size_t N>
[[nodiscard]] constexpr inline std::size_t getSize(const char (&)[N]) noexcept
{
    return N;
}

[[nodiscard]] constexpr inline std::size_t getSize(const char* s) noexcept
{
    const char* end = s;
    while(*end++ != 0) {}
    return end - s - 1;
}

[[nodiscard]] constexpr inline std::size_t getSize(const char&) noexcept { return 1; }

[[nodiscard]] inline std::size_t getSize(const std::string& s) noexcept { return s.size(); }

[[nodiscard]] constexpr inline std::size_t getSize(const std::string_view& s) noexcept
{
    return s.size();
}

template <typename... Ts>
auto concatInto(std::string& result,
                const Ts&... xs) -> std::enable_if_t<AllConvertibleToStringView<Ts...>, void>
{
    const std::size_t space = (1 + ... + getSize(xs));
    result.reserve(result.size() + space);
    ((result += xs), ...);
}

template <typename... Ts>
[[nodiscard]] auto
concat(const Ts&... xs) -> std::enable_if_t<AllConvertibleToStringView<Ts...>, std::string>
{
    std::string result;
    concatInto(result, xs...);
    return result;
}

// Function for types convertible to std::string_view
template <typename Sep, typename First, typename... Rest>
[[nodiscard]] auto concat(Sep sep, const First& first, const Rest&... rest)
    -> std::enable_if_t<AllConvertibleToStringView<First, Rest...>, std::string>
{
    std::string result;
    result += first;
    ((result += sep, result += rest), ...);
    return result;
}

// Function for other types
template <typename Sep, typename First, typename... Rest>
[[nodiscard]] auto concat(Sep sep, const First& first, const Rest&... rest)
    -> std::enable_if_t<!AllConvertibleToStringView<First, Rest...>, std::string>
{
    using ::operator<<;
    thread_local std::ostringstream oss;
    oss.str("");
    oss << first;
    ((oss << sep << rest), ...);
    return oss.str();
}

} // namespace ck_tile
