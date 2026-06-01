// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/numeric/pk_int4.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

// this structure is used to pick up the <base> type inside
// using xxx = <base> __attribute__((ext_vector_type(N)));
// because clang only allow native type + bool in this term (custom type will fail)
// overload this structure to let proper <base> type

template <typename T>
struct native_t
{
    using type = remove_cvref_t<T>;
};

// we name this as ext_vector purposely, because clang ext_vector_type extention only accept literay
// basic type to construct a ext_vector_type you must be very careful using this, or will have lot
// of compiler errors e.g. struct A; using Ax2_t = A __attribute__((ext_vector_type(2)));  -> will
// have compiler error
namespace impl {

template <typename T_, index_t N_, typename = void>
struct ext_vector;

template <typename T_, index_t N_>
struct ext_vector<T_, N_, std::enable_if_t<!std::is_class_v<typename native_t<T_>::type>>>
{
    static constexpr index_t N = N_;
    // struct type is not supported for ext_vector
    using value_type = typename native_t<T_>::type;
    static_assert(!std::is_class_v<value_type>);
    using type = value_type __attribute__((ext_vector_type(N))); // this is danguous
};

template <typename T_, index_t N_>
struct ext_vector<T_, N_, std::enable_if_t<std::is_class_v<typename native_t<T_>::type>>>
{
    static constexpr index_t N = N_;
    // struct type is not supported for ext_vector
    using value_type = typename native_t<T_>::type::type;
    static_assert(!std::is_class_v<value_type>);
    using type = value_type __attribute__((ext_vector_type(N))); // this is danguous
};

template <typename V_, index_t Vs_, index_t N_>
struct ext_vector<V_ __attribute__((ext_vector_type(Vs_))),
                  N_,
                  std::enable_if_t<!std::is_class_v<typename native_t<V_>::type>>>
{
    static constexpr index_t N = Vs_ * N_;
    using value_type           = typename native_t<remove_cvref_t<V_>>::type;
    static_assert(!std::is_class_v<value_type>);
    using type = value_type __attribute__((ext_vector_type(N))); // this is danguous
};

template <typename V_, index_t Vs_, index_t N_>
struct ext_vector<V_ __attribute__((ext_vector_type(Vs_))),
                  N_,
                  std::enable_if_t<std::is_class_v<typename native_t<V_>::type>>>
{
    static constexpr index_t N = Vs_ * N_;
    using value_type           = typename native_t<remove_cvref_t<V_>>::type::type;
    static_assert(!std::is_class_v<value_type>);
    using type = value_type __attribute__((ext_vector_type(N))); // this is danguous
};

} // namespace impl

template <typename T, index_t N>
using ext_vector_t = typename impl::ext_vector<T, N>::type;

// by default, any type will result in a vector_size=1 with scalar_type=T traits.
// ... unless we have other vector_traits specialization
template <typename T, typename = void>
struct vector_traits
{
    using scalar_type =
        std::conditional_t<std::is_same_v<remove_cvref_t<T>, pk_int4_t>, int8_t, remove_cvref_t<T>>;
    static constexpr index_t vector_size = 1;
};

// specialization for ext_vector_type()
template <typename T, index_t N>
struct vector_traits<T __attribute__((ext_vector_type(N))), void>
{
    using scalar_type = std::conditional_t<std::is_same_v<T, pk_int4_t>, int8_t, T>;
    static constexpr index_t vector_size = N;
};

template <typename X, typename Y>
using has_same_scalar_type = std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                          typename vector_traits<remove_cvref_t<Y>>::scalar_type>;

// below are some pre-defines of ext_vector_type
// attention! 2 vector type could be just the same type
// fp64
using fp64_t   = double;
using fp64x2_t = double __attribute__((ext_vector_type(2)));
using fp64x4_t = double __attribute__((ext_vector_type(4)));

// fp32
using fp32_t    = float;
using fp32x2_t  = float __attribute__((ext_vector_type(2)));
using fp32x4_t  = float __attribute__((ext_vector_type(4)));
using fp32x8_t  = float __attribute__((ext_vector_type(8)));
using fp32x16_t = float __attribute__((ext_vector_type(16)));
using fp32x32_t = float __attribute__((ext_vector_type(32)));
using fp32x64_t = float __attribute__((ext_vector_type(64)));

// fp16
// using fp16_t = ...
using fp16x2_t  = _Float16 __attribute__((ext_vector_type(2)));
using fp16x4_t  = _Float16 __attribute__((ext_vector_type(4)));
using fp16x8_t  = _Float16 __attribute__((ext_vector_type(8)));
using fp16x16_t = _Float16 __attribute__((ext_vector_type(16)));
using fp16x32_t = _Float16 __attribute__((ext_vector_type(32)));
using fp16x64_t = _Float16 __attribute__((ext_vector_type(64)));

// bf16
// using bf16_t = ...
using bf16x2_t  = bf16_raw_t __attribute__((ext_vector_type(2)));
using bf16x4_t  = bf16_raw_t __attribute__((ext_vector_type(4)));
using bf16x8_t  = bf16_raw_t __attribute__((ext_vector_type(8)));
using bf16x16_t = bf16_raw_t __attribute__((ext_vector_type(16)));
using bf16x32_t = bf16_raw_t __attribute__((ext_vector_type(32)));
using bf16x64_t = bf16_raw_t __attribute__((ext_vector_type(64)));

// i32
// using int32_t = ...
using int32x2_t  = int32_t __attribute__((ext_vector_type(2)));
using int32x4_t  = int32_t __attribute__((ext_vector_type(4)));
using int32x8_t  = int32_t __attribute__((ext_vector_type(8)));
using int32x16_t = int32_t __attribute__((ext_vector_type(16)));
using int32x32_t = int32_t __attribute__((ext_vector_type(32)));
using int32x64_t = int32_t __attribute__((ext_vector_type(64)));

// u32
// using uint32_t = ...
using uint32x2_t  = uint32_t __attribute__((ext_vector_type(2)));
using uint32x4_t  = uint32_t __attribute__((ext_vector_type(4)));
using uint32x8_t  = uint32_t __attribute__((ext_vector_type(8)));
using uint32x16_t = uint32_t __attribute__((ext_vector_type(16)));
using uint32x32_t = uint32_t __attribute__((ext_vector_type(32)));
using uint32x64_t = uint32_t __attribute__((ext_vector_type(64)));

// i16
// using int16_t = ...
using int16x2_t  = int16_t __attribute__((ext_vector_type(2)));
using int16x4_t  = int16_t __attribute__((ext_vector_type(4)));
using int16x8_t  = int16_t __attribute__((ext_vector_type(8)));
using int16x16_t = int16_t __attribute__((ext_vector_type(16)));
using int16x32_t = int16_t __attribute__((ext_vector_type(32)));
using int16x64_t = int16_t __attribute__((ext_vector_type(64)));

// u16
// using uint16_t
using uint16x2_t  = uint16_t __attribute__((ext_vector_type(2)));
using uint16x4_t  = uint16_t __attribute__((ext_vector_type(4)));
using uint16x8_t  = uint16_t __attribute__((ext_vector_type(8)));
using uint16x16_t = uint16_t __attribute__((ext_vector_type(16)));
using uint16x32_t = uint16_t __attribute__((ext_vector_type(32)));
using uint16x64_t = uint16_t __attribute__((ext_vector_type(64)));

// i8
// using int8_t
using int8x2_t  = int8_t __attribute((ext_vector_type(2)));
using int8x4_t  = int8_t __attribute((ext_vector_type(4)));
using int8x8_t  = int8_t __attribute((ext_vector_type(8)));
using int8x16_t = int8_t __attribute((ext_vector_type(16)));
using int8x32_t = int8_t __attribute((ext_vector_type(32)));
using int8x64_t = int8_t __attribute((ext_vector_type(64)));

// ui8
// using uint8_t
using uint8x2_t  = uint8_t __attribute((ext_vector_type(2)));
using uint8x4_t  = uint8_t __attribute((ext_vector_type(4)));
using uint8x8_t  = uint8_t __attribute((ext_vector_type(8)));
using uint8x16_t = uint8_t __attribute((ext_vector_type(16)));
using uint8x32_t = uint8_t __attribute((ext_vector_type(32)));
using uint8x64_t = uint8_t __attribute((ext_vector_type(64)));

#if CK_TILE_USE_CUSTOM_DATA_TYPE
// f8
// using fp8_t
using fp8x2_t  = fp8_raw_t __attribute((ext_vector_type(2)));
using fp8x4_t  = fp8_raw_t __attribute((ext_vector_type(4)));
using fp8x8_t  = fp8_raw_t __attribute((ext_vector_type(8)));
using fp8x16_t = fp8_raw_t __attribute((ext_vector_type(16)));
using fp8x32_t = fp8_raw_t __attribute((ext_vector_type(32)));
using fp8x64_t = fp8_raw_t __attribute((ext_vector_type(64)));

// bf8
// using bf8_t
using bf8x2_t  = bf8_raw_t __attribute((ext_vector_type(2)));
using bf8x4_t  = bf8_raw_t __attribute((ext_vector_type(4)));
using bf8x8_t  = bf8_raw_t __attribute((ext_vector_type(8)));
using bf8x16_t = bf8_raw_t __attribute((ext_vector_type(16)));
using bf8x32_t = bf8_raw_t __attribute((ext_vector_type(32)));
using bf8x64_t = bf8_raw_t __attribute((ext_vector_type(64)));
#else
// f8
// using fp8_t
using fp8x2_t  = fp8_t __attribute((ext_vector_type(2)));
using fp8x4_t  = fp8_t __attribute((ext_vector_type(4)));
using fp8x8_t  = fp8_t __attribute((ext_vector_type(8)));
using fp8x16_t = fp8_t __attribute((ext_vector_type(16)));
using fp8x32_t = fp8_t __attribute((ext_vector_type(32)));
using fp8x64_t = fp8_t __attribute((ext_vector_type(64)));

// bf8
// using bf8_t
using bf8x2_t  = bf8_t __attribute((ext_vector_type(2)));
using bf8x4_t  = bf8_t __attribute((ext_vector_type(4)));
using bf8x8_t  = bf8_t __attribute((ext_vector_type(8)));
using bf8x16_t = bf8_t __attribute((ext_vector_type(16)));
using bf8x32_t = bf8_t __attribute((ext_vector_type(32)));
using bf8x64_t = bf8_t __attribute((ext_vector_type(64)));
#endif

// pk_int4_t
// using pk_int4_t
using pk_int4x2_t  = int8_t __attribute((ext_vector_type(2)));
using pk_int4x4_t  = int8_t __attribute((ext_vector_type(4)));
using pk_int4x8_t  = int8_t __attribute((ext_vector_type(8)));
using pk_int4x16_t = int8_t __attribute((ext_vector_type(16)));
using pk_int4x32_t = int8_t __attribute((ext_vector_type(32)));
} // namespace ck_tile
