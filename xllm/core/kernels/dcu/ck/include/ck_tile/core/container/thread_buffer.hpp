// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/tuple.hpp"

namespace ck_tile {

#if CK_TILE_THREAD_BUFFER_DEFAULT == CK_TILE_THREAD_BUFFER_USE_TUPLE
template <typename T, index_t N>
using thread_buffer = tuple_array<T, N>;

template <typename... Ts>
CK_TILE_HOST_DEVICE constexpr auto make_thread_buffer(Ts&&... ts)
{
    return make_tuple(ts...);
}
#else

#if 0
template <typename T, index_t N>
using thread_buffer = array<T, N>;

template <typename... Ts>
CK_TILE_HOST_DEVICE constexpr auto make_thread_buffer(Ts&&... ts)
{
    return make_array(ts...);
}

#endif

// clang-format off
template<typename T_, index_t N_>
struct thread_buffer {
    using value_type = remove_cvref_t<T_>;
    static constexpr index_t N = N_;

    value_type data[N];

    // TODO: this ctor can't ignore
    CK_TILE_HOST_DEVICE constexpr thread_buffer() : data{} {}
    CK_TILE_HOST_DEVICE constexpr thread_buffer(const value_type & o) : data{o} {}
    // CK_TILE_HOST_DEVICE constexpr thread_buffer(const value_type & o) : data{} {
    //     static_for<0, N, 1>{}(
    //         [&](auto i) { data[i] = o; }
    //     );
    // }

    CK_TILE_HOST_DEVICE static constexpr auto size() { return N; }
    CK_TILE_HOST_DEVICE auto & get() {return data; }
    CK_TILE_HOST_DEVICE const auto & get() const {return data; }
    CK_TILE_HOST_DEVICE auto & get(index_t i) {return data[i]; }
    CK_TILE_HOST_DEVICE const auto & get(index_t i) const {return data[i]; }
    CK_TILE_HOST_DEVICE constexpr const auto& operator[](index_t i) const { return get(i); }
    CK_TILE_HOST_DEVICE constexpr auto& operator[](index_t i)             { return get(i); }
    CK_TILE_HOST_DEVICE constexpr auto& operator()(index_t i)             { return get(i); }     // TODO: compatible
    CK_TILE_HOST_DEVICE constexpr auto& at(index_t i)                                   { return get(i); }
    CK_TILE_HOST_DEVICE constexpr const auto& at(index_t i) const                       { return get(i); }
    template <index_t I> CK_TILE_HOST_DEVICE constexpr auto& at()                       { return get(I); }
    template <index_t I> CK_TILE_HOST_DEVICE constexpr const auto& at() const           { return get(I); }
    template <index_t I> CK_TILE_HOST_DEVICE constexpr auto& at(number<I>)              { return get(I); }
    template <index_t I> CK_TILE_HOST_DEVICE constexpr const auto& at(number<I>) const  { return get(I); }
    
    template <typename X_,
              typename std::enable_if<has_same_scalar_type<value_type, X_>::value, bool>::type = false>
    CK_TILE_HOST_DEVICE constexpr auto _get_as() const
    {
        using X = remove_cvref_t<X_>;

        constexpr index_t kSPerX = vector_traits<X>::vector_size;
        static_assert(N % kSPerX == 0);

        union {
            thread_buffer<X_, N / kSPerX> data {};
            // tuple_array<value_type, kSPerX> sub_data;
            value_type sub_data[N];
        } vx;
        static_for<0, N, 1>{}(
            [&](auto j) { vx.sub_data[j] = data[j]; });
        return vx.data;
    }

    template <typename X_,
              index_t Is,
              typename std::enable_if<has_same_scalar_type<value_type, X_>::value, bool>::type = false>
    CK_TILE_HOST_DEVICE const constexpr remove_reference_t<X_> _get_as(number<Is> is) const
    {
        using X = remove_cvref_t<X_>;

        constexpr index_t kSPerX = vector_traits<X>::vector_size;

        union {
            X_ data {};
            tuple_array<value_type, kSPerX> sub_data;
        } vx;
        static_for<0, kSPerX, 1>{}(
            [&](auto j) { vx.sub_data(j) = operator[]((is * number<sizeof(X_)/sizeof(value_type)>{}) + j); });
        return vx.data;
    }

    // To support 8bit data type get_as
    template <typename X_,
              typename std::enable_if<((!has_same_scalar_type<value_type, X_>::value) && sizeof(value_type) == 1),
                                      bool>::type = false>
    CK_TILE_HOST_DEVICE constexpr auto _get_as() const
    {
        using X = remove_cvref_t<X_>;
        static_assert(sizeof(X) % sizeof(value_type) == 0);

        constexpr index_t kSPerX = sizeof(X) / sizeof(value_type);
        static_assert(kSPerX > 0);
        static_assert(N % kSPerX == 0);

        union {
            thread_buffer<X_, N / kSPerX> data {};
            value_type sub_data[N];
        } vx;
        static_for<0, N, 1>{}(
            [&](auto j) { vx.sub_data[j] = data[j]; });
        return vx.data;
    }

    template <typename X_,
              index_t Is,
              typename std::enable_if<((!has_same_scalar_type<value_type, X_>::value) && sizeof(value_type) == 1),
                                      bool>::type = false>
    CK_TILE_HOST_DEVICE const constexpr remove_reference_t<X_> _get_as(number<Is> is) const
    {
        using X = remove_cvref_t<X_>;
        static_assert(sizeof(X) % sizeof(value_type) == 0);

        constexpr index_t kSPerX = sizeof(X) / sizeof(value_type);

        static_assert(kSPerX > 0);
        union {
            X_ data {};
            tuple_array<value_type, kSPerX> sub_data;
        } vx;
        static_for<0, kSPerX, 1>{}(
            [&](auto j) {
                vx.sub_data(j) = operator[]((is * number<kSPerX>{}) + j);
            });
        return vx.data;
    }

#if 0
    template <typename X_,
              index_t Is,
              typename std::enable_if<has_same_scalar_type<value_type, X_>::value, bool>::type = false>
    CK_TILE_HOST_DEVICE constexpr void _set_as(number<Is> is, X_ x)
    {
        using X = remove_cvref_t<X_>;

        constexpr index_t kSPerX = vector_traits<X>::vector_size;

        union {
            X_ data;
            tuple_array<value_type, kSPerX> sub_data;
        } vx {x};

        static_for<0, kSPerX, 1>{}(
           [&](auto j) { operator()((is * number<sizeof(X_)/sizeof(value_type)>{}) + j) = vx.sub_data[j]; });
    }
#endif


#define TB_COMMON_AS() \
            static_assert(sizeof(value_type) * N % sizeof(Tx) == 0); \
            constexpr int vx = sizeof(value_type) * N / sizeof(Tx)

    template<typename Tx>
    CK_TILE_HOST_DEVICE auto & get_as() {TB_COMMON_AS();
            return reinterpret_cast<thread_buffer<Tx, vx>&>(data);}
    template<typename Tx>
    CK_TILE_HOST_DEVICE constexpr auto get_as() const {TB_COMMON_AS();
            if constexpr(sizeof(value_type) <= 1 )
            return _get_as<Tx>();   // TODO: current compiler for 8bit data need use union to get data back, should fix in the future
            else
            return reinterpret_cast<const thread_buffer<Tx, vx>&>(data);}
    template<typename Tx, index_t I>
    CK_TILE_HOST_DEVICE auto & get_as(number<I>) {TB_COMMON_AS();
            return reinterpret_cast<thread_buffer<Tx, vx>&>(data).get(number<I>{});}
    template<typename Tx, index_t I>
    CK_TILE_HOST_DEVICE constexpr auto get_as(number<I>) const {TB_COMMON_AS();
            if constexpr(sizeof(value_type) <= 1 )
            return _get_as<Tx>(number<I>{});   // TODO: current compiler for 8bit data need use union to get data back, should fix in the future
            else
            return reinterpret_cast<const thread_buffer<Tx, vx>&>(data).get(number<I>{});}

    template <typename Tx> CK_TILE_HOST_DEVICE constexpr void set_as(index_t i, const Tx & x)
            { TB_COMMON_AS();    reinterpret_cast<thread_buffer<Tx, vx>&>(data).at(i) = x; }
    template <typename Tx, index_t I> CK_TILE_HOST_DEVICE constexpr void set_as(number<I>, const Tx & x)
            { TB_COMMON_AS();    reinterpret_cast<thread_buffer<Tx, vx>&>(data).at(number<I>{}) = x; }

#undef TB_COMMON_AS
};
// clang-format on

template <typename, typename>
struct vector_traits;

// specialization for array
template <typename T, index_t N>
struct vector_traits<thread_buffer<T, N>, std::enable_if_t<!std::is_class_v<T>>>
{
    using scalar_type                    = T;
    static constexpr index_t vector_size = N;
};

template <typename T, index_t N>
struct vector_traits<thread_buffer<T, N>, std::enable_if_t<std::is_class_v<T>>>
{
    using scalar_type                    = typename T::type;
    static constexpr index_t vector_size = N;
};

#endif

} // namespace ck_tile
