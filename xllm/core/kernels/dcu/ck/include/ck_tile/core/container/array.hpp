// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.

#pragma once

#include <initializer_list>
#include <vector>

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/utility/functional.hpp"

namespace ck_tile {

// use aggregate initialization for this type
// e.g. array<index_t, 4> buf {0};  => {0, 0, 0, 0}, clean
//      array<index_t, 4> buf {3, 2}; => {3, 2, 2, 2} (not {3,2,0,0})
// use make_array_with({...}) to construct an array with compatible behavior as old ck
// TODO: manually added constructor same as old ck
template <typename T_, index_t N_>
struct array
{
    using value_type           = T_;
    static constexpr index_t N = N_;
    // TODO: do we need this?
    // using bulk_type = uint8_t __attribute__((ext_vector_type(N * sizeof(value_type))));
    // union {
    value_type data[N];
    //     bulk_type __content;
    //};
    CK_TILE_HOST_DEVICE constexpr array() : data{} {}
    // TODO: will initialize the data[] with the last value repeatedly
    //       behavior different from std
    CK_TILE_HOST_DEVICE constexpr array(std::initializer_list<value_type> ilist)
    {
        constexpr index_t list_size = std::initializer_list<value_type>{}.size();
        static_assert(list_size <= N, "out of bound");

        index_t i        = 0;
        value_type vlast = value_type{};

        for(const value_type& val : ilist)
        {
            data[i] = val;
            vlast   = val;
            ++i;
        }
        for(; i < N; ++i)
        {
            data[i] = vlast;
        }
    }

    template <typename Y,
              typename = std::enable_if_t<std::is_convertible_v<Y, value_type> ||
                                          std::is_constructible_v<Y, value_type>>>
    CK_TILE_HOST_DEVICE explicit constexpr array(Y c)
    {
        for(auto i = 0; i < size(); i++)
            data[i] = static_cast<value_type>(c);
    }

    // template <typename Y>
    // CK_TILE_HOST_DEVICE constexpr array(const array& o)
    // {
    //     // static_assert(ArrayType::size() == size(), "wrong! size not the same");
    //     __content = o.__content;
    // }
    // CK_TILE_HOST_DEVICE constexpr array& operator=(const array& o)
    // {
    //     // static_assert(ArrayType::size() == size(), "wrong! size not the same");
    //     __content = o.__content;
    //     return *this;
    // }

    CK_TILE_HOST_DEVICE static constexpr auto size() { return N; }
    CK_TILE_HOST_DEVICE static constexpr bool is_static() { return is_static_v<value_type>; }

    // clang-format off
    CK_TILE_HOST_DEVICE constexpr auto& get()                                           { return data; }
    CK_TILE_HOST_DEVICE constexpr const auto& get() const                               { return data; }
    CK_TILE_HOST_DEVICE constexpr auto& get(index_t i)                                  { return data[i]; }
    CK_TILE_HOST_DEVICE constexpr const auto& get(index_t i) const                      { return data[i]; }
    template <index_t I> CK_TILE_HOST_DEVICE constexpr auto& get()                      { return data[I]; }
    template <index_t I> CK_TILE_HOST_DEVICE constexpr const auto& get() const          { return data[I]; }
    template <index_t I> CK_TILE_HOST_DEVICE constexpr auto& get(number<I>)             { return data[I]; }
    template <index_t I> CK_TILE_HOST_DEVICE constexpr const auto& get(number<I>) const { return data[I]; }

    CK_TILE_HOST_DEVICE constexpr auto& at(index_t i)                                   { return get(i); }
    CK_TILE_HOST_DEVICE constexpr const auto& at(index_t i) const                       { return get(i); }
    template <index_t I> CK_TILE_HOST_DEVICE constexpr auto& at()                       { return get(I); }
    template <index_t I> CK_TILE_HOST_DEVICE constexpr const auto& at() const           { return get(I); }
    template <index_t I> CK_TILE_HOST_DEVICE constexpr auto& at(number<I>)              { return get(I); }
    template <index_t I> CK_TILE_HOST_DEVICE constexpr const auto& at(number<I>) const  { return get(I); }

    CK_TILE_HOST_DEVICE constexpr const value_type& operator[](index_t i) const { return get(i); }
    CK_TILE_HOST_DEVICE constexpr value_type& operator[](index_t i)             { return get(i); }
    CK_TILE_HOST_DEVICE constexpr value_type& operator()(index_t i)             { return get(i); }     // TODO: compatible
#if 0
    template <typename ArrayLike>
    CK_TILE_HOST_DEVICE constexpr auto operator=(const ArrayLike& arr)
    {
        static_assert(ArrayLike::size() == size(), "wrong! size not the same");
        for(index_t i = 0; i < size(); ++i)
        {
            data[i] = arr[i];
        }
        return *this;
    }
#endif
    // type punning (strict aliasing) member functions for read/write
    // aliasing this array of type "T", "N" elements
    // as array of type "Tx", sizeof(T)*N/sizeof(Tx) elements
#define AR_AS_COM_()                                         \
    static_assert(sizeof(value_type) * N % sizeof(Tx) == 0); \
    constexpr int vx = sizeof(value_type) * N / sizeof(Tx)

    template <typename Tx> CK_TILE_HOST_DEVICE constexpr auto& get_as()
            { AR_AS_COM_();    return reinterpret_cast<array<Tx, vx>&>(data); }
    template <typename Tx> CK_TILE_HOST_DEVICE constexpr const auto& get_as() const
            { AR_AS_COM_();    return reinterpret_cast<const array<Tx, vx>&>(data); }

    // below index is for index *AFTER* type convert, not before
    template <typename Tx> CK_TILE_HOST_DEVICE constexpr auto& get_as(index_t i)
            { AR_AS_COM_();    return reinterpret_cast<array<Tx, vx>&>(data).at(i); }
    template <typename Tx> CK_TILE_HOST_DEVICE constexpr const auto& get_as(index_t i) const
            { AR_AS_COM_();    return reinterpret_cast<const array<Tx, vx>&>(data).at(i); }
    template <typename Tx, index_t I> CK_TILE_HOST_DEVICE constexpr auto& get_as(number<I>)
            { AR_AS_COM_();    return reinterpret_cast<array<Tx, vx>&>(data).at(number<I>{}); }
    template <typename Tx, index_t I> CK_TILE_HOST_DEVICE constexpr const auto& get_as(number<I>) const
            { AR_AS_COM_();    return reinterpret_cast<const array<Tx, vx>&>(data).at(number<I>{}); }
    
    template <typename Tx> CK_TILE_HOST_DEVICE constexpr void set_as(index_t i, const Tx & x)
            { AR_AS_COM_();    reinterpret_cast<array<Tx, vx>&>(data).at(i) = x; }
    template <typename Tx, index_t I> CK_TILE_HOST_DEVICE constexpr void set_as(number<I>, const Tx & x)
            { AR_AS_COM_();    reinterpret_cast<array<Tx, vx>&>(data).at(number<I>{}) = x; }
#undef AR_AS_COM_
    // clang-format on
};

// empty Array

template <typename T>
struct array<T, 0>
{
    using value_type = T;

    CK_TILE_HOST_DEVICE constexpr array() {}
    CK_TILE_HOST_DEVICE static constexpr index_t size() { return 0; }
    CK_TILE_HOST_DEVICE static constexpr bool is_static() { return is_static_v<T>; };
    CK_TILE_HOST_DEVICE void print() const { printf("array{size: 0, data: []}"); }
};

template <typename, typename>
struct vector_traits;

// specialization for array
template <typename T, index_t N>
struct vector_traits<array<T, N>, void>
{
    using scalar_type                    = T;
    static constexpr index_t vector_size = N;
};

namespace details {
template <class>
struct is_ref_wrapper : std::false_type
{
};
template <class T>
struct is_ref_wrapper<std::reference_wrapper<T>> : std::true_type
{
};

template <class T>
using not_ref_wrapper = std::negation<is_ref_wrapper<std::decay_t<T>>>;

template <class D, class...>
struct return_type_helper
{
    using type = D;
};
template <class... Ts>
struct return_type_helper<void, Ts...> : std::common_type<Ts...>
{
    static_assert(std::conjunction_v<not_ref_wrapper<Ts>...>,
                  "Ts cannot contain reference_wrappers when D is void");
};

template <class D, class... Ts>
using return_type = array<typename return_type_helper<D, Ts...>::type, sizeof...(Ts)>;
} // namespace details

template <typename D = void, typename... Ts>
CK_TILE_HOST_DEVICE constexpr details::return_type<D, Ts...> make_array(Ts&&... ts)
{
    return {std::forward<Ts>(ts)...};
}

// // make empty array
// template <typename T>
// CK_TILE_HOST_DEVICE constexpr auto make_array()
// {
//     return array<T, 0>{};
// }

// compatible with old ck's initializer, make an array and fill it withe the last element from
// initializer_list
template <typename T, index_t Size>
CK_TILE_HOST_DEVICE constexpr auto make_array_with(std::initializer_list<T> ilist)
{
    return array<T, Size>(ilist);
}

template <typename T, index_t Size>
CK_TILE_HOST_DEVICE constexpr bool operator==(const array<T, Size>& a, const array<T, Size>& b)
{
    bool same = true;

    for(index_t i = 0; i < Size; ++i)
    {
        if(a[i] != b[i])
        {
            same = false;
            break;
        }
    }

    return same;
}

template <typename T, index_t Size>
CK_TILE_HOST_DEVICE constexpr bool operator!=(const array<T, Size>& a, const array<T, Size>& b)
{
    return !(a == b);
}

template <typename T, index_t N, typename X>
CK_TILE_HOST_DEVICE constexpr auto to_array(const std::vector<X>& x)
{
    array<T, N> arr;

    static_for<0, N, 1>{}([&x, &arr](auto i) { arr(i) = x[i]; });

    return arr;
}

template <typename T, index_t N, typename X>
CK_TILE_HOST_DEVICE constexpr auto to_array(const X& x)
{
    static_assert(N <= X::size(), "");

    array<T, N> arr;

    static_for<0, N, 1>{}([&x, &arr](auto i) { arr(i) = x[i]; });

    return arr;
}

} // namespace ck_tile
