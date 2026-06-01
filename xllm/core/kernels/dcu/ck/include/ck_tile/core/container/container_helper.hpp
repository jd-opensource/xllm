// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/map.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/utility/functional.hpp"

namespace ck_tile {

template <typename TData, index_t NSize>
CK_TILE_HOST_DEVICE constexpr auto container_push_back(const array<TData, NSize>& a, const TData& x)
{
    array<TData, NSize + 1> r;
    static_for<0, NSize, 1>{}([&r, &a ](auto i) constexpr { r(i) = a[i]; });
    r[number<NSize>{}] = x;
    return r;
}

template <typename... Ts, typename T>
CK_TILE_HOST_DEVICE constexpr auto container_push_front(const tuple<Ts...>& a, const T& x)
{
    return container_concat(make_tuple(x), a);
}

template <typename... Ts, typename T>
CK_TILE_HOST_DEVICE constexpr auto container_push_back(const tuple<Ts...>& a, const T& x)
{
    return container_concat(a, make_tuple(x));
}

// reorder array
template <typename TData, index_t NSize, index_t... IRs>
CK_TILE_HOST_DEVICE constexpr auto
container_reorder_given_new2old(const array<TData, NSize>& old_array, sequence<IRs...> /*new2old*/)
{
    static_assert(NSize == sizeof...(IRs), "wrong! size not consistent");
    static_assert(is_valid_sequence_map<sequence<IRs...>>{}, "wrong! invalid reorder map");
    return make_array<remove_cvref_t<TData>>(old_array[IRs]...);
}

template <typename TData, index_t NSize, index_t... IRs>
CK_TILE_HOST_DEVICE constexpr auto
container_reorder_given_old2new(const array<TData, NSize>& old_array, sequence<IRs...> old2new)
{
    return container_reorder_given_new2old(
        old_array, typename sequence_map_inverse<decltype(old2new)>::type{});
}

// reorder array
template <typename TData, index_t NSize>
CK_TILE_HOST_DEVICE constexpr auto
container_reorder_given_new2old(const array<TData, NSize>& old_array,
                                const map<index_t, index_t>& new2old)
{
    array<TData, NSize> new_array;

    for(const auto& [new_pos, old_pos] : new2old)
    {
        new_array(new_pos) = old_array[old_pos];
    }

    return new_array;
}

template <typename TData, index_t NSize>
CK_TILE_HOST_DEVICE constexpr auto
container_reorder_given_old2new(const array<TData, NSize>& old_array,
                                const map<index_t, index_t>& old2new)
{
    array<TData, NSize> new_array;

    for(const auto& [old_pos, new_pos] : old2new)
    {
        new_array(new_pos) = old_array[old_pos];
    }

    return new_array;
}

// reorder tuple
template <typename... Ts, index_t... IRs>
CK_TILE_HOST_DEVICE constexpr auto container_reorder_given_new2old(const tuple<Ts...>& old_tuple,
                                                                   sequence<IRs...> /*new2old*/)
{
    static_assert(sizeof...(Ts) == sizeof...(IRs), "wrong! size not consistent");

    static_assert(is_valid_sequence_map<sequence<IRs...>>{}, "wrong! invalid reorder map");

    return make_tuple(old_tuple[number<IRs>{}]...);
}

template <typename... Ts, index_t... IRs>
CK_TILE_HOST_DEVICE constexpr auto container_reorder_given_old2new(const tuple<Ts...>& old_tuple,
                                                                   sequence<IRs...> old2new)
{
    return container_reorder_given_new2old(
        old_tuple, typename sequence_map_inverse<decltype(old2new)>::type{});
}

// reorder sequence
template <index_t... Is, index_t... IRs>
CK_TILE_HOST_DEVICE constexpr auto container_reorder_given_new2old(sequence<Is...> /* old_seq */,
                                                                   sequence<IRs...> /*new2old*/)
{
    static_assert(sizeof...(Is) == sizeof...(IRs), "wrong! size not consistent");

    static_assert(is_valid_sequence_map<sequence<IRs...>>{}, "wrong! invalid reorder map");

    return sequence<sequence<Is...>::at(number<IRs>{})...>{};
}

template <index_t... Is, index_t... IRs>
CK_TILE_HOST_DEVICE constexpr auto container_reorder_given_old2new(sequence<Is...> old_seq,
                                                                   sequence<IRs...> /* old2new */)
{
    static_assert(sizeof...(Is) == sizeof...(IRs), "wrong! size not consistent");

    static_assert(is_valid_sequence_map<sequence<IRs...>>{}, "wrong! invalid reorder map");

    constexpr auto new2old = typename sequence_map_inverse<sequence<IRs...>>::type{};

    return container_reorder_given_new2old(old_seq, new2old);
}

#if 0
// rocm-4.1 compiler would crash for recursive lambda
template <typename Container,
          typename Reduce,
          typename Init,
          index_t IBegin = 0,
          index_t IEnd   = Container::size(),
          index_t IStep  = 1>
CK_TILE_HOST_DEVICE constexpr auto container_reduce(const Container& x,
                                                    Reduce reduce,
                                                    Init init,
                                                    number<IBegin> = number<0>{},
                                                    number<IEnd>   = number<Container::size()>{},
                                                    number<IStep>  = number<1>{})
{
    static_assert((IEnd - IBegin) % IStep == 0, "wrong!");

    // f is recursive function, fs is a dummy of f
    // i is index, y_old is current scan, r_old is current reduction
    auto f = [&](auto fs, auto i, auto r_old) {
        auto r_new = reduce(x[i], r_old);

        if constexpr(i.value < IEnd - IStep)
        {
            // recursively call f/fs
            return fs(fs, i + number<IStep>{}, r_new);
        }
        else
        {
            return r_new;
        }
    };

    // start recursion
    return f(f, number<IBegin>{}, init);
}
#else
// i is index, y_old is current scan, r_old is current reduction
template <typename Container,
          typename Reduce,
          typename ROld,
          index_t I,
          index_t IEnd,
          index_t IStep>
CK_TILE_HOST_DEVICE constexpr auto container_reduce_impl(
    const Container& x, Reduce reduce, ROld r_old, number<I> i, number<IEnd>, number<IStep>)
{
    auto r_new = reduce(x[i], r_old);

    if constexpr(i.value < IEnd - IStep)
    {
        return container_reduce_impl(
            x, reduce, r_new, i + number<IStep>{}, number<IEnd>{}, number<IStep>{});
    }
    else
    {
        return r_new;
    }
}

// rocm-4.1 compiler would crash for recursive lambda
// container reduce with initial value
template <typename Container,
          typename Reduce,
          typename Init,
          index_t IBegin = 0,
          index_t IEnd   = Container::size(),
          index_t IStep  = 1>
CK_TILE_HOST_DEVICE constexpr auto container_reduce(const Container& x,
                                                    Reduce reduce,
                                                    Init init,
                                                    number<IBegin> = number<0>{},
                                                    number<IEnd>   = number<Container::size()>{},
                                                    number<IStep>  = number<1>{})
{
    static_assert((IEnd - IBegin) % IStep == 0, "wrong!");

    if constexpr(IEnd > IBegin)
    {
        return container_reduce_impl(
            x, reduce, init, number<IBegin>{}, number<IEnd>{}, number<IStep>{});
    }
    else
    {
        return init;
    }
}
#endif

template <typename TData, index_t NSize, typename Reduce>
CK_TILE_HOST_DEVICE constexpr auto
container_reverse_inclusive_scan(const array<TData, NSize>& x, Reduce f, TData init)
{
    array<TData, NSize> y;

    TData r = init;

    static_for<NSize - 1, 0, -1>{}([&](auto i) {
        r    = f(r, x[i]);
        y(i) = r;
    });

    r              = f(r, x[number<0>{}]);
    y(number<0>{}) = r;

    return y;
}

template <typename TData, index_t NSize, typename Reduce, typename Init>
CK_TILE_HOST_DEVICE constexpr auto
container_reverse_exclusive_scan(const array<TData, NSize>& x, Reduce f, Init init)
{
#if 0
    array<TData, NSize> y;

    TData r = init;

    static_for<NSize - 1, 0, -1>{}([&](auto i) {
        y(i) = r;
        r    = f(r, x[i]);
    });

    y(number<0>{}) = r;

    return y;
#else
    array<TData, NSize> y;

    TData r = init;

    for(index_t i = NSize - 1; i > 0; --i)
    {
        y(i) = r;
        r    = f(r, x[i]);
    }

    y(0) = r;

    return y;
#endif
}

template <index_t... Is, typename Reduce, index_t Init>
CK_TILE_HOST_DEVICE constexpr auto
container_reverse_exclusive_scan(const sequence<Is...>& seq, Reduce f, number<Init>)
{
    return reverse_exclusive_scan_sequence(seq, f, number<Init>{});
}

#if 0
// rocm4.1 compiler would crash with recursive lambda
template <typename... Xs, typename Reduce, typename Init>
CK_TILE_HOST_DEVICE constexpr auto
container_reverse_exclusive_scan(const tuple<Xs...>& x, Reduce reduce, Init init)
{
    constexpr index_t NSize = sizeof...(Xs);

    // f is recursive function, fs is a dummy of f
    // i is index, y_old is current scan, r_old is current reduction
    auto f = [&](auto fs, auto i, auto y_old, auto r_old) {
        auto r_new = reduce(x[i], r_old);

        auto y_new = container_push_front(y_old, r_new);

        if constexpr(i.value > 1)
        {
            // recursively call f/fs
            return fs(fs, i - number<1>{}, y_new, r_new);
        }
        else
        {
            return y_new;
        }
    };

    // start recursion
    return f(f, number<NSize - 1>{}, make_tuple(init), init);
}
#else
// i is index, y_old is current scan, r_old is current reduction
template <typename... Xs, typename Reduce, index_t I, typename YOld, typename ROld>
CK_TILE_HOST_DEVICE constexpr auto container_reverse_exclusive_scan_impl(
    const tuple<Xs...>& x, Reduce reduce, number<I> i, YOld y_old, ROld r_old)
{
    auto r_new = reduce(x[i], r_old);

    auto y_new = container_push_front(y_old, r_new);

    if constexpr(i.value > 1)
    {
        // recursively call f/fs
        return container_reverse_exclusive_scan_impl(x, reduce, i - number<1>{}, y_new, r_new);
    }
    else
    {
        return y_new;
    }
}

template <typename... Xs, typename Reduce, typename Init>
CK_TILE_HOST_DEVICE constexpr auto
container_reverse_exclusive_scan(const tuple<Xs...>& x, Reduce reduce, Init init)
{
    constexpr index_t NSize = sizeof...(Xs);

    return container_reverse_exclusive_scan_impl(
        x, reduce, number<NSize - 1>{}, make_tuple(init), init);
}
#endif

// TODO: update to like container_reverse_exclusive_scan to deal with tuple of Numebr<>
template <typename... Xs, typename Reduce, typename TData>
CK_TILE_HOST_DEVICE constexpr auto
container_reverse_inclusive_scan(const tuple<Xs...>& x, Reduce f, TData init)
{
    constexpr index_t NSize = sizeof...(Xs);

    tuple<Xs...> y;

    TData r = init;

    static_for<NSize - 1, 0, -1>{}([&](auto i) {
        r    = f(r, x[i]);
        y(i) = r;
    });

    r              = f(r, x[number<0>{}]);
    y(number<0>{}) = r;

    return y;
}

template <typename X, typename... Ys>
CK_TILE_HOST_DEVICE constexpr auto container_concat(const X& x, const Ys&... ys)
{
    return container_concat(x, container_concat(ys...));
}

template <typename T, index_t NX, index_t NY>
CK_TILE_HOST_DEVICE constexpr auto container_concat(const array<T, NX>& ax, const array<T, NY>& ay)
{
    return unpack2(
        [&](auto&&... zs) { return make_array<T>(std::forward<decltype(zs)>(zs)...); }, ax, ay);
}

template <typename... X, typename... Y>
CK_TILE_HOST_DEVICE constexpr auto container_concat(const tuple<X...>& tx, const tuple<Y...>& ty)
{
    return unpack2(
        [&](auto&&... zs) { return make_tuple(std::forward<decltype(zs)>(zs)...); }, tx, ty);
}

template <typename Container>
CK_TILE_HOST_DEVICE constexpr auto container_concat(const Container& x)
{
    return x;
}

template <typename T, index_t N, index_t... Is>
CK_TILE_HOST_DEVICE constexpr auto get_container_subset(const array<T, N>& arr, sequence<Is...>)
{
    static_assert(N >= sizeof...(Is), "wrong! size");

    if constexpr(sizeof...(Is) > 0)
    {
        return make_array<T>(arr[Is]...);
    }
    else
    {
        return array<T, 0>{};
    }
}

template <typename... Ts, index_t... Is>
CK_TILE_HOST_DEVICE constexpr auto get_container_subset(const tuple<Ts...>& tup, sequence<Is...>)
{
    static_assert(sizeof...(Ts) >= sizeof...(Is), "wrong! size");

    if constexpr(sizeof...(Is) > 0)
    {
        return make_tuple(tup[number<Is>{}]...);
    }
    else
    {
        return tuple<>{};
    }
}

template <typename T, index_t N, index_t... Is>
CK_TILE_HOST_DEVICE constexpr void
set_container_subset(array<T, N>& y, sequence<Is...> picks, const array<T, sizeof...(Is)>& x)
{
    static_assert(N >= sizeof...(Is), "wrong! size");

    if constexpr(sizeof...(Is) > 0)
    {
        for(index_t i = 0; i < picks.size(); ++i)
        {
            y(picks[i]) = x[i];
        }
    }
}

template <typename Y, typename X, index_t... Is>
CK_TILE_HOST_DEVICE constexpr void set_container_subset(Y& y, sequence<Is...> picks, const X& x)
{
    static_assert(Y::size() >= sizeof...(Is) && X::size() == sizeof...(Is), "wrong! size");

    if constexpr(sizeof...(Is) > 0)
    {
        static_for<0, sizeof...(Is), 1>{}([&](auto i) { y(picks[i]) = x[i]; });
    }
}

// return the index of first occurance in the sequence.
// return seq.size(), if not found
template <index_t... Is>
constexpr index_t container_find(sequence<Is...> seq, index_t value)
{
    for(auto i = 0; i < seq.size(); i++)
    {
        if(seq[i] == value)
            return i;
    }

    return seq.size();
}

template <index_t... Is>
CK_TILE_HOST_DEVICE constexpr auto sequence_to_tuple_of_number(sequence<Is...>)
{
    using Seq = sequence<Is...>;

    return generate_tuple(
        [&](auto i) {
            constexpr index_t tmp = Seq::at(i);
            return number<tmp>{};
        },
        number<Seq::size()>{});
}

#if 0
#define TO_TUPLE_OF_SEQUENCE(a_of_b_impl, a_size, bs_sizes)             \
    [a_of_b_impl, a_size, bs_sizes] {                                   \
        return ck_tile::generate_tuple(                                 \
            [=](auto i) {                                               \
                constexpr auto b_impl    = a_of_b_impl[i];              \
                constexpr index_t b_size = bs_sizes[i];                 \
                constexpr auto b         = TO_SEQUENCE(b_impl, b_size); \
                return b;                                               \
            },                                                          \
            ck_tile::number<a_size>{});                                 \
    }()
#else
// constexpr index_t can't be captured "-Wunused-lambda-capture"
// TODO: this is ugly
#define TO_TUPLE_OF_SEQUENCE(a_of_b_impl, a_size, bs_sizes)             \
    [a_of_b_impl, bs_sizes] {                                           \
        return ck_tile::generate_tuple(                                 \
            [=](auto i) {                                               \
                constexpr auto b_impl    = a_of_b_impl[i];              \
                constexpr index_t b_size = bs_sizes[i];                 \
                constexpr auto b         = TO_SEQUENCE(b_impl, b_size); \
                return b;                                               \
            },                                                          \
            ck_tile::number<a_size>{});                                 \
    }()
#endif

} // namespace ck_tile
