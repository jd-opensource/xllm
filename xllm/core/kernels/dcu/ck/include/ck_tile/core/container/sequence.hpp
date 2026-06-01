// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/to_sequence.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/utility/functional.hpp"

namespace ck_tile {

template <index_t, index_t, index_t>
struct static_for;

template <index_t...>
struct sequence;

template <typename Seq, index_t I>
struct sequence_split;

template <typename>
struct sequence_reverse;

template <typename>
struct sequence_map_inverse;

template <typename>
struct is_valid_sequence_map;

template <index_t I, index_t... Is>
CK_TILE_HOST_DEVICE constexpr auto sequence_pop_front(sequence<I, Is...>);

template <typename Seq>
CK_TILE_HOST_DEVICE constexpr auto sequence_pop_back(Seq);

namespace impl {
// static_assert(__has_builtin(__type_pack_element), "can't find __type_pack_element");
template <index_t I, typename... Ts>
using at_index_t = __type_pack_element<I, Ts...>;
} // namespace impl

// we could implement as below, similiar to std. But let's reduce the symbol name...
// template< class T, T... Ints >
// class integer_sequence;

template <index_t... Is>
struct sequence
{
    using type       = sequence;
    using value_type = index_t;

    CK_TILE_HOST_DEVICE static constexpr index_t size() { return sizeof...(Is); }
    CK_TILE_HOST_DEVICE static constexpr bool is_static() { return true; };

    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr auto get()
    {
        static_assert(I < size(), "wrong! I too large");
        return number<impl::at_index_t<I, constant<Is>...>{}>{};
    }

    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr auto get(number<I>)
    {
        static_assert(I < size(), "wrong! I too large");
        return number<get<I>()>{};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t at(index_t I)
    {
        // the last dummy element is to prevent compiler complain about empty array, when mSize = 0
        const index_t mData[size() + 1] = {Is..., 0};
        return mData[I];
    }

    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr auto at()
    {
        static_assert(I < size(), "wrong! I too large");
        return number<impl::at_index_t<I, constant<Is>...>{}>{};
    }

    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr auto at(number<I>)
    {
        static_assert(I < size(), "wrong! I too large");
        return number<get<I>()>{};
    }

    template <typename I>
    CK_TILE_HOST_DEVICE constexpr auto operator[](I i) const
    {
        return at(i);
    }

    template <index_t... IRs>
    CK_TILE_HOST_DEVICE static constexpr auto reorder_new_to_old(sequence<IRs...> /*new2old*/)
    {
        static_assert(sizeof...(Is) == sizeof...(IRs),
                      "wrong! reorder map should have the same size as sequence to be rerodered");

        static_assert(is_valid_sequence_map<sequence<IRs...>>::value, "wrong! invalid reorder map");

        return sequence<type::get(number<IRs>{})...>{};
    }

    // MapOld2New is sequence<...>
    template <typename MapOld2New>
    CK_TILE_HOST_DEVICE static constexpr auto reorder_old_to_new(MapOld2New)
    {
        static_assert(MapOld2New::size() == size(),
                      "wrong! reorder map should have the same size as sequence to be rerodered");

        static_assert(is_valid_sequence_map<MapOld2New>::value, "wrong! invalid reorder map");

        return reorder_new_to_old(typename sequence_map_inverse<MapOld2New>::type{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto reverse()
    {
        return typename sequence_reverse<type>::type{};
    }

    CK_TILE_HOST_DEVICE static constexpr auto front()
    {
        static_assert(size() > 0, "wrong!");
        return get(number<0>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto back()
    {
        static_assert(size() > 0, "wrong!");
        return get(number<size() - 1>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto pop_front() { return sequence_pop_front(type{}); }

    CK_TILE_HOST_DEVICE static constexpr auto pop_back() { return sequence_pop_back(type{}); }

    template <index_t... Xs>
    CK_TILE_HOST_DEVICE static constexpr auto push_front(sequence<Xs...>)
    {
        return sequence<Xs..., Is...>{};
    }

    template <index_t... Xs>
    CK_TILE_HOST_DEVICE static constexpr auto push_front(number<Xs>...)
    {
        return sequence<Xs..., Is...>{};
    }

    template <index_t... Xs>
    CK_TILE_HOST_DEVICE static constexpr auto push_back(sequence<Xs...>)
    {
        return sequence<Is..., Xs...>{};
    }

    template <index_t... Xs>
    CK_TILE_HOST_DEVICE static constexpr auto push_back(number<Xs>...)
    {
        return sequence<Is..., Xs...>{};
    }

    // pickup element at index <Ids...>
    template <index_t... Ids>
    CK_TILE_HOST_DEVICE static constexpr auto extract(number<Ids>...)
    {
        return sequence<type::get(number<Ids>{})...>{};
    }

    template <index_t... Ids>
    CK_TILE_HOST_DEVICE static constexpr auto extract(sequence<Ids...>)
    {
        return sequence<type::get(number<Ids>{})...>{};
    }

    // modify element at index "I" with value "X"
    template <index_t I, index_t X>
    CK_TILE_HOST_DEVICE static constexpr auto modify(number<I>, number<X>)
    {
        static_assert(I < size(), "wrong!");

        using seq_split          = sequence_split<type, I>;
        constexpr auto seq_left  = typename seq_split::left_type{};
        constexpr auto seq_right = typename seq_split::right_type{}.pop_front();

        return seq_left.push_back(number<X>{}).push_back(seq_right);
    }

    template <typename F>
    CK_TILE_HOST_DEVICE static constexpr auto transform(F f)
    {
        return sequence<f(Is)...>{};
    }

    CK_TILE_HOST_DEVICE static void print()
    {
        printf("sequence{size: %d, data: [", size());
        ((printf("%d ", Is)), ...);
        printf("]}");
    }
};

namespace impl {
template <typename T, T... Ints>
struct __integer_sequence;

template <index_t... Ints>
struct __integer_sequence<index_t, Ints...>
{
    using seq_type = sequence<Ints...>;
};
} // namespace impl

// similiar
template <index_t N>
using make_index_sequence =
    typename __make_integer_seq<impl::__integer_sequence, index_t, N>::seq_type;

// merge sequence
template <typename Seq, typename... Seqs>
struct sequence_merge
{
    using type = typename sequence_merge<Seq, typename sequence_merge<Seqs...>::type>::type;
};

template <index_t... Xs, index_t... Ys>
struct sequence_merge<sequence<Xs...>, sequence<Ys...>>
{
    using type = sequence<Xs..., Ys...>;
};

template <typename Seq>
struct sequence_merge<Seq>
{
    using type = Seq;
};

// generate sequence
template <index_t NSize, typename F>
struct sequence_gen
{
    template <index_t IBegin, index_t NRemain, typename G>
    struct sequence_gen_impl
    {
        static constexpr index_t NRemainLeft  = NRemain / 2;
        static constexpr index_t NRemainRight = NRemain - NRemainLeft;
        static constexpr index_t IMiddle      = IBegin + NRemainLeft;

        using type = typename sequence_merge<
            typename sequence_gen_impl<IBegin, NRemainLeft, G>::type,
            typename sequence_gen_impl<IMiddle, NRemainRight, G>::type>::type;
    };

    template <index_t I, typename G>
    struct sequence_gen_impl<I, 1, G>
    {
        static constexpr index_t Is = G{}(number<I>{});
        using type                  = sequence<Is>;
    };

    template <index_t I, typename G>
    struct sequence_gen_impl<I, 0, G>
    {
        using type = sequence<>;
    };

    using type = typename sequence_gen_impl<0, NSize, F>::type;
};

// arithmetic sequence
template <index_t IBegin, index_t IEnd, index_t Increment>
struct arithmetic_sequence_gen
{
    struct F
    {
        CK_TILE_HOST_DEVICE constexpr index_t operator()(index_t i) const
        {
            return i * Increment + IBegin;
        }
    };

    using type0 = typename sequence_gen<(IEnd - IBegin) / Increment, F>::type;
    using type1 = sequence<>;

    static constexpr bool kHasContent =
        (Increment > 0 && IBegin < IEnd) || (Increment < 0 && IBegin > IEnd);

    using type = typename std::conditional<kHasContent, type0, type1>::type;
};

template <index_t IEnd>
struct arithmetic_sequence_gen<0, IEnd, 1>
{
    using type = make_index_sequence<IEnd>;
};

// uniform sequence
template <index_t NSize, index_t I>
struct uniform_sequence_gen
{
    struct F
    {
        CK_TILE_HOST_DEVICE constexpr index_t operator()(index_t) const { return I; }
    };

    using type = typename sequence_gen<NSize, F>::type;
};

// reverse inclusive scan (with init) sequence
template <typename, typename, index_t>
struct sequence_reverse_inclusive_scan;

template <index_t I, index_t... Is, typename Reduce, index_t Init>
struct sequence_reverse_inclusive_scan<sequence<I, Is...>, Reduce, Init>
{
    using old_scan = typename sequence_reverse_inclusive_scan<sequence<Is...>, Reduce, Init>::type;

    static constexpr index_t new_reduce = Reduce{}(I, old_scan{}.front());

    using type = typename sequence_merge<sequence<new_reduce>, old_scan>::type;
};

template <index_t I, typename Reduce, index_t Init>
struct sequence_reverse_inclusive_scan<sequence<I>, Reduce, Init>
{
    using type = sequence<Reduce{}(I, Init)>;
};

template <typename Reduce, index_t Init>
struct sequence_reverse_inclusive_scan<sequence<>, Reduce, Init>
{
    using type = sequence<>;
};

// split sequence
template <typename Seq, index_t I>
struct sequence_split
{
    static constexpr index_t NSize = Seq{}.size();

    using range0 = typename arithmetic_sequence_gen<0, I, 1>::type;
    using range1 = typename arithmetic_sequence_gen<I, NSize, 1>::type;

    using left_type  = decltype(Seq::extract(range0{}));
    using right_type = decltype(Seq::extract(range1{}));
};

#if 0
// reverse sequence
template <typename Seq>
struct sequence_reverse
{
    static constexpr index_t NSize = Seq{}.size();

    using seq_split = sequence_split<Seq, NSize / 2>;
    using type      = typename sequence_merge<
        typename sequence_reverse<typename seq_split::right_type>::type,
        typename sequence_reverse<typename seq_split::left_type>::type>::type;
};

template <index_t I>
struct sequence_reverse<sequence<I>>
{
    using type = sequence<I>;
};

template <index_t I0, index_t I1>
struct sequence_reverse<sequence<I0, I1>>
{
    using type = sequence<I1, I0>;
};
#endif

namespace impl {
template <typename Id, index_t... Ns>
struct seq_reverse;

template <index_t... Ids, index_t... Ns>
struct seq_reverse<sequence<Ids...>, Ns...>
{
    template <index_t I>
    using element = impl::at_index_t<I, constant<Ns>...>;
    using type    = sequence<element<(sizeof...(Ns) - 1 - Ids)>::value...>;
};
} // namespace impl

template <index_t... Ns>
struct sequence_reverse<sequence<Ns...>>
    : impl::seq_reverse<make_index_sequence<sizeof...(Ns)>, Ns...>
{
};

// template <index_t... Ns>
// using sequence_reverse_t = typename sequence_reverse<Ns...>::type;

#if 1
template <typename Reduce, typename Seq, typename... Seqs>
struct sequence_reduce
{
    using type = typename sequence_reduce<Reduce,
                                          Seq,
                                          typename sequence_reduce<Reduce, Seqs...>::type>::type;
};

template <typename Reduce, index_t... Xs, index_t... Ys>
struct sequence_reduce<Reduce, sequence<Xs...>, sequence<Ys...>>
{
    using type = sequence<Reduce{}(Xs, Ys)...>;
};

template <typename Reduce, typename Seq>
struct sequence_reduce<Reduce, Seq>
{
    using type = Seq;
};
#endif

template <typename Values, typename Ids, typename Compare>
struct sequence_sort_impl
{
    template <typename LeftValues,
              typename LeftIds,
              typename RightValues,
              typename RightIds,
              typename MergedValues,
              typename MergedIds,
              typename Comp>
    struct sorted_sequence_merge_impl
    {
        static constexpr bool choose_left = LeftValues::front() < RightValues::front();

        static constexpr index_t chosen_value =
            choose_left ? LeftValues::front() : RightValues::front();
        static constexpr index_t chosen_id = choose_left ? LeftIds::front() : RightIds::front();

        using new_merged_values = decltype(MergedValues::push_back(number<chosen_value>{}));
        using new_merged_ids    = decltype(MergedIds::push_back(number<chosen_id>{}));

        using new_left_values = typename std::
            conditional<choose_left, decltype(LeftValues::pop_front()), LeftValues>::type;
        using new_left_ids =
            typename std::conditional<choose_left, decltype(LeftIds::pop_front()), LeftIds>::type;

        using new_right_values = typename std::
            conditional<choose_left, RightValues, decltype(RightValues::pop_front())>::type;
        using new_right_ids =
            typename std::conditional<choose_left, RightIds, decltype(RightIds::pop_front())>::type;

        using merge = sorted_sequence_merge_impl<new_left_values,
                                                 new_left_ids,
                                                 new_right_values,
                                                 new_right_ids,
                                                 new_merged_values,
                                                 new_merged_ids,
                                                 Comp>;
        // this is output
        using merged_values = typename merge::merged_values;
        using merged_ids    = typename merge::merged_ids;
    };

    template <typename LeftValues,
              typename LeftIds,
              typename MergedValues,
              typename MergedIds,
              typename Comp>
    struct sorted_sequence_merge_impl<LeftValues,
                                      LeftIds,
                                      sequence<>,
                                      sequence<>,
                                      MergedValues,
                                      MergedIds,
                                      Comp>
    {
        using merged_values = typename sequence_merge<MergedValues, LeftValues>::type;
        using merged_ids    = typename sequence_merge<MergedIds, LeftIds>::type;
    };

    template <typename RightValues,
              typename RightIds,
              typename MergedValues,
              typename MergedIds,
              typename Comp>
    struct sorted_sequence_merge_impl<sequence<>,
                                      sequence<>,
                                      RightValues,
                                      RightIds,
                                      MergedValues,
                                      MergedIds,
                                      Comp>
    {
        using merged_values = typename sequence_merge<MergedValues, RightValues>::type;
        using merged_ids    = typename sequence_merge<MergedIds, RightIds>::type;
    };

    template <typename LeftValues,
              typename LeftIds,
              typename RightValues,
              typename RightIds,
              typename Comp>
    struct sorted_sequence_merge
    {
        using merge = sorted_sequence_merge_impl<LeftValues,
                                                 LeftIds,
                                                 RightValues,
                                                 RightIds,
                                                 sequence<>,
                                                 sequence<>,
                                                 Comp>;

        using merged_values = typename merge::merged_values;
        using merged_ids    = typename merge::merged_ids;
    };

    static constexpr index_t nsize = Values::size();

    using split_unsorted_values = sequence_split<Values, nsize / 2>;
    using split_unsorted_ids    = sequence_split<Ids, nsize / 2>;

    using left_unsorted_values = typename split_unsorted_values::left_type;
    using left_unsorted_ids    = typename split_unsorted_ids::left_type;
    using left_sort          = sequence_sort_impl<left_unsorted_values, left_unsorted_ids, Compare>;
    using left_sorted_values = typename left_sort::sorted_values;
    using left_sorted_ids    = typename left_sort::sorted_ids;

    using right_unsorted_values = typename split_unsorted_values::right_type;
    using right_unsorted_ids    = typename split_unsorted_ids::right_type;
    using right_sort = sequence_sort_impl<right_unsorted_values, right_unsorted_ids, Compare>;
    using right_sorted_values = typename right_sort::sorted_values;
    using right_sorted_ids    = typename right_sort::sorted_ids;

    using merged_sorted = sorted_sequence_merge<left_sorted_values,
                                                left_sorted_ids,
                                                right_sorted_values,
                                                right_sorted_ids,
                                                Compare>;

    using sorted_values = typename merged_sorted::merged_values;
    using sorted_ids    = typename merged_sorted::merged_ids;
};

template <index_t ValueX, index_t ValueY, index_t IdX, index_t IdY, typename Compare>
struct sequence_sort_impl<sequence<ValueX, ValueY>, sequence<IdX, IdY>, Compare>
{
    static constexpr bool choose_x = Compare{}(ValueX, ValueY);

    using sorted_values = typename std::
        conditional<choose_x, sequence<ValueX, ValueY>, sequence<ValueY, ValueX>>::type;
    using sorted_ids =
        typename std::conditional<choose_x, sequence<IdX, IdY>, sequence<IdY, IdX>>::type;
};

template <index_t Value, index_t Id, typename Compare>
struct sequence_sort_impl<sequence<Value>, sequence<Id>, Compare>
{
    using sorted_values = sequence<Value>;
    using sorted_ids    = sequence<Id>;
};

template <typename Compare>
struct sequence_sort_impl<sequence<>, sequence<>, Compare>
{
    using sorted_values = sequence<>;
    using sorted_ids    = sequence<>;
};

template <typename Values, typename Compare>
struct sequence_sort
{
    using unsorted_ids = typename arithmetic_sequence_gen<0, Values::size(), 1>::type;
    using sort         = sequence_sort_impl<Values, unsorted_ids, Compare>;

    // this is output
    using type                = typename sort::sorted_values;
    using sorted2unsorted_map = typename sort::sorted_ids;
};

template <typename Values, typename Less, typename Equal>
struct sequence_unique_sort
{
    template <typename RemainValues,
              typename RemainIds,
              typename UniquifiedValues,
              typename UniquifiedIds,
              typename Eq>
    struct sorted_sequence_uniquify_impl
    {
        static constexpr index_t current_value = RemainValues::front();
        static constexpr index_t current_id    = RemainIds::front();

        static constexpr bool is_unique_value = (current_value != UniquifiedValues::back());

        using new_remain_values = decltype(RemainValues::pop_front());
        using new_remain_ids    = decltype(RemainIds::pop_front());

        using new_uniquified_values =
            typename std::conditional<is_unique_value,
                                      decltype(UniquifiedValues::push_back(
                                          number<current_value>{})),
                                      UniquifiedValues>::type;

        using new_uniquified_ids =
            typename std::conditional<is_unique_value,
                                      decltype(UniquifiedIds::push_back(number<current_id>{})),
                                      UniquifiedIds>::type;

        using uniquify = sorted_sequence_uniquify_impl<new_remain_values,
                                                       new_remain_ids,
                                                       new_uniquified_values,
                                                       new_uniquified_ids,
                                                       Eq>;

        // this is output
        using uniquified_values = typename uniquify::uniquified_values;
        using uniquified_ids    = typename uniquify::uniquified_ids;
    };

    template <typename UniquifiedValues, typename UniquifiedIds, typename Eq>
    struct sorted_sequence_uniquify_impl<sequence<>,
                                         sequence<>,
                                         UniquifiedValues,
                                         UniquifiedIds,
                                         Eq>
    {
        using uniquified_values = UniquifiedValues;
        using uniquified_ids    = UniquifiedIds;
    };

    template <typename SortedValues, typename SortedIds, typename Eq>
    struct sorted_sequence_uniquify
    {
        using uniquify = sorted_sequence_uniquify_impl<decltype(SortedValues::pop_front()),
                                                       decltype(SortedIds::pop_front()),
                                                       sequence<SortedValues::front()>,
                                                       sequence<SortedIds::front()>,
                                                       Eq>;

        using uniquified_values = typename uniquify::uniquified_values;
        using uniquified_ids    = typename uniquify::uniquified_ids;
    };

    using sort          = sequence_sort<Values, Less>;
    using sorted_values = typename sort::type;
    using sorted_ids    = typename sort::sorted2unsorted_map;

    using uniquify = sorted_sequence_uniquify<sorted_values, sorted_ids, Equal>;

    // this is output
    using type                = typename uniquify::uniquified_values;
    using sorted2unsorted_map = typename uniquify::uniquified_ids;
};

template <typename SeqMap>
struct is_valid_sequence_map
    : std::is_same<typename arithmetic_sequence_gen<0, SeqMap::size(), 1>::type,
                   typename sequence_sort<SeqMap, less<index_t>>::type>
{
};

template <typename SeqMap>
struct sequence_map_inverse
{
    template <typename X2Y, typename WorkingY2X, index_t XBegin, index_t XRemain>
    struct sequence_map_inverse_impl
    {
        static constexpr auto new_y2x =
            WorkingY2X::modify(X2Y::get(number<XBegin>{}), number<XBegin>{});

        using type =
            typename sequence_map_inverse_impl<X2Y, decltype(new_y2x), XBegin + 1, XRemain - 1>::
                type;
    };

    template <typename X2Y, typename WorkingY2X, index_t XBegin>
    struct sequence_map_inverse_impl<X2Y, WorkingY2X, XBegin, 0>
    {
        using type = WorkingY2X;
    };

    using type =
        typename sequence_map_inverse_impl<SeqMap,
                                           typename uniform_sequence_gen<SeqMap::size(), 0>::type,
                                           0,
                                           SeqMap::size()>::type;
};

template <index_t... Xs, index_t... Ys>
CK_TILE_HOST_DEVICE constexpr bool operator==(sequence<Xs...>, sequence<Ys...>)
{
    return ((Xs == Ys) && ...);
}

template <index_t... Xs, index_t... Ys>
CK_TILE_HOST_DEVICE constexpr bool operator!=(sequence<Xs...> x, sequence<Ys...> y)
{
    return !(x == y);
}

template <index_t... Xs, index_t... Ys>
CK_TILE_HOST_DEVICE constexpr auto operator+(sequence<Xs...>, sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return sequence<(Xs + Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
CK_TILE_HOST_DEVICE constexpr auto operator-(sequence<Xs...>, sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return sequence<(Xs - Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
CK_TILE_HOST_DEVICE constexpr auto operator*(sequence<Xs...>, sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return sequence<(Xs * Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
CK_TILE_HOST_DEVICE constexpr auto operator/(sequence<Xs...>, sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return sequence<(Xs / Ys)...>{};
}

template <index_t... Xs, index_t... Ys>
CK_TILE_HOST_DEVICE constexpr auto operator%(sequence<Xs...>, sequence<Ys...>)
{
    static_assert(sizeof...(Xs) == sizeof...(Ys), "wrong! inconsistent size");

    return sequence<(Xs % Ys)...>{};
}

template <index_t... Xs, index_t Y>
CK_TILE_HOST_DEVICE constexpr auto operator+(sequence<Xs...>, number<Y>)
{
    return sequence<(Xs + Y)...>{};
}

template <index_t... Xs, index_t Y>
CK_TILE_HOST_DEVICE constexpr auto operator-(sequence<Xs...>, number<Y>)
{
    return sequence<(Xs - Y)...>{};
}

template <index_t... Xs, index_t Y>
CK_TILE_HOST_DEVICE constexpr auto operator*(sequence<Xs...>, number<Y>)
{
    return sequence<(Xs * Y)...>{};
}

template <index_t... Xs, index_t Y>
CK_TILE_HOST_DEVICE constexpr auto operator/(sequence<Xs...>, number<Y>)
{
    return sequence<(Xs / Y)...>{};
}

template <index_t... Xs, index_t Y>
CK_TILE_HOST_DEVICE constexpr auto operator%(sequence<Xs...>, number<Y>)
{
    return sequence<(Xs % Y)...>{};
}

template <index_t Y, index_t... Xs>
CK_TILE_HOST_DEVICE constexpr auto operator+(number<Y>, sequence<Xs...>)
{
    return sequence<(Y + Xs)...>{};
}

template <index_t Y, index_t... Xs>
CK_TILE_HOST_DEVICE constexpr auto operator-(number<Y>, sequence<Xs...>)
{
    return sequence<(Y - Xs)...>{};
}

template <index_t Y, index_t... Xs>
CK_TILE_HOST_DEVICE constexpr auto operator*(number<Y>, sequence<Xs...>)
{
    return sequence<(Y * Xs)...>{};
}

template <index_t Y, index_t... Xs>
CK_TILE_HOST_DEVICE constexpr auto operator/(number<Y>, sequence<Xs...>)
{
    return sequence<(Y / Xs)...>{};
}

template <index_t Y, index_t... Xs>
CK_TILE_HOST_DEVICE constexpr auto operator%(number<Y>, sequence<Xs...>)
{
    return sequence<(Y % Xs)...>{};
}

template <index_t I, index_t... Is>
CK_TILE_HOST_DEVICE constexpr auto sequence_pop_front(sequence<I, Is...>)
{
    return sequence<Is...>{};
}

template <typename Seq>
CK_TILE_HOST_DEVICE constexpr auto sequence_pop_back(Seq)
{
    static_assert(Seq::size() > 0, "wrong! cannot pop an empty sequence!");
    return sequence_pop_front(Seq::reverse()).reverse();
}

template <typename... Seqs>
CK_TILE_HOST_DEVICE constexpr auto merge_sequences(Seqs...)
{
    return typename sequence_merge<Seqs...>::type{};
}

template <typename F, index_t... Xs>
CK_TILE_HOST_DEVICE constexpr auto transform_sequences(F f, sequence<Xs...>)
{
    return sequence<f(Xs)...>{};
}

template <typename F, index_t... Xs, index_t... Ys>
CK_TILE_HOST_DEVICE constexpr auto transform_sequences(F f, sequence<Xs...>, sequence<Ys...>)
{
    static_assert(sequence<Xs...>::size() == sequence<Ys...>::size(), "Dim not the same");

    return sequence<f(Xs, Ys)...>{};
}

template <typename F, index_t... Xs, index_t... Ys, index_t... Zs>
CK_TILE_HOST_DEVICE constexpr auto
transform_sequences(F f, sequence<Xs...>, sequence<Ys...>, sequence<Zs...>)
{
    static_assert(sequence<Xs...>::size() == sequence<Ys...>::size() &&
                      sequence<Xs...>::size() == sequence<Zs...>::size(),
                  "Dim not the same");

    return sequence<f(Xs, Ys, Zs)...>{};
}

template <typename Seq, typename Reduce, index_t Init>
CK_TILE_HOST_DEVICE constexpr auto reverse_inclusive_scan_sequence(Seq, Reduce, number<Init>)
{
    return typename sequence_reverse_inclusive_scan<Seq, Reduce, Init>::type{};
}

template <typename Seq, typename Reduce, index_t Init>
CK_TILE_HOST_DEVICE constexpr auto reverse_exclusive_scan_sequence(Seq, Reduce, number<Init>)
{
    return reverse_inclusive_scan_sequence(Seq::pop_front(), Reduce{}, number<Init>{})
        .push_back(number<Init>{});
}

template <typename Seq, typename Reduce, index_t Init>
CK_TILE_HOST_DEVICE constexpr auto inclusive_scan_sequence(Seq, Reduce, number<Init>)
{
    return reverse_inclusive_scan_sequence(Seq{}.reverse(), Reduce{}, number<Init>{}).reverse();
}

// e.g. Seq<2, 3, 4> --> Seq<0, 2, 5>, Init=0, Reduce=Add
//      ResultSeq  TargetSeq  Reduce
template <typename, typename, typename>
struct sequence_exclusive_scan;

template <index_t... Xs, index_t Y, index_t... Ys, typename Reduce>
struct sequence_exclusive_scan<sequence<Xs...>, sequence<Y, Ys...>, Reduce>
{
    using old_scan = typename sequence_merge<sequence<Xs...>,
                                             sequence<Reduce{}(Y, sequence<Xs...>{}.back())>>::type;
    using type     = typename sequence_exclusive_scan<old_scan, sequence<Ys...>, Reduce>::type;
};

template <index_t... Xs, index_t Y, typename Reduce>
struct sequence_exclusive_scan<sequence<Xs...>, sequence<Y>, Reduce>
{
    using type = sequence<Xs...>;
};

template <index_t... Xs, typename Reduce>
struct sequence_exclusive_scan<sequence<Xs...>, sequence<>, Reduce>
{
    using type = sequence<Xs...>;
};

template <typename Seq, typename Reduce, index_t Init>
constexpr auto exclusive_scan_sequence(Seq, Reduce, number<Init>)
{
    // TODO: c++20 and later can pass in Reduce with a lambda expression
    return typename sequence_exclusive_scan<sequence<Init>, Seq, Reduce>::type{};
}

template <typename Seq>
constexpr auto prefix_sum_sequence(Seq)
{
    return typename sequence_exclusive_scan<sequence<0>,
                                            typename sequence_merge<Seq, sequence<0>>::type,
                                            plus<index_t>>::type{};
}

template <typename Seq, index_t... Is>
CK_TILE_HOST_DEVICE constexpr auto pick_sequence_elements_by_ids(Seq, sequence<Is...> /* ids */)
{
    return sequence<Seq::get(number<Is>{})...>{};
}

#if 1
namespace detail {
template <typename WorkSeq, typename RemainSeq, typename RemainMask>
struct pick_sequence_elements_by_mask_impl
{
    using new_work_seq = typename std::conditional<RemainMask::front(),
                                                   decltype(WorkSeq::push_back(RemainSeq::front())),
                                                   WorkSeq>::type;

    using type =
        typename pick_sequence_elements_by_mask_impl<new_work_seq,
                                                     decltype(RemainSeq::pop_front()),
                                                     decltype(RemainMask::pop_front())>::type;
};

template <typename WorkSeq>
struct pick_sequence_elements_by_mask_impl<WorkSeq, sequence<>, sequence<>>
{
    using type = WorkSeq;
};

} // namespace detail

template <typename Seq, typename Mask>
CK_TILE_HOST_DEVICE constexpr auto pick_sequence_elements_by_mask(Seq, Mask)
{
    static_assert(Seq::size() == Mask::size(), "wrong!");

    return typename detail::pick_sequence_elements_by_mask_impl<sequence<>, Seq, Mask>::type{};
}

namespace detail {
template <typename WorkSeq, typename RemainValues, typename RemainIds>
struct modify_sequence_elements_by_ids_impl
{
    using new_work_seq = decltype(WorkSeq::modify(RemainIds::front(), RemainValues::front()));

    using type =
        typename modify_sequence_elements_by_ids_impl<new_work_seq,
                                                      decltype(RemainValues::pop_front()),
                                                      decltype(RemainIds::pop_front())>::type;
};

template <typename WorkSeq>
struct modify_sequence_elements_by_ids_impl<WorkSeq, sequence<>, sequence<>>
{
    using type = WorkSeq;
};
} // namespace detail

template <typename Seq, typename Values, typename Ids>
CK_TILE_HOST_DEVICE constexpr auto modify_sequence_elements_by_ids(Seq, Values, Ids)
{
    static_assert(Values::size() == Ids::size() && Seq::size() >= Values::size(), "wrong!");

    return typename detail::modify_sequence_elements_by_ids_impl<Seq, Values, Ids>::type{};
}
#endif

template <typename Seq, typename Reduce, index_t Init>
CK_TILE_HOST_DEVICE constexpr index_t
reduce_on_sequence(Seq, Reduce f, number<Init> /*initial_value*/)
{
    index_t result = Init;

    for(index_t i = 0; i < Seq::size(); ++i)
    {
        result = f(result, Seq::at(i));
    }

    return result;
}

// TODO: a generic any_of for any container
template <typename Seq, typename F>
CK_TILE_HOST_DEVICE constexpr bool sequence_any_of(Seq, F f)
{
    bool flag = false;

    for(index_t i = 0; i < Seq::size(); ++i)
    {
        flag = flag || f(Seq::at(i));
    }

    return flag;
}

// TODO: a generic all_of for any container
template <typename Seq, typename F>
CK_TILE_HOST_DEVICE constexpr bool sequence_all_of(Seq, F f)
{
    bool flag = true;

    for(index_t i = 0; i < Seq::size(); ++i)
    {
        flag = flag && f(Seq::at(i));
    }

    return flag;
}

template <typename... Seqs>
using sequence_merge_t = typename sequence_merge<Seqs...>::type;

template <index_t NSize, index_t I>
using uniform_sequence_gen_t = typename uniform_sequence_gen<NSize, I>::type;

template <index_t... Is>
CK_TILE_HOST_DEVICE constexpr auto make_sequence(number<Is>...)
{
    return sequence<Is...>{};
}

// F() returns index_t
// F use default constructor, so F cannot be lambda function
template <typename F, index_t N>
CK_TILE_HOST_DEVICE constexpr auto generate_sequence(F, number<N>)
{
    return typename sequence_gen<N, F>::type{};
}

// F() returns number<>
// F could be lambda function
template <typename F, index_t N>
CK_TILE_HOST_DEVICE constexpr auto generate_sequence_v2(F&& f, number<N>)
{
    return unpack([&f](auto&&... xs) { return make_sequence(f(xs)...); },
                  typename arithmetic_sequence_gen<0, N, 1>::type{});
}

template <class... T>
struct tuple;

template <index_t... Is>
CK_TILE_HOST_DEVICE constexpr auto to_sequence(tuple<number<Is>...>)
{
    return sequence<Is...>{};
}

namespace detail {
template <index_t h_idx, typename SeqSortedSamples, typename SeqRange>
struct sorted_sequence_histogram;

template <index_t h_idx, index_t x, index_t... xs, index_t r, index_t... rs>
struct sorted_sequence_histogram<h_idx, sequence<x, xs...>, sequence<r, rs...>>
{
    template <typename Histogram>
    constexpr auto operator()(Histogram& h)
    {
        if constexpr(x < r)
        {
            h.template at<h_idx>() += 1;
            sorted_sequence_histogram<h_idx, sequence<xs...>, sequence<r, rs...>>{}(h);
        }
        else
        {
            h.template at<h_idx + 1>() = 1;
            sorted_sequence_histogram<h_idx + 1, sequence<xs...>, sequence<rs...>>{}(h);
        }
    }
};

template <index_t h_idx, index_t x, index_t r, index_t... rs>
struct sorted_sequence_histogram<h_idx, sequence<x>, sequence<r, rs...>>
{
    template <typename Histogram>
    constexpr auto operator()(Histogram& h)
    {
        if constexpr(x < r)
        {
            h.template at<h_idx>() += 1;
        }
    }
};
} // namespace detail

template <typename, index_t>
struct array; // declare for later use (array->seq utility)

// SeqSortedSamples: <0, 2, 3, 5, 7>, SeqRange: <0, 3, 6, 9> -> SeqHistogram : <2, 2, 1>
template <typename SeqSortedSamples, index_t r, index_t... rs>
CK_TILE_HOST_DEVICE constexpr auto histogram_sorted_sequence(SeqSortedSamples, sequence<r, rs...>)
{
    constexpr auto bins      = sizeof...(rs); // or categories
    constexpr auto histogram = [&]() {
        array<index_t, bins> h{0}; // make sure this can clear all element to zero
        detail::sorted_sequence_histogram<0, SeqSortedSamples, sequence<rs...>>{}(h);
        return h;
    }();

    return TO_SEQUENCE(histogram, bins);
}

template <typename F, index_t N>
CK_TILE_HOST_DEVICE constexpr auto generate_array(F&& f, number<N>)
{
    using T = remove_cvref_t<decltype(f(number<0>{}))>;

    return unpack([&f](auto&&... is) { return array<T, N>{f(is)...}; },
                  typename arithmetic_sequence_gen<0, N, 1>::type{});
}

namespace impl {
template <typename, typename, typename, index_t>
struct reverse_slice_sequence_impl;

template <index_t x,
          index_t... xs,
          index_t m,
          index_t... ms,
          index_t id,
          index_t... ids,
          index_t SliceSize>
struct reverse_slice_sequence_impl<sequence<x, xs...>,
                                   sequence<m, ms...>,
                                   sequence<id, ids...>,
                                   SliceSize>
{
    using old_scan =
        reverse_slice_sequence_impl<sequence<xs...>, sequence<ms...>, sequence<ids...>, SliceSize>;

    static constexpr auto slice_size = old_scan::remaining_slice_sizes::front().value;
    static constexpr auto slice_length =
        std::conditional_t<m, number<gcd(x, slice_size)>, number<x>>::value;

    using dim_lengths =
        typename sequence_merge<sequence<slice_length>, typename old_scan::dim_lengths>::type;
    using dim_slices =
        typename sequence_merge<sequence<x / slice_length>, typename old_scan::dim_slices>::type;
    using remaining_slice_sizes = typename sequence_merge<
        std::conditional_t<m, sequence<slice_size / slice_length>, sequence<slice_size>>,
        typename old_scan::remaining_slice_sizes>::type;

    // the first idx that sliced length not equal to original length
    static constexpr index_t _flag =
        slice_length != x && remaining_slice_sizes{}.front().value == 1;
    static constexpr index_t _split_flag = std::conditional_t<m, number<_flag>, number<0>>::value;
    static constexpr index_t _split_idx =
        std::conditional_t<_split_flag, number<id>, number<0>>::value;

    static constexpr index_t split_flag = _split_flag || old_scan::split_flag;
    static constexpr index_t split_idx  = std::
        conditional_t<old_scan::split_flag, number<old_scan::split_idx>, number<_split_idx>>::value;
};

template <index_t x, index_t m, index_t id, index_t SliceSize>
struct reverse_slice_sequence_impl<sequence<x>, sequence<m>, sequence<id>, SliceSize>
{
    static constexpr auto slice_size = SliceSize;
    static constexpr auto slice_length =
        std::conditional_t<m, number<gcd(x, slice_size)>, number<x>>::value;

    using dim_lengths = sequence<slice_length>;
    using dim_slices  = sequence<x / slice_length>;
    using remaining_slice_sizes =
        std::conditional_t<m, sequence<slice_size / slice_length>, sequence<slice_size>>;

    // the first idx that sliced length not equal to original length
    static constexpr index_t _flag =
        slice_length != x && remaining_slice_sizes{}.front().value == 1;
    static constexpr index_t split_flag = std::conditional_t<m, number<_flag>, number<0>>::value;
    static constexpr index_t split_idx =
        std::conditional_t<split_flag, number<id>, number<0>>::value;
};
} // namespace impl

// clang-format off
// input a sequence(with optional mask), and the SliceSize : size per slice
// output the sequence each slice, and number of slices
//
// e.g. <2, 1, 4, 2>, 8     -> lengths:<1, 1, 4, 2>    , nums: <2, 1, 1, 1>    : 2 slices  , slice_idx: 0
//      <4, 2, 4, 1, 2>, 4  -> lengths:<1, 1, 2, 1, 2> , nums: <4, 2, 2, 1, 1> : 16 slices , slice_idx: 2
//      <4, 2, 4, 1, 6>, 4  -> lengths:<1, 1, 2, 1, 2> , nums: <4, 2, 2, 1, 3> : 48 slices , slice_idx: 2
//      <4, 2, 5, 1, 2>, 10 -> lengths:<1, 1, 5, 1, 2> , nums: <4, 2, 1, 1, 1> : 8 slices  , slice_idx: 1
//
//      <4, 2, 8>, 64       -> lengths:<4, 2, 8>       , nums: <1, 1, 1>       : 1  slices , slice_idx: 0
//      <4, 2, 8>, 32       -> lengths:<2, 2, 8>       , nums: <2, 1, 1>       : 2  slices , slice_idx: 0
//      <4, 2, 8>, 16       -> lengths:<1, 2, 8>       , nums: <4, 1, 1>       : 4  slices , slice_idx: 0
//      <4, 2, 8>, 8        -> lengths:<1, 1, 8>       , nums: <4, 2, 1>       : 8  slices , slice_idx: 1
//      <4, 2, 8>, 4        -> lengths:<1, 1, 4>       , nums: <4, 2, 2>       : 16 slices , slice_idx: 2
//      <4, 2, 8>, 2        -> lengths:<1, 1, 2>       , nums: <4, 2, 4>       : 32 slices , slice_idx: 2
//      <4, 2, 8>, 1        -> lengths:<1, 1, 1>       , nums: <4, 2, 8>       : 64 slices , slice_idx: 2
//
//      <4, 2, 1, 4, 2> / 4 ->
// mask:<1, 1, 1, 0, 1>,    -> lengths:<1, 2, 1, 4, 2> , nums: <4, 1, 1, 1, 1> : 8 slices  , slice_idx: 0
//
// return tuple<slice_lengths, slice_nums, slice_index>, slice_index is at which index will start
// have split slices (right -> left)
//  or the first index that sliced length is different from the original length
// clang-format on
template <typename Seq,
          index_t SliceSize,
          typename Mask = typename uniform_sequence_gen<Seq::size(), 1>::type>
constexpr auto reverse_slice_sequence(Seq,
                                      number<SliceSize>,
                                      Mask = typename uniform_sequence_gen<Seq::size(), 1>::type{})
{
    static_assert(Seq::size() == Mask::size());
    using sliced_type =
        impl::reverse_slice_sequence_impl<Seq,
                                          Mask,
                                          typename arithmetic_sequence_gen<0, Seq::size(), 1>::type,
                                          SliceSize>;
    static_assert(sliced_type::remaining_slice_sizes::front().value == 1,
                  "can not evenly divide this sequence, please check");
    return make_tuple(typename sliced_type::dim_lengths{},
                      typename sliced_type::dim_slices{},
                      number<sliced_type::split_idx>{});
}

template <typename Seq,
          index_t SliceSize,
          typename Mask = typename uniform_sequence_gen<Seq::size(), 1>::type>
constexpr auto slice_sequence(Seq,
                              number<SliceSize>,
                              Mask = typename uniform_sequence_gen<Seq::size(), 1>::type{})
{
    constexpr auto r =
        reverse_slice_sequence(Seq{}.reverse(), number<SliceSize>{}, Mask{}.reverse());
    return make_tuple(r[number<0>{}].reverse(),
                      r[number<1>{}].reverse(),
                      number<Seq::size() - r[number<2>{}] - 1>{});
}

} // namespace ck_tile
