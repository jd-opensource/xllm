// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <cmath>
#include <iterator>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>
#include <unordered_set>

#include "ck_tile/core.hpp"
#include "ck_tile/host/joinable_thread.hpp"

namespace ck_tile {

template <typename T>
struct FillUniformDistribution
{
    float a_{-5.f};
    float b_{5.f};
    std::optional<uint32_t> seed_{11939};
    // ATTENTION: threaded does not guarantee the distribution between thread
    bool threaded = false;

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        if(threaded)
        {
            uint32_t num_thread  = std::thread::hardware_concurrency();
            auto total           = static_cast<std::size_t>(std::distance(first, last));
            auto work_per_thread = static_cast<std::size_t>((total + num_thread - 1) / num_thread);

            std::vector<joinable_thread> threads(num_thread);
            for(std::size_t it = 0; it < num_thread; ++it)
            {
                std::size_t iw_begin = it * work_per_thread;
                std::size_t iw_end   = std::min((it + 1) * work_per_thread, total);
                auto thread_f        = [this, total, iw_begin, iw_end, &first] {
                    if(iw_begin > total || iw_end > total)
                        return;
                    // need to make each thread unique, add an offset to current seed
                    std::mt19937 gen(seed_.has_value() ? (*seed_ + iw_begin)
                                                              : std::random_device{}());
                    std::uniform_real_distribution<float> dis(a_, b_);
                    std::generate(first + iw_begin, first + iw_end, [&dis, &gen]() {
                        return ck_tile::type_convert<T>(dis(gen));
                    });
                };
                threads[it] = joinable_thread(thread_f);
            }
        }
        else
        {
            std::mt19937 gen(seed_.has_value() ? *seed_ : std::random_device{}());
            std::uniform_real_distribution<float> dis(a_, b_);
            std::generate(
                first, last, [&dis, &gen]() { return ck_tile::type_convert<T>(dis(gen)); });
        }
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const FillUniformDistribution&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

namespace impl {

// clang-format off
template<index_t bytes> struct RawIntegerType_ {};
template<> struct RawIntegerType_<1> { using type = uint8_t;};
template<> struct RawIntegerType_<2> { using type = uint16_t;};
template<> struct RawIntegerType_<4> { using type = uint32_t;};
template<> struct RawIntegerType_<8> { using type = uint64_t;};
// clang-format on

template <typename T>
using RawIntegerType = typename RawIntegerType_<sizeof(T)>::type;
} // namespace impl

// Note: this struct will have no const-ness will generate random
template <typename T>
struct FillUniformDistribution_Unique
{
    float a_{-5.f};
    float b_{5.f};
    std::optional<uint32_t> seed_{11939};

    std::mt19937 gen_{};
    std::unordered_set<impl::RawIntegerType<T>> set_{};

    FillUniformDistribution_Unique(float a                      = -5.f,
                                   float b                      = 5.f,
                                   std::optional<uint32_t> seed = {11939})
        : a_(a),
          b_(b),
          seed_(seed),
          gen_{seed_.has_value() ? *seed_ : std::random_device{}()},
          set_{}
    {
    }

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last)
    {
        std::mt19937& gen = gen_;
        std::uniform_real_distribution<float> dis(a_, b_);
        auto& set = set_;
        std::generate(first, last, [&dis, &gen, &set]() {
            T v = static_cast<T>(0);
            do
            {
                v = ck_tile::type_convert<T>(dis(gen));
            } while(set.count(bit_cast<impl::RawIntegerType<T>>(v)) == 1);
            set.insert(bit_cast<impl::RawIntegerType<T>>(v));

            return v;
        });
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range)
        -> std::void_t<decltype(std::declval<FillUniformDistribution_Unique&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }

    void clear() { set_.clear(); }
};

template <typename T>
struct FillNormalDistribution
{
    float mean_{0.f};
    float variance_{1.f};
    std::optional<uint32_t> seed_{11939};
    // ATTENTION: threaded does not guarantee the distribution between thread
    bool threaded = false;

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        if(threaded)
        {
            uint32_t num_thread  = std::thread::hardware_concurrency();
            auto total           = static_cast<std::size_t>(std::distance(first, last));
            auto work_per_thread = static_cast<std::size_t>((total + num_thread - 1) / num_thread);

            std::vector<joinable_thread> threads(num_thread);
            for(std::size_t it = 0; it < num_thread; ++it)
            {
                std::size_t iw_begin = it * work_per_thread;
                std::size_t iw_end   = std::min((it + 1) * work_per_thread, total);
                auto thread_f        = [this, total, iw_begin, iw_end, &first] {
                    if(iw_begin > total || iw_end > total)
                        return;
                    // need to make each thread unique, add an offset to current seed
                    std::mt19937 gen(seed_.has_value() ? (*seed_ + iw_begin)
                                                              : std::random_device{}());
                    std::normal_distribution<float> dis(mean_, std::sqrt(variance_));
                    std::generate(first + iw_begin, first + iw_end, [&dis, &gen]() {
                        return ck_tile::type_convert<T>(dis(gen));
                    });
                };
                threads[it] = joinable_thread(thread_f);
            }
        }
        else
        {
            std::mt19937 gen(seed_.has_value() ? *seed_ : std::random_device{}());
            std::normal_distribution<float> dis(mean_, std::sqrt(variance_));
            std::generate(
                first, last, [&dis, &gen]() { return ck_tile::type_convert<T>(dis(gen)); });
        }
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const FillNormalDistribution&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

// Normally FillUniformDistributionIntegerValue should use std::uniform_int_distribution as below.
// However this produces segfaults in std::mt19937 which look like inifite loop.
//      template <typename T>
//      struct FillUniformDistributionIntegerValue
//      {
//          int a_{-5};
//          int b_{5};
//
//          template <typename ForwardIter>
//          void operator()(ForwardIter first, ForwardIter last) const
//          {
//              std::mt19937 gen(11939);
//              std::uniform_int_distribution<int> dis(a_, b_);
//              std::generate(
//                  first, last, [&dis, &gen]() { return ck_tile::type_convert<T>(dis(gen)); });
//          }
//      };

// Workaround for uniform_int_distribution not working as expected. See note above.<
template <typename T>
struct FillUniformDistributionIntegerValue
{
    float a_{-5.f};
    float b_{5.f};
    std::optional<uint32_t> seed_{11939};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::mt19937 gen(seed_.has_value() ? *seed_ : std::random_device{}());
        std::uniform_real_distribution<float> dis(a_, b_);
        std::generate(
            first, last, [&dis, &gen]() { return ck_tile::type_convert<T>(std::round(dis(gen))); });
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const FillUniformDistributionIntegerValue&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

template <typename T>
struct FillNormalDistributionIntegerValue
{
    float mean_{0.f};
    float variance_{1.f};
    std::optional<uint32_t> seed_{11939};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::mt19937 gen(seed_.has_value() ? *seed_ : std::random_device{}());
        std::normal_distribution<float> dis(mean_, std::sqrt(variance_));
        std::generate(
            first, last, [&dis, &gen]() { return ck_tile::type_convert<T>(std::round(dis(gen))); });
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const FillNormalDistributionIntegerValue&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

template <typename T>
struct FillMonotonicSeq
{
    T init_value_{0};
    T step_{1};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::generate(first, last, [=, n = init_value_]() mutable {
            auto tmp = n;
            if constexpr(std::is_same_v<decltype(tmp), pk_int4_t>)
            {
                n.data += step_.data;
            }
            else
            {
                n += step_;
            }
            return tmp;
        });
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const
        -> std::void_t<decltype(std::declval<const FillMonotonicSeq&>()(
            std::begin(std::forward<ForwardRange>(range)),
            std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

template <typename T, bool IsAscending = true>
struct FillStepRange
{
    float start_value_{0};
    float end_value_{3};
    float step_{1};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::generate(first, last, [=, n = start_value_]() mutable {
            auto tmp = n;
            n += step_;
            if constexpr(IsAscending)
            {
                if(n > end_value_)
                    n = start_value_;
            }
            else
            {
                if(n < end_value_)
                    n = start_value_;
            }

            return type_convert<T>(tmp);
        });
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const -> std::void_t<
        decltype(std::declval<const FillStepRange&>()(std::begin(std::forward<ForwardRange>(range)),
                                                      std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

template <typename T>
struct FillConstant
{
    T value_{0};

    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        std::fill(first, last, value_);
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const -> std::void_t<
        decltype(std::declval<const FillConstant&>()(std::begin(std::forward<ForwardRange>(range)),
                                                     std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

template <typename T, bool UseCos = true, bool UseAbs = false>
struct FillTrigValue
{
    template <typename T_, bool UseCos_ = true, bool UseAbs_ = false>
    struct LinearTrigGen
    {
        int i{0};
        auto operator()()
        {
            float v = 0;
            if constexpr(UseCos_)
            {
                v = cos(i);
            }
            else
            {
                v = sin(i);
            }
            if constexpr(UseAbs_)
                v = abs(v);
            i++;
            return ck_tile::type_convert<T_>(v);
        }
    };
    template <typename ForwardIter>
    void operator()(ForwardIter first, ForwardIter last) const
    {
        LinearTrigGen<T, UseCos, UseAbs> gen;
        std::generate(first, last, gen);
    }

    template <typename ForwardRange>
    auto operator()(ForwardRange&& range) const -> std::void_t<
        decltype(std::declval<const FillTrigValue&>()(std::begin(std::forward<ForwardRange>(range)),
                                                      std::end(std::forward<ForwardRange>(range))))>
    {
        (*this)(std::begin(std::forward<ForwardRange>(range)),
                std::end(std::forward<ForwardRange>(range)));
    }
};

} // namespace ck_tile
