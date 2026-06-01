// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/container/tuple.hpp"

namespace ck_tile {

// naive map
template <typename key, typename data, index_t max_size = 128>
struct map
{
    using pair_type = tuple<key, data>;
    using impl_type = array<pair_type, max_size>;

    impl_type impl_;
    index_t size_;

    struct iterator
    {
        impl_type& impl_;
        index_t pos_;

        CK_TILE_HOST_DEVICE constexpr iterator(impl_type& impl, index_t pos)
            : impl_{impl}, pos_{pos}
        {
        }

        CK_TILE_HOST_DEVICE constexpr iterator& operator++()
        {
            pos_++;
            return *this;
        }

        CK_TILE_HOST_DEVICE constexpr bool operator!=(const iterator& other) const
        {
            return other.pos_ != pos_;
        }

        CK_TILE_HOST_DEVICE constexpr pair_type& operator*() { return impl_.at(pos_); }
    };

    struct const_iterator
    {
        const impl_type& impl_;
        index_t pos_;

        CK_TILE_HOST_DEVICE constexpr const_iterator(const impl_type& impl, index_t pos)
            : impl_{impl}, pos_{pos}
        {
        }

        CK_TILE_HOST_DEVICE constexpr const_iterator& operator++()
        {
            pos_++;

            return *this;
        }

        CK_TILE_HOST_DEVICE constexpr bool operator!=(const const_iterator& other) const
        {
            return other.pos_ != pos_;
        }

        CK_TILE_HOST_DEVICE constexpr const pair_type& operator*() const { return impl_.at(pos_); }
    };

    CK_TILE_HOST_DEVICE constexpr map() : impl_{}, size_{0} {}

    CK_TILE_HOST_DEVICE constexpr index_t size() const { return size_; }

    CK_TILE_HOST_DEVICE void clear() { size_ = 0; }

    CK_TILE_HOST_DEVICE constexpr index_t find_position(const key& k) const
    {
        for(index_t i = 0; i < size(); i++)
        {
            if(impl_[i].template at<0>() == k)
            {
                return i;
            }
        }

        return size_;
    }

    CK_TILE_HOST_DEVICE constexpr const_iterator find(const key& k) const
    {
        return const_iterator{impl_, find_position(k)};
    }

    CK_TILE_HOST_DEVICE constexpr iterator find(const key& k)
    {
        return iterator{impl_, find_position(k)};
    }

    CK_TILE_HOST_DEVICE constexpr const data& operator[](const key& k) const
    {
        const auto it = find(k);

        // FIXME
        // assert(it.pos_ < size());

        return impl_[it.pos_].template at<1>();
    }

    CK_TILE_HOST_DEVICE constexpr data& operator()(const key& k)
    {
        auto it = find(k);

        // if entry not found
        if(it.pos_ == size())
        {
            impl_(it.pos_).template at<0>() = k;
            size_++;
        }

        // FIXME
        // assert(size_ <= max_size);

        return impl_(it.pos_).template at<1>();
    }

    // WARNING: needed by compiler for C++ range-based for loop only, don't use this function!
    CK_TILE_HOST_DEVICE constexpr const_iterator begin() const { return const_iterator{impl_, 0}; }

    // WARNING: needed by compiler for C++ range-based for loop only, don't use this function!
    CK_TILE_HOST_DEVICE constexpr const_iterator end() const
    {
        return const_iterator{impl_, size_};
    }

    // WARNING: needed by compiler for C++ range-based for loop only, don't use this function!
    CK_TILE_HOST_DEVICE constexpr iterator begin() { return iterator{impl_, 0}; }

    // WARNING: needed by compiler for C++ range-based for loop only, don't use this function!
    CK_TILE_HOST_DEVICE constexpr iterator end() { return iterator{impl_, size_}; }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("map{size_: %d, ", size_);
        //
        printf("impl_: [");
        //
        for(const auto& [k, d] : *this)
        {
            printf("{key: ");
            print(k);
            printf(", data: ");
            print(d);
            printf("}, ");
        }
        //
        printf("]");
        //
        printf("}");
    }
};

} // namespace ck_tile
