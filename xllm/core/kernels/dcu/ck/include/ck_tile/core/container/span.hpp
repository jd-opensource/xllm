// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include <cstddef>
#include <array>
#include <type_traits>

namespace ck_tile {

// implement the c++20 std::span, lightweight, non-owning reference to a sequence
// weather it is dynamic or static range. Or can be seen as a view of a contiguous sequence
// TODO: do we need in device consider this is pointer?
template <typename T>
class span
{
    public:
    using element_type    = T;
    using value_type      = std::remove_cv_t<element_type>;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer         = element_type*;
    using const_pointer   = const element_type*;
    using reference       = element_type&;
    using const_reference = const element_type&;
    using iterator        = pointer;
    using const_iterator  = pointer;

    CK_TILE_HOST_DEVICE constexpr span() : span(nullptr, size_type{0}) {}

    CK_TILE_HOST_DEVICE constexpr span(pointer first, size_type count) : ptr_(first), size_(count)
    {
    }

    CK_TILE_HOST_DEVICE constexpr span(pointer first, pointer last) : span(first, last - first) {}

    template <std::size_t N>
    CK_TILE_HOST_DEVICE constexpr span(element_type (&arr)[N]) noexcept : span(arr, N)
    {
    }

    template <std::size_t N>
    CK_TILE_HOST_DEVICE constexpr span(std::array<value_type, N>& arr) noexcept
        : span(arr.data(), N)
    {
    }

    template <typename Container>
    CK_TILE_HOST_DEVICE constexpr span(const Container& container)
        : span(container.data(), container.size())
    {
    }

    CK_TILE_HOST_DEVICE constexpr iterator begin() const noexcept { return ptr_; }
    CK_TILE_HOST_DEVICE constexpr const_iterator cbegin() const noexcept { return begin(); }

    CK_TILE_HOST_DEVICE constexpr iterator end() const noexcept { return begin() + size(); }
    CK_TILE_HOST_DEVICE constexpr const_iterator cend() const noexcept { return end(); }

    CK_TILE_HOST_DEVICE constexpr reference front() const { return *begin(); }
    CK_TILE_HOST_DEVICE constexpr reference back() const { return *(--end()); }

    CK_TILE_HOST_DEVICE constexpr reference operator[](size_type idx) const
    {
        return *(begin() + idx);
    }
    CK_TILE_HOST_DEVICE constexpr pointer data() const noexcept { return ptr_; }

    CK_TILE_HOST_DEVICE constexpr size_type size() const noexcept { return size_; }

    private:
    pointer ptr_;
    size_type size_;
};

} // namespace ck_tile
