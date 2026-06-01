// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename F, typename... Fs>
struct composes : private composes<F>
{
    template <typename FirstArg, typename... RestArgs>
    CK_TILE_HOST_DEVICE constexpr explicit composes(FirstArg&& firstArg, RestArgs&&... restArgs)
        : composes<F>(std::forward<FirstArg>(firstArg)), inner_(std::forward<RestArgs>(restArgs)...)
    {
    }

    template <typename Arg>
    CK_TILE_HOST_DEVICE constexpr auto operator()(Arg&& arg) const
    {
        return static_cast<const composes<F>&>(*this)(inner_(std::forward<Arg>(arg)));
    }

    private:
    composes<Fs...> inner_;
};

template <typename F>
struct composes<F>
{
    static_assert(!std::is_reference_v<F>);

    template <typename Arg, typename = std::enable_if_t<std::is_constructible_v<F, Arg>>>
    CK_TILE_HOST_DEVICE constexpr explicit composes(Arg&& arg) : f_(std::forward<Arg>(arg))
    {
    }

    template <typename Arg,
              typename = std::enable_if_t<std::is_invocable_v<std::add_const_t<F>&, Arg>>>
    CK_TILE_HOST_DEVICE constexpr auto operator()(Arg&& arg) const
    {
        return f_(std::forward<Arg>(arg));
    }

    private:
    F f_;
};

/// FIXME: create macro to replace '__host__ __device__' and nothing more
template <typename... Ts>
__host__ __device__ composes(Ts&&...)->composes<remove_cvref_t<Ts>...>;

template <typename To>
struct saturates
{
    template <typename From>
    CK_TILE_HOST_DEVICE constexpr auto operator()(const From& from) const
        -> std::enable_if_t<std::is_arithmetic_v<From>, From>
    {
        return clamp(from,
                     type_convert<From>(numeric<To>::lowest()),
                     type_convert<From>(numeric<To>::max()));
    }
};

} // namespace ck_tile
