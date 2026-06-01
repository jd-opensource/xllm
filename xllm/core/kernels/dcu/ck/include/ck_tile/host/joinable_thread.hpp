// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include <thread>
#include <utility>

namespace ck_tile {

struct joinable_thread : std::thread
{
    template <typename... Xs>
    joinable_thread(Xs&&... xs) : std::thread(std::forward<Xs>(xs)...)
    {
    }

    joinable_thread(joinable_thread&&) = default;
    joinable_thread& operator=(joinable_thread&&) = default;

    ~joinable_thread()
    {
        if(this->joinable())
            this->join();
    }
};
} // namespace ck_tile
