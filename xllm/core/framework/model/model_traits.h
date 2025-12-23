/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <type_traits>

namespace xllm {
namespace layer {
class LmHead;
class WordEmbedding;
#if defined(USE_NPU)
class NpuLmHead;
class NpuWordEmbedding;
#endif
}  // namespace layer

namespace detail {
template <typename T, typename = void>
struct has_get_lm_head : std::false_type {};

template <typename T>
struct has_get_lm_head<T,
                       std::void_t<decltype(std::declval<T>()->get_lm_head())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_set_lm_head : std::false_type {};

template <typename T>
struct has_set_lm_head<T,
                       std::void_t<decltype(std::declval<T>()->set_lm_head(
                           std::declval<layer::LmHead&>()))>> : std::true_type {
};

template <typename T, typename = void>
struct has_get_word_embedding : std::false_type {};

template <typename T>
struct has_get_word_embedding<
    T,
    std::void_t<decltype(std::declval<T>()->get_word_embedding())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_set_word_embedding : std::false_type {};

template <typename T>
struct has_set_word_embedding<
    T,
    std::void_t<decltype(std::declval<T>()->set_word_embedding(
        std::declval<layer::WordEmbedding&>()))>> : std::true_type {};

#if defined(USE_NPU)
template <typename T, typename = void>
struct has_get_npu_lm_head : std::false_type {};

template <typename T>
struct has_get_npu_lm_head<
    T,
    std::void_t<decltype(std::declval<T>()->get_npu_lm_head())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_set_npu_lm_head : std::false_type {};

template <typename T>
struct has_set_npu_lm_head<
    T,
    std::void_t<decltype(std::declval<T>()->set_npu_lm_head(
        std::declval<layer::NpuLmHead&>()))>> : std::true_type {};

template <typename T, typename = void>
struct has_get_npu_word_embedding : std::false_type {};

template <typename T>
struct has_get_npu_word_embedding<
    T,
    std::void_t<decltype(std::declval<T>()->get_npu_word_embedding())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_set_npu_word_embedding : std::false_type {};

template <typename T>
struct has_set_npu_word_embedding<
    T,
    std::void_t<decltype(std::declval<T>()->set_npu_word_embedding(
        std::declval<layer::NpuWordEmbedding&>()))>> : std::true_type {};
#endif
}  // namespace detail
}  // namespace xllm
