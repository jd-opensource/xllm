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

namespace xllm {

#define PROPERTY(T, property)                                                 \
 public:                                                                      \
  [[nodiscard]] const T& property() const& noexcept { return property##_; }   \
  [[nodiscard]] T& property() & noexcept { return property##_; }              \
  [[nodiscard]] T&& property() && noexcept { return std::move(property##_); } \
                                                                              \
  auto property(const T& value) & -> decltype(*this) {                        \
    property##_ = value;                                                      \
    return *this;                                                             \
  }                                                                           \
                                                                              \
  auto property(T&& value) & -> decltype(*this) {                             \
    property##_ = std::move(value);                                           \
    return *this;                                                             \
  }                                                                           \
                                                                              \
  void property(const T& value) && = delete;                                  \
  void property(T&& value) && = delete;                                       \
                                                                              \
  T property##_

#define SAFE_CONCAT(a, b) (a##b)

#ifndef UNUSED_PARAMETER
#define UNUSED_PARAMETER(x) ((void)(x))
#endif

#ifndef DISALLOW_COPY_AND_ASSIGN
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;      \
  void operator=(const TypeName&) = delete
#endif

// Define a macro to simplify adding elements from a vector to a repeated field
#define ADD_VECTOR_TO_PROTO(proto_field, vec) \
  do {                                        \
    proto_field->Reserve(vec.size());         \
    for (const auto& value : vec) {           \
      *proto_field->Add() = value;            \
    }                                         \
  } while (0)

#define CALLBACK_WITH_ERROR_ARGS2(CODE, MSG) callback(Status{CODE, MSG})
#define CALLBACK_WITH_ERROR_ARGS3(CODE, MSG, ID) \
  callback({Status{CODE, MSG}, ID})

#define GET_MACRO(_1, _2, _3, NAME, ...) NAME
#define CALLBACK_WITH_ERROR(...)       \
  GET_MACRO(__VA_ARGS__,               \
            CALLBACK_WITH_ERROR_ARGS3, \
            CALLBACK_WITH_ERROR_ARGS2)(__VA_ARGS__)

#define CHECK_ACL_SUCCESS(expr, msg)             \
  do {                                           \
    auto _ret = (expr);                          \
    if (_ret != ACL_SUCCESS) {                   \
      LOG(FATAL) << "CHECK ACL FAILED: " << msg; \
    }                                            \
  } while (0)

#define NOT_IMPLEMENTED()                            \
  do {                                               \
    LOG(FATAL) << __func__ << " is not implemented"; \
  } while (0)
}  // namespace xllm
