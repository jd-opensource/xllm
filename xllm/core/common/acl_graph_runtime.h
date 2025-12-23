/* Copyright 2025 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *==============================================================================*/

#pragma once

namespace xllm {

namespace detail {
// Thread-local flag indicating whether current thread is inside ACL graph
// capture region.
inline thread_local bool g_in_acl_graph_capture = false;
}  // namespace detail

// Return true if current thread is inside ACL graph capture.
inline bool in_acl_graph_capture() { return detail::g_in_acl_graph_capture; }

// RAII guard that marks current thread as inside ACL graph capture for its
// lifetime.
class AclGraphCaptureGuard {
 public:
  AclGraphCaptureGuard() { detail::g_in_acl_graph_capture = true; }
  ~AclGraphCaptureGuard() { detail::g_in_acl_graph_capture = false; }
};

}  // namespace xllm
