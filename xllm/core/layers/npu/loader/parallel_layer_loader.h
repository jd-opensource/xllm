/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <functional>
#include <string>

namespace xllm::npu::model {

// Parallel-loading helper for the NPU manual-loader weight pipeline.
//
// The helper splits decoder weight loading into:
//   - Phase 1: pure CPU work that can run on a thread pool
//              (`prepare_manual_loader_weights` -> `merge_host_at_weights`
//               + `init_weight_slices`).
//   - Phase 2: NPU/host work that must stay on the main thread
//              (`finalize_manual_loader_*` -> `copy_weights_to_*` +
//               view rebuild + `init_layer`).
//
// The orchestration uses a sliding window: as soon as Phase 1 for layer
// `i` is done, the main thread immediately runs Phase 2 for layer `i`,
// while Phase 1 for layer `i+W` (W = parallelism) is dispatched. This
// keeps device-side bookkeeping serialised while overlapping host-side
// heavy work on `W` worker threads.

// Resolve the worker-thread parallelism from FLAGS_weight_load_parallelism.
//   FLAGS_weight_load_parallelism == -1 : serial (returns 0)
//   FLAGS_weight_load_parallelism ==  0 : auto =
//       min(num_cores/4, num_layers/4, 8), clamped to [1, 32].
//   FLAGS_weight_load_parallelism  > 0 : fixed window size (capped by
//                                        num_layers and 32).
// `num_layers` may be 0; in that case the helper still returns 0
// (serial) so callers can fall through to the legacy path.
int resolve_load_parallelism(int num_layers);

// Returns true iff parallel loading is allowed for this run:
//   * `parallelism > 0`, AND
//   * `num_layers > 0`, AND
//   * every layer index i (0..num_layers-1) reports
//     `per_layer_supports(i) == true`.
// If any layer falls back (e.g. no manual loader or eager mode), callers
// should run the entire load serially to keep behaviour consistent.
bool should_run_parallel(int num_layers,
                         int parallelism,
                         const std::function<bool(int)>& per_layer_supports);

// Stage 0 helper used for `load_state_dict`: runs `task(i)` for all
// `i` in [0, num_layers). When `parallelism > 0`, tasks are executed on
// a private ThreadPool with a sliding window; otherwise they run on the
// main thread sequentially. Errors propagate via abort/CHECK from inside
// `task`. Logs aggregated per-layer timings under `tag` for bottleneck
// analysis.
void parallel_run_per_layer(int num_layers,
                            int parallelism,
                            const std::function<void(int)>& task,
                            const std::string& tag);

// Stage 1 + Stage 2 helper used for `merge_loaded_weights` /
// `merge_and_move_pinned_host`:
//   * `phase1(i)` runs the host-side prepare on a worker thread (or the
//     main thread when `parallelism <= 0`).
//   * `phase2(i)` always runs on the main thread, in layer order, only
//     after `phase1(i)` finishes.
// The helper measures and logs per-phase totals, per-layer max/min/avg,
// the time the main thread spent waiting for Phase 1, and effective
// parallelism (=sum of phase1 durations / total wall time).
void parallel_prepare_serial_finalize(int num_layers,
                                      int parallelism,
                                      const std::function<void(int)>& phase1,
                                      const std::function<void(int)>& phase2,
                                      const std::string& tag);

}  // namespace xllm::npu::model
