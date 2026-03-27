/* Copyright 2026 The xLLM Authors. All Rights Reserved.
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
 * ==============================================================================*/

#include <ATen/cuda/CUDAContext.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <torch/torch.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/common/global_flags.h"
#include "core/kernels/cuda/utils.h"
#include "core/kernels/ops_api.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/cuda/flashinfer_planinfo.h"
#include "core/layers/cuda/flashinfer_workspace.h"
#include "core/platform/device.h"

namespace xllm {
namespace {

class FlashinferAotTestEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    google::InitGoogleLogging("flashinfer_aot_integration_test");
    google::SetStderrLogging(google::INFO);

    if (torch::cuda::is_available()) {
      xllm::Device xllm_device(0);
      xllm_device.set_device();
    }

    // Keep tests small and deterministic.
    FLAGS_block_size = 1;
    FLAGS_max_tokens_per_batch = 16;
    // Give FlashInfer workspace a modest size to avoid OOM.
    if (FLAGS_flashinfer_workspace_buffer_size < 16 * 1024 * 1024) {
      FLAGS_flashinfer_workspace_buffer_size = 16 * 1024 * 1024;
    }

    torch::manual_seed(0);
  }

  void TearDown() override { google::ShutdownGoogleLogging(); }
};

::testing::Environment* const flashinfer_aot_env =
    ::testing::AddGlobalTestEnvironment(new FlashinferAotTestEnvironment);

torch::Device InitCudaDeviceForFlashinferTest(int32_t device_index = 0) {
  xllm::Device xllm_device(device_index);
  return xllm_device.unwrap();
}

bool IsFlashinferAvailable() {
  const char* ops_path = std::getenv("FLASHINFER_OPS_PATH");
  if (ops_path == nullptr || std::string(ops_path).empty()) {
    return false;
  }
  return true;
}

bool IsCudaSm90OrLater(int32_t device_index = 0) {
  auto* prop = at::cuda::getDeviceProperties(device_index);
  if (prop == nullptr) {
    return false;
  }
  return prop->major >= 9;
}

int64_t GetEnvInt64OrDefault(const char* env_name,
                             int64_t default_value,
                             int64_t min_value = 1) {
  const char* raw = std::getenv(env_name);
  if (raw == nullptr || std::string(raw).empty()) {
    return default_value;
  }
  char* end = nullptr;
  errno = 0;
  const long long parsed = std::strtoll(raw, &end, 10);
  if (errno != 0 || end == raw || (end != nullptr && *end != '\0') ||
      parsed < min_value) {
    LOG(WARNING) << "Invalid " << env_name << "=" << raw << ", fallback to "
                 << default_value;
    return default_value;
  }
  return static_cast<int64_t>(parsed);
}

ffi::Tensor ToFfiTensorWithExplicitStrides(const torch::Tensor& torch_tensor) {
  // In newer tvm-ffi versions Tensor::as_strided is removed.
  // kernel::cuda::to_ffi_tensor already wraps torch tensor via DLPack and keeps
  // shape/stride metadata.
  return kernel::cuda::to_ffi_tensor(torch_tensor);
}

// Build a minimal prefill AttentionMetadata that routes to FlashInfer
layer::AttentionMetadata MakeMinimalPrefillMetadata(const torch::Device& device,
                                                    int32_t q_seq_len,
                                                    int32_t kv_seq_len) {
  layer::AttentionMetadata meta;

  auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(device);
  meta.q_seq_lens = torch::tensor({0, q_seq_len}, iopt);
  meta.kv_seq_lens = torch::tensor({0, kv_seq_len}, iopt);
  meta.q_cu_seq_lens = meta.q_seq_lens;
  meta.kv_cu_seq_lens = meta.kv_seq_lens;
  meta.slot_mapping =
      torch::arange(0, kv_seq_len, iopt);  // not used directly by plan
  meta.block_table =
      torch::arange(0, kv_seq_len, iopt).unsqueeze(0);  // 1 x kv_seq_len

  meta.max_query_len = q_seq_len;
  meta.max_seq_len = kv_seq_len;
  meta.compute_dtype = "bfloat16";
  meta.is_prefill = true;
  meta.is_chunked_prefill = false;
  meta.is_dummy = false;
  meta.is_causal = true;

  // FlashInfer-specific tensors for decode; not needed for prefill here.
  meta.paged_kv_indptr = torch::Tensor();
  meta.paged_kv_indices = torch::Tensor();
  meta.paged_kv_last_page_len = torch::Tensor();
  meta.qo_indptr = torch::Tensor();

  // PlanInfo will be allocated by the caller.
#if defined(USE_CUDA) || defined(USE_MUSA)
  meta.plan_info = std::make_shared<layer::PlanInfo>();
#endif

  meta.enable_cuda_graph = false;

  meta.full_k_cache = torch::Tensor();
  meta.full_v_cache = torch::Tensor();
  meta.unshared_k_cache = torch::Tensor();
  meta.unshared_v_cache = torch::Tensor();
  meta.step_tensor = torch::Tensor();
  meta.attn_mask = torch::Tensor();

  return meta;
}

struct PlanTestConfig {
  std::string backend;
};

class FlashinferRealPlanTest : public ::testing::TestWithParam<PlanTestConfig> {
};

TEST_P(FlashinferRealPlanTest, UpdatePlanInfoUsesRealSo) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available at runtime.";
  }
  if (!IsFlashinferAvailable()) {
    GTEST_SKIP()
        << "FLASHINFER_OPS_PATH is not set; skipping real FlashInfer tests.";
  }

  const auto cfg = GetParam();

  const torch::Device device =
      InitCudaDeviceForFlashinferTest(/*device_index=*/0);
  layer::flashinfer::FlashinferWorkspace::get_instance().initialize(device);

  // Minimal prefill configuration: batch=1, q_len=1, kv_len=4.
  auto meta =
      MakeMinimalPrefillMetadata(device, /*q_seq_len=*/1, /*kv_seq_len=*/4);
  ASSERT_NE(meta.plan_info, nullptr);
  meta.plan_info->layer_id = 0;

  const c10::ScalarType q_dtype = c10::kBFloat16;
  const c10::ScalarType kv_dtype = c10::kBFloat16;
  const c10::ScalarType o_dtype = c10::kBFloat16;

  const int32_t head_dim_qk = 64;
  const int32_t head_dim_vo = 64;
  const int32_t num_qo_heads = 1;
  const int32_t num_kv_heads = 1;
  const int32_t block_size = 1;
  const int32_t window_size_left = -1;
  const bool enable_cuda_graph = false;
  const bool causal = true;
  const bool use_tensor_core = false;

  try {
    LOG(INFO) << "Update plan info with backend: " << cfg.backend;
    layer::flashinfer::update_plan_info(meta.plan_info,
                                        cfg.backend,
                                        meta,
                                        q_dtype,
                                        kv_dtype,
                                        o_dtype,
                                        head_dim_qk,
                                        head_dim_vo,
                                        num_qo_heads,
                                        num_kv_heads,
                                        block_size,
                                        window_size_left,
                                        enable_cuda_graph,
                                        causal,
                                        use_tensor_core);
  } catch (const std::exception& e) {
    if (cfg.backend == "fa2" &&
        std::string(e.what()).find(
            "Mismatched number of arguments when calling: `plan") !=
            std::string::npos) {
      GTEST_SKIP() << "FlashInfer fa2 plan signature mismatch on this build: "
                   << e.what();
    }
    throw;
  }

  // uri should be non-empty and correspond to an existing .so file.
  ASSERT_FALSE(meta.plan_info->uri.empty());
  const std::string so_path =
      kernel::cuda::path_to_uri_so_lib(meta.plan_info->uri);
  LOG(INFO) << "FlashInfer plan uri: " << meta.plan_info->uri
            << ", so_path: " << so_path;

  struct stat st{};
  ASSERT_EQ(::stat(so_path.c_str(), &st), 0)
      << "FlashInfer .so file does not exist at: " << so_path;

  ASSERT_TRUE(meta.plan_info->plan_info.defined());
  ASSERT_GT(meta.plan_info->plan_info.size(), 0u);
}

INSTANTIATE_TEST_SUITE_P(Backends,
                         FlashinferRealPlanTest,
                         ::testing::Values(PlanTestConfig{"fa3"}));

struct BatchPrefillTestConfig {
  std::string backend;
};

class FlashinferRealBatchPrefillTest
    : public ::testing::TestWithParam<BatchPrefillTestConfig> {};

TEST_P(FlashinferRealBatchPrefillTest, RaggedRunProducesFiniteOutput) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available at runtime.";
  }
  if (!IsFlashinferAvailable()) {
    GTEST_SKIP()
        << "FLASHINFER_OPS_PATH is not set; skipping real FlashInfer tests.";
  }

  const auto cfg = GetParam();

  const torch::Device device =
      InitCudaDeviceForFlashinferTest(/*device_index=*/0);
  layer::flashinfer::FlashinferWorkspace::get_instance().initialize(device);

  auto meta =
      MakeMinimalPrefillMetadata(device, /*q_seq_len=*/1, /*kv_seq_len=*/4);
  ASSERT_NE(meta.plan_info, nullptr);
  meta.plan_info->layer_id = 0;

  const c10::ScalarType q_dtype = c10::kBFloat16;
  const c10::ScalarType kv_dtype = c10::kBFloat16;
  const c10::ScalarType o_dtype = c10::kBFloat16;

  const int32_t head_dim_qk = 64;
  const int32_t head_dim_vo = 64;
  const int32_t num_qo_heads = 1;
  const int32_t num_kv_heads = 1;
  const int32_t block_size = 1;
  const int32_t window_size_left = -1;
  const bool enable_cuda_graph = false;
  const bool causal = true;
  const bool use_tensor_core = false;

  try {
    LOG(INFO) << "Update plan info with backend: " << cfg.backend;
    layer::flashinfer::update_plan_info(meta.plan_info,
                                        cfg.backend,
                                        meta,
                                        q_dtype,
                                        kv_dtype,
                                        o_dtype,
                                        head_dim_qk,
                                        head_dim_vo,
                                        num_qo_heads,
                                        num_kv_heads,
                                        block_size,
                                        window_size_left,
                                        enable_cuda_graph,
                                        causal,
                                        use_tensor_core);
  } catch (const std::exception& e) {
    if (cfg.backend == "fa2" &&
        std::string(e.what()).find(
            "Mismatched number of arguments when calling: `plan") !=
            std::string::npos) {
      GTEST_SKIP() << "FlashInfer fa2 plan signature mismatch on this build: "
                   << e.what();
    }
    throw;
  }

  ASSERT_FALSE(meta.plan_info->uri.empty());
  ASSERT_TRUE(meta.plan_info->plan_info.defined());
  ASSERT_GT(meta.plan_info->plan_info.size(), 0u);

  auto& ws = layer::flashinfer::FlashinferWorkspace::get_instance();
  torch::Tensor float_workspace = ws.get_float_workspace_buffer();
  torch::Tensor int_workspace = ws.get_int_workspace_buffer();
  torch::Tensor page_locked_int_workspace =
      ws.get_page_locked_int_workspace_buffer();

  auto topt = torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(device);

  const int64_t qo_heads = num_qo_heads;
  const int64_t kv_heads = num_kv_heads;

  // FlashInfer batch prefill 期望的 NHD 布局：形状 [T, H, D]。
  const int64_t q_len = 1;
  const int64_t kv_len = 4;

  torch::Tensor query =
      torch::randn({q_len, qo_heads, head_dim_qk}, topt);  // [T_q, H_q, D_qk]
  torch::Tensor key = torch::randn({kv_len, kv_heads, head_dim_qk},
                                   topt);  // [T_kv, H_kv, D_qk]
  torch::Tensor value = torch::randn({kv_len, kv_heads, head_dim_vo},
                                     topt);  // [T_kv, H_kv, D_vo]

  // Use explicit int32 vectors to avoid Int/Long dtype mismatch in Torch API.
  std::vector<int32_t> q_lens_vec{0, static_cast<int32_t>(q_len)};
  std::vector<int32_t> kv_lens_vec{0, static_cast<int32_t>(kv_len)};
  torch::Tensor q_cu_seq_lens = torch::tensor(q_lens_vec, iopt);  // [0, q_len]
  torch::Tensor kv_cu_seq_lens =
      torch::tensor(kv_lens_vec, iopt);  // [0, kv_len]

  torch::Tensor output =
      torch::zeros({q_len, qo_heads, head_dim_vo}, topt);  // [T_q, H_q, D_vo]
  torch::Tensor output_before = output.clone();

  std::optional<torch::Tensor> output_lse;

  const int64_t window_left = -1;
  const double sm_scale = 1.0 / std::sqrt(static_cast<double>(head_dim_qk));

  // 模拟生产路径：通过 AttentionParams + xllm::kernel::batch_prefill
  // 调用最终的 FlashInfer ragged_run，而不是直接走底层 cuda::batch_prefill。
  kernel::AttentionParams attn_params(meta);
  attn_params.query = query;
  attn_params.key = key;
  attn_params.value = value;
  attn_params.output = output;
  attn_params.output_lse = output_lse;
  attn_params.window_size_left = window_left;
  attn_params.scale = sm_scale;
  attn_params.float_workspace_buffer = float_workspace;
  attn_params.int_workspace_buffer = int_workspace;
  attn_params.page_locked_int_workspace_buffer = page_locked_int_workspace;

  kernel::batch_prefill(attn_params);

  ASSERT_EQ(output.sizes(), output_before.sizes());
  ASSERT_TRUE(torch::isfinite(output).all().item<bool>());
  ASSERT_FALSE(torch::allclose(output, output_before))
      << "Output should be modified by FlashInfer ragged_run";
}

INSTANTIATE_TEST_SUITE_P(Backends,
                         FlashinferRealBatchPrefillTest,
                         ::testing::Values(BatchPrefillTestConfig{"fa3"}));

// ---------------------------------------------------------------------------//
// TileLang TVM FFI integration test: load simple_matmul.so and run once.
// ---------------------------------------------------------------------------//

struct TilelangMatmulConfig {
  std::string so_path;
  std::string meta_path;
};

TilelangMatmulConfig GetTilelangMatmulConfigFromEnvOrDefault() {
  TilelangMatmulConfig cfg;

  const char* so_env = std::getenv("TILELANG_KERNEL_SO_PATH");
  const char* meta_env = std::getenv("TILELANG_KERNEL_META_PATH");

  if (so_env != nullptr && std::string(so_env).size() > 0) {
    cfg.so_path = std::string(so_env);
  } else {
    cfg.so_path =
        "/export/home/wangyifan/tmp/tilelang/test_tilelang/kernels/"
        "simple_matmul.so";
  }

  if (meta_env != nullptr && std::string(meta_env).size() > 0) {
    cfg.meta_path = std::string(meta_env);
  } else {
    cfg.meta_path =
        "/export/home/wangyifan/tmp/tilelang/test_tilelang/kernels/"
        "simple_matmul_meta.json";
  }

  return cfg;
}

// TEST(TilelangMatmulSoTest, LoadAndRunSimpleMatmul) {
//   if (!torch::cuda::is_available()) {
//     GTEST_SKIP() << "CUDA is not available at runtime.";
//   }

//   TilelangMatmulConfig cfg = GetTilelangMatmulConfigFromEnvOrDefault();

//   // 1) 读取 meta JSON，提取 M/N/K 以及 ffi_symbol。
//   std::ifstream meta_ifs(cfg.meta_path);
//   ASSERT_TRUE(meta_ifs.is_open())
//       << "Failed to open TileLang meta json at: " << cfg.meta_path;

//   std::string meta_content((std::istreambuf_iterator<char>(meta_ifs)),
//                            std::istreambuf_iterator<char>());
//   meta_ifs.close();

//   int64_t M = 0;
//   int64_t N = 0;
//   int64_t K = 0;
//   std::string kernel_name = "simple_matmul";
//   std::string global_symbol = "simple_matmul";
//   std::string ffi_symbol = "tvm_ffi_simple_matmul";

//   auto extract_int = [&](const std::string& key) -> int64_t {
//     const std::string pattern = "\"" + key + "\"";
//     auto pos = meta_content.find(pattern);
//     if (pos == std::string::npos) return 0;
//     pos = meta_content.find(":", pos);
//     if (pos == std::string::npos) return 0;
//     ++pos;
//     while (pos < meta_content.size() &&
//            std::isspace(static_cast<unsigned char>(meta_content[pos]))) {
//       ++pos;
//     }
//     int64_t value = 0;
//     while (pos < meta_content.size() &&
//            std::isdigit(static_cast<unsigned char>(meta_content[pos]))) {
//       value = value * 10 + (meta_content[pos] - '0');
//       ++pos;
//     }
//     return value;
//   };

//   auto extract_string = [&](const std::string& key,
//                             const std::string& default_value) -> std::string
//                             {
//     const std::string pattern = "\"" + key + "\"";
//     auto pos = meta_content.find(pattern);
//     if (pos == std::string::npos) return default_value;
//     pos = meta_content.find(":", pos);
//     if (pos == std::string::npos) return default_value;
//     pos = meta_content.find("\"", pos);
//     if (pos == std::string::npos) return default_value;
//     ++pos;
//     std::string result;
//     while (pos < meta_content.size() && meta_content[pos] != '\"') {
//       result.push_back(meta_content[pos]);
//       ++pos;
//     }
//     return result;
//   };

//   M = extract_int("M");
//   N = extract_int("N");
//   K = extract_int("K");
//   kernel_name = extract_string("kernel_name", kernel_name);
//   global_symbol = extract_string("global_symbol", global_symbol);
//   ffi_symbol = extract_string("ffi_symbol", ffi_symbol);

//   ASSERT_GT(M, 0);
//   ASSERT_GT(N, 0);
//   ASSERT_GT(K, 0);
//   ASSERT_FALSE(kernel_name.empty());
//   ASSERT_FALSE(global_symbol.empty());
//   ASSERT_FALSE(ffi_symbol.empty());

//   // 2) 加载 TVM FFI Module，并获取函数。
//   ffi::Module mod = ffi::Module::LoadFromFile(cfg.so_path);
//   std::vector<std::string> candidates;
//   auto push_unique = [&](const std::string& s) {
//     if (s.empty()) return;
//     if (std::find(candidates.begin(), candidates.end(), s) ==
//         candidates.end()) {
//       candidates.push_back(s);
//     }
//   };
//   // TileLang/TVM versions may use different exported packed function names.
//   push_unique(ffi_symbol);
//   push_unique(global_symbol);
//   push_unique(kernel_name);
//   push_unique("__tvm_ffi_main");
//   push_unique("__tvm_main__");
//   push_unique("main");

//   std::optional<ffi::Function> resolved_func;
//   std::string resolved_symbol;
//   for (const auto& sym : candidates) {
//     auto opt = mod->GetFunction(sym);
//     if (opt.defined()) {
//       resolved_func = opt.value();
//       resolved_symbol = sym;
//       break;
//     }
//   }

//   std::string candidates_str;
//   for (size_t i = 0; i < candidates.size(); ++i) {
//     candidates_str += candidates[i];
//     if (i + 1 < candidates.size()) {
//       candidates_str += ", ";
//     }
//   }

//   ASSERT_TRUE(resolved_func.has_value())
//       << "No TVM function found in module: " << cfg.so_path
//       << ". Tried candidates: [" << candidates_str
//       << "]. Check TileLang-exported symbol names in meta JSON/host source.";
//   ffi::Function func = resolved_func.value();
//   LOG(INFO) << "Resolved TVM function symbol: " << resolved_symbol;

//   // 3) 构造输入输出张量，并封装为 ffi::Tensor。
//   const torch::Device device =
//       InitCudaDeviceForFlashinferTest(/*device_index=*/0);

//   auto topt = torch::TensorOptions().dtype(torch::kFloat16).device(device);

//   torch::Tensor A = torch::randn({M, K}, topt);
//   torch::Tensor B = torch::randn({K, N}, topt);
//   torch::Tensor C = torch::zeros({M, N}, topt);
//   torch::Tensor C_before = C.clone();

//   ffi::Tensor A_ffi = kernel::cuda::to_ffi_tensor(A);
//   ffi::Tensor B_ffi = kernel::cuda::to_ffi_tensor(B);
//   ffi::Tensor C_ffi = kernel::cuda::to_ffi_tensor(C);

//   // 4) 绑定 TVM FFI 的流到当前 Torch CUDA 流，并调用 kernel。
//   kernel::cuda::bind_tvmffi_stream_to_current_torch_stream(device);

//   func(A_ffi, B_ffi, C_ffi);

//   torch::cuda::synchronize(device.index());

//   // 5) 结果检查：输出有限且被写入。
//   ASSERT_EQ(C.sizes(), C_before.sizes());
//   ASSERT_TRUE(torch::isfinite(C).all().item<bool>());
//   ASSERT_FALSE(torch::allclose(C, C_before))
//       << "TileLang matmul kernel should modify output tensor C.";

//   // 可选：在较小的子块上和 Torch matmul 对比。
//   const int64_t sub_M = std::min<int64_t>(32, M);
//   const int64_t sub_N = std::min<int64_t>(32, N);
//   const int64_t sub_K = std::min<int64_t>(64, K);

//   torch::Tensor A_sub = A.index({torch::indexing::Slice(0, sub_M),
//                                  torch::indexing::Slice(0, sub_K)})
//                             .to(torch::kFloat32);
//   torch::Tensor B_sub = B.index({torch::indexing::Slice(0, sub_K),
//                                  torch::indexing::Slice(0, sub_N)})
//                             .to(torch::kFloat32);
//   torch::Tensor C_sub_ref = torch::matmul(A_sub, B_sub).to(torch::kFloat16);

//   torch::Tensor C_sub = C.index(
//       {torch::indexing::Slice(0, sub_M), torch::indexing::Slice(0, sub_N)});

//   // 允许一定误差。
//   ASSERT_TRUE(torch::allclose(C_sub,
//                               C_sub_ref,
//                               /*rtol=*/1e-1,
//                               /*atol=*/1e-1))
//       << "TileLang matmul result mismatch against torch::matmul on
//              sub -
//              block.";
// }

// TEST(TilelangAddSoTest, LoadAndRunSimpleAddSmoke) {
//   if (!torch::cuda::is_available()) {
//     GTEST_SKIP() << "CUDA is not available at runtime.";
//   }

//   const char* so_env = std::getenv("TILELANG_KERNEL_SO_PATH");
//   const std::string so_path =
//       (so_env != nullptr && std::string(so_env).size() > 0)
//           ? std::string(so_env)
//           : "/export/home/wangyifan/tmp/tilelang/test_tilelang/kernels/"
//             "simple_matmul.so";

//   // Keep these constants consistent with generate_kernel_add_smoke.py.
//   const int64_t M = 1024;
//   const int64_t N = 2048;
//   const int64_t K = 2048;

//   // 1) 加载 module，固定只取 main 符号。
//   // 最小化链路，便于排障。
//   ffi::Module mod = ffi::Module::LoadFromFile(so_path);
//   auto func_opt = mod->GetFunction("main");
//   ASSERT_TRUE(func_opt.defined())
//       << "Function `main` not found in module: " << so_path;
//   ffi::Function func = func_opt.value();
//   LOG(INFO) << "Resolved TVM function symbol: main";

//   // 2) 构造输入输出张量并调用。
//   const torch::Device device =
//       InitCudaDeviceForFlashinferTest(/*device_index=*/0);
//   auto topt = torch::TensorOptions().dtype(torch::kFloat16).device(device);

//   torch::Tensor A = torch::randn({M, K}, topt);
//   torch::Tensor B = torch::randn({K, N}, topt);
//   torch::Tensor C = torch::zeros({M, N}, topt);

//   ffi::Tensor A_ffi = kernel::cuda::to_ffi_tensor(A);
//   ffi::Tensor B_ffi = kernel::cuda::to_ffi_tensor(B);
//   ffi::Tensor C_ffi = kernel::cuda::to_ffi_tensor(C);

//   kernel::cuda::bind_tvmffi_stream_to_current_torch_stream(device);
//   func(A_ffi, B_ffi, C_ffi);
//   torch::cuda::synchronize(device.index());

//   // 3) Add smoke 校验：C[row,col] = A[row,0] + B[0,col]
//   torch::Tensor A_col0 =
//       A.index({torch::indexing::Slice(),
//       0}).unsqueeze(1).to(torch::kFloat32);
//   torch::Tensor B_row0 =
//       B.index({0,
//       torch::indexing::Slice()}).unsqueeze(0).to(torch::kFloat32);
//   torch::Tensor C_ref = (A_col0 + B_row0).to(torch::kFloat16);

//   ASSERT_TRUE(torch::isfinite(C).all().item<bool>());
//   ASSERT_TRUE(torch::allclose(C,
//                               C_ref,
//                               /*rtol=*/1e-3,
//                               /*atol=*/1e-3))
//       << "TileLang add smoke result mismatch: expected "
//          "C[row,col] = A[row,0] + B[0,col].";
// }

TEST(TilelangMatmulTmaSoTest, LoadAndRunSimpleMatmulTma) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available at runtime.";
  }
  if (!IsCudaSm90OrLater(/*device_index=*/0)) {
    GTEST_SKIP() << "TMA matmul requires sm90+ G PU.";
  }
  auto tma_create_tiled =
      ffi::Function::GetGlobal("__tvm_tensormap_create_tiled");
  ASSERT_TRUE(tma_create_tiled.has_value())
      << "Cannot find __tvm_tensormap_create_tiled. "
      << "Please ensure libtilelang.so is correctly linked and loaded.";

  const char* so_env = std::getenv("TILELANG_TMA_KERNEL_SO_PATH");
  const std::string so_path =
      (so_env != nullptr && std::string(so_env).size() > 0)
          ? std::string(so_env)
          : "/export/home/wangyifan/tmp/tilelang/test_tilelang/kernels/"
            "simple_matmul.so";

  // Keep these constants consistent with the exported simple_matmul.so.
  const int64_t M = 1024;
  const int64_t N = 1024;
  const int64_t K = 512;

  ffi::Module mod = ffi::Module::LoadFromFile(so_path);
  auto func_opt = mod->GetFunction("main");
  ASSERT_TRUE(func_opt.defined())
      << "Function `main` not found in module: " << so_path;
  ffi::Function func = func_opt.value();
  LOG(INFO) << "Resolved TVM function symbol: main";

  const torch::Device device =
      InitCudaDeviceForFlashinferTest(/*device_index=*/0);
  auto topt = torch::TensorOptions().dtype(torch::kFloat16).device(device);

  torch::Tensor A = torch::randn({M, K}, topt);
  torch::Tensor B = torch::randn({K, N}, topt);
  torch::Tensor C = torch::zeros({M, N}, topt);

  ffi::Tensor A_ffi = ToFfiTensorWithExplicitStrides(A);
  ffi::Tensor B_ffi = ToFfiTensorWithExplicitStrides(B);
  ffi::Tensor C_ffi = ToFfiTensorWithExplicitStrides(C);

  kernel::cuda::bind_tvmffi_stream_to_current_torch_stream(device);
  func(A_ffi, B_ffi, C_ffi);
  torch::cuda::synchronize(device.index());

  // Validate on a small sub-block to reduce overhead.
  const int64_t sub_M = std::min<int64_t>(32, M);
  const int64_t sub_N = std::min<int64_t>(32, N);

  // Keep K dimension full, consistent with Python smoke and kernel semantics.
  // We only slice output rows/cols to lower validation overhead.
  torch::Tensor A_sub =
      A.index({torch::indexing::Slice(0, sub_M), torch::indexing::Slice()})
          .to(torch::kFloat32);
  torch::Tensor B_sub =
      B.index({torch::indexing::Slice(), torch::indexing::Slice(0, sub_N)})
          .to(torch::kFloat32);
  torch::Tensor C_sub_ref = torch::matmul(A_sub, B_sub).to(torch::kFloat16);

  torch::Tensor C_sub = C.index(
      {torch::indexing::Slice(0, sub_M), torch::indexing::Slice(0, sub_N)});

  ASSERT_TRUE(torch::allclose(C_sub,
                              C_sub_ref,
                              /*rtol=*/1e-1,
                              /*atol=*/1e-1))
      << "TileLang TMA matmul result mismatch against torch::matmul on "
         "sub-block.";

  const int64_t load_iters =
      GetEnvInt64OrDefault("TILELANG_TMA_BENCH_LOAD_ITERS", 10);
  const int64_t warmup_iters =
      GetEnvInt64OrDefault("TILELANG_TMA_BENCH_WARMUP_ITERS", 5, 0);
  const int64_t exec_iters =
      GetEnvInt64OrDefault("TILELANG_TMA_BENCH_EXEC_ITERS", 50);

  double load_total_ms = 0.0;
  double resolve_total_ms = 0.0;
  for (int64_t i = 0; i < load_iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    ffi::Module bench_mod = ffi::Module::LoadFromFile(so_path);
    const auto t1 = std::chrono::steady_clock::now();
    auto bench_func_opt = bench_mod->GetFunction("main");
    const auto t2 = std::chrono::steady_clock::now();
    ASSERT_TRUE(bench_func_opt.defined())
        << "Function `main` not found in module during load benchmark: "
        << so_path;
    load_total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    resolve_total_ms +=
        std::chrono::duration<double, std::milli>(t2 - t1).count();
  }

  for (int64_t i = 0; i < warmup_iters; ++i) {
    func(A_ffi, B_ffi, C_ffi);
  }
  torch::cuda::synchronize(device.index());

  const auto exec_begin = std::chrono::steady_clock::now();
  for (int64_t i = 0; i < exec_iters; ++i) {
    func(A_ffi, B_ffi, C_ffi);
    torch::cuda::synchronize(device.index());
  }
  const auto exec_end = std::chrono::steady_clock::now();
  const double exec_total_ms =
      std::chrono::duration<double, std::milli>(exec_end - exec_begin).count();
  const double exec_avg_ms = exec_total_ms / static_cast<double>(exec_iters);

  LOG(INFO) << "[TileLang TMA Perf] so=" << so_path
            << ", load_iters=" << load_iters
            << ", avg_load_ms=" << (load_total_ms / load_iters)
            << ", avg_resolve_symbol_ms=" << (resolve_total_ms / load_iters)
            << ", warmup_iters=" << warmup_iters
            << ", exec_iters=" << exec_iters
            << ", total_exec_ms=" << exec_total_ms
            << ", avg_exec_ms=" << exec_avg_ms;
}

}  // namespace
}  // namespace xllm
