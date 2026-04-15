/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/python.h>

#include <optional>

#include "api_service/call.h"
#include "core/common/global_flags.h"
#include "core/common/options.h"
#include "core/common/types.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/distributed_runtime/vlm_master.h"
#include "core/framework/request/mm_data.h"
#include "core/framework/request/request_output.h"
#include "core/framework/request/request_params.h"
#include "core/framework/request/sample_slot.h"
#include "models/model_registry.h"

namespace xllm {
namespace py = pybind11;
using namespace pybind11::literals;

namespace {

void configure_runtime_flags(
    const std::optional<bool>& enable_prefix_cache,
    const std::optional<bool>& enable_chunked_prefill,
    const std::optional<bool>& enable_schedule_overlap,
    const std::optional<bool>& enable_beam_search_kernel,
    const std::optional<bool>& enable_rec_fast_sampler,
    const std::optional<bool>& enable_shm,
    const std::optional<bool>& use_contiguous_input_buffer,
    const std::optional<bool>& enable_graph,
    const std::optional<bool>& enable_graph_mode_decode_no_padding,
    const std::optional<bool>& enable_prefill_piecewise_graph,
    const std::optional<bool>& enable_block_copy_kernel) {
  if (enable_prefix_cache.has_value()) {
    FLAGS_enable_prefix_cache = enable_prefix_cache.value();
  }
  if (enable_chunked_prefill.has_value()) {
    FLAGS_enable_chunked_prefill = enable_chunked_prefill.value();
  }
  if (enable_schedule_overlap.has_value()) {
    FLAGS_enable_schedule_overlap = enable_schedule_overlap.value();
  }
  if (enable_beam_search_kernel.has_value()) {
    FLAGS_enable_beam_search_kernel = enable_beam_search_kernel.value();
  }
  if (enable_rec_fast_sampler.has_value()) {
    FLAGS_enable_rec_fast_sampler = enable_rec_fast_sampler.value();
  }
  if (enable_shm.has_value()) {
    FLAGS_enable_shm = enable_shm.value();
  }
  if (use_contiguous_input_buffer.has_value()) {
    FLAGS_use_contiguous_input_buffer = use_contiguous_input_buffer.value();
  }
  if (enable_graph.has_value()) {
    FLAGS_enable_graph = enable_graph.value();
  }
  if (enable_graph_mode_decode_no_padding.has_value()) {
    FLAGS_enable_graph_mode_decode_no_padding =
        enable_graph_mode_decode_no_padding.value();
  }
  if (enable_prefill_piecewise_graph.has_value()) {
    FLAGS_enable_prefill_piecewise_graph =
        enable_prefill_piecewise_graph.value();
  }
  if (enable_block_copy_kernel.has_value()) {
    FLAGS_enable_block_copy_kernel = enable_block_copy_kernel.value();
  }

#if !defined(USE_NPU) && !defined(USE_CUDA)
  FLAGS_enable_block_copy_kernel = false;
#endif
}

}  // namespace

PYBIND11_MODULE(xllm_export, m) {
  m.def("configure_runtime_flags",
        &configure_runtime_flags,
        py::arg("enable_prefix_cache") = py::none(),
        py::arg("enable_chunked_prefill") = py::none(),
        py::arg("enable_schedule_overlap") = py::none(),
        py::arg("enable_beam_search_kernel") = py::none(),
        py::arg("enable_rec_fast_sampler") = py::none(),
        py::arg("enable_shm") = py::none(),
        py::arg("use_contiguous_input_buffer") = py::none(),
        py::arg("enable_graph") = py::none(),
        py::arg("enable_graph_mode_decode_no_padding") = py::none(),
        py::arg("enable_prefill_piecewise_graph") = py::none(),
        py::arg("enable_block_copy_kernel") = py::none());

  // 1. export Options
  py::class_<Options>(m, "Options")
      .def(py::init())
      .def_readwrite("model_path", &Options::model_path_)
      .def_readwrite("devices", &Options::devices_)
      .def_readwrite("draft_model_path", &Options::draft_model_path_)
      .def_readwrite("draft_devices", &Options::draft_devices_)
      .def_readwrite("backend", &Options::backend_)
      .def_readwrite("block_size", &Options::block_size_)
      .def_readwrite("max_cache_size", &Options::max_cache_size_)
      .def_readwrite("max_memory_utilization",
                     &Options::max_memory_utilization_)
      .def_readwrite("enable_prefix_cache", &Options::enable_prefix_cache_)
      .def_readwrite("max_tokens_per_batch", &Options::max_tokens_per_batch_)
      .def_readwrite("max_seqs_per_batch", &Options::max_seqs_per_batch_)
      .def_readwrite("max_tokens_per_chunk_for_prefill",
                     &Options::max_tokens_per_chunk_for_prefill_)
      .def_readwrite("num_speculative_tokens",
                     &Options::num_speculative_tokens_)
      .def_readwrite("num_request_handling_threads",
                     &Options::num_request_handling_threads_)
      .def_readwrite("communication_backend", &Options::communication_backend_)
      .def_readwrite("rank_tablefile", &Options::rank_tablefile_)
      .def_readwrite("expert_parallel_degree",
                     &Options::expert_parallel_degree_)
      .def_readwrite("task_type", &Options::task_type_)
      .def_readwrite("enable_chunked_prefill",
                     &Options::enable_chunked_prefill_)
      .def_readwrite("enable_prefill_sp", &Options::enable_prefill_sp_)
      .def_readwrite("master_node_addr", &Options::master_node_addr_)
      .def_readwrite("nnodes", &Options::nnodes_)
      .def_readwrite("node_rank", &Options::node_rank_)
      .def_readwrite("dp_size", &Options::dp_size_)
      .def_readwrite("ep_size", &Options::ep_size_)
      .def_readwrite("instance_name", &Options::instance_name_)
      .def_readwrite("enable_disagg_pd", &Options::enable_disagg_pd_)
      .def_readwrite("enable_pd_ooc", &Options::enable_pd_ooc_)
      .def_readwrite("enable_schedule_overlap",
                     &Options::enable_schedule_overlap_)
      .def_readwrite("instance_role", &Options::instance_role_)
      .def_readwrite("kv_cache_transfer_mode",
                     &Options::kv_cache_transfer_mode_)
      .def_readwrite("device_ip", &Options::device_ip_)
      .def_readwrite("transfer_listen_port", &Options::transfer_listen_port_)
      .def_readwrite("disable_ttft_profiling",
                     &Options::disable_ttft_profiling_)
      .def_readwrite("enable_forward_interruption",
                     &Options::enable_forward_interruption_)
      .def_readwrite("enable_offline_inference",
                     &Options::enable_offline_inference_)
      .def_readwrite("spawn_worker_path", &Options::spawn_worker_path_)
      .def_readwrite("enable_graph", &Options::enable_graph_)
      .def_readwrite("enable_shm", &Options::enable_shm_)
      .def_readwrite("input_shm_size", &Options::input_shm_size_)
      .def_readwrite("output_shm_size", &Options::output_shm_size_)
      .def_readwrite("is_local", &Options::is_local_)
      .def_readwrite("kv_cache_dtype", &Options::kv_cache_dtype_);

  // 2. export LLMMaster
  py::class_<LLMMaster>(m, "LLMMaster")
      .def(py::init<const Options&>(),
           py::arg("options"),
           py::call_guard<py::gil_scoped_release>())
      .def("handle_request",
           py::overload_cast<std::string,
                             std::optional<std::vector<int>>,
                             RequestParams,
                             std::optional<Call*>,
                             OutputCallback>(&LLMMaster::handle_request),
           py::call_guard<py::gil_scoped_release>())
      .def("handle_request",
           py::overload_cast<std::vector<Message>,
                             std::optional<std::vector<int>>,
                             RequestParams,
                             std::optional<Call*>,
                             OutputCallback>(&LLMMaster::handle_request),
           py::call_guard<py::gil_scoped_release>())
      .def("handle_batch_request",
           py::overload_cast<std::vector<std::string>,
                             std::vector<RequestParams>,
                             BatchOutputCallback>(
               &LLMMaster::handle_batch_request),
           py::call_guard<py::gil_scoped_release>())
      .def("handle_batch_request",
           py::overload_cast<std::vector<std::vector<Message>>,
                             std::vector<RequestParams>,
                             BatchOutputCallback>(
               &LLMMaster::handle_batch_request),
           py::call_guard<py::gil_scoped_release>())
      .def("run", &LLMMaster::run, py::call_guard<py::gil_scoped_release>())
      .def("generate",
           &LLMMaster::generate,
           py::call_guard<py::gil_scoped_release>())
      .def("options",
           &LLMMaster::options,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "build_sample_slots",
          [](const LLMMaster& self,
             const std::string& request_id,
             const std::string& prompt,
             const std::string& literal) {
            std::vector<SampleSlot> sample_slots;
            const bool ok = xllm::build_sample_slots(
                request_id, prompt, literal, self.tokenizer(), &sample_slots);
            return std::make_pair(ok, sample_slots);
          },
          py::arg("request_id"),
          py::arg("prompt"),
          py::arg("literal"),
          py::call_guard<py::gil_scoped_release>())
      .def("get_rate_limiter",
           &LLMMaster::get_rate_limiter,
           py::call_guard<py::gil_scoped_release>())
      .def("__repr__", [](const LLMMaster& self) {
        return "LLMMaster({})"_s.format(self.options());
      });

  // 3. export SampleSlot
  py::class_<SampleSlot>(m, "SampleSlot")
      .def(py::init())
      .def_readwrite("request_id", &SampleSlot::request_id)
      .def_readwrite("sample_id", &SampleSlot::sample_id)
      .def_readwrite("token_position", &SampleSlot::token_position);

  // 4. export RequestParams
  py::class_<RequestParams>(m, "RequestParams")
      .def(py::init())
      .def(py::init([](py::kwargs kwargs) {
        RequestParams params;
        py::object obj = py::cast(params);
        for (const auto& item : kwargs) {
          if (!py::isinstance<py::str>(item.first)) {
            throw py::type_error("Keyword argument name must be a string");
          }
          py::setattr(obj, item.first, item.second);
        }
        return obj.cast<RequestParams>();
      }))
      .def_readwrite("request_id", &RequestParams::request_id)
      .def_readwrite("service_request_id", &RequestParams::service_request_id)
      .def_readwrite("x_request_id", &RequestParams::x_request_id)
      .def_readwrite("x_request_time", &RequestParams::x_request_time)
      .def_readwrite("max_tokens", &RequestParams::max_tokens)
      .def_readwrite("n", &RequestParams::n)
      .def_readwrite("best_of", &RequestParams::best_of)
      .def_readwrite("echo", &RequestParams::echo)
      .def_readwrite("frequency_penalty", &RequestParams::frequency_penalty)
      .def_readwrite("presence_penalty", &RequestParams::presence_penalty)
      .def_readwrite("repetition_penalty", &RequestParams::repetition_penalty)
      .def_readwrite("temperature", &RequestParams::temperature)
      .def_readwrite("top_p", &RequestParams::top_p)
      .def_readwrite("top_k", &RequestParams::top_k)
      .def_readwrite("logprobs", &RequestParams::logprobs)
      .def_readwrite("top_logprobs", &RequestParams::top_logprobs)
      .def_readwrite("skip_special_tokens", &RequestParams::skip_special_tokens)
      .def_readwrite("ignore_eos", &RequestParams::ignore_eos)
      .def_readwrite("is_embeddings", &RequestParams::is_embeddings)
      .def_readwrite("stop", &RequestParams::stop)
      .def_readwrite("stop_token_ids", &RequestParams::stop_token_ids)
      .def_readwrite("beam_width", &RequestParams::beam_width)
      .def_readwrite("add_special_tokens", &RequestParams::add_special_tokens)
      .def_readwrite("is_sample_request", &RequestParams::is_sample_request)
      .def_readwrite("sample_slots", &RequestParams::sample_slots);

  // 4. export Usage
  py::class_<Usage>(m, "Usage")
      .def(py::init())
      .def_readwrite("num_prompt_tokens", &Usage::num_prompt_tokens)
      .def_readwrite("num_generated_tokens", &Usage::num_generated_tokens)
      .def_readwrite("num_total_tokens", &Usage::num_total_tokens)
      .def_property_readonly(
          "prompt_tokens",
          [](const Usage& self) { return self.num_prompt_tokens; })
      .def_property_readonly(
          "completion_tokens",
          [](const Usage& self) { return self.num_generated_tokens; })
      .def_property_readonly("total_tokens", [](const Usage& self) {
        return self.num_total_tokens;
      });
  // 5. export RequestOutput
  py::class_<RequestOutput>(m, "RequestOutput")
      .def(py::init())
      .def_readwrite("request_id", &RequestOutput::request_id)
      .def_readwrite("service_request_id", &RequestOutput::service_request_id)
      .def_readwrite("prompt", &RequestOutput::prompt)
      .def_readwrite("status", &RequestOutput::status)
      .def_readwrite("outputs", &RequestOutput::outputs)
      .def_readwrite("usage", &RequestOutput::usage)
      .def_readwrite("finished", &RequestOutput::finished)
      .def_readwrite("cancelled", &RequestOutput::cancelled);

  // 6. export StatusCode
  py::enum_<StatusCode>(m, "StatusCode")
      .value("OK", StatusCode::OK)
      .value("CANCELLED", StatusCode::CANCELLED)
      .value("UNKNOWN", StatusCode::UNKNOWN)
      .value("INVALID_ARGUMENT", StatusCode::INVALID_ARGUMENT)
      .value("DEADLINE_EXCEEDED", StatusCode::DEADLINE_EXCEEDED)
      .value("RESOURCE_EXHAUSTED", StatusCode::RESOURCE_EXHAUSTED)
      .export_values();

  // 7. export Status
  py::class_<Status>(m, "Status")
      .def(py::init<StatusCode, const std::string&>(),
           py::arg("code"),
           py::arg("message"))
      .def_property_readonly("code", &Status::code)
      .def_property_readonly("message", &Status::message)
      .def_property_readonly("ok", &Status::ok)
      .def("__repr__", [](const Status& self) {
        if (self.message().empty()) {
          return "Status(code={})"_s.format(self.code());
        }
        return "Status(code={}, message={!r})"_s.format(self.code(),
                                                        self.message());
      });

  // 8. export LogProbData
  py::class_<LogProbData>(m, "LogProbData")
      .def(py::init())
      .def_readwrite("token", &LogProbData::token)
      .def_readwrite("token_id", &LogProbData::token_id)
      .def_readwrite("logprob", &LogProbData::logprob)
      .def_readwrite("finished_token", &LogProbData::finished_token)
      .def("__repr__", [](const LogProbData& self) {
        return "LogProbData(token={!r}, token_id={}, logprob={})"_s.format(
            self.token, self.token_id, self.logprob);
      });

  // 9. export LogProb
  py::class_<LogProb, LogProbData>(m, "LogProb")
      .def(py::init())
      .def_readwrite("top_logprobs", &LogProb::top_logprobs)
      .def("__repr__", [](const LogProb& self) {
        return "LogProb(token={!r}, token_id={}, logprob={})"_s.format(
            self.token, self.token_id, self.logprob);
      });

  // 10. export SequenceOutput
  py::class_<SequenceOutput>(m, "SequenceOutput")
      .def(py::init())
      .def_readwrite("index", &SequenceOutput::index)
      .def_readwrite("text", &SequenceOutput::text)
      .def_readwrite("embedding", &SequenceOutput::embedding)
      .def_readwrite("token_ids", &SequenceOutput::token_ids)
      .def_readwrite("finish_reason", &SequenceOutput::finish_reason)
      .def_readwrite("logprobs", &SequenceOutput::logprobs)
      .def_readwrite("embeddings", &SequenceOutput::embeddings)
      .def("__repr__", [](const SequenceOutput& self) {
        return "SequenceOutput({}: {!r})"_s.format(self.index, self.text);
      });

  // 11. export MMType
  py::enum_<MMType::Value>(m, "MMType")
      .value("NONE", MMType::Value::NONE)
      .value("IMAGE", MMType::Value::IMAGE)
      .value("VIDEO", MMType::Value::VIDEO)
      .value("AUDIO", MMType::Value::AUDIO)
      .export_values();

  // 12. export MMData
  py::class_<MMData>(m, "MMData")
      .def(py::init<int, const MMDict&>(), py::arg("ty"), py::arg("data"))
      .def("get",
           [](const MMData& self, const MMKey& key) -> py::object {
             auto value = self.get<torch::Tensor>(key);
             if (value.has_value()) {
               return py::cast(value.value());
             }
             return py::none();
           })
      .def("get_list",
           [](const MMData& self, const MMKey& key) -> py::object {
             auto value = self.get<std::vector<torch::Tensor>>(key);
             if (value.has_value()) {
               return py::cast(value.value());
             }
             return py::none();
           })
      .def("__repr__", [](const MMData& self) {
        std::stringstream ss;
        ss << "MMData(" << self.type() << ": " << self.size() << " items)";
        return ss.str();
      });

  // 13. export VLMMaster
  py::class_<VLMMaster>(m, "VLMMaster")
      .def(py::init<const Options&>(),
           py::arg("options"),
           py::call_guard<py::gil_scoped_release>())
      .def("handle_batch_request",
           py::overload_cast<std::vector<std::string>,
                             std::vector<MMData>,
                             std::vector<RequestParams>,
                             BatchOutputCallback>(
               &VLMMaster::handle_batch_request),
           py::call_guard<py::gil_scoped_release>())
      .def("handle_batch_request_with_image_urls",
           py::overload_cast<std::vector<std::string>,
                             std::vector<std::vector<std::string>>,
                             std::vector<RequestParams>,
                             BatchOutputCallback>(
               &VLMMaster::handle_batch_request_with_image_urls),
           py::call_guard<py::gil_scoped_release>())
      .def("generate",
           &VLMMaster::generate,
           py::call_guard<py::gil_scoped_release>())
      .def("__repr__", [](const VLMMaster& self) {
        return "VLMMaster({})"_s.format(self.options());
      });

  // 12. export helpers
  m.def("get_model_backend",
        &ModelRegistry::get_model_backend,
        py::arg("model_type"));
}

}  // namespace xllm
