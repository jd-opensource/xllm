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

#include "api_service/call.h"
#include "core/common/options.h"
#include "core/common/types.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/distributed_runtime/vlm_master.h"
#include "core/framework/request/mm_data.h"
#include "core/framework/request/request_output.h"
#include "core/framework/request/request_params.h"

namespace xllm {
namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(xllm_export, m) {
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
      .def_readwrite("enable_mla", &Options::enable_mla_)
      .def_readwrite("enable_chunked_prefill",
                     &Options::enable_chunked_prefill_)
      .def_readwrite("master_node_addr", &Options::master_node_addr_)
      .def_readwrite("nnodes", &Options::nnodes_)
      .def_readwrite("node_rank", &Options::node_rank_)
      .def_readwrite("dp_size", &Options::dp_size_)
      .def_readwrite("ep_size", &Options::ep_size_)
      .def_readwrite("xservice_addr", &Options::xservice_addr_)
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
      .def_readwrite("enable_shm", &Options::enable_shm_)
      .def_readwrite("is_local", &Options::is_local_);

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
      .def("get_cache_info",
           &LLMMaster::get_cache_info,
           py::call_guard<py::gil_scoped_release>())
      .def("link_cluster",
           &LLMMaster::link_cluster,
           py::call_guard<py::gil_scoped_release>())
      .def("unlink_cluster",
           &LLMMaster::unlink_cluster,
           py::call_guard<py::gil_scoped_release>())
      .def("options",
           &LLMMaster::options,
           py::call_guard<py::gil_scoped_release>())
      .def("get_rate_limiter",
           &LLMMaster::get_rate_limiter,
           py::call_guard<py::gil_scoped_release>())
      .def("__repr__", [](const LLMMaster& self) {
        return "LLMMaster({})"_s.format(self.options());
      });

  // 3. export RequestParams
  py::class_<RequestParams>(m, "RequestParams")
      .def(py::init())
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
      .def_readwrite("stop_token_ids", &RequestParams::stop_token_ids);

  // 4. export RequestOutput
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

  // 5. export StatusCode
  py::enum_<StatusCode>(m, "StatusCode")
      .value("OK", StatusCode::OK)
      .value("CANCELLED", StatusCode::CANCELLED)
      .value("UNKNOWN", StatusCode::UNKNOWN)
      .value("INVALID_ARGUMENT", StatusCode::INVALID_ARGUMENT)
      .value("DEADLINE_EXCEEDED", StatusCode::DEADLINE_EXCEEDED)
      .value("RESOURCE_EXHAUSTED", StatusCode::RESOURCE_EXHAUSTED)
      .export_values();

  // 6. export Status
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

  // 7. export SequenceOutput
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

  // 8. export MMType
  py::enum_<MMType::Value>(m, "MMType")
      .value("NONE", MMType::Value::NONE)
      .value("IMAGE", MMType::Value::IMAGE)
      .value("VIDEO", MMType::Value::VIDEO)
      .value("AUDIO", MMType::Value::AUDIO)
      .export_values();

  // 9. export MMData
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

  // 10. export VLMMaster
  py::class_<VLMMaster>(m, "VLMMaster")
      .def(py::init<const Options&>(),
           py::arg("options"),
           py::call_guard<py::gil_scoped_release>())
      .def("handle_batch_request",
           py::overload_cast<const std::vector<std::string>&,
                             const std::vector<MMData>&,
                             const std::vector<RequestParams>&,
                             BatchOutputCallback>(
               &VLMMaster::handle_batch_request),
           py::call_guard<py::gil_scoped_release>())
      .def("generate",
           &VLMMaster::generate,
           py::call_guard<py::gil_scoped_release>())
      .def("__repr__", [](const VLMMaster& self) {
        return "VLMMaster({})"_s.format(self.options());
      });
}

}  // namespace xllm
