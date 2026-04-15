import json
import os
import signal
import sys
import time
import uuid
from . import util
from typing import Any, Dict, List, Optional, Sequence, Union

import xllm_export
from xllm_export import (LLMMaster, VLMMaster, Options, RequestOutput,
                         RequestParams, configure_runtime_flags)
from .errors import ValidationError
from .params import (
    BeamSearchParams,
    PoolingParams,
    SamplingParams,
    to_request_params,
    to_request_params_list,
)

def _read_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_model_backend(model_path: str) -> str:
    model_index_path = os.path.join(model_path, "model_index.json")
    if os.path.exists(model_index_path):
        data = _read_json(model_index_path)
        if "_diffusers_version" in data:
            return "dit"

    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise ValueError(
            "config.json or model_index.json is required for backend detection"
        )
    data = _read_json(config_path)
    model_type = data.get("model_type") or data.get("model_name")
    if not model_type:
        raise ValueError("config.json must contain model_type or model_name")

    get_backend = getattr(xllm_export, "get_model_backend", None)
    if not callable(get_backend):
        raise ValueError(
            "xllm_export.get_model_backend is not available. "
            "Please rebuild xllm_export or explicitly specify backend."
        )
    try:
        backend = get_backend(model_type)
    except Exception as exc:
        raise ValueError(f"Failed to resolve backend for model_type: {model_type}") from exc
    if not backend:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return backend


class BeamSearchOutput:
    def __init__(self, output: RequestOutput):
        self.prompt = output.prompt
        self.sequences = output.outputs
        self.status = output.status
        self.usage = output.usage
        self.request_output = output


class EmbeddingOutputs:
    def __init__(self, output: RequestOutput):
        embedding = []
        if output.outputs and len(output.outputs) > 0:
            embedding = output.outputs[0].embeddings
        self.embedding = embedding
        self.embeddings = embedding


class EmbeddingOutput:
    def __init__(self, output: RequestOutput):
        self.prompt = output.prompt
        self.outputs = EmbeddingOutputs(output)
        self.status = output.status
        self.usage = output.usage
        self.request_output = output


class LLM:
    @staticmethod
    def _is_vllm_style_inputs(prompts: object) -> bool:
        if isinstance(prompts, dict):
            return True
        if isinstance(prompts, list) and prompts and all(isinstance(x, dict) for x in prompts):
            return True
        return False

    def __init__(
        self,
        model: str,
        task: str = "generate",
        runner: Optional[str] = None,
        devices: str = 'auto',
        draft_model: Optional[str] = None,
        draft_devices: Optional[str] = None,
        block_size: int = 128,
        max_cache_size: int = 0,
        max_memory_utilization: float = 0.9,
        disable_prefix_cache: bool = False,
        max_tokens_per_batch: int = 20480,
        max_seqs_per_batch: int = 256,
        max_tokens_per_chunk_for_prefill: int = -1,
        num_speculative_tokens: int = 0,
        num_request_handling_threads: int = 4,
        communication_backend: str = 'hccl',
        rank_tablefile: str = '',
        expert_parallel_degree: int = 0,
        disable_chunked_prefill: bool = False,
        enable_prefill_sp: bool = False,
        instance_role: str = 'DEFAULT',
        device_ip: str = '',
        transfer_listen_port: int = 26000,
        nnodes: int = 1,
        node_rank: int = 0,
        dp_size: int = 1,
        ep_size: int = 1,
        instance_name: str = '',
        enable_disagg_pd: bool = False,
        enable_pd_ooc: bool = False,
        enable_schedule_overlap: bool = False,
        kv_cache_transfer_mode: str = 'PUSH',
        disable_ttft_profiling: bool = False,
        enable_forward_interruption: bool = False,
        enable_shm: bool = False,
        is_local: bool = True,
        input_shm_size: int = 1024,
        output_shm_size: int = 128,
        enable_beam_search_kernel: bool = False,
        enable_rec_fast_sampler: bool = False,
        use_contiguous_input_buffer: bool = False,
        enable_graph: bool = False,
        enable_graph_mode_decode_no_padding: bool = False,
        enable_prefill_piecewise_graph: bool = False,
        enable_block_copy_kernel: bool = False,
        **kwargs: Any,
    ) -> None:
        signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
        signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

        if runner is not None:
            if runner != "pooling":
                raise ValueError(f"unsupported runner: {runner}")
            task = "embed"

        if not os.path.exists(model):
            raise ValueError(f"model {model} not exists")

        backend = _infer_model_backend(model)
        if backend == "dit":
            raise ValueError("LLM does not support DiT backend models")
        if backend == "vlm" and task != "generate":
            raise ValueError("VLM backend only supports generate task in LLM")

        resolved_enable_prefix_cache = not disable_prefix_cache
        resolved_enable_chunked_prefill = not disable_chunked_prefill
        configure_runtime_flags(
            enable_prefix_cache=resolved_enable_prefix_cache,
            enable_chunked_prefill=resolved_enable_chunked_prefill,
            enable_schedule_overlap=enable_schedule_overlap,
            enable_beam_search_kernel=enable_beam_search_kernel,
            enable_rec_fast_sampler=enable_rec_fast_sampler,
            enable_shm=enable_shm,
            use_contiguous_input_buffer=use_contiguous_input_buffer,
            enable_graph=enable_graph,
            enable_graph_mode_decode_no_padding=enable_graph_mode_decode_no_padding,
            enable_prefill_piecewise_graph=enable_prefill_piecewise_graph,
            enable_block_copy_kernel=enable_block_copy_kernel,
        )

        options = Options()
        options.model_path = model
        options.task_type = task
        options.devices = devices
        options.draft_model_path = draft_model
        options.draft_devices = draft_devices
        options.backend = backend
        options.block_size = block_size
        options.max_cache_size = max_cache_size
        options.max_memory_utilization = max_memory_utilization
        options.enable_prefix_cache = resolved_enable_prefix_cache
        options.max_tokens_per_batch = max_tokens_per_batch
        options.max_seqs_per_batch = max_seqs_per_batch
        options.max_tokens_per_chunk_for_prefill = max_tokens_per_chunk_for_prefill
        options.num_speculative_tokens = num_speculative_tokens
        options.num_request_handling_threads = num_request_handling_threads
        options.communication_backend = communication_backend
        options.rank_tablefile = rank_tablefile
        options.expert_parallel_degree = expert_parallel_degree
        options.enable_chunked_prefill = resolved_enable_chunked_prefill
        options.enable_prefill_sp = enable_prefill_sp
        free_port = util.get_free_port()
        options.master_node_addr = "127.0.0.1:" + str(free_port)
        options.device_ip = device_ip
        options.transfer_listen_port = transfer_listen_port
        options.nnodes = nnodes
        options.node_rank = node_rank
        options.dp_size = dp_size
        options.ep_size = ep_size
        options.instance_name = instance_name
        options.enable_disagg_pd = enable_disagg_pd
        options.enable_schedule_overlap = enable_schedule_overlap
        options.enable_pd_ooc = enable_pd_ooc
        options.kv_cache_transfer_mode = kv_cache_transfer_mode
        options.disable_ttft_profiling = disable_ttft_profiling
        options.enable_forward_interruption = enable_forward_interruption
        options.enable_offline_inference = True
        options.spawn_worker_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        options.enable_shm = enable_shm
        options.is_local = is_local
        options.input_shm_size = input_shm_size
        options.output_shm_size = output_shm_size
        options.enable_graph = enable_graph
        self._backend = backend
        if backend == "vlm":
            self.master = VLMMaster(options)
        else:
            self.master = LLMMaster(options)

    def finish(self) -> None:
        try:
            #os.kill(os.getpid(), signal.SIGTERM)
            #os.kill(os.getpid(), signal.SIGKILL)
            util.terminate_process(os.getpid())
        except Exception as e:
            pass

    def generate(
        self,
        prompts: Union[
            str,
            List[str],
            Dict[str, object],
            List[Dict[str, object]],
        ],
        sampling_params: Optional[Union[
            SamplingParams,
            List[SamplingParams],
        ]] = None,
        wait_for_schedule: bool = True,
        **kwargs: Any,
    ) -> List[RequestOutput]:
        request_params = kwargs.pop("request_params", None)
        if kwargs:
            unknown = ", ".join(kwargs.keys())
            raise TypeError(f"Unexpected keyword arguments: {unknown}")
        if request_params is None:
            request_params = sampling_params
        elif sampling_params is not None:
            raise ValueError("sampling_params and request_params cannot both be set")

        mm_datas = None
        image_urls = None
        if self._is_vllm_style_inputs(prompts):
            from . import mm_utils
            prompts, mm_datas, image_urls = mm_utils.normalize_vllm_style_inputs(prompts)
        else:
            if isinstance(prompts, str):
                prompts = [prompts]
            if not isinstance(prompts, list) or not all(isinstance(x, str) for x in prompts):
                raise TypeError("prompts must be str/list[str] or vLLM-style dicts")

        request_params_list = to_request_params_list(
            request_params, default_cls=SamplingParams)
        if len(request_params_list) not in (1, len(prompts)):
            raise ValueError(
                "The number of request_params must be 1 or equal to the "
                "number of prompts."
            )

        outputs = [None] * len(prompts)
        def callback(index: int, output: RequestOutput) -> bool:
            outputs[index] = output
            return True

        # schedule all requests
        if self._backend == "vlm":
            if mm_datas is not None:
                self.master.handle_batch_request(
                    prompts, mm_datas, request_params_list, callback
                )
            else:
                if image_urls is None:
                    image_urls = [[] for _ in prompts]
                self.master.handle_batch_request_with_image_urls(
                    prompts, image_urls, request_params_list, callback
                )
        else:
            has_images = image_urls is not None and any(image_urls)
            if mm_datas is not None or has_images:
                raise ValueError("multi_modal_data is only supported for VLM models")
            self.master.handle_batch_request(
                prompts, request_params_list, callback
            )

        # TODO: add wait later
        if wait_for_schedule:
            pass

        # generate
        self.master.generate()

        count = len(prompts)
        idx = 0
        while idx < count:
            # wait async output
            if outputs[idx] is None:
                continue
            if outputs[idx].status is not None and not outputs[idx].status.ok:
                raise ValidationError(outputs[idx].status.code, outputs[idx].status.message)
            outputs[idx].prompt = prompts[idx]
            idx += 1

        return outputs

    def beam_search(
        self,
        prompts: Union[str, Dict[str, str], List[Union[str, Dict[str, str]]]],
        params: Optional[Union[RequestParams, BeamSearchParams]] = None,
        wait_for_schedule: bool = True,
    ) -> List[BeamSearchOutput]:
        if isinstance(prompts, (str, dict)):
            prompts = [prompts]

        parsed_prompts: List[str] = []
        for prompt in prompts:
            if isinstance(prompt, str):
                parsed_prompts.append(prompt)
                continue
            if isinstance(prompt, dict):
                if "prompt" not in prompt:
                    raise ValueError("beam_search prompt dict must contain key 'prompt'")
                parsed_prompts.append(prompt["prompt"])
                continue
            raise TypeError("prompts must be str or dict with key 'prompt'")

        params = to_request_params(params, default_cls=BeamSearchParams)
        if params.beam_width <= 0:
            raise ValueError("beam_width must be greater than 0")
        else:
            # Beam search relies on top-k logprob candidates from sampler.
            # Keep this aligned with vLLM's internal default behavior.
            params.logprobs = True
            if params.top_logprobs == 0:
                # if not set top_logprobs, default to returning 2x candidates for better deduplication
                params.top_logprobs = 2 * params.beam_width

        outputs = self.generate(parsed_prompts,
                                request_params=params,
                                wait_for_schedule=wait_for_schedule)
        return [BeamSearchOutput(output) for output in outputs]

    def embed(
        self,
        prompts: Union[str, List[str]],
        pooling_params: Optional[Union[
            RequestParams,
            PoolingParams,
            List[Union[RequestParams, PoolingParams]],
        ]] = None,
        wait_for_schedule: bool = True,
    ) -> List[EmbeddingOutput]:
        request_params_list = to_request_params_list(
            pooling_params, default_cls=PoolingParams)
        for params in request_params_list:
            params.is_embeddings = True

        use_params: Union[RequestParams, List[RequestParams]]
        if len(request_params_list) == 1:
            use_params = request_params_list[0]
        else:
            use_params = request_params_list

        outputs = self.generate(prompts,
                                request_params=use_params,
                                wait_for_schedule=wait_for_schedule)
        return [EmbeddingOutput(output) for output in outputs]

    @staticmethod
    def _normalize_selector_values(
        prompts: Sequence[str],
        selector: Union[str, dict, Sequence[Union[str, dict]]],
    ) -> List[str]:
        def get_literal(value: Union[str, dict]) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                selector_type = value.get("type", "literal")
                literal = value.get("value", "")
                if selector_type != "literal":
                    raise ValueError("selector.type must be literal")
                if not isinstance(literal, str) or not literal:
                    raise ValueError("selector.value is required")
                return literal
            raise ValueError("selector must be a string or dict")

        if isinstance(selector, (str, dict)):
            literal = get_literal(selector)
            return [literal for _ in prompts]

        selector_values = list(selector)
        if len(selector_values) != len(prompts):
            raise ValueError("selector count must match prompts count")
        return [get_literal(item) for item in selector_values]

    @staticmethod
    def _build_request_params_list(
        prompts: Sequence[str],
        request_params: Optional[Union[RequestParams, Sequence[RequestParams]]],
    ) -> List[RequestParams]:
        if request_params is None:
            return [RequestParams() for _ in prompts]
        if isinstance(request_params, RequestParams):
            if len(prompts) != 1:
                raise ValueError(
                    "request_params must be a list when prompts has multiple items"
                )
            return [request_params]

        params_list = list(request_params)
        if len(params_list) != len(prompts):
            raise ValueError("request_params count must match prompts count")
        return params_list

    def sample(
        self,
        prompts: Union[str, List[str]],
        selector: Union[str, dict, Sequence[Union[str, dict]]],
        request_params: Optional[Union[RequestParams, Sequence[RequestParams]]] = None,
        logprobs: int = 5,
        wait_schedule_done: bool = True,
    ) -> List[RequestOutput]:
        if isinstance(prompts, str):
            prompts = [prompts]
        if not prompts:
            return []

        selector_values = self._normalize_selector_values(prompts, selector)
        params_list = self._build_request_params_list(prompts, request_params)
        if len(params_list) > 1:
            # sample() 会原地修改每个 RequestParams（如 request_id/sample_slots）。
            # 若复用同一个对象，会在并发批处理时互相覆盖。
            unique_param_objects = {id(p) for p in params_list}
            if len(unique_param_objects) != len(params_list):
                raise ValueError(
                    "request_params contains duplicated RequestParams objects. "
                    "Please create one RequestParams instance per prompt."
                )

        for i, prompt in enumerate(prompts):
            params = params_list[i]
            if not params.request_id:
                params.request_id = "sample-" + uuid.uuid4().hex

            params.max_tokens = 1
            params.n = 1
            params.best_of = 1
            params.logprobs = True
            params.top_logprobs = logprobs
            params.add_special_tokens = True
            params.is_sample_request = True

            ok, sample_slots = self.master.build_sample_slots(
                params.request_id,
                prompt,
                selector_values[i],
            )
            if not ok:
                raise ValueError(
                    "Failed to build sample slots. "
                    "selector.value must be a stable single special token."
                )
            params.sample_slots = sample_slots

        outputs = [None] * len(prompts)

        def callback(index: int, output: RequestOutput) -> bool:
            outputs[index] = output
            return True

        self.master.handle_batch_request(prompts, params_list, callback)

        if wait_schedule_done:
            pass

        self.master.generate()

        for i in range(len(outputs)):
            while outputs[i] is None:
                time.sleep(0.01)
            if outputs[i].status is not None and not outputs[i].status.ok:
                raise RuntimeError(
                    f"sample request failed: {outputs[i].status.message}"
                )
            outputs[i].prompt = prompts[i]

        return outputs
