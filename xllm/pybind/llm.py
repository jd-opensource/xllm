import os
import signal
import sys
import time
import uuid
from . import util
from typing import List, Optional, Sequence, Union

from xllm_export import (LLMMaster, Options, RequestOutput,
                         RequestParams)

class LLM:
    def __init__(
        self,
        model: str,
        task: str = "generate",
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
        communication_backend: str = 'lccl',
        rank_tablefile: str = '',
        expert_parallel_degree: int = 0,
        enable_mla: bool = False,
        disable_chunked_prefill: bool = False,
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
        **kwargs,
    ) -> None:
        signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
        signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

        if not os.path.exists(model):
            raise ValueError(f"model {model} not exists")

        options = Options()
        options.model_path = model
        options.task_type = task
        options.devices = devices
        options.draft_model_path = draft_model
        options.draft_devices = draft_devices
        options.backend = "llm"
        options.block_size = block_size
        options.max_cache_size = max_cache_size
        options.max_memory_utilization = max_memory_utilization
        if disable_prefix_cache:
            options.enable_prefix_cache = False
        else:
            options.enable_prefix_cache = True
        options.max_tokens_per_batch = max_tokens_per_batch
        options.max_seqs_per_batch = max_seqs_per_batch
        options.max_tokens_per_chunk_for_prefill = max_tokens_per_chunk_for_prefill
        options.num_speculative_tokens = num_speculative_tokens
        options.num_request_handling_threads = num_request_handling_threads
        options.communication_backend = communication_backend
        options.rank_tablefile = rank_tablefile
        options.expert_parallel_degree = expert_parallel_degree
        options.enable_mla = enable_mla
        if disable_chunked_prefill:
            options.enable_chunked_prefill = False
        else:
            options.enable_chunked_prefill = True
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
        options.enable_schedule_overlap = False
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
        self.master = LLMMaster(options)

    def finish(self):
        try:
            #os.kill(os.getpid(), signal.SIGTERM)
            #os.kill(os.getpid(), signal.SIGKILL)
            util.terminate_process(os.getpid())
        except Exception as e:
            pass

    def generate(
        self,
        prompts: Union[str, List[str]],
        request_params: Optional[Union[RequestParams, List[RequestParams]]] = None,
        wait_schedule_done: bool = True,
    ) -> List[RequestOutput]:
        if request_params is None:
            request_params = RequestParams()
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(request_params, RequestParams):
            request_params = [request_params]

        outputs = [None] * len(prompts)
        def callback(index: int, output: RequestOutput) -> bool:
            outputs[index] = output
            return True

        # schedule all requests
        self.master.handle_batch_request(
            prompts, request_params, callback
        )

        # TODO: add wait later
        if wait_schedule_done:
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
