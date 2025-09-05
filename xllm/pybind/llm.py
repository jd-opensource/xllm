import os
from typing import List, Optional, Union

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
        max_tokens_per_batch: int = 20000,
        max_seqs_per_batch: int = 256,
        max_tokens_per_chunk_for_prefill: int = 512,
        num_speculative_tokens: int = 0,
        num_handling_threads: int = 4,
        communication_backend: str = 'lccl',
        rank_tablefile: str = '',
        expert_parallel_degree: int = 0,
        enable_mla: bool = False,
        disable_chunked_prefill: bool = False,
        master_node_addr: str = '',
        instance_role: str = 'DEFAULT',
        device_ip: str = '',
        transfer_listen_port: int = 26000,
        nnodes: int = 1,
        node_rank: int = 0,
        dp_size: int = 1,
        ep_size: int = 1,
        xservice_addr: str = '',
        instance_name: str = '',
        enable_disagg_pd: bool = False,
        enable_schedule_overlap: bool = False,
        kv_cache_transfer_mode: str = 'PUSH',
        **kwargs,
    ) -> None:

        if not os.path.exists(model):
            raise ValueError(f"model {model} not exists")

        options = Options()
        options.model_path = model
        options.task_type = task
        options.devices = devices
        options.draft_model_path = draft_model
        options.draft_devices = draft_devices
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
        options.num_handling_threads = num_handling_threads
        options.communication_backend = communication_backend
        options.rank_tablefile = rank_tablefile
        options.expert_parallel_degree = expert_parallel_degree
        options.enable_mla = enable_mla
        if disable_chunked_prefill:
            options.enable_chunked_prefill = False
        else:
            options.enable_chunked_prefill = True
        options.master_node_addr = master_node_addr
        options.instance_role = instance_role
        options.device_ip = device_ip
        options.transfer_listen_port = transfer_listen_port
        options.nnodes = nnodes
        options.node_rank = node_rank
        options.dp_size = dp_size
        options.ep_size = ep_size
        options.xservice_addr = xservice_addr
        options.instance_name = instance_name
        options.enable_disagg_pd = enable_disagg_pd
        options.enable_schedule_overlap = enable_schedule_overlap
        options.kv_cache_transfer_mode = kv_cache_transfer_mode
        self.master = LLMMaster(options)

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

        for index, output in enumerate(outputs):
            if output is None:
                raise RuntimeError("Generate failed, no outputs return.")
            if output.status is not None and not output.status.ok:
                raise ValidationError(output.status.code, output.status.message)
            output.prompt = prompts[index]

        return outputs
