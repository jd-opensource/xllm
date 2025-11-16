import os
import signal
import time
from . import util
from typing import List, Optional, Union

from xllm_export import (LLMMaster, Options, RequestOutput,
                         RequestParams)

class Embedding:
    def __init__(
        self,
        model: str,
        devices: str = 'auto',
        block_size: int = 128,
        max_cache_size: int = 0,
        max_memory_utilization: float = 0.9,
        disable_prefix_cache: bool = False,
        max_tokens_per_batch: int = 20000,
        max_seqs_per_batch: int = 256,
        max_tokens_per_chunk_for_prefill: int = 512,
        num_request_handling_threads: int = 4,
        communication_backend: str = 'lccl',
        rank_tablefile: str = '',
        expert_parallel_degree: int = 0,
        enable_mla: bool = False,
        disable_chunked_prefill: bool = False,
        instance_role: str = 'DEFAULT',
        nnodes: int = 1,
        node_rank: int = 0,
        dp_size: int = 1,
        ep_size: int = 1,
        enable_shm: bool = False,
        is_local: bool = True,
        **kwargs,
    ) -> None:
        signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
        signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

        if not os.path.exists(model):
            raise ValueError(f"model {model} not exists")

        options = Options()
        options.model_path = model
        options.task_type = "embed"
        options.devices = devices
        options.draft_model_path = None
        options.draft_devices = None
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
        options.nnodes = nnodes
        options.node_rank = node_rank
        options.dp_size = dp_size
        options.ep_size = ep_size
        options.enable_disagg_pd = False
        options.enable_schedule_overlap = False
        options.enable_offline_inference = True
        options.spawn_worker_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        options.enable_shm = enable_shm
        options.is_local = is_local
        self.master = LLMMaster(options)

    def finish(self):
        try:
            #os.kill(os.getpid(), signal.SIGTERM)
            #os.kill(os.getpid(), signal.SIGKILL)
            util.terminate_process(os.getpid())
        except Exception as e:
            pass

    def embedding(
        self,
        inputs: Union[str, List[str]],
        request_params: Optional[Union[RequestParams, List[RequestParams]]] = None,
        wait_schedule_done: bool = True,
    ) -> List[RequestOutput]:
        if request_params is None:
            request_params = RequestParams()
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(request_params, RequestParams):
            request_params.is_embeddings = True
            request_params = [request_params]
        else:
            for i in range(len(request_params)):
                request_params[i].is_embeddings = True

        outputs = [None] * len(inputs)
        def callback(index: int, output: RequestOutput) -> bool:
            outputs[index] = output
            return True

        # schedule all requests
        self.master.handle_batch_request(
            inputs, request_params, callback
        )

        # TODO: add wait later
        if wait_schedule_done:
            pass

        # generate
        self.master.generate()

        # wait async output
        for i in range(len(outputs)):
            while outputs[i] is None:
                time.sleep(0.01)
            if outputs[i].status is not None and not outputs[i].status.ok:
                raise ValidationError(outputs[i].status.code, outputs[i].status.message)
            outputs[i].prompt = inputs[i]

        return outputs
