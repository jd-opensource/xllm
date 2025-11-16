import os
import signal
import time
from . import util
from typing import List, Optional, Union

from xllm_export import (VLMMaster, Options, RequestOutput,
                         RequestParams, MMInputData, MMChatMessage)

class VLM:
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
        xservice_addr: str = '',
        instance_name: str = '',
        enable_disagg_pd: bool = False,
        enable_schedule_overlap: bool = False,
        kv_cache_transfer_mode: str = 'PUSH',
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
        options.task_type = task
        options.devices = devices
        options.draft_model_path = draft_model
        options.draft_devices = draft_devices
        options.backend ="vlm"
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
        options.xservice_addr = xservice_addr
        options.instance_name = instance_name
        options.enable_disagg_pd = enable_disagg_pd
        options.enable_schedule_overlap = False
        options.kv_cache_transfer_mode = kv_cache_transfer_mode
        options.enable_offline_inference = True
        options.spawn_worker_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        options.enable_shm = enable_shm
        options.is_local = is_local
        self.master = VLMMaster(options)

    def finish(self):
        try:
            #os.kill(os.getpid(), signal.SIGTERM)
            #os.kill(os.getpid(), signal.SIGKILL)
            util.terminate_process(os.getpid())
        except Exception as e:
            pass

    # NOTE: support one request currently
    #
    def generate(
        self,
        input_data: Union[MMChatMessage, List[MMChatMessage]],
        request_params: RequestParams = None,
        wait_schedule_done: bool = True,
    ) -> List[RequestOutput]:
        if request_params is None:
            request_params = RequestParams()
        if isinstance(input_data, MMChatMessage):
            input_data = [input_data]

        outputs = [None]
        def callback(output: RequestOutput) -> bool:
            outputs[0] = output
            return True

        # schedule all requests
        self.master.handle_request(
            input_data, request_params, callback
        )

        # TODO: add wait later
        if wait_schedule_done:
            pass

        # generate
        self.master.generate()

        # wait async output
        while outputs[0] is None:
            time.sleep(0.01)
        if outputs[0].status is not None and not outputs[0].status.ok:
            raise ValidationError(outputs[0].status.code, outputs[0].status.message)

        return outputs[0]
