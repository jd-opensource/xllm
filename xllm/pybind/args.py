import argparse
from argparse import Namespace


def _parse_bool_flag(value: str) -> bool:
    normalized_value = value.strip().lower()
    if normalized_value in ('1', 'true', 't', 'yes', 'y', 'on'):
        return True
    if normalized_value in ('0', 'false', 'f', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError(f'invalid boolean value: {value}')


class ArgumentParser:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--model', type=str, help='"Name or path of the huggingface model to use."')
        self.parser.add_argument('--task', type=str, default="generate", help='The task to use the model for. generate/embed.')
        self.parser.add_argument('--runner', type=str, choices=['pooling'], default=None, help='Optional runner mode for LLM. Currently supports: pooling.')
        self.parser.add_argument('--devices', type=str, default='auto', help='Devices to run the model on, e.g. cpu, cuda:0, cuda:0,cuda:1, or auto to use all available gpus.')
        self.parser.add_argument('--draft_model', type=str, default='', help='draft hf model path to the model file.')
        self.parser.add_argument('--draft_devices', type=str, default='auto', help='Devices to run the draft model on, e.g. cpu, cuda:0, cuda:0,cuda:1, or auto to use all available gpus.')
        self.parser.add_argument('--block_size', type=int, default=128, help='Number of slots per kv cache block. Default is 128.')
        self.parser.add_argument('--max_cache_size', type=int, default=0, help='Max gpu memory size for kv cache. Default is 0, which means cache size is caculated by available memory.')
        self.parser.add_argument('--max_memory_utilization', type=float, default=0.9, help='The fraction of GPU memory to be used for model inference, including model weights and kv cache.')
        self.parser.add_argument('--disable_prefix_cache', action='store_true', help='Disable the prefix cache for the block manager.')
        self.parser.add_argument('--max_tokens_per_batch', type=int, default=20000, help='Max number of tokens per batch.')
        self.parser.add_argument('--max_seqs_per_batch', type=int, default=256, help='Max number of sequences per batch.')
        self.parser.add_argument('--max_tokens_per_chunk_for_prefill', type=int, default=512, help='Max number of tokens per chunk for request in prefill stage.')
        self.parser.add_argument('--num_speculative_tokens', type=int, default=0, help='Number of speculative tokens.')
        self.parser.add_argument('--num_request_handling_threads', type=int, default=4, help='Number of handling threads.')
        self.parser.add_argument('--communication_backend', type=str, default='hccl', help='npu communication backend.')
        self.parser.add_argument('--rank_tablefile', type=str, default='', help='atb hccl rank table file')
        self.parser.add_argument('--expert_parallel_degree', type=int, default=0, help='ep degree')
        self.parser.add_argument('--disable_chunked_prefill', action='store_true', help='Disable chunked prefill.')
        self.parser.add_argument('--enable_prefill_sp', action='store_true', help='Enable prefill-only sequence parallel.')
        self.parser.add_argument('--master_node_addr', type=str, default='', help='The master address for multi-node distributed serving(e.g. 10.18.1.1:9999).')
        self.parser.add_argument('--instance_role', type=str, default='DEFAULT', help='The role of instance(e.g. DEFAULT, PREFILL, DECODE, MIX).')
        self.parser.add_argument('--device_ip', type=str, default='', help='The device ip.')
        self.parser.add_argument('--transfer_listen_port', type=int, default=26000, help='The KVCacheTranfer listen port.')
        self.parser.add_argument('--nnodes', type=int, default=1, help='The number of multi-nodes.')
        self.parser.add_argument('--node_rank', type=int, default=0, help='The node rank.')
        self.parser.add_argument('--dp_size', type=int, default=1, help='Data parallel size for MLA attention.')
        self.parser.add_argument('--ep_size', type=int, default=1, help='Expert parallel size for MoE model.')
        self.parser.add_argument('--instance_name', type=str, default='', help='instance name')
        self.parser.add_argument('--enable_disagg_pd', action='store_true', help='Enable disaggregated prefill and decode execution.')
        self.parser.add_argument('--enable_pd_ooc', action='store_true', help='Enable online-offline co-location in disaggregated prefill-decoding mode.')
        self.parser.add_argument(
            '--enable_schedule_overlap',
            type=_parse_bool_flag,
            nargs='?',
            const=True,
            default=False,
            help='Whether to enable schedule overlap.',
        )
        self.parser.add_argument('--kv_cache_transfer_mode', type=str, default='PUSH', help='The mode of kv cache transfer(e.g. PUSH, PULL).')
        self.parser.add_argument('--enable_multi_stream_parallel', action='store_true', help='Whether to enable computation communication overlap.')
        self.parser.add_argument('--disable_ttft_profiling', action='store_true', help='Whether to disable TTFT profiling.')
        self.parser.add_argument('--enable_forward_interruption', action='store_true', help='Whether to enable forward interruption.')
        self.parser.add_argument(
            '--enable_shm',
            type=_parse_bool_flag,
            nargs='?',
            const=True,
            default=False,
            help='Use shared memory for inter-process communication in the single-machine multi-GPU scenario.',
        )
        self.parser.add_argument(
            '--use_contiguous_input_buffer',
            type=_parse_bool_flag,
            nargs='?',
            const=True,
            default=False,
            help='Whether to use contiguous device input buffer when shared memory is enabled.',
        )
        self.parser.add_argument(
            '--enable_graph',
            type=_parse_bool_flag,
            nargs='?',
            const=True,
            default=False,
            help='Whether to enable graph execution.',
        )
        self.parser.add_argument(
            '--enable_graph_mode_decode_no_padding',
            type=_parse_bool_flag,
            nargs='?',
            const=True,
            default=False,
            help='Whether to enable no-padding graph decode mode.',
        )
        self.parser.add_argument(
            '--enable_prefill_piecewise_graph',
            type=_parse_bool_flag,
            nargs='?',
            const=True,
            default=False,
            help='Whether to enable piecewise graph execution for prefill phase.',
        )
        self.parser.add_argument(
            '--enable_block_copy_kernel',
            type=_parse_bool_flag,
            nargs='?',
            const=True,
            default=False,
            help='Whether to enable block copy kernel on supported backends.',
        )
        self.parser.add_argument(
            '--enable_beam_search_kernel',
            type=_parse_bool_flag,
            nargs='?',
            const=True,
            default=False,
            help='Whether to enable beam search kernel.',
        )
        self.parser.add_argument(
            '--enable_rec_fast_sampler',
            type=_parse_bool_flag,
            nargs='?',
            const=True,
            default=False,
            help='Whether to enable RecSampler fast sampling path.',
        )
        self.parser.add_argument('--input_shm_size', type=int, default=1024, help='The size of input shared memory in MB.')
        self.parser.add_argument('--output_shm_size', type=int, default=128, help='The size of output shared memory in MB.')
        self.parser.add_argument('--kv_cache_dtype', type=str, default='auto', help='KV cache data type. "auto" (default) aligns with model dtype, "int8" enables INT8 quantization (MLU only).')

    def parse_args(self) -> Namespace:
        return self.parser.parse_args()
