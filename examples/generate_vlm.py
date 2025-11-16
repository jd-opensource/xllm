# python generate_vlm.py --model /path/to/Qwen2.5-VL-7B-Instruct/ --disable_prefix_cache --disable_chunked_prefill --max_seqs_per_batch 4 --devices='npu:0' --enable_shm
# python generate_vlm.py --model /path/to/Qwen2.5-VL-7B-Instruct/ --disable_prefix_cache --disable_chunked_prefill --max_seqs_per_batch 4 --devices='npu:0,npu:1' --enable_shm

import os
import signal
from xllm import ArgumentParser, VLM, RequestParams, MMChatMessage, MMInputData

# Create an VLM.
parser = ArgumentParser()
vlm = VLM(**vars(parser.parse_args()))

# Create a reqeust params, include sampling params
request_params = RequestParams()
request_params.temperature = 0.8
request_params.top_p = 0.95
request_params.max_tokens = 100

# input_data
mm_input_data1 = MMInputData()
mm_input_data1.type = 'text'
mm_input_data1.text = 'Please briefly introduce this picture.'
mm_input_data2 = MMInputData()
mm_input_data2.type = 'image_url'
mm_input_data2.image_url = 'https://img2.baidu.com/it/u=2376489989,3127732063&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=657'
mm_chat_msg = MMChatMessage()
mm_chat_msg.role = 'user'
mm_chat_msg.content = [mm_input_data1, mm_input_data2]

output = vlm.generate(mm_chat_msg, request_params, True)

prompt = output.prompt
generated_text = output.outputs[0].text
print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

vlm.finish()

