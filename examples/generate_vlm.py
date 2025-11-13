# python generate_vlm.py --model /path/to/Qwen2.5-VL-7B-Instruct/ --disable_prefix_cache --disable_chunked_prefill --max_seqs_per_batch 4

import os
import signal

from xllm import ArgumentParser, VLM, RequestParams, MMType, MMData

from PIL import Image
from transformers import AutoImageProcessor

# Create an VLM.
parser = ArgumentParser()
args = parser.parse_args()

vlm = VLM(**vars(args))
processor = AutoImageProcessor.from_pretrained(args.model, trust_remote_code=True)

questions = ["简单介绍下图片"]
prompts = [
    (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    for question in questions
]

paths = ["00307664d4ce393b.png"]
images = []
for path in paths:
    images.append(Image.open(path).convert("RGB"))

multi_modal_datas = []
for idx in range(len(images)):
    print(f"Processing image: {paths[idx]}")
    image = images[idx]

    data = processor.preprocess([image], return_tensors="pt").data
    mm_data = {
        "pixel_values": data['pixel_values'],
        "image_grid_thw": data['image_grid_thw'],            
    }
    multi_modal_datas.append(MMData(MMType.IMAGE, mm_data))


# Create a reqeust params, include sampling params
request_params = RequestParams()
request_params.temperature = 0
request_params.max_tokens = 1024

outputs = vlm.generate(prompts, multi_modal_datas, request_params, True)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

vlm.finish()


