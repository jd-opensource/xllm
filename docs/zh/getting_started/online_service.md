# 在线服务
先按照[xllm启动文档](launch_xllm.md)启动xllm服务。下面给出LLM和VLM的客户端调用示例，需要根据实际情况修改其中的参数。

## LLM 客户端调用
### HTTP 调用

chat模式：
```bash
curl http://localhost:9977/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2-7B-Instruct",
    "max_tokens": 10,
    "temperature": 0,
    "stream": true,
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "hello xllm"
      }
    ]
  }'
```

completions模式：
```bash
curl http://127.0.0.1:9977/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen2-7B-Instruct",
    "prompt": "hello xllm",
    "max_tokens": 10,
    "temperature": 0,
    "stream": true
  }'
```

### Python调用
```python
import requests
import json

url = f"http://localhost:9977/v1/chat/completions"
messages = [
    {'role': 'user', 'content': "列出三个国家和他的首都。"}
]

request_data = {
    "model": "Qwen2-7B-Instruct",
    "messages": messages,
    "stream": False, 
    "temperature": 0.6, 
    "max_tokens": 2048, 
}

response = requests.post(url, json=request_data)
if response.status_code != 200:
    print(response.status_code, response.text)
else:
    ans = json.loads(response.text)["choices"]
    print(ans[0]['message'])
```


## VLM 客户端调用
### HTTP API

```python
import base64
import requests

api_url = "http://localhost:12345/v1/chat/completions"
image_url = ""

def encode_image(url: str) -> str:
    with requests.get(url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode("utf-8")

    return result

image_base64 = encode_image(image_url)
payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "介绍下这张图片"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ],
        }
    ],
    "model": "Qwen2.5-VL-7B-Instruct",
    "max_completion_tokens": 128,
}

response = requests.post(
    api_url,
    json=payload,
    headers={"Content-Type": "application/json"}
)
print(response.json())
```


### OpenAI API
```python
from openai import OpenAI
import base64
import requests

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:12345/v1"
image_url = ""

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def encode_image(url: str) -> str:
    with requests.get(url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode("utf-8")

    return result

image_base64 = encode_image(image_url)
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "介绍下这张图片"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ],
        }
    ],
    model="Qwen2.5-VL-7B-Instruct",
    max_completion_tokens=128,
)

result = chat_completion.choices[0].message.content
print("Chat completion output:", result)
```