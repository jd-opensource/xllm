# Single Node Deployment

Start the single-node `xllm` service directly:
```bash linenums="1"
./build/xllm/core/server/xllm \
    --model=/path/to/your/qwen2-7b  \
    --backend=llm \
    --port=9977 \
    --max_memory_utilization 0.90
```

## Client 
### Curl
Chat mode:
```bash linenums="1"
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

Completions mode:
```bash linenums="1"
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

### Python
```python linenums="1"
import requests
import json

url = f"http://localhost:9977/v1/chat/completions"
messages = [
    {'role': 'user', 'content': "List three countries and their capitals."}
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