import requests

# 文本生成
# response = requests.post(
#     "http://localhost:6068/api/generate",
#     json={
#         "model": "qwen3:32b",
#         "prompt": "RESTful API 是什么？",
#         #"messages": [{"role": "user", "content": "你好！"}],
#         "stream": False
#     }
# )
# print(response.json()["response"])

# 对话模式（流式）
response = requests.post(
    "http://localhost:6068/api/chat",
    json={
        "model": "qwen3:32b",
        "messages": [{"role": "user", "content": " 你好！"}],
        "stream": True
    },
    stream=True
)
for line in response.iter_lines():
    if line:
        decoded = line.decode("utf-8")
        print(decoded)  # 输出 JSON 字符串（需解析）