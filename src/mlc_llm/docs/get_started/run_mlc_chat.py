from mlc_llm import MLCEngine

# 指定模型
model = "HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC"

# 创建引擎（自动下载并编译模型）
# engine = MLCEngine(model)
engine = MLCEngine(model, mode="local")

# 运行一次聊天请求（流式输出）
for response in engine.chat.completions.create(
    messages=[{"role": "user", "content": "What is the meaning of life?"}],
    model=model,
    stream=True,
):
    for choice in response.choices:
        print(choice.delta.content, end="", flush=True)

print("\n")
engine.terminate()

