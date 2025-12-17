# MLC LLM 快速启动指南

## ✅ 环境配置总结

### 已完成配置
1. **Conda 环境**: `mlc-llm-env` (Python 3.11)
2. **依赖安装**: PyTorch, Transformers, MLC LLM (CUDA 12.1版本)
3. **Git LFS**: 已安装并初始化
4. **仓库克隆**: `/home/liangtaodai/dailt_workplace/mlc_llm/`
5. **CUDA 工具链**: 通过 module load 加载 CUDA 12.4

## 🚀 使用方法

### 每次使用时必须执行的命令

```bash
# 1. 加载 CUDA 模块（重要！）
module load cuda

# 2. 激活 conda 环境
conda activate mlc-llm-env

# 3. 运行 MLC LLM
mlc_llm chat HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC
```

### 快捷方式

您也可以使用我创建的启动脚本：
```bash
cd /home/liangtaodai/dailt_workplace/mlc_llm
chmod +x run_mlc.sh
./run_mlc.sh
```

## 📝 运行示例

### 命令行聊天
```bash
mlc_llm chat HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC
```

聊天界面特殊命令：
- `/help` - 查看帮助
- `/exit` - 退出
- `/reset` - 重置对话
- `/stats` - 显示性能统计
- `/metrics` - 显示引擎指标
- `/set temperature=0.5;top_p=0.8` - 设置生成参数

### Python API 使用

```python
from mlc_llm import MLCEngine

model = "HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC"
engine = MLCEngine(model)

response = engine.chat.completions.create(
    messages=[{"role": "user", "content": "你好，介绍一下你自己"}],
    model=model,
    stream=True
)

for r in response:
    for choice in r.choices:
        print(choice.delta.content, end="", flush=True)
print("\n")

engine.terminate()
```

### REST Server

```bash
# 启动服务器
mlc_llm serve HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC

# 在另一个终端发送请求
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
        "model": "HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC",
        "messages": [{"role": "user", "content": "你好！"}]
  }' \
  http://127.0.0.1:8000/v1/chat/completions
```

## 🔧 常见问题

### 问题1: nvcc 未找到
**解决方法**: 
```bash
module load cuda
```

### 问题2: 模型下载失败
**解决方法**: 首次运行会自动下载模型（约5GB），确保网络连接正常。

### 问题3: GPU 显存不足
**解决方法**: 
- 使用更小的模型
- 减少 `context_window_size`
- 使用量化版本模型

## 📊 性能指标

根据运行日志：
- **CUDA 架构**: sm_89 (RTX 4090)
- **显存使用**: ~8GB
- **模型参数**: 4308 MB
- **KV Cache**: 1234 MB
- **临时缓冲区**: 2626 MB

## 📚 更多资源

- 官方文档: https://llm.mlc.ai/docs/
- 示例代码: `mlc_llm/examples/python/sample_mlc_engine.py`
- 快速开始: `mlc_llm/docs/get_started/quick_start.rst`

## 🎉 恭喜！

现在您可以开始使用 MLC LLM 进行 LLM 推理和部署了！

