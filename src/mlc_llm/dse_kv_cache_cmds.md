### KV Cache 参数 DSE 三个方案（命令合集）

#### 方案 1：显存占用最低（prefill 小，batch 小）

- **目标**：尽量压低显存，适合资源紧张、单机测试
- **参数**：`context_window_size=4096, prefill_chunk_size=1024, max_batch_size=16`

```bash
# 编译
python3.11 -m mlc_llm compile \
  /home/liangtaodai/.cache/mlc_llm/model_weights/hf/mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC \
  --device vulkan:0 \
  --overrides "context_window_size=4096;prefill_chunk_size=1024;max_batch_size=16" \
  --output /tmp/test_dse_small.so

# chat
MLC_JIT_POLICY=OFF mlc_llm chat \
  HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC \
  --model-lib /tmp/test_dse_small.so \
  --device vulkan:0
```

(Parameters: 4308.133 MB. KVCache: 722.068 MB. Temporary buffer: 442.011 MB)

---

#### 方案 2：中等设置（折中）

- **目标**：在显存和速度之间做折中
- **参数**：`context_window_size=4096, prefill_chunk_size=2048, max_batch_size=32`

```bash
# 编译
python3.11 -m mlc_llm compile \
  /home/liangtaodai/.cache/mlc_llm/model_weights/hf/mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC \
  --device vulkan:0 \
  --overrides "context_window_size=4096;prefill_chunk_size=2048;max_batch_size=32" \
  --output /tmp/test_dse_mid.so

# chat
MLC_JIT_POLICY=OFF mlc_llm chat \
  HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC \
  --model-lib /tmp/test_dse_mid.so \
  --device vulkan:0
```

(Parameters: 4308.133 MB. KVCache: 722.068 MB. Temporary buffer: 754.011 MB)

---

#### 方案 3：相对激进（窗口不变，只略降）

- **目标**：保持较大上下文，略微减小 prefill/batch，观察显存与速度
- **参数**：`context_window_size=8192, prefill_chunk_size=4096, max_batch_size=64`

```bash
# 编译
python3.11 -m mlc_llm compile \
  /home/liangtaodai/.cache/mlc_llm/model_weights/hf/mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC \
  --device vulkan:0 \
  --overrides "context_window_size=8192;prefill_chunk_size=4096;max_batch_size=64" \
  --output /tmp/test_dse_large.so

# chat
MLC_JIT_POLICY=OFF mlc_llm chat \
  HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC \
  --model-lib /tmp/test_dse_large.so \
  --device vulkan:0
```


(Parameters: 4308.133 MB. KVCache: 1234.072 MB. Temporary buffer: 1378.011 MB)
---




### 使用 `kv_cache_dse_bench.py` 做自动测速

- **脚本位置**：`kv_cache_dse_bench.py`（当前工作区根目录）
- **作用**：给定 KV 相关参数（`context_window_size / prefill_chunk_size / max_batch_size / sliding_window_size`），自动：
  - 配置引擎；
  - 连续跑多次固定问句；
  - 打印脚本本地吞吐（approx chars/s）；
  - 以及 MLC 引擎内部的官方 metrics（`prefill_tokens_per_s / decode_tokens_per_s` 等）。

#### 基本用法

```bash
python3.11 kv_cache_dse_bench.py \
  --model HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC \
  --device vulkan:0 \
  --context-window-size 4096 \
  --prefill-chunk-size 1024 \
  --max-batch-size 2 \
  --sliding-window-size -1 \
   --attention-sink-size 0 \
  --num-runs 3 \
  --max-tokens 128
```

- **关键指标看哪里**：
  - 本地统计：`approx chars/s`
  - 引擎指标：`prefill_tokens_per_s`、`decode_tokens_per_s`、`last_finished_request_*latency_s` 等
  - 结果落盘：`kv_cache_dse_results.csv`（每个 run 的参数 + time_s + approx_chars + GPU 显存估算），
    `kv_cache_dse_metrics.csv`（每个 run 的全部 Prometheus 指标）

#### 对应上面三种方案的脚本命令（示例）

- **方案 1（小显存）**

```bash
python3.11 kv_cache_dse_bench.py \
  --model HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC \
  --device vulkan:0 \
  --context-window-size 4096 \
  --prefill-chunk-size 1024 \
  --max-batch-size 16 \
  --sliding-window-size -1 \
  --attention-sink-size 0 \
  --num-runs 3 \
  --max-tokens 128
```

- **方案 2（折中）**

```bash
python3.11 kv_cache_dse_bench.py \
  --model HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC \
  --device vulkan:0 \
  --context-window-size 4096 \
  --prefill-chunk-size 2048 \
  --max-batch-size 32 \
  --sliding-window-size -1 \
  --attention-sink-size 0 \
  --num-runs 3 \
  --max-tokens 128
```

- **方案 3（相对激进）**

```bash
python3.11 kv_cache_dse_bench.py \
  --model HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC \
  --device vulkan:0 \
  --context-window-size 8192 \
  --prefill-chunk-size 4096 \
  --max-batch-size 64 \
  --sliding-window-size -1 \
  --attention-sink-size 0 \
  --num-runs 3 \
  --max-tokens 128
```

#### 全参数空间 DSE 枚举用法

```bash
python3.11 kv_cache_dse_bench.py \
  --model HF://mlc-ai/Llama-3-8B-Instruct-q4f32_1-MLC \
  --device vulkan:0 \
  --num-runs 3 \
  --max-tokens 128 \
  --dse

python3.11 kv_cache_dse_bench.py \
  --model HF://mlc-ai/Llama-3.2-1B-Instruct-q0f16-MLC \
  --device vulkan:0 \
  --num-runs 3 \
  --max-tokens 128 \
  --dse

python3.11 kv_cache_dse_bench.py \
  --model HF://mlc-ai/Llama-3.2-1B-Instruct-q0f16-MLC \
  --device vulkan:0 \
  --num-runs 3 \
  --max-tokens 128 \
  --dse
```

- **说明**：
  - 会自动枚举 `PARAM_SPACE` 和 `generate_dse_points()` 中的所有合法组合；
  - 所有点的单次 run 结果会写入 `kv_cache_dse_results.csv`，同时对应的引擎 metrics 写入 `kv_cache_dse_metrics.csv`。
