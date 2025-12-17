### WikiText-2 PPL 评估说明（配合 `eval_ppl_wikitext2.py`）

#### 1. 这个脚本在做什么？

- **目标**：在 WikiText-2 上计算和 MLC 模型对应的 **基座 HF 模型** 的困惑度（PPL），作为精度参考。  
- **方式**：直接用 HuggingFace 的 `Meta-Llama-3-8B` 等模型计算 PPL（可选 4bit/8bit 量化），因为当前 MLC 推理接口不暴露 logits，无法直接在 MLC engine 上算 PPL。

#### 2. 基本用法：未量化基线（fp16）

```bash
cd mlc_llm

python eval_ppl_wikitext2.py \
  --mlc-model HF://mlc-ai/Llama-3-8B-Instruct-q0f16-MLC \
  --device cuda
```

- 会自动映射到 HF 模型：`meta-llama/Meta-Llama-3-8B`  
- 在 WikiText-2 (test) 上计算 PPL，结果可视为 **未量化基线**。

#### 3. 近似量化 PPL（模拟 MLC 量化模型）

安装依赖（在当前 conda 环境内）：

```bash
python -m pip install bitsandbytes "accelerate>=0.26.0"
```

4-bit 量化（近似 `q4f16_1-MLC`）：

```bash
python eval_ppl_wikitext2.py \
  --mlc-model HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC \
  --device cuda \
  --load-in-4bit
```

8-bit 量化（近似 `q8f16_1-MLC`）：

```bash
python eval_ppl_wikitext2.py \
  --mlc-model HF://mlc-ai/Llama-3-8B-Instruct-q8f16_1-MLC \
  --device cuda \
  --load-in-8bit
```

> 说明：这里的量化发生在 HF 侧（bitsandbytes），位宽接近 MLC 的 q4/q8，但实现细节不完全相同，因此是 **高相关的近似 PPL**，而不是对 MLC runtime 的精确 PPL。

#### 4. 模型选择小结

- `--mlc-model`：写你实际在用的 MLC 模型标识（例如上面的 q0f16 / q4f16_1 / q8f16_1）。  
- `--hf-model`：如果自动推断不满足需求，可以手动指定 HF 模型名称，脚本会优先使用这里的值。  
- 默认会选与 MLC 模型对应的 **Base HF 模型**（如 `Meta-Llama-3-8B`），更适合做语言建模 PPL 评估。


