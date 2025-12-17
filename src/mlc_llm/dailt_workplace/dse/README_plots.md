## 概览

这几个脚本都是围绕 KV cache DSE 和 HF PPL 评估生成图表的工具，统一假定当前目录为项目根：

```bash
cd /home/liangtaodai/dailt_workplace/mlc_llm
```

所有图片默认写到 `dailt_workplace/dse/mem_plots/`，CSV 输入在 `dailt_workplace/dse/`。

---

## 1. `plot_kv_cache_memory.py`

- **作用**：从 `kv_cache_dse_results_*.csv` 画 **显存 vs 设计空间参数**。
- **输入 CSV**：`kv_cache_dse_results_<model_tag>.csv`（由 `kv_cache_dse_bench.py` 生成）。
- **主要关注列**：`gpu_mem_kvcache_mb`, `gpu_mem_total_mb` 与 `context_window_size / max_batch_size / sliding_window_size / attention_sink_size`。

**命令示例：**

```bash
python dailt_workplace/dse/plot_kv_cache_memory.py \
  --csv dailt_workplace/dse/kv_cache_dse_results_Llama-3-8B-Instruct-q0f16-MLC.csv \
  --out-dir dailt_workplace/dse/mem_plots
```

**输出图片：**

- `*_mem_kvcache_vs_context.png`  
  - y：`gpu_mem_kvcache_mb`  
  - x：`context_window_size`  
  - `col = max_batch_size`, `hue = sliding_window_size`, `style = attention_sink_size`
- `*_mem_total_vs_context.png`  
  - y：`gpu_mem_total_mb`
- `*_mem_kvcache_vs_sliding.png`  
  - y：`gpu_mem_kvcache_mb`  
  - x：`sliding_window_size`，`col = context_window_size`

---

## 2. `plot_kv_cache_quantization.py`

- **作用**：对比 **不同量化配置** 在同一 design space 下的 **KV 显存 / total 显存**。
- **输入 CSV**：多个 `kv_cache_dse_results_*.csv`（不同 quant 的同一个模型）。
- **固定 design space**：`max_batch_size=1, sliding_window_size<=0, attention_sink_size=0`。

**命令示例：**

```bash
python dailt_workplace/dse/plot_kv_cache_quantization.py \
  --csv \
    dailt_workplace/dse/kv_cache_dse_results_Llama-3-8B-Instruct-q0f16-MLC.csv \
    dailt_workplace/dse/kv_cache_dse_results_Llama-3-8B-Instruct-q3f16_1-MLC.csv \
    dailt_workplace/dse/kv_cache_dse_results_Llama-3-8B-Instruct-q4f16_1-MLC.csv \
    dailt_workplace/dse/kv_cache_dse_results_Llama-3-8B-Instruct-q4f32_1-MLC.csv \
  --out dailt_workplace/dse/mem_plots/kv_cache_quant_compare.png
```

**输出图片：**

- `kv_cache_quant_compare.png`：  
  - y：`gpu_mem_kvcache_mb`  
  - x：`context_window_size`  
  - `hue = model_tag`（不同量化）。
- `kv_cache_quant_compare_total.png`：  
  - y：`gpu_mem_total_mb`，其它同上。

---

## 3. `eval_ppl_wikitext2.py` + `plot_wikitext2_ppl.py`

### 3.1 PPL 评估脚本 `eval_ppl_wikitext2.py`

- **作用**：用 HF 模型在文本数据集上算 PPL，并把结果记到 CSV。
- **支持数据集**：`--dataset wikitext2 | wikitext103`（默认 `wikitext2`）。
- **量化配置**：
  - FP16：不加额外参数。
  - fake 3bit：`--fake-q3bit`（离线 int3 近似）。
  - 4bit：`--load-in-4bit`（bitsandbytes 4bit）。
- **输出 CSV**：
  - `dse/eval_ppl_wikitext2.py` 所在目录下：
    - `wikitext2_ppl_results.csv`
    - `wikitext103_ppl_results.csv`
  - 列：`mlc_model, hf_model, device, max_length, stride, split, load_in_4bit, load_in_8bit, fake_q3bit, ppl`

**8B 示例（wikitext2）：**

```bash
# FP16
python dailt_workplace/dse/eval_ppl_wikitext2.py \
  --hf-model meta-llama/Meta-Llama-3-8B \
  --mlc-model HF://mlc-ai/Llama-3-8B-Instruct-q0f16-MLC \
  --device cuda \
  --dataset wikitext2 \
  --split test

# fake 3bit
python dailt_workplace/dse/eval_ppl_wikitext2.py \
  --hf-model meta-llama/Meta-Llama-3-8B \
  --mlc-model HF://mlc-ai/Llama-3-8B-Instruct-q3f16_1-MLC \
  --device cuda \
  --dataset wikitext2 \
  --split test \
  --fake-q3bit

# 4bit
python dailt_workplace/dse/eval_ppl_wikitext2.py \
  --hf-model meta-llama/Meta-Llama-3-8B \
  --mlc-model HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC \
  --device cuda \
  --dataset wikitext2 \
  --split test \
  --load-in-4bit
```

### 3.2 PPL 画图脚本 `plot_wikitext2_ppl.py`

- **作用**：从 `*_ppl_results.csv` 画 **不同量化方式的 PPL 柱状图**，柱顶带数值。
- **自动识别量化**：
  - `fake_q3bit=1` → `fake_q3`
  - `load_in_4bit=1` → `4bit`
  - `load_in_8bit=1` → `8bit`
  - 其它 → `fp16`

**命令示例：**

```bash
# wikitext-2
python dailt_workplace/dse/plot_wikitext2_ppl.py \
  --csv dailt_workplace/dse/wikitext2_ppl_results.csv \
  --out dailt_workplace/dse/mem_plots/wikitext2_ppl_vs_quant.png

# wikitext-103
python dailt_workplace/dse/plot_wikitext2_ppl.py \
  --csv dailt_workplace/dse/wikitext103_ppl_results.csv \
  --out dailt_workplace/dse/mem_plots/wikitext103_ppl_vs_quant.png
```

**输出图片：**

- `wikitext2_ppl_vs_quant.png`
- `wikitext103_ppl_vs_quant.png`  
  - x：`quant`（`fp16/fake_q3/4bit/...`）  
  - y：`ppl`，柱顶显示具体数值。

---

## 4. `plot_throughput_quantization.py`

- **作用**：基于 `kv_cache_dse_metrics_*.csv`，画 **吞吐（TPS）+ 延迟 vs 各设计参数 / 量化配置**。
- **输入 CSV**：多个 `kv_cache_dse_metrics_<model_tag>.csv`（不同量化、同模型）。
- **内部提取的 metric**：
  - 吞吐：`prefill_tokens_per_s`, `decode_tokens_per_s`
  - 延迟：`last_finished_request_end_to_end_latency_s`, `last_finished_request_ttft_s`

**命令示例：**

```bash
python dailt_workplace/dse/plot_throughput_quantization.py \
  --csv \
    dailt_workplace/dse/kv_cache_dse_metrics_Llama-3-8B-Instruct-q0f16-MLC.csv \
    dailt_workplace/dse/kv_cache_dse_metrics_Llama-3-8B-Instruct-q3f16_1-MLC.csv \
    dailt_workplace/dse/kv_cache_dse_metrics_Llama-3-8B-Instruct-q4f16_1-MLC.csv \
    dailt_workplace/dse/kv_cache_dse_metrics_Llama-3-8B-Instruct-q4f32_1-MLC.csv \
  --out-dir dailt_workplace/dse/mem_plots
```

**主要输出图片（按含义分组）：**

- **吞吐 vs context（固定 batch=1, sliding=-1, sink=0）**
  - `throughput_prefill_vs_context.png`：`prefill_tokens_per_s`
  - `throughput_decode_vs_context.png`：`decode_tokens_per_s`

- **吞吐 vs batch（固定 sliding<=0）**
  - `throughput_prefill_vs_batch.png`
  - `throughput_decode_vs_batch.png`

- **吞吐 vs sliding / sink（只看 sliding>0 的点）**
  - `throughput_prefill_vs_sliding.png`
  - `throughput_prefill_vs_sink.png`
  - `throughput_decode_vs_sliding.png`
  - `throughput_decode_vs_sink.png`

- **延迟 vs context（固定 batch=1, sliding=-1, sink=0）**
  - `latency_e2e_vs_context.png`：`last_finished_request_end_to_end_latency_s`
  - `latency_ttft_vs_context.png`：`last_finished_request_ttft_s`

- **延迟 vs batch（固定 sliding<=0）**
  - `latency_e2e_vs_batch.png`
  - `latency_ttft_vs_batch.png`

- **延迟 vs sliding / sink（只看 sliding>0）**
  - `latency_e2e_vs_sliding.png`
  - `latency_e2e_vs_sink.png`
  - `latency_ttft_vs_sliding.png`
  - `latency_ttft_vs_sink.png`

这些图组合在一起，可以从三个维度看设计空间 + 量化的影响：

- **显存**：KV / total vs context / sliding / sink / quant  
- **吞吐**：prefill / decode tokens/s vs context / batch / sliding / sink / quant  
- **时延**：end-to-end / TTFT vs 上述相同参数。


