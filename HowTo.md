## MLC-LLM Usage and Project Workspace Guide (`dailt_workplace/`)

This document explains **how to use the MLC-LLM framework in this project**, and how to run the scripts you created under `mlc_llm/dailt_workplace` (KV cache design space exploration, quantization comparison, PPL evaluation, and visualization).

All paths below are relative to the **course repository root**:

```bash
cd /home/liangtaodai/dailt_workplace/ai-hardware-project-proposal-eat-sleep-do-ai
```

---

## 1. Code and Directory Layout

- **MLC-LLM source root**:  
  `src/mlc_llm/`
- **Project-specific workspace (your scripts)**:  
  `src/mlc_llm/dailt_workplace/`
  - `dse/`: KV cache design space exploration, performance evaluation, and plotting scripts  
  - `test/`: simple test scripts (e.g., `tvm_show.py`)

Enter the MLC-LLM root:

```bash
cd src/mlc_llm
```

---

## 2. Environment Setup

### 2.1 Use the provided script (recommended)

This repo provides a simple startup script `run_mlc.sh`, which will:

- Load CUDA 12.4 module (for the cluster environment)
- Activate the `conda` environment `mlc-llm-env`
- Start a default MLC-LLM chat session

Usage (from `src/mlc_llm`):

```bash
cd src/mlc_llm
bash run_mlc.sh
```

If you are on a local machine (non-cluster) and do not have `module load`, you can comment out or remove the module-loading lines at the top of the script.

### 2.2 Manual environment setup (if you want to modify/replicate)

1. **Create a Conda environment (example)**:

   ```bash
   conda create -n mlc-llm-env python=3.11 -y
   conda activate mlc-llm-env
   ```

2. **Install MLC-LLM Python package from source (editable)**:

   ```bash
   cd /home/liangtaodai/dailt_workplace/ai-hardware-project-proposal-eat-sleep-do-ai/src/mlc_llm
   pip install -e .
   ```

3. Make sure the `mlc_llm` CLI is available:

   ```bash
   mlc_llm --help
   ```

---

## 3. Basic Usage: Compilation + Chat Inference

MLC-LLM provides both Python APIs and a CLI. Here we focus on the **CLI**.

### 3.1 Direct chat (auto-download weights)

From `src/mlc_llm`:

```bash
mlc_llm chat HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC
```

On the first run, it will automatically download the corresponding MLC weights from Hugging Face/MLC (cached under `~/.cache/mlc_llm/model_weights` by default).

### 3.2 Explicitly compile to `.so`, then chat

Example (adapted from `dse/dse_kv_cache_cmds.md`):

```bash
python3.11 -m mlc_llm compile \
  /home/liangtaodai/.cache/mlc_llm/model_weights/hf/mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC \
  --device vulkan:0 \
  --overrides "context_window_size=4096;prefill_chunk_size=1024;max_batch_size=16" \
  --output /tmp/test_dse_small.so

MLC_JIT_POLICY=OFF mlc_llm chat \
  HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC \
  --model-lib /tmp/test_dse_small.so \
  --device vulkan:0
```

This highlights two key steps in the framework:

- **`python -m mlc_llm compile`**: compile the pre-quantized MLC model into a hardware-specific `.so` (e.g., for `vulkan:0` or `cuda:0`), using the TVM compilation pipeline registered in `python/mlc_llm/compiler_pass/pipeline.py`.
- **`mlc_llm chat`**: load the `.so` and corresponding weights to start the inference engine (OpenAI-style chat interface).

---

## 4. KV Cache Design Space Exploration and Benchmarking (`dailt_workplace/dse/`)

### 4.1 Three representative KV configurations (command collection)

File: `src/mlc_llm/dailt_workplace/dse/dse_kv_cache_cmds.md`  
It contains three typical design points:

- **Config 1**: minimum VRAM usage (small `prefill_chunk_size`, small `max_batch_size`)
- **Config 2**: trade-off between VRAM and speed
- **Config 3**: relatively aggressive (larger context window, larger batch)

You can copy the commands from this file and run them from the `src/mlc_llm` directory:

```bash
cd /home/liangtaodai/dailt_workplace/ai-hardware-project-proposal-eat-sleep-do-ai/src/mlc_llm
# 然后按照 dse_kv_cache_cmds.md 中的命令依次运行
```

### 4.2 Automated KV DSE benchmark: `kv_cache_dse_bench.py`

Script path:  
`src/mlc_llm/dailt_workplace/dse/kv_cache_dse_bench.py`

This script:

- Takes a set of KV-related parameters:  
  `context_window_size, prefill_chunk_size, max_batch_size, sliding_window_size, attention_sink_size`
- Configures the MLC engine and runs a fixed prompt multiple times
- Records:
  - Approximate **character throughput**: `approx chars/s`
  - Internal MLC metrics: `prefill_tokens_per_s`, `decode_tokens_per_s`, etc.
  - Writes results to CSV: `kv_cache_dse_results_*.csv` and `kv_cache_dse_metrics_*.csv`

#### Basic usage example

From the MLC-LLM root:

```bash
cd /home/liangtaodai/dailt_workplace/ai-hardware-project-proposal-eat-sleep-do-ai/src/mlc_llm

python3.11 dailt_workplace/dse/kv_cache_dse_bench.py \
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

More detailed examples (for the three configs and full DSE over the parameter space) are already written in  
`dailt_workplace/dse/dse_kv_cache_cmds.md`.

---

## 5. PPL Evaluation (HF + MLC models) and Plotting

Relevant scripts are under: `src/mlc_llm/dailt_workplace/dse/`  
The summary document `README_plots.md` gives more details; here is a structured overview.

### 5.1 PPL evaluation: `eval_ppl_wikitext2.py`

Script path:  
`src/mlc_llm/dailt_workplace/dse/eval_ppl_wikitext2.py`

Function:

- Use the **original Hugging Face model** to compute PPL on `wikitext2 / wikitext103` and similar datasets
- Record the pairing with the **corresponding MLC quantized model**, so you can later plot HF vs MLC/quantization

Example (run from `src/mlc_llm`):

```bash
cd /home/liangtaodai/dailt_workplace/ai-hardware-project-proposal-eat-sleep-do-ai/src/mlc_llm

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

Outputs:

- `dailt_workplace/dse/wikitext2_ppl_results.csv`
- `dailt_workplace/dse/wikitext103_ppl_results.csv`

### 5.2 PPL visualization: `plot_wikitext2_ppl.py`

Script path:  
`src/mlc_llm/dailt_workplace/dse/plot_wikitext2_ppl.py`

Example:

```bash
cd /home/liangtaodai/dailt_workplace/ai-hardware-project-proposal-eat-sleep-do-ai/src/mlc_llm

python dailt_workplace/dse/plot_wikitext2_ppl.py \
  --csv dailt_workplace/dse/wikitext2_ppl_results.csv \
  --out dailt_workplace/dse/mem_plots/wikitext2_ppl_vs_quant.png
```

The script automatically detects the quantization type (`fp16 / fake_q3 / 4bit / 8bit`, etc.) and produces a **bar chart of PPL vs quantization config**.

---

## 6. KV Memory / Throughput / Latency Visualization

### 6.1 Memory vs design space: `plot_kv_cache_memory.py`

Path: `src/mlc_llm/dailt_workplace/dse/plot_kv_cache_memory.py`

Input CSV: `kv_cache_dse_results_*.csv` (generated by `kv_cache_dse_bench.py`)  
Example:

```bash
cd /home/liangtaodai/dailt_workplace/ai-hardware-project-proposal-eat-sleep-do-ai/src/mlc_llm

python dailt_workplace/dse/plot_kv_cache_memory.py \
  --csv dailt_workplace/dse/kv_cache_dse_results_Llama-3-8B-Instruct-q0f16-MLC.csv \
  --out-dir dailt_workplace/dse/mem_plots
```

Outputs include:

- `*_mem_kvcache_vs_context.png`
- `*_mem_total_vs_context.png`
- `*_mem_kvcache_vs_sliding.png`

### 6.2 KV memory vs quantization configs: `plot_kv_cache_quantization.py`

Path: `src/mlc_llm/dailt_workplace/dse/plot_kv_cache_quantization.py`

Example:

```bash
python dailt_workplace/dse/plot_kv_cache_quantization.py \
  --csv \
    dailt_workplace/dse/kv_cache_dse_results_Llama-3-8B-Instruct-q0f16-MLC.csv \
    dailt_workplace/dse/kv_cache_dse_results_Llama-3-8B-Instruct-q3f16_1-MLC.csv \
    dailt_workplace/dse/kv_cache_dse_results_Llama-3-8B-Instruct-q4f16_1-MLC.csv \
    dailt_workplace/dse/kv_cache_dse_results_Llama-3-8B-Instruct-q4f32_1-MLC.csv \
  --out dailt_workplace/dse/mem_plots/kv_cache_quant_compare.png
```

### 6.3 Throughput and latency vs quantization / design parameters: `plot_throughput_quantization.py`

Path: `src/mlc_llm/dailt_workplace/dse/plot_throughput_quantization.py`

Input CSV: `kv_cache_dse_metrics_*.csv` (also generated by `kv_cache_dse_bench.py`)  
Example:

```bash
python dailt_workplace/dse/plot_throughput_quantization.py \
  --csv \
    dailt_workplace/dse/kv_cache_dse_metrics_Llama-3-8B-Instruct-q0f16-MLC.csv \
    dailt_workplace/dse/kv_cache_dse_metrics_Llama-3-8B-Instruct-q3f16_1-MLC.csv \
    dailt_workplace/dse/kv_cache_dse_metrics_Llama-3-8B-Instruct-q4f16_1-MLC.csv \
    dailt_workplace/dse/kv_cache_dse_metrics_Llama-3-8B-Instruct-q4f32_1-MLC.csv \
  --out-dir dailt_workplace/dse/mem_plots
```

Together, these plots characterize:

- **Memory**: KV / total vs context / sliding / sink / quant
- **Throughput**: prefill / decode tokens/s vs context / batch / sliding / sink / quant
- **Latency**: end-to-end / TTFT vs the same set of parameters

---

## 7. Compilation Pipeline (TVM Passes) – Optional Reading

If you want to understand the MLC-LLM framework from a **compiler perspective**, read:

- File: `src/mlc_llm/python/mlc_llm/compiler_pass/pipeline.py`
- Key symbol: `@register_pipeline("mlc_llm")`

This file defines the TVM pass sequence from Relax to TIR and finally VM bytecode, including:

- KV cache creation and memory estimation (`DispatchKVCacheCreation`, `AttachMetadataWithMemoryUsage`)
- Quantization-related fusion passes (`FuseDequantizeMatmulEwise`, `FuseFTDequantizeEpilogue`, etc.)
- DLight-level scheduling (`dl.ApplyDefaultSchedule(...)`)

All these passes are automatically executed when you run `python -m mlc_llm compile ...`. In most cases, you **do not need to modify** them unless you are doing compiler research.

---

## 8. Summary: Shortest Path to Reproduce Experiments

1. **Enter the repo and activate the environment**  

   ```bash
   cd /home/liangtaodai/dailt_workplace/ai-hardware-project-proposal-eat-sleep-do-ai/src/mlc_llm
   conda activate mlc-llm-env   # or use run_mlc.sh
   ```

2. **Verify that basic chat works**  

   ```bash
   mlc_llm chat HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC
   ```

3. **Run KV DSE benchmarks** (`kv_cache_dse_bench.py` + commands in `dse_kv_cache_cmds.md`)

4. **Run PPL evaluation and plotting scripts**:  
   `eval_ppl_wikitext2.py`、`plot_wikitext2_ppl.py`、`plot_kv_cache_memory.py`、`plot_kv_cache_quantization.py`、`plot_throughput_quantization.py`
   
Following these steps, you can go from **“using the MLC-LLM framework for basic inference”** to **“systematically evaluating memory / throughput / latency / PPL under different quantization and KV cache settings”**, and fully reproduce the experiments in this project.

