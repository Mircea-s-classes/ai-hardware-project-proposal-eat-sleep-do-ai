[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/v3c0XywZ)
# Towards Fast and Efficient Deep Neural Network Deployment: Supporting Diverse Quantization in the TVM Compiler
ECE 4332 / ECE 6332 — AI Hardware Design and Implementation  
Fall 2025

## 🧭 Overview
This repository contains the code, scripts, and documentation for our AI Hardware course project.  
We build on **MLC-LLM** and **TVM** to study **efficient deployment of quantized large language models (LLMs)** on GPUs, with a focus on:
- Supporting diverse quantization formats in the compilation pipeline
- Exploring KV cache design spaces (context window, batch size, sliding window, etc.)
- Understanding trade-offs between memory footprint, throughput, latency, and model quality (PPL)

## 🗂 Folder Structure
- `docs/` – project proposal and documentation for this specific project  
- `presentations/` – midterm and final presentation slides for our team  
- `report/` – final written report and related materials  
- `src/` – source code for software, hardware, and experiments  
  - most of our work lives in `src/mlc_llm/` (see **Code Layout in This Repo** below)  
- `data/` – datasets or pointers to data used in our experiments

## 🧑‍🤝‍🧑 Team
- **Team name**: Eat, Sleep, Do AI  
- **Members**:
  - Liangtao Dai  
  - Elton Jhang  
  - Haonan Ke  
  - Xinwei Li  

More details about roles and responsibilities are documented in `docs/Project_Proposal.md`.

## 📋 Project Artifacts
By the end of the semester, this repo will contain:
1. **Reproducible MLC-LLM + TVM scripts** for quantized LLM deployment and KV cache DSE (under `src/mlc_llm/dailt_workplace/`)  
2. **CSV logs and plots** analyzing memory, throughput, latency, and PPL (under `src/mlc_llm/dailt_workplace/dse/`)  
3. **Presentation slides** summarizing methodology and results (under `presentations/`)  
4. **Final report** with technical details and analysis (under `report/`)  

## 🚀 How to Use This Repository
If you want to **run our code or reproduce results**, start from:

1. The **Quick Start for Our Code** section below (environment setup + basic commands)  
2. The detailed usage guide in  
   `src/mlc_llm/dailt_workplace/HowTo.md`  

## 🧪 This Team's Project

Our team project focuses on **efficient deployment of quantized large language models using the MLC-LLM framework and TVM**.  
We study how different **quantization schemes** and **KV cache configurations** affect:
- GPU memory footprint
- Throughput (tokens/s or chars/s)
- End-to-end and first-token latency
- Model quality (PPL on language modeling benchmarks)

High-level goals:
- Support **diverse quantization formats** in the compilation flow.
- Explore **design spaces** (context window, batch size, sliding window, etc.) for KV cache.
- Provide **reproducible scripts** and **plots** to visualize the trade-offs between memory, speed, and accuracy.

## 🔧 Code Layout in This Repo

- `docs/` – proposal and project documentation (including technical plan and responsibilities)  
- `presentations/` – project slides (midterm + final)  
- `report/` – final report sources  
- `src/` – main source code:
  - `src/hardware/` – hardware-related experiments or configurations (if any)
  - `src/model/` – model-related scripts (if any)
  - `src/mlc_llm/` – **MLC-LLM framework (upstream source + our extensions)**
    - `dailt_workplace/` – **our working directory inside MLC-LLM**
      - `dse/` – KV cache design space exploration, PPL evaluation and plotting scripts
      - `test/` – small helper scripts for quick tests

For details on how to use the MLC-LLM framework **within this project**, see:  
`src/mlc_llm/dailt_workplace/HowTo.md`

## 🚀 Quick Start for Our Code

From the repo root:

```bash
cd src/mlc_llm
```

1. **Environment (cluster or local)**  
   - Create/activate Conda env (example):
     ```bash
     conda create -n mlc-llm-env python=3.11 -y
     conda activate mlc-llm-env
     ```
   - Install MLC-LLM from source:
     ```bash
     pip install -e .
     ```
   - Or use the helper script (if on the course cluster):
     ```bash
     bash run_mlc.sh
     ```

2. **Basic MLC-LLM Chat**  
   ```bash
   mlc_llm chat HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC
   ```

3. **Run our KV cache DSE and plots**  
   - KV cache benchmark + metrics: see commands in  
     `src/mlc_llm/dailt_workplace/dse/dse_kv_cache_cmds.md`
   - PPL evaluation and visualization:  
     `eval_ppl_wikitext2.py`, `plot_wikitext2_ppl.py`  
   - Memory / throughput / latency visualization:  
     `plot_kv_cache_memory.py`, `plot_kv_cache_quantization.py`, `plot_throughput_quantization.py`

These scripts together reproduce our experiments on **quantization-aware deployment of LLMs** using **MLC-LLM + TVM**.

## 📜 License
This project is released under the MIT License.
