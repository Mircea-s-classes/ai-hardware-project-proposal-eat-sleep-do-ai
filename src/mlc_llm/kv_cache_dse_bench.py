import argparse
import contextlib
import csv
import os
import re
import sys
import threading
import time
from itertools import product

from mlc_llm import MLCEngine
from mlc_llm.serve.config import EngineConfig


# KV Cache DSE 参数搜索空间（表格）
# 每个字段是一维离散取值，真正用于 DSE 时建议通过
# generate_dse_points() 生成满足约束的组合。
PARAM_SPACE = {
    "context_window_size": [4096, 8192, 16384],
    "prefill_chunk_size": [1024, 2048, 4096],
    "max_batch_size": [1, 2, 4, 8],
    "sliding_window_size": [-1, 2048, 4096],  # -1 表示不开 sliding window
    # attention_sink_size 的候选集合，具体可用约束过滤
    "attention_sink_size": [0, 32, 64, 128, 256],
}

_GPU_MEM_LOG_PATTERN = re.compile(
    r"Estimated total single GPU memory usage:\s*([0-9.]+)\s*MB\s*"
    r"\(Parameters:\s*([0-9.]+)\s*MB\. KVCache:\s*([0-9.]+)\s*MB\. "
    r"Temporary buffer:\s*([0-9.]+)\s*MB\)"
)


@contextlib.contextmanager
def _capture_engine_stderr():
    """捕获 MLCEngine 初始化期间的 stderr 日志，用于解析 GPU 内存估算信息。"""
    orig_fd = os.dup(2)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)
    os.close(write_fd)

    lines = []

    def _reader():
        with os.fdopen(read_fd, "r", buffering=1) as rf:
            for line in rf:
                lines.append(line)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    try:
        yield lines
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(orig_fd, 2)
        os.close(orig_fd)
        t.join(timeout=1.0)


def _parse_gpu_mem_from_logs(log_lines):
    """从 C++ 日志里解析 GPU 内存估算结果。"""
    text = "".join(log_lines)
    m = _GPU_MEM_LOG_PATTERN.search(text)
    if not m:
        return None
    total_mb, params_mb, kv_mb, temp_mb = map(float, m.groups())
    return {
        "gpu_mem_total_mb": total_mb,
        "gpu_mem_params_mb": params_mb,
        "gpu_mem_kvcache_mb": kv_mb,
        "gpu_mem_temp_mb": temp_mb,
    }


def _model_tag(model: str) -> str:
    """根据模型名字生成一个适合放在文件名里的 tag。"""
    name = model
    # 去掉 schema（例如 HF://）
    if "://" in name:
        name = name.split("://", 1)[1]
    # 取最后一段 path
    name = name.rstrip("/").split("/")[-1]
    # 只保留安全字符，其它变成下划线
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _append_result_row(
    *,
    model: str,
    device: str,
    context_window_size: int,
    prefill_chunk_size: int,
    max_batch_size: int,
    sliding_window_size: int,
    attention_sink_size: int,
    num_runs: int,
    max_tokens: int,
    run_id: int,
    dt: float,
    gen_tokens: int,
    gpu_mem_total_mb=None,
    gpu_mem_params_mb=None,
    gpu_mem_kvcache_mb=None,
    gpu_mem_temp_mb=None,
):
    """把当前 run 的参数和结果追加到一个 CSV 表格里。"""
    results_csv = f"kv_cache_dse_results_{_model_tag(model)}.csv"
    file_exists = os.path.exists(results_csv)
    with open(results_csv, "a", newline="") as f:
        fieldnames = [
            "model",
            "device",
            "context_window_size",
            "prefill_chunk_size",
            "max_batch_size",
            "sliding_window_size",
            "attention_sink_size",
            "num_runs",
            "max_tokens",
            "run_id",
            "time_s",
            "approx_chars",
            "gpu_mem_total_mb",
            "gpu_mem_params_mb",
            "gpu_mem_kvcache_mb",
            "gpu_mem_temp_mb",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "model": model,
                "device": device,
                "context_window_size": context_window_size,
                "prefill_chunk_size": prefill_chunk_size,
                "max_batch_size": max_batch_size,
                "sliding_window_size": sliding_window_size,
                "attention_sink_size": attention_sink_size,
                "num_runs": num_runs,
                "max_tokens": max_tokens,
                "run_id": run_id,
                "time_s": dt,
                "approx_chars": gen_tokens,
                "gpu_mem_total_mb": gpu_mem_total_mb,
                "gpu_mem_params_mb": gpu_mem_params_mb,
                "gpu_mem_kvcache_mb": gpu_mem_kvcache_mb,
                "gpu_mem_temp_mb": gpu_mem_temp_mb,
            }
        )


def _append_metrics_rows(
    *,
    model: str,
    device: str,
    context_window_size: int,
    prefill_chunk_size: int,
    max_batch_size: int,
    sliding_window_size: int,
    attention_sink_size: int,
    num_runs: int,
    max_tokens: int,
    metrics_text: str,
):
    """把 engine.metrics().prometheus_text() 中的所有指标写入 CSV，每个 metric 一行。"""
    metrics_csv = f"kv_cache_dse_metrics_{_model_tag(model)}.csv"
    file_exists = os.path.exists(metrics_csv)
    with open(metrics_csv, "a", newline="") as f:
        fieldnames = [
            "model",
            "device",
            "context_window_size",
            "prefill_chunk_size",
            "max_batch_size",
            "sliding_window_size",
            "attention_sink_size",
            "num_runs",
            "max_tokens",
            "metric_name",
            "metric_value",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for line in metrics_text.splitlines():
            line = line.strip()
            # 忽略注释和空行
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            # 形如：name  value  或  name{label=...}  value
            metric_name = " ".join(parts[:-1])
            metric_value = parts[-1]
            writer.writerow(
                {
                    "model": model,
                    "device": device,
                    "context_window_size": context_window_size,
                    "prefill_chunk_size": prefill_chunk_size,
                    "max_batch_size": max_batch_size,
                    "sliding_window_size": sliding_window_size,
                    "attention_sink_size": attention_sink_size,
                    "num_runs": num_runs,
                    "max_tokens": max_tokens,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                }
            )


def generate_dse_points():
    """生成用于 DSE 的参数组合，每个元素是一个 dict。

    约束：
    - prefill_chunk_size <= context_window_size
    - sliding_window_size <= 0 时，attention_sink_size = 0
    - sliding_window_size > 0 时，0 <= attention_sink_size <= sliding_window_size / 8
    """

    for (
        context_window_size,
        prefill_chunk_size,
        max_batch_size,
        sliding_window_size,
    ) in product(
        PARAM_SPACE["context_window_size"],
        PARAM_SPACE["prefill_chunk_size"],
        PARAM_SPACE["max_batch_size"],
        PARAM_SPACE["sliding_window_size"],
    ):
        # 约束：prefill_chunk_size 不能大于 context_window_size
        if prefill_chunk_size > context_window_size:
            continue

        for attention_sink_size in PARAM_SPACE["attention_sink_size"]:
            # 不开 sliding window 时，强制 sink 为 0
            if sliding_window_size <= 0:
                if attention_sink_size != 0:
                    continue
            else:
                # 开启 sliding window 时，sink 不能太大
                max_sink = max(0, sliding_window_size // 8)
                if not (0 <= attention_sink_size <= max_sink):
                    continue

            yield {
                "context_window_size": context_window_size,
                "prefill_chunk_size": prefill_chunk_size,
                "max_batch_size": max_batch_size,
                "sliding_window_size": sliding_window_size,
                "attention_sink_size": attention_sink_size,
            }


def run_bench(
    model: str,
    device: str,
    context_window_size: int,
    prefill_chunk_size: int,
    max_batch_size: int,
    sliding_window_size: int,
    attention_sink_size: int,
    num_runs: int,
    max_tokens: int,
):
    # 构造 EngineConfig（使用当前版本支持的字段）
    # - 用 max_total_sequence_length / max_single_sequence_length 表达 context_window_size
    # - 用 max_num_sequence 近似表达 max_batch_size
    engine_config = EngineConfig(
        model=model,
        mode="local",  # local / interactive / server，会影响默认容量
        max_total_sequence_length=context_window_size,
        max_single_sequence_length=context_window_size,
        max_num_sequence=max_batch_size,
        prefill_chunk_size=prefill_chunk_size,
        sliding_window_size=sliding_window_size,
        attention_sink_size=attention_sink_size,
    )

    print("=== KV DSE Bench Config ===")
    print(f"model                = {model}")
    print(f"device               = {device}")
    print(f"context_window_size  = {context_window_size}")
    print(f"prefill_chunk_size   = {prefill_chunk_size}")
    print(f"max_batch_size       = {max_batch_size}")
    print(f"sliding_window_size  = {sliding_window_size}")
    print(f"attention_sink_size  = {attention_sink_size}")
    print(f"num_runs             = {num_runs}")
    print(f"max_tokens_per_run   = {max_tokens}")
    print("============================\n", flush=True)

    # 直接把 engine_config 传给 MLCEngine，并在初始化阶段捕获 C++ 日志，解析 GPU 内存估算信息
    with _capture_engine_stderr() as err_lines:
        engine = MLCEngine(model, device=device, mode="local", engine_config=engine_config)
    gpu_mem_info = _parse_gpu_mem_from_logs(err_lines)

    prompt = "What is AI?\n What is Hardware?\n What is AI Hardware?\n"

    total_time = 0.0
    total_tokens = 0

    # 简单跑 num_runs 次，统计整体吞吐（我们自己算一份方便看）
    for run_id in range(1, num_runs + 1):
        messages = [{"role": "user", "content": prompt}]
        start = time.time()
        gen_tokens = 0

        for response in engine.chat.completions.create(
            messages=messages,
            model=model,
            stream=True,
            max_tokens=max_tokens,
        ):
            for choice in response.choices:
                text = choice.delta.content or ""
                gen_tokens += len(text)  # 粗略当作 token 数
                print(text, end="", flush=True)

        end = time.time()
        dt = end - start
        total_time += dt
        total_tokens += gen_tokens

        print("\n")
        print(f"[Run {run_id}] time = {dt:.3f} s, approx chars = {gen_tokens}")
        print("-" * 40)

        # 把这一轮的参数 + 结果写入 CSV 表格
        _append_result_row(
            model=model,
            device=device,
            context_window_size=context_window_size,
            prefill_chunk_size=prefill_chunk_size,
            max_batch_size=max_batch_size,
            sliding_window_size=sliding_window_size,
            attention_sink_size=attention_sink_size,
            num_runs=num_runs,
            max_tokens=max_tokens,
            run_id=run_id,
            dt=dt,
            gen_tokens=gen_tokens,
            gpu_mem_total_mb=(gpu_mem_info or {}).get("gpu_mem_total_mb"),
            gpu_mem_params_mb=(gpu_mem_info or {}).get("gpu_mem_params_mb"),
            gpu_mem_kvcache_mb=(gpu_mem_info or {}).get("gpu_mem_kvcache_mb"),
            gpu_mem_temp_mb=(gpu_mem_info or {}).get("gpu_mem_temp_mb"),
        )

    if total_time > 0:
        tps = total_tokens / total_time
    else:
        tps = 0.0

    print("\n=== KV DSE Bench Result (script local) ===")
    print(f"total_time      = {total_time:.3f} s")
    print(f"total_chars     = {total_tokens}")
    print(f"approx chars/s  = {tps:.2f}")
    print("=========================================\n")

    # 再从引擎查询一次官方 metrics（由 MLC 提供）
    metrics = engine.metrics()
    metrics_text = metrics.prometheus_text()
    print("=== Engine Metrics (from MLC) ===")
    # 直接打印 dict，或者用 prometheus_text 更易读
    print(metrics_text)
    print("=================================")

    # 把所有 metrics（prometheus_text 里的每一项）也写入一个 CSV
    _append_metrics_rows(
        model=model,
        device=device,
        context_window_size=context_window_size,
        prefill_chunk_size=prefill_chunk_size,
        max_batch_size=max_batch_size,
        sliding_window_size=sliding_window_size,
        attention_sink_size=attention_sink_size,
        num_runs=num_runs,
        max_tokens=max_tokens,
        metrics_text=metrics_text,
    )

    engine.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC",
    )
    parser.add_argument("--device", type=str, default="vulkan:0")

    parser.add_argument("--context-window-size", type=int, default=8192)
    parser.add_argument("--prefill-chunk-size", type=int, default=8192)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--sliding-window-size", type=int, default=-1)
    parser.add_argument("--attention-sink-size", type=int, default=0)

    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)

    # 是否运行 DSE：遍历 generate_dse_points() 中的所有组合
    parser.add_argument(
        "--dse",
        action="store_true",
        help="Enumerate all parameter combinations in PARAM_SPACE and run DSE.",
    )

    args = parser.parse_args()
    # 单点运行 或 全参数 DSE
    if args.dse:
        print("=== Running KV Cache DSE over PARAM_SPACE ===")
        t0 = time.time()
        num_points = 0
        for num_points, point in enumerate(generate_dse_points(), start=1):
            print(f"\n===== DSE Point {num_points} =====")
            print(point)
            run_bench(
                model=args.model,
                device=args.device,
                context_window_size=point["context_window_size"],
                prefill_chunk_size=point["prefill_chunk_size"],
                max_batch_size=point["max_batch_size"],
                sliding_window_size=point["sliding_window_size"],
                attention_sink_size=point["attention_sink_size"],
                num_runs=args.num_runs,
                max_tokens=args.max_tokens,
            )
        t1 = time.time()
        print(
            f"\n=== DSE finished: {num_points} points, total wall time = {t1 - t0:.3f} s ==="
        )
    else:
        run_bench(
            model=args.model,
            device=args.device,
            context_window_size=args.context_window_size,
            prefill_chunk_size=args.prefill_chunk_size,
            max_batch_size=args.max_batch_size,
            sliding_window_size=args.sliding_window_size,
            attention_sink_size=args.attention_sink_size,
            num_runs=args.num_runs,
            max_tokens=args.max_tokens,
        )