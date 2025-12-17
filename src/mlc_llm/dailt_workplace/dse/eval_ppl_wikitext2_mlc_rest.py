"""
用 MLC-LLM 的 REST API（/v1/chat/completions + logprobs）在 WikiText-2 上近似评估 PPL。

重要说明：
- 这里用到的是「模型自己生成的 token 的 logprob」，不是对给定 reference 文本的 teacher-forcing 评分；
- 因此这个 PPL 是一个近似指标，只能反映模型在该生成过程中的「自一致性」，不是标准基准 PPL。
"""

import argparse
import math
from typing import Optional

import requests
from datasets import load_dataset


def request_with_logprobs(
    server_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> Optional[dict]:
    """向 MLC-LLM REST 服务器发起一次 chat.completions 请求，要求返回 logprobs。"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": 0,
    }
    resp = requests.post(f"{server_url}/v1/chat/completions", json=payload, timeout=60)
    if resp.status_code != 200:
        print(f"[WARN] HTTP {resp.status_code}: {resp.text[:200]}")
        return None
    return resp.json()


def collect_nll_from_response(resp: dict) -> tuple[float, int]:
    """
    从一次 chat.completions 的响应中，累积生成 token 的负对数似然（NLL）和 token 数。
    这里依赖 OpenAI 风格的 logprobs 字段（每个 choice 带有 token_logprobs）。
    """
    if "choices" not in resp or not resp["choices"]:
        return 0.0, 0

    choice = resp["choices"][0]
    logprobs = choice.get("logprobs")
    if not logprobs:
        return 0.0, 0

    # 兼容两种可能的结构：
    # 1) OpenAI 老格式：logprobs["token_logprobs"] -> List[float]
    # 2) OpenAI 新格式：logprobs["content"] -> List[{"token": str, "logprob": float, ...}]
    nll = 0.0
    count = 0

    # 情况 1：token_logprobs 直接是一个 float 列表
    token_logprobs = logprobs.get("token_logprobs")
    if isinstance(token_logprobs, list) and token_logprobs:
        for lp in token_logprobs:
            if lp is None:
                continue
            nll += -float(lp)
            count += 1
        return nll, count

    # 情况 2：content 列表，每个元素里有 token 和 logprob
    content = logprobs.get("content")
    if isinstance(content, list) and content:
        for item in content:
            # 预期结构：{"token": "...", "logprob": float, "top_logprobs": [...]}
            lp = item.get("logprob")
            if lp is None:
                continue
            nll += -float(lp)
            count += 1
        return nll, count

    # 未识别到有效结构
    return 0.0, 0


def main():
    parser = argparse.ArgumentParser("通过 MLC-LLM REST API（logprobs）在 WikiText-2 上近似评估 PPL")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="MLC-LLM REST 服务器地址（不含路径），例如 http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="传给 REST API 的 model 字段，需与 mlc_llm serve 启动时一致，例如 ./dist/Llama-3-8B-Instruct-q4f16_1-MLC/",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "validation"],
        help="WikiText-2 的数据划分",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="使用多少条样本做近似评估（太大会很慢）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="每个样本生成的最大 token 数",
    )
    parser.add_argument(
        "--prompt-prefix",
        type=str,
        default="Continue the following text as naturally as possible:\n\n",
        help="加在每条 WikiText-2 文本前面的提示前缀",
    )

    args = parser.parse_args()

    print("=== MLC-LLM REST PPL (approx) ===")
    print(f"Server URL : {args.server_url}")
    print(f"Model      : {args.model}")
    print(f"Split      : wikitext-2-raw-v1 / {args.split}")
    print(f"Max samples: {args.max_samples}")
    print(f"Max tokens : {args.max_tokens}")
    print("=================================\n")

    # 1) 加载 WikiText-2 数据
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=args.split)
    texts = [ex["text"] for ex in ds if ex["text"].strip()]
    if args.max_samples is not None and args.max_samples > 0:
        texts = texts[: args.max_samples]

    print(f"使用样本数: {len(texts)}\n")

    # 2) 遍历样本，通过 REST API 获取生成 token 的 logprobs
    total_nll = 0.0
    total_tokens = 0

    for idx, text in enumerate(texts, start=1):
        prompt = args.prompt_prefix + text.strip()

        resp = request_with_logprobs(
            server_url=args.server_url,
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
        )
        if resp is None:
            continue

        nll, count = collect_nll_from_response(resp)
        total_nll += nll
        total_tokens += count

        if idx % 10 == 0:
            print(f"[{idx}/{len(texts)}] 累积 tokens = {total_tokens}")

    if total_tokens == 0:
        print("没有有效的 logprobs 数据，无法计算 PPL。")
        return

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)

    print("\n===== 近似结果（基于生成 token logprobs）=====")
    print(f"总 token 数 : {total_tokens}")
    print(f"平均 NLL    : {avg_nll:.6f}")
    print(f"近似 PPL    : {ppl:.4f}")
    print("===========================================")


if __name__ == "__main__":
    main()


