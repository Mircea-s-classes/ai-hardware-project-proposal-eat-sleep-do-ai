"""
简单脚本：在 WikiText-2 上计算与 MLC-LLM 使用的同一基座模型的 PPL。

思路：
- MLC-LLM 的模型来自 HuggingFace（如 Meta-Llama-3），只是被转换+量化；
- 这里直接用 HuggingFace 的原始 / 量化模型算 PPL，作为 MLC 模型精度的近似参考。
"""

import argparse
import math
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# 可选：bitsandbytes 量化
try:
    from transformers import BitsAndBytesConfig

    HAS_BNB = True
except Exception:  # pragma: no cover - 仅用于运行时环境
    BitsAndBytesConfig = None
    HAS_BNB = False


def map_mlc_to_hf(mlc_model: str) -> str:
    """根据常见 MLC 模型名，映射到对应的 HF 基座模型（只覆盖当前你在用的 Llama-3-8B-Instruct）。"""
    if "Llama-3-8B-Instruct" in mlc_model:
        # PPL 评估用 Base 模型更合理
        return "meta-llama/Meta-Llama-3-8B"
    raise ValueError(f"无法从 mlc 模型名推断 HF 模型，请用 --hf-model 显式指定: {mlc_model}")


def build_model_and_tokenizer(
    hf_model: str,
    device: str,
    cache_dir: Optional[str],
    load_in_4bit: bool,
    load_in_8bit: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(hf_model, cache_dir=cache_dir, trust_remote_code=True)

    # 处理设备（vulkan 回退到 cuda / cpu）
    actual_device = device
    if device.startswith("vulkan"):
        if torch.cuda.is_available():
            print("注意：Transformers 不支持 vulkan，改用 cuda")
            actual_device = "cuda"
        else:
            print("注意：Transformers 不支持 vulkan，且无 cuda，改用 cpu")
            actual_device = "cpu"

    # 构造加载参数
    load_kwargs = {"cache_dir": cache_dir, "trust_remote_code": True}

    quant_config = None
    if load_in_4bit or load_in_8bit:
        if not HAS_BNB:
            raise RuntimeError("需要安装 bitsandbytes 才能使用 --load-in-4bit/--load-in-8bit")
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print("使用 4-bit 量化加载 HF 模型（近似模拟 MLC 量化模型）")
        elif load_in_8bit:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            print("使用 8-bit 量化加载 HF 模型（近似模拟 MLC 量化模型）")

    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config
        # 量化通常需要 cuda
        if actual_device != "cuda" and torch.cuda.is_available():
            print("量化模型将使用 cuda 设备")
            actual_device = "cuda"
    else:
        load_kwargs["torch_dtype"] = torch.float16 if actual_device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(hf_model, **load_kwargs)
    if quant_config is None:
        model = model.to(actual_device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, actual_device


def apply_fake_int3_group_quant(model, group_size: int = 40):
    """
    对 HF 模型做一个“假的”3bit 分组量化（仅权重），用于近似评估 PPL。

    实现细节：
    - 只量化 Linear 的 weight，激活保持不变。
    - 在 CPU 上用张量运算做分组量化，避免在 GPU 上用 Python for 循环导致非常慢。
    - 每 group_size 个输入通道一组，对称 int3（[-3, 3]）+ per-group scale。
    """
    max_int = 3  # 对称 3bit: [-3, 3]

    # 为了速度，把模型临时搬到 CPU 上做量化，再搬回原设备
    orig_device = next(model.parameters()).device
    model.to("cpu")

    def _quant_weight(w: torch.Tensor):
        # w: [out_features, in_features] on CPU
        with torch.no_grad():
            out_dim, in_dim = w.shape
            # padding 使得 in_dim 可以整除 group_size
            pad = (group_size - in_dim % group_size) % group_size
            if pad > 0:
                w_padded = torch.nn.functional.pad(w, (0, pad), mode="constant", value=0.0)
            else:
                w_padded = w

            out_dim, in_dim_padded = w_padded.shape
            num_groups = in_dim_padded // group_size

            w_view = w_padded.view(out_dim, num_groups, group_size)
            # per-group max abs
            max_abs = w_view.abs().max(dim=2, keepdim=True)[0]  # [out_dim, num_groups, 1]
            # 避免除零
            max_abs = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs)
            scale = max_abs / max_int

            q = torch.round(w_view / scale).clamp_(-max_int, max_int)
            w_q = (q * scale).view(out_dim, in_dim_padded)

            # 去掉 padding，并写回原 weight
            if pad > 0:
                w.copy_(w_q[:, :in_dim])
            else:
                w.copy_(w_q)

    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and module.weight is not None:
            _quant_weight(module.weight.data)

    model.to(orig_device)


def compute_ppl_hf_style(
    model,
    tokenizer,
    texts,
    device: str,
    max_length: int = 2048,
    stride: int = 512,
) -> float:
    """
    标准 HF 做法：把所有文本拼成一长串，用滑动窗口计算平均 NLL，再 exp 得到 PPL。
    """
    model.eval()

    joined = "\n\n".join(t for t in texts if t.strip())
    encodings = tokenizer(joined, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    seq_len = input_ids.size(1)
    if seq_len < 2:
        return float("inf")

    nlls = []
    total_target_tokens = 0

    with torch.no_grad():
        for i in range(0, seq_len, stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, seq_len)
            trg_len = end_loc - i
            if trg_len <= 0:
                continue

            input_ids_slice = input_ids[:, begin_loc:end_loc]
            target_ids = input_ids_slice.clone()
            # 只在最后 trg_len 个 token 上计算 loss，其余位置设为 -100 忽略
            target_ids[:, :-trg_len] = -100

            outputs = model(input_ids_slice, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood)
            total_target_tokens += trg_len

    if total_target_tokens == 0 or not nlls:
        return float("inf")

    total_nll = torch.stack(nlls).sum()
    avg_nll = total_nll / total_target_tokens
    return math.exp(avg_nll.item())


def main():
    parser = argparse.ArgumentParser("在 WikiText-2 上计算与 MLC 模型对应的 HF 模型 PPL")
    parser.add_argument(
        "--mlc-model",
        type=str,
        default="HF://mlc-ai/Llama-3-8B-Instruct-q0f16-MLC",
        help="MLC 模型标识，仅用于推断 HF 模型名称",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help="直接指定 HF 模型名称（覆盖 --mlc-model 推断）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda / cpu / vulkan:0（vulkan 会自动回退）",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="评估时的最大序列长度",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="滑动窗口 stride",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "validation"],
        help="WikiText-2 的数据划分",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HF 缓存目录",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="使用 bitsandbytes 的 4-bit 量化加载 HF 模型（近似 MLC 量化）",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="使用 bitsandbytes 的 8-bit 量化加载 HF 模型（近似 MLC 量化）",
    )
    parser.add_argument(
        "--fake-q3bit",
        action="store_true",
        help="在 HF 模型上做简单 3bit 分组量化（权重 int3 近似，激活不变，仅用于 PPL 评估）",
    )

    args = parser.parse_args()

    print("=== WikiText-2 PPL Evaluation ===")
    print(f"MLC Model   : {args.mlc_model}")
    print(f"Device      : {args.device}")
    print(f"Max Length  : {args.max_length}")
    print(f"Stride      : {args.stride}")
    print(f"Split       : {args.split}")
    print("==============================\n")

    # 1) 选择 HF 模型（尽量与 MLC 使用的基座模型一致）
    if args.hf_model:
        hf_model = args.hf_model
    else:
        hf_model = map_mlc_to_hf(args.mlc_model)

    print(f"使用 HF 模型: {hf_model}")

    # 2) 加载模型 + tokenizer（可选 4/8bit 量化）
    model, tokenizer, actual_device = build_model_and_tokenizer(
        hf_model=hf_model,
        device=args.device,
        cache_dir=args.cache_dir,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    # 如果需要 3bit 近似量化，则在 HF 模型权重上做一次离线 int3 分组量化
    if args.fake_q3bit:
        print("对 HF 模型权重应用 3bit 分组量化近似（group_size=40，对标 MLC q3f16_1）...")
        apply_fake_int3_group_quant(model, group_size=40)

    print("模型与 tokenizer 加载完成。\n")

    # 3) 加载 WikiText-2 文本
    print("加载 WikiText-2 数据集...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=args.split)
    texts = [ex["text"] for ex in ds if ex["text"].strip()]
    print(f"样本数: {len(texts)}\n")

    # 4) 计算 PPL
    print("开始计算 PPL（这一步可能比较慢）...")
    ppl = compute_ppl_hf_style(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=actual_device,
        max_length=args.max_length,
        stride=args.stride,
    )

    print("\n===== 结果 =====")
    print(f"HF 模型 : {hf_model}")
    print(f"数据集  : WikiText-2 ({args.split})")
    print(f"PPL     : {ppl:.4f}")
    print("================")


if __name__ == "__main__":
    main()

