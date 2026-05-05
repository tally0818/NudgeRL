import argparse
import gc
import json
import math
import os
import random
import re
import time
from datetime import datetime, timezone
from typing import Any

import torch
import yaml
from peft import PeftConfig
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from .verify import reward


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_vllm_engine(
    *,
    model_name: str,
    max_seq_len: int | None,
    enable_lora: bool,
) -> tuple[LLM, Any]:
    llm_kwargs: dict[str, Any] = {
        "model": model_name,
        "trust_remote_code": True,
        "enable_lora": bool(enable_lora),
        "max_lora_rank": 32,
    }
    if max_seq_len is not None:
        llm_kwargs["max_model_len"] = int(max_seq_len)
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer

def _fallback_chat_text(messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
    lines: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user")).strip()
        content = msg.get("content", "")
        if isinstance(content, list):
            parts: list[str] = []
            for piece in content:
                if isinstance(piece, dict):
                    txt = piece.get("text", "")
                    if txt:
                        parts.append(str(txt))
            content = "\n".join(parts)
        lines.append(f"{role}: {content}")
    if add_generation_prompt:
        lines.append("assistant:")
    return "\n\n".join(lines)

def render_chat_prompt(
    tokenizer,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool = True,
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            if isinstance(text, str):
                return text
        except Exception:
            pass
    return _fallback_chat_text(messages, add_generation_prompt=add_generation_prompt)

def generate_completions_vllm(
    *,
    llm: LLM,
    tokenizer,
    prompt_messages: list[dict[str, Any]],
    lora_request: LoRARequest | None,
    n_samples: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> list[str]:
    prompt_text = render_chat_prompt(
        tokenizer=tokenizer,
        messages=prompt_messages,
        add_generation_prompt=True,
    )
    completions: list[str] = []
    remaining = int(n_samples)
    chunk_idx = 0

    while remaining > 0:
        cur_batch = min(int(batch_size), remaining)
        if float(temperature) <= 0.0:
            chunk_sampling_params = SamplingParams(
                n=int(cur_batch),
                temperature=0.0,
                max_tokens=int(max_new_tokens),
                seed=int(seed + chunk_idx),
            )
        else:
            chunk_sampling_params = SamplingParams(
                n=int(cur_batch),
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_new_tokens),
                seed=int(seed + chunk_idx),
            )

        generate_kwargs: dict[str, Any] = {"use_tqdm": False}
        if lora_request is not None:
            generate_kwargs["lora_request"] = lora_request

        outputs = llm.generate(
            [prompt_text],
            chunk_sampling_params,
            **generate_kwargs,
        )
        req_out = outputs[0] if outputs else None
        chunk_outputs = getattr(req_out, "outputs", []) if req_out is not None else []
        for out in chunk_outputs:
            text = str(getattr(out, "text", "") or "")
            completions.append(text.strip())
        remaining -= cur_batch
        chunk_idx += 1

    return completions


def estimate_pass_at_k(
    *,
    num_samples: int,
    num_correct: int,
    k: int,
) -> float:
    n = int(num_samples)
    c = max(0, min(int(num_correct), n))
    k = int(k)

    if n < 1:
        return 0.0
    if k < 1:
        raise ValueError("pass@k requires k >= 1")
    if k > n:
        raise ValueError(f"pass@k requires k <= n, got k={k}, n={n}")
    if c == 0:
        return 0.0
    if k == 1:
        return float(c / n)
    if (n - c) < k:
        return 1.0
    return float(1.0 - (math.comb(n - c, k) / math.comb(n, k)))

def evaluate_model_on_dataset(
    *,
    model_label: str,
    llm: LLM,
    tokenizer,
    lora_request: LoRARequest | None,
    dataset: list[dict[str, Any]],
    num_samples: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    estimate_k: int | None = None,
    dataset_label: str | None = None,
) -> dict[str, Any]:
    total = len(dataset)
    pass1_sum = 0.0
    passn_sum = 0.0
    per_problem: list[dict[str, Any]] = []
    source_stats: dict[str, dict[str, float]] = {}
    estimate_k = int(estimate_k) if estimate_k is not None else int(num_samples)
    if estimate_k < 1:
        raise ValueError("estimate_k must be >= 1")
    if estimate_k > int(num_samples):
        raise ValueError(f"estimate_k must be <= num_samples, got estimate_k={estimate_k}, num_samples={num_samples}")

    progress_desc = f"Eval {model_label}" if not dataset_label else f"Eval {model_label} [{dataset_label}]"
    progress = tqdm(dataset, desc=progress_desc, dynamic_ncols=True)
    start_time = time.time()
    passk_key = f"pass@{int(estimate_k)}"

    for idx, sample in enumerate(progress):
        set_seed(int(seed) + idx)
        completions = generate_completions_vllm(
            llm=llm,
            tokenizer=tokenizer,
            prompt_messages=sample["prompt"],
            lora_request=lora_request,
            n_samples=num_samples,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=int(seed) + idx * 1000,
        )
        scores = [int(reward(txt, sample["answer"])) for txt in completions]

        num_correct = int(sum(scores))
        pass1 = estimate_pass_at_k(
            num_samples=num_samples,
            num_correct=num_correct,
            k=1,
        )
        passk = estimate_pass_at_k(
            num_samples=num_samples,
            num_correct=num_correct,
            k=int(estimate_k),
        )

        pass1_sum += pass1
        passn_sum += passk

        source = str(sample.get("source_dataset", "unknown"))
        stat = source_stats.setdefault(source, {"count": 0.0, "pass1_sum": 0.0, "passn_sum": 0.0})
        stat["count"] += 1.0
        stat["pass1_sum"] += float(pass1)
        stat["passn_sum"] += float(passk)

        per_problem.append(
            {
                "global_id": int(sample.get("global_id", idx)),
                "original_id": sample.get("original_id"),
                "source_dataset": source,
                "pass@1": float(pass1),
                passk_key: passk,
                "num_correct_out_of_n": num_correct,
                "is_correct": [int(s) for s in scores],
            }
        )

        seen = idx + 1
        progress.set_postfix(
            pass1=f"{pass1_sum / seen:.3f}",
            passk=f"{passn_sum / seen:.3f}",
        )

    elapsed = time.time() - start_time
    metrics = {
        "num_examples": int(total),
        "num_samples_per_problem": int(num_samples),
        "estimate_k": int(estimate_k),
        "pass@1": float(pass1_sum / max(total, 1)),
        passk_key: float(passn_sum / max(total, 1)),
        "avg_correct_count": float(
            sum(item["num_correct_out_of_n"] for item in per_problem) / max(total, 1)
        ),
        "elapsed_sec": float(elapsed),
        "sec_per_problem": float(elapsed / max(total, 1)),
    }

    source_breakdown: dict[str, dict[str, float]] = {}
    for source, stat in sorted(source_stats.items()):
        count = max(int(stat["count"]), 1)
        source_breakdown[source] = {
            "count": int(stat["count"]),
            "pass@1": float(stat["pass1_sum"] / count),
            passk_key: float(stat["passn_sum"] / count),
        }

    return {
        "model_label": model_label,
        "metrics": metrics,
        "source_breakdown": source_breakdown,
        "per_problem": per_problem,
        "metric_notes": {
            "pass@1": (
                f"Average correctness over the {int(num_samples)} sampled completions "
                f"for each problem; equivalent to num_correct_out_of_n / {int(num_samples)}."
            ),
            passk_key: (
                f"Chen et al. (2021) unbiased pass@{int(estimate_k)} estimate "
                f"computed as 1 - C(n-c, k) / C(n, k) with n={int(num_samples)}, "
                f"k={int(estimate_k)}, c=num_correct_out_of_n."
            ),
            "is_correct": "Per-problem binary correctness sequence for each sampled completion (1=correct, 0=incorrect).",
        },
    }


def evaluate_model_on_aime(
    *,
    model_label: str,
    llm: LLM,
    tokenizer,
    lora_request: LoRARequest | None,
    dataset: list[dict[str, Any]],
    num_samples: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    estimate_k: int | None = None,
) -> dict[str, Any]:
    return evaluate_model_on_dataset(
        model_label=model_label,
        llm=llm,
        tokenizer=tokenizer,
        lora_request=lora_request,
        dataset=dataset,
        num_samples=num_samples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        estimate_k=estimate_k,
        dataset_label="AIME",
    )


def sanitize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "model"
