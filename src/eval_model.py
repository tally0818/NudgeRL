import argparse
import gc
import json
import os
from datetime import datetime, timezone
from typing import Any

from peft import PeftConfig
from vllm.lora.request import LoRARequest

try:
    from .data.loaders import (
        load_aime_dataset,
        load_amc_dataset,
        load_apex_shortlist_dataset,
        load_math500_dataset,
    )
    from .eval_utils import create_vllm_engine, evaluate_model_on_dataset, sanitize_name
except ImportError:
    from src.data.loaders import (
        load_aime_dataset,
        load_amc_dataset,
        load_apex_shortlist_dataset,
        load_math500_dataset,
    )
    from src.eval_utils import create_vllm_engine, evaluate_model_on_dataset, sanitize_name


SUPPORTED_DATASETS = ("AIME", "AMC23", "MATH500", "APEX_SHORTLIST")
DEFAULT_DATASETS = SUPPORTED_DATASETS
DEFAULT_DATASETS_ARG = ",".join(DEFAULT_DATASETS)
DATASET_ALIASES = {
    "AIME": "AIME",
    "AMC23": "AMC23",
    "AMC": "AMC23",
    "MATH500": "MATH500",
    "MATH": "MATH500",
    "APEX_SHORTLIST": "APEX_SHORTLIST",
    "APEX-SHORTLIST": "APEX_SHORTLIST",
    "APEX": "APEX_SHORTLIST",
}
DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "AIME": {
        "loader": load_aime_dataset,
        "data_dir_arg": "aime_data_dir",
        "display_name": "AIME (combined via load_aime_dataset)",
    },
    "AMC23": {
        "loader": load_amc_dataset,
        "data_dir_arg": "amc23_data_dir",
        "display_name": "AMC23 (combined via load_amc_dataset)",
    },
    "MATH500": {
        "loader": load_math500_dataset,
        "data_dir_arg": "math500_data_dir",
        "display_name": "MATH500 (combined via load_math500_dataset)",
    },
    "APEX_SHORTLIST": {
        "loader": load_apex_shortlist_dataset,
        "data_dir_arg": "apex_shortlist_data_dir",
        "display_name": "Apex Shortlist (combined via load_apex_shortlist_dataset)",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Evaluate a single LoRA adapter on selected datasets ({DEFAULT_DATASETS_ARG})."
    )
    parser.add_argument("--lora-path", type=str, required=True, help="Path to trained LoRA adapter")
    parser.add_argument(
        "--datasets",
        type=str,
        default=DEFAULT_DATASETS_ARG,
        help=f"Comma-separated dataset names. Default: {DEFAULT_DATASETS_ARG}",
    )
    parser.add_argument("--aime-data-dir", type=str, default="./data/aime", help="AIME dataset cache directory")
    parser.add_argument("--amc23-data-dir", type=str, default="./data/amc23", help="AMC23 dataset cache directory")
    parser.add_argument("--math500-data-dir", type=str, default="./data/math500", help="MATH500 dataset cache directory")
    parser.add_argument("--apex-shortlist-data-dir", type=str, default="./data/apex_shortlist", help="Apex Shortlist dataset cache directory")
    parser.add_argument("--output-json", type=str, default=None, help="Where to save evaluation JSON")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N problems per dataset")
    parser.add_argument("--num-samples", type=int, default=128, help="Number of samples per problem")
    parser.add_argument(
        "--estimate-k",
        type=int,
        default=16,
        help="k for pass@k estimation (default: 16)",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Generation batch size for repeated prompt sampling")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature; <=0 means greedy")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p for sampling")
    parser.add_argument("--max-new-tokens", type=int, default=6144, help="Max generated tokens")
    parser.add_argument("--max-seq-len", type=int, default=8192, help="Override max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def _metric_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if value is None and key.startswith("pass@"):
        suffix = key.split("@", 1)[1]
        alt = metrics.get(f"p@{suffix}")
        if alt is not None:
            return float(alt)
    if value is None and key.startswith("p@"):
        suffix = key.split("@", 1)[1]
        alt = metrics.get(f"pass@{suffix}")
        if alt is not None:
            return float(alt)
    if value is None:
        raise KeyError(f"Metric not found: {key}")
    return float(value)


def parse_dataset_names(raw_value: str) -> list[str]:
    dataset_names: list[str] = []
    for token in raw_value.split(","):
        name = token.strip().upper()
        if not name:
            continue
        canonical = DATASET_ALIASES.get(name)
        if canonical is None:
            supported = ", ".join(SUPPORTED_DATASETS)
            raise ValueError(f"Unsupported dataset: {token!r}. Supported datasets: {supported}")
        if canonical not in dataset_names:
            dataset_names.append(canonical)
    if not dataset_names:
        raise ValueError("No valid datasets provided to --datasets")
    return dataset_names


def make_default_output_path(
    *,
    lora_path: str,
    base_model_name: str,
    datasets: list[str],
    num_samples: int,
    estimate_k: int,
    limit: int | None,
) -> str:
    lora_id = sanitize_name(os.path.basename(os.path.normpath(lora_path)))
    model_id = sanitize_name(base_model_name.split("/")[-1])
    dataset_tag = sanitize_name("-".join(ds.lower() for ds in datasets))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_limit{int(limit)}" if limit is not None else ""
    out_dir = os.path.join("outputs", "evals", "single_lora_math_bench")
    os.makedirs(out_dir, exist_ok=True)
    metric_suffix = f"_k{int(estimate_k)}" if int(estimate_k) != int(num_samples) else ""
    return os.path.join(
        out_dir,
        f"{model_id}__{lora_id}__{dataset_tag}_p{int(num_samples)}{metric_suffix}{suffix}__{ts}.json",
    )


def print_summary(
    *,
    dataset_name: str,
    lora_path: str,
    base_model_name: str,
    num_samples: int,
    estimate_k: int,
    metrics: dict[str, Any],
) -> None:
    passk_key = f"pass@{int(estimate_k)}"
    print(f"\n=== {dataset_name} Evaluation (Single LoRA) ===")
    print(
        f"Metric definition: pass@1 = average correctness across the {int(num_samples)} sampled rollouts, "
        f"{passk_key} = Chen et al. unbiased pass@{int(estimate_k)} estimate from those rollouts"
    )
    print(f"Base model: {base_model_name}")
    print(f"LoRA path: {lora_path}")
    print(f"pass@1: {_metric_value(metrics, 'pass@1'):.4f}")
    if passk_key != "pass@1":
        print(f"{passk_key}: {_metric_value(metrics, passk_key):.4f}")


def load_datasets(
    *,
    args: argparse.Namespace,
    dataset_names: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    loaded_datasets: dict[str, list[dict[str, Any]]] = {}
    dataset_meta: dict[str, dict[str, Any]] = {}

    for dataset_name in dataset_names:
        conf = DATASET_CONFIGS[dataset_name]
        loader = conf["loader"]
        data_dir_arg = str(conf["data_dir_arg"])
        data_dir = str(getattr(args, data_dir_arg))

        print(f"Loading {dataset_name} dataset from {data_dir} ...")
        examples = loader(data_dir)
        if args.limit is not None:
            examples = examples[: int(args.limit)]
        print(f"Evaluating {len(examples)} {dataset_name} problems")

        loaded_datasets[dataset_name] = examples
        dataset_meta[dataset_name] = {
            "name": str(conf["display_name"]),
            "num_examples": int(len(examples)),
            "limit": int(args.limit) if args.limit is not None else None,
            "data_dir": data_dir,
        }
    return loaded_datasets, dataset_meta


def main() -> None:
    args = parse_args()
    dataset_names = parse_dataset_names(args.datasets)

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    estimate_k = int(args.estimate_k) if args.estimate_k is not None else int(args.num_samples)
    if estimate_k < 1:
        raise ValueError("--estimate-k must be >= 1")
    if estimate_k > int(args.num_samples):
        raise ValueError("--estimate-k must be <= --num-samples")
    if not os.path.exists(args.lora_path):
        raise FileNotFoundError(f"LoRA path not found: {args.lora_path}")

    peft_cfg = PeftConfig.from_pretrained(args.lora_path)
    base_model_name = str(peft_cfg.base_model_name_or_path)
    lora_request = LoRARequest("eval", 1, args.lora_path)

    print(f"Selected datasets: {', '.join(dataset_names)}")
    datasets, dataset_meta = load_datasets(args=args, dataset_names=dataset_names)

    print(f"Launching vLLM for base model (+LoRA): {base_model_name}")
    llm, tokenizer = create_vllm_engine(
        model_name=base_model_name,
        max_seq_len=args.max_seq_len,
        enable_lora=True,
    )
    results_by_dataset: dict[str, dict[str, Any]] = {}
    try:
        for dataset_name in dataset_names:
            result = evaluate_model_on_dataset(
                model_label="lora",
                llm=llm,
                tokenizer=tokenizer,
                lora_request=lora_request,
                dataset=datasets[dataset_name],
                num_samples=int(args.num_samples),
                batch_size=int(args.batch_size),
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                seed=int(args.seed),
                estimate_k=int(estimate_k),
                dataset_label=dataset_name,
            )
            results_by_dataset[dataset_name] = result
            print_summary(
                dataset_name=dataset_name,
                lora_path=args.lora_path,
                base_model_name=base_model_name,
                num_samples=int(args.num_samples),
                estimate_k=int(estimate_k),
                metrics=result["metrics"],
            )
    finally:
        del llm
        gc.collect()

    output_json = args.output_json or make_default_output_path(
        lora_path=args.lora_path,
        base_model_name=base_model_name,
        datasets=dataset_names,
        num_samples=int(args.num_samples),
        estimate_k=int(estimate_k),
        limit=args.limit,
    )
    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    passk_key = f"pass@{int(estimate_k)}"
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "aime_data_dir": args.aime_data_dir,
        "amc23_data_dir": args.amc23_data_dir,
        "math500_data_dir": args.math500_data_dir,
        "apex_shortlist_data_dir": args.apex_shortlist_data_dir,
        "lora_path": args.lora_path,
        "base_model_name": base_model_name,
        "requested_datasets": dataset_names,
        "generation": {
            "num_samples_per_problem": int(args.num_samples),
            "estimate_k": int(estimate_k),
            "batch_size": int(args.batch_size),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_new_tokens": int(args.max_new_tokens),
            "max_seq_len": int(args.max_seq_len) if args.max_seq_len is not None else None,
            "seed": int(args.seed),
        },
        "datasets": dataset_meta,
        "metric_definitions": {
            "pass@1": (
                f"Per problem, sample N rollouts; pass@1 is the average correctness across those "
                f"N samples, i.e. num_correct_out_of_n / N. Averaged over problems."
            ),
            passk_key: (
                f"Chen et al. (2021) unbiased pass@{int(estimate_k)} estimate from N sampled "
                f"rollouts, using 1 - C(n-c, k) / C(n, k) with n=N, k={int(estimate_k)}."
            ),
            "is_correct": "Per problem binary list (length N): sampled completion correctness in generation order.",
        },
        "lora_results_by_dataset": {
            dataset_name: {"model_ref": args.lora_path, **result}
            for dataset_name, result in results_by_dataset.items()
        },
    }

    if len(dataset_names) == 1:
        only_dataset = dataset_names[0]
        payload["dataset"] = dataset_meta[only_dataset]
        payload["lora"] = {
            "model_ref": args.lora_path,
            **results_by_dataset[only_dataset],
        }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved evaluation JSON to {output_json}")


if __name__ == "__main__":
    main()
