import argparse
import os
import random

import jsonlines as jsonl
import torch
import yaml
from datasets import load_from_disk
from openai import OpenAI

from ..generate_oracle_solutions import get_oracle_solution_per_prob


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_oracle_solution(raw_solution) -> str:
    if isinstance(raw_solution, list):
        raw_solution = raw_solution[0] if raw_solution else ""
    if raw_solution is None:
        return ""
    if isinstance(raw_solution, str):
        return raw_solution
    return str(raw_solution)


def _build_oracle_solution_list(
    prompt: str,
    answer: str,
    client: OpenAI,
    model_name: str,
    max_tries: int,
    filter_correct: bool,
    num_contexts: int,
) -> list[str]:
    oracle_solutions: list[str] = []
    for _ in range(num_contexts):
        raw_solution = get_oracle_solution_per_prob(
            prompt,
            answer,
            client,
            model_name,
            max_tries,
            filter_correct,
        )
        oracle_solutions.append(_normalize_oracle_solution(raw_solution))
    return oracle_solutions


def make_pope_dataset(
    dataset,
    model_name: str,
    num_samples: int,
    api_key: str,
    max_tries: int,
    filter_correct: bool,
    num_contexts: int,
    save_path: str,
) -> None:
    dataset = dataset.select(range(num_samples))
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def add_oracle_solutions(example):
        example["oracle_solution"] = _build_oracle_solution_list(
            prompt=example["prompt"],
            answer=example["solution"],
            client=client,
            model_name=model_name,
            max_tries=max_tries,
            filter_correct=filter_correct,
            num_contexts=num_contexts,
        )
        return example

    dataset = dataset.map(add_oracle_solutions)

    keep_cols = ["prompt", "solution", "oracle_solution"]
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep_cols])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with jsonl.open(save_path, "w") as writer:
        writer.write_all(dataset.to_list())


def _dataset_matches_requested_contexts(save_path: str, num_contexts: int) -> bool:
    if not os.path.exists(save_path):
        return False

    try:
        with jsonl.open(save_path, "r") as reader:
            first_row = next(iter(reader), None)
    except Exception:
        return False

    if first_row is None:
        return False

    oracle_solutions = first_row.get("oracle_solution")
    return isinstance(oracle_solutions, list) and len(oracle_solutions) >= num_contexts


def main():
    parser = argparse.ArgumentParser(description="Build POPE-style oracle solution dataset.")
    parser.add_argument("--config", type=str, required=True, help="Path to context config file.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_name = config["pope_style"]["model_name"]
    num_samples = int(config["num_samples"])
    num_contexts = int(config.get("num_contexts", config.get("num_context", 1)))
    if num_contexts <= 0:
        raise ValueError(f"num_contexts must be positive, got {num_contexts}.")

    api_key = config["pope_style"]["Deepseek_api_key"]
    seed = int(config["seed"])
    max_tries = int(config["pope_style"]["Max_tries"])
    filter_correct = bool(config["pope_style"].get("filter_correct", True))

    save_path = f"data/dapo17k_pope_{num_samples}.jsonl"
    if _dataset_matches_requested_contexts(save_path, num_contexts):
        print(f"Dataset already exists at {save_path} with >= {num_contexts} oracle solutions per sample.")
        return
    if os.path.exists(save_path):
        print(f"Rebuilding dataset at {save_path} to match num_contexts={num_contexts}.")

    set_seed(seed)
    print("\nLoading Arrow dataset...")
    dataset = load_from_disk("data/dapo-17k")["train"]

    make_pope_dataset(
        dataset=dataset,
        model_name=model_name,
        num_samples=num_samples,
        api_key=api_key,
        max_tries=max_tries,
        filter_correct=filter_correct,
        num_contexts=num_contexts,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
