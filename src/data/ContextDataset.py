import argparse
import json
from ..generate_hints import get_hint_per_prob
from datasets import load_from_disk
import random
import torch
import os
import jsonlines as jsonl
import yaml
from openai import OpenAI

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_context_dataset(dataset, model_name, num_contexts, num_samples, api_key, save_path):
    dataset = dataset.select(range(num_samples))
    client = OpenAI(api_key=api_key)

    def add_hints(example):
        example["hints"] = get_hint_per_prob(example["prompt"], client, model_name, num_contexts)
        return example

    dataset = dataset.map(add_hints)

    keep_cols = ["prompt", "solution", "hints"]
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep_cols])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with jsonl.open(save_path, "w") as writer:
        writer.write_all(dataset.to_list())

def main():
    parser = argparse.ArgumentParser(description='Process context dataset.')
    parser.add_argument('--config', type=str, required=True, help='The file path to the context dataset')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file) 
    model_name = config['model_name']
    num_contexts = config['num_contexts']
    num_samples = config['num_samples']
    api_key = config['api_key']   
    seed = config['seed']
    save_path = "data/dapo17k"+ f"_contexts_{num_contexts}_samples_{num_samples}.jsonl"
    if save_path and os.path.exists(save_path):
        print(f"Dataset existing at {save_path}")
        return 
    set_seed(seed)
    print("\nLoading Arrow dataset...")
    data_path = "data/dapo-17k"
    dataset = load_from_disk(data_path)['train']
    make_context_dataset(dataset, model_name, num_contexts, num_samples, api_key, save_path)
    return 

if __name__ == "__main__":
    main()
