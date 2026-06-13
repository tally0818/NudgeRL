import os
import json
import requests
from typing import List, Dict, Any
from datasets import load_dataset


reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"
solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"

system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""
def download_apex_shortlist_dataset(data_dir: str = "./data/apex_shortlist") -> str:
    os.makedirs(data_dir, exist_ok=True)
    combined_filepath = os.path.join(data_dir, "apex_shortlist.jsonl")
    if os.path.exists(combined_filepath):
        print(f"✅ Apex Shortlist dataset already exists at {combined_filepath}")
        return combined_filepath

    print("Downloading Apex Shortlist dataset from Hugging Face...")
    dataset = load_dataset("MathArena/apex-shortlist")

    all_problems = []
    global_id = 0
    for split_name, split_data in dataset.items():
        print(f"  Processing split: {split_name} ({len(split_data)} samples)")

        for i, item in enumerate(split_data):
            problem = {
                "global_id": global_id,
                "original_id": i,
                "source_dataset": split_name,
                "problem": item.get("problem", ""),
                "answer": str(item.get("answer", "")),
                "solution": item.get("solution", ""),
                "url": item.get("url", "")
            }
            all_problems.append(problem)
            global_id += 1
    with open(combined_filepath, "w", encoding="utf-8") as f:
        for problem in all_problems:
            f.write(json.dumps(problem, ensure_ascii=False) + "\n")

    print(f"✅ Combined {len(all_problems)} problems from Apex Shortlist dataset")
    print(f"   Saved to: {combined_filepath}")

    return combined_filepath

def load_apex_shortlist_dataset(data_dir: str = "./data/apex_shortlist") -> List[Dict[str, Any]]:
    filepath = download_apex_shortlist_dataset(data_dir)

    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    formatted_example = {
                        "global_id": data.get("global_id", line_num),
                        "original_id": data.get("original_id", data.get("id", line_num)),
                        "source_dataset": data.get("source_dataset", "unknown"),
                        "problem": data["problem"],
                        "answer": str(data["answer"]),
                        "solution": data.get("solution", ""),
                        "url": data.get("url", ""),
                        "prompt": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Problem: {data['problem']}\n\nSolve this step by step and provide your final numerical answer."}
                        ]
                    }
                    examples.append(formatted_example)

                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num + 1}: {e}")
                    continue

    print(f"Loaded {len(examples)} problems from combined Apex Shortlist dataset")
    source_counts = {}
    for example in examples:
        source = example['source_dataset']
        source_counts[source] = source_counts.get(source, 0) + 1

    for source, count in source_counts.items():
        print(f"  {source}: {count} problems")

    return examples

def download_and_combine_aime_datasets(data_dir: str = "./data/aime") -> str:

    datasets = {
        "test2024": "https://raw.githubusercontent.com/GAIR-NLP/AIME-Preview/main/eval/data/aime/test2024.jsonl",
        "test2025-I": "https://raw.githubusercontent.com/GAIR-NLP/AIME-Preview/main/eval/data/aime/test2025-I.jsonl",
        "test2025-II": "https://raw.githubusercontent.com/GAIR-NLP/AIME-Preview/main/eval/data/aime/test2025-II.jsonl"
    }

    os.makedirs(data_dir, exist_ok=True)
    combined_filepath = os.path.join(data_dir, "aime.jsonl")
    if os.path.exists(combined_filepath):
        print(f"Combined AIME dataset already exists at {combined_filepath}")
        return combined_filepath

    print("Downloading and combining AIME datasets...")

    all_problems = []
    global_id = 0

    for dataset_name, url in datasets.items():
        print(f"  Downloading {dataset_name}...")

        try:
            response = requests.get(url)
            response.raise_for_status()
            for line_num, line in enumerate(response.text.strip().split('\n')):
                if line.strip():
                    try:
                        data = json.loads(line)
                        data['source_dataset'] = dataset_name
                        data['original_id'] = data.get('id', line_num)
                        data['global_id'] = global_id
                        global_id += 1
                        all_problems.append(data)
                    except json.JSONDecodeError as e:
                        print(f"    Warning: Error parsing line {line_num + 1} in {dataset_name}: {e}")
                        continue

        except requests.RequestException as e:
            print(f"    Error downloading {dataset_name}: {e}")
            continue
    if all_problems:
        with open(combined_filepath, 'w', encoding='utf-8') as f:
            for problem in all_problems:
                f.write(json.dumps(problem, ensure_ascii=False) + '\n')

        print(f"✅ Combined {len(all_problems)} problems from {len(datasets)} datasets")
        print(f"   Saved to: {combined_filepath}")
        for dataset_name in datasets.keys():
            count = sum(1 for p in all_problems if p['source_dataset'] == dataset_name)
            print(f"   {dataset_name}: {count} problems")

    else:
        raise RuntimeError("No problems were successfully downloaded")

    return combined_filepath
def load_aime_dataset(data_dir: str = "./data/aime") -> List[Dict[str, Any]]:
    filepath = download_and_combine_aime_datasets(data_dir)

    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    formatted_example = {
                        "global_id": data.get("global_id", line_num),
                        "original_id": data.get("original_id", data.get("id", line_num)),
                        "source_dataset": data.get("source_dataset", "unknown"),
                        "problem": data["problem"],
                        "answer": str(data["answer"]),
                        "solution": data.get("solution", ""),
                        "url": data.get("url", ""),
                        "prompt": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Problem: {data['problem']}\n\nSolve this step by step and provide your final numerical answer."}
                        ]
                    }
                    examples.append(formatted_example)

                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num + 1}: {e}")
                    continue

    print(f"Loaded {len(examples)} problems from combined AIME dataset")
    source_counts = {}
    for example in examples:
        source = example['source_dataset']
        source_counts[source] = source_counts.get(source, 0) + 1

    for source, count in source_counts.items():
        print(f"  {source}: {count} problems")

    return examples
def download_math500_dataset(data_dir: str = "./data/math500") -> str:
    os.makedirs(data_dir, exist_ok=True)
    combined_filepath = os.path.join(data_dir, "math500.jsonl")

    if os.path.exists(combined_filepath):
        print(f"✅ MATH-500 dataset already exists at {combined_filepath}")
        return combined_filepath

    print("Downloading MATH-500 dataset from Hugging Face...")
    dataset = load_dataset("HuggingFaceH4/MATH-500")

    all_problems = []
    global_id = 0

    for split_name, split_data in dataset.items():
        print(f"  Processing split: {split_name} ({len(split_data)} samples)")

        for i, item in enumerate(split_data):
          if item['level'] == 5:
              problem = {
                  "global_id": global_id,
                  "original_id": i,
                  "source_dataset": split_name,
                  "problem": item.get("problem", ""),
                  "answer": str((item.get("answer", ""))),
                  "solution": item.get("solution", ""),
                  "url": item.get("url", "")
              }
              all_problems.append(problem)
              global_id += 1

    with open(combined_filepath, "w", encoding="utf-8") as f:
        for problem in all_problems:
            f.write(json.dumps(problem, ensure_ascii=False) + "\n")

    print(f"   Saved to: {combined_filepath}")

    return combined_filepath

def load_math500_dataset(data_dir: str = "./data/math500") -> List[Dict[str, Any]]:
    filepath = download_math500_dataset(data_dir)

    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    formatted_example = {
                        "global_id": data.get("global_id", line_num),
                        "original_id": data.get("original_id", data.get("id", line_num)),
                        "source_dataset": data.get("source_dataset", "unknown"),
                        "problem": data["problem"],
                        "answer": str(data["answer"]),
                        "solution": data.get("solution", ""),
                        "url": data.get("url", ""),
                        "prompt": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Problem: {data['problem']}\n\nSolve this step by step and provide your final numerical answer."}
                        ]
                    }
                    examples.append(formatted_example)

                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num + 1}: {e}")
                    continue

    print(f"Loaded {len(examples)} problems from combined MATH-500 dataset")
    source_counts = {}
    for example in examples:
        source = example['source_dataset']
        source_counts[source] = source_counts.get(source, 0) + 1

    for source, count in source_counts.items():
        print(f"  {source}: {count} problems")

    return examples

def download_amc23_dataset(data_dir: str = "./data/amc23") -> str:
    os.makedirs(data_dir, exist_ok=True)
    combined_filepath = os.path.join(data_dir, "amc23.jsonl")
    if os.path.exists(combined_filepath):
        print(f"✅ Combined AMC23 dataset already exists at {combined_filepath}")
        return combined_filepath

    print("Downloading AMC23 dataset from Hugging Face...")
    dataset = load_dataset("zwhe99/amc23")

    all_problems = []
    global_id = 0

    for split_name, split_data in dataset.items():
        print(f"  Processing split: {split_name} ({len(split_data)} samples)")

        for i, item in enumerate(split_data):
            problem = {
                "global_id": global_id,
                "original_id": i,
                "source_dataset": split_name,
                "problem": item.get("question", ""),
                "answer": str(int(item.get("answer", ""))),
                "solution": item.get("solution", ""),
                "url": item.get("url", "")
            }
            all_problems.append(problem)
            global_id += 1

    with open(combined_filepath, "w", encoding="utf-8") as f:
        for problem in all_problems:
            f.write(json.dumps(problem, ensure_ascii=False) + "\n")

    print(f"✅ Combined {len(all_problems)} problems from AMC23 dataset")
    print(f"   Saved to: {combined_filepath}")

    return combined_filepath


def load_amc_dataset(data_dir: str = "./data/amc23") -> List[Dict[str, Any]]:
    filepath = download_amc23_dataset(data_dir)

    examples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    formatted_example = {
                        "global_id": data.get("global_id", line_num),
                        "original_id": data.get("original_id", data.get("id", line_num)),
                        "source_dataset": data.get("source_dataset", "unknown"),
                        "problem": data["problem"],
                        "answer": str(data["answer"]),
                        "solution": data.get("solution", ""),
                        "url": data.get("url", ""),
                        "prompt": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Problem: {data['problem']}\n\nSolve this step by step and provide your final numerical answer."}
                        ]
                    }
                    examples.append(formatted_example)

                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num + 1}: {e}")
                    continue

    print(f"Loaded {len(examples)} problems from combined AMC23 dataset")
    source_counts = {}
    for example in examples:
        source = example['source_dataset']
        source_counts[source] = source_counts.get(source, 0) + 1

    for source, count in source_counts.items():
        print(f"  {source}: {count} problems")

    return examples
