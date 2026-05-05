import argparse
import yaml
import json
import time
import random
import numpy as np
from datasets import load_from_disk
from openai import OpenAI
from tqdm import tqdm
from .verify import reward



REASONING_START = "<start_working_out>"
REASONING_END = "<end_working_out>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

SYSTEM_PROMPT = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {REASONING_START} and {REASONING_END}.
Then, provide your solution between {SOLUTION_START}{SOLUTION_END}"""

def extract_solution_between_tags(content: str) -> str:
    start_idx = content.find(SOLUTION_START)
    if start_idx == -1:
        return ""

    start_idx += len(SOLUTION_START)
    end_idx = content.find(SOLUTION_END, start_idx)
    if end_idx == -1:
        return ""

    return content[start_idx:end_idx].strip()

def get_oracle_solution_per_prob(problem: str,
                                 gt: str, 
                                 client: OpenAI, 
                                 model: str, 
                                 max_tries: int,
                                 filter_correct: bool = True,
                                 ) -> list:
    prompt = f"""Problem: {problem}"""

    if filter_correct and max_tries <= 0:
        return []

    num_attempts = max_tries if filter_correct else 1

    for attempt in range(1, num_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=16384
            )
            content = (response.choices[0].message.content or "").strip()
            oracle_solution = content

            if not filter_correct:
                return oracle_solution

            predicted_solution = extract_solution_between_tags(content)
            if predicted_solution and reward(predicted_solution, gt):
                return oracle_solution

            print(f"Incorrect oracle solution ({attempt}/{max_tries}), retrying...")
        except Exception as e:
            print(f"Error generating oracle solutions ({attempt}/{num_attempts}): {e}")

    if filter_correct:
        print(f"Failed to generate correct oracle solution after {max_tries} tries. Skipping.")
    else:
        print("Failed to generate oracle solution on first attempt. Skipping.")
    return []
