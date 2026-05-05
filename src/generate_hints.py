import argparse
import yaml
import json
import time
import random
import numpy as np
from datasets import load_from_disk
from openai import OpenAI
from tqdm import tqdm


def get_hint_per_prob(problem: str, client: OpenAI, gpt_model: str, num_hints: int) -> list:
    prompt = f"""Given the following math problem, generate {num_hints} different keyword hints that would help solve it. 
Each hint should be a specific mathematical concept, theorem, or technique (e.g., "Ceva's theorem", "Lifting the exponents", "Triangle inequality").

Problem:
{problem}

Please provide exactly {num_hints} hints in the following format (one hint per line, numbered):
1. [Hint 1]
2. [Hint 2]
...
{num_hints}. [Hint {num_hints}]

Make sure each hint is a distinct mathematical concept or theorem."""

    try:
        response = client.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides mathematical hints. Provide concise, specific mathematical concepts or theorems as hints."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        response_text = response.choices[0].message.content
        hints = []
        lines = response_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and any(line.startswith(f"{i}.") for i in range(1, num_hints + 1)):
                hint = line.split('. ', 1)[1] if '. ' in line else line
                hints.append(hint.strip())
        if len(hints) < num_hints:
            print(f"Warning: Got {len(hints)} hints instead of {num_hints}")
        elif len(hints) > num_hints:
            hints = hints[:num_hints]
            
        return hints
        
    except Exception as e:
        print(f"Error generating hints: {e}")
        return []
