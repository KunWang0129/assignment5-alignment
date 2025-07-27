import json
import os
import re
from typing import Callable, List, Dict

from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.vllm_helper import evaluate_vllm


# Constants
MODEL_PATH = '/kun-data/assignment5-alignment/models/Qwen/Qwen2.5-Math-1.5B'
DATA_PATH = '/kun-data/assignment5-alignment/data/gsm8k/test.jsonl'
OUTPUT_DIR = '/kun-data/assignment5-alignment/eval_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'gsm8k_qwen_zeroshot_results.jsonl')
with open('cs336_alignment/prompts/r1_zero.prompt', 'r') as f:
    R1_ZERO_PROMPT = f.read()


def load_jsonl(file_path):
    """Loads gsm8k data from a jsonl file."""
    data = []
    with open(file_path, "r") as f:
        prompt_data = [json.loads(json_line) for json_line in f]
    return prompt_data

def format_prompt_with_template(question: str, template_path: str) -> str:
    with open(template_path, "r") as f:
        template = f.read()
    return template.format(question=question)

def format_for_training(file_path: str):
    data = load_jsonl(file_path)
    formatted_data = []

    for ex in data:
        prompt_string = R1_ZERO_PROMPT.format(
            question=ex['question'],
        )
        answer = ex['answer']
        formatted_data.append({
            'prompt': prompt_string,
            'response': answer,
        })
    return formatted_data

def main():
    """Main function to run the evaluation."""
    print("Loading model...")
    # It may not be able to run on the current environment
    # llm = LLM(model=MODEL_PATH)

    llm = LLM(model="Qwen/Qwen2-1.5B-Instruct")
    TEMPLATE_PATH = "cs336_alignment/prompts/r1_zero.prompt" # for testing


    print("Loading data...")
    gsm8k_data = load_jsonl(DATA_PATH)
    validation_data = format_for_training(DATA_PATH)
    prompts = [ex['prompt'] for ex in validation_data]
    
    sampling_params = SamplingParams(
        temperature=0.0, # Set to 0 for deterministic output
        max_tokens=256,
    )
    
    counts, format_errors, answer_errors = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        data=gsm8k_data,
        prompts=prompts,
        eval_sampling_params=sampling_params,
    )

    print("\n--- Evaluation Summary ---")
    print(f"Correct: {counts['correct']}")
    print(f"Wrong Answer: {counts['wrong_answer']}")
    print(f"Wrong Format: {counts['wrong_format']}")

    output_path = os.path.join(OUTPUT_DIR, 'gsm8k_eval_results.jsonl')
    print(f"\nSaving detailed results to {output_path}...")
    with open(output_path, 'w') as f:
        for error in format_errors:
            f.write(json.dumps(error) + '\n')
        for error in answer_errors:
            f.write(json.dumps(error) + '\n')
    print("Done.")

if __name__ == "__main__":
    main()