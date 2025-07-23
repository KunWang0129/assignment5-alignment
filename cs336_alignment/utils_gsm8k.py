import json
import os
import re
from typing import Callable, List, Dict

from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


# Constants
MODEL_PATH = '/kun-data/assignment5-alignment/models/Qwen/Qwen2.5-Math-1.5B'
DATA_PATH = '/kun-data/assignment5-alignment/data/gsm8k/test.jsonl'
OUTPUT_DIR = '/kun-data/assignment5-alignment/eval_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'gsm8k_qwen_zeroshot_results.jsonl')

def load_jsonl(file_path):
    """Loads gsm8k data from a jsonl file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def format_prompt_with_template(question: str, template_path: str) -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    return template.format(question=question)

def get_gsm8k_answer(answer_str: str) -> str:
    """Extracts the final numerical answer from the gsm8k answer string."""
    return answer_str.split('####')[-1].strip()

def format_data(
    data: List[Dict[str, str]],
    prompt_fn: Callable[[str], str],
) -> List[Dict[str, str]]:
    """
    Formats the data using the provided prompt function.
    
    Args:
        data: A list of dictionaries, each with 'question' and 'answer' keys.
        prompt_fn: A function that takes a question string and returns a formatted prompt string.
        
    Returns:
        A list of dictionaries, each with 'prompt' and 'response' keys.
    """
    formatted_data = []
    for ex in data:
        formatted_data.append({
            'prompt': prompt_fn(ex['question']),
            'response': ex['answer'],
        })
    return formatted_data

def format_for_sft(file_path):
    data = load_jsonl(file_path)
    formatted_data = []

    for ex in data:
        question = ex['question']
        answer = get_gsm8k_answer(ex['answer'])
        formatted_data.append({
            'prompt': f"Question: {question}\nAnswer: ",
            'response': answer
        })

def main():
    """Main function to run the evaluation."""
    print("Loading model...")
    # It may not be able to run on the current environment
    # llm = LLM(model=MODEL_PATH)
    llm = LLM(model="Qwen/Qwen2-1.5B-Instruct")


    print("Loading data...")
    gsm8k_data = load_gsm8k_data(DATA_PATH)
    
    prompts = [format_prompt_r1_zero(ex['question']) for ex in gsm8k_data]
    
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
