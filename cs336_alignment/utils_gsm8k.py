import json
import os
import re
from typing import Callable, List, Dict

from vllm import LLM, SamplingParams

# Constants
MODEL_PATH = '/kun-data/assignment5-alignment/models/Qwen/Qwen2.5-Math-1.5B'
DATA_PATH = '/kun-data/assignment5-alignment/data/gsm8k/test.jsonl'
OUTPUT_DIR = '/kun-data/assignment5-alignment/eval_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'gsm8k_qwen_zeroshot_results.jsonl')

def load_gsm8k_data(path: str) -> List[Dict]:
    """Loads gsm8k data from a jsonl file."""
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def format_prompt_r1_zero(question: str) -> str:
    """Formats a question using the r1_zero prompt format."""
    return f"Question: {question}\nAnswer:"

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


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    data: List[Dict[str, str]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
) -> [Dict[str, int], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and return the results.
    """
    print(f"Generating responses for {len(prompts)} prompts...")
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    counts = {'correct': 0, 'wrong_answer': 0, 'wrong_format': 0}
    format_errors = []
    answer_errors = []

    print(f"Evaluating {len(outputs)} generated responses...")
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        ground_truth = data[i]['answer']
        
        reward_info = reward_fn(ground_truth, generated_text)
        
        if reward_info['correctness']:
            counts['correct'] += 1
        elif reward_info['format_is_correct']:
            counts['wrong_answer'] += 1
            answer_errors.append({
                'prompt': output.prompt,
                'generated_text': generated_text,
                'ground_truth': ground_truth,
                'parsed_answer': reward_info['parsed_answer'],
                'expected_answer': reward_info['expected_answer']
            })
        else:
            counts['wrong_format'] += 1
            format_errors.append({
                'prompt': output.prompt,
                'generated_text': generated_text,
                'ground_truth': ground_truth
            })

    return counts, format_errors, answer_errors


from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

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
