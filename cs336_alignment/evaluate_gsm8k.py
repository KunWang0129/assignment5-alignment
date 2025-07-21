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

def gsm8k_reward_fn(ground_truth: str, generated_text: str) -> Dict[str, float]:
    """
    Calculates the reward for a gsm8k prediction.
    The reward is 1.0 if the prediction matches the ground truth, and 0.0 otherwise.
    """
    ground_truth_answer = get_gsm8k_answer(ground_truth)
    
    # Extract the last number from the generated text
    numbers = re.findall(r'\d+', generated_text)
    if numbers:
        predicted_answer = numbers[-1]
    else:
        predicted_answer = ""

    is_correct = 1.0 if predicted_answer == ground_truth_answer else 0.0
    return {'accuracy': is_correct}


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], Dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
    output_path: str
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    print(f"Generating responses for {len(prompts)} prompts...")
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    results = []
    total_accuracy = 0.0
    
    print(f"Evaluating {len(outputs)} generated responses...")
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        ground_truth = ground_truths[i]
        
        scores = reward_fn(ground_truth, generated_text)
        total_accuracy += scores.get('accuracy', 0.0)
        
        result = {
            'prompt': prompt,
            'ground_truth': ground_truth,
            'generated_text': generated_text,
            'scores': scores
        }
        results.append(result)

    avg_accuracy = total_accuracy / len(outputs) if outputs else 0.0
    print(f"Average Accuracy: {avg_accuracy:.4f}")

    print(f"Saving results to {output_path}...")
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    print("Done.")


def main():
    """Main function to run the evaluation."""
    print("Loading model...")
    # It may not be able to run on the current environment
    # llm = LLM(model=MODEL_PATH)
    llm = LLM(model="Qwen/Qwen2-1.5B-Instruct")


    print("Loading data...")
    gsm8k_data = load_gsm8k_data(DATA_PATH)
    
    prompts = [format_prompt_r1_zero(ex['question']) for ex in gsm8k_data]
    ground_truths = [ex['answer'] for ex in gsm8k_data]
    
    sampling_params = SamplingParams(
        temperature=0.0, # Set to 0 for deterministic output
        max_tokens=256,
    )
    
    evaluate_vllm(
        vllm_model=llm,
        reward_fn=gsm8k_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        output_path=OUTPUT_PATH
    )

if __name__ == "__main__":
    main()
