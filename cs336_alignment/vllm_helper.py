from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from unittest.mock import patch
import torch
import re
import os
import json
from typing import Callable, List, Dict

ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")

MODEL_PATH = '/kun-data/assignment5-alignment/models/Qwen/Qwen2.5-Math-1.5B'

def init_policy(debug=False):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model.to('cuda:0')

    return model, tokenizer

def init_vllm(model_id: str, 
              device: str, 
              seed: int, 
              gpu_memory_utilization: float = 0.85, 
              debug=False
              ):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)

    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    
def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def extract_reference_answer(answer: str) -> str:
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip().replace(",", "")
    return "[invalid]"

def evaluate_vllm(
    data: List[Dict],
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams
) -> None:
    # 3. Generate outputs for each example
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    generated_texts = [output.outputs[0].text.strip() for output in outputs]

    # 4. Calculate evaluation metrics and collect examples
    results = []
    counts = {
        "correct": 0,
        "wrong_answer": 0,
        "wrong_format": 0
    }

    format_error_examples = []
    answer_error_examples = []

    for prompt, generated_text, example in zip(prompts, generated_texts, data):
        ground_truth = example["answer"]
        reference_answer = extract_reference_answer(ground_truth)
        metrics = reward_fn(generated_text, reference_answer)

        if metrics["format_reward"] == 1 and metrics["answer_reward"] == 1:
            counts["correct"] += 1
        elif metrics["format_reward"] == 1 and metrics["answer_reward"] == 0:
            counts["wrong_answer"] += 1
            answer_error_examples.append({
                "prompt": prompt,
                "response": generated_text,
                "reference_answer": ground_truth,
                "reference_answer_extracted": reference_answer
            })
        else:
            counts["wrong_format"] += 1
            format_error_examples.append({
                "prompt": prompt,
                "response": generated_text,
                "reference_answer": ground_truth,
                "reference_answer_extracted": reference_answer
            })

        results.append({
            "prompt": prompt,
            "response": generated_text,
            "reference_answer": ground_truth,
            "reference_answer_extracted": reference_answer,
            "metrics": metrics
        })

    # 5. Save results
    os.makedirs("outputs", exist_ok=True)
    OUTPUT_PATH = os.path.join("outputs", "eval_results.jsonl")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Evaluation results saved to {OUTPUT_PATH}")

    return counts, format_error_examples, answer_error_examples