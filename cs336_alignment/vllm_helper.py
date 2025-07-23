from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from unittest.mock import patch
import torch
from typing import Callable, List, Dict


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

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85, debug=False):
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