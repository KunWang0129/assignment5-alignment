import re
import json
import torch
import wandb
import random
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from argparse import ArgumentParser

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

from cs336_alignment.sft_helper import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
)
from utils_gsm8k import (
    load_jsonl,
    format_prompt_with_template,
    format_data,
)

from cs336_alignment.vllm_helper import (
    init_vllm, 
    load_policy_into_vllm_instance,
    evaluate_vllm
)

from cs336_alignment.sft_experiment import sft

SEED = 69
torch.manual_seed(SEED)
random.seed(SEED)

ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")


def extract_reference_answer(answer: str) -> str:
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip().replace(",", "")
    return "[invalid]"

def get_batch(formatted_train_prompts: list[str], train_data, batch_size: int) -> list[str]:
    batch_indices = random.sample(
        range(len(formatted_train_prompts)), batch_size
    )
    return [formatted_train_prompts[i] for i in batch_indices], [train_data[i] for i in batch_indices]

def get_sft_batch(
    tokenized_train_data: dict[str, torch.Tensor], batch_size: int, device: str
) -> dict[str, torch.Tensor]:
    batch_indices = random.sample(
        range(len(tokenized_train_data["input_ids"])), batch_size
    )
    return {k: v[batch_indices].to(device) for k, v in tokenized_train_data.items()}

def to_float(val):
    if isinstance(val, torch.Tensor):
        return val.float().item()
    return float(val)

def main(args):
    global_step = 0

    EI_num_G = args.EI_num_G
    EI_batch_size = args.EI_batch_size

    model_id = '/kun-data/assignment5-alignment/models/Qwen/Qwen2.5-Math-1.5B'
    device_vllm = "cuda:1"
    device_SFT = "cuda:0"
    train_file_path = "./data/gsm8k/train.jsonl"
    test_file_path = "./data/gsm8k/test.jsonl"
    TEMPLATE_PATH = "cs336_alignment/prompts/r1_zero.prompt" # for train & test

    n_expert_iteration_steps = 5
    sampling_temperature = 1.0
    sampling_max_tokens = 1024
    sampling_min_tokens = 4

    # init policy model
    EI_vllm = init_vllm(model_id, device_vllm, seed=SEED, gpu_memory_utilization=0.9)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_SFT,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # init reward function: r1...
    # init task question D (load questions)
    train_data = load_jsonl(train_file_path)
    test_data = load_jsonl(test_file_path)

    formatted_train_prompts = [
        format_prompt_with_template(example["question"], TEMPLATE_PATH) for example in train_data
    ]
    formatted_test_prompts = [
        format_prompt_with_template(example["question"], TEMPLATE_PATH) for example in test_data
    ]

    # for step in 1 ... n_expert_iteration_steps
    # sample batch question Db
    
    for idx in range(n_expert_iteration_steps):
        ei_step = idx + 1
        print(f"Starting Expert Iteration {ei_step}")
        # old policy model <- policy model
        # sample G outputs with (old policy model, Db)
        formatted_train_prompts_batch, train_data_batch = get_batch(formatted_train_prompts, train_data, EI_batch_size)

        sampling_params = SamplingParams(
            temperature=sampling_temperature,
            top_p=1.0,
            max_tokens=sampling_max_tokens,
            min_tokens=sampling_min_tokens,
            stop=["</answer>"],
            include_stop_str_in_output=True,
            n=EI_num_G,
            seed=SEED,
        )

        outputs = EI_vllm.generate(formatted_train_prompts_batch, sampling_params)
        all_generated_texts = [
            [o.text.strip() for o in output.outputs]
            for output in outputs
        ]
        # compute rewards for each output (reward function)
        results_per_rollout = []
        format_error_examples = []
        answer_error_examples = []
        
        for prompt_idx, (prompt, generated_answers, example) in enumerate(zip(formatted_train_prompts_batch, all_generated_texts, train_data_batch)):
            ground_truth = example["answer"]
            reference_answer = extract_reference_answer(ground_truth)

            for rollout_idx, generated_text in enumerate(generated_answers):
                metrics = r1_zero_reward_fn(generated_text, reference_answer)
                is_correct = metrics["reward"] == 1.0
                is_format_wrong = metrics["format_reward"] == 0.0
                is_answer_wrong = metrics["answer_reward"] == 0.0 and metrics["format_reward"] == 1.0

                results_per_rollout.append({
                    "prompt_idx": prompt_idx,
                    "rollout_idx": rollout_idx,
                    "metrics": metrics,
                    "prompt": prompt,
                    "response": generated_text,
                    "is_correct": is_correct
                })

                if is_format_wrong:
                    format_error_examples.append({
                        "prompt": prompt,
                        "response": generated_text,
                        "expected": reference_answer
                    })
                elif is_answer_wrong:
                    answer_error_examples.append({
                        "prompt": prompt,
                        "response": generated_text,
                        "expected": reference_answer
                    })

        # Print some examples of format and answer errors
        print("\n=== Format Error Examples ===")
        for i, ex in enumerate(format_error_examples[:1]):
            print(f"{i+1}. Prompt: {ex['prompt']}")
            print(f"   Response: {ex['response']}")
            print(f"   Expected Answer: {ex['expected']}\n")

        print("\n=== Answer Error Examples ===")
        for i, ex in enumerate(answer_error_examples[:1]):
            print(f"{i+1}. Prompt: {ex['prompt']}")
            print(f"   Response: {ex['response']}")
            print(f"   Expected Answer: {ex['expected']}\n")
            
        # ✅ Sanity Check: Print how many total responses and how many are correct
        total_responses = len(results_per_rollout)
        correct_responses = sum(1 for item in results_per_rollout if item["is_correct"])
        print(f"\nSanity Check in the end of Expert Iteration Step {idx + 1}:")
        print(f"Total generated responses: {total_responses}")
        print(f"Correct responses: {correct_responses}")
        print(f"Accuracy so far: {correct_responses / total_responses * 100:.2f}%\n")
        # filter out wrong output -> Dsft
        sft_data = []
        for item in results_per_rollout:
            if item["is_correct"]:
                sft_data.append({
                    "prompt": item["prompt"],
                    "response": item["response"]
                })
        # print(sft_data)
        # policy model <- SFT(policy model, Dsft)
        model, global_step = sft(
            args,
            sft_data,
            model,
            tokenizer,
            optimizer,
            EI_vllm,
            test_data,
            formatted_test_prompts,
            device_sft=device_SFT,
            global_step=global_step,
        )
        load_policy_into_vllm_instance(model, EI_vllm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--EI_num_G", type=int, default=5)
    parser.add_argument("--SFT_num_epochs", type=int, default=1)
    # n_expert_iteration = 5
    parser.add_argument("--EI_batch_size", type=int, default=512)
    args = parser.parse_args()

    wandb.init(
        entity="kunwang03",
        project="expert_iteration",
        config=args,
    )

    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")

    wandb.define_metric("train/*", step_metric="train_step")

    wandb.define_metric("eval/*", step_metric="eval_step")

    main(args)