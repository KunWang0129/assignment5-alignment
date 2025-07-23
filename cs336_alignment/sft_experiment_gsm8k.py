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
from cs336_alignment.evaluate_gsm8k import evaluate_vllm
from cs336_alignment.vllm_helper import init_vllm, load_policy_into_vllm_instance


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def format_prompt_with_template(question: str, template_path: str) -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    return template.format(question=question)


def get_batch(
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


def test(model, vllm, test_data, formatted_test_prompts, i_sft_step, output_dir, tokenizer):
    load_policy_into_vllm_instance(model, vllm)

    sampling_params =  SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )
    counts, format_errors, answer_errors = evaluate_vllm(
        vllm_model=vllm,
        reward_fn=r1_zero_reward_fn,
        data=test_data,
        prompts=formatted_test_prompts,
        eval_sampling_params=sampling_params
    )

    accuracy = counts['correct'] / len(formatted_test_prompts)
    print(f"\n📊 Evaluation Summary at Step {i_sft_step}:")
    print(f"Correct (format + answer): {counts['correct']}")
    print(f"Wrong answer (but correct format): {counts['wrong_answer']}")
    print(f"Wrong format: {counts['wrong_format']}")
    print(f"Accuracy: {accuracy}")
    wandb.log({
        "eval/correct": counts["correct"],
        "eval/wrong_answer": counts["wrong_answer"],
        "eval/wrong_format": counts["wrong_format"],
        "eval/accuracy": accuracy,
        "eval_step": i_sft_step,
    })

    model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)



def train_step(model, optimizer, amp_ctx, train_batch, n_grad_accum_steps, i_sft_step, j_grad_accum_step, device_train, tokenized_train_data, micro_batch_size):
    with amp_ctx:
        response_log_probs = get_response_log_probs(model, train_batch["input_ids"], train_batch["labels"], return_token_entropy=True)
        log_probs = response_log_probs["log_probs"]
        entropy = response_log_probs["token_entropy"]

        loss, _ = sft_microbatch_train_step(log_probs, train_batch["response_mask"], n_grad_accum_steps)
        loss.backward()

        if j_grad_accum_step == n_grad_accum_steps - 1:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"\n📊 Training Summary at Step {i_sft_step + 1}:")
            print(f"Loss: {loss.item():.6f}")
            print(f"Entropy: {entropy.mean().item():.6f}")
            wandb.log({
                "train/loss": to_float(loss),
                "train/entropy": to_float(entropy.mean()),
                "train_step": i_sft_step + 1,
            })
    
    return get_batch(tokenized_train_data, micro_batch_size, device_train)

def train(args, model, tokenizer, optimizer, tokenized_train_data, vllm, test_data, formatted_test_prompts):
    device_train = "cuda:2"
    micro_batch_size = 2
    n_sft_steps = 64
    n_grad_accum_steps = 32
    eval_steps = 8
    output_dir = "./outputs/sft"

    amp_ctx = torch.amp.autocast(
        device_type=device_train,
        dtype=torch.bfloat16,
    )

    train_batch = get_batch(tokenized_train_data, micro_batch_size, device_train)

    for i_sft_step in range(n_sft_steps):
        for j_grad_accum_step in range(n_grad_accum_steps):
            train_batch = train_step(model, optimizer, amp_ctx, train_batch, n_grad_accum_steps, i_sft_step, j_grad_accum_step, device_train, tokenized_train_data, micro_batch_size)

        if (i_sft_step + 1) % eval_steps == 0 or i_sft_step == 0:
            test(model, vllm, test_data, formatted_test_prompts, i_sft_step + 1, output_dir, tokenizer)
    
    return model, n_sft_steps, eval_steps



def main(args):
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)
    vllm_set_random_seed(SEED)

    num_train_sample = args.num_train_sample
    model_id = '/kun-data/assignment5-alignment/models/Qwen/Qwen2.5-Math-1.5B'
    device_train = "cuda:1"
    device_vllm = "cuda"
    output_dir = "./outputs/sft"

    train_file_path = "./data/gsm8k/train.jsonl"
    test_file_path = "./data/gsm8k/test.jsonl"
    TEMPLATE_PATH = "cs336_alignment/prompts/r1_zero.prompt"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_train,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    vllm = init_vllm(model_id, device_vllm, seed=SEED, gpu_memory_utilization=0.9)
    
    train_data = load_jsonl(train_file_path)
    if num_train_sample is not None:
        train_data = train_data[:num_train_sample]
    test_data = load_jsonl(test_file_path)

    tokenized_train_data = tokenize_prompt_and_output(
        [data["prompt"] for data in train_data],
        [data["response"] for data in train_data],
        tokenizer
    )
    
    formatted_test_prompts = [
        format_prompt_with_template(example["question"], TEMPLATE_PATH) for example in test_data
    ]

    model, n_sft_steps, eval_steps = train(args, model, tokenizer, optimizer, tokenized_train_data, vllm, test_data, formatted_test_prompts)
    
    if n_sft_steps > 0 and n_sft_steps % eval_steps != 0:
        test(model, vllm, test_data, formatted_test_prompts, n_sft_steps, output_dir, tokenizer)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_train_sample", type=int, default=None)
    parser.add_argument("--use_corrupted", action='store_true')
    args = parser.parse_args()

    wandb.init(
        entity="kunwang0129-test-ucsd",
        project="sft_experiment",
        config=args,
    )

    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")

    wandb.define_metric("train/*", step_metric="train_step")

    wandb.define_metric("eval/*", step_metric="eval_step")

    main(args)