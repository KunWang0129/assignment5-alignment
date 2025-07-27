import torch
from vllm import LLM, SamplingParams
import json
import random
import wandb
import sys
import argparse
import yaml
import os
import datetime
from argparse import ArgumentParser

from cs336_alignment.vllm_helper import (
    init_policy,
    init_vllm, 
    load_policy_into_vllm_instance,
    extract_reference_answer,
    evaluate_vllm
)
from cs336_alignment.sft_helper import tokenize_prompt_and_output, get_response_log_probs
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.grpo_helper import (
    compute_group_normalized_rewards,
    compute_naive_policy_gradient_loss,
    compute_grpo_clip_loss,
    compute_policy_gradient_loss,
    masked_mean,
    grpo_microbatch_train_step,
)

from cs336_alignment.utils_gsm8k import (
    format_for_training,
)

with open('cs336_alignment/prompts/r1_zero.prompt', 'r') as f:
    R1_ZERO_PROMPT = f.read()

def get_data(file_path: str):
    return format_for_training(file_path)

def sample_dataset(dataset, num_samples):
    sampled_data = random.sample(dataset, num_samples)

    ret = {
        'prompts': [],
        'answers': [],
    }

    for d in sampled_data:
        ret['prompts'].append(d['prompt'])
        ret['answers'].append(d['response'])

    return ret

def duplicate_data(arr, group_size):
    '''
    Ex: duplicate_data([1, 2, 3], 2) => [1, 1, 2, 2, 3, 3]
    '''

    return [x for x in arr for _ in range(group_size)]

def main(args):

    model_id = '/kun-data/assignment5-alignment/models/Qwen/Qwen2.5-Math-1.5B'
    device_train = "cuda:1"
    device_vllm = "cuda:0"
    output_dir = "./outputs/grpo"
    train_file_path = "./data/gsm8k/train.jsonl"
    eval_file_path = "./data/gsm8k/test.jsonl"
    TEMPLATE_PATH = "cs336_alignment/prompts/r1_zero.prompt" # for testing
    seed = 69
    eval_log_freq = 10
    eval_sample_size = 1024
    use_std_normalization = True
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, f"model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(model_dir, exist_ok=True)

    # Initialize variables from args
    n_grpo_steps = args.n_grpo_steps
    learning_rate = args.learning_rate
    advandate_eps = args.advandate_eps
    rollout_batch_size = args.rollout_batch_size
    group_size = args.group_size
    sampling_temperature = args.sampling_temperature
    sampling_min_tokens = args.sampling_min_tokens
    sampling_max_tokens = args.sampling_max_tokens
    epochs_per_rollout_batch = args.epochs_per_rollout_batch
    train_batch_size = args.train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    loss_type = args.loss_type

    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    n_prompts_per_rollout_batch  = rollout_batch_size // group_size
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

    training_data = get_data(train_file_path)
    eval_data = get_data(eval_file_path)

    policy, tokenizer = init_policy(device_train)
    device = policy.device

    vllm = init_vllm(
        model_id=model_id,
        device=device_vllm,
        seed=seed,
        gpu_memory_utilization=0.9,
    )

    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        top_p = 1.0,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
        logprobs=0,
    )

    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    train_step = 0
    eval_step = 0

    for grpo_step_idx in range(n_grpo_steps):
        load_policy_into_vllm_instance(policy, vllm)

        # Evaluation
        if grpo_step_idx % eval_log_freq == eval_log_freq - 1:
       
            # Sample eval data
            sampled_eval_data = sample_dataset(eval_data, eval_sample_size)
            prompts_batch = sampled_eval_data["prompts"]
            answer_batch = sampled_eval_data['answers']

            # Generate rollouts
            vllm_rollouts = vllm.generate(
                prompts_batch,
                sampling_params,
                )
            
            rollout_input_text = []
            rollout_response_text = []

            # Parse rollouts
            for rollout in vllm_rollouts:
                for r in rollout.outputs:
                    rollout_input_text.append(rollout.prompt)
                    rollout_response_text.append(r.text)
            
            # Compute rewards
            _,_, reward_metadata = compute_group_normalized_rewards(
                reward_fn=r1_zero_reward_fn,
                rollout_responses=rollout_response_text,
                repeated_ground_truths=answer_batch,
                group_size=1,
                advantage_eps=advandate_eps,
                normalize_by_std=use_std_normalization,
            )

            ## Visual: print a randomly sampled eval response
            eval_rand_idx = random.randrange(eval_sample_size)
            print(f"Eval Step {eval_step}:")
            print(f"Prompt:\n{rollout_input_text[eval_rand_idx]}")
            print(f"Correct Answer:\n{answer_batch[eval_rand_idx]}")
            print(f"Generated Answer:\n{rollout_response_text[eval_rand_idx]}")

            wandb.log({
                "eval_step": eval_step,
                "eval/accuracy": reward_metadata['mean'],
            })

            # Save model
            curr_model_dir = os.path.join(model_dir, f"eval_step_{eval_step}")
            policy.save_pretrained(curr_model_dir)
            tokenizer.save_pretrained(curr_model_dir)
            eval_step += 1
        
        # Policy gradient step per train_batch_size of data
        for rollout_batch_idx in range(0, train_batch_size, rollout_batch_size):
            sampled_training_data = sample_dataset(training_data, n_prompts_per_rollout_batch)
            prompts_batch = sampled_training_data["prompts"]
            answer_batch = sampled_training_data['answers']

            prompts_batch = duplicate_data(prompts_batch, group_size)
            answer_batch = duplicate_data(answer_batch, group_size)

            vllm_rollouts = vllm.generate(
                prompts_batch,
                sampling_params,
            )

            rollout_input_text = []
            rollout_response_text = []

            for rollout in vllm_rollouts:
                for r in rollout.outputs:
                    rollout_input_text.append(rollout.prompt)
                    rollout_response_text.append(r.text)

            advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
                reward_fn=r1_zero_reward_fn,
                rollout_responses=rollout_response_text,
                repeated_ground_truths=answer_batch,
                group_size=group_size,
                advantage_eps=advandate_eps,
                normalize_by_std=use_std_normalization,
            )

            wandb.log({
                "train_step": train_step,
                "train/mean_reward": reward_metadata['mean'],
            })

            rollout_data_tokenized = tokenize_prompt_and_output(
                prompt_strs=rollout_input_text,
                output_strs=rollout_response_text,
                tokenizer=tokenizer,
            )

            for _ in range(epochs_per_rollout_batch):
                optimizer.zero_grad()

                rollout_batch_loss = 0

                for microbatch_idx in range(n_microbatches_per_rollout_batch):
                    microbatch_slice = slice(
                        microbatch_idx * micro_train_batch_size,
                        (microbatch_idx + 1) * micro_train_batch_size
                    )

                    microbatch_input_ids = rollout_data_tokenized['input_ids'][microbatch_slice].to(device)
                    microbatch_labels = rollout_data_tokenized['labels'][microbatch_slice].to(device)
                    microbatch_response_mask = rollout_data_tokenized['response_mask'][microbatch_slice].to(device)

                    microbatch_advantages = advantages[microbatch_slice].to(device)
                    microbatch_raw_rewards = raw_rewards[microbatch_slice].to(device)

                    policy_log_probs_dict = get_response_log_probs(
                        model=policy,
                        input_ids=microbatch_input_ids,
                        labels=microbatch_labels,
                        return_token_entropy=True,
                    )

                    policy_log_probs = policy_log_probs_dict['log_probs']
                    policy_token_entropy = policy_log_probs_dict['token_entropy']

                    old_log_probs = policy_log_probs # change this when doing off-policy updates

                    microbatch_advantages = microbatch_advantages.unsqueeze(-1)  # Shape (micro_train_batch_size, 1)

                    loss, loss_metadata = grpo_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=microbatch_response_mask,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        loss_type=loss_type,
                        raw_rewards=microbatch_raw_rewards,
                        advantages=microbatch_advantages,
                        old_log_probs=old_log_probs,
                        cliprange=1.0,
                    )

                    rollout_batch_loss += loss.item()

                optimizer.step()

                rollout_batch_loss /= n_microbatches_per_rollout_batch
                wandb.log({
                    "train_step": train_step,
                    "train/loss": loss,
                })
                train_step += 1
    wandb.finish()

    model_final_dir = os.path.join(model_dir, "final")
    policy.save_pretrained(model_final_dir)
    tokenizer.save_pretrained(model_final_dir)

    print(f"Training complete. Final model saved to {model_final_dir}")




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_grpo_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--advandate_eps", type=float, default=1e-6)
    parser.add_argument("--rollout_batch_size", type=int, default=256)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--sampling_min_tokens", type=int, default=4)
    parser.add_argument("--sampling_max_tokens", type=int, default=1024)
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128)
    parser.add_argument("--loss_type", type=str, default="reinforce_with_baseline")
    args = parser.parse_args()

    assert args.loss_type in [
    "no_baseline",
    "reinforce_with_baseline",
    "grpo_clip",
    ]



    # Initialize wandb
    wandb.init(
        entity="kunwang03",
        project="grpo_experiment",
        config=args,
    )

    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")

    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    main(args)
