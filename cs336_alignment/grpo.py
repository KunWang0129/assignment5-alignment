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

with open('cs336_alignment/prompts/r1_zero.prompt', 'r') as f:
    R1_ZERO_PROMPT = f.read()


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
    args = parser.parse_args()


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
