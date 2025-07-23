import json
import torch
import wandb
import random
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from argparse import ArgumentParser
from typing import List, Callable

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

from cs336_alignment.sft_helper import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
    log_generations,
)
from cs336_alignment.utils_gsm8k import (
    load_gsm8k_data,
    format_data,
    evaluate_vllm,
    compute_metrics,
    load_MATH,
    make_prompts,
)
from cs336_alignment.vllm_helper import init_vllm, load_policy_into_vllm_instance


def train_setup(model_string: str, 
                seed: int = 42, 
                vllm_device: str = 'cuda', 
                model_device: str = 'cuda',
                gpu_memory_utilization: float = 0.2) -> tuple[LLM, PreTrainedModel, PreTrainedTokenizer]:
    
    # initialize vllm onto 1st GPU
    print("Initializing vllm...")
    vllm_model = init_vllm(model_string, device = vllm_device, seed = seed, gpu_memory_utilization=gpu_memory_utilization)

    # load model and tokenizer onto 1st GPU
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_string,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map = model_device
    )

    # get tokenizer, optimizer
    tokenizer = AutoTokenizer.from_pretrained(model_string)

    # load policy into vllm
    print("Loading policy into vllm...")
    load_policy_into_vllm_instance(model, vllm_model)

    return vllm_model, model, tokenizer


def full_train_step(prompts: List[str], 
                    answers: List[str], 
                    tokenizer: PreTrainedTokenizer, 
                    model: PreTrainedModel, 
                    gradient_accumulation_steps: int, 
                    do_backward: bool = True, 
                    device: str = 'cuda:0') -> tuple[float, float]:
    """
    Do a full SFT training step from prompts/answers
    """
    # tokenize prompts and responses --> input_ids, labels, response_mask
    tokenized_results = tokenize_prompt_and_output(prompts, answers, tokenizer, device = device)
    # tokenized results: batch_size x max_length

    # get log probs and entropy
    lp_dict = get_response_log_probs(model, tokenized_results["input_ids"], tokenized_results["labels"], return_token_entropy = True)
    log_probs = lp_dict["log_probs"]
    token_entropy = lp_dict["token_entropy"]
    token_entropy *= tokenized_results["response_mask"] # mask out padding tokens
    response_lengths = torch.sum(tokenized_results["response_mask"], dim = -1, keepdim = True)
    token_entropy /= response_lengths # normalize over tokens per response
    per_token_entropy = token_entropy.mean() # average over responses

    # compute loss and do backward pass
    loss, metadata = sft_microbatch_train_step(policy_log_probs = log_probs,
                                                        response_mask = tokenized_results["response_mask"],
                                                        gradient_accumulation_steps = gradient_accumulation_steps,
                                                        normalize_constant = 1.0,
                                                        do_backward = do_backward)
    
    return loss.item(), per_token_entropy.item()


def evaluate_loss(prompts: List[str], answers: List[str], tokenizer: PreTrainedTokenizer, model: PreTrainedModel, minibatch_size: int):
    """
    Evaluate the model on a set of prompts/answers
    """
    
    # set model to eval mode
    model.eval()
    avg_loss = 0
    avg_entropy = 0
    n_minibatches = len(prompts) // minibatch_size

    with torch.no_grad():
        for i in range(0, len(prompts), minibatch_size):
            minibatch_prompts = prompts[i:i+minibatch_size]
            minibatch_answers = answers[i:i+minibatch_size]
            
            loss, entropy = full_train_step(minibatch_prompts, 
                            minibatch_answers, 
                            tokenizer, model, 
                            gradient_accumulation_steps = 1, 
                            do_backward = False)
            avg_loss += loss.item()
            avg_entropy += entropy
    
    model.train()
    
    # average loss and entropy over minibatches
    avg_loss /= n_minibatches
    avg_entropy /= n_minibatches

    return avg_loss, avg_entropy


def evaluate(prompts: List[str], 
             answers: List[str], 
             vllm_model: LLM, 
             eval_sampling_params: SamplingParams, 
             reward_fn: Callable[[str, str], dict[str, float]]):
    """
    Evaluate the model on a set of prompts/answers
    """
    results = evaluate_vllm(vllm_model, 
                           reward_fn, 
                           prompts, 
                           answers, 
                           eval_sampling_params)

    # compute metrics
    correct_fraction, format_only_fraction, wrong_fraction = compute_metrics(results, printout = False)

    return correct_fraction, format_only_fraction


def load_SFT(path: str):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    return [d["prompt"] for d in data], [d["response"] for d in data], [d["ground_truth"] for d in data]


def train_run(config: dict,
              eval_sampling_params: SamplingParams = None,
              end_eval: bool = True):
    
    vllm_device = 'cuda:0'
    model_device = 'cuda:0'

    # do training setup
    vllm_model, model, tokenizer = train_setup(config['model'], config['seed'], vllm_device, model_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = config['learning_rate'])

    # load training and validation sets
    print("Loading training and validation sets...")
    train_prompts, train_responses, train_ground_truths = load_SFT(config['train_path'])
    print(f"Loaded {len(train_prompts)} training examples")

    # sample subset of unique prompts to use
    if config['n_unique']: # use random sample of unique prompts
        train_idxs = random.sample(range(len(train_prompts)), config['n_unique'])
        train_prompts = [train_prompts[i] for i in train_idxs]
        train_responses = [train_responses[i] for i in train_idxs]
        train_ground_truths = [train_ground_truths[i] for i in train_idxs]
    
    val_questions, val_answers = load_MATH(config['val_path'])
    val_prompts = make_prompts(val_questions)

    # run SFT training
    train_sft(train_prompts = train_prompts, 
            train_responses = train_responses, 
            train_ground_truths = train_ground_truths, 
            val_prompts = val_prompts, 
            val_answers = val_answers, 
            vllm_model = vllm_model, 
            model = model, 
            model_device = model_device, 
            optimizer = optimizer, 
            tokenizer = tokenizer, 
            eval_sampling_params = eval_sampling_params, 
            config = config,
            start_train_step = config['start_train_step'],
            end_eval = end_eval)


def wandb_setup():
    wandb.define_metric("train_step")  # x-axis for training and eval
    wandb.define_metric("iter")        # x-axis for iteration-level metrics

    # Both train/ and eval/ metrics use train_step as x-axis
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="train_step")  # Note: uses train_step, not eval_step

    # iter/ metrics use iter as x-axis
    wandb.define_metric("iter/*", step_metric="iter")


def train_sft(train_prompts: List[str], 
            train_responses: List[str], 
            train_ground_truths: List[str], 
            val_prompts: List[str], 
            val_answers: List[str], 
            vllm_model: LLM, 
            model: PreTrainedModel, 
            model_device: str,
            optimizer: torch.optim.Optimizer, 
            tokenizer: PreTrainedTokenizer, 
            eval_sampling_params: SamplingParams,
            config: dict,
            start_train_step: int = 0,
            end_eval: bool = True):
    
    minibatch_size = config['minibatch_size']
    batch_size = config['train_batch_size']
    n_epochs = config['n_epochs']
    log_every_n = config['log_every_n']
    eval_every_n = config['eval_every_n']
    n_unique = config['n_unique']
    learning_rate = config['learning_rate']
    
    print("Running SFT training...")
    gradient_accumulation_steps = batch_size // minibatch_size
    n_minibatches = len(train_prompts) // config['minibatch_size']
    train_step = start_train_step
    mini_train_step = 0
    log_train = True 
    log_eval = True
    
    print(f"Training for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        # shuffle train indices before each epoch
        train_indices = list(range(len(train_prompts)))
        random.shuffle(train_indices)
        train_prompts = [train_prompts[i] for i in train_indices]
        train_responses = [train_responses[i] for i in train_indices]
        train_ground_truths = [train_ground_truths[i] for i in train_indices]
        
        # run training steps
        n_minibatches = len(train_prompts) // minibatch_size
        print(f'Training on {len(train_indices)} examples in {n_minibatches} minibatches')
        
        for minibatch_idx in range(n_minibatches):
            minibatch_prompts = train_prompts[minibatch_idx * minibatch_size:(minibatch_idx + 1) * minibatch_size]
            minibatch_responses = train_responses[minibatch_idx * minibatch_size:(minibatch_idx + 1) * minibatch_size]

            loss_val, avg_entropy = full_train_step(minibatch_prompts, 
                            minibatch_responses, 
                            tokenizer, model, 
                            gradient_accumulation_steps, 
                            do_backward = True,
                            device = model_device)
            
            # backwards pass
            if (mini_train_step + 1) % gradient_accumulation_steps == 0:
                # do gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # perform gradient descent once accumulated
                optimizer.step()
                optimizer.zero_grad()
                train_step += 1

                print("Train step: ", train_step)
                if train_step % log_every_n == 0:
                    log_train = True
                if train_step % eval_every_n == 0:
                    log_eval = True
            
            # logging
            if log_train:
                log_train = False
                wandb.log({
                    "train/loss": loss_val,
                    "train/avg_entropy": avg_entropy,
                    "train_step": train_step,
                })

            if log_eval:
                multiplier = 2
                log_eval = False
                print("Train step: ", train_step)
                print(f"Evaluating on {batch_size * multiplier} prompts...")

                # load policy into vllm
                print("Loading policy into vllm...")
                load_policy_into_vllm_instance(model, vllm_model)

                # select random double-batch of eval prompts/answers
                val_batch_indices = random.sample(range(len(val_prompts)), batch_size * multiplier)
                val_batch_prompts = [val_prompts[i] for i in val_batch_indices] 
                val_batch_answers = [val_answers[i] for i in val_batch_indices]

                correct_fraction, format_fraction = evaluate(prompts = val_batch_prompts, 
                                                                   answers = val_batch_answers, 
                                                                   vllm_model = vllm_model, 
                                                                   eval_sampling_params = eval_sampling_params, 
                                                                   reward_fn = r1_zero_reward_fn)
                print('Logging metrics to wandb...')
                wandb.log({
                    "eval/correct": correct_fraction,
                    "eval/format": format_fraction,
                    "train_step": train_step,
                })

                log_indices = random.sample(range(len(val_prompts)), minibatch_size)
                log_prompts = [val_prompts[i] for i in log_indices]
                log_answers = [val_answers[i] for i in log_indices]
                print(f"Logging generations for {len(log_prompts)} prompts...")
                log_generations(vllm_model = vllm_model, 
                                        reward_fn = r1_zero_reward_fn, 
                                        prompts = log_prompts, 
                                        answers = log_answers, 
                                        sampling_params = eval_sampling_params,
                                        log_file = f'sft_results/sft_{n_unique}_{batch_size}_{minibatch_size}_{learning_rate}.txt')
                
            mini_train_step += 1
    
    partial_step = (mini_train_step % gradient_accumulation_steps)
    print(f"Partial gradient accumulation steps: {partial_step}")
    if partial_step > (gradient_accumulation_steps // 2):
        print("Performing partial gradient update...")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        train_step += 1
    else:
        print("Not performing gradient update...")
        # erase gradients
        optimizer.zero_grad()

    print("Training complete!")
    load_policy_into_vllm_instance(model, vllm_model) # load policy into vllm

    # do full val set evaluation
    if end_eval:
        print("Evaluating on full validation set...")
        correct_fraction, format_fraction = evaluate(prompts = val_prompts, 
                                                                answers = val_answers, 
                                                                vllm_model = vllm_model, 
                                                                eval_sampling_params = eval_sampling_params, 
                                                                reward_fn = r1_zero_reward_fn)
        print('Logging metrics to wandb...')
        wandb.log({
            "eval/correct": correct_fraction,
            "eval/format": format_fraction,
        }, step = train_step)

    return train_step


def main(args):
    # Setup
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)
    vllm_set_random_seed(SEED)
    
    wandb_setup()

    # Config
    config = {
        "model": '/kun-data/assignment5-alignment/models/Qwen/Qwen2.5-Math-1.5B',
        "seed": SEED,
        "learning_rate": 5e-5,
        "train_path": "./data/gsm8k/train.jsonl",
        "val_path": "./data/gsm8k/test.jsonl",
        "n_unique": args.num_train_sample,
        "minibatch_size": 1,
        "train_batch_size": 8,
        "n_epochs": 1,
        "log_every_n": 1,
        "eval_every_n": 10,
        "start_train_step": 0,
    }
    
    vllm_device = 'cuda:0'
    model_device = 'cuda:0'

    # do training setup
    vllm_model, model, tokenizer = train_setup(config['model'], config['seed'], vllm_device, model_device, gpu_memory_utilization=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr = config['learning_rate'])

    # load training and validation sets
    print("Loading training and validation sets...")
    train_data = load_gsm8k_data(config['train_path'])
    if config['n_unique'] is not None:
        train_data = train_data[:config['n_unique']]
    
    val_data = load_gsm8k_data(config['val_path'])

    # format data
    with open("cs336_alignment/prompts/r1_zero.prompt", "r", encoding="utf-8") as f:
        prompt_template = f.read()
    format_prompt_fn = lambda q: prompt_template.format(question=q)

    formatted_train_data = format_data(train_data, format_prompt_fn)
    formatted_val_data = format_data(val_data, format_prompt_fn)

    train_prompts = [d['prompt'] for d in formatted_train_data]
    train_responses = [d['response'] for d in formatted_train_data]
    # For GSM8K, the response is the ground truth
    train_ground_truths = train_responses

    val_prompts = [d['prompt'] for d in formatted_val_data]
    val_answers = [d['response'] for d in formatted_val_data]
    
    print(f"Loaded {len(train_prompts)} training examples")
    print(f"Loaded {len(val_prompts)} validation examples")

    eval_sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    # run SFT training
    train_sft(train_prompts = train_prompts, 
            train_responses = train_responses, 
            train_ground_truths = train_ground_truths, 
            val_prompts = val_prompts, 
            val_answers = val_answers, 
            vllm_model = vllm_model, 
            model = model, 
            model_device = model_device, 
            optimizer = optimizer, 
            tokenizer = tokenizer, 
            eval_sampling_params = eval_sampling_params, 
            config = config,
            start_train_step = config['start_train_step'],
            end_eval = True)


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

    main(args)
