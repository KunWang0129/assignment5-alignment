import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import List, Dict, Any, Callable

def tokenize_prompt_and_output(
    prompt_strs: List[str],
    output_strs: List[str],
    tokenizer: PreTrainedTokenizer,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is True for the response tokens and False for
    other tokens (prompt or padding).
    Args:
        prompt_strs: list[str] List of prompt strings.
        output_strs: list[str] List of output strings.
        tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.
    Returns:
        dict[str, torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of
        the tokenized prompt and output strings. Then the returned dictionary should have the
        following keys:
        input_ids torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
        the tokenized prompt and output strings, with the final token sliced off.
        labels torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
        shifted input ids, i.e., the input ids without the first token.
        response_mask torch.Tensor of shape (batch_size, max(prompt_and_output_lens) -
        1): a mask on the response tokens in the labels.
    """
    
    # Tokenize the prompts and outputs
    prompt_input_ids = []
    output_input_ids = []

    for prompt in prompt_strs:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_input_ids.append(torch.tensor(tokens))

    for output in output_strs:
        tokens = tokenizer.encode(output, add_special_tokens=False)
        output_input_ids.append(torch.tensor(tokens))

    # max len
    seq_lengths = [len(p_ids) + len(o_ids) for p_ids, o_ids in zip(prompt_input_ids, output_input_ids)]
    max_length = max(seq_lengths)
    
    concatenated_input_ids = []
    concatenated_labels = []
    response_masks = []

    for p_ids, o_ids in zip(prompt_input_ids, output_input_ids):
        # concat
        input_ids = torch.cat([p_ids, o_ids], dim=0)
        response_mask = torch.cat([
            torch.zeros_like(p_ids, dtype=torch.bool),  # False for prompt
            torch.ones_like(o_ids, dtype=torch.bool)     # True for output
        ], dim=0)
        # and then pad
        pad_length = max_length - input_ids.shape[0]
        padded_input_ids = torch.nn.functional.pad(input_ids, (0, pad_length), value=tokenizer.pad_token_id)
        padded_response_mask = torch.nn.functional.pad(response_mask, (0, pad_length), value=False)

        concatenated_input_ids.append(padded_input_ids[:-1])
        concatenated_labels.append(padded_input_ids[1:])
        response_masks.append(padded_response_mask[1:])

    input_ids_tensor = torch.stack(concatenated_input_ids)
    labels_tensor = torch.stack(concatenated_labels)
    response_mask_tensor = torch.stack(response_masks)

    return {
        "input_ids": input_ids_tensor,
        "labels": labels_tensor,
        "response_mask": response_mask_tensor,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Shannon entropy H(p) = -Σ p * log p  
    This implementation works on raw logits without converting them to explicit
    probabilities first and is fully vectorised.    
    
    Args:
        logits: torch.Tensor of shape (batch_size, seq_length, vocab_size) containing the logits.
    Returns:
        torch.Tensor of shape (batch_size, seq_length) containing the entropy for each next token.
    """

    # exp keps numberical stability
    p_numerator = torch.exp(logits)

    # Z = Σ_j e^{logit_j}, sum over vocab size j
    # shape: (batch_size, seq_length, 1)
    p_denom = torch.sum(p_numerator, dim=-1, keepdim=True)

    # log p_j = logit_j − log Σ_k e^{logit_k} 
    log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    # p * log p  (component-wise contribution to entropy)
    summands = (p_numerator / p_denom) * log_prob

    # H(p) = −Σ_j p_j log p_j
    return -torch.sum(summands, dim=-1)

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
    """
    Get the log probabilities of the labels given the input_ids.
    Args:
        model: PreTrainedModel The model to use for computing the logits.
        input_ids: torch.Tensor of shape (batch_size, seq_length) containing the input ids.
        labels: torch.Tensor of shape (batch_size, seq_length) containing the labels.
        return_token_entropy: bool Whether to return the entropy of the logits.
    Returns:
        dict[str, torch.Tensor]: A dictionary containing:
    """

    logits = model(input_ids).logits

    # Get the log probabilities of the labels
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    # Setup return dictionary
    result = {
        "log_probs": log_probs,
    }

    if return_token_entropy:
        # Compute the entropy of the logits
        result["token_entropy"] = compute_entropy(logits)
    
    return result
    

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None= None,
    ) -> torch.Tensor:
    """
    Sum over tensor elements and normaizes by a constant respected to a boolean mask.

    Args:
        tensor: torch.Tensor The tensor to sum over.
        mask: torch.Tensor A boolean mask of the same shape as tensor.
        normalize_constant: float The constant to normalize by.
        dim: int | None The dimension to sum over. If None, sum over all dimensions
    Returns:
        torch.Tensor: The normalized tensor.
    """

    tensor_sum = torch.sum(tensor * mask, dim=dim)
    return tensor_sum / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Perform a single training step on a microbatch of data.
    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, seq_length) containing the log probabilities of the policy.
        response_mask: torch.Tensor of shape (batch_size, seq_length) containing the response mask.
        gradient_accumulation_steps: int The number of gradient accumulation steps.
        normalize_constant: float The constant to normalize by.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: A tuple containing:
            - loss: torch.Tensor The computed loss.
            - loss_metadata: dict[str, torch.Tensor] Metadata about the loss, including gradient accumulation steps and normalize constant.
    """

    # Compute the loss
    loss = (-masked_normalize(policy_log_probs, response_mask, normalize_constant, -1)).mean()
    loss /= gradient_accumulation_steps

    loss.backward()

    loss_metadata = {
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'normalize_constant': normalize_constant
    }

    return (loss, loss_metadata)


def log_generations(
    prompts: List[str],
    generated_responses: List[str],
    ground_truth_responses: List[str],
    reward_infos: List[Dict[str, Any]],
    avg_token_entropies: List[float],
    response_lengths: List[int],
) -> Dict[str, Any]:
    """
    Logs pre-generated responses and computes aggregate statistics.

    This function is designed to be called within a training or generation loop
    with pre-computed data.

    Args:
        prompts: A list of prompts.
        generated_responses: A list of generated responses.
        ground_truth_responses: A list of ground-truth responses.
        reward_infos: A list of dictionaries containing reward information for each example.
        avg_token_entropies: A list of average token entropies for each generated response.
        response_lengths: A list of lengths for each generated response.

    Returns:
        A dictionary containing per-example logs and aggregate statistics.
    """
    per_example_logs = []
    total_response_len = 0
    correct_response_len = 0
    incorrect_response_len = 0
    num_correct = 0

    for i in range(len(prompts)):
        reward_info = reward_infos[i]
        response_len = response_lengths[i]

        per_example_logs.append({
            "prompt": prompts[i],
            "generated_response": generated_responses[i],
            "ground_truth_response": ground_truth_responses[i],
            "reward_info": reward_info,
            "avg_token_entropy": avg_token_entropies[i],
            "response_length": response_len,
        })

        total_response_len += response_len
        # Assuming 'answer_reward' of 1 means correct
        if reward_info.get("answer_reward", 0) == 1:
            num_correct += 1
            correct_response_len += response_len
        else:
            incorrect_response_len += response_len

    num_incorrect = len(prompts) - num_correct
    avg_response_len = total_response_len / len(prompts) if prompts else 0
    avg_correct_response_len = correct_response_len / num_correct if num_correct > 0 else 0
    avg_incorrect_response_len = incorrect_response_len / num_incorrect if num_incorrect > 0 else 0

    aggregate_stats = {
        "avg_response_length": avg_response_len,
        "avg_correct_response_length": avg_correct_response_len,
        "avg_incorrect_response_length": avg_incorrect_response_len,
        "num_correct": num_correct,
        "num_incorrect": num_incorrect,
        "total_samples": len(prompts),
    }

    return {
        "per_example_logs": per_example_logs,
        "aggregate_stats": aggregate_stats,
    }