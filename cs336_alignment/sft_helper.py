import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import List, Dict

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
    prompt_tokens = tokenizer(prompt_strs)['input_ids']
    output_tokens = tokenizer(output_strs)['input_ids']

    batch_sz = len(prompt_tokens)
    
    # find padding length
    prompt_and_output_lens = [len(p) + len(o) for p, o in zip(prompt_tokens, output_tokens)]
    padded_len = max(prompt_and_output_lens)

    # Initialize tensors for return
    input_ids = torch.empty((batch_sz, padded_len - 1), dtype=torch.long)
    labels = torch.empty((batch_sz, padded_len - 1), dtype=torch.long)
    response_mask = torch.zeros((batch_sz, padded_len - 1), dtype=torch.bool)
    
    # Build the tensors sample by sample
    for i, (p_toks, o_toks) in enumerate(zip(prompt_tokens, output_tokens)):
        # Concatenate prompt + output and convert to a torch tensor
        p_o_concat = torch.tensor(p_toks + o_toks)
        concat_len = len(p_o_concat)

        # Pad concatentaed sequence on the right with EOS token
        # Such that every sequence has the same length 'padded_len'
        p_o_concat_padded = F.pad(p_o_concat, 
                                    (0, padded_len - concat_len), 
                                    'constant', 
                                    tokenizer.eos_token_id)

        # Fill input_ids and labels with 1-token shift (standard LM objective)
        input_ids[i] = p_o_concat_padded[:-1] # exclude last token
        labels[i] = p_o_concat_padded[1:] # exclude first token

        # Mark which *label* position corresponds to the assistant's output
        #     Because of the 1-token shift, the first output token is at index len(prompt)-1
        o_start = len(p_toks) - 1 # inclusive
        o_end = concat_len - 1 # exclusive (lavels are shifted)
        response_mask[i, o_start:o_end] = True

    # Return three tensor in a dictionary
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
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
        result["entropy"] = compute_entropy(logits)
    
    return result
    