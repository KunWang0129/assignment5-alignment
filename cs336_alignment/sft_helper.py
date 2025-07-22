import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
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