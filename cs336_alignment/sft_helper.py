"""
4.1 Using HuggingFace Models
Loading a HuggingFace model and tokenizer. To load a HuggingFace model and tokenizer from a
local dir (in bfloat16 and with FlashAttention-2 to save memory), you can use the following starter code:
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
"/data/a5-alignment/models/Qwen2.5-Math-1.5B",
torch_dtype=torch.bfloat16,
attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained("/data/a5-alignment/models/Qwen2.5-Math-1.5B")
Forward pass. After we’ve loaded the model, we can run a forward pass on a batch of input IDs and get
the logits (with the .logits) attribute of the output. Then, we can compute the loss between the model’s
predicted logits and the actual labels:
input_ids = train_batch["input_ids"].to(device)
labels = train_batch["labels"].to(device)
logits = model(input_ids).logits
loss = F.cross_entropy(..., ...)
7
Saving a trained model. To save the model to a directory after training is finished, you can use the
.save_pretrained() function, passing in the path to the desired output directory. Make sure to save under
/data/yourusername since they can be quite large. We recommend also saving the tokenizer as well (even if
you didn’t modify it), just so the model and tokenizer are self-contained and loadable from a single directory.
# Save the model weights
model.save_pretrained(save_directory=output_dir)
tokenizer.save_pretrained(save_directory=output_dir)
Gradient accumulation. Despite loading the model in bfloat16 and using FlashAttention-2, even an
80GB GPU does not have enough memory to support reasonable batch sizes. To use larger batch sizes,
we can use a technique called gradient accumulation. The basic idea behind gradient accumulation is that
rather than updating our model weights (i.e., taking an optimizer step) after every batch, we’ll accumulate
the gradients over several batches before taking a gradient step. Intuitively, if we had a larger GPU, we
should get the same results from computing the gradient on a batch of 32 examples all at once, vs. splitting
them up into 16 batches of 2 examples each and then averaging at the end.
Gradient accumulation is straightforward to implement in PyTorch. Recall that each weight tensor has
an attribute .grad that stores its gradient. Before we call loss.backward(), the .grad attribute is None.
After we call loss.backward(), the .grad attribute contains the gradient. Normally, we’d take an optimizer
step, and then zero the gradients with optimizer.zero_grad(), which resets the .grad field of the weight
tensors:
for inputs, labels in data_loader:
# Forward pass.
logits = model(inputs)
loss = loss_fn(logits, labels)
# Backward pass.
loss.backward()
# Update weights.
optimizer.step()
# Zero gradients in preparation for next iteration.
optimizer.zero_grad()
To implement gradient accumulation, we’ll just call the optimizer.step() and optimizer.zero_grad()
every k steps, where k is the number of gradient accumulation steps. We divide the loss by gradient_ ⌋
accumulation_steps before calling loss.backward() so that the gradients are averaged across the gradient
accumulation steps.
gradient_accumulation_steps = 4
for idx, (inputs, labels) in enumerate(data_loader):
# Forward pass.
logits = model(inputs)
loss = loss_fn(logits, labels) / gradient_accumulation_steps
# Backward pass.
loss.backward()
if (idx + 1) % gradient_accumulation_steps == 0:
# Update weights every `gradient_accumulation_steps` batches.
optimizer.step()
8
# Zero gradients every `gradient_accumulation_steps` batches.
optimizer.zero_grad()
As a result, our effective batch size when training is multiplied by k, the number of gradient accumulation
steps.
4.2 SFT Helper Methods
Next, we will implement some helper methods that you will use during SFT and in the later RL experiments.
As a quick note on nomenclature: in the following sections, we will interchangeably refer to a model’s
completion given a prompt as an “output”, “completion”, or “response”.
Tokenizing prompts and outputs. For each pair of question and target output (q, o), we will tokenize
the question and output separately and concatenate them. Then, we can score the log-probabilities of the
output with our SFT model (or in later sections, our RL policy). Moreover, we will need to construct a
response_mask: a boolean mask that is True for all tokens in the response, and False for all question and
padding tokens. We will use this mask in the training loop to ensure that we only compute the loss on the
response tokens
"""


import torch
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
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    prompt_token_ids_list = tokenizer(prompt_strs, add_special_tokens=False).input_ids
    output_token_ids_list = tokenizer(output_strs, add_special_tokens=False).input_ids

    combined_ids_list = []
    response_masks_list = []

    bos_id = [tokenizer.bos_token_id] if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None else []
    eos_id = [tokenizer.eos_token_id] if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None else []

    for i in range(len(prompt_token_ids_list)):
        prompt_ids = prompt_token_ids_list[i]
        output_ids = output_token_ids_list[i] + eos_id

        combined_ids = bos_id + prompt_ids + output_ids
        combined_ids_list.append(combined_ids)

        prompt_len = len(bos_id) + len(prompt_ids)
        
        mask = [False] * (prompt_len - 1) + [True] * len(output_ids)
        response_masks_list.append(mask)

    tokenizer.padding_side = 'right'
    padded_data = tokenizer.pad(
        {"input_ids": combined_ids_list},
        padding=True,
        return_tensors="pt"
    )
    input_ids_full = padded_data.input_ids

    max_len = input_ids_full.shape[1]
    padded_masks = []
    for mask in response_masks_list:
        pad_len = max_len - 1 - len(mask)
        padded_masks.append(mask + [False] * pad_len)
    
    response_mask = torch.tensor(padded_masks, dtype=torch.bool)

    input_ids = input_ids_full[:, :-1]
    labels = input_ids_full[:, 1:].clone()
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }