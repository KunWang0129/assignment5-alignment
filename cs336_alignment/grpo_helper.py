import torch
from typing import Literal

def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
    ):

    """
    Compute group normalized rewards for a set of rollout responses against ground truth responses.
    Args:
            - reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against 
            the ground truths, producing a dict with keys "reward", "format_reward", and
            "answer_reward".
            - rollout_responses: list[str] Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
            repeated_ground_truths: list[str] The ground truths for the examples. The length of this
            list is rollout_batch_size, because the ground truth for each example is repeated
            group_size times.
            - group_size: int Number of responses per question (group).
            advantage_eps: float Small constant to avoid division by zero in normalization.
            normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise
s           ubtract only the group mean.
    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]] containing:
            - advantage: torch.Tensor The computed advantage for each response.
            - raw_rewards: torch.Tensor The raw rewards for each response.
            - metadata: dict Metadata about the rewards, including mean, std, max, and min.
    """

    raw_rewards = []

    for rollout_response, gt_response in zip(rollout_responses, repeated_ground_truths):
        curr_reward = reward_fn(rollout_response, gt_response)['reward']
        raw_rewards.append(curr_reward)
    
    # Compute mean reward for each group
    raw_rewards = torch.tensor(raw_rewards)
    rewards_per_group = raw_rewards.reshape((-1, group_size))
    mean_reward_per_group = torch.mean(rewards_per_group, dim=-1, keepdim=True)

    advantage = rewards_per_group - mean_reward_per_group

    if normalize_by_std:
        std_reward_per_group = torch.std(rewards_per_group, dim=-1, keepdim=True)

        advantage /= (std_reward_per_group + advantage_eps)
    
    advantage = advantage.flatten()

    metadata = {
        'mean': torch.mean(raw_rewards),
        'std': torch.std(raw_rewards),
        'max': torch.max(raw_rewards),
        'min': torch.min(raw_rewards),
    }

    return advantage, raw_rewards, metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    ) -> torch.Tensor:
    """
    Compute the policy gradient loss using raw rewards or advantages.
    Args:
        raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1). The raw rewards or advantages for each response.
        policy_log_probs: Shape (batch_size, sequence_length), The log probabilities of the policy for each response.
    Returns:
        torch.Tensor: Shape (batch_size, sequence_length), The computed policy gradient loss.
    """

    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the GRPO clipped loss.
    Args:
        - advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log
        probs from the policy being trained.
        - old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs
        from the old policy.
        - cliprange: float Clip parameter ε (e.g. 0.2).
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: A tuple containing:
            - loss: torch.Tensor The computed GRPO loss.
            - loss_metadata: dict[str, torch.Tensor] whether the tokens were clipped.
    """

    # Broadcast advantages to match policy_log_probs shape
    advantages = advantages.expand_as(policy_log_probs)

    # Compute log ratio
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    unclipped_ratio = ratio * advantages

    # Clipped ratio
    clipped_ratio = torch.clip(ratio, min=1.0 - cliprange, max=1.0 + cliprange)
    clipped_ratio *= advantages

    # Compute loss
    loss = -torch.minimum(unclipped_ratio, clipped_ratio)

    metadata = {
        'token_clipped': clipped_ratio < unclipped_ratio
    }

    return loss, metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the policy gradient loss based on the specified loss type.
    Args:
        - policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
        policy being trained.
        - loss_type One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards Required if loss_type == "no_baseline"; shape (batch_size, 1).
        - advantages Required for "reinforce_with_baseline" and "grpo_clip"; shape
        (batch_size, 1).
        - old_log_probs Required for "grpo_clip"; shape (batch_size, sequence_length).
        - cliprange Required for "grpo_clip"; scalar ε used for clipping.
    """

    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("raw_rewards must be provided for 'no_baseline' loss type.")
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}
    
    elif loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("advantages must be provided for 'reinforce_with_baseline' loss type.")
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}
    
    elif loss_type == "grpo_clip":
        if advantages is None or old_log_probs is None or cliprange is None:
            raise ValueError("advantages, old_log_probs, and cliprange must be provided for 'grpo_clip' loss type.")
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    return loss, metadata

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    ) -> torch.Tensor:
    """
    Compute the mean of a tensor along a specified dimension, respecting a boolean mask.
    Args:
        - tensor: torch.Tensor The data to be averaged.
        - mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.
        - dim: int | None Dimension over which to average. If None, compute the mean over all
        masked elements.
    Returns:
        torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics.
    """

    masked_tensor = tensor * mask

    return torch.sum(masked_tensor, dim=dim) / torch.sum(mask, dim=dim)

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Perform a single training step on a microbatch of data for GRPO.
    Args:
        - tensor: torch.Tensor The data to be averaged.
        - mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.
        - dim: int | None Dimension over which to average. If None, compute the mean over all
        masked elements.
    Returns:
        - torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics.
    """
    # Compute the loss
    logs, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    loss = masked_mean(logs, response_mask)
    loss /= gradient_accumulation_steps

    loss.backward()

    return loss, metadata