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

    raw_rewards = []

    # Compute rewards for each response in the group
    for rollout_response, gt_response in zip(rollout_responses, repeated_ground_truths):
        curr_reward = reward_fn(rollout_response, gt_response)["reward"]
        raw_rewards.append(curr_reward)
    
    # Compute the mean reward for each group
    raw_rewards = torch.tensor(raw_rewards)
    rewards_per_group = raw_rewards.reshape((-1, group_size))
    mean_reward = torch.mean(rewards_per_group, dim=1, keepdim=True)

    advantage = rewards_per_group - mean_reward

    if normalize_by_std:
        std_reward = torch.std(rewards_per_group, dim=-1, keepdim=True)
        advandage /= (std_reward + advantage_eps)
    advantage = advantage.flatten()

    metadata = {
        'mean': torch.mean(raw_rewards),
        'std': torch.std(raw_rewards),
        'max': torch.max(raw_rewards),
        'min': torch.min(raw_rewards),
    }

    return advantage, raw_rewards, metadata
