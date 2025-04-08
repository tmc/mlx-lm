# Copyright © 2024 Apple Inc.

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from ..models import cache
from ..generate import generation_stream
from .grpo_reward_functions import (
    RewardFunctions,
    r1_accuracy_reward_func,
    r1_count_xml,
    r1_extract_xml_answer,
    r1_int_reward_func,
    r1_soft_format_reward_func,
    r1_strict_format_reward_func,
)
from .trainer import TrainingArgs, TrainingCallback, average_gradients, grad_checkpoint


def nanmean(a, axis=None, keepdims=False):
    """Calculate mean ignoring NaN values (similar to torch.nanmean).
    
    Args:
        a: Input array
        axis: Axis along which to calculate mean
        keepdims: Whether to keep dimensions
        
    Returns:
        Mean of non-NaN values, or NaN if all values are NaN
    """
    mask = mx.logical_not(mx.isnan(a))
    count = mx.sum(mask, axis=axis, keepdims=keepdims)
    masked_a = mx.where(mask, a, mx.zeros_like(a))
    total = mx.sum(masked_a, axis=axis, keepdims=keepdims)
    # Create an array of NaNs with the same shape as total
    nan_array = mx.ones_like(total) * float('nan')
    return mx.where(count > 0, total / count, nan_array)


def nanstd(a, axis=None, keepdims=False):
    """Calculate standard deviation ignoring NaN values (similar to PyTorch implementation).
    
    Args:
        a: Input array
        axis: Axis along which to calculate std
        keepdims: Whether to keep dimensions
        
    Returns:
        Standard deviation of non-NaN values, or NaN if all values are NaN
    """
    mask = mx.logical_not(mx.isnan(a))
    count = mx.sum(mask, axis=axis, keepdims=True)
    mean = nanmean(a, axis=axis, keepdims=True)
    
    # Calculate squared differences from mean for non-NaN values
    diff_squared = mx.where(mask, (a - mean) ** 2, mx.zeros_like(a))
    
    # Use Bessel's correction (n-1) for unbiased estimation when count > 1
    # This matches PyTorch's behavior
    variance = mx.sum(diff_squared, axis=axis, keepdims=keepdims)
    denom = mx.maximum(count - 1, 1)  # Avoid division by 0
    variance = variance / denom
    
    # Return 0 if count <= 1 (std not defined), NaN if all values are NaN
    result = mx.sqrt(variance)
    if axis is not None or keepdims:
        # Create an array of NaNs with the same shape as result
        nan_array = mx.ones_like(result) * float('nan')
        return mx.where(count > 1, result, nan_array)
    else:
        return result if count.item() > 1 else mx.array(float('nan'))


@dataclass
class GRPOTrainingArgs(TrainingArgs):
    group_size: int = field(
        default=4,
        metadata={"help": "Number of responses per prompt."},
    )
    beta: float = field(default=0.1, metadata={"help": "KL penalty coefficient."})
    epsilon: float = field(
        default=1e-4, metadata={"help": "The Epsilon for numerical stability."}
    )
    max_completion_length: int = field(
        default=512, metadata={"help": "Number of tokens to generate per completion."}
    )
    reference_model_path: str = field(
        default=None,
        metadata={
            "help": "Path to reference model weights. If None, uses the same model."
        },
    )
    temperature: float = field(
        default=1.0,
        metadata={
            "help": "Temperature for sampling. The higher the temperature, the more random the completions."
        },
    )
    reward_weights: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are weighted equally with weight `1.0`."
        },
    )
    scale_rewards: bool = field(
        default=True,
        metadata={
            "help": "Whether to scale advantages by their overall standard deviation."
        },
    )
    wandb_log_samples: bool = field(
        default=True,
        metadata={
            "help": "Log sample tables to WandB during validation."
        },
    )
    wandb_num_samples: int = field(
        default=8,
        metadata={
            "help": "Number of samples to log per validation."
        },
    )
    wandb_sample_strategy: str = field(
        default="best_worst_median",
        metadata={
            "help": "Strategy for selecting samples to log. Options: best_worst_median, percentiles, random"
        },
    )
    wandb_sample_percentiles: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "Percentiles to log if strategy is 'percentiles'. Default: [10, 50, 90]"
        },
    )


def get_per_token_logps(model: nn.Module, inputs, lengths, temperature=1.0):
    logits = model(inputs).astype(mx.float16)
    logits = logits[:, :-1, :]
    targets = inputs[:, 1:]
    per_token_logps = []
    for i in range(logits.shape[0]):
        seq_len = int(lengths[i]) - 1
        if seq_len <= 0:
            # Handle empty or single token sequences
            per_token_logps.append(mx.array([], dtype=mx.float32))
            continue
        seq_logits = logits[i, :seq_len]
        # Scale logits by temperature before softmax
        seq_logits = seq_logits / temperature
        seq_targets = targets[i, :seq_len]
        log_probs = nn.log_softmax(seq_logits, axis=-1)
        token_log_probs = mx.take_along_axis(
            log_probs, seq_targets.reshape(seq_len, 1), axis=-1
        ).squeeze(-1)
        per_token_logps.append(token_log_probs)
    mx.eval(logits)
    return per_token_logps


def generate_step(
    prompt: mx.array,
    model: nn.Module,
    max_tokens: int = 256,
    sampler: Optional[Callable] = None,
    logits_processors: Optional[List[Callable]] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    if sampler is None:
        # Default to argmax if no sampler is provided
        def sampler(logprobs):
            return mx.argmax(logprobs, axis=-1)
    
    tokens = None
    y = prompt
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(model, max_kv_size=max_kv_size)
    def _step(y):
        with mx.stream(generation_stream):
            logits = model(y[None], cache=prompt_cache)
            logits = logits[:, -1, :]
            if logits_processors:
                nonlocal tokens
                tokens = mx.concat([tokens, y]) if tokens is not None else y
                for processor in logits_processors:
                    logits = processor(tokens, logits)
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            next_token = sampler(logprobs)
            return mx.stop_gradient(next_token), mx.stop_gradient(logprobs.squeeze(0))
    try:
        with mx.stream(generation_stream):
            y, logprobs = _step(y)
        mx.eval(y, logprobs)
        for n in range(max_tokens):
            yield y.item(), logprobs
            next_y, next_logprobs = _step(y)
            mx.eval(next_y, next_logprobs)
            y, logprobs = next_y, next_logprobs
            if (n + 1) % 32 == 0:
                mx.clear_cache()
    finally:
        mx.clear_cache()


def generate_grpo(
    model: nn.Module,
    tokenizer,
    prompt_tokens,
    max_tokens: int,
    group_size: int,
    temperature: float,
    batch_size: int,
    end_token: str = "</answer>"
):
    try:
        end_sequence = mx.array(tokenizer.encode(end_token))
        total_samples = len(prompt_tokens)
        all_completions = []
        all_completion_texts = []
        batch_indices = []
        
        def temp_sampler(logits):
            return mx.random.categorical(logits / temperature)
        
        for i in range(0, total_samples, batch_size):
            current_batch_size = min(batch_size, total_samples - i)
            batch_prompts = prompt_tokens[i : i + current_batch_size]
            
            max_prompt_len = max(len(p) for p in batch_prompts)
            padded_prompts = []
            for prompt in batch_prompts:
                padding = [tokenizer.pad_token_id] * (max_prompt_len - len(prompt))
                padded_prompts.append(prompt + padding)
                
            prompt_tensor = mx.stop_gradient(mx.array(padded_prompts))
            
            if len(prompt_tensor.shape) == 1:
                prompt_tensor = prompt_tensor[None, :]
            if prompt_tensor.shape[1] == 0:
                continue
                
            expanded_prompts = mx.repeat(prompt_tensor, group_size, axis=0)
            batch_results = []
            
            total_prompt_samples = expanded_prompts.shape[0]
            for prompt_idx in range(total_prompt_samples):
                current_tokens = []
                prompt_cache = cache.make_prompt_cache(model)
                
                for token, _ in generate_step(
                    expanded_prompts[prompt_idx],
                    model,
                    max_tokens=max_tokens,
                    sampler=temp_sampler,
                    prompt_cache=prompt_cache,
                ):
                    if token == tokenizer.eos_token_id:
                        break
                        
                    current_tokens.append(token)

                    if len(current_tokens) >= len(end_sequence) and mx.array_equal(
                        mx.array(current_tokens[-len(end_sequence):]), end_sequence
                    ):
                        break
                
                if current_tokens:
                    batch_results.append(mx.array(current_tokens))
            
            if batch_results:
                for j, completion_ids in enumerate(batch_results):
                    prompt_idx = i + (j // group_size)
                    if prompt_idx < total_samples:
                        batch_indices.append(prompt_idx)
                        completion_text = tokenizer.decode(completion_ids.tolist())
                        all_completions.append(mx.stop_gradient(completion_ids))
                        all_completion_texts.append(completion_text)
            
            mx.clear_cache()
    
    finally:
        mx.clear_cache()
    
    return all_completions, all_completion_texts, batch_indices


def grpo_loss(
    model,
    ref_model,
    tokenizer,
    batch,
    completions=None,
    completion_texts=None,
    batch_indices=None,
    reward_funcs: Optional[List[RewardFunctions]] = None,
    beta: float = 0.1,
    group_size: int = 4,
    epsilon: float = 1e-4,
    max_tokens: int = 64,
    temperature: float = 0.8,
    reward_weights: Optional[List[float]] = None,
    batch_size: int = 1,
    scale_rewards: bool = True,
    is_validation: bool = False
):
    # Default reward functions if none provided
    if reward_funcs is None:
        reward_funcs = [
            r1_accuracy_reward_func,
            r1_int_reward_func,
            r1_strict_format_reward_func,
            r1_soft_format_reward_func,
            r1_count_xml,
        ]
    
    prompt_tokens, _, prompt_text, answer_text = batch
    
    if completions is not None and completion_texts is not None and batch_indices is not None:
        all_completions = completions
        all_completion_texts = completion_texts
        batch_indices = batch_indices
    else:
        all_completions, all_completion_texts, batch_indices = generate_grpo(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            group_size=group_size,
            temperature=temperature,
            batch_size=batch_size
        )

    if not all_completions:
        raise ValueError(
            "No completions were generated. Please check your model and inputs."
        )

    expanded_answers = []
    expanded_prompts = []
    unique_prompt_indices = sorted(set(batch_indices))
    grouped_completions = {idx: [] for idx in unique_prompt_indices}

    for i, completion_idx in enumerate(batch_indices):
        grouped_completions[completion_idx].append(i)

    ordered_completions = []
    ordered_completion_texts = []
    ordered_batch_indices = []

    for prompt_idx in unique_prompt_indices:
        completion_indices = grouped_completions[prompt_idx]
        for idx in completion_indices:
            ordered_completions.append(all_completions[idx])
            ordered_completion_texts.append(all_completion_texts[idx])
            ordered_batch_indices.append(prompt_idx)
            expanded_prompts.append(prompt_text[prompt_idx])
            expanded_answers.append(answer_text[prompt_idx])

    all_completions = ordered_completions
    all_completion_texts = ordered_completion_texts
    batch_indices = ordered_batch_indices
    max_length = max(ids.shape[0] for ids in all_completions)
    padded_completions = []
    attention_masks = []

    for completion_ids in all_completions:
        completion_tensor = mx.array(completion_ids.tolist())
        
        padding_length = max_length - completion_tensor.shape[0]
        if padding_length > 0:
            padding = mx.zeros((padding_length,), dtype=completion_tensor.dtype)
            padded_ids = mx.concatenate([completion_tensor, padding])
            mask = mx.concatenate(
                [mx.ones_like(completion_tensor), mx.zeros_like(padding)]
            )
        else:
            padded_ids = completion_tensor
            mask = mx.ones_like(completion_tensor)
        padded_completions.append(padded_ids)
        attention_masks.append(mask)

    inputs = mx.stack(padded_completions)
    attention_mask = mx.stack(attention_masks)
    lengths = attention_mask.sum(axis=1)

    token_log_probs = get_per_token_logps(model, inputs, lengths, temperature=temperature)
    mx.eval(token_log_probs)

    if ref_model is None:
        ref_token_log_probs = token_log_probs
    else:
        ref_token_log_probs = get_per_token_logps(ref_model, inputs, lengths, temperature=temperature)
        mx.eval(ref_token_log_probs)

    max_len = max(x.shape[0] for x in token_log_probs)
    padded_log_probs = []
    padded_ref_log_probs = []

    for i in range(len(token_log_probs)):
        seq_len = token_log_probs[i].shape[0]
        padding = mx.zeros((max_len - seq_len,))

        padded_log_probs.append(mx.concatenate([token_log_probs[i], padding]))
        padded_ref_log_probs.append(mx.concatenate([ref_token_log_probs[i], padding]))

    token_log_probs = mx.stack(padded_log_probs)
    ref_token_log_probs = mx.stack(padded_ref_log_probs)

    all_func_rewards = []

    for reward_func in reward_funcs:
        # Get raw rewards from function
        raw_rewards = reward_func(
            prompts=expanded_prompts,
            completions=all_completion_texts,
            answer=expanded_answers,
        )
        
        # Handle None values by converting to NaN
        processed_rewards = []
        for r in raw_rewards:
            if r is None:
                processed_rewards.append(float('nan'))
            else:
                processed_rewards.append(float(r))
        
        # Convert to MLX array
        func_rewards = mx.array(processed_rewards, dtype=mx.float32)
        all_func_rewards.append(func_rewards)

    rewards_matrix = mx.stack(all_func_rewards, axis=1)

    if reward_weights is not None:
        if len(reward_weights) != len(reward_funcs):
            raise ValueError(
                f"Number of reward weights ({len(reward_weights)}) must match number of reward "
                f"functions ({len(reward_funcs)})"
            )
        reward_weights = mx.array(reward_weights, dtype=mx.float32)
    else:
        reward_weights = mx.ones(len(reward_funcs), dtype=mx.float32)

    # Apply reward weights to the reward matrix
    weighted_rewards = rewards_matrix * mx.expand_dims(reward_weights, 0)
    
    # Handle NaN values in rewards: replace with zeros for summing
    nan_mask = mx.isnan(weighted_rewards)
    weighted_rewards = mx.where(nan_mask, mx.zeros_like(weighted_rewards), weighted_rewards)
    
    # Sum across reward functions
    rewards = weighted_rewards.sum(axis=1)

    num_unique_prompts = len(unique_prompt_indices)

    rewards_by_prompt = [[] for _ in range(num_unique_prompts)]
    for i, prompt_idx in enumerate(batch_indices):
        prompt_position = unique_prompt_indices.index(prompt_idx)
        rewards_by_prompt[prompt_position].append(rewards[i])

    advantages = mx.zeros_like(rewards)
    for i, prompt_rewards_list in enumerate(rewards_by_prompt):
        # Only calculate advantages if we have more than one sample per prompt
        # (required for meaningful mean/std calculations)
        if len(prompt_rewards_list) > 1:
            prompt_rewards_tensor = mx.array(prompt_rewards_list)
            mean_reward = mx.mean(prompt_rewards_tensor)
            std_reward = mx.std(prompt_rewards_tensor)
            
            # Find indices of all completions for this prompt
            indices = [
                j
                for j, idx in enumerate(batch_indices)
                if idx == unique_prompt_indices[i]
            ]
            
            # Only standardize if there's actual variance in rewards
            if std_reward > 0:
                for j, idx in enumerate(indices):
                    advantages[idx] = (prompt_rewards_tensor[j] - mean_reward) / (
                        std_reward + epsilon
                    )
            else:
                # If all rewards are identical, use raw differences from mean
                # (which will be zero, but this is more explicit and future-proof)
                for j, idx in enumerate(indices):
                    advantages[idx] = prompt_rewards_tensor[j] - mean_reward
        else:
            # For prompts with only one completion, advantage is 0
            idx = batch_indices.index(unique_prompt_indices[i])
            advantages[idx] = 0.0
            
    # Scale advantages by their overall standard deviation if requested
    # This helps stabilize training by normalizing the magnitude of advantages
    if scale_rewards and advantages.size > 1:
        overall_std_adv = mx.std(advantages)
        # Only scale if there's meaningful variation
        if overall_std_adv > epsilon:
            advantages = advantages / (overall_std_adv + epsilon)

    # Compute KL divergence using Schulman's approximator
    kl_div = (
        mx.exp(ref_token_log_probs - token_log_probs)
        - (ref_token_log_probs - token_log_probs)
        - 1
    )

    # Create mask for valid tokens
    length_mask = mx.arange(inputs.shape[1] - 1)[None, :] < (lengths[:, None] - 1)

    # Compute policy ratio
    policy_ratio = mx.exp(
        mx.array(token_log_probs - mx.stop_gradient(ref_token_log_probs))
    )

    # Apply PPO like clipping
    policy_ratio_cliped = mx.clip(policy_ratio, 1 - epsilon, 1 + epsilon)

    # Calculate both unclipped and clipped objectives
    unclipped_obj = policy_ratio * advantages.reshape(-1, 1)
    clipped_obj = policy_ratio_cliped * advantages.reshape(-1, 1)

    # Take the minimum (pessimistic bound)
    per_token_loss = -mx.minimum(unclipped_obj, clipped_obj)
    
    # Calculate clip fraction for reporting
    is_clipped = mx.not_equal(unclipped_obj, clipped_obj)
    clip_fraction = (is_clipped * length_mask).sum() / (length_mask.sum() + epsilon)

    # Add KL penalty if beta is non-zero
    if beta != 0.0:
        per_token_loss = per_token_loss + beta * kl_div


    per_token_loss = per_token_loss * length_mask

    # Get total valid tokens for safer division
    total_valid_tokens = length_mask.sum()
    
    # Average over tokens with NaN protection
    if total_valid_tokens > 0:
        loss = (per_token_loss * length_mask).sum() / total_valid_tokens  # Matches the pytorch implementation
    else:
        loss = mx.array(0.0, dtype=mx.float32)  # Fallback to zero loss if no valid tokens

    # Calculate mean KL divergence for metrics with NaN protection
    if total_valid_tokens > 0:
        # Per-sample valid token counts
        valid_tokens_per_sample = length_mask.sum(axis=1)
        
        # Calculate sum of KL for each sample
        kl_sum_per_sample = (kl_div * length_mask).sum(axis=1)
        
        # Calculate mean KL per sample but avoid division by zero
        # by setting the result to 0 where token count is 0
        per_sample_kl = mx.where(
            valid_tokens_per_sample > 0,
            kl_sum_per_sample / valid_tokens_per_sample,
            mx.zeros_like(kl_sum_per_sample)
        )
        
        # Calculate the overall mean, making sure to only consider samples with valid tokens
        valid_sample_count = mx.sum(valid_tokens_per_sample > 0)
        if valid_sample_count > 0:
            # Sum all per-sample KLs and divide by number of valid samples
            mean_kl = mx.sum(per_sample_kl * (valid_tokens_per_sample > 0)) / valid_sample_count
        else:
            mean_kl = mx.array(0.0, dtype=mx.float32)
    else:
        mean_kl = mx.array(0.0, dtype=mx.float32)
        
    # --- Calculate Policy Stats ---
    # Calculate clip_ratio: fraction of tokens where PPO objective was clipped
    # Method 2: Explicitly check the conditions for clipping based on advantage sign
    adv_reshaped = advantages.reshape(-1, 1)
    ratio_lower_bound = 1.0 - epsilon
    ratio_upper_bound = 1.0 + epsilon
    is_clipped = mx.logical_or(
        mx.logical_and(policy_ratio < ratio_lower_bound, adv_reshaped < 0),
        mx.logical_and(policy_ratio > ratio_upper_bound, adv_reshaped > 0)
    )
    clip_ratio = (is_clipped * length_mask).sum() / total_valid_tokens if total_valid_tokens > 0 else mx.array(0.0)

    # Collect reward metrics
    reward_metrics = {}
    individual_rewards = {}
    
    for i, reward_func in enumerate(reward_funcs):
        func_name = reward_func.__name__
        # Get raw rewards
        raw_rewards = reward_func(
            prompts=expanded_prompts,
            completions=all_completion_texts,
            answer=expanded_answers,
        )
        
        # Handle None values in raw rewards
        processed_rewards = []
        for r in raw_rewards:
            if r is None:
                processed_rewards.append(float('nan'))
            else:
                processed_rewards.append(float(r))
        
        func_rewards = mx.array(processed_rewards)
        # Use nanmean/nanstd to correctly handle NaN values (matching TRL behavior)
        reward_metrics[f"{func_name}_mean"] = nanmean(func_rewards)
        reward_metrics[f"{func_name}_std"] = nanstd(func_rewards)
        
        # Store individual reward values for potential sample collection
        individual_rewards[func_name] = func_rewards

    grouped_rewards_mean = mx.array(
        [mx.mean(mx.array(rewards)) for rewards in rewards_by_prompt]
    )
    grouped_rewards_std = mx.array(
        [
            mx.std(mx.array(rewards)) if len(rewards) > 1 else mx.zeros(1)
            for rewards in rewards_by_prompt
        ]
    )

    # Calculate completion statistics to track terminated vs truncated sequences
    # Get true completion lengths (not padded) from actual completions
    completion_lengths = mx.array([len(comp) for comp in all_completions])
    
    # Identify terminated sequences (those ending with EOS or special end token)
    # We check by seeing if a completion is shorter than max_tokens
    terminated_mask = completion_lengths < max_tokens
    terminated_ratio = terminated_mask.sum() / len(completion_lengths) if len(completion_lengths) > 0 else 0.0
    clipped_ratio = 1.0 - terminated_ratio
    
    # Get statistics for terminated sequences without using boolean indexing
    # Convert boolean mask to indices first, as MLX doesn't support boolean indexing
    if mx.sum(terminated_mask) > 0:
        # Get indices where terminated_mask is True
        terminated_indices = mx.array([i for i, is_term in enumerate(terminated_mask.tolist()) if is_term])
        # Use take to get the corresponding lengths
        terminated_lengths = mx.take(completion_lengths, terminated_indices)
    else:
        terminated_lengths = mx.array([0])
    
    # Get mean, min, max of completion lengths
    comp_mean_len = mx.mean(completion_lengths).item() if completion_lengths.size > 0 else 0.0
    comp_max_len = mx.max(completion_lengths).item() if completion_lengths.size > 0 else 0.0
    comp_min_len = mx.min(completion_lengths).item() if completion_lengths.size > 0 else 0.0
    
    # Get mean, min, max of terminated completion lengths
    term_mean_len = mx.mean(terminated_lengths).item() if terminated_lengths.size > 0 else 0.0
    term_max_len = mx.max(terminated_lengths).item() if terminated_lengths.size > 0 else 0.0
    term_min_len = mx.min(terminated_lengths).item() if terminated_lengths.size > 0 else 0.0
    
    # Store individual reward metrics for easier access
    all_individual_rewards = {}
    for i, reward_func in enumerate(reward_funcs):
        func_name = reward_func.__name__
        all_individual_rewards[func_name] = rewards_matrix[:, i]
    
    # Create TRL-aligned metrics dictionary with hierarchical structure
    # Structure follows exactly what's described in the user experience document
    metrics = {
        # Loss and overall performance
        "loss": loss,
        
        # Primary reward metrics
        "reward/mean": nanmean(rewards),  # Primary indicator for model optimization
        "reward/std": nanstd(rewards),
        "reward/grouped_mean": nanmean(grouped_rewards_mean),
        "reward/grouped_std": nanmean(grouped_rewards_std),
        
        # Policy metrics 
        "policy/kl_mean": mean_kl,  # KL from reference policy
        "policy/clip_ratio": clip_ratio,  # How often updates are clipped
        
        # Completion statistics
        "completions/mean_length": comp_mean_len,
        "completions/max_length": comp_max_len,
        "completions/min_length": comp_min_len,
        "completions/clipped_ratio": float(clipped_ratio),
        "completions/terminated_ratio": float(terminated_ratio),
        
        # Terminated sequence statistics
        "completions/mean_terminated_length": term_mean_len,
        "completions/max_terminated_length": term_max_len,
        "completions/min_terminated_length": term_min_len,
        
        # System statistics
        "system/num_tokens": lengths.sum().item(),
    }
    
    # Add individual reward function stats with standardized naming
    # Crucial for monitoring individual reward components
    for fname in all_individual_rewards:
        metrics[f"rewards/{fname}/mean"] = nanmean(all_individual_rewards[fname])
        metrics[f"rewards/{fname}/std"] = nanstd(all_individual_rewards[fname])
    
    # Add aliases for backward compatibility
    metrics["reward"] = metrics["reward/mean"]
    metrics["reward_std"] = metrics["reward/std"]
    metrics["kl"] = metrics["policy/kl_mean"]
    metrics["clip_ratio"] = metrics["policy/clip_ratio"]

    if is_validation and all_completion_texts:
        try:
            print("\n=== Validation Sample Details ===")

            # Print the input context (prompt)
            last_prompt_idx = batch_indices[-1] if batch_indices else 0

            if last_prompt_idx < len(prompt_text):
                print(f"\n📋 Raw Prompt:\n{prompt_text[last_prompt_idx]}")
                print("\n" + "=" * 10 + "\n")

                # Get the actual tokenized prompt that was fed to the model
                if last_prompt_idx < len(prompt_tokens):
                    actual_prompt = tokenizer.decode(prompt_tokens[last_prompt_idx])
                    print(f"\n🔄 Model Input:\n{actual_prompt}")
                    print("\n" + "=" * 10 + "\n")

            if all_completion_texts:
                print(f"\n📝 Generation:\n{all_completion_texts[-1]}")
                print("\n" + "=" * 10 + "\n")

            # Make sure we have a valid index for answer_text
            if last_prompt_idx < len(answer_text):
                print(f"\n✅ Answer:\n{answer_text[last_prompt_idx]}")
                print("\n" + "=" * 10 + "\n")

            # Only try to extract if r1_extract_xml_answer is defined
            if "r1_extract_xml_answer" in globals() and all_completion_texts:
                print(
                    f"\n🔍 Extracted Answer:\n{r1_extract_xml_answer(all_completion_texts[-1])}"
                )
            print("\n" + "=" * 35 + "\n")
        except Exception as e:
            print(f"\nError printing validation details: {e}")
            print("Continuing with training...\n")

    # Collect rich sample data for WandB reporting
    sample_data = {
        # Basic sample information
        "prompts": expanded_prompts,
        "completions": all_completion_texts,
        "answers": expanded_answers,
        
        # Reward and policy metrics
        "rewards": rewards,
        "advantages": advantages,
        "kl_div_per_token": (kl_div * length_mask).sum(axis=1) / mx.maximum(length_mask.sum(axis=1), 1),
        "policy_ratio": policy_ratio,
        
        # Individual reward components
        "individual_rewards": all_individual_rewards,
        
        # Completion statistics
        "completion_lengths": completion_lengths,
        "terminated_mask": terminated_mask,
        
        # Stats for histograms
        "stats": {
            "mean_reward": nanmean(rewards).item(),
            "std_reward": nanstd(rewards).item(),
            "mean_advantage": nanmean(advantages).item(),
            "std_advantage": nanstd(advantages).item(),
            "mean_completion_length": comp_mean_len,
            "clip_ratio": clip_ratio.item(),
            "kl": mean_kl.item(),
        }
    }
    
    mx.clear_cache()

    return loss, length_mask.sum(axis=1).sum(), metrics, sample_data


def iterate_grpo_batches(dataset, batch_size, max_seq_length, train=False):
    if not dataset or not isinstance(dataset[0], tuple) or len(dataset[0]) != 4:
        raise ValueError(
            "Dataset must be list of (prompt_tokens, answer_tokens, prompt_str, answer_str) tuples"
        )

    def length_key(i):
        return len(dataset[i][0]) + len(dataset[i][1])

    idx = sorted(range(len(dataset)), key=length_key)

    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size} "
            f"examples but only has {len(dataset)}."
        )

    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")

    def batch_index_generator():
        for i in range(0, len(idx) - batch_size + 1, batch_size):
            yield idx[i : i + batch_size : step]

    while True:
        indices = (
            np.random.permutation(list(batch_index_generator()))
            if train
            else batch_index_generator()
        )

        for batch_idx in indices:
            current_batch = [dataset[j] for j in batch_idx]

            prompts_tokens = [item[0] for item in current_batch]
            answers_tokens = [item[1] for item in current_batch]
            prompts_text = [item[2] for item in current_batch]
            answers_text = [item[3] for item in current_batch]

            # Check and truncate prompts that exceed max_seq_length
            if any(len(p) > max_seq_length for p in prompts_tokens):
                print(
                    f"[WARNING] Some prompts are longer than {max_seq_length} tokens. "
                    "Long prompts will be truncated."
                )
                # Actually truncate the prompts
                prompts_tokens = [p[:max_seq_length] for p in prompts_tokens]

            yield prompts_tokens, answers_tokens, prompts_text, answers_text

        if not train:
            break


def evaluate_grpo(
    model: nn.Module,
    ref_model: Optional[nn.Module],
    dataset,
    tokenizer,
    batch_size,
    num_batches,
    beta: float,
    epsilon: float,
    group_size: int,
    max_seq_length: int,
    max_tokens: int,
    temperature: float,
    reward_funcs: Optional[List[RewardFunctions]] = [
        r1_accuracy_reward_func,
        r1_int_reward_func,
        r1_strict_format_reward_func,
        r1_soft_format_reward_func,
        r1_count_xml,
    ],
    reward_weights: Optional[List[float]] = None,
    scale_rewards: bool = True,
    wandb_log_samples: bool = True,
    wandb_num_samples: int = 8,
    wandb_sample_strategy: str = "best_worst_median",
    wandb_sample_percentiles: Optional[List[int]] = None,
    loss_fn: callable = grpo_loss,
    iterate_batches: callable = iterate_grpo_batches,
):
    # Use mx.array for accumulating values to ensure correct distributed behavior
    all_losses = mx.array(0.0, dtype=mx.float32)
    ntokens = mx.array(0, dtype=mx.int32)
    all_metrics = None
    
    # For collecting rich sample data for reporting and reward function analysis
    all_sample_data = {
        # Basic sample information
        "prompts": [],
        "completions": [],
        "answers": [],
        
        # Reward and policy metrics
        "rewards": [],
        "advantages": [],
        "completion_lengths": [],
        "kl_div_per_sample": [],
        
        # Individual reward components dictionary
        "individual_rewards": {},
        
        # Track metadata for each prompt to enable grouping analysis
        "prompt_indices": [],
        "batch_indices": []
    }

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    for _, batch in zip(
        index_iterator,
        iterate_batches(
            dataset=dataset,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        ),
    ):
        losses, toks, metrics, batch_samples = loss_fn(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            reward_funcs=reward_funcs,
            reward_weights=reward_weights,
            beta=beta,
            group_size=group_size,
            epsilon=epsilon,
            ref_model=ref_model,
            temperature=temperature,
            max_tokens=max_tokens,
            scale_rewards=scale_rewards,
            is_validation=True
        )

        all_losses += losses * toks
        ntokens += toks

        if all_metrics is None:
            all_metrics = {k: mx.array(v * toks, dtype=mx.float32) for k, v in metrics.items()}
        else:
            for k, v in metrics.items():
                all_metrics[k] += v * toks
        
        # Accumulate rich sample data for deep reward analysis
        all_sample_data["prompts"].extend(batch_samples["prompts"])
        all_sample_data["completions"].extend(batch_samples["completions"])
        all_sample_data["answers"].extend(batch_samples["answers"])
        
        # Store batch indices for grouping analysis
        current_batch_start = len(all_sample_data["prompts"]) - len(batch_samples["prompts"])
        all_sample_data["batch_indices"].extend([i // batch_size for i in range(current_batch_start, len(all_sample_data["prompts"]))])
        
        # Convert arrays to lists for easier accumulation
        all_sample_data["rewards"].extend(batch_samples["rewards"].tolist())
        all_sample_data["advantages"].extend(batch_samples["advantages"].tolist())
        
        # Store additional metrics if available
        if "completion_lengths" in batch_samples:
            all_sample_data["completion_lengths"].extend(batch_samples["completion_lengths"].tolist())
        
        if "kl_div_per_token" in batch_samples:
            all_sample_data["kl_div_per_sample"].extend(batch_samples["kl_div_per_token"].tolist())
        
        # Track prompt indices (for grouping completions by prompt)
        if "prompt_indices" in batch_samples:
            all_sample_data["prompt_indices"].extend(batch_samples["prompt_indices"])
        else:
            # Default to sequential indices if not provided
            all_sample_data["prompt_indices"].extend(range(
                len(all_sample_data["prompt_indices"]),
                len(all_sample_data["prompt_indices"]) + len(batch_samples["prompts"])
            ))
        
        # Accumulate individual reward values
        for func_name, rewards in batch_samples["individual_rewards"].items():
            if func_name not in all_sample_data["individual_rewards"]:
                all_sample_data["individual_rewards"][func_name] = []
            all_sample_data["individual_rewards"][func_name].extend(rewards.tolist())

        mx.eval(all_losses, ntokens)

    # Aggregate metrics across all workers
    all_losses = mx.distributed.all_sum(all_losses, stream=mx.cpu)
    ntokens = mx.distributed.all_sum(ntokens, stream=mx.cpu)
    all_metrics = {k: mx.distributed.all_sum(v) for k, v in all_metrics.items()}

    # NaN protection for metrics and loss calculation
    if ntokens > 0:
        avg_metrics = {k: (v / ntokens).item() for k, v in all_metrics.items()}
        avg_loss = (all_losses / ntokens).item()
    else:
        # Fallback to zeros if no tokens were processed
        avg_metrics = {k: 0.0 for k, v in all_metrics.items()}
        avg_loss = 0.0
    
    # Add a timestamp for logging purposes
    avg_metrics["val_time"] = time.perf_counter()
    
    # Convert sample data rewards back to mx.arrays for the callback
    if all_sample_data["rewards"]:
        world = mx.distributed.init()
        rank = world.rank()
        
        # Only provide sample data on rank 0 to avoid duplication
        if rank == 0:
            sample_data = {
                "prompts": all_sample_data["prompts"],
                "completions": all_sample_data["completions"],
                "answers": all_sample_data["answers"],
                "rewards": mx.array(all_sample_data["rewards"]),
                "advantages": mx.array(all_sample_data["advantages"]),
                "individual_rewards": {k: mx.array(v) for k, v in all_sample_data["individual_rewards"].items()}
            }
        else:
            sample_data = None
    else:
        sample_data = None

    return avg_loss, ntokens, avg_metrics, sample_data


def train_grpo(
    model: nn.Module,
    ref_model: Optional[nn.Module],
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset,
    reward_funcs: Optional[List[RewardFunctions]] = [
        r1_accuracy_reward_func,
        r1_int_reward_func,
        r1_strict_format_reward_func,
        r1_soft_format_reward_func,
        r1_count_xml,
    ],
    args: GRPOTrainingArgs = GRPOTrainingArgs(),
    loss_fn: callable = grpo_loss,
    iterate_batches: callable = iterate_grpo_batches,
    training_callback: TrainingCallback = None,
):
    print(
        f"Starting GRPO training with {len(reward_funcs)} reward functions..., iters: {args.iters}"
    )
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        if hasattr(model, "layers") and len(model.layers) > 0:
            grad_checkpoint(model.layers[0])
        else:
            print("[Warning] Gradient checkpointing enabled but model.layers not found or empty. Skipping.")

    state = [model.state, optimizer.state, mx.random.state]

    def step(batch):
        # Set model to training mode
        if hasattr(model, "train"):
            model.train()
        
        # Unpack batch
        prompt_tokens, targets, prompt_lens, target_lens = batch
        
        # First generate completions
        all_completions, all_completion_texts, batch_indices = generate_grpo(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            max_tokens=args.max_completion_length,
            group_size=args.group_size,
            temperature=args.temperature,
            batch_size=args.batch_size
        )
        
        # Define a more robust value and grad function
        # This directly uses value_and_grad on a custom loss function
        def custom_loss_for_grad(model_params):
            # Restore model parameters for this gradient computation
            state = [model.state]
            
            # Call loss_fn directly with all necessary parameters
            loss_val, token_count, metrics_dict, _ = loss_fn(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                batch=batch,
                completions=all_completions,
                completion_texts=all_completion_texts,
                batch_indices=batch_indices,
                reward_funcs=reward_funcs,
                beta=args.beta,
                group_size=args.group_size,
                epsilon=args.epsilon,
                max_tokens=args.max_completion_length,
                temperature=args.temperature,
                reward_weights=args.reward_weights,
                batch_size=args.batch_size,
                scale_rewards=args.scale_rewards
            )
            
            # Return only the loss value for gradient computation
            return loss_val
        
        # Get the value and gradient directly - mx.grad is in mlx.core, not mlx.nn
        loss_and_grad = nn.value_and_grad(custom_loss_for_grad)
        
        # Compute loss and gradient
        loss_val, grad = loss_and_grad(model.trainable_parameters())
        
        # Call loss_fn again to get metrics (without computing gradients)
        loss, toks, metrics, _ = loss_fn(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            batch=batch,
            completions=all_completions,
            completion_texts=all_completion_texts,
            batch_indices=batch_indices,
            reward_funcs=reward_funcs,
            beta=args.beta,
            group_size=args.group_size,
            epsilon=args.epsilon,
            max_tokens=args.max_completion_length,
            temperature=args.temperature,
            reward_weights=args.reward_weights,
            batch_size=args.batch_size,
            scale_rewards=args.scale_rewards
        )

        grad = average_gradients(grad)
        optimizer.update(model, grad)

        return loss, toks, metrics

    # Value and grad is now defined inside the step function

    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    # Initialize accumulated metrics with TRL-standardized names
    # Matches the exact structure from user experience document
    accumulated_metrics = {
        # Primary metrics
        "reward/mean": 0,
        "reward/std": 0,
        "reward/grouped_mean": 0,
        "reward/grouped_std": 0,
        
        # Policy metrics
        "policy/kl_mean": 0,
        "policy/clip_ratio": 0,
        
        # Completion statistics
        "completions/mean_length": 0,
        "completions/max_length": 0,
        "completions/min_length": 0,
        "completions/clipped_ratio": 0,
        "completions/terminated_ratio": 0,
    }
    
    # Add individual reward function metrics with standardized naming
    for reward_func in reward_funcs:
        func_name = reward_func.__name__
        accumulated_metrics[f"rewards/{func_name}/mean"] = 0
        accumulated_metrics[f"rewards/{func_name}/std"] = 0
    
    # Add aliases for backward compatibility during training loop
    accumulated_metrics["reward"] = 0
    accumulated_metrics["reward_std"] = 0
    accumulated_metrics["kl"] = 0
    accumulated_metrics["clip_ratio"] = 0

    start = time.perf_counter()
    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ),
    ):
        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
            stop = time.perf_counter()
            
            # Set model to evaluation mode
            if hasattr(model, "eval"):
                model.eval()
                
            val_loss, val_ntokens, val_metrics, val_samples = evaluate_grpo(
                model=model,
                dataset=val_dataset,
                loss_fn=loss_fn,
                ref_model=ref_model,
                reward_funcs=reward_funcs,
                reward_weights=args.reward_weights,
                tokenizer=tokenizer,
                group_size=args.group_size,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                max_tokens=args.max_completion_length,
                beta=args.beta,
                epsilon=args.epsilon,
                temperature=args.temperature,
                scale_rewards=args.scale_rewards,
                iterate_batches=iterate_batches,
            )
            
            # Set model back to training mode
            if hasattr(model, "train"):
                model.train()
            val_time = time.perf_counter() - stop
            if rank == 0:
                val_metrics_str = (
                    f"Val loss {val_loss:.3f}, "
                    f"Val reward {val_metrics.get('reward/mean', 0.0):.3f}, "
                    f"Val reward std {val_metrics.get('reward/std', 0.0):.3f}, "
                    f"Val grouped reward mean {val_metrics.get('reward/grouped_mean', 0.0):.3f}, "
                    f"Val grouped reward std {val_metrics.get('reward/grouped_std', 0.0):.3f}, "
                    f"Val KL {val_metrics.get('policy/kl_mean', 0.0):.3f}, "
                    f"Val clip ratio {val_metrics.get('policy/clip_ratio', 0.0):.3f}, "
                    f"Val length {val_metrics.get('completions/mean_length', 0.0):.1f}"
                )

                # Add individual reward function metrics
                for reward_func in reward_funcs:
                    func_name = reward_func.__name__
                    mean_key = f"rewards/{func_name}/mean"
                    std_key = f"rewards/{func_name}/std"
                    
                    if mean_key in val_metrics and std_key in val_metrics:
                        val_metrics_str += (
                            f", Val {func_name} {val_metrics[mean_key]:.3f}±{val_metrics[std_key]:.3f}"
                        )

                print(
                    f"Iter {it}: {val_metrics_str}, " f"Val took {val_time:.3f}s",
                    flush=True,
                )

            if training_callback is not None:
                # Prepare validation info with hierarchical metrics
                val_info = {
                    "iteration": it,
                    "val_loss": val_loss,
                    # Preserve the hierarchical structure for metrics
                    "val_metrics": val_metrics
                }
                
                # Use enhanced validation_end method with samples data
                training_callback.on_validation_end(val_info, val_samples)
            
            # Clear cache after validation to prevent memory buildup
            mx.clear_cache()
            start = time.perf_counter()

        loss, toks, metrics = step(batch)
        losses += loss
        n_tokens += toks
        steps += 1

        mx.clear_cache()

        for k, v in metrics.items():
            accumulated_metrics[k] += v

        mx.eval(state, losses, n_tokens)

        if it % args.steps_per_report == 0 or it == args.iters:
            stop = time.perf_counter()

            train_loss = mx.distributed.all_sum(losses, stream=mx.cpu).item()
            train_loss /= steps * mx.distributed.init().size()
            avg_metrics = {
                k: v / (steps * world_size) for k, v in accumulated_metrics.items()
            }
            n_tokens = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / (stop - start)
            tokens_sec = float(n_tokens) / (stop - start)
            trained_tokens += n_tokens
            # Use appropriate memory tracking based on platform
            try:
                # Try the current method
                peak_mem = mx.get_peak_memory() / 1e9
            except AttributeError:
                # Fall back to metal method if necessary
                try:
                    peak_mem = mx.metal.get_peak_memory() / 1e9
                except AttributeError:
                    # Provide a default if neither method works
                    peak_mem = 0.0

            if rank == 0:
                train_metrics_str = (
                    f"Train loss {train_loss:.3f}, "
                    f"Reward {avg_metrics.get('reward/mean', 0.0):.3f}, "
                    f"Reward std {avg_metrics.get('reward/std', 0.0):.3f}, "
                    f"Grouped reward mean {avg_metrics.get('reward/grouped_mean', 0.0):.3f}, "
                    f"Grouped reward std {avg_metrics.get('reward/grouped_std', 0.0):.3f}, "
                    f"KL {avg_metrics.get('policy/kl_mean', 0.0):.3f}, "
                    f"Clip ratio {avg_metrics.get('policy/clip_ratio', 0.0):.3f}, "
                    f"Avg length {avg_metrics.get('completions/mean_length', 0.0):.1f}"
                )

                # Add individual reward function metrics
                for reward_func in reward_funcs:
                    func_name = reward_func.__name__
                    mean_key = f"rewards/{func_name}/mean"
                    std_key = f"rewards/{func_name}/std"
                    
                    if mean_key in avg_metrics and std_key in avg_metrics:
                        train_metrics_str += (
                            f", {func_name} {avg_metrics[mean_key]:.3f}±{avg_metrics[std_key]:.3f}"
                        )

                print(
                    f"Iter {it}: {train_metrics_str}, "
                    f"Learning Rate {learning_rate:.3e}, "
                    f"It/sec {it_sec:.3f}, "
                    f"Tokens/sec {tokens_sec:.3f}, "
                    f"Peak mem {peak_mem:.3f} GB",
                    flush=True,
                )

            if training_callback is not None:
                # Prepare refined training info with hierarchical structure following TRL naming
                train_info = {
                    "iteration": it,
                    "train_loss": train_loss,
                    
                    # Use standardized hierarchical metrics structure
                    "train_metrics": {
                        # Use normalized keys to match TRL structure
                        k.replace("_", "/"): v for k, v in avg_metrics.items()
                    },
                    
                    # System metrics
                    "system": {
                        "learning_rate": learning_rate,
                        "iterations_per_second": it_sec,
                        "tokens_per_second": tokens_sec,
                        "trained_tokens": trained_tokens,
                        "peak_memory_gb": peak_mem,
                    }
                }
                
                training_callback.on_train_loss_report(train_info)

            losses = 0
            n_tokens = 0
            steps = 0
            start = time.perf_counter()

        if it % args.steps_per_save == 0 and rank == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            
            # Ensure parent directory exists
            Path(args.adapter_file).parent.mkdir(parents=True, exist_ok=True)
            
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint), adapter_weights)
            print(
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

    if rank == 0:
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(args.adapter_file), adapter_weights)
        print(f"Saved final weights to {args.adapter_file}.")
