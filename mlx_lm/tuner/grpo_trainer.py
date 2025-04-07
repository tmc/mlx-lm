"""
Core implementation of the Group Relative Policy Optimization (GRPO) training algorithm.

Includes functions for loss calculation, batch iteration, training loop, and evaluation
specific to the GRPO method. Enhanced with detailed logging for validation.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

# Attempt to import wandb - Optional dependency
try:
    import wandb
except ImportError:
    wandb = None


from ..models import cache
from ..generate import generation_stream # Corrected import path
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


@dataclass
class GRPOTrainingArgs(TrainingArgs):
    """
    Training arguments specific to Group Relative Policy Optimization (GRPO).
    Inherits from the base TrainingArgs.
    """
    group_size: int = field(
        default=4,
        metadata={"help": "Number of responses per prompt."},
    )
    beta: float = field(default=0.1, metadata={"help": "KL penalty coefficient."})
    epsilon: float = field(
        default=1e-4, metadata={"help": "The Epsilon for numerical stability."}
    )
    max_completion_length: int = field(
        default=512, metadata={"help": "Maximum number of tokens to generate per response."}
    )
    reference_model_path: str = field(
        default=None,
        metadata={
            "help": "Path to reference model weights. If None, uses the base model as reference."
        },
    )
    temperature: float = field(
        default=1.0,
        metadata={
            "help": "Temperature for sampling during generation."
        },
    )
    reward_weights: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "Weights for combining multiple reward functions. If None, equal weights are used."
        },
    )


def get_per_token_logps(model: nn.Module, inputs: mx.array, lengths: mx.array) -> List[mx.array]:
    """
    Calculate the log probabilities of the target tokens given the inputs.

    Args:
        model (nn.Module): The language model.
        inputs (mx.array): Input token IDs (batch_size, sequence_length).
        lengths (mx.array): The actual sequence lengths for each item in the batch.

    Returns:
        List[mx.array]: A list where each element is a 1D array of log probabilities
                        for the tokens in a sequence, up to its actual length.
    """
    logits = model(inputs).astype(mx.float16) # Ensure consistent type for log_softmax
    # Shift logits and targets for next token prediction log prob calculation
    logits = logits[:, :-1, :]
    targets = inputs[:, 1:]

    per_token_logps = []
    for i in range(logits.shape[0]): # Iterate through batch items
        seq_len = int(lengths[i]) - 1 # Length of target sequence
        if seq_len <= 0:
            per_token_logps.append(mx.array([], dtype=mx.float16)) # Handle empty sequences
            continue
        seq_logits = logits[i, :seq_len]
        seq_targets = targets[i, :seq_len]
        log_probs = nn.log_softmax(seq_logits, axis=-1)
        # Gather the log probabilities of the actual target tokens
        token_log_probs = mx.take_along_axis(
            log_probs, seq_targets.reshape(seq_len, 1), axis=-1
        ).squeeze(-1)
        per_token_logps.append(token_log_probs)
    mx.eval(logits) # Ensure logits computation is done if using async eval
    return per_token_logps


def generate_step(
    prompt: mx.array,
    model: nn.Module,
    max_tokens: int = 256,
    sampler: Optional[Callable] = None,
    logits_processors: Optional[List[Callable]] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[Any] = None,
) -> Generator[Tuple[int, mx.array], None, None]:
    """
    Generates tokens step-by-step using the provided model and prompt.

    Args:
        prompt (mx.array): The initial prompt tokens.
        model (nn.Module): The language model to use for generation.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 256.
        sampler (Optional[Callable], optional): Function to sample the next token from logits.
            Defaults to argmax.
        logits_processors (Optional[List[Callable]], optional): List of functions to modify logits
            before sampling. Defaults to None.
        max_kv_size (Optional[int], optional): Maximum size for the KV cache. Defaults to None.
        prompt_cache (Optional[Any], optional): Pre-computed KV cache. If provided, it's
            updated in-place. Defaults to None.

    Yields:
        Generator[Tuple[int, mx.array], None, None]: Yields tuples of (token_id, log_probabilities).
    """
    tokens = None
    y = prompt
    sampler = sampler or (lambda x: mx.argmax(x, axis=-1)) # Default to argmax sampling
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(model, max_kv_size=max_kv_size)

    def _step(y):
        with mx.stream(generation_stream): # Use dedicated stream for generation
            logits = model(y[None], cache=prompt_cache)
            logits = logits[:, -1, :] # Get logits for the last token
            if logits_processors:
                nonlocal tokens
                tokens = mx.concat([tokens, y]) if tokens is not None else y
                for processor in logits_processors:
                    logits = processor(tokens, logits)
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            next_token = sampler(logprobs)
            # Stop gradient flow for generated tokens and logprobs
            return mx.stop_gradient(next_token), mx.stop_gradient(logprobs.squeeze(0))

    try:
        # Initial step to process the prompt
        with mx.stream(generation_stream):
            y, logprobs = _step(y)
        mx.eval(y, logprobs) # Ensure the first token is computed

        for n in range(max_tokens):
            yield y.item(), logprobs # Yield the computed token and its logprobs
            next_y, next_logprobs = _step(y) # Compute the next token
            mx.eval(next_y, next_logprobs) # Ensure computation is done
            y, logprobs = next_y, next_logprobs # Update state for the next iteration
            if (n + 1) % 32 == 0: # Periodically clear Metal cache
                mx.metal.clear_cache()
    finally:
        mx.metal.clear_cache() # Clear cache at the end


def generate_grpo(
    model: nn.Module,
    tokenizer,
    prompt_tokens: List[List[int]],
    max_tokens: int,
    group_size: int,
    temperature: float,
    batch_size: int,
    end_token: str = "</answer>"
) -> Tuple[List[mx.array], List[str], List[int]]:
    """
    Generates multiple completions per prompt for GRPO training.

    Args:
        model (nn.Module): The policy model to generate completions.
        tokenizer: The tokenizer for decoding generated tokens.
        prompt_tokens (List[List[int]]): A list of tokenized prompts.
        max_tokens (int): Maximum number of tokens to generate for each completion.
        group_size (int): Number of completions to generate per prompt.
        temperature (float): Sampling temperature for generation.
        batch_size (int): Number of prompts to process in parallel during generation.
        end_token (str, optional): The sequence indicating the end of a valid completion.
            Defaults to "</answer>".

    Returns:
        Tuple[List[mx.array], List[str], List[int]]: A tuple containing:
            - List of generated completion token tensors (gradients stopped).
            - List of decoded completion texts.
            - List of indices mapping completions back to their original prompt index.
    """
    try:
        end_sequence = mx.array(tokenizer.encode(end_token))
        total_samples = len(prompt_tokens)
        all_completions = []
        all_completion_texts = []
        batch_indices = [] # Tracks original prompt index for each completion

        # Sampler with temperature
        def temp_sampler(logits):
            return mx.random.categorical(logits / temperature)

        # Process prompts in batches
        for i in range(0, total_samples, batch_size):
            current_batch_size = min(batch_size, total_samples - i)
            batch_prompts = prompt_tokens[i : i + current_batch_size]

            # Pad prompts within the batch to the same length
            max_prompt_len = max(len(p) for p in batch_prompts) if batch_prompts else 0
            if max_prompt_len == 0: continue # Skip if batch is effectively empty

            padded_prompts = []
            for prompt in batch_prompts:
                padding = [tokenizer.pad_token_id] * (max_prompt_len - len(prompt))
                padded_prompts.append(prompt + padding)

            prompt_tensor = mx.stop_gradient(mx.array(padded_prompts))

            if len(prompt_tensor.shape) == 1: # Ensure batch dimension
                prompt_tensor = prompt_tensor[None, :]
            if prompt_tensor.shape[1] == 0: # Skip empty prompts after padding (shouldn't happen)
                continue

            # Repeat each prompt 'group_size' times for multiple generations
            expanded_prompts = mx.repeat(prompt_tensor, group_size, axis=0)
            batch_results = [] # Store completions for the current batch of prompts

            # Generate completions for the expanded prompts
            total_prompt_samples = expanded_prompts.shape[0]
            for prompt_idx in range(total_prompt_samples):
                current_tokens = []
                prompt_cache = cache.make_prompt_cache(model) # Fresh cache for each generation

                for token, _ in generate_step(
                    expanded_prompts[prompt_idx],
                    model,
                    max_tokens=max_tokens,
                    sampler=temp_sampler,
                    prompt_cache=prompt_cache,
                ):
                    if token == tokenizer.eos_token_id: # Stop at EOS
                        break

                    current_tokens.append(token)

                    # Stop if the end sequence is generated
                    if len(current_tokens) >= len(end_sequence) and mx.array_equal(
                        mx.array(current_tokens[-len(end_sequence):]), end_sequence
                    ):
                        break

                if current_tokens: # Store valid completions
                    batch_results.append(mx.array(current_tokens))

            # Map generated completions back to original prompts and store
            if batch_results:
                for j, completion_ids in enumerate(batch_results):
                    original_prompt_idx = i + (j // group_size) # Index in the original prompt_tokens list
                    if original_prompt_idx < total_samples:
                        batch_indices.append(original_prompt_idx)
                        completion_text = tokenizer.decode(completion_ids.tolist())
                        all_completions.append(mx.stop_gradient(completion_ids)) # Store tokens with stopped gradient
                        all_completion_texts.append(completion_text)

            mx.metal.clear_cache() # Clear Metal cache after each batch

    finally:
        mx.metal.clear_cache() # Ensure cache is cleared even if errors occur

    return all_completions, all_completion_texts, batch_indices


def grpo_loss(
    model: nn.Module,
    ref_model: Optional[nn.Module],
    tokenizer,
    batch: Tuple,
    completions: Optional[List[mx.array]] = None,
    completion_texts: Optional[List[str]] = None,
    batch_indices: Optional[List[int]] = None,
    reward_funcs: Optional[List[RewardFunctions]] = None,
    beta: float = 0.1,
    group_size: int = 4,
    epsilon: float = 1e-4,
    max_tokens: int = 64,
    temperature: float = 0.8,
    reward_weights: Optional[List[float]] = None,
    batch_size: int = 1,
    is_validation: bool = False,
) -> Tuple[mx.array, mx.array, Dict[str, Any]]:
    """
    Calculates the GRPO loss and associated metrics for a batch of data.

    Args:
        model (nn.Module): The policy model being trained.
        ref_model (Optional[nn.Module]): The reference model (frozen). If None,
            the policy model's initial state is implicitly used as reference.
        tokenizer: The tokenizer.
        batch (Tuple): A batch containing (prompt_tokens, _, prompt_text, answer_text).
        completions (Optional[List[mx.array]], optional): Pre-generated completions. If None,
            completions will be generated internally. Defaults to None.
        completion_texts (Optional[List[str]], optional): Pre-decoded completion texts.
            Defaults to None.
        batch_indices (Optional[List[int]], optional): Indices mapping completions to original prompts.
            Defaults to None.
        reward_funcs (Optional[List[RewardFunctions]], optional): List of reward functions
            to evaluate completions. Defaults to R1-inspired functions.
        beta (float, optional): KL divergence penalty coefficient. Defaults to 0.1.
        group_size (int, optional): Number of completions generated per prompt. Defaults to 4.
        epsilon (float, optional): Small constant for numerical stability. Defaults to 1e-4.
        max_tokens (int, optional): Max tokens for internal generation if completions not provided.
            Defaults to 64.
        temperature (float, optional): Temperature for internal generation. Defaults to 0.8.
        reward_weights (Optional[List[float]], optional): Weights for combining reward functions.
            Defaults to None (equal weights).
        batch_size (int, optional): Batch size used for internal generation. Defaults to 1.
        is_validation (bool, optional): If True, includes raw data for sample analysis in the return dict
                                        and prints a console sample. Defaults to False.

    Returns:
        Tuple[mx.array, mx.array, Dict[str, Any]]: A tuple containing:
            - The calculated GRPO loss for the batch (scalar).
            - The total number of tokens considered in the loss calculation (scalar).
            - A dictionary of metrics including scalar values (rewards means/stds, KL, avg sequence length)
              and optionally lists of raw data under the 'raw_batch_data' key if is_validation is True.

    Raises:
        ValueError: If no completions are generated or provided, or if reward weights
                    don't match the number of reward functions.
    """
    prompt_tokens, _, prompt_text, answer_text = batch

    # Default reward functions if none provided
    if reward_funcs is None:
        reward_funcs = [
            r1_accuracy_reward_func, r1_int_reward_func,
            r1_strict_format_reward_func, r1_soft_format_reward_func, r1_count_xml
        ]

    # Initialize lists before the conditional assignment
    raw_completions, raw_completion_texts, raw_batch_indices = [], [], []

    # Generate completions if not provided
    if completions is not None and completion_texts is not None and batch_indices is not None:
        raw_completions = completions
        raw_completion_texts = completion_texts
        raw_batch_indices = batch_indices
    else:
        raw_completions, raw_completion_texts, raw_batch_indices = generate_grpo(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            group_size=group_size,
            temperature=temperature,
            batch_size=batch_size
        )

    # Prepare data by grouping completions and aligning prompts/answers
    expanded_answers = []
    expanded_prompts = []
    unique_prompt_indices = sorted(list(set(raw_batch_indices))) # Get unique prompt indices in order
    grouped_completions = {idx: [] for idx in unique_prompt_indices} # Map prompt index to list of completion indices

    # Group completion indices by their original prompt index
    for i, completion_idx in enumerate(raw_batch_indices):
        grouped_completions[completion_idx].append(i)

    # Order completions, texts, and indices based on the original prompt order
    ordered_completions = []
    ordered_completion_texts = []
    ordered_batch_indices = []
    completion_lengths = [] # Store lengths of generated completions
    for prompt_idx in unique_prompt_indices:
        completion_indices_for_prompt = grouped_completions[prompt_idx]
        for idx in completion_indices_for_prompt:
            completion = raw_completions[idx]
            ordered_completions.append(completion)
            completion_lengths.append(completion.shape[0]) # Get length before padding
            ordered_completion_texts.append(raw_completion_texts[idx])
            ordered_batch_indices.append(prompt_idx)
            expanded_prompts.append(prompt_text[prompt_idx]) # Expand prompts to match completions
            expanded_answers.append(answer_text[prompt_idx]) # Expand answers to match completions

    # Use the ordered lists from now on
    all_completions = ordered_completions
    all_completion_texts = ordered_completion_texts
    batch_indices = ordered_batch_indices

    # === Check if completions list is empty after processing ===
    if not all_completions:
        print("[Warning] No valid completions remained after processing/reordering. Skipping loss calculation for this batch.")
        return mx.array(0.0), mx.array(0), {} # Return scalar zeros and empty metrics

    # Pad completions for model input
    max_length = max(completion_lengths) if completion_lengths else 0
    padded_completions = []
    attention_masks = []
    for i, completion_ids in enumerate(all_completions):
        completion_tensor = mx.array(completion_ids.tolist())
        padding_length = max_length - completion_lengths[i] # Use stored length
        if padding_length > 0:
            padding = mx.zeros((padding_length,), dtype=completion_tensor.dtype)
            padded_ids = mx.concatenate([completion_tensor, padding])
            mask = mx.concatenate([mx.ones_like(completion_tensor), mx.zeros_like(padding)])
        else:
            padded_ids = completion_tensor
            mask = mx.ones_like(completion_tensor)
        padded_completions.append(padded_ids)
        attention_masks.append(mask)

    inputs = mx.stack(padded_completions) # This should no longer error
    attention_mask = mx.stack(attention_masks)
    # Use 'lengths' derived from original completion lengths for log prob calc
    logp_lengths = mx.array(completion_lengths)

    # Calculate log probabilities
    token_log_probs = get_per_token_logps(model, inputs, logp_lengths)
    mx.eval(token_log_probs)

    if ref_model is None:
        ref_token_log_probs = [mx.stop_gradient(tlp) for tlp in token_log_probs]
    else:
        ref_token_log_probs = get_per_token_logps(ref_model, inputs, logp_lengths)
        mx.eval(ref_token_log_probs)
        ref_token_log_probs = [mx.stop_gradient(tlp) for tlp in ref_token_log_probs]

    # Pad log probability sequences
    max_len = max(x.shape[0] for x in token_log_probs) if token_log_probs else 0
    padded_log_probs = []
    padded_ref_log_probs = []
    for i in range(len(token_log_probs)):
        seq_len = token_log_probs[i].shape[0]
        padding = mx.zeros((max_len - seq_len,))
        padded_log_probs.append(mx.concatenate([token_log_probs[i], padding]))
        padded_ref_log_probs.append(mx.concatenate([ref_token_log_probs[i], padding]))

    token_log_probs = mx.stack(padded_log_probs)
    ref_token_log_probs = mx.stack(padded_ref_log_probs)

    # Calculate rewards
    all_func_rewards = []
    for reward_func in reward_funcs:
        func_rewards = mx.array(
            reward_func(
                prompts=expanded_prompts,
                completions=all_completion_texts,
                answer=expanded_answers,
            )
        )
        all_func_rewards.append(func_rewards)

    # Combine rewards
    rewards = mx.stack(all_func_rewards, axis=1)
    if reward_weights is not None:
        if len(reward_weights) != len(reward_funcs):
            raise ValueError(
                f"Number of reward weights ({len(reward_weights)}) must match number of reward "
                f"functions ({len(reward_funcs)})"
            )
        reward_weights = mx.array(reward_weights, dtype=mx.float32)
    else:
        reward_weights = mx.ones(len(reward_funcs), dtype=mx.float32)
    rewards = (rewards * mx.expand_dims(reward_weights, 0)).sum(axis=1)

    # Calculate advantages
    num_unique_prompts = len(unique_prompt_indices)
    rewards_by_prompt = [[] for _ in range(num_unique_prompts)]
    for i, prompt_idx in enumerate(batch_indices):
        prompt_position = unique_prompt_indices.index(prompt_idx)
        rewards_by_prompt[prompt_position].append(rewards[i])

    advantages = mx.zeros_like(rewards)
    for i, prompt_rewards in enumerate(rewards_by_prompt):
        if len(prompt_rewards) > 1:
            prompt_rewards_arr = mx.array(prompt_rewards) # Convert list to array for calculation
            mean_reward = mx.mean(prompt_rewards_arr)
            std_reward = mx.std(prompt_rewards_arr)
            indices_for_prompt = [j for j, idx in enumerate(batch_indices) if idx == unique_prompt_indices[i]]
            for j, global_idx in enumerate(indices_for_prompt):
                advantages[global_idx] = (prompt_rewards_arr[j] - mean_reward) / (std_reward + epsilon)
        else:
            idx = batch_indices.index(unique_prompt_indices[i])
            advantages[idx] = 0.0

    # Compute KL divergence
    kl_div = (
        mx.exp(ref_token_log_probs - token_log_probs)
        - (ref_token_log_probs - token_log_probs)
        - 1
    )

    # Mask for valid tokens in the loss calculation (based on original lengths)
    loss_length_mask = mx.arange(inputs.shape[1] - 1)[None, :] < (logp_lengths[:, None] - 1)

    # Compute policy ratio
    policy_ratio = mx.exp(
        mx.array(token_log_probs - mx.stop_gradient(ref_token_log_probs))
    )

    # Apply PPO clipping
    policy_ratio_clipped = mx.clip(policy_ratio, 1 - epsilon, 1 + epsilon)
    unclipped_obj = policy_ratio * advantages.reshape(-1, 1)
    clipped_obj = policy_ratio_clipped * advantages.reshape(-1, 1)
    per_token_loss = -mx.minimum(unclipped_obj, clipped_obj)

    # Add KL penalty
    if beta != 0.0:
        per_token_loss = per_token_loss + beta * kl_div

    # Apply mask and calculate loss
    per_token_loss = per_token_loss * loss_length_mask
    total_valid_tokens = loss_length_mask.sum()
    loss = (per_token_loss * loss_length_mask).sum() / total_valid_tokens if total_valid_tokens > 0 else mx.array(0.0)

    # Calculate scalar metrics for reporting
    mean_kl = ((kl_div * loss_length_mask).sum() / total_valid_tokens).item() if total_valid_tokens > 0 else 0.0
    avg_completion_len = mx.mean(mx.array(completion_lengths)).item() if completion_lengths else 0.0

    reward_metrics = {}
    for i, reward_func in enumerate(reward_funcs):
        func_name = reward_func.__name__
        func_rewards = all_func_rewards[i]
        reward_metrics[f"rewards/{func_name}_mean"] = mx.mean(func_rewards) # Add prefix
        reward_metrics[f"rewards/{func_name}_std"] = mx.std(func_rewards) # Add prefix

    grouped_rewards_mean_list = [mx.mean(mx.array(r)) for r in rewards_by_prompt if r] # Ensure non-empty
    grouped_rewards_std_list = [mx.std(mx.array(r)) if len(r) > 1 else 0.0 for r in rewards_by_prompt]

    metrics = {
        "reward": mx.mean(rewards), # Keep top-level reward
        "reward_std": mx.std(rewards), # Keep top-level reward std
        "rewards/grouped_mean": mx.mean(mx.array(grouped_rewards_mean_list)) if grouped_rewards_mean_list else mx.array(0.0),
        "rewards/grouped_std": mx.mean(mx.array(grouped_rewards_std_list)) if grouped_rewards_std_list else mx.array(0.0),
        "kl": mx.array(mean_kl), # Ensure it's an array for item() later
        "avg_completion_length": mx.array(avg_completion_len), # Added avg length
        **reward_metrics, # Add prefixed individual reward metrics
    }

    # Include raw data for validation callback if requested
    if is_validation:
        metrics["raw_batch_data"] = {
            "prompts": expanded_prompts,
            "completions": all_completion_texts,
            "answers": expanded_answers,
            "combined_rewards": rewards,
            "individual_rewards": {rf.__name__: r for rf, r in zip(reward_funcs, all_func_rewards)},
            "advantages": advantages # Include advantages for sample analysis
        }
        # Print one sample to console for quick check
        if all_completion_texts: # Ensure we have something to print
            print("\n=== Validation Sample Details (Console) ===")
            last_idx = len(all_completion_texts) - 1
            last_prompt_idx = batch_indices[last_idx] if batch_indices else 0
            if last_prompt_idx < len(prompt_text):
                print(f"\n📋 Raw Prompt:\n{prompt_text[last_prompt_idx]}")
                if last_prompt_idx < len(prompt_tokens):
                    actual_prompt = tokenizer.decode(prompt_tokens[last_prompt_idx])
                    print(f"\n🔄 Model Input:\n{actual_prompt}")
            print(f"\n📝 Generation:\n{all_completion_texts[last_idx]}")
            if last_prompt_idx < len(answer_text):
                print(f"\n✅ Answer:\n{answer_text[last_prompt_idx]}")
            if "r1_extract_xml_answer" in globals():
                 print(f"\n🔍 Extracted Answer:\n{r1_extract_xml_answer(all_completion_texts[last_idx])}")
            print(f"\n📈 Combined Reward: {rewards[last_idx].item():.4f}")
            print(f"\n📊 Advantage: {advantages[last_idx].item():.4f}")
            print("\n" + "=" * 35 + "\n")

    mx.metal.clear_cache()

    # Return scalar loss, scalar token count, and metrics dictionary
    return loss, total_valid_tokens, metrics


def iterate_grpo_batches(dataset: List[Tuple], batch_size: int, max_seq_length: int, train: bool = False) -> Generator[Tuple, None, None]:
    """
    Iterates over the GRPO dataset, yielding batches of tokenized data.

    Sorts the dataset by the combined length of prompt and answer tokens
    to potentially improve padding efficiency. Handles distributed training
    by splitting batches across workers.

    Args:
        dataset (List[Tuple]): The dataset, expected to be a list of tuples,
            where each tuple is (prompt_tokens, answer_tokens, prompt_str, answer_str).
        batch_size (int): The size of each batch.
        max_seq_length (int): The maximum sequence length. Prompts longer than this
            will be truncated (with a warning).
        train (bool, optional): If True, shuffles the batch order in each epoch.
            Defaults to False.

    Yields:
        Generator[Tuple, None, None]: Yields batches, where each batch is a tuple:
        (prompts_tokens, answers_tokens, prompts_text, answers_text).

    Raises:
        ValueError: If the dataset format is incorrect, dataset size is smaller
            than batch size, or batch size is not divisible by the number of
            distributed workers.
    """
    if not dataset or not isinstance(dataset[0], tuple) or len(dataset[0]) != 4:
        raise ValueError(
            "Dataset must be list of (prompt_tokens, answer_tokens, prompt_str, answer_str) tuples"
        )

    # Sort indices by combined length of prompt + answer tokens
    def length_key(i):
        # Ensure indices are within dataset bounds
        if i < 0 or i >= len(dataset): return float('inf') # Should not happen with valid indices
        prompt_len = len(dataset[i][0]) if dataset[i][0] is not None else 0
        answer_len = len(dataset[i][1]) if dataset[i][1] is not None else 0
        return prompt_len + answer_len

    idx = sorted(range(len(dataset)), key=length_key)


    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size} "
            f"examples but only has {len(dataset)}."
        )

    # Ensure batch size is divisible by the number of workers in distributed training
    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")

    # Generator function for batch indices, handling distributed training step
    def batch_index_generator():
        for i in range(0, len(idx) - batch_size + 1, batch_size):
            # Ensure the slice indices are valid
            start_idx = i
            end_idx = i + batch_size
            if start_idx < 0 or end_idx > len(idx): continue # Skip invalid range

            yield idx[start_idx : end_idx : step]


    # Loop indefinitely for training, or once for evaluation
    while True:
        # Shuffle indices for training, otherwise process in sorted order
        indices = (
            np.random.permutation(list(batch_index_generator()))
            if train
            else batch_index_generator()
        )

        for batch_idx in indices:
             # Additional check for empty batch_idx which can happen with small datasets/batch sizes
            if len(batch_idx) == 0: # Use len() check for list/array
                continue

            current_batch = [dataset[j] for j in batch_idx if j < len(dataset)] # Check index validity
            if not current_batch: # Skip if batch became empty after index check
                continue

            # Unzip the batch into separate lists
            prompts_tokens = [item[0] for item in current_batch]
            answers_tokens = [item[1] for item in current_batch]
            prompts_text = [item[2] for item in current_batch]
            answers_text = [item[3] for item in current_batch]

            # Warn if any prompt exceeds max_seq_length (truncation happens implicitly later)
            if any(len(p) > max_seq_length for p in prompts_tokens if p is not None):
                print(
                    f"[WARNING] Some prompts are longer than {max_seq_length} tokens. "
                    "Long prompts will be truncated."
                )

            yield prompts_tokens, answers_tokens, prompts_text, answers_text

        # Break after one pass if not in training mode
        if not train:
            break


def evaluate_grpo(
    model: nn.Module,
    ref_model: Optional[nn.Module],
    dataset: List[Tuple],
    tokenizer,
    batch_size: int,
    num_batches: int,
    beta: float,
    epsilon: float,
    group_size: int,
    max_seq_length: int,
    max_tokens: int,
    temperature: float,
    reward_funcs: Optional[List[RewardFunctions]] = None, # Use default if None
    loss_fn: callable = grpo_loss,
    iterate_batches: callable = iterate_grpo_batches,
) -> Tuple[float, int, Dict[str, float], Optional[dict]]:
    """
    Evaluates the model using the GRPO loss function on a validation or test set.

    Args:
        model (nn.Module): The policy model to evaluate.
        ref_model (Optional[nn.Module]): The reference model.
        dataset (List[Tuple]): The evaluation dataset.
        tokenizer: The tokenizer.
        batch_size (int): Batch size for evaluation.
        num_batches (int): Number of batches to evaluate. Use -1 for the entire dataset.
        beta (float): KL divergence penalty coefficient.
        epsilon (float): Stability constant for advantage normalization.
        group_size (int): Number of completions per prompt used during training (or eval).
        max_seq_length (int): Maximum sequence length for input.
        max_tokens (int): Maximum number of tokens to generate for completions during evaluation.
        temperature (float): Sampling temperature for generation during evaluation.
        reward_funcs (Optional[List[RewardFunctions]], optional): List of reward functions.
            Defaults to R1-inspired functions if None.
        loss_fn (callable, optional): The loss function to use. Defaults to grpo_loss.
        iterate_batches (callable, optional): Function to iterate over batches.
            Defaults to iterate_grpo_batches.

    Returns:
        Tuple[float, int, Dict[str, float], Optional[dict]]: A tuple containing:
            - Average loss over the evaluated batches.
            - Total number of tokens processed.
            - Dictionary of average scalar metrics (rewards, KL divergence, etc.).
            - Dictionary containing selected validation samples and raw data for callbacks,
              or None if no batches were successfully evaluated.
    """
    all_losses = 0.0
    ntokens = 0
    accumulated_metrics = {} # Initialize metrics dictionary
    all_raw_batch_data = [] # Accumulate raw data from batches

    # Default reward functions if none provided
    if reward_funcs is None:
        reward_funcs = [
            r1_accuracy_reward_func, r1_int_reward_func,
            r1_strict_format_reward_func, r1_soft_format_reward_func, r1_count_xml
        ]

    # Determine the number of batches to iterate over
    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    # Iterate through the evaluation dataset
    num_evaluated_batches = 0
    for _, batch in zip(
        index_iterator,
        iterate_batches(
            dataset=dataset,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            train=False # Evaluation mode, no shuffling
        ),
    ):
        num_evaluated_batches += 1
        # Calculate loss and metrics for the current batch
        loss, toks, metrics = loss_fn(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            reward_funcs=reward_funcs,
            beta=beta,
            group_size=group_size,
            epsilon=epsilon,
            ref_model=ref_model,
            temperature=temperature,
            max_tokens=max_tokens,
            is_validation=True # Enable collecting raw data and console print
        )

        # Check if metrics dictionary is empty (means loss calculation was skipped)
        if not metrics:
             continue # Skip accumulating results for this batch

        raw_batch_data = metrics.pop("raw_batch_data", None) # Extract raw data if present
        if raw_batch_data:
            all_raw_batch_data.append(raw_batch_data)

        # Accumulate weighted loss and token count
        loss_val = loss.item()
        toks_val = toks.item()
        all_losses += loss_val * toks_val
        ntokens += toks_val

        # Accumulate scalar metrics, weighted by the number of tokens in the batch
        for k, v in metrics.items():
            if k not in accumulated_metrics:
                accumulated_metrics[k] = 0.0
            # Ensure v is a scalar or properly handled array before multiplication
            metric_val = v.item() if isinstance(v, mx.array) else float(v) # Ensure float
            accumulated_metrics[k] += metric_val * toks_val # Accumulate weighted metric value

        mx.eval(loss, toks) # Evaluate accumulated values periodically if needed

    # Check if any batches were actually evaluated
    if num_evaluated_batches == 0 or ntokens == 0:
        print("[Warning] No batches were successfully evaluated.")
        return 0.0, 0, {}, None

    # Aggregate scalar results across distributed workers if applicable
    all_losses = mx.distributed.all_sum(mx.array(all_losses), stream=mx.cpu).item()
    ntokens = mx.distributed.all_sum(mx.array(ntokens), stream=mx.cpu).item()
    avg_metrics = {k: mx.distributed.all_sum(mx.array(v)).item() / ntokens if ntokens > 0 else 0.0
                   for k, v in accumulated_metrics.items()}
    avg_loss = (all_losses / ntokens) if ntokens > 0 else 0.0

    # --- Process accumulated raw data to select samples ---
    selected_samples_info = None
    if all_raw_batch_data:
        # Combine data from all batches
        all_prompts = [item for batch_data in all_raw_batch_data for item in batch_data["prompts"]]
        all_gens = [item for batch_data in all_raw_batch_data for item in batch_data["completions"]]
        all_answers = [item for batch_data in all_raw_batch_data for item in batch_data["answers"]]
        all_combined_rewards = mx.concatenate([batch_data["combined_rewards"] for batch_data in all_raw_batch_data])
        all_advantages = mx.concatenate([batch_data["advantages"] for batch_data in all_raw_batch_data]) # Added
        all_individual_rewards = {}
        for func_name in all_raw_batch_data[0]["individual_rewards"].keys():
             all_individual_rewards[func_name] = mx.concatenate(
                 [batch_data["individual_rewards"][func_name] for batch_data in all_raw_batch_data]
             )

        selected_samples_info = {}
        # Define columns including Advantage
        sample_columns = ["Prompt", "Generation", "Answer", "Combined Reward", "Advantage"]
        for func_name in all_individual_rewards.keys():
            sample_columns.append(f"{func_name}_Reward")

        # Select percentile samples for combined reward
        percentiles_to_log = [10, 50, 90] # Reduced percentiles
        percentile_samples_data = []
        if len(all_combined_rewards) > 0:
            try:
                percentile_values = np.percentile(all_combined_rewards.tolist(), percentiles_to_log)
                for p, p_val in zip(percentiles_to_log, percentile_values):
                    closest_idx = mx.argmin(mx.abs(all_combined_rewards - p_val)).item()
                    if closest_idx < len(all_prompts):
                        row_data = [
                            f"p{p}", all_prompts[closest_idx], all_gens[closest_idx],
                            all_answers[closest_idx], all_combined_rewards[closest_idx].item(),
                            all_advantages[closest_idx].item() # Add advantage
                        ]
                        for func_name, func_rewards in all_individual_rewards.items():
                            # Add individual reward score for this sample
                            row_data.append(func_rewards[closest_idx].item())
                        percentile_samples_data.append(row_data)
            except Exception as e:
                print(f"Warning: Could not compute combined reward percentiles. Error: {e}")
        selected_samples_info["percentile_samples"] = {
            # Update columns to include Percentile label
            "columns": ["Percentile"] + sample_columns,
            "data": percentile_samples_data
        }


        # Select best sample based on combined reward
        if len(all_combined_rewards) > 0:
            best_idx = mx.argmax(all_combined_rewards).item()
            if best_idx < len(all_prompts):
                best_sample_data = [[
                    all_prompts[best_idx], all_gens[best_idx], all_answers[best_idx],
                    all_combined_rewards[best_idx].item(),
                    all_advantages[best_idx].item() # Add advantage
                ]]
                # Add individual reward scores for the best sample
                for func_name, func_rewards in all_individual_rewards.items():
                    best_sample_data[0].append(func_rewards[best_idx].item())
                selected_samples_info["best_sample"] = {"columns": sample_columns, "data": best_sample_data}


    return avg_loss, ntokens, avg_metrics, selected_samples_info


def train_grpo(
    model: nn.Module,
    ref_model: Optional[nn.Module],
    tokenizer,
    optimizer,
    train_dataset: List[Tuple],
    val_dataset: List[Tuple],
    reward_funcs: Optional[List[RewardFunctions]] = None, # Use default if None
    args: GRPOTrainingArgs = GRPOTrainingArgs(),
    loss_fn: callable = grpo_loss,
    iterate_batches: callable = iterate_grpo_batches,
    training_callback: Optional[TrainingCallback] = None,
):
    """
    Main training loop for Group Relative Policy Optimization (GRPO).

    Args:
        model (nn.Module): The policy model to train.
        ref_model (Optional[nn.Module]): The reference model (frozen).
        tokenizer: The tokenizer.
        optimizer: The optimizer for updating model parameters.
        train_dataset (List[Tuple]): The training dataset.
        val_dataset (List[Tuple]): The validation dataset.
        reward_funcs (Optional[List[RewardFunctions]], optional): List of reward functions.
            Defaults to R1-inspired functions if None.
        args (GRPOTrainingArgs, optional): GRPO-specific training arguments.
            Defaults to GRPOTrainingArgs().
        loss_fn (callable, optional): The loss function to use. Defaults to grpo_loss.
        iterate_batches (callable, optional): Function to iterate over batches.
            Defaults to iterate_grpo_batches.
        training_callback (Optional[TrainingCallback], optional): Callback for reporting progress.
            Defaults to None.
    """
    # Default reward functions if none provided
    if reward_funcs is None:
        reward_funcs = [
            r1_accuracy_reward_func, r1_int_reward_func,
            r1_strict_format_reward_func, r1_soft_format_reward_func, r1_count_xml
        ]

    print(
        f"Starting GRPO training with {len(reward_funcs)} reward functions..., iters: {args.iters}"
    )

    # Initialize distributed training if applicable
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    # Enable gradient checkpointing if requested
    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0]) # Assuming checkpointing the first layer is sufficient

    # State for compiled training step
    state = [model.state, optimizer.state, mx.random.state]

    # Define the training step function
    def step(batch):
        # Generate completions first without gradient tracking
        prompt_tokens, targets, prompt_lens, target_lens = batch # Assuming batch structure
        all_completions, all_completion_texts, batch_indices = generate_grpo(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            max_tokens=args.max_completion_length,
            group_size=args.group_size,
            temperature=args.temperature,
            batch_size=args.batch_size # Pass batch size for internal generation batching
        )

        # Calculate loss and gradients using the pre-generated completions
        (loss, toks, metrics), grad = loss_value_and_grad(
            model, # The model passed here should have trainable parameters
            tokenizer=tokenizer,
            batch=(prompt_tokens, targets, prompt_lens, target_lens), # Original batch data
            completions=all_completions,
            completion_texts=all_completion_texts,
            batch_indices=batch_indices,
            reward_funcs=reward_funcs,
            beta=args.beta,
            group_size=args.group_size,
            epsilon=args.epsilon,
            ref_model=ref_model, # Pass the reference model here
        )

        # Check if metrics dictionary is empty (means loss calculation was skipped)
        if not metrics:
            # Return zeros or handle as appropriate if a step failed
            print("[Warning] Training step skipped due to failed completion generation/processing.")
            return mx.zeros(1), mx.zeros(1), {}


        # Average gradients across workers in distributed training
        grad = average_gradients(grad)
        # Update model parameters
        optimizer.update(model, grad)

        return loss, toks, metrics

    # Get the function that computes loss, gradients, and metrics
    loss_value_and_grad = nn.value_and_grad(model, loss_fn)

    # Initialize training statistics
    losses = 0.0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    # Initialize dictionary to accumulate metrics
    accumulated_metrics = {
        "reward": 0.0, "reward_std": 0.0, # Simplified names
        "kl": 0.0, "advantage_mean": 0.0, "avg_completion_length": 0.0
    }
    # Add individual reward metrics dynamically if needed for detailed logging
    for reward_func in reward_funcs:
        func_name = reward_func.__name__
        accumulated_metrics[f"rewards/{func_name}_mean"] = 0.0
        # accumulated_metrics[f"rewards/{func_name}_std"] = 0.0 # Optional

    start = time.perf_counter()

    # Main training loop
    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_batches(
            dataset=train_dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True, # Shuffle batches during training
        ),
    ):
        # Validation step
        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
            model.eval() # Set model to evaluation mode
            tic_eval = time.perf_counter()
            # Capture the sample dictionary from evaluate_grpo
            val_loss, val_ntokens, val_metrics, val_samples = evaluate_grpo(
                model=model,
                dataset=val_dataset,
                loss_fn=loss_fn,
                ref_model=ref_model,
                reward_funcs=reward_funcs,
                tokenizer=tokenizer,
                group_size=args.group_size,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                max_tokens=args.max_completion_length,
                beta=args.beta,
                epsilon=args.epsilon,
                temperature=args.temperature,
                iterate_batches=iterate_batches,
            )
            val_time = time.perf_counter() - tic_eval
            model.train() # Set model back to training mode

            if rank == 0: # Log validation results only on rank 0
                # Build the metric string dynamically using desired summary keys
                val_metric_items = [f"Val loss {val_loss:.3f}"]
                val_metric_items.append(f"Val reward {val_metrics.get('reward', 0.0):.3f}")
                val_metric_items.append(f"Val reward_std {val_metrics.get('reward_std', 0.0):.3f}")
                val_metric_items.append(f"Val kl {val_metrics.get('kl', 0.0):.3f}")
                val_metric_items.append(f"Val avg_completion_length {val_metrics.get('avg_completion_length', 0.0):.1f}")
                val_metrics_str = ", ".join(val_metric_items)
                print(
                    f"Iter {it}: {val_metrics_str}, Val took {val_time:.3f}s",
                    flush=True,
                )

            # Trigger validation callback with both metrics and samples
            if training_callback is not None:
                 # Structure data for wandb logging with desired keys
                val_log_data = {
                    "iteration": it,
                    "val/loss": val_loss,
                    "val/reward": val_metrics.get("reward", None),
                    "val/reward_std": val_metrics.get("reward_std", None),
                    "val/kl": val_metrics.get("kl", None),
                    "val/avg_completion_length": val_metrics.get("avg_completion_length", None),
                    "val/time": val_time,
                     # Optionally log detailed metrics under subgroups
                    "val/rewards/grouped_mean": val_metrics.get("grouped_rewards_mean", None),
                    "val/rewards/grouped_std": val_metrics.get("grouped_rewards_std", None),
                }
                for reward_func in reward_funcs:
                     func_name = reward_func.__name__
                     val_log_data[f"val/rewards/{func_name}_mean"] = val_metrics.get(f"rewards/{func_name}_mean", None)
                     val_log_data[f"val/rewards/{func_name}_std"] = val_metrics.get(f"rewards/{func_name}_std", None)

                val_log_data = {k: v for k, v in val_log_data.items() if v is not None}
                training_callback.on_validation_end(val_log_data, val_samples)

            start = time.perf_counter() # Reset timer after validation

        # Perform one training step
        loss, toks, metrics = step(batch)

        # Skip accumulation if the step was skipped (metrics is empty)
        if not metrics:
            continue

        # Accumulate loss, tokens, and metrics
        losses += loss
        n_tokens += toks
        steps += 1
        for k, v in metrics.items():
             # Ensure key exists before accumulating
            if k in accumulated_metrics:
                accumulated_metrics[k] += v
            elif f"rewards/{k}" in accumulated_metrics: # Handle individual reward metrics
                 accumulated_metrics[f"rewards/{k}"] += v
            else:
                # This case should ideally not happen if metrics dict is consistent
                print(f"[Warning] Metric key '{k}' from step not found in accumulator during training.")
                accumulated_metrics[k] = v # Initialize if missing, might skew average


        # Ensure all computations are finished
        mx.eval(state, losses, n_tokens)
        mx.metal.clear_cache() # Clear cache after step

        # Reporting step
        if (it % args.steps_per_report == 0 or it == args.iters) and steps > 0: # Ensure steps > 0 before reporting
            stop = time.perf_counter()

            # Aggregate and average metrics across workers and steps
            train_loss = mx.distributed.all_sum(losses, stream=mx.cpu).item()
            train_loss /= steps * world_size
            avg_metrics = {
                k: mx.distributed.all_sum(mx.array(v), stream=mx.cpu).item() / (steps * world_size)
                for k, v in accumulated_metrics.items()
            }
            n_tokens_agg = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = steps * world_size / (stop - start) # Use actual steps and world size
            tokens_sec = float(n_tokens_agg) / (stop - start) if (stop - start) > 0 else 0.0
            trained_tokens += n_tokens_agg
            peak_mem = mx.metal.get_peak_memory() / 1e9

            if rank == 0: # Log training progress only on rank 0
                # Build the metric string dynamically, mapping to desired names
                train_metric_items = [f"Train loss {train_loss:.3f}"]
                train_metric_items.append(f"reward {avg_metrics.get('reward', 0.0):.3f}")
                train_metric_items.append(f"reward_std {avg_metrics.get('reward_std', 0.0):.3f}")
                train_metric_items.append(f"kl {avg_metrics.get('kl', 0.0):.3f}")
                train_metric_items.append(f"avg_completion_length {avg_metrics.get('avg_completion_length', 0.0):.1f}")
                train_metrics_str = ", ".join(train_metric_items)

                print(
                    f"Iter {it}: {train_metrics_str}, "
                    f"Learning Rate {learning_rate:.3e}, "
                    f"It/sec {it_sec:.3f}, "
                    f"Tokens/sec {tokens_sec:.3f}, "
                    f"Peak mem {peak_mem:.3f} GB",
                    flush=True,
                )

            # Trigger training callback
            if training_callback is not None:
                 # Structure data for wandb logging with desired keys
                log_data = {
                    "iteration": it,
                    "train/loss": train_loss,
                    "train/reward": avg_metrics.get("reward", None),
                    "train/reward_std": avg_metrics.get("reward_std", None),
                    "train/kl": avg_metrics.get("kl", None),
                    "train/avg_completion_length": avg_metrics.get("avg_completion_length", None),
                    "train/learning_rate": learning_rate,
                    "train/iterations_per_second": it_sec,
                    "train/tokens_per_second": tokens_sec,
                    "train/trained_tokens": trained_tokens,
                    "train/peak_memory_gb": peak_mem,
                    # Optionally log detailed metrics under subgroups
                    "train/rewards/grouped_mean": avg_metrics.get("grouped_rewards_mean", None),
                    "train/rewards/grouped_std": avg_metrics.get("grouped_rewards_std", None),
                }
                for reward_func in reward_funcs:
                     func_name = reward_func.__name__
                     log_data[f"train/rewards/{func_name}_mean"] = avg_metrics.get(f"rewards/{func_name}_mean", None)
                     # log_data[f"train/rewards/{func_name}_std"] = avg_metrics.get(f"rewards/{func_name}_std", None) # Optional

                # Remove None values before logging
                log_data = {k: v for k, v in log_data.items() if v is not None}
                training_callback.on_train_loss_report(log_data)

            # Reset accumulators for the next reporting period
            losses = 0.0
            n_tokens = 0
            steps = 0
            for k in accumulated_metrics: accumulated_metrics[k] = 0.0
            start = time.perf_counter()

        # Saving step
        if it % args.steps_per_save == 0 and rank == 0: # Save only on rank 0
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            adapter_file_path = Path(args.adapter_file)
            mx.save_safetensors(str(adapter_file_path), adapter_weights)
            checkpoint_path = (
                adapter_file_path.parent / f"{it:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint_path), adapter_weights)
            print(
                f"Iter {it}: Saved adapter weights to "
                f"{adapter_file_path} and {checkpoint_path}."
            )

    # Save final adapter weights after training completion
    if rank == 0:
        adapter_weights = dict(tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(str(args.adapter_file), adapter_weights)
        print(f"Saved final weights to {args.adapter_file}.")
