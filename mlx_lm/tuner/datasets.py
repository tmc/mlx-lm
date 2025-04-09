import json
import types
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from transformers import PreTrainedTokenizer


class TextDataset:
    """
    Light-weight wrapper to hold a dataset.
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        text_key: str = "text",
    ):
        self._data = [d for d in data]
        self.tokenizer = tokenizer
        self.text_key = text_key

    def process(self, d):
        d = self.tokenizer.encode(d[self.text_key])
        if d[-1] != self.tokenizer.eos_token_id:
            d.append(self.tokenizer.eos_token_id)
        return d

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


class ChatDataset:
    """
    A dataset for chat data in the format of {"messages": [...]}
    https://platform.openai.com/docs/guides/fine-tuning/example-format
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        chat_key: str = "messages",
        mask_prompt: bool = False,
    ):
        self._data = [d for d in data]
        self.chat_key = chat_key
        self.mask_prompt = mask_prompt
        self.tokenizer = tokenizer

    def process(self, d):
        messages = d[self.chat_key]
        tools = d.get("tools", None)
        tokens = self.tokenizer.apply_chat_template(messages, tools=tools)
        if self.mask_prompt:
            messages = messages[:-1]
            offset = len(self.tokenizer.apply_chat_template(messages, tools=tools))
            return (tokens, offset)
        else:
            return tokens

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


class CompletionsDataset:
    """
    A dataset for prompt-completion data in the format of {"prompt": ..., "completion": ...}
    or using user-provided keys for prompt and completion values
    https://platform.openai.com/docs/guides/fine-tuning/example-format
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        prompt_key: str,
        completion_key: str,
        mask_prompt: bool,
    ):
        self._data = [d for d in data]
        self.prompt_key = prompt_key
        self.completion_key = completion_key
        self.mask_prompt = mask_prompt
        self.tokenizer = tokenizer

    def process(self, d):
        tokens = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": d[self.prompt_key]},
                {"role": "assistant", "content": d[self.completion_key]},
            ],
        )
        if self.mask_prompt:
            offset = len(
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": d[self.prompt_key]}]
                )
            )
            return (tokens, offset)

        return tokens

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


class ConcatenatedDataset:
    def __init__(self, data: List[Any]):
        self._data = data
        self._len = sum(len(d) for d in self._data)

    def __getitem__(self, idx: int):
        for data_idx, data in enumerate(self._data):
            j = idx - len(data)
            if j < 0:
                break
            idx = j
        datum = data[idx]
        datum["_dataset"] = data_idx
        return datum

    def process(self, d):
        return self._data[d["_dataset"]].process(d)

    def __len__(self):
        return self._len


class CacheDataset:
    def __init__(self, data: Any):
        self._data = data
        self._proc_data = [None] * len(data)

    def itemlen(self, idx: int):
        return len(self._data[idx])

    def __getitem__(self, idx: int):
        if self._proc_data[idx] is None:
            self._proc_data[idx] = self._data.process(self._data[idx])
        return self._proc_data[idx]

    def __len__(self):
        return len(self._data)


# Define the default system prompt text
DEFAULT_R1_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The User asks a question, "
    "and the Assistant solves it. The Assistant first thinks about the reasoning process "
    "in the mind and then provides the User with the answer. The reasoning process is "
    "enclosed within <think> </think> and answer is enclosed within <answer> </answer> "
    "tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
)

Message = Dict[str, str]

@dataclass
class GRPODataset:
    """
    Simplified dataset for GRPO training (eager processing, using print).

    Processes prompt/answer or chat formats, handling system prompts and prefill
    according to precedence rules during initialization. Stores processed tokenized data.

    - System Prompt Precedence: forced > key > messages[0] > default_arg > DEFAULT_R1_SYSTEM_PROMPT
    - Prefill Precedence: forced > messages[-1] > default_arg > ""

    Returns: (prompt_tokens, answer_tokens, prompt_str_for_reward, answer_str, system_prompt, prefill)
    """

    # --- Configuration ---
    data: List[Dict[str, Any]] = field(repr=False)
    tokenizer: Any = field(repr=False)
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    messages_key: str = "messages"
    system_prompt_key: Optional[str] = None
    forced_system_prompt: Optional[str] = None
    default_system_prompt_arg: Optional[str] = None
    use_chat_template: bool = True
    forced_assistant_prefill: Optional[str] = None
    default_assistant_prefill_arg: Optional[str] = None
    compress_system_to_user: bool = False

    # --- Internal State ---
    _processed_data: List[Tuple[List[int], List[int], str, str, str, str]] = field(init=False, default_factory=list, repr=False)

    def __post_init__(self):
        """Initialize tokenizer padding and process all data items."""
        if self.tokenizer.pad_token_id is None:
            # Replaced logger.warning
            print("[Warning] Tokenizer missing pad token ID. Using EOS token ID.")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        initial_count = len(self.data) # Use self.data
        # Replaced logger.info
        print(f"[GRPODataset] Initializing and processing {initial_count} items...")
        # Log config concisely
        config_summary = (
            f"Use Chat Template: {self.use_chat_template}, "
            f"Compress System: {self.compress_system_to_user}, "
            f"Forced SysPrompt: {'Yes' if self.forced_system_prompt else 'No'}, "
            f"Forced Prefill: {'Yes' if self.forced_assistant_prefill else 'No'}"
        )
        # Replaced logger.info
        print(f"[GRPODataset] Config: {config_summary}")


        processed_count = 0
        skipped_indices = []
        for index, item in enumerate(self.data): # Use self.data
            try:
                processed_item = self._process_single_item(item, index)
                if processed_item:
                    self._processed_data.append(processed_item)
                    processed_count += 1
                else:
                    skipped_indices.append(index) # Reason already printed in _process_single_item
            except Exception as e:
                # Replaced logger.error with exc_info=True
                print(f"[Error] Error processing item {index}: {e}. Skipping.")
                # Optionally print traceback if needed:
                # import traceback
                # traceback.print_exc()
                skipped_indices.append(index)

        # Report outcome
        skipped_count = len(skipped_indices)
        if skipped_count > 0:
            # Replaced logger.warning
            print(f"[Warning][GRPODataset] Processed {processed_count}/{initial_count} items. "
                  f"{skipped_count} items skipped (e.g., indices: {skipped_indices[:10]}{'...' if skipped_count > 10 else ''}).")
        else:
             # Replaced logger.info
            print(f"[GRPODataset] Successfully processed all {processed_count} items.")

        # Release raw data reference if no longer needed
        del self.data # Use self.data

    def _determine_system_prompt(self, item: Dict[str, Any]) -> Tuple[Optional[str], str]:
        """Helper to determine the effective system prompt (keeps precedence logic clear)."""
        # (No logging needed here, logic remains the same)
        source = "none" ; system_prompt = None

        if self.forced_system_prompt is not None:
            return self.forced_system_prompt, "cli_forced"

        if self.system_prompt_key and self.system_prompt_key in item:
            return str(item[self.system_prompt_key]), "data_key"

        messages = item.get(self.messages_key)
        if isinstance(messages, list) and messages and messages[0].get("role") == "system":
            return messages[0].get("content", ""), "data_messages"

        if self.default_system_prompt_arg is not None:
            return self.default_system_prompt_arg, "cli_default"

        return DEFAULT_R1_SYSTEM_PROMPT, "hardcoded_default"

    def _process_single_item(self, item: Dict[str, Any], index: int) -> Optional[Tuple[List[int], List[int], str, str, str, str]]:
        """Processes a single item, including determining inputs, prefill, answer, and tokenizing."""

        # --- 1. Determine System Prompt ---
        effective_system_prompt, system_source = self._determine_system_prompt(item)

        # --- 2. Extract Content Messages & Base Answer/Prefill ---
        prompt_content_messages = []
        data_prefill_content: Optional[str] = None
        answer_str = ""
        prompt_str_for_reward_base = "" # Base string before adding system prompt potentially

        original_messages = item.get(self.messages_key)
        start_index = 1 if system_source == "data_messages" else 0
        data_ends_with_assistant = False

        if isinstance(original_messages, list) and original_messages:
            # Using 'messages' format
            content_messages = list(original_messages[start_index:])
            if content_messages and content_messages[-1].get("role") == "assistant":
                data_ends_with_assistant = True
                data_prefill_content = content_messages[-1].get("content", "")
                prompt_content_messages = content_messages[:-1]
                answer_str = str(item.get(self.answer_key, data_prefill_content))
            else:
                prompt_content_messages = content_messages
                answer_str = str(item.get(self.answer_key, ""))
            prompt_str_for_reward_base = json.dumps(prompt_content_messages)

        elif self.prompt_key in item:
            # Using 'prompt'/'answer' format
            prompt_content = str(item[self.prompt_key])
            answer_str = str(item.get(self.answer_key, ""))
            prompt_content_messages = [{"role": "user", "content": prompt_content}]
            prompt_str_for_reward_base = prompt_content

        else:
            # Replaced logger.warning
            print(f"[Warning] Skipping item {index}: Lacks required keys ('{self.messages_key}' or '{self.prompt_key}').")
            return None

        # --- 3. Determine Effective Prefill ---
        effective_prefill = ""
        prefill_source = "none"
        if self.forced_assistant_prefill is not None:
            effective_prefill = self.forced_assistant_prefill ; prefill_source = "cli_forced"
        elif data_ends_with_assistant and data_prefill_content is not None:
            effective_prefill = data_prefill_content ; prefill_source = "data_messages"
        elif self.default_assistant_prefill_arg is not None:
            effective_prefill = self.default_assistant_prefill_arg ; prefill_source = "cli_default"

        # --- 4. Validate Answer Existence ---
        if not answer_str and prefill_source not in ["cli_forced", "cli_default", "data_messages"]:
             # Replaced logger.warning
            print(f"[Warning] Skipping item {index}: Missing target answer/prefill. Needs '{self.answer_key}' or a valid prefill source.")
            return None

        # --- 5. Build Final Tokenizer Input Messages & Reward String ---
        messages_for_template: List[Message] = []
        processed_content_messages = [m.copy() for m in prompt_content_messages]

        # Add/Compress System Prompt
        if effective_system_prompt:
            if self.compress_system_to_user:
                first_user_idx = next((i for i, msg in enumerate(processed_content_messages) if msg.get("role") == "user"), -1)
                if first_user_idx != -1:
                    original_content = processed_content_messages[first_user_idx].get("content", "")
                    processed_content_messages[first_user_idx]["content"] = f"System: {effective_system_prompt}\n\nUser: {original_content}"
                else:
                    processed_content_messages.insert(0, {"role": "user", "content": f"System: {effective_system_prompt}\n\nUser: "})
            else:
                messages_for_template.append({"role": "system", "content": effective_system_prompt})

        messages_for_template.extend(processed_content_messages)

        # Add assistant role+content for prefill if using chat template
        if effective_prefill and self.use_chat_template:
             messages_for_template.append({"role": "assistant", "content": effective_prefill})

        # Construct final reward string
        final_prompt_str_for_reward = prompt_str_for_reward_base
        if effective_system_prompt and system_source != "data_messages":
            try:
                reward_list = json.loads(prompt_str_for_reward_base)
                if isinstance(reward_list, list):
                    reward_list.insert(0, {"role": "system", "content": effective_system_prompt})
                    final_prompt_str_for_reward = json.dumps(reward_list)
                else:
                     final_prompt_str_for_reward = f"System: {effective_system_prompt}\n\n{prompt_str_for_reward_base}"
            except json.JSONDecodeError:
                 final_prompt_str_for_reward = f"System: {effective_system_prompt}\n\n{prompt_str_for_reward_base}"


        # --- 6. Tokenization ---
        try:
            prompt_tokens, answer_tokens = self._tokenize(
                messages_for_template, effective_prefill, answer_str, index
            )
        except ValueError as e:
             # Replaced logger.warning
             print(f"[Warning] Skipping item {index}: Tokenization failed. Reason: {e}")
             return None
        except Exception as e: # Catch unexpected tokenizer errors
            # Replaced logger.error
            print(f"[Error] Skipping item {index}: Unexpected error during tokenization: {e}")
            return None

        # --- 7. Return processed data ---
        return (
            prompt_tokens,
            answer_tokens,
            final_prompt_str_for_reward,
            answer_str,
            effective_system_prompt or "",
            effective_prefill
        )

    def _tokenize(
        self, messages_for_template: List[Message], prefill: str, answer_str: str, index: int
    ) -> Tuple[List[int], List[int]]:
        """Tokenizes prompt and answer, handling chat template and fallback."""
        prompt_tokens = None

        if self.use_chat_template:
            try:
                continue_message = bool(prefill)
                prompt_tokens = self.tokenizer.apply_chat_template(
                    messages_for_template,
                    tokenize=True,
                    add_generation_prompt=not continue_message,
                    continue_final_message=prefill,
                )
            except Exception as e:
                 # Replaced logger.warning
                print(f"[Warning][Item {index}] Error applying chat template: {e}. Falling back to basic format.")
                prompt_tokens = None # Trigger fallback

        # Basic formatting fallback
        if prompt_tokens is None:
            prompt_text = ""
            msgs_to_render = messages_for_template
            if prefill and self.use_chat_template and msgs_to_render and msgs_to_render[-1].get("role") == "assistant" and msgs_to_render[-1].get("content") == prefill:
                 msgs_to_render = messages_for_template[:-1]

            for msg in msgs_to_render:
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")
                if role == "System":
                    prompt_text += f"System: {content}\n"
                else:
                    prompt_text += f"{role}: {content}\n"

            prompt_text += "Assistant:"
            if prefill:
                prompt_text += " " + prefill
            prompt_tokens = self.tokenizer.encode(prompt_text)

        # Encode Answer
        answer_tokens = self.tokenizer.encode(answer_str, add_special_tokens=False)

        if not prompt_tokens:
            # This error should ideally be caught by the caller (_process_single_item)
            # but raising it here ensures it's handled if called directly.
            raise ValueError("Failed to produce prompt tokens.")

        return prompt_tokens, answer_tokens

    def __len__(self) -> int:
        """Return the number of successfully processed items."""
        return len(self._processed_data)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int], str, str, str, str]:
        """Returns the pre-processed item at the given index."""
        return self._processed_data[index]
def create_dataset(
    args,
    data,
    tokenizer: PreTrainedTokenizer,
    config,
):
    mask_prompt = getattr(config, "mask_prompt", False)
    prompt_feature = getattr(config, "prompt_feature", "prompt")
    text_feature = getattr(config, "text_feature", "text")
    completion_feature = getattr(config, "completion_feature", "completion")
    chat_feature = getattr(config, "chat_feature", "messages")
    answer_feature = getattr(config, "answer_feature", "answer")
    sample = data[0]
    if prompt_feature in sample and completion_feature in sample:
        return CompletionsDataset(
            data, tokenizer, prompt_feature, completion_feature, mask_prompt
        )
    elif chat_feature in sample:
        return ChatDataset(
            data, tokenizer, chat_key=chat_feature, mask_prompt=mask_prompt
        )
    elif text_feature in sample:
        if mask_prompt:
            raise ValueError("Prompt masking not supported for text dataset.")
        return TextDataset(data, tokenizer, text_key=text_feature)
    elif prompt_feature in sample and answer_feature in sample:
        # Check if we're dealing with GRPO-specific arguments
        prompt_key = getattr(args, "prompt_key", prompt_feature)
        answer_key = getattr(args, "answer_key", answer_feature)
        messages_key = getattr(args, "messages_key", chat_feature)
        
        is_grpo_prompt_answer = prompt_key in sample and answer_key in sample
        is_grpo_messages = messages_key in sample and isinstance(sample[messages_key], list)
        
        if is_grpo_prompt_answer or is_grpo_messages:
            return GRPODataset(
                data=data,
                tokenizer=tokenizer,
                prompt_key=prompt_key,
                answer_key=answer_key,
                messages_key=messages_key,
                system_prompt_key=getattr(args, "system_prompt_key", None),
                # System prompt handling
                forced_system_prompt=getattr(args, "system_prompt", None),
                default_system_prompt_arg=getattr(args, "default_system_prompt", None),
                # Template and prefill handling
                use_chat_template=getattr(args, "use_chat_template", True),
                forced_assistant_prefill=getattr(args, "assistant_prefill", None),
                default_assistant_prefill_arg=getattr(args, "default_assistant_prefill", None),
                # System prompt compression for models like Gemma
                compress_system_to_user=getattr(args, "compress_system_to_user", False)
            )
    else:
        raise ValueError(
            "Unsupported data format, check the supported formats here:\n"
            "https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md#data."
        )


def load_local_dataset(
    args,
    data_path: Path,
    tokenizer: PreTrainedTokenizer,
    config,
):
    def load_subset(path):
        if not path.exists():
            return []
        with open(path, "r") as fid:
            data = [json.loads(l) for l in fid]
        return create_dataset(args, data, tokenizer, config)

    names = ("train", "valid", "test")
    train, valid, test = [load_subset(data_path / f"{n}.jsonl") for n in names]
    return train, valid, test


def load_hf_dataset(
    args,
    data_id: str,
    tokenizer: PreTrainedTokenizer,
    config,
):
    from datasets import exceptions, load_dataset
    try:
        dataset = load_dataset(data_id)
        names = ("train", "valid", "test")
        train, valid, test = [
            (
                create_dataset(args, dataset[n], tokenizer, config)
                if n in dataset.keys()
                else []
            )
            for n in names
        ]
    except exceptions.DatasetNotFoundError:
        raise ValueError(f"Not found Hugging Face dataset: {data_id} .")
    return train, valid, test


def load_custom_hf_dataset(args, tokenizer: PreTrainedTokenizer):
    import datasets

    def create_hf_dataset(dataset_name, config, split, hf_config):
        ds = datasets.load_dataset(
            dataset_name,
            split=split,
            **hf_config,
        )
        return create_dataset(ds, tokenizer, config)

    dataset_collection = args.hf_dataset
    if isinstance(dataset_collection, dict):
        dataset_collection = [dataset_collection]

    collection = []
    for ds in dataset_collection:
        ds_path = ds["path"]
        print(f"Loading Hugging Face dataset {ds_path}.")
        ds["mask_prompt"] = getattr(args, "mask_prompt", False)
        config = types.SimpleNamespace(**ds)
        hf_config = ds.get("config", {})
        if args.train:
            train_split = ds.get("train_split", "train[:80%]")
            valid_split = ds.get("valid_split", "train[-10%:]")
            train = create_hf_dataset(
                ds_path,
                config,
                train_split,
                hf_config,
            )
            valid = create_hf_dataset(
                ds_path,
                config,
                valid_split,
                hf_config,
            )
        else:
            train, valid = [], []

        if args.test:
            test_split = ds.get("test_split")
            test = create_hf_dataset(
                ds_path,
                config,
                test_split,
                hf_config,
            )
        else:
            test = []

        collection.append((train, valid, test))

    if len(collection) == 1:
        return collection[0]

    # Otherwise concatenate them
    return tuple(map(ConcatenatedDataset, zip(*collection)))


def load_dataset(args, tokenizer: PreTrainedTokenizer):
    if getattr(args, "hf_dataset", False):
        train, valid, test = load_custom_hf_dataset(args, tokenizer)
    else:
        data_path = Path(args.data)
        if data_path.exists():
            train, valid, test = load_local_dataset(args, data_path, tokenizer, args.config)
        else:
            print(f"Loading Hugging Face dataset {args.data}.")
            train, valid, test = load_hf_dataset(args, args.data, tokenizer, args.config)

    if args.train and len(train) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if args.train and len(valid) == 0:
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for fine-tuning."
        )
    if args.test and len(test) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test set for evaluation."
        )
    return train, valid, test
