import re
import importlib
import inspect
from typing import Callable, Dict, List, Optional, Union

# Allow reward functions to return List[float] or List[Optional[float]]
# None values will be converted to NaN in the metrics calculation
RewardFunctions = Callable[[List[str], List[str], List[str]], List[Union[float, None]]]

def reward_metadata(name: str, max_value: float = 1.0):
    """
    Decorator to add metadata to reward functions.
    
    Args:
        name: A human-readable name for the reward function
        max_value: The maximum possible value this reward function can return
        
    Returns:
        Decorator function that adds these attributes to the reward function
        
    Example:
        @reward_metadata(name="Accuracy Score", max_value=1.0)
        def my_reward_function(prompts, completions, answer):
            # function implementation
            return rewards
    """
    def decorator(func: RewardFunctions) -> RewardFunctions:
        # Add attributes to the function
        func.name = name
        func.max_value = max_value
        return func
    return decorator


def load_reward_functions_from_module(module_path: str) -> Dict[str, RewardFunctions]:
    """
    Dynamically load reward functions from a Python module.
    
    Args:
        module_path: String path to the Python module (e.g., "my_package.my_module")
        
    Returns:
        Dictionary of function name to function object for all callables in the module
        that match the RewardFunctions signature.
    
    Example:
        A custom reward module should contain functions with this signature:
        ```python
        # my_custom_rewards.py
        def my_custom_reward(prompts: List[str], completions: List[str], answer: List[str], **kwargs) -> List[float]:
            # Calculate and return a list of reward values
            return [1.0 if completion.strip() else 0.0 for completion in completions]
        ```
        
        Then specify the module path when running GRPO training:
        ```
        python -m mlx_lm.tuner ... --reward_functions_module="path.to.my_custom_rewards"
        ```
    """
    try:
        module = importlib.import_module(module_path)
        reward_functions = {}
        
        for name, obj in inspect.getmembers(module):
            # Check if it's a callable (function or method)
            if inspect.isfunction(obj):
                # Check if the signature matches our RewardFunctions type
                sig = inspect.signature(obj)
                params = list(sig.parameters.keys())
                
                # Check for prompts, completions, answer parameters
                required_params = ['prompts', 'completions', 'answer']
                if all(param in params for param in required_params):
                    # Add metadata attributes if not present
                    if not hasattr(obj, 'name'):
                        obj.name = name.replace('_', ' ').title()
                    if not hasattr(obj, 'max_value'):
                        obj.max_value = 1.0
                    reward_functions[name] = obj
        
        return reward_functions
            
    except ImportError as e:
        raise ImportError(f"Failed to import module {module_path}: {e}")
    except Exception as e:
        raise Exception(f"Error loading reward functions from {module_path}: {e}")


def _extract_xml_value(tag: str, text: str) -> str:
    try:
        result = text.split(f"<{tag}>")[-1]
        result = result.split(f"</{tag}>")[0]
        return result.strip()
    except:
        print("_extract_xml_value returned empty string")
        return ""

def r1_extract_xml_answer(text: str) -> str:
    try:
        return _extract_xml_value('answer', text)
    except:
        print("r1_extract_xml_answer returned empty string")
        return ""

@reward_metadata(name="Integer Check", max_value=0.5)
def r1_int_reward_func(
    prompts: List[str], completions: List[str], answer: List[str], **kwargs
) -> List[Union[float, None]]:
    if not completions:
        return [0.0] * len(prompts)
    extracted_responses = [r1_extract_xml_answer(r) for r in completions]
    return [0.5 if r and r.isdigit() else 0.0 for r in extracted_responses]


@reward_metadata(name="Accuracy", max_value=2.0)
def r1_accuracy_reward_func(
    prompts: List[str], completions: List[str], answer: List[str], **kwargs
) -> List[Union[float, None]]:
    if not completions or not answer:
        return [0.0] * len(prompts)
    extracted_responses = [r1_extract_xml_answer(r) for r in completions]
    return [
        2.0 if r and a and r == a else 0.0 for r, a in zip(extracted_responses, answer)
    ]

@reward_metadata(name="<think> and <answer> present", max_value=0.5)
def r1_soft_format_reward_func(
    prompts: List[str], completions: List[str], answer: List[str], **kwargs
) -> List[Union[float, None]]:
    if not completions:
        return [0.0] * len(prompts)

    scores = []
    for completion in completions:
        if not completion:
            scores.append(0.0)
            continue

        reason_start = completion.find("<think>")
        reason_end = completion.find("</think>")
        answer_start = completion.find("<answer>")
        answer_end = completion.find("</answer>")

        if (
            reason_start != -1
            and reason_end != -1
            and answer_start != -1
            and answer_end != -1
            and reason_start < reason_end < answer_start < answer_end
        ):
            reason_content = completion[reason_start + 13 : reason_end].strip()
            answer_content = completion[answer_start + 8 : answer_end].strip()
            if reason_content and answer_content:
                scores.append(0.5)
                continue
        scores.append(0.0)
    return scores


@reward_metadata(name="Proper XML Structure", max_value=0.5)
def r1_strict_format_reward_func(
    prompts: List[str], completions: List[str], answer: List[str], **kwargs
) -> List[Union[float, None]]:
    if not completions:
        return [0.0] * len(prompts)
    pattern = r"<think> .*? </think><answer> .*? </answer>"
    matches = [bool(re.search(pattern, r)) if r else False for r in completions]
    return [0.5 if match else 0.0 for match in matches]


@reward_metadata(name="XML Tag Count", max_value=0.5)
def r1_count_xml(
    prompts: List[str], completions: List[str], answer: List[str], **kwargs
) -> List[Union[float, None]]:
    if not completions:
        return [0.0] * len(prompts)
    scores = []
    for text in completions:
        if not text:
            scores.append(0.0)
            continue
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.125
        if text.count("</think>") == 1:
            count += 0.125
        if text.count("<answer>") == 1:
            count += 0.125
        if text.count("</answer>") == 1:
            count += 0.125
        end_text = text.split("</answer>")[-1]
        count -= len(end_text) * 0.001 if len(end_text) > 0 else 0
        scores.append(max(0.0, count))
    return scores
