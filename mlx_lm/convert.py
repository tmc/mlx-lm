# Copyright © 2023-2024 Apple Inc.

import argparse
import glob
import shutil
from pathlib import Path
from typing import Callable, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .utils import (
    dequantize_model,
    fetch_from_hub,
    get_model_path,
    quantize_model,
    save_config,
    save_weights,
    upload_to_hub,
)


def mixed_quant_predicate_builder(
    recipe: str, model: nn.Module
) -> Callable[[str, nn.Module, dict], Union[bool, dict]]:

    if recipe == "mixed_2_6":
        low_bits = 2
    elif recipe == "mixed_3_6":
        low_bits = 3
    elif recipe == "mixed_4_6":
        low_bits = 4
    else:
        raise ValueError("Invalid quant recipe {recipe}")
    high_bits = 6
    group_size = 64

    down_keys = [k for k, _ in model.named_modules() if "down_proj" in k]
    if len(down_keys) == 0:
        raise ValueError("Model does not have expected keys for mixed quant.")

    # Look for the layer index location in the path:
    for layer_location, k in enumerate(down_keys[0].split(".")):
        if k.isdigit():
            break
    num_layers = len(model.layers)

    def mixed_quant_predicate(
        path: str,
        module: nn.Module,
        config: dict,
    ) -> Union[bool, dict]:
        """Implements mixed quantization predicates with similar choices to, for example, llama.cpp's Q4_K_M.
        Ref: https://github.com/ggerganov/llama.cpp/blob/917786f43d0f29b7c77a0c56767c0fa4df68b1c5/src/llama.cpp#L5265
        By Alex Barron: https://gist.github.com/barronalex/84addb8078be21969f1690c1454855f3
        """

        if not hasattr(module, "to_quantized"):
            return False

        index = (
            int(path.split(".")[layer_location])
            if len(path.split(".")) > layer_location
            else 0
        )
        use_more_bits = (
            index < num_layers // 8
            or index >= 7 * num_layers // 8
            or (index - num_layers // 8) % 3 == 2
        )
        if "v_proj" in path and use_more_bits:
            return {"group_size": group_size, "bits": high_bits}
        if "down_proj" in path and use_more_bits:
            return {"group_size": group_size, "bits": high_bits}
        if "lm_head" in path:
            return {"group_size": group_size, "bits": high_bits}

        return {"group_size": group_size, "bits": low_bits}

    return mixed_quant_predicate


QUANT_RECIPES = ["mixed_2_6", "mixed_3_6", "mixed_4_6"]


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
    dtype: str = "float16",
    upload_repo: str = None,
    revision: Optional[str] = None,
    dequantize: bool = False,
    quant_predicate: Optional[
        Union[Callable[[str, nn.Module, dict], Union[bool, dict]], str]
    ] = None,
):
    # Check the save path is empty
    if isinstance(mlx_path, str):
        mlx_path = Path(mlx_path)

    if mlx_path.exists():
        raise ValueError(
            f"Cannot save to the path {mlx_path} as it already exists."
            " Please delete the file/directory or specify a new path to save to."
        )

    print("[INFO] Loading")
    model_path = get_model_path(hf_path, revision=revision)
    model, config, tokenizer = fetch_from_hub(model_path, lazy=True)

    if isinstance(quant_predicate, str):
        quant_predicate = mixed_quant_predicate_builder(quant_predicate, model)


    if config.get("model_type") == "llama4":
        print("[INFO] Using Llama 4 specific quantization predicate.")
        num_layers = config["text_config"]["num_hidden_layers"] # Get num_layers from text_config
        # Assuming SwitchGLU is used as in the draft llama4.py
        target_patterns = [
            ".feed_forward.experts.gate_proj",
            ".feed_forward.experts.up_proj",
            ".feed_forward.experts.down_proj",
            # Add shared_expert if applicable and desired
            # ".feed_forward.shared_expert.gate_proj",
            # ".feed_forward.shared_expert.up_proj",
            # ".feed_forward.shared_expert.down_proj",
        ]

        def llama4_quant_predicate(path: str, module: nn.Module, config: dict) -> Union[bool, dict]:
            # Skip first and last layer blocks entirely
            if path.startswith(f"language_model.model.layers.0.") or \
               path.startswith(f"language_model.model.layers.{num_layers - 1}."):
                 return False
            # Check if it's one of the target MoE FFN linear layers
            if any(p in path for p in target_patterns):
                 # Can return True for default group/bits, or dict for custom per-layer
                 return True
            # Quantize only specific layer types if desired (e.g., only Linear)
            # if not isinstance(module, (nn.Linear, nn.QuantizedLinear, SwitchLinear, QuantizedSwitchLinear)):
            #     return False
            # Default: Don't quantize other layers
            return False
        def llama4_quant_predicate_wrap(path: str, module: nn.Module, config: dict) -> Union[bool, dict]:
            r = llama4_quant_predicate(path, module, config)
            print(f"quant-pred: {path}: {r}")
            return r
        quant_predicate = llama4_quant_predicate_wrap

    weights = dict(tree_flatten(model.parameters()))
    dtype = getattr(mx, dtype)
    if hasattr(model, "cast_predicate"):
        cast_predicate = model.cast_predicate()
    else:
        cast_predicate = lambda _: True
    weights = {
        k: v.astype(dtype) if cast_predicate(k) else v for k, v in weights.items()
    }

    if quantize and dequantize:
        raise ValueError("Choose either quantize or dequantize, not both.")

    if quantize:
        print("[INFO] Quantizing")
        model.load_weights(list(weights.items()))
        weights, config = quantize_model(
            model, config, q_group_size, q_bits, quant_predicate=quant_predicate
        )

    if dequantize:
        print("[INFO] Dequantizing")
        model = dequantize_model(model)
        weights = dict(tree_flatten(model.parameters()))

    del model
    save_weights(mlx_path, weights, donate_weights=True)

    py_files = glob.glob(str(model_path / "*.py"))
    for file in py_files:
        shutil.copy(file, mlx_path)

    tokenizer.save_pretrained(mlx_path)

    save_config(config, config_path=mlx_path / "config.json")

    if upload_repo is not None:
        upload_to_hub(mlx_path, upload_repo, hf_path)


def configure_parser() -> argparse.ArgumentParser:
    """
    Configures and returns the argument parser for the script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face model to MLX format"
    )

    parser.add_argument("--hf-path", type=str, help="Path to the Hugging Face model.")
    parser.add_argument(
        "--mlx-path", type=str, default="mlx_model", help="Path to save the MLX model."
    )
    parser.add_argument(
        "-q", "--quantize", help="Generate a quantized model.", action="store_true"
    )
    parser.add_argument(
        "--q-group-size", help="Group size for quantization.", type=int, default=64
    )
    parser.add_argument(
        "--q-bits", help="Bits per weight for quantization.", type=int, default=4
    )
    parser.add_argument(
        "--quant-predicate",
        help=f"Mixed-bit quantization recipe.",
        choices=QUANT_RECIPES,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--dtype",
        help="Type to save the non-quantized parameters.",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-d",
        "--dequantize",
        help="Dequantize a quantized model.",
        action="store_true",
        default=False,
    )
    return parser


def main():
    parser = configure_parser()
    args = parser.parse_args()
    convert(**vars(args))


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.convert ...` directly is deprecated."
        " Use `mlx_lm.convert ...` or `python -m mlx_lm convert ...` instead."
    )
    main()
