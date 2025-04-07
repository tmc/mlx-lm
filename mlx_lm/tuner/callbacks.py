"""
Callback definitions for reporting training progress and integrating with
tools like Weights & Biases.
"""
from typing import Optional, Dict, Any

try:
    import wandb
except ImportError:
    wandb = None

import mlx.core as mx
import numpy as np # Needed for percentile calculation


class TrainingCallback:
    """
    Base class for training callbacks.

    Subclass this to implement custom reporting or actions during training.
    """

    def on_train_loss_report(self, train_info: dict):
        """
        Called when a training loss report is generated.

        Args:
            train_info (dict): Dictionary containing training metrics like
                iteration, loss, learning rate, throughput, etc., potentially
                structured with prefixes like 'train/reward'.
        """
        pass

    def on_val_loss_report(self, val_info: dict, val_samples: Optional[dict] = None):
        """
        (Deprecated) Called when scalar validation metrics report is generated.
        Prefer on_validation_end.

        Args:
            val_info (dict): Dictionary containing scalar validation metrics.
        """
        pass

    def on_validation_end(self, val_info: dict, val_samples: Optional[dict] = None):
        """
        Called at the end of a validation phase, providing both aggregated scalar
        metrics and potentially detailed sample information.

        Args:
            val_info (dict): Dictionary containing scalar validation metrics, potentially
                             structured with prefixes like 'val/reward'.
            val_samples (Optional[dict]): Dictionary containing selected validation samples.
                                          Defaults to None.
        """
        pass


class WandBCallback(TrainingCallback):
    """
    A training callback that logs metrics and samples to Weights & Biases.

    Requires the `wandb` library to be installed (`pip install wandb`). Assumes
    metrics are passed with desired WandB key structure (e.g., 'train/loss').

    Args:
        project_name (str): The name of the WandB project.
        log_dir (str): Directory where WandB logs should be stored locally.
        config (dict): A dictionary containing hyperparameters or other run
            configuration details to log to WandB.
        wrapped_callback (TrainingCallback, optional): Another callback to call
            after logging to WandB. Defaults to None.
    """

    def __init__(
        self,
        project_name: str,
        log_dir: str,
        config: dict,
        wrapped_callback: TrainingCallback = None,
    ):
        if wandb is None:
            raise ImportError(
                "wandb is not installed. Please install it to use WandBCallback."
            )
        self.wrapped_callback = wrapped_callback
        wandb.init(project=project_name, dir=log_dir, config=config)

    def on_train_loss_report(self, train_info: dict):
        """Logs training information directly to WandB."""
        wandb.log(_convert_arrays(train_info)) # Log directly, assumes keys are structured
        if self.wrapped_callback:
            self.wrapped_callback.on_train_loss_report(train_info)

    def on_val_loss_report(self, val_info: dict):
        """(Deprecated) Called by default on_validation_end if not overridden."""
        if self.wrapped_callback:
            self.wrapped_callback.on_val_loss_report(val_info)

    def on_validation_end(self, val_info: dict, val_samples: Optional[dict] = None):
        """Logs aggregated validation metrics and detailed samples to WandB."""
        log_data = _convert_arrays(val_info) # Convert scalar metrics

        # Log sample tables if available
        if val_samples and wandb:
             # Log combined reward percentile samples table
            if "percentile_samples" in val_samples and val_samples["percentile_samples"]:
                percentile_table = wandb.Table(
                    columns=val_samples["percentile_samples"]["columns"], # Use dynamic columns
                    data=val_samples["percentile_samples"]["data"]
                )
                log_data["validation/combined_reward_percentile_samples"] = percentile_table

            # Log best sample table
            if "best_sample" in val_samples and val_samples["best_sample"]:
                best_sample_table = wandb.Table(
                    columns=val_samples["best_sample"]["columns"], # Use dynamic columns
                    data=val_samples["best_sample"]["data"]
                )
                log_data["validation/best_sample"] = best_sample_table

        # Log everything (scalars and potentially tables)
        wandb.log(log_data)

        # Call wrapped callback if exists
        if self.wrapped_callback:
            # Pass the original (non-converted) info to the wrapped callback
            self.wrapped_callback.on_validation_end(val_info, val_samples)


def _convert_arrays(info):
    """
    Recursively traverses a dictionary or list and converts mlx.core.array types
    to native Python types (lists or scalars) suitable for JSON serialization
    required by WandB.

    Args:
        info (Any): The data structure (dict, list, array, or other) to convert.

    Returns:
        Any: The converted data structure with mlx arrays replaced by native types.
    """
    if isinstance(info, dict):
        return {k: _convert_arrays(v) for k, v in info.items()}
    elif isinstance(info, list):
        return [_convert_arrays(item) for item in info]
    elif isinstance(info, mx.array):
        # Convert scalar arrays to Python scalars, others to lists
        return info.item() if info.size == 1 else info.tolist()
    else:
        return info
