try:
    import wandb
except ImportError:
    wandb = None


class TrainingCallback:
    def on_train_loss_report(self, train_info: dict):
        """Called to report training loss at specified intervals."""
        pass

    def on_val_loss_report(self, val_info: dict):
        """Called to report validation loss at specified intervals or the beginning."""
        pass

    def on_validation_end(self, val_info: dict, val_samples: dict = None):
        """Called after validation with both metrics and sample data for rich reporting.

        Args:
            val_info: Dictionary with validation metrics
            val_samples: Dictionary with sample data for visualization (prompts, completions, etc.)
        """
        # Default implementation calls legacy method for backward compatibility
        self.on_val_loss_report(val_info)


class WandBCallback(TrainingCallback):
    def __init__(
        self,
        project_name: str,
        log_dir: str,
        config: dict,
        group_name: str = None,
        wrapped_callback: TrainingCallback = None,
    ):
        """Initialize WandB callback.

        Args:
            project_name: WandB project name
            log_dir: Directory for WandB logs
            config: Configuration to log to WandB
            group_name: Group name for distributed training runs (default: None)
            wrapped_callback: Another callback to chain with
        """
        if wandb is None:
            raise ImportError("wandb is not installed. Please install it to use WandBCallback.")

        self.wrapped_callback = wrapped_callback

        import mlx.core as mx

        # Only log full config from rank 0 to avoid conflicts
        rank = 0
        try:
            rank = mx.distributed.get_rank()
            log_config = config if rank == 0 else None
        except:
            # Not in distributed mode
            log_config = config

        wandb.init(
            project=project_name,
            dir=log_dir,
            config=log_config,
            group=group_name,
        )

        if rank == 0:
            print(f"WandB initialized with project: {project_name}, group: {group_name or 'None'}")

    def on_train_loss_report(self, train_info: dict):
        """Log training metrics with hierarchical structure.

        Args:
            train_info: Dictionary with train metrics and system info
        """
        # Convert train_info for proper logging format
        if "train_metrics" in train_info:
            # Handle hierarchical metrics structure
            log_data = {
                f"train/{k}": v for k, v in train_info["train_metrics"].items()
            }
            # Add non-metrics fields
            for k, v in train_info.items():
                if k != "train_metrics" and k != "system":
                    log_data[f"train/{k}"] = v
                    
            # Handle system metrics separately
            if "system" in train_info:
                for k, v in train_info["system"].items():
                    log_data[f"system/{k}"] = v
        else:
            # Legacy format - convert flat dict to hierarchical
            log_data = {f"train/{k}": v for k, v in train_info.items()}
            
        # Log with step if available
        wandb.log(_convert_arrays(log_data), step=train_info.get("iteration"))
        
        # Chain to wrapped callback if exists
        if self.wrapped_callback:
            self.wrapped_callback.on_train_loss_report(train_info)

    def on_val_loss_report(self, val_info: dict):
        """Legacy method for validation reporting.

        Args:
            val_info: Dictionary with validation metrics
        """
        # Forward to enhanced method if possible
        if hasattr(self, "on_validation_end"):
            self.on_validation_end(val_info, None)
        else:
            # Legacy behavior
            wandb.log(_convert_arrays(val_info), step=val_info.get("iteration"))
            if self.wrapped_callback:
                self.wrapped_callback.on_val_loss_report(val_info)
                
    def on_validation_end(self, val_info: dict, val_samples: dict = None):
        """Enhanced method for rich validation reporting with samples.
        
        Args:
            val_info: Dictionary with validation metrics
            val_samples: Dictionary with sample data for visualization (prompts, completions, etc.)
        """
        import mlx.core as mx
        
        try:
            rank = mx.distributed.get_rank()
        except:
            rank = 0
            
        # Only log rich data from rank 0 in distributed mode
        if rank == 0:
            # Convert val_info for proper logging format
            if "val_metrics" in val_info:
                # Handle hierarchical metrics structure
                log_data = {
                    f"val/{k}": v for k, v in val_info["val_metrics"].items()
                }
                # Add non-metrics fields
                for k, v in val_info.items():
                    if k != "val_metrics" and k != "validation_samples":
                        log_data[f"val/{k}"] = v
            else:
                # Legacy format - convert flat dict to hierarchical
                log_data = {f"val/{k}": v for k, v in val_info.items() if k != "validation_samples"}

            # Process sample data if available
            if val_samples and wandb:
                try:
                    # Create table for prompt/completion examples
                    columns = ["Prompt", "Completion", "Answer", "Reward"]
                    data = []
                    
                    # Extract data for table
                    sample_count = min(10, len(val_samples["prompts"]))  # Limit to 10 samples
                    for i in range(sample_count):
                        data.append([
                            val_samples["prompts"][i],
                            val_samples["completions"][i],
                            val_samples["answers"][i] if "answers" in val_samples else "",
                            val_samples["rewards"][i].item() if "rewards" in val_samples else 0.0
                        ])
                    
                    # Create and add tables with simplified paths
                    # Main comprehensive table with everything
                    table = wandb.Table(columns=columns, data=data)
                    log_data["samples/all"] = table
                    
                    # Special table for top K best samples only (easier to review)
                    if len(data) > 0:
                        # Sort data by reward score (descending)
                        sorted_data = sorted(data, key=lambda x: x[3], reverse=True)
                        # Take top 3 samples
                        best_data = sorted_data[:3]
                        best_table = wandb.Table(columns=columns, data=best_data)
                        log_data["samples/best"] = best_table
                    
                    # Add histogram of rewards if available
                    if "rewards" in val_samples:
                        log_data["validation/reward_dist"] = wandb.Histogram(
                            val_samples["rewards"].tolist()
                        )
                except Exception as e:
                    print(f"Warning: Failed to process validation samples: {e}")
                    
            # Log data
            wandb.log(_convert_arrays(log_data), step=val_info.get("iteration"))
            
        # Chain to wrapped callback if exists
        if self.wrapped_callback:
            self.wrapped_callback.on_validation_end(val_info, val_samples)


def _convert_arrays(info):
    """
    Recursively traverses a dictionary and converts mlx.core.array types to lists.
    This is to prevent wandb's json encoding from failing on mlx.core.array instances.
    """
    import mlx.core
    if isinstance(info, dict):
        return {k: _convert_arrays(v) for k, v in info.items()}
    elif isinstance(info, list):
        return [_convert_arrays(item) for item in info]
    elif isinstance(info, mlx.core.array):
        return info.tolist()
    else:
        return info
