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
    def __init__(self, 
                 project_name: str, 
                 log_dir: str, 
                 config: dict, 
                 group_name: str = None,
                 wrapped_callback: TrainingCallback = None):
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
        try:
            import mlx.core as mx
            
            try:
                rank = mx.distributed.get_rank()
            except:
                rank = 0
                
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
        except Exception as e:
            print(f"Warning: Failed to log training metrics to WandB: {e}")
        
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
                    # Select samples using strategy from user documentation
                    # Default to 8 samples or user-configured number
                    num_samples = min(8, len(val_samples["prompts"]))
                    indices_to_log = set()
                    
                    if len(val_samples["prompts"]) > 0:
                        # Get indices for min/max samples to understand the range
                        rewards_array = mx.array(val_samples["rewards"])
                        best_idx = rewards_array.argmax().item()
                        worst_idx = rewards_array.argmin().item()
                        indices_to_log.add(best_idx)
                        indices_to_log.add(worst_idx)
                        
                        # Get samples at different quality levels to understand distribution
                        if len(rewards_array) > 3:
                            sorted_indices = mx.argsort(rewards_array)
                            # Get median (50th percentile)
                            median_idx = sorted_indices[len(sorted_indices) // 2].item()
                            indices_to_log.add(median_idx)
                            
                            # Get 25th and 75th percentiles for a more complete distribution view
                            p25_idx = sorted_indices[len(sorted_indices) // 4].item()
                            p75_idx = sorted_indices[3 * len(sorted_indices) // 4].item()
                            indices_to_log.add(p25_idx)
                            indices_to_log.add(p75_idx)
                            
                        # Look for interesting outliers in advantage
                        if "advantages" in val_samples and len(val_samples["advantages"]) > 0:
                            advantages_array = mx.array(val_samples["advantages"])
                            # Add samples with highest advantage (unexpectedly good)
                            high_adv_idx = advantages_array.argmax().item()
                            # Add samples with lowest advantage (unexpectedly bad)
                            low_adv_idx = advantages_array.argmin().item()
                            indices_to_log.add(high_adv_idx)
                            indices_to_log.add(low_adv_idx)
                        
                        # Add samples with extremes for individual reward functions
                        if "individual_rewards" in val_samples and val_samples["individual_rewards"]:
                            for func_name, rewards in val_samples["individual_rewards"].items():
                                func_rewards = mx.array(rewards)
                                # Get best and worst for this specific reward function
                                func_best_idx = func_rewards.argmax().item()
                                func_worst_idx = func_rewards.argmin().item()
                                # Only add if we haven't reached our limit yet
                                if len(indices_to_log) < num_samples:
                                    indices_to_log.add(func_best_idx)
                                if len(indices_to_log) < num_samples:
                                    indices_to_log.add(func_worst_idx)
                        
                        # Add random samples to reach desired count if needed
                        import random
                        if len(indices_to_log) < num_samples:
                            all_indices = set(range(len(val_samples["prompts"])))
                            remaining_indices = all_indices - indices_to_log
                            random_indices = random.sample(
                                remaining_indices, 
                                min(num_samples - len(indices_to_log), len(remaining_indices))
                            )
                            indices_to_log.update(random_indices)
                    
                    # Create rich WandB table for reward function analysis as described in user experience doc
                    columns = [
                        "ID",                # For reference in the table
                        "BatchIdx",          # To track batches
                        "PromptIdx",         # To group completions for same prompt
                        "Prompt",            # Input context
                        "Completion",        # Model generation
                        "Answer",            # Ground truth (if available)
                        "Combined_Reward",   # Final combined reward score
                        "Advantage",         # Calculated advantage vs other completions
                        "Length",            # Completion length
                        "KL_Divergence",     # KL from reference policy (if tracked)
                    ]
                    
                    # Add columns for individual reward functions with consistent naming
                    if "individual_rewards" in val_samples:
                        for func_name in val_samples["individual_rewards"]:
                            # Add standardized column name for this reward function
                            columns.append(f"{func_name}_Reward")
                    
                    data = []
                    
                    # Build rows from selected indices with rich metadata for sorting/filtering in WandB
                    for i, idx in enumerate(sorted(indices_to_log)):
                        # Start with metadata and core values
                        row = [
                            i,  # ID for reference
                            val_samples["batch_indices"][idx] if "batch_indices" in val_samples and idx < len(val_samples["batch_indices"]) else 0,
                            val_samples["prompt_indices"][idx] if "prompt_indices" in val_samples and idx < len(val_samples["prompt_indices"]) else 0,
                            val_samples["prompts"][idx],
                            val_samples["completions"][idx],
                            val_samples["answers"][idx] if "answers" in val_samples and idx < len(val_samples["answers"]) else "",
                            val_samples["rewards"][idx] if isinstance(val_samples["rewards"][idx], (float, int)) else val_samples["rewards"][idx].item(),
                            val_samples["advantages"][idx] if "advantages" in val_samples and idx < len(val_samples["advantages"]) else 0.0,
                            val_samples["completion_lengths"][idx] if "completion_lengths" in val_samples and idx < len(val_samples["completion_lengths"]) else 0,
                            val_samples["kl_div_per_sample"][idx] if "kl_div_per_sample" in val_samples and idx < len(val_samples["kl_div_per_sample"]) else 0.0,
                        ]
                        
                        # Add individual reward values for deep analysis
                        if "individual_rewards" in val_samples:
                            for func_name in val_samples["individual_rewards"]:
                                if idx < len(val_samples["individual_rewards"][func_name]):
                                    value = val_samples["individual_rewards"][func_name][idx]
                                    # Handle different value types
                                    if isinstance(value, (float, int)):
                                        row.append(value)
                                    else:
                                        row.append(value.item())
                                else:
                                    row.append(0.0)  # Default if index out of range
                        
                        data.append(row)
                    
                    # Create and add table
                    table = wandb.Table(columns=columns, data=data)
                    log_data["validation/samples"] = table
                    
                    # Enhanced histograms for detailed reward distribution analysis
                    try:
                        # Primary metrics histograms
                        histograms = {
                            # Combined reward shows overall distribution
                            "validation/dist/combined_reward": wandb.Histogram(val_samples["rewards"]),
                            
                            # Advantage shows how completions compare to others for same prompt
                            "validation/dist/advantage": wandb.Histogram(val_samples["advantages"]) if "advantages" in val_samples else None,
                            
                            # Completion length distribution helps understand generation patterns
                            "validation/dist/completion_length": wandb.Histogram(val_samples["completion_lengths"]) if "completion_lengths" in val_samples else None,
                            
                            # KL divergence shows how much policy deviates from reference
                            "validation/dist/kl_divergence": wandb.Histogram(val_samples["kl_div_per_sample"]) if "kl_div_per_sample" in val_samples else None,
                        }
                        
                        # Add individual reward function histograms for component analysis
                        if "individual_rewards" in val_samples:
                            for func_name, rewards in val_samples["individual_rewards"].items():
                                if rewards and len(rewards) > 0:
                                    # Use standardized naming for reward function histograms
                                    histograms[f"validation/dist/rewards/{func_name}"] = wandb.Histogram(rewards)
                        
                        # Add all valid histograms to log data
                        for hist_name, hist_obj in histograms.items():
                            if hist_obj is not None:
                                log_data[hist_name] = hist_obj
                    except Exception as e:
                        print(f"Warning: Error creating histograms: {e}")
                    
                    # Add summary statistics if available
                    if "stats" in val_samples:
                        for stat_name, stat_value in val_samples["stats"].items():
                            log_data[f"validation/stats/{stat_name}"] = stat_value
                            
                except Exception as e:
                    print(f"Warning: Failed to process validation samples: {e}")
                    import traceback
                    traceback.print_exc()
                    
            # Log data
            try:
                wandb.log(_convert_arrays(log_data), step=val_info.get("iteration"))
            except Exception as e:
                print(f"Warning: Failed to log to WandB: {e}")
            
        # Chain to wrapped callback if exists
        if self.wrapped_callback:
            self.wrapped_callback.on_validation_end(val_info, val_samples)


def _convert_arrays(info):
    """
    Simple conversion of MLX arrays to JSON-serializable types.
    """
    import mlx.core
    import numpy as np

    # Handle None
    if info is None:
        return None
        
    # Handle dicts
    if isinstance(info, dict):
        return {k: _convert_arrays(v) for k, v in info.items()}
        
    # Handle lists
    elif isinstance(info, list):
        return [_convert_arrays(item) for item in info]
        
    # Handle MLX arrays
    elif isinstance(info, mlx.core.array):
        try:
            # For single values, convert to Python scalar
            if info.size == 1:
                return float(info.item())
            # For arrays, convert to list
            return [float(x) for x in info.tolist()]
        except:
            # If conversion fails, use a string placeholder
            return "MLX_ARRAY"
            
    # Handle numpy arrays
    elif isinstance(info, np.ndarray):
        try:
            if info.size == 1:
                return float(info.item())
            return [float(x) for x in info.tolist()]
        except:
            return "NP_ARRAY"
            
    # Handle basic Python types
    elif isinstance(info, (int, float, str, bool)):
        return info
        
    # Last resort: string conversion
    else:
        try:
            return str(info)
        except:
            return "UNKNOWN_TYPE"
