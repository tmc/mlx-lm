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


class WandBCallback(TrainingCallback):
    def __init__(self, project_name: str, log_dir: str, config: dict, wrapped_callback: TrainingCallback = None):
        if wandb is None:
            raise ImportError("wandb is not installed. Please install it to use WandBCallback.")
        self.wrapped_callback = wrapped_callback
        wandb.init(project=project_name, dir=log_dir, config=config)

    def on_train_loss_report(self, train_info: dict):
        wandb.log(_convert_arrays(train_info))
        if self.wrapped_callback:
            self.wrapped_callback.on_train_loss_report(train_info)

    def on_val_loss_report(self, val_info: dict):
        wandb.log(_convert_arrays(val_info))
        if self.wrapped_callback:
            self.wrapped_callback.on_val_loss_report(val_info)


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
