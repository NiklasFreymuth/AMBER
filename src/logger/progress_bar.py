import sys

from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm


class CustomProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The train progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        return Tqdm(
            desc=self.validation_description,
            position=(2 * self.process_position + has_main_bar),
            disable=True,
            leave=not has_main_bar,
            dynamic_ncols=True,
            file=sys.stdout,
            bar_format=self.BAR_FORMAT,
        )

    def on_validation_epoch_start(self, trainer, pl_module):
        pass  # Prevents the validation progress bar from being created

    def get_metrics(self, trainer, pl_module):
        # Only keep the metrics you want in the progress bar
        items = super().get_metrics(trainer, pl_module)
        kept_metrics = {"loss": items.get("loss", 0.0), "dataset_size": items.get("dataset_size", 0)}

        return kept_metrics  # Keeps only selected metrics
