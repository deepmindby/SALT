"""Data preprocessing and checkpoint utilities shared by training and evaluation."""

import shutil
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from stable_pretraining import data as spt_data
from stable_worldmodel.data import get_cache_dir
from stable_worldmodel.wm.utils import save_pretrained


def image_preprocessor(source: str, target: str, img_size: int = 224):
    """ImageNet normalization followed by a resize, applied to a sample dict column."""
    stats = spt_data.dataset_stats.ImageNet
    to_image = spt_data.transforms.ToImage(**stats, source=source, target=target)
    resize = spt_data.transforms.Resize(img_size, source=source, target=target)
    return spt_data.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer (survives DataLoader worker spawning)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def column_normalizer(dataset, source: str, target: str):
    """Z-score normalizer fitted on a dataset column (NaN rows excluded)."""
    data = torch.from_numpy(np.array(dataset.get_col_data(source)))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return spt_data.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)


class SaveCheckpointCallback(Callback):
    """Saves ``weights_epoch_{n}.pt`` after every epoch and ``weights.pt`` after the last one.

    Checkpoints are written to ``$STABLEWM_HOME/checkpoints/<run_name>/`` together with the
    model config (``config.json``) so that they can be loaded with
    ``stable_worldmodel.wm.utils.load_pretrained``.
    """

    def __init__(self, run_name: str, model_cfg):
        super().__init__()
        self.run_name = run_name
        self.model_cfg = model_cfg

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        save_pretrained(
            pl_module.model,
            run_name=self.run_name,
            config=self.model_cfg,
            filename=f"weights_epoch_{epoch}.pt",
        )
        if epoch == trainer.max_epochs:
            ckpt_dir = Path(get_cache_dir(sub_folder="checkpoints"), self.run_name)
            shutil.copyfile(ckpt_dir / f"weights_epoch_{epoch}.pt", ckpt_dir / "weights.pt")
