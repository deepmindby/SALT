"""Train SALT or the reproduced LeWM baseline.

Examples
--------
    python salt/train.py --config-name salt data=pusht
    python salt/train.py --config-name lewm data=pusht

Datasets are read from ``$STABLEWM_HOME/datasets`` and checkpoints are written to
``$STABLEWM_HOME/checkpoints/<run_name>``.
"""

from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf, open_dict

from salt.data import SaveCheckpointCallback, column_normalizer, image_preprocessor
from salt.sigreg import SIGReg

OmegaConf.register_new_resolver("eval", eval, replace=True)


def rollout_loss(model, emb, act_emb, history_size):
    """Recursive multi-step objective (Eq. 9): the rollout starts from the first encoded
    latent, every prediction is fed back into the predictor, and all T-1 predicted latents
    are supervised by the encoded targets (not detached)."""
    n_frames = emb.size(1)
    z = emb[:, :1]
    preds = []
    for t in range(n_frames - 1):
        ctx = min(t + 1, history_size)
        step = model.predict(z[:, -ctx:], act_emb[:, t - ctx + 1 : t + 1])
        preds.append(step[:, -1])
        z = torch.cat([z, step[:, -1:]], dim=1)
    return (torch.stack(preds, dim=1) - emb[:, 1:]).pow(2).mean()


def onestep_loss(model, emb, act_emb, history_size, num_preds):
    """One-step objective of LeWM (Eq. 1): predict from encoded latents only."""
    pred = model.predict(emb[:, :history_size], act_emb[:, :history_size])
    return (pred - emb[:, num_preds:]).pow(2).mean()


def forward(self, batch, stage, cfg):
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb, act_emb = output["emb"], output["act_emb"]

    if cfg.objective == "rollout":
        output["pred_loss"] = rollout_loss(self.model, emb, act_emb, cfg.history_size)
    elif cfg.objective == "onestep":
        output["pred_loss"] = onestep_loss(self.model, emb, act_emb, cfg.history_size, cfg.num_preds)
    else:
        raise ValueError(f"Unknown objective: {cfg.objective}")

    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + cfg.sigreg.weight * output["sigreg_loss"]

    self.log_dict(
        {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k},
        on_step=True,
        sync_dist=True,
    )
    return output


@hydra.main(version_base=None, config_path="../configs/train", config_name="salt")
def run(cfg):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    dataset = swm.data.load_dataset(dataset_name, transform=None, **dataset_cfg)

    transforms = [image_preprocessor("pixels", "pixels", cfg.img_size)]
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if not col.startswith("pixels"):
                transforms.append(column_normalizer(dataset, col, col))
        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
    dataset.transform = spt.data.transforms.Compose(*transforms)

    generator = torch.Generator().manual_seed(cfg.split_seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=generator
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=generator
    )
    val_loader = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)

    module = spt.Module(
        model=hydra.utils.instantiate(cfg.model),
        sigreg=SIGReg(knots=cfg.sigreg.knots, num_proj=cfg.sigreg.num_proj),
        forward=partial(forward, cfg=cfg),
        optim={
            "model_opt": {
                "modules": "model",
                "optimizer": dict(cfg.optimizer),
                "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
                "interval": "epoch",
            }
        },
    )

    run_dir = Path(swm.data.get_cache_dir(sub_folder="checkpoints"), cfg.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "train_config.yaml")

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[SaveCheckpointCallback(cfg.run_name, cfg.model)],
        num_sanity_val_steps=1,
        logger=False,
        enable_checkpointing=True,
    )
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
        seed=cfg.seed,
    )
    manager()


if __name__ == "__main__":
    run()
