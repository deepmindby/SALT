"""Compose the released Hydra configs and instantiate the models (no data or GPU needed)."""

from pathlib import Path

import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
OmegaConf.register_new_resolver("eval", eval, replace=True)


def compose_train(name, env):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS / "train")):
        return compose(config_name=name, overrides=[f"data={env}"])


def test_salt_config_matches_paper():
    cfg = compose_train("salt", "pusht")
    assert cfg.objective == "rollout" and cfg.history_size + cfg.num_preds == 6
    assert cfg.data.dataset.num_steps == 6
    assert cfg.trainer.max_epochs == 10 and cfg.sigreg.weight == 0.09
    cfg.model.action_encoder.input_dim = 10
    model = instantiate(cfg.model)
    assert isinstance(model.pred_proj, nn.Identity)
    assert sum(p.numel() for p in model.predictor.parameters()) == 703_872


def test_lewm_config_instantiates():
    cfg = compose_train("lewm", "cube")
    assert cfg.objective == "onestep" and cfg.data.dataset.num_steps == 4
    cfg.model.action_encoder.input_dim = 25
    model = instantiate(cfg.model)
    assert not isinstance(model.pred_proj, nn.Identity)


def test_eval_configs_compose():
    for env in ["tworoom", "reacher", "pusht", "cube"]:
        with initialize_config_dir(version_base=None, config_dir=str(CONFIGS / "eval")):
            cfg = compose(config_name=env, overrides=["seed=1"])
        assert cfg.plan_config.horizon == 5 and cfg.eval.goal_offset_steps == 25
        assert cfg.solver.num_samples == 300 and cfg.solver.n_steps == 30 and cfg.solver.topk == 30
