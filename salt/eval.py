"""Closed-loop planning evaluation (CEM model-predictive control).

Examples
--------
    python salt/eval.py --config-name pusht checkpoint=salt/pusht/weights.pt seed=1
    python salt/eval.py --config-name pusht checkpoint=null seed=1      # random policy

``checkpoint`` is resolved relative to ``$STABLEWM_HOME/checkpoints``. Results are appended
to ``$STABLEWM_HOME/results/<env>/<checkpoint or random>/results.txt``; per-episode videos are
written next to it.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from stable_worldmodel.planning import GoalMSE, ShootingCostEvaluator
from torchvision.transforms import v2 as transforms


def image_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def episode_column(dataset) -> str:
    return "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"


def episode_lengths(dataset, episodes):
    episode_idx = dataset.get_col_data(episode_column(dataset))
    step_idx = dataset.get_col_data("step_idx")
    return np.array([np.max(step_idx[episode_idx == ep]) + 1 for ep in episodes])


def build_policy(cfg, dataset):
    """Random policy, or CEM planning with the world model given by ``cfg.checkpoint``."""
    if cfg.checkpoint is None:
        return swm.policy.RandomPolicy()

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        scaler = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        scaler.fit(col_data[~np.isnan(col_data).any(axis=1)])
        process[col] = scaler
        if col != "action":
            process[f"goal_{col}"] = scaler

    transform = {"pixels": image_transform(cfg.eval.img_size), "goal": image_transform(cfg.eval.img_size)}

    model = swm.wm.utils.load_pretrained(cfg.checkpoint).to("cuda").eval()
    model.requires_grad_(False)
    cost = ShootingCostEvaluator(model, GoalMSE())
    solver = hydra.utils.instantiate(cfg.solver, cost=cost)
    return swm.policy.WorldModelPolicy(
        solver=solver, config=swm.PlanConfig(**cfg.plan_config), process=process, transform=transform
    )


@hydra.main(version_base=None, config_path="../configs/eval", config_name="pusht")
def run(cfg: DictConfig):
    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget, (
        "Planning horizon must not exceed the evaluation budget"
    )

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name, keys_to_cache=list(cfg.dataset.keys_to_cache)
    )
    col = episode_column(dataset)
    episodes = np.unique(dataset.get_col_data(col))

    # Sample evaluation start states: any dataset row whose episode still has
    # ``goal_offset_steps`` steps ahead of it. The goal is the observation that many steps later.
    lengths = episode_lengths(dataset, episodes)
    max_start = dict(zip(episodes, lengths - cfg.eval.goal_offset_steps - 1))
    max_start_per_row = np.array([max_start[ep] for ep in dataset.get_col_data(col)])
    valid = np.nonzero(dataset.get_col_data("step_idx") <= max_start_per_row)[0]

    rng = np.random.default_rng(cfg.seed)
    rows = np.sort(valid[rng.choice(len(valid) - 1, size=cfg.eval.num_eval, replace=False)])
    eval_episodes = dataset.get_row_data(rows)[col]
    eval_starts = dataset.get_row_data(rows)["step_idx"]
    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(build_policy(cfg, dataset))

    run_name = "random" if cfg.checkpoint is None else str(Path(cfg.checkpoint).parent)
    results_dir = Path(swm.data.get_cache_dir(sub_folder="results"), cfg.env_name, run_name)
    results_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_starts.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
        video=results_dir,
    )
    elapsed = time.time() - start
    print(metrics)

    with (results_dir / "results.txt").open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        f.write(f"seed: {cfg.seed}\n")
        f.write(f"success_rate: {metrics['success_rate']:.4f}\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {elapsed:.1f} seconds\n")


if __name__ == "__main__":
    run()
