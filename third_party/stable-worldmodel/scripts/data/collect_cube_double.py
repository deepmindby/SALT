from pathlib import Path
import hydra
import numpy as np
from loguru import logger as logging
from omegaconf import DictConfig, OmegaConf
import stable_worldmodel as swm
from stable_worldmodel.envs.ogbench import ExpertPolicy

@hydra.main(version_base=None, config_path='./config', config_name='ogb')
def run(cfg: DictConfig):
    world = swm.World(
        'swm/OGBCube-v0',
        **cfg.world,
        env_type='double',            # 改 1: single -> double
        multiview=False,              # 改 2: 单视角 (对齐训练 h5 的 pixels 列, 省一半空间)
        width=224, height=224,
        visualize_info=False,
        terminate_at_goal=False,
        mode='data_collection',
    )
    options = cfg.get('options')
    options = OmegaConf.to_object(options) if options is not None else None
    rng = np.random.default_rng(cfg.seed)
    world.set_policy(ExpertPolicy())
    world.collect(
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir()) / 'datasets' / f'ogbench/{cfg.out_name}.lance',
        episodes=cfg.num_traj, seed=rng.integers(0, 1_000_000).item(), options=options,
    )
    logging.success('done')

if __name__ == '__main__':
    run()
