# SALT: State-Affine Latent Transition for Reliable Visual Planning

## Setup

Linux with an NVIDIA H100 80GB GPU.

```bash
conda env create -f environment.yml
conda activate salt
python -m pip install --upgrade stable-pretraining==0.1.8 scikit-learn==1.7.2

export STABLEWM_HOME=/absolute/path/to/salt_runtime
export MUJOCO_GL=egl
```

The dependency override provides the `vit_hf` API required by the model configs.

## Data

Follow [LeWorldModel](https://github.com/lucas-maes/le-wm) for dataset download and setup.

## Pretrained Models

| Environment  | Checkpoint                                                |
| ------------ | --------------------------------------------------------- |
| Two-Room     | [salt-tworoom](https://huggingface.co/ByDM/salt-tworoom)  |
| Reacher      | [salt-reacher](https://huggingface.co/ByDM/salt-reacher)  |
| PushT        | [salt-pusht](https://huggingface.co/<hf-user>/salt-pusht) |
| OGBench-Cube | [salt-cube](https://huggingface.co/<hf-user>/salt-cube)   |

All checkpoints are also grouped in the [SALT collection](https://huggingface.co/collections/ByDM/salt-6ab809f57ee0273c01af653a).
Download them into the layout expected by `scripts/eval.sh`:

```bash
python - <<'PY'
import os
from huggingface_hub import snapshot_download
for env in ["tworoom", "reacher", "pusht", "cube"]:
    snapshot_download(
        f"<hf-user>/salt-{env}",
        local_dir=os.path.join(os.environ["STABLEWM_HOME"], "checkpoints", "salt", env),
        allow_patterns=["weights.pt", "config.json", "training_config.yaml"],
    )
PY
```

## Training & Evaluation

Train and evaluate SALT on all four environments:

```bash
bash scripts/train_salt.sh 'run_name=salt-trained/${data.env_name}'
bash scripts/eval.sh salt-trained
```

Training uses the paper configuration (10 epochs, batch size 128, bf16). New checkpoints are saved under `$STABLEWM_HOME/checkpoints/salt-trained/`.

To evaluate the pretrained models:

```bash
bash scripts/eval.sh salt
```

Evaluation runs seeds 1–3 with 50 episodes each and horizon 5, then prints the summary. Outputs are saved under `$STABLEWM_HOME/results/`.

## Results

Paper-reported planning success rates (%), mean ± standard deviation:

| Method | Two-Room | Reacher | PushT | OGBench-Cube | Avg. |
|---|---:|---:|---:|---:|---:|
| Random | 2.00±2.00 | 12.00±4.00 | 5.30±2.30 | 42.00±4.00 | 15.33 |
| LeWM (reproduced) | 90.67±4.60 | 85.33±2.30 | 87.33±4.71 | 67.33±4.20 | 82.67 |
| SALT | **98.00±2.00** | **90.70±3.10** | **88.00±5.20** | **94.00±2.00** | **92.68** |

Paper-reported efficiency on an NVIDIA H100 80GB:

| Predictor | Params | Forward (ms) | Planning (s) |
|---|---:|---:|---:|
| LeWM (ViT-S) | 11.58M | 2.36 | 35.2 |
| SALT (Affine) | 0.70M (×16.5) | 0.29 (×8.1) | 11.4 (×3.1) |

Params and forward time include the projection head. Planning time is averaged over four environments and horizons {5, 10, 15, 20}.

## Acknowledgments

This project is based on [LeWorldModel](https://github.com/lucas-maes/le-wm). We thank the authors for open-sourcing their project.
