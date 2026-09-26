# SALT: State-Affine Latent Transition for Reliable Visual Planning

## Setup

Linux with an NVIDIA H100 GPU. Run from the repository root:

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

Deploy the extracted `SALT_weights` bundle:

```bash
export SALT_WEIGHTS_DIR=/absolute/path/to/SALT_weights
(cd "$SALT_WEIGHTS_DIR" && sha256sum -c SHA256SUMS)
for env in tworoom reacher pusht cube; do
  mkdir -p "$STABLEWM_HOME/checkpoints/salt/$env"
  cp "$SALT_WEIGHTS_DIR/salt-$env/"{weights.pt,config.json,training_config.yaml} \
     "$STABLEWM_HOME/checkpoints/salt/$env/"
done
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
