"""Convert a checkpoint produced with the pre-release code base to the released layout.

Renames the transition parameters to the paper notation (W_u, W_v, s, W_g, N, B, b) and
rewrites ``config.json`` to the released module paths. The model definition is unchanged,
so converted weights reproduce the original predictions exactly.

Usage:
    python scripts/convert_legacy_checkpoint.py <legacy_dir> <output_dir>

``legacy_dir`` must contain ``weights.pt`` (or ``weights_epoch_N.pt``) and ``config.json``.
"""

import json
import shutil
import sys
from pathlib import Path

import torch

TARGETS = {
    "jepa.JEPA": "salt.models.JEPA",
    "koopman.KoopmanPredictor": "salt.models.StateAffineTransition",
    "module.ARPredictor": "salt.models.TransformerPredictor",
    "module.Embedder": "salt.models.ActionEncoder",
    "module.MLP": "salt.models.MLP",
}
PREDICTOR_KEYS = {"skew_u": "W_u", "skew_v": "W_v", "G_c.weight": "W_g.weight", "C": "N"}
DROPPED_PREDICTOR_ARGS = {"bilinear_rank", "velocity"}
RENAMED_PREDICTOR_ARGS = {"gate_rank": "num_modes"}


def convert_config(cfg: dict) -> dict:
    out = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            v = convert_config(v)
        if k == "_target_":
            v = TARGETS.get(v, v)
        out[k] = v
    if out.get("_target_") == "salt.models.StateAffineTransition":
        for k in DROPPED_PREDICTOR_ARGS:
            out.pop(k, None)
        for old, new in RENAMED_PREDICTOR_ARGS.items():
            if old in out:
                out[new] = out.pop(old)
    return out


def main(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    weights = sorted(src.glob("weights*.pt"))
    if not weights:
        raise FileNotFoundError(f"No weights*.pt in {src}")
    for path in weights:
        state = torch.load(path, map_location="cpu")
        for old, new in PREDICTOR_KEYS.items():
            if f"predictor.{old}" in state:
                state[f"predictor.{new}"] = state.pop(f"predictor.{old}")
        torch.save(state, dst / path.name)
    cfg = json.loads((src / "config.json").read_text())
    (dst / "config.json").write_text(json.dumps(convert_config(cfg), indent=2) + "\n")
    for extra in ("training_config.yaml", "README.md"):
        if (src / extra).exists():
            shutil.copyfile(src / extra, dst / extra)
    print(f"Converted {len(weights)} checkpoint file(s) -> {dst}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
