#!/usr/bin/env bash
# Train SALT on the four environments with the paper configuration (Table 1).
# Checkpoints: $STABLEWM_HOME/checkpoints/salt/<env>/weights.pt
set -euo pipefail
cd "$(dirname "$0")/.."
for ENV in ${ENVS:-tworoom reacher pusht cube}; do
  python salt/train.py --config-name salt data=$ENV "$@"
done
