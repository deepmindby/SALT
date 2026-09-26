#!/usr/bin/env bash
# Train the reproduced LeWM baseline on the four environments.
# Checkpoints: $STABLEWM_HOME/checkpoints/lewm/<env>/weights.pt
set -euo pipefail
cd "$(dirname "$0")/.."
for ENV in ${ENVS:-tworoom reacher pusht cube}; do
  python salt/train.py --config-name lewm data=$ENV "$@"
done
