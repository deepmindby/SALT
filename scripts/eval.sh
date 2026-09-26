#!/usr/bin/env bash
# Standard evaluation protocol (Table 1): H=5, replanning every 5 model steps,
# goal 25 environment steps ahead, 50-step budget, 50 episodes, evaluation seeds 1-3.
#
#   bash scripts/eval.sh salt          # SALT checkpoints  (checkpoints/salt/<env>/weights.pt)
#   bash scripts/eval.sh lewm          # LeWM checkpoints  (checkpoints/lewm/<env>/weights.pt)
#   bash scripts/eval.sh random        # random policy
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:-salt}
for ENV in ${ENVS:-tworoom reacher pusht cube}; do
  for SEED in ${SEEDS:-1 2 3}; do
    if [ "$MODEL" = "random" ]; then
      python salt/eval.py --config-name $ENV checkpoint=null seed=$SEED
    else
      python salt/eval.py --config-name $ENV checkpoint=$MODEL/$ENV/weights.pt seed=$SEED
    fi
  done
done
python scripts/summarize_results.py
