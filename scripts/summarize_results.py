"""Aggregate evaluation results into a Table 1 style summary (mean +- std over seeds).

Usage: python scripts/summarize_results.py [results_root]
The default root is $STABLEWM_HOME/results.
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ENVS = ["tworoom", "reacher", "pusht", "cube"]
PATTERN = re.compile(r"^seed: (\d+)\s*$|^success_rate: ([0-9.]+)\s*$", re.M)


def collect(root: Path):
    rows = defaultdict(dict)  # rows[model][env] = {seed: success}
    for path in root.glob("*/**/results.txt"):
        env = path.relative_to(root).parts[0]
        model = path.parent.relative_to(root / env).as_posix().removesuffix("/" + env)
        seed = None
        for m in PATTERN.finditer(path.read_text()):
            if m.group(1) is not None:
                seed = int(m.group(1))
            elif seed is not None:
                rows[model].setdefault(env, {})[seed] = float(m.group(2))
                seed = None
    return rows


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("STABLEWM_HOME", "~/.stable_worldmodel"))
    root = root.expanduser()
    if root.name != "results":
        root = root / "results"
    rows = collect(root)
    if not rows:
        print(f"No results found under {root}")
        return
    header = f"{'model':<24}" + "".join(f"{e:>16}" for e in ENVS) + f"{'avg':>10}"
    print(header)
    print("-" * len(header))
    for model in sorted(rows):
        cells, means = [], []
        for env in ENVS:
            seeds = rows[model].get(env)
            if not seeds:
                cells.append(f"{'-':>16}")
                continue
            v = np.array(list(seeds.values()))
            cells.append(f"{v.mean():6.1f} +- {v.std(ddof=1) if len(v) > 1 else 0:4.1f} (n={len(v)})".rjust(16))
            means.append(v.mean())
        avg = f"{np.mean(means):.1f}" if len(means) == len(ENVS) else "-"
        print(f"{model:<24}" + "".join(cells) + f"{avg:>10}")


if __name__ == "__main__":
    main()
