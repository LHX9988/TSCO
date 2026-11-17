#!/usr/bin/env python3
"""Run real sensitivity simulations; launch from the installed project root.

Examples
--------
python examples/run_studies.py --data data/processed --checkpoint outputs/train/best_feasible.pt \
    --grid res_capacity --values .6 .8 1 1.2 1.4 --seeds 0 1 2 3 4 5 6 7 8 9 \
    --hours 168 --output outputs/res_sensitivity --plots

python examples/run_studies.py --data data/processed --checkpoint outputs/train/best_feasible.pt \
    --grid carbon_budget --values 1300 1375 1450 1525 1600 --hours 168 \
    --seeds 0 1 2 3 4 5 6 7 8 9 --output outputs/budget_sensitivity

Reward sensitivity uses a JSON map of multiplier -> separately trained path:
{"0.5":"reward_cost_05/seed_{seed}/best_feasible.pt", ...}.
Relative paths in that map are resolved against the map file's directory.
Each checkpoint must contain matching reward weights and the same frozen
normalization per paired seed. A single unchanged checkpoint is rejected.
"""
from pathlib import Path
import argparse
import json
import sys

# Also permit direct execution before editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tsco.config import load_config
from tsco.data import load_dataset
from tsco.studies import DEFAULT_GRIDS, StudyRunner, plot_study


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config")
    checkpoints = parser.add_mutually_exclusive_group(required=True)
    checkpoints.add_argument("--checkpoint")
    checkpoints.add_argument("--checkpoint-map", help="JSON multiplier->separately trained checkpoint for reward grids")
    parser.add_argument("--grid", choices=DEFAULT_GRIDS, required=True)
    parser.add_argument("--values", nargs="+", type=float)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--hours", type=float, default=168.)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--planning", choices=("benders", "extensive"), default="benders")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    training = load_dataset(args.data, "train", config)
    test = load_dataset(args.data, "test", config)
    checkpoint = args.checkpoint
    if args.checkpoint_map:
        file = Path(args.checkpoint_map)
        mapping = json.loads(file.read_text())
        checkpoint = {key: str(Path(value) if Path(value).is_absolute() else file.parent/value)
                      for key, value in mapping.items()}
    try:
        runner = StudyRunner(config, training, test, checkpoint, args.device, args.planning)
        runs = runner.run_grid(args.grid, args.values, args.seeds, args.output, args.hours)
        if args.plots:
            print(plot_study(args.output))
        print(json.dumps({"runs": len(runs), "failed": sum(bool(r["failed"]) for r in runs),
                          "output": str(Path(args.output).resolve())}, indent=2))
        return 2 if any(r["failed"] for r in runs) else 0
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
