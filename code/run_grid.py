"""Run a grid of experiments, either from a list of YAML configs or a sweep spec.

Two modes:

1) List mode — iterate a list of paths:

       python code/run_grid.py --configs configs/weather_patchtst_96.yaml configs/weather_patchtst_336.yaml

2) Sweep mode — start from one base config and override fields:

       python code/run_grid.py --base configs/weather_patchtst_96.yaml \
           --sweep '{"patch_len": [8, 16, 32], "stride": [4, 8, 16]}' --zip

   ``--zip`` zips parallel lists; otherwise we take the cartesian product.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_experiment import load_config
    from train import train_from_config
else:
    from .run_experiment import load_config
    from .train import train_from_config


def _expand(base: dict, sweep: dict, zipped: bool) -> list[dict]:
    keys = list(sweep.keys())
    values = [sweep[k] for k in keys]
    combos = zip(*values) if zipped else itertools.product(*values)
    out = []
    for combo in combos:
        new = dict(base)
        for k, v in zip(keys, combo):
            new[k] = v
        # tag the run_name with the swept fields so artifacts don't collide
        if not new.get("run_name"):
            tag = "_".join(f"{k}{v}" for k, v in zip(keys, combo))
            new["run_name"] = (new.get("dataset", "ds") + "_" + new.get("model", "m")
                                + "_" + tag)
        out.append(new)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=[])
    p.add_argument("--base", default=None)
    p.add_argument("--sweep", default=None, help="JSON dict of field->[values]")
    p.add_argument("--zip", dest="zipped", action="store_true",
                   help="zip parallel lists instead of cartesian product")
    p.add_argument("--project_root", default=None)
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    runs: list[dict] = []
    if args.configs:
        runs.extend(load_config(c) for c in args.configs)
    if args.base and args.sweep:
        base = load_config(args.base)
        sweep = json.loads(args.sweep)
        runs.extend(_expand(base, sweep, args.zipped))
    if not runs:
        p.error("provide --configs or --base/--sweep")

    for cfg in runs:
        if args.project_root:
            cfg["project_root"] = args.project_root
        if args.output_dir:
            cfg["output_dir"] = args.output_dir

    print(f"[grid] {len(runs)} runs queued")
    summaries = []
    for i, cfg in enumerate(runs, 1):
        print(f"\n========== run {i}/{len(runs)}: {cfg.get('run_name') or '(auto)'} ==========")
        try:
            summaries.append(train_from_config(cfg))
        except Exception as e:
            print(f"[grid] run failed: {e}")
            summaries.append({"run_name": cfg.get("run_name"), "error": str(e)})

    print("\n========== grid summary ==========")
    for s in summaries:
        if "error" in s:
            print(f"  {s.get('run_name')}: ERROR {s['error']}")
        else:
            print(f"  {s['run_name']}: mse={s['test_mse_inverse']:.4f} "
                  f"mae={s['test_mae_inverse']:.4f}")


if __name__ == "__main__":
    main()
