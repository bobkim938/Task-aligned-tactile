"""
Section 3-C
Usage:
  python evaluation/per_object_runs.py # wrench
  python evaluation/per_object_runs.py --object lego_duplo  # substring; pools all matches

Reads:
  results/downstream_performance/multiseed_eval_all_raw.csv   (written by evaluation/evaluation.py)

Writes:
  results/downstream_performance/per_object_runs_<object>.csv
  stdout: per-run table
"""

import argparse
import csv
from collections import defaultdict

import numpy as np

from per_category_analysis import (
    MODE_LABEL,
    MODE_ORDER,
    PRIMARY_TRAIN_SEED,
    RAW_CSV,
    RESULTS_DIR,
    base_mode,
    load_raw,
    normalize_model_id,
    train_seed_of,
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--object", default="wrench",
                    help="substring of the object name(s), e.g. 'wrench' or 'lego_duplo' "
                         "(all matching objects are pooled; default: wrench)")
    args = ap.parse_args()

    if not RAW_CSV.exists():
        raise FileNotFoundError(f"{RAW_CSV} not found. Run evaluation/evaluation.py first.")

    rows = [r for r in load_raw(RAW_CSV) if args.object in normalize_model_id(r["model_id"])]
    if not rows:
        raise SystemExit(f"No objects matching '{args.object}' in {RAW_CSV}")
    objects = sorted({normalize_model_id(r["model_id"]) for r in rows})

    grp = defaultdict(list)  # (cond, eval_seed) -> list of 0/1
    for r in rows:
        grp[(r["condition"], r["seed"])].append(r["success"])

    # run rate = mean over eval seeds of the per-seed success rate
    run_rate, run_episodes = {}, {}
    for cond in {c for c, _ in grp}:
        seeds = sorted(s for c, s in grp if c == cond)
        run_rate[cond] = float(np.mean([100.0 * np.mean(grp[(cond, s)]) for s in seeds]))
        run_episodes[cond] = sum(len(grp[(cond, s)]) for s in seeds)

    def train_seed(cond):
        seed = train_seed_of(cond)
        return seed if seed is not None else PRIMARY_TRAIN_SEED

    modes = sorted(
        {base_mode(c) for c in run_rate},
        key=lambda m: MODE_ORDER.index(m) if m in MODE_ORDER else len(MODE_ORDER),
    )
    # primary run first, then robustness runs in seed order
    train_seeds = sorted({train_seed(c) for c in run_rate},
                         key=lambda s: (s != PRIMARY_TRAIN_SEED, s))

    table = []  # one entry per mode
    for mode in modes:
        by_seed = {train_seed(c): run_rate[c] for c in run_rate if base_mode(c) == mode}
        rates = [by_seed[s] for s in train_seeds if s in by_seed]
        table.append(dict(
            mode=mode,
            by_seed=by_seed,
            mean=float(np.mean(rates)),
            std=float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
            n_runs=len(rates),
        ))

    # ---- Print table ----
    eps = sorted(set(run_episodes.values()))
    label_w = max(len(MODE_LABEL.get(m, m)) for m in modes) + 2
    table_w = label_w + 10 * len(train_seeds) + 28
    print(f"Object(s): {', '.join(objects)}")
    print(f"Episodes per run: {', '.join(map(str, eps))}")
    print("=" * table_w)
    print(f"{'Condition':<{label_w}}" + "".join(f"{'seed' + str(s):>10}" for s in train_seeds)
          + f"{'mean +/- std over runs':>24}{'N':>4}")
    print("-" * table_w)
    for t in table:
        cells = "".join(
            f"{t['by_seed'][s]:>10.1f}" if s in t["by_seed"] else f"{'--':>10}"
            for s in train_seeds
        )
        print(f"{MODE_LABEL.get(t['mode'], t['mode']):<{label_w}}{cells}"
              f"{t['mean']:>16.1f} +/- {t['std']:<4.1f}{t['n_runs']:>4}")
    print("=" * table_w)

    # ---- Write CSV ----
    out_csv = RESULTS_DIR / f"per_object_runs_{args.object}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition"] + [f"seed{s}_success_pct" for s in train_seeds]
                   + ["mean_success_pct", "std_success_pct", "n_train_runs"])
        for t in table:
            w.writerow([MODE_LABEL.get(t["mode"], t["mode"])]
                       + [f"{t['by_seed'][s]:.2f}" if s in t["by_seed"] else "" for s in train_seeds]
                       + [f"{t['mean']:.2f}", f"{t['std']:.2f}", t["n_runs"]])
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
