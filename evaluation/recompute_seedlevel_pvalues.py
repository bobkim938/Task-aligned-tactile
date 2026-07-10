"""
recompute_seedlevel_pvalues.py
==============================
Recomputes per-variant (per-object) and per-category significance tests at the
SEED level.
  - mean +/- std of seed-level success rates for each condition
  - the success-rate gap
  - Welch's t-test on the 10 seed-level rates (unpaired)
  - paired t-test on seed-matched rates (if seeds align across conditions)
  - Mann-Whitney U on seed-level rates (nonparametric backup)

Usage:
  python evaluation/recompute_seedlevel_pvalues.py \
      --csv results/multiseed_eval_raw.csv \
      --cond-a visual_encoder --cond-b visual_raw_tact

Edit CATEGORY_MAP / the column-name guesses below if your CSV differs.
"""

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Object category map
# ---------------------------------------------------------------------------
CATEGORY_MAP = {
    "symmetric": [
        "002_master_chef_can", "005_tomato_soup_can", "007_tuna_fish_can",
        "008_pudding_box", "009_gelatin_box", "010_potted_meat_can",
        "016_pear", "053_mini_soccer_ball", "054_softball", "055_baseball",
        "056_tennis_ball", "057_racquetball", "058_golf_ball",
        "062_dice", "063-a_marbles", "063-b_marbles",
        "012_strawberry", "013_apple", "014_lemon", "015_peach",
        "017_orange", "018_plum",
    ],
    "thin_elongated": [
        "030_fork", "031_spoon", "032_knife", "033_spatula",
        "040_large_marker", "048_hammer",
        "043_phillips_screwdriver", "044_flat_screwdriver",
        "037_scissors",
        "042_adjustable_wrench", "011_banana",
    ],
    "irregular": [
        # cups
        "065-a_cups", "065-b_cups", "065-c_cups", "065-d_cups", "065-e_cups",
        "065-f_cups", "065-g_cups", "065-h_cups", "065-i_cups", "065-j_cups",
        "025_mug",
        # toy airplanes
        "072-c_toy_airplane", "072-d_toy_airplane", "072-e_toy_airplane",
        # lego duplo
        "073-a_lego_duplo", "073-b_lego_duplo", "073-c_lego_duplo",
        "073-d_lego_duplo", "073-e_lego_duplo", "073-f_lego_duplo",
        "073-g_lego_duplo",
        # other irregular
        "038_padlock", "050_medium_clamp", "051_large_clamp",
        "061_foam_brick", "070-a_colored_wood_blocks",
        "070-b_colored_wood_blocks", "071_nine_hole_peg_test",
        "077_rubiks_cube",
    ],
}


# Candidate column names
OBJ_COLS = ["model_id", "object", "obj_name", "object_id"]
COND_COLS = ["condition", "cond", "ablation"]
SEED_COLS = ["seed", "eval_seed", "seed_id", "evaluation_seed"]
SUCCESS_COLS = ["success", "is_success", "succeeded"]


def pick(colnames, candidates, what):
    for c in candidates:
        if c in colnames:
            return c
    raise SystemExit(
        f"[fatal] Could not find a '{what}' column. Looked for {candidates}; "
        f"CSV has {colnames}. Edit the *_COLS list at the top of the script."
    )


def load_rows(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        obj_c = pick(cols, OBJ_COLS, "object")
        cond_c = pick(cols, COND_COLS, "condition")
        succ_c = pick(cols, SUCCESS_COLS, "success")
        # seed column is required for seed-level analysis
        seed_c = None
        for c in SEED_COLS:
            if c in cols:
                seed_c = c
                break
        if seed_c is None:
            raise SystemExit(
                "[fatal] No seed column found. Seed-level tests are impossible "
                f"without one. Looked for {SEED_COLS}; CSV has {cols}. If your "
                "episodes are not tagged by seed, the per-variant p-values "
                "cannot be computed correctly -- the episode-level numbers are "
                "the pseudoreplicated ones we are trying to replace."
            )
        rows = []
        for r in reader:
            rows.append({
                "obj": r[obj_c],
                "cond": r[cond_c],
                "seed": r[seed_c],
                "success": int(float(r[succ_c])),
            })
    print(f"Loaded {len(rows)} episode rows from {path}")
    print(f"  columns used: object='{obj_c}', condition='{cond_c}', "
          f"seed='{seed_c}', success='{succ_c}'\n")
    return rows


def seed_rates(rows, obj, cond):
    """Return dict seed -> success rate for one (object, condition)."""
    by_seed = defaultdict(lambda: [0, 0])  # seed -> [successes, n]
    for r in rows:
        if r["obj"] == obj and r["cond"] == cond:
            by_seed[r["seed"]][0] += r["success"]
            by_seed[r["seed"]][1] += 1
    return {s: sc / n for s, (sc, n) in by_seed.items() if n > 0}


def compare(rows, obj, cond_a, cond_b):
    """Seed-level comparison of cond_a vs cond_b for one object."""
    ra = seed_rates(rows, obj, cond_a)
    rb = seed_rates(rows, obj, cond_b)
    if not ra or not rb:
        return None

    common = sorted(set(ra) & set(rb))
    a_all = np.array([ra[s] for s in sorted(ra)])
    b_all = np.array([rb[s] for s in sorted(rb)])

    out = {
        "obj": obj,
        "n_seed_a": len(a_all), "n_seed_b": len(b_all),
        "mean_a": a_all.mean(), "std_a": a_all.std(ddof=1) if len(a_all) > 1 else 0.0,
        "mean_b": b_all.mean(), "std_b": b_all.std(ddof=1) if len(b_all) > 1 else 0.0,
        "gap": a_all.mean() - b_all.mean(),
        "n_common": len(common),
    }

    # Welch (unpaired) on all seed-level rates
    if len(a_all) > 1 and len(b_all) > 1:
        t, p = stats.ttest_ind(a_all, b_all, equal_var=False)
        out["welch_t"], out["welch_p"] = float(t), float(p)
    else:
        out["welch_t"], out["welch_p"] = float("nan"), float("nan")

    # Paired t-test on seed-matched rates (if seeds align)
    out["paired_t"], out["paired_p"] = float("nan"), float("nan")
    if len(common) >= 2:
        a_c = np.array([ra[s] for s in common])
        b_c = np.array([rb[s] for s in common])
        if np.allclose(a_c, b_c):
            out["paired_t"], out["paired_p"] = float("nan"), 1.0
        else:
            t, p = stats.ttest_rel(a_c, b_c)
            out["paired_t"], out["paired_p"] = float(t), float(p)

    # Mann-Whitney on the full (unpaired) seed-level rates
    out["mw_p"] = float("nan")
    if len(a_all) > 1 and len(b_all) > 1:
        try:
            _, pmw = stats.mannwhitneyu(a_all, b_all, alternative="two-sided")
            out["mw_p"] = float(pmw)
        except ValueError:
            out["mw_p"] = float("nan")

    return out


def fmt_p(p):
    if p != p:  # nan
        return "  n/a "
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.4f}"


def category_of(obj):
    for cat, objs in CATEGORY_MAP.items():
        if obj in objs:
            return cat
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--cond-a", default="visual_encoder",
                    help="condition A (default: Encoder)")
    ap.add_argument("--cond-b", default="visual_raw_tact",
                    help="condition B (default: Raw Tactile)")
    ap.add_argument("--filter", default=None,
                    help="only objects whose name contains this substring "
                         "(e.g. 'lego_duplo' or 'wrench')")
    args = ap.parse_args()

    rows = load_rows(args.csv)

    conds = sorted({r["cond"] for r in rows})
    print(f"Conditions in file: {conds}")
    if args.cond_a not in conds or args.cond_b not in conds:
        raise SystemExit(f"[fatal] --cond-a/--cond-b must be in {conds}")
    print(f"Comparing A='{args.cond_a}' vs B='{args.cond_b}'\n")

    objs = sorted({r["obj"] for r in rows})
    if args.filter:
        objs = [o for o in objs if args.filter in o]

    # ---- Per-object (per-variant) seed-level tests ----
    print("=" * 100)
    print("PER-OBJECT seed-level comparison (A - B)")
    print("=" * 100)
    print(f"{'object':<24}{'A mean':>9}{'B mean':>9}{'gap':>8}"
          f"{'Welch p':>11}{'paired p':>11}{'MW p':>10}{'#seed':>7}")
    print("-" * 100)

    results = []
    for o in objs:
        c = compare(rows, o, args.cond_a, args.cond_b)
        if c is None:
            continue
        results.append(c)
        print(f"{o:<24}{100*c['mean_a']:>8.1f}%{100*c['mean_b']:>8.1f}%"
              f"{100*c['gap']:>7.1f}"
              f"{fmt_p(c['welch_p']):>11}{fmt_p(c['paired_p']):>11}"
              f"{fmt_p(c['mw_p']):>10}{c['n_common']:>7}")

    # ---- Count how many clear common thresholds (uses paired if available) ----
    def pref_p(c):
        # prefer paired when seeds align, else Welch
        return c["paired_p"] if (c["n_common"] >= 2 and c["paired_p"] == c["paired_p"]) else c["welch_p"]

    if args.filter:
        gains17 = [c for c in results if c["gap"] * 100 >= 17]
        sig01 = [c for c in gains17 if pref_p(c) < 0.01]
        sig05 = [c for c in gains17 if pref_p(c) < 0.05]
        print("\n  Variants with gap >= 17 points: "
              f"{len(gains17)}/{len(results)}")
        print(f"    of those, p < 0.01 (preferred test): {len(sig01)}")
        print(f"    of those, p < 0.05 (preferred test): {len(sig05)}")

    # Per-category seed-level tests
    if CATEGORY_MAP:
        print("\n" + "=" * 100)
        print("PER-CATEGORY seed-level comparison")
        print("  (objects collapsed within each seed -> one rate per seed per")
        print("   condition; unit of analysis = seed, n = #seeds; paired A vs B)")
        print("=" * 100)
        print(f"{'category':<18}{'#obj':>6}{'#seed':>7}{'A mean':>9}{'B mean':>9}"
              f"{'gap':>8}{'paired p':>12}{'MW p':>11}")
        print("-" * 100)
        cat_objs = defaultdict(list)
        for o in {r["obj"] for r in rows}:
            cat = category_of(o)
            if cat:
                cat_objs[cat].append(o)
        for cat, olist in cat_objs.items():
            objset = set(olist)
            # seed -> [successes, n] pooled across the category's objects
            a_by_seed = defaultdict(lambda: [0, 0])
            b_by_seed = defaultdict(lambda: [0, 0])
            for r in rows:
                if r["obj"] not in objset:
                    continue
                if r["cond"] == args.cond_a:
                    a_by_seed[r["seed"]][0] += r["success"]
                    a_by_seed[r["seed"]][1] += 1
                elif r["cond"] == args.cond_b:
                    b_by_seed[r["seed"]][0] += r["success"]
                    b_by_seed[r["seed"]][1] += 1
            common = sorted(set(a_by_seed) & set(b_by_seed))
            if len(common) < 2:
                continue
            a = np.array([a_by_seed[s][0] / a_by_seed[s][1] for s in common])
            b = np.array([b_by_seed[s][0] / b_by_seed[s][1] for s in common])
            if np.allclose(a, b):
                p = 1.0
            else:
                _, p = stats.ttest_rel(a, b)
            try:
                _, pmw = stats.mannwhitneyu(a, b, alternative="two-sided")
            except ValueError:
                pmw = float("nan")
            print(f"{cat:<18}{len(olist):>6}{len(common):>7}{100*a.mean():>8.1f}%"
                  f"{100*b.mean():>8.1f}%{100*(a.mean()-b.mean()):>7.1f}"
                  f"{fmt_p(p):>12}{fmt_p(pmw):>11}")
    else:
        print("\n[note] CATEGORY_MAP is empty -- per-category section skipped.")
        print("       Fill CATEGORY_MAP at the top of the script to enable it.")


if __name__ == "__main__":
    main()