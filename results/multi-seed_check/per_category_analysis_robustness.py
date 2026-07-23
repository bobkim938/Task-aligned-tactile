"""
per_category_analysis_robustness.py
========================
Per-category success-rate breakdown from the multi-seed evaluation raw CSV.

Groups the 62 YCB objects into the three categories: Symmetric/Uniform, Thin/Elongated, and Irregular. 
For each (condition x category), computes mean +/- std of success rate across seeds.

Reads:
  ./multiseed_eval_raw.csv   (columns: condition, seed, model_id, env_idx, success)

Writes:
  ./per_category_summary.csv
  stdout: per-category table + the "where does the encoder help most" gap analysis
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent
RAW_CSV = RESULTS_DIR / "multiseed_eval_raw.csv"

# ---------------------------------------------------------------------------
# Object categorization
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

MODE_ORDER = ["oracle", "visual_only", "visual_raw_tact", "visual_encoder"]
MODE_LABEL = {
    "oracle": "Oracle",
    "visual_encoder": "Encoder (Ours)",
    "visual_raw_tact": "Raw Tactile",
    "visual_only": "Visual-only",
}
CATEGORY_LABEL = {
    "symmetric": "Symmetric/Uniform",
    "thin_elongated": "Thin/Elongated",
    "irregular": "Irregular",
}


def base_mode(condition):
    """Strip a trailing '_seed<N>' training-seed suffix, e.g.
    'visual_raw_tact_seed4' -> 'visual_raw_tact'."""
    return condition.split("_seed")[0]


def train_seed_of(condition):
    """Return the training seed encoded in the condition name, or None for
    a bare mode name with no seed suffix."""
    base = base_mode(condition)
    suffix = condition[len(base):]
    if suffix.startswith("_seed") and suffix[len("_seed"):].isdigit():
        return int(suffix[len("_seed"):])
    return None


def condition_label(condition):
    seed = train_seed_of(condition)
    label = MODE_LABEL.get(base_mode(condition), base_mode(condition))
    return f"{label} (seed{seed})" if seed is not None else label


def build_object_to_category():
    obj2cat = {}
    for cat, objs in CATEGORY_MAP.items():
        for o in objs:
            obj2cat[o] = cat
    return obj2cat


def load_raw(csv_path):
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "condition": r["condition"],
                "seed": int(r["seed"]),
                "model_id": r["model_id"],
                "env_idx": int(r["env_idx"]),
                "success": int(r["success"]),
            })
    return rows


def normalize_model_id(model_id):
    """
    The env's all_model_ids may carry suffixes; strip any trailing -N idx
    that the env appends when building actors (e.g. '058_golf_ball-0')
    """
    parts = model_id.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return model_id


def main():
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_CSV} not found. Run eval_multiseed.py first."
        )

    obj2cat = build_object_to_category()
    rows = load_raw(RAW_CSV)

    seen_objs = {normalize_model_id(r["model_id"]) for r in rows}
    uncategorized = sorted(o for o in seen_objs if o not in obj2cat)
    if uncategorized:
        print("WARNING: these objects appear in the data but are not in any " "category (they will be skipped):")
        for o in uncategorized:
            print(f"  {o}")
        print()

    categorized_with_data = {o for o in obj2cat if o in seen_objs}
    missing = sorted(o for o in obj2cat if o not in seen_objs)
    if missing:
        print("NOTE: these categorized objects have no rows in the data " "(check naming):")
        for o in missing:
            print(f"  {o}")
        print()

    conditions_present = sorted(
        {r["condition"] for r in rows},
        key=lambda c: (
            MODE_ORDER.index(base_mode(c)) if base_mode(c) in MODE_ORDER else len(MODE_ORDER),
            train_seed_of(c) if train_seed_of(c) is not None else -1,
        ),
    )

    grp = defaultdict(list)  # (cond, cat, eval_seed) -> list of 0/1
    for r in rows:
        obj = normalize_model_id(r["model_id"])
        cat = obj2cat.get(obj)
        if cat is None:
            continue
        grp[(r["condition"], cat, r["seed"])].append(r["success"])

    summary = {}  # (cond, cat) -> dict(mean, std, n_seeds, n_objs)
    cats = list(CATEGORY_MAP.keys())
    for cond in conditions_present:
        for cat in cats:
            eval_seeds = sorted({s for (c, ct, s) in grp if c == cond and ct == cat})
            if not eval_seeds:
                continue
            per_seed_rates = []
            for s in eval_seeds:
                vals = grp[(cond, cat, s)]
                per_seed_rates.append(100.0 * np.mean(vals))
            arr = np.array(per_seed_rates)
            # objects in this category that actually had data
            n_objs = len({normalize_model_id(r["model_id"]) for r in rows
                          if r["condition"] == cond
                          and obj2cat.get(normalize_model_id(r["model_id"])) == cat})
            summary[(cond, cat)] = dict(
                mean=float(arr.mean()),
                std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                n_seeds=len(eval_seeds),
                n_objs=n_objs,
            )

    modes_present = sorted(
        {base_mode(c) for c in conditions_present},
        key=lambda m: MODE_ORDER.index(m) if m in MODE_ORDER else len(MODE_ORDER),
    )
    pooled = {}  # (mode, cat) -> dict(mean, std, n_train_seeds)
    for mode in modes_present:
        conds_for_mode = [c for c in conditions_present if base_mode(c) == mode]
        for cat in cats:
            per_run_means = [summary[(c, cat)]["mean"] for c in conds_for_mode if (c, cat) in summary]
            if not per_run_means:
                continue
            arr = np.array(per_run_means)
            pooled[(mode, cat)] = dict(
                mean=float(arr.mean()),
                std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                n_train_seeds=len(per_run_means),
            )

    # ---- Print per-condition (per training-seed run) table ----
    col_w = max(24, max((len(condition_label(c)) for c in conditions_present), default=0) + 2)
    table_w = 20 + col_w * len(conditions_present)
    print("=" * table_w)
    print("Per-category success rate (mean +/- std across eval seeds), per training run")
    print("=" * table_w)
    header = f"{'Category':<20}" + "".join(
        f"{condition_label(c):>{col_w}}" for c in conditions_present
    )
    print(header)
    print("-" * table_w)
    for cat in cats:
        line = f"{CATEGORY_LABEL[cat]:<20}"
        for cond in conditions_present:
            stats = summary.get((cond, cat))
            cell = "--" if stats is None else f"{stats['mean']:.1f}+/-{stats['std']:.1f}"
            line += f"{cell:>{col_w}}"
        print(line)
    print("=" * table_w)

    # object counts per category (from data)
    print("\nObjects per category (with data):")
    for cat in cats:
        n = next((summary[(c, cat)]["n_objs"] for c in conditions_present if (c, cat) in summary), 0)
        print(f"  {CATEGORY_LABEL[cat]:<20}: {n} objects")

    # ---- Pooled table across training seeds, per base mode ----
    print("\n" + "=" * 78)
    print("Per-category success rate, pooled across training seeds per mode")
    print("=" * 78)
    header = f"{'Category':<20}" + "".join(
        f"{MODE_LABEL.get(m, m):>16}" for m in modes_present
    )
    print(header)
    print("-" * 78)
    for cat in cats:
        line = f"{CATEGORY_LABEL[cat]:<20}"
        for mode in modes_present:
            stats = pooled.get((mode, cat))
            if stats is None:
                line += f"{'--':>16}"
            else:
                line += f"{stats['mean']:>10.1f}+/-{stats['std']:<4.1f}"
        print(line)
    print("=" * 78)

    print("\n" + "=" * 78)
    print("Encoder gain over Raw Tactile, and over Visual-only, per category")
    print("(pooled across training seeds; this is the methodology-validation result)")
    print("=" * 78)
    print(f"{'Category':<20} {'Enc - RawTact':>16} {'Enc - Visual':>16}")
    print("-" * 78)
    for cat in cats:
        enc = pooled.get(("visual_encoder", cat))
        raw = pooled.get(("visual_raw_tact", cat))
        vis = pooled.get(("visual_only", cat))
        if enc and raw and vis:
            gain_raw = enc["mean"] - raw["mean"]
            gain_vis = enc["mean"] - vis["mean"]
            print(f"{CATEGORY_LABEL[cat]:<20} {gain_raw:>+15.1f} {gain_vis:>+15.1f}")
    print("=" * 78)

    # ---- Write summary CSV (per training-seed run + pooled rows) ----
    out_csv = RESULTS_DIR / "per_category_summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "category", "mean_success_pct",
                    "std_success_pct", "n_eval_seeds", "n_objects"])
        for cond in conditions_present:
            for cat in cats:
                stats = summary.get((cond, cat))
                if stats is None:
                    continue
                w.writerow([
                    condition_label(cond), CATEGORY_LABEL[cat],
                    f"{stats['mean']:.2f}", f"{stats['std']:.2f}",
                    stats["n_seeds"], stats["n_objs"],
                ])
        for mode in modes_present:
            for cat in cats:
                stats = pooled.get((mode, cat))
                if stats is None:
                    continue
                w.writerow([
                    f"{MODE_LABEL.get(mode, mode)} (pooled)", CATEGORY_LABEL[cat],
                    f"{stats['mean']:.2f}", f"{stats['std']:.2f}",
                    stats["n_train_seeds"], "",
                ])
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()