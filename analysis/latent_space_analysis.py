"""
latent_space_analysis.py
====================================
Panel A: 3-class phase classification accuracy from three feature sets:
         norm-only (1D), direction-only (16D), full z_t (16D), via 5-fold CV.
         Tests whether phase information lives in latent direction vs magnitude

Panel B: Pairwise L2 distance distributions in z_t-mean-during-transport space,
         split into within-category vs between-category pairs across the 3
         paper categories (Symmetric, Thin/Elongated, Irregular)

Panel C: Pearson correlation heatmap (16 dimensions x 5 physical quantities)
         over grasped timesteps (phase >= 1)

Reads:
    data/v2/deployment_records.npz

Writes:
    results/latent_space_analysis/latent_space_analysis.pdf
    results/latent_space_analysis/latent_space_analysis.png
    results/latent_space_analysis/latent_panelA_phase_accuracy.csv
    results/latent_space_analysis/latent_panelB_pairwise_distances.csv
    results/latent_space_analysis/latent_panelB_summary.csv
    results/latent_space_analysis/latent_panelC_correlations.csv

Console output also prints the underlying numbers.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Unified deployment schema: data/{tag}/deployment_records.npz
NPZ_PATH = (PROJECT_ROOT / "data"
            / "v2" / "deployment_records.npz")
OUT_DIR = PROJECT_ROOT / "results" / "latent_space_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   7.5,
    "font.family":       "sans-serif",
    "font.sans-serif":   ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.linewidth":    0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth":   1.0,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})


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
        "011_banana", "012_strawberry", "013_apple", "014_lemon", "015_peach",
        "017_orange", "018_plum",
    ],
    "thin_elongated": [
        "030_fork", "031_spoon", "032_knife", "033_spatula",
        "040_large_marker", "048_hammer",
        "043_phillips_screwdriver", "044_flat_screwdriver",
        "037_scissors", "042_adjustable_wrench",
    ],
    "irregular": [
        "065-a_cups", "065-b_cups", "065-c_cups", "065-d_cups", "065-e_cups",
        "065-f_cups", "065-g_cups", "065-h_cups", "065-i_cups", "065-j_cups",
        "025_mug", "072-c_toy_airplane", "072-d_toy_airplane", "072-e_toy_airplane",
        "073-a_lego_duplo", "073-b_lego_duplo", "073-c_lego_duplo",
        "073-d_lego_duplo", "073-e_lego_duplo", "073-f_lego_duplo",
        "038_padlock", "050_medium_clamp", "051_large_clamp",
        "061_foam_brick", "070-a_colored_wood_blocks",
        "070-b_colored_wood_blocks", "071_nine_hole_peg_test",
        "077_rubiks_cube",
    ],
}


def build_obj_to_cat():
    m = {}
    for cat, objs in CATEGORY_MAP.items():
        for o in objs:
            m[o] = cat
    return m


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    d = np.load(NPZ_PATH, allow_pickle=True)
    return {
        "z_t":         d["z_t"],
        "tactile_raw": d["tactile_raw"],
        "is_grasped":  d["is_grasped"],
        "phase":       d["phase"],
        "obj_names":   d["obj_names"],
        "success":     d["success"].astype(bool),
    }


# ---------------------------------------------------------------------------
# Panel A — Phase classification probe (norm vs direction vs full)
# ---------------------------------------------------------------------------
def panel_a(ax, data):
    z_t = data["z_t"]        # [N, T, 16]
    phase = data["phase"]    # [N, T]

    z_flat = z_t.reshape(-1, z_t.shape[-1])
    p_flat = phase.reshape(-1)
    mask = (p_flat >= 0) & (p_flat <= 2)
    X = z_flat[mask].astype(np.float32)
    y = p_flat[mask].astype(np.int32)

    counts = {p: int((y == p).sum()) for p in (0, 1, 2)}
    print(f"  Panel A: {len(X)} timesteps "
          f"(free={counts[0]}, grasp={counts[1]}, transport={counts[2]})")

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = norms
    X_dir = X / (norms + 1e-8)
    X_full = X

    def cv_acc(X_in, label):
        clf = LogisticRegression(max_iter=2000)
        scores = cross_val_score(clf, X_in, y, cv=5, scoring="accuracy", n_jobs=-1)
        print(f"    {label:<14}: {scores.mean():.4f} (std {scores.std():.4f})")
        return scores.mean(), scores.std()

    acc_norm, std_norm = cv_acc(X_norm, "Norm (1D)")
    acc_dir,  std_dir  = cv_acc(X_dir,  "Direction (16D)")
    acc_full, std_full = cv_acc(X_full, "Full (16D)")

    methods = ["Norm", "Direction", r"Full $z_t$"]
    accs = [acc_norm, acc_dir, acc_full]
    colors = ["#95a5a6", "#3498db", "#1f4e79"]
    bars = ax.bar(methods, accs, width=0.6, color=colors, alpha=0.9,
                  edgecolor="black", linewidth=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{acc:.3f}", ha="center", fontsize=8.5, fontweight="bold")

    class_freqs = np.bincount(y) / len(y)
    majority_baseline = class_freqs.max()
    ax.axhline(majority_baseline, color="red", linestyle=":",
               linewidth=0.8, alpha=0.7)
    ax.text(0.97, majority_baseline + 0.02,
            f"majority ({majority_baseline:.2f})",
            transform=ax.get_yaxis_transform(),
            fontsize=7, color="red", ha="right", va="bottom")

    ax.set_ylabel("Phase accuracy (5-fold CV)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    return {
        "rows": [
            ("norm_1d",       acc_norm, std_norm),
            ("direction_16d", acc_dir,  std_dir),
            ("full_16d",      acc_full, std_full),
        ],
        "majority_baseline": float(majority_baseline),
        "counts": counts,
    }


# ---------------------------------------------------------------------------
# Panel B — Within vs between category L2 distance
# ---------------------------------------------------------------------------
def panel_b(ax, data):
    z_t = data["z_t"]
    phase = data["phase"]
    obj_names = data["obj_names"]

    obj_to_cat = build_obj_to_cat()

    per_ep_mean = []
    per_ep_cat = []
    for i in range(len(z_t)):
        m = phase[i] == 2
        if m.sum() < 5:
            continue
        per_ep_mean.append(z_t[i][m].mean(axis=0))
        per_ep_cat.append(obj_to_cat.get(str(obj_names[i]), None))

    per_ep_mean = np.array(per_ep_mean)
    per_ep_cat = np.array(per_ep_cat)

    valid = per_ep_cat != None  # noqa: E711
    per_ep_mean = per_ep_mean[valid]
    per_ep_cat = per_ep_cat[valid]

    print(f"  Panel B: {len(per_ep_mean)} categorized episodes")
    for cat in CATEGORY_MAP:
        print(f"    {cat:<16}: {int((per_ep_cat == cat).sum())}")

    N = len(per_ep_mean)
    diffs = per_ep_mean[:, None, :] - per_ep_mean[None, :, :]
    pair_dist = np.linalg.norm(diffs, axis=-1)
    same = per_ep_cat[:, None] == per_ep_cat[None, :]
    iu = np.triu_indices(N, k=1)
    within = pair_dist[iu][same[iu]]
    between = pair_dist[iu][~same[iu]]

    print(f"    within pairs:  n={len(within)}, mean={within.mean():.3f}")
    print(f"    between pairs: n={len(between)}, mean={between.mean():.3f}")
    ratio_bw = between.mean() / within.mean()
    print(f"    ratio (between/within): {ratio_bw:.3f}")

    hi = max(within.max(), between.max()) * 1.05
    bins = np.linspace(0, hi, 38)

    ax.hist(within, bins=bins, density=True, alpha=0.55,
            color="#1f78b4", edgecolor="#1f78b4", linewidth=0.6)
    ax.hist(between, bins=bins, density=True, alpha=0.55,
            color="#e31a1c", edgecolor="#e31a1c", linewidth=0.6)

    ax.axvline(within.mean(),  color="#1f78b4", linestyle="--", linewidth=0.9, alpha=0.8)
    ax.axvline(between.mean(), color="#e31a1c", linestyle="--", linewidth=0.9, alpha=0.8)

    ax.set_xlabel(r"L2 distance in $z_t$ space")
    ax.set_ylabel("Density")
    ax.set_xlim(0, hi)

    box_x, box_y_top = 0.97, 0.97
    line_h = 0.075

    lines = [
        ("within:",                       "#1f78b4", True),
        (f"$\\mu={within.mean():.3f}$",   "#1f78b4", False),
        (f"($n$={len(within)})",          "#1f78b4", False),
        ("between:",                      "#e31a1c", True),
        (f"$\\mu={between.mean():.3f}$",  "#e31a1c", False),
        (f"($n$={len(between)})",         "#e31a1c", False),
        (f"ratio: ${ratio_bw:.2f}$",      "black",   False),
    ]

    from matplotlib.patches import FancyBboxPatch
    n_lines = len(lines)
    box_w = 0.40
    bg = FancyBboxPatch(
        (box_x - box_w, box_y_top - line_h * n_lines - 0.02),
        box_w, line_h * n_lines + 0.04,
        boxstyle="round,pad=0.01", transform=ax.transAxes,
        facecolor="white", edgecolor="gray", linewidth=0.5, alpha=0.95,
        zorder=4,
    )
    ax.add_patch(bg)

    for i, (txt, color, bold) in enumerate(lines):
        ax.text(box_x - 0.01, box_y_top - 0.01 - i * line_h, txt,
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=color,
                fontweight="bold" if bold else "normal", zorder=5)

    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    return {
        "within": within,
        "between": between,
        "ratio": float(ratio_bw),
        "n_episodes": int(len(per_ep_mean)),
        "cat_counts": {cat: int((per_ep_cat == cat).sum()) for cat in CATEGORY_MAP},
    }


# ---------------------------------------------------------------------------
# Panel C — Per-dimension correlation heatmap
# ---------------------------------------------------------------------------
def panel_c(ax, cax, data):
    z_t = data["z_t"]
    tactile_raw = data["tactile_raw"]
    phase = data["phase"]

    z_flat = z_t.reshape(-1, z_t.shape[-1])
    raw_flat = tactile_raw.reshape(-1, tactile_raw.shape[-1])
    phase_flat = phase.reshape(-1)
    mask = (phase_flat >= 1) & (phase_flat <= 2)
    z = z_flat[mask]
    raw = raw_flat[mask]
    print(f"  Panel C: {len(z)} grasped timesteps")

    f_left = np.linalg.norm(raw[:, :3], axis=1)
    f_right = np.linalg.norm(raw[:, 3:], axis=1)
    asym = np.abs(f_left - f_right) / (f_left + f_right + 1e-6)
    f_mag = np.linalg.norm(raw, axis=1)

    quantities = {
        "Asymmetry":           asym,
        "Force mag":           f_mag,
        r"$F_\mathrm{left}$":  f_left,
        r"$F_\mathrm{right}$": f_right,
    }
    quantity_csv_names = ["asymmetry", "force_mag", "F_left", "F_right"]

    n_dim = z.shape[1]
    n_q = len(quantities)
    corr = np.zeros((n_dim, n_q))
    for q_idx, q_vals in enumerate(quantities.values()):
        if q_vals.std() < 1e-8:
            corr[:, q_idx] = 0.0
            continue
        for d in range(n_dim):
            if z[:, d].std() < 1e-8:
                corr[d, q_idx] = 0.0
            else:
                r, _ = stats.pearsonr(z[:, d], q_vals)
                corr[d, q_idx] = r

    print("    Top-3 dimensions per quantity:")
    for q_idx, q_name in enumerate(quantities.keys()):
        order = np.argsort(np.abs(corr[:, q_idx]))[::-1][:3]
        s = ", ".join(f"z[{d}]={corr[d, q_idx]:+.3f}" for d in order)
        print(f"      {q_name:<14}: {s}")

    thr = 0.25
    print(f"    Dimensions with |r| > {thr} per quantity:")
    for q_idx, q_name in enumerate(quantities.keys()):
        order = np.argsort(np.abs(corr[:, q_idx]))[::-1]
        strong = [d for d in order if abs(corr[d, q_idx]) > thr]
        s = ", ".join(f"z[{d}]={corr[d, q_idx]:+.3f}" for d in strong) or "(none)"
        print(f"      {q_name:<14}: {s}")

    vmax = 0.6
    im = ax.imshow(corr, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    ax.set_yticks(range(n_dim))
    ax.set_yticklabels([f"{d}" for d in range(n_dim)], fontsize=7)
    ax.set_ylabel(r"$z_t$ dimension")

    ax.set_xticks(range(n_q))
    ax.set_xticklabels(list(quantities.keys()), rotation=35, ha="right", fontsize=7.5)

    for d in range(n_dim):
        for q_idx in range(n_q):
            r = corr[d, q_idx]
            if abs(r) > 0.3:
                color = "white" if abs(r) > 0.55 else "black"
                ax.text(q_idx, d, f"{r:+.2f}", ha="center", va="center",
                        fontsize=6.5, fontweight="bold", color=color)

    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Pearson $r$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    return {"corr": corr, "quantity_names": quantity_csv_names, "n_dim": n_dim}


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def export_panel_a_csv(out_path, a_data):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature_set", "cv_accuracy", "cv_std"])
        for name, acc, std in a_data["rows"]:
            w.writerow([name, f"{acc:.6f}", f"{std:.6f}"])
        w.writerow([])
        w.writerow(["metric", "value"])
        w.writerow(["majority_baseline", f"{a_data['majority_baseline']:.6f}"])
        w.writerow(["n_free", a_data["counts"][0]])
        w.writerow(["n_grasp", a_data["counts"][1]])
        w.writerow(["n_transport", a_data["counts"][2]])
    print(f"Wrote {out_path}")


def export_panel_b_csv(dist_path, summary_path, b_data):
    with open(dist_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_type", "l2_distance"])
        for v in b_data["within"]:
            w.writerow(["within", f"{v:.6f}"])
        for v in b_data["between"]:
            w.writerow(["between", f"{v:.6f}"])
    print(f"Wrote {dist_path}")

    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_type", "n", "mean"])
        wn, bn = b_data["within"], b_data["between"]
        w.writerow(["within", len(wn), f"{wn.mean():.6f}"])
        w.writerow(["between", len(bn), f"{bn.mean():.6f}"])
        w.writerow([])
        w.writerow(["metric", "value"])
        w.writerow(["ratio_between_over_within", f"{b_data['ratio']:.6f}"])
        w.writerow(["n_categorized_episodes", b_data["n_episodes"]])
        for cat, n in b_data["cat_counts"].items():
            w.writerow([f"n_{cat}", n])
    print(f"Wrote {summary_path}")


def export_panel_c_csv(out_path, c_data):
    corr = c_data["corr"]
    names = c_data["quantity_names"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["z_dim"] + names)
        for d in range(c_data["n_dim"]):
            w.writerow([d] + [f"{corr[d, q]:.6f}" for q in range(len(names))])
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data = load_data()
    n_ep = len(data["z_t"])
    n_succ = int(data["success"].sum())
    print(f"Loaded {n_ep} episodes from {NPZ_PATH}")
    print(f"  Success: {n_succ}/{n_ep} ({100*n_succ/n_ep:.1f}%)\n")

    fig = plt.figure(figsize=(7.16, 2.8))
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[1.65, 2.20, 2.30, 0.10],
        wspace=0.55,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    cax  = fig.add_subplot(gs[0, 3])

    print("Panel A: Phase classification probe")
    a_data = panel_a(ax_a, data)
    print("\nPanel B: Within-vs-between category L2 distance")
    b_data = panel_b(ax_b, data)
    print("\nPanel C: Per-dimension correlation heatmap")
    c_data = panel_c(ax_c, cax, data)

    for ax, letter, x_off in [(ax_a, "(a)", -0.35),
                              (ax_b, "(b)", -0.18),
                              (ax_c, "(c)", -0.10)]:
        ax.text(x_off, 1.02, letter, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="left")

    plt.subplots_adjust(left=0.07, right=0.96, top=0.93, bottom=0.20)

    pos_c = ax_c.get_position()
    cax.set_position([pos_c.x1 + 0.012, pos_c.y0, 0.012, pos_c.height])

    out_pdf = OUT_DIR / "latent_space_analysis.pdf"
    out_png = OUT_DIR / "latent_space_analysis.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)

    print(f"\nWrote {out_pdf}")
    print(f"Wrote {out_png}")

    print()
    export_panel_a_csv(OUT_DIR / "latent_panelA_phase_accuracy.csv", a_data)
    export_panel_b_csv(OUT_DIR / "latent_panelB_pairwise_distances.csv",
                       OUT_DIR / "latent_panelB_summary.csv", b_data)
    export_panel_c_csv(OUT_DIR / "latent_panelC_correlations.csv", c_data)


if __name__ == "__main__":
    main()