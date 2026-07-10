"""
analyze_generalization.py
====================================
Out-of-distribution latent-structure analysis on held-out YCB objects.

Panel A: Pairwise L2 distance distributions in z_t-mean-during-transport space,
         split into within-category vs between-category pairs across the 3
         paper categories (Symmetric, Thin/Elongated, Irregular). Shows the
         encoder organizes by physical interaction properties, not identity
         (within- vs between-category ratio replicates in-distribution).

Panel B: Pearson correlation heatmap (16 dimensions x physical quantities)
         over grasped timesteps (phase >= 1). Shows the encoder compresses
         force signals into a low-dimensional structured subspace, and reports
         the top-ranked dimension per quantity (which shifts under the
         distribution change).

Reads:
    data/figure_deployment/4_4_heldout/latent_records.npz

Writes:
    figures/section_4_4_heldout_latent_analysis.pdf
    figures/section_4_4_heldout_latent_analysis.png

Console output also prints the underlying numbers (use these in the paper prose).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NPZ_PATH = (PROJECT_ROOT / "data" / "4_4_heldout" / "latent_records.npz")
OUT_DIR = PROJECT_ROOT / "results" / "heldout_latent_analysis"
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
# Object categorization — extended with held-out objects for the generalization test
CATEGORY_MAP = {
    "symmetric": [
        # In-distribution
        "002_master_chef_can", "005_tomato_soup_can", "007_tuna_fish_can",
        "008_pudding_box", "009_gelatin_box", "010_potted_meat_can",
        "016_pear", "053_mini_soccer_ball", "054_softball", "055_baseball",
        "056_tennis_ball", "057_racquetball", "058_golf_ball",
        "062_dice", "063-a_marbles", "063-b_marbles",
        "011_banana", "012_strawberry", "013_apple", "014_lemon", "015_peach",
        "017_orange", "018_plum",
        # held-out (uniform geometry, symmetric profile)
        "003_cracker_box", "004_sugar_box", "026_sponge",
        "029_plate", "036_wood_block",
    ],
    "thin_elongated": [
        # In-distribution
        "030_fork", "031_spoon", "032_knife", "033_spatula",
        "040_large_marker", "048_hammer",
        "043_phillips_screwdriver", "044_flat_screwdriver",
        "037_scissors", "042_adjustable_wrench",
        # held-out
        "052_extra_large_clamp",
    ],
    "irregular": [
        # In-distribution
        "065-a_cups", "065-b_cups", "065-c_cups", "065-d_cups", "065-e_cups",
        "065-f_cups", "065-g_cups", "065-h_cups", "065-i_cups", "065-j_cups",
        "025_mug", "072-c_toy_airplane", "072-d_toy_airplane", "072-e_toy_airplane",
        "073-a_lego_duplo", "073-b_lego_duplo", "073-c_lego_duplo",
        "073-d_lego_duplo", "073-e_lego_duplo", "073-f_lego_duplo",
        "038_padlock", "050_medium_clamp", "051_large_clamp",
        "061_foam_brick", "070-a_colored_wood_blocks",
        "070-b_colored_wood_blocks", "071_nine_hole_peg_test",
        "077_rubiks_cube",
        # held-out (bottles, containers, asymmetric geometries)
        "006_mustard_bottle", "019_pitcher_base", "021_bleach_cleanser",
        "022_windex_bottle", "024_bowl", "028_skillet_lid",
        "035_power_drill", "059_chain",
        "072-a_toy_airplane", "072-b_toy_airplane",
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
# Panel A — Within vs between category L2 distance
# ---------------------------------------------------------------------------
def panel_a(ax, data):
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

    print(f"  Panel A: {len(per_ep_mean)} categorized episodes")
    for cat in CATEGORY_MAP:
        print(f"    {cat:<16}: {int((per_ep_cat == cat).sum())}")

    # Guard: need at least two episodes from different categories to form pairs
    if len(per_ep_mean) < 2 or len(np.unique(per_ep_cat)) < 2:
        msg = (f"insufficient data\n({len(per_ep_mean)} episodes,\n"
               f"{len(np.unique(per_ep_cat))} categories)")
        print(f"  Panel A: SKIPPED -- {msg}")
        ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="gray",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="gray", linewidth=0.5))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r"L2 distance in $z_t$ space")
        ax.set_ylabel("Density")
        return

    # All pairwise distances
    N = len(per_ep_mean)
    diffs = per_ep_mean[:, None, :] - per_ep_mean[None, :, :]
    pair_dist = np.linalg.norm(diffs, axis=-1)
    same = per_ep_cat[:, None] == per_ep_cat[None, :]
    iu = np.triu_indices(N, k=1)
    within = pair_dist[iu][same[iu]]
    between = pair_dist[iu][~same[iu]]

    # Guard: need both within and between pairs
    if len(within) == 0 or len(between) == 0:
        msg = (f"need both within and between pairs\n"
               f"(within: {len(within)}, between: {len(between)})")
        print(f"  Panel A: SKIPPED -- {msg}")
        ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="gray",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="gray", linewidth=0.5))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r"L2 distance in $z_t$ space")
        ax.set_ylabel("Density")
        return

    print(f"    within pairs:  n={len(within)}, mean={within.mean():.3f}")
    print(f"    between pairs: n={len(between)}, mean={between.mean():.3f}")
    ratio_bw = between.mean() / within.mean()
    print(f"    ratio (between/within): {ratio_bw:.3f}")

    # Histograms
    hi = max(within.max(), between.max()) * 1.05
    bins = np.linspace(0, hi, 38)

    ax.hist(within, bins=bins, density=True, alpha=0.55,
            color="navy", edgecolor="navy", linewidth=0.6)
    ax.hist(between, bins=bins, density=True, alpha=0.55,
            color="red", edgecolor="red", linewidth=0.6)

    ax.axvline(within.mean(),  color="navy", linestyle="--", linewidth=0.9, alpha=0.8)
    ax.axvline(between.mean(), color="red", linestyle="--", linewidth=0.9, alpha=0.8)

    ax.set_xlabel(r"L2 distance in $z_t$ space")
    ax.set_ylabel("Density")
    ax.set_xlim(0, hi)

    from matplotlib.offsetbox import TextArea, VPacker, AnchoredOffsetbox

    lines = [
        ("within:",                              "navy", "bold"),
        (f"$\\mu={within.mean():.3f}$",          "navy", "normal"),
        (f"($n$={len(within)})",                 "navy", "normal"),
        ("between:",                             "red", "bold"),
        (f"$\\mu={between.mean():.3f}$",         "red", "normal"),
        (f"($n$={len(between)})",                "red", "normal"),
        (f"ratio = ${ratio_bw:.2f}$",            "black",  "normal"),
    ]
    text_areas = [
        TextArea(txt, textprops=dict(color=color, fontsize=7.5,
                                     fontweight=weight))
        for txt, color, weight in lines
    ]
    packed = VPacker(children=text_areas, align="right", pad=0, sep=1.5)
    anchored = AnchoredOffsetbox(
        loc="upper right", child=packed, pad=0.35, borderpad=0.3,
        bbox_to_anchor=(0.97, 0.97), bbox_transform=ax.transAxes,
        frameon=True,
    )
    anchored.patch.set(facecolor="white", edgecolor="gray",
                       linewidth=0.5, alpha=0.95)
    anchored.set_zorder(5)
    ax.add_artist(anchored)

    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Panel B — Per-dimension correlation heatmap
# ---------------------------------------------------------------------------
def panel_b(ax, cax, data):
    z_t = data["z_t"]
    tactile_raw = data["tactile_raw"]
    phase = data["phase"]

    # Grasped timesteps (phase 1 or 2)
    z_flat = z_t.reshape(-1, z_t.shape[-1])
    raw_flat = tactile_raw.reshape(-1, tactile_raw.shape[-1])
    phase_flat = phase.reshape(-1)
    mask = (phase_flat >= 1) & (phase_flat <= 2)
    z = z_flat[mask]
    raw = raw_flat[mask]
    print(f"  Panel B: {len(z)} grasped timesteps")

    f_left = np.linalg.norm(raw[:, :3], axis=1)
    f_right = np.linalg.norm(raw[:, 3:], axis=1)
    asym = np.abs(f_left - f_right) / (f_left + f_right + 1e-6)
    f_mag = np.linalg.norm(raw, axis=1)
    
    # contact constant within phase>=1 (by definition of phase), so excluded
    quantities = {
        "Asymmetry":           asym,
        "Force mag":           f_mag,
        r"$F_\mathrm{left}$":  f_left,
        r"$F_\mathrm{right}$": f_right,
    }

    n_dim = z.shape[1]
    n_q = len(quantities)
    corr = np.zeros((n_dim, n_q))
    for q_idx, q_vals in enumerate(quantities.values()):
        if q_vals.std() < 1e-8:
            # Quantity is constant within this subset; correlation undefined
            corr[:, q_idx] = 0.0
            continue
        for d in range(n_dim):
            if z[:, d].std() < 1e-8:
                corr[d, q_idx] = 0.0
            else:
                r, _ = stats.pearsonr(z[:, d], q_vals)
                corr[d, q_idx] = r

    # Print top-3 dimensions per quantity
    print("    Top-3 dimensions per quantity:")
    for q_idx, q_name in enumerate(quantities.keys()):
        order = np.argsort(np.abs(corr[:, q_idx]))[::-1][:3]
        s = ", ".join(f"z[{d}]={corr[d, q_idx]:+.3f}" for d in order)
        print(f"      {q_name:<14}: {s}")

    vmax = 0.6
    im = ax.imshow(corr, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    ax.set_yticks(range(n_dim))
    ax.set_yticklabels([f"{d}" for d in range(n_dim)], fontsize=7)
    ax.set_ylabel(r"$z_t$ dimension")

    ax.set_xticks(range(n_q))
    ax.set_xticklabels(list(quantities.keys()), rotation=35, ha="right", fontsize=7.5)

    # Annotate strong cells
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data = load_data()
    n_ep = len(data["z_t"])
    n_succ = int(data["success"].sum())
    print(f"Loaded {n_ep} episodes from {NPZ_PATH}")
    print(f"  Success: {n_succ}/{n_ep} ({100*n_succ/n_ep:.1f}%)\n")

    fig = plt.figure(figsize=(5.3, 2.8))
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[2.20, 2.45],
        wspace=0.45,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    gs_b = gs[0, 1].subgridspec(1, 2, width_ratios=[2.30, 0.10], wspace=0.10)
    ax_b = fig.add_subplot(gs_b[0, 0])
    cax  = fig.add_subplot(gs_b[0, 1])

    print("Panel A: Within-vs-between category L2 distance")
    try:
        panel_a(ax_a, data)
    except Exception as e:
        print(f"  Panel A: FAILED with {type(e).__name__}: {e}")
        ax_a.text(0.5, 0.5, "Panel A failed", transform=ax_a.transAxes,
                  ha="center", va="center", color="red")

    print("\nPanel B: Per-dimension correlation heatmap")
    panel_b(ax_b, cax, data)

    for ax, letter, x_off in [(ax_a, "(a)", -0.18),
                              (ax_b, "(b)", -0.10)]:
        ax.text(x_off, 1.02, letter, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="left")

    plt.subplots_adjust(left=0.07, right=0.96, top=0.93, bottom=0.20)

    out_pdf = OUT_DIR / "section_4_4_heldout_latent_analysis.pdf"
    out_png = OUT_DIR / "section_4_4_heldout_latent_analysis.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)

    print(f"\nWrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()