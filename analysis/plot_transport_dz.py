"""
plot_transport_dz.py
================================
two-panel comparison of transport-phase
latent oscillation between V1 (no hinge loss) and V2 (with hinge loss).

Panel A (log-scale): All four distributions on the same axis to visually
    showing the compression in baseline oscillation.

Panel B (linear-scale, V2 only): Zoomed view showing that despite the
    compression, the success/failure separation in V2 is preserved
    (success mean 0.0432 vs failure mean 0.0800, MW p<1e-13).

Reads:
    data/v1/deployment_records.npz
    data/v2/deployment_records.npz

Writes:
    results/transport_dz_v1_vs_v2/transport_dz_v1_vs_v2.pdf  
    results/transport_dz_v1_vs_v2/transport_dz_v1_vs_v2.png  
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V1_NPZ = (PROJECT_ROOT / "data" / "v1"
          / "deployment_records.npz")
V2_NPZ = (PROJECT_ROOT / "data" / "v2"
          / "deployment_records.npz")
OUT_DIR = PROJECT_ROOT / "results" / "transport_dz_v1_vs_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.0,
    "pdf.fonttype": 42,   
    "ps.fonttype": 42,
})

COLOR_SUCCESS = "green"  
COLOR_FAILURE = "red" 


# ---------------------------------------------------------------------------
# Load data: compute per-episode transport-phase ||dz_t||
# ---------------------------------------------------------------------------
def per_episode_transport_dz(npz_path):
    """
    For each episode, compute the mean step-to-step latent oscillation
    ||z_t - z_{t-1}||_2 over transport-phase steps (phase == 2)
    """
    d = np.load(npz_path, allow_pickle=True)
    z_t = d["z_t"]                    # [N, T, 16]
    phase = d["phase"]               # [N, T]
    success = d["success"].astype(bool)  # [N]

    N = z_t.shape[0]
    dz = np.linalg.norm(np.diff(z_t, axis=1), axis=-1)   # [N, T-1] step distances
    # A step t belongs to transport if both endpoints are transport phase,
    # i.e. phase[t]==2 and phase[t+1]==2 -- this isolates within-transport
    # oscillation and excludes the grasp->transport boundary jump
    transport_step = (phase[:, :-1] == 2) & (phase[:, 1:] == 2)  # [N, T-1]

    per_ep = np.full(N, np.nan, dtype=np.float64)
    for i in range(N):
        m = transport_step[i]
        if m.sum() >= 2:
            per_ep[i] = dz[i][m].mean()
    return per_ep, success


def split(dz, success):
    """Drop NaN, return (success_array, failure_array)."""
    valid = ~np.isnan(dz)
    dz_v = dz[valid]
    s_v = success[valid]
    return dz_v[s_v], dz_v[~s_v]


# ---------------------------------------------------------------------------
# Panel A — horizontal box plot on log x-axis (compression view)
# ---------------------------------------------------------------------------
def panel_a(ax, v1_s, v1_f, v2_s, v2_f):
    """
    Four horizontal boxes (V1 succ, V1 fail, V2 succ, V2 fail) on a shared log x-axis
    """
    groups = [
        (v1_f, 4, COLOR_FAILURE, f"V1 failure (n={len(v1_f)})"),
        (v1_s, 3, COLOR_SUCCESS, f"V1 success (n={len(v1_s)})"),
        (v2_f, 2, COLOR_FAILURE, f"V2 failure (n={len(v2_f)})"),
        (v2_s, 1, COLOR_SUCCESS, f"V2 success (n={len(v2_s)})"),
    ]

    for data, y, color, _label in groups:
        bp = ax.boxplot(
            data, positions=[y], vert=False, widths=0.6,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor=color, alpha=0.55,
                          edgecolor=color, linewidth=1.0),
            medianprops=dict(color="black", linewidth=1.2),
            whiskerprops=dict(color=color, linewidth=1.0),
            capprops=dict(color=color, linewidth=1.0),
        )
        # Mean as a diamond marker
        ax.plot(data.mean(), y, marker="D", color=color,
                markeredgecolor="black", markeredgewidth=0.6,
                markersize=5, zorder=5)

    # Y-axis labels
    ax.set_yticks([4, 3, 2, 1])
    ax.set_yticklabels([
        f"V1 fail\n(n={len(v1_f)})",
        f"V1 succ\n(n={len(v1_s)})",
        f"V2 fail\n(n={len(v2_f)})",
        f"V2 succ\n(n={len(v2_s)})",
    ])

    # Visual separator between V1 and V2 groups
    ax.axhline(2.5, color="gray", linewidth=0.6, linestyle=":", alpha=0.7)

    ax.set_xscale("log")
    ax.set_xlabel(r"Transport-phase $\|\Delta z_t\|_2$ (log scale)")
    ax.set_xlim(5e-3, 3.0)
    ax.set_ylim(0.3, 4.7)

    # Compression annotation — placed in the visual gap between V1 and V2 groups
    compression = v1_s.mean() / v2_s.mean()
    gap_x = np.sqrt(v1_s.mean() * v2_s.mean())   # geometric mean of the two means
    ax.annotate(
        f"$\\sim${compression:.0f}$\\times$ compression",
        xy=(gap_x, 2.5), xycoords="data",
        ha="center", va="center", fontsize=9, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="gray", linewidth=0.6, alpha=0.95),
    )

    # Mean-marker legend (small, lower-left so it doesn't fight the compression box)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
                      markeredgecolor="black", markersize=5,
                      label="mean")]
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.9,
              edgecolor="gray", fancybox=False, borderpad=0.3,
              handlelength=1.0)

    ax.grid(True, axis="x", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Panel B — V2-only linear-scale, signal-preservation view
# ---------------------------------------------------------------------------
def panel_b(ax, v2_s, v2_f):
    # Linear bins covering V2's range
    upper = np.percentile(v2_f, 98)   # avoid blowing out for outliers
    bins = np.linspace(0, upper, 32)

    ax.hist(v2_s, bins=bins, density=True,
            color=COLOR_SUCCESS, alpha=0.6, edgecolor=COLOR_SUCCESS,
            linewidth=0.8, label=f"Success (n={len(v2_s)})")
    ax.hist(v2_f, bins=bins, density=True,
            color=COLOR_FAILURE, alpha=0.6, edgecolor=COLOR_FAILURE,
            linewidth=0.8, label=f"Failure (n={len(v2_f)})")

    # Mean lines
    ax.axvline(v2_s.mean(), color=COLOR_SUCCESS, linewidth=1.0,
               linestyle="-", alpha=0.85)
    ax.axvline(v2_f.mean(), color=COLOR_FAILURE, linewidth=1.0,
               linestyle="-", alpha=0.85)

    ax.set_xlabel(r"Transport-phase $\|\Delta z_t\|_2$ (V2 only)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, upper)

    _, p_val = mannwhitneyu(v2_s, v2_f, alternative="two-sided")
    ratio = v2_f.mean() / v2_s.mean()
    p_exp = int(np.floor(np.log10(p_val)))

    stats_text = (
        f"$\\mu_{{\\mathrm{{succ}}}} = {v2_s.mean():.4f} \\pm {v2_s.std(ddof=1):.4f}$\n"
        f"$\\mu_{{\\mathrm{{fail}}}} = {v2_f.mean():.4f} \\pm {v2_f.std(ddof=1):.4f}$\n"
        f"ratio: ${ratio:.2f}\\times$,  $p < 10^{{{p_exp}}}$"
    )

    # Give the panel ~30% extra vertical headroom so the stats box sits cleanly above all bars.
    ylim_cur = ax.get_ylim()
    ax.set_ylim(0, ylim_cur[1] * 1.35)

    ax.text(
        0.97, 0.97, stats_text,
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="gray", linewidth=0.6, alpha=0.95),
    )

    ax.legend(loc="upper left", frameon=True, framealpha=0.95,
              edgecolor="gray", fancybox=False, borderpad=0.4,
              handlelength=1.5)

    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def export_panel_a_csv(out_path, v1_s, v1_f, v2_s, v2_f):
    """
    Panel A data: every episode's per-episode mean transport ||dz_t||, tagged
    by encoder version (V1/V2) and outcome (success/failure). One row per
    episode
    """
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["version", "outcome", "transport_dz"])
        for val in v1_s:
            w.writerow(["V1", "success", f"{val:.6f}"])
        for val in v1_f:
            w.writerow(["V1", "failure", f"{val:.6f}"])
        for val in v2_s:
            w.writerow(["V2", "success", f"{val:.6f}"])
        for val in v2_f:
            w.writerow(["V2", "failure", f"{val:.6f}"])
    print(f"Wrote {out_path}")


def export_panel_b_csv(out_path, v2_s, v2_f):
    """
    Panel B data: every V2 episode's per-episode mean transport ||dz_t||,
    tagged by outcome. One row per episode
    """
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["outcome", "transport_dz"])
        for val in v2_s:
            w.writerow(["success", f"{val:.6f}"])
        for val in v2_f:
            w.writerow(["failure", f"{val:.6f}"])
    print(f"Wrote {out_path}")


def export_summary_csv(out_path, v1_s, v1_f, v2_s, v2_f):
    """
    Summary statistics for both panels: per-(version, outcome) n / mean / std /
    median, plus the V1->V2 success compression ratio and the V2 success-vs-
    failure separation ratio and Mann-Whitney p-value
    """
    def stats(arr):
        return (len(arr), float(np.mean(arr)), float(np.std(arr, ddof=1)),
                float(np.median(arr)))

    rows = [
        ("V1", "success", *stats(v1_s)),
        ("V1", "failure", *stats(v1_f)),
        ("V2", "success", *stats(v2_s)),
        ("V2", "failure", *stats(v2_f)),
    ]
    compression = v1_s.mean() / v2_s.mean()
    sep_ratio = v2_f.mean() / v2_s.mean()
    _, p_val = mannwhitneyu(v2_s, v2_f, alternative="two-sided")

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["version", "outcome", "n", "mean", "std", "median"])
        for version, outcome, n, mean, std, med in rows:
            w.writerow([version, outcome, n,
                        f"{mean:.6f}", f"{std:.6f}", f"{med:.6f}"])
        w.writerow([])
        w.writerow(["metric", "value"])
        w.writerow(["compression_v1_v2_success", f"{compression:.6f}"])
        w.writerow(["separation_ratio_v2_fail_over_succ", f"{sep_ratio:.6f}"])
        w.writerow(["mannwhitney_p_v2_succ_vs_fail", f"{p_val:.6e}"])
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    v1_dz, v1_success = per_episode_transport_dz(V1_NPZ)
    v2_dz, v2_success = per_episode_transport_dz(V2_NPZ)
    v1_s, v1_f = split(v1_dz, v1_success)
    v2_s, v2_f = split(v2_dz, v2_success)

    print("Loaded per-episode transport Δz_t arrays:")
    print(f"  V1: {len(v1_s)} success, {len(v1_f)} failure")
    print(f"  V2: {len(v2_s)} success, {len(v2_f)} failure")
    print(f"  V1 success mean: {v1_s.mean():.4f}")
    print(f"  V2 success mean: {v2_s.mean():.4f}")
    print(f"  Compression ratio (V1/V2, success): "
          f"{v1_s.mean()/v2_s.mean():.2f}x\n")

    # --- Panel B numbers (V2 success vs failure separation) ---
    _, p_val = mannwhitneyu(v2_s, v2_f, alternative="two-sided")
    sep_ratio = v2_f.mean() / v2_s.mean()
    print("--- Panel B: V2 success vs failure separation ---")
    print(f"  V2 success: n={len(v2_s):4d}  mean={v2_s.mean():.4f}  "
          f"std={v2_s.std(ddof=1):.4f}  median={np.median(v2_s):.4f}")
    print(f"  V2 failure: n={len(v2_f):4d}  mean={v2_f.mean():.4f}  "
          f"std={v2_f.std(ddof=1):.4f}  median={np.median(v2_f):.4f}")
    print(f"  Separation ratio (failure/success): {sep_ratio:.2f}x")
    print(f"  Mann-Whitney U p-value: {p_val:.3e}\n")

    # --- Export CSVs for external plotting ---
    export_panel_a_csv(OUT_DIR / "transport_dz_panelA.csv",
                       v1_s, v1_f, v2_s, v2_f)
    export_panel_b_csv(OUT_DIR / "transport_dz_panelB.csv", v2_s, v2_f)
    export_summary_csv(OUT_DIR / "transport_dz_summary.csv",
                       v1_s, v1_f, v2_s, v2_f)
    print()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    panel_a(axes[0], v1_s, v1_f, v2_s, v2_f)
    panel_b(axes[1], v2_s, v2_f)

    for ax, letter in zip(axes, ["(a)", "(b)"]):
        ax.text(-0.16, 1.02, letter, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="left")

    plt.tight_layout(pad=0.5, w_pad=2.0)

    out_pdf = OUT_DIR / "transport_dz_v1_vs_v2.pdf"
    out_png = OUT_DIR / "transport_dz_v1_vs_v2.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)

    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()