"""
plot_grasp_asymmetry_separation.py
Section 3-D
==================================
grasp-phase tactile force asymmetry (successful vs failed episodes)

Asymmetry per timestep:
    a_t = | ||F_left|| - ||F_right|| | / ( ||F_left|| + ||F_right|| )
aggregated as the per-episode mean over grasped timesteps (phase >= 1, grasp + transport). 
We plot the two distributions (success vs failure) and annotate the separation ratio and a Mann-Whitney U test.

This supports the claim that the auxiliary supervision target (force
asymmetry) is itself outcome-discriminative: failed episodes carry markedly
higher bilateral asymmetry than successful ones.

Reads:   data/v2/deployment_records.npz
Writes:  results/grasp_asymmetry_separation.{pdf,png}
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NPZ_PATH = (PROJECT_ROOT / "data" / "v2"
            / "deployment_records.npz")
OUT_DIR = PROJECT_ROOT / "results" / "grasp_asymmetry_separation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# IEEE single-column style
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

COLOR_SUCCESS = "green"   # green
COLOR_FAILURE = "red"   # red


def per_episode_asymmetry(npz_path):
    """Per-episode mean grasp-phase asymmetry (phase >= 1), split by outcome."""
    d = np.load(npz_path, allow_pickle=True)
    raw = d["tactile_raw"]                 # [N, T, 6]
    phase = d["phase"]                     # [N, T]
    success = d["success"].astype(bool)    # [N]
    N = raw.shape[0]

    f_left = np.linalg.norm(raw[..., :3], axis=-1)    # [N, T]
    f_right = np.linalg.norm(raw[..., 3:], axis=-1)   # [N, T]
    asym = np.abs(f_left - f_right) / (f_left + f_right + 1e-6)

    per_ep, per_ep_succ = [], []
    for i in range(N):
        m = phase[i] >= 1
        if m.sum() == 0:
            continue
        per_ep.append(asym[i][m].mean())
        per_ep_succ.append(success[i])

    per_ep = np.array(per_ep)
    per_ep_succ = np.array(per_ep_succ)
    return per_ep[per_ep_succ], per_ep[~per_ep_succ]


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------
def export_data_csv(out_path, succ, fail):
    """Per-episode asymmetry, one row per episode, tagged by outcome"""
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["outcome", "asymmetry"])
        for v in succ:
            w.writerow(["success", f"{v:.6f}"])
        for v in fail:
            w.writerow(["failure", f"{v:.6f}"])
    print(f"Wrote {out_path}")


def export_summary_csv(out_path, succ, fail, ratio, p):
    """Summary stats for annotations (means, n, separation ratio, MW p)."""
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["outcome", "n", "mean", "std", "median"])
        w.writerow(["success", len(succ), f"{succ.mean():.6f}",
                    f"{succ.std(ddof=1):.6f}", f"{np.median(succ):.6f}"])
        w.writerow(["failure", len(fail), f"{fail.mean():.6f}",
                    f"{fail.std(ddof=1):.6f}", f"{np.median(fail):.6f}"])
        w.writerow([])
        w.writerow(["metric", "value"])
        w.writerow(["separation_ratio_fail_over_succ", f"{ratio:.6f}"])
        w.writerow(["mannwhitney_p", f"{p:.6e}"])
    print(f"Wrote {out_path}")


def main():
    succ, fail = per_episode_asymmetry(NPZ_PATH)
    print(f"  success episodes: n={len(succ)}, mean={succ.mean():.4f}")
    print(f"  failure episodes: n={len(fail)}, mean={fail.mean():.4f}")
    ratio = fail.mean() / succ.mean()
    _, p = mannwhitneyu(succ, fail, alternative="two-sided")
    print(f"  separation ratio (failure/success): {ratio:.2f}x")
    print(f"  Mann-Whitney U p-value: {p:.3e}")

    # --- Export CSVs for external plotting ---
    export_data_csv(OUT_DIR / "grasp_asymmetry_data.csv", succ, fail)
    export_summary_csv(OUT_DIR / "grasp_asymmetry_summary.csv",
                       succ, fail, ratio, p)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    upper = np.percentile(fail, 98)
    bins = np.linspace(0, max(upper, succ.max()), 36)

    ax.hist(succ, bins=bins, density=True, color=COLOR_SUCCESS, alpha=0.6,
            edgecolor=COLOR_SUCCESS, linewidth=0.8,
            label=f"Success (n={len(succ)})")
    ax.hist(fail, bins=bins, density=True, color=COLOR_FAILURE, alpha=0.6,
            edgecolor=COLOR_FAILURE, linewidth=0.8,
            label=f"Failure (n={len(fail)})")

    # Mean lines
    ax.axvline(succ.mean(), color=COLOR_SUCCESS, linewidth=1.2, linestyle="-")
    ax.axvline(fail.mean(), color=COLOR_FAILURE, linewidth=1.2, linestyle="-")

    ax.set_xlabel(r"Grasp-phase force asymmetry "
                  r"$|\,\|F_L\| - \|F_R\|\,| / (\|F_L\| + \|F_R\|)$")
    ax.set_ylabel("Density")
    ax.set_xlim(0, max(upper, succ.max()))

    # Stats annotation
    p_exp = int(np.floor(np.log10(p))) if p > 0 else -300
    stats_text = (
        f"$\\mu_{{\\mathrm{{succ}}}} = {succ.mean():.3f}$\n"
        f"$\\mu_{{\\mathrm{{fail}}}} = {fail.mean():.3f}$\n"
        f"${ratio:.1f}\\times$ separation\n"
        f"$p < 10^{{{p_exp}}}$"
    )
    ylim = ax.get_ylim()
    ax.set_ylim(0, ylim[1] * 1.30)
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", linewidth=0.6, alpha=0.95))

    ax.legend(loc="upper left", frameon=True, framealpha=0.95,
              edgecolor="gray", fancybox=False, borderpad=0.4)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)

    plt.tight_layout(pad=0.4)
    out_pdf = OUT_DIR / "grasp_asymmetry_separation.pdf"
    out_png = OUT_DIR / "grasp_asymmetry_separation.png"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()