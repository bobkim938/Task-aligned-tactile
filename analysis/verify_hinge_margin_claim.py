"""
verify_hinge_margin_claim.py
============================
Compares step-to-step latent oscillation ||z_t - z_{t-1}||_2 between
V1 (no hinge loss) and V2 (with hinge loss), split into within-phase steps
and phase-transition steps. Establishes:

  (1) hinge compresses within-phase oscillation (V1 -> V2).
  (2) compression is selective: the within-phase-to-transition ratio is
      preserved across the compression, so phase structure is not erased.
  (3) separation is carried by the angular (direction) component, matching
      the Panel A finding that phase lives in latent direction.

Reads:
    data/v2/deployment_records.npz   (V2, with hinge)
    data/v1/deployment_records.npz   (V1, no hinge)

Run:
    python analysis/verify_hinge_margin_claim.py
"""

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_NPZ = (PROJECT_ROOT / "data" / "v2"
          / "deployment_records.npz")
V1_NPZ = (PROJECT_ROOT / "data" / "v1"
          / "deployment_records.npz")


def step_stats(z_t, phase):
    """Return dict of within/transition step-distance stats + angular/mag."""
    dz = np.diff(z_t, axis=1)
    dz_norm = np.linalg.norm(dz, axis=-1)

    z_prev, z_curr = z_t[:, :-1], z_t[:, 1:]
    np_prev = np.linalg.norm(z_prev, axis=-1) + 1e-8
    np_curr = np.linalg.norm(z_curr, axis=-1) + 1e-8
    cos = np.clip((z_prev * z_curr).sum(-1) / (np_prev * np_curr), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos))
    norm_diff = np.abs(np_curr - np_prev)

    is_trans = phase[:, :-1] != phase[:, 1:]
    is_within = ~is_trans

    return {
        "within": dz_norm[is_within],
        "transition": dz_norm[is_trans],
        "angle_within": angle_deg[is_within],
        "angle_trans": angle_deg[is_trans],
        "norm_within": norm_diff[is_within],
        "norm_trans": norm_diff[is_trans],
        "dz_norm": dz_norm,
        "phase_prev": phase[:, :-1],
        "phase_curr": phase[:, 1:],
    }


def report(tag, s):
    w, t = s["within"], s["transition"]
    print("")
    print(tag + ": step-to-step ||z_t - z_{t-1}||_2")
    print("  within-phase  (n=%6d): mean=%.4f  median=%.4f"
          % (len(w), w.mean(), np.median(w)))
    print("  transition    (n=%6d): mean=%.4f  median=%.4f"
          % (len(t), t.mean(), np.median(t)))
    print("  ratio (transition/within): mean=%.2fx  median=%.2fx"
          % (t.mean() / w.mean(), np.median(t) / np.median(w)))
    print("  angular: within %.2f deg  transition %.2f deg"
          % (s["angle_within"].mean(), s["angle_trans"].mean()))
    print("  norm   : within %.4f  transition %.4f"
          % (s["norm_within"].mean(), s["norm_trans"].mean()))


def per_transition_breakdown(s):
    transitions = [
        (0, 1, "free -> grasp"), (1, 2, "grasp -> transport"),
        (2, 1, "transport -> grasp"), (1, 0, "grasp -> free"),
        (2, 0, "transport -> free"),
    ]
    print("")
    print("  %-22s%8s%10s%10s" % ("Transition", "count", "mean dz", "median"))
    print("  " + "-" * 48)
    for src, dst, name in transitions:
        m = (s["phase_prev"] == src) & (s["phase_curr"] == dst)
        if m.sum() == 0:
            continue
        steps = s["dz_norm"][m]
        print("  %-22s%8d%10.4f%10.4f"
              % (name, m.sum(), steps.mean(), np.median(steps)))


def load(npz):
    d = np.load(npz, allow_pickle=True)
    return d["z_t"], d["phase"]


def main():
    print("=" * 70)
    print("HINGE LOSS: V1 (no hinge) vs V2 (with hinge) step-distance comparison")
    print("=" * 70)

    if not V2_NPZ.exists():
        raise FileNotFoundError("V2 file not found: %s" % V2_NPZ)
    z2, p2 = load(V2_NPZ)
    print("")
    print("V2: %d episodes x %d steps" % (z2.shape[0], z2.shape[1]))
    s2 = step_stats(z2, p2)
    report("V2 (with hinge loss)", s2)
    per_transition_breakdown(s2)

    if V1_NPZ.exists():
        z1, p1 = load(V1_NPZ)
        print("")
        print("V1: %d episodes x %d steps" % (z1.shape[0], z1.shape[1]))
        s1 = step_stats(z1, p1)
        report("V1 (no hinge loss)", s1)

        print("")
        print("=" * 70)
        print("Summary for the hinge section:")
        print("=" * 70)
        r1 = np.median(s1["transition"]) / np.median(s1["within"])
        r2 = np.median(s2["transition"]) / np.median(s2["within"])
        print("  V1 within median = %.3f, transition median = %.3f, ratio = %.1fx"
              % (np.median(s1["within"]), np.median(s1["transition"]), r1))
        print("  V2 within median = %.3f, transition median = %.3f, ratio = %.1fx"
              % (np.median(s2["within"]), np.median(s2["transition"]), r2))
        print("  Within-phase compression V1->V2 (medians): %.1fx"
              % (np.median(s1["within"]) / np.median(s2["within"])))
        print("  Ratio preserved across compression: %.1fx (V1) -> %.1fx (V2)"
              % (r1, r2))
        print("  V2 angular: within %.1f deg, transition %.1f deg"
              % (s2["angle_within"].mean(), s2["angle_trans"].mean()))
    else:
        print("")
        print("[V1] file not found (%s); V1 comparison skipped." % V1_NPZ)
        print("     Re-collect V1 with the unified collector to enable it.")


if __name__ == "__main__":
    main()