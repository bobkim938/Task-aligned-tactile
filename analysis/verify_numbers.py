"""
verify_numbers.py
========================
Verifies grasp-asymmetry numbers from the unified V2 deployment records
(data/v2/deployment_records.npz):

Grasp asymmetry split by episode success.
   Definition: asym_t = |F_left - F_right| / (F_left + F_right),
   evaluated on grasp-phase timesteps, then averaged per episode and
   per group (success / failure).

Run:
    python analysis/verify_numbers.py
"""

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NPZ_PATH = (PROJECT_ROOT / "data" / "v2"
            / "deployment_records.npz")


def load():
    print(f"Loading {NPZ_PATH}")
    d = np.load(NPZ_PATH, allow_pickle=True)
    tactile = d["tactile_raw"] if "tactile_raw" in d.files else d["tactile_smoothed"]
    out = {
        "z_t":         d["z_t"],          # [N, T, 16]
        "tactile_raw": tactile,           # [N, T, 6]
        "phase":       d["phase"],        # [N, T]
        "success":     d["success"].astype(bool),  # [N]
    }
    N, T, D = out["z_t"].shape
    print(f"  z_t shape: {N} episodes x {T} steps x {D} dims")
    print(f"  tactile shape: {out['tactile_raw'].shape}")
    print(f"  success: {int(out['success'].sum())}/{N}\n")
    return out


# ---------------------------------------------------------------------------
# Check: grasp asymmetry, split by success/failure
# ---------------------------------------------------------------------------
def check_grasp_asymmetry(data, thesis_success=0.037, thesis_failure=0.384):
    print("\n" + "=" * 70)
    print("Grasp-phase tactile asymmetry, success vs failure")
    print(f"  Thesis reported: success={thesis_success}, "
          f"failure={thesis_failure}")
    print("=" * 70)

    raw = data["tactile_raw"]              # [N, T, 6]
    phase = data["phase"]                  # [N, T]
    success = data["success"]              # [N]
    N = raw.shape[0]

    f_left = np.linalg.norm(raw[..., :3], axis=-1)   # [N, T]
    f_right = np.linalg.norm(raw[..., 3:], axis=-1)  # [N, T]
    asym = np.abs(f_left - f_right) / (f_left + f_right + 1e-6)  # [N, T]

    # Two interpretations of "grasp" phase:
    #   strict: phase == 1 only (the actual grasping phase)
    #   broad : phase >= 1 (grasp + transport, i.e. "gripper is holding")
    for tag, mask_fn in [
        ("phase == 1 (grasp only)",       lambda p: p == 1),
        ("phase >= 1 (grasp + transport)", lambda p: p >= 1),
    ]:
        print(f"\n  Definition: {tag}")
        print(f"  {'-' * 60}")

        ph_mask = mask_fn(phase)           # [N, T]

        per_ep_mean = []
        per_ep_n = []
        per_ep_success = []
        for i in range(N):
            m = ph_mask[i]
            n = int(m.sum())
            if n == 0:
                continue
            per_ep_mean.append(asym[i][m].mean())
            per_ep_n.append(n)
            per_ep_success.append(success[i])

        per_ep_mean = np.array(per_ep_mean)
        per_ep_n = np.array(per_ep_n)
        per_ep_success = np.array(per_ep_success)

        print(f"  Episodes with any matching timesteps: {len(per_ep_mean)}/{N}")
        print(f"    success episodes: {int(per_ep_success.sum())}")
        print(f"    failure episodes: {int((~per_ep_success).sum())}")

        for label, sel in [("success", per_ep_success),
                           ("failure", ~per_ep_success)]:
            vals = per_ep_mean[sel]
            ns = per_ep_n[sel]
            if len(vals) == 0:
                print(f"    {label}: no episodes")
                continue
            mean_per_ep = vals.mean()
            mean_pooled = (vals * ns).sum() / ns.sum()
            print(f"    {label}: n={len(vals):>4}  "
                  f"per-ep mean={mean_per_ep:.4f}  "
                  f"median={np.median(vals):.4f}  "
                  f"pooled={mean_pooled:.4f}  "
                  f"total grasp steps={ns.sum()}")

        succ_mean = per_ep_mean[per_ep_success].mean() if per_ep_success.any() else float("nan")
        fail_mean = per_ep_mean[~per_ep_success].mean() if (~per_ep_success).any() else float("nan")
        if succ_mean > 0:
            print(f"    Separation ratio (failure/success): {fail_mean/succ_mean:.2f}x")
        print(f"    Delta vs thesis: "
              f"success {succ_mean - thesis_success:+.4f}, "
              f"failure {fail_mean - thesis_failure:+.4f}")


def main():
    data = load()
    check_grasp_asymmetry(data)


if __name__ == "__main__":
    main()