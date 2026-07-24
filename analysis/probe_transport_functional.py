"""
probe_transport_functional.py
=============================
  TRANSPORT OUTCOME PROBE
    Predict episode success from the per-episode mean representation over
    TRANSPORT timesteps (phase == 2), comparing:
        Raw tactile (6D)         -- mean tactile_smoothed
        Raw mean+std (12D)       -- [mean, std] of tactile_smoothed (dimension
                                     objection control: richer raw statistics,
                                     the natural cheap proxy for what a
                                     frame-stack policy would see)
        Random projection (16D)  -- raw -> fixed random linear map (dim control)
        Trained encoder (16D)    -- mean z_t
    Reports ROC-AUC (5-fold stratified CV). The headline is the encoder-vs-raw
    gap at transport, to be compared against the grasp-phase gap.

Reads:  data/v2/deployment_records.npz

Usage:  python analysis/probe_transport_functional.py
"""

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NPZ_PATH = (PROJECT_ROOT / "data" / "v2"
            / "deployment_records.npz")

# Pre-committed constants for PART 2 (slip)
K_FUTURE = 5             # horizon (steps) for "near-future"
SLIP_THRESH = 0.005      # meters of RELATIVE height loss that counts as slip


# ---------------------------------------------------------------------------
# Shared CV helpers
# ---------------------------------------------------------------------------
def run_probe_full(X, y, n_splits=5):
    """5-fold stratified CV; returns ROC-AUC mean/std."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=5000, class_weight="balanced",
                                 solver="lbfgs")
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return {
        "dim": X.shape[1],
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs, ddof=1)),
    }


def cv_auc(X, y, n_splits=5):
    """5-fold stratified CV AUC only (for the slip part)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=5000, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1))


# ===========================================================================
# PART 1 -- transport outcome probe
# ===========================================================================
def outcome_probe(d):
    print("=" * 92)
    print("TRANSPORT OUTCOME PROBE")
    print("=" * 92)

    z_t = d["z_t"]
    tactile = d["tactile_smoothed"] if "tactile_smoothed" in d.files else d["tactile_raw"]
    phase = d["phase"]
    success = d["success"].astype(int)
    N, T, _ = z_t.shape

    rng = np.random.default_rng(seed=42)
    W_rand = rng.standard_normal(size=(6, 16)) / np.sqrt(6)

    X_raw, X_rawms, X_lat, X_rand = [], [], [], []
    for i in range(N):
        m = phase[i] == 2            # TRANSPORT only
        if m.sum() == 0:
            X_raw.append(np.full(6, np.nan))
            X_rawms.append(np.full(12, np.nan))
            X_lat.append(np.full(16, np.nan))
            X_rand.append(np.full(16, np.nan))
            continue
        raw_ep = tactile[i][m]
        X_raw.append(raw_ep.mean(axis=0))
        X_rawms.append(np.concatenate([raw_ep.mean(axis=0), raw_ep.std(axis=0)]))
        X_lat.append(z_t[i][m].mean(axis=0))
        X_rand.append((raw_ep @ W_rand).mean(axis=0))

    X_raw = np.array(X_raw); X_rawms = np.array(X_rawms)
    X_lat = np.array(X_lat); X_rand = np.array(X_rand)
    valid = (~np.isnan(X_raw).any(1) & ~np.isnan(X_rawms).any(1)
             & ~np.isnan(X_lat).any(1) & ~np.isnan(X_rand).any(1))
    X_raw, X_rawms, X_lat, X_rand = (X_raw[valid], X_rawms[valid],
                                      X_lat[valid], X_rand[valid])
    y = success[valid]

    print(f"  Episodes with transport steps: {len(y)} "
          f"(success={int(y.sum())}, failure={int((1-y).sum())})\n")

    sets = [("Raw tactile", X_raw), ("Raw mean+std", X_rawms),
            ("Random projection", X_rand), ("Trained encoder", X_lat)]
    res = {}
    for label, X in sets:
        r = run_probe_full(X, y)
        res[label] = r
        print(f"  {label:<20} ROC-AUC = {r['auc_mean']:.4f} ± {r['auc_std']:.3f}")

    print()
    print(f"  {'Feature set':<20}{'Dim':>4}{'ROC-AUC':>16}")
    print("  " + "-" * 40)
    for label, _ in sets:
        r = res[label]
        print(f"  {label:<20}{r['dim']:>4}"
              f"{r['auc_mean']:>9.4f}±{r['auc_std']:.3f}")

    raw_auc = res["Raw tactile"]["auc_mean"]
    rawms_auc = res["Raw mean+std"]["auc_mean"]
    rand_auc = res["Random projection"]["auc_mean"]
    lat_auc = res["Trained encoder"]["auc_mean"]
    print()
    print(f"  Encoder transport AUC : {lat_auc:.4f}")
    print(f"  Raw     transport AUC : {raw_auc:.4f}")
    print(f"  Raw mean+std AUC (12D): {rawms_auc:.4f}  "
          f"(dimension-objection control: {rawms_auc-raw_auc:+.4f} vs 6D raw mean, "
          f"{lat_auc-rawms_auc:+.4f} encoder still ahead)")
    print(f"  Random proj.    AUC   : {rand_auc:.4f}  "
          f"(dim control: {rand_auc-raw_auc:+.4f} vs raw)")
    print(f"  Encoder - Raw (gap)   : {lat_auc-raw_auc:+.4f}")
    return lat_auc, raw_auc, rand_auc, rawms_auc



def main():
    if not NPZ_PATH.exists():
        raise SystemExit(f"Not found: {NPZ_PATH}")
    d = np.load(NPZ_PATH, allow_pickle=True)
    N = d["z_t"].shape[0]
    print(f"Loaded {N} episodes from {NPZ_PATH.name}")
    print(f"  success: {int(d['success'].sum())}/{N} "
          f"({100*int(d['success'].sum())/N:.1f}%)\n")

    p1 = outcome_probe(d)

if __name__ == "__main__":
    main()