"""
probe_grasp_outcome.py
======================
Probe experiment: does the encoder's latent representation carry more
outcome-discriminative information during grasp than the raw tactile signal?

Three feature sets, all aggregated as per-episode mean over grasped timesteps
(phase >= 1):
  1. Raw tactile (6D)       : baseline; mean of tactile_smoothed over grasped steps
  2. Random projection (16D): raw tactile -> fixed random linear projection.
     Controls for "does more dimensionality help" with a linear control.
  3. Trained encoder (16D)  : the actual encoder used in deployment;
     mean of z_t over grasped steps.

For each feature set, train logistic regression with 5-fold stratified CV
to predict episode success y in {0, 1}, and report ROC-AUC, PR-AUC,
F1 on the failure class, and accuracy.

Reads:
    data/v2/deployment_records.npz

Usage:
    python analysis/probe_grasp_outcome.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------------
# Paths and imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NPZ_PATH = (PROJECT_ROOT / "data"
            / "v2" / "deployment_records.npz")
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Load data and aggregate per-episode features
# ---------------------------------------------------------------------------
def load_and_aggregate():
    d = np.load(NPZ_PATH, allow_pickle=True)
    z_t = d["z_t"]
    # Use tactile_smoothed
    tactile_raw = d["tactile_smoothed"] if "tactile_smoothed" in d.files else d["tactile_raw"]
    phase = d["phase"]
    success = d["success"].astype(int)

    N, T, _ = z_t.shape
    print(f"Loaded {N} episodes from {NPZ_PATH.name}")
    print(f"  z_t shape: {z_t.shape}, tactile shape: {tactile_raw.shape}")
    print(f"  Success: {success.sum()}/{N} "
          f"({100*success.sum()/N:.1f}%)\n")

    rng = np.random.default_rng(seed=42)
    W_rand = rng.standard_normal(size=(6, 16)) / np.sqrt(6)

    X_raw_list = []
    X_lat_list = []
    X_rand_list = []

    for i in range(N):
        m = phase[i] >= 1
        if m.sum() == 0:
            X_raw_list.append(np.full(6, np.nan))
            X_lat_list.append(np.full(16, np.nan))
            X_rand_list.append(np.full(16, np.nan))
            continue

        raw_ep = tactile_raw[i][m]
        lat_ep = z_t[i][m]
        rand_ep = raw_ep @ W_rand

        X_raw_list.append(raw_ep.mean(axis=0))
        X_lat_list.append(lat_ep.mean(axis=0))
        X_rand_list.append(rand_ep.mean(axis=0))

    X_raw = np.array(X_raw_list)
    X_lat = np.array(X_lat_list)
    X_rand = np.array(X_rand_list)

    valid = (
        ~np.isnan(X_raw).any(axis=1)
        & ~np.isnan(X_lat).any(axis=1)
        & ~np.isnan(X_rand).any(axis=1)
    )
    X_raw = X_raw[valid]
    X_lat = X_lat[valid]
    X_rand = X_rand[valid]
    y = success[valid]

    print(f"\nAfter filtering: {len(y)} episodes "
          f"(success={int(y.sum())}, failure={int((1-y).sum())})\n")

    return X_raw, X_lat, X_rand, y


def run_probe(X, y, label, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, prs, f1s, accs = [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
        )
        clf.fit(X_train, y_train)
        p_success = clf.predict_proba(X_test)[:, 1]
        y_pred = (p_success >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_test, p_success))
        prs.append(average_precision_score(1 - y_test, 1 - p_success))
        f1s.append(f1_score(y_test, y_pred, pos_label=0))
        accs.append(accuracy_score(y_test, y_pred))

    return {
        "label":    label,
        "dim":      X.shape[1],
        "auc_mean": float(np.mean(aucs)),
        "auc_std":  float(np.std(aucs, ddof=1)),
        "pr_mean":  float(np.mean(prs)),
        "pr_std":   float(np.std(prs, ddof=1)),
        "f1_mean":  float(np.mean(f1s)),
        "f1_std":   float(np.std(f1s, ddof=1)),
        "acc_mean": float(np.mean(accs)),
        "acc_std":  float(np.std(accs, ddof=1)),
    }


def print_results(results, n_episodes, n_success, n_failure):
    print("=" * 96)
    print("Probe results: predicting episode success from grasp-phase representation")
    print("=" * 96)
    print(f"  Episodes: {n_episodes}  (success={n_success}, failure={n_failure})")
    print(f"  Majority baseline accuracy: {n_success/n_episodes:.4f}")
    print()
    print(f"{'Feature set':<28} {'Dim':>4} "
          f"{'ROC-AUC':>16} {'PR-AUC':>16} {'F1 (fail)':>16} {'Acc':>16}")
    print("-" * 96)
    for r in results:
        print(f"{r['label']:<28} {r['dim']:>4} "
              f"{r['auc_mean']:>8.4f} ± {r['auc_std']:.3f}  "
              f"{r['pr_mean']:>7.4f} ± {r['pr_std']:.3f}  "
              f"{r['f1_mean']:>7.4f} ± {r['f1_std']:.3f}  "
              f"{r['acc_mean']:>7.4f} ± {r['acc_std']:.3f}")
    print()


def interpret(results):
    by_label = {r["label"]: r for r in results}
    raw = by_label["Raw tactile"]
    rand = by_label["Random projection"]
    trained = by_label["Trained encoder"]

    total = trained["auc_mean"] - raw["auc_mean"]

    print("=" * 96)
    print("Grasp-phase outcome discriminability (ROC-AUC):")
    print("=" * 96)
    print(f"  Raw tactile (6D)         : {raw['auc_mean']:.4f}")
    print(f"  Random projection (16D)  : {rand['auc_mean']:.4f}   "
          f"(linear control: {rand['auc_mean']-raw['auc_mean']:+.4f} vs raw)")
    print(f"  Trained encoder (16D)    : {trained['auc_mean']:.4f}   "
          f"(preservation: {total:+.4f} vs raw)")
    print()
    print("  Framing: the encoder's bottleneck PRESERVES the outcome-")
    print("  discriminative information present in the raw signal. This is")
    print("  a preservation claim, not an enhancement claim -- see diagnostic")
    print("  script for the architectural control (untrained encoder).")
    print()


def main():
    X_raw, X_lat, X_rand, y = load_and_aggregate()

    print("Running probes (5-fold stratified CV, balanced class weights)...\n")
    results = []
    for X, label in [
        (X_raw,  "Raw tactile"),
        (X_rand, "Random projection"),
        (X_lat,  "Trained encoder"),
    ]:
        r = run_probe(X, y, label)
        results.append(r)
        print(f"  {label:<28} done (ROC-AUC = {r['auc_mean']:.4f} ± {r['auc_std']:.3f})")
    print()

    n_episodes = len(y)
    n_success = int(y.sum())
    n_failure = int((1 - y).sum())
    print_results(results, n_episodes, n_success, n_failure)
    interpret(results)


if __name__ == "__main__":
    main()