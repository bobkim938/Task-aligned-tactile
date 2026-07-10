"""
Encoder Pre-training Data Analysis
====================================
Produces:
  1. Phase-segmented force signature plots per object category
  2. Slip event temporal profiles
  3. Class balance report for all auxiliary heads
  4. PCA dimensionality analysis
  5. Temporal discriminability check (single-frame vs window)
  6. Delta_E_slip analysis — continuous slip precursor validation

Usage:
  python tactile_encoder/analyze_encoder_data.py \
      --data-dir data/encoder_pretraining \
      --output-dir analysis/encoder_data \
      --window-size 16
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report


# =========================================================================
# Data loading
# =========================================================================

def load_all_data(data_dir: str):
    """Load and concatenate all mix HDF5 files into flat arrays."""
    records = defaultdict(list)
    metadata = []

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".h5"):
            continue
        fpath = os.path.join(data_dir, fname)
        mix_name = fname.replace(".h5", "")
        print(f"Loading {fpath}...")

        with h5py.File(fpath, "r") as f:
            for batch_key in sorted(f.keys()):
                if not batch_key.startswith("batch_"):
                    continue
                grp = f[batch_key]
                T = grp["tactile_smoothed"].shape[0]
                N = grp["tactile_smoothed"].shape[1]

                # Load arrays [T, N, ...]
                tactile_smoothed = grp["tactile_smoothed"][:]
                tactile_raw = grp["tactile_raw"][:]
                is_grasped = grp["is_grasped"][:]
                obj_pose = grp["obj_pose"][:]
                tcp_pose = grp["tcp_pose"][:]
                gripper_qpos = grp["gripper_qpos"][:]
                e_slip = grp["e_slip"][:]
                delta_e_slip = grp["delta_e_slip"][:]
                phase = grp["phase"][:]
                success = grp["success"][:]

                # Retrospective labels
                grasp_stability = grp["label_grasp_stability"][:]
                slip_imminent = grp["label_slip_imminent"][:]
                force_magnitude = grp["label_force_magnitude"][:]
                slip_rate = grp["label_slip_rate"][:]

                # Object IDs
                obj_ids = [x.decode() if isinstance(x, bytes) else x
                           for x in grp["object_ids"][:]]
                friction = grp["friction"][:]

                # Flatten [T, N] → [T*N] for per-timestep analysis
                for env_i in range(N):
                    records["tactile_smoothed"].append(tactile_smoothed[:, env_i, :])
                    records["tactile_raw"].append(tactile_raw[:, env_i, :])
                    records["is_grasped"].append(is_grasped[:, env_i])
                    records["e_slip"].append(e_slip[:, env_i])
                    records["delta_e_slip"].append(delta_e_slip[:, env_i])
                    records["phase"].append(phase[:, env_i])
                    records["grasp_stability"].append(grasp_stability[:, env_i])
                    records["slip_imminent"].append(slip_imminent[:, env_i])
                    records["force_magnitude"].append(force_magnitude[:, env_i])
                    records["slip_rate"].append(slip_rate[:, env_i])
                    records["obj_pose"].append(obj_pose[:, env_i, :])
                    records["tcp_pose"].append(tcp_pose[:, env_i, :])
                    records["gripper_qpos"].append(gripper_qpos[:, env_i, :])

                    metadata.append({
                        "mix": mix_name,
                        "batch": batch_key,
                        "env_idx": env_i,
                        "object_id": obj_ids[env_i] if env_i < len(obj_ids) else "unknown",
                        "friction": float(friction[env_i]) if env_i < len(friction) else 1.0,
                        "success": bool(success[env_i] > 0.5) if env_i < len(success) else False,
                        "T": T,
                    })

    print(f"Loaded {len(metadata)} trajectories")
    return records, metadata


# =========================================================================
# Analysis 1: Phase-segmented force statistics
# =========================================================================

def analyze_force_by_phase(records, metadata, output_dir):
    """Force signature statistics per phase, grouped by object success rate."""
    print("\n=== Analysis 1: Force signatures by phase ===")

    phase_names = {0: "free", 1: "grasp", 2: "transport", 3: "post_slip"}

    # Group objects by success rate
    obj_success = defaultdict(list)
    for m in metadata:
        obj_success[m["object_id"]].append(m["success"])
    obj_success_rate = {k: np.mean(v) for k, v in obj_success.items()}

    # Buckets
    buckets = {
        "high_success (>90%)": [],
        "mid_success (30-90%)": [],
        "low_success (<30%)": [],
    }
    for obj_id, rate in obj_success_rate.items():
        if rate > 0.9:
            buckets["high_success (>90%)"].append(obj_id)
        elif rate > 0.3:
            buckets["mid_success (30-90%)"].append(obj_id)
        else:
            buckets["low_success (<30%)"].append(obj_id)

    print("Object success buckets:")
    for bucket_name, obj_list in buckets.items():
        print(f"  {bucket_name}: {len(obj_list)} objects")

    # Compute per-phase, per-bucket force statistics
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle("Force Magnitude by Phase and Success Bucket", fontsize=14)

    for row_idx, (bucket_name, obj_list) in enumerate(buckets.items()):
        obj_set = set(obj_list)
        traj_indices = [i for i, m in enumerate(metadata) if m["object_id"] in obj_set]

        for col_idx, (phase_id, phase_name) in enumerate(phase_names.items()):
            ax = axes[row_idx, col_idx]
            all_forces = []

            for traj_idx in traj_indices:
                tact = records["tactile_raw"][traj_idx]  # [T, 6]
                phases = records["phase"][traj_idx]       # [T]
                mask = phases == phase_id
                if mask.sum() > 0:
                    force_mag = np.linalg.norm(tact[mask], axis=1)
                    all_forces.extend(force_mag.tolist())

            if all_forces:
                ax.hist(all_forces, bins=50, alpha=0.7, density=True)
                ax.set_title(f"{phase_name}\nmean={np.mean(all_forces):.1f}N")
            else:
                ax.set_title(f"{phase_name}\nno data")

            if col_idx == 0:
                ax.set_ylabel(bucket_name)
            ax.set_xlabel("Force (N)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "force_by_phase_and_success.png"), dpi=150)
    plt.close()
    print("  Saved force_by_phase_and_success.png")


# =========================================================================
# Analysis 2: Contact symmetry
# =========================================================================

def analyze_contact_symmetry(records, metadata, output_dir):
    """Force asymmetry during grasp phase, grouped by success."""
    print("\n=== Analysis 2: Contact symmetry ===")

    asymmetries_success = []
    asymmetries_fail = []

    for i, m in enumerate(metadata):
        tact = records["tactile_raw"][i]   # [T, 6]
        phases = records["phase"][i]        # [T]
        grasp_mask = (phases == 1) | (phases == 2)  # grasp or transport

        if grasp_mask.sum() < 3:
            continue

        # Left finger: channels 0,1,2. Right finger: channels 3,4,5
        f_left = np.linalg.norm(tact[grasp_mask, :3], axis=1)
        f_right = np.linalg.norm(tact[grasp_mask, 3:], axis=1)
        denom = f_left + f_right + 1e-6
        asym = np.abs(f_left - f_right) / denom
        mean_asym = asym.mean()

        if m["success"]:
            asymmetries_success.append(mean_asym)
        else:
            asymmetries_fail.append(mean_asym)

    fig, ax = plt.subplots(figsize=(8, 5))
    if asymmetries_success:
        ax.hist(asymmetries_success, bins=40, alpha=0.6, label=f"Success (n={len(asymmetries_success)})", density=True)
    if asymmetries_fail:
        ax.hist(asymmetries_fail, bins=40, alpha=0.6, label=f"Failure (n={len(asymmetries_fail)})", density=True)
    ax.set_xlabel("Force Asymmetry |L-R|/(L+R)")
    ax.set_ylabel("Density")
    ax.set_title("Contact Symmetry: Success vs Failure")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "contact_symmetry.png"), dpi=150)
    plt.close()
    print(f"  Success mean asymmetry: {np.mean(asymmetries_success):.3f}" if asymmetries_success else "  No success data")
    print(f"  Failure mean asymmetry: {np.mean(asymmetries_fail):.3f}" if asymmetries_fail else "  No failure data")


# =========================================================================
# Analysis 3: Slip event temporal profiles
# =========================================================================

def analyze_slip_events(records, metadata, output_dir, pre_window=15, post_window=5):
    """Extract and plot force profiles around slip events."""
    print("\n=== Analysis 3: Slip event temporal profiles ===")

    slip_force_profiles = []     # [pre+post, 6]
    slip_delta_e_profiles = []   # [pre+post]
    near_slip_force_profiles = []

    total_slips = 0
    total_near_slips = 0

    for i in range(len(metadata)):
        is_grasped = records["is_grasped"][i]     # [T]
        tact = records["tactile_smoothed"][i]      # [T, 6]
        delta_e = records["delta_e_slip"][i]       # [T]
        phase = records["phase"][i]                 # [T]
        T = len(is_grasped)

        for t in range(1, T):
            # Slip event: grasped -> not grasped during transport
            if is_grasped[t-1] > 0.5 and is_grasped[t] < 0.5 and phase[t-1] == 2:
                total_slips += 1
                start = max(0, t - pre_window)
                end = min(T, t + post_window)
                if t - start >= pre_window:  # only full windows
                    profile = tact[start:end]
                    if len(profile) == pre_window + post_window:
                        slip_force_profiles.append(profile)
                        slip_delta_e_profiles.append(delta_e[start:end])

            # Near-slip: force drops >30% over 3 steps but grasp holds
            if t >= 3 and is_grasped[t] > 0.5 and phase[t] == 2:
                force_now = np.linalg.norm(tact[t])
                force_prev = np.linalg.norm(tact[t-3])
                if force_prev > 1.0 and (force_now / force_prev) < 0.7:
                    total_near_slips += 1
                    start = max(0, t - pre_window)
                    end = min(T, t + post_window)
                    if t - start >= pre_window and len(tact[start:end]) == pre_window + post_window:
                        near_slip_force_profiles.append(tact[start:end])

    print(f"  Total slip events: {total_slips}")
    print(f"  Full-window slip profiles: {len(slip_force_profiles)}")
    print(f"  Total near-slip events: {total_near_slips}")
    print(f"  Full-window near-slip profiles: {len(near_slip_force_profiles)}")

    # Plot average slip profile
    if slip_force_profiles:
        profiles = np.array(slip_force_profiles)  # [n_events, window, 6]
        mean_profile = profiles.mean(axis=0)       # [window, 6]
        force_mag = np.linalg.norm(mean_profile, axis=1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        time_axis = np.arange(-pre_window, post_window)

        # Force magnitude
        ax = axes[0]
        ax.plot(time_axis, force_mag, 'b-', linewidth=2)
        ax.axvline(x=0, color='r', linestyle='--', label='Slip event')
        ax.set_xlabel("Timestep relative to slip")
        ax.set_ylabel("Force magnitude (normalized)")
        ax.set_title(f"Average Force Profile at Slip (n={len(slip_force_profiles)})")
        ax.legend()

        # Per-component
        ax = axes[1]
        labels = ["F_lx", "F_ly", "F_lz", "F_rx", "F_ry", "F_rz"]
        for ch in range(6):
            ax.plot(time_axis, mean_profile[:, ch], label=labels[ch])
        ax.axvline(x=0, color='r', linestyle='--')
        ax.set_xlabel("Timestep relative to slip")
        ax.set_ylabel("Force component (normalized)")
        ax.set_title("Per-Component Force at Slip")
        ax.legend(fontsize=8)

        # delta_E_slip
        if slip_delta_e_profiles:
            de_profiles = np.array(slip_delta_e_profiles)
            mean_de = de_profiles.mean(axis=0)
            ax = axes[2]
            ax.plot(time_axis, mean_de, 'g-', linewidth=2)
            ax.axvline(x=0, color='r', linestyle='--', label='Slip event')
            ax.set_xlabel("Timestep relative to slip")
            ax.set_ylabel("ΔE_slip (m)")
            ax.set_title("Average ΔE_slip Profile at Slip")
            ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "slip_event_profiles.png"), dpi=150)
        plt.close()
        print("  Saved slip_event_profiles.png")
    else:
        print("  WARNING: No slip events with full windows found!")


# =========================================================================
# Analysis 4: Class balance audit
# =========================================================================

def analyze_class_balance(records, metadata, output_dir):
    """Count label distributions for each auxiliary head."""
    print("\n=== Analysis 4: Class balance audit ===")

    # Flatten all trajectories
    all_is_grasped = np.concatenate(records["is_grasped"])
    all_stability = np.concatenate(records["grasp_stability"])
    all_slip = np.concatenate(records["slip_imminent"])
    all_phase = np.concatenate(records["phase"])

    total = len(all_is_grasped)

    report = []

    # Contact classifier
    contact_pos = (all_is_grasped > 0.5).sum()
    contact_neg = total - contact_pos
    report.append(f"Contact classifier:  True={contact_pos} ({contact_pos/total:.1%})  "
                  f"False={contact_neg} ({contact_neg/total:.1%})")

    # Stability predictor (only when grasped)
    grasped_mask = all_is_grasped > 0.5
    if grasped_mask.sum() > 0:
        stab = all_stability[grasped_mask]
        stab_pos = (stab > 0.5).sum()
        stab_neg = len(stab) - stab_pos
        report.append(f"Stability predictor: Stable={stab_pos} ({stab_pos/len(stab):.1%})  "
                      f"Unstable={stab_neg} ({stab_neg/len(stab):.1%})")
    else:
        report.append("Stability predictor: No grasped timesteps!")

    # Slip predictor (only when grasped)
    if grasped_mask.sum() > 0:
        slip = all_slip[grasped_mask]
        slip_pos = (slip > 0.5).sum()
        slip_neg = len(slip) - slip_pos
        report.append(f"Slip predictor:      Imminent={slip_pos} ({slip_pos/len(slip):.1%})  "
                      f"Safe={slip_neg} ({slip_neg/len(slip):.1%})")
        if slip_pos / len(slip) < 0.02:
            report.append("  !!!  SEVERE IMBALANCE: <2% slip events. Consider focal loss or oversampling.")
    else:
        report.append("Slip predictor: No grasped timesteps!")

    # Phase classifier
    phase_counts = {}
    phase_names = {0: "free", 1: "grasp", 2: "transport", 3: "post_slip"}
    for pid, pname in phase_names.items():
        count = (all_phase == pid).sum()
        phase_counts[pname] = count
        report.append(f"Phase '{pname}':       {count} ({count/total:.1%})")

    # Print and save
    for line in report:
        print(f"  {line}")

    with open(os.path.join(output_dir, "class_balance_report.txt"), "w") as f:
        f.write("Class Balance Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total timesteps: {total}\n\n")
        for line in report:
            f.write(line + "\n")
    print("  Saved class_balance_report.txt")

    return dict(slip_ratio=slip_pos / max(len(slip), 1) if grasped_mask.sum() > 0 else 0)


# =========================================================================
# Analysis 5: PCA dimensionality
# =========================================================================

def analyze_dimensionality(records, metadata, output_dir, window_size=16):
    """PCA on force windows to check how many dimensions capture the variance."""
    print(f"\n=== Analysis 5: PCA dimensionality (window={window_size}) ===")

    # Construct windows from transport phase
    windows = []
    labels_success = []
    labels_phase = []
    max_windows = 50000  # cap for memory

    for i in range(len(metadata)):
        tact = records["tactile_smoothed"][i]  # [T, 6]
        phase = records["phase"][i]             # [T]
        T = len(tact)

        for t in range(window_size, T):
            if phase[t] >= 1:  # grasp or transport
                window = tact[t-window_size:t].flatten()  # [window*6]
                windows.append(window)
                labels_success.append(metadata[i]["success"])
                labels_phase.append(phase[t])

                if len(windows) >= max_windows:
                    break
        if len(windows) >= max_windows:
            break

    windows = np.array(windows)
    labels_success = np.array(labels_success)
    labels_phase = np.array(labels_phase)

    print(f"  Constructed {len(windows)} windows of shape {windows.shape}")

    # PCA
    pca = PCA(n_components=min(50, windows.shape[1]))
    pca.fit(windows)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_90 = np.searchsorted(cumvar, 0.90) + 1
    n_95 = np.searchsorted(cumvar, 0.95) + 1

    print(f"  Components for 90% variance: {n_90}")
    print(f"  Components for 95% variance: {n_95}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Explained variance
    ax = axes[0]
    ax.plot(cumvar, 'b-o', markersize=3)
    ax.axhline(y=0.90, color='r', linestyle='--', label='90%')
    ax.axhline(y=0.95, color='g', linestyle='--', label='95%')
    ax.axvline(x=n_90-1, color='r', linestyle=':', alpha=0.5)
    ax.axvline(x=n_95-1, color='g', linestyle=':', alpha=0.5)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA: Explained Variance")
    ax.legend()

    # PC1 vs PC2 colored by success
    pc_scores = pca.transform(windows)[:, :3]
    ax = axes[1]
    scatter = ax.scatter(pc_scores[:, 0], pc_scores[:, 1],
                         c=labels_success.astype(float), cmap='RdYlGn',
                         alpha=0.3, s=2)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA: Colored by Success")
    plt.colorbar(scatter, ax=ax, label="Success")

    # PC1 vs PC2 colored by phase
    ax = axes[2]
    scatter = ax.scatter(pc_scores[:, 0], pc_scores[:, 1],
                         c=labels_phase, cmap='viridis',
                         alpha=0.3, s=2)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA: Colored by Phase")
    plt.colorbar(scatter, ax=ax, label="Phase")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_analysis.png"), dpi=150)
    plt.close()
    print("  Saved pca_analysis.png")


# =========================================================================
# Analysis 6: Temporal discriminability
# =========================================================================

def analyze_temporal_discriminability(records, metadata, output_dir, window_size=16):
    """Can predict slip from single-frame vs temporal window?"""
    print(f"\n=== Analysis 6: Temporal discriminability ===")

    # Collect samples: force at t, force window [t-K:t], slip label
    X_single = []
    X_window = []
    y_labels = []

    for i in range(len(metadata)):
        tact = records["tactile_smoothed"][i]  # [T, 6]
        slip = records["slip_imminent"][i]      # [T]
        is_grasped = records["is_grasped"][i]   # [T]
        T = len(tact)

        for t in range(window_size, T):
            if is_grasped[t] < 0.5:
                continue
            X_single.append(tact[t])
            X_window.append(tact[t-window_size:t].flatten())
            y_labels.append(slip[t])

    X_single = np.array(X_single)
    X_window = np.array(X_window)
    y_labels = np.array(y_labels)

    print(f"  Samples: {len(y_labels)} (positive: {(y_labels>0.5).sum()}, "
          f"negative: {(y_labels<0.5).sum()})")

    if (y_labels > 0.5).sum() < 10:
        print("  !!!  Too few positive samples for meaningful classification!")
        print("  Cannot assess temporal discriminability. Need more slip events.")
        return

    # Train/test split (simple 80/20)
    n = len(y_labels)
    perm = np.random.RandomState(42).permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]

    # Single-frame logistic regression
    clf_single = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf_single.fit(X_single[train_idx], y_labels[train_idx] > 0.5)
    pred_single = clf_single.predict(X_single[test_idx])
    acc_single = accuracy_score(y_labels[test_idx] > 0.5, pred_single)
    f1_single = f1_score(y_labels[test_idx] > 0.5, pred_single, zero_division=0)

    # Window logistic regression
    clf_window = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf_window.fit(X_window[train_idx], y_labels[train_idx] > 0.5)
    pred_window = clf_window.predict(X_window[test_idx])
    acc_window = accuracy_score(y_labels[test_idx] > 0.5, pred_window)
    f1_window = f1_score(y_labels[test_idx] > 0.5, pred_window, zero_division=0)

    print(f"\n  Single-frame (6D):  Accuracy={acc_single:.3f}  F1={f1_single:.3f}")
    print(f"  Window ({window_size}x6={window_size*6}D): Accuracy={acc_window:.3f}  F1={f1_window:.3f}")

    if f1_window > f1_single + 0.05:
        print("    Temporal window provides meaningful improvement -> Conv1D justified")
    elif f1_single > 0.9:
        print("  !!!  Single frame already strong — temporal window adds little")
        print("     Consider simpler MLP encoder instead of Conv1D")
    else:
        print("  !!!  Both weak — slip signal may not be learnable from force alone")
        print("     Check if slip events have consistent force precursors")

    # Save report
    with open(os.path.join(output_dir, "temporal_discriminability.txt"), "w") as f:
        f.write("Temporal Discriminability Analysis\n")
        f.write("=" * 50 + "\n")
        f.write(f"Window size: {window_size}\n")
        f.write(f"Total samples: {n}\n")
        f.write(f"Positive (slip imminent): {(y_labels>0.5).sum()}\n")
        f.write(f"Negative (safe): {(y_labels<0.5).sum()}\n\n")
        f.write(f"Single-frame:  Acc={acc_single:.4f}  F1={f1_single:.4f}\n")
        f.write(f"Window:        Acc={acc_window:.4f}  F1={f1_window:.4f}\n")
    print("  Saved temporal_discriminability.txt")


# =========================================================================
# Analysis 7: delta_E_slip continuous slip indicator
# =========================================================================

def analyze_delta_e_slip(records, metadata, output_dir):
    """Validate delta_E_slip as a continuous slip precursor."""
    print("\n=== Analysis 7: delta_E_slip analysis ===")

    # Compare delta_E_slip distributions for stable vs pre-slip timesteps
    delta_e_stable = []
    delta_e_preslip = []

    for i in range(len(metadata)):
        is_grasped = records["is_grasped"][i]
        delta_e = records["delta_e_slip"][i]
        slip = records["slip_imminent"][i]
        T = len(is_grasped)

        for t in range(T):
            if is_grasped[t] < 0.5:
                continue
            if slip[t] > 0.5:
                delta_e_preslip.append(delta_e[t])
            else:
                delta_e_stable.append(delta_e[t])

    print(f"  Stable timesteps:   {len(delta_e_stable)}")
    print(f"  Pre-slip timesteps: {len(delta_e_preslip)}")

    if not delta_e_preslip:
        print("  !!!  No pre-slip timesteps found!")
        return

    delta_e_stable = np.array(delta_e_stable)
    delta_e_preslip = np.array(delta_e_preslip)

    print(f"  Stable delta_E_slip:   mean={delta_e_stable.mean():.4f}  std={delta_e_stable.std():.4f}")
    print(f"  Pre-slip delta_E_slip: mean={delta_e_preslip.mean():.4f}  std={delta_e_preslip.std():.4f}")

    # Separation metric
    if delta_e_stable.std() > 0:
        cohens_d = abs(delta_e_preslip.mean() - delta_e_stable.mean()) / delta_e_stable.std()
        print(f"  Cohen's d (effect size): {cohens_d:.3f}")
        if cohens_d > 0.8:
            print("    Strong separation — delta_E_slip is a powerful slip indicator")
        elif cohens_d > 0.5:
            print("  ~ Moderate separation — delta_E_slip is useful but noisy")
        else:
            print("  !!!  Weak separation — delta_E_slip may not distinguish slip well")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(delta_e_stable, bins=50, alpha=0.6, density=True,
            label=f"Stable (n={len(delta_e_stable)})")
    ax.hist(delta_e_preslip, bins=50, alpha=0.6, density=True,
            label=f"Pre-slip (n={len(delta_e_preslip)})")
    ax.set_xlabel("delta_E_slip (m)")
    ax.set_ylabel("Density")
    ax.set_title("delta_E_slip Distribution: Stable vs Pre-Slip")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "delta_e_slip_analysis.png"), dpi=150)
    plt.close()
    print("  Saved delta_e_slip_analysis.png")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze encoder pre-training data")
    parser.add_argument("--data-dir", type=str, default="data/encoder_pretraining")
    parser.add_argument("--output-dir", type=str, default="analysis/pretraining_data")
    parser.add_argument("--window-size", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    records, metadata = load_all_data(args.data_dir)

    if not metadata:
        print("ERROR: No data found! Run collect_encoder_data.py first.")
        return

    # Run all analyses
    analyze_force_by_phase(records, metadata, args.output_dir)
    analyze_contact_symmetry(records, metadata, args.output_dir)
    analyze_slip_events(records, metadata, args.output_dir)
    balance_info = analyze_class_balance(records, metadata, args.output_dir)
    analyze_dimensionality(records, metadata, args.output_dir, args.window_size)
    analyze_temporal_discriminability(records, metadata, args.output_dir, args.window_size)
    analyze_delta_e_slip(records, metadata, args.output_dir)


if __name__ == "__main__":
    main()