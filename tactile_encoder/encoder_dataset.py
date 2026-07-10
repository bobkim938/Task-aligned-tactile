"""
Encoder Pre-training Dataset (v2 — with temporal consistency pairs)

Returns consecutive force windows (t-1, t) for temporal consistency loss.

Each sample:
    force_window:      [K, 6]  smoothed+normalized tactile window ending at t
    force_window_prev: [K, 6]  smoothed+normalized tactile window ending at t-1
    same_phase:        scalar  1.0 if phase(t) == phase(t-1), else 0.0
    contact:           scalar  is_grasped at timestep t
    symmetry:          scalar  force asymmetry at t
    stability:         scalar  will grasp hold for next 15 steps?
    phase:             long    {0=free, 1=grasp, 2=transport, 3=post_slip}
    force_mag:         scalar  log(1 + ||F_raw||) at t, normalized
"""

import os
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TactileWindowDataset(Dataset):
    """
    Indexes every valid (trajectory, timestep) pair across all HDF5 files.
    Returns consecutive window pairs for temporal consistency training.
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = 16,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
        max_files: int = None,
    ):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.split = split

        # Find all h5 files
        h5_files = sorted(self.data_dir.glob("*.h5"))
        if max_files is not None:
            h5_files = h5_files[:max_files]
        if not h5_files:
            raise FileNotFoundError(f"No .h5 files in {data_dir}")

        self.h5_paths = [str(p) for p in h5_files]
        print(f"Found {len(self.h5_paths)} HDF5 files:")
        for p in self.h5_paths:
            print(f"  {p}")

        # Build index: list of (file_idx, batch_key, env_idx, t)
        self.samples: List[Tuple[int, str, int, int]] = []
        trajectory_ids: List[str] = []
        trajectory_sample_indices: List[List[int]] = []

        print("\nIndexing samples (this may take a minute)...")
        for file_idx, h5_path in enumerate(self.h5_paths):
            with h5py.File(h5_path, "r") as f:
                batch_keys = sorted([k for k in f.keys() if k.startswith("batch_")])
                for batch_key in batch_keys:
                    grp = f[batch_key]
                    T = grp["tactile_smoothed"].shape[0]
                    N = grp["tactile_smoothed"].shape[1]

                    for env_idx in range(N):
                        traj_id = f"{file_idx}_{batch_key}_{env_idx}"
                        traj_sample_idxs = []
                        # Start from window_size (not window_size-1) so that
                        # t-1 also has a full window of K frames behind it
                        for t in range(window_size, T):
                            sample_idx = len(self.samples)
                            self.samples.append((file_idx, batch_key, env_idx, t))
                            traj_sample_idxs.append(sample_idx)
                        trajectory_ids.append(traj_id)
                        trajectory_sample_indices.append(traj_sample_idxs)

        print(f"Indexed {len(self.samples):,} samples across "
              f"{len(trajectory_ids):,} trajectories")

        # Train/val split BY TRAJECTORY
        rng = np.random.RandomState(seed)
        n_traj = len(trajectory_ids)
        perm = rng.permutation(n_traj)
        n_val = int(val_fraction * n_traj)

        if split == "train":
            selected_traj = perm[n_val:]
        elif split == "val":
            selected_traj = perm[:n_val]
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split}")

        selected_samples = set()
        for traj_idx in selected_traj:
            selected_samples.update(trajectory_sample_indices[traj_idx])
        self._valid_indices = sorted(selected_samples)

        print(f"  Split '{split}': {len(self._valid_indices):,} samples "
              f"from {len(selected_traj):,} trajectories")

        self._h5_handles = {}

    def __len__(self):
        return len(self._valid_indices)

    def _get_handle(self, file_idx: int):
        if file_idx not in self._h5_handles:
            self._h5_handles[file_idx] = h5py.File(self.h5_paths[file_idx], "r")
        return self._h5_handles[file_idx]

    def __getitem__(self, idx: int):
        sample_idx = self._valid_indices[idx]
        file_idx, batch_key, env_idx, t = self.samples[sample_idx]
        h5f = self._get_handle(file_idx)
        grp = h5f[batch_key]

        K = self.window_size

        # --- Current window [K, 6] ending at t ---
        start_curr = t - K + 1
        end_curr = t + 1
        force_window = grp["tactile_smoothed"][start_curr:end_curr, env_idx, :]  # [K, 6]

        # --- Previous window [K, 6] ending at t-1 ---
        start_prev = t - K
        end_prev = t
        force_window_prev = grp["tactile_smoothed"][start_prev:end_prev, env_idx, :]  # [K, 6]

        # --- Same phase flag ---
        phase_curr = int(grp["phase"][t, env_idx])
        phase_prev = int(grp["phase"][t - 1, env_idx])
        same_phase = float(phase_curr == phase_prev)

        # --- Labels at timestep t ---
        is_grasped = float(grp["is_grasped"][t, env_idx])
        stability = float(grp["label_grasp_stability"][t, env_idx])
        force_mag = float(grp["label_force_magnitude"][t, env_idx])

        # --- Contact symmetry from raw tactile at t ---
        raw_t = grp["tactile_raw"][t, env_idx, :]  # [6]
        f_left = float(np.linalg.norm(raw_t[:3]))
        f_right = float(np.linalg.norm(raw_t[3:]))
        denom = f_left + f_right + 1e-6
        symmetry = abs(f_left - f_right) / denom

        force_mag_norm = force_mag / 6.5

        return dict(
            force_window=torch.from_numpy(force_window).float(),
            force_window_prev=torch.from_numpy(force_window_prev).float(),
            same_phase=torch.tensor(same_phase, dtype=torch.float32),
            contact=torch.tensor(is_grasped, dtype=torch.float32),
            symmetry=torch.tensor(symmetry, dtype=torch.float32),
            stability=torch.tensor(stability, dtype=torch.float32),
            phase=torch.tensor(phase_curr, dtype=torch.long),
            force_mag=torch.tensor(force_mag_norm, dtype=torch.float32),
        )

    def close(self):
        for h in self._h5_handles.values():
            h.close()
        self._h5_handles.clear()


def collate_fn(batch):
    return {
        k: torch.stack([b[k] for b in batch]) for k in batch[0].keys()
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/encoder_pretraining")
    parser.add_argument("--window-size", type=int, default=16)
    args = parser.parse_args()

    ds_train = TactileWindowDataset(
        data_dir=args.data_dir,
        window_size=args.window_size,
        split="train",
    )
    ds_val = TactileWindowDataset(
        data_dir=args.data_dir,
        window_size=args.window_size,
        split="val",
    )

    print(f"\nTrain samples: {len(ds_train):,}")
    print(f"Val samples:   {len(ds_val):,}")

    sample = ds_train[0]
    print("\nSample shapes and values:")
    for k, v in sample.items():
        if v.dim() == 0:
            print(f"  {k}: scalar = {v.item():.4f}")
        else:
            print(f"  {k}: shape {tuple(v.shape)}, dtype {v.dtype}")

    from torch.utils.data import DataLoader
    loader = DataLoader(ds_train, batch_size=32, shuffle=True,
                        num_workers=0, collate_fn=collate_fn)
    batch = next(iter(loader))
    print("\nBatch shapes:")
    for k, v in batch.items():
        print(f"  {k}: shape {tuple(v.shape)}, dtype {v.dtype}")