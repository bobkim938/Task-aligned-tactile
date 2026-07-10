"""
Multi-seed evaluation across all ablation conditions on Franka.

For each (condition, seed, object) combination, runs 10 parallel episodes
and records per-episode success. Saves raw results to CSV

Usage:
    $ python evaluation/eval_multiseed.py

Output:
    results/multiseed_eval_raw.csv     -- one row per (condition, seed, obj, env_idx)
    results/multiseed_eval_summary.csv -- per-condition mean +/- std across seeds
    stdout: headline table
"""

import csv
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

# Add project root to sys.path BEFORE importing project modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import env.PickAndPlaceYCBToTray as env_module
from training.ppo import Agent
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


# =============================================================================
# CONFIG -- edit these
# =============================================================================

CHECKPOINTS = {
    "oracle":          str(PROJECT_ROOT / "checkpoints" / "policy_oracle.pt"),
    "visual_only":     str(PROJECT_ROOT / "checkpoints" / "policy_visual_only.pt"),
    "visual_raw_tact": str(PROJECT_ROOT / "checkpoints" / "policy_raw_tactile.pt"),
    "visual_encoder":  str(PROJECT_ROOT / "checkpoints" / "policy_encoder.pt"),
}

SEEDS = [42, 123, 456, 789, 1000, 1234, 2025, 7777, 31415, 65535]
NUM_PARALLEL_ENVS = 10        # episodes per (condition, seed, object)
NUM_EVAL_STEPS = 100          # match env's max_episode_steps
OUTPUT_DIR = Path("results")

ROBOT_UID = "panda_v2"


# =============================================================================
# Core eval logic
# =============================================================================

def evaluate_seed_condition(condition, seed, checkpoint_path, device):
    """Run full per-object evaluation for one (condition, seed) pair.

    Returns
    -------
    list of dict, one row per (object, env_idx), with keys:
        condition, seed, model_id, env_idx, success
    """
    env_module.EXPERIMENT_MODE = "visual_encoder" if condition.startswith("visual_encoder") else condition

    torch.manual_seed(seed)
    np.random.seed(seed)

    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="physx_cuda",
        control_mode="pd_joint_delta_pos",
        robot_uids=ROBOT_UID,
    )

    envs = gym.make(
        "PickAndPlaceYCBToTray-v1",
        num_envs=NUM_PARALLEL_ENVS,
        reconfiguration_freq=1,
        **env_kwargs,
    )
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)

    envs_vec = ManiSkillVectorEnv(
        envs, NUM_PARALLEL_ENVS, ignore_terminations=True, record_metrics=True
    )

    # Build and load policy
    base_env = envs.unwrapped
    if hasattr(base_env, "_tactile_dropout_rate"):
        base_env._tactile_dropout_rate = 0.0

    agent = Agent(envs_vec).to(device)
    agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.eval()

    # Snapshot the full object list, then iterate by restricting one at a time
    all_objects = base_env.all_model_ids.copy()
    rows = []

    for obj_id in all_objects:
        # Restrict environment to this single object and reconfigure
        base_env.all_model_ids = np.array([obj_id])
        eval_obs, _ = envs_vec.reset(seed=seed, options=dict(reconfigure=True))

        # Track success
        successes = torch.zeros(NUM_PARALLEL_ENVS, dtype=torch.bool, device=device)

        with torch.no_grad():
            for _ in range(NUM_EVAL_STEPS):
                action = agent.get_action(eval_obs, deterministic=True)
                eval_obs, _, _, _, eval_infos = envs_vec.step(action)
                if "success" in eval_infos:
                    successes |= eval_infos["success"]

        for env_idx in range(NUM_PARALLEL_ENVS):
            rows.append({
                "condition": condition,
                "seed": seed,
                "model_id": obj_id,
                "env_idx": env_idx,
                "success": int(successes[env_idx].item()),
            })

    base_env.all_model_ids = all_objects
    envs_vec.close()

    return rows


# =============================================================================
# Aggregation and reporting
# =============================================================================
def aggregate_summary(all_rows):
    """Compute per-condition mean +/- std across seeds.
    Returns: dict {condition: {"per_seed": [...], "mean": float, "std": float}}
    """
    summary = {}
    conditions = sorted({r["condition"] for r in all_rows})
    for cond in conditions:
        cond_rows = [r for r in all_rows if r["condition"] == cond]
        seeds = sorted({r["seed"] for r in cond_rows})
        per_seed_rates = []
        for s in seeds:
            seed_rows = [r for r in cond_rows if r["seed"] == s]
            rate = 100.0 * np.mean([r["success"] for r in seed_rows])
            per_seed_rates.append(rate)
        summary[cond] = dict(
            per_seed=per_seed_rates,
            mean=float(np.mean(per_seed_rates)),
            std=float(np.std(per_seed_rates, ddof=1)) if len(per_seed_rates) > 1 else 0.0,
        )
    return summary


def write_raw_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(summary, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "mean_success_pct", "std_success_pct",
                         "n_seeds", "per_seed_rates"])
        for cond, stats in summary.items():
            writer.writerow([
                cond,
                f"{stats['mean']:.2f}",
                f"{stats['std']:.2f}",
                len(stats["per_seed"]),
                ";".join(f"{r:.2f}" for r in stats["per_seed"]),
            ])


def print_headline_table(summary):
    print()
    print("=" * 72)
    print(f"{'Condition':<22} {'Mean (%)':>10} {'Std (%)':>10} {'N seeds':>10}")
    print("-" * 72)
    for cond, stats in summary.items():
        print(f"{cond:<22} {stats['mean']:>10.2f} {stats['std']:>10.2f} "
              f"{len(stats['per_seed']):>10d}")
    print("=" * 72)


# =============================================================================
# Main
# =============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    active_conditions = []
    for cond, ckpt in CHECKPOINTS.items():
        if ckpt is None:
            print(f"[skip] {cond}: no checkpoint set")
            continue
        if not os.path.exists(ckpt):
            print(f"[skip] {cond}: checkpoint not found at {ckpt}")
            continue
        active_conditions.append((cond, ckpt))

    if not active_conditions:
        print("No valid checkpoints. Edit CHECKPOINTS at the top of this file.")
        sys.exit(1)

    print(f"Device      : {device}")
    print(f"Robot       : {ROBOT_UID}")
    print(f"Conditions  : {[c for c, _ in active_conditions]}")
    print(f"Seeds       : {SEEDS}")
    print(f"Output dir  : {OUTPUT_DIR.resolve()}")
    print()

    all_rows = []
    raw_path = OUTPUT_DIR / "multiseed_eval_raw.csv"
    t_start = time.time()

    for cond, ckpt in active_conditions:
        print(f"\n### Condition: {cond}  ({ckpt})")
        for seed in SEEDS:
            t_seed = time.time()
            rows = evaluate_seed_condition(cond, seed, ckpt, device)
            all_rows.extend(rows)

            write_raw_csv(all_rows, raw_path)

            # Per-seed quick summary
            successes = [r["success"] for r in rows]
            rate = 100.0 * np.mean(successes)
            n_objs = len({r["model_id"] for r in rows})
            elapsed = time.time() - t_seed
            print(f"  seed={seed:>5d}  rate={rate:5.1f}%  "
                  f"objects={n_objs}  episodes={len(successes)}  "
                  f"({elapsed:.1f}s)")

    # Final aggregation
    summary = aggregate_summary(all_rows)
    write_summary_csv(summary, OUTPUT_DIR / "multiseed_eval_summary.csv")
    print_headline_table(summary)

    total_elapsed = (time.time() - t_start) / 60.0
    print(f"\nTotal wall-clock: {total_elapsed:.1f} min")
    print(f"Raw results    : {raw_path.resolve()}")
    print(f"Summary        : {(OUTPUT_DIR / 'multiseed_eval_summary.csv').resolve()}")


if __name__ == "__main__":
    main()