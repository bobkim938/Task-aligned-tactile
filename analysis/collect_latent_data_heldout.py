"""
collect_latent_data_heldout.py
==============================
Collect per-episode V2-deployment data on the 16 YCB objects that were held
out from both encoder training and PPO policy training. The encoder has never
seen their force signatures; the policy has never trained against them. This
is out-of-distribution test of the encoder's representational
generalization.

Strategy: with the held-out objects re-added to the env's asset pool, each
reset samples from 62+16 = 78 objects with ~20% chance of being a held-out
target

Saves: data/figure_deployment/4_4_heldout/latent_records.npz
       data/figure_deployment/4_4_heldout/meta.json

Usage:
    python collect_latent_data_heldout.py --policy   checkpoints/policy_encoder_seed4796.pt --encoder  checkpoints/tactile_encoder_hinge/best_encoder.pt --num-episodes 300
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "data" / "PickAndPlaceYCBToTray"))

import env.PickAndPlaceYCBToTray as env_module
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from tactile_encoder.tactile_encoder import TactileEncoder

NUM_STEPS = 100


# The 16 YCB objects held out from both encoder training AND PPO policy
# training. The encoder has never seen their force signatures.
HELDOUT_OBJECTS = {
    "003_cracker_box",
    "004_sugar_box",
    "006_mustard_bottle",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "022_windex_bottle",
    "024_bowl",
    "026_sponge",
    "028_skillet_lid",
    "029_plate",
    "035_power_drill",
    "036_wood_block",
    "052_extra_large_clamp",
    "059_chain",
    "072-a_toy_airplane",
    "072-b_toy_airplane",
}


# ---------------------------------------------------------------------------
# 3-phase scheme (post-slip merged into free)
# ---------------------------------------------------------------------------
def determine_phase(prev_phase, is_grasped, obj_z, initial_z):
    if is_grasped < 0.5:
        return 0
    if prev_phase != 2:
        return 2 if obj_z > initial_z + 0.04 else 1
    return 1 if obj_z < initial_z + 0.02 else 2


def reload_encoder(env_unwrapped, encoder_checkpoint, device):
    enc = TactileEncoder(in_channels=6, window_size=16, hidden_dim=64, z_dim=16)
    state = torch.load(encoder_checkpoint, map_location="cpu")
    enc.load_state_dict(state)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    env_unwrapped._tactile_encoder = enc.to(device)
    print(f"[hot-swap] Loaded encoder from {encoder_checkpoint}")


def rollout_episode(environment, policy, device, seed):
    """Reset, check the sampled object, run rollout only if it's a held-out target.

    Returns (record_dict, obj_name). record_dict is None if the sampled object
    was not in the held-out set (signals 'discard, try next seed').
    """
    obs, _ = environment.reset(seed=seed, options=dict(reconfigure=True))
    uw = environment.unwrapped

    initial_z = float(uw.obj.pose.p[0, 2].cpu().item())
    obj_name = uw._objs[0].name.rsplit("-", 1)[0] if uw._objs else "unknown"

    if obj_name not in HELDOUT_OBJECTS:
        return None, obj_name

    z_t_log = []
    tactile_raw_log = []
    is_grasped_log = []
    phase_log = []
    obj_pos_log = []
    prev_phase = 0
    success = False

    for _ in range(NUM_STEPS):
        with torch.no_grad():
            action = policy.get_action(obs.float().to(device), deterministic=True)
        obs, _, _, _, info = environment.step(action)

        prev_z = getattr(uw, "_prev_z_t", None)
        if prev_z is None:
            raise RuntimeError(
                "Env did not expose _prev_z_t. Confirm EXPERIMENT_MODE='visual_encoder'."
            )
        z_t_arr = prev_z[0].detach().cpu().numpy()
        raw_tact = uw._raw_tactile[0].cpu().numpy()
        is_grasped = float(info["is_grasped"][0].cpu().item())
        obj_pos = uw.obj.pose.p[0].cpu().numpy()
        obj_z = float(obj_pos[2])

        phase = determine_phase(prev_phase, is_grasped, obj_z, initial_z)
        prev_phase = phase

        z_t_log.append(z_t_arr)
        tactile_raw_log.append(raw_tact)
        is_grasped_log.append(is_grasped)
        phase_log.append(phase)
        obj_pos_log.append(obj_pos)

        if "success" in info and bool(info["success"][0].cpu().item()):
            success = True

    return {
        "z_t":          np.stack(z_t_log).astype(np.float32),
        "tactile_raw":  np.stack(tactile_raw_log).astype(np.float32),
        "is_grasped":   np.array(is_grasped_log, dtype=np.float32),
        "phase":        np.array(phase_log, dtype=np.int8),
        "obj_pos":      np.stack(obj_pos_log).astype(np.float32),
        "obj_name":     obj_name,
        "success":      bool(success),
        "initial_obj_z": float(initial_z),
        "seed":         int(seed),
    }, obj_name


def save_records(records, output_path):
    if len(records) == 0:
        return
    z_t          = np.stack([r["z_t"]         for r in records])
    tactile_raw  = np.stack([r["tactile_raw"] for r in records])
    is_grasped   = np.stack([r["is_grasped"]  for r in records])
    phase        = np.stack([r["phase"]       for r in records])
    obj_pos      = np.stack([r["obj_pos"]     for r in records])
    obj_names    = np.array([r["obj_name"]    for r in records])
    success      = np.array([r["success"]     for r in records], dtype=bool)
    initial_obj_z = np.array([r["initial_obj_z"] for r in records], dtype=np.float32)
    seeds        = np.array([r["seed"]        for r in records], dtype=np.int32)

    tmp = output_path.with_suffix(".npz.tmp")
    with open(tmp, 'wb') as f:
        np.savez_compressed(
            f,
            z_t=z_t, tactile_raw=tactile_raw, is_grasped=is_grasped,
            phase=phase, obj_pos=obj_pos, obj_names=obj_names,
            success=success, initial_obj_z=initial_obj_z, seeds=seeds,
        )
    tmp.replace(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy",          type=str, required=True)
    parser.add_argument("--encoder",         type=str, required=True)
    parser.add_argument("--num-episodes",    type=int, default=300,
                        help="Target number of held-out episodes")
    parser.add_argument("--max-seeds-to-try", type=int, default=3000,
                        help="Cap on total seeds tried (held-out + rejected)")
    parser.add_argument("--seed-start",      type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Save partial npz every N kept episodes")
    parser.add_argument("--output-dir",      type=str,
                        default=str(PROJECT_ROOT / "data" / "figure_deployment" / "4_4_heldout"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "latent_records.npz"
    meta_path = output_dir / "meta.json"

    env_module.EXPERIMENT_MODE = "visual_encoder"

    raw_env = gym.make(
        "PickAndPlaceYCBToTray-v1",
        num_envs=1,
        obs_mode="state",
        render_mode=None,
        sim_backend="physx_cuda",
        control_mode="pd_joint_delta_pos",
        reconfiguration_freq=1,
    )
    if isinstance(raw_env.action_space, gym.spaces.Dict):
        raw_env = FlattenActionSpaceWrapper(raw_env)
    environment = ManiSkillVectorEnv(raw_env, 1, ignore_terminations=True,
                                     record_metrics=True)

    reload_encoder(environment.unwrapped, args.encoder, device)

    from ppo import Agent
    policy = Agent(environment).to(device)
    policy.load_state_dict(torch.load(args.policy, map_location=device))
    policy.eval()
    print(f"Loaded policy: {args.policy}")
    print(f"Held-out targets: {len(HELDOUT_OBJECTS)} objects")
    print(f"Target episodes:  {args.num_episodes}\n")

    records = []
    seed = args.seed_start
    n_tried = 0
    rejected_objs = Counter()
    kept_objs = Counter()

    try:
        while (len(records) < args.num_episodes
               and n_tried < args.max_seeds_to_try):
            result, obj_name = rollout_episode(environment, policy, device, seed)
            n_tried += 1
            seed += 1

            if result is None:
                rejected_objs[obj_name] += 1
                if n_tried % 50 == 0:
                    hit_rate = 100 * len(records) / n_tried
                    print(f"  [progress] tried {n_tried}, kept {len(records)}, "
                          f"hit rate {hit_rate:.1f}%")
                continue

            records.append(result)
            kept_objs[obj_name] += 1

            n_grasp = int((result["phase"] >= 1).sum())
            n_transport = int((result["phase"] == 2).sum())
            print(f"  ep {len(records):4d}/{args.num_episodes} | "
                  f"seed={result['seed']:5d} | "
                  f"obj={obj_name:<28} | "
                  f"grasp={n_grasp:3d} transport={n_transport:3d} | "
                  f"success={int(result['success'])}")

            if len(records) % args.checkpoint_every == 0:
                save_records(records, output_path)
                print(f"  [checkpoint] saved {len(records)} episodes")
    finally:
        environment.close()
        save_records(records, output_path)

    if len(records) == 0:
        raise RuntimeError(
            "No held-out episodes collected. Confirm the env actually samples "
            "the held-out objects"
        )

    # ----- Coverage diagnostics -----
    print("\n" + "=" * 64)
    print("Per-object coverage:")
    print("=" * 64)
    for obj in sorted(HELDOUT_OBJECTS):
        n_kept = kept_objs.get(obj, 0)
        flag = " " if n_kept >= 5 else "!"   # flag sparse coverage
        print(f"  {flag} {obj:<32}: {n_kept:3d} ep")
    print()

    n_with_grasp = sum(1 for r in records if (r["phase"] >= 1).any())
    n_with_transport = sum(1 for r in records if (r["phase"] == 2).any())
    n_success = sum(1 for r in records if r["success"])
    total_grasp_steps = sum(int((r["phase"] >= 1).sum()) for r in records)

    print("Aggregate stats:")
    print(f"  Episodes with any grasp contact (phase>=1): "
          f"{n_with_grasp:3d}/{len(records)} ({100*n_with_grasp/len(records):.1f}%)")
    print(f"  Episodes reaching transport (phase==2):     "
          f"{n_with_transport:3d}/{len(records)} ({100*n_with_transport/len(records):.1f}%)")
    print(f"  Episodes succeeding:                        "
          f"{n_success:3d}/{len(records)} ({100*n_success/len(records):.1f}%)")
    print(f"  Total grasped timesteps for Panel C analysis: {total_grasp_steps}")
    if total_grasp_steps < 5000:
        print(f"  ! WARNING: Panel C correlations will be noisy with <5000 grasped timesteps.")
    print()

    meta = {
        "num_episodes":          len(records),
        "policy_checkpoint":     args.policy,
        "encoder_checkpoint":    args.encoder,
        "num_steps_per_episode": NUM_STEPS,
        "num_success":           int(n_success),
        "num_with_grasp":        int(n_with_grasp),
        "num_with_transport":    int(n_with_transport),
        "total_grasp_steps":     int(total_grasp_steps),
        "phase_scheme":          "3-class: 0=free, 1=grasp, 2=transport",
        "z_dim":                 16,
        "tactile_dim":           6,
        "seed_start":            args.seed_start,
        "seed_end":              seed - 1,
        "n_tried":               n_tried,
        "n_rejected":            n_tried - len(records),
        "hit_rate":              float(len(records) / n_tried),
        "heldout_objects":       sorted(HELDOUT_OBJECTS),
        "per_object_counts":     dict(kept_objs),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(records)} held-out episodes -> {output_path}")
    print(f"Meta -> {meta_path}")
    print()
    print("To render the figure on this data:")
    print("  1. Edit NPZ_PATH in plot_section_4_4_latent_analysis.py to point to")
    print(f"     {output_path}")
    print("  2. Run that plot script as usual.")
    print()
    print("OR temporarily symlink:")
    print(f"  cd {output_dir.parent / '4_4'}")
    print(f"  mv latent_records.npz latent_records.indist.npz")
    print(f"  ln -s ../4_4_heldout/latent_records.npz latent_records.npz")


if __name__ == "__main__":
    main()