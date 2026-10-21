"""
collect_.py
=============================
Run once per encoder version, with identical seeds so V1 and V2 episodes
are paired:

  # V1 (no-hinge encoder):
  python analysis/collect_.py \
      --policy checkpoints/policy_encoder_no_hinge.pt \
      --encoder checkpoints/tactile_encoder_no_hinge/best_encoder.pt \
      --tag v1 --num-episodes 1000 --seed-start 0

  # V2 (hinge encoder):
  python analysis/collect_.py \
      --policy checkpoints/policy_encoder_seed4796.pt \
      --encoder checkpoints/tactile_encoder_hinge/best_encoder.pt \
      --tag v2 --num-episodes 1000 --seed-start 0

Output: data/{tag}/deployment_records.npz
        data/{tag}/meta.json

Stored fields:
  z_t              [N, T, 16]  latent the policy consumed
  tactile_raw      [N, T, 6]   raw 6D contact force (physical units, unsmoothed)
  tactile_smoothed [N, T, 6]   smoothed force (env feeds encoder this / 100.0)
  is_grasped       [N, T]      privileged grasp flag
  phase            [N, T]      3-class label (0=free, 1=grasp, 2=transport)
  obj_pos          [N, T, 3]   object xyz (world frame)
  tcp_pos          [N, T, 3]   tool center point xyz (world frame); obj_pos -
                               tcp_pos gives object position relative to the
                               gripper -- a drop in relative height is SLIP,
                               not commanded lowering
  obj_names        [N]         object name per episode
  success          [N]         episode success flag
  initial_obj_z    [N]         initial object height
  seeds            [N]         seed per episode
"""

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

import env.PickAndPlaceYCBToTray as env_module
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from tactile_encoder.tactile_encoder import TactileEncoder

NUM_STEPS = 100


# ---------------------------------------------------------------------------
# 3-phase scheme (post-slip merged into free)
# ---------------------------------------------------------------------------
def determine_phase(prev_phase, is_grasped, obj_z, initial_z):
    """0=free, 1=grasp, 2=transport. Post-slip is folded into free."""
    if is_grasped < 0.5:
        return 0
    if prev_phase != 2:
        return 2 if obj_z > initial_z + 0.04 else 1
    return 1 if obj_z < initial_z + 0.02 else 2


# ---------------------------------------------------------------------------
# Hot-swap encoder (no need to edit env between runs)
# ---------------------------------------------------------------------------
def reload_encoder(env_unwrapped, encoder_checkpoint, device):
    enc = TactileEncoder(in_channels=6, window_size=16, hidden_dim=64, z_dim=16)
    state = torch.load(encoder_checkpoint, map_location="cpu")
    enc.load_state_dict(state)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    env_unwrapped._tactile_encoder = enc.to(device)
    print(f"[hot-swap] Loaded encoder from {encoder_checkpoint}")


# ---------------------------------------------------------------------------
# One episode
# ---------------------------------------------------------------------------
def rollout_episode(environment, policy, device, seed):
    obs, _ = environment.reset(seed=seed, options=dict(reconfigure=True))
    uw = environment.unwrapped

    initial_z = float(uw.obj.pose.p[0, 2].cpu().item())
    obj_name = uw._objs[0].name.rsplit("-", 1)[0] if uw._objs else "unknown"

    z_t_log = []
    tactile_raw_log = []
    tactile_smoothed_log = []
    is_grasped_log = []
    phase_log = []
    obj_pos_log = []
    tcp_pos_log = []

    prev_phase = 0
    success = False

    for _ in range(NUM_STEPS):
        with torch.no_grad():
            action = policy.get_action(obs.float().to(device), deterministic=True)
        obs, _, _, _, info = environment.step(action)

        # z_t the policy actually consumed at this step
        prev_z = getattr(uw, "_prev_z_t", None)
        if prev_z is None:
            raise RuntimeError(
                "Env did not expose _prev_z_t. Confirm EXPERIMENT_MODE='visual_encoder' "
                "and the env's _get_obs_extra writes _prev_z_t."
            )
        z_t_arr = prev_z[0].detach().cpu().numpy()             # [16]

        raw_tact = uw._raw_tactile[0].cpu().numpy()            # [6]

        # smoothed tactile
        smoothed_attr = getattr(uw, "_tactile_smoothed", None)
        if smoothed_attr is not None:
            smooth_tact = smoothed_attr[0].cpu().numpy()       # [6]
        else:
            smooth_tact = raw_tact

        is_grasped = float(info["is_grasped"][0].cpu().item())
        obj_pos = uw.obj.pose.p[0].cpu().numpy()               # [3]
        obj_z = float(obj_pos[2])

        # TCP (tool center point) position 
        tcp_pos = None
        try:
            tcp_pos = uw.agent.tcp.pose.p[0].cpu().numpy()     # [3]
        except Exception:
            try:
                tcp_pos = uw.agent.tcp_pose.p[0].cpu().numpy() # [3]
            except Exception:
                ptcp = getattr(uw, "_p_tcp", None)
                if ptcp is not None:
                    tcp_pos = ptcp[0].cpu().numpy()
        if tcp_pos is None:
            tcp_pos = np.full(3, np.nan, dtype=np.float32)

        phase = determine_phase(prev_phase, is_grasped, obj_z, initial_z)
        prev_phase = phase

        z_t_log.append(z_t_arr)
        tactile_raw_log.append(raw_tact)
        tactile_smoothed_log.append(smooth_tact)
        is_grasped_log.append(is_grasped)
        phase_log.append(phase)
        obj_pos_log.append(obj_pos)
        tcp_pos_log.append(tcp_pos)

        if "success" in info and bool(info["success"][0].cpu().item()):
            success = True

    return {
        "z_t":              np.stack(z_t_log).astype(np.float32),
        "tactile_raw":      np.stack(tactile_raw_log).astype(np.float32),
        "tactile_smoothed": np.stack(tactile_smoothed_log).astype(np.float32),
        "is_grasped":       np.array(is_grasped_log, dtype=np.float32),
        "phase":            np.array(phase_log, dtype=np.int8),
        "obj_pos":          np.stack(obj_pos_log).astype(np.float32),
        "tcp_pos":          np.stack(tcp_pos_log).astype(np.float32),
        "obj_name":         obj_name,
        "success":          bool(success),
        "initial_obj_z":    float(initial_z),
        "seed":             int(seed),
    }


# ---------------------------------------------------------------------------
# Save (atomic, single compressed npz)
# ---------------------------------------------------------------------------
def save_records(records, output_path):
    if len(records) == 0:
        return
    z_t              = np.stack([r["z_t"]              for r in records])
    tactile_raw      = np.stack([r["tactile_raw"]      for r in records])
    tactile_smoothed = np.stack([r["tactile_smoothed"] for r in records])
    is_grasped       = np.stack([r["is_grasped"]       for r in records])
    phase            = np.stack([r["phase"]            for r in records])
    obj_pos          = np.stack([r["obj_pos"]          for r in records])
    tcp_pos          = np.stack([r["tcp_pos"]          for r in records])
    obj_names        = np.array([r["obj_name"]         for r in records])
    success          = np.array([r["success"]          for r in records], dtype=bool)
    initial_obj_z    = np.array([r["initial_obj_z"]    for r in records], dtype=np.float32)
    seeds            = np.array([r["seed"]             for r in records], dtype=np.int32)

    tmp = output_path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            z_t=z_t,
            tactile_raw=tactile_raw,
            tactile_smoothed=tactile_smoothed,
            is_grasped=is_grasped,
            phase=phase,
            obj_pos=obj_pos,
            tcp_pos=tcp_pos,
            obj_names=obj_names,
            success=success,
            initial_obj_z=initial_obj_z,
            seeds=seeds,
        )
    tmp.replace(output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True,
                        help="PPO policy checkpoint")
    parser.add_argument("--encoder", type=str, required=True,
                        help="Tactile encoder checkpoint")
    parser.add_argument("--tag", type=str, required=True,
                        help="Output tag, e.g. 'v1' or 'v2'")
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--max-seeds-to-try", type=int, default=3000)
    parser.add_argument("--seed-start", type=int, default=0,
                        help="Use the SAME value for V1 and V2 to pair episodes")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = (Path(args.output_dir) if args.output_dir
                  else PROJECT_ROOT / "data" / args.tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "deployment_records.npz"
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
    print(f"Tag: {args.tag}  |  target episodes: {args.num_episodes}")
    print(f"Output: {output_path}\n")

    records = []
    seed = args.seed_start
    try:
        while (len(records) < args.num_episodes
               and seed < args.seed_start + args.max_seeds_to_try):
            rec = rollout_episode(environment, policy, device, seed)
            records.append(rec)
            n_free = int((rec["phase"] == 0).sum())
            n_grasp = int((rec["phase"] == 1).sum())
            n_transport = int((rec["phase"] == 2).sum())
            print(f"  ep {len(records):4d}/{args.num_episodes} | seed={seed:5d} | "
                  f"obj={rec['obj_name']:<28} | "
                  f"free={n_free:3d} grasp={n_grasp:3d} transport={n_transport:3d} | "
                  f"success={int(rec['success'])}")
            seed += 1

            if len(records) % args.checkpoint_every == 0:
                save_records(records, output_path)
                print(f"  [checkpoint] saved {len(records)} episodes")
    finally:
        environment.close()
        save_records(records, output_path)

    if len(records) == 0:
        raise RuntimeError("No episodes collected.")

    n_success = int(sum(1 for r in records if r["success"]))
    meta = {
        "tag":                   args.tag,
        "num_episodes":          len(records),
        "policy_checkpoint":     args.policy,
        "encoder_checkpoint":    args.encoder,
        "num_steps_per_episode": NUM_STEPS,
        "num_success":           n_success,
        "num_failure":           len(records) - n_success,
        "phase_scheme":          "3-class: 0=free (post-slip merged), 1=grasp, 2=transport",
        "z_dim":                 16,
        "tactile_dim":           6,
        "seed_start":            args.seed_start,
        "seed_end":              seed - 1,
        "stored_fields": [
            "z_t", "tactile_raw", "tactile_smoothed", "is_grasped",
            "phase", "obj_pos", "tcp_pos", "obj_names", "success",
            "initial_obj_z", "seeds",
        ],
        "tactile_note": ("encoder consumes tactile_smoothed/100.0; "
                         "stored tactile_smoothed is in physical units"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {len(records)} episodes -> {output_path}")
    print(f"Meta -> {meta_path}")
    print(f"Success: {n_success}/{len(records)} "
          f"({100*n_success/len(records):.1f}%)")


if __name__ == "__main__":
    main()