"""
Encoder Pre-training Data Collection Script
=============================================
Collects rollout data from the oracle expert policy for tactile encoder pre-training.

Three data mixes:
  1. Clean expert (60%)    — normal rollouts
  2. Noisy expert (25%)    — action noise sigma=0.15 for failure diversity
  3. Low-friction (15%)    — mu=0.3 for slip-rich data

Records per-timestep:
  - Smoothed tactile [6], raw tactile [6]
  - is_grasped, gripper_qpos [2]
  - obj_pose [7], tcp_pose [7]
  - E_slip (continuous slip distance)
  - Delta_E_slip (change from grasp baseline)
  - Phase labels, episode metadata

Usage:
  python tactile_encoder/collect_encoder_data.py \
      --checkpoint checkpoints/policy_oracle.pt \
      --output-dir data/encoder_pretraining \
      --num-envs 256 \
      --episodes-per-mix 200 \
      --seed 42
"""

import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import h5py

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

import mani_skill.envs
import env.PickAndPlaceYCBToTray
from ppo import Agent
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def make_env(num_envs: int, seed: int):
    """Instantiate the oracle-mode environment."""
    # Force oracle mode at module level before env creation

    import env.PickAndPlaceYCBToTray as env_module
    env_module.EXPERIMENT_MODE = "oracle"

    environment = gym.make(
        "PickAndPlaceYCBToTray-v1",
        num_envs=num_envs,
        obs_mode="state",
        render_mode=None,
        sim_backend="physx_cuda",
        control_mode="pd_joint_delta_pos",
        reconfiguration_freq=0,
    )
    environment.unwrapped._base_seed = seed
    return environment


def load_policy(checkpoint_path: str, device: str, env):
    """
    Load the trained PPO policy.
    """
    agent = Agent(env).to(device)
    agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.eval()
    
    return agent


def get_action(policy, obs, device, action_noise_std=0.0):
    """
    Get action from the policy, optionally adding Gaussian noise.
    Adjust this to match your policy's inference API.
    """
    with torch.no_grad():
        obs_tensor = obs.clone().detach().to(dtype=torch.float32, device=device) if isinstance(obs, torch.Tensor) else torch.tensor(obs, dtype=torch.float32, device=device)
        action = policy.get_action(obs_tensor, deterministic=True)
        
    if action_noise_std > 0:
        action += torch.randn_like(action) * action_noise_std
        action = action.clamp(-1.0, 1.0)
        
    return action


def flatten_obs(obs_dict):
    """Flatten the nested obs dict into a single tensor for the policy."""
    # Based on ManiSkill Vector Env standard flattening for 'state' mode
    if isinstance(obs_dict, torch.Tensor):
        return obs_dict
        
    parts = []
    if "agent" in obs_dict:
        for k, v in obs_dict["agent"].items():
            if isinstance(v, torch.Tensor):
                if v.dim() == 1:
                    v = v.unsqueeze(-1)
                parts.append(v.float())
    if "extra" in obs_dict:
        for k, v in obs_dict["extra"].items():
            if isinstance(v, torch.Tensor):
                if v.dim() == 1:
                    v = v.unsqueeze(-1)
                parts.append(v.float())
    
    if len(parts) == 0:
        return obs_dict # Fallback
        
    return torch.cat(parts, dim=-1)


# =========================================================================
# Per-timestep data extraction
# =========================================================================

def extract_step_data(env_unwrapped, obs, info, action, step_idx, episode_data):
    """
    Extract and append per-timestep data from the environment.
    All privileged state is read directly from the unwrapped env.
    """
    uw = env_unwrapped
    num_envs = uw.num_envs
    device = uw.device

    # - Raw and smoothed tactile -
    raw_tactile = uw._raw_tactile.cpu().numpy()                    # [N, 6]
    smoothed_tactile = (uw._tactile_smoothed / 100.0).cpu().numpy()  # [N, 6] normalized

    # - Privileged state -
    is_grasped = info["is_grasped"].cpu().numpy().astype(np.float32)  # [N]
    obj_pos = uw.obj.pose.p.cpu().numpy()                             # [N, 3]
    obj_pose = uw.obj.pose.raw_pose.cpu().numpy()                     # [N, 7]
    tcp_pos = uw.agent.tcp.pose.p.cpu().numpy()                       # [N, 3]
    tcp_pose = uw.agent.tcp.pose.raw_pose.cpu().numpy()               # [N, 7]
    gripper_qpos = uw.agent.robot.get_qpos()[..., -2:].cpu().numpy()  # [N, 2]
    waypoint_reached = info["waypoint_reached"].cpu().numpy().astype(np.float32)

    # - E_slip: continuous slip distance -
    # ||p_obj - p_tcp|| while grasped; 0.0 when not grasped
    e_slip = np.linalg.norm(obj_pos - tcp_pos, axis=1)  # [N]
    e_slip = e_slip * is_grasped  # zero out when not grasped

    # - delta_E_slip: change from grasp baseline -
    # On first grasped frame, record baseline distance per env
    # On subsequent grasped frames, compute delta
    for i in range(num_envs):
        if is_grasped[i] > 0.5:
            if episode_data["grasp_baseline_set"][i] == 0:
                # First grasped frame: set baseline
                episode_data["e_slip_baseline"][i] = e_slip[i]
                episode_data["grasp_baseline_set"][i] = 1
                episode_data["grasp_start_step"][i] = step_idx
        else:
            # reset baseline when grasp is lost (for potential re-grasp)
            episode_data["grasp_baseline_set"][i] = 0
            episode_data["e_slip_baseline"][i] = 0.0

    delta_e_slip = np.where(
        episode_data["grasp_baseline_set"] > 0.5,
        e_slip - episode_data["e_slip_baseline"],
        0.0,
    ).astype(np.float32)

    # - Phase label -
    # 0=free, 1=grasp (holding but not lifted), 2=transport (lifted), 3=post_slip
    obj_z = obj_pos[:, 2]
    initial_z = episode_data["initial_obj_z"]
    phase = np.zeros(num_envs, dtype=np.int32)
    for i in range(num_envs):
        if is_grasped[i] > 0.5:
            if obj_z[i] < initial_z[i] + 0.03:
                phase[i] = 1  # grasp formation
            else:
                phase[i] = 2  # transport
        else:
            if episode_data["was_grasped"][i] > 0.5:
                phase[i] = 3  # post-slip / dropped
            else:
                phase[i] = 0  # free-space

    # Update was_grasped tracker
    episode_data["was_grasped"] = np.maximum(
        episode_data["was_grasped"], is_grasped
    )

    # - Store -
    step_record = dict(
        tactile_smoothed=smoothed_tactile, # [N, 6]
        tactile_raw=raw_tactile, # [N, 6]
        is_grasped=is_grasped, # [N]
        obj_pose=obj_pose, # [N, 7]
        tcp_pose=tcp_pose, # [N, 7]
        gripper_qpos=gripper_qpos, # [N, 2]
        e_slip=e_slip.astype(np.float32), # [N]
        delta_e_slip=delta_e_slip, # [N]
        phase=phase, # [N]
        waypoint_reached=waypoint_reached, # [N]
        action=action.cpu().numpy() if isinstance(action, torch.Tensor) else action,
    )
    episode_data["steps"].append(step_record)


# =========================================================================
# Retrospective label computation (run after episode completes)
# =========================================================================

def compute_retrospective_labels(episode_steps, horizon_stability=15, horizon_slip=10):
    """
    Compute labels that require future knowledge:
      - grasp_stability: will the grasp hold for the next 'horizon_stability' steps?
      - slip_imminent:   will is_grasped go from True→False within 'horizon_slip' steps?
      - force_magnitude: log(1 + ||raw_tactile||) regression target

    Args:
        episode_steps: list of step_record dicts, each containing arrays of shape [N, ...]
        horizon_stability: number of future steps for stability check
        horizon_slip: number of future steps for slip check

    Returns:
        labels dict with arrays of shape [T, N] for each label
    """
    T = len(episode_steps)
    N = episode_steps[0]["is_grasped"].shape[0]

    # Stack temporal arrays: [T, N]
    is_grasped_seq = np.stack([s["is_grasped"] for s in episode_steps], axis=0)  # [T, N]
    tactile_raw_seq = np.stack([s["tactile_raw"] for s in episode_steps], axis=0)  # [T, N, 6]

    # Grasp stability: is_grasped stays True for the next K steps?
    grasp_stability = np.zeros((T, N), dtype=np.float32)
    for t in range(T):
        end = min(t + horizon_stability, T)
        if end > t:
            future_grasped = is_grasped_seq[t:end, :]  # [K, N]
            # Stable if grasped at current step AND stays grasped for all K future steps
            all_future_grasped = future_grasped.all(axis=0).astype(np.float32)
            grasp_stability[t] = is_grasped_seq[t] * all_future_grasped

    # Slip imminent: is_grasped transitions True→False within K steps?
    slip_imminent = np.zeros((T, N), dtype=np.float32)
    for t in range(T):
        if is_grasped_seq[t].sum() == 0:
            continue  # not grasped, slip is undefined
        end = min(t + horizon_slip, T)
        if end > t:
            future_grasped = is_grasped_seq[t:end, :]  # [K, N]
            # Slip = currently grasped but will lose grasp within K steps
            loses_grasp = (~future_grasped.all(axis=0)).astype(np.float32)
            slip_imminent[t] = is_grasped_seq[t] * loses_grasp

    # - Force magnitude: log(1 + ||F||) -
    force_magnitude = np.log1p(
        np.linalg.norm(tactile_raw_seq, axis=-1)
    ).astype(np.float32)  # [T, N]

    # - delta_E_slip rate: derivative of slip distance (computed from existing data) 
    delta_e_slip_seq = np.stack([s["delta_e_slip"] for s in episode_steps], axis=0)  # [T, N]
    # Rate of change: forward difference
    slip_rate = np.zeros_like(delta_e_slip_seq)
    slip_rate[:-1] = delta_e_slip_seq[1:] - delta_e_slip_seq[:-1]

    return dict(
        grasp_stability=grasp_stability,    # [T, N]
        slip_imminent=slip_imminent,        # [T, N]
        force_magnitude=force_magnitude,    # [T, N]
        slip_rate=slip_rate,                # [T, N]
    )


# =========================================================================
# Main collection loop
# =========================================================================

def collect_data(
    checkpoint_path: str,
    output_dir: str,
    num_envs: int = 256,
    episodes_per_mix: int = 200,
    seed: int = 42,
    device: str = "cuda",
):
    os.makedirs(output_dir, exist_ok=True)

    mixes = [
        {
            "name": "clean",
            "action_noise": 0.0,
            "low_friction": False,
            "fraction": 0.60,
        },
        {
            "name": "noisy",
            "action_noise": 0.15,
            "low_friction": False,
            "fraction": 0.25,
        },
        {
            "name": "low_friction",
            "action_noise": 0.0,
            "low_friction": True,
            "fraction": 0.15,
        },
    ]

    print(f"Initializing standard environment...")
    raw_env = make_env(num_envs, seed)
    environment = ManiSkillVectorEnv(raw_env, num_envs, ignore_terminations=True)
    
    if getattr(raw_env, 'spec', None) and raw_env.spec.max_episode_steps:
        max_steps = int(raw_env.spec.max_episode_steps)
    else:
        max_steps = 100
    
    uw = environment.unwrapped
    default_friction = None
    if hasattr(uw, "_obj_friction_values"):
        default_friction = uw._obj_friction_values.copy()

    # policy loading
    policy = load_policy(checkpoint_path, device, environment)
    print(f"Loaded policy from {checkpoint_path}")

    # collect each mix
    for mix in mixes:
        mix_name = mix["name"]
        n_episodes = max(1, int(episodes_per_mix * (mix["fraction"] / 0.60)))

        print(f"\n{'='*60}")
        print(f"Collecting mix: {mix_name}")
        print(f"  Action noise: {mix['action_noise']}")
        print(f"  Low friction: {mix['low_friction']}")
        print(f"  Episodes (batches of {num_envs}): {n_episodes}")
        print(f"{'='*60}")

        # Set or restore friction values
        if default_friction is not None:
            if mix["low_friction"]:
                uw._obj_friction_values = np.full_like(default_friction, 0.3)
            else:
                uw._obj_friction_values = default_friction.copy()

        # HDF5 file for this mix
        h5_path = os.path.join(output_dir, f"{mix_name}.h5")
        h5_file = h5py.File(h5_path, "w")

        # Metadata
        h5_file.attrs["mix_name"] = mix_name
        h5_file.attrs["action_noise"] = mix["action_noise"]
        h5_file.attrs["low_friction"] = mix["low_friction"]
        h5_file.attrs["num_envs"] = num_envs
        h5_file.attrs["max_steps"] = max_steps

        total_timesteps = 0
        total_episodes = 0
        total_successes = 0
        total_slips = 0

        for batch_idx in range(n_episodes):
            # Reset with reconfiguration to get diverse objects
            obs, info = environment.reset(
                seed=seed + batch_idx,
                options=dict(reconfigure=True),
            )
            uw = environment.unwrapped

            # Per-episode tracking state
            episode_data = {
                "steps": [],
                "initial_obj_z": uw.obj.pose.p[:, 2].cpu().numpy().copy(),
                "e_slip_baseline": np.zeros(num_envs, dtype=np.float32),
                "grasp_baseline_set": np.zeros(num_envs, dtype=np.float32),
                "grasp_start_step": np.zeros(num_envs, dtype=np.int32),
                "was_grasped": np.zeros(num_envs, dtype=np.float32),
            }

            # Record object IDs and friction for metadata
            obj_ids = []
            for obj_actor in uw._objs:
                obj_ids.append(obj_actor.name.rsplit("-", 1)[0])
            friction_vals = uw.obj_friction.cpu().numpy()

            ever_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
            for step in range(max_steps):
                # Get action from policy
                obs_flat = flatten_obs(obs)
                action = get_action(
                    policy, obs_flat, device,
                    action_noise_std=mix["action_noise"],
                )

                # Step environment
                obs, reward, terminated, truncated, info = environment.step(action)

                # Accumulate success across all steps (policy may move object away from goal after initial success)
                step_success = info.get("success", torch.zeros(num_envs, dtype=torch.bool, device=device))
                ever_success |= step_success

                # Extract and store data
                extract_step_data(uw, obs, info, action, step, episode_data)

            # Compute retrospective labels
            retro_labels = compute_retrospective_labels(episode_data["steps"])

            # count events 
            is_grasped_final = np.stack(
                [s["is_grasped"] for s in episode_data["steps"]], axis=0
            )
            success_flags = ever_success.cpu().numpy()

            # Count slip events: is_grasped goes 1 to 0
            slip_count = 0
            for i in range(num_envs):
                for t in range(1, len(episode_data["steps"])):
                    if is_grasped_final[t-1, i] > 0.5 and is_grasped_final[t, i] < 0.5:
                        slip_count += 1

            total_successes += success_flags.sum()
            total_slips += slip_count
            total_episodes += num_envs

            # save to HDF5 
            batch_grp = h5_file.create_group(f"batch_{batch_idx:04d}")
            batch_grp.attrs["num_envs"] = num_envs
            batch_grp.attrs["max_steps"] = max_steps
            batch_grp.attrs["success_rate"] = float(success_flags.mean())
            batch_grp.attrs["slip_count"] = slip_count

            # Object metadata (variable-length strings)
            dt = h5py.special_dtype(vlen=str)
            batch_grp.create_dataset("object_ids", data=obj_ids, dtype=dt)
            batch_grp.create_dataset("friction", data=friction_vals)
            batch_grp.create_dataset("initial_obj_z", data=episode_data["initial_obj_z"])

            # Per-timestep data: stack to [T, N, ...] and save
            T = len(episode_data["steps"])
            fields_to_save = [
                "tactile_smoothed", "tactile_raw", "is_grasped",
                "obj_pose", "tcp_pose", "gripper_qpos",
                "e_slip", "delta_e_slip", "phase", "waypoint_reached",
            ]
            for field in fields_to_save:
                arr = np.stack(
                    [s[field] for s in episode_data["steps"]], axis=0
                )  
                batch_grp.create_dataset(
                    field, data=arr, compression="gzip", compression_opts=4
                )

            # Retrospective labels
            for label_name, label_arr in retro_labels.items():
                batch_grp.create_dataset(
                    f"label_{label_name}", data=label_arr,
                    compression="gzip", compression_opts=4,
                )

            # Success per env
            batch_grp.create_dataset("success", data=success_flags.astype(np.float32))

            total_timesteps += T * num_envs

            print(
                f"  Batch {batch_idx+1}/{n_episodes}: "
                f"success={success_flags.mean():.2%}, "
                f"slips={slip_count}, "
                f"timesteps={T * num_envs}"
            )

        h5_file.attrs["total_timesteps"] = total_timesteps
        h5_file.attrs["total_episodes"] = total_episodes
        h5_file.attrs["total_successes"] = int(total_successes)
        h5_file.attrs["total_slips"] = total_slips
        h5_file.attrs["success_rate"] = float(total_successes / max(total_episodes, 1))
        h5_file.close()

        print(f"\n  Mix '{mix_name}' complete:")
        print(f"    Total timesteps: {total_timesteps:,}")
        print(f"    Total episodes:  {total_episodes}")
        print(f"    Success rate:    {total_successes/max(total_episodes,1):.2%}")
        print(f"    Slip events:     {total_slips}")
        print(f"    Saved to:        {h5_path}")

    environment.close()

    # final summary
    print(f"\n{'='*60}")
    print("DATA COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Files:")
    for mix in mixes:
        h5_path = os.path.join(output_dir, f"{mix['name']}.h5")
        size_mb = os.path.getsize(h5_path) / 1e6
        print(f"  {mix['name']}.h5 — {size_mb:.1f} MB")


# =========================================================================
# Entry point
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect oracle expert rollouts for tactile encoder pre-training"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the trained oracle PPO checkpoint",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/encoder_pretraining",
        help="Directory to save collected data",
    )
    parser.add_argument(
        "--num-envs", type=int, default=256,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--episodes-per-mix", type=int, default=200,
        help="Base number of episode batches for the majority (clean) mix",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for policy inference",
    )
    args = parser.parse_args()

    collect_data(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_envs=args.num_envs,
        episodes_per_mix=args.episodes_per_mix,
        seed=args.seed,
        device=args.device,
    )