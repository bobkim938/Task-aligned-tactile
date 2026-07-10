import os
import sys
import time
import torch
import numpy as np
import gymnasium as gym
from collections import defaultdict
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import environments and agent
import env.PickAndPlaceYCBToTray
from training.ppo import Agent
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Env params
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda", control_mode="pd_joint_delta_pos")
    
    # Run 10 parallel environments (10 rollouts per object)
    num_eval_envs = 10
    num_eval_steps = 100
    
    envs = gym.make("PickAndPlaceYCBToTray-v1", num_envs=num_eval_envs, reconfiguration_freq=1, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    
    envs_vec = ManiSkillVectorEnv(envs, num_eval_envs, ignore_terminations=True, record_metrics=True)
    
    # Load model
    agent = Agent(envs_vec).to(device)
    checkpoint_path = str(PROJECT_ROOT / "checkpoints" / "policy_encoder.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return
        
    print(f"Loading checkpoint {checkpoint_path}...")
    agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.eval()

    all_objects = envs.unwrapped.all_model_ids
    print(f"Found {len(all_objects)} objects to evaluate.")
    
    success_rates = {}
    slip_rates = {}
    
    for obj_id in all_objects:
        envs.unwrapped.all_model_ids = np.array([obj_id])
        
        eval_obs, _ = envs_vec.reset(seed=42, options=dict(reconfigure=True))
        
        successes = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)
        has_been_grasped = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)
        slips = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)

        with torch.no_grad():
            for _ in range(num_eval_steps):
                action = agent.get_action(eval_obs, deterministic=True)
                eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = envs_vec.step(action)
                # Check for success
                if "success" in eval_infos:
                    # eval_infos["success"] from standard environments is shape [num_envs]
                    successes |= eval_infos["success"]
                
                # Check for slip
                if "is_grasped" in eval_infos and "tcp_to_obj_dist" in eval_infos:
                    is_grasped = eval_infos["is_grasped"]
                    tcp_dist = eval_infos["tcp_to_obj_dist"]
                    
                    dist_to_goal = eval_infos.get("obj_to_final_dist", torch.ones_like(tcp_dist))
                    has_been_grasped |= is_grasped
                    current_slip = has_been_grasped & (~is_grasped) & (tcp_dist > 0.05) & (dist_to_goal > 0.15)
                    slips |= current_slip
                    
        success_rate = successes.float().mean().item() * 100
        slip_rate = slips.float().mean().item() * 100
        success_rates[obj_id] = success_rate
        slip_rates[obj_id] = slip_rate
        print(f"Object: {obj_id: <25} | Success Rate: {success_rate:.1f}%")

    # Output final summary
    print("\n==== Final Results ====")
    for obj in all_objects:
        s_rate = success_rates.get(obj, 0)
        sl_rate = slip_rates.get(obj, 0)
        print(f"{obj: <25}: Success: {s_rate:5.1f}%")
        
    avg_success_rate = np.mean(list(success_rates.values()))
    avg_slip_rate = np.mean(list(slip_rates.values()))
    print(f"{'-'*55}")
    print(f"{'Overall Rates': <25}: Success: {avg_success_rate:5.1f}%")

    envs_vec.close()

if __name__ == "__main__":
    main()
