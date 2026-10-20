"""
PickAndPlaceYCBToTray-v1  —  Visual-proxy observation model for tactile ablation experiments
=============================================================================================

Ablation rows (toggle via the EXPERIMENT_MODE variable below):
    "oracle"                — true obj_pose + is_grasped (upper bound)
    "visual_only"           — smooth occlusion + gripper_closed, no tactile
    "visual_raw_tact"       — smooth occlusion + gripper_closed + 6-D smoothed force
    "visual_raw_tact_stack" — smooth occlusion + gripper_closed + 16-frame stack of 6-D smoothed force
                              (96-dim; the encoder's input window, without the encoder)
    "visual_encoder"        — smooth occlusion + gripper_closed + learned z_t (16-dim)

All rows share the same reward function, success criterion, and evaluation logic.
Only the observation dict changes.
"""

from math import perm
import os
from typing import Any, Union

import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from .panda_v2 import PandaV2

WARNED_ONCE = False

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================
EXPERIMENT_MODE = "visual_raw_tact_stack"  # "oracle" | "visual_only" | "visual_raw_tact" | "visual_raw_tact_stack" | "visual_encoder"

# Path to the pre-trained tactile encoder checkpoint (only used when mode = "visual_encoder")
ENCODER_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "tactile_encoder_hinge", "best_encoder.pt"
)

# Encoder hyperparameters
ENCODER_WINDOW_SIZE = 16
ENCODER_Z_DIM = 16
ENCODER_HIDDEN_DIM = 64

# Modes that keep the ENCODER_WINDOW_SIZE-frame force history buffer
TEMPORAL_MODES = ("visual_encoder", "visual_raw_tact_stack")


@register_env("PickAndPlaceYCBToTray-v1", max_episode_steps=100, asset_download_ids=["ycb"])
class PickAndPlaceYCBToTrayEnv(BaseEnv):
    """
    Pick up a random YCB object and transport it through a waypoint to a tray.
    """

    SUPPORTED_ROBOTS = ["panda_v2"]
    agent: Union[PandaV2]
    waypoint_thresh = 0.08
    _tactile_dropout_rate = 0.0

    def __init__(
        self,
        *args,
        robot_uids="panda_v2",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.model_id = None
        self.all_model_ids = np.array(
            [
                k
                for k in load_json(
                    ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"
                ).keys()
                if k
                not in [
                    "003_cracker_box",
                    "019_pitcher_base",
                    "021_bleach_cleanser",
                    "024_bowl",
                    "035_power_drill",
                    "036_wood_block",
                    "072-a_toy_airplane",
                    "072-b_toy_airplane",
                    "022_windex_bottle",
                    "028_skillet_lid",
                    "029_plate",
                    "059_chain",
                    "006_mustard_bottle",
                    "004_sugar_box",
                    "052_extra_large_clamp",
                    "026_sponge",
                ]
            ]
        )
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0

        # ── Load tactile encoder if needed (before super().__init__) ──
        self._tactile_encoder = None
        if EXPERIMENT_MODE == "visual_encoder":
            self._load_tactile_encoder()

        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    def _load_tactile_encoder(self):
        """Load and freeze the pre-trained tactile encoder."""
        import sys
        # Add project root to path so we can import tactile_encoder module
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from tactile_encoder.tactile_encoder import TactileEncoder

        self._tactile_encoder = TactileEncoder(
            in_channels=6,
            window_size=ENCODER_WINDOW_SIZE,
            hidden_dim=ENCODER_HIDDEN_DIM,
            z_dim=ENCODER_Z_DIM,
        )

        # Load weights
        checkpoint_path = ENCODER_CHECKPOINT
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Tactile encoder checkpoint not found at {checkpoint_path}. "
                f"Train the encoder first with train_tactile_encoder.py"
            )
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self._tactile_encoder.load_state_dict(state_dict)
        self._tactile_encoder.eval()

        # Freeze all parameters
        for p in self._tactile_encoder.parameters():
            p.requires_grad = False

        print(f"[visual_encoder] Loaded tactile encoder from {checkpoint_path}")
        print(f"[visual_encoder] Encoder params: {sum(p.numel() for p in self._tactile_encoder.parameters()):,} (frozen)")

    # ------------------------------------------------------------------
    # Sim / sensor / camera configuration
    # ------------------------------------------------------------------
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**21, max_rigid_patch_count=2**21
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    # ------------------------------------------------------------------
    # Scene building
    # ------------------------------------------------------------------
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.55, 0, 0]))

    def _load_scene(self, options: dict):
        global WARNED_ONCE
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        model_ids = self._batched_episode_rng.choice(self.all_model_ids, replace=True)
        if (
            self.num_envs > 1
            and self.num_envs < len(self.all_model_ids)
            and self.reconfiguration_freq <= 0
            and not WARNED_ONCE
        ):
            WARNED_ONCE = True
            print(
                "There are less parallel environments than total available models to "
                "sample.  Not all models will be used during interaction even after "
                "resets unless you call env.reset(options=dict(reconfigure=True)) or "
                "set reconfiguration_freq to be >= 1."
            )

        self._obj_friction_values = self._batched_episode_rng.uniform(0.5, 1.0)

        self._objs: list[Actor] = []
        self.obj_heights = []
        for i, model_id in enumerate(model_ids):
            builder = actors.get_actor_builder(
                self.scene,
                id=f"ycb:{model_id}",
            )
            friction = float(self._obj_friction_values[i])
            mat = sapien.physx.PhysxMaterial(friction, friction, 0.0)
            for record in builder.collision_records:
                record.material = mat
            builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            builder.set_scene_idxs([i])
            self._objs.append(builder.build(name=f"{model_id}-{i}"))
            self.remove_from_state_dict_registry(self._objs[-1])
        self.obj = Actor.merge(self._objs, name="ycb_object")
        self.add_to_state_dict_registry(self.obj)

        self.waypoint_site = actors.build_sphere(
            self.scene,
            radius=self.waypoint_thresh,
            color=[0, 1, 0, 1],
            name="waypoint_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.waypoint_site)

        self.tray_half_size = [0.08, 0.08, 0.001]
        builder_left = self.scene.create_actor_builder()
        builder_left.initial_pose = sapien.Pose()
        builder_left.add_box_visual(
            half_size=self.tray_half_size,
            material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 1]),
        )
        self.left_tray = builder_left.build_kinematic(name="left_tray")

        builder_right = self.scene.create_actor_builder()
        builder_right.initial_pose = sapien.Pose()
        builder_right.add_box_visual(
            half_size=self.tray_half_size,
            material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 1]),
        )
        self.right_tray = builder_right.build_kinematic(name="right_tray")

    # ------------------------------------------------------------------
    # Post-reconfiguration setup
    # ------------------------------------------------------------------
    def _after_reconfigure(self, options: dict):
        self.object_zs = []
        for obj in self._objs:
            collision_mesh = obj.get_first_collision_mesh()
            self.object_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.object_zs = common.to_tensor(self.object_zs, device=self.device)

        FINGER_REACH = 0.04
        grasp_offsets = []
        for obj in self._objs:
            bb = obj.get_first_collision_mesh().bounding_box.bounds
            xy_radius = min(float(bb[1, 0]), float(bb[1, 1]))
            is_cup_or_mug = any(s in obj.name for s in ("cups", "mug"))
            grasp_offsets.append(
                max(0.0, xy_radius - FINGER_REACH) if is_cup_or_mug else 0.0
            )
        self.obj_grasp_offset = common.to_tensor(grasp_offsets, device=self.device)

        # Global episode state
        self.final_goal_xyz = torch.zeros((self.num_envs, 3), device=self.device)
        self.waypoint_reached = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Tactile EMA buffer and raw cache
        self._tactile_smoothed = torch.zeros((self.num_envs, 6), device=self.device)
        self._raw_tactile = torch.zeros((self.num_envs, 6), device=self.device)

        # Sticky vision-occlusion flag
        self._vision_occluded = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Friction record
        self.obj_friction = common.to_tensor(
            self._obj_friction_values, device=self.device
        )

        # force history buffer (visual_encoder / visual_raw_tact_stack modes)
        if EXPERIMENT_MODE in TEMPORAL_MODES:
            self._force_history = torch.zeros(
                (self.num_envs, ENCODER_WINDOW_SIZE, 6), device=self.device
            )
            # Move encoder to the correct device
            if self._tactile_encoder is not None:
                self._tactile_encoder = self._tactile_encoder.to(self.device)

    # ------------------------------------------------------------------
    # Episode initialization
    # ------------------------------------------------------------------
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # --- Tray placement ---
            go_left = torch.rand(b) < 0.5

            left_xyz = torch.zeros((b, 3))
            left_xyz[:, 0] = torch.rand(b) * 0.2
            left_xyz[:, 1] = -0.35
            left_xyz[:, 2] = 0.001
            self.left_tray.set_pose(Pose.create_from_pq(left_xyz))

            right_xyz = torch.zeros((b, 3))
            right_xyz[:, 0] = torch.rand(b) * 0.2
            right_xyz[:, 1] = 0.35
            right_xyz[:, 2] = 0.001
            self.right_tray.set_pose(Pose.create_from_pq(right_xyz))

            self.final_goal_xyz[env_idx, 0] = torch.where(
                go_left, left_xyz[:, 0], right_xyz[:, 0]
            )
            self.final_goal_xyz[env_idx, 1] = torch.where(
                go_left, left_xyz[:, 1], right_xyz[:, 1]
            )
            self.final_goal_xyz[env_idx, 2] = 0.05

            # Reset per-episode state
            self.waypoint_reached[env_idx] = False
            self._tactile_smoothed[env_idx] = 0.0
            self._vision_occluded[env_idx] = False

            # Reset force history buffer for resetting envs
            if EXPERIMENT_MODE in TEMPORAL_MODES:
                self._force_history[env_idx] = 0.0

            # --- Object placement ---
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 2] = self.object_zs[env_idx]
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            # --- Waypoint placement ---
            waypoint_xyz = torch.zeros((b, 3))
            base_waypoint_x = torch.where(
                go_left, left_xyz[:, 0], right_xyz[:, 0]
            )
            base_waypoint_y = torch.where(
                go_left, left_xyz[:, 1] + 0.15, right_xyz[:, 1] - 0.15
            )
            x_noise = (torch.rand(b, device=self.device) * 0.06) - 0.03
            y_noise = (torch.rand(b, device=self.device) * 0.06) - 0.03
            z_variance = torch.rand(b, device=self.device) * 0.05

            waypoint_xyz[:, 0] = base_waypoint_x + x_noise
            waypoint_xyz[:, 1] = base_waypoint_y + y_noise
            waypoint_xyz[:, 2] = xyz[:, 2] + 0.20 + z_variance
            self.waypoint_site.set_pose(Pose.create_from_pq(waypoint_xyz))

            # --- Robot initialization ---
            if self.robot_uids in ["panda_v2"]:
                qpos = np.array(
                    [0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04]
                )
                qpos[:-2] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(qpos) - 2
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.55, 0, 0]))
            else:
                raise NotImplementedError(self.robot_uids)

    # ------------------------------------------------------------------
    # Evaluation (always uses TRUE object pose)
    # ------------------------------------------------------------------
    def evaluate(self):
        obj_to_waypoint_pos = self.waypoint_site.pose.p - self.obj.pose.p
        currently_at_waypoint = (
            torch.linalg.norm(obj_to_waypoint_pos, axis=1) <= self.waypoint_thresh
        )
        self.waypoint_reached = self.waypoint_reached | currently_at_waypoint

        obj_to_final_pos = self.final_goal_xyz - self.obj.pose.p
        is_obj_placed = torch.linalg.norm(obj_to_final_pos, axis=1) <= 0.04

        is_grasped = self.agent.is_grasping(self.obj)
        is_robot_static = self.agent.is_static(0.2)

        success = (
            is_obj_placed & is_robot_static & is_grasped & self.waypoint_reached
        )

        return dict(
            is_grasped=is_grasped,
            obj_to_waypoint_pos=obj_to_waypoint_pos,
            obj_to_final_pos=obj_to_final_pos,
            waypoint_reached=self.waypoint_reached,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            success=success,
        )

    # ------------------------------------------------------------------
    # Helper: TCP-to-object surface vector
    # ------------------------------------------------------------------
    def _tcp_to_obj_surface_vec(self):
        vec = self.obj.pose.p - self.agent.tcp.pose.p
        dist = torch.linalg.norm(vec, axis=1, keepdim=True)
        unit = vec / (dist + 1e-6)
        offset = self.obj_grasp_offset.unsqueeze(-1)
        return vec - unit * offset

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs_extra(self, info: dict):
        waypoint_flag = info["waypoint_reached"].float().unsqueeze(-1)

        # ===============================================================
        # TACTILE PREPROCESSING — always runs
        # ===============================================================
        raw_tactile = self.agent.tactile_sensor.view(self.num_envs, 6)
        tactile_clamped = torch.clamp(raw_tactile, -1000.0, 1000.0)
        self._tactile_smoothed = (
            0.3 * tactile_clamped + 0.7 * self._tactile_smoothed
        )
        tactile_normalized = self._tactile_smoothed / 100.0
        self._raw_tactile = raw_tactile

        # ===============================================================
        # SMOOTH VISION OCCLUSION MODEL
        # ===============================================================
        gripper_qpos = self.agent.robot.get_qpos()[..., -2:]
        gripper_openness = gripper_qpos.mean(dim=-1)

        occlusion_alpha = 1.0 - torch.clamp(
            (gripper_openness - 0.005) / (0.015 - 0.005), 0.0, 1.0
        )

        self._vision_occluded = self._vision_occluded | (occlusion_alpha >= 0.99)
        fingers_fully_open = gripper_openness > 0.035
        self._vision_occluded = self._vision_occluded & (~fingers_fully_open)

        effective_alpha = torch.where(
            self._vision_occluded,
            torch.ones_like(occlusion_alpha),
            occlusion_alpha,
        ).unsqueeze(-1)

        visual_obj_pos = (
            (1.0 - effective_alpha) * self.obj.pose.p
            + effective_alpha * self.agent.tcp.pose.p
        )

        gripper_closed = (gripper_openness < 0.02).float().unsqueeze(-1)

        # ===============================================================
        # BUILD OBSERVATION DICT
        # ===============================================================
        mode = EXPERIMENT_MODE

        if mode == "oracle":
            obs = dict(
                tcp_pose=self.agent.tcp.pose.raw_pose,
                waypoint_pos=self.waypoint_site.pose.p,
                final_goal_pos=self.final_goal_xyz,
                waypoint_reached=waypoint_flag,
                is_grasped=info["is_grasped"].float().unsqueeze(-1),
                tactile=tactile_normalized,
            )
            if "state" in self.obs_mode or "rgbd" in self.obs_mode:
                obs.update(
                    tcp_to_waypoint_pos=self.waypoint_site.pose.p
                    - self.agent.tcp.pose.p,
                    tcp_to_final_pos=self.final_goal_xyz - self.agent.tcp.pose.p,
                    obj_pose=self.obj.pose.raw_pose,
                    tcp_to_obj_pos=self._tcp_to_obj_surface_vec(),
                    obj_to_waypoint_pos=self.waypoint_site.pose.p - self.obj.pose.p,
                    obj_to_final_pos=self.final_goal_xyz - self.obj.pose.p,
                )

        else:
            # ── Visual-proxy modes ──
            obs = dict(
                tcp_pose=self.agent.tcp.pose.raw_pose,
                waypoint_pos=self.waypoint_site.pose.p,
                final_goal_pos=self.final_goal_xyz,
                waypoint_reached=waypoint_flag,
                gripper_closed=gripper_closed,
            )

            # Shift force history buffer and insert new reading
            if mode in TEMPORAL_MODES:
                self._force_history[:, :-1] = self._force_history[:, 1:].clone()
                self._force_history[:, -1] = tactile_normalized

            if mode == "visual_raw_tact":
                obs["tactile"] = tactile_normalized

            elif mode == "visual_raw_tact_stack":
                # Flattened oldest -> newest
                obs["tactile_stack"] = self._force_history.reshape(self.num_envs, -1).clone()  # [b, 96]

            elif mode == "visual_encoder":
                # ── Encoder forward pass (frozen, no gradients) ──
                with torch.no_grad():
                    _h_t, z_t = self._tactile_encoder(self._force_history)


                    # ── Tactile dropout during training ──
                    # Prevents policy from over-relying on z_t at the expense of visual obs.
                    # Rate 0.4 = 40% of steps z_t is replaced with zeros.
                    # Disabled during evaluation by setting _tactile_dropout_rate = 0.0
                    if self._tactile_dropout_rate > 0:
                        dropout_mask = (
                            torch.rand(z_t.shape[0], 1, device=z_t.device)
                            > self._tactile_dropout_rate
                        ).float()
                        z_t = z_t * dropout_mask


                obs["tactile_repr"] = z_t  # [b, 16]
                
                ###############################################################
                if not hasattr(self, '_prev_z_t'):
                    self._prev_z_t = torch.zeros_like(z_t)

                z_t_delta = torch.norm(z_t - self._prev_z_t, dim=-1)  # [b]
                self._z_t_delta_log = z_t_delta.mean().item()  # scalar for logging
                self._prev_z_t = z_t.clone()
                ###############################################################
                # # test A: Shuffle z_t across env
                # perm = torch.randperm(z_t.shape[0], device=z_t.device)
                # obs["tactile_repr"] = z_t[perm]

                # # test B: Replace z_t with batch mean
                # obs["tactile_repr"] = z_t.mean(dim=0, keepdim=True).expand_as(z_t)

                # # test C: Add noise to z_t
                # obs["tactile_repr"] = z_t + torch.randn_like(z_t) * 2.0 
                ###############################################################

            if "state" in self.obs_mode or "rgbd" in self.obs_mode:
                obs.update(
                    tcp_to_waypoint_pos=self.waypoint_site.pose.p
                    - self.agent.tcp.pose.p,
                    tcp_to_final_pos=self.final_goal_xyz - self.agent.tcp.pose.p,
                    obj_pose_visual=visual_obj_pos,
                    tcp_to_obj_pos=visual_obj_pos - self.agent.tcp.pose.p,
                    obj_to_waypoint_pos=self.waypoint_site.pose.p - visual_obj_pos,
                    obj_to_final_pos=self.final_goal_xyz - visual_obj_pos,
                )

        return obs

    # ------------------------------------------------------------------
    # Dense reward (always uses TRUE object pose)
    # ------------------------------------------------------------------
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.obj.pose.p - self.agent.tcp.pose.p, axis=1
        )
        tcp_to_obj_dist = torch.clamp(
            tcp_to_obj_dist - self.obj_grasp_offset, min=0.0
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_waypoint_dist = torch.linalg.norm(
            self.waypoint_site.pose.p - self.obj.pose.p, axis=1
        )
        waypoint_reward = 1 - torch.tanh(5 * obj_to_waypoint_dist)
        waypoint_reward = torch.where(
            info["waypoint_reached"],
            torch.ones_like(waypoint_reward),
            waypoint_reward,
        )
        reward += waypoint_reward * is_grasped

        obj_to_final_dist = torch.linalg.norm(
            self.final_goal_xyz - self.obj.pose.p, axis=1
        )
        final_place_reward = 1 - torch.tanh(5 * obj_to_final_dist)
        final_place_reward = final_place_reward * info["waypoint_reached"].float()
        reward += final_place_reward * is_grasped

        raw_tactile = self._raw_tactile
        F_grip = torch.abs(raw_tactile[:, 2]) + torch.abs(raw_tactile[:, 5])
        F_threshold = 40.0
        F_scale = 20.0
        grip_force_penalty = -0.15 * torch.tanh(
            torch.clamp(F_grip - F_threshold, min=0.0) / F_scale
        )
        reward += grip_force_penalty * is_grasped

        reward += info["is_obj_placed"].float() * is_grasped

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        reward += static_reward * info["is_obj_placed"].float() * is_grasped

        reward[info["success"]] = 7.0

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 7.0
