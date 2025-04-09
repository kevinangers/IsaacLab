# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the vials task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def vial_placed_correctly(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    vial_cfg: SceneEntityCfg = SceneEntityCfg("vial"),
    goal_pos: torch.tensor = torch.tensor([0.5, 0.0, 0.0]),
    xy_threshold: float = 0.02,
    max_tilt_angle: float = 0.1,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
    atol=0.0001,
    rtol=0.0001,
):
    """Check if the vial is placed at the goal location and is upright.
    
    Args:
        env: The environment instance
        robot_cfg: The robot configuration
        vial_cfg: The vial configuration
        goal_pos: The goal position for the vial in world coordinates
        xy_threshold: Maximum allowable distance from the goal position in x-y plane
        max_tilt_angle: Maximum allowable tilt angle for the vial in radians
        gripper_open_val: The value of the gripper joint when open
        atol: Absolute tolerance for gripper position check
        rtol: Relative tolerance for gripper position check
    Returns:
        A boolean tensor indicating if vial is placed correctly for each environment
    """

    robot: Articulation = env.scene[robot_cfg.name]
    vial: RigidObject = env.scene[vial_cfg.name]

    # Get vial position and orientation
    vial_pos = vial.data.root_pos_w
    vial_quat = vial.data.root_quat_w

    # Check position - using only xy distance from goal
    pos_diff = vial_pos - goal_pos.to(env.device)
    xy_dist = torch.norm(pos_diff[:, :2], dim=1)
    
    # Convert quaternion to euler angles to check orientation
    vial_euler = euler_xyz_from_quat(vial_quat)
    
    # Check if vial is upright (roll and pitch should be close to 0)
    roll_pitch_ok = torch.logical_and(
        torch.abs(vial_euler[:, 0]) < max_tilt_angle,  # roll
        torch.abs(vial_euler[:, 1]) < max_tilt_angle   # pitch
    )

    # Position and orientation check
    correctly_placed = torch.logical_and(xy_dist < xy_threshold, roll_pitch_ok)

    # Check if gripper is open (not holding the vial)
    correctly_placed = torch.logical_and(
        torch.isclose(robot.data.joint_pos[:, -1], gripper_open_val.to(env.device), atol=atol, rtol=rtol),
        correctly_placed
    )
    correctly_placed = torch.logical_and(
        torch.isclose(robot.data.joint_pos[:, -2], gripper_open_val.to(env.device), atol=atol, rtol=rtol),
        correctly_placed
    )

    return correctly_placed