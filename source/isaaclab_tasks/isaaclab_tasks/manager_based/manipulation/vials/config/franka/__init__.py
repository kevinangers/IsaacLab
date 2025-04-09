# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import (
    agents,
    vials_ik_rel_blueprint_env_cfg,
    vials_ik_rel_env_cfg,
    vials_ik_rel_instance_randomize_env_cfg,
    vials_joint_pos_env_cfg,
    vials_joint_pos_instance_randomize_env_cfg,
)

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Vials-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": vials_joint_pos_env_cfg.FrankaVialsEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Vials-Instance-Randomize-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": vials_joint_pos_instance_randomize_env_cfg.FrankaVialsInstanceRandomizeEnvCfg,
    },
    disable_env_checker=True,
)


##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Vials-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": vials_ik_rel_env_cfg.FrankaVialsEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
        "robomimic_diffusion_policy_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/diffusion_policy_low_dim.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Vials-Franka-IK-Rel-Mimic-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": vials_ik_rel_env_cfg.FrankaVialsEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
        "robomimic_diffusion_policy_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/diffusion_policy_low_dim.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Vials-Instance-Randomize-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": vials_ik_rel_instance_randomize_env_cfg.FrankaVialsInstanceRandomizeEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Vials-Franka-IK-Rel-Blueprint-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": vials_ik_rel_blueprint_env_cfg.FrankaVialsBlueprintEnvCfg,
    },
    disable_env_checker=True,
)
