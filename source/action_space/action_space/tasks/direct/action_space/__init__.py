# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Intercept-Direct-v0",
    entry_point=f"{__name__}.action_space_env:ActionSpaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.action_space_env_cfg:ActionSpaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
gym.register(
    id="Intercept-Swarm-3v2-Direct-v0",
    entry_point=f"{__name__}.swarm_env:SwarmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.swarm_env_cfg:SwarmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SwarmPPORunnerCfg",
    },
)

gym.register(
    id="Intercept-Swarm-3v2-Bump-Direct-v0",
    entry_point=f"{__name__}.swarm_env:SwarmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.swarm_env_cfg:SwarmBumpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SwarmBumpPPORunnerCfg",
    },
)

gym.register(
    id="Intercept-Defense-3v2-v0",
    entry_point=f"{__name__}.defense_env:DefenseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.defense_env_cfg:DefenseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DefensePPORunnerCfg",
    },
)

gym.register(
    id="Intercept-Defense-3v2-Residual-v0",
    entry_point=f"{__name__}.defense_env:DefenseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.defense_env_cfg:DefenseResidualEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DefenseResidualPPORunnerCfg",
    },
)

gym.register(
    id="Intercept-Defense-3v2-StickyNet-v0",
    entry_point=f"{__name__}.defense_env:DefenseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.defense_env_cfg:DefenseStickyNetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DefensePPORunnerCfg",
    },
)
