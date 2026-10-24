# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 300
    save_interval = 50
    experiment_name = "intercept_direct"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class SwarmPPORunnerCfg(PPORunnerCfg):
    max_iterations = 1000
    save_interval = 100
    experiment_name = "intercept_swarm_3v2"

    def __post_init__(self):
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]


@configclass
class SwarmBumpPPORunnerCfg(SwarmPPORunnerCfg):
    max_iterations = 1500
    experiment_name = "intercept_swarm_3v2_bump"


@configclass
class DefensePPORunnerCfg(SwarmPPORunnerCfg):
    """Randomized 3v2 defense: longer rollouts (30 s rounds) and more iterations."""

    num_steps_per_env = 48
    max_iterations = 3000
    save_interval = 100
    experiment_name = "intercept_defense_3v2"


@configclass
class DefenseResidualPPORunnerCfg(DefensePPORunnerCfg):
    """Residual policy: start with small corrections so the scripted controller's skill is kept early on."""

    experiment_name = "intercept_defense_3v2_residual"

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.3
