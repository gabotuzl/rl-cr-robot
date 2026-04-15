from dataclasses import dataclass, field
from typing import List


@dataclass
class EnvConfig:
    # Workspace / target sampling
    x_variation: float = 0.1
    y_variation: float = 0.1
    z_variation: float = 0.1
    num_timesteps_per_step: int = 3000   # simulation substeps per RL step

    # Observation history
    state_history_len: int = 5
    reward_history_len: int = 10
    action_history_len: int = 5
    nodes_checked: int = 10  # Number of nodes checked for movement along the rod


@dataclass
class PPOConfig:
    n_envs: int = 12
    total_timesteps: int = 20_500_000
    n_steps: int = 2048
    batch_size: int = 4096
    n_epochs: int = 6
    learning_rate: float = 4e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.005
    vf_coef: float = 0.3
    max_grad_norm: float = 0.5
    net_arch_pi: List[int] = field(default_factory=lambda: [256, 128])
    net_arch_vf: List[int] = field(default_factory=lambda: [256, 256])


@dataclass
class PathConfig:
    checkpoint_dir: str = "./checkpoints/"
    log_dir: str = "./logs/"
    tensorboard_dir: str = "./ppo_tensorboard/"
    model_save_name: str = "ppo_CLAWAR"
    vecnorm_save_name: str = "ppo_CLAWAR_env.pkl"
    save_dir: str = "test_runs"


@dataclass
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    paths: PathConfig = field(default_factory=PathConfig)


# Ready-to-use singleton
CONFIG = TrainConfig()