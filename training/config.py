"""Small, explicit defaults for the first PPO experiment."""
from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 42
    num_envs: int = 4
    rollout_steps: int = 256
    hidden_size: int = 128
    episode_seconds: float = 30.0
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    epochs: int = 4
    minibatch_size: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    checkpoint_interval: int = 32768
    scenario: str = 'kickoff'
    opponent_version: str = 'v1'
    reward_name: str = 'goal10_touch0.1_v1'
    reward_recipe: str = 'custom'
    ground_only: bool = False
    approach_weight: float = 0.0
    velocity_weight: float = 0.0
    goal_weight: float = 10.0
    touch_weight: float = 0.1
    face_weight: float = 0.0
    air_weight: float = 0.0
    approach_positive_only: bool = False
    approach_until_first_touch: bool = False
