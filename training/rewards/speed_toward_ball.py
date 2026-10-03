"""SpeedTowardBallReward from RLGym's Rocket League training tutorial.

This is not included in the installed rlgym-rocket-league 2.0.1 reward package.
It is an experimental measurement component, not wired into PPO yet.
"""
import numpy as np
from rlgym.api import RewardFunction
from rlgym.rocket_league.common_values import CAR_MAX_SPEED


class SpeedTowardBallReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            # Match the current RLGym tutorial's team-relative frame.
            car_physics = car.physics if car.is_orange else car.inverted_physics
            ball_physics = state.ball if car.is_orange else state.inverted_ball
            to_ball = ball_physics.position - car_physics.position
            distance = float(np.linalg.norm(to_ball))
            if distance < 1e-8:
                rewards[agent] = 0.0  # Avoid an undefined direction at overlap.
                continue
            toward_speed = float(np.dot(car_physics.linear_velocity, to_ball / distance))
            rewards[agent] = max(toward_speed / CAR_MAX_SPEED, 0.0)
        return rewards
