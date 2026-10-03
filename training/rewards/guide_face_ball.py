"""Orientation-only FaceBall reward for the conventional early-stage recipe."""
import numpy as np
from rlgym.api import RewardFunction


class GuideFaceBallReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            car = state.cars[agent].physics
            to_ball = state.ball.position - car.position
            distance = float(np.linalg.norm(to_ball))
            rewards[agent] = (float(np.clip(np.dot(car.forward, to_ball / distance), -1, 1))
                              if distance > 1e-8 else 0.0)
        return rewards
