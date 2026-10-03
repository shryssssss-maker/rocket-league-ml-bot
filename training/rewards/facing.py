"""Reward facing the ball while moving; standing still cannot farm it."""
import numpy as np
from rlgym.api import RewardFunction
from rlgym.rocket_league.common_values import CAR_MAX_SPEED


class FacingBallReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            physics = state.cars[agent].physics
            toward = state.ball.position[:2] - physics.position[:2]
            distance = float(np.linalg.norm(toward))
            speed = float(np.linalg.norm(physics.linear_velocity[:2]))
            if distance < 1 or speed < 1:
                rewards[agent] = 0.0
            else:
                alignment = float(np.dot(physics.forward[:2], toward / distance))
                rewards[agent] = float(np.clip(alignment, -1, 1) *
                                       np.clip(speed / CAR_MAX_SPEED, 0, 1))
        return rewards
