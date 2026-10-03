"""Car motion toward the previous ball position, optionally without penalties."""
import numpy as np
from rlgym.api import RewardFunction
from rlgym.rocket_league.common_values import CAR_MAX_SPEED, TICKS_PER_SECOND


class ApproachReward(RewardFunction):
    def __init__(self, action_repeat=8, positive_only=False, until_first_touch=False):
        self.max_step_distance = CAR_MAX_SPEED * action_repeat / TICKS_PER_SECOND
        self.positive_only = positive_only
        self.until_first_touch = until_first_touch
        self.previous_cars = {}
        self.previous_ball = None
        self.has_touched = set()

    def reset(self, agents, initial_state, shared_info):
        self.previous_ball = initial_state.ball.position[:2].copy()
        self.has_touched = set()
        self.previous_cars = {
            agent: initial_state.cars[agent].physics.position[:2].copy()
            for agent in agents
        }

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            previous_car = self.previous_cars[agent]
            current_car = state.cars[agent].physics.position[:2]
            # Ball touches are reported for the current 8-tick action interval.
            # Exclude that entire interval: we cannot split pre/post-contact
            # physics inside it, and later approach credit stays disabled.
            if self.until_first_touch and state.cars[agent].ball_touches > 0:
                self.has_touched.add(agent)
            if self.until_first_touch and agent in self.has_touched:
                rewards[agent] = 0.0
            else:
                to_ball = self.previous_ball - previous_car
                distance = float(np.linalg.norm(to_ball))
                if distance < 1:
                    rewards[agent] = 0.0
                else:
                    # Ball movement alone earns nothing; sideways or away motion
                    # cannot harvest a positive approach reward.
                    car_motion = current_car - previous_car
                    progress = float(np.dot(car_motion, to_ball / distance))
                    lower = 0 if self.positive_only else -1
                    rewards[agent] = float(np.clip(progress / self.max_step_distance, lower, 1))
            self.previous_cars[agent] = current_car.copy()
        self.previous_ball = state.ball.position[:2].copy()
        return rewards
