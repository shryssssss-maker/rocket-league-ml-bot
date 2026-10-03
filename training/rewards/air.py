"""Tiny optional anti-forgetting reward for using aerial actions."""
from rlgym.api import RewardFunction


class InAirReward(RewardFunction):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        return {agent: float(not state.cars[agent].on_ground) for agent in agents}
