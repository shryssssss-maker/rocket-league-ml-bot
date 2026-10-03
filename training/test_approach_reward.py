"""Isolate the approach term from goals, touches, and opponent behavior."""
import unittest
from types import SimpleNamespace
import numpy as np
from rewards.approach import ApproachReward


def state(car_x, car_y, ball_x=0, ball_y=0, touches=0):
    return SimpleNamespace(
        ball=SimpleNamespace(position=np.array([ball_x, ball_y, 93], dtype=np.float32)),
        cars={'blue-0': SimpleNamespace(physics=SimpleNamespace(
            position=np.array([car_x, car_y, 17], dtype=np.float32)),
            ball_touches=touches)})


class ApproachRewardTests(unittest.TestCase):
    def setUp(self):
        self.reward = ApproachReward()
        self.reward.reset(['blue-0'], state(0, -500), {})

    def step(self, next_state):
        return self.reward.get_rewards(['blue-0'], next_state, {}, {}, {})['blue-0']

    def test_driving_toward_ball_is_positive(self):
        self.assertGreater(self.step(state(0, -400)), 0)

    def test_driving_away_is_negative(self):
        self.assertLess(self.step(state(0, -600)), 0)

    def test_sideways_motion_is_zero(self):
        self.assertAlmostEqual(self.step(state(100, -500)), 0)

    def test_ball_approaching_stationary_car_earns_nothing(self):
        self.assertEqual(self.step(state(0, -500, 0, -200)), 0)

    def test_stationary_or_close_car_earns_nothing(self):
        self.assertEqual(self.step(state(0, -500)), 0)
        self.reward.reset(['blue-0'], state(0, 0), {})
        self.assertEqual(self.step(state(0, 0)), 0)

    def test_large_teleport_is_bounded(self):
        self.assertEqual(self.step(state(0, 5000)), 1)

    def test_positive_only_keeps_approach_but_does_not_penalize_retreat(self):
        reward = ApproachReward(positive_only=True)
        reward.reset(['blue-0'], state(0, -500), {})
        away = reward.get_rewards(['blue-0'], state(0, -600), {}, {}, {})['blue-0']
        toward = reward.get_rewards(['blue-0'], state(0, -500), {}, {}, {})['blue-0']
        self.assertEqual(away, 0)
        self.assertGreater(toward, 0)

    def test_first_touch_gates_only_approach_and_resets_next_episode(self):
        reward = ApproachReward(until_first_touch=True)
        reward.reset(['blue-0'], state(0, -500), {})
        def step(next_state):
            return reward.get_rewards(['blue-0'], next_state, {}, {}, {})['blue-0']
        self.assertGreater(step(state(0, -400)), 0)
        self.assertEqual(step(state(0, -300, touches=1)), 0)
        self.assertEqual(step(state(0, -200)), 0)
        reward.reset(['blue-0'], state(0, -500), {})
        self.assertLess(step(state(0, -600)), 0)  # Signed formula is unchanged.

    def test_first_touch_gate_is_per_agent(self):
        reward = ApproachReward(until_first_touch=True)
        initial = state(0, -500)
        initial.cars['orange-0'] = SimpleNamespace(
            physics=SimpleNamespace(position=np.array([0, 500, 17], dtype=np.float32)),
            ball_touches=0)
        agents = ['blue-0', 'orange-0']
        reward.reset(agents, initial, {})
        next_state = state(0, -400, touches=1)
        next_state.cars['orange-0'] = SimpleNamespace(
            physics=SimpleNamespace(position=np.array([0, 400, 17], dtype=np.float32)),
            ball_touches=0)
        values = reward.get_rewards(agents, next_state, {}, {}, {})
        self.assertEqual(values['blue-0'], 0)
        self.assertGreater(values['orange-0'], 0)

    def test_touch_reward_is_not_gated(self):
        from rewards import baseline_reward
        reward = baseline_reward(goal_weight=0, touch_weight=1,
                                 approach_weight=.02, approach_until_first_touch=True)
        reward.reset(['blue-0'], state(0, -500), {})
        values = reward.get_rewards(['blue-0'], state(0, -400, touches=1),
                                    {}, {}, {})
        self.assertEqual(values['blue-0'], 1.0)


if __name__ == '__main__':
    unittest.main()
