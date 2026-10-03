"""Controlled checks for early directional, facing, and air rewards."""
import unittest
from types import SimpleNamespace
import numpy as np
from rewards import baseline_reward
from rewards.facing import FacingBallReward
from rewards.air import InAirReward


def fixture(velocity=(0, 0), forward=(0, 1), grounded=True):
    physics = SimpleNamespace(
        position=np.array([0, -500, 17], dtype=np.float32),
        linear_velocity=np.array([*velocity, 0], dtype=np.float32),
        forward=np.array([*forward, 0], dtype=np.float32))
    car = SimpleNamespace(physics=physics, on_ground=grounded)
    return SimpleNamespace(cars={'blue-0': car},
                           ball=SimpleNamespace(position=np.array([0, 0, 93], dtype=np.float32)))


class EarlyRewardTests(unittest.TestCase):
    def value(self, reward, state):
        reward.reset(['blue-0'], state, {})
        return reward.get_rewards(['blue-0'], state, {}, {}, {})['blue-0']

    def test_facing_while_moving(self):
        reward = FacingBallReward()
        self.assertGreater(self.value(reward, fixture((0, 1000))), 0)
        self.assertLess(self.value(reward, fixture((0, 1000), (0, -1))), 0)
        self.assertAlmostEqual(self.value(reward, fixture((0, 1000), (1, 0))), 0)
        self.assertEqual(self.value(reward, fixture()), 0)

    def test_air_only_when_airborne(self):
        reward = InAirReward()
        self.assertEqual(self.value(reward, fixture()), 0)
        self.assertEqual(self.value(reward, fixture(grounded=False)), 1)

    def test_old_and_new_reward_configs_build(self):
        self.assertIsNotNone(baseline_reward())
        self.assertIsNotNone(baseline_reward(goal_weight=0, touch_weight=1,
                                             approach_weight=.02, face_weight=.01,
                                             air_weight=.001))
        with self.assertRaises(ValueError):
            baseline_reward(goal_weight=0, touch_weight=0)


if __name__ == '__main__':
    unittest.main()
