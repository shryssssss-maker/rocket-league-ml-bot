"""Check tutorial velocity reward scale and team symmetry without simulation."""
import unittest
from types import SimpleNamespace
import numpy as np

from rewards.speed_toward_ball import SpeedTowardBallReward


def physics(position, velocity):
    return SimpleNamespace(position=np.array(position, dtype=np.float32),
                           linear_velocity=np.array(velocity, dtype=np.float32))


class SpeedTowardBallTests(unittest.TestCase):
    def value(self, velocity, position=(0, 0, 17), ball=(0, 1000, 93)):
        car_physics = physics(position, velocity)
        state = SimpleNamespace(
            cars={'orange-0': SimpleNamespace(is_orange=True, physics=car_physics)},
            ball=physics(ball, (0, 0, 0)))
        return SpeedTowardBallReward().get_rewards(['orange-0'], state, {}, {}, {})['orange-0']

    def test_toward_away_and_sideways(self):
        # Ball is slightly above the car, so use a level ball for exact 0.5.
        ball = (0, 1000, 17)
        self.assertAlmostEqual(self.value((0, 1150, 0), ball=ball), 0.5)
        self.assertEqual(self.value((0, -1150, 0), ball=ball), 0)
        self.assertEqual(self.value((1150, 0, 0), ball=ball), 0)
        self.assertEqual(self.value((0, 1150, 0), ball=(0, 0, 17)), 0)

    def test_blue_uses_inverted_physics_and_ball(self):
        car = SimpleNamespace(is_orange=False,
                              physics=physics((0, -1000, 17), (0, 1150, 0)),
                              inverted_physics=physics((0, 1000, 17), (0, -1150, 0)))
        state = SimpleNamespace(cars={'blue-0': car},
                                ball=physics((0, 0, 17), (0, 0, 0)),
                                inverted_ball=physics((0, 0, 17), (0, 0, 0)))
        reward = SpeedTowardBallReward().get_rewards(['blue-0'], state, {}, {}, {})
        self.assertAlmostEqual(reward['blue-0'], 0.5)


if __name__ == '__main__':
    unittest.main()
