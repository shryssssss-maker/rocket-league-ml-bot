"""Check the separate conventional early-contact reward recipe."""
import math
import unittest
from types import SimpleNamespace

import numpy as np
from rlgym.rocket_league.reward_functions import TouchReward

from environment import Match
from rewards import guide_early_reward
from rewards.air import InAirReward
from rewards.guide_face_ball import GuideFaceBallReward
from rewards.speed_toward_ball import SpeedTowardBallReward


class GuideRewardTests(unittest.TestCase):
    def test_exact_terms_and_weights_without_goal_or_approach(self):
        combined = guide_early_reward()
        self.assertEqual([type(term) for term in combined.reward_fns],
                         [TouchReward, SpeedTowardBallReward,
                          GuideFaceBallReward, InAirReward])
        self.assertEqual(combined.weights, (50., 5., 1., .15))

    def test_orientation_face_reward_does_not_require_car_motion(self):
        car = SimpleNamespace(physics=SimpleNamespace(
            position=np.array([0., -500., 17.]),
            forward=np.array([0., 1., 0.]),
            linear_velocity=np.zeros(3)))
        state = SimpleNamespace(cars={'blue-0': car},
                                ball=SimpleNamespace(position=np.array([0., 0., 17.])))
        values = GuideFaceBallReward().get_rewards(['blue-0'], state, {}, {}, {})
        self.assertAlmostEqual(values['blue-0'], 1.)
        car.physics.forward = np.array([0., -1., 0.])
        values = GuideFaceBallReward().get_rewards(['blue-0'], state, {}, {}, {})
        self.assertAlmostEqual(values['blue-0'], -1.)

    def test_real_rocketsim_step_uses_new_recipe(self):
        match = Match(42, 1., 'bridge', reward_recipe='guide_early_v1')
        try:
            self.assertEqual(match.env.reward_fn.weights, (50., 5., 1., .15))
            _, _, reward, _, _, _ = match.step(0)
            self.assertTrue(math.isfinite(reward))
        finally:
            match.close()


if __name__ == '__main__':
    unittest.main()
