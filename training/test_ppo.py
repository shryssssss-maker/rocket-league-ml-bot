"""Regression checks for bootstrapping at goals, timeouts, and rollout boundaries."""
import unittest
import numpy as np
from ppo import advantages


class AdvantageTests(unittest.TestCase):
    def test_goal_has_no_bootstrap(self):
        adv, returns = advantages(np.array([[10.]], dtype=np.float32), np.array([[2.]]),
            np.array([[100.]]), np.array([[1.]]), np.array([[0.]]), 0.9, 1)
        np.testing.assert_allclose(adv, [[8.]])
        np.testing.assert_allclose(returns, [[10.]])

    def test_timeout_bootstraps_without_crossing_reset(self):
        # Next episode's huge reward must not leak back across the timeout.
        adv, returns = advantages(np.array([[1.], [1000.]], dtype=np.float32),
            np.array([[2.], [0.]]), np.array([[3.], [0.]]), np.array([[0.], [1.]]),
            np.array([[1.], [0.]]), 0.9, 1)
        np.testing.assert_allclose(adv[0], [1.7])
        np.testing.assert_allclose(returns[0], [3.7])

    def test_rollout_cutoff_keeps_bootstrap(self):
        adv, returns = advantages(np.array([[1.], [2.]], dtype=np.float32),
            np.zeros((2, 1)), np.array([[0.], [3.]]), np.zeros((2, 1)),
            np.zeros((2, 1)), 0.9, 1)
        np.testing.assert_allclose(returns, [[5.23], [4.7]], rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
