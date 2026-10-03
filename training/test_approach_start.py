"""Small reset-only check for the stage-two RocketSim fixture."""
import unittest
import numpy as np
from environment import Match


class ApproachStartTests(unittest.TestCase):
    def test_bridge_overlaps_close_and_wider_starts(self):
        match = Match(42, 10, 'bridge')
        try:
            starts = []
            for _ in range(20):
                learner = match.env.state.cars[match.agent]
                x, y = learner.physics.position[:2]
                self.assertLessEqual(abs(float(x)), 300)
                self.assertGreaterEqual(abs(float(y)), 550)
                self.assertLessEqual(abs(float(y)), 1100)
                starts.append(abs(float(y)))
                match.reset()
            self.assertLess(min(starts), 700)
            self.assertGreater(max(starts), 950)
            self.assertEqual(match.obs.shape, (92,))
            self.assertTrue(np.isfinite(match.step(0)[2]))
        finally:
            match.close()

    def test_varied_starts_and_alternating_sides(self):
        match = Match(42, 10, 'approach')
        try:
            starts = []
            for _ in range(8):
                learner = match.env.state.cars[match.agent]
                opponent = match.env.state.cars[match.opponent]
                x, y = learner.physics.position[:2]
                self.assertLessEqual(abs(float(x)), 600)
                self.assertGreaterEqual(abs(float(y)), 900)
                self.assertLessEqual(abs(float(y)), 1700)
                self.assertAlmostEqual(float(np.linalg.norm(opponent.physics.position[:2])), 3500, delta=1)
                self.assertAlmostEqual(float(np.linalg.norm(match.env.state.ball.position[:2])), 0, delta=1)
                starts.append((match.agent, round(float(x)), round(float(y))))
                match.reset()
            self.assertEqual([side for side, _, _ in starts], ['blue-0', 'orange-0'] * 4)
            self.assertGreater(len(set(starts)), 4)
            self.assertEqual(match.obs.shape, (92,))
            self.assertTrue(np.isfinite(match.step(0)[2]))
        finally:
            match.close()


if __name__ == '__main__':
    unittest.main()
