"""Exercise metric collection through real RocketSim touch and timeout transitions."""
import unittest
import numpy as np
from environment import Match, ACTION_TABLE
from rewards.approach import ApproachReward


class MatchMetricTests(unittest.TestCase):
    def test_real_simulator_first_touch_gates_approach_only(self):
        match = Match(42, 2, 'touch', goal_weight=0, touch_weight=1,
                      approach_weight=.02, face_weight=.01, air_weight=.001,
                      approach_until_first_touch=True)
        try:
            car = match.env.state.cars[match.agent]
            direction = 1 if match.agent == 'blue-0' else -1
            car.physics.position = np.array([0, -direction * 150, 17], dtype=np.float32)
            car.physics.euler_angles = np.array([0, direction * np.pi / 2, 0], dtype=np.float32)
            car.physics.linear_velocity = np.array([0, direction * 1000, 0], dtype=np.float32)
            match.env.set_state(match.env.state)
            approach = next(fn for fn in match.env.reward_fn.reward_fns
                            if isinstance(fn, ApproachReward))
            idle = int(np.flatnonzero(np.all(ACTION_TABLE == 0, axis=1))[0])
            for _ in range(10):
                match.step(idle)
                if match.touches:
                    break
            self.assertGreater(match.touches, 0)
            self.assertIn(match.agent, approach.has_touched)
            match.step(idle)
            self.assertIn(match.agent, approach.has_touched)
        finally:
            match.close()

    def test_touch_event_and_episode_summary(self):
        match = Match(42, 2, 'touch')
        try:
            state = match.env.state
            car = state.cars[match.agent]
            direction = 1 if match.agent == 'blue-0' else -1
            car.physics.position = np.array([0, -direction * 150, 17], dtype=np.float32)
            car.physics.euler_angles = np.array([0, direction * np.pi / 2, 0], dtype=np.float32)
            car.physics.linear_velocity = np.array([0, direction * 1000, 0], dtype=np.float32)
            match.env.set_state(state)
            idle = int(np.flatnonzero(np.all(ACTION_TABLE == 0, axis=1))[0])
            for _ in range(30):
                *_, info = match.step(idle)
                if info is not None:
                    break
            self.assertIsNotNone(info)
            self.assertGreater(info['touches'], 0)
            self.assertGreaterEqual(info['raw_contacts'], info['touches'])
            self.assertGreater(info['ground_touch_steps'], 0)
            self.assertGreater(info['goalward_touch_steps'], 0)
            self.assertEqual(info['first_touch_by'], 'learner')
            self.assertIsNotNone(info['time_to_first_touch_seconds'])
            self.assertLessEqual(info['time_to_first_touch_seconds'], info['episode_seconds'])
            self.assertEqual(info['goals_for'], 0)
            self.assertEqual(info['goals_against'], 0)
            self.assertEqual(info['opponent_own_goal_proxy'], 0)
        finally:
            match.close()


if __name__ == '__main__':
    unittest.main()
