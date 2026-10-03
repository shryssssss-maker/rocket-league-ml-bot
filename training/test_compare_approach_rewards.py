"""Unit checks for the offline counterfactual; no simulator or training."""
import unittest

from compare_approach_rewards import score_report


def row(step, *, touch=0, throttle=1, velocity_change=100, distance=10):
    return dict(episode=0, step=step,
                action=dict(throttle=throttle, boost=0),
                state=dict(ball_touches=touch,
                           velocity_toward_ball_change=velocity_change,
                           distance_progress=distance,
                           on_ground_before=True, on_ground_after=True),
                reward=dict(approach=0.01))


class CounterfactualTests(unittest.TestCase):
    def test_touch_interval_and_all_later_intervals_are_gated(self):
        report = dict(format_version=1, contract=dict(action_repeat=8),
                      checkpoint_config=dict(approach_weight=0.02),
                      steps=[row(1), row(2, touch=1), row(3)])
        scored = score_report(report)
        self.assertEqual(scored['summary']['pre_touch_steps'], 1)
        self.assertEqual(scored['summary']['touch_steps'], 1)
        self.assertGreater(scored['steps'][0]['gated_closing_velocity_improvement'], 0)
        self.assertEqual(scored['steps'][1]['gated_closing_velocity_improvement'], 0)
        self.assertEqual(scored['steps'][2]['gated_closing_velocity_improvement'], 0)

    def test_coasting_and_retreat_are_distinguished(self):
        report = dict(format_version=1, contract=dict(action_repeat=8),
                      checkpoint_config=dict(approach_weight=0.02),
                      steps=[row(1, throttle=0, velocity_change=0),
                             row(2, throttle=0, velocity_change=-100),
                             row(3, throttle=1, velocity_change=100)])
        scored = score_report(report)
        candidate = scored['summary']['gated_closing_velocity_improvement']
        self.assertEqual(candidate['positive_steps'], 1)
        self.assertEqual(candidate['no_propulsion_positive_steps'], 0)


if __name__ == '__main__':
    unittest.main()
