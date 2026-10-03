"""The optional ground policy must keep valid lookup IDs and checkpoint behavior."""
import unittest
import torch
from environment import ACTION_TABLE
from policies import ActorCritic


class GroundPolicyTests(unittest.TestCase):
    def test_first_24_lookup_actions_are_ground_actions(self):
        self.assertEqual(ACTION_TABLE.shape, (90, 8))
        self.assertTrue((ACTION_TABLE[:24, 2] == 0).all())  # pitch
        self.assertTrue((ACTION_TABLE[:24, 4] == 0).all())  # roll
        self.assertTrue((ACTION_TABLE[:24, 5] == 0).all())  # jump

    def test_ground_policy_never_samples_an_aerial_id(self):
        policy = ActorCritic(128, ground_only=True)
        observations = torch.randn(200, 92)
        distribution, _ = policy(observations)
        self.assertEqual(distribution.logits.shape, (200, 90))
        self.assertTrue((distribution.sample() < 24).all().item())
        self.assertTrue((distribution.logits.argmax(-1) < 24).all().item())

    def test_unmasked_policy_retains_full_action_space(self):
        policy = ActorCritic(128)
        distribution, _ = policy(torch.zeros(1, 92))
        self.assertTrue((distribution.probs[0, 24:] > 0).all().item())


if __name__ == '__main__':
    unittest.main()
