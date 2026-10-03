"""Small categorical actor and independent value network."""
import torch
from torch import nn
from torch.distributions import Categorical
from observations import OBS_SIZE


class ActorCritic(nn.Module):
    def __init__(self, hidden_size=128, ground_only=False):
        super().__init__()
        # RLGym LookupTableAction indexes 0..23 are its ground-control block.
        # Masking keeps the checkpoint's 90-action output and action IDs intact.
        self.ground_only = ground_only
        def network(outputs):
            return nn.Sequential(nn.Linear(OBS_SIZE, hidden_size), nn.Tanh(),
                                 nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                                 nn.Linear(hidden_size, outputs))
        self.actor = network(90)
        self.critic = network(1)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2**0.5)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, obs):
        logits = self.actor(obs)
        if self.ground_only:
            logits = logits.clone()
            logits[..., 24:] = -1e9
        return Categorical(logits=logits), self.critic(obs).squeeze(-1)
