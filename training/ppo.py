"""PPO-Clip with GAE; true terminations and time limits have different masks."""
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_


def advantages(rewards, values, next_values, terminated, truncated, gamma, lam):
    result = np.zeros_like(rewards)
    carry = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        # Timeouts bootstrap V(final_obs), goals do not. Neither crosses a reset.
        delta = rewards[t] + gamma * next_values[t] * (1 - terminated[t]) - values[t]
        carry = delta + gamma * lam * (1 - np.maximum(terminated[t], truncated[t])) * carry
        result[t] = carry
    return result, result + values


def update(model, optimizer, data, config, device):
    obs, actions, old_logp, adv, returns = [torch.as_tensor(x, device=device) for x in data]
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    metrics = []
    model.train()
    for _ in range(config.epochs):
        epoch_kl = []
        for indices in torch.randperm(len(obs), device=device).split(config.minibatch_size):
            dist, value = model(obs[indices])
            log_ratio = dist.log_prob(actions[indices]) - old_logp[indices]
            ratio = log_ratio.exp()
            policy_loss = -torch.minimum(ratio * adv[indices],
                ratio.clamp(1-config.clip_ratio, 1+config.clip_ratio) * adv[indices]).mean()
            value_loss = (value - returns[indices]).square().mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite PPO loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if not torch.isfinite(norm):
                raise FloatingPointError('Non-finite PPO gradient')
            optimizer.step()
            kl = ((ratio - 1) - log_ratio).mean().item()
            metrics.append([policy_loss.item(), value_loss.item(), entropy.item(), kl])
            epoch_kl.append(kl)
        if np.mean(epoch_kl) > config.target_kl:
            break
    return dict(zip(('policy_loss', 'value_loss', 'entropy', 'approx_kl'),
                    np.mean(metrics, axis=0).tolist()))
