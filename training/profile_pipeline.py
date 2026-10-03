"""Time one disposable rollout/update using the current single-process pipeline.

The loaded checkpoint is never saved or modified on disk. The script performs
one in-memory PPO update to measure its real cost, then discards the model.
It does not measure multiprocessing or train a candidate baseline.
"""
import argparse
import json
import random
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from checkpoint import load_checkpoint
from config import Config
from environment import Match, contract
from ppo import advantages, update


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def profile(checkpoint, *, scenario, episode_seconds, num_envs, rollout_steps, seed):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model, data = load_checkpoint(checkpoint, device)
    config = Config(**data['config'])
    config.scenario = scenario
    config.episode_seconds = episode_seconds
    config.num_envs = num_envs
    config.seed = seed
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer.load_state_dict(data['optimizer'])
    matches = []
    try:
        for i in range(num_envs):
            matches.append(Match(seed + i, config.episode_seconds, config.scenario,
                                 config.opponent_version, config.approach_weight,
                                 config.goal_weight, config.touch_weight,
                                 config.face_weight, config.air_weight,
                                 config.approach_positive_only,
                                 config.approach_until_first_touch,
                                 config.reward_recipe))
        # Exclude CUDA context and first-forward setup from timed collection.
        with torch.no_grad():
            warm_obs = torch.as_tensor(np.stack([m.obs for m in matches]), device=device)
            for _ in range(3):
                model(warm_obs)
        synchronize(device)
        timers = dict(observation_batching=0., policy_inference=0.,
                      simulator_step=0., rollout_bookkeeping=0.,
                      bootstrap_and_gae=0., final_batching=0., ppo_update=0.)
        obs_buf, next_buf, act_buf, logp_buf, val_buf = [], [], [], [], []
        rew_buf, term_buf, trunc_buf = [], [], []
        completed_episodes = 0
        model.eval()
        wall_start = time.perf_counter()
        for _ in range(rollout_steps):
            start = time.perf_counter()
            obs = np.stack([match.obs for match in matches])
            timers['observation_batching'] += time.perf_counter() - start

            start = time.perf_counter()
            with torch.no_grad():
                dist, value = model(torch.as_tensor(obs, device=device))
                action = dist.sample()
                logp = dist.log_prob(action)
            action_np = action.cpu().numpy()  # Synchronizes GPU inference.
            logp_np = logp.cpu().numpy()
            value_np = value.cpu().numpy()
            timers['policy_inference'] += time.perf_counter() - start

            start = time.perf_counter()
            results = [match.step(int(a)) for match, a in zip(matches, action_np)]
            timers['simulator_step'] += time.perf_counter() - start

            start = time.perf_counter()
            obs_buf.append(obs)
            next_buf.append(np.stack([result[1] for result in results]))
            act_buf.append(action_np)
            logp_buf.append(logp_np)
            val_buf.append(value_np)
            rew_buf.append([result[2] for result in results])
            term_buf.append([result[3] for result in results])
            trunc_buf.append([result[4] for result in results])
            completed_episodes += sum(result[5] is not None for result in results)
            timers['rollout_bookkeeping'] += time.perf_counter() - start

        start = time.perf_counter()
        obs_buf = np.asarray(obs_buf)
        next_buf = np.asarray(next_buf)
        with torch.no_grad():
            next_values = model.critic(torch.as_tensor(
                next_buf.reshape(-1, 92), device=device))
            next_values = next_values.squeeze(-1).cpu().numpy().reshape(rollout_steps, num_envs)
        adv, returns = advantages(np.asarray(rew_buf, dtype=np.float32),
                                  np.asarray(val_buf), next_values,
                                  np.asarray(term_buf, dtype=np.float32),
                                  np.asarray(trunc_buf, dtype=np.float32),
                                  config.gamma, config.gae_lambda)
        synchronize(device)
        timers['bootstrap_and_gae'] = time.perf_counter() - start

        start = time.perf_counter()
        batch = [obs_buf.reshape(-1, 92), np.asarray(act_buf).reshape(-1),
                 np.asarray(logp_buf).reshape(-1), adv.reshape(-1),
                 returns.reshape(-1)]
        timers['final_batching'] = time.perf_counter() - start

        start = time.perf_counter()
        metrics = update(model, optimizer, batch, config, device)
        synchronize(device)
        timers['ppo_update'] = time.perf_counter() - start
        wall_seconds = time.perf_counter() - wall_start
    finally:
        for match in matches:
            match.close()
    transitions = rollout_steps * num_envs
    measured = sum(timers.values())
    return dict(format_version=1, checkpoint=str(checkpoint.resolve()),
                checkpoint_step=data['step'], device=str(device),
                config=asdict(config), contract=contract(),
                versions={package: version(package) for package in
                          ('torch', 'rlgym', 'rlgym-api', 'rlgym-rocket-league',
                           'rocketsim', 'numpy')},
                transitions=transitions, rollout_steps=rollout_steps,
                num_envs=num_envs, completed_episodes=completed_episodes,
                seconds=timers, wall_seconds=wall_seconds,
                unclassified_seconds=max(0., wall_seconds - measured),
                transitions_per_wall_second=transitions / wall_seconds,
                simulated_transitions_per_step_second=transitions / timers['simulator_step'],
                component_wall_fractions={name: value / wall_seconds
                                          for name, value in timers.items()},
                ppo_metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenario', choices=('touch', 'bridge', 'approach', 'kickoff'),
                        default='bridge')
    parser.add_argument('--episode-seconds', type=float, default=10.)
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--rollout-steps', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if (args.num_envs < 1 or args.rollout_steps < 2 or args.seed < 0 or
            not np.isfinite(args.episode_seconds) or args.episode_seconds <= 0):
        parser.error('num-envs must be >=1, rollout-steps >=2, episode-seconds positive, and seed nonnegative')
    if args.output.exists():
        parser.error('Refusing to overwrite an existing profile report')
    report = profile(args.model, scenario=args.scenario,
                     episode_seconds=args.episode_seconds,
                     num_envs=args.num_envs, rollout_steps=args.rollout_steps,
                     seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
        file.write('\n')
    visible = {key: report[key] for key in
               ('device', 'transitions', 'num_envs', 'wall_seconds',
                'transitions_per_wall_second', 'seconds', 'component_wall_fractions')}
    print(json.dumps(visible, indent=2))
    print(f'Report saved: {args.output}')


if __name__ == '__main__':
    main()
