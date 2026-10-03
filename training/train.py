"""Train one policy against a fixed scripted opponent; CUDA is required."""
import argparse
import csv
import json
import math
import random
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from checkpoint import load_checkpoint, save_checkpoint
from config import Config
from environment import Match
from policies import ActorCritic
from ppo import advantages, update

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=65536, help='Additional learner transitions')
    parser.add_argument('--num-envs', type=int, default=4, help='Batched CPU simulator instances')
    parser.add_argument('--run-name', default=datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate and print resolved configuration without training or writing files')
    parser.add_argument('--reward-recipe', choices=('custom', 'guide_early_v1'), default=None,
                        help='Use a named fixed reward recipe; default preserves checkpoint configuration')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scenario', choices=('kickoff', 'touch', 'bridge', 'approach'), default=None)
    parser.add_argument('--opponent', choices=('v1', 'v2'), default=None)
    parser.add_argument('--episode-seconds', type=float, default=None)
    parser.add_argument('--ground-only', action='store_true',
                        help='Mask aerial/jump actions; retain 90 output logits and lookup IDs')
    parser.add_argument('--approach-weight', type=float, default=None,
                        help='Weight for the optional signed ground-ball approach term')
    parser.add_argument('--goal-weight', type=float, default=None)
    parser.add_argument('--touch-weight', type=float, default=None)
    parser.add_argument('--face-weight', type=float, default=None)
    parser.add_argument('--air-weight', type=float, default=None)
    parser.add_argument('--approach-positive-only', action='store_true',
                        help='Clamp negative approach progress to zero; optional reward ablation')
    parser.add_argument('--approach-until-first-touch', action='store_true',
                        help='Stop only the signed approach term at the learner first-touch interval')
    args = parser.parse_args()
    if args.steps < 2 or args.num_envs < 1:
        parser.error('steps must be >=2 and num-envs >=1')
    if args.episode_seconds is not None and (not math.isfinite(args.episode_seconds) or args.episode_seconds <= 0):
        parser.error('episode-seconds must be positive and finite')
    for name in ('approach_weight', 'goal_weight', 'touch_weight', 'face_weight', 'air_weight'):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f'{name.replace("_", "-")} must be finite and nonnegative')
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        parser.error('run-name must be a simple directory name')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: run gpu_test.py before PPO')
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = Config(seed=args.seed, num_envs=args.num_envs)
    device = torch.device('cuda')
    data = None
    if args.resume:
        model, data = load_checkpoint(args.resume, device)
        config = Config(**data['config'])
        config.num_envs, config.seed = args.num_envs, args.seed
    else:
        model = ActorCritic(config.hidden_size).to(device)
    if args.scenario is not None:
        config.scenario = args.scenario
    if args.opponent is not None:
        config.opponent_version = args.opponent
    if args.episode_seconds is not None:
        config.episode_seconds = args.episode_seconds
    if args.ground_only:
        config.ground_only = True
    for name in ('approach_weight', 'goal_weight', 'touch_weight', 'face_weight', 'air_weight'):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    if args.approach_positive_only:
        config.approach_positive_only = True
    if args.approach_until_first_touch:
        config.approach_until_first_touch = True
    if args.reward_recipe is not None:
        config.reward_recipe = args.reward_recipe
    if config.reward_recipe == 'guide_early_v1':
        if (args.ground_only or args.approach_positive_only or
                args.approach_until_first_touch or
                any(getattr(args, name) is not None for name in
                    ('approach_weight', 'goal_weight', 'touch_weight',
                     'face_weight', 'air_weight'))):
            parser.error('guide_early_v1 has fixed weights and full 90-action controls')
        if config.ground_only:
            parser.error('guide_early_v1 cannot resume a ground-only checkpoint')
        config.approach_weight = 0.0
        config.velocity_weight = 5.0
        config.touch_weight = 50.0
        config.face_weight = 1.0
        config.air_weight = 0.15
        config.goal_weight = 0.0
        config.approach_positive_only = False
        config.approach_until_first_touch = False
    if config.approach_positive_only and config.approach_until_first_touch:
        parser.error('First-touch-gated branch must retain the signed approach reward')
    if not any(getattr(config, name) for name in
               ('approach_weight', 'goal_weight', 'touch_weight', 'face_weight', 'air_weight')):
        parser.error('At least one reward weight must be nonzero')
    config.reward_name = ('guide_early_v1' if config.reward_recipe == 'guide_early_v1' else
                          f'goal{config.goal_weight:g}_touch{config.touch_weight:g}_'
                          f'approach{config.approach_weight:g}_face{config.face_weight:g}_'
                          f'air{config.air_weight:g}' +
                          ('_approach_positive' if config.approach_positive_only else '') +
                          ('_approach_until_first_touch' if config.approach_until_first_touch else ''))
    if args.dry_run:
        print(json.dumps(asdict(config), indent=2))
        return
    model.ground_only = config.ground_only
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    step = 0 if data is None else data['step']
    if data is not None:
        optimizer.load_state_dict(data['optimizer'])
    target = step + args.steps
    checkpoint_dir = ROOT / 'checkpoints' / args.run_name
    log_dir = ROOT / 'logs' / args.run_name
    # Refuse to overwrite an earlier experiment's files.
    if checkpoint_dir.exists() or log_dir.exists():
        raise FileExistsError('Choose a new --run-name; experiment directory already exists')
    checkpoint_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    (log_dir / 'config.json').write_text(json.dumps(asdict(config), indent=2), encoding='utf-8')
    save_checkpoint(checkpoint_dir / f'baseline_step_{step}.pt', model, optimizer, asdict(config), step)
    print(f'CUDA: {torch.cuda.get_device_name(0)}; parameters={sum(p.numel() for p in model.parameters())}', flush=True)
    matches = []
    recent = deque(maxlen=100)
    start = time.perf_counter()
    start_step = step
    checkpoint_number = 1
    next_checkpoint = step + config.checkpoint_interval
    fields = ['training_step', 'episode_reward', 'average_reward', 'episode_length',
              'completed_episodes', 'touches', 'policy_loss', 'value_loss', 'entropy',
              'approx_kl', 'checkpoint_number', 'steps_per_second', 'gpu_peak_mib']
    try:
        for i in range(config.num_envs):
            matches.append(Match(config.seed + i, config.episode_seconds,
                                 config.scenario, config.opponent_version,
                                 config.approach_weight, config.goal_weight,
                                 config.touch_weight, config.face_weight,
                                 config.air_weight, config.approach_positive_only,
                                 config.approach_until_first_touch, config.reward_recipe))
        with (log_dir / 'metrics.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            while step < target:
                horizon = min(config.rollout_steps, math.ceil((target - step) / config.num_envs))
                obs_buf, next_buf, act_buf, logp_buf, val_buf = [], [], [], [], []
                rew_buf, term_buf, trunc_buf = [], [], []
                completed = []
                model.eval()
                for _ in range(horizon):
                    obs = np.stack([match.obs for match in matches])
                    with torch.no_grad():
                        dist, value = model(torch.as_tensor(obs, device=device))
                        action = dist.sample()
                        logp = dist.log_prob(action)
                    action_np = action.cpu().numpy()
                    results = [match.step(int(a)) for match, a in zip(matches, action_np)]
                    obs_buf.append(obs)
                    next_buf.append(np.stack([r[1] for r in results]))
                    act_buf.append(action_np)
                    logp_buf.append(logp.cpu().numpy())
                    val_buf.append(value.cpu().numpy())
                    rew_buf.append([r[2] for r in results])
                    term_buf.append([r[3] for r in results])
                    trunc_buf.append([r[4] for r in results])
                    completed.extend(r[5] for r in results if r[5] is not None)
                obs_buf, next_buf = np.asarray(obs_buf), np.asarray(next_buf)
                with torch.no_grad():
                    next_values = model.critic(torch.as_tensor(next_buf.reshape(-1, 92), device=device))
                    next_values = next_values.squeeze(-1).cpu().numpy().reshape(horizon, config.num_envs)
                adv, returns = advantages(np.asarray(rew_buf, dtype=np.float32), np.asarray(val_buf),
                    next_values, np.asarray(term_buf, dtype=np.float32), np.asarray(trunc_buf, dtype=np.float32),
                    config.gamma, config.gae_lambda)
                batch = [obs_buf.reshape(-1, 92), np.asarray(act_buf).reshape(-1),
                         np.asarray(logp_buf).reshape(-1), adv.reshape(-1), returns.reshape(-1)]
                metrics = update(model, optimizer, batch, config, device)
                step += horizon * config.num_envs
                recent.extend(completed)
                if step >= next_checkpoint or step >= target:
                    save_checkpoint(checkpoint_dir / f'baseline_step_{step}.pt', model, optimizer, asdict(config), step)
                    checkpoint_number += 1
                    next_checkpoint = step + config.checkpoint_interval
                # Empty fields mean no completed episode, rather than inventing a zero.
                row = dict(training_step=step,
                    episode_reward=completed[-1]['episode_reward'] if completed else '',
                    average_reward=float(np.mean([e['episode_reward'] for e in recent])) if recent else '',
                    episode_length=float(np.mean([e['episode_length'] for e in recent])) if recent else '',
                    completed_episodes=len(completed), touches=sum(e['touches'] for e in completed),
                    checkpoint_number=checkpoint_number,
                    steps_per_second=(step-start_step)/(time.perf_counter()-start),
                    gpu_peak_mib=torch.cuda.max_memory_allocated()/2**20, **metrics)
                writer.writerow(row)
                file.flush()
                if step == start_step + horizon * config.num_envs or step % 8192 == 0 or step >= target:
                    print(f"step={step} reward100={row['average_reward']} policy={metrics['policy_loss']:.4f} "
                          f"value={metrics['value_loss']:.4f} entropy={metrics['entropy']:.3f} "
                          f"samples/s={row['steps_per_second']:.0f}", flush=True)
    finally:
        for match in matches:
            match.close()
    print(f'Checkpoint: {checkpoint_dir / f"baseline_step_{step}.pt"}', flush=True)


if __name__ == '__main__':
    main()
