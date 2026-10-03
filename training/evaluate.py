"""Fixed-seed, alternating-side sudden-death matches against a scripted opponent."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
from checkpoint import load_checkpoint
from environment import Match, contract


def report_path(requested):
    """Preserve earlier reports and select a numbered filename before simulation."""
    candidate = requested
    number = 2
    while candidate.exists():
        candidate = requested.with_name(f'{requested.stem}_{number}{requested.suffix}')
        number += 1
    return candidate


def evaluate(model, games=20, seconds=30.0, seed=1000, mode='argmax',
             action_seed=3000, scenario='kickoff', opponent_version='v1',
             reward_weights=None):
    match = Match(seed, seconds, scenario, opponent_version, **(reward_weights or {}))
    metrics = dict(format_version=2, games=games, wins=0, losses=0, draws=0,
                   goals_for=0, goals_against=0)
    episodes = []
    histogram = np.zeros(90, dtype=int)
    active_episode = -1
    try:
        while len(episodes) < games:
            episode_index = len(episodes)
            if episode_index != active_episode:
                # Separate action randomness from the fixed simulator fixtures.
                rng = np.random.default_rng(action_seed + episode_index)
                generator = torch.Generator(device='cpu').manual_seed(action_seed + episode_index)
                active_episode = episode_index
            if mode == 'random':
                action = int(rng.integers(90))
            else:
                with torch.inference_mode():
                    dist, _ = model(torch.from_numpy(match.obs).unsqueeze(0))
                    action = (int(dist.logits.argmax(-1).item()) if mode == 'argmax' else
                              int(torch.multinomial(dist.probs.squeeze(0), 1,
                                                    generator=generator).item()))
            histogram[action] += 1
            _, _, _, _, _, info = match.step(action)
            if info is not None:
                episodes.append(info)
                metrics['goals_for'] += info['goals_for']
                metrics['goals_against'] += info['goals_against']
                outcome = 'wins' if info['goals_for'] else 'losses' if info['goals_against'] else 'draws'
                metrics[outcome] += 1
    finally:
        match.close()
    def mean(field):
        return float(np.mean([e[field] for e in episodes]))

    first_touch_times = [e['time_to_first_touch_seconds'] for e in episodes
                         if e['time_to_first_touch_seconds'] is not None]
    metrics.update(win_rate=metrics['wins']/games,
        wins_without_touch=sum(e['goals_for'] for e in episodes if e['touches'] == 0),
        touch_episode_rate=sum(e['touches'] > 0 for e in episodes) / games,
        mean_reward=mean('episode_reward'), mean_touches=mean('touches'),
        mean_raw_contacts=mean('raw_contacts'),
        mean_ground_touch_steps=mean('ground_touch_steps'),
        ground_touch_episode_rate=sum(e['ground_touch_steps'] > 0 for e in episodes) / games,
        mean_goalward_touch_steps=mean('goalward_touch_steps'),
        first_touch_rate=sum(e['first_touch_by'] == 'learner' for e in episodes) / games,
        opponent_first_touch_rate=sum(e['first_touch_by'] == 'opponent' for e in episodes) / games,
        same_step_first_touch_rate=sum(e['first_touch_by'] == 'same_step' for e in episodes) / games,
        mean_time_to_first_learner_touch_seconds=(float(np.mean(first_touch_times))
                                                  if first_touch_times else None),
        mean_ball_progress_toward_goal=mean('ball_progress_toward_goal'),
        opponent_own_goal_proxy=sum(e['opponent_own_goal_proxy'] for e in episodes),
        learner_own_goal_proxy=sum(e['learner_own_goal_proxy'] for e in episodes),
        unknown_goal_attribution=sum(e['unknown_goal_attribution'] for e in episodes),
        mean_episode_steps=mean('episode_length'), mean_episode_seconds=mean('episode_seconds'),
        jump_action_fraction=float(np.mean([e['jump_action_fraction'] for e in episodes])),
        seed=seed, action_seed=action_seed, mode=mode, seconds=seconds, scenario=scenario,
        opponent=(f'scripted_ball_chaser_{opponent_version}' if scenario == 'kickoff' else
                  {'touch': 'idle_touch_probe_v1', 'bridge': 'idle_bridge_v1',
                   'approach': 'idle_approach_v1'}[scenario]),
        action_histogram=histogram.tolist(), episodes=episodes)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--random', action='store_true', help='Random learner, same scripted opponent')
    parser.add_argument('--mode', choices=('argmax', 'sample', 'random'), default='argmax')
    parser.add_argument('--action-seed', type=int, default=3000)
    parser.add_argument('--games', type=int, default=20)
    parser.add_argument('--seconds', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--scenario', choices=('kickoff', 'touch', 'bridge', 'approach'), default='kickoff')
    parser.add_argument('--opponent', choices=('v1', 'v2'), default='v1')
    args = parser.parse_args()
    if args.random and args.mode != 'argmax':
        parser.error('--random is an alias for --mode random; use only one mode')
    mode = 'random' if args.random else args.mode
    if (bool(args.model) == (mode == 'random') or args.games < 1 or
            not np.isfinite(args.seconds) or args.seconds <= 0 or
            args.seed < 0 or args.action_seed < 0):
        parser.error('Use --model for argmax/sample, or --mode random without --model; games and seconds must be positive and seeds non-negative')
    destination = report_path(args.output) if args.output else None
    if destination and destination != args.output:
        print(f'Report already exists; saving this run to {destination}')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    model, data = (None, None) if mode == 'random' else load_checkpoint(args.model)
    reward_weights = {key: data['config'].get(key, default) for key, default in
                      (('approach_weight', 0.0), ('goal_weight', 10.0),
                       ('touch_weight', 0.1), ('face_weight', 0.0), ('air_weight', 0.0),
                       ('approach_positive_only', False),
                       ('approach_until_first_touch', False),
                       ('reward_recipe', 'custom'))} if data else None
    metrics = evaluate(model, args.games, args.seconds, args.seed, mode,
                       args.action_seed, args.scenario, args.opponent, reward_weights)
    if data is not None:
        metrics['checkpoint_step'] = data['step']
        metrics['checkpoint'] = str(args.model.resolve())
        metrics['training_config'] = data['config']
    metrics['reward_config'] = reward_weights or dict(goal_weight=10.0, touch_weight=0.1,
                                                      approach_weight=0.0, face_weight=0.0,
                                                      air_weight=0.0,
                                                      approach_positive_only=False,
                                                      approach_until_first_touch=False,
                                                      reward_recipe='custom')
    metrics['reward_config']['velocity_weight'] = (data['config'].get('velocity_weight', 0.0)
                                                   if data else 0.0)
    metrics['contract'] = contract()
    metrics['versions'] = {package: version(package) for package in
                           ('torch', 'rlgym', 'rlgym-api', 'rlgym-rocket-league', 'rocketsim', 'numpy')}
    output = json.dumps(metrics, indent=2)
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ('action_histogram', 'episodes', 'contract', 'training_config', 'versions')}, indent=2))
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also protects against another process creating it mid-run.
        with destination.open('x', encoding='utf-8') as file:
            file.write(output + '\n')
        print(f'Report saved: {destination}')


if __name__ == '__main__':
    main()
