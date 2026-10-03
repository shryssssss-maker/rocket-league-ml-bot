"""Trace actual reward, controls, and motion for a frozen checkpoint.

This is an evaluation-only tool. It does not update weights or change PPO's
environment. One row represents one 8-physics-tick action interval.
"""
import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from rlgym.api import RewardFunction

from checkpoint import load_checkpoint
from environment import ACTION_REPEAT, ACTION_TABLE, Match, contract, scripted_action
from observations import preprocess
from rewards.air import InAirReward
from rewards.facing import FacingBallReward
from rewards.guide_face_ball import GuideFaceBallReward
from rewards.speed_toward_ball import SpeedTowardBallReward
from rlgym.rocket_league.reward_functions import TouchReward


CONTROL_NAMES = contract()['control_order']
REWARD_NAMES = {'TouchReward': 'touch', 'ApproachReward': 'approach',
                'FacingBallReward': 'face', 'InAirReward': 'air',
                'GuideFaceBallReward': 'face', 'SpeedTowardBallReward': 'velocity_to_ball',
                'GoalReward': 'goal'}


class RewardTrace(RewardFunction):
    """Same calculation and call order as installed RLGym CombinedReward."""

    def __init__(self, combined):
        self.parts = list(zip(combined.reward_fns, combined.weights))
        self.last = {}

    def reset(self, agents, initial_state, shared_info):
        self.last = {}
        for reward_fn, _ in self.parts:
            reward_fn.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        totals = {agent: 0.0 for agent in agents}
        self.last = {agent: {name: 0.0 for name in (*REWARD_NAMES.values(), 'total')}
                     for agent in agents}
        for reward_fn, weight in self.parts:
            name = REWARD_NAMES[type(reward_fn).__name__]
            values = reward_fn.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
            for agent, raw in values.items():
                weighted = float(raw * weight)
                self.last[agent][name] += weighted
                totals[agent] += weighted
        for agent in agents:
            self.last[agent]['total'] = totals[agent]
        return totals


def motion(state, agent):
    car = state.cars[agent].physics
    ball = state.ball.position
    to_ball = ball[:2] - car.position[:2]
    distance = float(np.linalg.norm(to_ball))
    velocity = car.linear_velocity[:2]
    return dict(position=car.position[:2].copy(), ball_position=ball[:2].copy(),
                speed=float(np.linalg.norm(car.linear_velocity)),
                velocity_toward_ball=float(np.dot(velocity, to_ball / max(distance, 1))),
                distance_to_ball=distance,
                on_ground=bool(state.cars[agent].on_ground))


def trace(model, data, *, episodes, seconds, seed, scenario, mode, action_seed):
    config = data['config']
    reward_keys = ('approach_weight', 'goal_weight', 'touch_weight', 'face_weight',
                   'air_weight', 'approach_positive_only', 'approach_until_first_touch',
                   'reward_recipe')
    weights = {key: config[key] for key in reward_keys if key in config}
    match = Match(seed, seconds, scenario, config.get('opponent_version', 'v1'), **weights)
    # Match has already reset once. Initialize the wrapper at that same state,
    # avoiding a second reset that would change the selected fixture.
    reward_trace = RewardTrace(match.env.reward_fn)
    reward_trace.reset(match.env.agents, match.env.state, match.env.shared_info)
    match.env.reward_fn = reward_trace
    face_reward = (GuideFaceBallReward() if config.get('reward_recipe') == 'guide_early_v1'
                   else FacingBallReward())
    raw_terms = dict(touch=TouchReward(), velocity_to_ball=SpeedTowardBallReward(),
                     face=face_reward, air=InAirReward())
    rows = []
    idle = int(np.flatnonzero(np.all(ACTION_TABLE == 0, axis=1))[0])
    try:
        for episode in range(episodes):
            generator = torch.Generator(device='cpu').manual_seed(action_seed + episode)
            step = 0
            while True:
                agent, opponent = match.agent, match.opponent
                before = motion(match.env.state, agent)
                with torch.inference_mode():
                    distribution, _ = model(torch.from_numpy(match.obs).unsqueeze(0))
                    action = (int(distribution.logits.argmax(-1).item()) if mode == 'argmax' else
                              int(torch.multinomial(distribution.probs.squeeze(0), 1,
                                                    generator=generator).item()))
                controls = {name: float(value) for name, value in
                            zip(CONTROL_NAMES, ACTION_TABLE[action])}
                opponent_action = (scripted_action(match.env.state, opponent,
                                   config.get('opponent_version', 'v1'))
                                   if scenario == 'kickoff' else idle)
                observations, rewards, terminated, truncated = match.env.step({
                    agent: np.array([action], dtype=np.int64),
                    opponent: np.array([opponent_action], dtype=np.int64)})
                after = motion(match.env.state, agent)
                raw_rewards = {name: float(reward_fn.get_rewards(
                    [agent], match.env.state, terminated, truncated,
                    match.env.shared_info)[agent])
                    for name, reward_fn in raw_terms.items()}
                step += 1
                # Distance uses each endpoint's actual ball position. The car-only
                # projection isolates car travel toward the ball's starting point.
                toward = before['ball_position'] - before['position']
                unit = toward / max(float(np.linalg.norm(toward)), 1)
                car_progress = float(np.dot(after['position'] - before['position'], unit))
                row = dict(episode=episode, step=step,
                           time_seconds=step * ACTION_REPEAT / 120,
                           learner=agent, action_id=action, action=controls,
                           reward=reward_trace.last[agent].copy(),
                           raw_reward_terms=raw_rewards,
                           state=dict(speed_before=before['speed'], speed_after=after['speed'],
                                      speed_change=after['speed'] - before['speed'],
                                      velocity_toward_ball_before=before['velocity_toward_ball'],
                                      velocity_toward_ball_after=after['velocity_toward_ball'],
                                      velocity_toward_ball_change=(after['velocity_toward_ball'] -
                                                                   before['velocity_toward_ball']),
                                      distance_to_ball_before=before['distance_to_ball'],
                                      distance_to_ball_after=after['distance_to_ball'],
                                      distance_progress=before['distance_to_ball'] - after['distance_to_ball'],
                                      car_progress_toward_start_ball=car_progress,
                                      on_ground_before=before['on_ground'],
                                      on_ground_after=after['on_ground'],
                                      ball_touches=int(match.env.state.cars[agent].ball_touches)),
                           terminated=bool(terminated[agent]), truncated=bool(truncated[agent]))
                if not np.isclose(row['reward']['total'], rewards[agent], atol=1e-8):
                    raise AssertionError('Reward components do not sum to environment reward')
                rows.append(row)
                if terminated[agent] or truncated[agent]:
                    break
                match.obs = preprocess(observations[agent])
            if episode + 1 < episodes:
                match.reset()
    finally:
        match.close()

    positive = [row for row in rows if row['reward']['approach'] > 0]
    no_propulsion = [row for row in positive if row['action']['throttle'] == 0 and
                     row['action']['boost'] == 0]
    summary = dict(steps=len(rows), episodes=episodes,
                   positive_approach_steps=len(positive),
                   positive_approach_without_throttle_or_boost=len(no_propulsion),
                   no_propulsion_fraction_of_positive_approach=(len(no_propulsion) / len(positive)
                                                               if positive else None),
                   positive_approach_reward_without_throttle_or_boost=sum(
                       row['reward']['approach'] for row in no_propulsion),
                   total_approach_reward=sum(row['reward']['approach'] for row in rows),
                   total_touch_reward=sum(row['reward']['touch'] for row in rows))
    summary['raw_reward_stats'] = {
        name: dict(mean=float(np.mean(values)), p95=float(np.percentile(values, 95)),
                   max=float(np.max(values)),
                   positive_rate=float(np.mean(np.asarray(values) > 0)),
                   mean_when_positive=(float(np.mean([value for value in values if value > 0]))
                                       if any(value > 0 for value in values) else None))
        for name in raw_terms
        for values in [[row['raw_reward_terms'][name] for row in rows]]
    }
    return dict(format_version=1, checkpoint_step=data['step'],
                checkpoint_config=config, contract=contract(),
                versions={package: version(package) for package in
                          ('torch', 'rlgym', 'rlgym-api', 'rlgym-rocket-league',
                           'rocketsim', 'numpy')},
                fixture=dict(scenario=scenario, seconds=seconds, episodes=episodes,
                             seed=seed, mode=mode, action_seed=action_seed),
                summary=summary, steps=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scenario', choices=('touch', 'bridge', 'approach', 'kickoff'),
                        default='bridge')
    parser.add_argument('--mode', choices=('argmax', 'sample'), default='argmax')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--seed', type=int, default=5000)
    parser.add_argument('--action-seed', type=int, default=3000)
    args = parser.parse_args()
    if (args.episodes < 1 or not np.isfinite(args.seconds) or args.seconds <= 0 or
            args.seed < 0 or args.action_seed < 0):
        parser.error('Episodes and seconds must be positive; seeds must be nonnegative')
    if args.output.exists():
        parser.error('Refusing to overwrite an existing diagnostic report')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    model, data = load_checkpoint(args.model)
    result = trace(model, data, episodes=args.episodes, seconds=args.seconds,
                   seed=args.seed, scenario=args.scenario, mode=args.mode,
                   action_seed=args.action_seed)
    result['checkpoint'] = str(args.model.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    print(json.dumps(result['summary'], indent=2))
    print(f'Report saved: {args.output}')


if __name__ == '__main__':
    main()
