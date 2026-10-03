"""Inspect how frozen policies contact the ball; never changes training state.

One row is one existing eight-tick action interval. RocketSim's touch callback
provides a separate, finer snapshot at each collision. A callback snapshot is
*at* contact, not a claim about the preceding physics tick. RLBot-like action
delay remains enabled, so an interval's selected action may not yet be applied
at the instant of a callback in its first physics tick.
"""
import argparse
import json
import math
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from checkpoint import load_checkpoint
from environment import ACTION_REPEAT, ACTION_TABLE, Match, contract, scripted_action
from observations import preprocess


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / 'logs' / 'guide_early_1m_v1' / 'contact_diagnostics'
CHECKPOINTS = ROOT / 'checkpoints' / 'guide_early_1m_v1'
CONTROL_NAMES = contract()['control_order']
TICKS_PER_SECOND = contract()['physics_ticks_per_second']
INTERVAL_SECONDS = ACTION_REPEAT / TICKS_PER_SECOND
FIXTURES = (('bridge', 7000, 10.0), ('bridge', 8000, 10.0),
            ('touch', 3000, 8.0), ('approach', 3000, 10.0))


def numbered_path(path):
    candidate = path
    number = 2
    while candidate.exists():
        candidate = path.with_name(f'{path.stem}_{number}{path.suffix}')
        number += 1
    return candidate


def vector(values):
    return [float(value) for value in values]


def snapshot(state, agent):
    car = state.cars[agent]
    car_pos = car.physics.position
    car_vel = car.physics.linear_velocity
    ball_pos = state.ball.position
    ball_vel = state.ball.linear_velocity
    displacement = ball_pos - car_pos
    distance = float(np.linalg.norm(displacement))
    relative_velocity = car_vel - ball_vel
    return dict(car_position=vector(car_pos), car_velocity=vector(car_vel),
                car_z=float(car_pos[2]), wheels_with_contact=[bool(x) for x in car.wheels_with_contact],
                on_ground=bool(car.on_ground), ball_position=vector(ball_pos),
                ball_velocity=vector(ball_vel), distance_to_ball=distance,
                horizontal_distance_to_ball=float(np.linalg.norm(displacement[:2])),
                closing_velocity=float(np.dot(relative_velocity, displacement) / max(distance, 1e-9)))


def contact_snapshot(rocket_car, arena):
    """Read RocketSim's state inside its ball-touch callback, not after 8 ticks."""
    car = rocket_car.get_state()
    ball = arena.ball.get_state()
    wheels = tuple(bool(value) for value in car.wheels_with_contact)
    car_pos = car.pos.as_numpy()
    car_vel = car.vel.as_numpy()
    ball_pos = ball.pos.as_numpy()
    ball_vel = ball.vel.as_numpy()
    displacement = ball_pos - car_pos
    distance = float(np.linalg.norm(displacement))
    if len(wheels) == 4:
        # Same definition as installed RLGym Car.on_ground: >=3 wheel contacts.
        classification = 'ground' if sum(wheels) >= 3 else 'airborne'
    else:
        classification = 'unknown'
    return dict(classification=classification, car_position=vector(car_pos),
                car_velocity=vector(car_vel), car_z=float(car_pos[2]),
                wheels_with_contact=list(wheels), ball_position=vector(ball_pos),
                ball_velocity=vector(ball_vel), distance_to_ball=distance,
                horizontal_distance_to_ball=float(np.linalg.norm(displacement[:2])),
                closing_velocity=float(np.dot(car_vel - ball_vel, displacement) / max(distance, 1e-9)),
                has_jumped=bool(car.has_jumped),
                jump_control_at_callback=bool(car.last_controls.jump))


def mean_or_none(values):
    return float(np.mean(values)) if values else None


def episode_report(index, rows, contacts):
    first = contacts[0] if contacts else None
    first_row = first['interval'] - 1 if first else None
    first_time = first['interval_end_seconds'] if first else None
    previous_second = ([row['before']['distance_to_ball'] for row in rows
                        if row['interval'] <= first['interval'] and
                        row['time_seconds'] > first_time - 1.0]
                       if first else [])
    # Boundary-sampled estimate; excludes the touch interval because its
    # within-interval airborne duration is unknown without per-tick recording.
    airborne_before = (sum(not row['before']['on_ground'] for row in rows[:first_row]) *
                       INTERVAL_SECONDS if first else None)
    counts = {name: sum(c['classification'] == name for c in contacts)
              for name in ('ground', 'airborne', 'unknown')}
    classified = counts['ground'] + counts['airborne']
    actions = [row['action'] for row in rows]
    window = (rows[max(0, first_row - 10): min(len(rows), first_row + 6)]
              if first else [])
    sequence = [dict(relative_interval=row['interval'] - first['interval'],
                     interval=row['interval'], time_seconds=row['time_seconds'],
                     action_id=row['action_id'], action=row['action'],
                     before_on_ground=row['before']['on_ground'],
                     after_on_ground=row['after']['on_ground'],
                     before_distance_to_ball=row['before']['distance_to_ball'],
                     contact_count=row['contact_count'],
                     contact_classes=row['contact_classes']) for row in window]
    return dict(episode=index, duration_seconds=len(rows) * INTERVAL_SECONDS,
                total_touches=len(contacts), touch_intervals=sum(row['contact_count'] > 0 for row in rows),
                first_touch_time_seconds=first_time,
                first_touch_tick_time_seconds=first['tick_time_seconds'] if first else None,
                first_touch_classification=first['classification'] if first else None,
                ground_contacts=counts['ground'], airborne_contacts=counts['airborne'],
                unknown_contacts=counts['unknown'],
                fraction_touches_airborne=(counts['airborne'] / len(contacts) if contacts else None),
                fraction_classified_touches_airborne=(counts['airborne'] / classified
                                                      if classified else None),
                mean_distance_to_ball_final_second_before_first_touch=mean_or_none(previous_second),
                jump_fraction=mean_or_none([float(a['jump'] > 0) for a in actions]),
                throttle_fraction=mean_or_none([float(a['throttle'] > 0) for a in actions]),
                boost_fraction=mean_or_none([float(a['boost'] > 0) for a in actions]),
                mean_pitch_magnitude=mean_or_none([abs(a['pitch']) for a in actions]),
                airborne_boundary_seconds_before_first_touch_estimate=airborne_before,
                first_touch_action_sequence=sequence)


def summarize(episodes):
    count = len(episodes)
    contacts = sum(e['total_touches'] for e in episodes)
    class_counts = {name: sum(e[f'{name}_contacts'] for e in episodes)
                    for name in ('ground', 'airborne', 'unknown')}
    classified = class_counts['ground'] + class_counts['airborne']
    touched = [e for e in episodes if e['total_touches']]
    def average(field):
        return mean_or_none([e[field] for e in episodes])
    return dict(episodes=count, touch_episode_rate=len(touched) / count,
                mean_touches=average('touch_intervals'),
                mean_raw_contacts=contacts / count,
                ground_touch_episode_rate=sum(e['ground_contacts'] > 0 for e in episodes) / count,
                airborne_touch_episode_rate=sum(e['airborne_contacts'] > 0 for e in episodes) / count,
                first_touch_ground_episode_rate=sum(e['first_touch_classification'] == 'ground'
                                                    for e in episodes) / count,
                first_touch_airborne_episode_rate=sum(e['first_touch_classification'] == 'airborne'
                                                      for e in episodes) / count,
                fraction_touches_airborne=(class_counts['airborne'] / contacts if contacts else None),
                fraction_classified_touches_airborne=(class_counts['airborne'] / classified
                                                      if classified else None),
                contacts_by_class=class_counts,
                jump_fraction=average('jump_fraction'),
                throttle_fraction=average('throttle_fraction'),
                boost_fraction=average('boost_fraction'),
                mean_pitch_magnitude=average('mean_pitch_magnitude'),
                mean_first_touch_time_seconds=mean_or_none(
                    [e['first_touch_time_seconds'] for e in touched]),
                mean_distance_to_ball_final_second_before_first_touch=mean_or_none(
                    [e['mean_distance_to_ball_final_second_before_first_touch'] for e in touched]),
                mean_airborne_boundary_seconds_before_first_touch_estimate=mean_or_none(
                    [e['airborne_boundary_seconds_before_first_touch_estimate'] for e in touched]))


def trace(model, data, scenario, seed, seconds, mode, action_seed, games):
    config = data['config']
    keys = ('approach_weight', 'goal_weight', 'touch_weight', 'face_weight',
            'air_weight', 'approach_positive_only', 'approach_until_first_touch', 'reward_recipe')
    weights = {key: config[key] for key in keys if key in config}
    match = Match(seed, seconds, scenario, config.get('opponent_version', 'v1'), **weights)
    engine = match.env.transition_engine
    original_callback = engine._ball_touch_callback
    contacts_this_interval = []
    learner_id = match.agent
    interval = 0
    interval_start_tick = 0
    last_jump_interval = None

    def on_ball_touch(arena, car, data):
        # Preserve RLGym's own touch accounting and reward behavior.
        original_callback(arena, car, data)
        if engine._agent_ids.get(car.id) != learner_id:
            return
        event = contact_snapshot(car, arena)
        tick = max(1, min(ACTION_REPEAT, engine._tick_count - interval_start_tick + 1))
        event.update(interval=interval, tick_within_interval=tick,
                     tick_time_seconds=((interval - 1) * ACTION_REPEAT + tick) / TICKS_PER_SECOND,
                     interval_end_seconds=interval * INTERVAL_SECONDS,
                     previous_jump_intervals=(None if last_jump_interval is None else
                                              interval - last_jump_interval),
                     selected_jump_in_previous_10_intervals=(last_jump_interval is not None and
                                                             interval - last_jump_interval <= 10))
        contacts_this_interval.append(event)

    engine._arena.set_ball_touch_callback(on_ball_touch)
    idle_action = int(np.flatnonzero(np.all(ACTION_TABLE == 0, axis=1))[0])
    episodes = []
    all_rows = []
    all_contacts = []
    try:
        for episode in range(games):
            generator = torch.Generator(device='cpu').manual_seed(action_seed + episode)
            rows = []
            episode_contacts = []
            learner_id = match.agent
            last_jump_interval = None
            while True:
                agent, opponent = match.agent, match.opponent
                before = snapshot(match.env.state, agent)
                with torch.inference_mode():
                    distribution, _ = model(torch.from_numpy(match.obs).unsqueeze(0))
                    action_id = (int(distribution.logits.argmax(-1).item()) if mode == 'argmax' else
                                 int(torch.multinomial(distribution.probs.squeeze(0), 1,
                                                       generator=generator).item()))
                action = {name: float(value) for name, value in
                          zip(CONTROL_NAMES, ACTION_TABLE[action_id])}
                interval = len(rows) + 1
                interval_start_tick = engine._tick_count
                contacts_this_interval.clear()
                if action['jump'] > 0:
                    last_jump_interval = interval
                opponent_action = (scripted_action(match.env.state, opponent,
                                   config.get('opponent_version', 'v1'))
                                   if scenario == 'kickoff' else idle_action)
                observations, _, terminated, truncated = match.env.step({
                    agent: np.array([action_id], dtype=np.int64),
                    opponent: np.array([opponent_action], dtype=np.int64)})
                after = snapshot(match.env.state, agent)
                reported_touches = int(match.env.state.cars[agent].ball_touches)
                if reported_touches != len(contacts_this_interval):
                    raise AssertionError('RocketSim callback and RLGym touch count disagree')
                events = [event.copy() for event in contacts_this_interval]
                row = dict(episode=episode, interval=interval,
                           time_seconds=interval * INTERVAL_SECONDS,
                           action_id=action_id, action=action,
                           before=before, after=after,
                           contact_count=reported_touches,
                           contact_classes=[event['classification'] for event in events],
                           jump_used_in_previous_10_intervals=(last_jump_interval is not None and
                                                                interval - last_jump_interval <= 10),
                           terminated=bool(terminated[agent]), truncated=bool(truncated[agent]))
                rows.append(row)
                episode_contacts.extend(events)
                if terminated[agent] or truncated[agent]:
                    break
                match.obs = preprocess(observations[agent])
            episodes.append(episode_report(episode, rows, episode_contacts))
            all_rows.extend(rows)
            all_contacts.extend(dict(episode=episode, **event) for event in episode_contacts)
            if episode + 1 < games:
                match.reset()
    finally:
        engine._arena.set_ball_touch_callback(original_callback)
        match.close()
    return dict(format_version=1,
                fixture=dict(scenario=scenario, seed=seed, seconds=seconds, games=games,
                             mode=mode, action_seed=action_seed),
                checkpoint_step=data['step'], checkpoint_config=config,
                contract=contract(),
                versions={package: version(package) for package in
                          ('torch', 'rlgym', 'rlgym-api', 'rlgym-rocket-league',
                           'rocketsim', 'numpy')},
                classification_method=('RocketSim ball-touch callback wheel-contact flags; '
                                       'ground means at least three wheels contact at callback; '
                                       'this is contact-time state, not the previous physics tick'),
                time_method=('First-touch time in summary uses containing action interval end, '
                             'matching evaluate.py; tick_time_seconds is a callback-time estimate; '
                             'airborne duration uses action-boundary samples before that interval'),
                action_fraction_method=('Throttle fraction means throttle > 0 (forward input); '
                                        'jump and boost fractions mean button pressed during an '
                                        'action interval, not that propulsion or jump succeeded'),
                summary=summarize(episodes), episodes=episodes,
                contacts=all_contacts, intervals=all_rows)


def print_table(reports):
    print('fixture / mode / step | touch% | mean touches | ground% | air% | air contacts% | jump% | throttle% | boost% | first touch s')
    for report in reports:
        f = report['fixture']
        s = report['summary']
        def pct(value):
            return '-' if value is None else f'{value * 100:.1f}'
        first = '-' if s['mean_first_touch_time_seconds'] is None else f"{s['mean_first_touch_time_seconds']:.2f}"
        print(f"{f['scenario']}/{f['seed']} {f['mode']} {report['checkpoint_step']:>7} | "
              f"{pct(s['touch_episode_rate']):>6} | {s['mean_touches']:>12.3f} | "
              f"{pct(s['ground_touch_episode_rate']):>7} | "
              f"{pct(s['airborne_touch_episode_rate']):>4} | "
              f"{pct(s['fraction_touches_airborne']):>13} | "
              f"{pct(s['jump_fraction']):>5} | {pct(s['throttle_fraction']):>9} | "
              f"{pct(s['boost_fraction']):>6} | {first:>13}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='Run the fixed 16-case comparison')
    parser.add_argument('--scenario', choices=('bridge', 'touch', 'approach'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--mode', choices=('argmax', 'sample'))
    parser.add_argument('--step', type=int, choices=(0, 1000000))
    parser.add_argument('--games', type=int, default=40)
    parser.add_argument('--action-seed', type=int, default=3000)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.games < 1 or args.action_seed < 0:
        parser.error('games must be positive and action-seed non-negative')
    if args.all:
        if any(value is not None for value in
               (args.scenario, args.seed, args.seconds, args.mode, args.step)):
            parser.error('--all cannot be combined with individual fixture options')
        configurations = [(scenario, seed, seconds, mode, step)
                          for scenario, seed, seconds in FIXTURES
                          for mode in ('argmax', 'sample') for step in (0, 1000000)]
    else:
        if any(value is None for value in (args.scenario, args.seed, args.seconds, args.mode, args.step)):
            parser.error('Specify --all or scenario, seed, seconds, mode and step')
        if args.seed < 0 or not math.isfinite(args.seconds) or args.seconds <= 0:
            parser.error('seed and seconds must be valid and non-negative')
        configurations = [(args.scenario, args.seed, args.seconds, args.mode, args.step)]
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    reports = []
    paths = []
    models = {}
    for scenario, seed, seconds, mode, step in configurations:
        checkpoint = CHECKPOINTS / f'baseline_step_{step}.pt'
        if step not in models:
            models[step] = load_checkpoint(checkpoint)
        model, data = models[step]
        torch.manual_seed(seed)
        report = trace(model, data, scenario, seed, seconds, mode, args.action_seed, args.games)
        report['checkpoint'] = str(checkpoint.resolve())
        requested = args.output_dir / f'contact_{scenario}_seed{seed}_{mode}_step{step}.json'
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = numbered_path(requested)
        with path.open('x', encoding='utf-8') as file:
            # Full per-interval traces are large; compact encoding remains plain JSON.
            json.dump(report, file, separators=(',', ':'))
            file.write('\n')
        # Retain only the compact summary between configurations, not 16 full
        # trajectory logs in RAM at once.
        reports.append({key: report[key] for key in ('fixture', 'summary', 'checkpoint_step')})
        paths.append(path)
        print(f'Saved: {path}', flush=True)
    print_table(reports)
    print('Files created:')
    for path in paths:
        print(path)


if __name__ == '__main__':
    main()
