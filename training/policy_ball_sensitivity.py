"""Inference-only ball-position intervention for the frozen 1M actor.

No simulator is instantiated or advanced, and no optimizer is constructed.
Synthetic sweeps use the exact training DefaultObs builder. Live sweeps retain
all recorded slots except 0..2: DefaultObs stores ball position only there.
Rocket League's driver-right is PhysicsObject.right (its second matrix column).
"""
import hashlib
import inspect
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from checkpoint import load_checkpoint
from environment import ACTION_TABLE
from observations import OBS_CONFIG, obs_builder, preprocess
from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league.common_values import BOOST_LOCATIONS


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'training/checkpoints/guide_early_1m_v1/baseline_step_1000000.pt'
OUTPUT = ROOT / 'training/logs/guide_early_1m_v1/policy_sensitivity'
CONTROLS = ['throttle', 'steer', 'pitch', 'yaw', 'roll', 'jump', 'boost', 'handbrake']
DISTANCES = {'close': 250.0, 'medium': 1000.0, 'far': 3500.0}
ANGLES = {'ahead': 0, 'ahead_left': -45, 'ahead_right': 45,
          'left': -90, 'right': 90, 'behind': 180}


def physics(position, yaw=0.0, velocity=(0, 0, 0)):
    p = PhysicsObject()
    p.position = np.asarray(position, dtype=np.float32)
    p.linear_velocity = np.asarray(velocity, dtype=np.float32)
    p.angular_velocity = np.zeros(3, dtype=np.float32)
    p.euler_angles = np.asarray([0, yaw, 0], dtype=np.float32)
    return p


def car(team, position, yaw, speed=0.0):
    c = Car()
    c.team_num = team
    c.physics = physics(position, yaw)
    c.physics.linear_velocity = (speed * c.physics.forward).astype(np.float32)
    c.on_ground = True
    c.boost_amount = 33.0
    c.demo_respawn_timer = 0.0
    c.supersonic_time = 0.0
    c.boost_active_time = 0.0
    c.handbrake = 0.0
    c.is_holding_jump = False
    c.has_jumped = False
    c.is_jumping = False
    c.has_flipped = False
    c.has_double_jumped = False
    c.flip_time = 0.0
    c.air_time_since_jump = 0.0
    return c


def synthetic_template(name, team=0, yaw=None, speed=0.0):
    sign = 1 if team == 0 else -1
    yaw = sign * np.pi / 2 if yaw is None else yaw
    own = car(team, (0, -sign * 1000, 17), yaw, speed)
    opponent = car(1 - team, (0, sign * 3500, 17), -sign * np.pi / 2)
    state = GameState()
    state.tick_count = 0
    state.goal_scored = False
    state.cars = {'blue-0': own if team == 0 else opponent,
                  'orange-0': opponent if team == 0 else own}
    state.ball = physics((0, 0, 93))
    state.boost_pad_timers = np.zeros(len(BOOST_LOCATIONS), dtype=np.float32)
    agent = 'blue-0' if team == 0 else 'orange-0'
    builder = obs_builder()
    builder.reset([agent], state, {})

    def build(position):
        # A fresh PhysicsObject clears the orange inverted-ball cache.
        state.ball = physics(position)
        state._inverted_ball = None
        return preprocess(builder.build_obs([agent], state, {})[agent])

    return dict(name=name, team=team, position=own.physics.position,
                forward=own.physics.forward, right=own.physics.right,
                build=build, source='training DefaultObs built from fixed GameState')


def scan_live():
    summaries = []
    exemplars = {}
    sessions = {}
    for path in sorted((ROOT / 'python-example/logs/observation_diagnostics').glob('live_observations_*.jsonl')):
        events, phases, actions, errors = Counter(), Counter(), Counter(), Counter()
        count = 0
        start = end = None
        for line in path.open(encoding='utf-8'):
            record = json.loads(line)
            count += 1
            event = record.get('event', 'legacy_policy_decision')
            events[event] += 1
            phase = record.get('match', {}).get('phase')
            phases[str(phase)] += 1
            time = record.get('time_seconds')
            start = time if start is None else start
            end = time
            if event == 'policy_error':
                errors[record.get('error_type', '') + ': ' + record.get('error', '')] += 1
            if event != 'policy_decision':
                continue
            action = record['model_action']['index']
            actions[action] += 1
            session = record.get('recording', {}).get('session', path.stem)
            totals = sessions.setdefault(session, dict(events=Counter(), phases=Counter(), actions=Counter(),
                                                       jump=0, boost=0, ground=0, count=0))
            totals['count'] += 1
            totals['phases'][phase] += 1
            totals['actions'][action] += 1
            totals['jump'] += bool(record['model_action']['controls']['jump'])
            totals['boost'] += bool(record['model_action']['controls']['boost'])
            totals['ground'] += record['raw_rlbot']['air_state'] == 0
            if phase != 3:
                continue
            raw = record['raw_rlbot']
            pos = raw['car_position']
            if raw['air_state'] == 0 and pos[2] < 100 and abs(pos[0]) < 3600 and abs(pos[1]) < 3500:
                category = 'ground_action_' + str(action) if action in (51, 12) else None
            else:
                category = 'airborne_action_' + str(action) if raw['air_state'] != 0 and action == 41 else None
            if category:
                # Prefer later live sessions over earlier recording attempts.
                key = (session, category)
                exemplars.setdefault(key, (path.name, record))
        summaries.append(dict(file=path.name, records=count, first_time=start, last_time=end,
                              events=dict(events), phases=dict(phases), actions=dict(actions), errors=dict(errors)))
    live_templates = []
    if exemplars:
        latest_session = sorted({key[0] for key in exemplars})[-1]
        for (session, category), (filename, record) in exemplars.items():
            if session != latest_session:
                continue
            values = np.asarray([item['value'] for item in record['observation']], dtype=np.float32)
            team = record['team']
            inversion = np.asarray([-1, -1, 1] if team == 1 else [1, 1, 1], dtype=np.float32)

            def build(position, base=values.copy(), inv=inversion):
                obs = base.copy()
                obs[:3] = np.asarray(position, dtype=np.float32) * inv * OBS_CONFIG['pos_coef']
                return preprocess(obs)

            # Use the recorded 92D forward/up vectors, not the old right label.
            forward_world = values[55:58] * inversion
            up_world = values[58:61] * inversion
            right_world = np.cross(up_world, forward_world)
            live_templates.append(dict(name='live_' + category, team=team,
                                       position=np.asarray(record['raw_rlbot']['car_position'], dtype=np.float32),
                                       forward=forward_world, right=right_world, build=build,
                                       source=f'{filename}: frame {record["frame"]}',
                                       original_action=record['model_action']['index'],
                                       original_observation=values.tolist()))
    clean_sessions = {}
    for name, totals in sessions.items():
        count = totals['count']
        clean_sessions[name] = dict(decisions=count, phases=dict(totals['phases']), actions=dict(totals['actions']),
                                    jump_fraction=totals['jump'] / count,
                                    boost_fraction=totals['boost'] / count,
                                    grounded_fraction=totals['ground'] / count)
    return summaries, clean_sessions, live_templates


def evaluate_sweep(model, template):
    f = np.asarray(template['forward'][:2], dtype=np.float64)
    f /= np.linalg.norm(f)
    # Ground-plane direction tests: make tilted live states comparable while
    # keeping their actual pitch/roll and all observation slots fixed.
    r = np.asarray([-f[1], f[0]])
    cases = []
    reference = None
    for distance_name, distance in DISTANCES.items():
        for direction, degrees in ANGLES.items():
            angle = np.deg2rad(degrees)
            offset = distance * (np.cos(angle) * f + np.sin(angle) * r)
            position = template['position'].copy()
            position[:2] += offset
            position[2] = 93.0
            observation = template['build'](position)
            if reference is None:
                reference = observation.copy()
            if not np.array_equal(observation[3:], reference[3:]):
                raise AssertionError('A non-ball-position observation feature changed')
            with torch.no_grad():
                logits = model.actor(torch.from_numpy(observation).unsqueeze(0))[0]
                probabilities = torch.softmax(logits, dim=0)
            if logits.shape != (90,) or not torch.isfinite(logits).all():
                raise AssertionError('Expected 90 finite logits')
            action = int(logits.argmax().item())
            controls = dict(zip(CONTROLS, ACTION_TABLE[action].tolist()))
            cases.append(dict(distance=distance_name, horizontal_distance=distance, direction=direction,
                              ball_world_position=position.tolist(), observation=observation.tolist(),
                              logits=logits.tolist(), probabilities=probabilities.tolist(), action_index=action,
                              controls=controls, expected_steer=float(probabilities.double().numpy() @ ACTION_TABLE[:, 1])))
    changes = []
    for distance_name in DISTANCES:
        lookup = {case['direction']: case for case in cases if case['distance'] == distance_name}
        for left, right in [('ahead_left', 'ahead_right'), ('left', 'right')]:
            a, b = lookup[left], lookup[right]
            p, q = np.asarray(a['probabilities']), np.asarray(b['probabilities'])
            changes.append(dict(distance=distance_name, pair=[left, right],
                                action_changed=a['action_index'] != b['action_index'],
                                argmax_steer_changed=a['controls']['steer'] != b['controls']['steer'],
                                total_variation=float(np.abs(p - q).sum() / 2),
                                expected_steer_left=a['expected_steer'], expected_steer_right=b['expected_steer']))
    return dict(template=template['name'], source=template['source'], team=template['team'],
                car_world_position=template['position'].tolist(),
                fixed_non_ball_observation=reference[3:].tolist(),
                changed_indices_only=[0, 1, 2], cases=cases, left_right_pairs=changes,
                unique_argmax_actions=sorted({case['action_index'] for case in cases}),
                unique_argmax_steers=sorted({case['controls']['steer'] for case in cases}))


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    before_hash = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    model, checkpoint = load_checkpoint(MODEL, device='cpu')
    files, sessions, live_templates = scan_live()
    templates = [synthetic_template('ground_blue_facing_goal'),
                 synthetic_template('ground_orange_mirror', team=1),
                 synthetic_template('ground_blue_yaw_zero', yaw=0.0),
                 synthetic_template('ground_blue_forward_speed_500', speed=500.0)] + live_templates
    sweeps = [evaluate_sweep(model, template) for template in templates]
    if hashlib.sha256(MODEL.read_bytes()).hexdigest() != before_hash:
        raise AssertionError('Checkpoint bytes changed')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = next(OUTPUT / f'ball_sensitivity_{number:03d}.json'
                for number in range(1, 10000) if not (OUTPUT / f'ball_sensitivity_{number:03d}.json').exists())
    report = dict(checkpoint=str(MODEL), checkpoint_sha256=before_hash, checkpoint_step=checkpoint['step'],
                  versions={name: version(name) for name in ('torch', 'numpy', 'rlgym-rocket-league')},
                  obs_config=OBS_CONFIG, action_table=ACTION_TABLE.tolist(), control_order=CONTROLS,
                  method='Only observation indices 0..2 vary; no simulator stepping, training, or action assistance',
                  direction_convention='Rocket League driver-right: second rotation matrix column; yaw increases clockwise',
                  old_live_label_caveat='Prior live derived right field was incorrectly negated; not used here',
                  builder_source_sha256=hashlib.sha256(inspect.getsource(type(obs_builder())).encode()).hexdigest(),
                  live_recording_summaries=files, live_sessions=sessions, sweeps=sweeps)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(f'Report saved: {path}')
    print('Actions by distance: ahead, ahead-left, ahead-right, left, right, behind')
    for sweep in sweeps:
        print(sweep['template'])
        for distance in DISTANCES:
            cases = [case for case in sweep['cases'] if case['distance'] == distance]
            print(f'  {distance:6} actions={[case["action_index"] for case in cases]} '
                  f'steers={[case["controls"]["steer"] for case in cases]}')
    print('Live sessions:', json.dumps(sessions))


if __name__ == '__main__':
    main()
