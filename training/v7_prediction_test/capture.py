"""Live-only RLBot diagnostic capture; no prediction provider/comparison/training."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import struct
import importlib.metadata
import copy
import io
from types import SimpleNamespace
from datetime import datetime

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'training/live_reference_test'))
from launch import verify_sources, INTERPRETER, sha
from rlbot import flat
from rlbot.managers import Bot, MatchManager
sys.path.insert(0, str(ROOT / 'python-example-original/src'))
from util.ball_prediction_analysis import find_slice_at_time


def xyz(v):
    return None if v is None else [float(v.x), float(v.y), float(v.z)]


def physics(p):
    return dict(position=xyz(p.location), velocity=xyz(p.velocity),
                angular_velocity=xyz(p.angular_velocity),
                rotation_pyr=None if p.rotation is None else [float(p.rotation.pitch), float(p.rotation.yaw), float(p.rotation.roll)])


def fixtures():
    # Symmetric signs and perturbed paths; all requested states are recorded.
    seeds = {
        'stationary': ([0, 0, 93], [0, 0, 0]),
        'free_flight': ([0, 0, 1000], [900, 700, 400]),
        'rolling': ([0, 0, 93], [1100, 300, 0]),
        'spin': ([0, 0, 600], [900, 700, 100]),
        'floor': ([0, 0, 500], [600, 700, -800]),
        'wall': ([3300, 0, 500], [1400, 500, 100]),
        'corner': ([3200, 4300, 450], [1300, 1000, 100]),
        'ceiling': ([0, 0, 1750], [700, 500, 1100]),
        'goal_mouth': ([200, 4400, 160], [100, 1300, 50]),
        'car_near_path': ([0, 0, 500], [1100, 200, 0]),
    }
    result = []
    for group, (pos, vel) in seeds.items():
        for j in range(12):
            sx, sy = (-1 if j % 2 else 1), (-1 if (j // 2) % 2 else 1)
            p = [sx * pos[0], sy * pos[1], pos[2] + (j // 4) * 20]
            v = [sx * vel[0], sy * vel[1], vel[2]]
            ball = dict(position=p, velocity=v, angular_velocity=[sx * 3, sy * 2, 4] if group == 'spin' else [0, 0, 0], rotation_pyr=[0, 0, 0])
            car = dict(position=[-sx * 2200, -sy * 2500, 17], velocity=[0, 0, 0], angular_velocity=[0, 0, 0], rotation_pyr=[0, j * math.pi / 6, 0])
            other = dict(position=[sx * 3200, sy * 3000, 17], velocity=[0, 0, 0], angular_velocity=[0, 0, 0], rotation_pyr=[0, 0, 0])
            if group == 'car_near_path':
                other['position'] = [sx * 700, sy * 100, 17]
            result.append(dict(id=f'{group}_{j:02d}', group=group, ball=ball, car=car, other=other))
    return result


def desired(p):
    return flat.DesiredPhysics(location=flat.Vector3Partial(*p['position']),
        velocity=flat.Vector3Partial(*p['velocity']), angular_velocity=flat.Vector3Partial(*p['angular_velocity']),
        rotation=flat.RotatorPartial(*p['rotation_pyr']))


CONTROL_FIELDS = ('throttle', 'steer', 'pitch', 'yaw', 'roll', 'jump', 'boost', 'handbrake')
VERIFICATION_POLICY = dict(
    id='bounded_first_received_packet_proposal_v1', approved=True,
    verified_window=dict(max_frames=5, max_seconds=5/120+1e-4),
    near_window=dict(max_frames=12, max_seconds=.1001),
    ball=dict(position=100., velocity=100., angular_velocity=.25, rotation_pyr=.30),
    car=dict(position=35., velocity=100., angular_velocity=.25, rotation_pyr=.05),
    boost_error=.5, near_tolerance_multiplier=2.,
    norms='Euclidean position/velocity/angular-velocity error; maximum wrapped Euler-component error in radians',
    purpose='Application evidence only; no prediction modification or assumed physics extrapolation')


def controls(c):
    return {name: getattr(c, name) for name in CONTROL_FIELDS}


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def application_evidence(case, actual, request, receipt, T, frame):
    """Proposed bounded diagnostic comparison; never modifies received state."""
    errors = {}
    magnitudes = {}
    for body in ('ball', 'car', 'other'):
        for field in ('position', 'velocity', 'angular_velocity', 'rotation_pyr'):
            expected = [f32(v) for v in case[body][field]]
            got = actual.get(body)
            diff = None if got is None or got[field] is None else [a-b for a,b in zip(got[field], expected)]
            if diff is not None and field == 'rotation_pyr':
                diff = [math.atan2(math.sin(x),math.cos(x)) for x in diff]
            key=f'{body}.{field}'
            errors[key] = diff
            magnitudes[key] = None if diff is None else (max(abs(x) for x in diff) if field=='rotation_pyr' else math.sqrt(sum(x*x for x in diff)))
        if body != 'ball':
            got = actual.get(body)
            errors[f'{body}.boost'] = None if got is None else got.get('boost',float('nan'))
            magnitudes[f'{body}.boost'] = None if got is None else abs(errors[f'{body}.boost'])
    dt, frames = T-request['T'], frame-request['frame']
    received_after_request = receipt is not None and receipt['counter'] > request['packet_receipt_counter'] and receipt['monotonic'] >= request['monotonic'] and frames>0 and dt>=0
    def matches(multiplier, window):
        if not received_after_request or frames>window['max_frames'] or dt>window['max_seconds']:
            return False
        for key,value in magnitudes.items():
            body,field=key.split('.')
            limit=VERIFICATION_POLICY['boost_error'] if field=='boost' else VERIFICATION_POLICY['ball' if body=='ball' else 'car'][field]
            if value is None or not math.isfinite(value) or value>limit*multiplier:
                return False
        return True
    status = 'verified' if matches(1,VERIFICATION_POLICY['verified_window']) else 'near_match' if matches(2,VERIFICATION_POLICY['near_window']) else 'failed'
    return dict(status=status, criterion=VERIFICATION_POLICY['id'], policy_approved=VERIFICATION_POLICY['approved'],
        received_after_request=received_after_request, elapsed_seconds=dt, frame_difference=frames,
        component_errors=errors, request_id=request['id'], packet_receipt=receipt,
        error_magnitudes=magnitudes, coverage_eligible=status=='verified' and VERIFICATION_POLICY['approved'],
        interpretation='Conservative application evidence only; no bounce/goal outcome certification')


class CaptureBot(Bot):
    def __init__(self, *args, **kwargs):
        self.receipt_counter = 0
        self.packet_counter = self.prediction_counter = self.callback_counter = 0
        self.packet_receipt = self.prediction_receipt = None
        self.previous_prediction_counter = None
        self.previous_callback_prediction_counter = None
        self.log = self.event_log = self.error_log = None
        self.startup_events = []
        self.last_emitted = None
        self.emission_counter = 0
        self.halted = False
        super().__init__(*args, **kwargs)
        original_send = self._game_interface.send_msg

        def tracked_send(message):
            result = original_send(message)
            if isinstance(message, flat.PlayerInput):
                self.emission_counter += 1
                self.last_emitted = dict(counter=self.emission_counter, monotonic=time.monotonic(),
                    callback_counter=self.callback_counter, index=message.player_index,
                    controls=controls(message.controller_state), meaning='Successfully submitted to SocketRelay; not proof of game application')
                self.event('diagnostic_control_submission', self.last_emitted)
            return result
        self._game_interface.send_msg = tracked_send

    def event(self, kind, payload):
        row = dict(kind=kind, monotonic=time.monotonic(), payload=payload)
        if self.event_log is None:
            if len(self.startup_events) >= 10000:
                raise RuntimeError('Startup receipt log limit reached')
            self.startup_events.append(row)
        else:
            self.write(self.event_log, row)

    def receipt(self, kind, message):
        self.receipt_counter += 1
        if kind == 'packet':
            self.packet_counter += 1
            counter = self.packet_counter
        else:
            self.prediction_counter += 1
            counter = self.prediction_counter
        value = dict(counter=counter, delivery_order=self.receipt_counter,
                     monotonic=time.monotonic(), python_object_id=id(message),
                     identity_scope='Session-local delivery counter; not a server-side frame association')
        self.event(kind + '_receipt', value)
        return value

    def _handle_packet(self, packet):
        self.packet_receipt = self.receipt('packet', packet)
        super()._handle_packet(packet)

    def _handle_ball_prediction(self, prediction):
        self.prediction_receipt = self.receipt('prediction', prediction)
        super()._handle_ball_prediction(prediction)

    def write(self, stream, row):
        # Fail explicitly rather than silently truncate diagnostic evidence.
        if stream.tell() > 256 * 1024 * 1024:
            raise RuntimeError('Diagnostic file exceeded 256 MiB bounded capture limit')
        stream.write(json.dumps(row, allow_nan=False) + '\n')
        stream.flush()

    def initialize(self):
        self.folder = Path(os.environ['V7_CAPTURE_FOLDER'])
        self.suite = fixtures()
        self.i = 0
        self.pending = None
        self.count = 0
        self.failures = []
        self.fixture_results = []
        self.valid_snapshots = 0
        self.previous_T = None
        self.log = (self.folder / 'captures.jsonl').open('w', encoding='utf-8')
        self.event_log = (self.folder / 'delivery_events.jsonl').open('w', encoding='utf-8')
        self.error_log = (self.folder / 'errors.jsonl').open('w', encoding='utf-8')
        for row in self.startup_events:
            self.write(self.event_log, row)
        self.startup_events.clear()
        self.event('agent_initialized', dict(index=self.index, team=self.team, name=self.name,
            session=self.folder.name, prediction_source='RLBot Bot.ball_prediction at get_output callback',
            association='Latest delivered prediction; packet and prediction messages are asynchronous'))
        (self.folder/'agent_metadata.json').write_text(json.dumps(dict(agent_index=self.index,agent_team=self.team,
            name=self.name,python=sys.version,versions={n:importlib.metadata.version(n) for n in ('rlbot','rlbot-flatbuffers','psutil')},
            session=self.folder.name,verification_policy=VERIFICATION_POLICY),indent=2),encoding='utf-8')
        self.summary('ready')

    def summary(self, status, **extra):
        verified = [r for r in self.fixture_results if r['coverage_eligible'] and r['valid_snapshots'] == 3]
        value = dict(status=status, processed=self.i, total=len(self.suite), captures=self.count,
            valid_snapshots=self.valid_snapshots, fixture_results=self.fixture_results,
            verified_fixture_count=len(verified), verified_groups=sorted({r['group'] for r in verified}),
            failures=self.failures, agent_index=self.index, agent_team=self.team,
            session=self.folder.name, comparison_started=False, **extra)
        temp = self.folder / 'capture_summary.tmp'
        temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
        os.replace(temp, self.folder / 'capture_summary.json')

    def get_output(self, packet):
        if self.halted:
            return flat.ControllerState()
        self.callback_counter += 1
        try:
            self.capture(packet)
        except Exception as error:
            self.halted = True
            self.write(self.error_log, dict(kind='callback_error', callback_counter=self.callback_counter,
                error_type=type(error).__name__, error=repr(error), monotonic=time.monotonic()))
            self.summary('error', error=repr(error))
            raise
        return flat.ControllerState()

    def capture(self, packet):
        T = float(packet.match_info.seconds_elapsed)
        frame = int(packet.match_info.frame_num)
        previous_T, self.previous_T = self.previous_T, T
        if self.i == len(self.suite):
            return
        case = self.suite[self.i]
        actual = dict(ball=physics(packet.balls[0].physics) if packet.balls else None,
            car=physics(packet.players[self.index].physics) if len(packet.players)>self.index else None,
            other=physics(packet.players[1-self.index].physics) if len(packet.players)==2 else None)
        if actual['car'] is not None:
            actual['car']['boost']=float(packet.players[self.index].boost)
        if actual['other'] is not None:
            actual['other']['boost']=float(packet.players[1-self.index].boost)
        distance = None if actual['ball'] is None or actual['car'] is None else math.sqrt(sum(
            (a-b)**2 for a,b in zip(actual['ball']['position'], actual['car']['position'])))
        base = dict(case=case, canonical_T=T, frame=frame, phase=str(packet.match_info.match_phase),
            callback_counter=self.callback_counter, previous_callback_T=previous_T,
            actual=actual, agent_index=self.index, agent_team=self.team,
            packet_receipt=self.packet_receipt, prediction_receipt=self.prediction_receipt,
            prediction_reused_since_previous_callback=self.previous_callback_prediction_counter==self.prediction_counter,
            diagnostic_previous_control_submission=self.last_emitted,
            teacher_memory_observed=False,
            teacher_branch=dict(ball_present=bool(packet.balls), distance_3d=distance,
                far_ball=distance is not None and distance>1500,
                eligible_if_no_pending_sequence=distance is not None and distance>1500,
                pending_teacher_sequence='not_observed',
                actual_teacher_branch_eligibility=None if distance is not None and distance>1500 else False),
            probe_kind='Unconditional diagnostic selector probe; not an executed teacher decision')
        self.previous_callback_prediction_counter=self.prediction_counter
        self.event('callback_state',base)
        if not packet.balls:
            self.write(self.log, dict(**base, snapshot_status='missing_ball', selector_invoked=False,
                injection=None if self.pending is None else self.pending.copy()))
            self.count += 1
        if self.pending is None:
            if not packet.balls or len(packet.players) != 2 or packet.match_info.match_phase not in (flat.MatchPhase.Active, flat.MatchPhase.Kickoff):
                return
            # Installed Bot.set_game_state passes an unsupported fourth argument
            # to this helper. Use the same message path with its actual signature.
            from rlbot.utils import fill_desired_game_state
            self._game_interface.send_msg(fill_desired_game_state(
                balls={0: flat.DesiredBallState(desired(case['ball']))},
                cars={self.index: flat.DesiredCarState(desired(case['car']), 0),
                      1 - self.index: flat.DesiredCarState(desired(case['other']), 0)}))
            self.pending = dict(id=case['id'], frame=frame, T=T, monotonic=time.monotonic(), captured=0,
                valid_snapshots=0, status='unverified', evidence=None,
                packet_receipt_counter=self.packet_counter,
                prediction_receipt_counter=self.prediction_counter)
            self.event('fixture_request', dict(case=case, request=self.pending.copy(), actual_before_request=actual))
            return
        pending = self.pending
        # Decide exactly once from the first post-request delivered callback
        # with ball and both cars. Never search later callbacks for a nicer match.
        if pending['evidence'] is None and self.packet_receipt is not None and self.packet_receipt['counter']>pending['packet_receipt_counter'] and all(actual[k] is not None for k in ('ball','car','other')):
            evidence = application_evidence(case, actual, pending, self.packet_receipt,T,frame)
            pending['evidence'] = evidence
            pending['status']=evidence['status']
            self.event('fixture_application_evidence', dict(case=case['id'], evidence=evidence,
                actual=actual, canonical_T=T, frame=frame))
            if evidence['status']=='failed':
                self.failures.append(dict(case=case['id'],reason='fixture_application_failed',evidence=evidence))
        evidence=pending['evidence']
        age = T - pending['T']
        if time.monotonic() - pending['monotonic'] > 10:
            self.failures.append(dict(case=case['id'], reason='capture timeout', phase=str(packet.match_info.match_phase)))
            pending['status'] = 'failed'
            self.finish_case()
            return
        # Snapshot actual delivered state, never substitute the requested fixture.
        # Retain asynchronous prediction age/initial-state differences for V7.
        if frame - pending['frame'] < 3 or age < .10 * (pending['captured'] + 1) or not packet.balls:
            return
        prediction = self.ball_prediction
        slices = [dict(game_seconds=float(s.game_seconds), physics=physics(s.physics)) for s in prediction.slices]
        row = dict(**base, injection=pending.copy(), application_evidence=evidence, prediction=slices,
            prediction_payload_sha256=hashlib.sha256(json.dumps(slices,allow_nan=False).encode()).hexdigest(),
            prediction_reused_since_previous_probe=self.previous_prediction_counter == self.prediction_counter,
            ball=actual['ball'], car=actual['car'],
            players=[dict(index=i, team=int(p.team), physics=physics(p.physics),
                last_input=controls(p.last_input)) for i,p in enumerate(packet.players)],
            selector_requested_time=T+2, selector_invoked=True)
        self.previous_prediction_counter = self.prediction_counter
        try:
            selected = find_slice_at_time(prediction, T + 2)  # Original helper; errors propagate.
        except Exception as error:
            row.update(snapshot_status='selector_error', error_type=type(error).__name__, error=repr(error))
            self.write(self.log, row)  # Actual inputs are written before propagating.
            self.write(self.error_log, row)
            self.failures.append(dict(case=case['id'], reason='selector_error', error=repr(error)))
            pending['status'] = 'failed'
            self.finish_case()
            raise
        row.update(selected_index=int((T+2-prediction.slices[0].game_seconds)*120),
            selected=None if selected is None else dict(game_seconds=float(selected.game_seconds),physics=physics(selected.physics)),
            snapshot_status='valid' if selected is not None else 'selector_out_of_range')
        self.write(self.log, row)
        self.count += 1
        pending['captured'] += 1
        if selected is not None:
            self.valid_snapshots += 1
            pending['valid_snapshots'] += 1
        else:
            self.failures.append(dict(case=case['id'], reason='selector_out_of_range'))
            pending['status'] = 'failed'
        if pending['captured'] == 3:
            self.finish_case()

    def finish_case(self):
        self.fixture_results.append(dict(id=self.suite[self.i]['id'],group=self.suite[self.i]['group'],
            status=self.pending['status'], valid_snapshots=self.pending['valid_snapshots'],
            coverage_eligible=self.pending['status']=='verified' and VERIFICATION_POLICY['approved'],
            application_evidence=self.pending['evidence']))
        self.i += 1
        self.pending = None
        complete = not self.failures and len(self.fixture_results)==len(self.suite) and all(
            r['coverage_eligible'] and r['valid_snapshots']==3 for r in self.fixture_results)
        status = ('capture_complete' if complete else 'capture_incomplete') if self.i == len(self.suite) else 'capturing'
        self.summary(status)
        if self.i % 12 == 0:
            print(f'V7 CAPTURE {self.i}/{len(self.suite)} fixtures; {self.count} snapshots', flush=True)


def prepare(folder):
    folder.mkdir(parents=True)
    command = subprocess.list2cmdline([str(INTERPRETER), '-B', str(Path(__file__).resolve()), '--agent'])
    (folder / 'run_agent.cmd').write_text('@echo off\n' + command + '\n', encoding='utf-8')
    (folder / 'bot.toml').write_text('[settings]\nname = "V7 Prediction Capture"\nagent_id = "local/v7_capture"\nrun_command = "run_agent.cmd"\nroot_dir = ' + json.dumps(str(folder)) + '\nloadout_file = ' + json.dumps(str(ROOT / 'python-example-original/src/loadout.toml')) + '\n', encoding='utf-8')
    text = (ROOT / 'python-example-original/rlbot.toml').read_text(encoding='utf-8')
    for old, new in [('launcher = "Steam"', 'launcher = "Epic"'), ('config_file = "src/bot.toml"', 'config_file = "bot.toml"'), ('skip_replays = false', 'skip_replays = true'), ('start_without_countdown = false', 'start_without_countdown = true'), ('match_length = "FiveMinutes"', 'match_length = "Unlimited"')]:
        if text.count(old) != 1:
            raise RuntimeError(f'Unexpected original configuration: {old}')
        text = text.replace(old, new)
    (folder / 'match.toml').write_text(text, encoding='utf-8')
    from rlbot.config import load_player_config
    load_player_config(folder / 'bot.toml', team=1)
    return folder / 'match.toml'


def preflight_checks():
    """Diagnostic mechanics only. No service connection or populated prediction."""
    case=fixtures()[0]
    actual={k:copy.deepcopy(case[k]) for k in ('ball','car','other')}
    actual['car']['boost']=actual['other']['boost']=0.
    request=dict(id=case['id'],T=10.,frame=100,monotonic=1.,packet_receipt_counter=1)
    receipt=dict(counter=2,monotonic=1.01)
    statuses=[]
    def test(state,T,frame,expected):
        evidence=application_evidence(case,state,request,receipt,T,frame)
        assert evidence['status']==expected, evidence
        assert evidence['coverage_eligible'] == (expected=='verified' and VERIFICATION_POLICY['approved'])
        statuses.append(expected)
    test(actual,10.01,101,'verified')
    near=copy.deepcopy(actual)
    near['ball']['position'][0]+=150
    test(near,10.01,101,'near_match')
    bad=copy.deepcopy(actual)
    bad['car']['velocity'][0]+=1000
    test(bad,10.01,101,'failed')
    test(actual,10.2,124,'failed')
    missing=copy.deepcopy(actual)
    missing['ball']=None
    test(missing,10.01,101,'failed')
    # Exercise failure-before-propagation using only the empty native message;
    # no synthetic prediction slices are generated.
    bot=object.__new__(CaptureBot)
    bot.suite=fixtures(); bot.i=0; bot.index=0; bot.team=1
    bot.pending=dict(id=case['id'],T=10.,frame=100,monotonic=time.monotonic(),
        captured=0,valid_snapshots=0,status='unverified',evidence=None,
        packet_receipt_counter=1,prediction_receipt_counter=0)
    bot.count=bot.valid_snapshots=0; bot.failures=[]; bot.fixture_results=[]
    bot.callback_counter=0; bot.previous_T=10.; bot.last_emitted=None; bot.halted=False
    bot.packet_receipt=receipt; bot.prediction_receipt=None
    bot.prediction_counter=0; bot.previous_prediction_counter=None
    bot.previous_callback_prediction_counter=None
    bot.ball_prediction=flat.BallPrediction()
    bot.log=io.StringIO(); bot.event_log=io.StringIO(); bot.error_log=io.StringIO()
    bot.summary=lambda *args,**kwargs: None
    def p(state):
        return flat.Physics(location=flat.Vector3(*state['position']),velocity=flat.Vector3(*state['velocity']),
            angular_velocity=flat.Vector3(*state['angular_velocity']),rotation=flat.Rotator(*state['rotation_pyr']))
    packet=SimpleNamespace(match_info=SimpleNamespace(seconds_elapsed=10.3,frame_num=136,match_phase=flat.MatchPhase.Active),
        balls=[],players=[SimpleNamespace(physics=p(case[k]),boost=0.,team=i,last_input=flat.ControllerState()) for i,k in enumerate(('car','other'))])
    bot.get_output(packet)
    assert json.loads(bot.log.getvalue().splitlines()[0])['snapshot_status']=='missing_ball'
    packet.balls=[SimpleNamespace(physics=p(case['ball']))]
    try:
        bot.get_output(packet)
    except IndexError:
        pass
    else:
        raise AssertionError('Original empty prediction IndexError was suppressed')
    rows=[json.loads(line) for line in bot.log.getvalue().splitlines()]
    assert rows[-1]['snapshot_status']=='selector_error' and rows[-1]['error_type']=='IndexError'
    assert rows[-1]['prediction']==[] and bot.halted
    assert bot.fixture_results[-1]['status']=='failed' and not bot.fixture_results[-1]['coverage_eligible']
    assert 'IndexError' in bot.error_log.getvalue()
    # Counters distinguish repeated delivery of the same Python object.
    bot.receipt_counter=bot.packet_counter=bot.prediction_counter=0
    token=object()
    first=bot.receipt('prediction',token); second=bot.receipt('prediction',token)
    assert first['counter']==1 and second['counter']==2 and first['python_object_id']==second['python_object_id']
    return dict(application_status_cases=statuses,missing_ball_recorded=True,
        empty_prediction_evidence_written_before_IndexError=True,receipt_counters=True,
        original_selector_unchanged=True,synthetic_prediction_slices_created=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.agent:
        CaptureBot('local/v7_capture').run()
        return
    if Path(sys.executable).resolve() != INTERPRETER.resolve():
        raise RuntimeError('Use the existing python-example venv interpreter')
    manifest = verify_sources()
    folder = HERE / 'sessions' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    config = prepare(folder)
    (folder / 'metadata.json').write_text(json.dumps(dict(protected_hashes=manifest,
        runner_sha256=sha(Path(__file__)),fixtures=fixtures(),purpose='Live RLBot diagnostics only; comparison not performed',
        session_id=folder.name,versions={n:importlib.metadata.version(n) for n in ('rlbot','rlbot-flatbuffers','psutil')},
        python=sys.version,verification_policy=VERIFICATION_POLICY,
        source='Real Rocket League packets and actual delivered RLBot BallPrediction',
        config_sha256={name:sha(folder/name) for name in ('match.toml','bot.toml','run_agent.cmd')}),indent=2),encoding='utf-8')
    print(f'Session: {folder}', flush=True)
    if args.check:
        # Exercise installed state-setting serialization without starting a server.
        from rlbot.utils import fill_desired_game_state
        case = fixtures()[0]
        fill_desired_game_state({0: flat.DesiredBallState(desired(case['ball']))}, {0: flat.DesiredCarState(desired(case['car']), 0)}, None)
        checks=preflight_checks()
        verify_sources()
        (folder/'preflight_result.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
        print('PREFLIGHT PASS:',json.dumps(checks),flush=True)
        print(f"120 fixtures, config/imports and 43 protected hashes unchanged. No game launched. Policy approved: {VERIFICATION_POLICY['approved']}.")
        return
    if not VERIFICATION_POLICY['approved']:
        raise RuntimeError('Live capture blocked: proposed fixture verification tolerances await user approval. --check remains available.')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['V7_CAPTURE_FOLDER'] = str(folder)
    manager = MatchManager()
    outcome = 'interrupted'
    try:
        print('CAPTURE ONLY: do not control the blue human car. Approximately 1–3 minutes after launch.', flush=True)
        manager.start_match(config, wait_for_start=False)
        deadline = time.monotonic() + 360
        while time.monotonic() < deadline:
            path = folder / 'capture_summary.json'
            if path.exists():
                try:
                    summary = json.loads(path.read_text(encoding='utf-8'))
                except (PermissionError,FileNotFoundError,json.JSONDecodeError):
                    time.sleep(.25)
                    continue
                if summary['status'] in ('capture_complete', 'capture_incomplete', 'error'):
                    outcome = summary['status']
                    print(json.dumps(summary, indent=2), flush=True)
                    break
            time.sleep(.25)
        else:
            outcome = 'runner_timeout'
    finally:
        manager.stop_match()
        manager.disconnect()
        verify_sources()
        (folder / 'run_summary.json').write_text(json.dumps(dict(status=outcome, protected_hashes_unchanged=True, comparison_started=False), indent=2), encoding='utf-8')
        print(f'CAPTURE STOPPED: {outcome}. Results: {folder}', flush=True)


if __name__ == '__main__':
    main()
