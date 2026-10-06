"""V3 only: fresh-callback source boundaries; no sockets, physics or training."""
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python-example-original/src'))
sys.path.insert(0, str(ROOT / 'training/v2_observation_test'))
import bot as original
from rlbot import flat
from observation_contract import basis
from util.sequence import Sequence, ControlStep

CHANNELS = ('throttle', 'steer', 'pitch', 'yaw', 'roll', 'jump', 'boost', 'handbrake')
REVISION = 'fd061f457bf19175b4a9b3b3d7811a987044c64d'
TOL = 1e-12


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def xyz(v):
    return tuple(float(getattr(v, k)) for k in ('x', 'y', 'z'))


def controls(c):
    return [getattr(c, k) for k in CHANNELS]


class Renderer:
    cyan = 'cyan'
    white = 'white'

    def __init__(self):
        self.target = None

    def begin_rendering(self):
        pass

    def end_rendering(self):
        pass

    def draw_string_3d(self, *args):
        pass

    def draw_line_3d(self, start, end, color):
        if isinstance(start, original.CarAnchor):
            self.target = [end.x, end.y, end.z]


class Tracker:
    def update_boost_status(self, packet):
        pass


def source_result(packet, prediction, index):
    # Bypass the socket-owning constructor only. Execute original get_output,
    # begin_front_flip, prediction helper, steering helper and sequence tick.
    agent = object.__new__(original.MyBot)
    agent.index = index
    agent.active_sequence = None
    agent.boost_pad_tracker = Tracker()
    agent.renderer = Renderer()
    agent.ball_prediction = prediction
    events = []
    agent.send_match_comm = lambda *args: events.append('flip')
    lookup, steer = original.find_slice_at_time, original.steer_toward_target

    def traced_lookup(p, t):
        events.append('prediction')
        return lookup(p, t)

    def traced_steer(car, target):
        events.append('chase')
        return steer(car, target)

    with patch.object(original, 'find_slice_at_time', traced_lookup), patch.object(original, 'steer_toward_target', traced_steer):
        output = agent.get_output(packet)
    return dict(events=events, target=agent.renderer.target, controls=controls(output),
                sequence_created=agent.active_sequence is not None)


def adapted_result(packet, prediction, index):
    """Isolated source-branch reconstruction on raw physics, not a new expert."""
    car = packet.players[index].physics
    position, velocity = xyz(car.location), xyz(car.velocity)
    current = xyz(packet.balls[0].physics.location)
    distance = math.sqrt(sum((a-b)**2 for a, b in zip(position, current)))
    speed = math.sqrt(sum(v*v for v in velocity))
    target = current
    events = []
    if distance > 1500:
        events.append('prediction')
        # Keep original selection helper. Index semantics belong to V6.
        selected = original.find_slice_at_time(prediction, packet.match_info.seconds_elapsed + 2)
        if selected is not None:
            target = xyz(selected.physics.location)
    sequence = None
    if 750 < speed < 800:
        events.append('flip')
        sequence = Sequence([
            ControlStep(.05, flat.ControllerState(jump=True)),
            ControlStep(.05, flat.ControllerState(jump=False)),
            ControlStep(.2, flat.ControllerState(jump=True, pitch=-1)),
            ControlStep(.8, flat.ControllerState()),
        ])
        output = sequence.tick(packet)
    else:
        events.append('chase')
        axes = basis((car.rotation.pitch, car.rotation.yaw, car.rotation.roll))
        delta = tuple(a-b for a, b in zip(target, position))
        # Preserve Vec3.dot's arithmetic order, including signed zero.
        # sum() starts with +0 and can reverse atan2 at a behind-car target.
        def dot(axis):
            return delta[0]*axis[0] + delta[1]*axis[1] + delta[2]*axis[2]
        forward = dot(axes[0])
        right = dot(axes[1])
        output = flat.ControllerState(throttle=1, steer=max(-1, min(1, 5*math.atan2(right, forward))))
    return dict(events=events, target=list(target), controls=controls(output),
                sequence_created=sequence is not None), distance, speed


def packet_for(position, delta, velocity, rotation, team, index):
    car = flat.PlayerInfo(team=team, physics=flat.Physics(location=flat.Vector3(*position),
        velocity=flat.Vector3(*velocity), rotation=flat.Rotator(*rotation)))
    opponent = flat.PlayerInfo(team=1-team, physics=flat.Physics(location=flat.Vector3(4000, 4000, 17)))
    players = [opponent, opponent]
    players[index] = car
    return flat.GamePacket(players=players, balls=[flat.BallInfo(physics=flat.Physics(
        location=flat.Vector3(*(a+b for a, b in zip(position, delta)))))],
        match_info=flat.MatchInfo(seconds_elapsed=12))


def prediction_for(position, valid):
    # Fixed valid / out-of-range fixtures only: no indexing sweep or predictor.
    target = flat.Physics(location=flat.Vector3(position[0]+900, position[1]-700, position[2]+500))
    return flat.BallPrediction(slices=[flat.PredictionSlice(game_seconds=12+i/120, physics=target)
                                      for i in range(241 if valid else 1)])


def fixture_cases():
    poses = [(0,0,0), (0,math.pi/2,0), (.35,-.7,-.4), (math.pi/2,0,0),
             (0,0,math.pi/2), (0,math.pi,math.pi)]
    cases = []
    for pose_id, pose in enumerate(poses):
        for distance in (1499.9,1500.,1500.1):
            for speed in (749.9,750.,750.1,799.9,800.,800.1):
                for valid in (False,True):
                    for team in (0,1):
                        # Vertical separation/speed makes exact equalities exact
                        # while exposing any accidental 2D metric replacement.
                        position = (0.,0.,0.) if pose_id < 3 else (4096.,-1200.,1000.)
                        cases.append(dict(kind='boundary', position=position, delta=(0.,0.,distance),
                            velocity=(0.,0.,speed), rotation=pose, team=team,index=team,
                            prediction_valid=valid,nominal_distance=distance,nominal_speed=speed))
    for pose_id, pose in enumerate(poses):
        axes = basis(pose)
        for axis in axes:
            for sign in (-1,1):
                for team in (0,1):
                    cases.append(dict(kind='local_axis',position=(100.,-200.,1000.),
                        delta=tuple(sign*1000*x for x in axis),velocity=(300.,400.,0.),
                        rotation=pose,team=team,index=1-team,prediction_valid=True,
                        nominal_distance=1000.,nominal_speed=500.))
    # Full XYZ norms at exact equality: 900^2+1200^2=1500^2,
    # 450^2+600^2=750^2, and 480^2+640^2=800^2.
    for velocity in ((450.,600.,0.), (480.,640.,0.)):
        cases.append(dict(kind='pythagorean',position=(0.,0.,17.),delta=(900.,1200.,0.),
            velocity=velocity,rotation=(.4,.9,-.3),team=0,index=0,prediction_valid=True,
            nominal_distance=1500.,nominal_speed=math.hypot(*velocity[:2])))
    return cases


def hashes():
    manifest = json.loads((ROOT/'training/live_reference_test/source_hashes.json').read_text(encoding='utf-8-sig'))
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in manifest},manifest


def main():
    started = time.perf_counter()
    actual_revision = subprocess.run(['git','-C',str(ROOT/'python-example-original'),'rev-parse','HEAD'],
                                    capture_output=True,text=True,check=True).stdout.strip()
    require(actual_revision == REVISION, 'Recovered reference revision changed')
    before, manifest = hashes()
    require(before == manifest, 'Protected source/configuration hash mismatch')
    rows, max_error = [], 0.
    signed_zero_regression = None
    for wire in (False,True):
        for number, case in enumerate(fixture_cases()):
            packet = packet_for(case['position'],case['delta'],case['velocity'],case['rotation'],case['team'],case['index'])
            prediction = prediction_for(case['position'],case['prediction_valid'])
            if wire:
                packet = flat.CorePacket.unpack(flat.CorePacket(packet).pack()).message
                prediction = flat.CorePacket.unpack(flat.CorePacket(prediction).pack()).message
            expected = source_result(packet,prediction,case['index'])
            actual, distance, speed = adapted_result(packet,prediction,case['index'])
            if wire and number == 480:
                car = packet.players[case['index']].physics
                axes = basis((car.rotation.pitch,car.rotation.yaw,car.rotation.roll))
                delta = tuple(a-b for a,b in zip(xyz(packet.balls[0].physics.location),xyz(car.location)))
                bad_right = sum(a*b for a,b in zip(delta,axes[1]))
                bad_forward = sum(a*b for a,b in zip(delta,axes[0]))
                bad_steer = max(-1,min(1,5*math.atan2(bad_right,bad_forward)))
                require(expected['controls'][1] == -1 and bad_steer == 1,
                        'Signed-zero regression fixture no longer exposes the defect')
                signed_zero_regression = dict(case=number,wire=True,original_steer=-1,
                    corrected_steer=actual['controls'][1],sum_based_steer=bad_steer)
            branch = (['prediction'] if distance > 1500 else []) + (['flip'] if 750 < speed < 800 else ['chase'])
            label = f'{number}, wire={wire}'
            require(expected['events'] == actual['events'] == branch, f'Branch/order mismatch: {label}')
            require(expected['target'] == actual['target'], f'Target mismatch: {label}')
            require(expected['sequence_created'] == actual['sequence_created'] == (750 < speed < 800), f'Flip mismatch: {label}')
            for i, name in enumerate(CHANNELS):
                a,b = expected['controls'][i],actual['controls'][i]
                if i < 5:
                    require(math.isfinite(a) and math.isfinite(b), f'Nonfinite {label}')
                    error = abs(a-b)
                    max_error = max(max_error,error)
                    require(error <= TOL, f'Analog mismatch {name}: {label}')
                else:
                    require(type(a) is bool and type(b) is bool and a == b, f'Button mismatch {label}')
            for result in (expected, actual):
                require(result['controls'][3:5] == [0,0] and result['controls'][6:] == [False,False], f'Unsupported controls: {label}')
            if case['kind'] == 'boundary':
                # Assert nominal strict-boundary branches as well, independently
                # of the measured raw metric predicates above.
                nominal = (['prediction'] if case['nominal_distance'] > 1500 else []) + (['flip'] if 750 < case['nominal_speed'] < 800 else ['chase'])
                require(branch == nominal, f'Wire precision moved nominal branch: {label}')
            rows.append(dict(id=number,wire=wire,kind=case['kind'],nominal_distance=case['nominal_distance'],
                nominal_speed=case['nominal_speed'],actual_distance=distance,actual_speed=speed,
                team=case['team'],index=case['index'],rotation=case['rotation'],
                prediction_valid=case['prediction_valid'],events=branch,target=actual['target'],
                original_controls=expected['controls'],adapted_controls=actual['controls']))
    after,_ = hashes()
    require(after == before, 'Protected files changed during V3')
    report = dict(experiment='V3',status='PASS',reference_revision=actual_revision,
        paired_cases=len(rows),boundary_cases=sum(r['kind']=='boundary' for r in rows),
        local_axis_cases=sum(r['kind']=='local_axis' for r in rows),
        pythagorean_cases=sum(r['kind']=='pythagorean' for r in rows),
        maximum_analog_difference=max_error,analog_tolerance=TOL,buttons_exact=True,
        branch_and_order_exact=True,targets_exact=True,protected_hashes=len(before),
        signed_zero_regression=signed_zero_regression,
        prediction_before_flip_cases=sum(r['events']==['prediction','flip'] for r in rows),
        elapsed_seconds=time.perf_counter()-started,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Fresh callbacks only; original prediction selection and sequence primitives reused; no temporal sweep, physics, match or training.',
        cases=rows)
    path=ROOT/'training/reports/v3_source_boundaries_20261004.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f"V3 PASS: {len(rows)} paired cases; maximum analog difference {max_error}; {len(before)} protected hashes unchanged.")
    print(f"Prediction-before-flip cases: {report['prediction_before_flip_cases']}. Report: {path}")


if __name__ == '__main__':
    main()
