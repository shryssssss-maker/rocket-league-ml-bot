"""V1 only: diagnostic action codecs and installed transport round-trips.

No model, arena, live connection, observation builder or training is created.
Run with either existing interpreter; the coordinator uses both pinned venvs.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
ORDER = ('throttle', 'steer', 'pitch', 'yaw', 'roll', 'jump', 'boost', 'handbrake')
NAMES = ('Neutral', 'Chase', 'Jump', 'Front dodge')
LIMIT = 1e-6


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def decode(mode, steer=0.0):
    """Diagnostic reconstruction of a supported teacher label, not a student."""
    if type(mode) is not int or mode not in range(4):
        raise ValueError(f'Unsupported mode: {mode!r}')
    if isinstance(steer, bool) or not isinstance(steer, (int, float)) or not math.isfinite(steer):
        raise ValueError('Steering must be finite numeric data')
    if not -1 <= steer <= 1 or (mode != 1 and steer != 0):
        raise ValueError('Unsupported teacher steering/mode combination')
    rows = ([0., 0., 0., 0., 0., False, False, False],
            [1., float(steer), 0., 0., 0., False, False, False],
            [0., 0., 0., 0., 0., True, False, False],
            [0., 0., -1., 0., 0., True, False, False])
    return rows[mode]


def encode(row):
    """Fail closed on unsupported teacher controls; never choose a nearest row."""
    if len(row) != 8:
        raise ValueError('Expected exactly eight channels')
    for value in row[:5]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError('Invalid analog channel')
    for value in row[5:]:
        if not isinstance(value, (bool, int, float)) or not math.isfinite(value) or value not in (0, 1):
            raise ValueError('Buttons must be exactly binary')
    throttle, steer, pitch, yaw, roll, jump, boost, brake = row
    if yaw != 0 or roll != 0 or boost != 0 or brake != 0:
        raise ValueError('Teacher yaw/roll/boost/handbrake must remain zero/false')
    if throttle == 1 and pitch == 0 and jump == 0:
        return 1, steer
    if throttle == 0 and steer == 0:
        if pitch == 0:
            return (2 if jump else 0), 0.0
        if pitch == -1 and jump == 1:
            return 3, 0.0
    raise ValueError('Unsupported teacher controller combination')


def read_controls(control):
    return [getattr(control, name) for name in ORDER]


def compare(expected, actual, context, buttons_native=True):
    require(len(actual) == 8, f'{context}: wrong channel count')
    errors = [abs(float(a) - float(b)) for a, b in zip(expected[:5], actual[:5])]
    require(max(errors) <= LIMIT, f'{context}: analog discrepancy {errors}')
    for i in range(5, 8):
        require(actual[i] == expected[i], f'{context}: {ORDER[i]} mismatch')
        if buttons_native:
            require(type(actual[i]) is bool, f'{context}: {ORDER[i]} is not a native Boolean')
    return max(errors)


def live_worker(payload):
    from rlbot import flat
    # Confirm reference controls by invoking source methods without Bot.run(),
    # Bot construction, a game packet stream or sequence-timing experiments.
    sys.path.insert(0, str(ROOT / 'python-example-original' / 'src'))
    spec = importlib.util.spec_from_file_location('v1_reference', ROOT/'python-example-original/src/bot.py')
    teacher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(teacher)
    owner = SimpleNamespace(boost_pad_tracker=SimpleNamespace(update_boost_status=lambda _: None))
    neutral = teacher.MyBot.get_output(owner, SimpleNamespace(balls=[]))
    require(read_controls(neutral) == decode(0), 'Original no-ball controls differ')
    owner.send_match_comm = lambda *args: None
    first_jump = teacher.MyBot.begin_front_flip(owner, SimpleNamespace(match_info=SimpleNamespace(seconds_elapsed=0.0)))
    require(read_controls(first_jump) == decode(2), 'Original first-jump controls differ')
    steps = [read_controls(step.controls) for step in owner.active_sequence.steps]
    require(steps == [decode(2), decode(0), decode(3), decode(0)], 'Original flip controls differ')
    source_controls = {0: neutral, 2: first_jump, 3: owner.active_sequence.steps[2].controls}
    # Isolate only the steering calculation in memory, so source get_output
    # constructs each Chase controller itself. Geometry is deliberately not
    # tested in V1; no file or production function is modified.
    owner.index = 0
    owner.active_sequence = None
    owner.renderer = SimpleNamespace(begin_rendering=lambda: None, end_rendering=lambda: None,
                                     draw_line_3d=lambda *args: None, draw_string_3d=lambda *args: None,
                                     white=None, cyan=None)
    packet = SimpleNamespace(players=[SimpleNamespace(physics=SimpleNamespace(
        location=flat.Vector3(), velocity=flat.Vector3()))],
        balls=[SimpleNamespace(physics=SimpleNamespace(location=flat.Vector3(x=100.)))])
    rows = []
    maximum = 0.0
    original_steer = teacher.steer_toward_target
    for case in payload['cases'] + payload['sentinels']:
        if case.get('mode') == 1:
            teacher.steer_toward_target = lambda *args: case['steer']
            try:
                control = teacher.MyBot.get_output(owner, packet)
            finally:
                teacher.steer_toward_target = original_steer
        elif 'mode' in case:
            control = source_controls[case['mode']]
        else:
            control = flat.ControllerState(**dict(zip(ORDER, case['row'])), use_item=False)
        maximum = max(maximum, compare(case['row'], read_controls(control), 'RLBot named fields'))
        # This is SocketRelay.send_msg's actual envelope; no socket is opened.
        envelope = flat.InterfacePacket(flat.PlayerInput(7, control))
        received = flat.InterfacePacket.unpack(envelope.pack()).message
        require(type(received) is flat.PlayerInput and received.player_index == 7, 'PlayerInput identity changed')
        row = read_controls(received.controller_state)
        require(received.controller_state.use_item is False, 'Unexpected use_item')
        maximum = max(maximum, compare(case['row'], row, 'RLBot InterfacePacket wire'))
        rows.append(row)
    def encode_controller(control):
        if control.use_item is not False:
            raise ValueError('Teacher use_item must remain false')
        return encode(read_controls(control))
    try:
        encode_controller(flat.ControllerState(use_item=True))
    except ValueError as error:
        rejected_extra = dict(case='RLBot use_item=true',error=str(error))
    else:
        raise AssertionError('Unsupported use_item accepted')
    probe = flat.ControllerState(throttle=1., steer=0.123456789)
    standalone = read_controls(flat.ControllerState.unpack(probe.pack()))
    return dict(rows=rows, maximum_analog_error=maximum, original_sequence_controls=steps,
                source_chase_calls=1001, rejected_extra=rejected_extra,
                standalone_controller_probe=dict(input=read_controls(probe), output=standalone,
                                                 equal=read_controls(probe) == standalone),
                versions={x: importlib.metadata.version(x) for x in ('rlbot', 'rlbot-flatbuffers')})


def sim_worker(payload):
    import numpy as np
    from rlgym.rocket_league.sim import RocketSimEngine
    # Execute the installed method with inert arena/car sinks. No Arena is
    # constructed, no physics tick executes, and no observation is built.
    rows = payload['original_rows'] + payload['wire_rows']
    returned = []
    maximum = 0.0
    calls = 0
    for delay in (False, True):
        for row in rows:
            capture = []
            ticks = []
            fake = SimpleNamespace(_cars={'diagnostic': SimpleNamespace(set_controls=lambda c: capture.append(read_controls(c)))},
                                   _arena=SimpleNamespace(step=lambda n: ticks.append(n)),
                                   _rlbot_delay=delay, _tick_count=0, _get_state=lambda: None)
            RocketSimEngine.step(fake, {'diagnostic': np.asarray([row], dtype=np.float32)}, {})
            require(len(capture) == 1 and ticks == [1] and fake._tick_count == 1, 'Engine transport invocation changed')
            actual = capture[0]
            maximum = max(maximum, compare(row, actual, f'RocketSim engine delay={delay}'))
            returned.append(actual)
            calls += 1
    return dict(rows=returned, maximum_analog_error=maximum, installed_engine_calls=calls,
                versions={x: importlib.metadata.version(x) for x in ('RocketSim','rlgym-rocket-league','numpy')})


def reject_cases():
    rejected = []
    base = decode(0)
    modifications = [('yaw',3,.01), ('roll',4,-.01), ('boost',6,True), ('handbrake',7,True),
                     ('reverse',0,-1.), ('partial throttle',0,.5), ('neutral steering',1,.1),
                     ('pitch without jump',2,-1.), ('positive pitch',2,1.), ('partial pitch',2,-.5),
                     ('nonbinary jump',5,.5), ('nonbinary boost',6,2), ('nonbinary handbrake',7,-1),
                     ('NaN',1,float('nan')), ('infinity',1,float('inf')), ('out-of-range steer',1,1.01),
                     ('Boolean analog',0,True)]
    candidates = [(name, base[:i]+[value]+base[i+1:]) for name,i,value in modifications]
    candidates += [('Chase plus jump', [1.,.25,0.,0.,0.,True,False,False]),
                   ('Chase plus dodge pitch', [1.,.25,-1.,0.,0.,False,False,False]),
                   ('Jump plus steering', [0.,.25,0.,0.,0.,True,False,False]),
                   ('Dodge plus steering', [0.,.25,-1.,0.,0.,True,False,False]),
                   ('short vector', base[:-1]), ('ninth channel', base+[False])]
    for name,row in candidates:
        try:
            encode(row)
        except ValueError as error:
            rejected.append(dict(case=name, error=str(error)))
        else:
            raise AssertionError(f'Unsupported combination accepted: {name}')
    for name,mode,steer in [('invalid mode',4,0),('Boolean mode',True,0),('non-Chase steering',0,.1),('decode out-of-range',1,1.01)]:
        try:
            decode(mode,steer)
        except ValueError as error:
            rejected.append(dict(case=name,error=str(error)))
        else:
            raise AssertionError(f'Invalid label accepted: {name}')
    return rejected


def worker(which, payload):
    interpreter = ROOT/('python-example' if which == 'live' else 'training')/'venv/Scripts/python.exe'
    result = subprocess.run([str(interpreter), '-B', str(SCRIPT), '--worker', which],
                            input=json.dumps(payload), capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def hashes():
    manifest = json.loads((ROOT/'training/live_reference_test/source_hashes.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, f'Protected source changed: {name}')
    require(subprocess.check_output(['git','-C',str(ROOT/'python-example-original'),'rev-parse','HEAD'], text=True).strip()
            == 'fd061f457bf19175b4a9b3b3d7811a987044c64d', 'Reference revision changed')
    return len(manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=('live','sim'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        payload = json.load(sys.stdin)
        json.dump(live_worker(payload) if args.worker == 'live' else sim_worker(payload), sys.stdout)
        return
    started = time.perf_counter()
    before = hashes()
    cases = [dict(mode=m,steer=0.,row=decode(m)) for m in (0,2,3)]
    cases += [dict(mode=1,steer=(i-500)/500.,row=[1.,(i-500)/500.,0.,0.,0.,False,False,False]) for i in range(1001)]
    for case in cases:
        label = encode(case['row'])
        require(label == (case['mode'],case['steer']), 'Diagnostic label encoding changed')
        compare(case['row'], decode(*label), 'Diagnostic mode/vector round-trip')
    # Transport-only sentinels distinguish channels the teacher never uses.
    # These must NOT be accepted by the teacher codec.
    sentinels = [dict(row=[.11,-.22,.33,-.44,.55,False,False,False])]
    for i in (5,6,7):
        row = [.11,-.22,.33,-.44,.55,False,False,False]
        row[i] = True
        sentinels.append(dict(row=row))
    rejected = reject_cases()
    for case in sentinels:
        try:
            encode(case['row'])
        except ValueError:
            pass
        else:
            raise AssertionError('Transport-only sentinel accepted as teacher action')
    live = worker('live', dict(cases=cases, sentinels=sentinels))
    rejected.append(live['rejected_extra'])
    originals = [c['row'] for c in cases+sentinels]
    sim = worker('sim', dict(original_rows=originals, wire_rows=live['rows']))
    maximum = live['maximum_analog_error']
    total = len(originals)
    for delay_index in range(2):
        for input_index in range(2):
            offset = (delay_index*2+input_index)*total
            for i,case in enumerate(cases):
                actual = sim['rows'][offset+i]
                maximum = max(maximum,compare(case['row'],actual,'End-to-end teacher round-trip'))
                mode,steer = encode(actual)
                require(mode == case['mode'], 'Transport changed teacher mode')
                require(actual[3] == 0 and actual[4] == 0 and actual[6] is False and actual[7] is False,
                        'Fixed teacher controls changed')
                compare(actual,decode(mode,steer),'Returned command re-encoding')
    unique = len({r[1] for r in live['rows'][3:1004]})
    require(unique == 1001, 'Steering was quantized or collapsed')
    after = hashes()
    summary = dict(experiment='V1', status='PASS', control_order=ORDER, analog_tolerance=LIMIT,
                   teacher_cases=len(cases), chase_values=1001, unique_wire_steering_values=unique,
                   transport_only_sentinels=4, rejected_invalid_cases=rejected,
                   mode_results={name:'PASS' for name in NAMES}, maximum_analog_round_trip_error=maximum,
                   rlbot_wire_maximum_error=live['maximum_analog_error'],
                   rocketsim_maximum_error=sim['maximum_analog_error'],
                   installed_engine_calls=sim['installed_engine_calls'],
                   source_chase_calls=live['source_chase_calls'],
                   test_code_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
                   protected_hash_count_before=before, protected_hash_count_after=after,
                   versions=dict(live=live['versions'],simulation=sim['versions']),
                   standalone_controller_probe=live['standalone_controller_probe'],
                   source_sequence_controls=live['original_sequence_controls'],
                   elapsed_seconds=time.perf_counter()-started,
                   limits='Binary envelopes and native controls only; no network, arena, physics, timing or observations tested.')
    report = ROOT/'training/reports/v1_action_round_trip_20261004.json'
    report.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f'V1 PASS: {len(cases)} teacher cases, 1001 distinct Chase steering values, {len(rejected)} invalid cases rejected.')
    print(f'Maximum analog round-trip error: {maximum:.17g} (limit {LIMIT:g})')
    print(f'Installed engine transport calls: {sim["installed_engine_calls"]}; no physics executed.')
    print(f'Protected hashes: {after} unchanged. Report: {report}')
    if not live['standalone_controller_probe']['equal']:
        print('Discrepancy: standalone ControllerState pack/unpack shifts fields; actual InterfacePacket transport passed.')


if __name__ == '__main__':
    main()
