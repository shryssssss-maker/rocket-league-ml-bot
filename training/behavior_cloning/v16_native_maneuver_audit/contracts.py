"""V16 diagnostic contracts. No connections, model imports, or dataset reads."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LIVE_PYTHON = ROOT/'python-example/venv/Scripts/python.exe'
FEATURES = ('ball_forward','ball_right','ball_up','prediction_forward','prediction_right','prediction_up',
            'ball_distance','car_speed','ball_present','prediction_valid','prediction_horizon',
            'prediction_first_offset','callback_dt')
CHANNELS = ('throttle','steer','pitch','yaw','roll','jump','boost','handbrake')
MODES = ('Neutral','Chase','Jump','Front dodge')
HISTORY = (1,2,4,8,16,32,64)
NATIVE_FIELDS = ('air_state','has_jumped','has_double_jumped','has_dodged',
                 'dodge_timeout','dodge_elapsed','dodge_dir','last_input')
ANALOG_TOL = 1e-6

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def require(ok, message):
    if not ok: raise RuntimeError(message)

def write(path, value):
    path = Path(path)
    require(path.resolve().is_relative_to(HERE.resolve()) or path.parent.resolve()==(ROOT/'training/reports').resolve(),
            'Write outside isolated V16/report paths')
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False); f.write('\n')

def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

OBS = load('_v16_observation_contract',ROOT/'training/v2_observation_test/observation_contract.py')
require(tuple(OBS.FEATURES[:13])==FEATURES,'13D ordering changed')

def protected():
    hashes = read(ROOT/'training/live_reference_test/source_hashes.json')
    require(len(hashes)==43,'Expected all 43 protected sources')
    for name, expected in hashes.items(): require(sha(ROOT/name)==expected,'Protected hash mismatch: '+name)
    return hashes

def verify():
    seal = read(HERE/'source_manifest.json')
    for name, expected in seal['files'].items(): require(sha(ROOT/name)==expected,'Pinned V16/source mismatch: '+name)
    protected()
    return seal

def controls(value): return [getattr(value,name) for name in CHANNELS]

def mode(a):
    require(len(a)==8 and all(type(x) is bool for x in a[5:]),'Native eight-channel buttons')
    require(all(type(x) in (float,int) and math.isfinite(x) for x in a[:5]),'Finite analog controls')
    t,s,p,y,r,j,b,h = a
    require(y==r==0 and not b and not h,'Teacher unsupported controls')
    if t==1 and -1<=s<=1 and p==0 and not j: return 1
    if t==s==p==0: return 2 if j else 0
    if t==s==0 and p==-1 and j: return 3
    raise RuntimeError('Unsupported teacher action combination')

def agree(a,b):
    return len(a)==len(b)==8 and a[5:]==b[5:] and max(abs(x-y) for x,y in zip(a[:5],b[:5]))<=ANALOG_TOL

def xyz(v): return [float(v.x),float(v.y),float(v.z)]

def physics(p):
    return dict(position=xyz(p.location),velocity=xyz(p.velocity),angular_velocity=xyz(p.angular_velocity),
                rotation_pyr=[float(p.rotation.pitch),float(p.rotation.yaw),float(p.rotation.roll)])

def state(s):
    return None if s is None else dict(index=int(s.index),done=bool(s.done),
                                      starts=[step.start_time for step in s.steps],durations=[step.duration for step in s.steps])

def native(player):
    # Snapshot primitive values BEFORE invoking the teacher; never hold mutable native references.
    value=dict(air_state=int(player.air_state),air_state_name=str(player.air_state),
                has_jumped=player.has_jumped,has_double_jumped=player.has_double_jumped,has_dodged=player.has_dodged,
                dodge_timeout=float(player.dodge_timeout),dodge_elapsed=float(player.dodge_elapsed),
                dodge_dir=[float(player.dodge_dir.x),float(player.dodge_dir.y)],last_input=controls(player.last_input))
    a=value['last_input']
    require(all(type(v) in (int,float) and math.isfinite(v) and -1<=v<=1 for v in a[:5]) and all(type(v) is bool for v in a[5:]),
            'Invalid native last_input; record failure, do not repair')
    return value

def native_block(n):
    require(n['air_state'] in range(5),'Unknown AirState (do not pad/guess)')
    require(all(type(n[k]) is bool for k in ('has_jumped','has_double_jumped','has_dodged')),'Native Boolean types')
    out = [float(n['air_state']==i) for i in range(5)]
    out += [float(n[k]) for k in ('has_jumped','has_double_jumped','has_dodged')]
    out += [n['dodge_timeout']/1.45,n['dodge_elapsed']/1.0,*n['dodge_dir']]
    require(len(out)==12 and all(math.isfinite(x) for x in out),'Native C must be finite 12D')
    return out

def rebuild(row):
    car=row['car']; prediction=row['observation_prediction_probe']; prior=row['previous_submission']
    context=dict(elapsed=row['T'],previous_elapsed=row['previous_callback_T'],
                 previous_mode=0 if prior is None else mode(prior['controls']),
                 previous_steer=0. if prior is None else prior['controls'][1],
                 prediction_valid=prediction is not None,first_time=row['prediction_first_time'])
    if prediction is not None: context.update(prediction_position=prediction['position'],selected_time=prediction['timestamp'])
    return list(OBS.build(dict(position=car['position'],velocity=car['velocity'],basis=OBS.basis(car['rotation_pyr']),
                               ball_present=row['ball'] is not None,ball_position=None if row['ball'] is None else row['ball']['position']),context))[:13]
