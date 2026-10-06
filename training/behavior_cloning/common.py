"""V9 contracts and source protection; no model or training dependency."""
import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('_v9_preservation', ROOT/'training/live_reference_test/launch.py')
preserve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preserve)
sys.path.insert(0, str(ROOT/'training/v2_observation_test'))
from observation_contract import FEATURES, live_adapter

CHANNELS = ('throttle','steer','pitch','yaw','roll','jump','boost','handbrake')
SPLITS = {'pilot_01': 'train', 'pilot_02': 'validation', 'pilot_03': 'test'}
PILOT = HERE/'pilots/pilot_20261004'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def controls(c):
    return [getattr(c, k) for k in CHANNELS]

def mode(a):
    if len(a) != 8 or any(type(x) is not bool for x in a[5:]):
        raise ValueError('Invalid native controller buttons')
    t,s,p,y,r,j,b,h = a
    if y != 0 or r != 0 or b or h: raise ValueError('Unsupported teacher action')
    if t == 1 and -1 <= s <= 1 and p == 0 and not j: return 1
    if t == 0 and s == 0 and p == 0: return 2 if j else 0
    if t == 0 and s == 0 and p == -1 and j: return 3
    raise ValueError('Unsupported teacher action combination')

def xyz(v): return [float(v.x), float(v.y), float(v.z)]

def physics(p):
    return dict(position=xyz(p.location), velocity=xyz(p.velocity),
                angular_velocity=xyz(p.angular_velocity),
                rotation_pyr=[p.rotation.pitch,p.rotation.yaw,p.rotation.roll])

def sequence(s):
    return None if s is None else dict(index=s.index, done=s.done,
        starts=[step.start_time for step in s.steps], durations=[step.duration for step in s.steps])
