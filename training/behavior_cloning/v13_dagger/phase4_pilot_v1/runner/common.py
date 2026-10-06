"""Phase 3 pins and read-only dependencies; no dataset loader or game launch."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V13 = HERE.parents[1]
ROOT = V13.parents[2]
LIVE_PYTHON = ROOT/'python-example/venv/Scripts/python.exe'
TRAIN_PYTHON = ROOT/'training/venv/Scripts/python.exe'
CHECKPOINT = V13/'phase2_bootstrap_v1/B_best.pt'
CHECKPOINT_PIN = '579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7'
DATASET = ROOT/'training/behavior_cloning/datasets/v10_live_1v1_18d_20261005_v1'
MANIFEST_PIN = '080943fd8f9a81ada800e0a2f9ff3593b67f66d32272211d67030c03cf433a83'
OBS = ROOT/'training/v2_observation_test/observation_contract.py'
OBS_PIN = 'e0f59f763d33e59f6815a86e9c452b3fcade87fe26727ca53c910ebb7e6db750'
POLICY_SOURCE_PIN = '6c267c85293cc70ad2e12ca54894c31309092e9bded5a96d53460f196efb704f'
CHANNELS = ('throttle','steer','pitch','yaw','roll','jump','boost','handbrake')
MODES = ('Neutral','Chase','Jump','Front dodge')
ANALOG_TOL = 1e-6

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path, obj): Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def verify():
    for path,pin in ((CHECKPOINT,CHECKPOINT_PIN),(DATASET/'manifest.json',MANIFEST_PIN),(OBS,OBS_PIN),(V13/'bootstrap.py',POLICY_SOURCE_PIN)):
        if sha(path) != pin: raise RuntimeError('Pinned artifact mismatch: '+str(path))
    audit = read(V13/'phase1_context_audit_v1/audit.json')
    for path,pin in audit['v12_verified_hashes'].items():
        if sha(path) != pin: raise RuntimeError('V12 dependency changed: '+path)
    pm = read(V13/'projection_13d_v1/manifest.json')
    for path in (V13/'projection.py',Path(pm['projection_code_path'])):
        if sha(path) != pm['projection_code_sha256']: raise RuntimeError('Projection changed')
    preserve = load('_phase3_preservation',ROOT/'training/live_reference_test/launch.py')
    protected = preserve.verify_sources()
    if len(protected) != 43: raise RuntimeError('Expected 43 protected hashes')
    return protected

def immutable_snapshot():
    # Frozen version, V12 and completed V13 outputs. External dataset rows are
    # neither opened semantically nor used in this live diagnostic.
    folders = (DATASET,V13.parent/'v12',V13/'phase1_context_audit_v1',
               V13/'phase2_bootstrap_v1',V13/'projection_13d_v1',V13/'phase3_shadow_v1',V13/'phase3_shadow_v2')
    paths = [p for folder in folders for p in folder.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    paths += [V13/'bootstrap.py',V13/'projection.py',V13/'context_audit.py']
    return {str(p):sha(p) for p in sorted(paths)}

def controls(c): return [getattr(c,k) for k in CHANNELS]
def mode(c):
    if c[5] and c[2] == -1: return 'Front dodge'
    if c[5]: return 'Jump'
    if c[0] == 1: return 'Chase'
    return 'Neutral'
def xyz(v): return [float(v.x),float(v.y),float(v.z)]
def state(seq):
    return None if seq is None else dict(index=seq.index,done=seq.done,
        starts=[s.start_time for s in seq.steps],durations=[s.duration for s in seq.steps])
