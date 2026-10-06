"""V6 bounded source-informed probes. No provider selection or RLBot parity."""
from pathlib import Path
from importlib.metadata import distribution
import base64
import csv
import hashlib
import json
import math
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import RocketSim as rs

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'training/reports/v6_upstream_source_2da51b1dac7b8127127613a5ff30e490bdd70dd8'
REVISION = '2da51b1dac7b8127127613a5ff30e490bdd70dd8'
FILES = ('pyproject.toml','python-mtheall/Arena.cpp','python-mtheall/BallPredictor.cpp',
         'src/Sim/BallPredTracker/BallPredTracker.cpp','src/Sim/BallPredTracker/BallPredTracker.h',
         'src/Sim/Ball/Ball.cpp','python-mtheall/BallState.cpp','python-mtheall/Ball.cpp')


def require(condition,message):
    if not condition:
        raise AssertionError(message)


def snapshot(state):
    return dict(position=[float(x) for x in state.pos.as_numpy()],
                velocity=[float(x) for x in state.vel.as_numpy()],
                angular_velocity=[float(x) for x in state.ang_vel.as_numpy()])


def error(a,b):
    values = [abs(x-y) for k in a for x,y in zip(a[k],b[k])]
    require(all(math.isfinite(x) for x in values),'Nonfinite state')
    return max(values)


def initial():
    return rs.BallState(pos=rs.Vec(0,0,1200),vel=rs.Vec(600,200,100),ang_vel=rs.Vec(.1,.2,.3))


def arena():
    a = rs.Arena(rs.GameMode.SOCCAR)
    a.ball.set_state(initial())
    return a


def reference(num_ticks,start_state=None):
    a = arena()
    if start_state is not None:
        a.ball.set_state(start_state)
    rows = [snapshot(a.ball.get_state())]
    for _ in range(num_ticks):
        a.step(1)
        rows.append(snapshot(a.ball.get_state()))
    return rows


def protected():
    manifest = json.loads((ROOT/'training/live_reference_test/source_hashes.json').read_text(encoding='utf-8-sig'))
    actual = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in manifest}
    require(actual==manifest,'Protected sources changed')
    return actual


def source_metadata():
    git = subprocess.run(['git','-C',str(SOURCE),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    require(git == REVISION,'Unexpected source revision')
    rows = {}
    for relative in FILES:
        p = SOURCE/relative
        data = p.read_bytes()
        # git hash-object verifies checked-out content, accounting for CRLF.
        actual = subprocess.run(['git','-C',str(SOURCE),'hash-object','--path='+relative,str(p)],
                                capture_output=True,text=True,check=True).stdout.strip()
        expected = subprocess.run(['git','-C',str(SOURCE),'rev-parse','HEAD:'+relative],
                                  capture_output=True,text=True,check=True).stdout.strip()
        require(actual==expected,'Source file differs from pinned Git object: '+relative)
        rows[relative] = dict(sha256=hashlib.sha256(data).hexdigest(),git_blob=actual)
    return dict(revision=git,path=str(SOURCE),files=rows)


def installed_metadata():
    d = distribution('rocketsim')
    p = Path(rs.__file__)
    digest = hashlib.sha256(p.read_bytes()).digest()
    encoded = base64.urlsafe_b64encode(digest).decode().rstrip('=')
    record = list(csv.reader(d.read_text('RECORD').splitlines()))
    row = next(x for x in record if x[0]=='RocketSim.pyd')
    require(row[1] == 'sha256='+encoded and int(row[2])==p.stat().st_size,'Installed binary RECORD mismatch')
    return dict(version=d.version,path=str(p),sha256=digest.hex(),size=p.stat().st_size,
                record_verified=True,direct_url=d.read_text('direct_url.json'),
                wheel=d.read_text('WHEEL'),exact_source_commit_established=False)


def arena_probes():
    rows = []
    for count,stride in ((1,1),(12,1),(12,2),(12,3),(12,8),(241,1)):
        # Fresh arena per combination: do not enlarge an existing cache at dt=0.
        a = arena()
        before,ticks_before = snapshot(a.ball.get_state()),a.tick_count
        states = a.get_ball_prediction(count,stride)
        require(len(states)==count,'Unexpected returned count')
        expected = reference((count-1)*stride)
        differences = [error(snapshot(s),expected[i*stride]) for i,s in enumerate(states)]
        require(max(differences)==0,'Fresh-arena sample offset mismatch')
        require(a.tick_count==ticks_before and snapshot(a.ball.get_state())==before,'Prediction mutated caller arena')
        repeated = a.get_ball_prediction(count,stride)
        require([snapshot(s) for s in repeated]==[snapshot(s) for s in states],'Unchanged same-size repeat changed output')
        a.step(2)
        advanced = a.get_ball_prediction(count,stride)
        extended = reference(2+(count-1)*stride)
        advanced_errors = [error(snapshot(s),extended[2+i*stride]) for i,s in enumerate(advanced)]
        inferred = [min(range(len(extended)),key=lambda j:error(snapshot(s),extended[j])) for s in advanced]
        require(inferred==[2+i*stride for i in range(count)],'Arena cache offset mismatch')
        require(max(advanced_errors)<=.01,f'Arena cache error above offset-disambiguation bound: {count}/{stride}: {max(advanced_errors)}')
        rows.append(dict(count=count,stride=stride,offsets=[i*stride for i in range(count)],
            tick_rate=float(a.tick_rate),tick_time=float(a.tick_time),
            maximum_initial_error=max(differences),maximum_after_two_ticks_error=max(advanced_errors),
            inferred_offsets_after_two_ticks=inferred,offset_disambiguation_bound=.01,
            first_sample=snapshot(states[0]),last_sample=snapshot(states[-1]),
            unchanged_repeat_exact=True,caller_not_advanced_by_prediction=True))
    # Keyword spelling discovery is a safe Python argument-parser check.
    keyword = {}
    for name in ('num_ticks','num_states'):
        a = arena()
        try:
            value = a.get_ball_prediction(**{name:12,'tick_interval':1})
            keyword[name] = dict(accepted=True,length=len(value))
        except TypeError as ex:
            keyword[name] = dict(accepted=False,error=str(ex))
    return rows,keyword


def predictor_probes():
    rows = []
    for stride in (1,2,3,8):
        # Fixed count/stride for each predictor; no cache enlargement.
        predictor = rs.BallPredictor()
        supplied = initial()
        before = snapshot(supplied)
        states = predictor.get_ball_prediction(supplied,0,12,stride)
        require(len(states)==12,'Unexpected predictor count')
        expected = reference(11*stride)
        differences = [error(snapshot(s),expected[i*stride]) for i,s in enumerate(states)]
        # Record mismatches; do not turn this into a hidden initialization fix.
        rows.append(dict(stride=stride,count=12,ticks_since_last_update=0,
            input=before,input_unchanged=snapshot(supplied)==before,
            first_sample=snapshot(states[0]),maximum_reference_error=max(differences),
            sampling_matches_reference=max(differences)==0))
    # Separate fresh instances let us test elapsed-tick parameter semantics
    # without altering or repairing a previously observed failing cache.
    for elapsed in (1,2,12):
        predictor = rs.BallPredictor()
        states = predictor.get_ball_prediction(initial(),elapsed,12,1)
        expected = reference(11)
        differences = [error(snapshot(s),expected[i]) for i,s in enumerate(states)]
        rows.append(dict(stride=1,count=12,ticks_since_last_update=elapsed,
            first_sample=snapshot(states[0]),maximum_reference_error=max(differences),
            sampling_matches_reference=max(differences)==0))
    return rows


def predictor_cache_probes():
    predictor = rs.BallPredictor()
    caller = arena()
    predictor.get_ball_prediction(caller.ball.get_state(),0,12,1)
    rows = []
    for delta in (0,2,12):
        caller.step(delta)
        supplied = caller.ball.get_state()
        expected = reference(11,supplied)
        values = predictor.get_ball_prediction(supplied,delta,12,1)
        errors = [error(snapshot(s),expected[i]) for i,s in enumerate(values)]
        inferred = [min(range(len(expected)),key=lambda j:error(snapshot(s),expected[j])) for s in values]
        require(inferred == list(range(12)),'Predictor cache sample offsets mismatch')
        require(max(errors)<=.01,'Predictor cache error above offset-disambiguation bound')
        rows.append(dict(ticks_since_last_update=delta,maximum_error=max(errors),inferred_offsets=inferred,
                         first_sample=snapshot(values[0]),supplied=snapshot(supplied)))
    changed=rs.BallState(pos=rs.Vec(100,200,1400),vel=rs.Vec(-300,400,200),ang_vel=rs.Vec(.2,.1,0))
    expected=reference(11,changed)
    values=predictor.get_ball_prediction(changed,0,12,1)
    errors=[error(snapshot(s),expected[i]) for i,s in enumerate(values)]
    require(max(errors)==0,'Changed-state full prediction mismatch')
    rows.append(dict(changed_state=True,ticks_since_last_update=0,maximum_error=max(errors),
                     first_sample=snapshot(values[0]),supplied=snapshot(changed)))
    # A sub-margin change can intentionally re-use prior prediction data.
    # It is not an exact copy of the supplied state even at sample zero.
    slightly_changed=rs.BallState(pos=rs.Vec(100.1,200,1400),vel=rs.Vec(-300,400,200),ang_vel=rs.Vec(.2,.1,0))
    reused=predictor.get_ball_prediction(slightly_changed,0,12,1)
    require(snapshot(reused[0])==snapshot(values[0]),'Sub-margin cache probe diverges from source reuse')
    rows.append(dict(sub_margin_change=True,ticks_since_last_update=0,
                    supplied=snapshot(slightly_changed),first_sample=snapshot(reused[0]),
                    first_sample_input_difference=error(snapshot(reused[0]),snapshot(slightly_changed)),
                    previous_cache_reused=True))
    # Check both constructor styles and zero spin without a prior unsafe call.
    initialization=[]
    for style in ('keyword','assigned_fields'):
        supplied=rs.BallState(pos=rs.Vec(0,0,1200),vel=rs.Vec(600,200,100)) if style=='keyword' else rs.BallState()
        if style=='assigned_fields':
            supplied.pos=rs.Vec(0,0,1200)
            supplied.vel=rs.Vec(600,200,100)
        expected=reference(11,supplied)
        values=rs.BallPredictor().get_ball_prediction(supplied,0,12,1)
        errors=[error(snapshot(s),expected[i]) for i,s in enumerate(values)]
        initialization.append(dict(style=style,supplied=snapshot(supplied),first_sample=snapshot(values[0]),
                                   maximum_reference_error=max(errors),matches=max(errors)==0))
    return rows,initialization


def main():
    started = time.perf_counter()
    before = protected()
    selector_files=(ROOT/'training/v6_prediction_test/run_v6.py',ROOT/'training/reports/v6_prediction_selection_20261004.json')
    selector_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in selector_files}
    source,binary = source_metadata(),installed_metadata()
    meshes=ROOT/'training/venv/Lib/site-packages/rlgym/rocket_league/sim/collision_meshes'
    rs.init(str(meshes))
    arenas,keywords = arena_probes()
    predictors = predictor_probes()
    cache,initialization = predictor_cache_probes()
    require(protected()==before,'Protected files changed')
    require(installed_metadata()['sha256']==binary['sha256'],'Installed binary changed')
    require({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in selector_files}==selector_hashes,
            'Established teacher selector changed')
    report=dict(experiment='V6_NATIVE',status='MEASURED_PROVENANCE_OR_API_LIMITATIONS_REMAIN',
        source=source,installed=binary,arena_probes=arenas,arena_keyword_tests=keywords,
        predictor_probes=predictors,protected_hashes=len(before),
        predictor_cache_probes=cache,initialization_probes=initialization,
        teacher_selector_hashes_unchanged=selector_hashes,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        elapsed_seconds=time.perf_counter()-started,
        safety='Positive counts/strides; fresh caches for each combination; no horizon growth at zero elapsed ticks, no trimming, no production provider selected.',
        scope='Car-free native sampling against separately stepped RocketSim only; not RLBot/RocketSim parity.')
    path=ROOT/'training/reports/v6_native_sampling_20261004.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f"V6 native measured: {len(arenas)} Arena combinations; {len(predictors)} BallPredictor cases.")
    print('Fresh Arena states exact; advanced cache offsets verified, numerical differences recorded.')
    print('Predictor matches:',[p['sampling_matches_reference'] for p in predictors])
    print('Keywords:',keywords)
    print(f"{len(before)} protected hashes unchanged; exact binary source revision not established. Report: {path}")


if __name__=='__main__':
    main()
