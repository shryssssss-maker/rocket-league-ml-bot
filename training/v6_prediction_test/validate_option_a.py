"""Bounded contract checks and user-run fresh-arena latency benchmark."""
import argparse
import ast
import copy
import faulthandler
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import struct
import subprocess
import sys
import time
from unittest.mock import patch

sys.dont_write_bytecode=True
from constrained_provider import ConstrainedProvider,ROOT,BINARY_SHA256
import constrained_provider as provider_module


def require(condition,message):
    if not condition:
        raise AssertionError(message)


def protected():
    saved=json.loads((ROOT/'training/live_reference_test/source_hashes.json').read_text(encoding='utf-8-sig'))
    actual={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in saved}
    require(actual==saved,'Protected source/config hash mismatch')
    return actual


def f32(x):
    return struct.unpack('<f',struct.pack('<f',x))[0]


def ball_cases():
    return [dict(position=[x,y,z],velocity=[vx,vy,vz],angular_velocity=[ax,ay,az],rotation_pyr=[.2,.3,-.1])
        for x,y,z,vx,vy,vz,ax,ay,az in (
            (0,0,93.15,0,0,0,0,0,0),
            (0,0,1200,600,200,100,.1,.2,.3),
            (3000,0,300,1800,100,-100,1,2,3),
            (3500,4400,500,1500,1800,100,-2,1,3),
            (0,4000,250,0,1800,-100,0,1,0),
            (0,0,1700,500,-400,1200,3,-2,1))]


def static_contract():
    path=ROOT/'training/v6_prediction_test/constrained_provider.py'
    tree=ast.parse(path.read_text())
    constructors=[x for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='Arena']
    calls=[x for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='get_ball_prediction']
    require(len(constructors)==len(calls)==1,'Expected one arena construction and one prediction call site')
    require({k.arg:ast.literal_eval(k.value) for k in calls[0].keywords}=={'num_states':241,'tick_interval':1},'Approved native call changed')
    require(not any(isinstance(x,ast.Attribute) and x.attr=='BallPredictor' for x in ast.walk(tree)),'BallPredictor forbidden')
    assignments=[target for x in ast.walk(tree) if isinstance(x,ast.Assign) for target in x.targets]
    require(not any(isinstance(x,ast.Attribute) and x.attr in ('pos','vel','ang_vel','rot_mat') for x in assignments),'Post-construction native mutation forbidden')


def contract_checks(provider):
    original_arena=provider_module.rs.Arena
    events=[]
    live=set()
    counter=0
    class ArenaProbe:
        def __init__(self,*args,**kwargs):
            nonlocal counter
            self.number=counter
            counter+=1
            self.native=original_arena(*args,**kwargs)
            self.calls=0
            live.add(self.number)
            events.append(dict(event='created',id=self.number))
        @property
        def ball(self):
            return self.native.ball
        def get_ball_prediction(self,**kwargs):
            self.calls+=1
            require(self.calls==1,'Cache reuse detected')
            require(kwargs==dict(num_states=241,tick_interval=1),'Native call mismatch')
            result=self.native.get_ball_prediction(**kwargs)
            require(self.native.tick_count==0,'Prediction stepped caller arena')
            events.append(dict(event='predicted',id=self.number,count=len(result)))
            return result
        def __del__(self):
            del self.native
            live.discard(self.number)
            events.append(dict(event='released',id=self.number))
    rows=[]
    with patch.object(provider_module.rs,'Arena',ArenaProbe):
        for n,ball in enumerate(ball_cases()):
            for T in (0.,12.,f32(500.123)):
                before=copy.deepcopy(ball)
                result=provider.predict(ball,T)
                require(ball==before,'Supplied caller state mutated')
                require(len(result.states)==len(result.game_seconds)==241,'Prediction count mismatch')
                require(result.game_seconds==tuple(T+i/120 for i in range(241)),'Canonical timestamp rule mismatch')
                require(list(result.states[0].pos.as_numpy())==[f32(x) for x in ball['position']], 'Sample zero mismatch')
                require(list(result.states[0].vel.as_numpy())==[f32(x) for x in ball['velocity']], 'Sample zero velocity mismatch')
                gc.collect()
                require(not live,'Temporary arena was not released')
                rows.append(dict(case=n,T=T,count=241,caller_unchanged=True,arena_released=True))
    require(counter==len(rows),'Fresh arena count mismatch')
    # Use unchanged original helper in the existing live interpreter. No
    # interpolation or independent replacement selector in this validation.
    worker=r'''
import json,sys
from pathlib import Path
sys.dont_write_bytecode=True
root=Path.cwd()
sys.path.insert(0,str(root/'python-example-original/src'))
from rlbot import flat
from util.ball_prediction_analysis import find_slice_at_time
for c in json.load(sys.stdin):
    T=c['T']
    p=flat.BallPrediction(slices=[flat.PredictionSlice(game_seconds=T+i/120,physics=flat.Physics(location=flat.Vector3(i,0,100))) for i in range(241)])
    s=find_slice_at_time(p,T+2)
    assert s is not None
    expected=int(((T+2)-p.slices[0].game_seconds)*120)
    assert s.physics.location.x==expected and s.game_seconds==p.slices[expected].game_seconds
print('Original selector checks PASS')
'''
    run=subprocess.run([str(ROOT/'python-example/venv/Scripts/python.exe'),'-B','-c',worker],
        input=json.dumps(rows),text=True,capture_output=True,check=True,cwd=ROOT)
    return dict(cases=len(rows),results=rows,arena_events=events,original_selector_output=run.stdout.strip())


def quantile(values,p):
    ordered=sorted(values)
    index=(len(ordered)-1)*p
    lo=int(index)
    hi=min(lo+1,len(ordered)-1)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(index-lo)


def benchmark(provider,samples,watchdog_seconds):
    rng=random.Random(20261004)
    # Diversity fixture generator only; no policy actions or demonstrations.
    cases=[dict(position=[rng.uniform(-3500,3500),rng.uniform(-4500,4500),rng.uniform(95,1800)],
                velocity=[rng.uniform(-2200,2200),rng.uniform(-2200,2200),rng.uniform(-1200,1200)],
                angular_velocity=[rng.uniform(-4,4) for _ in range(3)],
                rotation_pyr=[rng.uniform(-math.pi,math.pi) for _ in range(3)]) for _ in range(samples)]
    warmup=[]
    print(f'Warmup: 20 unchanged prediction calls; watchdog={watchdog_seconds:g}s per call.',flush=True)
    for ball in cases[:min(20,samples)]:
        faulthandler.dump_traceback_later(watchdog_seconds,repeat=False)
        try:
            start=time.perf_counter_ns()
            provider.predict(ball,100.)
            warmup.append((time.perf_counter_ns()-start)/1e6)
        finally:
            faulthandler.cancel_dump_traceback_later()
    costs=[]
    wall_start=time.monotonic()
    print(f'Measured samples: 0/{samples}; progress every 100 samples.',flush=True)
    for n,ball in enumerate(cases):
        before=copy.deepcopy(ball)
        # Keep canonical-time conversion inside the measured expression, as
        # it was in the original benchmark. Timer setup stays outside it.
        faulthandler.dump_traceback_later(watchdog_seconds,repeat=False)
        try:
            start=time.perf_counter_ns()
            result=provider.predict(ball,f32(100+n/60))
            costs.append((time.perf_counter_ns()-start)/1e6)
        finally:
            faulthandler.cancel_dump_traceback_later()
        require(ball==before,'Benchmark caller state mutated')
        require(len(result.states)==241,'Benchmark count mismatch')
        if (n+1)%100==0 or n+1==samples:
            print(f'Samples {n+1}/{samples}; elapsed={time.monotonic()-wall_start:.1f}s; '
                  f'last={costs[-1]:.3f}ms; running mean={statistics.mean(costs):.3f}ms; '
                  f'max={max(costs):.3f}ms; caller/count checks PASS',flush=True)
    return dict(samples=samples,warmup_samples=len(warmup),seed=20261004,
        costs_ms=costs,warmup_costs_ms=warmup,min_ms=min(costs),mean_ms=statistics.mean(costs),
        median_ms=statistics.median(costs),p90_ms=quantile(costs,.90),p95_ms=quantile(costs,.95),
        p99_ms=quantile(costs,.99),maximum_ms=max(costs),
        over_one_physics_tick=sum(x>1000/120 for x in costs),
        over_nominal_two_tick_callback=sum(x>1000/60 for x in costs),
        diagnostics=dict(progress_every=100,watchdog_seconds=watchdog_seconds,
            watchdog_action='One Python stack dump per stalled call; no termination, retry, skip or exception suppression',
            timing_note='Watchdog scheduling/cancellation and progress output are outside measured intervals; diagnostic activity can perturb host scheduling.'),
        measurement='Entire predict call: constructor/state setup, prediction, validity checks, timestamp labels, arena release; excludes one-time initialization and result-consumer/IPC costs.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--samples',type=int,default=0)
    parser.add_argument('--watchdog-seconds',type=float,default=60.)
    args=parser.parse_args()
    require(math.isfinite(args.watchdog_seconds) and args.watchdog_seconds>=10,'Watchdog must be finite and at least 10 seconds')
    require(args.samples==0 or args.samples>=1000,'Meaningful benchmark requires at least 1000 measured calls')
    faulthandler.enable()
    print('V6 Option A: checking protected hashes and static contract...',flush=True)
    before=protected()
    static_contract()
    print('Initializing pinned RocketSim provider...',flush=True)
    faulthandler.dump_traceback_later(args.watchdog_seconds,repeat=False)
    start=time.perf_counter_ns()
    provider=ConstrainedProvider()
    initialization_ms=(time.perf_counter_ns()-start)/1e6
    faulthandler.cancel_dump_traceback_later()
    print(f'Initialization complete ({initialization_ms:.3f}ms). Running 18 contract checks...',flush=True)
    faulthandler.dump_traceback_later(args.watchdog_seconds,repeat=False)
    checks=contract_checks(provider)
    faulthandler.cancel_dump_traceback_later()
    print(f"Contract checks PASS: {checks['cases']} cases. Starting latency stage.",flush=True)
    latency=benchmark(provider,args.samples,args.watchdog_seconds) if args.samples else None
    require(protected()==before,'Protected files changed')
    report=dict(experiment='V6_OPTION_A',status='CONTRACT_PASS_LATENCY_PENDING' if latency is None else 'CONTRACT_AND_LATENCY_MEASURED',
        provider_identity=provider.identity,initialization_ms=initialization_ms,contract_checks=checks,
        benchmark=latency,protected_hashes=len(before),
        code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__),ROOT/'training/v6_prediction_test/constrained_provider.py')},
        timestamp_rule='game_seconds[i] = T + i / 120; no existing provider timestamps rewritten',
        scope='Constrained experiment only; final provider not accepted, no V7/provider parity, demonstrations or ML.')
    suffix='benchmark' if latency else 'contract'
    path=ROOT/f'training/reports/v6_option_a_{suffix}_20261004.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f"V6 Option A: {report['status']}; {checks['cases']} contract cases; {len(before)} protected hashes unchanged.")
    if latency:
        print('Latency ms:',{k:latency[k] for k in ('samples','mean_ms','median_ms','p90_ms','p95_ms','p99_ms','maximum_ms','over_one_physics_tick','over_nominal_two_tick_callback')})
    print('Report:',path)


if __name__=='__main__':
    try:
        main()
    finally:
        faulthandler.cancel_dump_traceback_later()
        protected()
