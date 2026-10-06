"""V2 only: synthetic matched states, no physics, prediction lookup or model."""
import argparse
import array
import copy
import hashlib
import importlib.metadata
import json
import math
import random
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode=True
from observation_contract import FEATURES,basis,build,live_adapter,simulation_adapter
ROOT=Path(__file__).resolve().parents[2]
TOL=1e-5
SEED=20261004


def check(ok,message):
    if not ok:
        raise AssertionError(message)


def f32(x):
    return struct.unpack('<f',struct.pack('<f',x))[0]


def fixtures():
    rng=random.Random(SEED)
    orientations=[(0,0),(.35,-.4),(math.pi/2,0),(0,math.pi/2),(0,math.pi)]
    cases=[]
    for n in range(500):
        availability=(n//25+n%5)%5  # Independent of orientation category.
        if n<200:
            pitch,roll=orientations[(n//5)%5]
            yaw=(n//25)*math.pi/4
            distance=(0,500,1500,1700,10000)[n%5]
            pos=[100.,-200.,17. if n%25<10 else 1000.]
            ball=[pos[0]+distance,pos[1]+distance*.2,pos[2]+distance*.1]
        else:
            pitch,roll=rng.uniform(-math.pi/2,math.pi/2),rng.uniform(-math.pi,math.pi)
            yaw=rng.uniform(-math.pi,math.pi)
            pos=[rng.uniform(-4096,4096),rng.uniform(-5120,5120),rng.uniform(17,1900)]
            ball=[rng.uniform(-5000,5000),rng.uniform(-6200,6200),rng.uniform(0,2200)]
        state=dict(position=[f32(x) for x in pos],velocity=[f32(rng.uniform(-2500,2500)) for _ in range(3)],
                   pyr=[f32(x) for x in (pitch,yaw,roll)],ball_position=[f32(x) for x in ball],
                   ball_velocity=[f32(rng.uniform(-4000,4000)) for _ in range(3)],ball_present=availability not in (3,4))
        now=f32(32+n*.125)
        init=n%17==0
        mode=0 if init else n%4
        ctx=dict(elapsed=now,previous_elapsed=None if init else f32(now-(1,2,3,5)[n%4]/120),
                 previous_mode=mode,previous_steer=f32(rng.uniform(-1,1)) if mode==1 else 0.,
                 prediction_valid=availability in (0,3),
                 prediction_position=[f32(x+rng.uniform(-2500,2500)) for x in ball],
                 selected_time=f32(now+2+(n%3-1)/120),first_time=f32(now+(1,-8,0)[n%3]/120))
        if n%20==0:
            ctx['prediction_position']=state['position'][:]
        for team in (0,1):
            cases.append(dict(id=len(cases),team=team,index=n%2,state=state,context=ctx))
    return cases


def live_worker(cases):
    from rlbot import flat
    sys.path.insert(0,str(ROOT/'python-example-original/src'))
    from util.orientation import Orientation,relative_location
    from util.vec import Vec3
    rows,oracles=[],[]
    guards=[]
    def xyz(v):
        return flat.Vector3(*v)
    for c in cases:
        s,ctx=c['state'],c['context']
        controlled=flat.PlayerInfo(team=c['team'],physics=flat.Physics(location=xyz(s['position']),
                    velocity=xyz(s['velocity']),rotation=flat.Rotator(*s['pyr'])))
        opponent=flat.PlayerInfo(team=1-c['team'],physics=flat.Physics(location=flat.Vector3(x=3500,y=-4000)))
        players=[opponent,opponent]
        players[c['index']]=controlled
        packet=flat.GamePacket(players=players,
            balls=[flat.BallInfo(physics=flat.Physics(location=xyz(s['ball_position']),velocity=xyz(s['ball_velocity'])))] if s['ball_present'] else [],
            match_info=flat.MatchInfo(seconds_elapsed=ctx['elapsed']))
        packet=flat.CorePacket.unpack(flat.CorePacket(packet).pack()).message
        selected=None
        native_context=dict(ctx)
        if ctx['prediction_valid']:
            # Preselected fixture slices only; no indexing or prediction run.
            prediction=flat.BallPrediction(slices=[flat.PredictionSlice(game_seconds=ctx['first_time']),
                flat.PredictionSlice(game_seconds=ctx['selected_time'],
                                    physics=flat.Physics(location=xyz(ctx['prediction_position'])))])
            prediction=flat.CorePacket.unpack(flat.CorePacket(prediction).pack()).message
            native_context['first_time']=prediction.slices[0].game_seconds
            selected=prediction.slices[1]
        out=live_adapter(packet,c['index'],native_context,selected)
        if c['id']==0:
            for name,index,slice_value in [('missing live car',-1,selected),('missing selected live slice',c['index'],None)]:
                try:live_adapter(packet,index,native_context,slice_value)
                except ValueError as error:guards.append(dict(case=name,error=str(error)))
                else:raise AssertionError(f'Live adapter accepted {name}')
        rows.append(list(out))
        # Independent original geometry plus independently written schema
        # expectations; this oracle never calls shared build()/basis().
        car=packet.players[c['index']].physics
        ori=Orientation(car.rotation)
        expected=[0.]*18
        expected[7]=Vec3(car.velocity).length()/2300
        expected[12]=0 if ctx['previous_elapsed'] is None else (packet.match_info.seconds_elapsed-ctx['previous_elapsed'])*60
        expected[13+ctx['previous_mode']]=1
        expected[17]=ctx['previous_steer']
        if packet.balls:
            ball=packet.balls[0].physics.location
            v=relative_location(Vec3(car.location),ori,Vec3(ball))
            expected[:3]=[v.x/6000,v.y/6000,v.z/6000]
            expected[6]=Vec3(car.location).dist(Vec3(ball))/6000
            expected[8]=1
            if ctx['prediction_valid']:
                v=relative_location(Vec3(car.location),ori,Vec3(*ctx['prediction_position']))
                expected[3:6]=[v.x/6000,v.y/6000,v.z/6000]
                expected[9]=1
                expected[10]=(ctx['selected_time']-packet.match_info.seconds_elapsed)/2
                expected[11]=(ctx['first_time']-packet.match_info.seconds_elapsed)*120
        oracles.append(list(array.array('f',expected)))
    return dict(rows=rows,oracles=oracles,features=FEATURES,adapter_guards=guards,
                versions={x:importlib.metadata.version(x) for x in ('rlbot','rlbot-flatbuffers')})


def sim_worker(cases):
    import numpy as np
    import RocketSim as rs
    from rlgym.rocket_league.api import GameState,Car,PhysicsObject
    rows,alternate=[],[]
    guards=[]
    for c in cases:
        s=c['state']
        state=GameState()
        car=Car()
        car.team_num=c['team']
        car.physics=PhysicsObject()
        car.physics.position=np.asarray(s['position'],dtype=np.float32)
        car.physics.linear_velocity=np.asarray(s['velocity'],dtype=np.float32)
        # Actual native Angle API takes yaw/pitch/roll; conversion matches
        # RocketSimEngine._get_state's transpose from the native matrix.
        pitch,yaw,roll=s['pyr']
        angle=rs.Angle(yaw=yaw,pitch=pitch,roll=roll)
        car.physics.rotation_mtx=np.ascontiguousarray(angle.as_rot_mat().as_numpy().reshape(3,3).transpose())
        state.cars={'controlled':car,'opponent':Car()}
        state.ball=PhysicsObject()
        state.ball.position=np.asarray(s['ball_position'],dtype=np.float32)
        state.ball.linear_velocity=np.asarray(s['ball_velocity'],dtype=np.float32)
        selected=None
        if c['context']['prediction_valid']:
            selected=rs.BallState()
            selected.pos=rs.Vec(*c['context']['prediction_position'])
        rows.append(list(simulation_adapter(state,'controlled',c['context'],s['ball_present'],selected)))
        if c['id']==0:
            for name,agent,pred in [('missing simulation car','missing',selected),('missing selected simulation state','controlled',None)]:
                try:simulation_adapter(state,agent,c['context'],s['ball_present'],pred)
                except ValueError as error:guards.append(dict(case=name,error=str(error)))
                else:raise AssertionError(f'Simulation adapter accepted {name}')
        car.physics.euler_angles=np.asarray(s['pyr'],dtype=np.float32)
        alternate.append(list(simulation_adapter(state,'controlled',c['context'],s['ball_present'],selected)))
    return dict(rows=rows,euler_rows=alternate,features=FEATURES,adapter_guards=guards,
                versions={x:importlib.metadata.version(x) for x in ('RocketSim','rlgym-rocket-league','numpy')})


def compare(a,b,label):
    check(len(a)==len(b)==18,f'{label}: dimension')
    check(all(math.isfinite(v) for v in a+b),f'{label}: nonfinite')
    errors=[abs(x-y) for x,y in zip(a,b)]
    check(max(errors)<=TOL,f'{label}: {max(errors)} exceeds tolerance at {FEATURES[errors.index(max(errors))]}')
    for i in (8,9,13,14,15,16):
        check(a[i]==b[i],f'{label}: mask/one-hot mismatch')
    return errors


def anchors_and_rejections():
    raw=dict(position=[0.,0.,0.],velocity=[0.,0.,2300.],basis=basis([0.,0.,0.]),
             ball_present=True,ball_position=[600.,-1200.,1800.])
    ctx=dict(elapsed=12.5,previous_elapsed=12.5-1/60,previous_mode=1,previous_steer=.25,
             prediction_valid=True,prediction_position=[1200.,600.,-600.],selected_time=14.5,first_time=12.5+1/120)
    expected=[.1,-.2,.3,.2,.1,-.1,math.sqrt(14)/10,1.,1.,1.,1.,1.,1.,0.,1.,0.,0.,.25]
    compare(list(build(raw,ctx)),expected,'all-feature analytic anchor')
    anchor_count=1
    for feature in ('missing_ball','missing_prediction','initialization','zero_prediction'):
        r,k,e=copy.deepcopy(raw),copy.deepcopy(ctx),expected[:]
        if feature=='missing_ball':
            r['ball_present']=False
            e[:7]=[0.]*7
            e[8:12]=[0.]*4
        elif feature=='missing_prediction':
            k['prediction_valid']=False
            e[3:6]=[0.]*3
            e[9:12]=[0.]*3
        elif feature=='initialization':
            k.update(previous_elapsed=None,previous_mode=0,previous_steer=0.)
            e[12:]=[0.,1.,0.,0.,0.,0.]
        else:
            k['prediction_position']=[0.,0.,0.]
            e[3:6]=[0.]*3  # Valid zero target is not missing information.
        compare(list(build(r,k)),e,feature)
        anchor_count+=1
    for pyr,ball,relative in [([0,0,0],[6000,0,0],[1,0,0]),([0,0,0],[0,6000,0],[0,1,0]),
                              ([0,math.pi/2,0],[6000,0,0],[0,-1,0]),
                              ([math.pi/2,0,0],[0,0,6000],[1,0,0]),([0,0,math.pi],[0,0,6000],[0,0,-1])]:
        r=copy.deepcopy(raw)
        r.update(basis=basis(pyr),ball_position=ball)
        check(max(abs(a-b) for a,b in zip(build(r,ctx)[:3],relative))<=TOL,'Analytic axis/sign anchor')
        anchor_count+=1
    bad=[]
    modifications=[('position','position',[0,0]),('nonfinite position','position',[0,0,float('nan')]),
                   ('nonfinite velocity','velocity',[0,0,float('inf')]),('bad basis','basis',[[1,0,0]]*3),
                   ('missing ball coordinates','ball_position',None),('nonboolean mask','ball_present',1)]
    for name,key,value in modifications:
        r=copy.deepcopy(raw);r[key]=value
        bad.append((name,r,ctx))
    for name,key,value in [('missing prediction coordinates','prediction_position',None),('missing slice time','selected_time',None),
                           ('missing first time','first_time',None),('invalid mode','previous_mode',4),
                           ('unsupported previous steering','previous_steer',1.1),('duplicate clock','previous_elapsed',12.5),
                           ('backward clock','previous_elapsed',13.),('nonfinite time','elapsed',float('nan')),
                           ('nonboolean prediction mask','prediction_valid',1)]:
        k=copy.deepcopy(ctx);k[key]=value
        bad.append((name,raw,k))
    k=copy.deepcopy(ctx);k.update(previous_mode=0,previous_steer=.25)
    bad.append(('non-Chase previous steering',raw,k))
    rejected=[]
    for name,r,k in bad:
        try:build(r,k)
        except (ValueError,TypeError) as error:rejected.append(dict(case=name,error=str(error)))
        else:raise AssertionError(f'Invalid observation accepted: {name}')
    # Verify the comparison catches familiar errors rather than only accepting
    # matching paths. These are comparator probes, not production mutations.
    detected=[]
    base=list(build(raw,ctx))
    for name,i,value in [('right sign',1,-base[1]),('wrong scale',0,base[0]*2),('mask lost',8,0),
                          ('mode order',14,0),('time scale',12,base[12]/120),('distance clipped',6,0)]:
        mutant=base[:];mutant[i]=value
        try:compare(base,mutant,name)
        except AssertionError:detected.append(name)
        else:raise AssertionError(f'Comparator missed mutation: {name}')
    r=copy.deepcopy(raw);r['velocity']=[0.,0.,4600.];r['ball_position']=[12000.,0.,0.]
    check(build(r,ctx)[6]==2 and build(r,ctx)[7]==2,'Silent clipping detected')
    return anchor_count+1,rejected,detected


def hashes():
    m=json.loads((ROOT/'training/live_reference_test/source_hashes.json').read_text(encoding='utf-8'))
    for name,digest in m.items():
        check(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,f'Protected source changed: {name}')
    return len(m)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',choices=('live','sim'),help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        cases=json.load(sys.stdin)
        json.dump(live_worker(cases) if args.worker=='live' else sim_worker(cases),sys.stdout)
        return
    start=time.perf_counter();hashes()
    cases=fixtures();results={}
    for name,env in [('live','python-example'),('sim','training')]:
        proc=subprocess.run([str(ROOT/env/'venv/Scripts/python.exe'),'-B',str(Path(__file__).resolve()),'--worker',name],
                            input=json.dumps(cases),capture_output=True,text=True,check=True)
        results[name]=json.loads(proc.stdout)
        check(results[name]['features']==list(FEATURES),'Schema names/order differ')
    maximum=[0.]*18;oracle_max=0.;euler_max=0.
    for i in range(1000):
        live=results['live']['rows'][i]
        for j,e in enumerate(compare(live,results['sim']['rows'][i],f'Native matrix pair {i}')):maximum[j]=max(maximum[j],e)
        oracle_max=max(oracle_max,max(compare(live,results['live']['oracles'][i],f'Original helper oracle {i}')))
        euler_max=max(euler_max,max(compare(live,results['sim']['euler_rows'][i],f'RLGym Euler pair {i}')))
        if i%2:
            check(live==results['live']['rows'][i-1],'Live team-dependent inversion')
            check(results['sim']['rows'][i]==results['sim']['rows'][i-1],'Simulation team-dependent inversion')
    anchors,rejected,detected=anchors_and_rejections()
    summary=dict(experiment='V2',status='PASS',seed=SEED,paired_states=1000,unique_physical_states=500,
                 features=FEATURES,dtype='float32',dimension=18,tolerance=TOL,
                 maximum_normalized_feature_difference=max(maximum),per_feature_maximum_difference=dict(zip(FEATURES,maximum)),
                 maximum_original_helper_oracle_difference=oracle_max,maximum_rlgym_euler_path_difference=euler_max,
                 analytic_anchors=anchors,rejected_invalid_inputs=rejected,comparator_mutations_detected=detected,
                 adapter_guard_cases=results['live']['adapter_guards']+results['sim']['adapter_guards'],
                 coverage=dict(blue_states=500,orange_states=500,structured_states=400,seeded_random_states=600,
                    ball_present=sum(c['state']['ball_present'] for c in cases),
                    ball_missing=sum(not c['state']['ball_present'] for c in cases),
                    effective_valid_prediction=sum(c['state']['ball_present'] and c['context']['prediction_valid'] for c in cases)),
                 protected_hashes=hashes(),versions={k:v['versions'] for k,v in results.items()},
                 code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                 fixture_sha256=hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest(),
                 elapsed_seconds=time.perf_counter()-start,
                 scope='Synthetic matched native objects; supplied selection/time/action context; no physics, prediction lookup, live match or training.')
    output=ROOT/'training/reports/v2_observation_parity_20261004.json'
    output.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f'V2 PASS: 1000 paired states; maximum native-path difference {max(maximum):.17g}; limit {TOL:g}.')
    print(f'Original geometry oracle difference {oracle_max:.17g}; Euler-path difference {euler_max:.17g}.')
    print(f'{anchors} analytic anchors; {len(rejected)} invalid inputs rejected; {len(detected)} comparator mutations detected.')
    print(f'43 protected hashes unchanged. Report: {output}')


if __name__=='__main__':main()
