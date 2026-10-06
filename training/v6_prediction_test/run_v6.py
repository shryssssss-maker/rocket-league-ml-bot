"""V6 teacher selection only; native RocketSim sampling remains a separate probe."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'training/v4_temporal_test'))
import run_v4 as v4
v3,flat = v4.v3,v4.flat
original_select = v3.original.find_slice_at_time


def reconstructed_select(prediction,requested):
    # Preserve empty-list exception, Python int truncation, and no interpolation.
    first = prediction.slices[0].game_seconds
    index = int((requested-first)*120)
    if 0 <= index < len(prediction.slices):
        return prediction.slices[index]


def prediction(first,count,wire,irregular=False):
    slices = [flat.PredictionSlice(game_seconds=first+(i/120 if not irregular else i/60+1),
        physics=flat.Physics(location=flat.Vector3(i,-i,100+i%3))) for i in range(count)]
    if irregular and slices:
        slices[0] = flat.PredictionSlice(game_seconds=first,physics=slices[0].physics)
    value = flat.BallPrediction(slices=slices)
    return flat.CorePacket.unpack(flat.CorePacket(value).pack()).message if wire else value


def result(fn,pred,requested):
    try:
        selected = fn(pred,requested)
        if selected is None:
            return dict(kind='none')
        return dict(kind='slice',index=int(selected.physics.location.x),
                    selected_time=float(selected.game_seconds),position=v3.xyz(selected.physics.location))
    except Exception as error:
        return dict(kind='exception',type=type(error).__name__,message=str(error))


def helper_cases():
    rows = []
    for wire in (False,True):
        for first in (0.,12.,500.):
            for count in (0,1,239,240,241,720):
                pred = prediction(first,count,wire)
                actual_first = float(pred.slices[0].game_seconds) if count else first
                indexes = (-2.,-1.,-.999,-.5,0.,.001,.999,1.,1.001,238.999,239.,239.001,
                           240.,240.001,float(count-1),float(count),float(count)+.5)
                requests = [actual_first+x/120 for x in indexes]
                for index in (0,1,239,240,count):
                    boundary = actual_first+index/120
                    requests += [math.nextafter(boundary,-math.inf),boundary,math.nextafter(boundary,math.inf)]
                for requested in requests:
                    a,b = result(original_select,pred,requested),result(reconstructed_select,pred,requested)
                    v3.require(a == b,'Helper selection mismatch')
                    if count:
                        numeric = int((requested-actual_first)*120)
                        v3.require(a['kind'] == ('slice' if 0 <= numeric < count else 'none'),'Range semantics mismatch')
                        if a['kind'] == 'slice':
                            v3.require(a['index'] == numeric,'Marker index mismatch')
                    else:
                        v3.require(a['kind']=='exception' and a['type']=='IndexError','Empty prediction exception changed')
                    rows.append(dict(wire=wire,first=actual_first,count=count,requested=requested,
                                     scaled_index=None if not count else (requested-actual_first)*120,result=a))
    # The helper uses the first time and nominal 120 Hz index, not a search
    # over actual timestamps. Deliberately irregular messages expose this.
    for wire in (False,True):
        pred = prediction(12.,241,wire,True)
        a,b = result(original_select,pred,14.),result(reconstructed_select,pred,14.)
        v3.require(a==b and a['index']==240 and a['selected_time']==17.,'Unexpected timestamp search/interpolation')
        rows.append(dict(wire=wire,first=12.,count=241,requested=14.,irregular=True,result=a))
    # Explicit regression: negative fractional scaled index truncates to zero.
    pred = prediction(12.,241,False)
    requested = 12.-.5/120
    actual = result(original_select,pred,requested)
    v3.require(actual['index']==0 and math.floor((requested-12.)*120)==-1,'Negative truncation anchor failed')
    return rows,dict(requested=requested,first=12.,scaled_index=(requested-12.)*120,
                     int_index=0,floor_index=-1)


def teacher_cases():
    rows = []
    for wire in (False,True):
        for team in (0,1):
            for now in (0.,12.,500.):
                for offset in (0.,1/120,2/120,-1/120,-1.,-5.):
                    for count in (0,240,241,720):
                        first = now+offset
                        pred = prediction(first,count,wire)
                        base = v3.packet_for((0.,0.,17.),(1700.,100.,0.),(0.,0.,0.),(0.,0.,0.),team,team)
                        packet = flat.GamePacket(players=base.players,balls=base.balls,
                            match_info=flat.MatchInfo(seconds_elapsed=now))
                        if wire:
                            packet = flat.CorePacket.unpack(flat.CorePacket(packet).pack()).message
                        actual_now = float(packet.match_info.seconds_elapsed)
                        requested = actual_now+2
                        expected = result(reconstructed_select,pred,requested)
                        agent = v4.make_original(team)
                        agent.ball_prediction = pred
                        calls = []
                        def traced(p,t):
                            calls.append(t)
                            return original_select(p,t)
                        try:
                            with patch.object(v3.original,'find_slice_at_time',traced):
                                out = v3.controls(agent.get_output(packet))
                            error = None
                        except Exception as exception:
                            out,error = None,type(exception).__name__
                        v3.require(calls == [requested],'Teacher request is not exact now+2')
                        if expected['kind']=='exception':
                            v3.require(error==expected['type']=='IndexError','Teacher empty-list exception swallowed')
                            target = None
                        else:
                            v3.require(error is None and out[0]==1,'Teacher did not chase')
                            target = list(expected['position']) if expected['kind']=='slice' else list(v3.xyz(packet.balls[0].physics.location))
                            v3.require(agent.renderer.target == target,'Teacher target/fallback mismatch')
                            expected_steer = v3.original.steer_toward_target(packet.players[team],v3.original.Vec3(*target))
                            v3.require(out[1] == expected_steer,'Teacher prediction steering mismatch')
                        actual_first = float(pred.slices[0].game_seconds) if count else None
                        rows.append(dict(wire=wire,team=team,elapsed=actual_now,nominal_first_offset=offset,
                            first_time=actual_first,count=count,requested=requested,result=expected,
                            horizon=(expected['selected_time']-actual_now) if expected['kind']=='slice' else None,
                            first_offset_ticks=(actual_first-actual_now)*120 if actual_first is not None else None,
                            target=target,controls=out,error=error))
    # Empty prediction is consulted only on the far-ball normal branch.
    guards = []
    for wire in (False,True):
        for mode in ('near_ball','missing_ball','pending_sequence'):
            agent = v4.make_original(0)
            agent.ball_prediction = prediction(0.,0,wire)
            if mode=='pending_sequence':
                agent.get_output(v4.packet(0.,0,0,wire))
            base = v3.packet_for((0.,0.,17.),(1000. if mode=='near_ball' else 1700.,100.,0.),
                                (0.,0.,1100.),(0.,0.,0.),0,0)
            p = flat.GamePacket(players=base.players,balls=[] if mode=='missing_ball' else base.balls,
                                match_info=flat.MatchInfo(seconds_elapsed=.02))
            if wire:
                p=flat.CorePacket.unpack(flat.CorePacket(p).pack()).message
            with patch.object(v3.original,'find_slice_at_time',side_effect=AssertionError('Unexpected lookup')):
                out = v3.controls(agent.get_output(p))
            guards.append(dict(wire=wire,mode=mode,controls=out))
    return rows,guards


def main():
    started = time.perf_counter()
    revision = subprocess.run(['git','-C',str(ROOT/'python-example-original'),'rev-parse','HEAD'],
                              capture_output=True,text=True,check=True).stdout.strip()
    v3.require(revision == v3.REVISION,'Reference revision changed')
    before,manifest = v3.hashes()
    v3.require(before == manifest,'Protected hash mismatch')
    helper,anchor = helper_cases()
    teacher,guards = teacher_cases()
    after,_ = v3.hashes()
    v3.require(after == before,'Protected files changed')
    report = dict(experiment='V6',status='TEACHER_SELECTION_PASS_NATIVE_SAMPLING_PENDING',
        reference_revision=revision,helper_cases=len(helper),teacher_cases=len(teacher),lookup_guard_cases=len(guards),
        helper_outcomes=dict(Counter(x['result']['kind'] for x in helper)),
        teacher_outcomes=dict(Counter(x['result']['kind'] for x in teacher)),
        selected_index_timestamp_position_exact=True,negative_truncation_anchor=anchor,
        protected_hashes=len(before),elapsed_seconds=time.perf_counter()-started,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Original helper and teacher request/target behavior; no forecasting/provider-accuracy claim.',
        helper_results=helper,teacher_results=teacher,lookup_guards=guards)
    path=ROOT/'training/reports/v6_prediction_selection_20261004.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f"V6 teacher selection PASS: {len(helper)} helper cases, {len(teacher)} teacher cases, {len(guards)} lookup guards.")
    print(f"Native sampling remains pending; {len(before)} protected hashes unchanged. Report: {path}")


if __name__ == '__main__':
    main()
