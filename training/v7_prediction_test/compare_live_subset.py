"""Read-only live-capture selector/18D comparison; no gameplay or training."""
import array
import collections
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
SESSION=ROOT/'training/v7_prediction_test/sessions/20261004_192953_891979'
sys.path.insert(0,str(ROOT/'training/live_reference_test'))
from launch import verify_sources,sha,INTERPRETER
sys.path.insert(0,str(ROOT/'python-example-original/src'))
from rlbot import flat
from util.ball_prediction_analysis import find_slice_at_time
from util.drive import steer_toward_target
from util.orientation import Orientation,relative_location
from util.vec import Vec3
sys.path.insert(0,str(ROOT/'training/v2_observation_test'))
from observation_contract import FEATURES,live_adapter

EXCLUDED={'stationary_00','wall_03','corner_07','goal_mouth_03','corner_04'}
OBS_TOL=1e-5  # Existing V2 float32 feature tolerance.
SIGN_DEADBAND=.02
STEERING_DIAGNOSTIC_LIMIT=.1  # Previously proposed V7 limit; no automatic gate approval.


def require(ok,message):
    if not ok:
        raise AssertionError(message)


def xyz(v):
    return None if v is None else [float(v.x),float(v.y),float(v.z)]


def encode_physics(p):
    return dict(position=xyz(p.location),velocity=xyz(p.velocity),angular_velocity=xyz(p.angular_velocity),
        rotation_pyr=None if p.rotation is None else [float(p.rotation.pitch),float(p.rotation.yaw),float(p.rotation.roll)])


def decode_physics(p):
    v=lambda key: None if p[key] is None else flat.Vector3(*p[key])
    native=flat.Physics(location=v('position'),velocity=v('velocity'),angular_velocity=v('angular_velocity'),
        rotation=None if p['rotation_pyr'] is None else flat.Rotator(*p['rotation_pyr']))
    require(encode_physics(native)==p,'Captured physics changed during replay reconstruction')
    return native


def digest(value):
    return hashlib.sha256(json.dumps(value,allow_nan=False).encode()).hexdigest()


def replay(row):
    """Restore captured fields, not a fabricated prediction/fixture trajectory.

    The original wire packet was not saved. Only a shared packet view of recorded
    fields consumed by selector/geometry/adapter is reconstructed; absent fields
    are not invented and the full gameplay bot is never executed.
    """
    players=[]
    for i,p in enumerate(row['players']):
        require(p['index']==i,'Player index order changed')
        players.append(SimpleNamespace(physics=decode_physics(p['physics']),team=p['team'],
            last_input=flat.ControllerState(**p['last_input'])))
    packet=SimpleNamespace(players=players,balls=[SimpleNamespace(physics=decode_physics(row['ball']))],
        match_info=SimpleNamespace(seconds_elapsed=row['canonical_T'],frame_num=row['frame'],match_phase=row['phase']))
    prediction=flat.BallPrediction(slices=[flat.PredictionSlice(game_seconds=s['game_seconds'],
        physics=decode_physics(s['physics'])) for s in row['prediction']])
    restored=[dict(game_seconds=float(s.game_seconds),physics=encode_physics(s.physics)) for s in prediction.slices]
    require(restored==row['prediction'],'Captured prediction values changed during reconstruction')
    require(digest(restored)==row['prediction_payload_sha256'],'Prediction payload integrity mismatch')
    index=row['agent_index']
    require(encode_physics(packet.players[index].physics)=={k:row['actual']['car'][k] for k in ('position','velocity','angular_velocity','rotation_pyr')},'Own packet state differs from actual capture')
    require(packet.match_info.seconds_elapsed==row['canonical_T'],'Canonical received time changed')
    return packet,prediction


def candidate_select(prediction,elapsed):
    # Independent diagnostic wrapper around the proposed live adapter.
    # V2 live_adapter consumes a preselected slice; this is not deployed code.
    first_slice_time=prediction.slices[0].game_seconds
    index=int((elapsed + 2 - first_slice_time) * 120)
    if 0 <= index < len(prediction.slices):
        return index,prediction.slices[index]
    return index,None


def previous_diagnostic_action(row):
    emitted=row['diagnostic_previous_control_submission']
    require(emitted is not None,'No observed diagnostic control submission')
    require(emitted['callback_counter']<row['callback_counter'],'Previous action is not past information')
    c=emitted['controls']
    require(c==dict(throttle=0.,steer=0.,pitch=0.,yaw=0.,roll=0.,jump=False,boost=False,handbrake=False),'This subset requires actual neutral diagnostic history')
    return 0,float(c['steer'])


def oracle_features(packet,index,prediction,selected,previous_T,mode,steer):
    # Independent schema oracle, using original geometry utilities. It does
    # not call candidate basis/build/live_adapter or candidate selector.
    car=packet.players[index].physics
    now=packet.match_info.seconds_elapsed
    out=[0.]*18
    out[7]=Vec3(car.velocity).length()/2300
    out[12]=0. if previous_T is None else (now-previous_T)/(1/60)
    out[13+mode]=1.
    out[17]=steer
    if packet.balls:
        center=Vec3(car.location)
        ori=Orientation(car.rotation)
        ball=Vec3(packet.balls[0].physics.location)
        rel=relative_location(center,ori,ball)
        out[:3]=[rel.x/6000,rel.y/6000,rel.z/6000]
        out[6]=center.dist(ball)/6000
        out[8]=1.
        if selected is not None:
            rel=relative_location(center,ori,Vec3(selected.physics.location))
            out[3:6]=[rel.x/6000,rel.y/6000,rel.z/6000]
            out[9]=1.
            out[10]=(selected.game_seconds-now)/2
            out[11]=(prediction.slices[0].game_seconds-now)/(1/120)
    return list(array.array('f',out)),out


def compare(row):
    packet,prediction=replay(row)
    before=digest(row)
    packet_before=[encode_physics(p.physics) for p in packet.players]
    prediction_before=[dict(game_seconds=s.game_seconds,physics=encode_physics(s.physics)) for s in prediction.slices]
    selected=find_slice_at_time(prediction,packet.match_info.seconds_elapsed+2)
    require(selected is not None,'Original selector returned no slice on eligible valid snapshot')
    # Observe which object the unmodified helper returned; do not independently
    # compute the reference index using the candidate's expression.
    reference_index=next(i for i,s in enumerate(prediction.slices) if s is selected)
    candidate_index,candidate=candidate_select(prediction,packet.match_info.seconds_elapsed)
    require(candidate is not None,'Candidate selector returned no slice')
    mode,steer=previous_diagnostic_action(row)
    context=dict(previous_elapsed=row['previous_callback_T'],previous_mode=mode,previous_steer=steer,
        prediction_valid=candidate is not None,first_time=prediction.slices[0].game_seconds)
    observation=list(live_adapter(packet,row['agent_index'],context,candidate))
    expected,unrounded=oracle_features(packet,row['agent_index'],prediction,selected,
        row['previous_callback_T'],mode,steer)
    require(len(observation)==len(expected)==18,'Observation dimension mismatch')
    require(all(math.isfinite(x) for x in observation+expected),'Nonfinite observation')
    car=packet.players[row['agent_index']]
    distance=Vec3(car.physics.location).dist(Vec3(packet.balls[0].physics.location))
    # Hypothetical normal Chase with no pending sequence; NOT a temporal-teacher
    # gameplay evaluation. Far/near uses actual received current-ball distance.
    target=selected.physics.location if distance>1500 else packet.balls[0].physics.location
    reference_steer=steer_toward_target(car,Vec3(target))
    candidate_distance=math.sqrt(sum((a-b)**2 for a,b in zip(row['actual']['car']['position'],row['actual']['ball']['position'])))
    offset=3 if candidate_distance>1500 else 0
    candidate_steer=max(-1.,min(1.,5*math.atan2(observation[offset+1],observation[offset])))
    # This is the original control-range limit, never a prediction-index clamp.
    target_delta=[a-b for a,b in zip(xyz(candidate.physics.location),xyz(selected.physics.location))]
    errors=dict(index=abs(candidate_index-reference_index),timestamp=abs(candidate.game_seconds-selected.game_seconds),
        target_position=math.sqrt(sum(x*x for x in target_delta)),steering=abs(candidate_steer-reference_steer))
    feature_errors=[abs(a-b) for a,b in zip(observation,expected)]
    sign_disagreement=reference_steer*candidate_steer<0 and abs(reference_steer)>SIGN_DEADBAND and abs(candidate_steer)>SIGN_DEADBAND
    failures=[]
    for key in ('index','timestamp','target_position'):
        if errors[key]!=0:
            failures.append(key+' mismatch')
    if reference_index!=row['selected_index'] or selected.game_seconds!=row['selected']['game_seconds'] or xyz(selected.physics.location)!=row['selected']['physics']['position']:
        failures.append('Reference replay differs from recorded live selector result')
    for i,error in enumerate(feature_errors):
        if error>OBS_TOL:
            failures.append(f'Feature {i} {FEATURES[i]} exceeds existing V2 tolerance')
    for i in (8,9,13,14,15,16,17):
        if observation[i]!=expected[i]:
            failures.append(f'Exact mask/previous-action field {i} mismatch')
    if (candidate_distance>1500)!=(distance>1500):
        failures.append('Actual-state far-ball branch mismatch')
    if errors['steering']>STEERING_DIAGNOSTIC_LIMIT:
        failures.append('Induced Chase steering exceeds prior provisional V7 diagnostic limit')
    if sign_disagreement:
        failures.append('Steering-sign disagreement outside deadband')
    require(before==digest(row),'Captured input mutated')
    require(packet_before==[encode_physics(p.physics) for p in packet.players],'Shared replay packet mutated')
    require(prediction_before==[dict(game_seconds=s.game_seconds,physics=encode_physics(s.physics)) for s in prediction.slices],'Shared replay prediction mutated')
    return dict(case=row['case']['id'],group=row['case']['group'],frame=row['frame'],canonical_T=row['canonical_T'],
        reference_index=reference_index,candidate_index=candidate_index,
        reference_timestamp=selected.game_seconds,candidate_timestamp=candidate.game_seconds,
        reference_target=xyz(selected.physics.location),candidate_target=xyz(candidate.physics.location),
        reference_steering=reference_steer,candidate_observation_induced_steering=candidate_steer,
        errors=errors,feature_errors=feature_errors,reference_features=expected,candidate_features=observation,
        reference_before_float32=unrounded,sign_disagreement=sign_disagreement,failures=failures,
        far_ball=distance>1500,teacher_gameplay_decision_equivalence='not_tested',
        previous_action_provenance='Actual past neutral diagnostic submission; not teacher sequence memory',
        prediction_receipt=row['prediction_receipt'],packet_receipt=row['packet_receipt'])


def aggregate(rows):
    def summary(values):
        return dict(maximum=max(values),median=statistics.median(values))
    return dict(snapshots=len(rows),fixtures=len({r['case'] for r in rows}),
        differences={key:summary([r['errors'][key] for r in rows]) for key in ('index','timestamp','target_position','steering')},
        feature_differences={name:dict(index=i,**summary([r['feature_errors'][i] for r in rows])) for i,name in enumerate(FEATURES)},
        failed_snapshots=sum(bool(r['failures']) for r in rows),
        steering_sign_disagreements=sum(r['sign_disagreement'] for r in rows),
        far_ball_snapshots=sum(r['far_ball'] for r in rows))


def main():
    require(Path(sys.executable).resolve()==INTERPRETER.resolve(),'Use the existing live venv interpreter')
    protected_before=verify_sources()
    files=('capture_summary.json','captures.jsonl','delivery_events.jsonl','metadata.json','agent_metadata.json','run_summary.json')
    inputs={name:sha(SESSION/name) for name in files}
    saved=json.loads((SESSION/'capture_summary.json').read_text())
    eligible={r['id']:r for r in saved['fixture_results'] if r['status']=='verified' and r['coverage_eligible'] and r['valid_snapshots']==3}
    require(len(eligible)==115 and not EXCLUDED.intersection(eligible),'Approved fixture subset mismatch')
    require(set(r['id'] for r in saved['fixture_results'])-set(eligible)==EXCLUDED,'Unexpected excluded cases')
    results=[]
    errors=[]
    seen=collections.Counter()
    print('V7 bounded comparison ONLY: reading 115 approved fixtures / 345 live snapshots.',flush=True)
    with (SESSION/'captures.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['case']['id'] not in eligible:
                continue
            require(row['snapshot_status']=='valid','Eligible row was not captured as valid')
            require(row['case']['group']==eligible[row['case']['id']]['group'],'Group label mismatch')
            seen[row['case']['id']]+=1
            try:
                results.append(compare(row))
            except Exception as error:
                errors.append(dict(case=row['case']['id'],frame=row['frame'],group=row['case']['group'],
                    error_type=type(error).__name__,error=repr(error)))
            if sum(seen.values())%100==0:
                print(f'Compared {sum(seen.values())}/345 snapshots; exceptions={len(errors)}',flush=True)
    require(sum(seen.values())==345 and all(seen[k]==3 for k in eligible),'Eligible sample multiplicity mismatch')
    require(verify_sources()==protected_before,'Protected source change')
    require(inputs=={name:sha(SESSION/name) for name in files},'Saved capture artifacts changed')
    groups=sorted({r['group'] for r in eligible.values()})
    report=dict(status='COMPARISON_COMPLETE_AWAITING_REVIEW',V7_pass_declared=False,
        session=str(SESSION),subset=dict(fixtures=115,snapshots=345,excluded=sorted(EXCLUDED)),
        input_sha256=inputs,protected_hashes_unchanged=len(protected_before),
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'training/v2_observation_test/observation_contract.py',ROOT/'python-example-original/src/util/ball_prediction_analysis.py',ROOT/'python-example-original/src/util/drive.py',ROOT/'python-example-original/src/util/orientation.py',ROOT/'python-example-original/src/util/vec.py')},
        versions={n:importlib.metadata.version(n) for n in ('rlbot','rlbot-flatbuffers')},
        criteria=dict(selector_index_timestamp_target='exact',observation_float32_tolerance=OBS_TOL,
            masks_previous_action='exact',prior_provisional_steering_limit=STEERING_DIAGNOSTIC_LIMIT,sign_deadband=SIGN_DEADBAND),
        method=dict(selector='Original helper versus independent exact-expression diagnostic wrapper',
            observation='Existing V2 live_adapter versus independently written schema oracle using original Orientation/Vec3 utilities',
            steering='Original steer_toward_target versus induced steering from candidate float32 local observation coordinates; no neural policy',
            replay='Shared read-only replay of captured fields and delivered slice values, with exact restoration checks; raw wire packet and unused fields were not captured',
            memory='Only captured past neutral diagnostic submissions; no teacher sequence reconstruction',
            branch='Conditional normal Chase on actual state assuming no pending sequence; not full gameplay equivalence'),
        overall=aggregate(results) if results else None,
        groups={g:aggregate([r for r in results if r['group']==g]) if any(r['group']==g for r in results) else None for g in groups},
        exceptions=errors,failures=[r for r in results if r['failures']],
        sign_disagreements=[r for r in results if r['sign_disagreement']],rows=results,
        limitations=['All eligible snapshots have ball present and valid prediction; missing/invalid masks not exercised here.',
            'Previous action is Neutral only; Chase/Jump/Front-dodge history and recurrent memory are not exercised.',
            'Human controls may be nonneutral; no fixed-input rollout, collision forecast accuracy or temporal-teacher equivalence claim.',
            'No live provider replacement, generated forecasts, prediction timestamp correction, production integration or ML.'])
    path=ROOT/'training/reports/v7_live_subset_comparison_20261004.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    write_markdown(report,path.with_suffix('.md'))
    print('RESULT:',json.dumps(dict(compared=len(results),exceptions=len(errors),failed_snapshots=sum(bool(r['failures']) for r in results),sign_disagreements=len(report['sign_disagreements']),overall=report['overall']['differences'] if results else None)),flush=True)
    print('Report:',path,flush=True)
    print('V7 NOT DECLARED PASSED. Stop for review.',flush=True)


def write_markdown(report,path):
    a=report['overall']
    lines=['# Bounded V7 live snapshot comparison','',
        '**Comparison complete; V7 pass is not declared. User review required.**','',
        'Only the unchanged-policy 115 verified fixtures / 345 eligible snapshots were used. The four boost-failed fixtures and corner_04 were excluded. Capture artifacts and all 43 protected hashes remained unchanged.','',
        'The captured packet fields and all delivered prediction slice values were restored without numerical changes. Both paths received the same shared replay objects and actual canonical elapsed time; mutation checks passed. This is replay of recorded live data, not a synthetic forecast. Original raw wire packets and unused packet fields were not saved, so full packet fidelity is not claimed.','',
        'Reference: unchanged original find_slice_at_time; selected index observed from the returned slice object. Candidate: isolated exact-expression selector wrapper plus the unchanged V2 live_adapter. No production adapter selector previously existed; this diagnostic wrapper is not deployed.','',
        'Reference 18D oracle independently writes the declared schema using original Orientation/relative_location/Vec3. Features are compared after their declared float32 cast. Original Vec3/basis intermediates and the candidate builder can differ in floating-point precision; all differences are recorded, never corrected.','',
        'Induced steering compares original steer_toward_target with steering computed from candidate normalized float32 local coordinates under the same Chase formula. No student/model runs. Chase is conditional on no pending sequence and uses the actual current-ball distance; this does not establish teacher gameplay decisions or temporal state.','',
        'Previous-action fields come only from the actual prior diagnostic control submission. Every retained prior action is Neutral, so other modes and teacher memory are outside this comparison.','',
        '## Overall numerical differences','',
        '| Quantity | Maximum absolute difference | Median absolute difference |','|---|---:|---:|']
    if a:
        for k,d in a['differences'].items():
            lines.append(f"| {k} | {d['maximum']:.12g} | {d['median']:.12g} |")
    lines+=['','Position is Euclidean error in UU; timestamps are seconds; steering is controller magnitude. Feature tolerance is the existing V2 1e-5; masks/previous-action values and selector outputs require exact agreement. The prior provisional V7 0.1 steering limit is shown only as a diagnostic threshold, not automatic gate approval.','',
        '## Scenario groups','',
        '| Group | Fixtures | Snapshots | Exceptions | Failed snapshots | Sign disagreements | Max time error (s) | Max target error (UU) | Max steer error | Median steer error |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g,v in report['groups'].items():
        e=sum(r['group']==g for r in report['exceptions'])
        if v:
            d=v['differences']
            lines.append(f"| {g} | {v['fixtures']} | {v['snapshots']} | {e} | {v['failed_snapshots']} | {v['steering_sign_disagreements']} | {d['timestamp']['maximum']:.9g} | {d['target_position']['maximum']:.9g} | {d['steering']['maximum']:.9g} | {d['steering']['median']:.9g} |")
    lines+=['','Every group\'s maximum/median for all metrics and all 18 features, plus every individual snapshot, is in the accompanying JSON report.','',
        '## Each 18D feature','', '| Index | Feature | Maximum difference | Median difference |','|---:|---|---:|---:|']
    if a:
        for name,d in a['feature_differences'].items():
            lines.append(f"| {d['index']} | {name} | {d['maximum']:.12g} | {d['median']:.12g} |")
    lines+=['','## All failures and steering-sign disagreements','',
        f"Exceptions: {len(report['exceptions'])}. Failed snapshots: {len(report['failures'])}. Sign disagreements outside ±0.02: {len(report['sign_disagreements'])}.",'']
    for r in report['exceptions']+report['failures']:
        lines.append(f"- {r['case']} frame {r['frame']}: {r.get('error',r.get('failures'))}")
    for r in report['sign_disagreements']:
        lines.append(f"- SIGN {r['case']} frame {r['frame']}: original={r['reference_steering']}, candidate={r['candidate_observation_induced_steering']}")
    lines+=['','## Limits of the evidence','']+['- '+s for s in report['limitations']]
    lines+=['','No V7 pass declaration, recapture, training or deployment followed this comparison.','',
        '## Exact sources','',
        '- Original selector: python-example-original/src/util/ball_prediction_analysis.py:find_slice_at_time.',
        '- Original geometry/steering: src/util/orientation.py:Orientation/relative_location; src/util/vec.py:Vec3; src/util/drive.py:steer_toward_target.',
        '- Candidate schema: training/v2_observation_test/observation_contract.py:FEATURES/build/live_adapter.',
        '- Diagnostic wrapper and oracle: training/v7_prediction_test/compare_live_subset.py:candidate_select/oracle_features/compare.',
        '- Input capture hashes, dependency versions and exact code hashes are included in JSON.']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
