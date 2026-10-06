"""Versioned read-only diagnostics. No torch, model loading, fitting or live SDK."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
WORK=Path(__file__).resolve().parent
REPORT=ROOT/'training/reports/v13_targeted_diagnostics_20261005'
SESSION=ROOT/'training/behavior_cloning/v13_dagger/phase4_full_match_v1/sessions/20261005_071223_978853'
FORENSICS=ROOT/'training/reports/v13_full_match_failure_forensics_20261005.json'
DATASET=ROOT/'training/behavior_cloning/datasets/v10_live_1v1_18d_20261005_v1'
MANIFEST_PIN='080943fd8f9a81ada800e0a2f9ff3593b67f66d32272211d67030c03cf433a83'
TRAIN=('pilot_01','v10_001','v10_004','v10_006')
FEATURES=['ball_forward','ball_right','ball_up','prediction_forward','prediction_right','prediction_up',
          'ball_distance','car_speed','ball_present','prediction_valid','prediction_horizon','prediction_first_offset','callback_dt']
ANCHORS=(456,4279,9815,19942,579,4380,8300,13200,13500,16500,20250,10335)
HISTORY=(1,16,64)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def up(r):
    p,y,roll=r['car']['physics']['rotation_pyr'];return math.cos(p)*math.cos(roll)
def speed(r):return math.dist(r['car']['physics']['velocity'],[0,0,0])
def orientation(u):return 'inverted' if u<-.5 else 'upright' if u>.8 else 'tilted'
def live(phase):return phase in ('MatchPhase.Active','MatchPhase.Kickoff')
def compact(r):
    return dict(callback=r['callback'],frame=r['frame'],T=r['T'],phase=r['match_phase'],callback_dt=r['callback_dt'],
        student_mode=r['student']['mode'],student_controls=r['student']['controls'],
        previous_student_submission=r['previous_submission'],shadow_mode=r['shadow_mode'],shadow_controls=r['shadow_controls'],
        shadow_sequence_before=r['shadow_before'],shadow_sequence_after=r['shadow_after'],
        shadow_sequence_id=r['shadow_sequence_id_after'],shadow_consumed_phase=r['shadow_detail']['consumed_phase'],
        car=r['car'],ball=r['ball'],opponent=r['opponent'],body_up_world_z=up(r),speed=speed(r),
        observation13=r['observation13'],prediction_probe=r['observation_probe'],shadow_selection=r['shadow_detail']['selected'],
        student_hidden_before=r['student']['hidden_before_sha256'],student_hidden_after=r['student']['hidden_after_sha256'],
        student_logits=r['student']['logits'])

def plan():
    if sha(DATASET/'manifest.json')!=MANIFEST_PIN:raise RuntimeError('Manifest pin mismatch')
    manifest=read(DATASET/'manifest.json')
    if manifest['features'][:13]!=FEATURES:raise RuntimeError('Feature order mismatch')
    matches=[m for m in manifest['matches'] if m['split']=='train']
    if tuple(m['match_id'] for m in matches)!=TRAIN:raise RuntimeError('Train assignment mismatch')
    files={}
    for m in matches:
        tensor=m['tensors']['observations']
        if tensor['shape']!=[m['callbacks'],18] or tensor['dtype']!='float32':raise RuntimeError('Frozen shape/dtype mismatch')
        for name in ('raw/trajectory.jsonl','processed/observations.bin'):
            item=m['files'][name];p=(ROOT/item['path']).resolve()
            if not p.is_relative_to(ROOT.resolve()) or not p.is_file() or p.stat().st_size!=item['bytes']:
                raise RuntimeError('Missing/out-of-scope artifact')
            files[str(p)]=item['sha256']
    return manifest,matches,files

def protected():
    pins=read(ROOT/'training/live_reference_test/source_hashes.json')
    if len(pins)!=43:raise RuntimeError('Protected count changed')
    for p,h in pins.items():
        if sha(ROOT/p)!=h:raise RuntimeError('Protected hash mismatch: '+p)
    return pins

def source_check():
    pin=read(SESSION/'diagnostic_integrity.json')
    for p,h in pin['files'].items():
        if sha(SESSION/p)!=h:raise RuntimeError('Capture artifact mismatch: '+p)
    for p,h in read(SESSION/'metadata.json')['code_hashes'].items():
        if sha(p)!=h:raise RuntimeError('Capture source changed')
    return {str(p):sha(p) for p in SESSION.rglob('*') if p.is_file()}

def runs(rows,a,b,key):
    result=[];start=a
    for i in range(a+1,b+2):
        if i==b+1 or key(rows[i])!=key(rows[start]):
            end=i-1;nextT=rows[i]['T'] if i<len(rows) else rows[end]['T']
            result.append(dict(start_callback=rows[start]['callback'],end_callback=rows[end]['callback'],
                start_T=rows[start]['T'],last_T=rows[end]['T'],sampled_span=rows[end]['T']-rows[start]['T'],
                next_callback_hold_span=nextT-rows[start]['T'],value=key(rows[start]),callbacks=end-start+1))
            start=i
    return result

def temporal(rows,forensics):
    results=[];windows=[]
    for k in forensics['kickoffs']:
        a=k['start_callback']-1;first=k['first_jump']['callback']-1
        seq=next((i for i in range(a,k['phase_end_callback']) if rows[i]['shadow_detail']['flip_trigger']),None)
        if seq is None:raise RuntimeError('No kickoff shadow sequence start')
        sid=rows[seq]['shadow_sequence_id_after']
        complete=next(i for i in range(seq,len(rows)) if rows[i]['shadow_sequence_id_after']==sid and rows[i]['shadow_after'] and rows[i]['shadow_after']['done'])
        phases=runs(rows,seq,complete,lambda r:r['shadow_detail']['consumed_phase'])
        after=min(len(rows)-1,complete+90)
        actions=runs(rows,max(a,seq-15),after,lambda r:r['student']['controls'])
        button=runs(rows,first,min(first+90,len(rows)-1),lambda r:r['student']['controls'][5])
        first_release=next((p for p in button[1:] if not p['value']),None)
        first_press_end=button[0]['end_callback']-1
        native_dodge_before_release=any(r['car']['has_dodged'] for r in rows[first:first_press_end+1])
        milestones={name:compact(rows[index]) for name,index in dict(pre_kickoff=max(0,a-1),kickoff=a,
            shadow_start=seq,first_jump_button=first,first_bad_orientation=k['first_bad_orientation']['callback']-1,
            first_inversion=k['first_inversion']['callback']-1).items()}
        if first_release:milestones['first_student_release']=compact(rows[first_release['start_callback']-1])
        for name,t in k['first_contacts'].items():
            if t:milestones[name+'_first_touch']=compact(rows[t['first_observed_callback']-1])
        # Full-match recovery avoids claiming the short original window exhausted recovery evidence.
        rec=next((e for e in forensics['bad_orientation_episodes'] if e['start_callback']==k['first_bad_orientation']['callback']),None)
        if rec and rec['stable_recovery_callback']:milestones['stable_recovery']=compact(rows[rec['stable_recovery_callback']-1])
        p2=next(p for p in phases if p['value']==2)
        release_during_shadow_release=[r for r in rows[seq:complete+1] if r['shadow_detail']['consumed_phase']==1]
        entry=dict(kickoff_id=k['id'],shadow_sequence_id=sid,shadow_start_callback=seq+1,
            shadow_start_T=rows[seq]['T'],student_first_jump_callback=first+1,
            student_first_jump_minus_shadow_start_seconds=rows[first]['T']-rows[seq]['T'],
            student_first_jump_mode=rows[first]['student']['mode'],
            student_front_dodge_before_reference_phase2=(rows[first]['student']['mode']=='Front dodge' and rows[first]['T']<p2['start_T']),
            continuous_first_jump_hold_seconds=button[0]['next_callback_hold_span'],
            first_release=first_release,native_has_dodged_before_first_release=native_dodge_before_release,
            teacher_reference_durations=[.05,.05,.20,.80],reference_consumed_phase_runs=phases,
            student_native_action_runs=actions,student_jump_button_runs=button,
            actual_student_false_jump_during_shadow_release_callbacks=sum(not r['student']['controls'][5] for r in release_during_shadow_release),
            shadow_release_callbacks=len(release_during_shadow_release),milestones=milestones,
            interpretation='Timing is relative to this stateful shadow, not a claimed optimal kickoff schedule. Actual packet effects follow earlier submitted controls.')
        results.append(entry)
        end=max(after,rec['stable_recovery_callback']+30 if rec and rec['stable_recovery_callback'] else after)
        windows.append(dict(kind='kickoff_temporal',id=k['id'],records=[compact(r) for r in rows[max(0,a-60):min(end,len(rows))]]))
    return results,windows

def recovery_pairs(rows,forensics):
    keys=(484,4307,7705,15667,8187,13070);out=[];windows=[]
    for key in keys:
        item=next((e for e in forensics['recovery_comparison'] if e['start_callback']==key),None)
        kind='fast' if key in (484,4307) else 'prolonged'
        if item is None:
            item=next(e for e in forensics['wall_events'] if e['start_callback']==key);kind='prolonged_neutral'
        a=item['start_callback']-1;b=item['end_callback']-1
        subset=rows[a:b+1];start=rows[a]['T'];end=rows[b]['T'];contacts=[]
        for r in rows[max(0,a-60):min(len(rows),b+121)]:
            for who,t in [('student',r['car']['latest_touch']),('human',r['opponent'][0]['latest_touch'])]:
                m=None if not t else re.search(r'game_seconds=([^,]+)',t)
                if m and start-1<=float(m[1])<=end+2:
                    evidence=dict(who=who,contact_T=float(m[1]),original_repr=t)
                    if evidence not in contacts:contacts.append(evidence)
        entry=dict(category=kind,source_interval={k:v for k,v in item.items() if k!='milestones'},
            onset=compact(rows[a]),midpoint=compact(rows[(a+b)//2]),offset=compact(rows[b]),
            after=compact(rows[min(len(rows)-1,b+90)]),nearby_recorded_ball_contacts=contacts,
            jump_button_callbacks=sum(r['student']['controls'][5] for r in subset),
            angular_velocity_magnitude_min=min(math.dist(r['car']['physics']['angular_velocity'],[0,0,0]) for r in subset),
            angular_velocity_magnitude_max=max(math.dist(r['car']['physics']['angular_velocity'],[0,0,0]) for r in subset),
            movement_resumed_within_two_seconds=any(speed(r)>150 for r in rows[b+1:min(len(rows),b+121)]),
            wall_collision_proven=False)
        out.append(entry);windows.append(dict(kind='recovery_pair',id=key,records=[compact(r) for r in rows[max(0,a-60):min(len(rows),b+121)]]))
    return out,windows

def qualification(rows,ids):
    starts={r['shadow_sequence_id_after']:r for r in rows if r['shadow_detail']['flip_trigger']}
    result=[]
    for n in ids:
        r=rows[n-1];car=r['car'];start=starts[r['shadow_sequence_id_after']];prev=r['previous_submission']
        actual_prior=[x for x in rows[start['callback']-1:n-1] if x['student']['controls'][5]]
        label='prerequisite missing'
        reason='Grounded actual state with has_jumped=false does not establish the airborne first-jump prerequisite for a physical dodge.'
        if car['demolished_timeout']>0:
            label='clearly not a meaningful direct correction';reason='Controlled car is demolished; immediate physical dodge is unavailable.'
        elif 'OnGround' not in car['air_state'] or car['has_jumped']:
            label='ambiguous';reason='Record is inconsistent with the original grounded/no-jump predicate; do not invent availability.'
        result.append(dict(callback=n,T=r['T'],classification=label,reason=reason,
            actual_state=car,previous_student_submission=prev,student_controls=r['student']['controls'],
            shadow_controls=r['shadow_controls'],shadow_consumed_phase=r['shadow_detail']['consumed_phase'],
            shadow_sequence_id=r['shadow_sequence_id_after'],time_since_shadow_start=r['T']-start['T'],
            prior_student_jump_command_callbacks=[x['callback'] for x in actual_prior],
            previous_jump_button=None if prev is None else prev['controls'][5],
            physical_prior_jump_established=car['has_jumped'],
            native_controls_are_legal=True,
            literal_controls_could_initiate_first_jump=('OnGround' in car['air_state'] and car['demolished_timeout']<=0 and prev is not None and not prev['controls'][5]),
            direct_correction_value='unknown; no outcome counterfactual. Do not equate missing intended-dodge prerequisite with illegal controller output.'))
    return result

def load_training(matches):
    arrays=[]
    for m in matches:
        print('TARGETED TRAIN READ:',m['match_id'],flush=True)
        obs=np.fromfile(ROOT/m['files']['processed/observations.bin']['path'],dtype='<f4').reshape(m['callbacks'],18)
        metadata=[]
        with (ROOT/m['files']['raw/trajectory.jsonl']['path']).open(encoding='utf-8') as f:
            for i,line in enumerate(f):
                r=json.loads(line)
                if r['callback']!=i+1 or not r['valid'] or not np.array_equal(obs[i],np.asarray(r['observation18'],np.float32)):
                    raise RuntimeError('Train raw/tensor mismatch')
                p,y,roll=r['car']['rotation_pyr']
                metadata.append(dict(callback=r['callback'],T=r['T'],phase=r['match_phase'],up_z=math.cos(p)*math.cos(roll),
                    car=r['car'],teacher_controls=r['teacher_controls'],teacher_mode=r['action_mode']))
        if len(metadata)!=len(obs):raise RuntimeError('Train row count mismatch')
        arrays.append(dict(match_id=m['match_id'],x=obs[:,:13].astype(np.float64),rows=metadata))
    return arrays

def nearest(query,L,arrays,phase):
    top=[]
    for match in arrays:
        x=match['x'];meta=match['rows'];count=len(x)
        # Causal windows only, no match-boundary crossing, no resampling.
        for a in range(L-1,count,4096):
            b=min(count,a+4096);s=np.zeros(b-a,np.float64)
            for lag in range(L):
                diff=x[a-lag:b-lag]-query[-1-lag]
                s+=np.einsum('ij,ij->i',diff,diff)
            eligible=np.asarray([meta[i]['phase']==phase for i in range(a,b)])
            s[~eligible]=np.inf
            for j in np.argsort(s)[:3]:
                i=a+int(j)
                if not math.isfinite(s[j]):continue
                rms=math.sqrt(float(s[j])/(L*13))
                top.append(dict(match_id=match['match_id'],start_callback=i-L+2,end_callback=i+1,
                    rms13=rms,exact=bool(s[j]==0.),actual_history_seconds=meta[i]['T']-meta[i-L+1]['T'],
                    terminal_metadata=meta[i],terminal_feature_deltas=(x[i]-query[-1]).tolist(),
                    all_history_feature_mae=np.mean(np.abs(x[i-L+1:i+1]-query),axis=0).tolist()))
    return sorted(top,key=lambda x:x['rms13'])[:5]

def run():
    if REPORT.with_suffix('.json').exists() or (WORK/'results.json').exists():raise RuntimeError('Version exists; no overwrite permitted')
    manifest,matches,files=plan();pins=protected();source=source_check()
    # All eight permitted inputs hash-verified before loading ANY full train samples.
    for i,(p,h) in enumerate(files.items(),1):
        print('TARGETED TRAIN HASH',i,'/',len(files),flush=True)
        if sha(p)!=h:raise RuntimeError('Train hash mismatch: '+p)
    forensic_pin=read(ROOT/'training/reports/v13_full_match_forensics_20261005/integrity.json')
    if sha(FORENSICS)!=forensic_pin['json_sha256']:raise RuntimeError('Forensic reference mismatch')
    forensics=read(FORENSICS)
    rows=[json.loads(l) for l in (SESSION/'callbacks.jsonl').read_text().splitlines()]
    temporal_events,windows=temporal(rows,forensics)
    pairs,pair_windows=recovery_pairs(rows,forensics);windows+=pair_windows
    labels=qualification(rows,forensics['possible_unexecutable_shadow_dodge_ids'])
    training=load_training(matches);nn=[]
    for n in ANCHORS:
        r=rows[n-1]
        for L in HISTORY:
            print('TARGETED CAUSAL SEARCH: callback',n,'history',L,flush=True)
            q=np.asarray([x['observation13'] for x in rows[n-L:n]],np.float64)
            neighbors=nearest(q,L,training,r['match_phase'])
            nn.append(dict(query_callback=n,query=compact(r),callback_history_length=L,
                actual_query_history_seconds=rows[n-1]['T']-rows[n-L]['T'],neighbors=neighbors,
                any_exact_match=any(x['exact'] for x in neighbors)))
    videos=[str(p) for base in (SESSION,ROOT/'training/reports/v13_full_match_forensics_20261005') for p in base.rglob('*')
        if p.suffix.lower() in ('.mp4','.mkv','.mov','.webm','.avi','.png','.jpg','.jpeg')]
    counts=dict(Counter(x['classification'] for x in labels))
    result=dict(version='v13_targeted_diagnostics_20261005_v1',source_session=str(SESSION),callbacks=len(rows),
        experiments=dict(kickoff_temporal=temporal_events,recovery_pairs=pairs,single_frame_and_causal_training_neighbors=nn,
            shadow_label_qualification=labels,video=dict(files_in_known_evidence_locations=videos,
                synchronized_video_available=False,limitation='No synchronized video was supplied or found in the session/forensic evidence directories; no visual event mapping made.',
                future_alignment='Capture video presentation timestamps and a visible callback/frame/canonical-time marker or synchronized sidecar; record capture latency/clock mapping. Goal transitions alone are coarse anchors, not exact alignment.')),
        metric=dict(features=FEATURES,distance='RMS Euclidean feature error over original normalized 13D values and actual ordered callbacks. No fitted scaling, interpolation or resampling.',
            history_lengths=list(HISTORY),phase_condition='Candidate ending phase equals query. Earlier native phases are retained, not reset; each history stays within one training match. Phase is cohort metadata, not an input feature.',
            leakage='Training-only reference; no future callbacks in a query/candidate window. Neighborhoods overlap and are not independent samples.',
            identifiability='Approximate closeness is not exact aliasing. Failure to find an identical history in finite data does not prove sufficiency; history comparisons do not evaluate a learned classifier or the GRU.'),
        qualification_counts={name:counts.get(name,0) for name in ('executable now','prerequisite missing','ambiguous','clearly not a meaningful direct correction')},
        legal_control_vs_physical_maneuver='A grounded jump+pitch label may legally initiate first jump; no claim all such controls are useless or must be discarded.',
        immutable_dataset_manifest_sha256=MANIFEST_PIN,training_input_hashes=files,protected_count=43,
        validation_or_test_samples_read=False,model_loaded_or_modified=False,training=False,aggregation=False,
        decision='B — bounded temporal training/objective experiment as a future proposal only; no fitting in this task.',
        decision_reason='All kickoff traces provide direct native button/phase collapse evidence. Test that failure with unchanged 13D before attributing it solely to missing features; hold DAgger behind a separate prerequisite-qualification gate.',
        decision_scope='Does not establish 13D observability sufficiency or exclude later representation/reference changes.')
    write(WORK/'callback_windows.json',windows);result['windows_path']=str(WORK/'callback_windows.json')
    result['windows_sha256']=sha(WORK/'callback_windows.json')
    write(WORK/'results.json',result);write(REPORT.with_suffix('.json'),result)
    lines=['# V13 targeted diagnostics — 2026-10-05','',
        'Read-only analysis of the existing complete match. No new collection, fitting, model/contract change or DAgger aggregation. Training-only nearest neighbors are descriptive.','',
        '## 1. Kickoff temporal hypothesis','',
        '| Kickoff | Shadow start | Student first jump button | Jump delay versus shadow | First jump mode | Continuous hold | Jump-false callbacks during reference release | Native dodge before release |',
        '|---|---:|---:|---:|---|---:|---:|---|',
        *[f"| K{e['kickoff_id']} | {e['shadow_start_callback']} | {e['student_first_jump_callback']} | {e['student_first_jump_minus_shadow_start_seconds']:.6f}s | {e['student_first_jump_mode']} | {e['continuous_first_jump_hold_seconds']:.6f}s | {e['actual_student_false_jump_during_shadow_release_callbacks']}/{e['shadow_release_callbacks']} | {e['native_has_dodged_before_first_release']} |" for e in temporal_events],'',
        'The table distinguishes pure Jump mode from the native jump button in Front-dodge output. Actual first-jump timing must be interpreted relative to shadow sequence start; a late jump can still carry pitch prematurely. A semantic Front-dodge label does not prove a physical dodge. Phase consumption and exact native controls are preserved in JSON.','',
        'All four first jump-button presses occur after the corresponding shadow start, not earlier. Each first press remains held for about 0.5s. Three start with pitch -1; the fourth starts with two pure Jump callbacks then adds pitch without an intervening release. Jump-false during a shadow release is not a correctly placed student release if the student has not yet pressed jump. These distinguish delayed initiation from missing own-sequence release.','',
        'Reference nominal durations are 0.05s Jump, 0.05s release, 0.20s jump+pitch, 0.80s coast. The original ControlStep uses strict elapsed>duration and returns finishing-step controls before the next callback consumes the following step. Therefore delivered phase spans need not equal nominal durations. No optimal-kickoff claim follows.','',
        'Milestones include pre-kickoff, first jump, first release, severe tilt, inversion, first actual touch and stable recovery. Full actual callback windows retain previous submitted actions, all native physical/jump fields, angular velocity, prediction selections and original sequence state.','',
        '## 2. Paired recovery evidence','',
        '| Pair category | Callbacks | Duration | Student jump-button callbacks | Movement resumes within 2s after interval | Nearby ball touches |',
        '|---|---|---:|---:|---|---:|',
        *[f"| {e['category']} | {e['source_interval']['start_callback']}–{e['source_interval']['end_callback']} | {e['source_interval']['duration_seconds']:.3f}s | {e['jump_button_callbacks']} | {e['movement_resumed_within_two_seconds']} | {len(e['nearby_recorded_ball_contacts'])} |" for e in pairs],'',
        'Two fast orientation episodes, two prolonged episodes and two prolonged Neutral episodes were selected from the existing forensic criteria, without changing those criteria. Onset/midpoint/offset/after states and aligned causal windows are preserved. No regional proximity is called proven wall collision. Nearby touch records may explain opportunity for interaction, not causal impulse attribution.','',
        '## 3. Single-frame and causal-history observability','',
        'Twelve relevant query anchors are compared against the four frozen training matches at 1, 16 and 64 callbacks. Current normalized features are unaltered; 16/64 callback durations vary with actual dt and are reported. Search candidates never cross match boundaries or use future callbacks.','',
        '| Query | History callbacks | Best train match / ending callback | RMS feature difference | Exact? | Query / neighbor history seconds | Neighbor terminal orientation |',
        '|---:|---:|---|---:|---|---|---|',
        *[f"| {e['query_callback']} | {e['callback_history_length']} | {e['neighbors'][0]['match_id']} / {e['neighbors'][0]['end_callback']} | {e['neighbors'][0]['rms13']:.8f} | {e['neighbors'][0]['exact']} | {e['actual_query_history_seconds']:.3f} / {e['neighbors'][0]['actual_history_seconds']:.3f} | {orientation(e['neighbors'][0]['terminal_metadata']['up_z'])} |" if e['neighbors'] else f"| {e['query_callback']} | {e['callback_history_length']} | no eligible neighbor | — | — | — | — |" for e in nn],'',
        'Per-feature deltas, history MAEs, terminal orientation/geometry and top-five neighbors are in JSON. Distances have no validated reliability threshold and are not classification accuracy. Exact equality and approximate resemblance are explicitly separated. A finite search cannot establish reliable 13D recovery-state identification.','',
        'Orientation influences relative target coordinates implicitly. Single-frame 13D does not uniquely specify world-up, contact constraints, angular motion or jump availability. A GRU can exploit temporal clues, but neither nearest histories nor hidden-state hashes prove that it can infer these hidden facts reliably. This audit does not claim orientation is completely absent or propose new features.','',
        '## 4. Expert-label qualification','',
        *[f"- {name}: {counts.get(name,0)}" for name in ('executable now','prerequisite missing','ambiguous','clearly not a meaningful direct correction')],'',
        'Scope is the previously flagged 216 grounded/no-prior-jump shadow Front-dodge callbacks, not all expert labels. Each record includes prior student command evidence, native availability flags, air state, sequence phase and elapsed sequence time. Executed command does not establish successful physical first jump; current has_jumped=false remains the observed state.','',
        'Grounded jump+pitch commands are legal and may initiate a first jump on a rising button edge. They cannot be assumed to realize the shadow’s intended second-jump/dodge. Thus missing prerequisites are a maneuver-semantic classification, not proof the direct control is useless. No label was filtered, repaired or used as a target.','',
        '## 5. Video alignment','',
        'No synchronized video was supplied or found in known session/forensic evidence folders. No mapping of the human’s corner observation to one episode is invented. A future capture would require video presentation timestamps mapped to callback/frame/canonical-time markers with logged clock offset/latency; goal/replay events alone provide only coarse anchors. No future capture launched.','',
        '## 6. Decision framework','',
        '| Finding | Evidence | Implication |','|---|---|---|',
        '| Temporal kickoff error | Native button-edge/release traces above; rotation before ball contact | Direct reason to test temporal supervision, not an assertion that missing features are the sole cause |',
        '| Recovery distinguishability | Exact versus approximate train-history neighbors, original orientation metadata | No validated information-sufficiency claim; representation remains an open question |',
        '| Prolonged Neutral | W1/W3 aligned windows; shadow often Chase on the same actual state | Student policy/temporal learning issue exists; physical constraint and hidden-state cause unproven |',
        '| Stateful prerequisite mismatch | 216 qualified labels with actual native state and prior submissions | Blind DAgger must remain blocked pending a label-semantics decision |',
        '| Teacher limitation | MyBot has speed-triggered sequence and no explicit recovery routine | Do not assume the shadow is an optimal recovery oracle |',
        '| Student-only discrepancy | Different phase/action timing from unchanged source | An isolated temporal experiment is supported; teacher-controlled outcome counterfactual absent |','',
        '**Exactly one recommended next research direction: B — a bounded temporal training/objective experiment**, requiring separate approval and a specification before fitting. Keep 13D/checkpoint/reference immutable during this audit. A future experiment would test sequence timing and rare-mode supervision with the same input contract, rather than declaring a feature repair from nearest-neighbor resemblance.','',
        'A is premature as a mandatory repair because the finite neighbor search does not prove causal-history indistinguishability. C is premature because maneuver prerequisites and label usefulness remain unresolved. D changes the reference before testing the directly measured imitation failure. E is not required to reproduce the native timing failure already present; missing video still limits wall-causality claims. F is not required as the next intervention: questions can be tested sequentially without changing multiple factors. B is a proposed experiment, not a claimed cure or authorization to train.','',
        '## 7. Integrity, limits and stop','',
        'All eight permitted training raw/observation artifacts were hash-verified before full sample reads, with manifest ordering/dtype checks. No validation/test sample was opened. No training tensors were persisted; only diagnostic neighbor metadata and callback windows were written. Source/capture hashes were checked before and after analysis.','',
        'Evidence is observational, packets may omit physics ticks, contacts are repr-based, GRU hidden vectors were not captured, and the teacher never controlled these physical trajectories. Report neither competency, causal distribution-shift diagnosis nor DAgger benefit.','',
        f"Callback windows: `{WORK/'callback_windows.json'}`. JSON companion: `{REPORT.with_suffix('.json')}`. STOP for human review."
    ]
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for p,h in files.items():
        if sha(p)!=h:raise RuntimeError('Training input changed during audit')
    assert sha(DATASET/'manifest.json')==MANIFEST_PIN
    assert source_check()==source and protected()==pins
    write(WORK/'integrity.json',dict(training_files_unchanged=True,existing_session_unchanged=True,protected_hashes=43,
        report_sha256=sha(REPORT.with_suffix('.md')),json_sha256=sha(REPORT.with_suffix('.json')),
        analysis_sha256=sha(Path(__file__)),windows_sha256=sha(WORK/'callback_windows.json')))
    print('TARGETED DIAGNOSTICS COMPLETE:',REPORT.with_suffix('.md'),flush=True)
    print('STOP. No model loading, fitting, aggregation, validation/test reads or live launch.',flush=True)

def preflight():
    _,matches,files=plan()
    assert len(files)==8 and sum(m['callbacks'] for m in matches)==107400
    assert len(protected())==43
    # Small numerical check only; no game packets, dataset tensors or trajectory constructed.
    a=np.asarray([[0.,1.],[1.,2.]],np.float64)
    assert np.einsum('ij,ij->i',a-a,a-a).tolist()==[0.,0.]
    print('TARGETED NON-LIVE PREFLIGHT PASS; 4 train matches / 8 files planned; 43 protected hashes verified.')
    print('No full training sample reads or nearest-neighbor searches in preflight.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.check:preflight()
    else:run()
