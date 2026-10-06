"""Read-only full-match forensics. Writes only new diagnostic reports; no ML."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
SESSION=ROOT/'training/behavior_cloning/v13_dagger/phase4_full_match_v1/sessions/20261005_071223_978853'
REPORTS=ROOT/'training/reports'
OUT=REPORTS/'v13_full_match_failure_forensics_20261005'
sys.path.insert(0,str(SESSION.parents[1]/'runner'))
from common import verify

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def live(r):return r['match_phase'] in ('MatchPhase.Kickoff','MatchPhase.Active')
def up(r):
    p,y,roll=r['car']['physics']['rotation_pyr']
    return math.cos(p)*math.cos(roll)
def speed(r):return math.dist(r['car']['physics']['velocity'],[0,0,0])
def magnitude(v):return math.dist(v,[0,0,0])
def touch(s):
    if not s:return None
    m=re.search(r'game_seconds=([^,]+)',s)
    return None if not m else dict(T=float(m[1]),original_repr=s)
def snapshot(r):
    return dict(callback=r['callback'],T=r['T'],frame=r['frame'],phase=r['match_phase'],callback_dt=r['callback_dt'],
        car=r['car'],opponent=r['opponent'],ball=r['ball'],speed=speed(r),body_up_world_z=up(r),
        ball_distance=None if not r['ball'] else math.dist(r['car']['physics']['position'],r['ball']['position']),
        opponent_distance=math.dist(r['car']['physics']['position'],r['opponent'][0]['physics']['position']),
        student=r['student'],shadow_controls=r['shadow_controls'],shadow_mode=r['shadow_mode'],
        sequence_before=r['shadow_before'],sequence_after=r['shadow_after'],sequence_id=r['shadow_sequence_id_after'],
        shadow_detail=r['shadow_detail'],observation13=r['observation13'],prediction_probe=r['observation_probe'],
        prediction_receipt=r['prediction_receipt'],previous_submission=r['previous_submission'])

def episodes(rows,predicate,minimum):
    out=[];start=None
    for i,r in enumerate(rows):
        good=predicate(r)
        if good and start is None:start=i
        if start is not None and (not good or i==len(rows)-1):
            end=i if good else i-1
            if rows[end]['T']-rows[start]['T']>=minimum:out.append((start,end))
            start=None
    return out

def summarize(rows,a,b):
    group=rows[a:b+1];positions=[r['car']['physics']['position'] for r in group]
    extrema=[[min(v[i] for v in positions),max(v[i] for v in positions)] for i in range(3)]
    return dict(start_callback=rows[a]['callback'],end_callback=rows[b]['callback'],start_T=rows[a]['T'],end_T=rows[b]['T'],
        duration_seconds=rows[b]['T']-rows[a]['T'],callbacks=len(group),position_ranges=extrema,
        displacement=math.dist(positions[0],positions[-1]),speed_min=min(map(speed,group)),speed_max=max(map(speed,group)),
        up_z_min=min(map(up,group)),up_z_max=max(map(up,group)),
        angular_speed_max=max(magnitude(r['car']['physics']['angular_velocity']) for r in group),
        student_modes=dict(Counter(r['student']['mode'] for r in group)),shadow_modes=dict(Counter(r['shadow_mode'] for r in group)),
        student_jump_callbacks=sum(r['student']['controls'][5] for r in group),
        student_throttle_callbacks=sum(r['student']['controls'][0]!=0 for r in group),
        shadow_chase_student_neutral=sum(r['shadow_mode']=='Chase' and r['student']['mode']=='Neutral' for r in group),
        has_dodged_callbacks=sum(r['car']['has_dodged'] for r in group))

def main():
    preserved={str(p):sha(p) for p in SESSION.rglob('*') if p.is_file()}
    pins=read(SESSION/'diagnostic_integrity.json')['files']
    for name,pin in pins.items():
        if sha(SESSION/name)!=pin:raise RuntimeError('Capture hash mismatch: '+name)
    protected=verify()
    meta=read(SESSION/'metadata.json')
    for name,pin in meta['code_hashes'].items():
        if sha(Path(name))!=pin:raise RuntimeError('Capture code changed: '+name)
    rows=[json.loads(line) for line in (SESSION/'callbacks.jsonl').read_text().splitlines()]
    report=read(SESSION/'contract_report.json')
    assert len(rows)==26635 and all(r['callback']==i+1 for i,r in enumerate(rows))
    wall=episodes(rows,lambda r:live(r) and (abs(r['car']['physics']['position'][0])>3500 or abs(r['car']['physics']['position'][1])>4300) and speed(r)<150,.5)
    bad=episodes(rows,lambda r:live(r) and up(r)<.5,.3)
    inverted=episodes(rows,lambda r:live(r) and up(r)<-.5,.3)
    stationary=episodes(rows,lambda r:live(r) and speed(r)<5,1.)
    evidence_windows=[]
    kickoffs=[]
    starts=[i for i,r in enumerate(rows) if r['phase_transition'] and r['match_phase']=='MatchPhase.Kickoff']
    for number,a in enumerate(starts,1):
        end=next(i for i in range(a+1,len(rows)) if rows[i]['match_phase']!='MatchPhase.Kickoff')
        finish=min(end+600,len(rows)-1)
        first_bad=next((i for i in range(a,finish+1) if up(rows[i])<.5),None)
        first_inv=next((i for i in range(a,finish+1) if up(rows[i])<-.5),None)
        jump=next((i for i in range(a,finish+1) if rows[i]['student']['controls'][5]),None)
        recovery=next((i for i in range(first_bad or finish,finish+1) if up(rows[i])>.8 and rows[i]['car']['physics']['position'][2]<120
            and i+30<=finish and all(up(r)>.8 for r in rows[i:i+31])),None)
        touches={};touch_followup={}
        next_kickoff=starts[number] if number<len(starts) else len(rows)
        for who in ('student','human'):
            found=[]
            for i in range(a,next_kickoff):
                item=touch(rows[i]['car']['latest_touch'] if who=='student' else rows[i]['opponent'][0]['latest_touch'])
                if item and item['T']>=rows[a]['T']:
                    found.append(dict(first_observed_callback=rows[i]['callback'],**item))
                    touch_followup[who]=[snapshot(rows[j]) for j in (i,i+1,i+10,i+30) if j<next_kickoff]
                    break
            touches[who]=found[0] if found else None
        marks=sorted(set([a,end]+[i for i in (jump,first_bad,first_inv,recovery) if i is not None]))
        human_touch=touches['human'];first_contact=human_touch['T'] if human_touch else None
        first_touch_who=min((who for who in touches if touches[who]),key=lambda who:touches[who]['T'],default=None)
        entry=dict(id=number,start_callback=rows[a]['callback'],start_T=rows[a]['T'],phase_end_callback=rows[end]['callback'],
            phase_end_T=rows[end]['T'],first_jump=None if jump is None else snapshot(rows[jump]),
            first_bad_orientation=None if first_bad is None else snapshot(rows[first_bad]),
            first_inversion=None if first_inv is None else snapshot(rows[first_inv]),first_contacts=touches,
            touch_search_end_callback=rows[next_kickoff-1]['callback'],first_recorded_touch_by=first_touch_who,
            actual_touch_followup=touch_followup,
            recovery_callback=None if recovery is None else rows[recovery]['callback'],
            classification='pre-contact airborne pitch rotation/inversion; '+('human first recorded touch' if first_touch_who=='human' else 'student eventually first recorded touch' if first_touch_who=='student' else 'no recorded touch'),
            snapshots=[snapshot(rows[i]) for i in marks],
            pre_contact_native_has_dodged=sum(rows[i]['car']['has_dodged'] for i in range(a,first_bad+1)),
            window_end_callback=rows[finish]['callback'])
        kickoffs.append(entry)
        evidence_windows.append(dict(kind='kickoff',id=number,records=[snapshot(r) for r in rows[max(0,a-30):finish+1]]))
    wall_events=[]
    for number,(a,b) in enumerate(wall,1):
        s=summarize(rows,a,b);s['id']=number
        s['candidate_confidence']='high: sustained inverted side-region standstill' if s['up_z_max']<-.5 and s['duration_seconds']>5 else 'high: prolonged mostly upright back-region Neutral pause' if s['duration_seconds']>5 else 'lower: brief low-speed continuation near back/side region'
        s['actual_wall_contact_proven']=False
        s['milestones']=[snapshot(rows[i]) for i in sorted(set([max(0,a-60),a,(a+b)//2,b,min(len(rows)-1,b+120)]))]
        wall_events.append(s)
        evidence_windows.append(dict(kind='wall_candidate',id=number,records=[snapshot(r) for r in rows[max(0,a-120):min(len(rows),b+181)]]))
    bad_events=[]
    for a,b in bad:
        s=summarize(rows,a,b)
        recovery=next((i for i in range(b+1,min(len(rows),b+300)) if live(rows[i]) and up(rows[i])>.8 and rows[i]['car']['physics']['position'][2]<120
            and all(live(r) and up(r)>.8 for r in rows[i:min(i+31,len(rows))])),None)
        s['stable_recovery_callback']=None if recovery is None else rows[recovery]['callback']
        s['minimum_inverted_up_z']=min(map(up,rows[a:b+1]))
        bad_events.append(s)
    recovery_comparison=[]
    for target in (484,4307,7705,15667,19970):
        item=next(s for s in bad_events if s['start_callback']==target)
        item=dict(item)
        a=target-1;b=item['end_callback']-1
        item['milestones']=[snapshot(rows[i]) for i in sorted(set([max(0,a-30),a,(a+b)//2,b,min(len(rows)-1,b+60)]))]
        recovery_comparison.append(item)
        evidence_windows.append(dict(kind='recovery_comparison',id=target,records=[snapshot(r) for r in rows[max(0,a-60):min(len(rows),b+121)]]))
    alias_examples=[]
    for n in (8300,13200,16500):
        q=rows[n-1];candidates=[r for r in rows if live(r) and (up(r)>.8 if up(q)<-.5 else up(r)<-.5) and speed(r)<150]
        nearest=min(candidates,key=lambda r:sum((a-b)**2 for a,b in zip(q['observation13'],r['observation13'])))
        d=math.sqrt(sum((a-b)**2 for a,b in zip(q['observation13'],nearest['observation13'])))
        alias_examples.append(dict(query=snapshot(q),nearest_opposite_orientation=snapshot(nearest),normalized_13D_L2=d,
            interpretation='Approximate similarity only, not an exact identical-input or hidden-state collision; never a proof of GRU indistinguishability.'))
    hidden_chain=all(r['student']['hidden_resets']==1 and r['student']['hidden_before_sha256']==(None if i==0 else rows[i-1]['student']['hidden_after_sha256']) for i,r in enumerate(rows))
    audit=dict(session=str(SESSION),callbacks=len(rows),student_modes=report['student_modes'],shadow_modes=report['shadow_modes'],
        report_input_correction='5,238 callbacks / 0 Jump / 9 sequences belonged to the previous 90-second pilot. Full match: 26,635 callbacks / 2 Jump / 33 sequences.',
        shadow_sequence_starts=33,shadow_sequence_completions=33,missed_shadow_starts=33,
        possible_unexecutable_shadow_dodge_callbacks=len(report['possibly_unexecutable_shadow_dodge_callbacks']),
        possible_unexecutable_shadow_dodge_ids=report['possibly_unexecutable_shadow_dodge_callbacks'],
        scores=rows[-1]['scores'],student_side='Blue',opponent='human_b Orange',kickoffs=kickoffs,wall_events=wall_events,
        bad_orientation_episodes=bad_events,inverted_episodes=[summarize(rows,a,b) for a,b in inverted],
        stationary_episodes=[summarize(rows,a,b) for a,b in stationary],recovery_comparison=recovery_comparison,
        approximate_input_similarity_examples=alias_examples,hidden_chain_verified=hidden_chain,
        criteria=dict(body_up_world_z='cos(pitch)*cos(roll), from actual rotation_pyr; up<0.5 marked bad orientation, up<-0.5 inverted; no dodge/collision identity inferred',
            wall_candidate='Actual Active/Kickoff; abs(x)>3500 OR abs(y)>4300; speed<150 UU/s for >=0.5s. Location proxy only; not collision-mesh distance.',
            stationary='speed<5 UU/s for >=1s during Active/Kickoff',stable_recovery='up>0.8 and z<120, sustained ~0.5s in ordinary play; no phase reset credited'),
        protected_hashes_verified=len(protected),diagnostic_hashes_verified=len(pins),prior_sources_unchanged=True,
        recommendation='E: targeted diagnostic evidence before deciding; no collection launched',
        limitations=['Actual packets contain rotation and jump/dodge flags, but no collision impulse/contact pair.',
            'Touch was logged as repr; parse only explicit game_seconds, preserve original strings. Null/missing new touch is not proof of no collision.',
            'No video timestamp identifies which wall episode the human meant. Rank candidates, do not assert a unique match.',
            'No collision mesh/field geometry captured; location thresholds are regional proxies.',
            'Only GRU hidden-state hashes and outputs captured, not hidden vectors; no identifiability/causal memory proof.',
            'Teacher never controlled physics in this match; matching/different shadow labels are not outcome counterfactuals.',
            'Processed callbacks only, no proof of lossless delivery; no new validation/test samples accessed.'])
    work=Path(__file__).resolve().parent
    write(work/'callback_windows.json',evidence_windows)
    audit['callback_windows_path']=str(work/'callback_windows.json');audit['callback_windows_sha256']=sha(work/'callback_windows.json')
    write(OUT.with_suffix('.json'),audit)
    md=['# V13 full-match failure forensics — 2026-10-05','',
        '## 1. Executive summary','',
        'Read-only review of the complete Blue-student/Orange-human_b match. **26,635 callbacks**, 33 completed shadow sequences, two pure student Jump callbacks. The 5,238/zero-Jump/nine-sequence figures in the request refer to the earlier pilot. Final score Blue 2–Orange 1. Overtime is inferred from the five-minute configuration, a tied 1–1 countdown at T=330.317, and subsequent play until T=447.117; an explicit overtime flag was not captured. One result does not establish competence.','',
        '**Four out of four detected kickoffs contain pre-contact orientation loss.** Ground-origin jump+pitch −1 outputs precede airborne pitching and inversion; no native dodge is recorded by orientation-loss onset. Three have a human first touch before the student. In the fourth, the student eventually records first contact at T=358.308, about 24 seconds after kickoff begins. Thus the initial failure is not supported as collision-induced flipping.','',
        'Two distinct prolonged pause patterns exist: inverted/Neutral near the side region, and mostly upright/Neutral near the back region. Recovery is intermittent. The current capture cannot identify a unique video-described corner episode or prove actual wall constraint. **Do not blindly aggregate these labels into DAgger.**','',
        '## 2. Exact kickoff events','',
        '| Event | Kickoff callbacks / canonical T | First student jump / bad orientation | First recorded human touch | Stable recovery in window | Outcome |',
        '|---|---|---|---|---|---|',
        *[f"| K{k['id']} | {k['start_callback']}–{k['phase_end_callback']}; {k['start_T']:.3f}–{k['phase_end_T']:.3f}s | {k['first_jump']['callback']} / {k['first_bad_orientation']['callback']} | {k['first_contacts']['human']['first_observed_callback'] if k['first_contacts']['human'] else 'unobserved'} | {k['recovery_callback'] or 'not in window'} | {k['classification']} |" for k in kickoffs],'',
        '| Event | Student first touch callback / T | Human first touch callback / T |','|---|---|---|',
        *[f"| K{k['id']} | {k['first_contacts']['student']['first_observed_callback']} / {k['first_contacts']['student']['T']:.6f}s | {k['first_contacts']['human']['first_observed_callback']} / {k['first_contacts']['human']['T']:.6f}s |" for k in kickoffs],'',
        'Touch searches cover the interval until the next kickoff/end, not only the short orientation timeline. Each touch’s original repr, first observed callback, and immediate subsequent packet states are preserved in JSON. These identify recorded ball touches, not every collision or the game application time of a submitted control.','',
        'Orientation criterion: body-up dot world-up = cos(pitch)×cos(roll). Values below −0.5 support inversion; below +0.5 indicate severe tilt. This is a physical orientation measure, not a semantic dodge/flip or collision detector. Native has_jumped/has_dodged are reported separately.','',
        'At initial K1, callback 456 outputs [0,0,−1,0,0,true,false,false] from OnGround; callback 484 has up_z≈0.48 at [222,−3157,181], and 494 has up_z≈−0.27. At callback 579 (human touch), up_z≈−1 with student Neutral. The shadow changes Jump→release→Front dodge→coast based on its independent sequence, whereas the student holds jump while pitching. K2/K3/K4 show the same ground-jump/airborne-pitch pattern. K3 alone has two pure Jump callbacks (9815–9816), followed by Front dodge without an intervening release.','',
        'A. Kickoff orientation loss is systematic in these four observations, not proven systematic over all matches. B. The temporal action error starts before first recorded contact. Steering/throttle differences are also recorded; their independent contribution is unproven. C. The shadow differs at relevant pre-contact callbacks, particularly sequence timing. D. Initial inversion precedes contact; later collisions or surfaces may prolong it. E. The original teacher also flips on a speed band, but this run cannot establish that it would suffer the same outcome under its own physics.','',
        '## 3. Exact side/back-region pause candidates','',
        '| Candidate | Callbacks | T / duration | Up-z range | Student modes | Shadow Chase while student Neutral | Confidence |',
        '|---|---|---|---|---|---:|---|',
        *[f"| W{e['id']} | {e['start_callback']}–{e['end_callback']} | {e['start_T']:.3f}s / {e['duration_seconds']:.3f}s | {e['up_z_min']:.3f}…{e['up_z_max']:.3f} | {e['student_modes']} | {e['shadow_chase_student_neutral']} | {e['candidate_confidence']} |" for e in wall_events],'',
        'W1 is the strongest inverted-side pause candidate: 8187–8752, 9.500 s; at 8300/8500/8752 the car is essentially fixed around [−3807,391,42], upside down, with Neutral controls. Shadow Chase differs on those same callbacks. W2 is a brief continuation before eventual recovery. Neither proves contact with a wall: x≈−3807 is only a regional proxy.','',
        'W3 is the strongest back-region pause candidate: 13070–13902, 14.008 s. It begins tilted, then is upright by 13200 at [1433,−4949,18], speed≈1 UU/s, still Neutral while shadow Chase. W4–W6 are briefer nearby pauses. The prolonged pause is not wholly explained by inversion. Repeated steering into a wall is not supported during all-zero Neutral portions; no effective recovery attempt is recorded there. Later motion can include gravity or human/ball interaction and is not credited solely to the model.','',
        'Every candidate has a callback-by-callback lead-up/pause/recovery window in callback_windows.json, including exact physics, orientations, both controls, sequence states, observations, prediction selections and callback dt. No rows were resampled or corrected.','',
        '## 4. Successful versus prolonged recovery','',
        '| Bad-orientation onset | End | Duration | Stable recovery | Student modes during episode |',
        '|---:|---:|---:|---:|---|',
        *[f"| {e['start_callback']} | {e['end_callback']} | {e['duration_seconds']:.3f}s | {e['stable_recovery_callback'] or 'unobserved'} | {e['student_modes']} |" for e in recovery_comparison],'',
        'The initial kickoff episode recovers near callback 666 and upright ground motion is visible by 9000 in the long side episode. In contrast, the episode 7705–8960 lasts 21.125 s and includes the 9.5 s inverted standstill. Another inversion 15667–16824 lasts 19.483 s, including a zero-speed inverted Neutral state around 16500. The overtime episode lasts 17.625 s. These are not permanent failures: prolonged cases eventually become upright without a goal-reset being counted as recovery.','',
        'Milestone snapshots preserve ball geometry/prediction, actual speed/orientation, controls and shadow labels. GRU continuity is verified through hidden hashes, but hashes cannot explain what state the GRU encoded. Actions use deterministic argmax/tanh inference, not stochastic sampling. Different recovery durations correlate with position, motion and Neutral/Chase periods; sparse examples cannot separate gravity/collision dynamics from action effects causally.','',
        '## 5. Teacher-versus-student failure classification','',
        '| Category | Evidence-based classification |',
        '|---|---|',
        '| Kickoff inversion | Primarily student temporal-action learning failure: premature jump+pitch and missing release precede inversion. Teacher speed-only flip trigger is also limited; its own outcome is untested. |',
        '| Upright back-region pause | Likely student-learning failure: prolonged Neutral while original shadow Chase on actual state. Wall constraint and hidden-state cause remain ambiguous. |',
        '| Inverted side pause / recovery | Mixed learning and representation limitation; inherited teacher limitation also matters because it has no explicit orientation-aware recovery routine. |',
        '| Missed jump timing / front-dodge sequence | Strong student-learning evidence: all 33 shadow sequence starts missed; only two pure Jump callbacks, and explicit no-release kickoff patterns. |',
        '| Later shadow dodge labels on learner physics | Stateful-label/action-prerequisite mismatch, not proof of helpful expert behavior. |','',
        'Source: python-example-original/src/bot.py, MyBot.get_output and begin_front_flip. Teacher ignores other logic while pending; at 750<speed<800 starts 0.05s Jump, 0.05s release, 0.2s jump+pitch−1, 0.8s coast. It has no dedicated kickoff strategy, wall navigation or inversion recovery; this is an inherited reference limitation, not a newly introduced bug.','',
        '## 6. 13D observability','',
        'Exact features remain ball-relative XYZ, predicted-relative XYZ, distance, speed, two availability masks, selected horizon, first-slice offset and callback dt. Positions are car-relative, so orientation influences them implicitly. It is incorrect to say there is no orientation information whatsoever. However, no world-up vector, absolute car position, angular velocity, velocity direction, native air/jump/dodge state, wall distance or executed action history is supplied.','',
        'A relative target vector does not uniquely determine orientation relative to gravity: multiple world/car configurations can produce the same relative features. Speed discards direction. Thus inversion versus upright or wall contact versus ordinary low-speed chasing is not generally identifiable from a single 13D observation. Temporal changes can offer clues, but do not guarantee identification: stationary periods, unobserved forces/contact and absent executed-action inputs can remain ambiguous. GRU state hashes do not prove reliable inference.','',
        'Approximate opposite-orientation feature neighbors from this match are included in JSON. They are explicitly not exact identical-input collisions, and neither prove nor disprove history-level ambiguity. No feature change or architecture is proposed/implemented in this audit.','',
        '## 7. DAgger implications','',
        f"Blind aggregation is **not technically justified**: {len(report['possibly_unexecutable_shadow_dodge_callbacks'])} shadow Front-dodge labels occur while the actual car is grounded with has_jumped=false. A shadow sequence follows its own prior expert actions even when the student did not execute them. The full-match evidence therefore requires an expert-label/prerequisite qualification decision before treating these as imitation targets. Observation identifiability and temporal supervision also deserve tests, but this audit does not choose a repair, filter, objective or altered teacher.",'',
        'Dropping labels, resetting the teacher, adding recovery rules or changing input features would each alter a contract and must be separately reviewed. No such alteration is made here.','',
        '## 8. Evidence limits and integrity','',*['- '+x for x in audit['limitations']],'',
        f"Verified {len(pins)} capture-file hashes and 43 protected sources before analysis. Rehashed all existing session files after report construction; no existing artifacts were written. No training/validation/test samples, fitting, aggregation, simulation or live launch used.",'',
        '## 9. Recommended next experiment only','',
        '**E — collect targeted diagnostic evidence before deciding**, subject to separate approval. Focus on naturally occurring kickoff pitch/jump/release timing and paired fast/prolonged recovery states, with time-aligned video and structured native touch/contact-prerequisite evidence. Use the unchanged student and faithful shadow; do not force expert-sequence state resets or treat labels as a dataset. Specify label qualification/observability hypotheses and success criteria before capture.','',
        'Why E: the capture establishes action-sequence failure and Neutral pauses, but cannot uniquely locate the human’s corner event, attribute wall impulses, explain GRU hidden state, or establish which counterfactual shadow labels are appropriate. This uncertainty makes direct DAgger (A) premature and does not yet justify selecting B, C or D as the sole repair. No next experiment is launched.','',
        f"Full callback evidence: `{work/'callback_windows.json'}`. JSON companion: `{OUT.with_suffix('.json')}`. Stop for review."
    ]
    OUT.with_suffix('.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    after={str(p):sha(p) for p in SESSION.rglob('*') if p.is_file()}
    assert after==preserved and verify()==protected
    write(work/'integrity.json',dict(existing_session_unchanged=True,protected_hashes=43,
        input_files=preserved,analysis_sha256=sha(Path(__file__)),
        report_sha256=sha(OUT.with_suffix('.md')),json_sha256=sha(OUT.with_suffix('.json')),
        callback_windows_sha256=sha(work/'callback_windows.json')))
    print('FORENSICS COMPLETE:',OUT.with_suffix('.md'))
    print('Kickoffs:',len(kickoffs),'Wall candidates:',len(wall_events),'Bad-orientation episodes:',len(bad_events))
    print('Capture unchanged; 43 protected hashes unchanged. No training, collection or contract changes.')

if __name__=='__main__':main()
