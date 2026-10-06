"""Quarantined pilot review; stdlib only, no training artifact/sample readers."""
from bisect import bisect_right
from collections import Counter
import json
import math
from pathlib import Path
from common import HERE,CHANNELS,ANALOG_TOL,sha,read,write
from diagnostics import SummaryReader,callback_accounting

PILOT=HERE.parent

def lines(path):
    if not path.exists():return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def quantile(values,p):
    if not values:return None
    a=sorted(values);k=(len(a)-1)*p;i=int(k)
    return a[i] if i==len(a)-1 else a[i]+(a[i+1]-a[i])*(k-i)

def describe(values):
    return dict(count=len(values),minimum=min(values) if values else None,
        median=quantile(values,.5),p95=quantile(values,.95),p99=quantile(values,.99),
        maximum=max(values) if values else None)

def js(a,b):
    if not sum(a) or not sum(b):return None
    p=[x/sum(a) for x in a];q=[x/sum(b) for x in b];m=[(x+y)/2 for x,y in zip(p,q)]
    return .5*sum(x*math.log2(x/z) for x,z in zip(p,m) if x)+.5*sum(y*math.log2(y/z) for y,z in zip(q,m) if y)

def eligible(x,i):
    if i in (0,1,2,6):return x[8]==1
    if i in (3,4,5,10,11):return x[8]==1 and x[9]==1
    if i==12:return x[12]>0
    return True

def compare_reference(rows,weights,reference,names):
    out={}
    for group,features in reference['features'].items():
        result={}
        for i,name in enumerate(names):
            base=features[name]
            if not base['count']:result[name]=dict(status='reference_empty');continue
            selected=[(r['observation13'][i],weights[k]) for k,r in enumerate(rows)
                if (group=='ALL' or r['match_phase']==group) and eligible(r['observation13'],i)]
            values=[v for v,w in selected];counts=[0]*len(base['counts']);hold=[0.]*len(counts)
            for v,w in selected:
                j=bisect_right(base['histogram_internal_edges'],v);counts[j]+=1;hold[j]+=w
            result[name]=dict(pilot=describe(values),reference_count=base['count'],
                outside_reference_minmax=sum(v<base['minimum'] or v>base['maximum'] for v in values),
                outside_reference_p01_p99=sum(v<base['p01'] or v>base['p99'] for v in values),
                callback_histogram=counts,hold_seconds_histogram=hold,
                callback_js_bits=js(counts,base['counts']),hold_js_bits=js(hold,base['hold_seconds']),
                reference_constant=base['constant'])
        out[group]=result
    out['unrepresented_pilot_phases']=sorted(set(r['match_phase'] for r in rows)-set(reference['features']))
    return out

def categories(r):
    x=r['observation13'];speed=x[7]*2300
    return dict(phase=r['match_phase'],ball_present='present' if x[8]==1 else 'missing',
        prediction_valid='valid' if x[9]==1 else 'unavailable',
        near_far='missing' if x[8]!=1 else 'far' if x[6]*6000>1500 else 'near',
        speed='below750' if speed<750 else '750to800' if speed<=800 else 'above800',
        geometry='missing' if x[8]!=1 else ('front' if x[0]>=0 else 'behind')+('_right' if x[1]>=0 else '_left'))

def review(folder):
    folder=Path(folder);meta=read(folder/'metadata.json');supervisor=read(folder/'supervisor.json')
    raw=lines(folder/'callbacks.jsonl');rows,duplicates,malformed=callback_accounting(raw)
    events=lines(folder/'events.jsonl');errors=lines(folder/'diagnostic_errors.jsonl')
    submissions=[e['payload'] for e in events if e['kind']=='student_controls_submitted']
    grouped={}
    for s in submissions:grouped.setdefault(s['callback'],[]).append(s)
    valid=[];failures=[];prior_hidden=None;verified=0
    for r in rows:
        n=r['callback'];student=r.get('student');sent=grouped.get(n,[])
        if not student or 'error' in r:
            failures.append(dict(callback=n,kind='callback_error',error=r.get('error')));continue
        if len(sent)!=1 or sent[0]['controls']!=student['controls']:
            failures.append(dict(callback=n,kind='student_transport',submissions=len(sent)))
        else:verified+=1
        if student['hidden_before_sha256']!=prior_hidden or student['hidden_resets']!=1:
            failures.append(dict(callback=n,kind='hidden_chain'))
        prior_hidden=student['hidden_after_sha256']
        if r['shadow_prediction_object_id']!=r['prediction_object_id'] or not r['same_native_shadow_packet']:
            failures.append(dict(callback=n,kind='shadow_input_provenance'))
        if len(r['observation13'])!=13 or r['private_teacher_state_in_student_input']:
            failures.append(dict(callback=n,kind='input_contract'))
        valid.append(r)
    weights=[0.]*len(valid);crossed=0;learner_states=[]
    for k,r in enumerate(valid):
        previous=r['previous_submission']
        if k and previous and previous['callback']==valid[k-1]['callback'] and len(grouped.get(previous['callback'],[]))==1:
            if previous==grouped[previous['callback']][0]:learner_states.append(r['callback'])
        if k+1<len(valid):
            nxt=valid[k+1];prior=nxt['previous_submission']
            linked=prior and prior['callback']==r['callback'] and len(grouped.get(r['callback'],[]))==1 and prior==grouped[r['callback']][0]
            if linked and nxt['T']>r['T']:
                if nxt['match_phase']==r['match_phase']:weights[k]=nxt['T']-r['T']
                else:crossed+=1
    starts=[r for r in valid if r['shadow_detail']['flip_trigger']]
    missed=[r for r in starts if r['student']['mode']!='Jump']
    missed_ids={r['shadow_sequence_id_after'] for r in missed}
    continuation=[r for r in valid if r['shadow_sequence_id_before'] in missed_ids and r['shadow_before'] and not r['shadow_before']['done']]
    unsupported=[r['callback'] for r in valid if r['shadow_mode']=='Front dodge' and 'OnGround' in r['car']['air_state'] and not r['car']['has_jumped']]
    completions=[r['callback'] for r in valid if r['shadow_after'] and r['shadow_after']['done'] and (not r['shadow_before'] or not r['shadow_before']['done'])]
    pending_missing=[r['callback'] for r in valid if r['ball'] is None and r['shadow_before'] and not r['shadow_before']['done']]
    pending_phases={phase:[r['callback'] for r in valid if phase in r['match_phase'] and r['shadow_before'] and not r['shadow_before']['done']]
        for phase in ('GoalScored','Replay','Countdown','Kickoff')}
    channels={}
    for i,name in enumerate(CHANNELS):
        differences=[abs(r['student']['controls'][i]-r['shadow_controls'][i]) for r in valid]
        channels[name]=dict(student=describe([r['student']['controls'][i] for r in valid]),
            shadow=describe([r['shadow_controls'][i] for r in valid]),absolute_error=describe(differences),
            exact_agreements=sum(r['student']['controls'][i]==r['shadow_controls'][i] for r in valid))
    agreement=sum(all(abs(a-b)<=ANALOG_TOL for a,b in zip(r['student']['controls'][:5],r['shadow_controls'][:5]))
        and r['student']['controls'][5:]==r['shadow_controls'][5:] for r in valid)
    sign=[r['callback'] for r in valid if abs(r['student']['controls'][1])>.02 and abs(r['shadow_controls'][1])>.02
        and r['student']['controls'][1]*r['shadow_controls'][1]<0]
    occupancy={}
    for k,r in enumerate(valid):
        for name,label in categories(r).items():
            value=occupancy.setdefault(name,{}).setdefault(label,dict(callbacks=0,hold_seconds=0.))
            value['callbacks']+=1;value['hold_seconds']+=weights[k]
    reference=HERE.parents[1]/'phase4_pilot_v1/train_reference_v1'
    integrity=read(reference/'integrity.json')
    path=reference/'statistics.json'
    if sha(path)!=integrity['statistics_sha256']:raise RuntimeError('Descriptive reference integrity failure')
    baseline=read(path) # Precomputed descriptions only: no training raw/tensor reads.
    comparisons={name:compare_reference(valid,weights,baseline[name],baseline['feature_order']) for name in ('primary','secondary')}
    summary=SummaryReader().read(folder);closed=read(folder/'agent_closed.json') if (folder/'agent_closed.json').exists() else {}
    final=max((r['callback'] for r in rows),default=0)
    final_sent=len(grouped.get(final,[]))==1 and final in {r['callback'] for r in valid}
    clean=not failures and not errors and not duplicates and not malformed and not summary.get('publication_errors')
    coverage=dict(actual_learner_induced_callbacks=len(learner_states),shadow_sequence_starts=len(starts),shadow_completions=len(completions),
        student_missed_shadow_starts=len(missed),offpolicy_shadow_continuation_callbacks=len(continuation),
        missing_ball_pending=len(pending_missing),pending_match_phases={k:len(v) for k,v in pending_phases.items()},
        long_callback_gaps_over_five_ticks=sum((r['callback_dt'] or 0)>5/120+.0001 for r in valid),
        actual_shadow_prediction_selections=sum(r['shadow_detail']['selected'] is not None for r in valid),
        near_far=dict(Counter(categories(r)['near_far'] for r in valid)))
    status='natural_match_complete_awaiting_review' if clean and supervisor['status']=='natural_match_ended' and closed and final_sent and supervisor['protected_ok'] and supervisor['immutable_ok'] else 'incomplete_or_failed_awaiting_review'
    result=dict(version='phase4_pilot_review_v1',status=status,quarantined_diagnostics_not_training_dataset=True,
        session=str(folder),opponent_id=meta['opponent_id'],supervisor=supervisor,agent_closed=closed,
        raw_records=len(raw),unique_callbacks=len(rows),valid_snapshots=len(valid),failures=failures,errors=errors,
        duplicate_callbacks=duplicates,malformed_records=len(malformed),verified_student_submissions=verified,
        total_student_submission_events=len(submissions),final_callback=final,final_callback_submission_verified=final_sent,
        publication_errors=summary.get('publication_errors',[]),coverage=coverage,
        student_modes=dict(Counter(r['student']['mode'] for r in valid)),shadow_modes=dict(Counter(r['shadow_mode'] for r in valid)),
        mode_agreement=sum(r['student_shadow_mode_agreement'] for r in valid),native_action_agreement_with_tolerance=agreement,
        channels=channels,steering_sign_disagreement_callbacks=sign,
        shadow_missed_start_callbacks=[r['callback'] for r in missed],shadow_continuation_callbacks=[r['callback'] for r in continuation],
        possibly_unexecutable_shadow_dodge_callbacks=unsupported,pending_missing_callbacks=pending_missing,pending_phases=pending_phases,
        callback_dt=describe([r['callback_dt'] for r in valid if r['callback_dt'] is not None]),
        inference_seconds=describe([r['student']['inference_seconds'] for r in valid]),
        callback_to_submission_seconds=describe([s['callback_to_submission_seconds'] for s in submissions]),
        packet_receipt_counters=len({r['packet_receipt']['counter'] for r in valid}),
        prediction_receipt_counters=len({r['prediction_receipt']['counter'] for r in valid}),
        consecutive_prediction_reuse=sum(a['prediction_receipt']['counter']==b['prediction_receipt']['counter'] for a,b in zip(valid,valid[1:])),
        occupancy=occupancy,train_only_descriptive_comparison=comparisons,
        crossed_phase_hold_intervals_excluded=crossed,final_hold_censored=True,
        reference_sha256=integrity['statistics_sha256'],
        limitations=['Processed callbacks only; receipt counters cannot prove lossless physics-tick delivery.',
            'One native five-minute match with countdown/replay and overtime; no wall-clock deadline or coverage forcing.',
            'Shadow equivalence was validated in Phase 3; this pilot has no executed-teacher physics counterfactual.',
            'Expert temporal labels may assume actions the learner never executed; they are not approved learning targets.',
            'Marginal train-reference differences describe one learner trajectory; they do not establish causal distribution shift.',
            'No competence gate, aggregation, fitting or additional run is authorized by report completion.'])
    write(folder/'contract_report.json',result)
    artifacts={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name not in ('contract_report.md','diagnostic_integrity.json')}
    write(folder/'diagnostic_integrity.json',dict(files=artifacts,not_training_dataset=True))
    (folder/'contract_report.md').write_text('\n'.join(['# Natural five-minute student-controlled match','',f"Status: **{status}**. Stop for review.",'',
        f"Session: `{folder}`. {len(rows)} unique processed callbacks; {verified} verified student submissions. Final callback submitted: {final_sent}.",'',
        'The 13D student alone submitted controls. The unchanged original teacher ran independently in shadow on actual delivered packets/predictions. No state-setting, receive pauses, replay manipulation, fallback or repair was used. Diagnostics remain quarantined.','',
        '## Observed learner trajectory and shadow behavior','', '```json',json.dumps(coverage,indent=2),'```','',
        f"Missed shadow sequence-start callbacks: {[r['callback'] for r in missed]}. Possibly unexecutable shadow dodge callbacks: {unsupported}.",'',
        'Zero counts identify unobserved categories; this bounded run must not be extended to force coverage. Linked prior submissions identify actual student-controlled successor callbacks. They do not establish what teacher-controlled physics would have been.','',
        '## Disagreement and timing','',f"Native action agreement within analog tolerance: {agreement}/{len(valid)}; mode agreement: {result['mode_agreement']}/{len(valid)}. Steering-sign disagreements outside ±0.02: {len(sign)}.",'',
        'Per-channel errors, mode counts, callback/inference/submission latency, all disagreement IDs, receipt reuse, phase transitions and failures are preserved in the JSON companion and raw diagnostics.','',
        '## Train-only descriptive comparison','',
        'Use the secondary all-four-train reference as the relevant descriptive comparison for this Blue student. The inherited primary is Orange-only and is retained as explicitly side-mismatched context, not a Blue-side matched baseline. No new reference samples were read. Phase/mask-conditioned feature comparisons do not tune the policy. Cross-phase and final-censored holds are excluded.','',
        '## Integrity, failures and limits','',f"Failures: {len(failures)}; error records: {len(errors)}; duplicate callbacks: {duplicates}; publication errors: {summary.get('publication_errors',[])}.",'',
        *result['limitations'],'',f"Exact evidence and file SHA256s: `{folder/'contract_report.json'}` and `{folder/'diagnostic_integrity.json'}`.",'',
        'No second pilot, DAgger aggregation, training, or progression follows automatically.'])+'\n',encoding='utf-8')
    return result
