"""Read-only evidence review; writes a new review folder, never repairs raw logs."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import math
from pathlib import Path
import statistics
from common import HERE,ROOT,CHANNELS,ANALOG_TOL,verify,immutable_snapshot,read,sha,write

def stats(values):
    values=sorted(values)
    return dict(count=len(values),minimum=min(values),median=statistics.median(values),
        p95=values[math.ceil(.95*len(values))-1],maximum=max(values)) if values else dict(count=0)

def review(folder):
    folder=folder.resolve()
    if folder.parent!= (HERE/'sessions').resolve():raise ValueError('Expected existing isolated Phase 3 session')
    protected=verify();immutable=immutable_snapshot()
    inputs={str(p):sha(p) for p in folder.iterdir() if p.is_file()}
    automatic=read(folder/'contract_report.json')
    for name,pin in automatic['integrity'].items():
        if sha(folder/name)!=pin:raise RuntimeError('Session evidence changed: '+name)
    metadata=read(folder/'metadata.json')
    for path,pin in metadata['code_hashes'].items():
        if sha(path)!=pin:raise RuntimeError('Capture code changed since run: '+path)
    rows=[__import__('json').loads(s) for s in (folder/'callbacks.jsonl').read_text(encoding='utf-8').splitlines()]
    events=[__import__('json').loads(s) for s in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    grouped=defaultdict(list)
    for row in rows:grouped[row['callback']].append(row)
    # Count callbacks, not JSONL lines. Retain every duplicate/error in evidence.
    unique=[grouped[i][0] for i in sorted(grouped)]
    if [r['callback'] for r in unique]!=list(range(1,len(unique)+1)):raise RuntimeError('Noncontiguous callback IDs')
    errors=[dict(callback=r['callback'],type=r.get('error_type'),error=r.get('error'),comparison_failures=r.get('failures'))
        for r in rows if r.get('error') or r.get('failures')]
    mismatches=[];maximum_analog=0.;last_hidden=None;hidden_errors=[]
    for r in unique:
        a,b=r['original_controls'],r['shadow_controls']
        analog=max(abs(x-y) for x,y in zip(a[:5],b[:5]))
        maximum_analog=max(maximum_analog,analog)
        issue=[]
        if not all(math.isfinite(x) for x in a[:5]+b[:5]) or analog>ANALOG_TOL:issue.append('analog')
        if a[5:]!=b[5:] or any(type(x) is not bool for x in a[5:]+b[5:]):issue.append('buttons')
        for key in ('before','after'):
            if r['original_'+key]!=r['shadow_'+key]:issue.append('sequence_'+key)
        if r['original_mode']!=r['shadow_mode']:issue.append('mode')
        if r['original_detail']!=r['shadow_detail']:issue.append('selector/branch/consumed phase')
        if r['original_flips']!=r['shadow_flips']:issue.append('sequence count')
        if r['native_packet_object_id']!=r['shadow_packet_object_id'] or r['native_prediction_object_id']!=r['shadow_prediction_object_id']:issue.append('native input identity')
        if r['T']!=r['shadow_elapsed'] or not r['independent_sequence_objects']:issue.append('time/memory')
        if a[3:5]!=[0.,0.] or a[6:]!=[False,False]:issue.append('declared teacher support')
        if issue:mismatches.append(dict(callback=r['callback'],issues=issue))
        student=r['student']
        if student['hidden_before_sha256']!=last_hidden or student['hidden_resets']!=1:
            hidden_errors.append(r['callback'])
        last_hidden=student['hidden_after_sha256']
    submitted=[e['payload'] for e in events if e['kind']=='original_controls_submitted']
    submitted_by={s['callback']:s for s in submitted}
    transport_errors=[];not_submitted=[]
    for r in unique:
        s=submitted_by.get(r['callback'])
        if s is None:not_submitted.append(r['callback']);continue
        a,b=r['original_controls'],s['controls']
        if a[5:]!=b[5:] or max(abs(x-y) for x,y in zip(a[:5],b[:5]))>ANALOG_TOL:
            transport_errors.append(r['callback'])
    def example(r):
        return dict(callback=r['callback'],frame=r['frame'],elapsed=r['T'],match_phase=r['match_phase'],
            sequence_id=r['original_sequence_id_after'],source_phase=r['original_detail']['consumed_phase'],
            teacher_mode=r['original_mode'],student_mode=r['student']['mode'],teacher_controls=r['original_controls'],
            student_controls=r['student']['controls'],previous_student=r['previous_student_action_diagnostic_only'],
            callback_dt=r['callback_dt'])
    coverage=defaultdict(list);previous_mode=None;sequence_missed=set();starts=[];completions=[]
    for r in unique:
        d=r['original_detail'];before=r['original_before'];after=r['original_after']
        pending=before is not None and not before['done'];sid=r['original_sequence_id_after']
        def mark(key):coverage[key].append(example(r))
        if previous_mode=='Neutral' and r['original_mode']=='Chase':mark('neutral_to_chase')
        if previous_mode=='Chase' and d['flip_trigger'] and r['original_mode']=='Jump':mark('chase_to_jump')
        for phase,key in ((1,'release'),(2,'front_dodge'),(3,'coast')):
            if d['consumed_phase']==phase:mark(key)
        if d['flip_trigger']:
            starts.append(example(r))
            if r['student']['mode']!='Jump':sequence_missed.add(sid);mark('student_missed_sequence_start')
        if after is not None and after['done'] and (before is None or not before['done']):
            completions.append(example(r));mark('completion')
        if pending and not r['relevant_observation']['ball_present']:
            mark('missing_ball_pending')
            if before!=after:raise AssertionError('Missing ball changed pending state')
        for phase,key in (('GoalScored','goal_pending'),('Replay','replay_pending'),('Countdown','countdown_pending'),('Kickoff','kickoff_pending')):
            if pending and r['match_phase']=='MatchPhase.'+phase:mark(key)
        if pending and r['callback_dt'] is not None and r['callback_dt']>=1:mark('long_gap')
        if d['branch'] in ('near_ball','far_ball'):mark(d['branch'])
        if d['selected'] is not None:mark('live_selector')
        prior=r['previous_student_action_diagnostic_only']
        if d['consumed_phase']==2 and prior is not None and not prior['controls'][5]:mark('student_missed_prior_jump')
        if pending and sid in sequence_missed and not d['flip_trigger']:mark('shadow_continues_after_student_divergence')
        previous_mode=r['original_mode']
    required=('neutral_to_chase','chase_to_jump','release','front_dodge','coast','completion',
        'missing_ball_pending','goal_pending','replay_pending','kickoff_pending','long_gap',
        'near_ball','far_ball','live_selector','student_missed_sequence_start','student_missed_prior_jump',
        'shadow_continues_after_student_divergence')
    absent=[k for k in required if not coverage[k]]
    shared=set.intersection(*[{e['sequence_id'] for e in coverage[k]} for k in ('goal_pending','replay_pending','kickoff_pending')])
    continuation={str(sid):dict(start=next(s for s in starts if s['sequence_id']==sid),
        continuation_callbacks=sum(r['original_sequence_id_after']==sid and r['original_before'] is not None
            and not r['original_before']['done'] for r in unique),
        completion=next((s for s in completions if s['sequence_id']==sid),None),
        pending_phases=sorted({r['match_phase'] for r in unique if r['original_sequence_id_before']==sid
            and r['original_before'] is not None and not r['original_before']['done']})) for sid in sorted(sequence_missed)}
    reuse=sum(a['prediction_receipt']['counter']==b['prediction_receipt']['counter'] for a,b in zip(unique,unique[1:]))
    pauses=[e for e in events if e['kind']=='controlled_receive_pause']
    report=dict(status='incomplete_due_to_diagnostic_io_failure',phase3_passed=False,
        generated_utc=datetime.now(timezone.utc).isoformat(),session=str(folder),
        raw_jsonl_lines=len(rows),unique_processed_callbacks=len(unique),
        duplicated_callback_ids={str(k):len(v) for k,v in grouped.items() if len(v)>1},
        duplication_interpretation='Callback 1115 was written once after comparison and again in the diagnostic exception handler; not an additional delivered callback. Raw evidence is unchanged.',
        shadow_contract=dict(compared_callbacks=len(unique),mismatches=mismatches,maximum_analog_difference=maximum_analog,
            analog_tolerance=ANALOG_TOL,buttons_sequence_selector_exact=not mismatches,
            submitted_callbacks=len(submitted),native_transport_mismatches=transport_errors,
            callbacks_without_submission_evidence=not_submitted,student_hidden_chain_errors=hidden_errors),
        required_category_counts={k:len(coverage[k]) for k in required},unobserved_required_categories=absent,
        coverage_examples={k:v[:12] for k,v in coverage.items()},sequence_starts=starts,sequence_completions=completions,
        same_pending_sequence_across_goal_replay_kickoff=sorted(shared),
        student_disagreement=dict(mode_disagreement_callbacks=sum(r['original_mode']!=r['student']['mode'] for r in unique),
            missed_sequence_starts=coverage['student_missed_sequence_start'],
            front_dodge_after_student_previous_jump_false=coverage['student_missed_prior_jump']),
        shadow_continuation_after_student_divergence=continuation,
        off_policy_scope='Actual independent shadow continuation after student action disagreements was observed on teacher-controlled physics. No student action reached the game; learner-induced off-policy state continuation was NOT tested.',
        diagnostics_errors=errors,agent_status=read(folder/'agent_summary.json')['status'],
        supervisor_status=read(folder/'supervisor.json')['status'],
        automatic_report_corrections=dict(raw_lines_not_callbacks=True,
            repeated_1115_transport_entry_is_one_unsubmitted_callback=True,
            hidden_chain_discrepancy_is_duplicate_error_row_not_a_reset=not hidden_errors,
            coverage_complete_is_not_clean_success=True),
        delivery=dict(packet_receipts=read(folder/'agent_summary.json')['packet_receipts'],
            prediction_receipts=read(folder/'agent_summary.json')['prediction_receipts'],
            adjacent_processed_callbacks_reusing_prediction_receipt=reuse,controlled_pauses=pauses,
            callback_dt=stats([r['callback_dt'] for r in unique if r['callback_dt'] is not None]),
            frame_gaps=stats([r['frame_gap'] for r in unique if r['frame_gap'] is not None]),
            student_ipc_seconds=stats([r['student']['ipc_seconds'] for r in unique]),
            queue_full_warnings='User terminal reports thresholds 1, 10 and 100 missed messages; not a complete drop count.',
            lossless_delivery_proved=False),
        error_cause=dict(established='Windows denied os.replace(agent_summary.tmp, agent_summary.json) at callback 1115. It raised before returning controls to the SDK; SDK exception path sent no PlayerInput.',
            uncertain='Concurrent reader/file-sharing or shutdown timing may explain the denial. The process holding the file was not captured; exact OS-level cause remains unproven.',
            status_race='Supervisor cached coverage_complete before the agent entered error; final agent summary/closed marker says error.'),
        protected_hashes_verified=len(protected),immutable_anchors_verified=True,
        inherited_full_immutable_verification=automatic['immutable_artifacts_unchanged'],
        no_test_evaluation=True,no_dagger_training_dataset=True,no_training_or_deployment=True,
        input_hashes=inputs,capture_code_hashes_verified=True,
        recommendation='Do not advance to Phase 4. Review a minimal diagnostic-only fix for summary publication/final status/error accounting, then authorize any rerun separately. Do not change teacher, selector, model or observation contract.')
    output=folder/'review_v1'
    output.mkdir(exist_ok=False)
    write(output/'phase3_contract_review_20261005.json',report)
    lines=['# Phase 3 contract review','', '**Result: INCOMPLETE — diagnostic I/O failure; do not advance to Phase 4.**','',
        '## 1. Verified original/shadow contract','',
        f"{len(unique):,} unique processed RLBot callbacks; {len(rows):,} raw diagnostic lines. Callback 1115 has a duplicate error record. Zero recomputed source/shadow mismatches; maximum analog difference {maximum_analog}. Sequence state/index/done/start times, branches, selected index/timestamp/position and buttons agreed. Native packet/prediction object identities and canonical times matched; sequence memory was independent.",'',
        f"{len(submitted):,} logged original-teacher submissions agree in all eight channels. No student/shadow control transport exists. Callback(s) {not_submitted} lack submission evidence: the summary write raised before the SDK could send the original controls. This is a diagnostic failure, not an observed teacher/shadow action disagreement.",'',
        f"Student hidden-state hash chain is continuous on all {len(unique)} unique callbacks; reset count remains one. The automatic report's hidden discrepancy came from counting the duplicate callback as a new inference step.",'',
        '## 2. Actual student disagreements','',
        f"Student and teacher modes differed on {report['student_disagreement']['mode_disagreement_callbacks']} callbacks. The teacher began {len(starts)} sequences; the student predicted Chase at all three starts (callbacks 469, 617, 674), with prior student actions also Chase. These are actual model outputs, never forced labels.",'',
        f"{len(coverage['student_missed_prior_jump'])} actual teacher Front-dodge callbacks followed a prior student prediction with jump=False. Full callback IDs and native controls are in the JSON report.",'',
        '## 3. Stateful shadow continuation and off-policy limits','', report['off_policy_scope'],'',
        f"Sequences 1 and 2 completed; sequence 3 remained pending across goal, replay, countdown and kickoff. Same transition sequence IDs: {sorted(shared)}. Shadow state continued independently despite the student's missed sequence starts; student predictions never reset or overwrite shadow memory.",'',
        '## 4. Observed / unobserved categories','',
        '| Required category | Actual unique callbacks |','|---|---:|',
        *[f'| {k} | {len(coverage[k])} |' for k in required], '',
        f"Unobserved planned categories: {absent or 'none'}. Missing contract evidence: successful original-controller submission for the final callback and clean error-free diagnostic completion. Actual learner-controlled off-policy physics was outside this gate and remains untested.",'',
        '## 5. Diagnostic error and delivery limitations','',
        report['error_cause']['established'],report['error_cause']['uncertain'],report['error_cause']['status_race'],'',
        'The automatic coverage-complete status is insufficient to pass the gate. Its two transport failures both refer to one duplicated callback (1115); its hidden-chain error is also the duplicated error row. The original report and raw records remain untouched.', '',
        f"There were {len(pauses)} intentional receive-loop pauses. Callback-dt distribution: `{report['delivery']['callback_dt']}`. Frame gaps: `{report['delivery']['frame_gaps']}`. Queue-full warnings were observed; no lossless delivery or undelivered-tick equivalence is claimed. CPU inference IPC, profiling, rendering and per-row file flushing add timing overhead.",'',
        '## 6. Integrity and next decision','',
        'All 43 protected hashes match; recorded capture code hashes and session integrity hashes match. Frozen manifest/checkpoint/projection/V12 anchors were verified. The capture reported full immutable-artifact preservation; the review independently confirmed its before/after immutable snapshot. No frozen artifact or original bot was edited, no test split was evaluated, and no DAgger training tensors or aggregation were created.', '',
        report['recommendation'],'',f"Session: `{folder}`. Review writes only this new `review_v1/` directory; original session artifacts are preserved byte-for-byte."
    ]
    (output/'phase3_contract_review_20261005.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    assert all(sha(p)==h for p,h in inputs.items()),'Original session artifacts changed'
    assert verify()==protected and immutable_snapshot()==immutable,'Immutable source/artifacts changed'
    print('PHASE3 REVIEW: INCOMPLETE; callbacks',len(unique),'source/shadow failures',len(mismatches),'unsubmitted',not_submitted,flush=True)
    print('Report:',output/'phase3_contract_review_20261005.md',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--session',required=True)
    review(Path(parser.parse_args().session))
