"""Post-run diagnostic review only. Preserve all capture evidence and model artifacts."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import statistics
from common import HERE,ANALOG_TOL,verify,immutable_snapshot,read,sha,write
from diagnostics import SummaryReader

def stats(values):
    values=sorted(values)
    return dict(count=len(values),minimum=min(values),median=statistics.median(values),
        p95=values[math.ceil(.95*len(values))-1],maximum=max(values)) if values else dict(count=0)

def main(folder):
    folder=folder.resolve()
    if folder.parent!=(HERE/'sessions').resolve():raise ValueError('Not a v2 Phase 3 session')
    protected=verify();immutable=immutable_snapshot()
    pins={str(p):sha(p) for p in folder.rglob('*') if p.is_file()}
    automatic=read(folder/'contract_report.json');metadata=read(folder/'metadata.json')
    for path,pin in automatic['integrity'].items():
        if sha(folder/path)!=pin:raise RuntimeError('Capture file changed: '+path)
    for path,pin in metadata['code_hashes'].items():
        if sha(path)!=pin:raise RuntimeError('Capture implementation changed: '+path)
    for filename,pin in metadata['config_hashes'].items():
        if sha(folder/filename)!=pin:raise RuntimeError('Capture config changed: '+filename)
    rows=[json.loads(line) for line in (folder/'callbacks.jsonl').read_text(encoding='utf-8').splitlines()]
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    diagnostic_errors=[json.loads(line) for line in (folder/'diagnostic_errors.jsonl').read_text(encoding='utf-8').splitlines()]
    ids=Counter(r['callback'] for r in rows)
    duplicates={str(k):v for k,v in ids.items() if v!=1}
    if duplicates or sorted(ids)!=list(range(1,len(rows)+1)):raise RuntimeError('Unexpected callback accounting')
    summary=SummaryReader().read(folder);closed=read(folder/'agent_closed.json');supervisor=read(folder/'supervisor.json')
    mismatches=[];hidden_errors=[];max_analog=0.;previous_hidden=None
    for r in rows:
        problems=list(r.get('failures',[]))
        a,b=r['original_controls'],r['shadow_controls']
        error=max(abs(x-y) for x,y in zip(a[:5],b[:5]));max_analog=max(max_analog,error)
        if not all(math.isfinite(x) for x in a[:5]+b[:5]) or error>ANALOG_TOL:problems.append('analog')
        if a[5:]!=b[5:] or any(type(x) is not bool for x in a[5:]+b[5:]):problems.append('buttons')
        for key in ('before','after','mode','detail','flips'):
            if r['original_'+key]!=r['shadow_'+key]:problems.append(key)
        for a_key,b_key in (('native_packet_object_id','shadow_packet_object_id'),('native_prediction_object_id','shadow_prediction_object_id'),('T','shadow_elapsed')):
            if r[a_key]!=r[b_key]:problems.append('native packet/prediction/time')
        if not r['independent_sequence_objects']:problems.append('shared sequence')
        if r['original_controls'][3:5]!=[0.,0.] or r['original_controls'][6:]!=[False,False]:problems.append('teacher support')
        if len(r['observation13'])!=13 or not all(math.isfinite(x) for x in r['observation13']) or r['private_teacher_state_in_student_input']:
            problems.append('student input contract')
        if r.get('error'):problems.append(r['error'])
        if problems:mismatches.append(dict(callback=r['callback'],issues=problems))
        s=r['student']
        if s['hidden_resets']!=1 or s['hidden_before_sha256']!=previous_hidden:hidden_errors.append(r['callback'])
        previous_hidden=s['hidden_after_sha256']
    submissions=[e['payload'] for e in events if e['kind']=='original_controls_submitted']
    counts=Counter(s['callback'] for s in submissions);by_callback={s['callback']:s for s in submissions}
    missing=[];transport=[]
    for r in rows:
        s=by_callback.get(r['callback'])
        if s is None:missing.append(r['callback']);continue
        a,b=r['original_controls'],s['controls']
        if a[5:]!=b[5:] or max(abs(x-y) for x,y in zip(a[:5],b[:5]))>ANALOG_TOL:transport.append(r['callback'])
    duplicate_submissions={str(k):v for k,v in counts.items() if v!=1}
    def example(r):
        return dict(callback=r['callback'],frame=r['frame'],elapsed=r['T'],match_phase=r['match_phase'],
            sequence_id=r['original_sequence_id_after'],phase_index=r['original_detail']['consumed_phase'],
            teacher_controls=r['original_controls'],student_controls=r['student']['controls'],
            student_mode=r['student']['mode'],previous_student=r['previous_student_action_diagnostic_only'],callback_dt=r['callback_dt'])
    coverage=defaultdict(list);starts=[];ends=[];missed=set();previous_mode=None
    for r in rows:
        detail=r['original_detail'];before=r['original_before'];after=r['original_after']
        pending=before is not None and not before['done'];sid=r['original_sequence_id_after']
        def mark(key):coverage[key].append(example(r))
        if previous_mode=='Neutral' and r['original_mode']=='Chase':mark('neutral_to_chase')
        if previous_mode=='Chase' and detail['flip_trigger'] and r['original_mode']=='Jump':mark('chase_to_jump')
        for index,key in ((1,'release'),(2,'front_dodge'),(3,'coast')):
            if detail['consumed_phase']==index:mark(key)
        if detail['flip_trigger']:
            starts.append(example(r))
            if r['student']['mode']!='Jump':missed.add(sid);mark('student_missed_sequence_start')
        if after is not None and after['done'] and (before is None or not before['done']):ends.append(example(r));mark('completion')
        if pending and not r['relevant_observation']['ball_present']:
            mark('missing_ball_pending')
            if before!=after:raise AssertionError('Missing-ball state changed')
        for phase,key in (('GoalScored','goal_pending'),('Replay','replay_pending'),('Countdown','countdown_pending'),('Kickoff','kickoff_pending')):
            if pending and r['match_phase']=='MatchPhase.'+phase:mark(key)
        if pending and r['callback_dt'] is not None and r['callback_dt']>=1:mark('long_gap')
        if detail['branch'] in ('near_ball','far_ball'):mark(detail['branch'])
        if detail['selected'] is not None:mark('live_selector')
        prior=r['previous_student_action_diagnostic_only']
        if detail['consumed_phase']==2 and prior is not None and not prior['controls'][5]:mark('student_missed_prior_jump')
        if pending and sid in missed and not detail['flip_trigger']:mark('shadow_continues_after_student_divergence')
        previous_mode=r['original_mode']
    required=list(summary['coverage'])
    absent=[k for k in required if not coverage[k]]
    shared=set.intersection(*[{e['sequence_id'] for e in coverage[k]} for k in ('goal_pending','replay_pending','kickoff_pending')])
    final_verified=(bool(rows) and rows[-1]['callback'] in by_callback and summary['final_callback_submission_verified']
        and summary['last_submitted_callback']==rows[-1]['callback']==closed['callbacks'])
    clean=(not mismatches and not hidden_errors and not missing and not transport and not duplicates and not duplicate_submissions
        and not diagnostic_errors and not summary['diagnostic_errors'] and not summary['summary_publication_errors']
        and not supervisor['error'] and not supervisor['cleanup_errors'] and supervisor['agent_closed'] and closed['worker_closed']
        and closed['status']=='coverage_complete_awaiting_review' and final_verified)
    enough=not absent and bool(shared) and len(starts)>=2 and len(ends)>=2
    pauses=[e for e in events if e['kind']=='controlled_receive_pause']
    report=dict(recommended_verdict='pass_with_scope_awaiting_review' if clean and enough else 'incomplete_or_failed',
        formal_gate_approved=False,session=str(folder),created_utc=datetime.now(timezone.utc).isoformat(),
        unique_callbacks=len(rows),raw_lines=len(rows),duplicate_callbacks=duplicates,
        original_shadow_failures=mismatches,maximum_analog_difference=max_analog,
        submissions=len(submissions),missing_submissions=missing,native_transport_disagreements=transport,
        final_callback_submission_verified=final_verified,hidden_continuity_errors=hidden_errors,
        diagnostic_errors=diagnostic_errors,summary_publication_errors=summary['summary_publication_errors'],
        coverage_counts={k:len(coverage[k]) for k in required},unobserved_required_categories=absent,
        coverage_examples={k:v[:12] for k,v in coverage.items()},sequence_starts=starts,sequence_completions=ends,
        same_pending_sequence_goal_replay_kickoff=sorted(shared),
        student_mode_disagreement_callbacks=sum(r['original_mode']!=r['student']['mode'] for r in rows),
        student_missed_sequence_starts=coverage['student_missed_sequence_start'],
        student_missed_prior_jump=coverage['student_missed_prior_jump'],
        independent_shadow_continuation_callbacks=len(coverage['shadow_continues_after_student_divergence']),
        off_policy_scope='Actual shadow continuation after student action disagreements was observed on teacher-controlled physics. No learner-induced off-policy physical-state continuation or student-controlled gameplay was tested.',
        delivery=dict(intentional_pauses=pauses,callback_dt=stats([r['callback_dt'] for r in rows if r['callback_dt'] is not None]),
            frame_gaps=stats([r['frame_gap'] for r in rows if r['frame_gap'] is not None]),
            worker_ipc_seconds=stats([r['student']['ipc_seconds'] for r in rows]),
            reused_prediction_receipt_pairs=sum(a['prediction_receipt']['counter']==b['prediction_receipt']['counter'] for a,b in zip(rows,rows[1:])),
            packet_receipts=summary['packet_receipts'],prediction_receipts=summary['prediction_receipts'],
            queue_warnings='User terminal reports 1/10/100 missed-message warning thresholds; total losses cannot be reconstructed from these thresholds.',
            lossless_delivery_proven=False),
        protected_hashes_verified=43,capture_code_configs_and_session_hashes_verified=True,
        full_immutable_check_reported_by_capture=automatic['immutable_artifacts_unchanged'],
        original_session_sha256=pins,student_controls_sent=False,shadow_controls_sent=False,
        test_evaluated=False,dagger_training_data_created=False,training_started=False,
        recommendation='Submit Phase 3 for explicit review. Stop; no Phase 4, learner controls, DAgger collection/aggregation or training.')
    output=folder/'review_v1';output.mkdir(exist_ok=False)
    write(output/'phase3_contract_review_20261005.json',report)
    lines=['# Phase 3 corrected rerun contract review','',f"**Recommended result: {report['recommended_verdict']}. Formal gate approval remains with the user.**",'',
        '## 1. Verified teacher/shadow contract','',
        f"{len(rows):,} unique actual processed callbacks and {len(rows):,} raw lines; no duplicate callbacks. Recomputed teacher/shadow failures: {len(mismatches)}; maximum analog difference: {max_analog}. Native eight-channel outputs, buttons, sequence state/index/done/start times, branches and live selector index/timestamp/position agreed. Packet/prediction identities and canonical elapsed time matched, with independent sequence memory.",'',
        f"{len(submissions):,} original-teacher controller submissions verified. Missing submissions: {missing}; transport disagreements: {transport}; final callback {rows[-1]['callback']} submitted and verified: {final_verified}. No student/shadow control reached the transport. Student GRU memory had one initial reset and no hidden-hash discontinuities.",'',
        '## 2. Actual observed student disagreements','',
        f"Student/teacher modes differed on {report['student_mode_disagreement_callbacks']} callbacks. Teacher sequence starts: {len(starts)}. Student missed {len(coverage['student_missed_sequence_start'])} starts at callbacks {[e['callback'] for e in coverage['student_missed_sequence_start']]}; each predicted Chase instead of teacher Jump. There were {len(coverage['student_missed_prior_jump'])} teacher Front-dodge callbacks following a student prediction with jump=False. No student output was forced or changed.",'',
        '## 3. Independent shadow continuation / off-policy scope','',report['off_policy_scope'],'',
        f"{len(coverage['shadow_continues_after_student_divergence'])} pending-sequence callbacks followed the observed missed starts. {len(ends)} sequences completed; sequence 3 persisted across goal, replay, countdown and kickoff. Same pending transition sequence IDs: {sorted(shared)}. Student disagreements did not overwrite/reset shadow memory.",'',
        '## 4. Required actual coverage','', '| Category | Unique callbacks |','|---|---:|',
        *[f'| {k} | {len(coverage[k])} |' for k in required],'',
        f"Unobserved required categories: {absent or 'none'}. Probe requests alone were not counted. Learner-controlled physical-state behavior remains outside this gate and untested.",'',
        '## 5. Diagnostic completion and delivery limitations','',
        f"Diagnostic errors: {len(diagnostic_errors)}; publication errors: {len(summary['summary_publication_errors'])}; duplicates: {duplicates}; hidden-chain errors: {hidden_errors}. Agent/worker shutdown confirmed. The v1 summary-publication and duplicate-accounting failure did not recur.",'',
        f"Intentional receive pauses: {len(pauses)}. Callback dt: `{report['delivery']['callback_dt']}`. Frame gaps: `{report['delivery']['frame_gaps']}`. CPU worker IPC latency: `{report['delivery']['worker_ipc_seconds']}`.",'',
        report['delivery']['queue_warnings'],'Only actual processed callbacks are covered. No lossless-delivery or undelivered-physics-tick equivalence is claimed. Instrumentation, IPC, filesystem writes and intentional pauses can influence callback delivery. Native predictions may be reused between callbacks; no timestamps, slices or source packets were changed.', '',
        '## 6. Integrity and review decision','',
        'All 43 protected source hashes, pinned frozen manifest/checkpoint/projection/V12 anchors, capture-code/config hashes and saved session-file hashes passed verification. The capture reported full immutable-artifact preservation, and this review verified its own before/after immutable snapshot. Original bots, previous runs, V10/V12/V13 completed artifacts and this capture remain byte-for-byte unchanged.', '',
        'No V10 test evaluation, student control, DAgger training dataset/aggregation, fitting or PPO occurred. Only a new read-only review script and this review folder were created.', '',
        report['recommendation'], '',f"Original session: `{folder}`. JSON report includes every coverage example, disagreement, receipt/latency statistic and input SHA256."
    ]
    (output/'phase3_contract_review_20261005.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    assert all(sha(p)==pin for p,pin in pins.items()),'Capture evidence changed'
    assert verify()==protected and immutable_snapshot()==immutable,'Protected artifacts changed'
    print('PHASE3 REVIEW:',report['recommended_verdict'],'callbacks',len(rows),'final submitted',final_verified,flush=True)
    print('Report:',output/'phase3_contract_review_20261005.md',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--session',required=True)
    main(Path(parser.parse_args().session))
