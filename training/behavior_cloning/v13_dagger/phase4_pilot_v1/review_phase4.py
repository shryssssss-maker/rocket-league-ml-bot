"""Read-only diagnostic review of the one completed pilot. No model/sample loading."""
from collections import Counter
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'runner'))
from common import sha,read,write,verify,CHANNELS

def main():
    session=HERE/'sessions/20261005_063422_960966'
    integrity=read(session/'diagnostic_integrity.json')
    for name,pin in integrity['files'].items():
        if sha(session/name)!=pin:raise RuntimeError('Diagnostic hash mismatch: '+name)
    protected=verify()
    report=read(session/'contract_report.json')
    rows=[json.loads(line) for line in (session/'callbacks.jsonl').read_text().splitlines()]
    events=[json.loads(line) for line in (session/'events.jsonl').read_text().splitlines()]
    sent=[e['payload'] for e in events if e['kind']=='student_controls_submitted']
    assert len(rows)==len(sent)==5238 and len({r['callback'] for r in rows})==5238
    assert all(r['callback']==s['callback'] and r['student']['controls']==s['controls'] for r,s in zip(rows,sent))
    metadata=read(session/'metadata.json')
    code_matches={p:sha(p)==pin for p,pin in metadata['code_hashes'].items()}
    assert all(code_matches.values())
    first=sent[0]['monotonic'];late=[dict(callback=s['callback'],offset_seconds=s['monotonic']-first) for s in sent if s['monotonic']-first>90]
    mean_errors={name:sum(abs(r['student']['controls'][i]-r['shadow_controls'][i]) for r in rows)/len(rows) for i,name in enumerate(CHANNELS)}
    sequences=[]
    for start in (r for r in rows if r['shadow_detail']['flip_trigger']):
        n=start['shadow_sequence_id_after']
        group=[r for r in rows if r['shadow_sequence_id_after']==n and (r['shadow_detail']['flip_trigger'] or r['shadow_before'] and not r['shadow_before']['done'])]
        sequences.append(dict(sequence_id=n,start_callback=start['callback'],student_start_mode=start['student']['mode'],
            pending_callbacks=len(group),shadow_jump_callbacks=sum(r['shadow_mode']=='Jump' for r in group),
            student_jump_button_during_shadow_jump=sum(r['shadow_mode']=='Jump' and r['student']['controls'][5] for r in group),
            shadow_dodge_callbacks=sum(r['shadow_mode']=='Front dodge' for r in group),
            student_modes=dict(Counter(r['student']['mode'] for r in group))))
    selected=[r for r in rows if r['shadow_detail']['selected']]
    selector_provenance=all(r['shadow_detail']['selected']==r['observation_probe'] for r in selected)
    reference=HERE/'train_reference_v1'
    ref_integrity=read(reference/'integrity.json')
    assert sha(reference/'statistics.json')==ref_integrity['statistics_sha256']
    descriptions=read(reference/'statistics.json')
    assert not descriptions['validation_read'] and not descriptions['test_read']
    review=dict(status='failed_strict_90_second_runner_limit_otherwise_complete_with_scope',
        original_automatic_status=report['status'],reason='Two submissions followed the strict 90-second cutoff; no live rerun authorized.',
        completed_pilot_report=report,verified_diagnostic_files=len(integrity['files']),
        protected_hashes_verified=len(protected),runner_code_matches=code_matches,
        first_to_last_submission_seconds=sent[-1]['monotonic']-first,post_90_second_submissions=late,
        canonical_elapsed_span=rows[-1]['T']-rows[0]['T'],frame_span=[rows[0]['frame'],rows[-1]['frame']],
        scores_first=rows[0]['scores'],scores_last=rows[-1]['scores'],per_channel_mae=mean_errors,
        shadow_selected_matches_actual_observation_probe=selector_provenance,
        actual_received_prediction_selection_count=len(selected),sequence_disagreement_evidence=sequences,
        shadow_jump_without_student_jump_callbacks=sum(r['shadow_mode']=='Jump' and not r['student']['controls'][5] for r in rows),
        possible_unexecutable_dodge_count=len(report['possibly_unexecutable_shadow_dodge_callbacks']),
        shutdown_log_interpretation='Pasted terminal reports SocketRelay disconnected unexpectedly during shutdown, after sessions began closing. Saved cleanup_errors empty, worker closed, final submission verified. Transport shutdown cannot be described as warning-free.',
        no_queue_full_warning_in_supplied_excerpt=True,complete_server_log_available=False,
        clock_limit='Submission monotonic clock records were coarse (many zero-duration callback latencies); CPU inference uses perf_counter. Clock resolution is not inferred as actual zero latency.',
        future_actions='Stopped; no correction, second run, aggregation, fitting, or PPO.',
        source_integrity_sha256=sha(session/'diagnostic_integrity.json'),reference_sha256=ref_integrity['statistics_sha256'])
    out=HERE/'review_v1';out.mkdir(exist_ok=False)
    write(out/'phase4_pilot_review_20261005.json',review)
    content=['# Phase 4 pilot review — 2026-10-05','',
        '**Classification: failed the strict 90-second runner limit; the single diagnostic pilot otherwise completed with scope.**', '',
        'The automatic report says `pilot_complete_awaiting_review`. Manual evidence review found two controls submitted after the approved cutoff: the final submission was at +90.031 seconds. This is a runner timing-contract violation, not a model-training result. No rerun or fix was performed.','',
        f"Session: `{session}`. Blue human_b versus Orange pinned 13D GRU student. One natural live 1v1; original teacher shadow-only.",'',
        '### 1. Actual learner-induced physical states','',
        '5,238 unique callbacks and 5,238 exact student-only transport submissions. 5,237 successor callbacks link to the preceding verified student submission. First/last frames: 13/10,645; canonical elapsed span: 88.600003 seconds. The first callback is not labeled learner-induced. All captured observations use the actual received packet; no requested or synthetic states were used.','',
        'Near ball: 3,839 callbacks; far ball: 1,399. Every snapshot had a ball and valid prediction. Scores remained Blue 0 / Orange 0; scoring ability, touch rate and useful contact were not independently measured.','',
        '### 2. Student versus independent shadow labels','',
        '| Mode | Student callbacks | Shadow callbacks |','|---|---:|---:|',
        *[f"| {mode} | {report['student_modes'].get(mode,0)} | {report['shadow_modes'].get(mode,0)} |" for mode in ('Neutral','Chase','Jump','Front dodge')],'',
        f"Mode agreement: 4,012/5,238 ({4012/5238:.2%}). Native eight-channel agreement within 1e-6 analog tolerance: 212/5,238 ({212/5238:.2%}). Steering-sign disagreements outside ±0.02: 379.",'',
        '| Channel | MAE versus shadow |','|---|---:|',
        *[f'| {name} | {value:.6f} |' for name,value in mean_errors.items()],'',
        'Yaw/roll remained exactly zero; boost/handbrake remained false in both paths. Steering absolute error: median 0.206024, p95 1.0, maximum 1.402245.','',
        'The shadow started and completed nine original sequences; the student missed all nine Jump-mode starts and emitted no pure Jump mode anywhere. Of 41 shadow Jump callbacks, 40 had no student jump button. The shadow continued independently through 631 pending callbacks after missed starts. There were 51 shadow Front-dodge callbacks while the actual learner car was grounded and had not jumped. These are potentially incompatible action prerequisites, not proof that the labels form useful learning targets.','',
        f"Missed starts: {report['shadow_missed_start_callbacks']}. Per-sequence evidence is in the JSON review.",'',
        'The shadow made 1,064 actual prediction selections; all matched the observation probe index/timestamp/position on those callbacks. Probe selection on other callbacks does not imply the original teacher selected a slice while a sequence was pending. This run verifies input provenance and independent continuation, not a new teacher-controlled-physics counterfactual.','',
        '### 3. Train-only descriptive distribution comparisons','',
        'Primary reference: 57,794 Orange-side train callbacks (pilot_01/v10_006). Secondary: all 107,400 train callbacks. No validation/test samples were accessed. Comparisons did not choose, modify or fit the policy.','',
        '| Feature (primary reference, ALL phase aggregate) | Outside train min/max | Outside p01–p99 | Callback JS bits | Hold-weighted JS bits |',
        '|---|---:|---:|---:|---:|',
        *[f"| {name} | {v.get('outside_reference_minmax','NA')} | {v.get('outside_reference_p01_p99','NA')} | {v.get('callback_js_bits',0):.6f} | {v.get('hold_js_bits',0):.6f} |" for name,v in report['train_only_descriptive_comparison']['primary']['ALL'].items()],'',
        'All-feature phase/mask-conditioned quantiles, excursions and primary/secondary histograms are preserved in the JSON report. Aggregate speed median was about 1,199.6 UU/s; median car-to-ball distance about 784.3 UU. These marginal comparisons describe the observed learner trajectory and do not establish causal distribution shift, joint-state coverage, or DAgger benefit.','',
        'Actual same-phase hold occupancy: Countdown 5.741667 s; Kickoff 2.958333 s; Active 79.866670 s. Two intervals crossing phase boundaries were excluded; final hold is censored. These are canonical-game-time weights, not proof that the wall window equaled active play.','',
        '### 4. Unobserved categories','',
        'No missing-ball callbacks, goal events, replay, pending-sequence goal/replay/countdown overlap, or gaps greater than five physics ticks were observed. Pending shadow sequences did overlap Kickoff for 71 callbacks. No probes, pauses, replay manipulation or additional run were used to force missing coverage.','',
        '### 5. Callback, queue and shutdown limits','',
        'Callback dt: median 16.6664 ms; p95 16.6702 ms; p99 25.0015 ms; max 41.6679 ms. Model inference: median 0.4394 ms; p95 0.6772 ms; p99 0.8162 ms; max 30.6122 ms.','',
        'Callback-to-submission clock: median 0 ms, p95/p99 16 ms, max 47 ms. The coarse monotonic readings must not be interpreted as zero processing cost. This metric includes shadow work, IPC/inference and diagnostic logging, and is not game-application latency.','',
        '5,238 distinct packet receipt counters and prediction receipt counters; no consecutive prediction-receipt reuse. These counts do not prove lossless physics-tick delivery. The pasted terminal excerpt contains no queue-full warning, but no complete server log is available for a global queue claim.','',
        'The supervisor stopped at its bounded completion condition; final callback 5,238 was verified submitted, worker closed, and agent_closed.json exists. Cleanup errors were empty. The SDK logged “SocketRelay disconnected unexpectedly!” during session shutdown; shutdown therefore cannot be called warning-free. agent_closed.status=running describes the internal agent before external stop, not evidence it kept running afterward.','',
        f"Strict cutoff failures: {late}. The supervisor polls every 50 ms and stops/disconnects outside the receive loop; this implementation did not enforce a hard transport cutoff. No timing semantics were changed after the run.",'',
        '### 6. Integrity and final disposition','',
        f"{len(integrity['files'])} diagnostic-file hashes verified; all runner code matches session metadata; all 43 protected hashes verified. Saved supervisor reports immutable V10/V12/V13 artifacts unchanged. Frozen dataset, original teacher, bot directories, input contract and prediction selector were untouched.",'',
        'Zero recorded callback/transport/hidden-chain/provenance/publication errors or duplicate callback records. The timing-limit failure is additional to these automatic counts and is explicitly recorded in this review.','',
        'The pilot produced real learner-induced trajectory evidence and faithfully independent shadow behavior within the validated processed-callback scope. It does not prove competence, label usefulness, lossless delivery, or DAgger improvement. Diagnostics remain quarantined. Stop for review: no aggregation, retraining, PPO or second run.','',
        f"Full machine evidence: `{out/'phase4_pilot_review_20261005.json'}`. Original report and capture are retained unchanged."
    ]
    (out/'phase4_pilot_review_20261005.md').write_text('\n'.join(content)+'\n',encoding='utf-8')
    write(out/'integrity.json',dict(review_code_sha256=sha(Path(__file__)),
        json_sha256=sha(out/'phase4_pilot_review_20261005.json'),markdown_sha256=sha(out/'phase4_pilot_review_20261005.md')))
    print('PHASE4 REVIEW:',review['status'])
    print('Verified diagnostic files:',len(integrity['files']),'Protected hashes:',len(protected))
    print('Report:',out/'phase4_pilot_review_20261005.md')
    print('Stopped. No new live run, dataset/sample reads, aggregation or fitting.')

if __name__=='__main__':main()
