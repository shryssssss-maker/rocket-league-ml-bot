"""Prepare a proposal only. No runnable pilot, connection, dataset reads or training."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
V13=HERE.parent
PHASE3=V13/'phase3_shadow_v2'
spec=importlib.util.spec_from_file_location('_phase4_readonly_pins',PHASE3/'common.py')
pins=importlib.util.module_from_spec(spec);spec.loader.exec_module(pins)

def main():
    protected=pins.verify()
    before=pins.immutable_snapshot()
    phase3_before={str(p):pins.sha(p) for p in PHASE3.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    destination=HERE/'launch_readiness_specification_20261005.json'
    if destination.exists():raise RuntimeError('Specification exists; preserve it and stop for review')
    projection=pins.read(V13/'projection_13d_v1/manifest.json')
    accepted=pins.read(PHASE3/'sessions/20261005_044629_296920/review_v1/phase3_contract_review_20261005.json')
    assert accepted['recommended_verdict']=='pass_with_scope_awaiting_review'
    proposal=dict(version='v13_phase4_student_pilot_spec_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        status='specification_proposal_awaiting_explicit_review',phase3_formally_accepted_by_user=True,
        phase4_launch_approved=False,runnable_live_launcher_created=False,live_launched=False,
        proposed_run=dict(matches=1,wall_seconds_after_first_verified_student_submission=90,startup_seconds=180,
            teacher_shadow_team='Orange',student_team='Orange',human_team='Blue',
            opponent_id='human_a only if the original pilot player actually participates; confirm identity before launch',
            natural_play=True,state_setting=False,intentional_receive_pauses=False,
            retain_replays=True,retain_countdown=True,
            scope='A bounded partial match, not a full-match win-rate experiment',
            decision_status='90-second configuration proposed; user preference question pending'),
        authoritative_path='Real Rocket League -> actual RLBot packet/BallPrediction -> unchanged 13D policy -> student ControllerState -> real Rocket League',
        controller=dict(sole_sender='13D student',teacher_controls_sent=False,teacher_fallback=False,
            student_native_channels=list(pins.CHANNELS),modes=list(pins.MODES),
            decode='Existing argmax 4-mode and continuous tanh steering decode; no quantization',
            required_zero_support=['yaw','roll','boost','handbrake'],
            inference_process='Existing training Python, CPU one thread, pinned accepted model',
            ipc_timeout_seconds=2,source='Reuse Phase3 worker/model/decoder; no fitting or parameter changes'),
        checkpoint=dict(path=str(pins.CHECKPOINT),sha256=pins.CHECKPOINT_PIN,parameters=26181,
            architecture='13 -> Linear64/ReLU -> causal one-layer GRU64 -> 4 mode logits + continuous steering'),
        observation=dict(features=projection['result_feature_list'],source_manifest_sha256=pins.MANIFEST_PIN,
            projection_sha256=projection['projection_code_sha256'],builder_sha256=pins.OBS_PIN,
            source_features='Accepted 18D builder, retain columns 0 through 12 unchanged',
            private_teacher_memory_in_inputs=False,previous_action_inputs=False,first_callback_dt=0,
            duplicate_or_backward_elapsed='Record violation and stop; no clock repair or hidden reset'),
        prediction=dict(provider='Actual delivered RLBot BallPrediction, reuse logged explicitly',
            selector='int((elapsed + 2 - first_slice_time) * 120)',
            gameplay_branch='Exact original shadow branch, including IndexError on empty prediction when lookup is reached',
            observation_probe='Same nonempty-prediction guard and original helper as accepted frozen recorder/Phase3; distinct from shadow gameplay decision',
            prohibited=['interpolation','timestamp shifts','clamping','synthetic slices','RocketSim','stale cutoff']),
        memory=dict(student_reset='Once at match/session initialization only',
            shadow_reset='Once at match/session initialization only',
            survives=['goal','replay','countdown','kickoff','missing ball','long callback gaps','student/shadow disagreement'],
            source='Byte-identical validated OriginalShadow source logic and independent tracker/sequence with non-game sink',
            labels='Temporal state is diagnostic only; neither copied into student nor reset to match student actions'),
        execution_order=['Receive actual native packet/prediction and record receipt provenance',
            'Build unchanged 13D input from current actual state/time and previous callback elapsed time',
            'Advance independent original shadow once on that same callback stream',
            'Advance student GRU once; decode and validate native controls',
            'Submit only the student ControllerState; record submission evidence and receipt-time latency',
            'On the next actually processed callback, link received physical state to prior student submission',
            'Publish immutable diagnostic snapshots; never let publication I/O suppress otherwise valid student controls'],
        diagnostics=dict(role='Quarantined Phase4 pilot diagnostics; NOT an approved DAgger training dataset',
            files=['metadata.json','callbacks.jsonl','events.jsonl','diagnostic_errors.jsonl','summary_snapshots/*.json','agent_closed.json','contract_report.md','contract_report.json'],
            record=['session/callback/frame IDs','canonical elapsed time and callback/frame gaps','match phase and agent/team identity',
                'packet/prediction receipt counters, monotonic time and object identities',
                'exact 13D float32 input','student logits, mode, continuous steering and exact eight controls',
                'actual student controller submission and hidden-state hash chain',
                'shadow eight controls, branch, selected index/timestamp/position and attempted selection',
                'shadow sequence ID, before/after phase/index/start/done state',
                'observation selector probe separately from shadow gameplay selection',
                'current actual car/ball/opponent physics, scores and transition events',
                'native air_state, has_jumped, has_double_jumped, has_dodged and demolition fields',
                'previous submitted student action and link to next actually received state'],
            no_training_tensors=True,no_split_assignment=True,no_aggregation=True,
            response_link_scope='Next received state is an observed result after student action hold, not a single-action counterfactual or exact per-physics-tick causal attribution'),
        disagreement_metrics=dict(mode_confusion=True,analog_mae_and_max_by_channel=True,
            buttons_precision_recall=True,undefined_rare_metrics='null with subset counts',
            steering_sign_deadband=.02,exact_native_agreement_tolerance=pins.ANALOG_TOL,
            subsets=['near/far','missing ball','playable/nonplayable phase','shadow sequence phase and boundary','long callback gaps'],
            interpretation='Live agreement with original shadow is distinct from gameplay competence or label usefulness'),
        state_distribution=dict(reference_split='train ONLY; never read/evaluate validation/test samples',
            primary_reference_matches=['pilot_01','v10_006'],primary_reason='Natural Orange-teacher matches, matching proposed student side',
            secondary_reference_matches=['pilot_01','v10_001','v10_004','v10_006'],
            verification='Verify frozen manifest and every required manifest-listed artifact before semantic reads; stop on missing/hash mismatch',
            preparation='Read-only reference statistics only after specification approval; user runs any lengthy audit command',
            conditioning=['Kickoff/Active separately from Countdown/Goal/Replay','ball-relative features only with ball present',
                'prediction-relative/time features only when prediction valid','first callback separated for dt'],
            metrics=['Counts/exposure by phase, masks, near/far, car-speed bands and relative geometry sectors',
                'Per-feature min, median, p05/p95 and p01/p99 under original normalized representation',
                'Fraction outside reference min/max and p01/p99 bounds',
                'Fixed training-reference quantile-bin occupancy and per-feature Jensen-Shannon divergence',
                'Both callback-count-weighted and actual elapsed-hold-time-weighted occupancy'],
            time_weighting='A logged submission receives the interval to the next processed callback; final unobserved hold excluded; crossed phases labeled transition intervals',
            degenerate_features='Record constant/repeated quantile edges explicitly; merge identical histogram edges for reporting only, never change observations',
            noncausal_limit='One short match, different opponents/sessions, human actions, phases and callback delivery confound attribution. No significance or competence claim from marginal divergence alone',
            frozen_records_modified=False,samples_rebalanced=False),
        off_policy_shadow=dict(definition='Actual states reached while student actions, not teacher actions, control the game',
            required_evidence=['Verified student-only submissions before the states labeled learner-driven',
                'Original shadow starts sequence while student does not produce the corresponding Jump',
                'Subsequent actual learner-controlled callbacks show persistent original shadow sequence state',
                'Shadow Front dodge after student omitted preceding jump, grouped by native air_state and has_jumped',
                'Actual measured timing, ground/air status, ball-relative geometry and prediction selection at those disagreements'],
            missing_evidence='Explicitly unobserved; no probes, teacher fallback, synthetic history or repeated rounds to manufacture coverage',
            usefulness_limit='A stateful original expert can emit committed sequence actions in a physical state where the student did not perform the prerequisite. Report such cases; do not silently reset/relabel or assume these labels are suitable for training'),
        stopping=dict(normal='90-second proposed wall limit after first verified student submission, or natural match end, whichever occurs first',
            immediate=['user Ctrl+C','source/shadow selector exception','nonfinite/malformed/unsupported model output',
                'missing controlled car','observation/timing contract violation','worker exit or 2-second timeout',
                'unexpected controller sender or native transport mismatch','diagnostic integrity failure','server disconnect'],
            action_on_error='Stop the match and disconnect; no teacher takeover, emergency scripted policy, hidden-state repair or synthetic inputs',
            log_limit_per_file_mib=256,prohibition='Never auto-start a second match'),
        review=dict(technical_pass='Every recorded callback accounted for, student-only native controls verified, final submission/shutdown clean, masks/times/input hashes valid, no source/artifact changes',
            research_evidence='Separately classify actual learner-driven state occupancy and sequence-divergence continuation as observed or unobserved; technical completion alone does not satisfy this research evidence',
            report=['match duration/callbacks and phases','student/shadow disagreement and rare-mode subsets','learner-driven state occupancy versus frozen train-only reference',
                'stateful off-policy examples and potentially unsupported shadow labels','unobserved categories','callback delivery/IPC/queue/error limits','integrity/source hashes','pass-with-scope versus incomplete/failure'],
            no_automatic_phase5_or_training=True),
        required_preflight=['43 protected hashes and frozen artifact/source pins match; prior phases preserved',
            'Shadow source/worker/decoder match validated code and student memory continuity survives phases',
            'No teacher/shadow PlayerInput transport exists; only student outputs can be submitted',
            'Natural 1v1 config: Epic, rendering optional, state-setting disabled, no receive-pause planner',
            'Immutable summary publication, error accounting and final student submission gating',
            'First callback/history/dt and exact 13D feature projection checks',
            'Phase4-only diagnostics paths and no dataset/test/training imports in live path',
            'Read-only training-reference metrics verified before any sample reads',
            'Bounded timeout/user-stop/worker-failure paths do not trigger teacher fallback'],
        preflight_complete=False,protected_hashes_verified=len(protected),
        frozen_baseline_statistics_computed=False,reference_tensor_samples_read=False,
        validated_phase3_dependency_hashes={name:pins.sha(PHASE3/name) for name in ('reference.py','student_worker.py','student_client.py','diagnostics.py')})
    pins.write(destination,proposal)
    (HERE/'launch_readiness_specification_20261005.md').write_text(render(proposal)+'\n',encoding='utf-8')
    assert pins.verify()==protected and pins.immutable_snapshot()==before
    assert all(pins.sha(path)==pin for path,pin in phase3_before.items())
    print('PHASE4 SPECIFICATION PREPARED ONLY. No runner/launch/training.',flush=True)
    print('Report:',HERE/'launch_readiness_specification_20261005.md',flush=True)

def render(p):
    return '\n'.join([
        '# Phase 4 student-controlled pilot — launch-readiness specification','',
        '**Proposal for explicit review. No Phase 4 live runner exists yet; no live launch or preflight has occurred.**','',
        'Phase 3 is formally accepted with scope: faithful independent shadow on processed callbacks under teacher control. Lossless delivery, learner-induced states, expert-label usefulness, student competence and DAgger benefit remain unproven.', '',
        '## Proposed run — preference pending','',
        'One natural 1v1 pilot: Orange 13D student against Blue human, 90 seconds of wall time after the first verified student submission, plus at most 180 seconds startup. Keep replay/countdown; stop earlier if the match ends. Confirm who plays and record their actual opponent ID. No controlled state-setting, deliberate receive pauses or extra collection rounds. This is a partial-match diagnostic, not a win-rate test.', '',
        'The 90-second duration is a proposal awaiting the user preference response; no dependent runner is implemented before that choice and the specification are approved.', '',
        '## Control, observation and memory contract','',
        f"Frozen checkpoint: `{p['checkpoint']['path']}`; SHA256 `{p['checkpoint']['sha256']}`. Architecture and decoder remain unchanged (26,181 parameters). Existing training Python performs CPU inference; existing RLBot Python handles the game.",'',
        'Only the student ControllerState reaches RLBot PlayerInput. Original teacher is shadow-only, using the byte-identical validated original-source actor with independent sequence/boost-tracker state and a non-game sink rejecting control/state messages. No teacher takeover or hybrid controller is permitted, including on error.', '',
        f"13D features: `{', '.join(p['observation']['features'])}`. Same frozen builder/scales/masks, columns 0–12 only, no previous-action inputs and no private sequence labels. First callback dt is zero. Duplicate/backward clock is an explicit contract failure, not an invented reset.",'',
        'Use the exact actual delivered native packet, prediction and canonical elapsed time. Preserve selector `int((elapsed + 2 - first_slice_time) * 120)`. Preserve the original empty-prediction exception in its gameplay lookup branch. The independent observation probe keeps the existing recorder guard and is labeled separately. No interpolation, timestamp repair, provider substitution or simulator.', '',
        'Student GRU and original shadow sequence reset once at match/session initialization, never on goal/replay/kickoff, missing ball, long gaps or disagreement. Advance each once per actual processed callback, without copying student actions into shadow sequence state.', '',
        '## Diagnostic evidence and action-to-state links','',
        *['- '+s for s in p['diagnostics']['record']], '',
        'Link each submitted student action to the next actually received state, retaining elapsed/frame gaps and last delivered prediction receipt. That next packet is an observed result after an action hold and intervening game physics/human play, not an exact isolated-action counterfactual. Stop/end leaves the final hold without a later observed state; report it as censored.', '',
        'Store isolated raw diagnostic JSONL and metadata only, with integrity hashes and immutable summary snapshots. Mark `not_training_dataset`; create no processed training tensors, train/validation/test assignment, DAgger aggregation or training job.', '',
        '## Learner-induced state distribution','',
        'Pre-register a train-only descriptive comparison. Primary reference: Orange-teacher natural training matches `pilot_01` and `v10_006`; secondary: all four training matches. Validate required artifact hashes before reading samples. No validation or test samples are read/evaluated. Do not compare the natural pilot to Phase 3 controlled-probe occupancy as if those distributions were equivalent.', '',
        *['- '+s for s in p['state_distribution']['metrics']], '',
        'Condition masks and phases: ball-relative features require ball-present, predicted features require prediction-valid, and Kickoff/Active are separated from nonplayable phases. Keep original normalized features; reference quantile bins are reporting only. Separate initial dt and crossed-phase hold intervals. Publish denominators and masks to prevent missing-ball zeros/replay exposure from masquerading as physical novelty.', '',
        p['state_distribution']['noncausal_limit'], '',
        '## Stateful shadow under actual learner control','',
        *['- '+s for s in p['off_policy_shadow']['required_evidence']], '',
        'Native physical-state diagnostics use the installed RLBot fields `air_state`, `has_jumped`, `has_double_jumped`, `has_dodged`, `dodge_elapsed` and demolition status. These are diagnostic labels only, never new policy inputs. Classify shadow Front-dodge commands issued while the student remained grounded/did not jump; do not automatically call them valid expert targets.', '',
        p['off_policy_shadow']['usefulness_limit'], '',
        'If rare sequence-divergence cases do not occur naturally, report them as unobserved and keep the research evidence incomplete. Do not manufacture coverage through probes or secretly run another match.', '',
        '## Bounds and failure policy','',
        *['- '+s for s in p['stopping']['immediate']], '',
        'On failure stop/disconnect and record the actual last student action hold and missing submissions. No substitute teacher/neutral/scripted action is introduced. Keep existing two-second worker timeout; measure inference/IPC/end-to-end control latency and actual callback timing. A final report requires verified last student submission and clean agent/worker shutdown; diagnostic publication cannot suppress a valid policy output.', '',
        '## Required preflight before live approval','',
        *['- '+s for s in p['required_preflight']], '',
        '## Review criteria and current status','',
        'Separate technical control/transport integrity from learner-state coverage, shadow-label behavior and gameplay competence. Report action agreement/error by mode/phase, rare subsets and sequence boundaries; undefined precision/recall stays null. Use existing 1e-6 analog agreement tolerance and ±0.02 steering-sign deadband for diagnostics, not controller quantization.', '',
        'Observed novelty or shadow disagreement does not establish that labels will improve a policy. No model-selection, retraining or test tuning follows. Formal pilot review must classify missing evidence and timing/queue constraints separately.', '',
        f"Protected hashes checked: {p['protected_hashes_verified']}; prior Phase 3 files and completed artifacts unchanged. Only this preparation script plus Markdown/JSON specification were created. No reference tensors were read, no distribution statistics computed, no launcher created and no game launched.",'',
        'Approve/revise the run configuration, distribution-reference/metrics and fail-stop protocol first. Then prepare the isolated runner and non-live preflight, and present their readiness evidence before explicit live-launch review. No executable launch command is supplied at this specification-only stage.'
    ])

if __name__=='__main__':main()
