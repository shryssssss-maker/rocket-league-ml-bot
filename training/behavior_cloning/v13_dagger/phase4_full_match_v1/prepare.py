"""Preparation only. Never starts a game or reads dataset samples."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
LAUNCH=HERE/'runner/launch.py'
body='''def readiness(folder,checks,reference,protected_ok,immutable_ok):
    complete=checks['status']=='code_preflight_passed' and reference['verified'] and protected_ok and immutable_ok
    report=dict(version='student_full_match_v1',status='ready' if complete else 'incomplete',
        nonlive_preflight_complete=complete,live_launched=False,checks=checks,reference=reference,
        protected_hashes_unchanged=43 if protected_ok else False,completed_artifacts_unchanged=immutable_ok,
        config_folder=str(folder),approved_protocol='One natural five-minute game: human_b Orange, pinned student Blue; native replays/countdown/overtime; original teacher shadow-only',
        runner_code_hashes={p.name:sha(p) for p in sorted(HERE.glob('*.py'))},
        approved_spec_sha256=sha(PILOT/'launch_readiness_specification_20261005.json'),
        model_sha256=CHECKPOINT_PIN,validation_or_test_samples_read=False,
        training_or_aggregation=False,one_run_only=True)
    write(PILOT/'launch_readiness_20261005.json',report)
    (PILOT/'launch_readiness_20261005.md').write_text(
        '# Five-minute match readiness\\n\\n'
        f"Status: {report['status']}. Non-live checks only; no game launched.\\n\\n"
        'Human_b plays Orange; the SAME pinned 13D GRU student plays Blue. Original teacher is shadow-only and has no control transport.\\n\\n'
        'One standard five-minute game ends at native MatchPhase.Ended, including replays/countdowns/overtime. No 90-second or five-minute wall cutoff. Ctrl+C or error makes the run incomplete. No second run is automatic.\\n\\n'
        'Same checkpoint, input projection, decoder, causal hidden state and original selector. No probes, intentional pauses, teacher fallback, training, aggregation or validation/test sample reads.\\n\\n'
        'Existing train-only descriptive statistics are reused unchanged. All-train secondary reference is relevant; inherited Orange-only primary is explicitly side-mismatched context for a Blue student.\\n\\n'
        f"43 protected hashes and all prior artifacts unchanged: {protected_ok and immutable_ok}.\\n",
        encoding='utf-8')

'''

def main():
    text=LAUNCH.read_text(encoding='utf-8')
    start=text.index('def readiness(');end=text.index('def run(',start)
    LAUNCH.write_text(text[:start]+body+text[end:],encoding='utf-8')
    report=HERE/'runner/pilot_report.py'
    text=report.read_text(encoding='utf-8').replace("reference=HERE.parents[2]/'phase4_pilot_v1/train_reference_v1'", "reference=HERE.parents[1]/'phase4_pilot_v1/train_reference_v1'")
    report.write_text(text,encoding='utf-8')
    with (HERE/'launch_readiness_specification_20261005.json').open('x',encoding='utf-8') as f:
        json.dump(dict(version='student_full_match_v1',matches=1,opponent_id='human_b',human_side='Orange',student_side='Blue',
            native_match_minutes=5,end_condition='actual native MatchPhase.Ended',overtime='Unlimited',
            original_teacher='shadow-only',student_checkpoint_sha256='579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7',
            state_setting=False,intentional_receive_pauses=False,replay_manipulation=False,
            teacher_fallback=False,training=False,aggregation=False,additional_runs=False),f,indent=2)

if __name__=='__main__':main()
