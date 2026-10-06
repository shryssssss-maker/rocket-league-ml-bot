"""--check is non-live. --run is blocked until separate recorded launch approval."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib
from common import *
from diagnostics import SummaryReader

PILOT=HERE.parent
REFERENCE=PILOT/'train_reference_v1'

def reference_check():
    if not (REFERENCE/'integrity.json').exists():return dict(status='pending_train_reference_build',verified=False)
    integrity=read(REFERENCE/'integrity.json')
    if sha(REFERENCE/'statistics.json')!=integrity['statistics_sha256']:raise RuntimeError('Reference statistics changed')
    if sha(HERE/'reference_statistics.py')!=integrity['builder_sha256']:raise RuntimeError('Reference builder changed')
    if integrity['source_manifest_sha256']!=MANIFEST_PIN:raise RuntimeError('Reference dataset mismatch')
    # Recheck byte hashes only; never import the reference reader into live path.
    for path,pin in integrity['input_hashes'].items():
        if sha(path)!=pin:raise RuntimeError('Required training input changed: '+path)
    stats=read(REFERENCE/'statistics.json')
    if stats['validation_read'] or stats['test_read'] or stats['model_loaded'] or stats['fitting']:raise RuntimeError('Reference scope violation')
    if stats['primary']['matches']!=['pilot_01','v10_006'] or stats['secondary']['callbacks']!=107400:
        raise RuntimeError('Reference assignments/count mismatch')
    return dict(status='verified_train_only_description',verified=True,
        sha256=integrity['statistics_sha256'],primary_callbacks=stats['primary']['callbacks'],
        all_train_callbacks=stats['secondary']['callbacks'],sample_use='description_only')

def prepare(folder):
    folder.mkdir(parents=True,exist_ok=False)
    command=subprocess.list2cmdline([str(LIVE_PYTHON),'-u','-B',str(HERE/'live_agent.py')])
    (folder/'run_agent.cmd').write_text('@echo off\n'+command+'\n',encoding='utf-8')
    (folder/'bot.toml').write_text('[settings]\nname = "Phase 4 13D Student Pilot"\nagent_id = "local/v13_phase4_student"\nrun_command = "run_agent.cmd"\nroot_dir = '+json.dumps(str(folder))+'\nloadout_file = '+json.dumps(str(ROOT/'python-example-original/src/loadout.toml'))+'\n',encoding='utf-8')
    text=(ROOT/'python-example-original/rlbot.toml').read_text(encoding='utf-8')
    for old,new in [('launcher = "Steam"','launcher = "Epic"'),('config_file = "src/bot.toml"','config_file = "bot.toml"'),('enable_state_setting = true','enable_state_setting = false')]:
        if text.count(old)!=1:raise RuntimeError('Unexpected original config')
        text=text.replace(old,new)
    (folder/'match.toml').write_text(text,encoding='utf-8')
    c=tomllib.loads(text)
    assert len(c['cars'])==2 and c['cars'][0]['type']=='Human' and c['cars'][0]['team']=='Blue' and c['cars'][1]['team']=='Orange'
    assert not c['match']['skip_replays'] and not c['match']['start_without_countdown'] and not c['match']['enable_state_setting']
    from rlbot.config import load_player_config
    load_player_config(folder/'bot.toml',team=1)
    return folder/'match.toml'

def preflight(folder):
    import ast
    import live_agent
    from reference import NonGameSink
    from student_client import StudentClient
    from rlbot import flat
    previous=V13/'phase3_shadow_v2'
    for name in ('reference.py','student_worker.py','student_client.py','diagnostics.py'):
        assert sha(HERE/name)==sha(previous/name),name+' changed'
    sink=NonGameSink()
    try:sink.send_msg(flat.PlayerInput(0,flat.ControllerState()))
    except RuntimeError:pass
    else:raise AssertionError('Shadow controls permitted')
    zero=[0.,0.,0.,0.,0.,False,False,False]
    live_agent.check_student_submission(flat.PlayerInput(0,flat.ControllerState(*zero)),0,zero)
    for message,expected in ((flat.PlayerInput(1,flat.ControllerState(*zero)),zero),
        (flat.PlayerInput(0,flat.ControllerState(throttle=1.)),zero),
        (flat.PlayerInput(0,flat.ControllerState(*zero)),None)):
        try:live_agent.check_student_submission(message,0,expected)
        except RuntimeError:pass
        else:raise AssertionError('Unexpected student transport permitted')
    tree=ast.parse((HERE/'live_agent.py').read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    assert cls.bases[0].id=='Bot'
    get_output=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_output')
    returns=[n for n in ast.walk(get_output) if isinstance(n,ast.Return)]
    assert len(returns)==1 and isinstance(returns[0].value,ast.Name) and returns[0].value.id=='output'
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('set_game_state','sleep') for n in ast.walk(tree))
    assert 'fill_desired_game_state' not in (HERE/'live_agent.py').read_text()
    forbidden=('reference_statistics','v11','bakeoff','bootstrap','rocketsim')
    imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any(any(word in name.lower() for word in forbidden) for name in imports)
    import pilot_report
    assert pilot_report.describe([])['median'] is None
    assert pilot_report.quantile([0.,1.,2.],.5)==1.
    assert pilot_report.js([1,0],[1,0])==0.
    assert pilot_report.js([1,0],[0,1])==1.
    assert pilot_report.js([0,0],[1,0]) is None
    assert not pilot_report.eligible([0.]*13,0)
    assert pilot_report.eligible([0.]*13,7)
    assert not pilot_report.eligible([0.]*13,12)
    worker=StudentClient(folder)
    try:
        prior=None; outputs=[]
        for i in range(1,5):
            result=worker.infer(i,[0.]*13)
            assert result['hidden_before_sha256']==prior and result['hidden_resets']==1
            prior=result['hidden_after_sha256']
            a=result['controls'];native=flat.ControllerState(*a)
            assert controls(native)==a and a[3:5]==[0.,0.] and a[6:]==[False,False]
            outputs.append(a)
        ready=worker.ready
    finally:worker.close()
    # Reuse non-live summary regression checks without constructing game inputs.
    from diagnostics import SummaryPublisher
    from unittest.mock import patch
    publisher=SummaryPublisher(folder/'summary_unit')
    (folder/'summary_unit').mkdir()
    first=publisher.publish(dict(status='running'))
    with first.open('rb'):
        assert publisher.publish(dict(status='running')) is not None
    with patch('diagnostics.os.rename',side_effect=PermissionError(13,'Injected denial')):
        assert publisher.publish(dict(status='running')) is None
    return dict(status='code_preflight_passed',worker=ready,shadow_transport_rejected=True,
        byte_identical_shadow_worker_decoder_client_and_publication=True,
        student_is_only_returned_controller=True,causal_hidden_chain=True,native_float_and_button_conversion_exact=True,
        unexpected_controls_wrong_agent_and_unset_expected_rejected=True,
        no_state_setting_or_intentional_pauses=True,no_dataset_or_training_reader_in_live_path=True,
        immutable_summary_and_injected_denial_checks=True,game_packet_or_prediction_constructed=False,
        actual_live_callbacks=0,preflight_outputs_diagnostic_only=True)

def readiness(folder,checks,reference,protected_ok,immutable_ok):
    complete=checks['status']=='code_preflight_passed' and reference['verified'] and protected_ok and immutable_ok
    report=dict(version='phase4_pilot_runner_v1',status='ready_for_explicit_launch_review' if complete else 'code_ready_reference_check_pending',
        nonlive_preflight_complete=complete,live_launched=False,checks=checks,reference=reference,
        protected_hashes_unchanged=43 if protected_ok else False,completed_artifacts_unchanged=immutable_ok,
        config_folder=str(folder),approved_protocol='Exactly one 90-second natural Blue-human/Orange-student run, no state-setting/pauses/replay manipulation',
        live_launch_approval_record_present=(PILOT/'live_launch_approval.json').exists(),
        launch_blocked_without_separate_approval=True,
        runner_code_hashes={p.name:sha(p) for p in sorted(HERE.glob('*.py'))},
        approved_spec_sha256=sha(PILOT/'launch_readiness_specification_20261005.json'),
        model_sha256=CHECKPOINT_PIN,input_features=read(V13/'projection_13d_v1/manifest.json')['result_feature_list'],
        inspection_order_note='One pilot_01 raw header was inspected before hashing during schema discovery. Its pinned hash was immediately verified; the production reference builder hashes all required training inputs before semantic reads.',
        training_or_aggregation=False,validation_or_test_samples_read=False,
        frozen_inputs_changed=False,
        reference_build_command="& '.\\training\\venv\\Scripts\\python.exe' -u -B '.\\training\\behavior_cloning\\v13_dagger\\phase4_pilot_v1\\runner\\reference_statistics.py' --build",
        final_preflight_command="& '.\\python-example\\venv\\Scripts\\python.exe' -u -B '.\\training\\behavior_cloning\\v13_dagger\\phase4_pilot_v1\\runner\\launch.py' --check")
    write(PILOT/'launch_readiness_20261005.json',report)
    (PILOT/'launch_readiness_20261005.md').write_text('\n'.join([
        '# Phase 4 launch readiness','',f"Status: **{report['status']}**. No live launch. Separate approval required.",'',
        '## Isolated runner and control isolation','',
        'StudentPilot inherits the RLBot transport base, not the original teacher. Its only returned ControllerState comes from the pinned 13D student. SDK PlayerInput is checked against that exact student output. OriginalShadow has no game socket and rejects control/state messages; the validated shadow, worker, decoder, client and immutable summary code are byte-identical to Phase 3.', '',
        'The temporary 1v1 configuration uses Epic, Blue human and Orange student. State-setting is disabled, native replay/countdown retained, and no probe planner or receive-loop sleeps exist. Exactly one launch reservation is allowed; additional attempts require review. A recorded separate launch approval is required before --run can connect to RLBot.', '',
        'Model/observation/prediction/reset contracts are unchanged. Both paths consume one actual callback stream, native RLBot prediction and canonical elapsed time. Shadow state never follows or resets to student actions. Only columns 0–12 reach the model; private labels are diagnostic. Empty-prediction errors remain unsuppressed in the original lookup branch.', '',
        '## Non-live checks','',f"`{json.dumps(checks,indent=2)}`",'',
        f"43 protected hashes unchanged: {protected_ok}; V10/V12/V13 completed artifacts and both prior Phase 3 versions unchanged: {immutable_ok}.",'',
        '## Train-only descriptive reference','',f"Status: `{reference['status']}`.",'',
        'The builder requires only four training matches: pilot_01, v10_001, v10_004, v10_006. It hashes eight required raw/observation files before any sample content is read; it never opens validation/test samples. It checks raw/float32 tensor parity and chronological submission links, then saves only phase/mask-conditioned quantiles/histograms/occupancy, never training tensors. Primary baseline is Orange-only pilot_01/v10_006; secondary is all four training matches. These descriptions cannot tune or change the model.', '',
        report['inspection_order_note'],'',
        '## Before launch review','',
        'If reference statistics are pending, run the following non-live commands. The first scans about 107,400 training rows and may take tens of seconds; progress lists hashes/matches. The second verifies reference integrity and repeats non-live runner checks. Neither launches Rocket League.', '',
        '```powershell',report['reference_build_command'],report['final_preflight_command'],'```','',
        'Paste their output. Full readiness is not claimed until both succeed. Do not run a live command yet.', '',
        '## Future pilot/report boundaries','',
        'After separate approval: one 90-second monotonic wall window starts at the first verified student submission, with at most 180 seconds startup. Natural match end/user stop/error can end earlier. No fallback controller or fabricated repair is returned on failure. Records remain quarantined diagnostics without a dataset split, DAgger aggregation, fitting or test evaluation.', '',
        'The report separately checks final student submission/clean shutdown, native transport, model continuity, actual learner-driven states, original shadow sequences after missed student jumps, potentially unsupported expert labels, train-reference occupancy, unobserved categories and queue/latency/callback limits. No competence, causal distribution-shift or DAgger-benefit claim follows from one partial match.', '',
        f"Prepared configs: `{folder}`. This preparation creates no live session."
    ])+'\n',encoding='utf-8')

def run(config,folder,opponent,protected,immutable):
    from rlbot import flat
    from rlbot.managers import MatchManager
    os.environ['V13_PHASE4_SESSION']=str(folder);os.environ['PYTHONDONTWRITEBYTECODE']='1'
    reader=SummaryReader();manager=MatchManager();status='interrupted';error=None
    try:
        print('ONE NATURAL 90-SECOND PILOT: Blue human plays normally; Orange STUDENT alone controls. No probes/pauses.',flush=True)
        manager.start_match(config,wait_for_start=False)
        startup=time.monotonic()+180;last_print=0
        while True:
            now=time.monotonic();summary=reader.read(folder)
            if manager.rlbot_server_process is not None and not manager.rlbot_server_process.is_running():status='server_exited';break
            if summary.get('status') not in (None,'running'):status=summary['status'];break
            first=summary.get('first_submission_monotonic')
            if first is not None and now-first>=90:status='bounded_pilot_complete';break
            if first is None and now>startup:status='startup_timeout';break
            packet=manager.packet
            if first is not None and packet is not None and packet.match_info.match_phase==flat.MatchPhase.Ended:status='natural_match_ended';break
            if now-last_print>=10:
                print('PHASE4',dict(callbacks=summary.get('callbacks',0),shadow_flips=summary.get('shadow_flips',0),
                    seconds_since_first_submission=None if first is None else round(now-first,2)),flush=True)
                last_print=now
            time.sleep(.05)  # Supervisor polling only; NEVER a receive-loop pause.
    except KeyboardInterrupt:status='stopped_by_user'
    except Exception as e:status='supervisor_error';error=repr(e)
    finally:
        cleanup=[]
        for action in (manager.stop_match,manager.disconnect):
            try:action()
            except Exception as e:cleanup.append(repr(e))
        end=time.monotonic()+5
        while not (folder/'agent_closed.json').exists() and time.monotonic()<end:time.sleep(.1)
        write(folder/'supervisor.json',dict(status=status,error=error,cleanup_errors=cleanup,
            protected_ok=verify()==protected,immutable_ok=immutable_snapshot()==immutable,
            agent_closed=(folder/'agent_closed.json').exists(),opponent_id=opponent))
        from pilot_report import review
        review(folder)
        print('PHASE4 STOPPED:',status,'Report:',folder/'contract_report.md',flush=True)
        print('Stop. No second match, aggregation or fitting.',flush=True)

def main():
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--check',action='store_true');g.add_argument('--run',action='store_true')
    parser.add_argument('--opponent-id',choices=['human_a','human_b','human_c']);args=parser.parse_args()
    if Path(sys.executable).resolve()!=LIVE_PYTHON.resolve():raise RuntimeError('Use existing RLBot Python')
    protected=verify();immutable=immutable_snapshot();reference=reference_check()
    if args.run:
        approval=PILOT/'live_launch_approval.json'
        if not approval.exists() or not read(approval).get('approved',False):raise RuntimeError('Separate live-launch approval not recorded; no game launched')
        if not args.opponent_id:raise RuntimeError('Actual opponent identity is required')
        if not reference['verified']:raise RuntimeError('Reference/preflight incomplete')
        ready=read(PILOT/'launch_readiness_20261005.json')
        if not ready['nonlive_preflight_complete']:raise RuntimeError('Full preflight has not passed')
        if ready['runner_code_hashes']!={p.name:sha(p) for p in sorted(HERE.glob('*.py'))}:
            raise RuntimeError('Runner changed since preflight; rerun non-live check')
        if ready['approved_spec_sha256']!=sha(PILOT/'launch_readiness_specification_20261005.json'):
            raise RuntimeError('Approved protocol changed')
        if ready['reference']!=reference:raise RuntimeError('Reference changed since preflight')
        if read(approval).get('readiness_sha256')!=sha(PILOT/'launch_readiness_20261005.json'):
            raise RuntimeError('Separate approval must bind exact readiness report')
        # Atomic reservation; prevents any second run without explicit review.
        with (PILOT/'pilot_launch_reservation.json').open('x',encoding='utf-8') as f:
            json.dump(dict(opponent=args.opponent_id,utc=datetime.now(timezone.utc).isoformat()),f)
    folder=PILOT/('prepared' if args.check else 'sessions')/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    config=prepare(folder)
    write(folder/'metadata.json',dict(version='phase4_student_pilot_v1',not_training_dataset=True,
        code_hashes={str(p):sha(p) for p in HERE.glob('*.py')},config_hashes={p.name:sha(p) for p in folder.iterdir() if p.is_file()},
        model_sha256=CHECKPOINT_PIN,source_manifest_sha256=MANIFEST_PIN,reference=reference,
        reference_description_only=True,opponent_id=args.opponent_id,student_only_controller=True,
        original_shadow_only=True,natural_run_seconds=90,state_setting=False,receive_pauses=False,
        versions={n:importlib.metadata.version(n) for n in ('rlbot','rlbot-flatbuffers','psutil')},
        phase4_spec_sha256=sha(PILOT/'launch_readiness_specification_20261005.json')))
    if args.check:
        checks=preflight(folder)
        unchanged=verify()==protected and immutable_snapshot()==immutable
        assert unchanged
        readiness(folder,checks,reference,True,unchanged)
        print('PHASE4 NON-LIVE CODE PREFLIGHT PASS; reference:',reference['status'],flush=True)
        print('Readiness report:',PILOT/'launch_readiness_20261005.md',flush=True)
    else:run(config,folder,args.opponent_id,protected,immutable)

if __name__=='__main__':main()
