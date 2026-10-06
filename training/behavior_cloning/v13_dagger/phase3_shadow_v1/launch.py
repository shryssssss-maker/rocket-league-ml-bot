"""--check is non-live preparation. --run requires a later human launch approval."""
import argparse
import ast
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import subprocess
import tomllib
from common import *

SCOPE = [
    'Only actual processed RLBot callbacks; not lossless delivery or equivalence on undelivered physics ticks.',
    'Receive pauses can cause queue-full warnings; IPC/profiling/logging can change delivered callback scheduling.',
    'Teacher drives; student outputs never reach the game. No learner-induced physical-state coverage is established.',
    'Shadow is an independent original-source actor; exact equivalence does not prove learned sequence competence.',
    'Requested probe states do not count as evidence; only actual later packets/sequence transitions do.',
    'Observation prediction probe is distinct from whether the original gameplay branch consulted prediction.',
    'Diagnostic callback logs are not a DAgger training dataset; no tensor export, aggregation, fitting or test evaluation.',
]

def prepare(folder,protected):
    folder.mkdir(parents=True,exist_ok=False)
    command=subprocess.list2cmdline([str(LIVE_PYTHON),'-u','-B',str(HERE/'live_agent.py')])
    (folder/'run_agent.cmd').write_text('@echo off\n'+command+'\n',encoding='utf-8')
    (folder/'bot.toml').write_text('[settings]\nname = "Phase 3 Original Teacher Shadow Diagnostic"\nagent_id = "local/v13_phase3_shadow"\nrun_command = "run_agent.cmd"\nroot_dir = '+json.dumps(str(folder))+'\nloadout_file = '+json.dumps(str(ROOT/'python-example-original/src/loadout.toml'))+'\n',encoding='utf-8')
    text=(ROOT/'python-example-original/rlbot.toml').read_text(encoding='utf-8')
    for old,new in [('launcher = "Steam"','launcher = "Epic"'),('config_file = "src/bot.toml"','config_file = "bot.toml"'),('enable_rendering = false','enable_rendering = true')]:
        if text.count(old)!=1: raise RuntimeError('Unexpected original launch config')
        text=text.replace(old,new)
    (folder/'match.toml').write_text(text,encoding='utf-8')
    config=tomllib.loads(text)
    assert len(config['cars'])==2 and config['cars'][0]['type']=='Human' and config['cars'][0]['team']=='Blue' and config['cars'][1]['team']=='Orange'
    assert not config['match']['skip_replays'] and not config['match']['start_without_countdown']
    assert config['match']['enable_state_setting']
    from rlbot.config import load_player_config
    load_player_config(folder/'bot.toml',team=1)
    write(folder/'metadata.json',dict(version='v13_phase3_shadow_v1',session=folder.name,
        prepared_utc=datetime.now(timezone.utc).isoformat(),controller='Unmodified original MyBot only',
        student='Pinned accepted 13D bootstrap; diagnostic CPU worker only',
        shadow='Unmodified original MyBot methods; independent sequence/tracker; non-game cosmetic sink',
        protected_hashes=protected,checkpoint_sha256=CHECKPOINT_PIN,source_manifest_sha256=MANIFEST_PIN,
        source_revision='fd061f457bf19175b4a9b3b3d7811a987044c64d',
        projection=read(V13/'projection_13d_v1/manifest.json'),observation_code_sha256=OBS_PIN,
        code_hashes={str(p):sha(p) for p in sorted(HERE.glob('*.py'))},
        config_hashes={p.name:sha(p) for p in folder.iterdir() if p.is_file()},
        versions={n:importlib.metadata.version(n) for n in ('rlbot','rlbot-flatbuffers','psutil')},
        packet_processing='RLBot SDK native callbacks; prediction receipts may be reused between callbacks',
        python=sys.version,analog_tolerance=ANALOG_TOL,
        intervention_limits=dict(state_requests=24,receive_pauses=6,live_seconds=300,startup_seconds=180,
            ordinary_pause_seconds=1.25,countdown_pause_seconds=5),
        controls_to_game='Original teacher ControllerState through RLBot SDK PlayerInput only',
        probe_preparation_approved=True,live_launch_approval_pending=True,scope=SCOPE))
    return folder/'match.toml'

def nonlive_preflight(folder):
    import reference
    import live_agent
    from student_client import StudentClient
    from rlbot import flat
    sink=reference.NonGameSink()
    sink.send_msg(flat.MatchComm(0,1,False,'test',b''))
    try: sink.send_msg(flat.PlayerInput(0,flat.ControllerState()))
    except RuntimeError: pass
    else: raise AssertionError('Shadow permits game controls')
    assert sink.forbidden==1
    # No GamePacket, BallPrediction or fabricated trajectories are constructed.
    assert reference.original.MyBot.get_output.__module__=='_phase3_original'
    # Construct an unconnected source actor using SDK default field metadata;
    # do not fabricate or process any game packet/prediction.
    owner=object.__new__(reference.original.MyBot)
    owner.index=0;owner.team=1;owner.active_sequence=None
    shadow=reference.OriginalShadow(owner)
    assert shadow.actor is not owner and shadow.actor.boost_pad_tracker is not owner.boost_pad_tracker
    assert not hasattr(shadow.sink,'connect') and shadow.active_sequence is None
    tree=ast.parse((HERE/'student_worker.py').read_text(encoding='utf-8'))
    imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
    assert not any('rlbot' in n.lower() or 'rocketsim' in n.lower() for n in imports)
    assert not any('rocketsim' in p.read_text(encoding='utf-8').lower() for p in HERE.glob('*.py') if p.name not in ('launch.py',))
    client=StudentClient(folder)
    try:
        results=[]
        for i in range(1,5):
            # Numerical protocol/hidden-continuity smoke test, never live evidence.
            results.append(client.infer(i,[0.]*13))
        for i,r in enumerate(results):
            assert len(r['controls'])==8 and len(r['logits'])==4 and r['hidden_resets']==1
            assert r['controls'][3:5]==[0.,0.] and r['controls'][6:]==[False,False]
            assert all(type(v) is bool for v in r['controls'][5:])
            assert r['hidden_before_sha256']==(None if i==0 else results[i-1]['hidden_after_sha256'])
        return dict(status='passed_nonlive',worker=client.ready,numerical_callbacks=4,
            worker_hidden_hash_chain=True,shadow_controller_guard=True,
            independent_original_actor_and_tracker=True,
            no_game_packet_or_prediction_constructed=True,live_categories_observed=0)
    finally: client.close()

def readiness(folder,preflight,protected,immutable_ok):
    report=dict(status='prepared_awaiting_launch_review',phase3_passed=False,live_launched=False,
        preflight=preflight,protected_hashes_unchanged=len(protected),completed_artifacts_unchanged=immutable_ok,
        prepared_config_folder=str(folder),scope=SCOPE,
        launch_command="& '.\\python-example\\venv\\Scripts\\python.exe' -u -B '.\\training\\behavior_cloning\\v13_dagger\\phase3_shadow_v1\\launch.py' --run",
        required_live_evidence=['Neutral->Chase','Chase->Jump','Release','Front dodge','Coast/completion',
            'At least two starts/two completions','Missing ball with pending sequence',
            'Same pending sequence across goal/replay/kickoff','Actual long gap >=1 second',
            'Near/far branches','Actual selector index/timestamp/position','Exact native eight-channel submission',
            'Student misses a teacher sequence start','Teacher dodge after student prior jump=False',
            'Independent shadow continuation after student divergence'])
    write(HERE/'launch_readiness_20261005.json',report)
    (HERE/'launch_readiness_20261005.md').write_text('\n'.join([
        '# Phase 3 launch readiness','', '**Prepared only. No live launch. Phase 3 has not passed.**','',
        f"Non-live checks: {preflight['status']}. All {len(protected)} protected hashes unchanged; completed V10/V12/V13 artifacts unchanged: {immutable_ok}.",'',
        '## Control and memory contract','',
        'The live wrapper returns only original MyBot.get_output. The independent shadow invokes the same original methods on the same native packet/prediction/time, with its own sequence and boost tracker. It has no socket; its sink discards cosmetic render/quickchat and rejects controls/state requests.', '',
        'The existing training Python hosts the accepted checkpoint on CPU. Its only input is callback index plus 13 float32 features. No private teacher state/history is sent. GRU memory persists across all phases and missing-ball callbacks, and resets only at session/match initialization.', '',
        'The unchanged frozen observation builder supplies columns 0–12. The observation selector probe follows the frozen recorder’s nonempty-prediction guard and original helper; the teacher’s far-ball branch retains its original empty-prediction IndexError. No interpolation, shifting, clamping, timestamp correction or simulator is used.', '',
        '## Bounded real-game probes','',
        'Orange original teacher, blue human. Keep hands off during probes: both cars and the ball can move following state-setting requests. At most 24 requests, 6 explicitly logged pauses, 300 seconds after the first callback plus 180 seconds startup. Two phase-specific pauses use 1.25 seconds; pending countdown probes use 5 seconds. Pauses occur only after teacher controls are submitted. Requests alone never count as coverage.', '',
        'Completion requires every explicit callback evidence category and two sequence completions, including an actual student disagreement at a teacher sequence start and subsequent independent shadow continuation. If such cases do not occur, the run remains incomplete; no forced student output or fabricated evidence is allowed.', '',
        '## Required evidence','', *['- '+s for s in report['required_live_evidence']], '',
        '## Limits','', *['- '+s for s in SCOPE], '',
        '## Launch after review','', '```powershell',report['launch_command'],'```','',
        'Expected: session path, original-teacher-only notice, periodic callbacks/flip/missing-category counts, bounded stop status and diagnostic contract-report paths. Ctrl+C produces an incomplete report. No DAgger dataset, fitting, test-set evaluation or student control follows.', '',
        f"Prepared configs: `{folder}`. Every later launch creates a new session; existing outputs are never overwritten."
    ])+'\n',encoding='utf-8')

def result_report(folder,status,protected_ok,immutable_ok,error=None):
    preserve=load('_phase3_summary_reader',ROOT/'training/live_reference_test/launch.py')
    summary=preserve.SummaryReader().read(folder/'agent_summary.json')
    rows=[];events=[]
    for filename,destination in (('callbacks.jsonl',rows),('events.jsonl',events)):
        p=folder/filename
        if p.exists():
            for line in p.read_text(encoding='utf-8').splitlines():
                try: destination.append(json.loads(line))
                except ValueError: destination.append(dict(error='Partial/malformed diagnostic row'))
    failures=[r for r in rows if r.get('error') or r.get('failures')]
    submissions=[e['payload'] for e in events if e.get('kind')=='original_controls_submitted']
    by_callback={s['callback']:s for s in submissions}
    transport_failures=[]
    hidden_failures=[];previous=None
    for r in rows:
        if 'original_controls' not in r: continue
        s=by_callback.get(r['callback'])
        if s is None or s['controls'][5:]!=r['original_controls'][5:] or max(abs(a-b) for a,b in zip(s['controls'][:5],r['original_controls'][:5]))>ANALOG_TOL:
            transport_failures.append(r['callback'])
        student=r.get('student')
        if student is None or student['hidden_before_sha256']!=previous or student['hidden_resets']!=1:
            hidden_failures.append(r['callback'])
        previous=None if student is None else student['hidden_after_sha256']
    coverage=summary.get('coverage',{})
    transition_ids=set.intersection(*[{e['sequence_id'] for e in coverage.get(k,[]) if e.get('sequence_id') is not None}
        for k in ('goal_pending','replay_pending','kickoff_pending')])
    all_observed=bool(coverage) and all(coverage.values()) and summary.get('original_flips',0)>=2 and len(coverage.get('completion',[]))>=2 and bool(transition_ids)
    report=dict(status=status,phase3_pass_declared=False,session=str(folder),callbacks=len(rows),
        summary=summary,coverage_complete=all_observed,transition_sequence_ids=sorted(transition_ids),
        comparison_error_rows=failures,transport_failure_callbacks=transport_failures,
        hidden_continuity_failure_callbacks=hidden_failures,supervisor_error=error,
        original_controls_only=True,student_controls_deployed=False,dagger_collected=False,
        test_evaluated=False,protected_hashes_unchanged=43 if protected_ok else False,
        immutable_artifacts_unchanged=immutable_ok,scope=SCOPE,
        integrity={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
    write(folder/'contract_report.json',report)
    (folder/'contract_report.md').write_text('\n'.join(['# Phase 3 shadow-teacher contract', '',
        f"Status: **{status}**. Review required; no automatic pass.", '',
        f"Callbacks: {len(rows)}; all categories observed: {all_observed}; comparison/error rows: {len(failures)}; transport discrepancies: {len(transport_failures)}; hidden-chain discrepancies: {len(hidden_failures)}.",'',
        '## Actual evidence', '', *[f"- {k}: {len(v)} retained callback examples" for k,v in coverage.items()], '',
        f"Same sequence across goal/replay/kickoff: {sorted(transition_ids)}.",'',
        f"43 protected hashes unchanged: {protected_ok}; completed immutable artifacts unchanged: {immutable_ok}.",'',
        '## Scope', '', *['- '+s for s in SCOPE], '',
        'Full native controls, source/shadow temporal states, live selector traces, student diagnostics, packet/prediction receipts and intervention events are retained in session JSONL. See JSON report for every discrepancy. Stop for review; no student-controlled gameplay or DAgger collection follows.'
    ])+'\n',encoding='utf-8')
    return report

def run(config,folder,before,immutable):
    from rlbot import flat
    from rlbot.managers import MatchManager
    preserve=load('_phase3_live_summary',ROOT/'training/live_reference_test/launch.py')
    reader=preserve.SummaryReader()
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['V13_PHASE3_SESSION']=str(folder)
    manager=MatchManager(); status='interrupted_incomplete';error=None
    try:
        print('ORIGINAL TEACHER ONLY controls Orange. Student and independent original shadow log only.',flush=True)
        print('Bounded probes reset both cars/ball. Keep Blue human hands off during probes. Logged pauses may cause queue-full warnings.',flush=True)
        manager.start_match(config,wait_for_start=False)
        startup=time.monotonic()+180;deadline=None;last_print=0
        while True:
            summary=reader.read(folder/'agent_summary.json');now=time.monotonic()
            server=manager.rlbot_server_process
            if server is not None and not server.is_running():
                status='server_exited_incomplete';break
            if summary.get('callbacks',0)>0 and deadline is None:deadline=now+300
            if summary.get('status') not in (None,'running'):
                status=summary['status'];break
            packet=manager.packet
            if deadline is not None and packet is not None and packet.match_info.match_phase==flat.MatchPhase.Ended:
                status='natural_match_ended_awaiting_review';break
            if (deadline is None and now>startup) or (deadline is not None and now>deadline):
                status='bounded_timeout_incomplete';break
            if now-last_print>=10:
                print('PHASE3',dict(callbacks=summary.get('callbacks',0),flips=summary.get('original_flips',0),
                    failures=len(summary.get('failures',[])),missing=[k for k,v in summary.get('coverage',{}).items() if not v]),flush=True)
                last_print=now
            time.sleep(.25)
    except KeyboardInterrupt: status='stopped_by_user_incomplete'
    except Exception as e: status='supervisor_error_incomplete';error=repr(e)
    finally:
        cleanup=[]
        for action in (manager.stop_match,manager.disconnect):
            try:action()
            except Exception as e:cleanup.append(repr(e))
        limit=time.monotonic()+5
        while not (folder/'agent_closed.json').exists() and time.monotonic()<limit:time.sleep(.1)
        closed=(folder/'agent_closed.json').exists()
        if not closed:status+='__agent_shutdown_unconfirmed'
        protected_ok=verify()==before
        immutable_ok=immutable_snapshot()==immutable
        write(folder/'supervisor.json',dict(status=status,error=error,cleanup_errors=cleanup,agent_closed=closed))
        result_report(folder,status,protected_ok,immutable_ok,error)
        print('PHASE3 STOPPED:',status,'Report:',folder/'contract_report.md',flush=True)
        print('Stop for review. No DAgger collection or student game controls.',flush=True)

def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if Path(sys.executable).resolve()!=LIVE_PYTHON.resolve():raise RuntimeError('Use existing live venv')
    before=verify();immutable=immutable_snapshot()
    prefix='prepared' if args.check else 'sessions'
    folder=HERE/prefix/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    config=prepare(folder,before)
    print('Session/config:',folder,flush=True)
    if args.check:
        preflight=nonlive_preflight(folder)
        assert verify()==before
        immutable_ok=immutable_snapshot()==immutable
        assert immutable_ok
        readiness(folder,preflight,before,immutable_ok)
        print('PHASE3 PREFLIGHT PASS: no live connection/launch; 43 protected hashes unchanged.',flush=True)
        print('Readiness report:',HERE/'launch_readiness_20261005.md',flush=True)
    else: run(config,folder,before,immutable)

if __name__=='__main__':main()
