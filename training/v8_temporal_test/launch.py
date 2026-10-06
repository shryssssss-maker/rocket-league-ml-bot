"""User-run real RLBot V8. --check never connects or launches the game."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib
from datetime import datetime

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'training/live_reference_test'))
# Explicit module loading avoids importing this launch.py under the same name.
import importlib.util
spec=importlib.util.spec_from_file_location('_v8_preservation',ROOT/'training/live_reference_test/launch.py')
preserve=importlib.util.module_from_spec(spec)
spec.loader.exec_module(preserve)
verify_sources,sha,INTERPRETER=preserve.verify_sources,preserve.sha,preserve.INTERPRETER
sys.path.insert(0,str(HERE))
from candidate import Sequence,controls,mode,state


def prepare(folder):
    folder.mkdir(parents=True,exist_ok=False)
    command=subprocess.list2cmdline([str(INTERPRETER),'-u','-B',str(HERE/'live_agent.py')])
    (folder/'run_agent.cmd').write_text('@echo off\n'+command+'\n',encoding='utf-8')
    (folder/'bot.toml').write_text('[settings]\nname = "V8 Original Teacher Shadow Check"\nagent_id = "local/v8_shadow"\nrun_command = "run_agent.cmd"\nroot_dir = '+json.dumps(str(folder))+'\nloadout_file = '+json.dumps(str(ROOT/'python-example-original/src/loadout.toml'))+'\n',encoding='utf-8')
    text=(ROOT/'python-example-original/rlbot.toml').read_text(encoding='utf-8')
    for old,new in [('launcher = "Steam"','launcher = "Epic"'),('config_file = "src/bot.toml"','config_file = "bot.toml"'),('enable_rendering = false','enable_rendering = true')]:
        if text.count(old)!=1:raise RuntimeError('Unexpected original config: '+old)
        text=text.replace(old,new)
    (folder/'match.toml').write_text(text,encoding='utf-8')
    parsed=tomllib.loads(text)
    if len(parsed['cars'])!=2 or parsed['cars'][0]['type']!='Human' or parsed['cars'][0]['team']!='Blue' or parsed['cars'][1]['team']!='Orange':
        raise RuntimeError('Expected real blue-human versus orange-source 1v1 roster')
    if parsed['match']['skip_replays'] or parsed['match']['start_without_countdown']:
        raise RuntimeError('Transition probes require original replays/countdown retained')
    from rlbot.config import load_player_config
    load_player_config(folder/'bot.toml',team=1)
    metadata=dict(session=folder.name,source_revision=preserve.REVISION,protected_hashes=verify_sources(),
        code_sha256={p.name:sha(p) for p in (HERE/'candidate.py',HERE/'live_agent.py',HERE/'launch.py')},
        configs={n:sha(folder/n) for n in ('bot.toml','match.toml','run_agent.cmd')},
        versions={n:importlib.metadata.version(n) for n in ('rlbot','rlbot-flatbuffers','psutil')},python=sys.version,
        controlled_probes_authorized=True,interventions=['Actual game state-setting requests','Two 1.25-second receive-loop pauses after source controls sent','One five-second pending-sequence pause over real countdown per goal-transition attempt'],
        analog_tolerance=1e-6,sequence_selection_buttons='exact',candidate_deployed=False,
        teacher='Original source methods executed unmodified; passive Python profiling observes selection and phase calls',
        matching='Both paths receive the same native callback packet and current delivered BallPrediction',
        scope='V8 only; no forecasts, RocketSim, production adapter, demonstrations, BC or PPO')
    (folder/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return folder/'match.toml'


def make_report(folder,status,protected_ok,error=None):
    summary_path=None if folder is None else folder/'agent_summary.json'
    summary=json.loads(summary_path.read_text(encoding='utf-8')) if summary_path is not None and summary_path.exists() else {}
    coverage=summary.get('coverage',{})
    definitions={
        '1 Neutral -> Chase':['neutral_to_chase'],
        '2 Chase -> Jump':['chase_to_jump'],
        '3 Jump release':['release'],
        '4 Front dodge':['front_dodge'],
        '5 Coast and completion':['coast','completion'],
        '6 Multiple sequences':['multiple_sequences'],
        '7 Missing ball while sequence pending':['missing_ball_pending'],
        '8 Goal/replay/kickoff while sequence pending':['goal_pending','replay_pending','kickoff_pending'],
        '9 Long callback gaps while pending':['long_gap'],
        '10 Near and far branches':['near_ball','far_ball'],
        '11 Actual live selector':['live_selector'],
        '12 Native eight-channel controllers':['native_controls']}
    rows=[]
    if folder is not None and (folder/'callbacks.jsonl').exists():
        rows=[json.loads(line) for line in (folder/'callbacks.jsonl').open(encoding='utf-8')]
    exact_rows=[r for r in rows if 'original_controls' in r and not r['failures']]
    special=dict(multiple_sequences=summary.get('original_flips',0)>=2 and len(coverage.get('completion',[]))>=2,
        native_controls=bool(rows) and len(exact_rows)==len(rows))
    requirements={name:dict(evidence={k:coverage.get(k,special.get(k,False)) for k in keys},
        observed=all(bool(coverage.get(k,special.get(k,False))) for k in keys)) for name,keys in definitions.items()}
    transition_ids=set.intersection(*[{e.get('sequence_id') for e in coverage.get(k,[]) if e.get('sequence_id') is not None}
        for k in ('goal_pending','replay_pending','kickoff_pending')])
    requirements['8 Goal/replay/kickoff while sequence pending']['same_sequence_ids']=sorted(transition_ids)
    requirements['8 Goal/replay/kickoff while sequence pending']['observed']=bool(transition_ids)
    failures=[r for r in rows if r.get('failures') or r.get('error')]
    metadata={} if folder is None else json.loads((folder/'metadata.json').read_text())
    report=dict(experiment='V8_FULL_TEMPORAL_LIVE',status=status,session=None if folder is None else str(folder),
        live_callbacks=len(rows),requirements=requirements,all_required_categories_observed=all(r['observed'] for r in requirements.values()),
        comparison_failures=len(failures),failure_rows=failures,
        original_flips=summary.get('original_flips',0),candidate_flips=summary.get('candidate_flips',0),
        maximum_analog_error=summary.get('maximum_analog_error'),analog_tolerance=1e-6,
        auxiliary_observation_errors=[dict(callback=r['callback'],error=r['auxiliary_observation_error']) for r in rows if 'auxiliary_observation_error' in r],
        original_controls_only=True,candidate_controls_deployed=False,protected_hashes_unchanged=43 if protected_ok else False,
        V8_pass_declared=False,supervisor_error=error,agent_summary=summary,metadata=metadata,
        notes=['No required category can be inferred from probe requests; actual live callback examples are mandatory.',
            'Passive profiling and diagnostic logging add overhead; pauses are explicitly labeled interventions.',
            'Full callback rows include actual physics, actual selection, both sequence states, controls, branches and transitions.',
            'Original raw packet/prediction wire messages are not modified or replaced; full prediction trajectories are not dumped every callback.',
            'Auxiliary observations are labeled separately; unavailable/invalid auxiliary features do not suppress teacher selector exceptions.',
            'Only live evidence can support V8; preflight results are not coverage.'])
    path=ROOT/'training/reports/v8_temporal_teacher_equivalence_20261004.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# V8 temporal teacher equivalence','',f'**Status: {status}. V8 is not declared passed.**','',
        f"Live callbacks: {len(rows)}. Comparison failures/errors: {len(failures)}. Original/candidate sequence starts: {report['original_flips']}/{report['candidate_flips']}.",
        f"Maximum analog difference: {report['maximum_analog_error']}; established tolerance: 1e-6. Sequence phase/index, done/start times, branch, selected index/timestamp/position and buttons require exact equality.",'',
        '| Required category | Actual live evidence present |','|---|---|']
    lines += [f"| {name} | {'Yes' if value['observed'] else 'No — incomplete'} |" for name,value in requirements.items()]
    lines+=['',f'All categories observed: {report["all_required_categories_observed"]}. Protected hashes unchanged: {report["protected_hashes_unchanged"]}.','',
        '## Method','',
        'The source MyBot initialization/get_output and original Sequence/helper functions execute unmodified on actual RLBot native callbacks and the current delivered BallPrediction. A passive sys.setprofile observer records source helper return locals and consumed phase; no source function/global is replaced. The independent candidate sees the same native objects sequentially and keeps its own temporal state. Only source ControllerState is returned to RLBot.','',
        'Both sequence snapshots are logged before/after every comparison callback, with emitted output modes and consumed phases kept distinct. The finishing callback returns old-phase controls while index/done may already advance. Missing-ball callbacks preserve both pending states. Previous-action observation provenance is the actual prior source control submission, not candidate output.','',
        'Approved controlled interventions request near/far positions, speed-triggered flips and a real goal while a sequence is pending. Actual phase/ball callbacks determine coverage, never requested state. Two labeled 1.25-second pauses and a five-second pending-sequence pause over real countdown occur after source controls are sent; no intermediate ticks/packets are invented. Replay and countdown are retained.','',
        'Long gap evidence requires at least one second of actual canonical callback delta with an already-pending sequence. Multiple-sequence coverage requires two observed starts and two completions. Goal, replay and kickoff each need separate pending-sequence evidence linked to the same observed sequence identity.','',
        '## Evidence and limitations','',
        f'Session: {report["session"]}. Full per-callback evidence: callbacks.jsonl; interventions: events.jsonl; metadata/config hashes: metadata.json. Required-category examples and every failure are included in the JSON report.','',
        'Coverage can remain incomplete if real transitions do not overlap the sequence as requested. Do not fabricate missing-ball/phase states, search for synthetic substitutions, or infer success from requests. Mismatch/error, natural match end, user interruption or the bounded supervisor timeout stops the test. No candidate deployment or ML follows.','',
        '## Exact implementation','',
        '- training/v8_temporal_test/candidate.py: independent Candidate/Sequence.',
        '- training/v8_temporal_test/live_agent.py: LiveHarness.observe_original/get_output/cover/plan.',
        '- Unmodified source: python-example-original/src/bot.py and src/util/sequence.py, ball_prediction_analysis.py.',
        '- Auxiliary declared observation: training/v2_observation_test/observation_contract.py:live_adapter.',
        '- Both protected bot directories remain untouched.']
    if error:lines+=['',f'Supervisor error: {error}']
    path.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if Path(sys.executable).resolve()!=INTERPRETER.resolve():raise RuntimeError('Use existing live venv interpreter')
    before=verify_sources()
    folder=HERE/'sessions'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    config=prepare(folder)
    print('Session:',folder,flush=True)
    if args.check:
        import live_agent
        seq=Sequence()
        out,phase=seq.tick(0.)
        assert phase==0 and mode(controls(out))=='Jump' and state(seq)['index']==0
        seq.tick(.05)
        assert seq.index==0
        out,phase=seq.tick(.06)
        assert phase==0 and seq.index==1 and seq.steps[1].start_time is None
        assert verify_sources()==before
        report=make_report(folder,'PREPARED_NOT_LIVE_VALIDATED',True)
        print('PREFLIGHT PASS: imports, native controls, strict step mechanics, 1v1 config, replay/countdown retained, 43 hashes unchanged. No game launched.',flush=True)
        print('All V8 live coverage remains unobserved. Reports:',ROOT/'training/reports/v8_temporal_teacher_equivalence_20261004.md',flush=True)
        return
    from rlbot import flat
    from rlbot.managers import MatchManager
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['V8_SESSION']=str(folder)
    manager=MatchManager()
    status='interrupted'; error=None
    reader=preserve.SummaryReader()
    try:
        print('V8 LIVE SHADOW ONLY: orange original teacher drives; candidate never drives.',flush=True)
        print('Do not control the blue car during probes. Both cars/ball may be reset. Replays/countdown remain; 1.25s and 5s diagnostic pauses are intentional.',flush=True)
        print('Allow up to five minutes of live testing plus game startup. Ctrl+C yields an incomplete report.',flush=True)
        manager.start_match(config,wait_for_start=False)
        startup_deadline=time.monotonic()+180
        live_deadline=None; last_print=0
        while True:
            summary=reader.read(folder/'agent_summary.json')
            now=time.monotonic()
            if summary.get('callbacks',0)>0 and live_deadline is None:live_deadline=now+300
            if summary.get('status') in ('error','comparison_failure','coverage_complete_awaiting_review'):
                status=summary['status']; break
            packet=manager.packet
            if live_deadline is not None and packet is not None and packet.match_info.match_phase==flat.MatchPhase.Ended:
                status='natural_match_ended'; break
            if (live_deadline is None and now>startup_deadline) or (live_deadline is not None and now>live_deadline):
                status='bounded_timeout_incomplete'; break
            if now-last_print>10:
                print('V8',dict(callbacks=summary.get('callbacks',0),flips=summary.get('original_flips',0),
                    missing=[k for k,v in summary.get('coverage',{}).items() if not v],failures=len(summary.get('failures',[]))),flush=True)
                last_print=now
            time.sleep(.25)
    except KeyboardInterrupt:
        status='stopped_by_user_incomplete'
    except Exception as exception:
        status='supervisor_error'; error=repr(exception)
    finally:
        for action in (manager.stop_match,manager.disconnect):
            try:action()
            except Exception as exception:error=(error or '')+' cleanup: '+repr(exception)
        protected_ok=verify_sources()==before
        report=make_report(folder,status,protected_ok,error)
        (folder/'run_summary.json').write_text(json.dumps(dict(status=status,error=error,protected_ok=protected_ok),indent=2),encoding='utf-8')
        print('V8 STOPPED:',status,flush=True)
        print('Observed all required categories:',report['all_required_categories_observed'],'Failures:',report['comparison_failures'],flush=True)
        print('Results:',folder,flush=True)
        print('Reports:',ROOT/'training/reports/v8_temporal_teacher_equivalence_20261004.md',flush=True)
        print('Stop for review. No ML or candidate deployment.',flush=True)


if __name__=='__main__':
    main()
