"""One natural original-teacher match per explicit --run. --check never connects."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from contracts import HERE,ROOT,LIVE_PYTHON,FEATURES,CHANNELS,load,read,write,sha,verify,require
diagnostics=load('_v16_summary_reader',ROOT/'training/behavior_cloning/v13_dagger/phase3_shadow_v2/diagnostics.py')

def config_text(side):
    text=(ROOT/'python-example-original/rlbot.toml').read_text(encoding='utf-8')
    for old,new in [('launcher = "Steam"','launcher = "Epic"'),('config_file = "src/bot.toml"','config_file = "bot.toml"'),
                    ('enable_rendering = false','enable_rendering = true'),('enable_state_setting = true','enable_state_setting = false')]:
        require(text.count(old)==1,'Unexpected original configuration: '+old); text=text.replace(old,new)
    if side=='Blue': text=text.replace('team = "Blue"','team = "SWAP"').replace('team = "Orange"','team = "Blue"').replace('team = "SWAP"','team = "Orange"')
    cfg=tomllib.loads(text)
    require(len(cfg['cars'])==2 and cfg['cars'][0]['type']=='Human' and cfg['cars'][1]['team']==side,'Natural 1v1 roster')
    require(not cfg['match']['enable_state_setting'] and not cfg['match']['skip_replays'] and not cfg['match']['start_without_countdown'],'Natural transitions only')
    require(cfg['mutators']['match_length']=='FiveMinutes' and cfg['mutators']['multi_ball']=='One','Five-minute single-ball match')
    return text

def prepare(m,folder):
    seal=verify()
    folder.mkdir(parents=True,exist_ok=False); (folder/'raw').mkdir()
    command=subprocess.list2cmdline([str(LIVE_PYTHON),'-u','-B',str(HERE/'collector/live_agent.py')])
    # Bind to THIS config, even when an already-running server has stale parent environment.
    (folder/'run_agent.cmd').write_text('@echo off\nset "V16_SESSION='+str(folder)+'"\nset "PYTHONDONTWRITEBYTECODE=1"\n'+command+'\n',encoding='utf-8')
    (folder/'bot.toml').write_text('[settings]\nname = "V16 Original Teacher Native Audit"\nagent_id = "local/v16_native_audit"\nrun_command = "run_agent.cmd"\nroot_dir = '+__import__('json').dumps(str(folder))+'\nloadout_file = '+__import__('json').dumps(str(ROOT/'python-example-original/src/loadout.toml'))+'\n',encoding='utf-8')
    (folder/'match.toml').write_text(config_text(m['teacher_side']),encoding='utf-8')
    from rlbot.config import load_player_config
    load_player_config(folder/'bot.toml',team=0 if m['teacher_side']=='Blue' else 1)
    write(folder/'metadata.json',dict(**m,session_id=m['match_id'],team=0 if m['teacher_side']=='Blue' else 1,
          version='v16_native_observability_v1',created_utc=datetime.now(timezone.utc).isoformat(),features=list(FEATURES),channels=list(CHANNELS),
          source_manifest_sha256=sha(HERE/'source_manifest.json'),source_hashes=seal['files'],
          config_hashes={name:sha(folder/name) for name in ('match.toml','bot.toml','run_agent.cmd')},
          versions={name:importlib.metadata.version(name) for name in ('rlbot','rlbot-flatbuffers','psutil')},
          provenance='Real game, actual delivered packets/predictions, unchanged original teacher only',
          state_setting=False,intentional_receive_pauses=False,student=False,policy_fitting=False,
          private_teacher_state='Diagnostic labels only; never in probe inputs',last_input_semantics='Pending observational timing characterization'))
    return folder

def check():
    verify()
    import live_agent
    require(live_agent.NativeRecorder.get_output.__module__=='live_agent','Recorder import failed')
    folder=HERE/'preflight'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    for side in ('Blue','Orange'):
        f=prepare(dict(match_id='non_live_'+side,opponent_id='preflight_not_a_person',play_session_id='non_live',teacher_side=side),folder/side)
        require('set "V16_SESSION='+str(f)+'"' in (f/'run_agent.cmd').read_text(encoding='utf-8'),'Batch must bind its own match path')
    # Non-game publication test catches the earlier Windows overwrite failure pattern.
    publisher=diagnostics.SummaryPublisher(folder)
    first=publisher.publish(dict(callbacks=1)); second=publisher.publish(dict(callbacks=2))
    require(first!=second and first.is_file() and second.is_file() and not publisher.errors,'Immutable publication check')
    verify()
    write(folder/'result.json',dict(status='non_live_collection_preflight_pass',protected=43,both_configs_parsed=True,
          teacher_imported=True,source_state='Unchanged original; no instance/socket constructed',game_launched=False,
          last_input_live_semantics='Still pending; preflight cannot establish engine meaning'))
    print('V16 NON-LIVE COLLECTION PREFLIGHT PASS:',folder,flush=True)
    print('No game/collector launched; native timing semantics remain pending live capture.',flush=True)

def collect(match_id):
    verify()
    plan=read(HERE/'collection_plan.json')
    m=next((x for x in plan['matches'] if x['match_id']==match_id),None)
    require(m is not None,'Match not assigned by approved user roster')
    folder=HERE/'matches'/match_id
    require(not folder.exists(),'Existing match/config/data cannot be overwritten or resumed')
    # Persist assignments, hashes and config BEFORE any launch/connection.
    prepare(m,folder)
    write(folder/'launch_started.json',dict(monotonic=time.monotonic(),utc=datetime.now(timezone.utc).isoformat()))
    from rlbot import flat
    from rlbot.managers import MatchManager
    import psutil
    manager=MatchManager(); reader=diagnostics.SummaryReader()
    os.environ['V16_SESSION']=str(folder); os.environ['PYTHONDONTWRITEBYTECODE']='1'
    status='incomplete'; error=None; end=None; cleanup=[]
    print('V16 ASSIGNMENT:',m,flush=True)
    print('Original teacher ONLY. Play naturally as', 'Orange' if m['teacher_side']=='Blue' else 'Blue',
          'human. One five-minute match, replays/countdowns/overtime retained. No probes/student.',flush=True)
    print('Session:',folder,flush=True)
    try:
        manager.start_match(folder/'match.toml',wait_for_start=False)
        server=manager.rlbot_server_process; deadline=time.monotonic()+180; live_deadline=None; progress=0
        while True:
            now=time.monotonic(); summary=reader.read(folder)
            if server is not None and (not server.is_running() or server.status()==psutil.STATUS_ZOMBIE): status='server_exited_incomplete'; break
            if summary.get('callbacks',0) and live_deadline is None: live_deadline=now+1200
            if summary.get('status','running')!='running' or summary.get('publication_errors'): status='diagnostic_or_teacher_error'; break
            if summary.get('callbacks',0) and summary.get('team')!=(0 if m['teacher_side']=='Blue' else 1): status='side_mismatch'; break
            packet=manager.packet
            if live_deadline is not None and packet is not None and packet.match_info.match_phase==flat.MatchPhase.Ended:
                status='natural_match_ended'; end=dict(T=float(packet.match_info.seconds_elapsed),frame=int(packet.match_info.frame_num)); break
            if now>(deadline if live_deadline is None else live_deadline): status='timeout_incomplete'; break
            if now-progress>=10:
                print('V16',dict(callbacks=summary.get('callbacks',0),starts=summary.get('starts',0),completions=summary.get('completions',0),teacher_side=m['teacher_side']),flush=True); progress=now
            time.sleep(.25)  # Supervisor only; no intentional agent receive-loop pauses.
    except KeyboardInterrupt: status='user_interrupted_incomplete'
    except Exception as e: status='supervisor_error'; error=repr(e)
    finally:
        for fn in (manager.stop_match,manager.disconnect):
            try: fn()
            except Exception as e: cleanup.append(repr(e))
        deadline=time.monotonic()+5
        while not (folder/'agent_closed.json').exists() and time.monotonic()<deadline: time.sleep(.1)
        closed=read(folder/'agent_closed.json') if (folder/'agent_closed.json').exists() else None
        if status=='natural_match_ended' and (closed is None or closed['status']!='running' or closed['last_submission'] is None or closed['last_submission']['callback']!=closed['callbacks']):
            status='closure_or_final_submission_incomplete'
        verify()
        write(folder/'run_summary.json',dict(status=status,error=error,cleanup_errors=cleanup,natural_end_packet=end,
              recorder_closed=closed is not None,protected_hashes_unchanged=43,monotonic=time.monotonic()))
        write(folder/'integrity.json',dict(files={p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()},
              hash_scope='all match configs, metadata, raw journals, summaries and closure artifacts; self excluded'))
        print('V16 STOPPED:',status,'Results:',folder,flush=True)
        print('No next match/analysis/training starts automatically.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true'); group.add_argument('--run',action='store_true')
    parser.add_argument('--match-id'); args=parser.parse_args()
    require(Path(sys.executable).resolve()==LIVE_PYTHON.resolve(),'Use pinned live interpreter')
    if args.check: check()
    else:
        require(args.match_id is not None,'Explicit single match ID required'); collect(args.match_id)
