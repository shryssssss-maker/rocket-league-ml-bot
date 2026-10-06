"""V10 natural-only collection; reuses the approved V9 recorder unchanged."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import tomllib

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE))
from common import ROOT,PILOT,FEATURES,CHANNELS,preserve,sha

ANCHOR=HERE/'pilot_preservation.json'
PLAN=HERE/'collection_plan.json'
MATCHES=HERE/'matches'

def write_json(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def anchor():
    files={p.relative_to(PILOT).as_posix():sha(p) for p in PILOT.rglob('*') if p.is_file()}
    if ANCHOR.exists():
        expected=json.loads(ANCHOR.read_text(encoding='utf-8'))
        if files!=expected: raise RuntimeError('Pilot contents changed; stop for review')
    else: write_json(ANCHOR,files)
    preserve.verify_sources()
    return files

def plan():
    if not PLAN.exists(): raise RuntimeError('No approved roster/split/coverage plan yet')
    p=json.loads(PLAN.read_text(encoding='utf-8'))
    if not p.get('user_approved'): raise RuntimeError('Collection plan requires user review')
    if p.get('pilot_opponent_id') not in p['opponents']: raise RuntimeError('Pilot opponent not identified')
    targets=p['coverage_targets']
    for key in ('jump_callbacks_total','dodge_callbacks_total','complete_sequences_total','min_matches_each_side_total','min_human_opponents'):
        if type(targets.get(key)) is not int or targets[key]<=0: raise RuntimeError('Invalid coverage target: '+key)
    if set(targets['completed_sequences_by_split'])!={'train','validation','test'} or any(type(v) is not int or v<=0 for v in targets['completed_sequences_by_split'].values()):
        raise RuntimeError('Explicit positive per-split sequence targets required')
    ids=set()
    sessions={}
    for m in p['matches']:
        if not re.fullmatch(r'v10_[0-9]{3}',m['match_id']) or m['match_id'] in ids: raise RuntimeError('Invalid/duplicate match ID')
        ids.add(m['match_id'])
        if m['split'] not in ('train','validation','test') or m['teacher_side'] not in ('Blue','Orange'): raise RuntimeError('Invalid split/side')
        if m['opponent_id'] not in p['opponents'] or not m['play_session_id'] or not m['play_style']: raise RuntimeError('Missing roster/session/style')
        previous=sessions.setdefault(m['play_session_id'],m['opponent_id'])
        if previous!=m['opponent_id']: raise RuntimeError('Play-session identity reused for a different opponent')
        existing=MATCHES/m['match_id']/'metadata.json'
        if existing.exists():
            frozen=json.loads(existing.read_text(encoding='utf-8'))
            if any(frozen[k]!=m[k] for k in ('match_id','split','teacher_side','opponent_id','play_session_id','play_style')):
                raise RuntimeError('Collected match assignment cannot be changed retroactively')
    return p

def config_text(side):
    text=(ROOT/'python-example-original/rlbot.toml').read_text(encoding='utf-8')
    for old,new in [('launcher = "Steam"','launcher = "Epic"'),('config_file = "src/bot.toml"','config_file = "bot.toml"'),('enable_rendering = false','enable_rendering = true'),('enable_state_setting = true','enable_state_setting = false')]:
        if text.count(old)!=1: raise RuntimeError('Unexpected original config')
        text=text.replace(old,new)
    if side=='Blue':
        # Swap team assignments, retaining original player types and game settings.
        text=text.replace('team = "Blue"','team = "V10_SWAP"').replace('team = "Orange"','team = "Blue"').replace('team = "V10_SWAP"','team = "Orange"')
    cfg=tomllib.loads(text)
    expected_human='Orange' if side=='Blue' else 'Blue'
    if len(cfg['cars'])!=2 or cfg['cars'][0]['type']!='Human' or cfg['cars'][0]['team']!=expected_human or cfg['cars'][1]['team']!=side:
        raise RuntimeError('Invalid 1v1 roster')
    if cfg['match']['enable_state_setting'] or cfg['match']['skip_replays'] or cfg['match']['start_without_countdown']: raise RuntimeError('Natural transition contract violated')
    return text

def prepare(m,p):
    folder=MATCHES/m['match_id']; folder.mkdir(parents=True,exist_ok=False)
    (folder/'raw').mkdir()
    command=subprocess.list2cmdline([str(preserve.INTERPRETER),'-u','-B',str(BASE/'recorder.py')])
    (folder/'run_agent.cmd').write_text('@echo off\n'+command+'\n',encoding='utf-8')
    (folder/'bot.toml').write_text('[settings]\nname = "V10 Original Teacher Recorder"\nagent_id = "local/v9_source_recorder"\nrun_command = "run_agent.cmd"\nroot_dir = '+json.dumps(str(folder))+'\nloadout_file = '+json.dumps(str(ROOT/'python-example-original/src/loadout.toml'))+'\n',encoding='utf-8')
    (folder/'match.toml').write_text(config_text(m['teacher_side']),encoding='utf-8')
    from rlbot.config import load_player_config
    load_player_config(folder/'bot.toml',team=0 if m['teacher_side']=='Blue' else 1)
    metadata=dict(**m,collection='V10_NATURAL',opponent_identity=p['opponents'][m['opponent_id']],
        session_id=m['match_id'],recording_session_id=m['match_id'],
        features=list(FEATURES),channels=list(CHANNELS),source_revision=preserve.REVISION,
        protected_hashes=preserve.verify_sources(),pilot_anchor_sha256=sha(ANCHOR),
        collection_plan_sha256=sha(PLAN),approved_plan=p,
        team=0 if m['teacher_side']=='Blue' else 1,
        code_sha256={str(f.relative_to(ROOT)):sha(f) for f in list(BASE.glob('*.py'))+list(HERE.glob('*.py'))+[ROOT/'training/v2_observation_test/observation_contract.py']},
        config_sha256={n:sha(folder/n) for n in ('match.toml','bot.toml','run_agent.cmd')},
        versions={n:importlib.metadata.version(n) for n in ('rlbot','rlbot-flatbuffers','psutil')},python=sys.version,
        provenance='Real Rocket League -> actual RLBot packet/prediction -> unchanged original teacher; unchanged approved V9 recorder',
        state_setting=False,intentional_receive_pauses=False,synthetic_trajectories=False,
        private_teacher_memory='Diagnostic only, excluded from 18D tensors',
        split_unit='Whole match assigned in approved plan before launch')
    write_json(folder/'metadata.json',metadata)
    return folder

def check():
    protected=preserve.verify_sources(); pilot=anchor()
    import recorder
    from rlbot.config import load_player_config
    from datetime import datetime
    folder=HERE/'preflight'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'); folder.mkdir(parents=True)
    for side in ('Blue','Orange'):
        f=folder/side; f.mkdir()
        (f/'match.toml').write_text(config_text(side),encoding='utf-8')
        (f/'bot.toml').write_text('[settings]\nname = "V10 Preflight"\nagent_id = "local/v9_source_recorder"\nrun_command = "run_agent.cmd"\nroot_dir = '+json.dumps(str(f))+'\n',encoding='utf-8')
        (f/'run_agent.cmd').write_text('@echo off\n',encoding='utf-8')
        load_player_config(f/'bot.toml',team=0 if side=='Blue' else 1)
    assert preserve.verify_sources()==protected and anchor()==pilot
    write_json(folder/'result.json',dict(status='non_live_config_preflight_pass',both_sides_parsed=True,
        protected_hashes=43,pilot_files_unchanged=len(pilot),game_launched=False,
        blue_teacher_live_verification='Pending first real blue-side match; config parsing is not live validation'))
    print('PREFLIGHT PASS: both side configs parsed, approved recorder imports, 43 hashes and',len(pilot),'pilot files unchanged. No live launch.',flush=True)

def collect(match_id):
    p=plan(); anchor(); before=preserve.verify_sources()
    m=next((m for m in p['matches'] if m['match_id']==match_id),None)
    if m is None: raise RuntimeError('Match not in preassigned plan')
    folder=MATCHES/match_id
    if folder.exists():
        # Permit a reviewed, prepared configuration once; never resume or overwrite data.
        expected_files={'metadata.json','bot.toml','match.toml','run_agent.cmd'}
        if {f.name for f in folder.iterdir()}!=expected_files|{'raw'} or not (folder/'raw').is_dir() or any((folder/'raw').iterdir()):
            raise RuntimeError('Existing match contains data or unexpected files; cannot overwrite')
        meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        if any(meta[k]!=m[k] for k in ('match_id','split','teacher_side','opponent_id','play_session_id','play_style')) or meta['collection_plan_sha256']!=sha(PLAN):
            raise RuntimeError('Prepared match differs from approved plan')
        for name,h in meta['config_sha256'].items():
            if sha(folder/name)!=h: raise RuntimeError('Prepared configuration changed')
        for name,h in meta['code_sha256'].items():
            if sha(ROOT/name)!=h: raise RuntimeError('Prepared recorder/launcher code changed; review before launch')
    else:
        folder=prepare(m,p)
    from rlbot import flat
    from rlbot.managers import MatchManager
    import psutil
    manager=MatchManager(); reader=preserve.SummaryReader()
    os.environ['V9_SESSION']=str(folder); os.environ['PYTHONDONTWRITEBYTECODE']='1'
    status='interrupted'; error=None; end=None; server_pid=None; cleanup_errors=[]
    human='Orange' if m['teacher_side']=='Blue' else 'Blue'
    print('V10:',m,flush=True)
    print('Play naturally as',human,'human. Teacher is',m['teacher_side']+'. No resets/probes. Ctrl+C retains an incomplete match.',flush=True)
    print('Session:',folder,flush=True)
    try:
        manager.start_match(folder/'match.toml',wait_for_start=False)
        server=manager.rlbot_server_process
        server_pid=server.pid if server is not None else None
        print('V10 server PID:',server_pid,flush=True)
        startup=time.monotonic()+180; live=None; progress=0
        while True:
            now=time.monotonic(); s=reader.read(folder/'agent_summary.json')
            if server is not None and (not server.is_running() or server.status()==psutil.STATUS_ZOMBIE):
                status='server_exited_before_callbacks' if not s.get('callbacks',0) else 'server_exited_incomplete'
                print('V10: RLBot server exited; stopping this incomplete attempt.',flush=True)
                break
            if s.get('callbacks',0)>0 and live is None: live=now+1200
            if s.get('callbacks',0)>0 and s.get('team')!=(0 if m['teacher_side']=='Blue' else 1):
                status='agent_side_mismatch'; break
            if s.get('status')=='teacher_error': status='teacher_error'; break
            packet=manager.packet
            if live is not None and packet is not None and packet.match_info.match_phase==flat.MatchPhase.Ended:
                status='natural_match_ended'; end=dict(frame=int(packet.match_info.frame_num),T=float(packet.match_info.seconds_elapsed),phase=str(packet.match_info.match_phase)); break
            if (live is None and now>startup) or (live is not None and now>live): status='bounded_timeout_incomplete'; break
            if now-progress>10:
                print('V10',dict(callbacks=s.get('callbacks',0),flips=s.get('flips',0),invalid=s.get('invalid_rows',0),teacher_side=m['teacher_side']),flush=True); progress=now
            time.sleep(.25)
    except KeyboardInterrupt: status='stopped_by_user_incomplete'
    except Exception as e: status='supervisor_error'; error=repr(e)
    finally:
        for fn in (manager.stop_match,manager.disconnect):
            try: fn()
            except Exception as e: cleanup_errors.append(repr(e))
        deadline=time.monotonic()+5
        while not (folder/'recorder_closed.json').exists() and time.monotonic()<deadline: time.sleep(.1)
        recorder_closed=(folder/'recorder_closed.json').exists()
        # Preserve the primary failure; unconfirmed closure only invalidates a natural end.
        if not recorder_closed and status=='natural_match_ended': status='recorder_shutdown_unconfirmed'
        protected=preserve.verify_sources()==before; anchor()
        write_json(folder/'run_summary.json',dict(status=status,error=error,cleanup_errors=cleanup_errors,
            recorder_shutdown_confirmed=recorder_closed,server_pid=server_pid,
            protected_ok=protected,natural_end_packet=end))
        files=[f for f in folder.rglob('*') if f.is_file() and f.name!='integrity.json']
        write_json(folder/'integrity.json',{f.relative_to(folder).as_posix():sha(f) for f in files})
        print('V10 STOPPED:',status,'Results:',folder,flush=True)
        print('Run the offline V10 review command next. No training or next match starts automatically.',flush=True)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--check',action='store_true'); parser.add_argument('--match-id'); parser.add_argument('--review',action='store_true')
    args=parser.parse_args()
    if Path(sys.executable).resolve()!=preserve.INTERPRETER.resolve(): raise RuntimeError('Use existing live interpreter')
    if args.check: return check()
    if args.review:
        import coverage_review
        return coverage_review.review()
    if not args.match_id: parser.error('--match-id, --check or --review required')
    collect(args.match_id)

if __name__=='__main__': main()
