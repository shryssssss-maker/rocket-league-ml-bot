"""User-run three-match natural pilot. --check never launches RLBot."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib
from common import ROOT,HERE,PILOT,SPLITS,FEATURES,CHANNELS,preserve,sha
from dataset import report


def prepare(match_id):
    folder=PILOT/match_id
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'raw').mkdir()
    cmd=subprocess.list2cmdline([str(preserve.INTERPRETER),'-u','-B',str(HERE/'recorder.py')])
    (folder/'run_agent.cmd').write_text('@echo off\n'+cmd+'\n',encoding='utf-8')
    (folder/'bot.toml').write_text('[settings]\nname = "Original Teacher Trajectory Recorder"\nagent_id = "local/v9_source_recorder"\nrun_command = "run_agent.cmd"\nroot_dir = '+json.dumps(str(folder))+'\nloadout_file = '+json.dumps(str(ROOT/'python-example-original/src/loadout.toml'))+'\n',encoding='utf-8')
    text=(ROOT/'python-example-original/rlbot.toml').read_text(encoding='utf-8')
    for old,new in [('launcher = "Steam"','launcher = "Epic"'),('config_file = "src/bot.toml"','config_file = "bot.toml"'),('enable_rendering = false','enable_rendering = true'),('enable_state_setting = true','enable_state_setting = false')]:
        if text.count(old)!=1: raise RuntimeError('Unexpected original configuration: '+old)
        text=text.replace(old,new)
    (folder/'match.toml').write_text(text,encoding='utf-8')
    cfg=tomllib.loads(text)
    if len(cfg['cars'])!=2 or cfg['cars'][0]['type']!='Human' or cfg['cars'][0]['team']!='Blue' or cfg['cars'][1]['team']!='Orange': raise RuntimeError('Unexpected roster')
    if cfg['match']['skip_replays'] or cfg['match']['start_without_countdown'] or cfg['match']['enable_state_setting']: raise RuntimeError('Non-natural match config')
    from rlbot.config import load_player_config
    load_player_config(folder/'bot.toml',team=1)
    meta=dict(match_id=match_id,session_id=match_id,pilot_id=PILOT.name,split=SPLITS[match_id],
        split_plan=SPLITS,source_revision=preserve.REVISION,protected_hashes=preserve.verify_sources(),
        features=list(FEATURES),channels=list(CHANNELS),versions={n:importlib.metadata.version(n) for n in ('rlbot','rlbot-flatbuffers','psutil')},python=sys.version,
        code_hashes={str(p.relative_to(ROOT)):sha(p) for p in list(HERE.glob('*.py'))+[ROOT/'training/v2_observation_test/observation_contract.py']},
        config_hashes={n:sha(folder/n) for n in ('run_agent.cmd','bot.toml','match.toml')},
        opponent='Blue human, approved three complete natural matches',team=1,
        provenance='Original MyBot source methods, passive profile observer, actual RLBot packet/prediction',
        action_history='Actual prior successful PlayerInput submission; initial Neutral only at fresh agent lifetime',
        prediction='Prospective 18D probe independent of actual teacher selection. Empty prediction marks observation prediction invalid, as validated V8; original teacher far-branch IndexError still propagates.',
        no_resampling=True,hidden_sequence_diagnostic_only=True,production_candidate=False)
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    return folder


def check():
    import recorder
    from dataset import reconstruct
    before=preserve.verify_sources()
    # Real archived V8 callbacks are used only for preflight parity, never pilot data.
    p=ROOT/'training/v8_temporal_test/sessions/20261004_205108_820323/callbacks.jsonl'
    count=0
    for line in p.open(encoding='utf-8'):
        r=json.loads(line); o=r['relevant_observation']; pred=None
        if o['ball_present'] and r['auxiliary_observation18'][9]:
            # Saved normalized features do not provide raw probe targets at every callback.
            # Check actual source-selected rows with recorded raw targets only.
            pred=r['original_detail']['selected']
            if pred is None: continue
        elif r['auxiliary_observation18'][9]: continue
        previous=o['previous_actual_teacher_submission']
        if previous is not None:
            previous=dict(previous,mode=__import__('common').mode(previous['controls']))
        raw=dict(car=o['car'],ball=o['ball'],ball_present=o['ball_present'],T=r['T'],
            previous_elapsed=None if r['callback']==1 else r['T']-r['callback_dt'],
            previous_submission=previous,observation_prediction=pred,prediction_first_time=o['first_prediction_time'])
        import struct
        if struct.pack('<18f',*reconstruct(raw))!=struct.pack('<18f',*r['auxiliary_observation18']): raise RuntimeError('Preflight 18D mismatch')
        count+=1
    if count<400: raise RuntimeError('Insufficient preflight evidence')
    # Parse an isolated preview configuration, then remove no files: retain it as provenance.
    preview=HERE/'preflight'/__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
    preview.mkdir(parents=True,exist_ok=False)
    text=(ROOT/'python-example-original/rlbot.toml').read_text().replace('launcher = "Steam"','launcher = "Epic"').replace('enable_state_setting = true','enable_state_setting = false')
    cfg=tomllib.loads(text)
    assert len(cfg['cars'])==2 and not cfg['match']['enable_state_setting']
    assert preserve.verify_sources()==before
    (preview/'result.json').write_text(json.dumps(dict(status='non_live_preflight_pass',archived_rows_compared=count,protected_hashes=43,game_launched=False,pilot_rows_created=0),indent=2))
    report()
    print('PREFLIGHT PASS:',count,'archived real rows, unchanged 18D reconstruction, native action validation, imports; 43 protected hashes unchanged. No live launch.',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--report',action='store_true')
    parser.add_argument('--match-id',choices=list(SPLITS))
    args=parser.parse_args()
    if Path(sys.executable).resolve()!=preserve.INTERPRETER.resolve(): raise RuntimeError('Use existing live interpreter')
    if args.check: return check()
    if args.report: return report()
    if not args.match_id: parser.error('--match-id is required for live collection')
    before=preserve.verify_sources()
    folder=prepare(args.match_id)
    from rlbot import flat
    from rlbot.managers import MatchManager
    manager=MatchManager(); reader=preserve.SummaryReader()
    os.environ['V9_SESSION']=str(folder); os.environ['PYTHONDONTWRITEBYTECODE']='1'
    status='interrupted'; error=None; end_evidence=None
    print('V9 natural 1v1:',args.match_id,'split',SPLITS[args.match_id],flush=True)
    print('Play normally as BLUE human against ORANGE original teacher. No resets or diagnostic pauses. Wait for natural match end; Ctrl+C makes this match incomplete.',flush=True)
    print('Session:',folder,flush=True)
    try:
        manager.start_match(folder/'match.toml',wait_for_start=False)
        startup=time.monotonic()+180; live=None; progress=0
        while True:
            now=time.monotonic(); s=reader.read(folder/'agent_summary.json')
            if s.get('callbacks',0)>0 and live is None: live=now+1200
            if s.get('status')=='teacher_error': status='teacher_error'; break
            p=manager.packet
            if live is not None and p is not None and p.match_info.match_phase==flat.MatchPhase.Ended:
                status='natural_match_ended'
                end_evidence=dict(frame=int(p.match_info.frame_num),T=float(p.match_info.seconds_elapsed),phase=str(p.match_info.match_phase),source='Actual supervisor RLBot packet')
                break
            if (live is None and now>startup) or (live is not None and now>live): status='bounded_timeout_incomplete'; break
            if now-progress>10:
                print('V9',dict(callbacks=s.get('callbacks',0),flips=s.get('flips',0),invalid=s.get('invalid_rows',0)),flush=True)
                progress=now
            time.sleep(.25)
    except KeyboardInterrupt: status='stopped_by_user_incomplete'
    except Exception as e: status='supervisor_error'; error=repr(e)
    finally:
        for fn in (manager.stop_match,manager.disconnect):
            try: fn()
            except Exception as e: error=(error or '')+' cleanup: '+repr(e)
        deadline=time.monotonic()+5
        while not (folder/'recorder_closed.json').exists() and time.monotonic()<deadline: time.sleep(.1)
        if not (folder/'recorder_closed.json').exists():
            status='recorder_shutdown_unconfirmed'; error=(error or '')+' Recorder close acknowledgement missing; raw data retained but not eligible.'
        protected=preserve.verify_sources()==before
        (folder/'run_summary.json').write_text(json.dumps(dict(status=status,error=error,protected_ok=protected,natural_end_packet=end_evidence),indent=2))
        files=[p for p in folder.rglob('*') if p.is_file() and p.name!='integrity.json']
        (folder/'integrity.json').write_text(json.dumps({p.relative_to(folder).as_posix():sha(p) for p in files},indent=2)+'\n')
        print('V9 STOPPED:',status,flush=True)
        report()
        print('Stop after the three-match pilot. No training.',flush=True)

if __name__=='__main__': main()
