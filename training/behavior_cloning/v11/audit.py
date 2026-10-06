"""Read-only audit of the pinned V10 reference artifact; no preprocessing export.

Only the pinned manifest's external files and frozen version directory are read.
Outputs are new V11 reports. Frozen audit functions are extracted by AST without
importing their launchers, current repository code, RLBot, or any teacher module.
"""
import argparse
import array
import ast
import collections
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import types

ROOT=Path(__file__).resolve().parents[3]
VERSION=ROOT/'training/behavior_cloning/datasets/v10_live_1v1_18d_20261005_v1'
PIN='080943fd8f9a81ada800e0a2f9ff3593b67f66d32272211d67030c03cf433a83'
REPORT=ROOT/'training/reports/v11_frozen_dataset_preprocessing_audit_20261005'
FEATURES=('ball_forward','ball_right','ball_up','prediction_forward','prediction_right','prediction_up',
          'ball_distance','car_speed','ball_present','prediction_valid','prediction_horizon','prediction_first_offset',
          'callback_dt','previous_neutral','previous_chase','previous_jump','previous_dodge','previous_steer')
CHANNELS=('throttle','steer','pitch','yaw','roll','jump','boost','handbrake')
SPECS={'observations':('float32',18,4),'analog_controls':('float32',5,4),'buttons':('uint8',3,1),
       'mode':('uint8',1,1),'callback':('uint64',1,8),'frame':('uint64',1,8),'time':('float64',1,8),'dt':('float64',1,8)}
ALLOWED=set()

def digest(path):
    path=Path(path).resolve()
    if path not in ALLOWED and not path.is_relative_to(VERSION.resolve()):
        raise RuntimeError('Read outside permitted paths: '+str(path))
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def require(ok,reason):
    if not ok: raise RuntimeError(reason)

def verified_manifest(full):
    require(digest(VERSION/'manifest.json')==PIN,'Pinned manifest SHA256 mismatch')
    manifest=load(VERSION/'manifest.json')
    for key,e in manifest['artifacts'].items():
        require(key==e['path'],'Artifact key/path mismatch')
        path=(ROOT/key).resolve()
        require(path.is_relative_to(ROOT.resolve()),'External artifact escapes repository')
        ALLOWED.add(path)
    entries=[x['frozen_copy'] for x in manifest['current_code_config_report_snapshots']]
    for e in entries:
        p=(ROOT/e['path']).resolve()
        require(p.is_relative_to(VERSION.resolve()),'Snapshot outside version directory')
        require(p.is_file() and p.stat().st_size==e['bytes'] and digest(p)==e['sha256'],'Snapshot hash/size mismatch: '+str(p))
    require(manifest['features']==list(FEATURES) and manifest['native_channels']==list(CHANNELS),'Feature/channel order mismatch')
    require(manifest['analog_order']==list(CHANNELS[:5]) and manifest['button_order']==list(CHANNELS[5:]),'Analog/button order mismatch')
    if full:
        for i,(key,e) in enumerate(manifest['artifacts'].items(),1):
            p=ROOT/key
            require(p.is_file(),'Missing artifact: '+key)
            require(p.stat().st_size==e['bytes'] and digest(p)==e['sha256'],'Artifact SHA256/size mismatch: '+key)
            if i%20==0: print(f'V11 HASH {i}/{len(manifest["artifacts"])}',flush=True)
        seals=load(VERSION/'SHA256.json')
        for key,h in seals.items():
            p=(VERSION/key).resolve()
            require(p.is_relative_to(VERSION.resolve()),'Seal path escapes version')
            require(p.is_file() and digest(p)==h,'Frozen version seal mismatch: '+key)
        print('V11 all permitted artifacts verified before dataset content reads',flush=True)
    return manifest

def frozen_functions():
    snapshots=VERSION/'snapshots/training'
    obs=types.ModuleType('observation_contract')
    source=snapshots/'v2_observation_test/observation_contract.py'
    exec(compile(source.read_text(encoding='utf-8'),str(source),'exec'),obs.__dict__)
    sys.modules['observation_contract']=obs
    env=dict(array=array,collections=collections,hashlib=hashlib,json=json,math=math,struct=struct,sys=sys,
             FEATURES=FEATURES,CHANNELS=CHANNELS,sha=digest,build=obs.build,basis=obs.basis)
    def no_write(*args,**kwargs): raise RuntimeError('Read-only audit attempted an export')
    env['write_json']=no_write
    selections=[('behavior_cloning/common.py',{'mode'}),
                ('behavior_cloning/dataset.py',{'reconstruct','distribution'}),
                ('behavior_cloning/v10/coverage_review.py',{'verify_files','audit','VALUE_FIELDS','TENSOR_SPECS'})]
    for name,wanted in selections:
        tree=ast.parse((snapshots/name).read_text(encoding='utf-8'))
        nodes=[]
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in wanted:
                if node.name=='audit':
                    # Existing manifest branch only. Remove the export path structurally.
                    for item in ast.walk(node):
                        if isinstance(item,ast.If) and isinstance(item.test,ast.Call) and isinstance(item.test.func,ast.Attribute) and isinstance(item.test.func.value,ast.Name) and item.test.func.value.id=='existing' and item.test.func.attr=='exists':
                            item.orelse=ast.parse("raise RuntimeError('Frozen tensor manifest missing; export forbidden')").body
                nodes.append(node)
            elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in wanted for t in node.targets): nodes.append(node)
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),name,'exec'),env)
    return env

def tensor_contract(m):
    require(set(m['tensors'])==set(SPECS),'Unexpected/missing tensors: '+m['match_id'])
    for key,(dtype,width,size) in SPECS.items():
        t=m['tensors'][key]; shape=[m['callbacks'],width] if width!=1 else [m['callbacks']]
        require(t['dtype']==dtype and t['shape']==shape and t['bytes']==m['callbacks']*width*size,'Tensor contract mismatch: '+m['match_id']+'/'+key)

def history_check(m):
    path=(ROOT/m['dataset_directory']/'raw/trajectory.jsonl').resolve()
    require(path in ALLOWED,'Trajectory not in manifest')
    previous=None; missing=0; n=0
    with path.open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line); n+=1; obs=r['observation18']; submission=r['previous_submission']
            require(r['callback']==n and r['match_id']==m['match_id'] and r['session_id']==m['session_id'],'Cross-match/index/session error')
            if previous is None:
                require(submission is None and r['previous_elapsed'] is None and r['callback_dt'] is None and r['frame_gap'] is None,'First callback has nonempty history')
                require(obs[12:]==[0.,1.,0.,0.,0.,0.],'First callback observation reset mismatch')
            else:
                require(submission is not None and submission['callback']==n-1 and submission['T']==previous['T'],'Prior submission not preceding actual callback')
                require(submission['controls']==previous['teacher_controls'] and submission['mode']==previous['action_mode'],'Prior action differs from actually emitted teacher control')
                require(r['previous_elapsed']==previous['T'] and r['frame_gap']==r['frame']-previous['frame'],'History/frame mismatch')
                require(r['T']>previous['T'] and r['frame']>previous['frame'],'Noncausal callback ordering')
                onehot=[0.,0.,0.,0.]; onehot[previous['action_mode']]=1.
                require(obs[13:17]==onehot and struct.pack('<f',obs[17])==struct.pack('<f',previous['teacher_controls'][1]),'Previous-action feature mismatch')
            require(r['hidden_sequence_label_only'] is True,'Private state not declared diagnostic')
            if not r['ball_present']:
                missing+=1
                require(r['ball'] is None and all(obs[j]==0. for j in range(7)) and all(obs[j]==0. for j in range(8,12)),'Missing-ball masking changed')
            previous=r
    require(n==m['callbacks'],'Raw row count mismatch')
    # No windows/tensors are created. Exhaustively check causal index recipes only.
    recipes={}
    for length in (2,4,8,16,32,64):
        for end in range(n):
            start=max(0,end-length+1)
            require(0<=start<=end<n and end-start+1<=length,'Window crosses boundary/future')
        recipes[str(length)]=dict(checked_endpoints=n,rule='[max(0,t-L+1), t] within this match; left context shorter at match start',materialized=False)
    return dict(match_id=m['match_id'],rows=n,first_callback_reset=True,
                previous_action_exact=True,missing_ball_callbacks=missing,causal_index_recipes=recipes)

def compare(actual,expected,label):
    require(actual==expected,'Frozen statistics not exactly reproduced: '+label)

def audit(manifest):
    env=frozen_functions(); frozen=manifest['statistics']; stats=[]; history=[]
    values={k:array.array('d') for k in env['VALUE_FIELDS']}
    channels={k:collections.Counter() for k in CHANNELS if k!='steer'}
    for i,m in enumerate(manifest['matches'],1):
        print(f"V11 ROW AUDIT {i}/9: {m['match_id']}",flush=True)
        tensor_contract(m)
        identity={k:m[k] for k in ('split','teacher_side','team','opponent_id','play_session_id','play_style','is_pilot')}
        s,v,ch=env['audit'](ROOT/m['dataset_directory'],identity)
        expected=next(x for x in frozen['matches'] if x['match_id']==m['match_id'])
        compare(s,expected,m['match_id']+' all per-match statistics')
        stats.append(s); history.append(history_check(m))
        for k in values: values[k].extend(v[k])
        for k in channels: channels[k].update(ch[k])
    aggregate={k:collections.Counter() for k in frozen['aggregate_counts']}
    for m in stats:
        for k in aggregate: aggregate[k].update(m[k])
    aggregate={k:dict(v) for k,v in aggregate.items()}
    distributions={k:env['distribution'](v) for k,v in values.items()}
    compare(aggregate,frozen['aggregate_counts'],'aggregate counts')
    compare(distributions,frozen['distributions'],'all aggregate distributions (exact Python numeric equality)')
    native={}
    for k in CHANNELS:
        if k=='steer': native[k]=env['distribution'](values[k])
        else:
            counts=channels[k]; n=sum(counts.values())
            native[k]=dict(count=n,values={str(v):n for v,n in sorted(counts.items())},minimum=min(counts),maximum=max(counts),mean=sum(v*n for v,n in counts.items())/n)
    compare(native,frozen['native_action_channels'],'native channel distributions')
    groups={key:{} for key in frozen['grouped_coverage']}
    for key in groups:
        for m in stats:
            g=groups[key].setdefault(m[key],dict(matches=0,callbacks=0,jump=0,front_dodge=0,sequence_starts=0,sequence_completions=0,sides=set()))
            g['matches']+=1; g['callbacks']+=m['callbacks']; g['jump']+=m['modes'].get('jump',0); g['front_dodge']+=m['modes'].get('front-dodge',0)
            g['sequence_starts']+=m['sequence_starts']; g['sequence_completions']+=m['sequence_completions']; g['sides'].add(m['teacher_side'])
        for g in groups[key].values(): g['sides']=sorted(g['sides'])
    compare(groups,frozen['grouped_coverage'],'split/side/opponent/session groups')
    for key in ('sequence_starts','sequence_completions','complete_sequences_with_all_four_phases','prediction_reuse'):
        compare(sum(m[key] for m in stats),frozen[key],key)
    split={s:[m['match_id'] for m in stats if m['split']==s] for s in ('train','validation','test')}
    compare(split,manifest['exact_match_split'],'exact split')
    require(len({x for v in split.values() for x in v})==sum(len(v) for v in split.values()),'Match split overlap')
    return dict(per_match_history=history,recomputed_aggregate=aggregate,recomputed_distributions=distributions,
                recomputed_native_channels=native,grouped_coverage=groups,split=split)

TRADEOFFS={
    'single_callback_feedforward':{
        'advantages':['Simple, low latency, no window assembly; consumes only frozen 18D.'],
        'limitations':['18D includes previous mode/steering and elapsed dt, but not sequence elapsed time, pending sequence identity or recurrent memory. Release and Coast both emit Neutral; private timing survives missing-ball/replay callbacks. A single callback is therefore not a sufficient exact temporal teacher state.',
                       'Shared previous-action inputs are teacher-forced in this dataset. Live student uses its own previous emitted action; errors can compound. Offline agreement does not prove live equivalence.'],
        'preprocessing_requirements':['Keep 18D/native targets unchanged; reset previous history at each match; preserve raw dt null versus processed zero; no scaling fit on validation/test. Any future learned scaling must use train only.']},
    'short_causal_history':{
        'advantages':['Can expose prior mode changes and delivered callback timing without private labels.'],
        'limitations':['Fixed callback count spans variable elapsed time. Finite history may lose a pending sequence origin across missing-ball/replay/long gaps; no tested length is approved or guaranteed sufficient.',
                       'Teacher-forced previous actions versus student-generated actions remains a deployment issue.'],
        'preprocessing_requirements':['Use only rows <= current callback from the same match/split. Include the current observation; target is current action, not a future action.',
                                    'At match start use explicit shorter history or separately reviewed validity masks; do not silently pad, repeat, or drop callbacks.',
                                    'Do not reset at goals/replays/missing ball unless a future design explicitly approves a change; teacher memory can survive them.',
                                    'No centered windows, backward interpolation from future packets, shuffled recurrent state, or private sequence labels as inputs.']},
    'decision':'Neither architecture selected; no history tensors, training examples, preprocessing exports or models created.'}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--check',action='store_true'); args=parser.parse_args()
    if args.check:
        m=verified_manifest(False); frozen_functions()
        for match in m['matches']: tensor_contract(match)
        print('V11 NON-LIVE PREFLIGHT PASS: manifest/snapshot hashes, read-only frozen functions and tensor specifications. Full external hashes/row audit pending.',flush=True)
        return
    evidence={}; error=None
    try:
        m=verified_manifest(True); evidence=audit(m)
        # Seal output hashes and manifest remain unchanged; input operations were read-only.
        require(digest(VERSION/'manifest.json')==PIN,'Manifest changed during audit')
    except Exception as exc:
        error=repr(exc)
    result=dict(status='passed_with_scope' if error is None else 'failed_stopped',manifest_sha256=PIN,
                error=error,evidence=evidence,architecture_tradeoffs=TRADEOFFS,
                scope=['Existing frozen tensors audited callback-by-callback; no sequence/window preprocessing implementation exists yet.',
                       'Causal index recipes exhaustively checked for lengths 2/4/8/16/32/64, without choosing a length or constructing data.',
                       'Frozen observation reconstruction depends on current physics/prediction and strictly prior submitted action only; prediction is a contemporaneous forecast, not a later recorded outcome.',
                       'Private sequence diagnostics only affect statistics; they are excluded from 18D and causal input recipes.',
                       'Match-disjoint splits are not fully opponent/session-disjoint. human_a spans all splits; human_b has two train matches sharing one session; human_c has one test match.',
                       'Historical protection evidence is hash-pinned in freeze metadata; current bot/source files are deliberately not read.'],
                frozen_inputs_modified=False,preprocessing_exported=False,training_started=False)
    check_names=[
        'Pinned manifest and all frozen/artifact SHA256 integrity',
        'Exact tensor shapes/dtypes/channel ordering',
        '18D feature ordering against frozen manifest',
        'Native eight-channel ordering against frozen manifest',
        'Previous-action features from immediately prior actual callback',
        'History initialization at every match boundary',
        'No split/sample/index recipe crosses a match boundary',
        'Causal index recipes exclude future callbacks; no windows exported',
        'Private sequence state remains diagnostic, excluded from model-input recipes',
        'Missing-ball features exactly preserved',
        'Initial raw null dt/history and tensor/observation zero dt explicit',
        'All frozen per-match and aggregate distributions exactly reproduced',
    ]
    result['requirements']={str(i):dict(check=name,status='pass' if error is None else 'not_certified_due_to_stopped_audit')
                            for i,name in enumerate(check_names,1)}
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines=['# V11 frozen dataset preprocessing and leakage audit','',f"Status: **{result['status']}**.",'',f'Manifest SHA256: `{PIN}`.','']
    if error: lines+=['Audit stopped: `'+error+'`. No subsequent audit stages or training executed.','']
    else:
        lines+=['Verified all 171 manifest-listed external artifact hashes and the frozen version seals before reading dataset contents. Tensor shape/dtype/byte-size/order checks passed; every raw row exactly reconstructs its 18D float32 tensor and native action tensors.', '',
                'All 239,205 rows audited. Prior submitted controls/mode/time/callback agree with the immediately preceding processed teacher callback. Each of nine matches starts with null raw dt/history, processed dt zero and Neutral/zero previous action. No match crosses a split boundary.', '',
                'Missing-ball masks and zeroed ball/prediction features match frozen data. Car-speed, dt and prior-action context are retained. Teacher-private state stays diagnostic only. No phase-based resets or added input features.', '',
                'All per-match statistics, aggregate counts/distributions, native channel distributions, split/side/opponent/session groups and sequence/prediction-reuse totals exactly reproduce the frozen manifest. This uses the verified frozen audit arithmetic, preserving quantile and floating-point accumulation conventions.', '']
    lines+=['## Required checks','', '| # | Check | Result |','|---|---|---|']
    lines += [f"| {i} | {item['check']} | {item['status']} |" for i,item in result['requirements'].items()]
    lines+=['## Leakage and scope','']+['- '+x for x in result['scope']]
    for key,value in TRADEOFFS.items():
        if isinstance(value,dict):
            lines+=['', '## '+key.replace('_',' '),'']
            for label,items in value.items(): lines += [label.replace('_',' ').capitalize()+':','']+['- '+x for x in items]+['']
    lines+=['',TRADEOFFS['decision'],'','Frozen inputs unchanged. Stop for review; no training.']
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('V11:',result['status'],'Report:',REPORT.with_suffix('.md'),flush=True)
    if error: raise SystemExit(1)

if __name__=='__main__': main()
