"""Phase 2: pinned 13D GRU bootstrap; train/validation only, no live control.

All output is confined to this new experimental directory. V12 routines are
read-only, hash-verified and extracted without executing its launcher.
"""
import argparse
import ast
from datetime import datetime,timezone,timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import projection

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
V12=HERE.parent/'v12'
PIN='080943fd8f9a81ada800e0a2f9ff3593b67f66d32272211d67030c03cf433a83'
PROJECTION=HERE/'projection_13d_v1/manifest.json'
GATE=HERE/'phase2_bootstrap_v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

class Policy(nn.Module):
    def __init__(self,kind='B'):
        super().__init__();self.kind='B'
        self.encoder=nn.Sequential(nn.Linear(13,64),nn.ReLU())
        self.gru=nn.GRU(64,64,num_layers=1,batch_first=True,bidirectional=False)
        self.head=nn.Linear(64,5)
    def forward(self,x,h=None):
        z,h=self.gru(self.encoder(x).unsqueeze(0),h)
        out=self.head(z.squeeze(0))
        return out[:,:4],torch.tanh(out[:,4]),h

def prepared():
    audit=read(HERE/'phase1_context_audit_v1/audit.json')
    if audit['status']!='passed_with_scope':raise RuntimeError('Phase1 incomplete')
    for p,h in audit['v12_verified_hashes'].items():
        if sha(Path(p))!=h:raise RuntimeError('V12 artifact changed: '+p)
    pm=read(PROJECTION)
    if sha(HERE/'projection.py')!=pm['projection_code_sha256'] or sha(Path(pm['projection_code_path']))!=pm['projection_code_sha256']:
        raise RuntimeError('Projection code changed')
    spec=importlib.util.spec_from_file_location('v13_verified_reader',HERE.parent/'v11/audit.py')
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    reader.verified_manifest(False)
    env=dict(np=np,torch=torch,nn=nn,F=F,json=json,Path=Path,ROOT=ROOT,reader=reader,
             CHANNELS=reader.CHANNELS,MODES=('neutral','chase','jump','front-dodge'),
             random=random,time=time,sha=sha,write=write,Policy=Policy)
    tree=ast.parse((V12/'bakeoff.py').read_text(encoding='utf-8'))
    names={'decode','loss_parts','metrics','load_match','teacher_loss','train','degradation'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'verified_v12_readonly_functions','exec'),env)
    return reader,env,pm

def configuration():
    c=read(V12/'experiment.json')
    c.update(experiment_version='v13_13d_gru_bootstrap_20261005_v1',
             models={'B':{'type':'GRU','input':13,'projection':64,'hidden':64,'layers':1,'activation':'ReLU','bidirectional':False}},
             test_policy='No test evaluation; train fitting and validation selection/final diagnostics only',
             final_evaluation=['teacher_forced_13d','student_forced_13d_no_action_feedback'],
             student_forced_scope='Previous predicted action tracked but never fed to 13D input; recorded physics/prediction/timing unchanged',
             deployment=False,shadow_gate_controller='Original teacher only; student and shadow actions logged only')
    return c

def selftest(reader,env,pm):
    assert env['load_match'].__globals__['json'] is json
    torch.set_num_threads(1);torch.manual_seed(42)
    x=np.arange(90,dtype=np.float32).reshape(5,18)
    y=projection.project(x,pm['source_feature_list'])
    assert y.shape==(5,13) and np.array_equal(y,x[:,:13]) and not np.shares_memory(y,x)
    for invalid in (x.astype(np.float64),np.zeros((2,13),np.float32)):
        try:projection.project(invalid,pm['source_feature_list'])
        except ValueError:pass
        else:raise AssertionError('Projection accepted incompatible contract')
    model=Policy().eval()
    with torch.inference_mode():
        logits,s,h=model(torch.from_numpy(y)); a,b,carry=model(torch.from_numpy(y[:2]));d,e,_=model(torch.from_numpy(y[2:]),carry)
    assert logits.shape==(5,4) and s.shape==(5,) and h.shape==(1,1,64)
    assert torch.isfinite(logits).all() and torch.isfinite(s).all() and torch.isfinite(h).all()
    assert torch.allclose(torch.cat((a,d)),logits,atol=1e-6,rtol=1e-6)
    assert sum(p.numel() for p in model.parameters())==26181
    print('V13 PHASE2 PREFLIGHT PASS: exact 13D projection, finite tensor shapes, GRU continuity; no fitting/live launch.',flush=True)

def evaluate(model,match,condition,env,c):
    n=len(match['mode']); logits=np.empty((n,4),np.float32);steering=np.empty(n,np.float32);h=None;last_action=None
    input_hash=hashlib.sha256()
    with torch.inference_mode():
        width=c['chunk_callbacks'] if condition=='teacher_forced_13d' else 1
        for start in range(0,n,width):
            stop=min(start+width,n)
            # No previous-action inputs exist. Both conditions see identical bits.
            x=match['obs'][start:stop];input_hash.update(x.tobytes())
            out,s,h=model(torch.from_numpy(x),h)
            logits[start:stop]=out.numpy();steering[start:stop]=s.numpy()
            if width==1:last_action=env['decode'](out.argmax(1).numpy(),s.numpy())[0]
            if width==1 and start and start%10000==0:print('V13 callback evaluation',match['metadata']['match_id'],start,'/',n,flush=True)
    pred=dict(logits=logits,steering=steering,modes=logits.argmax(1))
    pred['native']=env['decode'](pred['modes'],steering)
    return pred,dict(match_id=match['metadata']['match_id'],input_sha256=input_hash.hexdigest(),last_predicted_action_tracked_only=None if last_action is None else last_action.tolist())

def run(evaluate_existing=False):
    reader,env,pm=prepared();c=configuration();env['config']=lambda:c
    selftest(reader,env,pm)
    if GATE.exists() and not evaluate_existing:raise RuntimeError('Phase2 run already exists; preserve it and stop for review')
    original_files={}
    if evaluate_existing:
        if not GATE.exists():raise RuntimeError('No existing bootstrap to evaluate')
        original_files={str(p):sha(p) for p in GATE.iterdir() if p.is_file()}
        saved_hashes=read(GATE/'code_hashes.json')
        for p,h in saved_hashes.items():
            if Path(p).resolve()!=Path(__file__).resolve() and sha(Path(p))!=h:
                raise RuntimeError('Original dependency/projection changed: '+p)
        if read(GATE/'experiment.json')!=c:raise RuntimeError('Original experiment choices changed')
        selection=read(GATE/'selection.json')
        if sha(Path(selection['checkpoint']))!=selection['checkpoint_sha256']:
            raise RuntimeError('Selected checkpoint hash mismatch')
        output=GATE/'evaluation_recovery_v1'
        if output.exists():raise RuntimeError('Recovery output exists; preserve it and stop for review')
    else:output=GATE
    manifest=reader.verified_manifest(True) # Test file bytes hashed for integrity only, never evaluated.
    output.mkdir(exist_ok=False)
    code=[Path(__file__).resolve(),HERE/'projection.py',PROJECTION,V12/'bakeoff.py',V12/'experiment.json',HERE.parent/'v11/audit.py']
    hashes={str(p):sha(p) for p in code}
    write(output/'experiment.json',c);write(output/'code_hashes.json',hashes)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    write(output/'runtime.json',dict(torch=torch.__version__,numpy=np.__version__,python=sys.version,device=str(device),
                                projection_manifest_sha256=sha(PROJECTION),source_manifest_sha256=PIN))
    if evaluate_existing:
        write(output/'recovery_provenance.json',dict(reason='Missing json in extracted helper namespace; dependency added, fitting unchanged',
            original_runner_sha256=saved_hashes[str(Path(__file__).resolve())],recovery_runner_sha256=sha(Path(__file__)),
            original_files_sha256=original_files,checkpoint_sha256=selection['checkpoint_sha256'],retrained=False))
    try:
        if not evaluate_existing:
            data={}
            for split in ('train','validation'):
                data[split]=[]
                for m in manifest['matches']:
                    if m['split']!=split:continue
                    x=env['load_match'](m,diagnostics=False)
                    x['obs']=projection.project(x['obs'],manifest['features'])
                    data[split].append(x)
            selection=env['train']('B',data['train'],data['validation'],GATE,c,device)
            write(GATE/'selection.json',selection)
            del data
        model=Policy().eval();saved=torch.load(selection['checkpoint'],map_location='cpu',weights_only=True)
        assert saved['experiment']==c and saved['dataset_manifest_sha256']==PIN
        model.load_state_dict(saved['state_dict'])
        # Only after selecting checkpoint, read private validation diagnostics for metric subsets.
        validation=[]
        for m in manifest['matches']:
            if m['split']!='validation':continue
            x=env['load_match'](m,diagnostics=True);x['obs']=projection.project(x['obs'],manifest['features']);validation.append(x)
        results={};outputs={}
        target={k:np.concatenate([x[k] for x in validation]) for k in ('mode','native','phase','boundary','missing','around_missing','long_gap')}
        for condition in c['final_evaluation']:
            predictions=[];evidence=[]
            for match in validation:
                print('V13 validation',condition,match['metadata']['match_id'],flush=True)
                pred,identity=evaluate(model,match,condition,env,c);predictions.append(pred);evidence.append(identity)
            pred={k:np.concatenate([x[k] for x in predictions]) for k in ('logits','steering','modes','native')}
            outputs[condition]=pred
            subsets={name:env['metrics'](target,pred,target['mode']==i) for i,name in enumerate(env['MODES'])}
            for key in ('boundary','missing','around_missing','long_gap'):subsets[key]=env['metrics'](target,pred,target[key])
            results[condition]=dict(overall=env['metrics'](target,pred),by_mode_and_diagnostics=subsets,
                                   by_phase={name:env['metrics'](target,pred,target['phase']==i) for i,name in enumerate(('Jump','Release','Front dodge','Coast'))},
                                   input_evidence=evidence)
        a,b=c['final_evaluation'];equivalence={}
        assert [e['input_sha256'] for e in results[a]['input_evidence']]==[e['input_sha256'] for e in results[b]['input_evidence']]
        equivalence['input_bits_identical']=True
        equivalence['max_logit_difference']=float(np.max(np.abs(outputs[a]['logits']-outputs[b]['logits'])))
        equivalence['max_steering_difference']=float(np.max(np.abs(outputs[a]['steering']-outputs[b]['steering'])))
        equivalence['mode_disagreements']=int((outputs[a]['modes']!=outputs[b]['modes']).sum())
        reader.verified_manifest(True)
        assert all(sha(Path(p))==h for p,h in hashes.items()),'Code/projection changed during bootstrap'
        assert all(sha(Path(p))==h for p,h in original_files.items()),'Original bootstrap artifacts changed during recovery'
        report=dict(status='phase2_bootstrap_complete_awaiting_review',experiment=c,projection_manifest=pm,
                    selection=selection,validation=results,condition_equivalence=equivalence,
                    limitation='Removing previous-action inputs makes both conditions identical by construction. Differences from chunked versus stepwise floating-point evaluation are numerical only; similarity is not closed-loop robustness.',
                    private_diagnostics='Validation metric subsets only; excluded from fitting/selection and 13D inputs',
                    inputs_verified_before_and_after=True,test_evaluated=False,live_launched=False,
                    teacher_controls_phase3=True,dagger_collected=False,phase3_started=False,
                    evaluation_only_recovery=evaluate_existing,original_bootstrap_artifacts_preserved=True)
        write(output/'report.json',report)
        lines=['# V13 13D GRU bootstrap','', 'Status: **phase2_bootstrap_complete_awaiting_review**.', '',
               f"Checkpoint: `{selection['checkpoint']}`; SHA256 `{selection['checkpoint_sha256']}`; best epoch {selection['best_epoch']}; 26,181 parameters.", '',
               'Train/validation only; no test evaluation. Exact retained 13 features, identical scaling/masks/dt. Mode/continuous steering head and V12 training family/settings preserved. No private state inputs, balancing, resampling or new normalization.', '',
               '| Validation condition | Loss | Mode accuracy | Steer MAE | Jump precision | Jump recall |','|---|---:|---:|---:|---:|---:|']
        for key,value in results.items():
            v=value['overall'];lines.append(f"| {key} | {v['total_loss']:.6f} | {v['mode_accuracy']:.6f} | {v['steer_mae']:.6f} | {v['jump']['precision']} | {v['jump']['recall']} |")
        lines+=['', '## Interpretation','',report['limitation'],'',f"Numerical comparison: `{json.dumps(equivalence)}`.",'',
                'The callback-by-callback pass tracks its own predicted action but does not feed it into any feature. Physics, prediction and timing remain recorded. No real-game robustness or learner-induced state coverage is established.', '',
                'Full per-channel/action/phase/boundary/missing-ball/long-gap evidence is in report.json. Private labels are retrospective metric groupings only.', '',
                'Original V10/V12/source contracts unchanged; source hashes verified before/after. No live launch or DAgger collection. Phase3 requires this gate review and remains original-teacher-controlled.']
        (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print('V13 PHASE2 COMPLETE:',output/'report.md','No live launch or DAgger collection.',flush=True)
    except BaseException as e:
        write(output/'failure.json',dict(error=repr(e),status='incomplete_preserve_for_review'))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--run',action='store_true')
    group.add_argument('--evaluate-existing',action='store_true');args=parser.parse_args()
    if args.check:
        reader,env,pm=prepared();selftest(reader,env,pm)
    else:run(evaluate_existing=args.evaluate_existing)
