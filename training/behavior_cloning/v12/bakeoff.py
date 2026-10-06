"""Approved V12 offline BC; immutable input, no RLBot/RocketSim/deployment.

--check runs numerical checks only. --run verifies artifacts before fitting.
Every run is exclusively created; partial runs are preserved, never resumed.
"""
import argparse
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CONFIG=HERE/'experiment.json'
CONFIG_PIN='58ce29ff9a7005262c8449bcc0f0ed7696b0c80fa3722eb4737a356d23f4ed74'
REPORT=ROOT/'training/reports/v12_bc_architecture_bakeoff_20261005'
TEST_GATE=HERE/'test_suite_started.json'
spec=importlib.util.spec_from_file_location('v12_verified_reader',HERE.parent/'v11/audit.py')
reader=importlib.util.module_from_spec(spec); spec.loader.exec_module(reader)
CHANNELS=reader.CHANNELS
MODES=('neutral','chase','jump','front-dodge')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def config():
    if sha(CONFIG)!=CONFIG_PIN: raise RuntimeError('Frozen V12 experiment choices changed; stop for review')
    c=json.loads(CONFIG.read_text(encoding='utf-8'))
    if not c['user_approved'] or c['manifest_sha256']!=reader.PIN: raise RuntimeError('Approved experiment/dataset pin mismatch')
    return c

class Policy(nn.Module):
    def __init__(self,kind):
        super().__init__(); self.kind=kind
        if kind=='A': self.encoder=nn.Sequential(nn.Linear(18,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU())
        elif kind=='B':
            self.encoder=nn.Sequential(nn.Linear(18,64),nn.ReLU())
            self.gru=nn.GRU(64,64,num_layers=1,batch_first=True,bidirectional=False)
        else: raise ValueError(kind)
        self.head=nn.Linear(64,5)

    def forward(self,x,h=None):
        z=self.encoder(x)
        if self.kind=='B':
            z,h=self.gru(z.unsqueeze(0),h); z=z.squeeze(0)
        out=self.head(z)
        return out[:,:4],torch.tanh(out[:,4]),h

def loss_parts(logits,steer,mode,target_steer):
    ce=F.cross_entropy(logits,mode,reduction='none')
    steering=(steer-target_steer).square()*(mode==1)
    return ce,steering

def decode(mode,steer):
    a=np.zeros((len(mode),8),dtype=np.float32)
    chase=mode==1; dodge=mode==3
    a[chase,0]=1; a[chase,1]=steer[chase]
    a[dodge,2]=-1; a[:,5]=np.logical_or(mode==2,dodge)
    return a

def selftest():
    torch.set_num_threads(1); torch.manual_seed(42)
    mode=np.array([0,1,2,3]); steering=np.array([.7,-.372,.4,.6],dtype=np.float32)
    expected=np.array([[0,0,0,0,0,0,0,0],[1,-.372,0,0,0,0,0,0],[0,0,0,0,0,1,0,0],[0,0,-1,0,0,1,0,0]],dtype=np.float32)
    assert np.array_equal(decode(mode,steering),expected)
    # Numerical probes only, not fabricated trajectories or fitting examples.
    x=torch.zeros(12,18); x[:,13]=1; x[:,12]=1
    with torch.inference_mode():
        for kind in ('A','B'):
            model=Policy(kind).eval(); logits,steer,h=model(x)
            assert logits.shape==(12,4) and steer.shape==(12,) and torch.isfinite(logits).all()
            if kind=='B':
                a,s,carry=model(x[:5]); b,t,_=model(x[5:],carry)
                assert torch.allclose(torch.cat((a,b)),logits,atol=1e-6,rtol=1e-6)
                altered=x.clone(); altered[7:]=10
                future,_,_=model(altered)
                assert torch.equal(future[:7],logits[:7])
    assert sha(reader.VERSION/'manifest.json')==reader.PIN
    target=dict(mode=mode.astype(np.int64),native=expected)
    predicted=dict(modes=mode,steering=steering,logits=np.eye(4,dtype=np.float32)*20,native=expected.copy())
    check=metrics(target,predicted)
    assert check['exact_native_action_agreement']==1. and check['steer_mae']==0.
    assert check['jump']['precision']==1. and check['jump']['recall']==1.
    assert metrics(target,predicted,np.zeros(4,bool))['metrics'] is None
    print('V12 PREFLIGHT PASS: native decode, finite shapes, causal GRU and carried-state chunk equivalence. No training.',flush=True)

def load_match(m,diagnostics=False):
    reader.tensor_contract(m); folder=ROOT/m['dataset_directory']; tensors={}
    dtype={'float32':'<f4','float64':'<f8','uint8':'u1','uint64':'<u8'}
    for key,t in m['tensors'].items():
        path=(folder/'processed'/t['file']).resolve()
        if path not in reader.ALLOWED: raise RuntimeError('Tensor not manifest-listed')
        tensors[key]=np.fromfile(path,dtype=dtype[t['dtype']]).reshape(t['shape'])
    native=np.concatenate((tensors['analog_controls'],tensors['buttons'].astype(np.float32)),axis=1)
    if np.any(native[:,[3,4,6,7]]!=0): raise RuntimeError('Unexpected teacher constant channel variation')
    result=dict(metadata=m,obs=tensors['observations'],native=native,mode=tensors['mode'].astype(np.int64),dt=tensors['dt'])
    if diagnostics:
        n=m['callbacks']; phase=np.full(n,-1,dtype=np.int64); boundary=np.zeros(n,bool); missing=np.zeros(n,bool)
        previous_phase=None
        path=(folder/'raw/trajectory.jsonl').resolve()
        if path not in reader.ALLOWED: raise RuntimeError('Raw diagnostics not manifest-listed')
        with path.open(encoding='utf-8') as f:
            for i,line in enumerate(f):
                r=json.loads(line)
                if r['callback']!=i+1 or r['action_mode']!=result['mode'][i]: raise RuntimeError('Diagnostic callback/target mismatch')
                consumed=r['teacher_selection']['consumed_phase']
                phase[i]=-1 if consumed is None else consumed
                before,after=r['sequence_before'],r['sequence_after']
                complete=after is not None and after['done'] and (before is None or not before['done'])
                boundary[i]=bool(r['teacher_selection']['flip_trigger'] or complete or consumed!=previous_phase)
                missing[i]=not r['ball_present']; previous_phase=consumed
        radius=config()['boundary_radius_callbacks']
        def expand(mask):
            out=mask.copy()
            for shift in range(1,radius+1):
                out[shift:]|=mask[:-shift]; out[:-shift]|=mask[shift:]
            return out
        result.update(phase=phase,boundary=expand(boundary),missing=missing,around_missing=expand(missing),
                      long_gap=result['dt']>config()['long_gap_threshold_seconds'])
    return result

def teacher_loss(model,data,device,chunk):
    model.eval(); sums=np.zeros(3,dtype=np.float64)
    with torch.inference_mode():
        for match in data:
            h=None
            for start in range(0,len(match['mode']),chunk):
                end=start+chunk
                x=torch.as_tensor(match['obs'][start:end],device=device)
                modes=torch.as_tensor(match['mode'][start:end],device=device)
                target=torch.as_tensor(match['native'][start:end,1],device=device)
                logits,steer,h=model(x,h); ce,se=loss_parts(logits,steer,modes,target)
                sums+=np.array([ce.sum().item(),se.sum().item(),len(x)])
    return dict(total_loss=float((sums[0]+sums[1])/sums[2]),mode_cross_entropy=float(sums[0]/sums[2]),steering_loss=float(sums[1]/sums[2]))

def train(kind,train_data,val_data,run,c,device):
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(c['seed'])
    model=Policy(kind).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'])
    best=float('inf'); best_epoch=None; bad=0; history=[]; checkpoint=run/(kind+'_best.pt'); begun=time.monotonic(); last_progress=begun
    for epoch in range(1,c['max_epochs']+1):
        model.train(); sums=np.zeros(3,dtype=np.float64); states=0
        for match in train_data:
            h=None; states+=1
            for start in range(0,len(match['mode']),c['chunk_callbacks']):
                end=start+c['chunk_callbacks']; optimizer.zero_grad(set_to_none=True)
                x=torch.as_tensor(match['obs'][start:end],device=device)
                modes=torch.as_tensor(match['mode'][start:end],device=device)
                target=torch.as_tensor(match['native'][start:end,1],device=device)
                logits,steer,h=model(x,h); ce,se=loss_parts(logits,steer,modes,target); loss=(ce+se).mean()
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                loss.backward(); optimizer.step()
                if h is not None: h=h.detach()  # No numerical reset, only truncate gradient graph.
                sums+=np.array([ce.detach().sum().item(),se.detach().sum().item(),len(x)])
                if time.monotonic()-last_progress>20:
                    print(f"V12 {kind} epoch {epoch} train progress {int(sums[2])}/107400 callbacks",flush=True)
                    last_progress=time.monotonic()
        validation=teacher_loss(model,val_data,device,c['chunk_callbacks'])
        item=dict(epoch=epoch,train_total_loss=float((sums[0]+sums[1])/sums[2]),validation=validation,callbacks=int(sums[2]),match_state_initializations=states)
        history.append(item)
        if validation['total_loss']<best:
            best=validation['total_loss']; best_epoch=epoch; bad=0
            torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},kind=kind,epoch=epoch,
                            validation_loss=best,experiment=c,dataset_manifest_sha256=reader.PIN),checkpoint)
        else: bad+=1
        write(run/(kind+'_training.json'),dict(history=history,best_epoch=best_epoch,best_validation_loss=best))
        print(f"V12 {kind} epoch {epoch}/{c['max_epochs']} train={item['train_total_loss']:.6f} val={validation['total_loss']:.6f} best_epoch={best_epoch} elapsed={time.monotonic()-begun:.1f}s",flush=True)
        if bad>=c['patience']: break
    return dict(kind=kind,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),best_epoch=best_epoch,
                parameter_count=sum(p.numel() for p in model.parameters()),training_seconds=time.monotonic()-begun,history=history,
                state_note='GRU hidden values carried between chronological chunks; gradients detached. Hidden values computed under preceding chunk weights are carried after optimizer updates; never reset at phases.')

def predict(model,match,condition,chunk):
    n=len(match['mode']); logits=np.empty((n,4),np.float32); steering=np.empty(n,np.float32); h=None
    with torch.inference_mode():
        if condition=='teacher_forced':
            for start in range(0,n,chunk):
                stop=min(start+chunk,n); out,s,h=model(torch.from_numpy(match['obs'][start:stop]),h)
                logits[start:stop]=out.numpy(); steering[start:stop]=s.numpy()
        else:
            last_mode=0; last_steer=0.
            for i in range(n):
                x=match['obs'][i:i+1].copy()
                x[0,13:17]=0.; x[0,13+last_mode]=1.; x[0,17]=last_steer
                out,s,h=model(torch.from_numpy(x),h)
                logits[i]=out[0].numpy(); steering[i]=s[0].item()
                last_mode=int(out.argmax(1).item()); last_steer=float(s[0].item()) if last_mode==1 else 0.
                if i and i%10000==0: print(f"V12 offline student context {match['metadata']['match_id']} {i}/{n}",flush=True)
    if not np.isfinite(logits).all() or not np.isfinite(steering).all(): raise RuntimeError('Invalid evaluation output')
    modes=logits.argmax(1)
    return dict(logits=logits,steering=steering,modes=modes,native=decode(modes,steering))

def metrics(target,pred,mask=None):
    mask=np.ones(len(target['mode']),bool) if mask is None else mask
    n=int(mask.sum())
    if not n: return dict(callbacks=0,metrics=None)
    y=target['native'][mask]; p=pred['native'][mask]; ym=target['mode'][mask]; pm=pred['modes'][mask]
    error=np.abs(p.astype(np.float64)-y.astype(np.float64))
    ce,se=loss_parts(torch.from_numpy(pred['logits'][mask]),torch.from_numpy(pred['steering'][mask]),torch.from_numpy(ym),torch.from_numpy(y[:,1]))
    def binary(t,v):
        tp=int((t&v).sum()); fp=int((~t&v).sum()); fn=int((t&~v).sum())
        return dict(accuracy=float((t==v).mean()),precision=tp/(tp+fp) if tp+fp else None,
                    recall=tp/(tp+fn) if tp+fn else None,tp=tp,fp=fp,fn=fn)
    confusion=np.zeros((4,4),np.int64)
    np.add.at(confusion,(ym,pm),1)
    return dict(callbacks=n,total_loss=float((ce+se).mean()),mode_cross_entropy=float(ce.mean()),steering_loss=float(se.mean()),
        per_channel_loss={k:float((error[:,j]**2).mean()) for j,k in enumerate(CHANNELS)},per_channel_loss_definition='native decoded channel MSE, diagnostic; optimization uses mode CE plus masked steering MSE',
        per_channel_mae={k:float(error[:,j].mean()) for j,k in enumerate(CHANNELS)},
        throttle_mae=float(error[:,0].mean()),steer_mae=float(error[:,1].mean()),
        pitch_accuracy=float((p[:,2]==y[:,2]).mean()),pitch_mae=float(error[:,2].mean()),
        jump=binary(y[:,5].astype(bool),p[:,5].astype(bool)),front_dodge=binary(ym==3,pm==3),
        mode_accuracy=float((ym==pm).mean()),mode_confusion_matrix=confusion.tolist(),
        exact_native_action_agreement=float((p==y).all(1).mean()),
        analog_1e6_buttons_exact_agreement=float(((error[:,:5]<=1e-6).all(1)&(p[:,5:]==y[:,5:]).all(1)).mean()))

def evaluate(model,data,condition,c):
    per_match=[]; targets=[]; predictions=[]
    for match in data:
        print('V12 EVAL',model.kind,condition,match['metadata']['match_id'],flush=True)
        p=predict(model,match,condition,c['chunk_callbacks']); targets.append(match); predictions.append(p)
        per_match.append(dict(match_id=match['metadata']['match_id'],metrics=metrics(match,p)))
    target={k:np.concatenate([x[k] for x in targets]) for k in ('mode','native','phase','boundary','missing','around_missing','long_gap')}
    prediction={k:np.concatenate([x[k] for x in predictions]) for k in ('logits','steering','modes','native')}
    subsets={name:metrics(target,prediction,target['mode']==i) for i,name in enumerate(MODES)}
    subsets['jump_or_dodge']=metrics(target,prediction,np.logical_or(target['mode']==2,target['mode']==3))
    for k in ('boundary','missing','around_missing','long_gap'): subsets[k]=metrics(target,prediction,target[k])
    phase={name:metrics(target,prediction,target['phase']==i) for i,name in enumerate(('Jump','Release','Front dodge','Coast'))}
    return dict(overall=metrics(target,prediction),by_action_mode_and_diagnostics=subsets,
                by_consumed_sequence_phase=phase,per_match=per_match,
                sequence_phase_scope='Metrics stratified by private diagnostic labels only. No phase output head; no claim to recover hidden phase identity from Neutral outputs.')

def degradation(tf,sf):
    out={}
    for key in ('total_loss','throttle_mae','steer_mae','pitch_accuracy','mode_accuracy','exact_native_action_agreement'):
        out[key]=sf['overall'][key]-tf['overall'][key]
    for key in ('accuracy','precision','recall'):
        a=tf['overall']['jump'][key]; b=sf['overall']['jump'][key]
        out['jump_'+key]=None if a is None or b is None else b-a
    return dict(student_minus_teacher_forced=out,interpretation='Positive loss/MAE is worse; negative accuracy/agreement/precision/recall is worse. Offline robustness only, not closed-loop gameplay.')

def run():
    c=config(); selftest()
    if TEST_GATE.exists(): raise RuntimeError('Final test suite was already started; stop for review, do not repeat test access')
    manifest=reader.verified_manifest(True)
    now=datetime.now(timezone(timedelta(hours=5,minutes=30)))
    folder=HERE/'runs'/(c['experiment_version']+'_'+now.strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True,exist_ok=False)
    if REPORT.with_suffix('.json').exists() or REPORT.with_suffix('.md').exists():
        raise RuntimeError('Final V12 report already exists; no new test evaluation without review')
    # Protect against modifying the approved choices after training starts.
    code_paths=[Path(__file__).resolve(),CONFIG,HERE.parent/'v11/audit.py']
    hashes={str(p):sha(p) for p in code_paths}
    write(folder/'experiment.json',c)
    torch.set_num_threads(1); torch.backends.cudnn.benchmark=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    write(folder/'run_metadata.json',dict(started_at=now.isoformat(),code_sha256=hashes,manifest_sha256=reader.PIN,
        python=sys.version,torch=torch.__version__,numpy=np.__version__,device=str(device),
        cuda_device=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        repeatability='Seeded one-run comparison; GPU kernels may have residual nondeterminism. No multi-seed confidence claim.'))
    try:
        fitting=[load_match(m) for m in manifest['matches'] if m['split']=='train']
        validation=[load_match(m) for m in manifest['matches'] if m['split']=='validation']
        selections={kind:train(kind,fitting,validation,folder,c,device) for kind in ('A','B')}
        write(folder/'selection_complete_before_test.json',selections)
        del fitting,validation
        results={}
        for kind in ('A','B'):
            model=Policy(kind)
            saved=torch.load(selections[kind]['checkpoint'],map_location='cpu',weights_only=True)
            model.load_state_dict(saved['state_dict']); model.eval()
            results[kind]={}
            for split in ('validation','test'):
                if split=='test' and kind=='A':
                    with TEST_GATE.open('x',encoding='utf-8') as gate:
                        json.dump(dict(run_path=str(folder),selected_checkpoint_sha256={k:v['checkpoint_sha256'] for k,v in selections.items()},
                            policy='Single final suite for the two already-selected models, both contexts; interrupted test requires review'),gate,indent=2)
                data=[load_match(m,diagnostics=True) for m in manifest['matches'] if m['split']==split]
                conditions={cond:evaluate(model,data,cond,c) for cond in c['final_evaluation']}
                conditions['degradation']=degradation(conditions['teacher_forced'],conditions['student_forced_offline'])
                results[kind][split]=conditions
        # Full post-run integrity sweep; any mismatch invalidates this experiment.
        reader.verified_manifest(True)
        if any(sha(Path(p))!=h for p,h in hashes.items()): raise RuntimeError('Experiment code/config changed during run')
        report=dict(status='training_and_offline_evaluation_complete_awaiting_review',experiment=c,
                    run_path=str(folder),manifest_sha256=reader.PIN,selection=selections,results=results,
                    input_integrity_before_and_after=True,teacher_constants={'yaw':0,'roll':0,'boost':False,'handbrake':False},
                    frozen_dataset_modified=False,teacher_modified=False,observations_modified=False,
                    no_rebalancing=True,test_used_for_tuning=False,deployed=False,ppo_started=False,
                    limitations=manifest['limitations']+[
                        'One seed and bounded fixed architecture/loss comparison, not a hyperparameter sweep or generalization guarantee.',
                        'Student-forced diagnostic uses recorded physics/prediction, not physics resulting from student controls; not closed-loop evaluation.',
                        'Final exact native agreement is with frozen float32 analog targets; raw double-precision controller values remain preserved in raw records.',
                        'Private sequence labels used only for retrospective metric subsets, never model inputs, fitting weights or selection.',
                        'Both modes use identical CPU evaluation. Training on GPU and CPU evaluation may have small floating-point differences.'])
        write(folder/'results.json',report); write(REPORT.with_suffix('.json'),report)
        lines=['# V12 Behavior Cloning architecture bakeoff','',f"Status: **{report['status']}**. Offline only; no deployment or PPO.",'',
               f"Frozen manifest: `{reader.PIN}`. Inputs verified before and after; unchanged.", '',
               '## Fixed experiment','',
               'A: 18→128→64 ReLU MLP. B: 18→64 ReLU→GRU(64), one causal layer. Both use four mode logits and continuous tanh steering, decoded in throttle/steer/pitch/yaw/roll/jump/boost/handbrake order.', '',
               'Unweighted mode cross-entropy plus Chase-only steering squared error averaged over all callbacks. Adam 0.001, seed 42, max 20 epochs, patience 4. Chronological 512-callback chunks; GRU values carried across chunks, gradients detached. Reset only at match boundaries. No balancing or new normalization.', '',
               'Checkpoints selected on teacher-forced validation loss. Both selections completed before test tensor/label loading. Test evaluated once as a final suite with both conditions; no tuning from test.', '',
               '## Overall comparison','', '| Model | Split | Context | Loss | Throttle MAE | Steer MAE | Jump precision | Jump recall | Exact native agreement |','|---|---|---|---:|---:|---:|---:|---:|---:|']
        for kind in results:
            for split in results[kind]:
                for cond in c['final_evaluation']:
                    v=results[kind][split][cond]['overall']; j=v['jump']
                    lines.append(f"| {kind} | {split} | {cond} | {v['total_loss']:.6f} | {v['throttle_mae']:.6f} | {v['steer_mae']:.6f} | {j['precision']} | {j['recall']} | {v['exact_native_action_agreement']:.6f} |")
        lines+=['', '## Checkpoint selection','']
        for kind,s in selections.items():
            lines += [f"- {kind}: best epoch {s['best_epoch']}; {s['parameter_count']} parameters; `{s['checkpoint']}`; SHA256 `{s['checkpoint_sha256']}`."]
        lines+=['', '## Detailed evidence','',
                'JSON records every requested per-channel loss/MAE, throttle/steer MAE, pitch accuracy/error, jump precision/recall, mode confusion, Front-dodge precision/recall, Jump/Dodge subsets, sequence-phase-stratified action agreement, ±5-callback sequence-boundary neighborhoods, missing-ball neighborhoods and callbacks after dt >100 ms. Empty subsets and undefined precision/recall are reported as null, not success.', '',
                'Exact native agreement uses no tolerance. A separate analog ≤1e-6/buttons-exact agreement is also reported. Private phase identity has no output head: phase-stratified action accuracy is not phase reconstruction.', '',
                'Teacher-forced validation preserves recorded prior-action features. Student-forced evaluation copies each observation and replaces only prior-mode one-hot and prior steering with the preceding decoded prediction; all recorded physics/prediction/dt remain unchanged. JSON explicitly reports student-minus-teacher-forced degradation separately for validation and test.', '',
                'Yaw=0, roll=0, boost=false and handbrake=false for every teacher target. The decoded heads enforce that same support; no variation or ability outside teacher support is claimed.', '',
                '## Limitations','']+['- '+x for x in report['limitations']]+['', 'Stop for review. No live deployment, teacher replacement or RL fine-tuning.']
        REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        write(folder/'completion.json',dict(status=report['status'],report=str(REPORT.with_suffix('.md'))))
        print('V12 COMPLETE:',REPORT.with_suffix('.md'),'Run:',folder,flush=True)
    except BaseException as e:
        write(folder/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),status='incomplete_preserved_no_resume'))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(); mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true'); mode.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.check: config(); reader.verified_manifest(False); selftest()
    else: run()
