"""One approved offline V14 v2 experiment. No test/live/DAgger entry points.

--check: numerical preflight only, no dataset sample reads or fitting.
--baseline: validation-only evaluation of existing V13; no fitting.
--train: one fresh seeded v2 run, gated by the frozen baseline; never resumes.
"""
import argparse
import ast
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import boundary

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VERSION = ROOT/'training/behavior_cloning/datasets/v10_live_1v1_18d_20261005_v1'
CONFIG = HERE/'proposal_v2.json'
TRAIN = ('pilot_01','v10_001','v10_004','v10_006')
VAL = ('pilot_02','v10_002')
BASE = HERE/'baseline_v2'
RUN = HERE/'run_v2'
CHANNELS = ('throttle','steer','pitch','yaw','roll','jump','boost','handbrake')
FEATURES = ('ball_forward','ball_right','ball_up','prediction_forward','prediction_right','prediction_up',
            'ball_distance','car_speed','ball_present','prediction_valid','prediction_horizon','prediction_first_offset',
            'callback_dt','previous_neutral','previous_chase','previous_jump','previous_dodge','previous_steer')
SPECS = {'observations':('float32',18,4),'analog_controls':('float32',5,4),'buttons':('uint8',3,1),
         'mode':('uint8',1,1),'callback':('uint64',1,8),'frame':('uint64',1,8),'time':('float64',1,8),'dt':('float64',1,8)}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path=Path(path)
    if not path.resolve().is_relative_to(HERE.resolve()):raise RuntimeError('Output outside V14')
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def require(ok,reason):
    if not ok:raise RuntimeError(reason)


def pins():
    seal=read(HERE/'implementation_seal_v2.json')
    for path,h in seal['files'].items():require(sha(ROOT/path)==h,'Implementation/config changed: '+path)
    c=read(CONFIG)
    require(c['version']=='v14_boundary_temporal_objective_proposal_v2' and c['objective']['lambda']==boundary.LAMBDA==.1,
            'Wrong approved experiment')
    for path,h in c['source_hashes'].items():require(sha(ROOT/path)==h,'Pinned source changed: '+path)
    protected=read(ROOT/'training/live_reference_test/source_hashes.json')
    for path,h in protected.items():require(sha(ROOT/path)==h,'Protected source changed: '+path)
    require(len(protected)==43,'Protected list changed')
    return c,seal


def helpers():
    """Extract only pinned definitions. Never import old launchers/readers."""
    env=dict(np=np,torch=torch,nn=nn,F=F,CHANNELS=CHANNELS)
    sources=[('training/behavior_cloning/v13_dagger/bootstrap.py',{'Policy'}),
             ('training/behavior_cloning/v12/bakeoff.py',{'decode','loss_parts','metrics','teacher_loss'})]
    for file,names in sources:
        tree=ast.parse((ROOT/file).read_text(encoding='utf-8'))
        nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        require(len(nodes)==len(names),'Pinned helper definitions missing')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),file,'exec'),env)
    env['config']=lambda:dict(boundary_radius_callbacks=5,long_gap_threshold_seconds=.1)
    return env


def authorized_manifest(ids,c):
    """Hash all selected artifacts BEFORE any selected dataset content reads."""
    require(set(ids)<=set(TRAIN+VAL),'Test/unknown match prohibited')
    manifest=read(VERSION/'manifest.json')  # pins() verified SHA before this read.
    require(manifest['features']==list(FEATURES) and manifest['native_channels']==list(CHANNELS),'Frozen ordering mismatch')
    require(manifest['analog_order']==list(CHANNELS[:5]) and manifest['button_order']==list(CHANNELS[5:]),'Native transport order changed')
    for split,expected in (('train',TRAIN),('validation',VAL)):
        require(tuple(m['match_id'] for m in manifest['matches'] if m['split']==split)==expected,'Frozen match order mismatch')
    by_id={m['match_id']:m for m in manifest['matches']}
    selected=[by_id[i] for i in ids]
    allowed={}
    for m in selected:
        require(m['split']==('train' if m['match_id'] in TRAIN else 'validation'),'Frozen split changed')
        require(set(m['tensors'])==set(SPECS),'Tensor keys changed')
        for key,(dtype,width,size) in SPECS.items():
            t=m['tensors'][key];shape=[m['callbacks'],width] if width!=1 else [m['callbacks']]
            require(t['dtype']==dtype and t['shape']==shape and t['bytes']==m['callbacks']*width*size,'Tensor contract changed')
            path=(ROOT/m['dataset_directory']/'processed'/t['file']).resolve()
            require(path.is_relative_to(ROOT.resolve()),'Tensor escapes workspace')
            rel=path.relative_to(ROOT.resolve()).as_posix()
            require(rel in manifest['artifacts'],'Tensor not manifest listed')
            e=manifest['artifacts'][rel]
            require(e['sha256']==t['sha256'],'Manifest tensor hash disagreement')
            allowed[rel]=e
        if m['split']=='validation':
            rel=(Path(m['dataset_directory'])/'raw/trajectory.jsonl').as_posix()
            require(rel in manifest['artifacts'],'Validation diagnostics absent from manifest')
            allowed[rel]=manifest['artifacts'][rel]
    for i,(rel,e) in enumerate(allowed.items(),1):
        p=(ROOT/rel).resolve()
        require(p.is_relative_to(ROOT.resolve()) and e['path']==rel,'Artifact path escapes manifest')
        require(p.is_file() and p.stat().st_size==e['bytes'] and sha(p)==e['sha256'],'Artifact integrity failed: '+rel)
        if i%10==0:print('V14 AUTHORIZED HASH',i,'/',len(allowed),flush=True)
    print('V14 selected artifacts verified before content reads; no test access.',flush=True)
    return selected,allowed


def load_match(m,allowed,env,diagnostics=False):
    require(m['match_id'] in TRAIN+VAL and (not diagnostics or m['match_id'] in VAL),'Unauthorized sample/diagnostic read')
    dtype={'float32':'<f4','float64':'<f8','uint8':'u1','uint64':'<u8'};a={}
    for key,t in m['tensors'].items():
        rel=(Path(m['dataset_directory'])/'processed'/t['file']).as_posix()
        require(rel in allowed,'Unverified tensor read')
        a[key]=np.fromfile(ROOT/rel,dtype=dtype[t['dtype']]).reshape(t['shape'])
    native=np.concatenate((a['analog_controls'],a['buttons'].astype(np.float32)),axis=1)
    mode=a['mode'].astype(np.int64);obs=a['observations']
    require(np.isfinite(obs).all() and np.isfinite(native).all(),'Nonfinite frozen inputs')
    require(np.all((mode>=0)&(mode<=3)),'Unsupported teacher modes')
    require(np.array_equal(native,env['decode'](mode,native[:,1])),'Native decoder/target mismatch')
    require(np.array_equal(a['callback'],np.arange(1,len(mode)+1,dtype=np.uint64)),'Callback order changed')
    require(np.all((obs[:,8]==0)|(obs[:,8]==1)),'Invalid ball mask')
    require(np.array_equal(obs[:,12],(a['dt']/(1/60)).astype(np.float32)),'Frozen dt input changed')
    require(np.all(obs[obs[:,8]==0,:7]==0) and np.all(obs[obs[:,8]==0,8:12]==0),'Missing-ball masking changed')
    eligible=boundary.eligibility(a['time'],a['dt'],obs[:,8])
    x=dict(metadata=m,obs=obs[:,:13].copy(),mode=mode,native=native,dt=a['dt'],time=a['time'],eligible=eligible)
    # Copy preserves exact column bits and never adds diagnostic labels to inputs.
    require(x['obs'].dtype==np.float32 and np.array_equal(x['obs'],obs[:,:13]),'13D projection changed')
    if diagnostics:
        n=len(mode);phase=np.full(n,-1,np.int64);b=np.zeros(n,bool);missing=np.zeros(n,bool);previous=None
        rel=(Path(m['dataset_directory'])/'raw/trajectory.jsonl').as_posix()
        require(rel in allowed,'Unverified diagnostic read')
        i=-1
        with (ROOT/rel).open(encoding='utf-8') as f:
            for i,line in enumerate(f):
                r=json.loads(line)
                require(i<n and r['callback']==i+1 and r['action_mode']==mode[i] and r['T']==x['time'][i],'Raw/tensor diagnostic mismatch')
                require(r['match_id']==m['match_id'],'Cross-match raw row')
                consumed=r['teacher_selection']['consumed_phase'];phase[i]=-1 if consumed is None else consumed
                before,after=r['sequence_before'],r['sequence_after']
                complete=after is not None and after['done'] and (before is None or not before['done'])
                b[i]=bool(r['teacher_selection']['flip_trigger'] or complete or consumed!=previous)
                missing[i]=not r['ball_present'];previous=consumed
        require(i+1==n and np.array_equal(missing,obs[:,8]==0),'Raw diagnostic count/mask mismatch')
        def expand(mask):
            out=mask.copy()
            for shift in range(1,6):out[shift:]|=mask[:-shift];out[:-shift]|=mask[shift:]
            return out
        x.update(phase=phase,boundary=expand(b),missing=missing,around_missing=expand(missing),long_gap=x['dt']>.1)
    return x


def predict(model,m,env):
    model.eval();h=None;outputs=[];steers=[]
    with torch.inference_mode():
        for start in range(0,len(m['mode']),512):
            z,s,h=model(torch.from_numpy(m['obs'][start:start+512]),h)
            require(torch.isfinite(z).all() and torch.isfinite(s).all() and torch.isfinite(h).all(),'Nonfinite evaluation')
            outputs.append(z.numpy());steers.append(s.numpy())
    z=np.concatenate(outputs);s=np.concatenate(steers);modes=z.argmax(1)
    return dict(logits=z,steering=s,modes=modes,native=env['decode'](modes,s))


def evaluate(model,data,env,folder):
    predictions=[];temporal=[];per_match=[];boundary_records=[]
    for m in data:
        print('V14 VALIDATION',m['metadata']['match_id'],flush=True)
        p=predict(model,m,env);predictions.append(p);temporal.append(boundary.temporal(m,p['modes']))
        per_match.append(dict(match_id=m['metadata']['match_id'],metrics=env['metrics'](m,p)))
        np.savez(folder/(m['metadata']['match_id']+'_predictions.npz'),**p)
        cache=None
        with torch.inference_mode():
            for start in range(0,len(m['mode']),512):
                end=start+512
                bl,cache,stats=boundary.loss(torch.from_numpy(p['logits'][start:end]),torch.from_numpy(m['mode'][start:end]),
                                            torch.from_numpy(m['eligible'][start:end]),cache)
                boundary_records.append(dict(match_id=m['metadata']['match_id'],start=start,callbacks=len(m['mode'][start:end]),
                                              boundary_loss=float(bl),**stats))
    target={k:np.concatenate([m[k] for m in data]) for k in ('mode','native','phase','boundary','missing','around_missing','long_gap')}
    pred={k:np.concatenate([p[k] for p in predictions]) for k in ('logits','steering','modes','native')}
    subsets={name:env['metrics'](target,pred,target['mode']==i) for i,name in enumerate(('neutral','chase','jump','front-dodge'))}
    for k in ('boundary','missing','around_missing','long_gap'):subsets[k]=env['metrics'](target,pred,target[k])
    subsets['jump_or_dodge']=env['metrics'](target,pred,(target['mode']==2)|(target['mode']==3))
    phases={name:env['metrics'](target,pred,target['phase']==i) for i,name in enumerate(('Jump','Release','Front dodge','Coast'))}
    t,v=target['mode']==2,pred['modes']==2;tp=int((t&v).sum());fp=int((~t&v).sum());fn=int((t&~v).sum())
    return dict(overall=env['metrics'](target,pred),by_mode_and_diagnostics=subsets,by_phase=phases,
                pure_jump=dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None),
                per_match=per_match,temporal=boundary.temporal_summary(temporal),
                boundary_loss_diagnostics=dict(callback_weighted_chunk_mean=sum(x['boundary_loss']*x['callbacks'] for x in boundary_records)/len(target['mode']),
                                               chunks=boundary_records,checkpoint_selection_uses_this=False),
                private_labels_scope='Retrospective metric subsets only, excluded from objective/inputs',
                context_scope='No previous-action feedback exists in 13D; fixed recorded physics, not live robustness')


def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)


def initial_hash(model):
    h=hashlib.sha256()
    for name,v in model.state_dict().items():h.update(name.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def begin(folder,c,seal,allowed):
    folder.mkdir(exist_ok=False)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    write(folder/'metadata.json',dict(started_at=datetime.now(timezone(timedelta(hours=5,minutes=30))).isoformat(),
          python=sys.version,torch=torch.__version__,numpy=np.__version__,device=device,
          cuda_device=torch.cuda.get_device_name(0) if device=='cuda' else None,
          cudnn_benchmark=False,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
          approved_proposal=c,implementation_seal=seal,authorized_artifacts=allowed,
          test_access=False,live=False,dagger=False,repeatability='Same seeded V13 setup; GPU nondeterminism remains possible'))
    return torch.device(device)


def finish(folder,c,seal,allowed):
    pins()
    for rel,e in allowed.items():require(sha(ROOT/rel)==e['sha256'],'Input changed during run')
    files={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='integrity.json'}
    write(folder/'integrity.json',dict(files=files,protected_hashes_unchanged=43,authorized_inputs_verified_after=True,
                                     implementation_seal=seal,test_access=False))


def publish_final():
    # Only the two explicitly requested reports may be published outside V14.
    for ext in ('md','json'):
        destination=ROOT/f'training/reports/v14_temporal_objective_20261005.{ext}'
        destination.write_bytes((RUN/f'report.{ext}').read_bytes())


def baseline():
    c,seal=pins();env=helpers();selected,allowed=authorized_manifest(VAL,c)
    begin(BASE,c,seal,allowed)
    try:
        model=env['Policy']().eval()
        checkpoint=ROOT/'training/behavior_cloning/v13_dagger/phase2_bootstrap_v1/B_best.pt'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(saved['state_dict'])
        data=[load_match(m,allowed,env,True) for m in selected]
        result=evaluate(model,data,env,BASE)
        old=read(ROOT/'training/behavior_cloning/v13_dagger/phase2_bootstrap_v1/evaluation_recovery_v1/report.json')
        old=old['validation']['teacher_forced_13d']['overall']
        for key in ('mode_accuracy','steer_mae','total_loss','exact_native_action_agreement'):
            require(abs(result['overall'][key]-old[key])<=1e-6,'Saved V13 baseline not reproduced: '+key)
        temporal=result['temporal'];ready=temporal['eligible']>=20 and temporal['score'] is not None and temporal['score']<=.85
        result.update(status='baseline_ready_before_fitting' if ready else 'baseline_requires_review_no_fitting',
                      checkpoint_sha256=sha(checkpoint),saved_v13_aggregate=old,training_performed=False)
        write(BASE/'report.json',result);finish(BASE,c,seal,allowed)
        print('V14 BASELINE:',result['status'],'eligible cores',temporal['eligible'],'score',temporal['score'],flush=True)
    except BaseException as e:
        write(BASE/'failure.json',dict(error=repr(e),status='incomplete_preserve_for_review'));raise


def baseline_gate(seal):
    integrity=read(BASE/'integrity.json')
    require(integrity['implementation_seal']==seal,'Baseline implementation differs; stop for review')
    for name,h in integrity['files'].items():require(sha(BASE/name)==h,'Baseline artifact changed')
    b=read(BASE/'report.json')
    require(b['status']=='baseline_ready_before_fitting','Baseline gate not ready; stop before fitting')
    return b


def train():
    c,seal=pins();base=baseline_gate(seal);env=helpers()
    selected,allowed=authorized_manifest(TRAIN+VAL,c)
    device=begin(RUN,c,seal,allowed)
    try:
        train_data=[load_match(m,allowed,env) for m in selected if m['split']=='train']
        val_data=[load_match(m,allowed,env) for m in selected if m['split']=='validation']
        require(sum(len(m['mode']) for m in train_data)==107400 and sum(len(m['mode']) for m in val_data)==47271,'Split count mismatch')
        seed();model=env['Policy']().to(device);init_hash=initial_hash(model)
        require(sum(p.numel() for p in model.parameters())==26181,'Parameter count changed')
        write(RUN/'initialization.json',dict(seed=42,state_sha256=init_hash,source='fresh exact V13 Policy, not pretrained'))
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False)
        best=float('inf');bad=0;history=[];best_epoch=None;begun=time.monotonic()
        for epoch in range(1,21):
            model.train();totals=np.zeros(4,np.float64);chunk_records=[];last_progress=time.monotonic()
            support=dict(eligible_pairs=0,cross_chunk_pairs=0,empty_eligible_chunk=0,clamp_count=0,positive=[0]*4,negative=[0]*4)
            for m in train_data:
                h=None;cache=None
                for start in range(0,len(m['mode']),512):
                    end=start+512;optimizer.zero_grad(set_to_none=True)
                    x=torch.as_tensor(m['obs'][start:end],device=device);y=torch.as_tensor(m['mode'][start:end],device=device)
                    target=torch.as_tensor(m['native'][start:end,1],device=device)
                    z,s,h=model(x,h)
                    ce,se=env['loss_parts'](z,s,y,target)
                    bl,cache,stats=boundary.loss(z,y,torch.as_tensor(m['eligible'][start:end],device=device),cache)
                    loss=(ce+se).mean()+boundary.LAMBDA*bl
                    require(torch.isfinite(ce).all() and torch.isfinite(se).all() and torch.isfinite(h).all() and torch.isfinite(loss),'Nonfinite training component')
                    loss.backward()
                    require(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),'Nonfinite gradient; no repair')
                    optimizer.step();h=h.detach()
                    require(all(torch.isfinite(p).all() for p in model.parameters()),'Nonfinite parameters')
                    totals+=np.array([ce.detach().sum().item(),se.detach().sum().item(),bl.detach().item()*len(x),len(x)])
                    for key in ('eligible_pairs','cross_chunk_pairs','empty_eligible_chunk','clamp_count'):support[key]+=stats[key]
                    for key in ('positive','negative'):support[key]=[a+b for a,b in zip(support[key],stats[key])]
                    chunk_records.append(dict(match_id=m['metadata']['match_id'],start=start,callbacks=len(x),
                                              original_loss=float((ce+se).mean().detach()),boundary_loss=float(bl.detach()),**stats))
                    if time.monotonic()-last_progress>20:
                        print('V14 v2 training progress epoch',epoch,'callbacks',int(totals[3]),'/107400',flush=True)
                        last_progress=time.monotonic()
            validation=env['teacher_loss'](model,val_data,device,512)
            require(all(np.isfinite(v) for v in validation.values()),'Nonfinite validation loss')
            item=dict(epoch=epoch,original_train_loss=float((totals[0]+totals[1])/totals[3]),
                      boundary_loss_callback_weighted_chunk_mean=float(totals[2]/totals[3]),
                      objective_train_loss=float((totals[0]+totals[1]+.1*totals[2])/totals[3]),support=support,validation=validation,
                      chunks=chunk_records)
            history.append(item)
            if validation['total_loss']<best:
                best=validation['total_loss'];best_epoch=epoch;bad=0
                torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},kind='B',epoch=epoch,
                                validation_loss=best,experiment=c,dataset_manifest_sha256=c['source_hashes'][str(VERSION.relative_to(ROOT)/'manifest.json').replace('\\','/')]),RUN/'B_best.pt')
            else:bad+=1
            write(RUN/'history.json',dict(history=history,best_epoch=best_epoch,best_loss=best))
            print(f'V14 v2 epoch {epoch}/20 original={item["original_train_loss"]:.6f} boundary={item["boundary_loss_callback_weighted_chunk_mean"]:.6f} val={validation["total_loss"]:.6f} best_epoch={best_epoch} elapsed={time.monotonic()-begun:.1f}s',flush=True)
            if bad>=4:break
        selection=dict(checkpoint=str(RUN/'B_best.pt'),checkpoint_sha256=sha(RUN/'B_best.pt'),best_epoch=best_epoch,validation_loss=best)
        write(RUN/'selection.json',selection)
        model=env['Policy']().eval();model.load_state_dict(torch.load(RUN/'B_best.pt',map_location='cpu',weights_only=True)['state_dict'])
        diagnostics=[load_match(m,allowed,env,True) for m in selected if m['split']=='validation']
        results=evaluate(model,diagnostics,env,RUN)
        t=results['temporal'];bt=base['temporal'];g=c['regression_guards'];chase=results['by_mode_and_diagnostics']['chase']
        require(t['eligible']==bt['eligible'] and [m['eligible'] for m in t['by_match']]==[m['eligible'] for m in bt['by_match']],
                'Temporal reference denominator changed; no success verdict')
        checks=dict(temporal_score=t['score']>=max(.25,bt['score']+.15),extra_core_rate=t['extra_cores_per_minute']<=bt['extra_cores_per_minute']+.5,
                    chase_accuracy=chase['mode_accuracy']>=g['chase_accuracy_min'],chase_steering=chase['steer_mae']<=g['chase_steer_mae_max'],
                    overall_accuracy=results['overall']['mode_accuracy']>=g['overall_accuracy_min'])
        temporal_pass=checks['temporal_score'] and checks['extra_core_rate'];guards=checks['chase_accuracy'] and checks['chase_steering'] and checks['overall_accuracy']
        verdict='offline_success_with_scope' if temporal_pass and guards else 'partial_temporal_improvement_regression' if temporal_pass else 'unsuccessful_for_temporal_purpose'
        recommendation='Seek approval for one small live validation' if temporal_pass and guards else 'Review one further scoped offline experiment' if temporal_pass else 'Investigate representation; do not change it automatically'
        report=dict(status=verdict,checks=checks,recommendation=recommendation,selection=selection,validation=results,
                    baseline=base,proposal=c,training_history=history,test_access=False,live=False,dagger=False,
                    limitation='Recorded teacher-controlled physics only; no closed-loop or DAgger benefit claim')
        write(RUN/'report.json',report)
        metric_rows=[]
        for name,bv,nv in [
            ('Original total loss',base['overall']['total_loss'],results['overall']['total_loss']),
            ('Overall mode accuracy',base['overall']['mode_accuracy'],results['overall']['mode_accuracy']),
            ('Chase accuracy',base['by_mode_and_diagnostics']['chase']['mode_accuracy'],chase['mode_accuracy']),
            ('Native Jump precision',base['overall']['jump']['precision'],results['overall']['jump']['precision']),
            ('Native Jump recall',base['overall']['jump']['recall'],results['overall']['jump']['recall']),
            ('Front-dodge precision',base['overall']['front_dodge']['precision'],results['overall']['front_dodge']['precision']),
            ('Front-dodge recall',base['overall']['front_dodge']['recall'],results['overall']['front_dodge']['recall']),
            ('Pure Jump-mode accuracy',base['by_mode_and_diagnostics']['jump']['mode_accuracy'],results['by_mode_and_diagnostics']['jump']['mode_accuracy']),
            ('Boundary mode accuracy',base['by_mode_and_diagnostics']['boundary']['mode_accuracy'],results['by_mode_and_diagnostics']['boundary']['mode_accuracy']),
            ('Steer MAE, all callbacks',base['overall']['steer_mae'],results['overall']['steer_mae']),
            ('Steer MAE, Chase',base['by_mode_and_diagnostics']['chase']['steer_mae'],chase['steer_mae']),
            ('Exact native agreement',base['overall']['exact_native_action_agreement'],results['overall']['exact_native_action_agreement']),
            ('Timed-core score',bt['score'],t['score'])]:
            metric_rows.append(f'| {name} | {bv} | {nv} |')
        lines=['# V14 v2 temporal objective results','',f'Status: **{verdict}**.','',
               'Original V13 loss + 0.1 × approved adjacent-boundary BCE; one fresh seed-42 GRU. No v1/test/live/DAgger.',
               '', '## Baseline and validation comparison','', '| Metric | Pinned V13 | V14 v2 |','|---|---:|---:|',*metric_rows,
               '', '## Objective and training','',
               'The exact formula, edge construction, numerical/masking rules and lambda justification are retained in proposal_v2.md/json. Same 26,181-parameter model, input projection, native decoder, optimizer, seed and schedule. Selection uses original unweighted validation loss, not the boundary or temporal score.',
               f'Selected checkpoint: `{selection["checkpoint"]}`; SHA256 `{selection["checkpoint_sha256"]}`.',
               '', '## Temporal sequence and regression gates','',f'Checks: `{json.dumps(checks)}`.',
               f'Timed cores: V13 {bt["successful"]}/{bt["eligible"]}; V14 {t["successful"]}/{t["eligible"]}. Extra cores/minute: V13 {bt["extra_cores_per_minute"]}; V14 {t["extra_cores_per_minute"]}.',
               '', 'Complete per-mode/channel, temporal edge/duration, missing-ball/gap and regression evidence is in report.json.',
               '', '## Limitations','',report['limitation'],
               'No previous-action feedback exists in 13D. Local edge supervision does not ensure global/physical maneuver validity. Censored/interrupted cores are reported separately. Match-disjoint validation retains opponent overlap; checkpoint selection may reject temporally better epochs. One seed does not establish statistical significance.',
               '', '## One recommendation','',f'{recommendation}. Stop for review.']
        (RUN/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        for ext in ('md','json'):
            prior=ROOT/f'training/reports/v14_temporal_objective_20261005.{ext}'
            if prior.exists():(RUN/f'pre_run_design_report.{ext}').write_bytes(prior.read_bytes())
        baseline_gate(seal)  # Existing baseline must also remain unchanged.
        finish(RUN,c,seal,allowed)
        publish_final()
        print('V14 v2 COMPLETE:',RUN/'report.md','STOP. No deployment/live/DAgger/PPO.',flush=True)
    except BaseException as e:
        write(RUN/'failure.json',dict(error=repr(e),status='incomplete_preserve_for_review'));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true');group.add_argument('--baseline',action='store_true');group.add_argument('--train',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
    if args.check:
        from preflight import check
        check()
    elif args.baseline:baseline()
    else:train()
