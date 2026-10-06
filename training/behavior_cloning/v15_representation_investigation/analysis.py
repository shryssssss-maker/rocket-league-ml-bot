"""V15 read-only observability investigation. No policy/model/checkpoint use.

--check numerical/source checks only; --run user-run bounded diagnostic.
Only train/validation artifacts and explicitly pinned existing forensic queries.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
VERSION=ROOT/'training/behavior_cloning/datasets/v10_live_1v1_18d_20261005_v1'
PIN='080943fd8f9a81ada800e0a2f9ff3593b67f66d32272211d67030c03cf433a83'
TRAIN=('pilot_01','v10_001','v10_004','v10_006');VAL=('pilot_02','v10_002')
FEATURES=('ball_forward','ball_right','ball_up','prediction_forward','prediction_right','prediction_up','ball_distance',
          'car_speed','ball_present','prediction_valid','prediction_horizon','prediction_first_offset','callback_dt',
          'previous_neutral','previous_chase','previous_jump','previous_dodge','previous_steer')
SPECS={'observations':('float32',18,4),'analog_controls':('float32',5,4),'buttons':('uint8',3,1),'mode':('uint8',1,1),
       'callback':('uint64',1,8),'frame':('uint64',1,8),'time':('float64',1,8),'dt':('float64',1,8)}
HISTORY=(1,2,4,8,16);KS=(1,8,32,128);MODES=('Neutral','Chase','Jump','Front dodge')
SESSION=ROOT/'training/behavior_cloning/v13_dagger/phase4_full_match_v1/sessions/20261005_071223_978853'
ANCHORS=(456,4279,9815,19942,579,4380,8300,13200,13500,16500,20250,10335)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def require(ok,why):
    if not ok:raise RuntimeError(why)
def write(p,v):
    p=Path(p);require(p.resolve().is_relative_to(HERE.resolve()) or p.parent.resolve()==(ROOT/'training/reports').resolve(),'Output path outside V15')
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def verify_sources():
    seal=read(HERE/'source_seal.json')
    for p,h in seal['files'].items():require(sha(ROOT/p)==h,'V15/pinned source changed: '+p)
    for p,h in read(ROOT/'training/live_reference_test/source_hashes.json').items():require(sha(ROOT/p)==h,'Protected file changed: '+p)
    old=read(ROOT/'training/behavior_cloning/v14_temporal_objective/run_v2/integrity.json')
    for p,h in old['files'].items():require(sha(ROOT/'training/behavior_cloning/v14_temporal_objective/run_v2'/p)==h,'V14 artifact changed: '+p)
    return seal


def extracted_reader():
    """Use verified V14 tensor contract/loader, but never its launcher or fitter."""
    import types
    boundary_env={'np':np}
    src=ROOT/'training/behavior_cloning/v14_temporal_objective/boundary.py'
    node=next(n for n in ast.parse(src.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='eligibility')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),boundary_env)
    decoder_env={'np':np}
    src=ROOT/'training/behavior_cloning/v12/bakeoff.py'
    node=next(n for n in ast.parse(src.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='decode')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),decoder_env)
    env=dict(ROOT=ROOT,VERSION=VERSION,TRAIN=TRAIN,VAL=VAL,FEATURES=FEATURES,SPECS=SPECS,
             CHANNELS=('throttle','steer','pitch','yaw','roll','jump','boost','handbrake'),np=np,
             sha=sha,read=read,require=require,json=json,Path=Path,
             boundary=types.SimpleNamespace(eligibility=boundary_env['eligibility']))
    src=ROOT/'training/behavior_cloning/v14_temporal_objective/experiment.py'
    nodes=[n for n in ast.parse(src.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in ('authorized_manifest','load_match')]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),env)
    return env,decoder_env


def plan():
    require(sha(VERSION/'manifest.json')==PIN,'Manifest hash mismatch')
    r,decoder=extracted_reader();selected,allowed=r['authorized_manifest'](TRAIN+VAL,{})
    manifest=read(VERSION/'manifest.json')
    for m in selected:
        key=(Path(m['dataset_directory'])/'raw/trajectory.jsonl').as_posix()
        require(key in manifest['artifacts'],'Raw artifact absent')
        e=manifest['artifacts'][key];p=ROOT/key
        require(p.is_file() and p.stat().st_size==e['bytes'] and sha(p)==e['sha256'],'Raw hash mismatch: '+key)
        allowed[key]=e;print('V15 RAW HASH',m['match_id'],flush=True)
    # Forensic inputs are quarantined queries, never a training reference library.
    pin=read(SESSION/'diagnostic_integrity.json')
    diag={}
    for name in ('callbacks.jsonl','metadata.json'):
        require(name in pin['files'],'Forensic artifact not sealed')
        require(sha(SESSION/name)==pin['files'][name],'Forensic hash mismatch: '+name)
        diag[str((SESSION/name).relative_to(ROOT)).replace('\\','/')]=pin['files'][name]
    print('V15 all permitted artifacts verified before sample reads.',flush=True)
    return selected,allowed,r,decoder,diag


def load(m,allowed,reader,decoder):
    x=reader['load_match'](m,allowed,decoder,m['split']=='validation')
    n=len(x['mode']);a=np.empty((n,1));b=np.empty((n,3));e=np.empty((n,3));d=np.empty((n,5));phase=[]
    i=-1;previous=None;car_fields=set()
    with (ROOT/m['dataset_directory']/'raw/trajectory.jsonl').open(encoding='utf-8') as f:
        for i,line in enumerate(f):
            row=json.loads(line);car=row['car'];car_fields.update(car)
            require(i<n and row['callback']==i+1 and row['T']==x['time'][i] and row['action_mode']==x['mode'][i],'Raw ordering/time/label mismatch')
            require(row['match_id']==m['match_id'],'Raw cross-match row')
            require(np.array_equal(np.asarray(row['observation18'][:13],np.float32),x['obs'][i]),'Raw 13D mismatch')
            pitch,yaw,roll=car['rotation_pyr'];a[i,0]=math.cos(pitch)*math.cos(roll)
            b[i]=np.asarray(car['angular_velocity'],np.float64)/5.5;e[i]=np.asarray(car['position'],np.float64)/6000
            prior=row['previous_submission'];d[i]=0
            if prior is None:
                require(i==0,'Missing noninitial previous submission');d[i,0]=1
            else:
                require(previous is not None and prior['callback']==i and prior['controls']==previous['teacher_controls'],
                        'Prior submitted teacher action not preceding callback')
                require(prior['T']==previous['T'] and prior['mode']==previous['action_mode'],'Prior submission identity mismatch')
                d[i,int(prior['mode'])]=1;d[i,4]=np.float32(prior['controls'][1])
            require(np.array_equal(np.asarray(row['observation18'][13:],np.float32),d[i].astype(np.float32)),'Group D not exact frozen prior features')
            phase.append(row['match_phase']);previous=row
            if i and i%10000==0:print('V15 RAW PROGRESS',m['match_id'],i,'/',n,flush=True)
    require(i+1==n,'Raw callback count mismatch')
    require(all(np.isfinite(g).all() for g in (a,b,d,e)),'Nonfinite candidate state')
    x.update(groups={k:g.astype(np.float32).astype(np.float64) for k,g in {'A':a,'B':b,'D':d,'E':e}.items()},
             phase=np.asarray(phase),car_fields=sorted(car_fields))
    return x


def masks(x):
    n=len(x['mode']);up=x['groups']['A'][:,0];pos=x['groups']['E']*6000
    neutral=np.zeros(n,bool)
    starts=np.r_[0,np.flatnonzero(x['mode'][1:]!=x['mode'][:-1])+1,n]
    for a,b in zip(starts[:-1],starts[1:]):
        if x['mode'][a]==0 and x['time'][b-1]-x['time'][a]>=2:
            neutral[a:b]=True
    return dict(representative=np.arange(n)%128==0,boundary=x.get('boundary',np.zeros(n,bool)),
        kickoff=x['phase']=='MatchPhase.Kickoff',inverted=up<-.5,tilted=(up>=-.5)&(up<=.8),
        long_neutral=neutral,corner_back=(np.abs(pos[:,1])>4000)|((np.abs(pos[:,0])>3000)&(np.abs(pos[:,1])>3000)),
        missing=x['obs'][:,8]==0)


def choose(mask,k):
    idx=np.flatnonzero(mask)
    return idx if len(idx)<=k else idx[np.linspace(0,len(idx)-1,k,dtype=int)]


def query_indices(x):
    ms=masks(x);indices=set(np.flatnonzero(ms['representative']).tolist())
    indices.update(np.flatnonzero((x['mode']==2)|(x['mode']==3)).tolist())
    indices.update(choose(ms['boundary'],256).tolist())
    for k in ('kickoff','inverted','tilted','long_neutral','corner_back','missing'):indices.update(choose(ms[k],64).tolist())
    return np.asarray(sorted(i for i in indices if i>=15),np.int64),ms


def history(x,h,ends):
    require(np.all(ends>=h-1) and np.all(ends<len(x['mode'])),'History crosses match boundary')
    index=ends[:,None]+np.arange(-h+1,1)[None,:]
    return x['obs'][index].reshape(len(ends),13*h).astype(np.float64)/math.sqrt(13*h)


def top(values,ids,k):
    """Deterministic exact tie inclusion by global training endpoint ID."""
    k=min(k,len(values));cut=np.partition(values,k-1)[k-1]
    below=np.flatnonzero(values<cut);equal=np.flatnonzero(values==cut)
    equal=equal[np.argsort(ids[equal],kind='stable')[:k-len(below)]]
    take=np.r_[below,equal];order=np.lexsort((ids[take],values[take]))
    return values[take][order],ids[take][order]


def neighbors(train,query,k=128):
    """Exhaustive double-precision block search, direct returned-distance check."""
    indices=np.empty((len(query),k),np.int64);dist=np.empty((len(query),k));begun=time.monotonic()
    tn=np.einsum('ij,ij->i',train,train);qn=np.einsum('ij,ij->i',query,query)
    for qs in range(0,len(query),32):
        q=query[qs:qs+32];bestv=[np.empty(0) for _ in q];besti=[np.empty(0,np.int64) for _ in q]
        for ts in range(0,len(train),8192):
            t=train[ts:ts+8192];v=qn[qs:qs+len(q),None]+tn[ts:ts+len(t)][None,:]-2*(q@t.T)
            v=np.maximum(v,0);ids=np.arange(ts,ts+len(t))
            for j in range(len(q)):bestv[j],besti[j]=top(np.r_[bestv[j],v[j]],np.r_[besti[j],ids],k)
        for j in range(len(q)):
            # Direct recomputation removes norm/dot cancellation from reported distances.
            direct=np.einsum('ij,ij->i',train[besti[j]]-q[j],train[besti[j]]-q[j])
            vv,ii=top(direct,besti[j],k);dist[qs+j]=np.sqrt(vv);indices[qs+j]=ii
        if qs%256==0:print('V15 SEARCH',qs,'/',len(query),'dimensions',train.shape[1],'elapsed',round(time.monotonic()-begun,1),flush=True)
    return indices,dist


def metrics(y,pred,subset):
    y=y[subset];pred=pred[subset]
    if not len(y):return dict(rows=0)
    def recall(mask,hit):return float(hit[mask].mean()) if np.any(mask) else None
    return dict(rows=len(y),mode_accuracy=float((y==pred).mean()),jump_recall=recall((y==2)|(y==3),(pred==2)|(pred==3)),
        pure_jump_accuracy=recall(y==2,pred==2),front_dodge_recall=recall(y==3,pred==3),chase_accuracy=recall(y==1,pred==1))


def summarize(y,tm,nn,dist,subsets,query_ids,train_native,query_native):
    result={};votes=tm[nn];pred=np.array([np.argmax(np.bincount(v[:32],minlength=4)) for v in votes])
    result['probe']={key:metrics(y,pred,mask) for key,mask in subsets.items()}
    result['probe']['targeted_union']=metrics(y,pred,np.ones(len(y),bool))
    steering=np.asarray([float(train_native[ids[:32][tm[ids[:32]]==1],1].mean())
                         if np.any(tm[ids[:32]]==1) else 0. for ids in nn])
    decoded_steer=np.where(pred==1,steering,0.)
    result['steering_probe']={key:dict(rows=int(mask.sum()),mae=float(np.abs(decoded_steer-query_native[:,1])[mask].mean()) if mask.any() else None,
        chase_mae=float(np.abs(decoded_steer-query_native[:,1])[mask&(y==1)].mean()) if np.any(mask&(y==1)) else None)
        for key,mask in subsets.items()}
    result['by_mode']={MODES[i]:metrics(y,pred,y==i) for i in range(4)}
    result['by_match']={key:{s:metrics(y,pred,mask&(query_ids==key)) for s,mask in subsets.items()}
                        for key in np.unique(query_ids)}
    result['neighbors']={}
    for k in KS:
        same=(votes[:,:k]==y[:,None]).mean(1)
        majority=np.array([np.bincount(v[:k],minlength=4).max()/k for v in votes])
        entropy=[]
        for v in votes:
            prob=np.bincount(v[:k],minlength=4)/k;prob=prob[prob>0]
            entropy.append(float(-(prob*np.log2(prob)).sum()))
        entry=dict(mean_true_label_probability=float(same.mean()),mean_majority_purity=float(majority.mean()),mean_label_entropy_bits=float(np.mean(entropy)),
                   by_mode={MODES[i]:dict(rows=int((y==i).sum()),mean_true_label_probability=float(same[y==i].mean()) if np.any(y==i) else None) for i in range(4)},
                   by_distance_bin={})
        for low,high in ((0,.005),(.005,.02),(.02,.05),(.05,.1),(.1,float('inf'))):
            mask=(dist[:,0]<=high)&(dist[:,0]>low if low else dist[:,0]>=0)
            entry['by_distance_bin'][f'{low}..{high}']=dict(rows=int(mask.sum()),mean_true_label_probability=float(same[mask].mean()) if mask.any() else None)
        result['neighbors'][str(k)]=entry
    result['nearest_distance']=dict(median=float(np.median(dist[:,0])),maximum=float(dist[:,0].max()))
    return result,pred


def forensic_queries():
    need=set()
    for anchor in ANCHORS:need.update(range(anchor-15,anchor+1))
    rows={}
    with (SESSION/'callbacks.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['callback'] in need:rows[r['callback']]=r
    require(set(rows)==need,'Forensic causal history callback missing')
    query=[]
    for anchor in ANCHORS:
        hist=[rows[i] for i in range(anchor-15,anchor+1)];r=hist[-1];car=r['car']['physics'];p,y,roll=car['rotation_pyr']
        prior=r['previous_submission'];d=np.zeros(5)
        if prior is None:d[0]=1
        else:
            mode=MODES.index(prior['mode']);d[mode]=1;d[4]=prior['controls'][1]
        query.append(dict(anchor=anchor,obs=np.asarray([x['observation13'] for x in hist],np.float64),
            teacher_mode=MODES.index(r['shadow_mode']),student_mode=MODES.index(r['student']['mode']),
            groups={k:g.astype(np.float32).astype(np.float64) for k,g in
                    {'A':np.array([math.cos(p)*math.cos(roll)]),'B':np.asarray(car['angular_velocity'])/5.5,
                     'D':d,'E':np.asarray(car['position'])/6000}.items()},
            native_car={k:v for k,v in r['car'].items() if k!='physics'},
            time=np.asarray([x['T'] for x in hist]),scope='Stateful shadow query label on student physics; not approved learning target'))
    return query


def check():
    verify_sources()
    # Numerical fixtures only, not synthesized demonstrations or probe fitting.
    t=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]]);q=np.array([[.2,.1]])
    ni,di=neighbors(t,q,4);expected=np.argsort(((t-q[0])**2).sum(1),kind='stable')
    require(np.array_equal(ni[0],expected),'Block neighbor search fails scalar oracle')
    require(np.allclose(di[0],np.linalg.norm(t[expected]-q[0],axis=1),atol=1e-12),'Distances fail scalar oracle')
    _,tie=top(np.array([0.,0.,0.]),np.array([3,1,2]),2);require(tie.tolist()==[1,2],'Tie rule changed')
    obs=np.arange(20*13).reshape(20,13);x={'obs':obs,'mode':np.zeros(20)}
    for h in HISTORY:
        got=history(x,h,np.array([15]));require(np.array_equal(got,obs[16-h:16].reshape(1,-1)/math.sqrt(13*h)),'Noncausal history')
    print('V15 SOURCE/NUMERICAL CHECK PASS. No dataset samples/probe results/policy fitting.',flush=True)


def run():
    verify_sources();output=HERE/'results_v1';output.mkdir(exist_ok=False)
    try:
        selected,allowed,reader,decoder,diag=plan()
        data={}
        for m in selected:
            print('V15 RAW/TENSOR READ',m['match_id'],flush=True);data[m['match_id']]=load(m,allowed,reader,decoder)
        training=[data[k] for k in TRAIN];validation=[data[k] for k in VAL]
        train_ends=[np.arange(15,len(x['mode'])) for x in training]
        val_ends=[];subset_masks={};qid=[];qy=[];spans=[]
        for x in validation:
            ends,ms=query_indices(x);val_ends.append(ends);qid.extend([x['metadata']['match_id']]*len(ends));qy.extend(x['mode'][ends])
            spans.extend((x['time'][ends]-x['time'][ends-15]).tolist())
            for key,mask in ms.items():subset_masks.setdefault(key,[]).extend(mask[ends].tolist())
        qid=np.asarray(qid);qy=np.asarray(qy);subset_masks={k:np.asarray(v,bool) for k,v in subset_masks.items()}
        tm=np.concatenate([x['mode'][ends] for x,ends in zip(training,train_ends)])
        tn=np.concatenate([x['native'][ends] for x,ends in zip(training,train_ends)])
        train_identity=[(x['metadata']['match_id'],int(i)+1) for x,ends in zip(training,train_ends) for i in ends]
        groups_t={k:np.concatenate([x['groups'][k][i] for x,i in zip(training,train_ends)]) for k in ('A','B','D','E')}
        groups_q={k:np.concatenate([x['groups'][k][i] for x,i in zip(validation,val_ends)]) for k in ('A','B','D','E')}
        qnative=np.concatenate([x['native'][i] for x,i in zip(validation,val_ends)])
        forensic=forensic_queries();experiments={};raw_neighbors={};e_gate=False
        for h in HISTORY:
            train=np.concatenate([history(x,h,i) for x,i in zip(training,train_ends)])
            query=np.concatenate([history(x,h,i) for x,i in zip(validation,val_ends)])
            fq=np.asarray([r['obs'][-h:].reshape(-1)/math.sqrt(13*h) for r in forensic])
            print('V15 CAUSAL HISTORY',h,'validation queries',len(query),flush=True)
            nn,dist=neighbors(train,np.concatenate((query,fq)))
            summary,pred=summarize(qy,tm,nn[:len(query)],dist[:len(query)],subset_masks,qid,tn,qnative)
            key=f'13D_H{h}';experiments[key]=summary
            np.savez(output/(key+'_neighbors.npz'),indices=nn,distances=dist,query_modes=qy,probe_modes=pred)
            raw_neighbors[key]=(nn[:len(query)],dist[:len(query)])
            examples=[]
            for j in range(len(query)):
                nearest=nn[j,0];ids=nn[j,:128];native=tn[ids]
                conflict=tm[ids]!=qy[j];material=conflict | np.any(native[:,[2,5]]!=qnative[j,[2,5]],axis=1) | ((tm[ids]==1)&(qy[j]==1)&(np.abs(native[:,1]-qnative[j,1])>.2))
                close=dist[j,:128]<=.02
                other=next((int(i) for i in ids if tm[i]!=qy[j]),None)
                examples.append(dict(query_index=j,query_match=str(qid[j]),query_mode=int(qy[j]),distance=float(dist[j,0]),
                    neighbor=train_identity[nearest],neighbor_mode=int(tm[nearest]),
                    mode_distribution_128=np.bincount(tm[ids],minlength=4).tolist(),
                    native_jump_fraction=float(native[:,5].mean()),native_pitch_mean=float(native[:,2].mean()),
                    native_steer_min=float(native[:,1].min()),native_steer_max=float(native[:,1].max()),
                    by_k={str(k):dict(mode_counts=np.bincount(tm[ids[:k]],minlength=4).tolist(),
                         jump_button_fraction=float(native[:k,5].mean()),pitch_values=dict(Counter(str(v) for v in native[:k,2])),
                         steer_mean=float(native[:k,1].mean()),steer_std=float(native[:k,1].std()),
                         material_action_disagreements=int(material[:k].sum())) for k in KS},
                    materially_different_neighbors_128=int(material.sum()),close_conflicting_neighbors=int((close&conflict).sum()),
                    distinct_neighbor_matches=len(set(train_identity[i][0] for i in ids)),
                    exact_nearest_feature_equality=bool(np.array_equal(query[j],train[nearest])),
                    query_up_z=float(groups_q['A'][j,0]),neighbor_up_z=float(groups_t['A'][nearest,0]),
                    position_difference_uu=float(np.linalg.norm(groups_q['E'][j]-groups_t['E'][nearest])*6000),
                    angular_difference_rad_s=float(np.linalg.norm(groups_q['B'][j]-groups_t['B'][nearest])*5.5),
                    conflicting_neighbor=None if other is None else dict(identity=train_identity[other],mode=int(tm[other]),
                        base_distance=float(np.linalg.norm(query[j]-train[other])),exact=bool(np.array_equal(query[j],train[other])),
                        up_z=float(groups_t['A'][other,0]),position=groups_t['E'][other].tolist())))
            write(output/(key+'_query_evidence.json'),examples)
            summary['close_conflict_queries']=sum(x['close_conflicting_neighbors']>0 for x in examples)
            summary['exact_conflicting_queries']=sum(x['conflicting_neighbor'] is not None and x['conflicting_neighbor']['exact'] for x in examples)
            if h==1:
                position_support=0
                for j in range(len(query)):
                    if not any(subset_masks[s][j] for s in ('kickoff','inverted','tilted','corner_back')):continue
                    ids=nn[j];close=dist[j]<=.02
                    position_support+=int(np.any(close&(tm[ids]!=qy[j])&(np.linalg.norm(groups_t['E'][ids]-groups_q['E'][j],axis=1)*6000>=1000)))
                e_gate=position_support>=20;summary['E_position_support_queries']=position_support
            summary['forensic_queries']=[]
            for j,r in enumerate(forensic):
                ids=nn[len(query)+j];counts=np.bincount(tm[ids[:32]],minlength=4)
                summary['forensic_queries'].append(dict(anchor=r['anchor'],shadow_mode=r['teacher_mode'],student_mode=r['student_mode'],
                    history_seconds=float(r['time'][-1]-r['time'][-h]),nearest=train_identity[ids[0]],distance=float(dist[len(query)+j,0]),
                    mode_distribution_32=counts.tolist(),probe_mode=int(counts.argmax()),native_car=r['native_car'],scope=r['scope']))
            if h in (1,16):
                for group in ('A','B','D','E'):
                    if group=='E' and not e_gate:continue
                    print('V15 INDIVIDUAL GROUP',group,'history',h,flush=True)
                    gtrain=np.c_[train,groups_t[group]/math.sqrt(groups_t[group].shape[1])]
                    gquery=np.c_[query,groups_q[group]/math.sqrt(groups_q[group].shape[1])]
                    gf=np.c_[fq,np.asarray([r['groups'][group] for r in forensic])/math.sqrt(groups_t[group].shape[1])]
                    gn,gd=neighbors(gtrain,np.concatenate((gquery,gf)))
                    gs,gpred=summarize(qy,tm,gn[:len(query)],gd[:len(query)],subset_masks,qid,tn,qnative)
                    baseline_nn=nn[:len(query)];baseline_dist=dist[:len(query)]
                    ambiguous=np.any((baseline_dist<=.02)&(tm[baseline_nn]!=qy[:,None]),axis=1)&(subset_masks['boundary']|subset_masks['inverted']|subset_masks['tilted']|subset_masks['kickoff'])
                    oldprob=(tm[baseline_nn[:,:32]]==qy[:,None]).mean(1);newprob=(tm[gn[:len(query),:32]]==qy[:,None]).mean(1)
                    gs['close_ambiguous_subset']=dict(rows=int(ambiguous.sum()),true_label_probability_gain=float((newprob-oldprob)[ambiguous].mean()) if ambiguous.any() else None)
                    gs['forensic_queries']=[dict(anchor=r['anchor'],shadow_mode=r['teacher_mode'],
                        probe_mode=int(np.bincount(tm[gn[len(query)+j,:32]],minlength=4).argmax()),distance=float(gd[len(query)+j,0]),
                        base13_distance_to_nearest=float(np.linalg.norm(fq[j]-train[gn[len(query)+j,0]])),scope=r['scope']) for j,r in enumerate(forensic)]
                    experiments[f'13D_{group}_H{h}']=gs
                    np.savez(output/f'13D_{group}_H{h}_neighbors.npz',indices=gn,distances=gd,probe_modes=gpred)
        # Preregistered comparison, not iterative feature/model tuning.
        def gains(new,old):
            nb,ob=new['probe']['boundary'],old['probe']['boundary']
            checks={}
            for key,threshold in (('mode_accuracy',.10),('pure_jump_accuracy',.05),('front_dodge_recall',.05)):
                aa=nb.get(key);bb=ob.get(key);checks[key]=aa is not None and bb is not None and aa-bb>=threshold
            aa=new['probe']['representative'].get('mode_accuracy');bb=old['probe']['representative'].get('mode_accuracy')
            checks['overall_guard']=aa is not None and bb is not None and aa>=bb-.01
            return checks
        histchecks=gains(experiments['13D_H16'],experiments['13D_H1']);history_help=all(histchecks.values())
        candidates={};qualified=[]
        for group in ('A','B','E'):
            key=f'13D_{group}_H16'
            if key not in experiments:continue
            g=experiments[key];checks=gains(g,experiments['13D_H16'])
            close=g['close_ambiguous_subset'];checks['close_support']=close['rows']>=20
            checks['close_gain']=close['true_label_probability_gain'] is not None and close['true_label_probability_gain']>=.10
            checks['boundary_support']=g['probe']['boundary']['rows']>=30
            per=[]
            for match in VAL:
                aa=g['by_match'][match]['boundary'];bb=experiments['13D_H16']['by_match'][match]['boundary']
                per.append(all(aa.get(k) is not None and bb.get(k) is not None and aa[k]>bb[k] for k in ('mode_accuracy','pure_jump_accuracy','front_dodge_recall')))
            checks['both_validation_matches']=all(per);candidates[group]=checks
            if all(checks.values()):qualified.append(group)
        classification='C' if history_help and qualified else 'B' if qualified else 'A' if history_help else 'D'
        winner=next((g for g in ('A','B','E') if g in qualified),None)
        recommendation={'A':'Investigate temporal/modeling limitations without changing official13D',
            'B':f'Propose one new candidate contract: original13D plus Group {winner}, pending review; do not train',
            'C':f'Propose one scoped causal-context plus Group {winner} investigation, pending review; do not train',
            'D':'Perform one focused native-maneuver-state observability audit; current evidence does not justify a feature/policy change'}[classification]
        for rel,e in allowed.items():require(sha(ROOT/rel)==e['sha256'],'Authorized dataset artifact changed')
        for rel,h in diag.items():require(sha(ROOT/rel)==h,'Forensic input changed')
        seal=verify_sources()
        report=dict(status='investigation_complete_awaiting_review',classification=classification,recommendation=recommendation,
                    qualified_groups=qualified,history_checks=histchecks,candidate_checks=candidates,experiments=experiments,
                    E_probe_justified=e_gate,C_probe='unavailable in V10, not inferred',
                    group_definitions={'A':'up.z=cos(pitch)*cos(roll), appended scalar','B':'native world angular XYZ /5.5, appended in XYZ order',
                                       'C':'native flags/air_state absent from frozen raw; no probe','D':'previous submitted mode one-hot then steer; teacher-forced diagnostic only',
                                       'E':'world XYZ/6000, appended XYZ order; conditional evidence gate'},
                    source_audit=(HERE/'protocol.md').read_text(encoding='utf-8'),
                    native_sdk_audit=read(HERE/'native_sdk_audit.json'),
                    recorded_car_fields={k:data[k]['car_fields'] for k in TRAIN+VAL},
                    runtime=dict(numpy=np.__version__,search='exhaustive float64 CPU NumPy block distance; no policy/checkpoint loaded'),
                    input13=list(FEATURES[:13]),normalization='unchanged frozen column bits; protocol.md exact scales',
                    data_ids={'train':list(TRAIN),'validation':list(VAL),'test_access':False},
                    validation_query_count=len(qy),query_ids=qid.tolist(),query_modes=qy.tolist(),
                    query_callbacks=np.concatenate([i+1 for i in val_ends]).tolist(),
                    causal_history_seconds_16={'median':float(np.median(spans)),'maximum':float(np.max(spans))},
                    excluded_first_callbacks_per_match=15,queries_not_population_estimates=True,
                    artifacts_read={**{rel:e['sha256'] for rel,e in allowed.items()},**diag},source_seal=seal,
                    all_protected_hashes_unchanged=43,frozen_artifacts_unchanged=True,
                    no_policy_fitting=True,no_test=True,no_live=True,no_dagger=True,no_contract_change=True,
                    limitations=['Finite histories do not exhaust GRU memory or prove observability/insufficiency',
                        'H16 is shorter than the complete original maneuver; representation-versus-horizon classification is exploratory',
                        'Original teacher has no explicit recovery policy; predicting its labels is not optimal-recovery evidence',
                        'Neighbor scale and correlated callbacks affect purity; no significance or causal claim',
                        'Diagnostic query subset, not full validation population; match/opponent overlap remains',
                        'Group D is prior submitted teacher controls, not packet last_input or student-feedback robustness',
                        'Native maneuver flags/boost not recorded in V10; Group C cannot be compared',
                        'Shadow query labels on learner physics have unqualified maneuver prerequisites',
                        'Region proximity does not prove wall contact; no synchronized video',
                        'Double-precision norm/dot search has rounding; returned distances recomputed directly'])
        write(output/'report.json',report)
        stem=ROOT/'training/reports/v15_representation_investigation_20261006'
        write(stem.with_suffix('.json'),report)
        lines=['# V15 representation investigation — 2026-10-06','', '## 1. Executive conclusion','',
               f'Classification **{classification}**. {recommendation}. No policy or official contract changed.',
               '', '## 2. Exact 13D contract','', 'Unchanged original columns 0–12: '+', '.join(FEATURES[:13])+'.',
               'Car-native forward/right/up geometry; no team inversion. XYZ/distance /6000, speed /2300, masks 0/1, horizon /2 s, first offset /(1/120 s), dt /(1/60 s). Exact selector int((elapsed+2-first_slice_time)*120); no interpolation/clamping. Missing-ball handling unchanged.',
               '', '## 3. Native field availability audit','', 'See protocol.md Phase 0 table and exact installed SDK/source locations (also embedded in JSON source_audit). V10 records physics and prior submissions; native maneuver flags/packet last_input/boost are absent. No reconstruction is invented.',
               '', '## 4. 13D ambiguity analysis','', '| Condition | Query mode accuracy | Pure Jump accuracy | Dodge recall | Boundary accuracy |','|---|---:|---:|---:|---:|']
        for key,r in experiments.items():
            o=r['probe']['targeted_union'];b=r['probe']['boundary']
            lines.append(f'| {key} | {o.get("mode_accuracy")} | {o.get("pure_jump_accuracy")} | {o.get("front_dodge_recall")} | {b.get("mode_accuracy")} |')
        lines+=['', 'Probe is fixed 32-neighbor voting, not a fitted/deployed policy. Overall representative-grid metrics, distance bins, k=1/8/32/128 label distributions and physical examples are in JSON/diagnostic files.',
                '', '## 5. Temporal-history analysis','',f'Fixed H=1,2,4,8,16 causal histories, common endpoints; checks: `{json.dumps(histchecks)}`. Context spans use actual clocks; no future callbacks or match crossing.',
                '', '## 6. Candidate feature groups','',f'A/B/D tested individually; E evidence gate={e_gate}; C unavailable. No combinations or Group F. Candidate checks: `{json.dumps(candidates)}`.',
                '', '## 7. Feature sufficiency probe results','', 'Full per-mode/native jump/dodge and boundary/subset metrics are in report.json. Group D gains describe teacher-forced history association only. No validation-driven k/scale/architecture tuning.',
                '', '## 8. Kickoff-specific findings','', '| Condition | Validation kickoff rows | Mode accuracy | Pure Jump recall | Dodge recall | V13 kickoff anchor agreements |','|---|---:|---:|---:|---:|---:|']
        for key,r in experiments.items():
            k=r['probe']['kickoff'];f=[x for x in r['forensic_queries'] if x['anchor'] in ANCHORS[:4]]
            lines.append(f'| {key} | {k["rows"]} | {k.get("mode_accuracy")} | {k.get("pure_jump_accuracy")} | {k.get("front_dodge_recall")} | {sum(x["probe_mode"]==x["shadow_mode"] for x in f)}/4 |')
        lines+=['','Shadow anchor labels are diagnostic on learner-induced physics, not approved training corrections. Passing this probe is not closed-loop maneuver success.',
                '', '## 9. Recovery/inversion-specific findings','', '| Condition | Inverted rows / accuracy | Tilted rows / accuracy | Region rows / accuracy | Recovery anchor agreements |','|---|---:|---:|---:|---:|']
        for key,r in experiments.items():
            i=r['probe']['inverted'];t=r['probe']['tilted'];region=r['probe']['corner_back'];f=[x for x in r['forensic_queries'] if x['anchor'] in ANCHORS[4:-1]]
            lines.append(f'| {key} | {i["rows"]} / {i.get("mode_accuracy")} | {t["rows"]} / {t.get("mode_accuracy")} | {region["rows"]} / {region.get("mode_accuracy")} | {sum(x["probe_mode"]==x["shadow_mode"] for x in f)}/{len(f)} |')
        lines+=['', 'Native world-up alignment, inverted/tilted subsets and query-anchor physical metadata distinguish geometry from contact/maneuver prerequisites. Long-Neutral subset in teacher demonstrations is Neutral-label context, not evidence that this teacher was stuck. Known V13 long-pause windows are shadow-label queries only. Field proximity is not wall collision.',
                '', '## 10. Representation versus temporal classification','',f'{classification}; finite observational criteria, not proof. Native C data unavailability limits this classification.',
                '', '## 11. One recommended next experiment','',recommendation,
                '', '## 12. Experiments NOT authorized','', 'No full policy training, test evaluation, live game, DAgger, deployment, official observation/architecture/teacher change.',
                '', '## 13. Integrity/hash results','', 'All authorized inputs hash-verified before/after; 43 protected hashes unchanged. Frozen V10 and V13/V14 inputs remain unchanged.',
                '', '## 14. Exact artifacts read','', 'report.json artifacts_read/source_seal lists paths and SHA256s. Native source-only audit preceded all sample analysis.',
                '', '## 15. Limitations','']+[f'- {l}' for l in report['limitations']]+['','STOP for review.']
        md='\n'.join(lines)+'\n';(output/'report.md').write_text(md,encoding='utf-8');stem.with_suffix('.md').write_text(md,encoding='utf-8')
        write(output/'integrity.json',dict(files={p.name:sha(p) for p in output.iterdir() if p.is_file()},report_sha256=sha(stem.with_suffix('.md')),
                                          json_sha256=sha(stem.with_suffix('.json')),protected=43))
        print('V15 INVESTIGATION COMPLETE:',stem.with_suffix('.md'),'Classification',classification,'STOP. No policy training/live/test/DAgger.',flush=True)
    except BaseException as ex:
        write(output/'failure.json',dict(error=repr(ex),status='incomplete_preserve_for_review'));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--check',action='store_true');g.add_argument('--run',action='store_true')
    args=p.parse_args()
    if args.check:check()
    else:run()
