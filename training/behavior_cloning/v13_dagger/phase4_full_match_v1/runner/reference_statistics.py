"""Read-only TRAIN reference descriptions. No model, validation/test rows or training."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
from common import HERE,ROOT,DATASET,MANIFEST_PIN,sha,read,write

REFERENCE=HERE.parent/'train_reference_v1'
TRAIN_IDS=('pilot_01','v10_001','v10_004','v10_006')
PRIMARY_IDS=('pilot_01','v10_006')
PROBS=[0.,.01,.05,.25,.5,.75,.95,.99,1.]

def plan():
    if sha(DATASET/'manifest.json')!=MANIFEST_PIN:raise RuntimeError('Frozen manifest mismatch')
    manifest=read(DATASET/'manifest.json')
    matches=[m for m in manifest['matches'] if m['split']=='train']
    if tuple(m['match_id'] for m in matches)!=TRAIN_IDS:raise RuntimeError('Training assignments changed')
    artifacts={}
    for m in matches:
        for key in ('raw/trajectory.jsonl','processed/observations.bin'):
            item=m['files'][key];p=(ROOT/item['path']).resolve()
            if not p.is_relative_to(ROOT.resolve()):raise ValueError('Artifact outside workspace')
            if not p.is_file() or p.stat().st_size!=item['bytes']:raise RuntimeError('Required training artifact absent/size mismatch')
            artifacts[str(p)]=item['sha256']
    return manifest,matches,artifacts

def distributions(x,phase,weights,names):
    out={}
    for group in sorted(set(phase.tolist()))+['ALL']:
        group_mask=np.ones(len(x),bool) if group=='ALL' else phase==group
        features={}
        for i,name in enumerate(names):
            mask=group_mask.copy()
            if i in (0,1,2,6):mask &= x[:,8]==1
            if i in (3,4,5,10,11):mask &= (x[:,8]==1)&(x[:,9]==1)
            if i==12:mask &= x[:,12]>0 # First callback handled separately.
            values=x[mask,i].astype(np.float64);w=weights[mask]
            if not len(values):features[name]=dict(count=0);continue
            q=np.quantile(values,PROBS).tolist();edges=np.unique(np.asarray(q,dtype=np.float64))
            # Extend fixed bins to include later out-of-reference values;
            # min/max/quantile excursions remain separately reported.
            internal=edges[1:-1].tolist() if len(edges)>1 else edges.tolist()
            hist_edges=np.asarray([-np.inf]+internal+[np.inf])
            counts=np.histogram(values,bins=hist_edges)[0]
            weighted=np.histogram(values,bins=hist_edges,weights=w)[0]
            features[name]=dict(count=len(values),minimum=q[0],median=q[4],p01=q[1],p05=q[2],p95=q[6],p99=q[7],maximum=q[8],
                quantile_probabilities=PROBS,quantiles=q,constant=len(edges)==1,
                histogram_internal_edges=internal,counts=counts.tolist(),hold_seconds=weighted.tolist(),
                included_hold_seconds=float(w.sum()))
        out[group]=features
    return out

def occupancy(x,phase,weights):
    categories=dict(phase=phase,ball_present=np.where(x[:,8]==1,'present','missing'),
        prediction_valid=np.where(x[:,9]==1,'valid','unavailable'),
        near_far=np.where(x[:,8]!=1,'missing',np.where(x[:,6]*6000>1500,'far','near')),
        speed=np.where(x[:,7]*2300<750,'below750',np.where(x[:,7]*2300<=800,'750to800','above800')),
        geometry=np.where(x[:,8]!=1,'missing',np.char.add(np.where(x[:,0]>=0,'front','behind'),
            np.where(x[:,1]>=0,'_right','_left'))))
    return {name:{str(v):dict(callbacks=int((a==v).sum()),hold_seconds=float(weights[a==v].sum()))
        for v in np.unique(a)} for name,a in categories.items()}

def build():
    manifest,matches,artifacts=plan()
    if REFERENCE.exists():raise RuntimeError('Reference already exists; immutable after creation')
    # Hash ALL required train-only inputs before any sample/tensor semantic reads.
    for i,(path,pin) in enumerate(artifacts.items(),1):
        print('PHASE4 TRAIN HASH',i,'/',len(artifacts),Path(path).name,flush=True)
        if sha(path)!=pin:raise RuntimeError('Training artifact mismatch: '+path)
    arrays=[];phases=[];holds=[];ids=[];summaries=[]
    for m in matches:
        print('PHASE4 TRAIN DESCRIPTIONS:',m['match_id'],flush=True)
        x=np.fromfile(ROOT/m['files']['processed/observations.bin']['path'],dtype='<f4').reshape(m['callbacks'],18)
        if x.shape!=(m['callbacks'],18) or not np.isfinite(x).all():raise RuntimeError('Training tensor contract mismatch')
        times=[];p=[];last=None;submissions=[]
        with (ROOT/m['files']['raw/trajectory.jsonl']['path']).open(encoding='utf-8') as f:
            for i,line in enumerate(f):
                r=json.loads(line)
                if not r['valid'] or r['match_id']!=m['match_id'] or r['callback']!=i+1:raise RuntimeError('Training row integrity error')
                if not np.array_equal(np.asarray(r['observation18'],dtype=np.float32),x[i]):raise RuntimeError('Raw/tensor observation mismatch')
                if last is not None and r['T']<=last:raise RuntimeError('Nonmonotonic training time')
                last=r['T'];times.append(r['T']);p.append(r['match_phase']);submissions.append(r['previous_submission'])
        if len(times)!=m['callbacks']:raise RuntimeError('Training count mismatch')
        times=np.asarray(times,np.float64);p=np.asarray(p);w=np.zeros(len(x),np.float64)
        for i in range(len(x)-1):
            if submissions[i+1] is None or submissions[i+1]['callback']!=i+1:raise RuntimeError('Prior native submission link mismatch')
            if p[i]==p[i+1]:w[i]=times[i+1]-times[i]
        crossed=int(np.count_nonzero(p[:-1]!=p[1:]))
        arrays.append(x[:,:13].copy());phases.append(p);holds.append(w);ids += [m['match_id']]*len(x)
        summaries.append(dict(match_id=m['match_id'],split='train',teacher_side=m['teacher_side'],callbacks=len(x),
            phase_crossing_holds_excluded=crossed,final_hold_censored=True,first_callback_dt=float(x[0,12]),
            total_same_phase_hold_seconds=float(w.sum())))
    x=np.concatenate(arrays);phase=np.concatenate(phases);weights=np.concatenate(holds);ids=np.asarray(ids)
    primary=np.isin(ids,PRIMARY_IDS)
    # No rows are saved as tensors: only descriptive histograms/quantiles.
    result=dict(version='phase4_train_reference_v1',description_only=True,source_manifest_sha256=MANIFEST_PIN,
        feature_order=manifest['features'][:13],reference_matches=summaries,input_hashes=artifacts,
        validation_read=False,test_read=False,model_loaded=False,fitting=False,
        builder_sha256=sha(Path(__file__)),
        primary=dict(matches=list(PRIMARY_IDS),callbacks=int(primary.sum()),
            features=distributions(x[primary],phase[primary],weights[primary],manifest['features'][:13]),
            occupancy=occupancy(x[primary],phase[primary],weights[primary])),
        secondary=dict(matches=list(TRAIN_IDS),callbacks=len(x),
            features=distributions(x,phase,weights,manifest['features'][:13]),occupancy=occupancy(x,phase,weights)),
        limitations=['Descriptive only; no policy selection/tuning/fitting.',
            'Masks and phases conditioned; crossed-phase holds and unobserved final holds excluded.',
            'Marginal feature differences do not establish causal policy effects or label usefulness.'])
    for path,pin in artifacts.items():
        if sha(path)!=pin:raise RuntimeError('Training input changed during description build')
    if sha(DATASET/'manifest.json')!=MANIFEST_PIN:raise RuntimeError('Manifest changed')
    REFERENCE.mkdir(exist_ok=False)
    write(REFERENCE/'statistics.json',result)
    write(REFERENCE/'integrity.json',dict(statistics_sha256=sha(REFERENCE/'statistics.json'),input_hashes=artifacts,
        source_manifest_sha256=MANIFEST_PIN,builder_sha256=sha(Path(__file__))))
    print('PHASE4 TRAIN-ONLY REFERENCE PASS:',REFERENCE/'statistics.json',flush=True)
    print('No validation/test samples read; no training or live launch.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build',action='store_true',required=True)
    parser.parse_args();build()
