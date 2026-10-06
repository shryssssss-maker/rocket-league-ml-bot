"""Bounded numerical checks, not trajectories, training or dataset access."""
import math
import numpy as np
import torch
import boundary


def check():
    import experiment as exp
    c,seal=exp.pins();env=exp.helpers();torch.set_num_threads(1)
    tests=[]
    def passed(name):tests.append(name)
    # Exact original Policy/initialization and causal shape/continuity.
    exp.seed();a=env['Policy']().eval();h1=exp.initial_hash(a)
    exp.seed();b=env['Policy']().eval();assert exp.initial_hash(b)==h1
    assert sum(p.numel() for p in a.parameters())==26181
    x=torch.arange(130,dtype=torch.float32).reshape(10,13)/100
    with torch.inference_mode():
        z,s,h=a(x);z1,s1,h1c=a(x[:4]);z2,s2,_=a(x[4:],h1c)
        assert z.shape==(10,4) and s.shape==(10,) and h.shape==(1,1,64)
        assert torch.allclose(torch.cat((z1,z2)),z,atol=1e-6,rtol=1e-6)
        xx=x.clone();xx[5:]+=5;future,_,_=a(xx);assert torch.equal(z[:5],future[:5])
    passed('Exact V13 architecture, seeded parameter identity, causal hidden continuity')
    expected=np.array([[0,0,0,0,0,0,0,0],[1,.3,0,0,0,0,0,0],
                       [0,0,0,0,0,1,0,0],[0,0,-1,0,0,1,0,0]],np.float32)
    assert np.array_equal(env['decode'](np.arange(4),np.array([0,.3,0,0],np.float32)),expected)
    passed('Pinned eight-channel native decoder')
    # Independent scalar oracle, includes both groups and single-group cases.
    y=torch.tensor([1,2,2,0,0,3,3,0,1],dtype=torch.int64)
    logits=torch.tensor([[.4,.2,-.3,.1]]*len(y),dtype=torch.float32,requires_grad=True)
    eligible=torch.tensor([False]+[True]*(len(y)-1))
    value,cache,stats=boundary.loss(logits,y,eligible)
    p=torch.softmax(logits.detach(),1).numpy();oracle=[]
    for k in range(4):
        positive=[];negative=[]
        for t in range(1,len(y)):
            prev,cur=int(y[t-1]),int(y[t])
            qs=[(1-p[t-1,2])*p[t,2],p[t-1,2]*p[t,0],p[t-1,0]*p[t,3],p[t-1,3]*p[t,0]]
            bs=[prev!=2 and cur==2,prev==2 and cur==0,prev==0 and cur==3,prev==3 and cur==0]
            q=max(boundary.EPSILON,min(1-boundary.EPSILON,float(qs[k])))
            (positive if bs[k] else negative).append(-math.log(q) if bs[k] else -math.log1p(-q))
        groups=[sum(g)/len(g) for g in (positive,negative) if g]
        oracle.append(sum(groups)/len(groups))
    assert abs(float(value.detach())-sum(oracle)/4)<2e-7
    assert stats['positive']==[1,1,1,1] and stats['eligible_pairs']==8
    assert cache[0].requires_grad is False
    value.backward();assert torch.isfinite(logits.grad).all() and torch.any(logits.grad!=0)
    passed('All four exact boundary scores/targets, balanced scalar oracle, finite gradients')
    for onepair in ([1,1],[1,2],[2,0],[0,3],[3,0]):
        zz=torch.zeros(2,4,requires_grad=True);yy=torch.tensor(onepair)
        val,_,st=boundary.loss(zz,yy,torch.tensor([False,True]));val.backward()
        assert torch.isfinite(val) and torch.isfinite(zz.grad).all()
        assert sum(st['positive'])==int(onepair!=[1,1])
    val,_,st=boundary.loss(torch.zeros(1,4,requires_grad=True),torch.tensor([0]),torch.tensor([False]))
    assert float(val.detach())==0 and st['empty_eligible_chunk']==1
    passed('Only-positive/only-negative and empty-pair reductions')
    # Cross-chunk first pair uses cached old probabilities and carries no graph.
    prior=torch.tensor([.1,.2,.6,.1],requires_grad=True)
    zz=torch.zeros(1,4,requires_grad=True)
    val,_,st=boundary.loss(zz,torch.tensor([0]),torch.tensor([True]),(prior.detach(),2))
    val.backward();assert prior.grad is None and zz.grad is not None
    assert st['cross_chunk_pairs']==1 and st['positive']==[0,1,0,0]
    passed('Detached cross-chunk probability cache, first pair included without extra forward')
    tt=np.array([0.,.0625,.125,.25,.3125]);dt=np.r_[0.,np.diff(tt)]
    assert boundary.eligibility(tt,dt,np.array([1,1,0,1,1])).tolist()==[False,True,False,False,True]
    assert boundary.eligibility(np.array([0.,.1]),np.array([0.,.1]),np.ones(2)).tolist()==[False,True]
    assert boundary.eligibility(np.array([0.,0.,-.1]),np.array([0.,0.,-.1]),np.ones(3)).tolist()==[False,False,False]
    try:boundary.eligibility(tt,dt+.0001,np.ones(5))
    except ValueError:pass
    else:raise AssertionError('Invalid dt accepted')
    passed('Exact dt validation; missing/gap/time masks; no across-match pair')
    extreme=torch.tensor([[1000.,-1000.,-1000.,-1000.],[-1000.,-1000.,1000.,-1000.]],requires_grad=True)
    val,_,st=boundary.loss(extreme,torch.tensor([0,2]),torch.tensor([False,True]));val.backward()
    assert st['clamp_count']>0 and torch.isfinite(val) and torch.isfinite(extreme.grad).all()
    try:boundary.loss(torch.full((1,4),float('nan')),torch.tensor([0]),torch.tensor([False]))
    except ValueError:pass
    else:raise AssertionError('Nonfinite logits accepted')
    passed('Finite clamp endpoints, log stability and nonfinite rejection')
    # Mode order, real release and edge times; delayed/held Jump cannot pass.
    modes=np.array([1,2,2,0,0,3,3,3,0,0,1])
    times=np.arange(len(modes),dtype=np.float64)/64
    match=dict(metadata={'match_id':'numerical_fixture_only'},mode=modes,time=times,dt=np.r_[0,np.diff(times)],obs=np.ones((len(modes),13),np.float32))
    t=boundary.temporal(match,modes);assert t['eligible']==1 and t['successful']==1
    collapsed=np.array([1,3,3,3,3,3,3,3,0,0,1]);assert boundary.temporal(match,collapsed)['successful']==0
    held=np.array([1,2,2,2,2,2,0,3,0,0,1]);assert boundary.temporal(match,held)['successful']==0
    match['obs'][3,8]=0;t=boundary.temporal(match,modes);assert t['eligible']==0 and t['excluded_complete_patterns']
    passed('Preregistered timed-core success, missed release, late edges and interrupted exclusion')
    assert c['regression_guards']=={'chase_accuracy_min':0.9896232223028739,'chase_steer_mae_max':0.1695616946664056,'overall_accuracy_min':0.9522601595058281}
    assert boundary.LAMBDA==.1
    result=dict(status='non_dataset_numerical_preflight_pass',tests=tests,source_hashes_unchanged=43,
                initial_state_sha256=h1,implementation_seal=seal,dataset_samples_read=False,
                baseline_evaluated=False,fitting=False,test_access=False,live=False,v1_run=False,
                pending='User-run validation baseline then one v2 training command; no automatic execution')
    exp.write(exp.HERE/'preflight_v2.json',result)
    print('V14 v2 NON-DATASET PREFLIGHT PASS:',len(tests),'checks. No baseline/fitting/test/live/DAgger.',flush=True)
    return result
