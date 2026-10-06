"""V16 fixed diagnostic analysis. New audit matches only; no neural fitting."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
import math
import time
import numpy as np
from contracts import HERE,ROOT,FEATURES,CHANNELS,MODES,HISTORY,read,write,sha,require,verify,agree,mode,native_block,rebuild

K=32
CLOSE=.02
REPORT_STEM='v16_native_maneuver_observability_20261006'

def distribution(values):
    a=np.asarray(values,dtype=np.float64)
    if not len(a): return dict(count=0,minimum=None,median=None,p10=None,p90=None,maximum=None)
    return dict(count=len(a),minimum=float(a.min()),median=float(np.median(a)),p10=float(np.quantile(a,.1)),p90=float(np.quantile(a,.9)),maximum=float(a.max()))

def authorized():
    verify(); plan=read(HERE/'collection_plan.json'); inventory=[]
    # Verify EVERY permitted artifact before opening raw data. No other dataset accessed.
    for m in plan['matches']:
        folder=HERE/'matches'/m['match_id']; require(folder.is_dir(),'Missing audit match: '+m['match_id'])
        pin=read(folder/'integrity.json')
        require({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p.name!='integrity.json'}==set(pin['files']),
                'Unexpected/unsealed match artifact')
        for name,expected in pin['files'].items():
            path=folder/name
            require(path.resolve().is_relative_to(folder.resolve()),'Artifact escapes match')
            require(sha(path)==expected,'Missing/hash-mismatched V16 artifact: '+str(path))
        metadata=read(folder/'metadata.json'); run=read(folder/'run_summary.json'); closed=read(folder/'agent_closed.json')
        require(all(metadata[k]==v for k,v in m.items()),'Assignment differs from preregistered plan')
        require(metadata['source_manifest_sha256']==sha(HERE/'source_manifest.json'),'Collected with different source seal')
        require(run['status']=='natural_match_ended' and closed['status']=='running' and not closed['diagnostic_errors'] and not closed['publication_errors'],
                'Incomplete or diagnostic-failed match; do not silently exclude/retry')
        inventory.append(dict(**m,folder=folder,hashes=pin['files'],closed=closed))
        print('V16 MATCH HASH PASS',m['match_id'],flush=True)
    require(len(inventory)>=6,'Need at least six complete natural matches')
    require({m['teacher_side'] for m in inventory}=={'Blue','Orange'},'Both teacher sides required')
    require(len({m['play_session_id'] for m in inventory})>=2,'At least two actual identified sessions required')
    return inventory

def native_bins(n):
    def timer(x):
        if x==-1: return 'sentinel_-1'
        for lo,hi,name in ((-math.inf,0,'other_negative'),(0,.05,'[0,.05)'),(.05,.1,'[.05,.1)'),(.1,.2,'[.1,.2)'),(.2,.8,'[.2,.8)'),(.8,1.45,'[.8,1.45)')):
            if lo<=x<hi: return name
        return '[1.45,+inf)'
    dx,dy=n['dodge_dir']
    out=dict(air_state=str(n['air_state']),has_jumped=str(n['has_jumped']),has_double_jumped=str(n['has_double_jumped']),
                has_dodged=str(n['has_dodged']),dodge_timeout=timer(n['dodge_timeout']),dodge_elapsed=timer(n['dodge_elapsed']),
                dodge_dir_signs=str((int(np.sign(dx)),int(np.sign(dy)))),
                last_input_buttons=str(tuple(n['last_input'][5:])))
    for channel,value in zip(CHANNELS,n['last_input']):
        if type(value) is bool: label=str(value)
        elif value==0: label='exact_zero'
        elif value<-.5: label='[-1,-.5)'
        elif value<-.02: label='[-.5,-.02)'
        elif value<=.02: label='[-.02,.02]_nonzero'
        elif value<=.5: label='(.02,.5]'
        else: label='(.5,1]'
        out['last_input_'+channel]=label
    return out

def read_match(entry):
    folder=entry['folder']; submissions={}
    with (folder/'raw/submissions.jsonl').open(encoding='utf-8') as f:
        for line in f:
            s=json.loads(line); require(s['callback'] not in submissions,'Duplicate submission')
            require(s['status']=='local_send_returned','Transport failure'); submissions[s['callback']]=s
    obs=[]; labels=[]; action=[]; times=[]; frames=[]; native_values=[]; prior_actions=[]; phase=[]; private_phase=[]; event=[]
    rotations=[]; angular=[]; position=[]; receipt=[]; lag=[]; conditional=defaultdict(Counter); state_by_label=defaultdict(Counter)
    raw_timers=defaultdict(list); callback_errors=[]; previous=None; starts=completions=0; last_input_changed=0; previous_submitted_changed=0
    last_input_matches=Counter(); changed_input_matches=Counter(); eligible_lags=Counter(); changed_eligible_lags=Counter()
    flags_transitions=Counter(); native_equal_current=0; documented_reset_observations=Counter()
    long_gap=missing=reuse=0; recent=[]; sequence_begin={}; sequence_end={}; first_mono=last_mono=None
    with (folder/'raw/callbacks.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line); idx=len(obs); cb=idx+1
            require(row['callback']==cb and row['match_id']==entry['match_id'],'Duplicate/gap/cross-match processed rows')
            require(row['valid'] and not row['errors'],'Invalid row; stop instead of dropping')
            s=submissions.get(cb); require(s is not None and agree(row['teacher_controls'],s['controls']),'Unconfirmed current teacher submission')
            mode(row['teacher_controls']); require(row['mode']==mode(row['teacher_controls']),'Action label mismatch')
            require(row['side']==entry['teacher_side'],'Side mismatch')
            require(row['packet_receipt']['object_id']==row['packet_object_id'] and row['packet_receipt']['T']==row['T'] and row['packet_receipt']['frame']==row['frame'],'Packet provenance mismatch')
            r=row['prediction_receipt']
            require(r is None or r['object_id']==row['prediction_object_id'],'Prediction provenance mismatch')
            require(row['packet_receipt']['order']<row['callback_order']<s['order'],'Message processing order')
            require(row['packet_receipt']['monotonic']<=row['callback_started_monotonic']<=row['native_captured_monotonic']<=row['teacher_started_monotonic']<=row['teacher_finished_monotonic']<=s['before_send_monotonic']<=s['after_send_monotonic'],
                    'Pre-action snapshot/submission monotonic ordering')
            if r is not None: require(r['order']<row['callback_order'] and r['monotonic']<=row['native_captured_monotonic'],'Future prediction receipt')
            if previous is None:
                require(row['previous_callback_T'] is None and row['callback_dt'] is None and row['previous_submission'] is None and row['observation13'][12]==0,'First callback dt/history')
                pa=[0.,0.,0.,0.,0.,False,False,False]
            else:
                require(row['previous_callback_T']==previous['T'] and row['callback_dt']==row['T']-previous['T'] and row['T']>previous['T'],'Causal callback time')
                require(row['previous_submission']==submissions[cb-1],'Previous submission must be preceding confirmed callback')
                require(row['previous_submission']['after_send_monotonic']<=row['callback_started_monotonic'],'Future prior action')
                pa=row['previous_submission']['controls']
                if r is not None and previous['prediction_receipt'] is not None and r['counter']==previous['prediction_receipt']['counter']: reuse+=1
                if not agree(pa,recent[-1]['controls']): raise RuntimeError('Previous-action journal mismatch')
                if len(recent)>1 and not agree(pa,recent[-2]['controls']): previous_submitted_changed+=1
            require(np.array_equal(np.asarray(row['observation13'],np.float32),np.asarray(rebuild(row),np.float32)),'Exact 13D reconstruction failed')
            n=row['native']; nb=native_block(n); li=n['last_input']
            require(len(li)==8 and all(type(v) is bool for v in li[5:]) and all(math.isfinite(v) for v in li[:5]),'Native input schema')
            require(agree(li,li),'Finite native control sanity')
            changed=previous is not None and not agree(li,previous['native']['last_input'])
            last_input_changed+=int(changed)
            hits=[]
            for back in range(1,min(8,len(recent))+1):
                yes=agree(li,recent[-back]['controls']); last_input_matches[str(back)]+=int(yes)
                eligible_lags[str(back)]+=1
                if changed:
                    changed_input_matches[str(back)]+=int(yes); changed_eligible_lags[str(back)]+=1
                if yes: hits.append(back)
            native_equal_current+=int(agree(li,s['controls']))
            lag.append(dict(callback=cb,T=row['T'],native_last_input=li,previous_submission=None if not recent else recent[-1],
                            prior_matching_lags=hits,native_input_changed=changed,
                            current_control_coincidence=agree(li,s['controls']),
                            warning='Coincidence/lag match is not an engine application acknowledgement'))
            lab=MODES[row['mode']]
            consumed=row['consumed_phase_diagnostic']
            if row['mode']==0 and consumed in (1,3): lab='Neutral release' if consumed==1 else 'Neutral coast'
            for field,value in native_bins(n).items(): conditional[field][(value,lab)]+=1; state_by_label[field][(lab,value)]+=1
            for field in ('dodge_timeout','dodge_elapsed'): raw_timers[field].append(n[field])
            if n['air_state']==0:
                documented_reset_observations['ground_callbacks']+=1
                for field in ('has_double_jumped','has_dodged'):
                    documented_reset_observations[field+'_true_on_ground']+=int(n[field])
                documented_reset_observations['dodge_elapsed_nonzero_on_ground']+=int(n['dodge_elapsed']!=0)
            if previous:
                for field in ('air_state','has_jumped','has_double_jumped','has_dodged'):
                    if n[field]!=previous['native'][field]: flags_transitions[field]+=1
            starts+=int(row['teacher_trace']['flip_trigger'])
            before,after=row['sequence_before'],row['sequence_after']
            finished=after is not None and after['done'] and (before is None or not before['done'])
            completions+=int(finished)
            if row['teacher_trace']['flip_trigger']: sequence_begin[row['sequence_id_after']]=idx
            if finished: sequence_end[row['sequence_id_after']]=idx
            if first_mono is None: first_mono=row['callback_started_monotonic']
            last_mono=s['after_send_monotonic']
            obs.append(row['observation13']); labels.append(row['mode']); action.append(s['controls']); times.append(row['T']); frames.append(row['frame'])
            native_values.append(n); prior_actions.append(pa); phase.append(row['match_phase']); private_phase.append(-1 if consumed is None else consumed)
            # Diagnostic independence key only, never a neighbor input.
            active_event=row['sequence_id_after'] if consumed is not None else None
            event.append(('sequence',active_event) if active_event is not None else ('timebin',int(row['T']//2)))
            rotations.append(row['car']['rotation_pyr']); angular.append(row['car']['angular_velocity']); position.append(row['car']['position'])
            receipt.append(dict(packet=row['packet_receipt']['counter'],prediction=None if r is None else r['counter']))
            long_gap+=int(row['callback_dt'] is not None and row['callback_dt']>=1); missing+=int(row['ball'] is None)
            recent.append(s); recent=recent[-8:]; previous=row
            if cb%10000==0: print('V16 ROW AUDIT',entry['match_id'],cb,flush=True)
    require(len(obs)==len(submissions)==entry['closed']['callbacks'],'No dropped processed callbacks/submissions')
    require(len(obs)>64,'Match too short for common causal endpoints')
    x=dict(id=entry['match_id'],meta={k:v for k,v in entry.items() if k not in ('folder','hashes','closed')},obs=np.asarray(obs,np.float32),
           label=np.asarray(labels,np.int64),actions=np.asarray(action,np.float64),T=np.asarray(times,np.float64),frame=np.asarray(frames,np.int64),
           C=np.asarray([native_block(n) for n in native_values],np.float32),B=np.asarray([n['last_input'] for n in native_values],np.float32),
           A=np.asarray(prior_actions,np.float32),native=native_values,phase=np.asarray(phase),private_phase=np.asarray(private_phase),event=event,
           rotation=np.asarray(rotations),angular=np.asarray(angular),position=np.asarray(position),receipt=receipt,
           timing=dict(callback_dt=distribution(np.diff(times)),long_gaps=long_gap,missing_ball=missing,prediction_reuse=reuse),
           counts=dict(callbacks=len(obs),starts=starts,completions=completions,Jump=int(np.sum(np.asarray(labels)==2)),
                       Front_dodge=int(np.sum(np.asarray(labels)==3)),release=int(np.sum(np.asarray(private_phase)==1)),
                       modes=dict(Counter(MODES[i] for i in labels)),phases=dict(Counter(phase))),
           synchronization=dict(valid_rows=len(obs),invalid_rows=0,dropped_processed_rows=0,confirmed_local_submissions=len(submissions),
                                future_values_detected=0,pre_action_native_snapshot=True,engine_application_acknowledged=False),
           last_input=dict(lag_matches=dict(last_input_matches),changed_input_lag_matches=dict(changed_input_matches),
                           lag_denominators=dict(eligible_lags),changed_input_lag_denominators=dict(changed_eligible_lags),
                           changed_native_inputs=last_input_changed,changed_previous_submissions=previous_submitted_changed,
                           matches_current_label=native_equal_current,lag_evidence=lag),
           conditional={k:[dict(state=a,label=b,count=c) for (a,b),c in v.items()] for k,v in conditional.items()},
           reverse_conditional={k:[dict(label=a,state=b,count=c) for (a,b),c in v.items()] for k,v in state_by_label.items()},
           native_distributions={k:distribution(v) for k,v in raw_timers.items()},native_transitions=dict(flags_transitions),
           documented_reset_observations=dict(documented_reset_observations),
           completed_sequence_bounds=[(sequence_begin[k],v) for k,v in sequence_end.items() if k in sequence_begin])
    n=len(obs); boundary=np.zeros(n,bool)
    transitions=np.flatnonzero(x['label'][1:]!=x['label'][:-1])+1
    for t in transitions:
        # Labels select evaluation subsets only, never the feature vectors.
        if x['obs'][t,8] and x['obs'][t-1,8] and x['T'][t]-x['T'][t-1]<=5/120+.0001:
            boundary[max(0,t-5):min(n,t+6)]=True
    up=np.cos(x['rotation'][:,0])*np.cos(x['rotation'][:,2])
    x['masks']=dict(boundary=boundary,kickoff=x['phase']=='MatchPhase.Kickoff',inverted=up<0,tilted=up<.5,
                    missing=x['obs'][:,8]==0,long_gap=np.r_[False,np.diff(x['T'])>=1],representative=np.arange(n)%128==0)
    x['counts'].update(kickoff=int(x['masks']['kickoff'].sum()),inverted=int(x['masks']['inverted'].sum()),tilted=int(x['masks']['tilted'].sum()))
    x['timing'].update(canonical_duration=float(x['T'][-1]-x['T'][0]),processed_callback_wall_duration=float(last_mono-first_mono))
    candidates=set(range(63,n,128)); candidates.update(np.flatnonzero((x['label']==2)|(x['label']==3)).tolist())
    for mask in (boundary,x['masks']['kickoff'],x['masks']['inverted'],x['masks']['tilted'],x['masks']['missing'],x['masks']['long_gap']):
        indexes=np.flatnonzero(mask); indexes=indexes[indexes>=63]
        if len(indexes): candidates.update(indexes[np.linspace(0,len(indexes)-1,min(256,len(indexes)),dtype=int)].tolist())
    # Keep first callback/history exclusions explicit. Common endpoints H64 for every comparison.
    x['query']=np.asarray(sorted(i for i in candidates if i>=63),np.int64)
    x['masks']['representative']=np.zeros(n,bool); x['masks']['representative'][np.arange(63,n,128)]=True
    return x

def history(x,indexes,h):
    return np.asarray(x['obs'][indexes[:,None]+np.arange(1-h,1)],np.float64).reshape(len(indexes),h*13)/math.sqrt(h*13)

def nearest(q,reference,h,group):
    qidx=q['query']; qbase=history(q,qidx,h); blocks=[]; info=[]
    for r in reference:
        idx=np.arange(63,len(r['label'])); base=history(r,idx,h)
        values=base if group is None else np.c_[base,np.asarray(r[group][idx],np.float64)/math.sqrt(r[group].shape[1])]
        blocks.append(values); info.extend((r['id'],int(i)) for i in idx)
    train=np.concatenate(blocks); query=qbase if group is None else np.c_[qbase,np.asarray(q[group][qidx],np.float64)/math.sqrt(q[group].shape[1])]
    require(len(train)>=K,'Not enough match-disjoint reference endpoints')
    result=[]; distances=[]
    for start in range(0,len(query),16):
        a=query[start:start+16]; best=np.empty((len(a),0),np.int64); best_d=np.empty((len(a),0))
        for b in range(0,len(train),8192):
            t=train[b:b+8192]; d=np.maximum((a*a).sum(1)[:,None]+(t*t).sum(1)[None,:]-2*a@t.T,0)
            ids=np.broadcast_to(np.arange(b,b+len(t)),d.shape)
            all_d=np.c_[best_d,d]; all_i=np.c_[best,ids]
            # Stable distance/id ordering, no k or distance tuning.
            order=np.lexsort((all_i,all_d),axis=1)[:,:K]
            best=np.take_along_axis(all_i,order,1); best_d=np.take_along_axis(all_d,order,1)
        direct=np.sqrt(((a[:,None,:]-train[best])**2).sum(2))
        order=np.lexsort((best,direct),axis=1); best=np.take_along_axis(best,order,1); direct=np.take_along_axis(direct,order,1)
        result.extend(best.tolist()); distances.extend(direct.tolist())
        if start%256==0: print('V16 LOMO',q['id'],'H',h,'group',group,'queries',start,'/',len(query),flush=True)
    return np.asarray(result),np.asarray(distances),info

def measure(q,refs,h,group):
    ids,distance,lookup=nearest(q,refs,h,group); byid={r['id']:r for r in refs}; qi=q['query']
    labels=np.asarray([[byid[lookup[j][0]]['label'][lookup[j][1]] for j in row] for row in ids])
    actions=np.asarray([[byid[lookup[j][0]]['actions'][lookup[j][1]] for j in row] for row in ids])
    probs=np.stack([(labels==k).mean(1) for k in range(4)],1); predicted=np.argmax(probs,1)
    steering=np.zeros(len(qi))
    for i in range(len(qi)):
        chase=labels[i]==1
        if predicted[i]==1 and chase.any(): steering[i]=actions[i,chase,1].mean()
    truth=q['label'][qi]; trueprob=probs[np.arange(len(qi)),truth]
    njump=(predicted==2)|(predicted==3); truthjump=q['actions'][qi,5].astype(bool)
    base_distance=np.empty_like(distance)
    for i,t in enumerate(qi):
        base_distance[i]=[np.sqrt(np.mean((q['obs'][t].astype(float)-byid[lookup[j][0]]['obs'][lookup[j][1]].astype(float))**2)) for j in ids[i]]
    close=base_distance<=CLOSE; conflicting=(labels!=truth[:,None])&close
    def metrics(mask):
        def mean(v,selection): return float(v[selection].mean()) if selection.any() else None
        return dict(rows=int(mask.sum()),mode_accuracy=mean(predicted==truth,mask),pure_jump_recall=mean(predicted==2,mask&(truth==2)),
                    native_jump_recall=mean(njump,mask&truthjump),front_dodge_recall=mean(predicted==3,mask&(truth==3)),
                    chase_accuracy=mean(predicted==1,mask&(truth==1)),chase_steer_MAE=mean(np.abs(steering-q['actions'][qi,1]),mask&(truth==1)),
                    true_label_probability=mean(trueprob,mask),close_neighbor_true_label_probability=mean(trueprob,mask&close.any(1)),
                    ambiguity_rate=mean(conflicting.any(1),mask),close_query_count=int((mask&close.any(1)).sum()),
                    pure_jump_rows=int((mask&(truth==2)).sum()),front_dodge_rows=int((mask&(truth==3)).sum()),
                    native_jump_rows=int((mask&truthjump).sum()),chase_rows=int((mask&(truth==1)).sum()))
    subsets={'all_targeted_queries':np.ones(len(qi),bool),**{name:mask[qi] for name,mask in q['masks'].items()}}
    result=dict(query_count=len(qi),metrics={k:metrics(v) for k,v in subsets.items()},h=h,group=group)
    evidence=dict(neighbor_ids=ids,neighbor_distances=distance,current13_distance=base_distance,neighbor_lookup=lookup,
                  probabilities=probs,predicted=predicted,steer=steering,trueprob=trueprob,conflicting=conflicting,close=close)
    return result,evidence

def pair_details(q,refs,evidence):
    byid={r['id']:r for r in refs}; pairs=[]; used_q=set(); used_r=set()
    for i,qi in enumerate(q['query']):
        for j,refid in enumerate(evidence['neighbor_ids'][i]):
            rid,ri=evidence['neighbor_lookup'][refid]; r=byid[rid]
            if evidence['current13_distance'][i,j]>CLOSE: continue
            qa,ra=q['actions'][qi],r['actions'][ri]
            material=q['label'][qi]!=r['label'][ri] or qa[5]!=ra[5] or qa[2]!=ra[2] or (q['label'][qi]==r['label'][ri]==1 and abs(qa[1]-ra[1])>.2)
            if not material: continue
            qkey=(q['id'],*q['event'][qi]); rkey=(rid,*r['event'][ri])
            independent=qkey not in used_q and rkey not in used_r
            if independent: used_q.add(qkey); used_r.add(rkey)
            def endpoint(x,idx): return dict(match_id=x['id'],callback=int(idx+1),T=float(x['T'][idx]),frame=int(x['frame'][idx]),
                mode=MODES[int(x['label'][idx])],controls=x['actions'][idx].tolist(),native=x['native'][idx],
                rotation_pyr=x['rotation'][idx].tolist(),angular_velocity=x['angular'][idx].tolist(),
                diagnostic_event_key=x['event'][idx])
            pairs.append(dict(query=endpoint(q,qi),reference=endpoint(r,ri),distance13=float(evidence['current13_distance'][i,j]),
                              different_native_C=bool(not np.array_equal(q['C'][qi],r['C'][ri])),
                              maneuver_relevant=bool(q['private_phase'][qi]>=0 or r['private_phase'][ri]>=0 or q['label'][qi] in (2,3) or r['label'][ri] in (2,3)),
                              independent_event_pair=independent,clock_difference=float(q['T'][qi]-r['T'][ri]),
                              clock_difference_scope='Different matches have independent clocks; not a causal interval'))
            break  # One nearest material conflict per query, not thousands of correlated pairs.
    return pairs

def aggregate(results,condition,subset):
    entries=[r[condition]['metrics'][subset] for r in results.values()]
    out={}
    for key in ('mode_accuracy','pure_jump_recall','native_jump_recall','front_dodge_recall','chase_accuracy','chase_steer_MAE','true_label_probability','ambiguity_rate'):
        weight='pure_jump_rows' if key=='pure_jump_recall' else 'front_dodge_rows' if key=='front_dodge_recall' else 'rows'
        if key=='native_jump_recall': weight='native_jump_rows'
        if key in ('chase_accuracy','chase_steer_MAE'): weight='chase_rows'
        valid=[e for e in entries if e[key] is not None and e[weight]>0]
        out[key]=None if not valid else sum(e[key]*e[weight] for e in valid)/sum(e[weight] for e in valid)
    out['rows']=sum(e['rows'] for e in entries)
    return out

def promising(results,base,candidate,independent):
    b=aggregate(results,base,'boundary'); c=aggregate(results,candidate,'boundary')
    bj=aggregate(results,base,'all_targeted_queries'); cj=aggregate(results,candidate,'all_targeted_queries')
    br=aggregate(results,base,'representative'); cr=aggregate(results,candidate,'representative')
    def gain(a,b): return None if a is None or b is None else a-b
    gains=dict(boundary=gain(c['mode_accuracy'],b['mode_accuracy']),pure_jump=gain(cj['pure_jump_recall'],bj['pure_jump_recall']),
               dodge=gain(cj['front_dodge_recall'],bj['front_dodge_recall']),representative=gain(cr['mode_accuracy'],br['mode_accuracy']))
    each=[]
    for name,rs in results.items():
        aa,bb=rs[base]['metrics'],rs[candidate]['metrics']
        each.append(all(aa[s][k] is not None and bb[s][k] is not None and bb[s][k]>aa[s][k]
                        for s,k in (('boundary','mode_accuracy'),('all_targeted_queries','pure_jump_recall'),('all_targeted_queries','front_dodge_recall'))))
    checks=dict(independent_close_cases=independent>=30,boundary=gains['boundary'] is not None and gains['boundary']>=.1,
                pure_jump=gains['pure_jump'] is not None and gains['pure_jump']>=.05,dodge=gains['dodge'] is not None and gains['dodge']>=.05,
                representative_guard=gains['representative'] is not None and gains['representative']>=-.01,every_match_positive=all(each))
    return dict(gains=gains,checks=checks,promising=all(checks.values()))

def run():
    require(not (HERE/'results_v1').exists(),'Results already exist; no overwrite/reanalysis with changed thresholds')
    for suffix in ('.md','.json'): require(not (ROOT/'training/reports'/(REPORT_STEM+suffix)).exists(),'Final report exists')
    inventory=authorized(); out=HERE/'results_v1'; out.mkdir()
    data=[read_match(e) for e in inventory]; results={}; all_pairs=[]; spans={}; gate_inputs={}
    for q in data:
        refs=[r for r in data if r is not q]; rs={}; qspans={}; baseline=None
        for h in HISTORY:
            indexes=np.arange(h-1,len(q['label'])); duration=q['T'][indexes]-q['T'][indexes-h+1]
            qspans[str(h)]=dict(**distribution(duration),maneuver_events=len({q['event'][i] for i in indexes if q['private_phase'][i]>=0}),
                              fully_contained_completed_sequences=sum(1 for start,end in q['completed_sequence_bounds'] if end-start+1<=h and max(end,h-1)<=min(start+h-1,len(q['label'])-1)),
                              complete_sequence_span_fraction=float((duration>=1.1).mean()),definition='Actual processed callback elapsed span; gaps retained, no resets/padding')
            for group in (None,'C'):
                name=f'H{h}_'+('13D' if group is None else 'native_C')
                result,evidence=measure(q,refs,h,group); rs[name]=result
                np.savez_compressed(out/(q['id']+'_'+name+'_neighbors.npz'),query=q['query'],**{k:v for k,v in evidence.items() if isinstance(v,np.ndarray)})
                write(out/(q['id']+'_'+name+'_lookup.json'),evidence['neighbor_lookup'])
                if h==1 and group is None:
                    baseline=evidence; pairs=pair_details(q,refs,evidence); all_pairs.extend(pairs)
            if h==1:
                # Same base-defined close-conflict queries in both conditions; no post-hoc support selection.
                base_name='H1_13D'; candidate_name='H1_native_C'
                conflict=baseline['conflicting'].any(1)
                # Candidate evidence here belongs to native C (last group in fixed loop).
                rs[candidate_name]['base_close_conflict_diagnostic']=dict(rows=int(conflict.sum()),
                    baseline_true_label_probability=None if not conflict.any() else float(baseline['trueprob'][conflict].mean()),
                    native_true_label_probability=None if not conflict.any() else float(evidence['trueprob'][conflict].mean()),
                    baseline_mode_accuracy=None if not conflict.any() else float((baseline['predicted'][conflict]==q['label'][q['query']][conflict]).mean()),
                    native_mode_accuracy=None if not conflict.any() else float((evidence['predicted'][conflict]==q['label'][q['query']][conflict]).mean()))
            # Separate previous submitted input/native last_input probes, never mixed into primary C.
            if h in (1,64):
                for group in ('A','B'):
                    name=f'H{h}_'+('prior_submission' if group=='A' else 'last_input')
                    rs[name],ev=measure(q,refs,h,group)
                    np.savez_compressed(out/(q['id']+'_'+name+'_neighbors.npz'),query=q['query'],**{k:v for k,v in ev.items() if isinstance(v,np.ndarray)})
                    write(out/(q['id']+'_'+name+'_lookup.json'),ev['neighbor_lookup'])
        gate_inputs[q['id']]=dict(close_conflict_query_count=int(baseline['conflicting'].any(1).sum()),
                                  excluded_initial_callbacks=63,common_query_endpoints=len(q['query']))
        results[q['id']]=rs; spans[q['id']]=qspans
        write(out/(q['id']+'_last_input_timing.json'),q['last_input'])
    # Deduplicate shared event pairs across folds, not just within a query fold.
    used_events=set(); independent=0
    for p in all_pairs:
        endpoints=(str((p['query']['match_id'],p['query']['diagnostic_event_key'])),str((p['reference']['match_id'],p['reference']['diagnostic_event_key'])))
        independent_case=p['maneuver_relevant'] and not any(e in used_events for e in endpoints)
        p['independent_event_pair']=independent_case
        if independent_case: independent+=1; used_events.update(endpoints)
    native_checks=promising(results,'H64_13D','H64_native_C',independent)
    history_checks=promising(results,'H1_13D','H64_13D',independent)
    # D by default: failure to qualify is not evidence of absence.
    classification='D'
    if native_checks['promising']: classification='C' if history_checks['promising'] else 'A'
    elif history_checks['promising']:
        gains=native_checks['gains']
        if independent>=30 and all(gains[k] is not None and abs(gains[k])<.02 for k in ('boundary','pure_jump','dodge')): classification='B'
    # Repeat source and ALL collected artifact hashes after analysis, before reporting.
    authorized()
    conditional={q['id']:dict(P_label_given_native=q['conditional'],P_native_given_label=q['reverse_conditional'],
                             raw_timer_distributions=q['native_distributions'],native_transitions=q['native_transitions'],
                             documented_reset_observations=q['documented_reset_observations']) for q in data}
    # Convert exact count tables to explicit conditional probabilities.
    for value in conditional.values():
        for direction,first in (('P_label_given_native','state'),('P_native_given_label','label')):
            for table in value[direction].values():
                denominator=Counter()
                for row in table: denominator[row[first]]+=row['count']
                for row in table: row['probability']=row['count']/denominator[row[first]]
    limitation=['Teacher label predictability is not optimal control or policy success.',
                'LOMO is match-disjoint, not opponent/session-disjoint; literal user session IDs overlap paired matches.',
                'H64 and even native augmentation are finite diagnostic contexts, not GRU sufficiency proof.',
                'Current packet fields are copied pre-decision, but engine sampling/last_input application phase has no installed producer-source proof.',
                'Lag equality and transition association do not prove successful engine execution.',
                'Correlated callbacks and conservatively deduplicated events are not statistical independence proof.',
                'Heterogeneous equal-block distance is fixed but scale-sensitive; no statistical significance claim.',
                'Missing balls, replays and callback gaps remain in histories; they may carry non-play labels.',
                'Recovery strata use measured orientation; tilted/inverted is not native contact or optimal recovery.',
                'Exhaustive norm/dot neighbor search has floating roundoff; returned distances recomputed directly.',
                '32-neighbor close conflicts cover the returned neighbor set, not every possible close pair.']
    report=dict(status='diagnostic_analysis_complete_awaiting_review',classification=classification,
                native_evidence=native_checks,history_evidence=history_checks,independent_usable_conflict_events=independent,
                inventory=[dict(**q['meta'],counts=q['counts'],timing=q['timing'],synchronization=q['synchronization']) for q in data],
                total_callbacks=sum(len(q['label']) for q in data),results=results,temporal_coverage=spans,conditional_purity=conditional,
                close_conflict_pairs=all_pairs,gate_inputs=gate_inputs,last_input_summary={q['id']:{k:v for k,v in q['last_input'].items() if k!='lag_evidence'} for q in data},
                last_input_conclusion='Observed lag/transition association only. Do not assert last_input == previous_submission or successful execution; inspect changed-action evidence.',
                source_audit=read(HERE/'source_audit.json'),protocol_sha256=sha(HERE/'protocol.md'),source_manifest_sha256=sha(HERE/'source_manifest.json'),
                input_hashes={e['match_id']:e['hashes'] for e in inventory},protected_hashes_unchanged=43,
                no_policy_fitting=True,no_test_access=True,no_live_launch_by_analysis=True,no_contract_change=True,no_dagger=True,
                recommendation='Review native-state timing and the preregistered per-match evidence before authorizing any feature or model experiment.',
                limitations=limitation,created_utc=datetime.now(timezone.utc).isoformat())
    write(out/'report.json',report)
    sections=['Executive conclusion','Exact native SDK version/source audit','Field semantics and timing','Pre-action vs post-action classification',
              'New collection protocol','Dataset inventory','13D conflict analysis','Native state conditional purity','Fixed 32-neighbor observability results',
              'last_input vs previous-submission analysis','H1/H2/H4/H8/H16/H32/H64 temporal coverage','Kickoff findings','Flip/Jump/Dodge findings',
              'Recovery/inversion findings','Evidence classification A/B/C/D','One next recommendation','Experiments explicitly NOT authorized',
              'Integrity/hash results','Exact artifacts read/created','Limitations']
    bodies=[f"Classification **{classification}**, diagnostic only; {report['total_callbacks']} processed callbacks. No policy or official feature change.",
            'Installed rlbot 2.0.0b55 / rlbot-flatbuffers 0.19.0. source_audit.json contains exact installed documentation/source lines and unresolved engine semantics.',
            'All raw fields retained before the current teacher action; seconds-based timers retain sentinel -1. AirState is not world-up. No invented reset/direction frame.',
            'Received packet snapshot is pre-current-submission. Engine state may describe prior execution; last_input application/tick semantics remain observational, not asserted.',
            'Six approved natural teacher-controlled matches, alternating sides. No probes, student, state setting, pauses, resampling or game-state repair.',
            json.dumps(report['inventory'],indent=2),'Close RMS <= .02; materially different mode/jump/pitch or Chase steer > .2. '+str(independent)+' conservatively deduplicated event-pair cases; see JSON physical examples.',
            'Exact categorical and fixed timer-bin P(label|state) / P(state|label), including private diagnostic Release/Coast labels, in JSON. No private labels in probe inputs.',
            'Fixed 32-neighbor LOMO at all seven causal histories, native C excluding last_input. Equal-block weights and preprocessing fixed before collection.',
            json.dumps(report['last_input_summary'],indent=2)+'\n'+report['last_input_conclusion'],json.dumps(spans,indent=2),
            'Per-match kickoff results in results[match][condition].metrics.kickoff; not maneuver-success evidence.',
            'Pure Jump/native jump/Front-dodge and boundary metrics reported separately in JSON; sequences are diagnostic label strata only.',
            'Per-match inverted/tilted strata and conflict physical orientation/angular velocity recorded; not an optimal-recovery test.',
            json.dumps(dict(classification=classification,native=native_checks,history=history_checks),indent=2),report['recommendation'],
            'No policy fitting, checkpoint training, test access, DAgger, deployment, teacher/contract modifications or automatic new collection.',
            'All collected artifacts verified before raw reads and again after analysis; 43 protected hashes unchanged.',
            'Only V16 raw journals/configs/source audit read as sample data. Hash manifest and JSON inventory enumerate exact artifacts; V10 samples unused.',
            '\n'.join('- '+x for x in limitation)]
    text='# V16 native maneuver-state observability — 2026-10-06\n\n'
    for i,(name,body) in enumerate(zip(sections,bodies),1): text+=f'## {i}. {name}\n\n{body}\n\n'
    (out/'report.md').write_text(text,encoding='utf-8')
    write(ROOT/'training/reports'/(REPORT_STEM+'.json'),report)
    (ROOT/'training/reports'/(REPORT_STEM+'.md')).write_text(text,encoding='utf-8')
    write(out/'integrity.json',dict(files={p.relative_to(out).as_posix():sha(p) for p in out.iterdir() if p.is_file()},
                                  published_reports={suffix:sha(ROOT/'training/reports'/(REPORT_STEM+suffix)) for suffix in ('.md','.json')}))
    print('V16 ANALYSIS COMPLETE: classification',classification,'Report:',ROOT/'training/reports'/(REPORT_STEM+'.md'),flush=True)
    print('STOP. Diagnostic evidence only. No training/feature change authorized.',flush=True)

def check():
    verify()
    from contracts import native_block
    n=dict(air_state=0,has_jumped=False,has_double_jumped=False,has_dodged=False,dodge_timeout=-1.,dodge_elapsed=0.,dodge_dir=[0.,0.])
    require(len(native_block(n))==12 and native_block(n)[8]==-1/1.45,'Sentinel/encoding preflight')
    # Algebra-only diagnostic inputs, not packets, predictions or trajectories.
    a=np.arange(7*13,dtype=np.float32).reshape(7,13)
    for h in (1,2,4):
        x=history(dict(obs=a),np.array([6]),h)
        require(x.shape==(1,13*h) and np.allclose(x.reshape(h,13)*math.sqrt(h*13),a[7-h:7]),'Causal flattening check')
    require(native_bins(dict(n,last_input=[0.,0.,0.,0.,0.,False,False,False]))['dodge_timeout']=='sentinel_-1','Fixed descriptive bins')
    print('V16 NON-LIVE ANALYSIS PREFLIGHT PASS: encoding/causal indexing; no dataset/sample read, game, fitting or test.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--check',action='store_true'); g.add_argument('--run',action='store_true'); args=p.parse_args()
    check() if args.check else run()
