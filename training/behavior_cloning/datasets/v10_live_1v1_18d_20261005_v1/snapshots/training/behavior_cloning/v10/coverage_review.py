"""Offline natural-trajectory audit/export and cumulative coverage, no training."""
import array
import collections
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from collection import HERE,BASE,MATCHES,PLAN,anchor,write_json,plan as validate_plan
sys.path.insert(0,str(BASE))
from common import ROOT,PILOT,SPLITS,FEATURES,CHANNELS,sha,mode,preserve
from dataset import reconstruct,distribution

VALUE_FIELDS=('steer','callback_dt','car_speed','ball_forward','ball_right','ball_up','ball_distance',
              'ball_height','ball_speed','ball_spin','prediction_horizon')
TENSOR_SPECS={'observations':('f',18),'analog_controls':('f',5),'buttons':('B',3),
              'mode':('B',1),'callback':('Q',1),'frame':('Q',1),'time':('d',1),'dt':('d',1)}

def verify_files(folder,expected):
    for name,h in expected.items():
        if sha(folder/name)!=h: raise RuntimeError('Integrity mismatch: '+str(folder/name))

def audit(folder,identity):
    run=json.loads((folder/'run_summary.json').read_text(encoding='utf-8'))
    if run['status']!='natural_match_ended' or run['error'] or not run['protected_ok']:
        raise RuntimeError('Incomplete/failed natural match')
    summary=json.loads((folder/'agent_summary.json').read_text(encoding='utf-8'))
    if summary['invalid_rows']: raise RuntimeError('Invalid raw rows retained; match not coverage-eligible')
    expected=json.loads((folder/'integrity.json').read_text(encoding='utf-8')); verify_files(folder,expected)
    tensors={k:array.array(t) for k,(t,_) in TENSOR_SPECS.items()}
    values={k:array.array('d') for k in VALUE_FIELDS}
    counters={k:collections.Counter() for k in ('modes','branches','near_far','phases','speed_bands','geometry')}
    channel_counts={k:collections.Counter() for k in CHANNELS if k!='steer'}
    seq={}; n=0; previous_time=None; phase_transitions=0; prediction_valid=0; prediction_empty=0
    missing_pending=0; prediction_reuse=0; previous_prediction_counter=None
    dt_over_100ms=0
    with (folder/'raw/trajectory.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line); n+=1
            if r['callback']!=n or r['match_id']!=folder.name or not r['valid']: raise RuntimeError('Invalid/omitted/mixed-match raw callback')
            if r['team']!=identity['team']: raise RuntimeError('Actual teacher side differs from planned side')
            if r['previous_elapsed']!=previous_time: raise RuntimeError('Callback-history mismatch')
            obs=reconstruct(r)
            if struct.pack('<18f',*obs)!=struct.pack('<18f',*r['observation18']): raise RuntimeError('18D observation reconstruction mismatch')
            a=r['teacher_controls']; action=mode(a)
            if action!=r['action_mode']: raise RuntimeError('Unsupported/mislabeled native controls')
            if r['callback_dt'] is not None:
                if r['callback_dt']!=r['T']-previous_time or r['callback_dt']<=0: raise RuntimeError('Callback clock mismatch')
                values['callback_dt'].append(r['callback_dt']); dt_over_100ms+=int(r['callback_dt']>.1)
            elif n!=1: raise RuntimeError('Missing noninitial callback dt')
            previous_time=r['T']
            tensors['observations'].extend(obs); tensors['analog_controls'].extend(a[:5]); tensors['buttons'].extend(a[5:]); tensors['mode'].append(action)
            tensors['callback'].append(n); tensors['frame'].append(r['frame']); tensors['time'].append(r['T']); tensors['dt'].append(0. if r['callback_dt'] is None else r['callback_dt'])
            counters['modes'][r['branch']]+=1; counters['branches'][r['teacher_logic_branch']]+=1; counters['phases'][r['match_phase']]+=1
            for k,v in zip(CHANNELS,a):
                if k=='steer': values['steer'].append(v)
                else: channel_counts[k][v]+=1
            speed=math.sqrt(sum(v*v for v in r['car']['velocity'])); values['car_speed'].append(speed)
            band='below_750' if speed<=750 else 'strict_flip_window_750_800' if speed<800 else '800_to_1410' if speed<=1410 else 'above_1410'
            counters['speed_bands'][band]+=1
            if r['ball_present']:
                distance=math.dist(r['car']['position'],r['ball']['position']); values['ball_distance'].append(distance)
                counters['near_far']['far' if distance>1500 else 'near']+=1
                from observation_contract import basis
                delta=[b-c for b,c in zip(r['ball']['position'],r['car']['position'])]
                relative=[sum(v*w for v,w in zip(delta,axis)) for axis in basis(r['car']['rotation_pyr'])]
                for key,v in zip(('ball_forward','ball_right','ball_up'),relative): values[key].append(v)
                counters['geometry']['ahead' if relative[0]>=0 else 'behind']+=1
                counters['geometry']['right' if relative[1]>0 else 'left' if relative[1]<0 else 'center']+=1
                counters['geometry']['above' if relative[2]>0 else 'below_or_level']+=1
                values['ball_height'].append(r['ball']['position'][2])
                values['ball_speed'].append(math.sqrt(sum(v*v for v in r['ball']['velocity'])))
                values['ball_spin'].append(math.sqrt(sum(v*v for v in r['ball']['angular_velocity'])))
            else: counters['near_far']['missing_ball']+=1
            prediction_valid+=int(obs[9]==1); prediction_empty+=int(r['prediction_count']==0)
            if obs[9]: values['prediction_horizon'].append(r['observation_prediction']['timestamp']-r['T'])
            receipt=r['prediction_receipt']; counter=None if receipt is None else receipt['counter']
            prediction_reuse+=int(counter is not None and counter==previous_prediction_counter); previous_prediction_counter=counter
            phase_transitions+=int(r['phase_transition'] is not None)
            sid=r['sequence_id_after']
            if sid is not None:
                entry=seq.setdefault(sid,dict(start_seen=False,completion_seen=False,consumed_phases=set(),pending_match_phases=set(),jump_callbacks=0,dodge_callbacks=0,missing_pending_callbacks=0))
                entry['start_seen']|=r['teacher_selection']['flip_trigger']
                consumed=r['teacher_selection']['consumed_phase']
                if consumed is not None: entry['consumed_phases'].add(consumed)
                entry['jump_callbacks']+=int(action==2); entry['dodge_callbacks']+=int(action==3)
                before,after=r['sequence_before'],r['sequence_after']
                entry['completion_seen']|=after is not None and after['done'] and (before is None or not before['done'])
                if before is not None and not before['done']:
                    entry['pending_match_phases'].add(r['match_phase'])
                    if not r['ball_present']:
                        missing_pending+=1; entry['missing_pending_callbacks']+=1
                        if before!=after: raise RuntimeError('Missing-ball teacher memory changed unexpectedly')
    if n!=summary['callbacks'] or n!=json.loads((folder/'recorder_closed.json').read_text())['callbacks']: raise RuntimeError('Recorder close/count mismatch')
    if prediction_reuse!=summary['prediction_reuse_callbacks']: raise RuntimeError('Receipt reuse mismatch')
    existing=folder/'processed/manifest.json'
    dtype={'f':'float32','d':'float64','B':'uint8','Q':'uint64'}
    entries={}
    if existing.exists():
        manifest=json.loads(existing.read_text(encoding='utf-8'))
        if manifest['split']!=identity['split'] or manifest['features']!=list(FEATURES) or manifest['channels']!=list(CHANNELS): raise RuntimeError('Frozen processed contract mismatch')
        for k,buf in tensors.items():
            h=hashlib.sha256(buf.tobytes()).hexdigest()
            if manifest['tensors'][k]['sha256']!=h or sha(folder/'processed'/manifest['tensors'][k]['file'])!=h: raise RuntimeError('Exact raw/tensor round-trip mismatch')
    else:
        if identity['is_pilot']: raise RuntimeError('Pilot processed tensors missing; never rebuild pilot')
        out=folder/'processed'; out.mkdir(exist_ok=False)
        for k,buf in tensors.items():
            t,width=TENSOR_SPECS[k]; path=out/(k+'.bin')
            with path.open('wb') as f: buf.tofile(f)
            entries[k]=dict(file=path.name,dtype=dtype[t],shape=[n,width] if width!=1 else [n],bytes=path.stat().st_size,sha256=sha(path))
        write_json(existing,dict(match_id=folder.name,split=identity['split'],features=list(FEATURES),channels=list(CHANNELS),endianness='little',tensors=entries,
            raw_integrity=expected,hidden_teacher_state_not_in_tensors=True,temporal_policy='All actual processed callbacks in order; no resampling, rebalancing, chunks or phase resets.'))
    serial=[]
    for sid,s in seq.items(): serial.append(dict(sequence_id=sid,**{k:sorted(v) if isinstance(v,set) else v for k,v in s.items()}))
    stats=dict(**identity,match_id=folder.name,callbacks=n,invalid_rows=0,dropped_processed_rows=0,
        modes=dict(counters['modes']),branches=dict(counters['branches']),near_far=dict(counters['near_far']),phases=dict(counters['phases']),
        speed_bands=dict(counters['speed_bands']),geometry_sectors=dict(counters['geometry']),
        distributions={k:distribution(v) for k,v in values.items()},
        sequences=serial,sequence_starts=sum(s['start_seen'] for s in seq.values()),sequence_completions=sum(s['completion_seen'] for s in seq.values()),
        complete_sequences_with_all_four_phases=sum(s['completion_seen'] and s['consumed_phases']=={0,1,2,3} for s in seq.values()),
        missing_ball_pending_callbacks=missing_pending,phase_transitions=phase_transitions,
        prediction_reuse=prediction_reuse,prediction_valid_callbacks=prediction_valid,prediction_empty_callbacks=prediction_empty,
        callback_dt_over_100ms=dt_over_100ms,record_path_max_seconds=summary['maximum_record_seconds'],
        integrity=dict(raw_verified=True,tensor_roundtrip_exact=True,manifest_sha256=sha(existing)))
    return stats,values,channel_counts

def review():
    if sys.byteorder!='little': raise RuntimeError('Little-endian tensor contract required')
    preserve.verify_sources(); pilot_anchor=anchor()
    p=None if not PLAN.exists() else json.loads(PLAN.read_text(encoding='utf-8'))
    plan_authorized=p is not None and p.get('user_approved',False)
    if plan_authorized: p=validate_plan()
    approved=plan_authorized and p.get('coverage_targets_approved',False)
    targets={} if not approved else p['coverage_targets']
    source=[]
    for name,split in SPLITS.items():
        source.append((PILOT/name,dict(split=split,teacher_side='Orange',team=1,
            opponent_id=p['pilot_opponent_id'] if plan_authorized else 'pilot_human_unmapped',
            play_session_id='pilot_20261004_session_identity_not_separately_recorded',play_style='Natural pilot; style not separately recorded',is_pilot=True)))
    for folder in sorted(MATCHES.glob('v10_*')):
        if not (folder/'run_summary.json').exists(): continue
        meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        source.append((folder,{k:meta[k] for k in ('split','teacher_side','team','opponent_id','play_session_id','play_style')}|dict(is_pilot=False)))
    matches=[]; failures=[]; values={k:array.array('d') for k in VALUE_FIELDS}; channel_counts={k:collections.Counter() for k in CHANNELS if k!='steer'}
    for folder,identity in source:
        print('V10 auditing:',folder.name,flush=True)
        try:
            stats,v,channels=audit(folder,identity)
            matches.append(stats)
            for k in values: values[k].extend(v[k])
            for k in channel_counts: channel_counts[k].update(channels[k])
        except Exception as e:
            summary_path=folder/'agent_summary.json'
            s={} if not summary_path.exists() else json.loads(summary_path.read_text())
            failures.append(dict(match_id=folder.name,error=repr(e),recorded_callbacks=s.get('callbacks'),invalid_rows=s.get('invalid_rows'),eligible=False))
    aggregate={k:collections.Counter() for k in ('modes','branches','near_far','phases','speed_bands','geometry_sectors')}
    for m in matches:
        for k in aggregate: aggregate[k].update(m[k])
    group={key:{} for key in ('split','teacher_side','opponent_id','play_session_id')}
    for key in group:
        for m in matches:
            item=group[key].setdefault(m[key],dict(matches=0,callbacks=0,jump=0,front_dodge=0,sequence_starts=0,sequence_completions=0,sides=set()))
            item['matches']+=1; item['callbacks']+=m['callbacks']; item['jump']+=m['modes'].get('jump',0); item['front_dodge']+=m['modes'].get('front-dodge',0)
            item['sequence_starts']+=m['sequence_starts']; item['sequence_completions']+=m['sequence_completions']; item['sides'].add(m['teacher_side'])
        for item in group[key].values(): item['sides']=sorted(item['sides'])
    answers={}
    if approved:
        answers['jump_coverage_sufficient']=aggregate['modes']['jump']>=targets['jump_callbacks_total']
        answers['front_dodge_coverage_sufficient']=aggregate['modes']['front-dodge']>=targets['dodge_callbacks_total']
        answers['teacher_side_diversity_sufficient']=all(group['teacher_side'].get(s,{}).get('matches',0)>=targets['min_matches_each_side_total'] for s in ('Blue','Orange')) and all(len(group['split'].get(s,{}).get('sides',[]))==2 for s in ('train','validation','test'))
        # Opponent identities are declared by the user; sessions are not persons.
        opponents=set(m['opponent_id'] for m in matches if m['opponent_id'] in p['opponents'])
        answers['opponent_session_diversity_sufficient']=len(opponents)>=targets['min_human_opponents']
        answers['temporal_sequence_coverage_sufficient']=sum(m['complete_sequences_with_all_four_phases'] for m in matches)>=targets['complete_sequences_total'] and all(sum(m['complete_sequences_with_all_four_phases'] for m in matches if m['split']==split)>=minimum for split,minimum in targets['completed_sequences_by_split'].items())
    else:
        answers={key:None for key in ('jump_coverage_sufficient','front_dodge_coverage_sufficient','teacher_side_diversity_sufficient','opponent_session_diversity_sufficient','temporal_sequence_coverage_sufficient')}
    covered=approved and all(answers.values()) and not failures
    answers['dataset_ready_to_freeze_for_bc']='Yes under the approved collection criteria, subject to user freeze review; no automatic freeze or guarantee of BC success' if covered else 'No'
    native={}
    for k in CHANNELS:
        if k=='steer': native[k]=distribution(values['steer'])
        else:
            c=channel_counts[k]; n=sum(c.values())
            native[k]=dict(count=n,values={str(v):n for v,n in sorted(c.items())},minimum=min(c) if c else None,maximum=max(c) if c else None,mean=sum(v*n for v,n in c.items())/n if n else None)
    remaining=[k for k,v in answers.items() if k!='dataset_ready_to_freeze_for_bc' and v is not True]
    if failures: remaining.append('Failed/incomplete matches excluded; inspect each failure before freeze')
    remaining+=['Coverage thresholds are provisional heuristics, not proof of learned recurrent behavior or generalization.',
        'Natural collection does not guarantee extreme states or very long pending-sequence interruptions.',
        'Match-level splitting does not establish opponent/session-disjoint evaluation; report identities explicitly.']
    report=dict(status='coverage_targets_met_awaiting_review' if covered else 'collection_in_progress',
        match_plan_authorized=plan_authorized,coverage_targets_approved=approved,coverage_targets=targets,answers=answers,remaining_gaps=remaining,
        opponent_session_assessment='The approved independent-person target gates this answer. Recorded play sessions are reported separately; no extra numeric session stopping threshold was approved. Historical pilot sitting identities were not separately captured and are not invented.',
        eligible_matches=len(matches),new_matches=sum(not m['is_pilot'] for m in matches),eligible_callbacks=sum(m['callbacks'] for m in matches),
        aggregate_counts={k:dict(v) for k,v in aggregate.items()},native_action_channels=native,
        distributions={k:distribution(v) for k,v in values.items()},grouped_coverage=group,matches=matches,failures=failures,
        sequence_starts=sum(m['sequence_starts'] for m in matches),sequence_completions=sum(m['sequence_completions'] for m in matches),
        complete_sequences_with_all_four_phases=sum(m['complete_sequences_with_all_four_phases'] for m in matches),
        prediction_reuse=sum(m['prediction_reuse'] for m in matches),invalid_eligible_rows=0,dropped_processed_eligible_rows=0,
        exact_match_split={s:[m['match_id'] for m in matches if m['split']==s] for s in ('train','validation','test')},
        pilot_files_unchanged=len(pilot_anchor),protected_hashes_unchanged=43,
        limitations=['All figures describe actually processed RLBot callbacks, not lossless transport or undelivered physics ticks.',
            'Hidden sequence labels are diagnostic only; 18D features/actions remain unchanged. No resampling or rebalancing.',
            'Opponent and play-session identities are declared by the user, not independently detected.'],
        dataset_frozen=False,training_started=False)
    path=ROOT/'training/reports/v10_collection_review_20261005.json'; write_json(path,report)
    lines=['# V10 natural demonstration collection review','',f"Status: **{report['status']}**. Dataset not frozen; no training.",'',
        f"Eligible matches: {len(matches)} ({report['new_matches']} new); callbacks: {report['eligible_callbacks']:,}.",
        f"Cumulative Jump: {aggregate['modes']['jump']:,}; Front dodge: {aggregate['modes']['front-dodge']:,}; sequence starts/completions: {report['sequence_starts']}/{report['sequence_completions']}.",'',
        '## Coverage decisions','', '| Question | Current answer |','|---|---|']
    lines += [f'| {k} | {v if v is not None else "Unresolved: approved targets/roster required"} |' for k,v in answers.items()]
    lines += ['', '## Match assignments','', '| Match | Split | Teacher side | Opponent | Play session | Callbacks | Jump | Dodge | Completed sequences |','|---|---|---|---|---|---:|---:|---:|---:|']
    lines += [f"| {m['match_id']} | {m['split']} | {m['teacher_side']} | {m['opponent_id']} | {m['play_session_id']} | {m['callbacks']} | {m['modes'].get('jump',0)} | {m['modes'].get('front-dodge',0)} | {m['sequence_completions']} |" for m in matches]
    lines += ['', '## Integrity and distributions','',
        'Every eligible raw row was audited: contiguous callback index, actual side, native action support, canonical elapsed/dt, bitwise 18D reconstruction and exact raw/tensor round-trip. Raw/tensor SHA256 and protected sources were checked. Original pilot files were not rewritten.',
        'The JSON includes side/opponent/session/split grouping, all native channels, rare modes, diagnostic per-sequence phase coverage, near/far, missing-ball, match phases, speed bands and distributions, car-local ball geometry, ball speed/spin/height, prediction horizon/reuse and invalid/incomplete match evidence.',
        '', '## Remaining gaps','']+[f'- {gap}' for gap in remaining]+['',
        'Stop after coverage review. No synthetic trajectories, controlled probes, prediction substitutions, additional matches, dataset freeze or training start automatically.']
    path.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    anchor(); preserve.verify_sources()
    print('V10 REVIEW:',report['status'],'Jump',aggregate['modes']['jump'],'Dodge',aggregate['modes']['front-dodge'],flush=True)
    print('Report:',path.with_suffix('.md'),flush=True)
    return report
