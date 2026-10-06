"""Lossless callback indexing and tensor export, never training."""
import array
import collections
import json
import math
import struct
import sys
from common import ROOT, PILOT, SPLITS, FEATURES, CHANNELS, sha, preserve
from observation_contract import build, basis


def reconstruct(r):
    car = r['car']; p = r['observation_prediction']; previous = r['previous_submission']
    return list(build(dict(position=car['position'],velocity=car['velocity'],basis=basis(car['rotation_pyr']),
        ball_present=r['ball_present'],ball_position=None if r['ball'] is None else r['ball']['position']),
        dict(elapsed=r['T'],previous_elapsed=r['previous_elapsed'],
        previous_mode=0 if previous is None else previous['mode'],
        previous_steer=0. if previous is None else previous['controls'][1],
        prediction_valid=p is not None,prediction_position=None if p is None else p['position'],
        selected_time=None if p is None else p['timestamp'],first_time=r['prediction_first_time'])))


def distribution(values):
    v = sorted(values)
    if not v: return dict(count=0)
    def pct(q):
        x=(len(v)-1)*q; i=int(x)
        return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)
    return dict(count=len(v),minimum=v[0],maximum=v[-1],mean=sum(v)/len(v),
        p50=pct(.5),p90=pct(.9),p95=pct(.95),p99=pct(.99))


def process(folder):
    if sys.byteorder!='little': raise RuntimeError('Declared tensor format requires little endian')
    run=json.loads((folder/'run_summary.json').read_text())
    if run['status']!='natural_match_ended': raise RuntimeError('Incomplete match is not eligible: '+folder.name)
    if run['error'] or not run['protected_ok']: raise RuntimeError('Collection cleanup/protection error: '+folder.name)
    expected=json.loads((folder/'integrity.json').read_text())
    for name,h in expected.items():
        if sha(folder/name)!=h: raise RuntimeError('Raw integrity mismatch: '+name)
    out=folder/'processed'
    out.mkdir(exist_ok=False)
    specs={'observations':('f',18), 'analog_controls':('f',5), 'buttons':('B',3),
           'mode':('B',1), 'callback':('Q',1), 'frame':('Q',1), 'time':('d',1), 'dt':('d',1)}
    buffers={k:array.array(t) for k,(t,_) in specs.items()}
    counts={k:collections.Counter() for k in ('branches','logic_branches','near_far','phases')}
    channels={k:[] for k in CHANNELS}; dt=[]; invalid=[]; n=0; flips=0; missing=0; completions=0
    with (folder/'raw/trajectory.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line); n+=1
            if r['callback']!=n: raise RuntimeError('Callback missing/duplicate: '+str(n))
            if r['match_id']!=folder.name: raise RuntimeError('Cross-match row contamination')
            if not r['valid']:
                invalid.append(dict(callback=n,errors=r['errors'])); continue
            obs=reconstruct(r)
            if struct.pack('<18f',*obs)!=struct.pack('<18f',*r['observation18']):
                raise RuntimeError('Observation reconstruction mismatch: '+str(n))
            a=r['teacher_controls']
            from common import mode
            if mode(a)!=r['action_mode']: raise RuntimeError('Action-label mismatch')
            buffers['observations'].extend(obs); buffers['analog_controls'].extend(a[:5]); buffers['buttons'].extend(a[5:])
            buffers['mode'].append(r['action_mode']); buffers['callback'].append(n); buffers['frame'].append(r['frame'])
            buffers['time'].append(r['T']); buffers['dt'].append(0. if r['callback_dt'] is None else r['callback_dt'])
            counts['branches'][r['branch']]+=1; counts['logic_branches'][r['teacher_logic_branch']]+=1
            counts['phases'][r['match_phase']]+=1
            if r['ball_present']:
                distance=math.dist(r['car']['position'],r['ball']['position'])
                counts['near_far']['far' if distance>1500 else 'near']+=1
            else: missing+=1
            if r['callback_dt'] is not None: dt.append(r['callback_dt'])
            flips+=int(r['teacher_selection']['flip_trigger'])
            before,after=r['sequence_before'],r['sequence_after']
            completions+=int(after is not None and after['done'] and (before is None or not before['done']))
            for k,v in zip(CHANNELS,a): channels[k].append(float(v))
    if invalid:
        (out/'invalid_rows.json').write_text(json.dumps(invalid,indent=2))
        raise RuntimeError('Invalid rows preserved in raw; match rejected for tensors: '+folder.name)
    summary=json.loads((folder/'agent_summary.json').read_text())
    if summary['callbacks']!=n: raise RuntimeError('Recorded callback count differs from agent summary')
    tensors={}
    dtype={'f':'float32','d':'float64','B':'uint8','Q':'uint64'}
    for name,buf in buffers.items():
        t,width=specs[name]; path=out/(name+'.bin')
        with path.open('wb') as f: buf.tofile(f)
        shape=[n,width] if width!=1 else [n]
        tensors[name]=dict(file=path.name,dtype=dtype[t],shape=shape,sha256=sha(path),bytes=path.stat().st_size)
    stats=dict(match_id=folder.name,split=SPLITS[folder.name],callbacks=n,
        branches=dict(counts['branches']),teacher_logic_branches=dict(counts['logic_branches']),
        near_far=dict(counts['near_far']),phases=dict(counts['phases']),missing_ball=missing,
        sequence_starts=flips,sequence_completions=completions,callback_dt_seconds=distribution(dt),
        action_channels={k:dict(**distribution(v),zero_count=sum(x==0 for x in v),
            nonzero_count=sum(x!=0 for x in v),minus_one_count=sum(x==-1 for x in v),plus_one_count=sum(x==1 for x in v)) for k,v in channels.items()},
        invalid_rows=0,dropped_processed_rows=0,transport_loss_claim='Unknown; received callbacks only',
        prediction_reuse_callbacks=summary['prediction_reuse_callbacks'])
    stats['missing_ball_fraction']=missing/n if n else None
    stats['sequence_starts_per_1000_callbacks']=flips*1000/n if n else None
    stats['branch_fractions']={k:v/n for k,v in counts['branches'].items()} if n else {}
    stats['callback_dt_nonpositive_count']=sum(v<=0 for v in dt)
    stats['callback_dt_over_100ms_count']=sum(v>.1 for v in dt)
    stats['callback_dt_over_1s_count']=sum(v>=1 for v in dt)
    manifest=dict(match_id=folder.name,split=SPLITS[folder.name],endianness='little',
        features=list(FEATURES),channels=list(CHANNELS),tensors=tensors,stats=stats,
        raw_integrity=expected,hidden_teacher_state_not_in_tensors=True,
        temporal_policy='One row per processed callback, dt=0 only first callback; no chunks/resampling or phase resets. Buttons remain uint8 0/1; analog controls float32. All match rows share one split.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def report():
    preserve.verify_sources()
    matches=[]; failures=[]
    for match in SPLITS:
        folder=PILOT/match
        if not (folder/'run_summary.json').exists(): continue
        try:
            if (folder/'processed/manifest.json').exists():
                m=json.loads((folder/'processed/manifest.json').read_text())
                for n,h in m['raw_integrity'].items():
                    if sha(folder/n)!=h: raise RuntimeError('Raw hash changed: '+n)
                for v in m['tensors'].values():
                    if sha(folder/'processed'/v['file'])!=v['sha256']: raise RuntimeError('Tensor hash changed')
            else: m=process(folder)
            matches.append(m)
        except Exception as e: failures.append(dict(match_id=match,error=repr(e)))
    complete=len(matches)==3 and not failures
    combined={key:collections.Counter() for key in ('branches','teacher_logic_branches','near_far','phases')}
    for m in matches:
        for key in combined: combined[key].update(m['stats'][key])
    diversity=dict(action_modes={k:combined['branches'][k] for k in ('neutral','chase','jump','front-dodge')},
        near_far=dict(combined['near_far']),match_phases=dict(combined['phases']),
        sequence_starts=sum(m['stats']['sequence_starts'] for m in matches),
        missing_ball_callbacks=sum(m['stats']['missing_ball'] for m in matches))
    basic_diversity=all(diversity['action_modes'].values()) and all(combined['near_far'].get(k,0)>0 for k in ('near','far'))
    sufficient=(complete and basic_diversity)
    r=dict(status='pilot_complete_awaiting_review' if complete else 'pilot_incomplete',
        matches_collected=len(matches),callbacks_collected=sum(m['stats']['callbacks'] for m in matches),
        matches=[m['stats'] for m in matches],failures=failures,
        exact_split={split:[match] for match,split in SPLITS.items()},
        split_unit='Whole match, assigned before collection; never callbacks',
        integrity='Every raw and tensor SHA256 checked for accepted matches',protected_hashes_unchanged=43,
        observed_diversity=diversity,
        larger_collection_functionally_supported=sufficient,
        sufficient_for_larger_collection=('The complete pilot supports a larger collection experiment: all four action modes and both near/far cases were observed, and integrity/reconstruction passed. User review is still required; three matches do not establish adequate coverage, class balance or statistical generalization.' if sufficient else 'Not established: '+('the pilot is incomplete.' if not complete else 'basic action-mode or near/far diversity is missing. Review the observed groups before expanding.')),
        training_started=False,candidate_deployed=False)
    p=ROOT/'training/reports/v9_dataset_pilot_20261004.json'
    p.write_text(json.dumps(r,indent=2)+'\n')
    lines=['# V9 teacher dataset pilot','',f"Status: **{r['status']}**. No model training.",'',
        f"Complete accepted matches: {len(matches)}; callbacks: {r['callbacks_collected']}.",'',
        'Real Rocket League / RLBot / unmodified original teacher. No controlled resets, receive pauses, synthetic trajectories or RocketSim.',
        'Every processed callback remains in raw records. Invalid/incomplete matches are rejected explicitly, never silently filtered into training tensors.',
        'Teacher private sequence state is diagnostic-only. The unchanged 18D features contain previous actually submitted controls, not private sequence labels.',
        'Processed tensors are little-endian typed binary arrays with explicit shapes/dtypes/hashes, separate from raw JSONL. No sequences/chunks are constructed yet.',
        '', '## Match splits', '', '| Match | Split | Callbacks | Flips | Missing ball |','|---|---|---:|---:|---:|']
    lines += [f"| {m['match_id']} | {m['split']} | {m['stats']['callbacks']} | {m['stats']['sequence_starts']} | {m['stats']['missing_ball']} |" for m in matches]
    lines += ['', 'Reserved split: pilot_01=train, pilot_02=validation, pilot_03=test. Incomplete matches are not counted.',
        '', '## Distributions and integrity', '', 'The accompanying JSON records per-match action-channel distributions (min/max/mean/percentiles and zero/±1 counts), output and source-branch distributions, near/far geometry, phases, flips/completions, missing-ball counts, callback-dt distribution, prediction reuse, invalid/dropped rows and all collection failures.',
        'All accepted raw inputs and exported tensors are hash-verified. All 43 protected sources remain unchanged. Packet/receipt counters document provenance; this does not claim lossless transport or coverage of undelivered ticks.',
        '', '## Larger collection decision', '', r['sufficient_for_larger_collection'],
        'Stop for pilot review; no additional collection or training is automatically approved.']
    if failures: lines += ['', 'Failures: '+json.dumps(failures)]
    p.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print('V9:',r['status'],'matches',len(matches),'callbacks',r['callbacks_collected'],flush=True)
    print('Reports:',p.with_suffix('.md'),flush=True)
    return r
