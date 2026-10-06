"""Approved V14 v2 boundary term only; no data access or model changes."""
import numpy as np
import torch

LAMBDA = 0.1
EPSILON = 1e-7


def eligibility(time, dt, present):
    time = np.asarray(time, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    present = np.asarray(present)
    if time.ndim != 1 or dt.shape != time.shape or present.shape != time.shape:
        raise ValueError('Adjacent metadata shape mismatch')
    if not np.isfinite(time).all() or not np.isfinite(dt).all():
        raise ValueError('Nonfinite callback time/dt')
    if len(dt) and dt[0] != 0:
        raise ValueError('First callback dt must preserve frozen zero')
    if not np.array_equal(np.diff(time), dt[1:]):
        raise ValueError('Frozen dt is not the exact canonical-time difference')
    out = np.zeros(len(time), dtype=bool)
    if len(time) > 1:
        out[1:] = (present[:-1] == 1) & (present[1:] == 1) & (np.diff(time) > 0) & (dt[1:] > 0) & (dt[1:] <= .1)
    return out


def loss(logits, labels, eligible, previous=None):
    """previous=(detached probabilities, preceding label) at chunk edges.

    Return boundary loss, detached next cache, support/clamp diagnostics.
    eligible is precomputed on the full actual match, never across matches.
    """
    if logits.dtype != torch.float32 or logits.ndim != 2 or logits.shape[1] != 4:
        raise ValueError('Expected float32 [callbacks,4] logits')
    n = len(logits)
    if not n or labels.dtype != torch.int64 or labels.shape != (n,):
        raise ValueError('Expected nonempty logits and int64 labels')
    if eligible.dtype != torch.bool or eligible.shape != (n,):
        raise ValueError('Expected boolean eligibility per current callback')
    if not torch.isfinite(logits).all() or torch.any((labels < 0) | (labels > 3)):
        raise ValueError('Invalid logits/teacher modes')
    p = torch.softmax(logits, dim=1)
    if not torch.isfinite(p).all():
        raise ValueError('Nonfinite mode probabilities')
    if previous is None:
        if bool(eligible[0]):
            raise ValueError('First match callback cannot have a preceding pair')
        prev_p = torch.cat((p[:1].detach(), p[:-1]), dim=0)
        prev_y = torch.cat((labels[:1], labels[:-1]), dim=0)
    else:
        cache_p, cache_y = previous
        if cache_p.requires_grad or cache_p.shape != (4,) or cache_p.dtype != torch.float32:
            raise ValueError('Cross-chunk probabilities must be detached float32 [4]')
        if not torch.isfinite(cache_p).all():
            raise ValueError('Invalid probability cache')
        prev_p = torch.cat((cache_p[None], p[:-1]), dim=0)
        prev_y = torch.cat((labels.new_tensor([cache_y]), labels[:-1]), dim=0)
    q = torch.stack(((1-prev_p[:, 2])*p[:, 2], prev_p[:, 2]*p[:, 0],
                     prev_p[:, 0]*p[:, 3], prev_p[:, 3]*p[:, 0]), dim=1)
    b = torch.stack(((prev_y != 2) & (labels == 2), (prev_y == 2) & (labels == 0),
                     (prev_y == 0) & (labels == 3), (prev_y == 3) & (labels == 0)), dim=1)
    q = q[eligible]
    b = b[eligible]
    stats = dict(eligible_pairs=len(q), cross_chunk_pairs=int(previous is not None and bool(eligible[0])),
                 empty_eligible_chunk=int(len(q) == 0), positive=[], negative=[], clamp_count=0)
    if len(q):
        stats['clamp_count'] = int(((q < EPSILON) | (q > 1-EPSILON)).sum().item())
    qbar = torch.clamp(q, EPSILON, 1-EPSILON)
    terms = []
    for k in range(4):
        pos, neg = b[:, k], ~b[:, k]
        stats['positive'].append(int(pos.sum().item()))
        stats['negative'].append(int(neg.sum().item()))
        groups = []
        if bool(pos.any()):
            groups.append(-torch.log(qbar[pos, k]).mean())
        if bool(neg.any()):
            groups.append(-torch.log1p(-qbar[neg, k]).mean())
        terms.append(torch.stack(groups).mean() if groups else logits.sum()*0)
    value = torch.stack(terms).mean()
    if not torch.isfinite(value):
        raise ValueError('Nonfinite boundary loss')
    stats['per_type_loss'] = [float(x.detach()) for x in terms]
    return value, (p[-1].detach().clone(), int(labels[-1])), stats


def cores(mode):
    """All observed J,N,D,N run-start patterns; no future supervision."""
    mode = np.asarray(mode)
    if not len(mode):
        return []
    starts = np.r_[0, np.flatnonzero(mode[1:] != mode[:-1])+1]
    result = []
    for j in range(len(starts)-3):
        edges = starts[j:j+4]
        if np.array_equal(mode[edges], [2, 0, 3, 0]):
            result.append(edges.astype(int).tolist())
    return result


def temporal(match, predicted_modes):
    """Preregistered per-match observed timed-core evidence, not training labels."""
    times, dt = match['time'], match['dt']
    eligible, excluded = [], []
    for edges in cores(match['mode']):
        a, z = edges[0], edges[-1]
        reasons = []
        if not np.all(match['obs'][a:z+1, 8] == 1): reasons.append('missing_ball')
        if np.any(dt[a:z+1] > .1): reasons.append('long_gap')
        if np.any(np.diff(times[a:z+1]) <= 0): reasons.append('nonincreasing_time')
        (excluded if reasons else eligible).append(dict(edges=edges, reasons=reasons))
    candidates = cores(predicted_modes)
    used, records = set(), []
    for item in eligible:
        edges = item['edges']; onset = times[edges[0]]
        options = [j for j,x in enumerate(candidates) if j not in used and abs(times[x[0]]-onset) <= 2/120]
        j = min(options, key=lambda i:(abs(times[candidates[i][0]]-onset), times[candidates[i][0]], i)) if options else None
        record = dict(teacher_edges=edges, teacher_times=times[edges].tolist(), matched=j is not None, success=False)
        if j is not None:
            used.add(j); pe = candidates[j]; err = times[pe]-times[edges]
            # Decoder maps Neutral to jump=false; require an actual release callback.
            release = pe[2] > pe[1] and np.all(np.asarray(predicted_modes)[pe[1]:pe[2]] == 0)
            record.update(predicted_edges=pe, predicted_times=times[pe].tolist(), edge_errors=err.tolist(),
                          duration_errors=(np.diff(times[pe])-np.diff(times[edges])).tolist(),
                          success=bool(release and np.all(np.abs(err) <= 2/120+1e-6)))
        records.append(record)
    minutes = float((times[-1]-times[0])/60) if len(times)>1 else 0.
    # Incomplete/censored or interrupted patterns are not silently successes.
    jump_starts = np.flatnonzero((match['mode'] == 2) & np.r_[True, match['mode'][:-1] != 2])
    complete_starts = {x[0] for x in cores(match['mode'])}
    incomplete=[]
    for i,start in enumerate(jump_starts):
        if int(start) in complete_starts:continue
        stop=int(jump_starts[i+1]) if i+1<len(jump_starts) else len(times)
        interval=slice(int(start),stop)
        incomplete.append(dict(jump_start=int(start),end_exclusive=stop,
            reaches_match_end=stop==len(times),missing_ball=bool(np.any(match['obs'][interval,8]!=1)),
            long_gap=bool(np.any(dt[interval]>.1)),reason='No four-consecutive-run J,N,D,N pattern; not an eligible success'))
    return dict(match_id=match['metadata']['match_id'], eligible=len(eligible), successful=sum(x['success'] for x in records),
                matched_ordered=sum(x['matched'] for x in records), predicted_cores=len(candidates),
                extra_cores=len(candidates)-len(used), minutes=minutes, cores=records,
                excluded_complete_patterns=excluded, jump_runs_without_complete_pattern=len(incomplete),
                incomplete_or_censored_jump_runs=incomplete)


def temporal_summary(matches):
    n = sum(x['eligible'] for x in matches); successes = sum(x['successful'] for x in matches)
    minutes = sum(x['minutes'] for x in matches); extras = sum(x['extra_cores'] for x in matches)
    def distribution(values):
        if not values:return None
        v = np.asarray(values, np.float64)
        return dict(median=float(np.median(v)), p95=float(np.quantile(v,.95)), maximum=float(np.max(v)))
    edges = [r['edge_errors'] for m in matches for r in m['cores'] if r['matched']]
    durations = [r['duration_errors'] for m in matches for r in m['cores'] if r['matched']]
    return dict(eligible=n, successful=successes, score=successes/n if n else None,
                ordering_only_rate=sum(x['matched_ordered'] for x in matches)/n if n else None,
                extra_cores=extras, validation_minutes=minutes, extra_cores_per_minute=extras/minutes if minutes else None,
                edge_errors={name:dict(signed=distribution([v[i] for v in edges]),absolute=distribution([abs(v[i]) for v in edges]))
                             for i,name in enumerate(('Jump','Release','Dodge','Coast'))},
                duration_errors={name:dict(signed=distribution([v[i] for v in durations]),absolute=distribution([abs(v[i]) for v in durations]))
                                 for i,name in enumerate(('Jump','Release','Dodge'))}, by_match=matches)
