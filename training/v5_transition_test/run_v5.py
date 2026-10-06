"""V5 only: ball absence, phase/position transitions and interrupted timers."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'training/v4_temporal_test'))
import run_v4 as v4
v3,flat = v4.v3,v4.flat
NEUTRAL = list(v4.STEP_CONTROLS[3])
PREFIXES = {
    'Jump_started':[0.],
    'Release_unstarted':[0.,.06],
    'Release_started':[0.,.06,.07],
    'Dodge_unstarted':[0.,.06,.07,.13],
    'Dodge_started':[0.,.06,.07,.13,.14],
    'Coast_unstarted':[0.,.06,.07,.13,.14,.35],
    'Coast_started':[0.,.06,.07,.13,.14,.35,.36],
}


class Reconstruction(v4.Reconstruction):
    def output(self,packet,index):
        # Source's availability check precedes sequence.tick. No timer/reset
        # or phase-specific rule is added to the existing independent machine.
        if not packet.balls:
            return tuple(NEUTRAL)
        return super().output(packet,index)


def make_packet(now,team,wire,present=True,phase='Active',speed=1100.,teleport=False,score=0):
    position = (2048.,2560.,17.) if teleport else (0.,0.,17.)
    base = v3.packet_for(position,(700.,100.,200.),(0.,0.,speed),
                        (.35,.7,-.4) if teleport else (0.,0.,0.),team,team)
    p = flat.GamePacket(players=base.players,balls=base.balls if present else [],
        teams=[flat.TeamInfo(team_index=0,score=score),flat.TeamInfo(team_index=1,score=0)],
        match_info=flat.MatchInfo(seconds_elapsed=now,frame_num=round(now*120),
                                  match_phase=getattr(flat.MatchPhase,phase)))
    if wire:
        p = flat.CorePacket.unpack(flat.CorePacket(p).pack()).message
    return p


class Trial:
    def __init__(self,name,team,wire):
        self.name,self.team,self.wire = name,team,wire
        self.source,self.adapted = v4.make_original(team),Reconstruction()
        self.counts = Counter()
        self.examples = []
        self.digest = hashlib.sha256()
        self.now = None

    def call(self,now,present=True,phase='Active',speed=1100.,teleport=False,score=0,label=''):
        p = make_packet(now,self.team,self.wire,present,phase,speed,teleport,score)
        actual_now = float(p.match_info.seconds_elapsed)
        v3.require(self.now is None or actual_now > self.now,'Non-increasing test clock')
        self.now = actual_now
        before = v4.state(self.source)
        old_sequence,old_candidate = self.source.active_sequence,self.adapted.active_sequence
        old_flips = self.source.v4_flips
        a,b = v3.controls(self.source.get_output(p)),list(self.adapted.output(p,self.team))
        v4.check_output(a,b,f'{self.name}/{label}')
        after = v4.state(self.source)
        v3.require(after == v4.state(self.adapted),'Sequence snapshot mismatch')
        v3.require(self.source.v4_flips == self.adapted.flips,'Flip count mismatch')
        v3.require(a[3:5] == [0,0] and a[6:] == [False,False],'Unsupported controls')
        self.counts['callbacks'] += 1
        if not present:
            self.counts['missing_ball'] += 1
            v3.require(a == NEUTRAL,'Missing ball must return Neutral')
            v3.require(after == before and self.source.v4_flips == old_flips,'Missing ball mutated state')
            v3.require(self.source.active_sequence is old_sequence and self.adapted.active_sequence is old_candidate,
                       'Missing ball reset sequence identity')
        elif before is not None and not before['done']:
            self.counts['pending_ball_present'] += 1
            i = before['index']
            start = actual_now if before['starts'][i] is None else before['starts'][i]
            finished = actual_now-start > v4.DURATIONS[i]
            v3.require(after['index'] == i+int(finished),'Must advance at most one step')
            v3.require(after['starts'][i] == start,'Incorrect timer survival/first use')
            v3.require(a == list(v4.STEP_CONTROLS[i]),'Old step controls must survive finishing callback')
            v3.require(self.source.active_sequence is old_sequence and self.adapted.active_sequence is old_candidate,
                       'Phase or position change replaced pending sequence')
            v3.require(self.source.v4_flips == old_flips,'Pending sequence retriggered')
            if i < 3:
                v3.require(after['starts'][i+1] is None,'Next step started prematurely')
            if finished:
                self.counts['step_finishes'] += 1
        if after is not None and after['done'] and (before is None or not before['done']):
            self.counts['completions'] += 1
        if self.source.v4_flips > old_flips:
            self.counts['flip_starts'] += 1
        row = dict(elapsed=actual_now,phase=phase,ball_present=present,teleport=teleport,score=score,
                   label=label,controls=a,before=before,after=after)
        self.digest.update((json.dumps(row,separators=(',',':'))+'\n').encode())
        # Explicit interruptions/resumptions and state changes are bounded.
        if len(self.examples) < 12 and (label not in ('prefix','drain') or
                (label == 'drain' and after is not None and after['done'] and before != after)):
            self.examples.append(row)
        return row

    def seed(self,prefix):
        for n,now in enumerate(prefix):
            self.call(now,speed=775. if n==0 else 1100.,label='prefix')

    def finish(self):
        # Drain ball-present Active callbacks only, then confirm a new ordinary
        # decision. Completing the coast itself must still return Neutral.
        for _ in range(180):
            if self.source.active_sequence is not None and self.source.active_sequence.done:
                break
            self.call(self.now+1/60,label='drain')
        v3.require(self.source.active_sequence is not None and self.source.active_sequence.done,'Sequence did not finish')
        old = self.source.active_sequence
        row = self.call(self.now+1/60,label='normal_after_completion')
        v3.require(row['controls'][0] == 1 and self.source.active_sequence is old,'Normal Chase changed completed sequence')
        row = self.call(self.now+1/60,speed=775.,label='fresh_trigger_after_completion')
        v3.require(row['controls'] == list(v4.STEP_CONTROLS[0]) and self.source.active_sequence is not old,
                   'Fresh trigger failed to replace completed sequence')

    def result(self):
        return dict(name=self.name,team=self.team,wire=self.wire,counts=dict(self.counts),
                    examples=self.examples,trace_sha256=self.digest.hexdigest(),
                    final_state=v4.state(self.source))


def interrupted(prefix_name,prefix,gap,kind,team,wire):
    trial = Trial(f'{prefix_name}/{gap}/{kind}',team,wire)
    trial.seed(prefix)
    base = trial.now
    initial = v4.state(trial.source)
    if kind == 'missing_ball':
        phases,present = ['Active']*4,False
    elif kind == 'goal_replay_kickoff_missing':
        phases,present = ['GoalScored','Replay','Countdown','Kickoff'],False
    elif kind == 'phase_changes_ball_present':
        phases,present = ['GoalScored','Replay','Countdown','Kickoff'],True
    else:
        phases,present = [],True  # no callbacks until the long-gap resume
    for phase,fraction in zip(phases,(.1,.4,.7,1.)):
        trial.call(base+gap*fraction,present=present,phase=phase,
                   teleport=phase in ('Countdown','Kickoff'),score=1,label='interruption')
    resumed = trial.call(base+gap+1/120,phase='Active',teleport=True,score=1,label='resume')
    if kind != 'phase_changes_ball_present':
        i = initial['index']
        v3.require(resumed['controls'] == list(v4.STEP_CONTROLS[i]),'Resume must return interrupted phase controls')
        if initial['starts'][i] is None:
            v3.require(resumed['after']['index'] == i and resumed['after']['starts'][i] == resumed['elapsed'],
                       'Unstarted step borrowed interruption time')
        else:
            finished = resumed['elapsed']-initial['starts'][i] > v4.DURATIONS[i]
            v3.require(resumed['after']['index'] == i+int(finished),'Started step did not retain original timer')
    trial.finish()
    result = trial.result()
    result.update(prefix=prefix_name,gap=gap,kind=kind,interrupted_state=initial,resume=resumed)
    return result


def phase_matrix(team,wire):
    results = []
    for seed in ('fresh_chase','completed_sequence'):
        for presence in ('all_present','all_missing','alternating'):
            trial = Trial(f'phase_matrix/{seed}/{presence}',team,wire)
            if seed == 'completed_sequence':
                trial.seed([0.,.06,.07,.13,.14,.35,.36,1.3])
                v3.require(trial.source.active_sequence.done,'Bad completed prefix')
            else:
                trial.call(0.,label='fresh_chase')
            retained = v4.state(trial.source)
            # Match phase/score and physical spawn reset metadata do not affect
            # the source's own branch logic. No agent-lifetime restart implied.
            for n,phase in enumerate(('GoalScored','Replay','Countdown','Kickoff','Active','Paused','Ended','Inactive')):
                present = presence=='all_present' or (presence=='alternating' and n%2==0)
                row = trial.call(trial.now+.1,present=present,phase=phase,
                                 teleport=True,score=1,label='phase_matrix')
                v3.require(row['controls'][0] == (1 if present else 0),'Phase invented an output gate')
                v3.require(row['after'] == retained,'Phase invented a reset')
            results.append(trial.result())
    return results


def main():
    started = time.perf_counter()
    revision = subprocess.run(['git','-C',str(ROOT/'python-example-original'),'rev-parse','HEAD'],
                              text=True,capture_output=True,check=True).stdout.strip()
    v3.require(revision == v3.REVISION,'Reference revision changed')
    before,manifest = v3.hashes()
    v3.require(before == manifest,'Protected hash mismatch')
    cases = []
    for team in (0,1):
        for wire in (False,True):
            for name,prefix in PREFIXES.items():
                for gap in (.1,1.,5.,10.):
                    for kind in ('missing_ball','goal_replay_kickoff_missing','phase_changes_ball_present','no_callbacks_long_gap'):
                        cases.append(interrupted(name,prefix,gap,kind,team,wire))
            cases += phase_matrix(team,wire)
    after,_ = v3.hashes()
    v3.require(after == before,'Protected files changed')
    counts = Counter()
    for case in cases:
        counts.update(case['counts'])
    report = dict(experiment='V5',status='PASS',reference_revision=revision,
        cases=len(cases),interruption_cases=sum('prefix' in c for c in cases),
        phase_matrix_cases=sum('prefix' not in c for c in cases),counts=dict(counts),
        maximum_analog_difference=0.,buttons_exact=True,sequence_state_exact=True,
        protected_hashes=len(before),elapsed_seconds=time.perf_counter()-started,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Synthetic packets on the same agent lifetime; independent sequence machine with explicit missing-ball early return. No prediction, physics, match or ML.',
        trial_results=cases)
    path = ROOT/'training/reports/v5_transition_memory_20261004.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f"V5 PASS: {len(cases)} trials, {counts['callbacks']} callbacks; maximum analog difference 0.0.")
    print(f"Missing-ball callbacks {counts['missing_ball']}; sequence states exact; {len(before)} protected hashes unchanged.")
    print(f"Internal duration {report['elapsed_seconds']:.2f}s. Report: {path}")


if __name__ == '__main__':
    main()
