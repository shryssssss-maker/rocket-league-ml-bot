"""V4 only: independent temporal reconstruction versus untouched source."""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training/v3_boundary_test'))
import run_v3 as v3
flat, original = v3.flat, v3.original
from util.sequence import ControlStep as OriginalStep

DURATIONS = (.05, .05, .2, .8)
STEP_CONTROLS = (
    (0.,0.,0.,0.,0.,True,False,False),
    (0.,0.,0.,0.,0.,False,False,False),
    (0.,0.,-1.,0.,0.,True,False,False),
    (0.,0.,0.,0.,0.,False,False,False),
)
LIVE = ROOT/'training/live_reference_test/sessions/20261004_055830_563928'


@dataclass
class Step:
    duration: float
    output: tuple
    start_time: float | None = None

    def tick(self, now):
        if self.start_time is None:
            self.start_time = now
        return self.output, now-self.start_time > self.duration


class Sequence:
    """Independent code; never calls original Sequence or ControlStep."""
    def __init__(self):
        self.steps = [Step(d,c) for d,c in zip(DURATIONS, STEP_CONTROLS)]
        self.index = 0
        self.done = False

    def tick(self, now):
        if self.index >= len(self.steps):
            self.done = True
            return STEP_CONTROLS[3]
        output, finished = self.steps[self.index].tick(now)
        if finished:
            self.index += 1
            self.done = self.index == len(self.steps)
        # Even a finishing callback returns the OLD phase's controls.
        return output


class Reconstruction:
    def __init__(self):
        self.active_sequence = None
        self.flips = 0

    def output(self, packet, index):
        now = packet.match_info.seconds_elapsed
        if self.active_sequence is not None and not self.active_sequence.done:
            return self.active_sequence.tick(now)
        car = packet.players[index].physics
        velocity = v3.xyz(car.velocity)
        speed = math.sqrt(velocity[0]**2+velocity[1]**2+velocity[2]**2)
        if 750 < speed < 800:
            self.flips += 1
            self.active_sequence = Sequence()
            return self.active_sequence.tick(now)
        # V4 fixtures always keep the ball within 1500: no prediction lookup.
        position, ball = v3.xyz(car.location),v3.xyz(packet.balls[0].physics.location)
        delta = tuple(a-b for a,b in zip(ball,position))
        axes = v3.basis((car.rotation.pitch,car.rotation.yaw,car.rotation.roll))
        def dot(axis):
            return delta[0]*axis[0]+delta[1]*axis[1]+delta[2]*axis[2]
        steer = max(-1,min(1,5*math.atan2(dot(axes[1]),dot(axes[0]))))
        return (1.,steer,0.,0.,0.,False,False,False)


def state(agent):
    seq = agent.active_sequence
    if seq is None:
        return None
    return dict(index=seq.index,done=seq.done,
        starts=[step.start_time for step in seq.steps],
        durations=[step.duration for step in seq.steps])


def make_original(index):
    agent = object.__new__(original.MyBot)
    agent.index = index
    agent.active_sequence = None
    agent.boost_pad_tracker = v3.Tracker()
    agent.renderer = v3.Renderer()
    agent.ball_prediction = flat.BallPrediction()
    agent.v4_flips = 0
    def comm(*args):
        agent.v4_flips += 1
    agent.send_match_comm = comm
    return agent


def packet(now, number, team, wire):
    # Every 160 callbacks supplies a fresh trigger; speed/heading/ball target
    # vary while a pending sequence must suppress fresh chase/flip decisions.
    speed = 775. if number%160 == 0 or number%7 == 3 else (0. if number%3 == 0 else 1100.)
    pose = (0.,0.,0.) if number%2 == 0 else (.35,.7,-.4)
    p = v3.packet_for((0.,0.,17.),(700.,100. if number%2 == 0 else -200.,200.),
                      (0.,0.,speed),pose,team,team)
    p = flat.GamePacket(players=p.players,balls=p.balls,
        match_info=flat.MatchInfo(seconds_elapsed=now,frame_num=round(now*120)))
    if wire:
        p = flat.CorePacket.unpack(flat.CorePacket(p).pack()).message
    return p


def check_output(a,b,label):
    v3.require(len(a)==len(b)==8, 'Controller shape '+label)
    error = max(abs(x-y) for x,y in zip(a[:5],b[:5]))
    v3.require(all(math.isfinite(x) for x in a[:5]+b[:5]),'Nonfinite '+label)
    v3.require(error == 0, 'Analog mismatch '+label)
    v3.require(all(type(x) is bool and type(y) is bool and x==y for x,y in zip(a[5:],b[5:])), 'Button mismatch '+label)
    return error


def run_schedule(name,times,wire,team):
    source, adapted = make_original(team),Reconstruction()
    transitions,completions,returns_after_completion = [],0,0
    transition_count = 0
    transition_digest = hashlib.sha256()
    last_transition = None
    max_error = 0.
    previous_now = None
    for n, now in enumerate(times):
        p = packet(now,n,team,wire)
        actual_now = float(p.match_info.seconds_elapsed)
        v3.require(previous_now is None or actual_now > previous_now,'Non-increasing fixture clock')
        previous_now = actual_now
        before = state(source)
        flips_before = source.v4_flips
        out = v3.controls(source.get_output(p))
        candidate = list(adapted.output(p,team))
        label = f'{name}, wire={wire}, team={team}, callback={n}'
        max_error = max(max_error,check_output(out,candidate,label))
        after = state(source)
        v3.require(after == state(adapted),'Sequence state mismatch '+label)
        v3.require(source.v4_flips == adapted.flips,'Retrigger mismatch '+label)
        if before is not None and not before['done']:
            i = before['index']
            start = actual_now if before['starts'][i] is None else before['starts'][i]
            finished = actual_now-start > DURATIONS[i]
            v3.require(after['index'] == i+int(finished), 'Wrong strict transition '+label)
            v3.require(out == list(STEP_CONTROLS[i]), 'Finishing callback returned next phase '+label)
            v3.require(after['starts'][i] == start,'Wrong first-use timestamp '+label)
            if i+1 < 4:
                v3.require(after['starts'][i+1] is None,'Next phase started too early '+label)
        if before is not None and before['done']:
            returns_after_completion += 1
            v3.require(out[0] == 1 or source.v4_flips == flips_before+1,'No normal decision after completion '+label)
        changed = before != after
        if after is not None and after['done'] and (before is None or not before['done']):
            completions += 1
        if changed or flips_before != source.v4_flips:
            last_transition = dict(callback=n,elapsed=actual_now,controls=out,before=before,after=after,
                                   flip_count=source.v4_flips)
            transition_count += 1
            transition_digest.update((json.dumps(last_transition,separators=(',',':'))+'\n').encode())
            # All callbacks are asserted; keep bounded diagnostic examples.
            if len(transitions) < 24:
                transitions.append(last_transition)
    v3.require(completions > 0 and returns_after_completion > 0,'Incomplete schedule coverage '+name)
    return dict(name=name,wire=wire,team=team,callbacks=len(times),flips=source.v4_flips,
                completions=completions,callbacks_after_completion=returns_after_completion,
                maximum_analog_error=max_error,transition_count=transition_count,
                transition_sha256=transition_digest.hexdigest(),transitions=transitions,
                last_transition=last_transition,final_state=state(source))


def strict_step_tests():
    results = []
    for i,duration in enumerate(DURATIONS):
        for when,expected in ((math.nextafter(duration,-math.inf),False),
                              (duration,False),(math.nextafter(duration,math.inf),True)):
            source = OriginalStep(duration,flat.ControllerState(*STEP_CONTROLS[i]))
            adapted = Step(duration,STEP_CONTROLS[i])
            p = flat.GamePacket(match_info=flat.MatchInfo(seconds_elapsed=0.))
            source.tick(p)
            adapted.tick(0.)
            p = flat.GamePacket(match_info=flat.MatchInfo(seconds_elapsed=when))
            a = source.tick(p)
            b,done = adapted.tick(when)
            v3.require(a.done == done == expected,'Strict duration predicate mismatch')
            check_output(v3.controls(a.controls),list(b),'strict duration')
            v3.require(source.start_time == adapted.start_time == 0,'Strict anchor start mismatch')
            results.append(dict(step=i,duration=duration,elapsed=when,done=done))
    return results


def equality_sequence():
    # Original instance, first trigger at exactly zero. At .05 it stays in
    # Jump; next representable time finishes Jump but still returns Jump.
    # At the next callback Release starts, rather than borrowing Jump's timer.
    times = [0.,.05,math.nextafter(.05,math.inf),.06,.11,.12,.13,.33,.34,.35,1.15,1.16,1.17,1.18]
    # This fixture needs a constant trigger only at callback zero; later
    # synthetic speed variation does not alter a pending sequence.
    return run_schedule('explicit_equality_and_next_callback',times,False,0)


def history_witness():
    fresh, pending = make_original(0),make_original(0)
    trigger = packet(0.,0,0,False)
    pending.get_output(trigger)
    same = packet(.02,1,0,False)  # speed1100: fresh chases; pending jumps.
    a,b = v3.controls(fresh.get_output(same)),v3.controls(pending.get_output(same))
    v3.require(a != b and a[0] == 1 and b == list(STEP_CONTROLS[0]),'History witness failed')
    return dict(elapsed=.02,identical_current_packet=True,fresh_controls=a,pending_controls=b,
                fresh_state=state(fresh),pending_state=state(pending))


def identical_observation_witness():
    from observation_contract import live_adapter
    histories = [('release_finished',[.86,.92,.93,.99]),
                 ('coast_pending',[0.,.06,.07,.13,.14,.35,.36,.99])]
    agents, traces = [],[]
    for name,times in histories:
        source, adapted = make_original(0),Reconstruction()
        trace = []
        for n,now in enumerate(times):
            p = packet(now,0 if n==0 else 1,0,False)
            a,b = v3.controls(source.get_output(p)),list(adapted.output(p,0))
            check_output(a,b,'observation witness history')
            v3.require(state(source) == state(adapted),'Witness history state mismatch')
            trace.append(dict(elapsed=now,controls=a,state=state(source)))
        v3.require(trace[-1]['controls'] == list(STEP_CONTROLS[3]),'Expected previous Neutral')
        agents.append((source,adapted))
        traces.append(dict(name=name,history=trace))
    same = packet(1.,1,0,False)
    # Same packet, previous callback clock and transmitted Neutral action.
    # Private sequence state is intentionally not supplied to the builder.
    context = dict(previous_elapsed=.99,previous_mode=0,previous_steer=0.,
        prediction_valid=False,prediction_position=None,selected_time=None,first_time=None)
    observations = [list(live_adapter(same,0,dict(context))) for _ in agents]
    outputs = []
    for source,adapted in agents:
        a,b = v3.controls(source.get_output(same)),list(adapted.output(same,0))
        check_output(a,b,'identical observation current output')
        v3.require(state(source) == state(adapted),'Identical-observation state mismatch')
        outputs.append(a)
    v3.require(observations[0] == observations[1], 'Candidate observations differ')
    v3.require(outputs == [list(STEP_CONTROLS[2]),list(STEP_CONTROLS[3])], 'Temporal ambiguity witness failed')
    return dict(elapsed=1.,previous_elapsed=.99,previous_mode='Neutral',
        observation=observations[0],identical_18D_observations=True,
        outputs=outputs,histories=traces,
        meaning='Same 18D candidate including previous action/delta requires Front dodge versus Coast Neutral depending on earlier history.')


def mixed_schedules():
    summary_path,event_path = LIVE/'bot_summary.json',LIVE/'bot_events.jsonl'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    intervals = []
    for line in event_path.read_text(encoding='utf-8').splitlines():
        event = json.loads(line)
        interval = event.get('interval')
        if interval is not None:
            v3.require(interval['frame_gap'] in range(1,6) and interval['game_dt'] > 0,'Bad retained live interval')
            intervals.append(interval)
    v3.require(len(intervals)>0,'No retained live intervals')
    def accumulate(deltas):
        result = [0.]
        for delta in deltas:
            result.append(result[-1]+delta)
        return result
    histogram = {int(k):v for k,v in summary['frame_gap_histogram'].items()}
    gaps = [gap for gap,count in histogram.items() for _ in range(count)]
    random.Random(20261004).shuffle(gaps)
    v3.require(Counter(gaps) == Counter(histogram),'Histogram schedule mismatch')
    provenance = dict(session=str(LIVE.relative_to(ROOT)),retained_intervals=len(intervals),
        retained_gap_histogram=dict(Counter(x['frame_gap'] for x in intervals)),
        full_gap_histogram=histogram,synthetic_seed=20261004,
        original_complete_callback_order_available=False,
        note='Concatenated retained individual intervals preserve only their logged order, not adjacency. Full histogram shuffle is synthetic.',
        sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (summary_path,event_path)})
    return [('retained_live_intervals_concatenated',accumulate(x['game_dt'] for x in intervals)),
            ('synthetic_exact_live_histogram',accumulate(x/120 for x in gaps))],provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--approved-mixed-proxies',action='store_true',
                        help='Only use after explicit approval of incomplete live-order proxy coverage')
    args = parser.parse_args()
    started = time.perf_counter()
    revision = subprocess.run(['git','-C',str(ROOT/'python-example-original'),'rev-parse','HEAD'],
                              text=True,capture_output=True,check=True).stdout.strip()
    v3.require(revision == v3.REVISION,'Reference revision changed')
    before,manifest = v3.hashes()
    v3.require(before == manifest,'Protected hashes mismatch')
    schedules = [(f'constant_gap_{gap}',[n*gap/120 for n in range(721)]) for gap in range(1,6)]
    provenance = None
    if args.approved_mixed_proxies:
        mixed,provenance = mixed_schedules()
        schedules += mixed
    runs = []
    for name,times in schedules:
        for wire in (False,True):
            for team in (0,1):
                runs.append(run_schedule(name,times,wire,team))
    runs.append(equality_sequence())
    anchors,witness = strict_step_tests(),history_witness()
    observation_witness = identical_observation_witness()
    after,_ = v3.hashes()
    v3.require(after == before,'Protected files changed')
    report = dict(experiment='V4',status='PASS' if args.approved_mixed_proxies else 'PARTIAL_FIXED_SCHEDULES_PASS',
        reference_revision=revision,independent_sequence_implementation=True,
        callbacks=sum(x['callbacks'] for x in runs),schedule_runs=len(runs),
        strict_duration_anchors=anchors,history_witness=witness,mixed_provenance=provenance,
        identical_observation_witness=observation_witness,
        maximum_analog_error=max(x['maximum_analog_error'] for x in runs),
        sequence_index_done_start_times_exact=True,buttons_exact=True,protected_hashes=len(before),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        elapsed_seconds=time.perf_counter()-started,runs=runs,
        scope='Ball-present callbacks; regular schedules use gaps 1-5 ticks, explicit duration anchors separately. No phase transitions, prediction lookup, physics, match or ML.')
    path = ROOT/'training/reports/v4_temporal_equivalence_20261004.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f"V4 {report['status']}: {report['callbacks']} callbacks, {len(runs)} runs; maximum analog error {report['maximum_analog_error']}.")
    print(f"State timestamps/index/done exact; {len(anchors)} strict-duration anchors; {len(before)} protected hashes unchanged.")
    print(f"Internal duration: {report['elapsed_seconds']:.2f}s. Report: {path}")


if __name__ == '__main__':
    main()
