"""Independent shadow reference. Never submits controls or uses a simulator."""
from dataclasses import dataclass
import math
from rlbot import flat

DURATIONS=(.05,.05,.2,.8)
CHANNELS=('throttle','steer','pitch','yaw','roll','jump','boost','handbrake')
STEPS=((0.,0.,0.,0.,0.,True,False,False),
       (0.,0.,0.,0.,0.,False,False,False),
       (0.,0.,-1.,0.,0.,True,False,False),
       (0.,0.,0.,0.,0.,False,False,False))


def xyz(v):
    return [float(v.x),float(v.y),float(v.z)]


def state(seq):
    return None if seq is None else dict(index=seq.index,done=seq.done,
        starts=[s.start_time for s in seq.steps],durations=[s.duration for s in seq.steps])


def controls(c):
    return [getattr(c,k) for k in CHANNELS]


def mode(c):
    if c[5] and c[2]==-1:
        return 'Front dodge'
    if c[5]:
        return 'Jump'
    if c[0]==1:
        return 'Chase'
    return 'Neutral'


@dataclass
class Step:
    duration:float
    start_time:float|None=None


class Sequence:
    def __init__(self):
        self.index=0
        self.done=False
        self.steps=[Step(d) for d in DURATIONS]

    def tick(self,T):
        if self.index>=4:
            self.done=True
            return flat.ControllerState(*STEPS[3]),None
        consumed=self.index
        step=self.steps[consumed]
        if step.start_time is None:
            step.start_time=T
        if T-step.start_time>step.duration:
            self.index+=1
            self.done=self.index==4
        return flat.ControllerState(*STEPS[consumed]),consumed


class Candidate:
    def __init__(self):
        self.active_sequence=None
        self.flips=0

    def output(self,packet,index,prediction):
        T=packet.match_info.seconds_elapsed
        detail=dict(branch=None,selected=None,consumed_phase=None)
        if not packet.balls:
            detail['branch']='missing_ball'
            return flat.ControllerState(),detail
        if self.active_sequence is not None and not self.active_sequence.done:
            detail['branch']='pending_sequence'
            output,detail['consumed_phase']=self.active_sequence.tick(T)
            return output,detail
        p=packet.players[index].physics
        position,ball,velocity=xyz(p.location),xyz(packet.balls[0].physics.location),xyz(p.velocity)
        delta=[a-b for a,b in zip(ball,position)]
        distance=math.sqrt(sum(x*x for x in delta))
        speed=math.sqrt(sum(x*x for x in velocity))
        detail.update(distance=distance,speed=speed)
        target=ball
        if distance>1500:
            # Empty predictions preserve IndexError. No correction/clamp/filter.
            first_slice_time=prediction.slices[0].game_seconds
            selected_index=int((T + 2 - first_slice_time) * 120)
            detail['selection_attempt']=dict(index=selected_index,first_time=first_slice_time,requested_time=T+2)
            if 0<=selected_index<len(prediction.slices):
                selected=prediction.slices[selected_index]
                target=xyz(selected.physics.location)
                detail['selected']=dict(index=selected_index,timestamp=selected.game_seconds,position=target)
            detail['branch']='far_ball'
        else:
            detail['branch']='near_ball'
        if 750<speed<800:
            detail['decision']='flip_trigger'
            self.flips+=1
            self.active_sequence=Sequence()
            output,detail['consumed_phase']=self.active_sequence.tick(T)
            return output,detail
        pitch,yaw,roll=float(p.rotation.pitch),float(p.rotation.yaw),float(p.rotation.roll)
        cp,sp,cy,sy,cr,sr=math.cos(pitch),math.sin(pitch),math.cos(yaw),math.sin(yaw),math.cos(roll),math.sin(roll)
        forward=(cp*cy,cp*sy,sp)
        right=(cy*sp*sr-cr*sy,sy*sp*sr+cr*cy,-cp*sr)
        d=[a-b for a,b in zip(target,position)]
        dot=lambda a:d[0]*a[0]+d[1]*a[1]+d[2]*a[2]
        steer=max(-1.,min(1.,5*math.atan2(dot(right),dot(forward))))
        detail['decision']='chase'
        return flat.ControllerState(throttle=1.,steer=steer),detail
