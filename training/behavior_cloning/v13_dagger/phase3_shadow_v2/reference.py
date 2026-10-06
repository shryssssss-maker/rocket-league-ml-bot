"""Two original-source actors, independent memory; passive traces only."""
import sys
from common import ROOT,load,controls,state,xyz

sys.path.insert(0,str(ROOT/'python-example-original/src'))
original = load('_phase3_original',ROOT/'python-example-original/src/bot.py')
from rlbot import flat
from rlbot.managers.rendering import Renderer

def selected_info(s,index):
    return None if s is None else dict(index=index,timestamp=s.game_seconds,position=xyz(s.physics.location))

def observe(actor,packet):
    trace = dict(selected=None,selection_attempt=None,consumed_phase=None,flip_trigger=False)
    def profile(frame,event,arg):
        if frame.f_code is original.find_slice_at_time.__code__ and event=='return':
            local=frame.f_locals
            if 'approx_index' in local:
                i=local['approx_index']
                trace['selection_attempt']=dict(index=i,first_time=local['start_time'],requested_time=local['game_time'])
                trace['selected']=selected_info(arg,i)
        elif frame.f_code is original.Sequence.tick.__code__ and event=='call':
            trace['consumed_phase']=frame.f_locals['self'].index
        elif frame.f_code is original.MyBot.begin_front_flip.__code__ and event=='call':
            trace['flip_trigger']=True
    previous=sys.getprofile()
    sys.setprofile(profile)
    try: out=original.MyBot.get_output(actor,packet)
    finally: sys.setprofile(previous)
    return out,trace

class NonGameSink:
    can_render=False
    def __init__(self): self.discarded={}; self.forbidden=0
    def send_msg(self,message):
        if not isinstance(message,(flat.MatchComm,flat.RenderGroup,flat.RemoveRenderGroup)):
            self.forbidden+=1
            raise RuntimeError('Shadow attempted forbidden transport: '+type(message).__name__)
        name=type(message).__name__
        self.discarded[name]=self.discarded.get(name,0)+1

class OriginalShadow:
    """API compatible with diagnostic harness; source MyBot, not reconstructed logic."""
    def __init__(self,owner):
        self.actor=object.__new__(original.MyBot)  # No socket construction/connection.
        self.sink=NonGameSink()
        self.actor.index=owner.index; self.actor.team=owner.team
        self.actor.field_info=owner.field_info
        self.actor.active_sequence=None
        self.actor.boost_pad_tracker=original.BoostPadTracker()
        self.actor._game_interface=self.sink
        self.actor.renderer=Renderer(self.sink)
        self.actor.renderer._current_renders=[]
        self.actor.renderer._used_group_ids=set()
        original.MyBot.initialize(self.actor)
        self.flips=0
        if self.actor.boost_pad_tracker is owner.boost_pad_tracker:
            raise RuntimeError('Shadow boost tracker shares teacher memory')

    @property
    def active_sequence(self): return self.actor.active_sequence

    def output(self,packet,index,prediction):
        if index!=self.actor.index: raise RuntimeError('Agent index changed')
        before=state(self.active_sequence)
        self.actor.ball_prediction=prediction  # Same native delivered object.
        out,trace=observe(self.actor,packet)
        branch=('missing_ball' if not packet.balls else
                'pending_sequence' if before is not None and not before['done'] else
                'far_ball' if trace['selection_attempt'] is not None else 'near_ball')
        decision='flip_trigger' if trace['flip_trigger'] else 'chase' if branch in ('near_ball','far_ball') else None
        if trace['flip_trigger']: self.flips+=1
        return out,dict(**trace,branch=branch,decision=decision)
