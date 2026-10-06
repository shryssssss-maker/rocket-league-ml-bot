"""Natural student-only pilot. Original shadow logic is imported unchanged."""
import json
import math
import os
from pathlib import Path
import sys
import time
from common import ROOT,controls,mode,state,xyz,verify,write
from reference import original,OriginalShadow,selected_info
from student_client import StudentClient
from diagnostics import SummaryPublisher
from rlbot import flat
from rlbot.managers import Bot

sys.path.insert(0,str(ROOT/'training/v2_observation_test'))
from observation_contract import live_adapter

def physics(p):
    return dict(position=xyz(p.location),velocity=xyz(p.velocity),angular_velocity=xyz(p.angular_velocity),
        rotation_pyr=[p.rotation.pitch,p.rotation.yaw,p.rotation.roll])

def player(p):
    return dict(physics=physics(p.physics),team=p.team,air_state=str(p.air_state),
        has_jumped=p.has_jumped,has_double_jumped=p.has_double_jumped,has_dodged=p.has_dodged,
        dodge_elapsed=p.dodge_elapsed,dodge_timeout=p.dodge_timeout,demolished_timeout=p.demolished_timeout,
        boost=p.boost,latest_touch=None if p.latest_touch is None else repr(p.latest_touch))

def check_student_submission(message,index,expected):
    if isinstance(message,flat.PlayerInput):
        a=controls(message.controller_state)
        if message.player_index!=index or expected is None or a!=expected:
            raise RuntimeError('Nonstudent controller submission denied')

class StudentPilot(Bot):
    def __init__(self,*args,**kwargs):
        verify()
        self.folder=Path(os.environ['V13_PHASE4_SESSION'])
        self.packet_receipts=self.prediction_receipts=0
        self.packet_receipt=self.prediction_receipt=None
        self.callback_count=0;self.last_T=self.last_frame=None
        self.last_submitted=None;self.expected=None
        self.first_submission_monotonic=None
        super().__init__(*args,**kwargs)
        native_send=self._game_interface.send_msg
        def send(message):
            if isinstance(message,flat.PlayerInput):
                a=controls(message.controller_state)
                try:check_student_submission(message,self.index,self.expected)
                except RuntimeError:
                    self.failure('Unexpected/nonstudent controller submission')
                    raise
            result=native_send(message)
            if isinstance(message,flat.PlayerInput):
                now=time.monotonic()
                self.last_submitted=dict(callback=self.callback_count,T=self.last_T,frame=self.last_frame,controls=a,
                    mode=mode(a),monotonic=now,callback_to_submission_seconds=now-self.callback_started)
                if self.first_submission_monotonic is None:self.first_submission_monotonic=now
                self.event('student_controls_submitted',self.last_submitted)
                if self.callback_count==1 or self.callback_count%120==0:self.summary()
            return result
        self._game_interface.send_msg=send

    def initialize(self):
        self.callback_log=(self.folder/'callbacks.jsonl').open('w',encoding='utf-8')
        self.event_log=(self.folder/'events.jsonl').open('w',encoding='utf-8')
        self.error_log=(self.folder/'diagnostic_errors.jsonl').open('w',encoding='utf-8')
        self.publisher=SummaryPublisher(self.folder);self.errors=[];self.status='running'
        # Compatibility attribute only: shadow constructs its own initialized
        # tracker. No live original teacher actor or control path is constructed.
        self.boost_pad_tracker=original.BoostPadTracker()
        self.shadow=OriginalShadow(self)
        self.student=StudentClient(self.folder)
        self.previous_phase=None
        self.event('initialized',dict(agent_index=self.index,team=self.team,controller='13D student only',
            teacher_role='independent original shadow; non-game sink',worker=self.student.ready))
        self.summary()

    def record(self,file,value):
        if file.tell()>256*1024*1024:raise RuntimeError('Diagnostic file bound exceeded')
        file.write(json.dumps(value,allow_nan=False)+'\n');file.flush()

    def event(self,kind,payload):
        self.record(self.event_log,dict(kind=kind,monotonic=time.monotonic(),payload=payload))

    def failure(self,error):
        self.status='error'
        evidence=dict(callback=self.callback_count,error=str(error),monotonic=time.monotonic())
        self.errors.append(evidence)
        try:self.record(self.error_log,evidence)
        except OSError:pass
        self.summary()

    def summary(self):
        value=dict(status='diagnostic_publication_error' if self.publisher.errors else self.status,
            callbacks=self.callback_count,shadow_flips=self.shadow.flips,errors=self.errors,
            publication_errors=self.publisher.errors,packet_receipts=self.packet_receipts,prediction_receipts=self.prediction_receipts,
            first_submission_monotonic=self.first_submission_monotonic,
            last_submitted_callback=0 if self.last_submitted is None else self.last_submitted['callback'],
            final_callback_submission_verified=self.last_submitted is not None and self.last_submitted['callback']==self.callback_count,
            student_controls_only=True,shadow_forbidden_messages=self.shadow.sink.forbidden,
            shadow_cosmetic_discards=self.shadow.sink.discarded,agent_index=self.index,agent_team=self.team)
        if self.publisher.publish(value) is None:
            error=dict(callback=self.callback_count,error=self.publisher.errors[-1],stage='summary_publication')
            self.errors.append(error)
            try:self.record(self.error_log,error)
            except OSError:pass

    def _handle_packet(self,packet):
        self.packet_receipts+=1
        self.packet_receipt=dict(counter=self.packet_receipts,monotonic=time.monotonic(),object_id=id(packet))
        super()._handle_packet(packet)

    def _handle_ball_prediction(self,prediction):
        self.prediction_receipts+=1
        self.prediction_receipt=dict(counter=self.prediction_receipts,monotonic=time.monotonic(),object_id=id(prediction))
        super()._handle_ball_prediction(prediction)

    def _packet_processor(self,packet):
        if self.status!='running':return  # Abort; retain native last-action hold until supervisor stops match.
        if len(packet.players)<=self.index:
            self.failure('Controlled car absent in actual packet')
            self.event('missing_controlled_car',dict(frame=packet.match_info.frame_num,T=packet.match_info.seconds_elapsed,
                packet_receipt=self.packet_receipt))
            return
        super()._packet_processor(packet)

    def get_output(self,packet):
        self.callback_count+=1;self.callback_started=time.monotonic()
        T=float(packet.match_info.seconds_elapsed);frame=int(packet.match_info.frame_num)
        pred=self.ball_prediction
        row=dict(session_id=self.folder.name,callback=self.callback_count,frame=frame,T=T,
            callback_dt=None if self.last_T is None else T-self.last_T,frame_gap=None if self.last_frame is None else frame-self.last_frame,
            match_phase=str(packet.match_info.match_phase),agent_index=self.index,team=self.team,
            packet_receipt=self.packet_receipt,prediction_receipt=self.prediction_receipt,
            packet_object_id=id(packet),prediction_object_id=id(pred),prediction_count=len(pred.slices),
            first_prediction_time=None if not pred.slices else pred.slices[0].game_seconds,
            previous_submission=self.last_submitted,car=player(packet.players[self.index]),
            opponent=[player(p) for i,p in enumerate(packet.players) if i!=self.index],
            ball=None if not packet.balls else physics(packet.balls[0].physics),
            scores={str(t.team_index):t.score for t in packet.teams},
            phase_transition=None if self.previous_phase==str(packet.match_info.match_phase) else dict(before=self.previous_phase,after=str(packet.match_info.match_phase)),
            not_training_dataset=True)
        attempted=False
        try:
            if not self.prediction_receipt:raise RuntimeError('No actually delivered prediction receipt yet')
            selected=None;probe_index=None
            if pred.slices:
                probe_index=int((T+2-pred.slices[0].game_seconds)*120)
                selected=original.find_slice_at_time(pred,T+2)
            prior_mode=0 if self.last_submitted is None else {'Neutral':0,'Chase':1,'Jump':2,'Front dodge':3}[self.last_submitted['mode']]
            ctx=dict(previous_elapsed=self.last_T,previous_mode=prior_mode,
                previous_steer=0. if self.last_submitted is None else self.last_submitted['controls'][1],
                prediction_valid=selected is not None,first_time=row['first_prediction_time'])
            row['observation13']=list(live_adapter(packet,self.index,ctx,selected))[:13]
            row['observation_probe']=selected_info(selected,probe_index)
            row['observation_probe_is_shadow_decision']=False
            before=state(self.shadow.active_sequence)
            row.update(shadow_before=before,shadow_sequence_id_before=None if before is None else self.shadow.flips)
            teacher,detail=self.shadow.output(packet,self.index,pred)
            row.update(shadow_controls=controls(teacher),shadow_mode=mode(controls(teacher)),shadow_detail=detail,
                shadow_after=state(self.shadow.active_sequence),shadow_sequence_id_after=self.shadow.flips,
                same_native_shadow_packet=True,shadow_prediction_object_id=id(self.shadow.actor.ball_prediction),shadow_elapsed=T)
            result=self.student.infer(self.callback_count,row['observation13'])
            a=result['controls']
            if len(a)!=8 or not all(math.isfinite(x) and -1<=x<=1 for x in a[:5]) or any(type(x) is not bool for x in a[5:]):
                raise ValueError('Invalid native student output')
            if a[3:5]!=[0.,0.] or a[6:]!=[False,False] or result['hidden_resets']!=1:
                raise ValueError('Student support/memory contract violated')
            output=flat.ControllerState(*a)
            if controls(output)!=a:raise ValueError('Native controller conversion altered output')
            row.update(student=result,student_shadow_mode_agreement=result['mode']==row['shadow_mode'],
                private_teacher_state_in_student_input=False)
            attempted=True;self.record(self.callback_log,row)
            self.last_T=T;self.last_frame=frame;self.previous_phase=row['match_phase'];self.expected=a
            return output  # ONLY the pinned student's output reaches SDK PlayerInput.
        except Exception as error:
            if not attempted:
                row.update(error=repr(error),error_type=type(error).__name__)
                try:self.record(self.callback_log,row)
                except OSError:pass
            self.failure(repr(error))
            raise  # No shadow/teacher/neutral fallback or state repair.

    def retire(self):
        if hasattr(self,'student'):
            self.summary();self.student.close()
            for file in (self.callback_log,self.event_log,self.error_log):file.close()
            write(self.folder/'agent_closed.json',dict(status=self.status,callbacks=self.callback_count,worker_closed=True,
                last_submitted_callback=0 if self.last_submitted is None else self.last_submitted['callback']))
        super().retire()

if __name__=='__main__':StudentPilot('local/v13_fullmatch_student').run()
