"""Phase 3 teacher-controlled diagnostic. No training trajectory export."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode=True
from common import ROOT,HERE,verify,write as write_json
from reference import original,OriginalShadow,observe,selected_info
from rlbot import flat
from rlbot.utils import fill_desired_game_state
from common import CHANNELS,controls,mode,state,xyz
from student_client import StudentClient
sys.path.insert(0,str(ROOT/'training/v2_observation_test'))
from observation_contract import live_adapter

ANALOG_TOL=1e-6  # Established V1 native analog transport tolerance.
LONG_GAP_SECONDS=1.
PHASES=('Jump','Release','Front dodge','Coast')


def physics(p):
    return dict(position=xyz(p.location),velocity=xyz(p.velocity),
        angular_velocity=xyz(p.angular_velocity),rotation_pyr=[p.rotation.pitch,p.rotation.yaw,p.rotation.roll])


def selected_info(s,index):
    return None if s is None else dict(index=index,timestamp=s.game_seconds,position=xyz(s.physics.location))


class LiveHarness(original.MyBot):
    def __init__(self,*args,**kwargs):
        self.packet_receipts=self.prediction_receipts=0
        self.latest_packet_receipt=self.latest_prediction_receipt=None
        self.pause_requested=None
        self.last_submitted=None
        self.callback_count=0
        self.folder=Path(os.environ['V13_PHASE3_SESSION'])
        self.request_count=0
        self.request_limit=24
        self.pause_count=0
        self.pause_limit=6
        verify()
        super().__init__(*args,**kwargs)
        send=self._game_interface.send_msg
        def tracked_send(message):
            if isinstance(message,flat.PlayerInput):
                actual=controls(message.controller_state)
                if message.player_index!=self.index or self.expected_original_controls is None:
                    raise RuntimeError('Unexpected controller submission')
                expected=self.expected_original_controls
                if actual[5:]!=expected[5:] or max(abs(x-y) for x,y in zip(actual[:5],expected[:5]))>ANALOG_TOL:
                    raise RuntimeError('Submitted controls differ from original teacher output')
            result=send(message)
            if isinstance(message,flat.PlayerInput):
                self.last_submitted=dict(callback=self.callback_count,T=self.last_T,controls=controls(message.controller_state),monotonic=time.monotonic())
                self.event('original_controls_submitted',self.last_submitted)
                if self.pause_requested is not None:
                    seconds,label=self.pause_requested
                    self.pause_requested=None
                    if self.pause_count<self.pause_limit:
                        self.pause_count+=1
                        self.event('controlled_receive_pause',dict(seconds=seconds,label=label,controls_already_submitted=self.last_submitted))
                        time.sleep(seconds)  # Approved diagnostic pause, not synthetic ticks.
                        self.event('controlled_receive_resume',dict(label=label))
            return result
        self._game_interface.send_msg=tracked_send

    def _handle_packet(self,packet):
        self.packet_receipts+=1
        self.latest_packet_receipt=dict(counter=self.packet_receipts,monotonic=time.monotonic(),object_id=id(packet))
        super()._handle_packet(packet)

    def _handle_ball_prediction(self,prediction):
        self.prediction_receipts+=1
        self.latest_prediction_receipt=dict(counter=self.prediction_receipts,monotonic=time.monotonic(),object_id=id(prediction))
        super()._handle_ball_prediction(prediction)

    def initialize(self):
        original.MyBot.initialize(self)  # Actual source initialization and field info.
        self.shadow=OriginalShadow(self)
        self.student=StudentClient(self.folder)
        self.expected_original_controls=None
        self.previous_student=None
        self.divergence_sequences={}
        self.callback_log=(self.folder/'callbacks.jsonl').open('w',encoding='utf-8')
        self.event_log=(self.folder/'events.jsonl').open('w',encoding='utf-8')
        self.coverage={k:[] for k in ('neutral_to_chase','chase_to_jump','release','front_dodge','coast','completion',
            'missing_ball_pending','goal_pending','replay_pending','kickoff_pending','long_gap',
            'near_ball','far_ball','live_selector','student_missed_sequence_start',
            'student_missed_prior_jump','shadow_continues_after_student_divergence')}
        self.failures=[]
        self.original_flips=0
        self.last_T=None
        self.last_frame=None
        self.last_mode=None
        self.max_analog_error=0.
        self.status='running'
        self.started=time.monotonic()
        self.planner_stage='near'
        self.planner_wait=None
        self.planner_attempts={}
        self.goal_sequence=None
        self.countdown_T=None
        self.pause_phase_attempted=set()
        self.transition_pause_attempted=False
        self.event('initialized',dict(index=self.index,team=self.team,
            source='Unmodified MyBot.get_output on actual native packet/BallPrediction',
            shadow_role='Independent original-source MyBot; renderer/quickchat discarded by non-game sink',
            student_role='Pinned 13D GRU diagnostic only; no controller transport',worker=self.student.ready))
        self.summary()

    def write(self,file,row):
        if file.tell()>256*1024*1024:
            raise RuntimeError('Bounded diagnostic log exceeded 256 MiB')
        file.write(json.dumps(row,allow_nan=False)+'\n')
        file.flush()

    def event(self,kind,payload):
        if hasattr(self,'event_log'):
            self.write(self.event_log,dict(kind=kind,monotonic=time.monotonic(),payload=payload))

    def summary(self):
        value=dict(status=self.status,callbacks=self.callback_count,original_flips=self.original_flips,
            shadow_flips=self.shadow.flips,coverage=self.coverage,failures=self.failures,
            maximum_analog_error=self.max_analog_error,packet_receipts=self.packet_receipts,
            prediction_receipts=self.prediction_receipts,agent_index=self.index,agent_team=self.team,
            shadow_controls_submitted=False,student_controls_submitted=False,planner_stage=self.planner_stage,
            student_divergence_sequences=self.divergence_sequences,shadow_forbidden_messages=self.shadow.sink.forbidden,
            shadow_discarded_cosmetic_messages=self.shadow.sink.discarded,
            state_requests=self.request_count,receive_pauses=self.pause_count,student_hidden_reset_policy='match initialization only')
        temp=self.folder/'agent_summary.tmp'
        temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
        os.replace(temp,self.folder/'agent_summary.json')

    def count(self,key,row):
        example=dict(callback=row['callback'],frame=row['frame'],T=row['T'],phase=row['match_phase'],
            sequence_id=row['original_sequence_id_after'] if row['original_detail']['flip_trigger'] else row['original_sequence_id_before'],
            teacher_mode=row['original_mode'],student_mode=row['student']['mode'],
            previous_student=row['previous_student_action_diagnostic_only'],
            source_phase=row['original_detail']['consumed_phase'])
        if len(self.coverage[key])<12:
            self.coverage[key].append(example)
        elif example['sequence_id'] not in {e['sequence_id'] for e in self.coverage[key]}:
            # Preserve evidence for a later transition attempt instead of
            # allowing twelve frames of an earlier replay to hide it.
            self.coverage[key][-1]=example

    def transition_sequence_ids(self):
        return set.intersection(*[{e['sequence_id'] for e in self.coverage[k] if e['sequence_id'] is not None}
            for k in ('goal_pending','replay_pending','kickoff_pending')])

    def complete(self):
        return self.original_flips>=2 and len(self.coverage['completion'])>=2 and all(self.coverage.values()) and bool(self.transition_sequence_ids()) and not self.failures

    def observe_original(self,packet):
        return observe(self,packet)

    def get_output(self,packet):
        # Keep comparing/logging until the supervisor actually disconnects.
        # Coverage-complete stops probes, not callback accounting.
        self.callback_count+=1
        T=float(packet.match_info.seconds_elapsed)
        before=state(self.active_sequence)
        shadow_before=state(self.shadow.active_sequence)
        pred=self.ball_prediction
        row=dict(callback=self.callback_count,T=T,frame=int(packet.match_info.frame_num),match_phase=str(packet.match_info.match_phase),
            callback_dt=None if self.last_T is None else T-self.last_T,
            frame_gap=None if self.last_frame is None else int(packet.match_info.frame_num)-self.last_frame,
            original_before=before,shadow_before=shadow_before,
            original_sequence_id_before=None if before is None else self.original_flips,
            shadow_sequence_id_before=None if shadow_before is None else self.shadow.flips,
            packet_receipt=self.latest_packet_receipt,prediction_receipt=self.latest_prediction_receipt,
            native_packet_object_id=id(packet),native_prediction_object_id=id(pred),
            relevant_observation=dict(car=physics(packet.players[self.index].physics),
                ball=None if not packet.balls else physics(packet.balls[0].physics),
                ball_present=bool(packet.balls),prediction_count=len(pred.slices),
                first_prediction_time=None if not pred.slices else pred.slices[0].game_seconds,
                previous_actual_teacher_submission=self.last_submitted),
            source_sequence_memory_observed=True)
        try:
            out,trace=self.observe_original(packet)
            shadow,detail=self.shadow.output(packet,self.index,pred)
            a,b=controls(out),controls(shadow)
            after,shadow_after=state(self.active_sequence),state(self.shadow.active_sequence)
            original_branch='missing_ball' if not packet.balls else 'pending_sequence' if before is not None and not before['done'] else 'far_ball' if trace['selection_attempt'] is not None else 'near_ball'
            original_decision='flip_trigger' if trace['flip_trigger'] else 'chase' if original_branch in ('far_ball','near_ball') else None
            problems=[]
            if self.active_sequence is not None and self.active_sequence is self.shadow.active_sequence:
                problems.append('shared sequence memory')
            if self.boost_pad_tracker is self.shadow.actor.boost_pad_tracker:
                problems.append('shared boost tracker')
            if self.shadow.actor.ball_prediction is not pred:
                problems.append('different native prediction object')
            analog=max(abs(x-y) for x,y in zip(a[:5],b[:5]))
            self.max_analog_error=max(self.max_analog_error,analog)
            if not all(math.isfinite(x) for x in a[:5]+b[:5]) or analog>ANALOG_TOL:
                problems.append('analog channels')
            if any(type(x) is not bool or type(y) is not bool or x!=y for x,y in zip(a[5:],b[5:])):
                problems.append('buttons')
            if mode(a)!=mode(b):problems.append('output mode')
            if a[3:5]!=[0.,0.] or a[6:]!=[False,False]:
                problems.append('teacher declared zero/false support')
            if before!=shadow_before or after!=shadow_after:problems.append('sequence state/index/done/start times')
            if trace['consumed_phase']!=detail['consumed_phase']:problems.append('consumed sequence phase')
            if trace['selected']!=detail['selected'] or trace['selection_attempt']!=detail.get('selection_attempt'):problems.append('prediction selection')
            if original_branch!=detail['branch'] or original_decision!=detail.get('decision'):problems.append('branch/decision')
            if trace['flip_trigger']:self.original_flips+=1
            if self.original_flips!=self.shadow.flips:problems.append('sequence trigger count')
            row.update(original_controls=a,shadow_controls=b,original_mode=mode(a),shadow_mode=mode(b),
                original_after=after,shadow_after=shadow_after,
                original_detail=dict(**trace,branch=original_branch,decision=original_decision),shadow_detail=detail,
                analog_error=analog,failures=problems,
                original_flips=self.original_flips,shadow_flips=self.shadow.flips,
                original_sequence_id_after=None if after is None else self.original_flips,
                shadow_sequence_id_after=None if shadow_after is None else self.shadow.flips,
                state_transition=dict(original=before!=after,shadow=shadow_before!=shadow_after))
            # Same observation probe as frozen recorder, distinct from gameplay
            # branch. All prior-action columns are discarded before worker IPC.
            previous_mode=0 if self.last_submitted is None else {'Neutral':0,'Chase':1,'Jump':2,'Front dodge':3}[mode(self.last_submitted['controls'])]
            previous_steer=0. if self.last_submitted is None else self.last_submitted['controls'][1]
            selected=None; probe_index=None
            if pred.slices:
                probe_index=int((T+2-pred.slices[0].game_seconds)*120)
                selected=original.find_slice_at_time(pred,T+2)
            ctx=dict(previous_elapsed=self.last_T,previous_mode=previous_mode,previous_steer=previous_steer,
                prediction_valid=selected is not None,first_time=None if not pred.slices else pred.slices[0].game_seconds)
            row['observation13']=list(live_adapter(packet,self.index,ctx,selected))[:13]
            row['observation_prediction_probe']=selected_info(selected,probe_index)
            row['observation_probe_is_teacher_decision']=False
            row['previous_student_action_diagnostic_only']=self.previous_student
            row['student']=self.student.infer(self.callback_count,row['observation13'])
            row['student_teacher_mode_agreement']=row['student']['mode']==row['original_mode']
            row['private_teacher_state_in_student_input']=False
            row['independent_sequence_objects']=self.active_sequence is None or self.active_sequence is not self.shadow.active_sequence
            row['shadow_packet_object_id']=id(packet)
            row['shadow_prediction_object_id']=id(self.shadow.actor.ball_prediction)
            row['shadow_elapsed']=float(packet.match_info.seconds_elapsed)
            if row['student']['hidden_resets']!=1: problems.append('unexpected student hidden reset')
            if problems:
                self.failures.append(dict(callback=self.callback_count,frame=row['frame'],failures=problems))
                self.status='comparison_failure'
            else:
                self.cover(row)
            self.write(self.callback_log,row)
            self.last_T=T
            self.last_frame=row['frame']
            self.last_mode=mode(a)
            self.previous_student=dict(callback=self.callback_count,controls=row['student']['controls'],mode=row['student']['mode'])
            self.expected_original_controls=a
            if self.complete():
                self.status='coverage_complete_awaiting_review'
            if self.status=='running':self.plan(packet,row)
            if self.callback_count%120==0 or self.status!='running':self.summary()
            return out  # Only original source controls are returned to RLBot.
        except Exception as error:
            row.update(error_type=type(error).__name__,error=repr(error))
            self.write(self.callback_log,row)
            self.failures.append(dict(callback=self.callback_count,error=repr(error)))
            self.status='error'
            self.summary()
            raise

    def cover(self,row):
        before,after=row['original_before'],row['original_after']
        d=row['original_detail']
        pending=before is not None and not before['done']
        if self.last_mode=='Neutral' and row['original_mode']=='Chase':self.count('neutral_to_chase',row)
        if self.last_mode=='Chase' and d['flip_trigger'] and row['original_mode']=='Jump':self.count('chase_to_jump',row)
        for i,key in ((1,'release'),(2,'front_dodge'),(3,'coast')):
            if d['consumed_phase']==i:self.count(key,row)
        if after is not None and after['done'] and (before is None or not before['done']):self.count('completion',row)
        if pending and not row['relevant_observation']['ball_present']:self.count('missing_ball_pending',row)
        for phase,key in (('GoalScored','goal_pending'),('Replay','replay_pending'),('Kickoff','kickoff_pending')):
            if pending and row['match_phase']=='MatchPhase.'+phase:self.count(key,row)
        if row['callback_dt'] is not None and row['callback_dt']>=LONG_GAP_SECONDS and pending:self.count('long_gap',row)
        if d['branch'] in ('near_ball','far_ball'):self.count(d['branch'],row)
        if d['selection_attempt'] is not None and d['selected'] is not None:self.count('live_selector',row)
        sid=row['original_sequence_id_after']
        if d['flip_trigger'] and row['student']['mode']!='Jump':
            self.divergence_sequences[str(sid)]=dict(start=row['callback'],student_mode=row['student']['mode'])
            self.count('student_missed_sequence_start',row)
        previous=row['previous_student_action_diagnostic_only']
        if d['consumed_phase']==2 and previous is not None and not previous['controls'][5]:
            self.count('student_missed_prior_jump',row)
        if pending and str(sid) in self.divergence_sequences and row['callback']>self.divergence_sequences[str(sid)]['start']:
            self.count('shadow_continues_after_student_divergence',row)

    def request(self,packet,label,position=None,velocity=None,ball=None):
        if self.request_count>=self.request_limit:
            self.status='probe_budget_exhausted_incomplete'
            self.event('probe_budget_exhausted',dict(limit=self.request_limit))
            self.summary()
            return
        self.request_count+=1
        cars={}
        if position is not None:
            cars[self.index]=flat.DesiredCarState(flat.DesiredPhysics(location=flat.Vector3Partial(*position),
                velocity=flat.Vector3Partial(*velocity),rotation=flat.RotatorPartial(0,math.pi/2,0),angular_velocity=flat.Vector3Partial(0,0,0)))
            cars[1-self.index]=flat.DesiredCarState(flat.DesiredPhysics(location=flat.Vector3Partial(3000,0,18.34),
                velocity=flat.Vector3Partial(0,0,0),rotation=flat.RotatorPartial(0,0,0),angular_velocity=flat.Vector3Partial(0,0,0)))
        balls={} if ball is None else {0:flat.DesiredBallState(flat.DesiredPhysics(
            location=flat.Vector3Partial(*ball[0]),velocity=flat.Vector3Partial(*ball[1]),angular_velocity=flat.Vector3Partial(0,0,0),rotation=flat.RotatorPartial(0,0,0)))}
        message=fill_desired_game_state(balls,cars,None)
        self._game_interface.send_msg(message)
        self.event('controlled_state_request',dict(label=label,T=packet.match_info.seconds_elapsed,frame=packet.match_info.frame_num,
            position=position,velocity=velocity,ball=ball,meaning='Request only; coverage comes exclusively from actual later callbacks'))

    def plan(self,packet,row):
        # Approved live interventions. No fabricated packet/time/ball absence.
        T=row['T']; phase=packet.match_info.match_phase
        if self.planner_stage=='near' and phase in (flat.MatchPhase.Active,flat.MatchPhase.Kickoff) and not (self.active_sequence is not None and not self.active_sequence.done):
            self.request(packet,'near-ball start',(0,-900,17),(0,0,0),((0,0,93.15),(0,0,0)))
            self.planner_wait=T; self.planner_stage='wait_near'
        elif self.planner_stage=='wait_near' and self.coverage['near_ball'] and T-self.planner_wait>.25:
            self.request(packet,'far-ball start',(0,-3500,17),(0,0,0),((0,0,93.15),(0,0,0)))
            self.planner_wait=T; self.planner_stage='wait_far'
        elif self.planner_stage=='wait_far' and self.coverage['far_ball'] and T-self.planner_wait>.25:
            self.planner_stage='regular_flips'
        elif self.planner_stage=='regular_flips' and phase==flat.MatchPhase.Active:
            pending=self.active_sequence is not None and not self.active_sequence.done
            if not pending and len(self.coverage['completion'])<2:
                if self.planner_wait is None or T-self.planner_wait>.5:
                    self.request(packet,'regular flip trigger',(0,-2500,17),(0,775,0),((0,0,93.15),(0,0,0)))
                    self.planner_wait=T
            elif pending and self.active_sequence.index in (0,2) and self.active_sequence.index not in self.pause_phase_attempted:
                i=self.active_sequence.index
                self.pause_phase_attempted.add(i)
                self.pause_requested=(1.25,'phase '+str(i)+' long-gap probe')
            elif len(self.coverage['completion'])>=2 and not pending:
                self.request(packet,'goal-overlap flip trigger',(0,-2500,17),(0,775,0),((0,0,93.15),(0,0,0)))
                self.planner_stage='goal_trigger_wait'; self.planner_wait=T
        elif self.planner_stage=='goal_trigger_wait':
            if self.active_sequence is not None and not self.active_sequence.done and self.active_sequence.index==1:
                self.request(packet,'goal during pending Release',ball=((0,5100,120),(0,2800,0)))
                self.planner_stage='goal_wait'; self.planner_wait=T
                self.transition_pause_attempted=False
            elif T-self.planner_wait>4 and phase==flat.MatchPhase.Active:
                self.request(packet,'retry goal-overlap flip trigger',(0,-2500,17),(0,775,0),((0,0,93.15),(0,0,0)))
                self.planner_wait=T
        elif self.planner_stage=='goal_wait':
            if phase==flat.MatchPhase.Countdown:
                pending=self.active_sequence is not None and not self.active_sequence.done
                if pending and not self.transition_pause_attempted:
                    self.transition_pause_attempted=True
                    self.pause_requested=(5.,'pending sequence across real countdown/kickoff')
                    self.planner_stage='kickoff_wait'; self.planner_wait=T
            elif T-self.planner_wait>12 and phase==flat.MatchPhase.Active:
                self.planner_stage='goal_trigger_wait'; self.planner_wait=T
                self.request(packet,'retry goal-overlap after no observed goal',(0,-2500,17),(0,775,0),((0,0,93.15),(0,0,0)))
        elif self.planner_stage=='kickoff_wait':
            if T-self.planner_wait>12 and phase==flat.MatchPhase.Active and not self.complete():
                self.countdown_T=None; self.planner_stage='goal_trigger_wait'; self.planner_wait=T
                self.request(packet,'repeat transition coverage probe',(0,-2500,17),(0,775,0),((0,0,93.15),(0,0,0)))

    def retire(self):
        if hasattr(self,'callback_log'):
            self.summary()
            self.student.close()
            self.callback_log.close()
            self.event_log.close()
            write_json(self.folder/'agent_closed.json',dict(callbacks=self.callback_count,status=self.status,worker_closed=True))
        super().retire()


if __name__=='__main__':
    LiveHarness('local/v13_phase3_shadow').run()
