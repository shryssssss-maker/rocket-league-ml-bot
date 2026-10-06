"""Original teacher only. Natural, pre-action native snapshots; no student/probes."""
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from contracts import ROOT,HERE,load,verify,read,write,controls,mode,agree,physics,native,native_block,state,xyz,OBS

sys.path.insert(0,str(ROOT/'python-example-original/src'))
original=load('_v16_original',ROOT/'python-example-original/src/bot.py')
from rlbot import flat
diagnostics=load('_v16_immutable_summary',ROOT/'training/behavior_cloning/v13_dagger/phase3_shadow_v2/diagnostics.py')

class NativeRecorder(original.MyBot):
    def __init__(self,*args,**kwargs):
        verify()
        self.folder=Path(os.environ['V16_SESSION'])
        self.meta=read(self.folder/'metadata.json')
        self.receipts={'packet':0,'prediction':0}; self.latest={'packet':None,'prediction':None}
        self.order=0; self.n=0; self.last_submission=None; self.pending=None
        self.diagnostic_errors=[]; self.last_T=self.last_frame=self.last_phase=None
        self.starts=0; self.completions=0; self.status='running'
        super().__init__(*args,**kwargs)
        send=self._game_interface.send_msg
        def tracked(message):
            if isinstance(message,flat.DesiredGameState): raise RuntimeError('V16 state setting forbidden')
            if isinstance(message,flat.PlayerInput):
                if self.pending is None or message.player_index!=self.index or not agree(controls(message.controller_state),self.pending['teacher_controls']):
                    raise RuntimeError('Only exact current original teacher output may be submitted')
                self.order+=1
                submission=dict(callback=self.n,T=self.pending['T'],frame=self.pending['frame'],
                                before_send_monotonic=time.monotonic(),order=self.order,controls=controls(message.controller_state))
                try: result=send(message)
                except Exception as e:
                    submission.update(status='send_failed',error=repr(e),after_send_monotonic=time.monotonic())
                    self.safe_log(self.submissions,submission); self.status='transport_error'; self.publish(); raise
                submission.update(status='local_send_returned',after_send_monotonic=time.monotonic(),
                                  engine_application_acknowledged=False)
                self.last_submission=submission
                self.safe_log(self.submissions,submission); self.pending=None
                if self.n%120==0 or self.status!='running': self.publish()
                return result
            return send(message)
        self._game_interface.send_msg=tracked

    def receipt(self,kind,obj):
        self.order+=1; self.receipts[kind]+=1
        value=dict(counter=self.receipts[kind],monotonic=time.monotonic(),object_id=id(obj),order=self.order)
        if kind=='packet': value.update(T=float(obj.match_info.seconds_elapsed),frame=int(obj.match_info.frame_num))
        else: value.update(count=len(obj.slices),first_time=None if not obj.slices else float(obj.slices[0].game_seconds))
        self.latest[kind]=value

    def _handle_packet(self,packet):
        self.receipt('packet',packet); super()._handle_packet(packet)

    def _handle_ball_prediction(self,prediction):
        self.receipt('prediction',prediction); super()._handle_ball_prediction(prediction)

    def initialize(self):
        original.MyBot.initialize(self)
        self.log=(self.folder/'raw/callbacks.jsonl').open('x',encoding='utf-8')
        self.submissions=(self.folder/'raw/submissions.jsonl').open('x',encoding='utf-8')
        self.errors=(self.folder/'raw/errors.jsonl').open('x',encoding='utf-8')
        self.publisher=diagnostics.SummaryPublisher(self.folder)
        self.publish()

    def safe_log(self,file,value):
        import json
        try: file.write(json.dumps(value,allow_nan=False)+'\n'); file.flush()
        except (OSError,ValueError) as e:
            self.diagnostic_errors.append(dict(callback=self.n,error=repr(e))); self.status='diagnostic_error'
            # No replacement controller, no duplicate callback/error row.

    def publish(self):
        value=dict(status=self.status,callbacks=self.n,starts=self.starts,completions=self.completions,
                   team=self.team,agent_index=self.index,receipts=self.receipts,diagnostic_errors=self.diagnostic_errors,
                   publication_errors=self.publisher.errors,last_submitted_callback=0 if self.last_submission is None else self.last_submission['callback'],
                   final_submission_verified=self.last_submission is not None and self.last_submission['callback']==self.n)
        self.publisher.publish(value)
        if self.publisher.errors: self.status='publication_error'

    def get_output(self,packet):
        self.n+=1; self.order+=1
        T=float(packet.match_info.seconds_elapsed); frame=int(packet.match_info.frame_num)
        row=dict(match_id=self.meta['match_id'],session_id=self.meta['session_id'],opponent_id=self.meta['opponent_id'],
                 play_session_id=self.meta['play_session_id'],callback=self.n,frame=frame,T=T,
                 packet_timestamp=T,previous_callback_T=self.last_T,
                 callback_dt=None if self.last_T is None else T-self.last_T,frame_gap=None if self.last_frame is None else frame-self.last_frame,
                 match_phase=str(packet.match_info.match_phase),agent_index=self.index,team=self.team,
                 side='Blue' if self.team==0 else 'Orange',packet_receipt=dict(self.latest['packet']) if self.latest['packet'] else None,
                 prediction_receipt=dict(self.latest['prediction']) if self.latest['prediction'] else None,
                 packet_object_id=id(packet),prediction_object_id=id(self.ball_prediction),callback_order=self.order,
                 callback_started_monotonic=time.monotonic(),previous_submission=self.last_submission,
                 sequence_before=state(self.active_sequence),sequence_id_before=self.starts or None,
                 phase_transition=None if self.last_phase==str(packet.match_info.match_phase) else dict(previous=self.last_phase,current=str(packet.match_info.match_phase)),
                 current_submission='pending_separate_submission_journal',valid=False,errors=[])
        trace=dict(selected=None,selection_attempt=None,consumed_phase=None,flip_trigger=False)
        def profile(f,event,arg):
            if f.f_code is original.find_slice_at_time.__code__ and event=='return' and 'approx_index' in f.f_locals:
                i=f.f_locals['approx_index']
                trace['selection_attempt']=dict(index=i,first_time=f.f_locals['start_time'],requested_time=f.f_locals['game_time'])
                trace['selected']=None if arg is None else dict(index=i,timestamp=arg.game_seconds,position=xyz(arg.physics.location))
            elif f.f_code is original.Sequence.tick.__code__ and event=='call': trace['consumed_phase']=f.f_locals['self'].index
            elif f.f_code is original.MyBot.begin_front_flip.__code__ and event=='call': trace['flip_trigger']=True
        teacher_error=None; out=None
        try:
            player=packet.players[self.index]
            row['native']=native(player); native_block(row['native'])
            row['native_captured_monotonic']=time.monotonic()
            row['native_provenance']='same actual packet; immutable scalar copy before teacher decision'
            row['car']=physics(player.physics); row['ball']=None if not packet.balls else physics(packet.balls[0].physics)
            row['auxiliary_native']=dict(boost=float(player.boost),demolished_timeout=float(player.demolished_timeout),is_supersonic=player.is_supersonic)
            row['teacher_started_monotonic']=time.monotonic()
        except Exception as e: row['errors'].append(dict(stage='pre_action_snapshot',error=repr(e)))
        saved=sys.getprofile()
        try:
            sys.setprofile(profile); out=original.MyBot.get_output(self,packet)
        except Exception as e: teacher_error=e; row['errors'].append(dict(stage='original_teacher',type=type(e).__name__,error=repr(e)))
        finally: sys.setprofile(saved)
        row['teacher_finished_monotonic']=time.monotonic()
        if trace['flip_trigger']: self.starts+=1
        row.update(teacher_trace=trace,sequence_after=state(self.active_sequence),sequence_id_after=self.starts or None,
                   consumed_phase_diagnostic=trace['consumed_phase'],private_state_is_diagnostic_only=True)
        before,after=row['sequence_before'],row['sequence_after']
        if after is not None and after['done'] and (before is None or not before['done']): self.completions+=1
        if teacher_error is None:
            try:
                row['teacher_controls']=controls(out); row['mode']=mode(row['teacher_controls'])
                row['teacher_branch']='missing_ball' if not packet.balls else 'pending_sequence' if before is not None and not before['done'] else 'far_ball' if trace['selection_attempt'] is not None else 'near_ball'
                pred=self.ball_prediction; selected=None; index=None
                row['prediction_count']=len(pred.slices); row['prediction_first_time']=None if not pred.slices else float(pred.slices[0].game_seconds)
                if pred.slices:
                    index=int((T+2-pred.slices[0].game_seconds)*120); selected=original.find_slice_at_time(pred,T+2)
                row['observation_prediction_probe']=None if selected is None else dict(index=index,timestamp=selected.game_seconds,position=xyz(selected.physics.location))
                row['probe_is_gameplay_decision']=False
                prior=self.last_submission
                context=dict(previous_elapsed=self.last_T,previous_mode=0 if prior is None else mode(prior['controls']),
                             previous_steer=0. if prior is None else prior['controls'][1],prediction_valid=selected is not None,first_time=row['prediction_first_time'])
                row['observation13']=list(OBS.live_adapter(packet,self.index,context,selected))[:13]
                if row['packet_receipt'] is None or row['packet_receipt']['object_id']!=id(packet): raise RuntimeError('Packet receipt identity mismatch')
                if row['prediction_receipt'] is not None and row['prediction_receipt']['object_id']!=id(pred): raise RuntimeError('Prediction receipt identity mismatch')
                if self.team!=self.meta['team']: raise RuntimeError('Teacher side differs from assignment')
            except Exception as e: row['errors'].append(dict(stage='observation_diagnostic',error=repr(e)))
        row['valid']=not row['errors']
        row['record_finished_monotonic']=time.monotonic()
        self.safe_log(self.log,row)
        if row['errors']: self.safe_log(self.errors,dict(callback=self.n,errors=row['errors'])); self.status='callback_error'
        self.last_T,self.last_frame,self.last_phase=T,frame,row['match_phase']
        if teacher_error is not None: self.publish(); raise teacher_error
        self.pending=row
        return out  # SDK submits only this unchanged original ControllerState.

    def retire(self):
        if hasattr(self,'log'):
            self.publish()
            for file in (self.log,self.submissions,self.errors): file.close()
            write(self.folder/'agent_closed.json',dict(callbacks=self.n,status=self.status,last_submission=self.last_submission,
                                                      diagnostic_errors=self.diagnostic_errors,publication_errors=self.publisher.errors))
        super().retire()

if __name__=='__main__': NativeRecorder('local/v16_native_audit').run()
