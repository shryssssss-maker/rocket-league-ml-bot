"""Natural live source-teacher recorder. No state setting, pauses or candidate."""
import importlib.util
import json
import math
import os
import sys
import time
from common import ROOT, controls, mode, physics, sequence, live_adapter, xyz

sys.path.insert(0, str(ROOT/'python-example-original/src'))
spec = importlib.util.spec_from_file_location('_v9_original', ROOT/'python-example-original/src/bot.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
from rlbot import flat


class Recorder(original.MyBot):
    def __init__(self, *args, **kwargs):
        self.folder = __import__('pathlib').Path(os.environ['V9_SESSION'])
        self.receipts = {'packet':0, 'prediction':0}
        self.latest = {'packet':None, 'prediction':None}
        self.last_submitted = None
        self.n = 0
        super().__init__(*args, **kwargs)
        send = self._game_interface.send_msg
        def track(message):
            result = send(message)
            if isinstance(message, flat.PlayerInput):
                a = controls(message.controller_state)
                self.last_submitted = dict(callback=self.n, T=self.last_T,
                    controls=a, mode=mode(a), monotonic=time.monotonic())
            return result
        self._game_interface.send_msg = track

    def _handle_packet(self, packet):
        self.receipt('packet',packet)
        super()._handle_packet(packet)

    def _handle_ball_prediction(self, prediction):
        self.receipt('prediction',prediction)
        super()._handle_ball_prediction(prediction)

    def receipt(self, kind, obj):
        self.receipts[kind] += 1
        self.latest[kind] = dict(counter=self.receipts[kind],monotonic=time.monotonic(),object_id=id(obj))

    def initialize(self):
        original.MyBot.initialize(self)
        self.log = (self.folder/'raw/trajectory.jsonl').open('x',encoding='utf-8')
        self.errors = (self.folder/'raw/errors.jsonl').open('x',encoding='utf-8')
        self.last_T = self.last_frame = self.last_phase = None
        self.flips = 0
        self.invalid = 0
        self.status = 'recording'
        self.last_prediction_counter = None
        self.reused_predictions = 0
        self.maximum_record_seconds = 0.
        self.summary()

    def summary(self):
        p = self.folder/'agent_summary.tmp'
        p.write_text(json.dumps(dict(status=self.status,callbacks=self.n,invalid_rows=self.invalid,
            flips=self.flips,packet_receipts=self.receipts['packet'],prediction_receipts=self.receipts['prediction'],
            prediction_reuse_callbacks=self.reused_predictions,maximum_record_seconds=self.maximum_record_seconds,
            agent_index=self.index,team=self.team)),encoding='utf-8')
        os.replace(p,self.folder/'agent_summary.json')

    def write(self, row):
        self.log.write(json.dumps(row,allow_nan=False)+'\n')
        self.log.flush()  # Preserve each processed callback; no resampling/buffering loss.

    def get_output(self, packet):
        start = time.monotonic()
        self.n += 1
        T = float(packet.match_info.seconds_elapsed)
        frame = int(packet.match_info.frame_num)
        phase = str(packet.match_info.match_phase)
        pred = self.ball_prediction
        before = sequence(self.active_sequence)
        row = dict(match_id=self.folder.name,session_id=self.folder.name,callback=self.n,frame=frame,T=T,
            callback_dt=None if self.last_T is None else T-self.last_T,
            frame_gap=None if self.last_frame is None else frame-self.last_frame,
            match_phase=phase,agent_index=self.index,team=self.team,side='Orange' if self.team==1 else 'Blue',
            packet_receipt=self.latest['packet'],prediction_receipt=self.latest['prediction'],
            previous_submission=self.last_submitted,previous_elapsed=self.last_T,
            phase_transition=None if phase==self.last_phase else dict(previous=self.last_phase,current=phase),
            sequence_before=before,sequence_id_before=None if before is None else self.flips,
            observation18=None,teacher_controls=None,valid=False,errors=[])
        trace = dict(selected=None,selection_attempt=None,consumed_phase=None,flip_trigger=False)
        def profile(f,event,arg):
            if f.f_code is original.find_slice_at_time.__code__ and event=='return' and 'approx_index' in f.f_locals:
                i = f.f_locals['approx_index']
                trace['selection_attempt'] = dict(index=i,first_time=f.f_locals['start_time'],requested_time=f.f_locals['game_time'])
                trace['selected'] = None if arg is None else dict(index=i,timestamp=arg.game_seconds,position=xyz(arg.physics.location))
            elif f.f_code is original.Sequence.tick.__code__ and event=='call': trace['consumed_phase']=f.f_locals['self'].index
            elif f.f_code is original.MyBot.begin_front_flip.__code__ and event=='call': trace['flip_trigger']=True
        saved = sys.getprofile()
        source_error = None
        try:
            sys.setprofile(profile)
            out = original.MyBot.get_output(self,packet)
        except Exception as e:
            source_error = e
            row['errors'].append(dict(stage='original_teacher',type=type(e).__name__,message=repr(e)))
        finally:
            sys.setprofile(saved)
        if trace['flip_trigger']: self.flips += 1
        row.update(teacher_selection=trace,sequence_after=sequence(self.active_sequence),
            sequence_id_after=None if self.active_sequence is None else self.flips,
            hidden_sequence_label_only=True,
            sequence_consumed_phase_label=None if trace['consumed_phase'] is None else ('Jump','Release','Front dodge','Coast')[trace['consumed_phase']])
        if source_error is None:
            row['teacher_controls'] = controls(out)
        try:
            row['car'] = physics(packet.players[self.index].physics)
            row['ball'] = None if not packet.balls else physics(packet.balls[0].physics)
            row['ball_present'] = bool(packet.balls)
            row['prediction_count'] = len(pred.slices)
            row['prediction_first_time'] = None if not pred.slices else float(pred.slices[0].game_seconds)
            row['observation_prediction'] = None
            s = None
            # Prospective prediction probe, independent of teacher's private branch.
            if pred.slices:
                i = int((T+2-pred.slices[0].game_seconds)*120)
                row['observation_prediction_index_attempt'] = i
                s = original.find_slice_at_time(pred,T+2)
                if s is not None:
                    row['observation_prediction'] = dict(index=i,timestamp=s.game_seconds,position=xyz(s.physics.location))
            context = dict(previous_elapsed=self.last_T,
                previous_mode=0 if self.last_submitted is None else self.last_submitted['mode'],
                previous_steer=0. if self.last_submitted is None else self.last_submitted['controls'][1],
                prediction_valid=s is not None, first_time=row['prediction_first_time'])
            row['observation18'] = list(live_adapter(packet,self.index,context,s))
            if source_error is None:
                a = controls(out)
                row['teacher_controls'] = a
                row['action_mode'] = mode(a)
                row['branch'] = ('neutral','chase','jump','front-dodge')[mode(a)]
                row['teacher_logic_branch'] = 'missing_ball' if not packet.balls else 'pending_sequence' if before is not None and not before['done'] else 'far_ball' if trace['selection_attempt'] is not None else 'near_ball'
                if not all(math.isfinite(v) for v in a[:5]): raise ValueError('Nonfinite teacher output')
        except Exception as e:
            row['errors'].append(dict(stage='record_contract',type=type(e).__name__,message=repr(e)))
        row['valid'] = not row['errors'] and row['teacher_controls'] is not None
        counter = None if self.latest['prediction'] is None else self.latest['prediction']['counter']
        if counter is not None and counter==self.last_prediction_counter: self.reused_predictions += 1
        self.last_prediction_counter = counter
        self.write(row)
        if row['errors']:
            self.invalid += 1
            self.errors.write(json.dumps(dict(callback=self.n,errors=row['errors']))+'\n')
            self.errors.flush()
        self.last_T,self.last_frame,self.last_phase = T,frame,phase
        self.maximum_record_seconds=max(self.maximum_record_seconds,time.monotonic()-start)
        if source_error is not None:
            self.status='teacher_error'
            self.summary()
            raise source_error  # Original IndexError and other source exceptions preserved.
        if self.n%120==0: self.summary()
        return out  # Unchanged source output, including on diagnostic invalid rows.

    def retire(self):
        if hasattr(self,'log'):
            self.summary()
            self.log.close()
            self.errors.close()
            (self.folder/'recorder_closed.json').write_text(json.dumps(dict(callbacks=self.n,monotonic=time.monotonic())),encoding='utf-8')
        super().retire()

if __name__=='__main__': Recorder('local/v9_source_recorder').run()
