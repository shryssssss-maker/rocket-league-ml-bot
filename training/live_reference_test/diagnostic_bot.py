"""Temporary observation-only wrapper; original gameplay source stays untouched."""
import argparse
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT / 'python-example-original' / 'src'
sys.path.insert(0, str(SOURCE))
spec = importlib.util.spec_from_file_location('_recovered_teacher', SOURCE / 'bot.py')
original = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = original
spec.loader.exec_module(original)

# Forward the original lookup exactly once, capturing its actual returned slice.
# This changes only the in-memory function binding, never the source file.
_find_slice = original.find_slice_at_time
_observer = None


def traced_find_slice(prediction, game_time):
    if _observer is not None:
        _observer.prediction_requests += 1
        _observer.prediction = {'requested_time': game_time}
    result = _find_slice(prediction, game_time)  # Original errors propagate.
    if _observer is not None:
        try:
            _observer.prediction_slices_selected += int(result is not None)
            _observer.prediction.update(
                slice_count=len(prediction.slices),
                first_slice_time=prediction.slices[0].game_seconds,
                selected_index=int((game_time - prediction.slices[0].game_seconds) * 120),
                selected_time=None if result is None else result.game_seconds,
                target=None if result is None else xyz(result.physics.location),
            )
        except Exception as error:
            _observer.note_diagnostic_error(error)
    return result


original.find_slice_at_time = traced_find_slice


def xyz(vector):
    return [float(vector.x), float(vector.y), float(vector.z)]


class Diagnostics:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.log = (self.folder / 'bot_events.jsonl').open('w', encoding='utf-8')
        self.bytes_written = 0
        self.truncated = False
        self.callbacks = self.errors = self.invalid_outputs = 0
        self.prediction_requests = self.prediction_slices_selected = 0
        self.prediction_steering_callbacks = self.flip_starts = self.sequence_completions = 0
        self.diagnostic_errors = 0
        self.frame_gaps = Counter()
        self.phases = Counter()
        self.duplicate_frames = self.frame_rollbacks = self.clock_rollbacks = 0
        self.wall_ms_sum = self.wall_ms_max = self.game_dt_max = 0.0
        self.interval_count = 0
        self.previous = None
        self.current_interval = None
        self.last_write = -math.inf
        self.prediction = None
        self.last_touch = None
        self.observed_touch_events = 0
        self.last_error = None
        self.latest = None
        self.initialized = False
        self.summary_replace_retries = 0
        self.summary_deferred_writes = 0
        self.summary_pending_since = None

    def emit(self, event):
        try:
            line = json.dumps(event, allow_nan=False) + '\n'
            size = len(line.encode('utf-8'))
            if self.bytes_written + size <= 1024 * 1024:
                self.log.write(line)
                self.log.flush()
                self.bytes_written += size
            else:
                self.truncated = True
        except Exception as error:
            self.note_diagnostic_error(error)

    def note_diagnostic_error(self, error):
        self.diagnostic_errors += 1
        if self.diagnostic_errors == 1:
            print(f'OG diagnostics error (controls unchanged): {error}', flush=True)

    def callback(self, packet):
        wall = time.perf_counter()
        frame, game = int(packet.match_info.frame_num), float(packet.match_info.seconds_elapsed)
        self.callbacks += 1
        self.phases[str(packet.match_info.match_phase)] += 1
        self.current_interval = None
        if self.previous is not None:
            old_wall, old_frame, old_game = self.previous
            gap, dt, wall_ms = frame - old_frame, game - old_game, (wall - old_wall) * 1000
            self.frame_gaps[str(gap) if -120 <= gap <= 120 else 'outside_120'] += 1
            self.duplicate_frames += int(gap == 0)
            self.frame_rollbacks += int(gap < 0)
            self.clock_rollbacks += int(dt < 0)
            self.interval_count += 1
            self.wall_ms_sum += wall_ms
            self.wall_ms_max = max(self.wall_ms_max, wall_ms)
            self.game_dt_max = max(self.game_dt_max, dt)
            self.current_interval = dict(frame_gap=gap, game_dt=dt, wall_ms=wall_ms)
        self.previous = (wall, frame, game)
        return wall

    def snapshot(self, bot, packet, controls, returned_phase, original_ms):
        car = packet.players[bot.index]
        ball = packet.balls[0] if packet.balls else None
        touch = car.latest_touch
        if touch is not None and touch.game_seconds != self.last_touch:
            self.last_touch = touch.game_seconds
            self.observed_touch_events += 1
            self.emit(dict(event='observed_teacher_touch', frame=packet.match_info.frame_num,
                           touch_time=touch.game_seconds))
        seq = bot.active_sequence
        sample = dict(
            event='sample', frame=int(packet.match_info.frame_num),
            elapsed=float(packet.match_info.seconds_elapsed),
            match_phase=str(packet.match_info.match_phase),
            player_count=len(packet.players), ball_count=len(packet.balls),
            speed=original.Vec3(car.physics.velocity).length(),
            distance=None if ball is None else original.Vec3(car.physics.location).dist(
                original.Vec3(ball.physics.location)),
            car_position=xyz(car.physics.location),
            ball_position=None if ball is None else xyz(ball.physics.location),
            prediction=self.prediction,
            sequence_active=seq is not None and not seq.done,
            sequence_index_after=None if seq is None else seq.index,
            returned_sequence_phase=returned_phase,
            prediction_used_for_steering=bool(returned_phase is None and self.prediction and
                                             self.prediction.get('target') is not None),
            interval=self.current_interval, original_logic_ms=original_ms,
            scores={str(t.team_index): t.score for t in packet.teams},
            air_state=str(car.air_state),
            controls={key: getattr(controls, key) for key in
                      ('throttle', 'steer', 'pitch', 'yaw', 'roll', 'jump', 'boost', 'handbrake')},
        )
        if returned_phase is not None:
            step = seq.steps[returned_phase]
            sample['returned_sequence_step'] = dict(
                start_time=step.start_time, nominal_duration=step.duration,
                elapsed=None if step.start_time is None else sample['elapsed'] - step.start_time)
        if ball is not None and self.prediction and self.prediction.get('target') is not None:
            sample['prediction_target_distance_from_current_ball'] = math.dist(
                self.prediction['target'], xyz(ball.physics.location))
        self.latest = sample
        return sample

    def save(self, final=False):
        """Retry sharing races briefly, or defer to the next periodic save.

        Ordinary callbacks never sleep. Retirement permits a bounded 200 ms
        flush so a pending final snapshot is not silently abandoned.
        """
        deadline = time.perf_counter() + 0.2 if final else None
        attempts = 0
        try:
            while True:
                summary = dict(
                    initialized=self.initialized,
                    callbacks=self.callbacks, original_errors=self.errors,
                    last_error=self.last_error, diagnostic_errors=self.diagnostic_errors,
                    invalid_outputs=self.invalid_outputs,
                    prediction_requests=self.prediction_requests,
                    prediction_slices_selected=self.prediction_slices_selected,
                    prediction_steering_callbacks=self.prediction_steering_callbacks,
                    flip_starts=self.flip_starts, sequence_completions=self.sequence_completions,
                    observed_teacher_touch_events=self.observed_touch_events,
                    touch_count_note='Distinct latest-touch timestamps observed; skipped packets can hide touches.',
                    phase_callback_counts=dict(self.phases), frame_gap_histogram=dict(self.frame_gaps),
                    duplicate_frames=self.duplicate_frames, frame_rollbacks=self.frame_rollbacks,
                    game_clock_rollbacks=self.clock_rollbacks,
                    callback_wall_ms_mean=self.wall_ms_sum / max(self.interval_count, 1),
                    callback_wall_ms_max=self.wall_ms_max, game_dt_max=self.game_dt_max,
                    events_log_capped=self.truncated, latest_sample=self.latest,
                    summary_replace_retries=self.summary_replace_retries,
                    summary_deferred_writes=self.summary_deferred_writes,
                    timing_note='Callbacks include instrumentation overhead; frame gaps are observations, not automatic failures.',
                )
                path = self.folder / 'bot_summary.json'
                temp = path.with_suffix('.tmp')
                temp.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
                try:
                    temp.replace(path)
                except PermissionError:
                    self.summary_replace_retries += 1
                    attempts += 1
                    now = time.perf_counter()
                    if self.summary_pending_since is None:
                        self.summary_pending_since = now
                    if final:
                        if now >= deadline:
                            raise
                        time.sleep(0.01)  # Only retirement; never a control callback.
                        continue
                    if now - self.summary_pending_since >= 10:
                        raise
                    if attempts < 3:
                        continue  # Bounded immediate retries without sleeping.
                    self.summary_deferred_writes += 1
                    return False  # Keep the last complete file; next save retries.
                self.summary_pending_since = None
                return True
        except Exception as error:
            self.note_diagnostic_error(error)
            return False


class InstrumentedBot(original.MyBot):
    def __init__(self, folder):
        self.diagnostics = Diagnostics(folder)
        self._flip_started = False
        self._last_sequence_key = None
        super().__init__('rlbot_community/python_example')

    def initialize(self):
        try:
            super().initialize()
        except Exception as error:
            self.record_error(error, 'initialize')
            raise
        self.diagnostics.initialized = True
        self.diagnostics.emit(dict(event='initialized', source=str(SOURCE / 'bot.py'),
                                   player_index=self.index, team=self.team))
        self.diagnostics.save()
        print(f'OG reference ready; bounded diagnostics: {self.diagnostics.folder}', flush=True)

    def record_error(self, error, stage):
        d = self.diagnostics
        d.errors += 1
        d.last_error = dict(stage=stage, type=type(error).__name__, message=str(error))
        d.emit(dict(event='original_error', **d.last_error))
        if d.errors == 1 or time.perf_counter() - d.last_write >= 2:
            d.save()
            d.last_write = time.perf_counter()

    def begin_front_flip(self, packet):
        controls = super().begin_front_flip(packet)
        self._flip_started = True
        self.diagnostics.flip_starts += 1
        self.diagnostics.emit(dict(event='flip_started', frame=packet.match_info.frame_num,
                                   elapsed=packet.match_info.seconds_elapsed))
        return controls

    def get_output(self, packet):
        global _observer
        d = self.diagnostics
        started = d.callback(packet)
        d.prediction = None
        self._flip_started = False
        seq = self.active_sequence
        returned_phase = seq.index if seq is not None and not seq.done and packet.balls else None
        _observer = d
        try:
            controls = super().get_output(packet)  # Exactly one original decision.
        except Exception as error:
            self.record_error(error, 'get_output')
            raise  # Never hide an original failure or supply replacement controls.
        finally:
            _observer = None
        original_ms = (time.perf_counter() - started) * 1000
        try:
            if self._flip_started:
                returned_phase = 0
            d.prediction_steering_callbacks += int(returned_phase is None and d.prediction is not None
                                                  and d.prediction.get('target') is not None)
            invalid = any(not math.isfinite(getattr(controls, k)) or not -1 <= getattr(controls, k) <= 1
                          for k in ('throttle', 'steer', 'pitch', 'yaw', 'roll'))
            d.invalid_outputs += int(invalid)
            seq = self.active_sequence
            key = None if seq is None else (id(seq), seq.index, seq.done, returned_phase)
            changed = key != self._last_sequence_key
            if seq is not None and seq.done and (self._last_sequence_key is None or
                                                not self._last_sequence_key[2]):
                d.sequence_completions += 1
            periodic = started - d.last_write >= 2
            # Touch counts are checked each callback, independently of sampled logging.
            car = packet.players[self.index]
            if car.latest_touch is not None and car.latest_touch.game_seconds != d.last_touch:
                d.last_touch = car.latest_touch.game_seconds
                d.observed_touch_events += 1
                d.emit(dict(event='observed_teacher_touch', frame=packet.match_info.frame_num,
                            touch_time=car.latest_touch.game_seconds))
            if changed or periodic:
                sample = d.snapshot(self, packet, controls, returned_phase, original_ms)
                if changed:
                    d.emit(dict(sample, event='sequence_state_changed'))
                if periodic:
                    d.emit(sample)
                    d.save()
                    print(f"OG frame={sample['frame']} t={sample['elapsed']:.2f} "
                          f"speed={sample['speed']:.0f} distance={sample['distance']} "
                          f"prediction={bool(d.prediction and d.prediction.get('target'))} "
                          f"seq={returned_phase}->{sample['sequence_index_after']} "
                          f"throttle={controls.throttle:+.1f} steer={controls.steer:+.3f} "
                          f"pitch={controls.pitch:+.1f} jump={int(controls.jump)}", flush=True)
                    d.last_write = started
                self._last_sequence_key = key
        except Exception as error:
            d.note_diagnostic_error(error)
        return controls  # Same controller object, with no field changes.

    def retire(self):
        self.diagnostics.save(final=True)
        self.diagnostics.log.close()
        super().retire()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=Path, required=True)
    args = parser.parse_args()
    InstrumentedBot(args.log_dir).run()
