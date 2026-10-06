"""Isolated Epic 1v1 live check. Run --check without launching any process."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ORIGINAL = PROJECT / 'python-example-original'
INTERPRETER = PROJECT / 'python-example' / 'venv' / 'Scripts' / 'python.exe'
REVISION = 'fd061f457bf19175b4a9b3b3d7811a987044c64d'
AGENT_NAME = 'Recovered Python Example'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources():
    revision = subprocess.check_output(['git', '-C', str(ORIGINAL), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise RuntimeError(f'Original revision changed: {revision}')
    manifest = json.loads((HERE / 'source_hashes.json').read_text(encoding='utf-8'))
    changed = [name for name, expected in manifest.items()
               if not (PROJECT / name).is_file() or sha(PROJECT / name) != expected]
    if changed:
        raise RuntimeError(f'Protected source files changed since preparation: {changed}')
    return manifest


def prepare(folder, plain):
    folder.mkdir(parents=True, exist_ok=False)
    command = ([str(INTERPRETER), '-B', str(ORIGINAL / 'src' / 'bot.py')] if plain else
               [str(INTERPRETER), '-B', str(HERE / 'diagnostic_bot.py'), '--log-dir', str(folder)])
    # RLBotServer starts run_command through cmd.exe. A leading quoted
    # executable path can lose its quotes in that extra shell layer.
    # Use a space-free local batch name; quote paths inside the batch itself.
    help_command = [str(INTERPRETER), '-B', str(HERE / 'diagnostic_bot.py'), '--help']
    (folder / 'run_agent.cmd').write_text(
        '@echo off\n'
        'if /I "%~1"=="--check-launch" goto check\n'
        f'{subprocess.list2cmdline(command)}\n'
        'exit /b %errorlevel%\n'
        ':check\n'
        f'{subprocess.list2cmdline(help_command)}\n'
        'exit /b %errorlevel%\n', encoding='utf-8')
    (folder / 'bot.toml').write_text(
        '[settings]\n'
        'name = "Recovered Python Example"\n'
        f'loadout_file = {json.dumps(str(ORIGINAL / "src" / "loadout.toml"))}\n'
        f'root_dir = {json.dumps(str(folder))}\n'
        'run_command = "run_agent.cmd"\n'
        'agent_id = "rlbot_community/python_example"\n', encoding='utf-8')
    # Preserve all original match/mutator settings; change only launcher,
    # rendering visibility, and the isolated agent configuration path.
    text = (ORIGINAL / 'rlbot.toml').read_text(encoding='utf-8')
    if text.count('launcher = "Steam"') != 1 or text.count('config_file = "src/bot.toml"') != 1:
        raise RuntimeError('Unexpected original launch config; stop for review.')
    text = text.replace('launcher = "Steam"', 'launcher = "Epic"')
    text = text.replace('enable_rendering = false', 'enable_rendering = true')
    text = text.replace('config_file = "src/bot.toml"', 'config_file = "bot.toml"')
    (folder / 'match.toml').write_text(text, encoding='utf-8')
    parsed = tomllib.loads(text)
    cars = parsed['cars']
    if len(cars) != 2 or cars[0].get('type') != 'Human' or cars[0].get('team') != 'Blue' or cars[1].get('team') != 'Orange':
        raise RuntimeError('Expected blue human versus orange original bot.')
    from rlbot.config import load_player_config
    load_player_config(folder / 'bot.toml', team=1)
    # MatchManager's proven path route sends StartCommand to RLBotServer.
    # Do not substitute the Python helper's separate match-enum parser.
    return folder / 'match.toml'


class SummaryReader:
    """Keep the last complete snapshot across brief Windows file-sharing races."""

    def __init__(self):
        self.snapshot = {}
        self.read_failures = 0
        self.failure_since = None

    def read(self, path):
        try:
            snapshot = json.loads(path.read_text(encoding='utf-8'))
        except (PermissionError, FileNotFoundError, json.JSONDecodeError):
            if not self.snapshot and not path.exists():
                return {}  # The agent has not written its first snapshot yet.
            self.read_failures += 1
            now = time.monotonic()
            if self.failure_since is None:
                self.failure_since = now
            if now - self.failure_since >= 10:
                raise RuntimeError('Bot diagnostics unreadable for 10 seconds; inspect agent output.')
            return self.snapshot
        self.snapshot = snapshot
        self.failure_since = None
        return snapshot


def agent_startup(folder, packet, plain, summary_reader=None):
    """Match activity alone does not establish that our agent started."""
    roster_ok = packet is not None and len(packet.players) == 2 and sorted(
        int(player.team) for player in packet.players) == [0, 1]
    teacher = next((player for player in packet.players
                    if int(player.team) == 1 and player.name == AGENT_NAME), None) if roster_ok else None
    input_observed = teacher is not None and any(
        bool(getattr(teacher.last_input, field)) for field in
        ('throttle', 'steer', 'pitch', 'yaw', 'roll', 'jump', 'boost', 'handbrake'))
    if plain:
        return dict(ready=bool(roster_ok and input_observed), initialized=None,
                    controls_observed=bool(input_observed), original_error=None)
    path = folder / 'bot_summary.json'
    summary = (summary_reader or SummaryReader()).read(path)
    return dict(ready=bool(teacher is not None and summary.get('initialized') and
                          summary.get('latest_sample') and summary.get('callbacks', 0) > 0),
                initialized=bool(summary.get('initialized')),
                controls_observed=bool(input_observed),
                original_error=summary.get('last_error'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Preflight only; no RLBotServer/game launch.')
    parser.add_argument('--plain', action='store_true', help='Execute original bot directly without wrapper.')
    parser.add_argument('--v0', action='store_true', help='Full-match V0 harness verification; retain instrumentation.')
    args = parser.parse_args()
    if args.v0 and args.plain:
        parser.error('V0 requires instrumentation; do not combine --v0 and --plain.')
    if Path(sys.executable).resolve() != INTERPRETER.resolve():
        raise RuntimeError(f'Use the existing live interpreter: {INTERPRETER}')
    manifest = verify_sources()
    folder = HERE / 'sessions' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    config = prepare(folder, args.plain)
    metadata = dict(source_revision=REVISION, mode='plain' if args.plain else 'instrumented',
                    experiment='V0' if args.v0 else 'live_reference',
                    interpreter=sys.executable, python=sys.version,
                    versions={name: importlib.metadata.version(name) for name in ('rlbot', 'rlbot-flatbuffers', 'psutil')},
                    source_hashes=manifest,
                    launch_changes=['Steam to Epic', 'rendering enabled', 'isolated agent config/interpreter'],
                    match_config_sha256=sha(folder / 'match.toml'),
                    agent_config_sha256=sha(folder / 'bot.toml'),
                    agent_launcher_sha256=sha(folder / 'run_agent.cmd'))
    (folder / 'session_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Session: {folder}', flush=True)
    print('1v1: blue human versus orange recovered example; five-minute match; replays/countdown retained.', flush=True)
    if args.check:
        # Import only, to verify the original source/dependency path. No bot instance or connection.
        import diagnostic_bot
        # Exercise the actual batch/shell quoting without connecting any agent.
        subprocess.run([os.environ.get('COMSPEC', 'cmd.exe'), '/d', '/c',
                        'run_agent.cmd', '--check-launch'], cwd=folder, check=True)
        print('PREFLIGHT PASS: source hashes, imports, 1v1 config, and batch command quoting. No game launched.', flush=True)
        return
    from rlbot import flat
    from rlbot.managers import MatchManager
    manager = MatchManager()
    summary_reader = SummaryReader()
    outcome = dict(status='starting', final_scores=None, phases_seen=[], protected_sources_unchanged=None,
                   experiment='V0' if args.v0 else 'live_reference',
                   teacher_ready=False, teacher_initialized=None, teacher_controls_observed=False)
    try:
        if args.v0:
            print('V0 ONLY: let the full match end naturally, including overtime if tied. Ctrl+C makes V0 incomplete.', flush=True)
        else:
            print('Launching Epic/RLBot. Play or observe for 3-5 minutes; Ctrl+C ends the test.', flush=True)
        # Avoid writes from child imports in either protected directory.
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        manager.start_match(config, wait_for_start=False)
        deadline = time.monotonic() + 180
        last_print = 0
        ready = False
        while True:
            packet = manager.packet
            agent = agent_startup(folder, packet, args.plain, summary_reader)
            outcome['teacher_initialized'] = agent['initialized']
            outcome['teacher_controls_observed'] |= agent['controls_observed']
            if agent['original_error']:
                raise RuntimeError(f"Original bot error: {agent['original_error']}. See bot_events.jsonl and agent output.")
            if packet is not None:
                phase = str(packet.match_info.match_phase)
                if phase not in outcome['phases_seen']:
                    outcome['phases_seen'].append(phase)
                if agent['ready'] and packet.match_info.match_phase not in (flat.MatchPhase.Inactive, flat.MatchPhase.Ended):
                    if not ready:
                        print('TEACHER READY: named orange agent and live control callbacks verified.', flush=True)
                    ready = True
                    outcome['teacher_ready'] = True
                    outcome['status'] = 'running'
                outcome['final_scores'] = {str(t.team_index): t.score for t in packet.teams}
                outcome['last_frame'] = packet.match_info.frame_num
                outcome['last_game_elapsed'] = packet.match_info.seconds_elapsed
                if time.monotonic() - last_print >= 10:
                    outcome['last_roster'] = [dict(name=p.name, team=int(p.team)) for p in packet.players]
                    label = 'LIVE' if ready else 'WAITING FOR TEACHER'
                    print(f"{label} frame={packet.match_info.frame_num} phase={phase} scores={outcome['final_scores']}", flush=True)
                    last_print = time.monotonic()
                if ready and packet.match_info.match_phase == flat.MatchPhase.Ended:
                    outcome['status'] = 'match_ended'
                    print('MATCH ENDED: natural end phase observed; saving results for review.', flush=True)
                    break
            if not ready and time.monotonic() > deadline:
                raise TimeoutError('Teacher did not initialize and produce controls within 180 seconds. Match activity alone is insufficient; inspect agent startup output.')
            time.sleep(0.1)
    except KeyboardInterrupt:
        outcome['status'] = 'stopped_by_user'
        print('Stopping live test.', flush=True)
    except Exception as error:
        outcome.update(status='error', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        outcome['diagnostic_read_failures'] = summary_reader.read_failures
        try:
            manager.stop_match()
        except Exception as error:
            outcome['cleanup_error'] = str(error)
        try:
            manager.disconnect()
        except Exception as error:
            outcome['disconnect_error'] = str(error)
        try:
            verify_sources()
            outcome['protected_sources_unchanged'] = True
        except Exception as error:
            outcome['protected_sources_unchanged'] = False
            outcome['source_check_error'] = str(error)
        (folder / 'run_summary.json').write_text(json.dumps(outcome, indent=2), encoding='utf-8')
        print(f'Live result files: {folder}', flush=True)
        print('Share run_summary.json, bot_summary.json (instrumented mode), and your gameplay observations.', flush=True)


if __name__ == '__main__':
    main()
