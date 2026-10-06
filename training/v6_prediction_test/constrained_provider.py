"""Approved Option A experiment only; no deployment/training integration."""
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import math
from pathlib import Path
import RocketSim as rs

ROOT = Path(__file__).resolve().parents[2]
BINARY_SHA256 = 'e3ee24ca82445b4bfcc754583f6778d7b0d8b7a7f7d64f872be8c65e621a63d0'
COUNT = 241
TICK_INTERVAL = 1


def finite(value):
    if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):
        raise ValueError('Expected finite numeric value')
    return float(value)


def vector(value):
    if len(value)!=3:
        raise ValueError('Expected three components')
    return tuple(finite(x) for x in value)


@dataclass(frozen=True)
class Prediction:
    states: tuple
    game_seconds: tuple


class ConstrainedProvider:
    def __init__(self):
        if version('rocketsim')!='2.2.1':
            raise RuntimeError('Requires pinned RocketSim 2.2.1')
        path=Path(rs.__file__)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=BINARY_SHA256:
            raise RuntimeError('RocketSim binary hash mismatch')
        meshes=ROOT/'training/venv/Lib/site-packages/rlgym/rocket_league/sim/collision_meshes'
        rs.init(str(meshes))
        self.identity=dict(version='2.2.1',binary_sha256=BINARY_SHA256,exact_build_commit=None,
            meshes={str(p.relative_to(meshes)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(meshes.rglob('*')) if p.is_file()})

    def predict(self,ball,T):
        T=finite(T)
        position=vector(ball['position'])
        velocity=vector(ball['velocity'])
        angular=vector(ball['angular_velocity'])
        pitch,yaw,roll=vector(ball['rotation_pyr'])
        # All BallState fields are supplied at construction. No subsequent
        # assignment/copy synchronization workaround is used.
        supplied=rs.BallState(pos=rs.Vec(*position),vel=rs.Vec(*velocity),
            ang_vel=rs.Vec(*angular),rot_mat=rs.Angle(yaw=yaw,pitch=pitch,roll=roll).as_rot_mat())
        arena=rs.Arena(rs.GameMode.SOCCAR)
        try:
            arena.ball.set_state(supplied)
            states=arena.get_ball_prediction(num_states=241,tick_interval=1)
            if len(states)!=COUNT:
                raise RuntimeError('Expected exactly 241 returned states; no trimming permitted')
            for state in states:
                for v in (state.pos,state.vel,state.ang_vel):
                    if not all(math.isfinite(float(x)) for x in v.as_numpy()):
                        raise RuntimeError('Nonfinite native prediction state')
            # Absolute labels only: no correction, interpolation or shifting.
            result=Prediction(tuple(states),tuple(T+i/120 for i in range(COUNT)))
        finally:
            # Each call owns one fresh arena and releases it before returning.
            del arena
        return result
