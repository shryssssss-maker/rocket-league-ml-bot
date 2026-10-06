"""Isolated V2 candidate builder; not integrated into training/deployment."""
import array
import math

FEATURES = ('ball_forward','ball_right','ball_up','prediction_forward','prediction_right','prediction_up',
            'ball_distance','car_speed','ball_present','prediction_valid','prediction_horizon',
            'prediction_first_offset','callback_dt','previous_neutral','previous_chase',
            'previous_jump','previous_dodge','previous_steer')


def finite(x):
    if isinstance(x, bool) or not isinstance(x, (int,float)) or not math.isfinite(x):
        raise ValueError('Expected finite numeric input')
    return float(x)


def vector(v):
    if v is None or len(v) != 3:
        raise ValueError('Expected exactly three vector components')
    return tuple(finite(x) for x in v)


def basis(pyr):
    p,y,r = vector(pyr)
    cp,cy,cr = math.cos(p),math.cos(y),math.cos(r)
    sp,sy,sr = math.sin(p),math.sin(y),math.sin(r)
    return ((cp*cy,cp*sy,sp), (cy*sp*sr-cr*sy,sy*sp*sr+cr*cy,-cp*sr),
            (-cr*cy*sp-sr*sy,-cr*sy*sp+sr*cy,cp*cr))


def build(raw, context):
    pos,vel = vector(raw['position']),vector(raw['velocity'])
    axes = raw['basis']
    if len(axes) != 3:
        raise ValueError('Expected three orientation axes')
    axes = tuple(vector(v) for v in axes)
    for i in range(3):
        for j in range(3):
            if abs(sum(a*b for a,b in zip(axes[i],axes[j]))-int(i==j)) > 1e-5:
                raise ValueError('Orientation basis is not orthonormal')
    if type(raw['ball_present']) is not bool or type(context['prediction_valid']) is not bool:
        raise ValueError('Availability masks must be Boolean')
    now = finite(context['elapsed'])
    previous = context['previous_elapsed']
    mode,steer = context['previous_mode'],finite(context['previous_steer'])
    if type(mode) is not int or mode not in range(4) or not -1 <= steer <= 1 or (mode != 1 and steer != 0):
        raise ValueError('Unsupported previous transmitted action')
    if previous is None:
        if mode != 0 or steer != 0:
            raise ValueError('Initialization must have previous Neutral action')
        dt = 0.
    else:
        dt = now-finite(previous)
        if dt <= 0:
            raise ValueError('Duplicate/backward callback clock')
    out = [0.]*18
    out[7] = math.sqrt(sum(x*x for x in vel))/2300.
    out[12] = dt/(1/60)
    out[13+mode] = 1.
    out[17] = steer
    def relative(target):
        delta = [a-b for a,b in zip(vector(target),pos)]
        return [sum(a*b for a,b in zip(delta,axis))/6000 for axis in axes], delta
    if raw['ball_present']:
        out[:3],delta = relative(raw['ball_position'])
        out[6] = math.sqrt(sum(x*x for x in delta))/6000.
        out[8] = 1.
        if context['prediction_valid']:
            out[3:6],_ = relative(context['prediction_position'])
            out[9] = 1.
            out[10] = (finite(context['selected_time'])-now)/2.
            out[11] = (finite(context['first_time'])-now)/(1/120)
    if not all(math.isfinite(x) for x in out):
        raise ValueError('Nonfinite observation')
    result = array.array('f',out)
    if result.itemsize != 4 or len(result) != len(FEATURES) or not all(math.isfinite(x) for x in result):
        raise ValueError('Expected finite 18D float32 observation')
    return result


def live_adapter(packet, index, context, selected_prediction=None):
    if type(index) is not int or not 0 <= index < len(packet.players):
        raise ValueError('Controlled car missing')
    if len(packet.balls) > 1:
        raise ValueError('Only the single-ball contract is supported')
    p = packet.players[index].physics
    xyz = lambda v: [float(v.x),float(v.y),float(v.z)]
    ctx = dict(context,elapsed=float(packet.match_info.seconds_elapsed))
    if packet.balls and ctx['prediction_valid']:
        if selected_prediction is None:
            raise ValueError('Valid prediction requires a selected native slice')
        ctx.update(prediction_position=xyz(selected_prediction.physics.location),
                   selected_time=float(selected_prediction.game_seconds))
    return build(dict(position=xyz(p.location),velocity=xyz(p.velocity),
                      basis=basis([p.rotation.pitch,p.rotation.yaw,p.rotation.roll]),
                      ball_present=bool(packet.balls),
                      ball_position=xyz(packet.balls[0].physics.location) if packet.balls else None),ctx)


def simulation_adapter(state, agent_id, context, ball_present, selected_prediction=None):
    if agent_id not in state.cars:
        raise ValueError('Controlled car missing')
    p = state.cars[agent_id].physics
    ctx = dict(context)
    if ball_present and ctx['prediction_valid']:
        if selected_prediction is None:
            raise ValueError('Valid prediction requires a selected native ball state')
        ctx['prediction_position'] = selected_prediction.pos.as_numpy().tolist()
    # Native non-inverted matrix columns, matching installed PhysicsObject.
    return build(dict(position=p.position.tolist(),velocity=p.linear_velocity.tolist(),
                      basis=[p.forward.tolist(),p.right.tolist(),p.up.tolist()],
                      ball_present=ball_present,
                      ball_position=state.ball.position.tolist() if ball_present else None),ctx)
