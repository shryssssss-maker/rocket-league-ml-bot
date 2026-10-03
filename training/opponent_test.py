"""Controlled check that v2 targets and scores toward each team's enemy goal."""
import numpy as np
from environment import build_env, scripted_action, ACTION_TABLE


def main():
    env = build_env()
    idle = int(np.flatnonzero(np.all(ACTION_TABLE == 0, axis=1))[0])
    try:
        for agent, direction in (('blue-0', 1), ('orange-0', -1)):
            env.reset()
            state = env.state
            other = 'orange-0' if agent == 'blue-0' else 'blue-0'
            state.cars[agent].physics.position = np.array([0, -direction * 1200, 17], dtype=np.float32)
            state.cars[agent].physics.euler_angles = np.array([0, direction * np.pi/2, 0], dtype=np.float32)
            state.cars[other].physics.position = np.array([3000, 0, 17], dtype=np.float32)
            env.set_state(state)
            touched = False
            for step in range(450):
                action = scripted_action(env.state, agent, 'v2')
                _, rewards, terminated, _ = env.step({agent: np.array([action]), other: np.array([idle])})
                touched |= bool(env.state.cars[agent].ball_touches)
                if any(terminated.values()):
                    assert touched and rewards[agent] >= 10 and rewards[other] == -10
                    print(f'PASS: {agent} v2 touches and scores at step {step+1}, rewards={rewards}')
                    break
            else:
                raise AssertionError(f'{agent} failed the controlled scoring check')
    finally:
        env.close()


if __name__ == '__main__':
    main()
