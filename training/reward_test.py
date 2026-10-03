"""Validate real simulator touches, goals for both teams, and timeouts."""
import numpy as np
from rlgym.rocket_league.action_parsers import LookupTableAction
from smoke_test import build_env


def main():
    env = build_env()
    table = LookupTableAction.make_lookup_table()
    idle = int(np.flatnonzero(np.all(table == 0, axis=1))[0])
    try:
        env.reset()
        state = env.state
        car = state.cars['blue-0']
        car.physics.position = np.array([0, -150, 17], dtype=np.float32)
        car.physics.euler_angles = np.array([0, np.pi / 2, 0], dtype=np.float32)
        car.physics.linear_velocity = np.array([0, 1000, 0], dtype=np.float32)
        env.set_state(state)
        actions = {agent: np.array([idle]) for agent in env.agents}
        touched = False
        for _ in range(15):
            obs, rewards, terminated, truncated = env.step(actions)
            assert set(rewards) == set(env.agents)
            if env.state.cars['blue-0'].ball_touches:
                assert rewards['blue-0'] == 0.1 and rewards['orange-0'] == 0
                assert not any(terminated.values()) and not any(truncated.values())
                print(f"Touch event detected: agent=blue-0 rewards={rewards}")
                touched = True
                break
        assert touched, 'Controlled collision did not produce a touch'
        for direction, scorer in ((1, 'blue-0'), (-1, 'orange-0')):
            env.reset()
            state = env.state
            state.ball.position = np.array([0, direction * 5000, 100], dtype=np.float32)
            state.ball.linear_velocity = np.array([0, direction * 2000, 0], dtype=np.float32)
            env.set_state(state)
            for _ in range(15):
                obs, rewards, terminated, truncated = env.step(actions)
                assert set(rewards) == set(env.agents)
                if any(terminated.values()):
                    assert all(terminated.values()) and not any(truncated.values())
                    assert rewards[scorer] == 10
                    assert rewards[next(a for a in env.agents if a != scorer)] == -10
                    print(f"Goal event detected: agent={scorer} rewards={rewards} terminated={terminated}")
                    break
            else:
                raise AssertionError('Controlled shot did not score')
        env.reset()
        for _ in range(30):
            _, rewards, terminated, truncated = env.step(actions)
        assert all(truncated.values()) and not any(terminated.values())
        print(f"Timeout detected: truncated={truncated}")
    finally:
        env.close()
    print('PASS: real touches, both goal reward signs, agent IDs, termination and truncation')


if __name__ == '__main__':
    main()
