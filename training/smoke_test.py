"""Exercise two short headless 1v1 episodes; no learner or live game needed."""

import random
import sys
from importlib.metadata import version

import numpy as np
import rlgym
import RocketSim
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    KickoffMutator,
    MutatorSequence,
)


def build_env() -> RLGym:
    return RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1), KickoffMutator()
        ),
        obs_builder=DefaultObs(zero_padding=None),
        action_parser=RepeatAction(LookupTableAction(), repeats=8),
        reward_fn=CombinedReward((GoalReward(), 10.0), (TouchReward(), 0.1)),
        termination_cond=GoalCondition(),
        # A short timeout guarantees that random play finishes the smoke test.
        truncation_cond=TimeoutCondition(timeout_seconds=2.0),
        transition_engine=RocketSimEngine(),
    )


def check_observations(env: RLGym, observations: dict) -> None:
    assert set(observations) == set(env.agents)
    for agent, obs in observations.items():
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (env.observation_space(agent)[1],)
        assert np.isfinite(obs).all(), f"Non-finite observation for {agent}"


def main() -> None:
    assert sys.version_info[:2] == (3, 12), "Use training/venv with Python 3.12"
    print(f"Python: {sys.version.split()[0]} | interpreter: {sys.executable}")
    for package in ("rlgym", "rlgym-api", "rlgym-rocket-league", "rocketsim", "numpy"):
        print(f"{package}: {version(package)}")
    print("Imports OK: rlgym, RocketSim")

    # KickoffMutator uses Python's RNG; actions use a separate seeded NumPy RNG.
    random.seed(42)
    rng = np.random.default_rng(42)
    env = build_env()
    try:
        for episode in range(1, 3):
            observations = env.reset()
            check_observations(env, observations)
            assert len(env.agents) == 2
            assert {car.team_num for car in env.state.cars.values()} == {0, 1}
            print(f"\nEpisode {episode}: reset OK; agents={env.agents}")
            for agent, obs in observations.items():
                print(f"  {agent}: type={type(obs).__name__}, shape={obs.shape}, "
                      f"dtype={obs.dtype}, sample={obs[:5]}")
            print(f"  Action spaces: {env.action_spaces}")
            initial_positions = {
                agent: car.physics.position.copy() for agent, car in env.state.cars.items()
            }
            totals = dict.fromkeys(env.agents, 0.0)
            for step in range(1, 61):
                # LookupTableAction expects a one-element array of discrete indices.
                actions = {
                    agent: rng.integers(0, space[1], size=(1,))
                    for agent, space in env.action_spaces.items()
                }
                previous_tick = env.state.tick_count
                # RLGym v2 returns four agent-keyed dicts, with no Gymnasium info dict.
                observations, rewards, terminated, truncated = env.step(actions)
                check_observations(env, observations)
                assert env.state.tick_count == previous_tick + 8
                for result in (rewards, terminated, truncated):
                    assert set(result) == set(env.agents)
                assert all(np.isfinite(reward) for reward in rewards.values())
                for agent, reward in rewards.items():
                    totals[agent] += reward
                done = any(terminated.values()) or any(truncated.values())
                if step <= 3 or step % 10 == 0 or done:
                    print(f"  step={step}, tick={env.state.tick_count}, rewards={rewards}, "
                          f"terminated={terminated}, truncated={truncated}")
                if done:
                    assert any(
                        np.linalg.norm(car.physics.position - initial_positions[agent]) > 1
                        for agent, car in env.state.cars.items()
                    ), "Cars did not move"
                    print(f"Episode {episode} finished after {step} steps; totals={totals}")
                    break
            else:
                raise AssertionError("Episode did not finish within the step limit")
    finally:
        env.close()
        print("Environment closed.")
    print("PASS: headless 1v1 creation, reset, actions, movement, observations, rewards, "
          "episode completion, second reset, and close.")


if __name__ == "__main__":
    main()
