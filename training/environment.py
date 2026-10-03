"""Raw RLGym environment and fixed scripted opponent; CPU simulation only."""
import random
import numpy as np
from rlgym.api import RLGym, StateMutator
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence
from observations import obs_builder, preprocess, OBS_CONFIG
from rewards import baseline_reward, guide_early_reward

ACTION_REPEAT = 8
ACTION_TABLE = LookupTableAction.make_lookup_table()


def contract():
    return dict(observation='RLGym DefaultObs 92 float32', obs_config=OBS_CONFIG,
                action_table=ACTION_TABLE.tolist(), action_repeat=ACTION_REPEAT,
                physics_ticks_per_second=120, rlbot_delay=True,
                control_order=['throttle', 'steer', 'pitch', 'yaw', 'roll', 'jump', 'boost', 'handbrake'])


class TouchStart(StateMutator):
    """Controlled 1v1 diagnostic: learner near ball, idle opponent far away."""
    def __init__(self):
        self.episode = 0

    def apply(self, state, shared_info):
        learner = 'blue-0' if self.episode % 2 == 0 else 'orange-0'
        self.episode += 1
        for agent, car in state.cars.items():
            direction = 1 if car.team_num == 0 else -1
            distance = 550 if agent == learner else 3500
            offset = random.uniform(-150, 150) if agent == learner else 0
            car.physics.position = np.array([offset, -direction * distance, 17], dtype=np.float32)
            car.physics.euler_angles = np.array([0, direction * np.pi / 2, 0], dtype=np.float32)


class ApproachStart(StateMutator):
    """Stationary ball with configurable distance, lateral, and heading variation."""
    def __init__(self, distance_range=(900, 1700), lateral_limit=600, heading_limit=0.65):
        self.episode = 0
        self.distance_range = distance_range
        self.lateral_limit = lateral_limit
        self.heading_limit = heading_limit

    def apply(self, state, shared_info):
        learner = 'blue-0' if self.episode % 2 == 0 else 'orange-0'
        self.episode += 1
        for agent, car in state.cars.items():
            direction = 1 if car.team_num == 0 else -1
            if agent == learner:
                distance = random.uniform(*self.distance_range)
                offset = random.uniform(-self.lateral_limit, self.lateral_limit)
                yaw_offset = random.uniform(-self.heading_limit, self.heading_limit)
            else:
                distance, offset, yaw_offset = 3500, 0, 0
            car.physics.position = np.array([offset, -direction * distance, 17], dtype=np.float32)
            car.physics.euler_angles = np.array(
                [0, direction * np.pi / 2 + yaw_offset, 0], dtype=np.float32)


def build_env(seconds=30.0, scenario='kickoff', approach_weight=0.0,
              goal_weight=10.0, touch_weight=0.1, face_weight=0.0, air_weight=0.0,
              approach_positive_only=False, approach_until_first_touch=False,
              reward_recipe='custom'):
    if scenario not in ('kickoff', 'touch', 'bridge', 'approach'):
        raise ValueError('Unknown start scenario')
    if reward_recipe not in ('custom', 'guide_early_v1'):
        raise ValueError('Unknown reward recipe')
    mutators = [FixedTeamSizeMutator(1, 1), KickoffMutator()]
    if scenario == 'touch':
        mutators.append(TouchStart())
    elif scenario == 'bridge':
        # Overlap both the proven 550-unit probe and the easiest wider starts.
        mutators.append(ApproachStart((550, 1100), 300, 0.3))
    elif scenario == 'approach':
        mutators.append(ApproachStart())
    return RLGym(
        state_mutator=MutatorSequence(*mutators),
        obs_builder=obs_builder(),
        action_parser=RepeatAction(LookupTableAction(), repeats=ACTION_REPEAT),
        reward_fn=(guide_early_reward() if reward_recipe == 'guide_early_v1' else
                   baseline_reward(approach_weight, goal_weight, touch_weight,
                                   face_weight, air_weight, approach_positive_only,
                                   approach_until_first_touch)),
        termination_cond=GoalCondition(),
        truncation_cond=TimeoutCondition(seconds), transition_engine=RocketSimEngine())


def scripted_action(state, agent, version='v1'):
    car = state.cars[agent]
    target = state.ball.position.copy()
    if version == 'v2':
        # Approach from behind the ball relative to the opponent's goal.
        direction = 1 if car.team_num == 0 else -1
        goal = np.array([0., direction * 5120.])
        shot = goal - state.ball.position[:2]
        shot /= max(np.linalg.norm(shot), 1.)
        relative = car.physics.position[:2] - state.ball.position[:2]
        lateral = abs(relative[0] * shot[1] - relative[1] * shot[0])
        if np.dot(relative, shot) > -100 or lateral > 250:
            target[:2] -= shot * 350
        else:
            target[:2] += shot * 300
    elif version != 'v1':
        raise ValueError('Unknown scripted opponent version')
    delta = target - car.physics.position
    angle = np.arctan2(delta[1], delta[0]) - car.physics.yaw
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    steer = int(np.sign(angle)) if abs(angle) > 0.12 else 0
    boost = int(abs(angle) < 0.25 and np.linalg.norm(delta[:2]) > 800)
    controls = np.array([1, steer, 0, steer, 0, 0, boost, 0])
    return int(np.flatnonzero(np.all(ACTION_TABLE == controls, axis=1))[0])


class Match:
    """One learner versus one scripted car; alternate learner's team on reset."""
    def __init__(self, seed, seconds, scenario='kickoff', opponent_version='v1',
                 approach_weight=0.0, goal_weight=10.0, touch_weight=0.1,
                 face_weight=0.0, air_weight=0.0, approach_positive_only=False,
                 approach_until_first_touch=False, reward_recipe='custom'):
        self.rng = random.Random(seed)
        self.scenario = scenario
        self.opponent_version = opponent_version
        self.env = build_env(seconds, scenario, approach_weight, goal_weight,
                             touch_weight, face_weight, air_weight, approach_positive_only,
                             approach_until_first_touch, reward_recipe)
        self.episode = 0
        self.reset()

    def reset(self):
        # KickoffMutator uses module-level random: isolate each match's RNG state.
        previous = random.getstate()
        random.setstate(self.rng.getstate())
        try:
            obs = self.env.reset()
            self.rng.setstate(random.getstate())
        finally:
            random.setstate(previous)
        self.agent = 'blue-0' if self.episode % 2 == 0 else 'orange-0'
        self.opponent = 'orange-0' if self.agent == 'blue-0' else 'blue-0'
        self.episode += 1
        self.total_reward = 0.0
        self.length = 0
        self.touches = 0
        self.raw_contacts = 0
        self.ground_touch_steps = 0
        self.goalward_touch_steps = 0
        self.first_touch_by = None
        self.time_to_first_touch_seconds = None
        self.last_touch_by = None
        self.initial_ball_y = float(self.env.state.ball.position[1])
        self.action_count = 0
        self.jump_actions = 0
        self.obs = preprocess(obs[self.agent])
        return self.obs

    def step(self, action):
        direction = 1 if self.agent == 'blue-0' else -1
        before_ball_y = float(self.env.state.ball.position[1])
        learner_grounded_before = bool(self.env.state.cars[self.agent].on_ground)
        self.action_count += 1
        self.jump_actions += int(bool(ACTION_TABLE[action, 5]))
        opponent_action = scripted_action(self.env.state, self.opponent, self.opponent_version) if self.scenario == 'kickoff' else int(
            np.flatnonzero(np.all(ACTION_TABLE == 0, axis=1))[0])
        actions = {self.agent: np.array([action], dtype=np.int64),
                   self.opponent: np.array([opponent_action])}
        observations, rewards, terminated, truncated = self.env.step(actions)
        final_obs = preprocess(observations[self.agent])
        reward = rewards[self.agent]
        term, trunc = bool(terminated[self.agent]), bool(truncated[self.agent])
        self.total_reward += reward
        self.length += 1
        learner_contacts = int(self.env.state.cars[self.agent].ball_touches)
        opponent_contacts = int(self.env.state.cars[self.opponent].ball_touches)
        learner_touched = learner_contacts > 0
        opponent_touched = opponent_contacts > 0
        self.raw_contacts += learner_contacts
        self.touches += int(learner_touched)  # Matches TouchReward: one credit per action step.
        if learner_touched and learner_grounded_before and self.env.state.cars[self.agent].on_ground:
            self.ground_touch_steps += 1  # Conservative ground-contact proxy at step boundaries.
        if learner_touched and not opponent_touched and direction * (
                float(self.env.state.ball.position[1]) - before_ball_y) > 0:
            self.goalward_touch_steps += 1  # Directional proxy, not attribution of a shot.
        if self.first_touch_by is None and (learner_touched or opponent_touched):
            self.first_touch_by = ('same_step' if learner_touched and opponent_touched else
                                   'learner' if learner_touched else 'opponent')
            if learner_touched:
                self.time_to_first_touch_seconds = self.length * ACTION_REPEAT / 120
        elif learner_touched and self.time_to_first_touch_seconds is None:
            self.time_to_first_touch_seconds = self.length * ACTION_REPEAT / 120
        if learner_touched or opponent_touched:
            self.last_touch_by = ('same_step' if learner_touched and opponent_touched else
                                  'learner' if learner_touched else 'opponent')
        info = None
        if term or trunc:
            team = self.env.state.cars[self.agent].team_num
            goal_for = int(term and self.env.state.scoring_team == team)
            goal_against = int(term and not goal_for)
            info = dict(episode_reward=self.total_reward, episode_length=self.length,
                        episode_index=self.episode - 1,
                        episode_seconds=self.length * ACTION_REPEAT / 120,
                        touches=self.touches, raw_contacts=self.raw_contacts,
                        ground_touch_steps=self.ground_touch_steps,
                        goalward_touch_steps=self.goalward_touch_steps,
                        first_touch_by=self.first_touch_by,
                        time_to_first_touch_seconds=self.time_to_first_touch_seconds,
                        ball_progress_toward_goal=direction * (
                            float(self.env.state.ball.position[1]) - self.initial_ball_y),
                        goals_for=goal_for, goals_against=goal_against,
                        # A goal's last recorded toucher is only an attribution proxy.
                        opponent_own_goal_proxy=int(goal_for and self.last_touch_by == 'opponent'),
                        learner_own_goal_proxy=int(goal_against and self.last_touch_by == 'learner'),
                        unknown_goal_attribution=int(term and self.last_touch_by in (None, 'same_step')),
                        jump_action_fraction=self.jump_actions / self.action_count,
                        learner_side=self.agent)
            self.reset()
        else:
            self.obs = final_obs
        # Keep pre-reset final observation for correct timeout bootstrapping.
        return self.obs, final_obs, reward, term, trunc, info

    def close(self):
        self.env.close()
