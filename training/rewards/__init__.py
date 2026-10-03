from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from .approach import ApproachReward
from .facing import FacingBallReward
from .air import InAirReward
from .guide_face_ball import GuideFaceBallReward
from .speed_toward_ball import SpeedTowardBallReward


def guide_early_reward():
    """ZealanL-style early contact recipe; no goal or custom approach terms."""
    return CombinedReward((TouchReward(), 50.0),
                          (SpeedTowardBallReward(), 5.0),
                          (GuideFaceBallReward(), 1.0),
                          (InAirReward(), 0.15))


def baseline_reward(approach_weight=0.0, goal_weight=10.0, touch_weight=0.1,
                    face_weight=0.0, air_weight=0.0, approach_positive_only=False,
                    approach_until_first_touch=False):
    terms = []
    if goal_weight:
        terms.append((GoalReward(), goal_weight))
    if touch_weight:
        terms.append((TouchReward(), touch_weight))
    if approach_weight:
        terms.append((ApproachReward(positive_only=approach_positive_only,
                                     until_first_touch=approach_until_first_touch),
                      approach_weight))
    if face_weight:
        terms.append((FacingBallReward(), face_weight))
    if air_weight:
        terms.append((InAirReward(), air_weight))
    if not terms:
        raise ValueError('At least one reward weight must be nonzero')
    return CombinedReward(*terms)
