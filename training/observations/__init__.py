"""Single observation definition for training, evaluation, and future deployment."""
import numpy as np
from rlgym.rocket_league.obs_builders import DefaultObs

OBS_SIZE = 92
OBS_CONFIG = dict(zero_padding=None, pos_coef=1 / 2300, ang_coef=1 / np.pi,
                  lin_vel_coef=1 / 2300, ang_vel_coef=1 / np.pi,
                  pad_timer_coef=1 / 10, boost_coef=1 / 100)


def obs_builder():
    return DefaultObs(**OBS_CONFIG)


def preprocess(obs):
    result = np.asarray(obs, dtype=np.float32)
    if result.shape != (OBS_SIZE,) or not np.isfinite(result).all():
        raise ValueError('Expected a finite 92-dimensional DefaultObs observation')
    return result
