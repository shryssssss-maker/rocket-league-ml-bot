"""v13_projection_13d_v1: lossless selection of frozen V10 columns 0..12.

This defines a new model-input view. It never writes source arrays or artifacts,
changes feature values/scales, synthesizes rows, or removes callbacks.
"""
import numpy as np

VERSION='v13_projection_13d_v1'
SOURCE_FEATURES=('ball_forward','ball_right','ball_up','prediction_forward','prediction_right','prediction_up',
                 'ball_distance','car_speed','ball_present','prediction_valid','prediction_horizon','prediction_first_offset',
                 'callback_dt','previous_neutral','previous_chase','previous_jump','previous_dodge','previous_steer')
KEEP_INDICES=tuple(range(13))
FEATURES=SOURCE_FEATURES[:13]
REMOVED_FEATURES=SOURCE_FEATURES[13:]

def project(observations,source_features):
    if tuple(source_features)!=SOURCE_FEATURES:
        raise ValueError('Frozen feature ordering mismatch; no reordering permitted')
    if not isinstance(observations,np.ndarray) or observations.dtype!=np.dtype('float32'):
        raise ValueError('Expected exact float32 observations')
    if observations.ndim!=2 or observations.shape[1]!=18:
        raise ValueError('Expected [callbacks,18]; no padding/truncation fallback')
    return observations[:,KEEP_INDICES].copy()
