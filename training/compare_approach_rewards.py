"""Offline counterfactual reward comparison over diagnose_steps.py JSON reports.

No simulator, checkpoint loading, gradients, or training are involved. These
scores describe recorded trajectories; they do not predict a retrained policy.
"""
import argparse
import json
from pathlib import Path

from rlgym.rocket_league.common_values import CAR_MAX_SPEED, TICKS_PER_SECOND


ACTION_REPEAT = 8
MAX_STEP_DISTANCE = CAR_MAX_SPEED * ACTION_REPEAT / TICKS_PER_SECOND


def score_report(report):
    if report.get('format_version') != 1 or not isinstance(report.get('steps'), list):
        raise ValueError('Expected a diagnose_steps.py format_version=1 report')
    if report.get('contract', {}).get('action_repeat') != ACTION_REPEAT:
        raise ValueError('Action-repeat contract differs from this counterfactual')
    approach_weight = report.get('checkpoint_config', {}).get('approach_weight')
    if approach_weight is None or approach_weight <= 0:
        raise ValueError('A positive checkpoint approach_weight is required')

    touched_episodes = set()
    rows = []
    for source in report['steps']:
        episode = source['episode']
        touched_before = episode in touched_episodes
        touch_now = source['state']['ball_touches'] > 0
        # One report row spans 8 physics ticks. If a touch occurs inside it,
        # pre- vs post-touch motion cannot be separated, so exclude that row.
        pre_touch = not touched_before and not touch_now
        if touch_now:
            touched_episodes.add(episode)
        no_propulsion = (source['action']['throttle'] == 0 and
                         source['action']['boost'] == 0)
        distance_delta = source['state']['distance_progress']
        closing_delta = source['state']['velocity_toward_ball_change']
        old = source['reward']['approach']
        # All candidate rewards are nonnegative and weighted equally so that
        # shares can be compared; reported old reward retains its signed value.
        gated_old_positive = max(0., old) if pre_touch else 0.
        distance = (approach_weight * min(1., max(0., distance_delta / MAX_STEP_DISTANCE))
                    if pre_touch else 0.)
        closing = (approach_weight * min(1., max(0., closing_delta / CAR_MAX_SPEED))
                   if pre_touch else 0.)
        rows.append(dict(episode=episode, step=source['step'],
                         pre_touch=pre_touch, touch_on_step=touch_now,
                         no_propulsion_command=no_propulsion,
                         airborne_interval=(not source['state']['on_ground_before'] or
                                            not source['state']['on_ground_after']),
                         throttle=source['action']['throttle'],
                         boost=source['action']['boost'],
                         distance_progress=distance_delta,
                         closing_velocity_change=closing_delta,
                         old_recorded_approach=old,
                         gated_old_positive_approach=gated_old_positive,
                         gated_distance_progress=distance,
                         gated_closing_velocity_improvement=closing))

    def aggregate(field):
        positive = [row for row in rows if row[field] > 0]
        total = sum(row[field] for row in rows)
        no_propulsion = [row for row in positive if row['no_propulsion_command']]
        no_propulsion_total = sum(row[field] for row in no_propulsion)
        airborne_no_propulsion = [row for row in no_propulsion if row['airborne_interval']]
        return dict(total=total, positive_steps=len(positive),
                    no_propulsion_positive_steps=len(no_propulsion),
                    no_propulsion_reward=no_propulsion_total,
                    no_propulsion_reward_share=(no_propulsion_total / total if total > 0 else None),
                    airborne_no_propulsion_positive_steps=len(airborne_no_propulsion))

    summary = dict(steps=len(rows), pre_touch_steps=sum(r['pre_touch'] for r in rows),
                   touch_steps=sum(r['touch_on_step'] for r in rows),
                   old_recorded_approach=aggregate('old_recorded_approach'),
                   gated_old_positive_approach=aggregate('gated_old_positive_approach'),
                   gated_distance_progress=aggregate('gated_distance_progress'),
                   gated_closing_velocity_improvement=aggregate(
                       'gated_closing_velocity_improvement'))
    return dict(checkpoint=report.get('checkpoint'),
                checkpoint_step=report.get('checkpoint_step'),
                fixture=report.get('fixture'), approach_weight=approach_weight,
                summary=summary, steps=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an existing comparison report')
    inputs = []
    for path in (args.reference, args.candidate):
        with path.open(encoding='utf-8') as file:
            inputs.append(json.load(file))
    if inputs[0].get('fixture') != inputs[1].get('fixture'):
        raise ValueError('Traces must use the same scenario, seed, mode, duration, and episode count')
    result = dict(format_version=1,
                  interpretation=('Offline counterfactual on fixed trajectories only; '
                                  'first-touch action intervals are excluded because an '
                                  '8-tick row cannot be split at the contact tick.'),
                  reference=score_report(inputs[0]), candidate=score_report(inputs[1]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, indent=2)
        file.write('\n')
    print(json.dumps({name: result[name]['summary'] for name in ('reference', 'candidate')},
                     indent=2))
    print(f'Report saved: {args.output}')


if __name__ == '__main__':
    main()
