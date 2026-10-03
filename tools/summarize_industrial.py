#!/usr/bin/env python3
"""Summarize recorded industrial flights without running inference or physics.

Usage: python tools/summarize_industrial.py results/industrial/RUN [OTHER_RUN ...]
Distances to the obstacle are measured center distances, never body clearance.
Passage phase describes recorded motion, not causal proof of learned avoidance.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def vector(value):
    return value if isinstance(value, list) and len(value) == 3 and all(number(x) for x in value) else None


def difference(a, b):
    return [x-y for x, y in zip(a, b)]


def norm(a):
    return math.sqrt(sum(x*x for x in a))


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def stats(values):
    values = [float(x) for x in values if number(x)]
    return dict(count=len(values), mean=statistics.fmean(values), maximum=max(values), total=sum(values)) if values else dict(count=0, mean=None, maximum=None, total=0.)


def window(rows, origin, lower, upper):
    selected = [row for row in rows if lower <= row['t'] <= upper]
    displacement = [norm(difference(row['position'], origin)) for row in selected] if origin else []
    distances = [norm(difference(row['position'], row['obstacle']['actual_position']))
                 for row in selected if vector(row.get('obstacle', {}).get('actual_position'))]
    return dict(samples=len(selected), recorded_time_range_s=[selected[0]['t'], selected[-1]['t']] if selected else None,
                vehicle_path_length_m=sum(norm(difference(b['position'], a['position'])) for a, b in zip(selected[:-1], selected[1:])),
                max_vehicle_displacement_from_start_m=max(displacement) if displacement else None,
                vehicle_left_start_by_more_than_0_2m=any(x > .2 for x in displacement) if displacement else None,
                minimum_actual_center_distance_m=min(distances) if distances else None)


def passage(rows, origin, goal, obstacle, first_departure):
    anchor = vector(obstacle.get('anchor_world'))
    if not origin or not goal or not anchor:
        return dict(classification='insufficient_route_or_obstacle_data', crossing=None)
    route = difference(goal, origin)
    length = norm(route)
    if length < 1e-8:
        return dict(classification='zero_length_route', crossing=None)
    direction = [x/length for x in route]
    crossing = None
    # A route-normal plane through the authored anchor is an evaluator landmark.
    # Interpolate adjacent 100 Hz logged positions; do not invent contact truth.
    for before, after in zip(rows[:-1], rows[1:]):
        a, b = dot(difference(before['position'], anchor), direction), dot(difference(after['position'], anchor), direction)
        if a < 0 <= b and b > a:
            alpha = -a/(b-a)
            t = before['t'] + alpha*(after['t']-before['t'])
            position = [x+alpha*(y-x) for x, y in zip(before['position'], after['position'])]
            offset = difference(position, origin)
            lateral = norm([x-dot(offset, direction)*d for x, d in zip(offset, direction)])
            before_box = vector(before.get('obstacle', {}).get('actual_position'))
            after_box = vector(after.get('obstacle', {}).get('actual_position'))
            center = [x+alpha*(y-x) for x, y in zip(before_box, after_box)] if before_box and after_box else None
            crossing = dict(time_s=t, interpolated_vehicle_position=position,
                            departure_from_straight_route_m=lateral,
                            actual_obstacle_center_distance_m=norm(difference(position, center)) if center else None)
            break
    if crossing is None:
        label = 'no_recorded_forward_passage'
    else:
        t = crossing['time_s']
        if t < 3:
            label = 'passed_before_obstacle_entered'
        elif t < 5:
            label = 'passed_during_obstacle_entry_motion'
        elif t <= 13:
            label = 'passed_during_commanded_blocking_window'
        elif t < 15:
            label = 'passed_during_obstacle_retreat'
        elif first_departure is not None and first_departure >= 13:
            label = 'waited_then_passed_after_obstacle_retreat'
        else:
            label = 'passed_after_obstacle_retreat'
    return dict(classification=label, crossing=crossing,
                interpretation='Recorded passage timing and route deviation only; center distance is not clearance, and passage is not proof of causal replanning.')


def summarize(directory):
    directory = Path(directory).resolve()
    report = json.loads((directory/'report.json').read_text())
    raw_rows = json.loads((directory/'trajectory.json').read_text())
    lidar = json.loads((directory/'lidar_frames.json').read_text())
    if not isinstance(report, dict) or not isinstance(raw_rows, list) or not isinstance(lidar, list):
        raise ValueError('expected report object and trajectory/lidar frame arrays')
    rows = [row for row in raw_rows if isinstance(row, dict) and number(row.get('t')) and vector(row.get('position'))]
    if any(b['t'] <= a['t'] for a, b in zip(rows[:-1], rows[1:])):
        raise ValueError('trajectory timestamps must be strictly increasing')
    origin = vector(report.get('initial_position')) or (rows[0]['position'] if rows else None)
    goal = vector(report.get('goal_position'))
    first_departure = next((row['t'] for row in rows if origin and norm(difference(row['position'], origin)) > .2), None)
    planner = report.get('planner') or {}
    plans = planner.get('plans', [])
    checkpoint = planner.get('checkpoint') or {}
    models = [event.get('model', {}) for event in plans]
    planning = stats([event.get('planning_seconds') for event in plans])
    ages = stats([event.get('observation_age_sim_seconds') for event in plans])
    periods = stats([event['activation_time']-event['scan_time'] for event in plans
                     if number(event.get('activation_time')) and number(event.get('scan_time'))])
    collision = report.get('collision') or {}
    obstacle = report.get('obstacle') or {}
    anchor = vector(obstacle.get('anchor_world'))
    axis = {'X': 0, 'Y': 1, 'Z': 2}.get(obstacle.get('motion_axis'))
    offsets = [row['obstacle']['actual_position'][axis]-anchor[axis] for row in rows
               if axis is not None and anchor and vector(row.get('obstacle', {}).get('actual_position'))]
    hit_counts = [entry['hits'] for entry in lidar if number(entry.get('hits'))]
    model_used = planner.get('model_used')
    if model_used is None:
        model_used = any(number(model.get('model_evaluations')) and model['model_evaluations'] > 0 for model in models)
    result = dict(
        directory=str(directory), mode=report.get('mode'), seed=report.get('seed'),
        passed=report.get('passed'), completed=report.get('completed'), error=report.get('error'),
        steps=report.get('steps'), simulation_seconds=report.get('simulation_seconds'),
        loop_wall_seconds=report.get('loop_wall_seconds'),
        goal=dict(reached=report.get('goal_reached'), first_goal_time_s=report.get('first_goal_time'),
                  final_distance_m=report.get('final_goal_distance_m'), final_speed_mps=report.get('final_speed_mps')),
        collision=dict(detected=collision.get('collision_detected'), truth_complete=collision.get('complete'),
                       contact_steps=collision.get('contact_steps'), max_penetration_m=collision.get('max_penetration_m'),
                       first=collision.get('first_collision')),
        planner=dict(failed=planner.get('failed'), failure_reason=planner.get('failure_reason'),
                     model_used=model_used, fixture_stub=bool(planner.get('test_hold_stub')),
                     completed_events=len(plans), statuses=dict(Counter(event.get('status', 'unspecified') for event in plans)),
                     guard_reasons=dict(Counter(event.get('reason', 'unspecified') for event in planner.get('guard_events', []))),
                     planning_wall_seconds=planning, model_inference_wall_seconds=stats([model.get('inference_seconds') for model in models]),
                     cumulative_completed_planning_period_sim_seconds=periods['total'],
                     activation_scan_age_sim_seconds=ages,
                     model_evaluations=sum(model['model_evaluations'] for model in models if number(model.get('model_evaluations'))),
                     algorithms=dict(Counter(model['algorithm'] for model in models if model.get('algorithm'))),
                     checkpoint_sha256=checkpoint.get('checkpoint_sha256'),
                     scope='Statistics cover recorded completed planning events; in-flight work without an event is not included.'),
        lidar=dict(frames=len(lidar), total_returns=sum(hit_counts), minimum_returns=min(hit_counts) if hit_counts else None,
                   maximum_returns=max(hit_counts) if hit_counts else None),
        vehicle=dict(first_departure_over_0_2m_s=first_departure,
                     recorded_rows=len(rows), invalid_rows=len(raw_rows)-len(rows)),
    )
    if obstacle or report.get('dynamic_obstacle'):
        result['obstacle_interaction'] = dict(
            motion_axis=obstacle.get('motion_axis'), actual_offset_range_m=[min(offsets), max(offsets)] if offsets else None,
            reported_max_tracking_error_m=obstacle.get('max_position_tracking_error_m'),
            blocking_window_5_to_13_s=window(rows, origin, 5., 13.),
            entry_motion_3_to_5_s=window(rows, origin, 3., 5.),
            retreat_motion_13_to_15_s=window(rows, origin, 13., 15.),
            overall_minimum_actual_center_distance_m=window(rows, origin, -math.inf, math.inf)['minimum_actual_center_distance_m'],
            passage=passage(rows, origin, goal, obstacle, first_departure),
            distance_semantics='Actual logged vehicle-origin to obstacle-center Euclidean distance, not surface/body clearance. Blocking window is commanded timing; inspect actual offset and tracking error.')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    args = parser.parse_args(argv)
    try:
        summaries = [summarize(directory) for directory in args.directories]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(summaries[0] if len(summaries) == 1 else summaries, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
