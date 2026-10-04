#!/usr/bin/env python3
"""Produce a compact geometry-encoder comparison JSON and PNG from real runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hnuter_dit.geometry_evaluation import DEFAULT_PROTOCOL, REPRESENTATIONS, SCHEMA, write_json
from hnuter_dit.temporal_data import load_dataset
from hnuter_dit.temporal_evaluation import file_sha256, industrial_evidence_valid


def relocate_path(path, old_root=None, new_root=None):
    """Map only real path descendants; keep raw evidence documents immutable."""
    value = Path(path).expanduser().resolve()
    if old_root is None:
        return value
    try:
        relative = value.relative_to(Path(old_root).expanduser().resolve())
    except ValueError:
        return value
    destination = Path(new_root).resolve() if new_root is not None else Path(__file__).resolve().parents[1]
    return destination/relative


def industrial_protocol_checks(report, protocol):
    expected = protocol['industrial']
    limits = report.get('planner', {}).get('motion_limits', {})
    goal = np.asarray(report.get('goal_position', []))
    return dict(seed=report.get('seed') == expected['seed'],
        requested_duration=report.get('requested_duration') == expected['duration_s'],
        goal=goal.shape == (3,) and bool(np.allclose(goal, expected['goal_world'], atol=1e-8, rtol=0)),
        profile=report.get('dynamic_profile') in expected['profiles'],
        physics_dt=report.get('physics_dt') == .001,
        speed_limit=limits.get('max_speed_mps') == protocol['evaluation']['max_speed_mps'],
        acceleration_limit=limits.get('max_acceleration_mps2') == protocol['evaluation']['max_acceleration_mps2'])


def compact_industrial(paths, checkpoint_sha, protocol=None, *, relocate_from=None, relocate_root=None):
    protocol = protocol or json.loads(DEFAULT_PROTOCOL.read_text())
    runs = []
    for path in paths:
        source_path = str(Path(path).resolve())
        path = relocate_path(path, relocate_from, relocate_root)
        report = json.loads(path.read_text())
        planner = report.get('planner', {})
        plans = planner.get('plans', [])
        statuses = sorted({p.get('status', 'unknown') for p in plans})
        value = {key: report.get(key) for key in (
            'passed', 'completed', 'mode', 'goal_reached', 'dynamic_profile', 'dynamic_obstacle',
            'steps', 'simulation_seconds', 'loop_wall_seconds', 'first_goal_time',
            'final_goal_distance_m', 'final_speed_mps', 'collision', 'seed', 'requested_duration',
            'physics_dt', 'goal_position', 'error', 'stop_reason', 'video')}
        checks = industrial_protocol_checks(report, protocol)
        if isinstance(value.get('video'), dict):
            value['video'] = dict(value['video'])
            for key in ('directory', 'manifest'):
                if value['video'].get(key):
                    value['video']['source_'+key] = value['video'][key]
                    value['video'][key] = str(relocate_path(value['video'][key], relocate_from, relocate_root))
        value.update(report_path=str(path), source_report_path=source_path, report_sha256=file_sha256(path),
            checkpoint_matches_selection=planner.get('checkpoint', {}).get('checkpoint_sha256') == checkpoint_sha,
            protocol_checks=checks, protocol_compliant=all(checks.values()),
            plan_status_counts={status: sum(p.get('status', 'unknown') == status for p in plans) for status in statuses},
            planner={key: planner.get(key) for key in ('failed', 'failure_reason', 'accepted_plans',
                'guard_rejections', 'guard_interventions', 'horizon_completions', 'time_in_state_seconds',
                'planning_seconds_mean', 'planning_seconds_max', 'temporal_checkpoint_used', 'checkpoint', 'motion_limits')})
        runs.append(value)
    profiles = {r['dynamic_profile'] for r in runs}
    result = dict(status='complete' if {'constant_incoming', 'constant_outgoing'} <= profiles else 'pending',
                  system_passed=bool({'constant_incoming', 'constant_outgoing'} <= profiles and
                                     all(r['passed'] is True and r['checkpoint_matches_selection'] and r['protocol_compliant'] for r in runs)), runs=runs)
    result['unassisted_model_passed'] = bool(result['system_passed'] and industrial_evidence_valid(result, checkpoint_sha))
    return result


def bind_video(video_path, industrial, checkpoint_sha, representation, protocol, *, relocate_from=None, relocate_root=None):
    """Follow MP4 -> sidecar -> capture manifest -> run -> selected weights."""
    from tools.export_flight_video import validate_flight_identity, validate_manifest
    video_path = relocate_path(video_path, relocate_from, relocate_root)
    sidecar_path = video_path.with_suffix('.json')
    if not video_path.is_file() or not sidecar_path.is_file():
        raise ValueError('video requires its exporter-generated evidence JSON sidecar')
    evidence = json.loads(sidecar_path.read_text())
    if evidence.get('schema') != 'hnuter_actual_flight_mp4_v1' or evidence.get('pose_replay') is not False:
        raise ValueError('video sidecar is not actual controlled-flight evidence')
    if evidence.get('sha256') != file_sha256(video_path):
        raise ValueError('MP4 digest differs from export evidence')
    manifest_path = relocate_path(evidence['manifest'], relocate_from, relocate_root)
    flight_path = relocate_path(evidence['report'], relocate_from, relocate_root)
    if (evidence.get('manifest_sha256') != file_sha256(manifest_path)
            or evidence.get('report_sha256') != file_sha256(flight_path)):
        raise ValueError('video manifest or flight report changed after export')
    runs = [run for run in industrial['runs'] if run['report_sha256'] == evidence['report_sha256']]
    if len(runs) != 1:
        raise ValueError('video must bind exactly one supplied industrial report')
    manifest = validate_manifest(manifest_path)  # Includes every recorded PNG hash and physical timestamp.
    flight = validate_flight_identity(manifest_path, manifest, flight_path)
    model = flight.get('planner', {}).get('checkpoint', {})
    if (evidence.get('checkpoint_sha256') != checkpoint_sha or model.get('checkpoint_sha256') != checkpoint_sha
            or evidence.get('input_representation') != representation or model.get('input_representation') != representation):
        raise ValueError('video weights/representation differ from validation-selected encoder')
    required = protocol['industrial']
    if (not all(industrial_protocol_checks(flight, protocol).values())
            or flight.get('dynamic_profile') != required['video_profile']
            or evidence.get('fps') != required['video_fps'] or manifest.get('fps') != required['video_fps']
            or manifest.get('camera') != required['camera']):
        raise ValueError('video flight/camera/fps differs from preregistered demonstration')
    for key in ('frame_count', 'first_simulation_time', 'last_simulation_time', 'width', 'height'):
        if evidence.get(key) != manifest.get(key):
            raise ValueError('video sidecar and manifest disagree on '+key)
    if not np.isclose(evidence.get('encoded_duration_seconds', -1), manifest['frame_count']/manifest['fps']):
        raise ValueError('video encoded duration differs from the recorded frame clock')
    return dict(status='available', bound_to_selected_run=True, path=str(video_path), sha256=evidence['sha256'],
                sidecar_path=str(sidecar_path), sidecar_sha256=file_sha256(sidecar_path),
                manifest_path=str(manifest_path), manifest_sha256=evidence['manifest_sha256'],
                source_manifest_path=evidence['manifest'], source_flight_report=evidence['report'],
                flight_report=str(flight_path), flight_report_sha256=evidence['report_sha256'],
                checkpoint_sha256=checkpoint_sha, input_representation=representation,
                frame_count=manifest['frame_count'], fps=manifest['fps'], camera=manifest['camera'],
                last_simulation_time=manifest['last_simulation_time'], flight_passed=flight.get('passed') is True,
                time_base='simulation time, 10 fps; not a wall-clock realtime measurement')


def build_compact(evaluation_path, industrial_paths=(), video_path=None, *, relocate_from=None, relocate_root=None):
    source_evaluation_path = str(Path(evaluation_path).resolve())
    evaluation_path = relocate_path(evaluation_path, relocate_from, relocate_root)
    report = json.loads(evaluation_path.read_text())
    if report.get('schema') != SCHEMA or report.get('phase') != 'test' or report.get('completed') is not True:
        raise ValueError('a completed held-out geometry test report is required')
    selection_path = relocate_path(report['selection_report'], relocate_from, relocate_root)
    if file_sha256(selection_path) != report['selection_sha256']:
        raise ValueError('validation selection report changed after test')
    selection = json.loads(selection_path.read_text())
    if selection['selected_new_encoder'] != report['selected_new_encoder']:
        raise ValueError('selected new encoder changed after validation')
    protocol = report['preregistered_protocol']
    protocol_path = relocate_path(protocol['path'], relocate_from, relocate_root)
    if file_sha256(protocol_path) != protocol['sha256']:
        raise ValueError('preregistered protocol changed')
    selected = report['selected_new_encoder']['representation']
    weights = report['variants'][selected]['checkpoint_sha256']
    variants = {}
    for name, variant in report['variants'].items():
        item = dict(checkpoint_sha256=variant['checkpoint_sha256'], summary=variant['summary'])
        if name in REPRESENTATIONS:
            runtime = variant['runtime']
            item.update(checkpoint=str(relocate_path(variant['checkpoint'], relocate_from, relocate_root)),
                source_checkpoint_path=variant['checkpoint'],
                selected_training_step=runtime['training_step'], training_budget=runtime['training_budget'],
                parameter_count=runtime['parameter_count'], encoder_parameter_count=runtime['encoder_parameter_count'],
                shared_architecture=runtime['architecture'], geometry_config=runtime['geometry_config'],
                geometry_config_sha256=variant['geometry_config_sha256'],
                preprocessing_split_batch_seconds=variant['preprocessing_split_batch_seconds'],
                preprocessing_timing_scope=variant['preprocessing_timing_scope'],
                validation_summary=selection['variants'][name]['summary'])
        variants[name] = item
    native = {name: {k: v for k, v in result.items() if k != 'rows'} for name, result in report['native'].items()}
    relocation = dict(relocate_from=relocate_from, relocate_root=relocate_root)
    industrial = compact_industrial(industrial_paths, weights, protocol['content'], **relocation)
    video = bind_video(video_path, industrial, weights, selected, protocol['content'], **relocation) if video_path else dict(status='pending')
    return dict(schema='geometry_encoder_pilot_summary_v1', date='2026-10-04',
        evaluation_report=dict(path=str(evaluation_path), source_path=source_evaluation_path, sha256=file_sha256(evaluation_path)),
        selection_report=dict(path=str(selection_path), source_path=report['selection_report'], sha256=report['selection_sha256']),
        preregistered_protocol=dict(protocol, path=str(protocol_path), source_path=protocol['path']),
        dataset=report['dataset_freeze'],
        relocation=dict(from_root=str(Path(relocate_from).resolve()) if relocate_from else None,
                        to_root=str(Path(relocate_root).resolve() if relocate_root else Path(__file__).resolve().parents[1]),
                        source_json_modified=False),
        selected_new_encoder=report['selected_new_encoder'], variants=variants,
        paired_comparisons=report['paired_comparisons'], history_ablations=report['history_ablations'],
        native_execution=native, industrial=industrial, video=video,
        automatic_expansion_authorized=False,
        limitations=[
            'This is a small encoder comparison, not permission to enlarge data or training.',
            'Validation chose point or voxel before test; being selected does not mean outperforming box.',
            'One fixed-seed sample per condition; no future-truth path selection or retiming.',
            'Timed 42-sphere geometry is a proxy; original actuator-driven native contacts are reported separately.',
            'Eight-second physical prefixes may legitimately wait and need not complete the full mission.',
            'Industrial guard stops/rejections are system behavior, not learned model success.',
            'Voxels are per-frame measured occupancy/free/unknown, not nvblox TSDF fusion.',
            'No contact-task, moving-start generalization or wall-clock realtime claim.',
        ])


def render_figure(compact, evaluation_path, output, *, relocate_from=None, relocate_root=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    evaluation_path = relocate_path(evaluation_path, relocate_from, relocate_root)
    evaluation = json.loads(evaluation_path.read_text())
    data = load_dataset(relocate_path(evaluation['dataset_path'], relocate_from, relocate_root))
    with np.load(evaluation_path.parent/'predictions.npz', allow_pickle=False) as result:
        indices = result['indices'].copy()
        predictions = {name: result[name].copy() for name in REPRESENTATIONS}
    if len(indices) < 2 or data['group_id'][indices[0]] != data['group_id'][indices[1]]:
        raise ValueError('figure requires the fixed first complete held-out pair')
    colors = dict(box='#6C7785', point='#2471A3', voxel='#D17A22')
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    labels = list(compact['variants'])
    rates = [100*compact['variants'][n]['summary']['group_balanced_feasible_safe_progress_rate'] for n in labels]
    axes[0, 0].bar([n.replace('_', '\n') for n in labels], rates,
                   color=[colors[n.split('_')[0]] for n in labels], alpha=.85)
    axes[0, 0].set(title='Held-out fixed-candidate results', ylabel='Feasible + safe + adequate progress (%)', ylim=(0, 105))
    for index, name in enumerate(REPRESENTATIONS):
        summary = compact['variants'][name]['summary']
        axes[0, 1].bar(index, summary['sampling_seconds_p95']*1000, color=colors[name])
    axes[0, 1].set(xticks=range(3), xticklabels=REPRESENTATIONS, ylabel='Sampling P95 (ms)',
                   title='Model + tensor transfer; excludes ray preprocessing/IPC')
    for local, axis in enumerate(axes[1]):
        index = indices[local]
        axis.plot(data['future_times'], data['positions'][index, :, 0], color='black', label='Validated teacher reference', linewidth=2.)
        for name in REPRESENTATIONS:
            axis.plot(data['future_times'], predictions[name][local, :, 0], color=colors[name], label=name)
        direction = 'incoming' if data['pair_variant'][index] == 0 else 'outgoing'
        axis.set(title=f'Fixed first test pair: {direction}, group {data["group_id"][index]}',
                 xlabel='Seconds after latest observation', ylabel='Goal-direction reference position (m)')
        axis.grid(alpha=.2)
    axes[1, 0].legend(fontsize=8)
    figure.suptitle('Geometry encoder pilot — same data, training budget and fixed-time trajectory head', fontsize=13)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--industrial', type=Path, action='append', default=[])
    parser.add_argument('--video', type=Path)
    parser.add_argument('--relocate-from', type=Path, help='Map evidence paths inside this former repository root to the current repository; source JSON and hashes remain unchanged')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    compact = build_compact(args.evaluation, args.industrial, args.video, relocate_from=args.relocate_from)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, compact)
    render_figure(compact, args.evaluation, args.output.with_suffix('.png'), relocate_from=args.relocate_from)
    print(json.dumps(dict(report=str(args.output), figure=str(args.output.with_suffix('.png')),
                          selected=compact['selected_new_encoder']['representation'],
                          industrial=compact['industrial']['status'],
                          automatic_expansion_authorized=False)))


if __name__ == '__main__':
    main()
