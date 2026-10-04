import json
from pathlib import Path

import numpy as np
import pytest

from hnuter_dit.flight_video import FrameLedger
from hnuter_dit.geometry_evaluation import DEFAULT_PROTOCOL
from hnuter_dit.temporal_evaluation import file_sha256
from tools.report_geometry_pilot import bind_video, compact_industrial, relocate_path, render_figure


def protocol_fields():
    return dict(seed=7,requested_duration=30,goal_position=[-25,-33,2.2],physics_dt=.001)


def test_industrial_guard_and_different_checkpoint_cannot_be_model_success(tmp_path):
    paths = []
    for profile in ('constant_incoming', 'constant_outgoing'):
        row = dict(passed=True, completed=True, mode='temporal', goal_reached=True,
                   **protocol_fields(),
                   dynamic_profile=profile, dynamic_obstacle=True, steps=8000,
                   collision=dict(samples=8000, complete=True, collision_detected=False),
                   planner=dict(failed=False, checkpoint={'checkpoint_sha256':'selected'},
                     motion_limits=dict(max_speed_mps=.5,max_acceleration_mps2=.6),
                     accepted_plans=1, guard_rejections=0, guard_interventions=1,
                     temporal_checkpoint_used=True, time_in_state_seconds={'model_reference':4.}))
        path = tmp_path/(profile+'.json')
        path.write_text(json.dumps(row))
        paths.append(path)
    result = compact_industrial(paths, 'selected')
    assert result['system_passed'] and not result['unassisted_model_passed']
    mismatched = compact_industrial(paths, 'different')
    assert not mismatched['system_passed'] and not mismatched['unassisted_model_passed']
    assert compact_industrial([], 'selected')['status'] == 'pending'
    modified = json.loads(paths[0].read_text())
    modified['seed'] = 8
    paths[0].write_text(json.dumps(modified))
    wrong_protocol = compact_industrial(paths, 'selected')
    assert not wrong_protocol['system_passed'] and not wrong_protocol['runs'][0]['protocol_compliant']


def test_video_requires_selected_weights_and_exact_manifest_report_chain_but_allows_failure(tmp_path):
    ledger = FrameLedger(tmp_path/'video', width=8, height=6, fps=10)
    ledger.save(np.zeros((6,8,3),np.uint8),physics_step=0,simulation_time=0.,
        state=dict(position=np.array([-25,-35,2.2]),velocity=np.zeros(3),quaternion=np.array([1,0,0,0])),
        planner_status='planner_failed',goal_distance_m=2.)
    video_info = ledger.close()
    report = dict(passed=False,completed=False,mode='temporal',steps=0,dynamic_obstacle=True,
        dynamic_profile='constant_outgoing',video=video_info,**protocol_fields(),
        planner=dict(checkpoint=dict(checkpoint_sha256='selected',input_representation='point'),
                     motion_limits=dict(max_speed_mps=.5,max_acceleration_mps2=.6)))
    report_path = tmp_path/'report.json'
    report_path.write_text(json.dumps(report))
    video_path = tmp_path/'flight.mp4'
    video_path.write_bytes(b'synthetic container fixture, not real flight video')
    manifest_path = tmp_path/'video/manifest.json'
    evidence = dict(schema='hnuter_actual_flight_mp4_v1',pose_replay=False,
        sha256=file_sha256(video_path),manifest=str(manifest_path),manifest_sha256=file_sha256(manifest_path),
        report=str(report_path),report_sha256=file_sha256(report_path),checkpoint_sha256='selected',
        input_representation='point',fps=10,frame_count=1,encoded_duration_seconds=.1,
        first_simulation_time=0.,last_simulation_time=0.,width=8,height=6)
    video_path.with_suffix('.json').write_text(json.dumps(evidence))
    industrial = compact_industrial([report_path],'selected')
    protocol = json.loads(DEFAULT_PROTOCOL.read_text())
    bound = bind_video(video_path,industrial,'selected','point',protocol)
    assert bound['bound_to_selected_run'] and not bound['flight_passed']
    with pytest.raises(ValueError,match='weights'):
        bind_video(video_path,industrial,'other','point',protocol)
    with pytest.raises(ValueError,match='exactly one'):
        bind_video(video_path,{'runs':[]},'selected','point',protocol)
    old_root = tmp_path/'removed_worktree'
    evidence.update(manifest=str(old_root/'video/manifest.json'),report=str(old_root/'report.json'))
    sidecar = video_path.with_suffix('.json')
    sidecar.write_text(json.dumps(evidence))
    source_digest = file_sha256(sidecar)
    relocated_industrial = compact_industrial([old_root/'report.json'],'selected',
        relocate_from=old_root,relocate_root=tmp_path)
    relocated = bind_video(old_root/'flight.mp4',relocated_industrial,'selected','point',protocol,
        relocate_from=old_root,relocate_root=tmp_path)
    assert relocated['manifest_path'] == str(manifest_path)
    assert relocated['source_manifest_path'] == str(old_root/'video/manifest.json')
    assert file_sha256(sidecar) == source_digest
    video_path.write_bytes(b'tampered video')
    with pytest.raises(ValueError,match='MP4 digest'):
        bind_video(video_path,industrial,'selected','point',protocol)


def test_figure_uses_fixed_first_pair_and_renders_standalone_png(tmp_path, monkeypatch):
    data = dict(group_id=np.array([3,3]), pair_variant=np.array([0,1]),
                future_times=np.arange(65)*.125, positions=np.zeros((2,65,3)))
    loaded_paths = []
    def load(path):
        loaded_paths.append(path)
        return data
    monkeypatch.setattr('tools.report_geometry_pilot.load_dataset', load)
    old_root = tmp_path/'removed_worktree'
    report = tmp_path/'report.json'
    report.write_text(json.dumps({'dataset_path':str(old_root/'frozen_data')}))
    original_hash = file_sha256(report)
    np.savez_compressed(tmp_path/'predictions.npz', indices=[0,1],
                        **{name:np.zeros((2,65,3)) for name in ('box','point','voxel')})
    compact = dict(variants={name:dict(summary={
        'group_balanced_feasible_safe_progress_rate':.5,'sampling_seconds_p95':.01})
        for name in ('box','point','voxel','point_frozen_history','voxel_frozen_history')})
    output = tmp_path/'fixture.png'
    render_figure(compact, old_root/'report.json', output, relocate_from=old_root, relocate_root=tmp_path)
    assert output.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert output.stat().st_size > 1000
    assert loaded_paths == [tmp_path/'frozen_data']
    assert file_sha256(report) == original_hash


def test_relocation_uses_path_boundaries_not_text_prefixes(tmp_path):
    old, new = tmp_path/'old-repo', tmp_path/'new-repo'
    assert relocate_path(old/'results/x.json', old, new) == new/'results/x.json'
    assert relocate_path(old, old, new) == new
    similar = tmp_path/'old-repo-other/results/x.json'
    assert relocate_path(similar, old, new) == similar
    unrelated = tmp_path/'outside/x.json'
    assert relocate_path(unrelated, old, new) == unrelated
    assert relocate_path(old/'results/x.json') == old/'results/x.json'
