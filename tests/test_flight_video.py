"""Simulation-clock capture evidence and export validation without Isaac/GPU."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from hnuter_dit.flight_video import FrameLedger

spec = importlib.util.spec_from_file_location('export_flight_video', Path(__file__).parents[1]/'tools/export_flight_video.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def state():
    return dict(position=np.array([-25., -35., 2.2]), velocity=np.zeros(3), quaternion=np.array([1., 0., 0., 0.]))


def save(ledger, step=0, time=0.):
    ledger.save(np.full((6, 8, 4), 127, dtype=np.uint8), physics_step=step,
        simulation_time=time, state=state(), planner_status='model_wait', goal_distance_m=2.)


def test_video_capture_keeps_measured_state_and_exact_physics_timestamps(tmp_path):
    ledger = FrameLedger(tmp_path/'video', width=8, height=6, fps=20)
    save(ledger)
    assert not ledger.due(49) and ledger.due(50)
    save(ledger, 50, .05)
    description = ledger.close()
    assert description['frame_count'] == 2 and description['pose_replay'] is False
    assert description['last_simulation_time'] == .05
    manifest = exporter.validate_manifest(tmp_path/'video/manifest.json')
    assert manifest['frames'][1]['position'] == [-25., -35., 2.2]
    assert manifest['frames'][1]['planner_status'] == 'model_wait'
    assert len(manifest['frames'][1]['sha256']) == 64


def test_capture_rejects_clock_drift_missing_frames_and_nonfinite_state(tmp_path):
    ledger = FrameLedger(tmp_path/'video', width=8, height=6)
    with pytest.raises(ValueError, match='physics clock'):
        save(ledger, 0, .001)
    with pytest.raises(ValueError, match='physics clock'):
        save(ledger, 100, .1)
    save(ledger)
    with pytest.raises(ValueError, match='physics clock'):
        save(ledger, 200, .2)
    with pytest.raises(ValueError, match='uint8'):
        ledger.save(np.zeros((6, 8, 3)), physics_step=100, simulation_time=.1,
            state=state(), planner_status='wait', goal_distance_m=2.)
    broken = state()
    broken['position'][0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        ledger.save(np.zeros((6, 8, 3), np.uint8), physics_step=100, simulation_time=.1,
            state=broken, planner_status='wait', goal_distance_m=2.)
    ledger.close()
    with pytest.raises(ValueError, match='empty'):
        FrameLedger(tmp_path/'video', width=8, height=6)


def test_export_rejects_tampered_or_outside_frames(tmp_path):
    ledger = FrameLedger(tmp_path/'video', width=8, height=6)
    save(ledger)
    ledger.close()
    path = tmp_path/'video/manifest.json'
    data = json.loads(path.read_text())
    data['frames'][0]['sha256'] = '0'*64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='digest mismatch'):
        exporter.validate_manifest(path)
    data['frames'][0]['filename'] = '../other.png'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='inside'):
        exporter.validate_manifest(path)


def test_export_binds_exact_manifest_to_flight_model_and_allows_failed_runs(tmp_path):
    ledger = FrameLedger(tmp_path/'video', width=8, height=6)
    save(ledger)
    evidence = ledger.close()
    manifest_path = tmp_path/'video/manifest.json'
    manifest = exporter.validate_manifest(manifest_path)
    report_path = tmp_path/'report.json'
    report = dict(passed=False, steps=0, physics_dt=.001, video=evidence,
                  planner=dict(checkpoint=dict(input_representation='voxel', checkpoint_sha256='test')))
    report_path.write_text(json.dumps(report))
    bound = exporter.validate_flight_identity(manifest_path, manifest, report_path)
    assert bound['passed'] is False and bound['planner']['checkpoint']['input_representation'] == 'voxel'
    report['video']['manifest_sha256'] = 'another flight'
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='exact video manifest'):
        exporter.validate_flight_identity(manifest_path, manifest, report_path)
    report['video']['manifest_sha256'] = exporter.digest(manifest_path)
    report['video']['fps'] = 20
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='fps'):
        exporter.validate_flight_identity(manifest_path, manifest, report_path)
    report['video']['fps'] = 10
    report['steps'] = 100
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='physics steps'):
        exporter.validate_flight_identity(manifest_path, manifest, report_path)
