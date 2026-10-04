"""Inspection acceptance: actual MuJoCo first hits and full-window timing."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hnuter_dit.task_inspection import inspection_measurement, summarize_inspection


def make_world(*, blocked=False, behind=False, yaw=0.):
    mj = pytest.importorskip('mujoco')
    quat = [np.cos(yaw/2), 0., 0., np.sin(yaw/2)]
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.],
                         [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
    position = np.array([0., 0., 2.8])
    target = position+rotation@np.array([-1.76 if behind else 2.24, 0., .025])
    blocker = position+rotation@np.array([1.24, 0., .025])
    numbers = lambda value: ' '.join(map(str, value))
    obstacle = (f'<geom name="blocker" type="sphere" pos="{numbers(blocker)}" size=".1"/>'
                if blocked else '')
    xml = f'''<mujoco><worldbody>
        <geom name="inspection_target" type="sphere" pos="{numbers(target)}" size=".05"/>
        {obstacle}<body name="drone" pos="{numbers(position)}" quat="{numbers(quat)}">
        <freejoint/><geom name="aircraft" type="sphere" group="5" size=".4" mass="1"/>
        </body></worldbody></mujoco>'''
    model = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(model); mj.mj_forward(model, data)
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'drone')
    return SimpleNamespace(mj=mj, model=model, data=data,
        robot_geoms=model.geom_bodyid == body_id,
        state=lambda: dict(position=data.xpos[body_id].copy(),
                           rotation_matrix=data.xmat[body_id].reshape(3, 3).copy()))


def measurement(**changes):
    return dict(dict(in_frustum=True, line_of_sight=True, axis_angle_deg=0., range_m=2.), **changes)


def trace(value):
    return [dict(t=float(t), measurement=value) for t in np.arange(0, 20.001, .01)]


def test_actual_ray_hits_target_and_excludes_aircraft():
    result = inspection_measurement(make_world(), {})
    assert result['first_hit_geom'] == 'inspection_target'
    assert result['effective']
    assert result['range_m'] == pytest.approx(2.)
    assert result['ray_distance_m'] == pytest.approx(1.95)


def test_actual_occluder_is_first_hit_and_rejects_visibility():
    result = inspection_measurement(make_world(blocked=True), {})
    assert result['first_hit_geom'] == 'blocker'
    assert result['ray_distance_m'] == pytest.approx(.9)
    assert not result['visible'] and not result['effective']


def test_line_of_sight_behind_camera_does_not_count_as_visible():
    result = inspection_measurement(make_world(behind=True), {})
    assert result['line_of_sight']
    assert not result['in_frustum'] and not result['effective']


def test_camera_extrinsic_rotates_with_actual_body():
    result = inspection_measurement(make_world(yaw=np.pi/2), {})
    assert result['effective']
    np.testing.assert_allclose(result['camera_position'], [0., .24, 2.825], atol=1e-12)
    np.testing.assert_allclose(result['optical_axis'], [0., 1., 0.], atol=1e-12)


def test_whole_operating_window_has_its_declared_denominator():
    result = summarize_inspection(trace(measurement()), {})
    assert result['passed']
    assert result['effective_fraction'] == pytest.approx(1.)
    assert result['longest_continuous_effective_s'] == pytest.approx(20/3)


def test_short_occlusion_breaks_required_continuous_dwell():
    records = trace(measurement())
    for row in records:
        if 9. <= row['t'] < 9.2:
            row['measurement'] = measurement(line_of_sight=False)
    result = summarize_inspection(records, {})
    assert result['effective_fraction'] == pytest.approx(.97)
    assert result['longest_continuous_effective_s'] == pytest.approx(40/3-9.2)
    assert not result['passed']


@pytest.mark.parametrize('records', [[], [dict(t=7., measurement=measurement()),
                                        dict(t=13., measurement=measurement())]])
def test_missing_samples_never_inflate_dwell(records):
    result = summarize_inspection(records, {})
    assert not result['passed']
    assert result['measured_fraction'] < .004


@pytest.mark.parametrize('changes', [dict(axis_angle_deg=10.01), dict(range_m=2.51),
                                   dict(in_frustum=False)])
def test_independent_observation_gates(changes):
    result = summarize_inspection(trace(measurement(**changes)), {})
    assert not result['passed'] and result['effective_fraction'] == 0.


def test_duplicate_timestamps_fail_closed():
    with pytest.raises(ValueError, match='strictly increasing'):
        summarize_inspection([dict(t=7., measurement=measurement())]*2, {})
