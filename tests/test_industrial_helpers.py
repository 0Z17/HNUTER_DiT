"""Physical evaluation semantics without launching Isaac or initializing CUDA."""
from types import SimpleNamespace

import numpy as np
import pytest

from hnuter_dit.industrial_obstacle import SlidingObstacle
from hnuter_dit.industrial_runner import ContactMonitor, GoalProgress, evaluation_passed


class RuntimeArray:
    """The runtime accessor used by Newton/Warp arrays."""
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value


def gpu_contacts(geom, distances, worlds=None, count=None, capacity=None):
    # Permutations differ across worlds to catch fixed geom IDs and world-0-only
    # lookup. A robot body with a non-robot shape label exercises body ownership.
    model = SimpleNamespace(shape_label=["/World/HNUTER/drone/mesh", "/World/Wall",
                                        "/ImportedRotorMesh", "/World/OtherObject"],
                            body_label=["/World/HNUTER/drone", "/World/HNUTER/rotor", "/World/OtherObject"],
                            shape_body=RuntimeArray([0, -1, 1, 2]))
    contact = SimpleNamespace(geom=RuntimeArray(np.asarray(geom).reshape(-1, 2)),
                              dist=RuntimeArray(distances),
                              worldid=RuntimeArray(worlds if worlds is not None else np.zeros(len(distances), dtype=int)))
    data = SimpleNamespace(nacon=RuntimeArray([len(distances) if count is None else count]),
                           naconmax=len(distances) if capacity is None else capacity, contact=contact)
    solver = SimpleNamespace(mjc_geom_to_newton_shape=RuntimeArray([[1, 2, 0, 3], [3, 0, 1, 2]]),
                             mjw_data=data)
    return SimpleNamespace(model=model, solver=solver, initialized=True,
                           cfg=SimpleNamespace(solver_cfg=SimpleNamespace(use_mujoco_cpu=False)))


def state(position=(0, 0, 0), velocity=(0, 0, 0)):
    return {"position": np.asarray(position, dtype=float), "velocity": np.asarray(velocity, dtype=float)}


def test_slide_schedule_is_continuous_and_starts_clear():
    expected = {0: -4., 3: -4., 4: -2., 5: 0., 12: 0., 13: 0., 14: -2., 15: -4., 20: -4.}
    for t, offset in expected.items():
        assert SlidingObstacle.target(t) == pytest.approx(offset)
    t = np.linspace(0, 20, 20001)
    q = np.array([SlidingObstacle.target(value) for value in t])
    assert q.min() >= -4 and q.max() <= 0
    for edge in (3, 5, 13, 15):
        h = 1e-5
        assert abs(SlidingObstacle.target(edge+h)-SlidingObstacle.target(edge-h)) < 1e-8
        assert abs((SlidingObstacle.target(edge+h)-SlidingObstacle.target(edge-h))/(2*h)) < 1e-4


def test_contacts_use_world_mapping_exclude_self_and_ignore_positive_margin():
    stage = gpu_contacts([[2, 0], [2, 1], [0, 3]], [.01, -.3, -.5])
    monitor = ContactMonitor(stage)
    assert not monitor.sample(.001)
    assert monitor.report()["minimum_contact_distance_m"] == pytest.approx(.01)
    stage.solver.mjw_data.contact.dist.value[0] = -.02
    assert monitor.sample(.002)
    first = monitor.report()["first_collision"]
    assert first["simulation_time"] == .002
    assert first["pairs"][0]["shape_b"] == "/World/Wall"
    assert monitor.report()["max_penetration_m"] == .02
    # Receding contact must not erase earlier impact or amplify self collision.
    stage.solver.mjw_data.contact.dist.value[0] = .02
    assert not monitor.sample(.003)
    assert monitor.report()["collision_detected"]
    assert monitor.report()["contact_steps"] == 1
    assert monitor.report()["first_collision"] == first
    assert monitor.report()["complete"]


def test_zero_distance_is_contact_and_world_one_is_not_world_zero():
    stage = gpu_contacts([[1, 0]], [0.], worlds=[1])
    monitor = ContactMonitor(stage)
    assert monitor.sample(.001)
    report = monitor.report()
    assert report["penetration_steps"] == 0
    assert report["first_collision"]["pairs"][0]["shape_a"] == "/World/HNUTER/drone/mesh"
    assert report["first_collision"]["pairs"][0]["shape_b"] == "/World/OtherObject"


def test_contact_overflow_and_unmapped_geometry_cannot_claim_complete():
    monitor = ContactMonitor(gpu_contacts([[2, 0]], [-.1], count=3, capacity=1))
    assert monitor.sample(.001)
    assert monitor.report()["overflow_steps"] == 1
    assert not monitor.report()["complete"]
    monitor = ContactMonitor(gpu_contacts([[-1, 0], [2, 0]], [-.1, -.2], worlds=[0, 3]))
    assert not monitor.sample(.001)
    assert monitor.report()["unmapped_contacts"] == 2
    assert not monitor.report()["complete"]


def test_empty_contacts_and_rebuilt_physics():
    stage = gpu_contacts([], [])
    monitor = ContactMonitor(stage)
    assert not monitor.sample(.001)
    assert monitor.report()["complete"]
    stage.initialized = False
    with pytest.raises(RuntimeError, match="reset/rebuilt"):
        monitor.sample(.002)


def test_cpu_contact_path_does_not_require_gpu_data():
    stage = gpu_contacts([], [])
    stage.cfg.solver_cfg.use_mujoco_cpu = True
    stage.solver.mj_data = SimpleNamespace(ncon=1, contact=[SimpleNamespace(geom1=2, geom2=0, dist=-.005)])
    del stage.solver.mjw_data
    monitor = ContactMonitor(stage)
    assert monitor.sample(.001)
    assert monitor.report()["backend"] == "native MuJoCo contacts"


def test_goal_requires_half_second_slow_hold_and_rejects_fast_pass():
    progress = GoalProgress([0, 0, 0])
    for t in np.arange(501) * .001:
        assert not progress.observe(state(velocity=(.11, 0, 0)), float(t))
    # Exact distance and speed boundaries qualify, but 499 ms do not.
    for step in range(501, 1001):
        assert not progress.observe(state((.35, 0, 0), (.1, 0, 0)), step * .001)
    assert progress.observe(state((.35, 0, 0), (.1, 0, 0)), 1.001)
    assert progress.report()["first_goal_time"] == 1.001
    assert progress.report()["goal_settled_seconds"] == pytest.approx(.5)


def test_goal_hold_restarts_on_speed_or_distance_violation():
    progress = GoalProgress([0, 0, 0])
    for step in range(500):
        assert not progress.observe(state(), step * .001)
    assert not progress.observe(state((.351, 0, 0)), .5)
    for step in range(501, 1001):
        assert not progress.observe(state(), step * .001)
    assert not progress.observe(state(velocity=(0, .101, 0)), 1.001)
    for step in range(1002, 1502):
        assert not progress.observe(state(), step * .001)
    assert progress.observe(state(), 1.502)
    assert progress.report()["first_goal_time"] == 1.502


def test_collision_latches_and_invalidates_previously_settled_goal():
    progress = GoalProgress([0, 0, 0])
    for step in range(501):
        progress.observe(state(), step * .001)
    assert progress.report()["goal_reached"]
    assert not progress.observe(state(), .501, collided=True)
    for step in range(502, 1200):
        assert not progress.observe(state(), step * .001)
    assert progress.report()["goal_settled_seconds"] == 0
    assert progress.report()["first_goal_time"] == .5


def test_goal_rejects_nonfinite_state_and_out_of_order_time():
    progress = GoalProgress([0, 0, 0])
    progress.observe(state(), 0.)
    with pytest.raises(ValueError, match="increasing"):
        progress.observe(state(), 0.)
    with pytest.raises(ValueError, match="finite"):
        progress.observe(state(velocity=(np.nan, 0, 0)), .001)


def test_settled_flight_still_requires_healthy_planner_and_complete_contact_truth():
    report = {"completed": True, "goal_reached": True, "planner": {"failed": False},
              "collision": {"complete": True, "collision_detected": False}}
    assert evaluation_passed(report)
    for planner in ({"failed": True, "failure_reason": "worker exited"}, {}, {"failed": None}):
        assert not evaluation_passed(dict(report, planner=planner))
    for collision in ({"complete": False, "collision_detected": False},
                      {"complete": True, "collision_detected": True}):
        assert not evaluation_passed(dict(report, collision=collision))
    assert not evaluation_passed(dict(report, completed=False))
    assert not evaluation_passed(dict(report, goal_reached=False))
