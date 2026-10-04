import numpy as np
import pytest

from hnuter_dit.industrial_obstacle import SlidingObstacle, ConstantVelocityObstacle


def test_constant_profiles_share_anchor_but_have_opposite_observable_histories():
    incoming = ConstantVelocityObstacle([0, 1, 2.2], 0, direction=1)
    outgoing = ConstantVelocityObstacle([0, 1, 2.2], 0, direction=-1)
    assert incoming.position_target(.6) == outgoing.position_target(.6) == -1
    assert incoming.position_target(0) < outgoing.position_target(0)
    assert incoming.position_target(8.6) == 1
    assert outgoing.position_target(8.6) == -3
    assert np.isclose(incoming.actuator_target(.6) + outgoing.actuator_target(.6), -2)
    assert 'motion_times_seconds' not in incoming.report()
    origin, goal = np.array([-25., -35., 2.2]), np.array([-25., -33., 2.2])
    np.testing.assert_allclose(ConstantVelocityObstacle.anchor_for_route(origin, goal, 1), [-25, -33.7, 2.2])
    np.testing.assert_allclose(SlidingObstacle.anchor_for_route(origin, goal, 1), [-25, -34, 2.2])


def test_legacy_schedule_is_preserved_and_constant_invalid_clock_rejected():
    obstacle = SlidingObstacle([0, 0, 0], 1)
    for t in (0, 3, 4, 5, 13, 14, 15, 40):
        assert obstacle.position_target(t) == obstacle.actuator_target(t) == SlidingObstacle.target(t)
    assert obstacle.joint_limits == (-4.1, .1)
    with pytest.raises(ValueError):
        ConstantVelocityObstacle([0, 0, 0], 1, direction=0)
    with pytest.raises(ValueError):
        ConstantVelocityObstacle([0, 0, 0], 1).position_target(float('nan'))
