from hnuter_dit.thermal_run import can_resume, should_pause


def test_temperature_hysteresis_and_missing_sensors():
    pause = dict(use_gpu=True, cpu_pause=90., gpu_pause=83.)
    resume = dict(use_gpu=True, cpu_resume=80., gpu_resume=75.)
    assert not should_pause(89., 82., **pause)
    assert should_pause(90., 70., **pause)
    assert should_pause(70., 83., **pause)
    assert should_pause(None, 70., **pause)
    assert should_pause(70., None, **pause)
    assert can_resume(80., 75., **resume)
    assert not can_resume(81., 75., **resume)
    assert not can_resume(80., 76., **resume)
    assert not can_resume(None, 70., **resume)
    assert can_resume(80., None, use_gpu=False, cpu_resume=80., gpu_resume=75.)
