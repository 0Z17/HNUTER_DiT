"""HNUTER environment-conditioned DiT inference package."""

__all__ = ["ConditionalDiT", "DiffusionSchedule", "DiTTrajectoryService", "ddim_sample"]


def __getattr__(name):
    # Simulation-side IPC clients must not import Torch into Isaac/Kit. Preserve
    # the existing public exports, loading the inference stack only on demand.
    if name == "DiTTrajectoryService":
        from .service import DiTTrajectoryService
        return DiTTrajectoryService
    if name in {"ConditionalDiT", "DiffusionSchedule", "ddim_sample"}:
        from . import model
        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
