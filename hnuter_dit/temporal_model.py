"""Trainable history-conditioned DiT for fixed-time cubic position splines.

This is a new model family. It does not load the static v003 weights and does
not generate attitude. Repeated positions/waits retain their physical times.
"""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.interpolate import BSpline
import torch
from torch import nn
from torch.nn import functional as F

from .geometry_observations import input_spec, GRID_SIZE, POINT_COUNT, ROI_RADIUS

from .model import DiTBlock, timestep_embedding

CHECKPOINT_SCHEMA = "hnuter_temporal_bspline_dit_v1"
CONDITION_SCHEMA = "canonical_goal3_velocity3_body_rotation6d_v1"
HISTORY_SCHEMA = "canonical_observed_box_history4x32_v1"
TRAJECTORY_SCHEMA = "fixed_time_cubic_control16_horizon8_v1"
HISTORY_FRAMES, BOXES_PER_FRAME, BOX_DIM, CONDITION_DIM = 4, 32, 10, 12
CONTROL_POINTS, FIXED_CONTROLS, POSITION_DIM = 16, 2, 3
FREE_CONTROLS = CONTROL_POINTS - FIXED_CONTROLS
HORIZON_SECONDS, FUTURE_DT = 8., .125
FUTURE_TIMES = np.arange(65, dtype=np.float64) * FUTURE_DT
HISTORY_TIMES = np.array([-.6, -.4, -.2, 0.], dtype=np.float64)
SPLINE_DEGREE = 3
KNOT_SPACING = HORIZON_SECONDS / (CONTROL_POINTS - SPLINE_DEGREE)


def spline_knots():
    return np.r_[np.zeros(4), np.arange(1, 13) * KNOT_SPACING,
                 np.full(4, HORIZON_SECONDS)]


def spline_basis(times=None, derivative=0):
    values = FUTURE_TIMES if times is None else np.asarray(times, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > HORIZON_SECONDS):
        raise ValueError("spline times must lie in the fixed [0,8] second horizon")
    if derivative not in (0, 1, 2, 3):
        raise ValueError("spline derivative must be between zero and three")
    return BSpline(spline_knots(), np.eye(CONTROL_POINTS), SPLINE_DEGREE)(values, nu=derivative)


def decode_control_points(control_points, times=None):
    """NumPy analytic decoding; accepts [...,16,3], never stretches time."""
    controls = np.asarray(control_points, dtype=np.float64)
    if controls.shape[-2:] != (CONTROL_POINTS, 3) or not np.isfinite(controls).all():
        raise ValueError("control_points must be finite [...,16,3]")
    values = FUTURE_TIMES if times is None else np.asarray(times, dtype=float)
    return {name: np.einsum("...cd,tc->...td", controls, spline_basis(values, derivative))
            for name, derivative in (("positions", 0), ("velocities", 1), ("accelerations", 2))}


def fit_control_points(positions, initial_velocity):
    """Fit timed teacher positions with exact current position/velocity.

    No arc-length resampling, target endpoint constraint, or time reversal is
    applied. Fit error must be reported separately from model prediction error.
    """
    positions, velocity = np.asarray(positions, dtype=float), np.asarray(initial_velocity, dtype=float)
    if positions.shape[-2:] != (65, 3) or velocity.shape != positions.shape[:-2] + (3,):
        raise ValueError("expected positions[...,65,3] and matching initial_velocity[...,3]")
    if not np.isfinite(positions).all() or not np.isfinite(velocity).all():
        raise ValueError("teacher positions and initial velocity must be finite")
    if np.max(np.abs(positions[..., 0, :])) > 1e-4:
        raise ValueError("canonical teacher positions must start at the origin")
    basis = spline_basis()
    controls = np.zeros(positions.shape[:-2] + (16, 3), dtype=float)
    controls[..., 1, :] = velocity * (KNOT_SPACING / 3.)
    residual = positions - basis[:, 1, None] * controls[..., 1, None, :]
    free = basis[:, FIXED_CONTROLS:]
    fit = np.linalg.solve(free.T @ free + 1e-8 * np.eye(FREE_CONTROLS), free.T)
    controls[..., FIXED_CONTROLS:, :] = np.einsum("ct,...td->...cd", fit, residual)
    return controls.astype(np.float32)


def normalization_from_training(controls, condition, history_tokens, history_mask, train_indices, input_representation="box"):
    """Fit normalization only on explicit training rows and bind their identity."""
    indices = np.asarray(train_indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices):
        raise ValueError("train_indices must be a nonempty set of unique row indices")
    controls, condition = np.asarray(controls), np.asarray(condition)
    tokens, mask = np.asarray(history_tokens), np.asarray(history_mask, dtype=bool)
    free = controls[indices, FIXED_CONTROLS:].astype(np.float64)
    # Geometry uses a fixed metric ROI scale, never a fit across held-out rows.
    active = tokens[indices][mask[indices]][:, :6].astype(np.float64) if input_representation == "box" else np.empty((0,6))
    current = condition[indices].astype(np.float64)
    if not len(active):
        environment_mean, environment_std = np.zeros(6), np.ones(6)
    else:
        environment_mean = active.mean(axis=0)
        environment_std = np.maximum(active.std(axis=0), .05)
    result = dict(control_mean=free.mean(axis=(0, 1)).tolist(),
        control_std=np.maximum(free.std(axis=(0, 1)), .05).tolist(),
        condition_mean=current.mean(axis=0).tolist(),
        condition_std=np.maximum(current.std(axis=0), .05).tolist(),
        environment_mean=environment_mean.tolist(), environment_std=environment_std.tolist(),
        fit_split="train", train_row_count=len(indices),
        train_indices_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest())
    return result


class TemporalDiT(nn.Module):
    """Velocity-prediction diffusion over 14 free three-dimensional controls.

    History observations are a permutation-invariant set within each frame.
    Their physical ages distinguish frames; no partition index/object ID is
    supplied. ``latest`` uses identical trainable architecture but masks every
    earlier frame, providing the controlled single-frame training ablation.
    """
    def __init__(self, dimension=128, depth=4, heads=4, history_depth=2, history_mode="temporal", input_representation="box"):
        super().__init__()
        if history_mode not in ("temporal", "latest"):
            raise ValueError("history_mode must be temporal or latest")
        if dimension <= 0 or heads <= 0 or dimension % heads or depth <= 0 or history_depth <= 0:
            raise ValueError("invalid TemporalDiT architecture")
        self.architecture = dict(dimension=int(dimension), depth=int(depth), heads=int(heads),
                                 history_depth=int(history_depth))
        self.history_mode = history_mode
        self.input_representation = input_representation
        self.input_spec = input_spec(input_representation)
        for name, shape, initial in (("control_mean", 3, 0.), ("control_std", 3, 1.),
                ("condition_mean", 12, 0.), ("condition_std", 12, 1.),
                ("environment_mean", 6, 0.), ("environment_std", 6, 1.)):
            self.register_buffer(name, torch.full((shape,), initial))
        self.register_buffer("position_basis", torch.tensor(spline_basis(), dtype=torch.float32), persistent=False)
        self.register_buffer("velocity_basis", torch.tensor(spline_basis(derivative=1), dtype=torch.float32), persistent=False)
        self.register_buffer("acceleration_basis", torch.tensor(spline_basis(derivative=2), dtype=torch.float32), persistent=False)
        self.input = nn.Linear(3, dimension)
        self.position_embedding = nn.Parameter(torch.randn(1, FREE_CONTROLS, dimension) * .02)
        self.time_mlp = nn.Sequential(nn.Linear(64, dimension), nn.SiLU(), nn.Linear(dimension, dimension))
        self.condition_mlp = nn.Sequential(nn.Linear(12, dimension), nn.SiLU(), nn.Linear(dimension, dimension))
        if input_representation == "box":
            # Preserve names and initialization order for v1 box checkpoints.
            self.box_mlp = nn.Sequential(nn.Linear(10, dimension), nn.SiLU(), nn.Linear(dimension, dimension))
        elif input_representation == "point":
            self.point_mlp = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, dimension))
        else:
            self.voxel_cnn = nn.Sequential(nn.Conv3d(3, 8, 4, stride=4), nn.SiLU(),
                nn.Conv3d(8, 32, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv3d(32, dimension, 3, stride=2, padding=1), nn.SiLU())
        if input_representation != "box":
            self.spatial_mlp = nn.Linear(3, dimension)
            # CNN receptive-field centers follow its exact kernel/stride/pad;
            # the point pools use uniform 4m cells. Do not label them alike.
            axis = torch.linspace(-.75,.75,4) if input_representation == "point" else torch.tensor([-7.5,-3.5,.5,4.5])/ROI_RADIUS
            centers = torch.stack(torch.meshgrid(*[axis]*3, indexing="ij"), dim=-1).reshape(64,3)
            self.register_buffer("spatial_centers", centers, persistent=False)
        self.age_mlp = nn.Sequential(nn.Linear(1, dimension), nn.SiLU(), nn.Linear(dimension, dimension))
        self.null_observation = nn.Parameter(torch.zeros(1, 1, dimension))
        self.history_blocks = nn.ModuleList([nn.TransformerEncoderLayer(dimension, heads,
            dim_feedforward=2*dimension, dropout=0., activation="gelu", batch_first=True,
            norm_first=True) for _ in range(history_depth)])
        self.blocks = nn.ModuleList([DiTBlock(dimension, heads, dimension, True, False)
                                     for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dimension, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dimension, 2*dimension))
        self.output = nn.Linear(dimension, 3)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def set_normalization(self, value):
        for name in ("control_mean", "control_std", "condition_mean", "condition_std",
                     "environment_mean", "environment_std"):
            target = getattr(self, name)
            source = torch.as_tensor(value[name], dtype=target.dtype, device=target.device)
            if source.shape != target.shape or not bool(torch.isfinite(source).all()):
                raise ValueError("invalid normalization " + name)
            if name.endswith("std") and bool((source <= 0).any()):
                raise ValueError("normalization standard deviations must be positive")
            target.copy_(source)

    def encoder_parameter_count(self):
        names = ("box_mlp",) if self.input_representation == "box" else (
            ("point_mlp", "spatial_mlp") if self.input_representation == "point" else ("voxel_cnn", "spatial_mlp"))
        return sum(p.numel() for name in names for p in getattr(self, name).parameters())

    def encode_history(self, history_tokens, history_mask, history_times):
        batch = len(history_tokens)
        if (tuple(history_tokens.shape) != (batch, *self.input_spec['tokens_shape'])
                or tuple(history_mask.shape) != (batch, *self.input_spec['mask_shape'])):
            raise ValueError("history tokens/mask shapes do not match " + self.input_representation)
        if history_times.shape == (4,):
            history_times = history_times[None].expand(batch, -1)
        if history_times.shape != (batch, 4):
            raise ValueError("expected history_times[4] or [B,4]")
        mask = history_mask.bool()
        if self.history_mode == "latest":
            mask = mask.clone()
            mask[:, :-1] = False
        if self.input_representation == "box":
            normalized = history_tokens.clone()
            normalized[..., :6] = (normalized[..., :6] - self.environment_mean) / self.environment_std
            normalized = torch.where(mask[..., None], normalized, torch.zeros_like(normalized))
            value = self.box_mlp(normalized)
            valid = mask
        elif self.input_representation == "point":
            normalized = history_tokens / ROI_RADIUS
            normalized = torch.where(mask[..., None], normalized, torch.zeros_like(normalized))
            features = self.point_mlp(normalized).reshape(batch*4, POINT_COUNT, -1)
            cell = ((normalized+1)*2).floor().long().clamp(0,3)
            cell_id = (cell[...,0]*16 + cell[...,1]*4 + cell[...,2]).reshape(batch*4,POINT_COUNT)
            active = mask.reshape(batch*4, POINT_COUNT)
            features = features.masked_fill(~active[...,None], -torch.inf)
            pooled = features.new_full((batch*4,64,features.shape[-1]), -torch.inf)
            pooled.scatter_reduce_(1, cell_id[...,None].expand_as(features), features, reduce="amax", include_self=True)
            counts = torch.zeros((batch*4,64), device=features.device, dtype=torch.int64)
            counts.scatter_add_(1, cell_id, active.long())
            valid = (counts > 0).reshape(batch,4,64)
            value = torch.where(valid[...,None], pooled.reshape(batch,4,64,-1), 0.)
            value = value + self.spatial_mlp(self.spatial_centers)[None,None]
        else:
            volume = history_tokens.to(self.input.weight.dtype).reshape(batch*4,3,GRID_SIZE,GRID_SIZE,GRID_SIZE)
            value = self.voxel_cnn(volume).flatten(2).transpose(1,2).reshape(batch,4,64,-1)
            # Match the CNN receptive fields, including their padded edges.
            known = mask.reshape(batch*4,1,GRID_SIZE,GRID_SIZE,GRID_SIZE).float()
            known = F.max_pool3d(known, 4, 4)
            known = F.max_pool3d(known, 3, 2, 1)
            valid = F.max_pool3d(known, 3, 2, 1).bool().reshape(batch,4,64)
            value = value + self.spatial_mlp(self.spatial_centers)[None,None]
        value = value + self.age_mlp(history_times[..., None])[:, :, None]
        value = value.reshape(batch, -1, value.shape[-1])
        valid = valid.reshape(batch, -1)
        value = torch.cat((self.null_observation.expand(batch, -1, -1), value), dim=1)
        valid = torch.cat((torch.ones((batch, 1), dtype=torch.bool, device=valid.device), valid), dim=1)
        for block in self.history_blocks:
            value = block(value, src_key_padding_mask=~valid)
        return value, valid

    def forward_with_context(self, noisy_controls, diffusion_timestep, condition_context, observations, observation_mask):
        condition = self.time_mlp(timestep_embedding(diffusion_timestep, 64)) + condition_context
        value = self.input(noisy_controls) + self.position_embedding
        for block in self.blocks:
            value = block(value, condition, observations, observation_mask, None)
        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        return self.output(self.final_norm(value) * (1+scale[:, None]) + shift[:, None])

    def condition_context(self, condition):
        return self.condition_mlp((condition-self.condition_mean) / self.condition_std)

    def forward(self, noisy_controls, diffusion_timestep, condition,
                history_tokens, history_mask, history_times):
        if noisy_controls.shape != (len(condition), FREE_CONTROLS, 3) or condition.shape != (len(condition), 12):
            raise ValueError("expected noisy_controls[B,14,3] and condition[B,12]")
        observations, valid = self.encode_history(history_tokens, history_mask, history_times)
        return self.forward_with_context(noisy_controls, diffusion_timestep,
                                         self.condition_context(condition), observations, valid)

    def full_controls(self, normalized_free_controls, condition):
        free = normalized_free_controls * self.control_std + self.control_mean
        known = torch.stack((torch.zeros_like(condition[:, :3]), condition[:, 3:6] * (KNOT_SPACING/3)), dim=1)
        return torch.cat((known, free), dim=1)

    def decode(self, full_controls):
        return {name: torch.einsum("lc,bcd->bld", basis, full_controls) for name, basis in
                (("positions", self.position_basis), ("velocities", self.velocity_basis),
                 ("accelerations", self.acceleration_basis))}
