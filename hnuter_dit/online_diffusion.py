"""Low-noise DiT refinement of explicit geometric priors using current LiDAR.

The checkpoint is unchanged. These are hand-designed translation proposals
refined by its actual conditional velocity predictions, not trajectories learned
from new LiDAR training. Attitude follows a bounded endpoint SLERP; the network's
attitude predictions are discarded. Neither these boxes nor this function certify
unobserved space or collision freedom. Validate the result before executing it.
Prior offsets and network/guidance corrections lie in the start-goal chord's
normal plane, so translating proposals make uniform forward chord progress.
This deliberately excludes routes that require reversing along that chord.
A sine-squared correction envelope preserves the prior's endpoint tangents;
it does not imply that those directions have valid LiDAR returns.
"""
from __future__ import annotations

import time
import numpy as np
import torch
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation, Slerp

from .geometry import CANONICAL_FRAME_VERSION, CanonicalFrame
from .guidance import CollisionGuidanceConfig, guide_clean_trajectory
from .model import hard_clamp_endpoints, pose7_to_pose9


def _pose(value, name):
    result = np.asarray(value, dtype=np.float64).copy()
    if result.shape != (7,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite xyz+wxyz pose")
    norm = np.linalg.norm(result[3:])
    if norm < 1e-8:
        raise ValueError(f"{name} quaternion must be nonzero")
    result[3:] /= norm
    return result


def _basis(length, control_points):
    """Bounded B-spline corrections preserving the prior endpoint tangents.

    Zero endpoint controls make the displacement O(s) near an endpoint. The
    sin² envelope makes it O(s³), so the correction and its first two phase
    derivatives vanish there. Nonnegative basis row sums remain <=1; bounded
    control vectors therefore still bound every interpolated displacement.
    """
    knots = np.r_[np.zeros(4), np.linspace(0, 1, control_points-2)[1:-1], np.ones(4)]
    phase = np.linspace(0, 1, length)
    full = BSpline(knots, np.eye(control_points), 3)(phase)
    basis = full[:, 1:-1] * np.sin(np.pi * phase)[:, None] ** 2
    second_difference = np.diff(np.eye(control_points), n=2, axis=0)[:, 1:-1]
    fit = np.linalg.solve(basis.T @ basis + .15 * second_difference.T @ second_difference
                          + 1e-6 * np.eye(control_points-2), basis.T)
    return basis.astype(np.float32), fit.astype(np.float32)


def _priors(start, goal, frame, count, length):
    phase = np.linspace(0, 1, length)
    raw_condition = pose7_to_pose9(np.stack((start, goal))).reshape(18)
    canonical = frame.pose9_to_canonical(raw_condition.reshape(2, 9))
    positions = (1-phase[:, None]) * canonical[0, :3] + phase[:, None] * canonical[1, :3]
    span = float(np.linalg.norm(goal[:3] - start[:3]))
    side, up = min(.9, .3 * span), min(.6, .2 * span)
    offsets = np.array([(0., 0.), (side, 0.), (-side, 0.), (0., up),
                        (.7*side, .7*up), (-.7*side, .7*up),
                        (1.3*side, .25*up), (-1.3*side, .25*up)])
    selected = offsets[np.arange(count) % len(offsets)]
    chord = canonical[1, :3] - canonical[0, :3]
    direction = chord / max(float(np.linalg.norm(chord)), 1e-12)
    normal_offsets = np.c_[np.zeros(count), selected]
    # "Up" has a longitudinal component when the goal changes altitude. Remove
    # it from the prior as well, so the progress constraint holds in 3D rather
    # than only for horizontal goals.
    normal_offsets -= (normal_offsets @ direction)[:, None] * direction
    pose9 = np.tile(canonical[0], (count, length, 1))
    pose9[..., :3] = positions
    # Zero position and tangent offsets at both endpoints.
    envelope = np.sin(np.pi * phase) ** 2
    pose9[..., :3] += normal_offsets[:, None] * envelope[None, :, None]
    rotations = Rotation.from_quat(np.stack((start, goal))[:, [4, 5, 6, 3]])
    quaternions = Slerp([0., 1.], rotations)(phase).as_quat()[:, [3, 0, 1, 2]]
    attitudes = pose7_to_pose9(np.c_[np.zeros((length, 3)), quaternions])
    pose9[..., 3:] = frame.pose9_to_canonical(attitudes)[..., 3:]
    return pose9.astype(np.float32), quaternions, selected, normal_offsets


def sample_online(runtime, tokens, mask, start_pose, goal_pose, seed, count=8, *,
                  start_timestep=10, denoising_steps=6,
                  max_position_correction_m=.25, control_points=10,
                  max_attitude_change_deg=45.):
    """Return world-frame ``[count,128,7]`` poses and auditable inference metadata.

    Each DDIM update calls the same checkpoint with all current LiDAR tokens.
    Its predicted clean positions are projected onto a 10-control-point cubic
    displacement field around the fixed geometric prior. Each control vector
    is norm bounded, so the nonnegative B-spline basis also bounds displacement
    between samples. Smooth OBB collision guidance then undergoes the same
    projection. Every displacement control is perpendicular to the full 3D
    start-goal chord. Together with perpendicular prior offsets, this fixes
    chord progress to the uniform waypoint phase and prevents longitudinal
    reversals/near-zero intervals. Coincident endpoints allow no translation.
    The displacement basis also has zero endpoint first/second derivatives,
    preserving the prior's initial/final direction rather than adding an abrupt
    learned slope. This does not establish visibility of the resulting prefix.
    This controls local refinement; it cannot find routes requiring backtracking.
    """
    if runtime.canonical_frame != CANONICAL_FRAME_VERSION:
        raise ValueError("online refinement requires the canonical v003-compatible checkpoint")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or not 1 <= count <= 16:
        raise ValueError("count must be an integer in [1,16]")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    schedule_length = len(runtime.schedule.alpha_bars)
    if not isinstance(start_timestep, int) or not 1 <= start_timestep < schedule_length:
        raise ValueError("start_timestep must be inside the checkpoint noise schedule")
    if not isinstance(denoising_steps, int) or not 2 <= denoising_steps <= start_timestep+1:
        raise ValueError("denoising_steps must be between 2 and start_timestep+1")
    if not isinstance(control_points, int) or not 4 <= control_points <= 32:
        raise ValueError("control_points must be an integer in [4,32]")
    if not np.isfinite(max_position_correction_m) or not 0 < max_position_correction_m <= 1.:
        raise ValueError("max_position_correction_m must be in (0,1]")
    if not np.isfinite(max_attitude_change_deg) or not 0 < max_attitude_change_deg <= 90:
        raise ValueError("max_attitude_change_deg must be in (0,90]")
    start, goal = _pose(start_pose, "start_pose"), _pose(goal_pose, "goal_pose")
    attitude_pair = Rotation.from_quat(np.stack((start, goal))[:, [4, 5, 6, 3]])
    attitude_change = float(np.rad2deg((attitude_pair[0].inv() * attitude_pair[1]).magnitude()))
    if attitude_change > max_attitude_change_deg + 1e-8:
        raise ValueError("endpoint attitude change exceeds the constrained online prototype limit")
    tokens, mask = np.asarray(tokens, dtype=np.float32), np.asarray(mask, dtype=bool)
    if tokens.shape != (32, 10) or mask.shape != (32,) or not np.isfinite(tokens).all():
        raise ValueError("expected finite tokens[32,10] and mask[32]")
    if np.any(tokens[mask, 3:6] <= 0) or np.any(np.linalg.norm(tokens[mask, 6:], axis=1) < 1e-8):
        raise ValueError("active tokens need positive full extents and nonzero wxyz quaternions")
    length = int(runtime.sequence_length)
    if length != 128:
        raise ValueError("this online adapter expects the 128-pose checkpoint")

    started = time.perf_counter()
    condition_world = pose7_to_pose9(np.stack((start, goal))).reshape(18)
    frame = CanonicalFrame.from_pose9_condition(condition_world)
    condition_metric = frame.pose9_to_canonical(condition_world.reshape(2, 9)).reshape(18)
    prior_np, attitudes, offsets, normal_offsets = _priors(start, goal, frame, count, length)
    device = runtime.device
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    mean, std = tensor(runtime.path_mean), tensor(runtime.path_std)
    if not torch.isfinite(std).all() or bool((std <= 0).any()):
        raise ValueError("checkpoint path standard deviations must be positive and finite")
    prior = tensor(prior_np)
    prior_normalized = (prior - mean) / std
    condition = ((tensor(condition_metric).reshape(2, 9) - mean) / std).reshape(18)
    conditions = condition[None].expand(count, -1)
    obstacles = tensor(frame.obstacles_to_canonical(tokens))[None].expand(count, -1, -1)
    obstacle_mask = torch.as_tensor(mask, device=device)[None].expand(count, -1)
    basis_np, fit_np = _basis(length, control_points)
    basis, fit = tensor(basis_np), tensor(fit_np)
    chord = condition_metric[9:12] - condition_metric[:3]
    chord_length = float(np.linalg.norm(chord))
    chord_direction = tensor(chord / max(chord_length, 1e-12))

    def project(clean):
        metric = clean * std + mean
        controls = torch.einsum("kl,blj->bkj", fit, metric[..., :3] - prior[..., :3])
        if chord_length > 1e-8:
            controls = controls - (controls * chord_direction).sum(dim=-1, keepdim=True) * chord_direction
        else:
            controls = torch.zeros_like(controls)
        magnitude = torch.linalg.vector_norm(controls, dim=-1, keepdim=True)
        controls = controls * torch.clamp(max_position_correction_m / magnitude.clamp_min(1e-12), max=1.)
        result = prior.clone()
        result[..., :3] += torch.einsum("lk,bkj->blj", basis, controls)
        return hard_clamp_endpoints((result - mean) / std, conditions)

    indices = np.unique(np.linspace(start_timestep, 0, denoising_steps).round().astype(int))[::-1].copy()
    generator = torch.Generator(device=device).manual_seed(int(seed))
    alpha = runtime.schedule.alpha_bars[int(indices[0])]
    noise = torch.randn(prior_normalized.shape, dtype=torch.float32, device=device, generator=generator)
    latent = hard_clamp_endpoints(torch.sqrt(alpha) * prior_normalized + torch.sqrt(1-alpha) * noise, conditions)
    guidance = CollisionGuidanceConfig(mode="smooth", steps=1, step_size_m=.12,
                                      control_points=control_points)
    updates = []
    # Use the same model lock as ordinary inference; guidance explicitly enables
    # autograd for its input displacement but never trains model weights.
    with runtime.lock, torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        neural_started = time.perf_counter()
        for index, timestep in enumerate(indices):
            step_tensor = torch.full((count,), int(timestep), dtype=torch.long, device=device)
            prediction = runtime.model(latent, step_tensor, conditions, obstacles, obstacle_mask)
            if prediction.shape != latent.shape or not bool(torch.isfinite(prediction).all()):
                raise RuntimeError("checkpoint returned an invalid velocity prediction")
            alpha = runtime.schedule.alpha_bars[int(timestep)]
            raw_clean = torch.sqrt(alpha) * latent - torch.sqrt(1-alpha) * prediction
            raw_correction = (raw_clean * std + mean)[..., :3] - prior[..., :3]
            learned = project(raw_clean)
            if bool(mask.any()):
                guided, _ = guide_clean_trajectory(learned, conditions, mean, std,
                                                   obstacles, obstacle_mask, guidance)
            else:
                guided = learned
            guided_correction = ((guided - learned) * std)[..., :3]
            clean = project(guided)
            if not bool(torch.isfinite(clean).all()):
                raise RuntimeError("online guidance produced a nonfinite trajectory")
            updates.append({
                "timestep": int(timestep),
                "raw_learned_correction_mean_m": float(torch.linalg.vector_norm(raw_correction, dim=-1).mean()),
                "raw_learned_correction_max_m": float(torch.linalg.vector_norm(raw_correction, dim=-1).max()),
                "projected_learned_correction_max_m": float(torch.linalg.vector_norm(((learned-prior_normalized)*std)[..., :3], dim=-1).max()),
                "guidance_displacement_mean_m": float(torch.linalg.vector_norm(guided_correction, dim=-1).mean()),
                "guidance_displacement_max_m": float(torch.linalg.vector_norm(guided_correction, dim=-1).max()),
                "network_velocity_prediction_rms": float(torch.sqrt(prediction.square().mean())),
            })
            predicted_noise = (latent - torch.sqrt(alpha) * clean) / torch.sqrt(1-alpha).clamp_min(1e-6)
            if index + 1 < len(indices):
                previous_alpha = runtime.schedule.alpha_bars[int(indices[index+1])]
                latent = hard_clamp_endpoints(torch.sqrt(previous_alpha) * clean
                    + torch.sqrt(1-previous_alpha) * predicted_noise, conditions)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - neural_started
        metric = (clean * std + mean).cpu().numpy()
    world = frame.pose9_to_world(metric)
    paths = np.concatenate((world[..., :3], np.broadcast_to(attitudes, (count, length, 4))), axis=-1)
    paths[:, 0], paths[:, -1] = start, goal
    paths = paths.astype(np.float32)
    metadata = {
        "algorithm": "low_noise_conditional_ddim_monotone_tangent_preserving_refinement_v3",
        "checkpoint_sha256": str(runtime.checkpoint_sha256),
        "weights_retrained": False,
        "prior_source": "hand_designed_straight_and_sine_squared_side_up_curves",
        "prior_side_up_offsets_m": offsets.tolist(),
        "prior_normal_plane_offsets_canonical_m": normal_offsets.tolist(),
        "translation_progress_policy": "uniform_forward_chord_parameter_normal_plane_corrections_only",
        "correction_envelope": "sin_squared_uniform_phase",
        "correction_endpoint_derivatives": "zero_value_first_and_second_phase_derivatives",
        "start_goal_distance_m": chord_length,
        "minimum_chord_progress_step_m": float((np.diff(paths[..., :3].astype(np.float64), axis=1)
            @ ((goal[:3]-start[:3]) / max(chord_length, 1e-12))).min()),
        "canonical_frame": CANONICAL_FRAME_VERSION,
        "conditioning": "same_current_32_lidar_box_tokens_cross_attention_at_every_model_call",
        "active_tokens": int(mask.sum()),
        "model_evaluations": len(indices), "timesteps": indices.tolist(),
        "seed": int(seed), "candidate_count": int(count),
        "attitude_policy": "endpoint_slerp_only_network_attitude_predictions_discarded",
        "endpoint_attitude_change_deg": attitude_change,
        "max_endpoint_attitude_change_deg": max_attitude_change_deg,
        "max_position_correction_m": max_position_correction_m,
        "control_points": control_points,
        "guidance_enabled": bool(mask.any()),
        "updates": updates,
        "final_max_prior_correction_m": float(np.linalg.norm(metric[..., :3]-prior_np[..., :3], axis=-1).max()),
        "inference_seconds": inference_seconds,
        "wall_seconds": time.perf_counter()-started,
        "collision_certified": False,
        "unknown_space_policy": "unknown; caller must validate before execution",
    }
    return paths, metadata
