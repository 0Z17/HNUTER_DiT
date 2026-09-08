"""Training entry point for canonical HNUTER DiT variants."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .collision_loss import collision_auxiliary_loss
from .geometry import (
    CANONICAL_FRAME_VERSION,
    canonicalize_condition_batch,
    canonicalize_obstacles_batch,
    canonicalize_pose9_batch,
)
from .model import (
    ConditionalDiT, DiffusionSchedule, build_model, hard_clamp_endpoints,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path("/home/z017/research/HNUTER_datasets/free_flight_159103traj_dit_v001")
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "base_free_flight_159103traj_dit_relgeom_v002"


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class FrozenDataset:
    paths: np.ndarray
    conditions: np.ndarray
    split_codes: np.ndarray
    obstacles: np.ndarray
    obstacle_mask: np.ndarray
    environment_indices: np.ndarray
    source_dataset: str

    @classmethod
    def load(cls, root: Path) -> "FrozenDataset":
        root = root.resolve()
        freeze = json.loads((root / "dataset_freeze.json").read_text(encoding="utf-8"))
        cache = root / "prepared_dataset.npz"
        if freeze.get("status") != "frozen" or not freeze.get("integrity_passed"):
            raise RuntimeError("dataset is not a verified frozen dataset")
        if sha256_file(cache) != freeze.get("prepared_cache_sha256"):
            raise RuntimeError("prepared_dataset.npz SHA-256 does not match dataset freeze")
        with np.load(cache, allow_pickle=False) as data:
            result = cls(
                paths=data["paths"].copy(), conditions=data["conditions"].copy(),
                split_codes=data["split_codes"].copy(),
                obstacles=data["obstacles"].copy(),
                obstacle_mask=data["obstacle_mask"].copy(),
                environment_indices=data["environment_indices"].copy(),
                source_dataset=str(data["source_dataset"]),
            )
        if result.paths.shape[1:] != (128, 9) or result.conditions.shape != (len(result.paths), 18):
            raise RuntimeError("unexpected prepared trajectory shapes")
        if result.obstacles.shape[1:] != (32, 10):
            raise RuntimeError("unexpected environment token shape")
        if result.source_dataset != str(root):
            raise RuntimeError("prepared dataset source path mismatch")
        return result


def _canonicalize_pose9_numpy(poses: np.ndarray, conditions: np.ndarray) -> np.ndarray:
    start = conditions[:, :3]
    delta = conditions[:, 9:12] - start
    horizontal = np.linalg.norm(delta[:, :2], axis=1)
    cosine = np.where(horizontal < 1e-6, 1.0, delta[:, 0] / np.maximum(horizontal, 1e-6))
    sine = np.where(horizontal < 1e-6, 0.0, delta[:, 1] / np.maximum(horizontal, 1e-6))
    def rotate(value: np.ndarray) -> np.ndarray:
        c, s = cosine[:, None], sine[:, None]
        return np.stack((
            c * value[..., 0] + s * value[..., 1],
            -s * value[..., 0] + c * value[..., 1],
            value[..., 2],
        ), axis=-1)
    return np.concatenate((
        rotate(poses[..., :3] - start[:, None]),
        rotate(poses[..., 3:6]), rotate(poses[..., 6:9]),
    ), axis=-1).astype(np.float32)


def _canonicalize_obstacles_numpy(obstacles: np.ndarray, conditions: np.ndarray) -> np.ndarray:
    start = conditions[:, :3]
    delta = conditions[:, 9:12] - start
    yaw = np.arctan2(delta[:, 1], delta[:, 0])
    cosine, sine = np.cos(yaw), np.sin(yaw)
    result = obstacles.copy()
    relative = obstacles[..., :3] - start[:, None]
    result[..., 0] = cosine[:, None] * relative[..., 0] + sine[:, None] * relative[..., 1]
    result[..., 1] = -sine[:, None] * relative[..., 0] + cosine[:, None] * relative[..., 1]
    result[..., 2] = relative[..., 2]
    ch, sh = np.cos(0.5 * yaw)[:, None], np.sin(0.5 * yaw)[:, None]
    w, x, y, z = np.moveaxis(obstacles[..., 6:10], -1, 0)
    quaternion = np.stack((
        ch * w + sh * z, ch * x + sh * y,
        ch * y - sh * x, ch * z - sh * w,
    ), axis=-1)
    quaternion *= np.where(quaternion[..., :1] < 0, -1.0, 1.0)
    result[..., 6:10] = quaternion
    return result.astype(np.float32)


def _swap_conditions(conditions: np.ndarray) -> np.ndarray:
    return np.concatenate((conditions[:, 9:], conditions[:, :9]), axis=1)


def canonical_statistics(data: FrozenDataset, cache_path: Path, chunk_size: int = 256) -> dict[str, Any]:
    if cache_path.is_file():
        value = json.loads(cache_path.read_text(encoding="utf-8"))
        if value.get("canonical_frame") == CANONICAL_FRAME_VERSION:
            return value
    indices = np.flatnonzero(data.split_codes == 0)
    path_sum = np.zeros(9, dtype=np.float64)
    path_square = np.zeros(9, dtype=np.float64)
    path_count = 0
    environment_sum = np.zeros(6, dtype=np.float64)
    environment_square = np.zeros(6, dtype=np.float64)
    environment_count = 0
    for begin in range(0, len(indices), chunk_size):
        selected = indices[begin:begin + chunk_size]
        source_path = data.paths[selected]
        source_condition = data.conditions[selected]
        source_obstacles = data.obstacles[data.environment_indices[selected]]
        source_mask = data.obstacle_mask[data.environment_indices[selected]]
        for reverse in (False, True):
            conditions = _swap_conditions(source_condition) if reverse else source_condition
            paths = source_path[:, ::-1] if reverse else source_path
            canonical_path = _canonicalize_pose9_numpy(paths, conditions).astype(np.float64)
            path_sum += canonical_path.sum(axis=(0, 1))
            path_square += np.square(canonical_path).sum(axis=(0, 1))
            path_count += canonical_path.shape[0] * canonical_path.shape[1]
            canonical_obstacles = _canonicalize_obstacles_numpy(
                source_obstacles, conditions
            )[..., :6].astype(np.float64)
            active = canonical_obstacles[source_mask]
            environment_sum += active.sum(axis=0)
            environment_square += np.square(active).sum(axis=0)
            environment_count += len(active)
        if begin == 0 or (begin // chunk_size) % 100 == 0:
            print(f"canonical statistics {min(begin + chunk_size, len(indices))}/{len(indices)}", flush=True)
    path_mean = path_sum / path_count
    path_std = np.sqrt(np.maximum(path_square / path_count - np.square(path_mean), 1e-8))
    environment_mean = environment_sum / environment_count
    environment_std = np.sqrt(np.maximum(
        environment_square / environment_count - np.square(environment_mean), 1e-8
    ))
    value = {
        "schema_version": "hnuter_dit_canonical_statistics_v001",
        "canonical_frame": CANONICAL_FRAME_VERSION,
        "fit_split": "train_with_reverse_augmentation",
        "path_mean": path_mean.tolist(), "path_std": path_std.tolist(),
        "environment_feature_mean": environment_mean.tolist(),
        "environment_feature_std": environment_std.tolist(),
        "path_value_count": path_count,
        "environment_token_count": environment_count,
    }
    write_json(cache_path, value)
    return value


class ResumableBatchSampler:
    def __init__(self, size: int, batch_size: int, seed: int) -> None:
        self.size, self.batch_size = size, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.permutation = torch.randperm(size, generator=self.generator)
        self.cursor = 0
        self.epoch = 0

    def next_indices(self) -> np.ndarray:
        if self.cursor >= self.size:
            self.permutation = torch.randperm(self.size, generator=self.generator)
            self.cursor = 0
            self.epoch += 1
        end = min(self.cursor + self.batch_size, self.size)
        result = self.permutation[self.cursor:end].numpy()
        self.cursor = end
        return result

    def state_dict(self) -> dict[str, Any]:
        return {"generator_state": self.generator.get_state(), "permutation": self.permutation,
                "cursor": self.cursor, "epoch": self.epoch, "size": self.size,
                "batch_size": self.batch_size}

    def load_state_dict(self, value: dict[str, Any]) -> None:
        if value["size"] != self.size or value["batch_size"] != self.batch_size:
            raise RuntimeError("resume sampler shape mismatch")
        self.generator.set_state(value["generator_state"].cpu())
        self.permutation = value["permutation"].cpu()
        self.cursor, self.epoch = int(value["cursor"]), int(value["epoch"])


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.values = {name: value.detach().clone() for name, value in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            if self.values[name].is_floating_point():
                self.values[name].lerp_(value.detach(), 1.0 - self.decay)
            else:
                self.values[name].copy_(value.detach())

    def cpu_state_dict(self) -> dict[str, Tensor]:
        return {name: value.detach().cpu() for name, value in self.values.items()}


def raw_batch(data: FrozenDataset, split_indices: np.ndarray, local_indices: np.ndarray,
              reverse_augmentation: bool) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    base_size = len(split_indices)
    reverse = reverse_augmentation & (local_indices >= base_size)
    selected = split_indices[local_indices % base_size]
    paths = data.paths[selected].copy()
    conditions = data.conditions[selected].copy()
    if np.any(reverse):
        paths[reverse] = paths[reverse, ::-1]
        conditions[reverse] = _swap_conditions(conditions[reverse])
    environment = data.environment_indices[selected]
    return (
        torch.from_numpy(paths), torch.from_numpy(conditions),
        torch.from_numpy(data.obstacles[environment].copy()),
        torch.from_numpy(data.obstacle_mask[environment].copy()),
    )


def prepare_batch(batch: tuple[Tensor, Tensor, Tensor, Tensor], device: torch.device,
                  path_mean: Tensor, path_std: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    world_path, world_condition, obstacles, mask = (
        value.to(device, non_blocking=True) for value in batch
    )
    canonical_path = canonicalize_pose9_batch(world_path.float(), world_condition.float())
    canonical_condition = canonicalize_condition_batch(world_condition.float())
    canonical_obstacles = canonicalize_obstacles_batch(obstacles.float(), world_condition.float())
    condition_mean = path_mean.repeat(2)
    condition_std = path_std.repeat(2)
    return (
        (canonical_path - path_mean) / path_std,
        (canonical_condition - condition_mean) / condition_std,
        canonical_obstacles, mask.bool(),
    )


def diffusion_collision_loss(
    model: nn.Module, clean: Tensor, conditions: Tensor,
    obstacles: Tensor, obstacle_mask: Tensor, schedule: DiffusionSchedule,
    path_mean: Tensor, path_std: Tensor, *, collision_weight: float,
    safe_distance: float, collision_sigma: float, timestep_power: float,
    collision_stride: int, generator: torch.Generator | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    batch = len(clean)
    timesteps = torch.randint(
        0, len(schedule.betas), (batch,), device=clean.device,
        generator=generator,
    )
    noise = torch.randn(clean.shape, dtype=clean.dtype, device=clean.device, generator=generator)
    noise[:, 0] = 0
    noise[:, -1] = 0
    alpha_bar = schedule.alpha_bars[timesteps].reshape(batch, 1, 1)
    noisy = torch.sqrt(alpha_bar) * clean + torch.sqrt(1 - alpha_bar) * noise
    noisy = hard_clamp_endpoints(noisy, conditions)
    predicted_velocity = model(noisy, timesteps, conditions, obstacles, obstacle_mask)
    target_velocity = torch.sqrt(alpha_bar) * noise - torch.sqrt(1 - alpha_bar) * clean
    diffusion = (predicted_velocity[:, 1:-1] - target_velocity[:, 1:-1]).square().mean()
    predicted_x0 = torch.sqrt(alpha_bar) * noisy - torch.sqrt(1 - alpha_bar) * predicted_velocity
    predicted_x0 = hard_clamp_endpoints(predicted_x0.clamp(-6, 6), conditions)
    collision, diagnostics = collision_auxiliary_loss(
        predicted_x0, path_mean[None, None], path_std[None, None],
        obstacles, obstacle_mask, alpha_bar,
        safe_distance=safe_distance, sigma=collision_sigma,
        timestep_power=timestep_power, waypoint_stride=collision_stride,
    )
    total = diffusion + collision_weight * collision
    return total, {"diffusion": diffusion.detach(), "collision": collision.detach(), **diagnostics}


@torch.no_grad()
def validate(model: nn.Module, data: FrozenDataset, indices: np.ndarray,
             args: argparse.Namespace, device: torch.device, schedule: DiffusionSchedule,
             path_mean: Tensor, path_std: Tensor) -> dict[str, float]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed + 777)
    totals: list[dict[str, float]] = []
    for begin in range(0, len(indices), args.batch_size):
        local = np.arange(begin, min(begin + args.batch_size, len(indices)))
        batch = prepare_batch(
            raw_batch(data, indices, local, False), device, path_mean, path_std
        )
        total, metrics = diffusion_collision_loss(
            model, *batch, schedule, path_mean, path_std,
            collision_weight=args.collision_weight,
            safe_distance=args.collision_safe_distance,
            collision_sigma=args.collision_sigma,
            timestep_power=args.collision_timestep_power,
            collision_stride=args.collision_waypoint_stride,
            generator=generator,
        )
        totals.append({"total": float(total), **{k: float(v) for k, v in metrics.items()}})
    model.train()
    return {key: float(np.mean([row[key] for row in totals])) for key in totals[0]}


def checkpoint_payload(model_type: str, model_state: dict[str, Tensor], step: int,
                       validation: dict[str, float], data: FrozenDataset,
                       stats: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_type": model_type, "model_state": model_state,
        "architecture": {
            "sequence_length": 128, "unet_channels": 64,
            "dit_dimension": 256, "dit_depth": 8, "dit_heads": 8,
        },
        "diffusion_steps": 100, "prediction_type": "velocity",
        "training_step": step, "validation_loss": validation["total"],
        "validation_diffusion_loss": validation["diffusion"],
        "validation_collision_loss": validation["collision"],
        "path_mean": np.asarray(stats["path_mean"], dtype=np.float32),
        "path_std": np.asarray(stats["path_std"], dtype=np.float32),
        "environment_feature_mean": np.asarray(
            stats["environment_feature_mean"], dtype=np.float32
        ),
        "environment_feature_std": np.asarray(
            stats["environment_feature_std"], dtype=np.float32
        ),
        "environment_token_schema_version": "box_geometry_canonical_v002",
        "canonical_frame": CANONICAL_FRAME_VERSION,
        "relative_geometry_attention": model_type == "dit_rel_cross",
        "relative_geometry_features": (
            ["body_delta_xyz", "distance", "obb_extent_xyz", "relative_rotation_6d"]
            if model_type == "dit_rel_cross" else []
        ),
        "collision_auxiliary": {
            "weight": args.collision_weight,
            "safe_distance_m": args.collision_safe_distance,
            "sigma_m": args.collision_sigma,
            "timestep_weight": f"alpha_bar^{args.collision_timestep_power}",
            "waypoint_stride": args.collision_waypoint_stride,
            "robot_spheres": 42,
        },
        "source_dataset": data.source_dataset,
    }


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = FrozenDataset.load(args.dataset)
    stats = canonical_statistics(data, args.output_dir / "canonical_statistics.json")
    write_json(args.output_dir / "resolved_training_config.json", {
        "schema_version": (
            "hnuter_dit_relative_geometry_training_v002"
            if args.model_type == "dit_rel_cross"
            else "hnuter_dit_canonical_cross_attention_training_v003"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset), "output_dir": str(args.output_dir),
        "canonical_frame": CANONICAL_FRAME_VERSION,
        "architecture": {
            "model_type": args.model_type, "sequence_length": 128,
            "dimension": 256, "depth": 8, "heads": 8,
            "pairwise_relative_geometry": args.model_type == "dit_rel_cross",
        },
        "training": vars(args),
    })
    if args.audit_only:
        print(json.dumps({"status": "audit_passed", "statistics": stats}))
        return args.output_dir
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = build_model(
        args.model_type, sequence_length=128, dit_dimension=256,
        dit_depth=8, dit_heads=8,
    ).to(device)
    assert isinstance(model, ConditionalDiT)
    path_mean = torch.as_tensor(stats["path_mean"], dtype=torch.float32, device=device)
    path_std = torch.as_tensor(stats["path_std"], dtype=torch.float32, device=device)
    if args.model_type == "dit_rel_cross":
        model.configure_path_normalization(path_mean, path_std)
    model.configure_environment_normalization(
        stats["environment_feature_mean"], stats["environment_feature_std"]
    )
    parameter_count = sum(value.numel() for value in model.parameters())
    print(f"device={device} model={args.model_type} parameters={parameter_count:,}", flush=True)
    train_indices = np.flatnonzero(data.split_codes == 0)
    validation_indices = np.flatnonzero(data.split_codes == 1)
    if args.validation_max_samples > 0:
        validation_indices = validation_indices[:args.validation_max_samples]
    sampler = ResumableBatchSampler(2 * len(train_indices), args.batch_size, args.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    warmup = min(args.warmup_steps, max(1, args.training_steps // 10))
    def lr_multiplier(step: int) -> float:
        if step < warmup:
            return max((step + 1) / warmup, 1e-3)
        progress = (step - warmup) / max(args.training_steps - warmup, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1)))
        return args.minimum_learning_rate_ratio + (1 - args.minimum_learning_rate_ratio) * cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    schedule = DiffusionSchedule.cosine(100, device)
    ema = ExponentialMovingAverage(model, args.ema_decay)
    use_amp = device.type == "cuda" and not args.disable_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_context = (
        (lambda: torch.autocast("cuda", dtype=torch.float16))
        if use_amp else (lambda: nullcontext())
    )
    model_dir = args.output_dir / "models" / args.model_type
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path, ema_path, last_path = (
        model_dir / "best.pt", model_dir / "ema.pt", model_dir / "last.pt"
    )
    history: list[dict[str, Any]] = []
    best_validation, best_step, start_step = float("inf"), 0, 1
    no_improvement = 0
    if last_path.is_file() and not args.force_retrain:
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        ema.values = {k: v.to(device) for k, v in state["ema_state"].items()}
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        sampler.load_state_dict(state["sampler_state"])
        history = state["history"]
        best_validation = float(state["best_validation_loss"])
        best_step = int(state["best_training_step"])
        no_improvement = int(state["validation_intervals_without_improvement"])
        start_step = int(state["training_step"]) + 1
        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_random_state"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if device.type == "cuda" and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng_state"]])
        print(f"resumed from step {start_step - 1}", flush=True)
    elif args.force_retrain:
        for path in (best_path, ema_path, last_path, model_dir / "training_complete.json"):
            if path.exists():
                path.unlink()
    started = time.monotonic()
    rolling: dict[str, float] = {"total": 0, "diffusion": 0, "collision": 0}
    rolling_count = 0
    final_step = start_step - 1
    for step in range(start_step, args.training_steps + 1):
        final_step = step
        batch = prepare_batch(
            raw_batch(data, train_indices, sampler.next_indices(), True),
            device, path_mean, path_std,
        )
        optimizer.zero_grad(set_to_none=True)
        with amp_context():
            total, metrics = diffusion_collision_loss(
                model, *batch, schedule, path_mean, path_std,
                collision_weight=args.collision_weight,
                safe_distance=args.collision_safe_distance,
                collision_sigma=args.collision_sigma,
                timestep_power=args.collision_timestep_power,
                collision_stride=args.collision_waypoint_stride,
            )
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)
        rolling["total"] += float(total.detach())
        rolling["diffusion"] += float(metrics["diffusion"])
        rolling["collision"] += float(metrics["collision"])
        rolling_count += 1
        if step % args.validation_interval == 0 or step == args.training_steps:
            live = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema.values)
            validation = validate(
                model, data, validation_indices, args, device, schedule,
                path_mean, path_std,
            )
            model.load_state_dict(live)
            entry = {
                "step": step,
                "training_total_loss": rolling["total"] / rolling_count,
                "training_diffusion_loss": rolling["diffusion"] / rolling_count,
                "training_collision_loss": rolling["collision"] / rolling_count,
                **{f"validation_{k}": v for k, v in validation.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_s": time.monotonic() - started,
            }
            history.append(entry)
            print(
                f"step={step:05d}/{args.training_steps} "
                f"train={entry['training_total_loss']:.5f} "
                f"diff={validation['diffusion']:.5f} col={validation['collision']:.5f} "
                f"total={validation['total']:.5f} elapsed={entry['elapsed_s']:.1f}s",
                flush=True,
            )
            rolling = {"total": 0, "diffusion": 0, "collision": 0}
            rolling_count = 0
            inference = checkpoint_payload(
                args.model_type, ema.cpu_state_dict(), step, validation,
                data, stats, args,
            )
            if validation["total"] < best_validation:
                best_validation, best_step, no_improvement = validation["total"], step, 0
                torch.save(inference, best_path)
            else:
                no_improvement += 1
            torch.save(inference, ema_path)
            torch.save(inference | {
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "ema_state": ema.cpu_state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "sampler_state": sampler.state_dict(),
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "history": history, "best_validation_loss": best_validation,
                "best_training_step": best_step,
                "validation_intervals_without_improvement": no_improvement,
            }, last_path)
            if args.early_stopping_patience and no_improvement >= args.early_stopping_patience:
                print("early stopping", flush=True)
                break
    write_json(model_dir / "training_history.json", {
        "model_type": args.model_type, "parameter_count": parameter_count,
        "train_sample_count_with_reverse_augmentation": 2 * len(train_indices),
        "validation_sample_count": len(validation_indices),
        "best_validation_loss": best_validation, "best_training_step": best_step,
        "wall_time_s": time.monotonic() - started, "device": str(device),
        "amp": use_amp, "history": history,
    })
    write_json(model_dir / "checkpoint_selection.json", {
        "selection_split": "validation", "criterion": "minimum_total_loss",
        "selected_checkpoint": "best.pt", "best_training_step": best_step,
        "best_validation_loss": best_validation,
    })
    write_json(model_dir / "training_complete.json", {
        "status": "complete", "final_training_step": final_step,
        "stopped_early": final_step < args.training_steps,
    })
    checkpoint_hash = sha256_file(best_path)
    write_json(args.output_dir / "run_status.json", {
        "status": "complete", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_checkpoint": str(best_path.relative_to(args.output_dir)),
        "selected_checkpoint_sha256": checkpoint_hash,
    })
    print(f"training complete: {best_path} sha256={checkpoint_hash}", flush=True)
    return best_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model-type", choices=("dit_cross", "dit_rel_cross"),
        default="dit_rel_cross",
    )
    parser.add_argument("--training-steps", type=int, default=48_000)
    parser.add_argument("--validation-interval", type=int, default=2_000)
    parser.add_argument(
        "--validation-max-samples", type=int, default=0,
        help="smoke-test only; 0 evaluates the complete validation split",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--collision-weight", type=float, default=0.002)
    parser.add_argument("--collision-safe-distance", type=float, default=0.04)
    parser.add_argument("--collision-sigma", type=float, default=0.05)
    parser.add_argument("--collision-timestep-power", type=float, default=1.0)
    parser.add_argument("--collision-waypoint-stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    args.dataset = args.dataset.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.training_steps <= 0 or args.validation_interval <= 0 or args.batch_size <= 0:
        parser.error("training steps, validation interval, and batch size must be positive")
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
