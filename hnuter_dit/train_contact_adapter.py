"""Reproducible frozen-v003 Contact adapter training, with exact step resume.

The dataset is immutable. Only 8,000 train experts may enter optimization;
validation uses all 1,000 held-out experts and a common, batch-independent
noise bank. Checkpoints contain adapter/optimizer state, never Base weights.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .collision_loss import collision_auxiliary_loss
from .contact_adapter import (
    ADAPTER_SCHEMA, ContactAdapter, ContactAdapterConfig, Task, load_frozen_base,
    prepare_contact_condition_batch, sha256_file,
)
from .geometry import CANONICAL_FRAME_VERSION, canonicalize_pose9_batch
from .model import DiffusionSchedule, hard_clamp_endpoints, quaternion_wxyz_to_matrix_torch, rotation_6d_to_matrix
from .training import FrozenDataset, ResumableBatchSampler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "artifacts/base_free_flight_159103traj_dit_canonical_cross_v003/models/dit_cross/best.pt"
DEFAULT_DATASET = Path("/home/z017/research/HNUTER_datasets/surface_multifamily_contact_10000traj_v001")
VALIDATION_NOISE_SEED = 202610110077
DATA_SUBSET_SEED = 20261011


class ContactBatchSampler(ResumableBatchSampler):
    """Always return a full batch, carrying the tail across shuffle boundaries."""
    def next_indices(self) -> np.ndarray:
        parts: list[Tensor] = []
        needed = self.batch_size
        while needed:
            if self.cursor >= self.size:
                self.permutation = torch.randperm(self.size, generator=self.generator)
                self.cursor = 0
                self.epoch += 1
            count = min(needed, self.size - self.cursor)
            parts.append(self.permutation[self.cursor:self.cursor + count])
            self.cursor += count
            needed -= count
        return torch.cat(parts).numpy()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def frozen_state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    # Base normalization buffers are deliberately non-persistent in model.py.
    # Include them explicitly as well as checkpoint weights in the freeze gate.
    values = dict(model.state_dict())
    values.update({f"buffer:{name}": value for name, value in model.named_buffers()})
    for name, value in sorted(values.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass
class ContactDataset:
    core: FrozenDataset
    features: np.ndarray
    feature_mask: np.ndarray
    supports: np.ndarray
    support_mask: np.ndarray
    forbidden: np.ndarray
    forbidden_mask: np.ndarray
    map_ids: np.ndarray
    condition_ids: np.ndarray
    families: dict[str, str]
    provenance: dict[str, Any]

    @classmethod
    def load(cls, root: Path, base: dict[str, Any]) -> "ContactDataset":
        root = root.resolve()
        core = FrozenDataset.load(root)
        manifest = json.loads((root / "manifest.json").read_text())
        provenance = json.loads((root / "provenance.json").read_text())
        normalization = manifest["normalization"]
        if normalization.get("fitted_on_contact_data") is not False or normalization.get("canonical_frame") != CANONICAL_FRAME_VERSION:
            raise RuntimeError("Contact dataset must reference frozen v003 canonical statistics")
        if provenance.get("base_checkpoint_sha256") != base["base_sha256"]:
            raise RuntimeError("Contact dataset and frozen Base checkpoint hashes differ")
        reference = Path(normalization["reference"])
        if sha256_file(reference) != normalization["sha256"]:
            raise RuntimeError("v003 normalization reference SHA-256 mismatch")
        stats = json.loads(reference.read_text())
        with np.load(root / "prepared_dataset.npz", allow_pickle=False) as data:
            for key in ("path_mean", "path_std", "environment_feature_mean", "environment_feature_std"):
                if not np.array_equal(data[key], np.asarray(base[key], np.float32)) or not np.array_equal(data[key], np.asarray(stats[key], np.float32)):
                    raise RuntimeError(f"Contact normalization {key} does not match frozen Base")
            if str(data["normalization_coordinate_frame"]) != CANONICAL_FRAME_VERSION or str(data["paths_coordinate_frame"]) != "world":
                raise RuntimeError("unexpected Contact coordinate convention")
            result = cls(core, data["surface_features_world"].copy(), data["surface_feature_mask"].copy(),
                         data["task_support_tokens_world"].copy(), data["task_support_mask"].copy(),
                         data["forbidden_regions_intrinsic"].copy(), data["forbidden_region_mask"].copy(),
                         data["map_ids"].copy(), data["condition_ids"].copy(), {}, {})
        count = len(core.paths)
        for field, shape in ((result.features, (count, 32)), (result.feature_mask, (count, 32)),
                             (result.supports, (count, 8, 10)), (result.support_mask, (count, 8)),
                             (result.forbidden, (count, 8, 4)), (result.forbidden_mask, (count, 8)),
                             (result.map_ids, (count,)), (result.condition_ids, (count,))):
            if field.shape != shape:
                raise RuntimeError(f"unexpected task array shape {field.shape}; expected {shape}")
        if not np.isfinite(result.features).all() or not np.allclose(result.features[:, :2].sum(axis=1), 1):
            raise RuntimeError("invalid Contact type or geometry features")
        for split in ("train", "validation", "test"):
            index = json.loads((root / split / "index.json").read_text())
            for row in index["samples"]:
                map_id = row["group_id"]
                family = row["family"]
                if map_id in result.families and result.families[map_id] != family:
                    raise RuntimeError("source map appears in multiple environment families")
                result.families[map_id] = family
        for map_id in np.unique(result.map_ids):
            if len(np.unique(core.split_codes[result.map_ids == map_id])) != 1:
                raise RuntimeError("source map leakage between train/validation/test")
            if map_id not in result.families:
                raise RuntimeError("source map family metadata missing")
        result.provenance = dict(dataset_root=str(root), dataset_freeze_sha256=sha256_file(root / "dataset_freeze.json"),
                                 dataset_manifest_sha256=sha256_file(root / "manifest.json"),
                                 prepared_cache_sha256=json.loads((root / "dataset_freeze.json").read_text())["prepared_cache_sha256"],
                                 normalization_reference=str(reference), normalization_sha256=normalization["sha256"])
        return result


def select_training_indices(data: ContactDataset, train_size: int,
                            data_subset_seed: int = DATA_SUBSET_SEED) -> tuple[np.ndarray, dict[str, Any]]:
    """Nested whole-map selection: hash ranks within families, then round-robin."""
    if train_size <= 0 or train_size % 20:
        raise ValueError("train_size must be a positive multiple of 20 expert paths")
    train = np.flatnonzero(data.core.split_codes == 0)
    groups: dict[str, list[str]] = {}
    rows: dict[str, np.ndarray] = {}
    for map_id in np.unique(data.map_ids[train]):
        indices = train[data.map_ids[train] == map_id]
        if len(indices) != 20:
            raise RuntimeError("whole-source-map subsets require exactly 20 experts per train map")
        rows[str(map_id)] = indices
        groups.setdefault(data.families[str(map_id)], []).append(str(map_id))
    for maps in groups.values():
        maps.sort(key=lambda map_id: hashlib.sha256(f"{data_subset_seed}:{map_id}".encode()).hexdigest())
    family_order = sorted(groups)
    ordered = [groups[family][round_index] for round_index in range(max(map(len, groups.values())))
               for family in family_order if round_index < len(groups[family])]
    requested_maps = train_size // 20
    if requested_maps > len(ordered):
        raise ValueError("requested expert budget exceeds the train split")
    selected_maps = ordered[:requested_maps]
    selected = np.sort(np.concatenate([rows[map_id] for map_id in selected_maps]))
    family_counts = {family: sum(data.families[map_id] == family for map_id in selected_maps) for family in family_order}
    subset = dict(strategy="whole_map_family_hash_rank_round_robin_v001", data_subset_seed=data_subset_seed,
                  train_size=train_size, map_count=len(selected_maps), selected_map_ids=selected_maps,
                  family_map_counts=family_counts, trajectory_indices=selected.tolist(),
                  expert_pairs_preserved=True, augmentation_does_not_increase_expert_count=True)
    return selected, subset


@dataclass
class ContactBatch:
    clean: Tensor
    conditions: Tensor
    obstacles: Tensor
    obstacle_mask: Tensor
    task: dict[str, Tensor]


def prepare_contact_batch(data: ContactDataset, indices: np.ndarray, device: torch.device,
                          path_mean: Tensor, path_std: Tensor,
                          reverse_flags: np.ndarray | None = None) -> ContactBatch:
    paths = data.core.paths[indices].copy()
    conditions = data.core.conditions[indices].copy()
    if reverse_flags is not None and np.any(reverse_flags):
        paths[reverse_flags] = paths[reverse_flags, ::-1]
        c = conditions[reverse_flags].copy()
        conditions[reverse_flags] = np.concatenate((c[:, 9:], c[:, :9]), axis=1)
    world_path = torch.from_numpy(paths).to(device)
    world_condition = torch.from_numpy(conditions).to(device)
    environment = data.core.environment_indices[indices]
    values = [data.core.obstacles[environment], data.core.obstacle_mask[environment],
              data.features[indices], data.feature_mask[indices], data.supports[indices], data.support_mask[indices]]
    tensors = [torch.from_numpy(np.asarray(value)).to(device) for value in values]
    c, o, m, task = prepare_contact_condition_batch(world_condition, *tensors, path_mean, path_std,
                    torch.from_numpy(data.forbidden[indices]).to(device),
                    torch.from_numpy(data.forbidden_mask[indices]).to(device))
    clean = (canonicalize_pose9_batch(world_path.float(), world_condition.float()) - path_mean) / path_std
    return ContactBatch(clean, c, o, m, task)


@dataclass(frozen=True)
class LossWeights:
    contact: float = 5.0
    normal: float = .5
    smooth: float = .1
    collision: float = .002


def contact_geometry_losses(path_metric: Tensor, task: Task, alpha_bar: Tensor) -> dict[str, Tensor]:
    """Plane/cylinder contact of the actual body tool, for arbitrary saved extrinsics."""
    features = task["features"].float()
    body_rotation = rotation_6d_to_matrix(path_metric[..., 3:9].float())
    tip = path_metric[..., :3].float() + torch.einsum("blij,bj->bli", body_rotation, features[:, 22:25])
    frame = features[:, 5:14].reshape(-1, 3, 3).transpose(1, 2)
    local = torch.einsum("bli,bij->blj", tip - features[:, None, 2:5], frame)
    radial_distance = torch.linalg.vector_norm(local[..., :2], dim=-1)
    radial_normal_local = torch.cat((local[..., :2] / radial_distance.clamp_min(1e-6)[..., None], torch.zeros_like(local[..., :1])), dim=-1)
    radial_normal = torch.einsum("bij,blj->bli", frame, radial_normal_local)
    cylinder = features[:, 1] > .5
    normal = torch.where(cylinder[:, None, None], radial_normal, frame[:, None, :, 2])
    error = torch.where(cylinder[:, None], radial_distance - features[:, None, 14], local[..., 2])
    tool_rotation = quaternion_wxyz_to_matrix_torch(features[:, 25:29])
    tool_axis = torch.einsum("blij,bj->bli", body_rotation, tool_rotation[:, :, 2])
    weight = alpha_bar.reshape(len(features), -1)[:, 0].float()
    distance_per_path = error[:, 1:-1].square().mean(dim=1)
    normal_per_path = (tool_axis[:, 1:-1] + normal[:, 1:-1]).square().sum(dim=-1).mean(dim=1)
    # Physical tool-position second differences avoid the atan2 branch cut of cylinder UV.
    smooth_per_path = torch.diff(tip, n=2, dim=1).square().mean(dim=(1, 2))
    return dict(contact=(weight * distance_per_path).mean(), normal=(weight * normal_per_path).mean(),
                smooth=(weight * smooth_per_path).mean())


def contact_training_loss(model: ContactAdapter, batch: ContactBatch, schedule: DiffusionSchedule,
                          weights: LossWeights = LossWeights(), *, noise: Tensor | None = None,
                          timesteps: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
    clean, conditions = batch.clean, batch.conditions
    if timesteps is None:
        timesteps = torch.randint(len(schedule.betas), (len(clean),), device=clean.device)
    if noise is None:
        noise = torch.randn_like(clean)
    noise = noise.clone()
    noise[:, 0] = noise[:, -1] = 0
    alpha = schedule.alpha_bars[timesteps].reshape(-1, 1, 1)
    noisy = hard_clamp_endpoints(alpha.sqrt() * clean + (1 - alpha).sqrt() * noise, conditions)
    prediction = model(noisy, timesteps, conditions, batch.obstacles, batch.obstacle_mask, task=batch.task)
    target = alpha.sqrt() * noise - (1 - alpha).sqrt() * clean
    diffusion = (prediction[:, 1:-1] - target[:, 1:-1]).float().square().mean()
    x0 = hard_clamp_endpoints((alpha.sqrt() * noisy - (1 - alpha).sqrt() * prediction).float().clamp(-6, 6), conditions)
    physical = x0 * model.path_std + model.path_mean
    geometry = contact_geometry_losses(physical, batch.task, alpha)
    # Supports are task-visible physical OBBs and should also enter the existing
    # differentiable sphere loss. Target plane/cylinder solids are audited by COAL.
    obstacles = torch.cat((batch.obstacles, batch.task["support_tokens"]), dim=1)
    masks = torch.cat((batch.obstacle_mask, batch.task["support_mask"]), dim=1)
    collision, _ = collision_auxiliary_loss(x0, model.path_mean[None, None], model.path_std[None, None],
                                            obstacles, masks, alpha, waypoint_stride=4)
    total = diffusion + weights.contact * geometry["contact"] + weights.normal * geometry["normal"] + weights.smooth * geometry["smooth"] + weights.collision * collision
    return total, dict(total=total, diffusion=diffusion, collision=collision, **geometry)


def fixed_validation_noise(count: int, sequence_length: int = 128) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(VALIDATION_NOISE_SEED)
    timesteps = torch.randint(100, (count,), generator=generator)
    noise = torch.randn(count, sequence_length, 9, generator=generator)
    noise[:, 0] = noise[:, -1] = 0
    return timesteps, noise


@torch.no_grad()
def validate(model: ContactAdapter, data: ContactDataset, indices: np.ndarray,
             schedule: DiffusionSchedule, batch_size: int, noise_bank: tuple[Tensor, Tensor]) -> dict[str, float]:
    model.eval()
    totals = {name: 0.0 for name in ("total", "diffusion", "contact", "normal", "smooth", "collision")}
    timesteps, noises = noise_bank
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        batch = prepare_contact_batch(data, indices[start:stop], model.path_mean.device, model.path_mean, model.path_std)
        _, metrics = contact_training_loss(model, batch, schedule, LossWeights(),
                         noise=noises[start:stop].to(model.path_mean.device),
                         timesteps=timesteps[start:stop].to(model.path_mean.device))
        for name in totals:
            totals[name] += float(metrics[name]) * (stop - start)
    model.train()
    return {name: value / len(indices) for name, value in totals.items()}


class AdapterEMA:
    def __init__(self, model: ContactAdapter, decay: float = .995) -> None:
        self.decay = decay
        self.values = {name: value.detach().clone() for name, value in model.trainable_state_dict().items()}

    @torch.no_grad()
    def update(self, model: ContactAdapter) -> None:
        for name, value in model.trainable_state_dict().items():
            self.values[name].lerp_(value.detach(), 1 - self.decay)

    def cpu(self) -> dict[str, Tensor]:
        return {name: value.detach().cpu() for name, value in self.values.items()}


def scaled_optimizer_step(scaler: Any, optimizer: torch.optim.Optimizer,
                          parameters: list[nn.Parameter]) -> bool:
    """Return whether an optimizer update happened, excluding AMP overflow skips."""
    scaler.unscale_(optimizer)
    gradient_norm = nn.utils.clip_grad_norm_(parameters, 1.)
    if not scaler.is_enabled() and not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("non-finite Contact gradients")
    old_scale = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    return float(scaler.get_scale()) >= old_scale


def source_hashes() -> dict[str, str]:
    directory = Path(__file__).parent
    return {name: sha256_file(directory / name) for name in
            ("contact_adapter.py", "train_contact_adapter.py", "model.py", "geometry.py", "collision_loss.py", "training.py")}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base", type=Path, default=DEFAULT_BASE)
    result.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--train-size", type=int, choices=(500, 1000, 2000, 4000, 8000), default=8000)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--data-subset-seed", type=int, default=DATA_SUBSET_SEED)
    result.add_argument("--steps", type=int, default=8000)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--validation-batch-size", type=int, default=64)
    result.add_argument("--validation-every", "--validation-interval", dest="validation_every", type=int, default=500)
    result.add_argument("--checkpoint-every", type=int, default=100)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--weight-decay", type=float, default=1e-4)
    result.add_argument("--warmup-steps", type=int, default=500)
    result.add_argument("--ema-decay", type=float, default=.995)
    result.add_argument("--contact-weight", type=float, default=5.)
    result.add_argument("--normal-weight", type=float, default=.5)
    result.add_argument("--smooth-weight", type=float, default=.1)
    result.add_argument("--collision-weight", type=float, default=.002)
    result.add_argument("--task-conditioning", choices=("geometry", "none"), default="geometry")
    result.add_argument("--device", default="cuda")
    result.add_argument("--disable-amp", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--stop-after", type=int, default=0, help="Deliberate resumable stop after this many updates in this invocation; 0 runs the planned budget")
    return result


def train(args: argparse.Namespace) -> Path:
    if min(args.steps, args.batch_size, args.validation_batch_size, args.validation_every, args.checkpoint_every) <= 0:
        raise ValueError("steps and batch/checkpoint intervals must be positive")
    weights = LossWeights(args.contact_weight, args.normal_weight, args.smooth_weight, args.collision_weight)
    if min(asdict(weights).values()) < 0 or args.learning_rate <= 0 or not 0 <= args.ema_decay < 1:
        raise ValueError("invalid optimizer/loss/EMA configuration")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    args.output = args.output.resolve()
    config_path = args.output / "resolved_training_config.json"
    if config_path.exists() and not args.resume:
        raise RuntimeError("output already contains a run; use --resume or a new directory")
    if args.resume and not (args.output / "last.pt").is_file():
        raise RuntimeError("--resume requires an existing last.pt")
    backbone, base = load_frozen_base(args.base, device)
    data = ContactDataset.load(args.dataset, base)
    selected, subset = select_training_indices(data, args.train_size, args.data_subset_seed)
    validation_indices = np.flatnonzero(data.core.split_codes == 1)
    if not len(validation_indices):
        raise RuntimeError("validation split is empty")
    config = ContactAdapterConfig(task_conditioning=args.task_conditioning)
    model = ContactAdapter(backbone, torch.as_tensor(base["path_mean"]), torch.as_tensor(base["path_std"]), config).to(device)
    initial_base_hash = frozen_state_hash(backbone)
    immutable = dict(schema_version=ADAPTER_SCHEMA, base_path=base["base_path"], base_sha256=base["base_sha256"],
                     **data.provenance, source_hashes=source_hashes(), adapter_config=asdict(config),
                     subset_sha256=json_hash(subset), train_size=args.train_size, seed=args.seed,
                     data_subset_seed=args.data_subset_seed, training_steps=args.steps, batch_size=args.batch_size,
                     validation_batch_size=args.validation_batch_size, validation_every=args.validation_every,
                     checkpoint_every=args.checkpoint_every, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                     warmup_steps=args.warmup_steps, ema_decay=args.ema_decay, loss_weights=asdict(weights),
                     validation_weights=asdict(LossWeights()), validation_noise_seed=VALIDATION_NOISE_SEED,
                     validation_count=len(validation_indices), reverse_augmentation=True, device=str(device),
                     amp=device.type == "cuda" and not args.disable_amp,
                     smooth_protocol="canonical_tool_position_second_difference_v001",
                     auxiliary_weight="alpha_bar", collision_protocol="42_body_spheres_vs_base_and_task_support_OBBs_stride4",
                     target_solid_collision="not_in_surrogate_loss;full_URDF_COAL_evaluation_required",
                     geometry_none_protocol="same_adapter_architecture;only_start_goal_tokens_active;real_environment_and_geometry_losses_retained",
                     optimizer_budget="successful_updates;AMP_overflow_skips_retried_and_counted",
                     batch_protocol="full_batches_cross_shuffle_boundary_v001",
                     library_versions=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, cuda=torch.version.cuda))
    contract_hash = json_hash(immutable)
    args.output.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous["contract_sha256"] != contract_hash:
            raise RuntimeError("resume contract mismatch (source, data, Base, subset or training configuration changed)")
        if json_hash(json.loads((args.output / "training_subset.json").read_text())) != immutable["subset_sha256"]:
            raise RuntimeError("resume subset manifest does not match the frozen run contract")
    else:
        write_json(config_path, immutable | dict(contract_sha256=contract_hash, created_at_utc=datetime.now(timezone.utc).isoformat()))
        write_json(args.output / "training_subset.json", subset)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup = min(args.warmup_steps, max(1, args.steps // 10))
    def lr_multiplier(step: int) -> float:
        if step < warmup:
            return max((step + 1) / warmup, 1e-3)
        progress = (step - warmup) / max(args.steps - warmup, 1)
        return .05 + .95 * .5 * (1 + math.cos(math.pi * min(progress, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    ema = AdapterEMA(model, args.ema_decay)
    sampler = ContactBatchSampler(2 * len(selected), args.batch_size, args.seed)
    schedule = DiffusionSchedule.cosine(100, device)
    noise_bank = fixed_validation_noise(len(validation_indices))
    use_amp = immutable["amp"]
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_context = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_amp else (lambda: nullcontext())
    history: list[dict[str, Any]] = []
    best_score, best_step, start_step = None, None, 1
    rolling = {name: 0.0 for name in ("total", "diffusion", "contact", "normal", "smooth", "collision")}
    rolling_samples = 0
    elapsed_previous = 0.0
    attempted_updates = skipped_updates = 0
    if args.resume:
        state = torch.load(args.output / "last.pt", map_location=device, weights_only=False)
        if state["contract_sha256"] != contract_hash or state["frozen_base_state_sha256"] != initial_base_hash:
            raise RuntimeError("resume checkpoint does not match its frozen run contract")
        model.load_trainable_state(state["adapter_state"])
        ema.values = {name: value.to(device) for name, value in state["ema_state"].items()}
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        sampler.load_state_dict(state["sampler_state"])
        history, best_score, best_step = state["history"], state["best_validation_score"], state["best_step"]
        rolling, rolling_samples = state["rolling_metrics"], state["rolling_samples"]
        elapsed_previous = state["elapsed_s"]
        attempted_updates, skipped_updates = state["attempted_updates"], state["skipped_updates"]
        start_step = state["step"] + 1
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    started = time.monotonic()
    final_step = start_step - 1
    def elapsed() -> float:
        return elapsed_previous + time.monotonic() - started
    def payload() -> dict[str, Any]:
        return dict(schema_version=ADAPTER_SCHEMA, adapter_config=asdict(config),
                    adapter_state={name: value.detach().cpu() for name, value in model.trainable_state_dict().items()},
                    ema_state=ema.cpu(), **base, **data.provenance, contract_sha256=contract_hash,
                    frozen_base_state_sha256=initial_base_hash, source_hashes=immutable["source_hashes"],
                    subset_sha256=immutable["subset_sha256"], config=immutable,
                    step=final_step, training_step=final_step, best_step=best_step,
                    best_validation_score=best_score, history=history, elapsed_s=elapsed(),
                    attempted_updates=attempted_updates, skipped_updates=skipped_updates)
    def status(status_name: str, **extra: Any) -> None:
        write_json(args.output / "run_status.json", dict(status=status_name, step=final_step,
                   planned_steps=args.steps, train_experts=len(selected), validation_experts=len(validation_indices),
                   trainable_parameters=sum(p.numel() for p in trainable), frozen_parameters=sum(p.numel() for p in backbone.parameters()),
                   best_step=best_step, best_validation_score=best_score, elapsed_s=elapsed(),
                   attempted_updates=attempted_updates, skipped_updates=skipped_updates,
                   successful_sample_presentations=final_step * args.batch_size,
                   contract_sha256=contract_hash, updated_at_utc=datetime.now(timezone.utc).isoformat(), **extra))
    def save_resume() -> None:
        state = payload() | dict(optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
                   scaler_state=scaler.state_dict(), sampler_state=sampler.state_dict(), python_rng=random.getstate(),
                   numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                   rolling_metrics=rolling, rolling_samples=rolling_samples)
        atomic_torch_save(args.output / "last.pt", state)
    status("running")
    print(f"contact_adapter seed={args.seed} train={len(selected)} maps={subset['map_count']} trainable={sum(p.numel() for p in trainable)} step={final_step}/{args.steps}", flush=True)
    try:
        for step in range(start_step, args.steps + 1):
            for attempt in range(16):
                local = sampler.next_indices()
                batch = prepare_contact_batch(data, selected[local % len(selected)], device, model.path_mean, model.path_std, local >= len(selected))
                optimizer.zero_grad(set_to_none=True)
                model.train()
                with amp_context():
                    total, metrics = contact_training_loss(model, batch, schedule, weights)
                if not bool(torch.isfinite(total)):
                    raise RuntimeError("non-finite Contact training loss")
                scaler.scale(total).backward()
                if any(parameter.grad is not None for parameter in backbone.parameters()):
                    raise RuntimeError("frozen Base received gradients")
                attempted_updates += 1
                if scaled_optimizer_step(scaler, optimizer, trainable):
                    break
                skipped_updates += 1
            else:
                raise RuntimeError("16 consecutive AMP overflow skips; planned optimizer update did not happen")
            scheduler.step()
            ema.update(model)
            final_step = step
            for name in rolling:
                rolling[name] += float(metrics[name].detach()) * len(local)
            rolling_samples += len(local)
            if step % args.validation_every == 0 or step == args.steps:
                live = {name: value.detach().clone() for name, value in model.trainable_state_dict().items()}
                model.load_trainable_state(ema.values)
                try:
                    validation = validate(model, data, validation_indices, schedule, args.validation_batch_size, noise_bank)
                finally:
                    model.load_trainable_state(live)
                if not all(math.isfinite(value) for value in validation.values()):
                    raise RuntimeError("non-finite common validation score")
                if frozen_state_hash(backbone) != initial_base_hash:
                    raise RuntimeError("frozen Base weights changed")
                entry = dict(step=step, train={name: value / rolling_samples for name, value in rolling.items()},
                             validation=validation, elapsed_s=elapsed(), learning_rate=optimizer.param_groups[0]["lr"])
                history.append(entry)
                if best_score is None or validation["total"] < best_score:
                    best_score, best_step = validation["total"], step
                    best = payload()
                    best["adapter_state"] = ema.cpu()
                    atomic_torch_save(args.output / "best.pt", best)
                write_json(args.output / "training_history.json", dict(history=history, selection="minimum_common_fixed_noise_validation_total"))
                print(f"step={step}/{args.steps} train={entry['train']['total']:.6f} validation={validation['total']:.6f} best={best_score:.6f} elapsed={elapsed():.1f}s", flush=True)
                rolling = {name: 0.0 for name in rolling}
                rolling_samples = 0
            stopping = bool(args.stop_after and step - start_step + 1 >= args.stop_after)
            if step % args.checkpoint_every == 0 or step % args.validation_every == 0 or step == args.steps or stopping:
                save_resume()
                status("running")
            if stopping and step < args.steps:
                status("stopped_resumable", reason="explicit_stop_after", checkpoint=str(args.output / "last.pt"))
                return args.output
        if frozen_state_hash(backbone) != initial_base_hash:
            raise RuntimeError("frozen Base weights changed")
        if final_step != args.steps or not (args.output / "best.pt").is_file():
            raise RuntimeError("run cannot be marked complete without its planned updates and selected checkpoint")
        status("complete", selected_checkpoint=str(args.output / "best.pt"), selected_checkpoint_sha256=sha256_file(args.output / "best.pt"), frozen_base_unchanged=True)
        return args.output
    except BaseException as exc:
        status("failed" if not isinstance(exc, KeyboardInterrupt) else "interrupted", error_type=type(exc).__name__, error=str(exc),
               last_checkpoint_step=(torch.load(args.output / "last.pt", map_location="cpu", weights_only=False)["step"] if (args.output / "last.pt").is_file() else None))
        raise


def main(argv: list[str] | None = None) -> None:
    train(parser().parse_args(argv))


if __name__ == "__main__":
    main()
