"""Task cross-attention residuals on an immutable canonical v003 DiT prior.

All public task tensors contain requested geometry, never an expert route or
mode. Coordinates passed to the model are canonical metres/body extrinsics.
Fixed physical scales below are architecture constants, not fitted statistics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .geometry import (
    CANONICAL_FRAME_VERSION, _rotate_vectors_world_to_canonical,
    _yaw_components_torch, canonicalize_condition_batch,
    canonicalize_obstacles_batch,
)
from .model import ConditionalDiT, build_model, timestep_embedding


ADAPTER_SCHEMA = "hnuter_contact_multifamily_adapter_v001"
TASK_FEATURE_DIM = 32
Task = Mapping[str, Tensor]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_contact_task(
    features_world: Tensor, world_conditions: Tensor, feature_mask: Tensor,
    support_tokens_world: Tensor, support_mask: Tensor,
    forbidden_regions: Tensor | None = None,
    forbidden_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Jointly transform task points/axes/OBBs; body and intrinsic fields stay fixed.

    This has no dependency on expert paths and is also the inference API. The
    frame is column-major in feature slots 5:14. Padded supports are zeroed so
    arbitrary values under a false mask cannot leak into a model prediction.
    """
    batch = len(world_conditions)
    if features_world.shape != (batch, 32) or feature_mask.shape != (batch, 32):
        raise ValueError("surface features and feature mask must be [batch,32]")
    if support_tokens_world.shape != (batch, 8, 10) or support_mask.shape != (batch, 8):
        raise ValueError("task supports must be [batch,8,10] with [batch,8] mask")
    features = features_world.float().clone()
    start, cosine, sine = _yaw_components_torch(world_conditions.float())
    features[:, 2:5] = _rotate_vectors_world_to_canonical(features[:, 2:5] - start, cosine, sine)
    for offset in (5, 8, 11):
        features[:, offset:offset + 3] = _rotate_vectors_world_to_canonical(
            features[:, offset:offset + 3], cosine, sine)
    supports = canonicalize_obstacles_batch(support_tokens_world.float(), world_conditions.float())
    supports = supports.masked_fill(~support_mask.bool()[..., None], 0)
    result = dict(features=features, feature_mask=feature_mask.bool(),
                  support_tokens=supports, support_mask=support_mask.bool())
    if (forbidden_regions is None) != (forbidden_mask is None):
        raise ValueError("forbidden regions and mask must be supplied together")
    if forbidden_regions is not None:
        if forbidden_regions.shape != (batch, 8, 4) or forbidden_mask.shape != (batch, 8):
            raise ValueError("intrinsic forbidden regions must be [batch,8,4]")
        result.update(forbidden_regions=forbidden_regions.float(), forbidden_mask=forbidden_mask.bool())
    return result


def prepare_contact_condition_batch(
    world_conditions: Tensor, obstacles_world: Tensor, obstacle_mask: Tensor,
    features_world: Tensor, feature_mask: Tensor,
    support_tokens_world: Tensor, support_mask: Tensor,
    path_mean: Tensor, path_std: Tensor,
    forbidden_regions: Tensor | None = None, forbidden_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
    """Inference conditioning from a benchmark batch, without expert labels."""
    canonical = canonicalize_condition_batch(world_conditions.float())
    condition = (canonical - path_mean.repeat(2)) / path_std.repeat(2)
    obstacles = canonicalize_obstacles_batch(obstacles_world.float(), world_conditions.float())
    obstacles = obstacles.masked_fill(~obstacle_mask.bool()[..., None], 0)
    task = canonicalize_contact_task(features_world, world_conditions, feature_mask,
                                     support_tokens_world, support_mask,
                                     forbidden_regions, forbidden_mask)
    return condition, obstacles, obstacle_mask.bool(), task


@dataclass(frozen=True)
class ContactAdapterConfig:
    blocks: tuple[int, ...] = (0, 3, 7)
    bottleneck: int = 64
    task_conditioning: str = "geometry"
    physical_length_scale_m: float = 2.0
    schema_version: str = ADAPTER_SCHEMA

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContactAdapterConfig":
        value = dict(value)
        value["blocks"] = tuple(value.get("blocks", (0, 3, 7)))
        return cls(**value)


def _encoder(width: int, dimension: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(width, dimension), nn.SiLU(), nn.Linear(dimension, dimension))


class ContactGeometryEncoder(nn.Module):
    """Four semantic tokens, two endpoint tokens, eight supports and eight UV regions."""
    groups = ((0, 14), (14, 22), (22, 29), (29, 32))

    def __init__(self, dimension: int, scale_m: float) -> None:
        super().__init__()
        self.scale_m = scale_m
        self.encoders = nn.ModuleList([_encoder(2 * (hi - lo), dimension) for lo, hi in self.groups])
        self.endpoint_encoder = _encoder(9, dimension)
        self.support_encoder = _encoder(10, dimension)
        self.forbidden_encoder = _encoder(4, dimension)
        self.token_types = nn.Parameter(torch.randn(8, dimension) * .02)
        scale = torch.ones(32)
        scale[2:5] = scale[14:20] = scale[22:25] = scale_m
        scale[20:22] = torch.pi
        scale[29], scale[30], scale[31] = 1e-3, 1e-2, .05
        self.register_buffer("feature_scales", scale, persistent=False)

    def forward(self, task: Task, metric_context: Tensor, conditioning: str) -> tuple[Tensor, Tensor]:
        features = task["features"]
        masks = task["feature_mask"].bool()
        batch = len(metric_context)
        if features.shape != (batch, 32) or masks.shape != (batch, 32):
            raise ValueError("canonical task features must be [batch,32]")
        if conditioning == "none":
            masks = torch.zeros_like(masks)
        # torch.where also removes NaN padding rather than multiplying NaN by zero.
        scaled = torch.where(masks, features / self.feature_scales, torch.zeros_like(features))
        tokens, active = [], []
        for index, (lo, hi) in enumerate(self.groups):
            value = torch.cat((scaled[:, lo:hi], masks[:, lo:hi].to(features.dtype)), dim=-1)
            tokens.append(self.encoders[index](value) + self.token_types[index])
            active.append(masks[:, lo:hi].any(dim=-1))
        endpoints = metric_context.reshape(batch, 2, 9).clone()
        endpoints[..., :3] = endpoints[..., :3] / self.scale_m
        for index in range(2):
            tokens.append(self.endpoint_encoder(endpoints[:, index]) + self.token_types[4 + index])
            active.append(torch.ones(batch, dtype=torch.bool, device=features.device))
        global_tokens = torch.stack(tokens, dim=1)
        global_mask = torch.stack(active, dim=1)
        supports = task["support_tokens"]
        support_mask = task["support_mask"].bool()
        if supports.shape != (batch, 8, 10) or support_mask.shape != (batch, 8):
            raise ValueError("canonical supports must be [batch,8,10]")
        if conditioning == "none":
            support_mask = torch.zeros_like(support_mask)
        supports = torch.where(support_mask[..., None], supports, torch.zeros_like(supports)).clone()
        supports[..., :6] = supports[..., :6] / self.scale_m
        support_tokens = self.support_encoder(supports) + self.token_types[6]
        forbidden = task.get("forbidden_regions", features.new_zeros((batch, 8, 4)))
        forbidden_mask = task.get("forbidden_mask", torch.zeros((batch, 8), dtype=torch.bool, device=features.device)).bool()
        if forbidden.shape != (batch, 8, 4) or forbidden_mask.shape != (batch, 8):
            raise ValueError("forbidden regions must be [batch,8,4]")
        if conditioning == "none":
            forbidden_mask = torch.zeros_like(forbidden_mask)
        forbidden = torch.where(forbidden_mask[..., None], forbidden, torch.zeros_like(forbidden))
        forbidden_tokens = self.forbidden_encoder(forbidden / self.scale_m) + self.token_types[7]
        return (torch.cat((global_tokens, support_tokens, forbidden_tokens), dim=1),
                torch.cat((global_mask, support_mask, forbidden_mask), dim=1))


class ContactResidual(nn.Module):
    def __init__(self, dimension: int, heads: int, bottleneck: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.down = nn.Linear(dimension, bottleneck)
        self.up = nn.Linear(bottleneck, dimension)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: Tensor, tokens: Tensor, mask: Tensor) -> Tensor:
        residual = self.attention(self.norm(hidden), tokens, tokens,
                                  key_padding_mask=~mask, need_weights=False)[0]
        return self.up(torch.nn.functional.silu(self.down(residual)))


class ContactAdapter(nn.Module):
    def __init__(self, backbone: ConditionalDiT, path_mean: Tensor, path_std: Tensor,
                 config: ContactAdapterConfig | None = None) -> None:
        super().__init__()
        self.config = config or ContactAdapterConfig()
        if self.config.schema_version != ADAPTER_SCHEMA or self.config.task_conditioning not in {"geometry", "none"}:
            raise ValueError("unsupported Contact adapter schema or conditioning")
        if self.config.bottleneck <= 0 or self.config.physical_length_scale_m <= 0:
            raise ValueError("adapter bottleneck and physical scale must be positive")
        if backbone.model_type != "dit_cross" or backbone.relative_geometry_attention:
            raise ValueError("Contact adapter requires the v003 dit_cross architecture")
        if not self.config.blocks or len(set(self.config.blocks)) != len(self.config.blocks) or any(i < 0 or i >= len(backbone.blocks) for i in self.config.blocks):
            raise ValueError("adapter block indices must be unique valid Base blocks")
        self.backbone = backbone
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        backbone.eval()
        self.register_buffer("path_mean", torch.as_tensor(path_mean).float().clone(), persistent=False)
        self.register_buffer("path_std", torch.as_tensor(path_std).float().clone(), persistent=False)
        self.geometry_encoder = ContactGeometryEncoder(backbone.dimension, self.config.physical_length_scale_m)
        heads = backbone.blocks[0].self_attention.num_heads
        self.adapters = nn.ModuleDict({str(i): ContactResidual(backbone.dimension, heads, self.config.bottleneck)
                                      for i in self.config.blocks})
        self.model_type = "dit_contact_adapter"

    def train(self, mode: bool = True) -> "ContactAdapter":
        super().train(mode)
        self.backbone.eval()
        return self

    def trainable_state_dict(self) -> dict[str, Tensor]:
        return {name: value for name, value in self.state_dict().items()
                if name.startswith(("geometry_encoder.", "adapters."))}

    def load_trainable_state(self, state: Mapping[str, Tensor]) -> None:
        expected = self.trainable_state_dict()
        if set(expected) != set(state):
            raise RuntimeError("adapter state keys do not match the declared architecture")
        full = self.state_dict()
        full.update(state)
        self.load_state_dict(full, strict=True)

    def forward(self, path: Tensor, timestep: Tensor, context: Tensor,
                obstacles: Tensor | None = None, obstacle_mask: Tensor | None = None,
                task: Task | None = None, *, adapter_enabled: bool = True) -> Tensor:
        if not adapter_enabled:
            return self.backbone(path, timestep, context, obstacles, obstacle_mask)
        if task is None:
            raise ValueError("enabled Contact adapter requires a canonical task payload")
        metric_context = context * self.path_std.repeat(2) + self.path_mean.repeat(2)
        tokens, token_mask = self.geometry_encoder(task, metric_context, self.config.task_conditioning)
        condition = self.backbone.time_mlp(timestep_embedding(timestep, 64)) + self.backbone.context_mlp(context)
        hidden = self.backbone.input(path) + self.backbone.position_embedding[:, :path.shape[1]]
        environment, environment_mask = self.backbone.encode_environment(path, obstacles, obstacle_mask)
        for index, block in enumerate(self.backbone.blocks):
            # Frozen weights must remain in the autograd graph after task residuals.
            hidden = block(hidden, condition, environment, environment_mask)
            key = str(index)
            if key in self.adapters:
                hidden = hidden + self.adapters[key](hidden, tokens, token_mask)
        shift, scale = self.backbone.final_modulation(condition).chunk(2, dim=-1)
        hidden = self.backbone.final_norm(hidden) * (1 + scale[:, None]) + shift[:, None]
        return self.backbone.output(hidden)


def load_frozen_base(base_path: str | Path, device: str | torch.device = "cpu",
                     expected_sha256: str | None = None) -> tuple[ConditionalDiT, dict[str, Any]]:
    base_path = Path(base_path).resolve()
    digest = sha256_file(base_path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError("frozen Base checkpoint SHA-256 mismatch")
    payload = torch.load(base_path, map_location="cpu", weights_only=False)
    if payload.get("model_type") != "dit_cross" or payload.get("canonical_frame") != CANONICAL_FRAME_VERSION:
        raise RuntimeError("a canonical v003 dit_cross Base checkpoint is required")
    if payload.get("prediction_type") != "velocity" or payload.get("diffusion_steps") != 100:
        raise RuntimeError("unsupported Base prediction parameterization")
    model = build_model(payload["model_type"], **payload["architecture"])
    model.load_state_dict(payload["model_state"], strict=True)
    model.configure_environment_normalization(payload["environment_feature_mean"], payload["environment_feature_std"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device).eval()
    metadata = {key: payload[key] for key in (
        "model_type", "architecture", "path_mean", "path_std", "environment_feature_mean",
        "environment_feature_std", "canonical_frame", "diffusion_steps", "prediction_type")}
    metadata.update(base_path=str(base_path), base_sha256=digest)
    return model, metadata


def load_contact_adapter(base_path: str | Path, adapter_checkpoint: str | Path,
                         device: str | torch.device = "cpu", use_ema: bool = True
                         ) -> tuple[ContactAdapter, dict[str, Any]]:
    checkpoint = torch.load(adapter_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != ADAPTER_SCHEMA:
        raise RuntimeError("unsupported Contact adapter checkpoint")
    base, metadata = load_frozen_base(base_path, device, checkpoint["base_sha256"])
    for key in ("path_mean", "path_std", "environment_feature_mean", "environment_feature_std"):
        if not torch.equal(torch.as_tensor(metadata[key]), torch.as_tensor(checkpoint[key])):
            raise RuntimeError(f"adapter checkpoint {key} differs from frozen Base")
    model = ContactAdapter(base, torch.as_tensor(metadata["path_mean"]), torch.as_tensor(metadata["path_std"]),
                           ContactAdapterConfig.from_dict(checkpoint["adapter_config"])).to(device)
    state = checkpoint.get("ema_state") if use_ema else checkpoint.get("adapter_state")
    if state is None:
        state = checkpoint["adapter_state"]
    model.load_trainable_state(state)
    model.eval()
    return model, checkpoint | metadata

