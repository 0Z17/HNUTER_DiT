"""Shared-latent v003/contact sampling without reading any expert or oracle."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .contact_adapter import load_contact_adapter, load_frozen_base, prepare_contact_condition_batch
from .geometry import decanonicalize_pose9_batch
from .model import DiffusionSchedule, hard_clamp_endpoints

INFERENCE_VERSION = 'contact_explicit_noise_ddim_v001'
LATENT_SEED = 20261011


def latent_seed(condition_id: str, candidate_index: int, seed: int = LATENT_SEED) -> int:
    # Scenario and model deliberately do not enter this key. Obstacle shift,
    # every data budget and projection on/off reuse the identical initial noise.
    key = f'{INFERENCE_VERSION}/{seed}/{condition_id}/{candidate_index}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % (2**63-1)


def latent_noise(condition_ids: Sequence[str], candidate_indices: Sequence[int],
                 seed: int = LATENT_SEED) -> Tensor:
    if len(condition_ids) != len(candidate_indices):
        raise ValueError('one candidate index is required per condition')
    return torch.stack([torch.randn((128, 9), generator=torch.Generator().manual_seed(
        latent_seed(str(case), int(index), seed)))
        for case, index in zip(condition_ids, candidate_indices)])


def ddim_with_noise(model: torch.nn.Module, conditions: Tensor, obstacles: Tensor,
                    obstacle_mask: Tensor, task: dict[str, Tensor] | None,
                    noise: Tensor, schedule: DiffusionSchedule, steps: int = 25,
                    clip_x0: float = 4.) -> Tensor:
    """The Base deterministic velocity DDIM equations, with explicit latents."""
    if noise.shape != (len(conditions), 128, 9) or not 1 <= steps <= len(schedule.betas):
        raise ValueError('invalid DDIM noise or sampling budget')
    path = hard_clamp_endpoints(noise.clone(), conditions)
    indices = torch.linspace(len(schedule.betas)-1, 0, steps, device=path.device).round().long().unique_consecutive()
    for index, value in enumerate(indices):
        timestep = torch.full((len(path),), int(value), dtype=torch.long, device=path.device)
        velocity = (model(path, timestep, conditions, obstacles, obstacle_mask, task=task)
                    if task is not None else model(path, timestep, conditions, obstacles, obstacle_mask))
        alpha = schedule.alpha_bars[value]
        clean = hard_clamp_endpoints((alpha.sqrt()*path - (1-alpha).sqrt()*velocity).clamp(-clip_x0, clip_x0), conditions)
        epsilon = (path-alpha.sqrt()*clean)/(1-alpha).sqrt().clamp_min(1e-6)
        if index == len(indices)-1:
            path = clean
        else:
            previous = schedule.alpha_bars[indices[index+1]]
            path = hard_clamp_endpoints(previous.sqrt()*clean + (1-previous).sqrt()*epsilon, conditions)
    return hard_clamp_endpoints(path, conditions)


class ContactInferenceRuntime:
    def __init__(self, base: str | Path, adapter: str | Path | None = None,
                 device: str = 'cuda') -> None:
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            torch.set_float32_matmul_precision('high')
        self.adapter = adapter is not None
        if self.adapter:
            self.model, self.metadata = load_contact_adapter(base, adapter, self.device)
        else:
            self.model, self.metadata = load_frozen_base(base, self.device)
        self.mean = torch.as_tensor(self.metadata['path_mean'], device=self.device).float()
        self.std = torch.as_tensor(self.metadata['path_std'], device=self.device).float()
        self.schedule = DiffusionSchedule.cosine(100, self.device)

    @torch.inference_mode()
    def sample(self, inputs: dict[str, np.ndarray], rows: Sequence[int],
               candidate_indices: Sequence[int], *, sampling_steps: int = 25) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        def tensor(key: str, dtype: torch.dtype = torch.float32) -> Tensor:
            return torch.as_tensor(inputs[key][rows], dtype=dtype, device=self.device)
        world = tensor('conditions')
        features = tensor('surface_features')
        feature_mask = tensor('surface_mask', torch.bool)
        if feature_mask.ndim == 1:
            feature_mask = feature_mask[:, None].expand_as(features)
        optional = {}
        if 'forbidden_regions' in inputs:
            optional = dict(forbidden_regions=tensor('forbidden_regions'), forbidden_mask=tensor('forbidden_mask', torch.bool))
        context, obstacles, mask, task = prepare_contact_condition_batch(
            world, tensor('obstacles'), tensor('obstacle_mask', torch.bool),
            features, feature_mask, tensor('support_tokens'), tensor('support_mask', torch.bool),
            self.mean, self.std, **optional)
        noise = latent_noise(inputs['condition_id'][rows], candidate_indices).to(self.device)
        normalized = ddim_with_noise(self.model, context, obstacles, mask, task if self.adapter else None,
                                     noise, self.schedule, sampling_steps)
        paths = decanonicalize_pose9_batch(normalized*self.std+self.mean, world)
        # A second exact clamp removes canonical transform/serialization rounding.
        paths[:, 0] = world[:, :9]
        paths[:, -1] = world[:, 9:]
        return paths.cpu().numpy()
