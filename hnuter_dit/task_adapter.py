"""Task residuals on a frozen SDFPoseDiT prior, with no geometric correction.

One instance supplies an independent task adapter (e.g. Surface or Inspection).
This pilot follows the paper's frozen-prior/zero-residual principles; it is not
its canonical v003 pose9 architecture. Inputs and normalization retain the SDF
prior's six-free-control, six-channel contract. The soft mask weights control
support only: downstream attention and spline support can propagate changes
outside a selected control, so it is not exact output-phase localization.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import torch
from torch import nn

from .model import DiffusionSchedule, timestep_embedding

ADAPTER_SCHEMA = 'frozen_sdf_task_residual_v1'
INJECTION_BLOCKS = (0, 2, 3)
DEFAULT_PHASE = (.35, .85, 1., 1., .85, .35)
BOTTLENECK = 32
TASK_TOKENS = 3


class _TaskResidual(nn.Module):
    def __init__(self, dimension, heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(dimension)
        self.hint = nn.Linear(2, dimension)
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=0., batch_first=True)
        self.down = nn.Linear(dimension, BOTTLENECK)
        self.up = nn.Linear(BOTTLENECK, dimension)
        self.activation = nn.SiLU()
        self.zero = nn.Linear(dimension, dimension)
        nn.init.zeros_(self.zero.weight)
        nn.init.zeros_(self.zero.bias)

    def forward(self, hidden, task_tokens, phase, progress):
        hints = torch.stack((progress.expand_as(phase), phase), dim=-1)
        query = self.query_norm(hidden) + self.hint(hints)
        attended, _ = self.attention(query, task_tokens, task_tokens, need_weights=False)
        residual = self.zero(self.up(self.activation(self.down(attended))))
        return phase[..., None] * residual


class FrozenSDFTaskAdapter(nn.Module):
    """Train only this instance's task branch while retaining the frozen prior.

    ``task`` is a generic canonical descriptor ``[B, task_dim]``; its geometric
    semantics and units belong to the caller. It does not replace the base's
    target condition. Use separate instances/checkpoints for independent tasks.
    ``adapter_state_dict`` excludes every base parameter and normalization
    buffer. Optimizers should receive ``adapter_parameters()`` explicitly.
    """
    def __init__(self, base, task_dim=24, output_residual=True):
        super().__init__()
        if isinstance(task_dim, bool) or not isinstance(task_dim, int) or task_dim < 1:
            raise ValueError('task_dim must be a positive integer')
        architecture = getattr(base, 'architecture', {})
        dimension, heads = architecture.get('dimension'), architecture.get('heads')
        if (not isinstance(dimension, int) or not isinstance(heads, int)
                or dimension < 1 or heads < 1 or dimension % heads
                or len(getattr(base, 'blocks', ())) != 4
                or tuple(getattr(base, 'mean', torch.empty(0)).shape) != (6, 6)
                or tuple(getattr(base, 'std', torch.empty(0)).shape) != (6, 6)
                or tuple(getattr(base, 'progress', torch.empty(0)).shape) != (6,)):
            raise ValueError('base must be a compatible four-block SDFPoseDiT with six free controls')
        if not isinstance(output_residual, bool):
            raise ValueError('output_residual must be boolean')
        self.output_residual = output_residual
        self.base = base
        self.base.eval()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.task_dim = task_dim
        self.dimension = dimension
        self.task_encoder = nn.Sequential(nn.Linear(task_dim, dimension), nn.SiLU(),
                                          nn.Linear(dimension, TASK_TOKENS * dimension))
        self.branches = nn.ModuleDict({str(index): _TaskResidual(dimension, heads)
                                      for index in INJECTION_BLOCKS})
        # A frozen LayerNorm + linear readout bounds the attainable normalized
        # diffusion velocity. This new SDF-pilot variant has a separate readout
        # for task distributions outside that range; it is not the manuscript's
        # exact block-only adapter. The last layer starts at zero.
        self.output_head = None
        if output_residual:
            self.output_head = nn.Sequential(nn.Linear(2*dimension, BOTTLENECK), nn.SiLU(),
                                             nn.Linear(BOTTLENECK, 6))
            nn.init.zeros_(self.output_head[-1].weight)
            nn.init.zeros_(self.output_head[-1].bias)
        self.register_buffer('default_phase', torch.tensor(DEFAULT_PHASE, dtype=torch.float32))
        # Match a loaded base's device/dtype immediately; callers may still use
        # the ordinary module .to(...) API on the complete wrapper afterward.
        reference = next(base.parameters())
        self.task_encoder.to(device=reference.device, dtype=reference.dtype)
        self.branches.to(device=reference.device, dtype=reference.dtype)
        if self.output_head is not None:
            self.output_head.to(device=reference.device, dtype=reference.dtype)
        self.default_phase = self.default_phase.to(device=reference.device, dtype=reference.dtype)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    @property
    def mean(self):
        return self.base.mean

    @property
    def std(self):
        return self.base.std

    def baseline(self, target):
        return self.base.baseline(target)

    def normalized(self, free, target):
        return self.base.normalized(free, target)

    def denormalized(self, value, target):
        return self.base.denormalized(value, target)

    def encode(self, fields, ages):
        # Environment memory is frozen and cacheable. Do not wrap downstream
        # denoising in no_grad: adapter gradients must traverse frozen blocks.
        with torch.no_grad():
            return self.base.encode(fields, ages)

    def adapter_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith('base.'):
                yield parameter

    def adapter_state_dict(self):
        """Independent task weights and mask only; never duplicates the base."""
        return OrderedDict((name, value.detach().clone()) for name, value in self.state_dict().items()
                           if not name.startswith('base.'))

    def load_adapter_state_dict(self, state):
        if not isinstance(state, Mapping):
            raise ValueError('adapter state must be a mapping')
        expected = self.adapter_state_dict()
        if set(state) != set(expected):
            raise ValueError('adapter state keys do not match; base state is never accepted')
        for name, reference in expected.items():
            value = state[name]
            if (not isinstance(value, torch.Tensor) or value.shape != reference.shape
                    or not bool(torch.isfinite(value).all())):
                raise ValueError('invalid adapter tensor: ' + name)
        phase = state['default_phase']
        if bool(((phase < 0) | (phase > 1)).any()):
            raise ValueError('checkpoint control-support mask must lie in [0,1]')
        # Full validation precedes any copies, so malformed inputs are atomic.
        own = self.state_dict()
        with torch.no_grad():
            for name, value in state.items():
                own[name].copy_(value)

    def adapter_configuration(self):
        return dict(schema=ADAPTER_SCHEMA, base_architecture=dict(self.base.architecture),
                    task_dim=self.task_dim, task_tokens=TASK_TOKENS,
                    injection_blocks=list(INJECTION_BLOCKS), injection='after each selected base block',
                    bottleneck=BOTTLENECK, output_residual=self.output_residual,
                    output_residual_design=('zero-initialized normalized-velocity residual from final hidden + mean task tokens; '
                        'bypasses frozen LayerNorm/readout capacity; new SDF-pilot extension, not canonical paper block-only'),
                    output_residual_phase='same soft control-support mask as hidden residuals',
                    default_phase=self.default_phase.detach().cpu().tolist(),
                    phase_semantics='soft control-support mask; not exact output localization',
                    parameter_count=sum(parameter.numel() for parameter in self.adapter_parameters()),
                    base_parameter_count=sum(parameter.numel() for parameter in self.base.parameters()),
                    frozen_base=True, frozen_normalization=True, output='raw six free pose controls',
                    projection=False)

    def _validated_phase(self, phase, x):
        value = self.default_phase if phase is None else phase
        try:
            value = torch.as_tensor(value, device=x.device, dtype=x.dtype)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError('phase must be a numeric control-support mask') from exc
        if value.shape == (6,):
            value = value[None].expand(len(x), -1)
        if value.shape != (len(x), 6) or not bool(torch.isfinite(value).all()):
            raise ValueError('phase must be finite [6] or [B,6]')
        if bool(((value < 0) | (value > 1)).any()):
            raise ValueError('phase must lie in [0,1]')
        return value

    def forward_cached(self, x, t, target, memory, task, phase=None, enabled=True):
        if not isinstance(enabled, bool):
            raise ValueError('enabled must be boolean')
        if not enabled:
            # Exact delegation includes the unchanged base's own semantics.
            return self.base.forward_cached(x, t, target, memory)
        if (x.ndim != 3 or tuple(x.shape[1:]) != (6, 6)
                or target.shape != (len(x), 6) or t.shape != (len(x),)
                or memory.ndim != 3 or memory.shape[0] != len(x)
                or memory.shape[1] < 1 or memory.shape[2] != self.dimension):
            raise ValueError('invalid cached SDF denoiser shapes')
        if (not isinstance(task, torch.Tensor) or task.shape != (len(x), self.task_dim)
                or not task.is_floating_point() or task.device != x.device
                or not bool(torch.isfinite(task).all())):
            raise ValueError('task must be a finite floating tensor [B,task_dim] on the model device')
        phase = self._validated_phase(phase, x)
        task_tokens = self.task_encoder(task.to(dtype=x.dtype)).reshape(len(x), TASK_TOKENS, self.dimension)
        context = self.base.time(timestep_embedding(t, 64)) + self.base.condition(target)
        value = self.base.input(x) + self.base.order
        observed = torch.ones(memory.shape[:2], device=value.device, dtype=torch.bool)
        for index, block in enumerate(self.base.blocks):
            value = block(value, context, memory, observed, None)
            branch = self.branches[str(index)] if index in INJECTION_BLOCKS else None
            if branch is not None:
                value = value + branch(value, task_tokens, phase, self.base.progress[None])
        prediction = self.base.output(self.base.norm(value))
        if self.output_head is not None:
            task_summary = task_tokens.mean(dim=1, keepdim=True).expand(-1, value.shape[1], -1)
            correction = self.output_head(torch.cat((value, task_summary), dim=-1))
            prediction = prediction + phase[..., None] * correction
        return prediction


@torch.inference_mode()
def sample_task(model, fields, ages, target, task, seed=23, steps=20,
                enabled=True, memory=None, phase=None):
    """Sample raw ``[B,6,6]`` free controls using the prior's unchanged DDIM.

    ``memory`` may be precomputed using the frozen ``encode``. In that case
    ``fields`` and ``ages`` may be None. No projection, endpoint editing,
    trajectory decoding, candidate filtering or retiming occurs here.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 100:
        raise ValueError('steps must be an integer from 1 through 100')
    if (not isinstance(target, torch.Tensor) or target.ndim != 2 or target.shape[1] != 6
            or not target.is_floating_point() or not bool(torch.isfinite(target).all())):
        raise ValueError('target must be finite floating [B,6]')
    if memory is None:
        if fields is None or ages is None:
            raise ValueError('fields and ages are required when memory is not supplied')
        memory = model.encode(fields, ages)
    schedule = DiffusionSchedule.cosine(100, target.device)
    generator = torch.Generator(device=target.device).manual_seed(int(seed))
    x = torch.randn((len(target), 6, 6), device=target.device, generator=generator)
    indices = torch.linspace(99, 0, steps, device=target.device).round().long()
    for index, timestep in enumerate(indices):
        velocity = model.forward_cached(x, timestep.expand(len(target)), target, memory,
                                        task, phase=phase, enabled=enabled)
        alpha = schedule.alpha_bars[timestep]
        clean = alpha.sqrt()*x - (1-alpha).sqrt()*velocity
        noise = (1-alpha).sqrt()*x + alpha.sqrt()*velocity
        if index+1 < len(indices):
            previous = schedule.alpha_bars[indices[index+1]]
            x = previous.sqrt()*clean + (1-previous).sqrt()*noise
        else:
            x = clean
    return model.denormalized(x, target)
