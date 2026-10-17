from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hnuter_dit.contact_adapter import (
    ContactAdapter, ContactAdapterConfig, canonicalize_contact_task,
    load_contact_adapter, load_frozen_base, prepare_contact_condition_batch,
    sha256_file,
)
from hnuter_dit.geometry import CANONICAL_FRAME_VERSION
from hnuter_dit.model import build_model, quaternion_wxyz_to_matrix_torch
from hnuter_dit.training import FrozenDataset
from hnuter_dit.train_contact_adapter import (
    ContactBatchSampler, ContactDataset, LossWeights, contact_geometry_losses, fixed_validation_noise,
    frozen_state_hash, parser, prepare_contact_batch, scaled_optimizer_step, select_training_indices, train,
)


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def features(batch=2):
    f = torch.zeros(batch, 32)
    f[:, 0] = 1
    f[:, 5:14] = torch.eye(3).T.reshape(-1)
    f[:, 15] = .2
    f[:, 16:22] = torch.tensor([-3, 3, -3, 3, -torch.pi, torch.pi])
    f[:, 24] = -.75
    f[:, 26] = 1  # tool rotation pi around X: its Z points down in the body.
    f[:, 29:32] = torch.tensor([1e-7, 1e-6, .01])
    mask = torch.ones_like(f, dtype=torch.bool)
    mask[:, 14] = False
    supports = torch.zeros(batch, 8, 10)
    supports[:, 0] = torch.tensor([4, 5, 6, 1, 1, 1, 1, 0, 0, 0])
    support_mask = torch.zeros(batch, 8, dtype=torch.bool)
    support_mask[:, 0] = True
    return dict(features=f, feature_mask=mask, support_tokens=supports, support_mask=support_mask)


def small_model(conditioning="geometry", sequence_length=16):
    base = build_model("dit_cross", sequence_length=sequence_length, dit_dimension=32, dit_depth=8, dit_heads=4)
    # A nonzero loaded-like readout makes freeze and adapter tests nontrivial.
    torch.nn.init.normal_(base.output.weight, std=.04)
    torch.nn.init.normal_(base.final_modulation[-1].weight, std=.02)
    return ContactAdapter(base, torch.zeros(9), torch.ones(9),
                          ContactAdapterConfig(bottleneck=8, task_conditioning=conditioning))


def input_batch(batch=2, length=16):
    path = torch.randn(batch, length, 9)
    context = torch.cat((path[:, 0], path[:, -1]), dim=-1)
    obstacles = torch.zeros(batch, 32, 10)
    obstacles[:, 0] = torch.tensor([1, 2, 3, 1, 2, 1, 1, 0, 0, 0])
    masks = torch.zeros(batch, 32, dtype=torch.bool)
    masks[:, 0] = True
    return path, torch.tensor([7, 71])[:batch], context, obstacles, masks


def activate_adapter(model):
    for residual in model.adapters.values():
        torch.nn.init.normal_(residual.up.weight, std=.02)


def test_zero_residual_matches_base_and_freezes_gradients():
    torch.manual_seed(7)
    model = small_model()
    args = input_batch()
    task = features()
    original = frozen_state_hash(model.backbone)
    expected = model.backbone(*args)
    assert torch.equal(model(*args, task=task), expected)
    assert torch.equal(model(*args, task=task, adapter_enabled=False), expected)
    assert not any(p.requires_grad for p in model.backbone.parameters())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    model(*args, task=task).square().mean().backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    # This proves the graph passes through frozen blocks after the first adapter.
    assert model.adapters["0"].up.weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model(*args, task=task).square().mean().backward()
    assert model.geometry_encoder.encoders[0][0].weight.grad.abs().sum() > 0
    assert frozen_state_hash(model.backbone) == original
    assert not any(name.startswith("backbone.") for name in model.trainable_state_dict())
    model.train()
    assert not model.backbone.training


def test_semantic_tokens_masks_and_none_negative_control():
    torch.manual_seed(4)
    model = small_model()
    activate_adapter(model)
    args, task = input_batch(), features()
    tokens, mask = model.geometry_encoder(task, args[2], "geometry")
    assert tokens.shape == (2, 22, 32)
    assert mask.sum(dim=1).tolist() == [7, 7]
    expected = model(*args, task=task)
    padding = {name: value.clone() for name, value in task.items()}
    padding["features"][:, 14] = float("nan")
    padding["support_tokens"][:, 1:] = float("nan")
    assert torch.equal(model(*args, task=padding), expected)
    changed = {name: value.clone() for name, value in task.items()}
    changed["features"][:, 2:5] += .5
    assert not torch.equal(model(*args, task=changed), expected)
    negative = small_model("none")
    activate_adapter(negative)
    a = negative(*args, task=task)
    changed["support_tokens"] += 10
    changed["feature_mask"][:] = False
    assert torch.equal(negative(*args, task=changed), a)
    _, negative_mask = negative.geometry_encoder(task, args[2], "none")
    assert negative_mask.sum(dim=1).tolist() == [2, 2]


def test_canonical_task_is_rigid_world_invariant_and_body_fields_unchanged():
    task = features(1)
    identity = torch.tensor([1, 0, 0, 0, 1, 0])
    conditions = torch.cat((torch.tensor([1., 2., .75]), identity,
                            torch.tensor([3., 2., .75]), identity))[None]
    a = canonicalize_contact_task(task["features"], conditions, task["feature_mask"], task["support_tokens"], task["support_mask"])
    angle = .8
    c, s = np.cos(angle), np.sin(angle)
    rotation = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)
    translation = torch.tensor([3., -4., 1.])
    moved = conditions.clone()
    for offset in (0, 9):
        moved[:, offset:offset+3] = conditions[:, offset:offset+3] @ rotation.T + translation
        for axis in (3, 6):
            moved[:, offset+axis:offset+axis+3] = conditions[:, offset+axis:offset+axis+3] @ rotation.T
    f = task["features"].clone()
    f[:, 2:5] = f[:, 2:5] @ rotation.T + translation
    for offset in (5, 8, 11):
        f[:, offset:offset+3] = f[:, offset:offset+3] @ rotation.T
    supports = task["support_tokens"].clone()
    supports[..., :3] = supports[..., :3] @ rotation.T + translation
    supports[:, 0, 6:10] = torch.tensor([np.cos(angle/2), 0, 0, np.sin(angle/2)])
    b = canonicalize_contact_task(f, moved, task["feature_mask"], supports, task["support_mask"])
    torch.testing.assert_close(a["features"], b["features"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(a["support_tokens"], b["support_tokens"], atol=1e-6, rtol=1e-6)
    assert torch.equal(a["features"][:, 14:], task["features"][:, 14:])
    # Endpoint preparation is independent of expert paths.
    prepared = prepare_contact_condition_batch(conditions, task["support_tokens"], task["support_mask"],
                                               task["features"], task["feature_mask"], task["support_tokens"], task["support_mask"],
                                               torch.zeros(9), torch.ones(9))
    assert prepared[0].shape == (1, 18)
    assert abs(float(prepared[0][0, 10])) < 1e-6


def test_general_tool_plane_and_cylinder_losses_and_alpha_weighting():
    task = features(2)
    f = task["features"]
    f[1, :2] = torch.tensor([0., 1.])
    f[1, 14] = 2
    f[1, 15] = 0
    angles = torch.linspace(-.3, .3, 16)
    normal = torch.stack((angles.cos(), angles.sin(), torch.zeros_like(angles)), dim=-1)
    tangent = torch.stack((-angles.sin(), angles.cos(), torch.zeros_like(angles)), dim=-1)
    tool_z = -normal
    tool_y = torch.cross(tool_z, tangent, dim=-1)
    tool_frame = torch.stack((tangent, tool_y, tool_z), dim=-1)
    tool_rotation = quaternion_wxyz_to_matrix_torch(f[1:2, 25:29])[0]
    body = tool_frame @ tool_rotation.T
    cylinder_positions = 2 * normal - torch.einsum("lij,j->li", body, f[1, 22:25])
    plane = torch.zeros(16, 9)
    plane[:, 0] = torch.linspace(-1, 1, 16)
    plane[:, 2] = .75
    plane[:, 3:9] = torch.tensor([1, 0, 0, 0, 1, 0])
    cylinder = torch.cat((cylinder_positions, body[..., :, 0], body[..., :, 1]), dim=-1)
    paths = torch.stack((plane, cylinder))
    result = contact_geometry_losses(paths, task, torch.ones(2, 1, 1))
    assert float(result["contact"]) < 1e-12
    assert float(result["normal"]) < 1e-12
    perturbed = paths.clone().requires_grad_()
    with torch.no_grad():
        perturbed[0, :, 2] += .2
        perturbed[1, :, :3] += .2 * normal
    losses = contact_geometry_losses(perturbed, task, torch.ones(2, 1, 1))
    assert float(losses["contact"].detach()) == pytest.approx(.04, rel=1e-5)
    sum(losses.values()).backward()
    assert torch.isfinite(perturbed.grad).all()
    zero = contact_geometry_losses(perturbed, task, torch.zeros(2, 1, 1))
    assert all(float(value.detach()) == 0 for value in zero.values())
    # A tool rotated 90 degrees around Y requires body Z tangent to the plane;
    # the old body-Z-equals-normal shortcut would give the wrong answer here.
    general = features(1)
    general["features"][:, 22:25] = torch.tensor([.3, -.2, .4])
    general["features"][:, 25:29] = torch.tensor([2**-.5, 0, 2**-.5, 0])
    rotation = quaternion_wxyz_to_matrix_torch(general["features"][:, 25:29])[0]
    tip = torch.stack((torch.linspace(-1, 1, 16), torch.zeros(16), torch.zeros(16)), dim=-1)
    position = tip - rotation @ general["features"][0, 22:25]
    general_path = torch.cat((position, rotation[:, 0].expand(16, -1), rotation[:, 1].expand(16, -1)), dim=-1)[None]
    losses = contact_geometry_losses(general_path, general, torch.ones(1, 1, 1))
    assert float(losses["contact"]) < 1e-12
    assert float(losses["normal"]) < 1e-12


def test_reverse_augmentation_recomputes_task_frame_and_condition_environment():
    task = features(1)
    task["features"][:, 2:5] = torch.tensor([1., -2., 0.])
    path = np.zeros((1, 128, 9), np.float32)
    path[:, :, 0] = 1
    path[:, :, 1] = np.linspace(-2, 0, 128)
    path[:, :, 2] = .75
    path[:, :, 3:9] = [1, 0, 0, 0, 1, 0]
    conditions = np.concatenate((path[:, 0], path[:, -1]), axis=1)
    obstacles = np.zeros((2, 32, 10), np.float32)
    obstacles[1, 0] = [4, 5, 6, 1, 2, 1, 1, 0, 0, 0]
    mask = np.zeros((2, 32), bool)
    mask[1, 0] = True
    core = FrozenDataset(path, conditions, np.zeros(1, np.int8), obstacles, mask, np.ones(1, int), "fixture")
    data = ContactDataset(core, task["features"].numpy(), task["feature_mask"].numpy(),
                          task["support_tokens"].numpy(), task["support_mask"].numpy(),
                          np.zeros((1, 8, 4), np.float32), np.zeros((1, 8), bool),
                          np.asarray(["map"]), np.asarray(["condition"]), {"map": "family"}, {})
    normal = prepare_contact_batch(data, np.asarray([0]), torch.device("cpu"), torch.zeros(9), torch.ones(9))
    reverse = prepare_contact_batch(data, np.asarray([0]), torch.device("cpu"), torch.zeros(9), torch.ones(9), np.asarray([True]))
    assert not torch.equal(normal.task["features"][:, 2:14], reverse.task["features"][:, 2:14])
    assert torch.equal(normal.task["features"][:, 14:], reverse.task["features"][:, 14:])
    assert reverse.obstacle_mask[0, 0]  # condition-specific environment index 1 was used.
    assert torch.equal(reverse.clean[:, 0], reverse.conditions[:, :9])
    assert torch.equal(reverse.clean[:, -1], reverse.conditions[:, 9:])
    for batch in (normal, reverse):
        losses = contact_geometry_losses(batch.clean, batch.task, torch.ones(1, 1, 1))
        assert float(losses["contact"]) < 1e-12 and float(losses["normal"]) < 1e-12


def subset_dataset():
    ids, families = [], {}
    for family in range(10):
        for index in range(40):
            name = f"map_{family}_{index}"
            families[name] = f"family_{family:02d}"
            ids.extend([name] * 20)
    count = len(ids)
    core = FrozenDataset(np.zeros((count, 2, 9)), np.zeros((count, 18)), np.zeros(count, np.int8),
                         np.zeros((1, 32, 10)), np.zeros((1, 32), bool), np.zeros(count, int), "fixture")
    return ContactDataset(core, np.empty((0, 32)), np.empty((0, 32)), np.empty((0, 8, 10)),
                          np.empty((0, 8)), np.empty((0, 8, 4)), np.empty((0, 8)), np.asarray(ids),
                          np.asarray(ids), families, {})


def test_whole_map_subsets_are_nested_balanced_and_seed_independent():
    data = subset_dataset()
    previous = set()
    for size in (500, 1000, 2000, 4000, 8000):
        selected, report = select_training_indices(data, size, 123)
        assert len(selected) == size and previous.issubset(set(selected))
        counts = list(report["family_map_counts"].values())
        assert max(counts) - min(counts) <= 1
        for map_id in np.unique(data.map_ids[selected]):
            assert (data.map_ids[selected] == map_id).sum() == 20
        previous = set(selected)
    a, _ = select_training_indices(data, 500, 123)
    b, _ = select_training_indices(data, 500, 124)
    assert not np.array_equal(a, b)
    assert len(set(a) & set(b)) < 500
    with pytest.raises(ValueError, match="multiple of 20"):
        select_training_indices(data, 510)


def test_validation_noise_does_not_follow_training_seed_or_batch_size():
    torch.manual_seed(17)
    a = fixed_validation_noise(31)
    torch.manual_seed(43)
    b = fixed_validation_noise(31)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert torch.equal(torch.cat([b[1][i:i+7] for i in range(0, 31, 7)]), a[1])
    assert torch.count_nonzero(a[1][:, [0, -1]]) == 0
    assert asdict(LossWeights()) == dict(contact=5., normal=.5, smooth=.1, collision=.002)


def test_amp_overflow_is_not_counted_as_an_optimizer_update():
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.SGD([parameter], lr=.1)
    class FakeScaler:
        scale_value = 64.
        skip = True
        def unscale_(self, optimizer):
            pass
        def is_enabled(self):
            return True
        def get_scale(self):
            return self.scale_value
        def step(self, optimizer):
            if not self.skip:
                optimizer.step()
        def update(self):
            if self.skip:
                self.scale_value /= 2
                self.skip = False
    scaler = FakeScaler()
    parameter.grad = torch.tensor([2.])
    assert not scaled_optimizer_step(scaler, optimizer, [parameter])
    assert float(parameter.detach()) == 1.
    parameter.grad = torch.tensor([2.])
    assert scaled_optimizer_step(scaler, optimizer, [parameter])
    assert float(parameter.detach()) == pytest.approx(.9)


def test_full_batches_cross_shuffle_boundary_and_resume_without_dropping_tail():
    sampler = ContactBatchSampler(6, 4, 17)
    first_permutation = sampler.permutation.numpy().copy()
    first = sampler.next_indices()
    second = sampler.next_indices()
    assert len(first) == len(second) == 4
    assert np.array_equal(np.concatenate((first, second[:2])), first_permutation)
    assert sampler.cursor == 2 and sampler.epoch == 1
    state = sampler.state_dict()
    resumed = ContactBatchSampler(6, 4, 43)
    resumed.load_state_dict(state)
    assert np.array_equal(sampler.next_indices(), resumed.next_indices())


def write_training_fixture(directory: Path):
    directory.mkdir()
    base_path = directory / "base.pt"
    torch.manual_seed(9)
    base = small_model(sequence_length=128).backbone
    torch.save(dict(model_type="dit_cross", architecture=dict(sequence_length=128, dit_dimension=32, dit_depth=8, dit_heads=4),
                    model_state=base.state_dict(), path_mean=np.zeros(9, np.float32), path_std=np.ones(9, np.float32),
                    environment_feature_mean=np.zeros(6, np.float32), environment_feature_std=np.ones(6, np.float32),
                    canonical_frame=CANONICAL_FRAME_VERSION, prediction_type="velocity", diffusion_steps=100), base_path)
    stats = dict(path_mean=[0.] * 9, path_std=[1.] * 9, environment_feature_mean=[0.] * 6, environment_feature_std=[1.] * 6)
    reference = directory / "stats.json"
    reference.write_text(json.dumps(stats))
    root = directory / "data"
    root.mkdir()
    paths = np.zeros((540, 128, 9), np.float32)
    paths[:, :, 0] = np.linspace(0, 2, 128)
    paths[:, :, 2] = .75
    paths[:, :, 3:9] = [1, 0, 0, 0, 1, 0]
    conditions = np.concatenate((paths[:, 0], paths[:, -1]), axis=1)
    ids = np.asarray([f"map_{i // 20}" for i in range(540)])
    codes = np.repeat([0] * 25 + [1, 2], 20).astype(np.int8)
    task = features(540)
    supports, support_masks = np.zeros((540, 8, 10), np.float32), np.zeros((540, 8), bool)
    np.savez(root / "prepared_dataset.npz", paths=paths, conditions=conditions, split_codes=codes,
             obstacles=np.zeros((540, 32, 10), np.float32), obstacle_mask=np.zeros((540, 32), bool),
             environment_indices=np.arange(540), source_dataset=str(root.resolve()), map_ids=ids,
             condition_ids=np.asarray([f"condition_{i}" for i in range(540)]),
             surface_features_world=task["features"].numpy(), surface_feature_mask=task["feature_mask"].numpy(),
             task_support_tokens_world=supports, task_support_mask=support_masks,
             forbidden_regions_intrinsic=np.zeros((540, 8, 4), np.float32), forbidden_region_mask=np.zeros((540, 8), bool),
             normalization_coordinate_frame=CANONICAL_FRAME_VERSION, paths_coordinate_frame="world", **stats)
    (root / "dataset_freeze.json").write_text(json.dumps(dict(status="frozen", integrity_passed=True,
           prepared_cache_sha256=sha256_file(root / "prepared_dataset.npz"))))
    (root / "manifest.json").write_text(json.dumps(dict(normalization=dict(reference=str(reference), sha256=sha256_file(reference),
           fitted_on_contact_data=False, canonical_frame=CANONICAL_FRAME_VERSION))))
    (root / "provenance.json").write_text(json.dumps(dict(base_checkpoint_sha256=sha256_file(base_path))))
    for code, split in enumerate(("train", "validation", "test")):
        (root / split).mkdir()
        rows = [dict(group_id=str(ids[i]), family=f"family_{(i // 20) % 5}") for i in np.flatnonzero(codes == code)]
        (root / split / "index.json").write_text(json.dumps(dict(samples=rows)))
    return base_path, root


def test_training_exact_resume_common_score_and_adapter_only_checkpoint(tmp_path):
    base_path, dataset = write_training_fixture(tmp_path / "fixture")
    common = ["--base", str(base_path), "--dataset", str(dataset), "--device", "cpu", "--train-size", "500",
              "--steps", "2", "--batch-size", "2", "--validation-batch-size", "5", "--validation-interval", "2", "--checkpoint-every", "1"]
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    train(parser().parse_args(common + ["--output", str(full)]))
    train(parser().parse_args(common + ["--output", str(resumed), "--stop-after", "1"]))
    assert json.loads((resumed / "run_status.json").read_text())["status"] == "stopped_resumable"
    train(parser().parse_args(common + ["--output", str(resumed), "--resume"]))
    a = torch.load(full / "last.pt", weights_only=False)
    b = torch.load(resumed / "last.pt", weights_only=False)
    for key in ("adapter_state", "ema_state"):
        assert all(torch.equal(a[key][name], b[key][name]) for name in a[key])
    assert a["history"][0]["validation"] == b["history"][0]["validation"]
    assert a["frozen_base_state_sha256"] == b["frozen_base_state_sha256"]
    assert a["attempted_updates"] == b["attempted_updates"] == 2
    assert a["skipped_updates"] == b["skipped_updates"] == 0
    assert "model_state" not in a and not any(k.startswith("backbone.") for k in a["adapter_state"])
    assert json.loads((resumed / "run_status.json").read_text())["status"] == "complete"
    loaded, metadata = load_contact_adapter(base_path, resumed / "best.pt", "cpu")
    assert metadata["base_sha256"] == sha256_file(base_path)
    assert frozen_state_hash(loaded.backbone) == a["frozen_base_state_sha256"]
    with pytest.raises(RuntimeError, match="resume contract mismatch"):
        train(parser().parse_args(common + ["--output", str(resumed), "--resume", "--normal-weight", "0"]))
    wrong_hash = "0" * 64
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        load_frozen_base(base_path, expected_sha256=wrong_hash)
