"""Contact adapter trajectory replay, online inference and physical execution.

The frozen experiment is read only. Replay scores retain their original gates;
the separately retimed contact curve is audited again before MuJoCo execution.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import gzip
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import urlsplit
import uuid

import numpy as np
import torch

from .contact_adapter import sha256_file
from .contact_benchmark import _parse_input_record, parse_input_condition
from .contact_inference import ContactInferenceRuntime, LATENT_SEED, INFERENCE_VERSION
from .contact_metrics import _infer_states, _quaternions, _rotations, evaluate_contact_path
from .contact_projection import ContactProjectionSpec, lift_contact_intrinsic, project_contact_pose9
from .contact_smooth_projection import project_smooth_contact_pose9
from .evaluate_contact_experiment import DEFAULT_BENCHMARK, load_benchmark
from .mujoco_execution import DEFAULT_MUJOCO_ROOT, ExecutionConfig, MujocoExecutionManager
from .service import DEFAULT_COLLECTOR, DEFAULT_MODEL, PROJECT_ROOT
from .train_contact_adapter import json_hash
from .web import DiTWebHandler

DEFAULT_EXPERIMENT = PROJECT_ROOT / 'artifacts/contact_adapter_multifamily_v001'
DEFAULT_MODEL_ID = 'geometry_n8000_seed17'
DEFAULT_CONDITION = 'surface_contact_adapter_benchmark_10000cond_v001/map_000527/condition_002'
WEB_ROOT = Path(__file__).resolve().parent / 'contact_web_assets'
SCENARIOS = ('original', 'obstacle_shift')
DECODERS = ('off', 'on', 'on_smooth')


def execution_policy(decoder: str, metrics: dict, diagnostics: dict, error: str | None = None) -> dict:
    """Keep formal scores; allow only the curvature part of shape to warn."""
    curvature = metrics.get('maximum_curvature_inv_m')
    curvature_warning = bool(curvature is not None and np.isfinite(curvature) and curvature > 8.+1e-8)
    failures = list(metrics.get('failures', []))
    blockers = [name for name in failures if name != 'shape_quality']
    if error or metrics.get('evaluation_error') or metrics.get('projection_failed'):
        blockers.append('invalid_input_or_backend_error')
    strict = bool(metrics.get('candidate_success') and metrics.get('dense_validation_completed'))
    diagnostic = False
    if not strict and not blockers and failures == ['shape_quality'] and curvature_warning:
        backtracking = metrics.get('cumulative_backtracking_m')
        chord = metrics.get('minimum_local_chord_arc_ratio')
        if backtracking is None or backtracking > .25+1e-8:
            blockers.append('shape_backtracking')
        if chord is None or chord < .65-1e-8:
            blockers.append('shape_chord_arc_ratio')
        if not metrics.get('physical_collision_checked') or not metrics.get('physical_collision_free'):
            blockers.append('physical_collision_unverified')
        diagnostic = not blockers
    elif not strict and not blockers:
        blockers.append('execution_gates_unverified')
    if decoder not in {'on', 'on_smooth'}:
        blockers.append('raw_contact_reference_unsupported')
    curvature_nodes = sum(value is not None and value > 8. for value in diagnostics.get('curvature_inv_m', []))
    return dict(eligible=bool((strict or diagnostic) and not blockers),
                curvature_warning=curvature_warning, curvature_threshold_inv_m=8.,
                maximum_curvature_inv_m=curvature, curvature_exceeded_node_count=curvature_nodes,
                warnings=['source_curvature_exceeds_8_inv_m'] if curvature_warning else [],
                blocking_failures=list(dict.fromkeys(blockers)),
                mode='curvature_diagnostic' if curvature_warning else 'strict',
                formal_source_success=bool(metrics.get('candidate_success')),
                dense_execution_recheck_required=True,
                geometry='exact_intrinsic_linear_segments_rest_at_nodes' if decoder == 'on'
                         else 'separately_audited_bounded_intrinsic_cubic' if decoder == 'on_smooth'
                         else 'unsupported_raw_contact_reference')


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf8'))


def _check_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f'文件哈希与冻结记录不一致: {path}')
    return actual


def _safe_path(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute() or any('oracle' in part.lower() or 'route' in part.lower() or part == '..' for part in Path(relative).parts):
        raise ValueError('只允许读取物理条件白名单文件')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('文件路径超出数据目录')
    return path


def _read_scores(path: Path) -> list[dict]:
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _integer(value: Any, minimum: int, maximum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f'{label} 必须是 [{minimum}, {maximum}] 范围的整数')
    return value


def trajectory_diagnostics(path: np.ndarray, spec: ContactProjectionSpec,
                           clearance: Any = None) -> dict:
    """Per-node geometry with the same interior curvature policy as metrics."""
    rotation = _rotations(path)
    tip = path[:, :3] + rotation @ spec.tool_offset
    local = (tip - spec.origin) @ spec.frame
    if spec.kind == 'plane':
        uv = np.clip(local[:, :2], spec.bounds[:2, 0], spec.bounds[:2, 1])
        surface = np.column_stack((uv, np.zeros(len(path))))
        normal = np.tile([0., 0., 1.], (len(path), 1))
    else:
        angle = np.arctan2(local[:, 1], local[:, 0])
        angle += 2*np.pi*np.round((spec.start[0]/spec.radius-angle[0])/(2*np.pi))
        u = np.clip(spec.radius*np.unwrap(angle), *spec.bounds[0])
        v = np.clip(local[:, 2], *spec.bounds[1])
        angle = u/spec.radius
        normal = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(len(path))))
        surface = spec.radius*normal + np.column_stack((np.zeros((len(path), 2)), v))
    surface = surface @ spec.frame.T + spec.origin
    normal = normal @ spec.frame.T
    tool_axis = rotation @ spec.tool_rotation[:, 2]
    error = np.linalg.norm(tip-surface, axis=1)
    normal_error = np.degrees(np.arctan2(np.linalg.norm(np.cross(tool_axis, -normal), axis=1),
                                       np.sum(tool_axis*-normal, axis=1)))
    step = np.diff(path[:, :3], axis=0)
    length = np.linalg.norm(step, axis=1)
    usable = (length[:-1] > 1e-8) & (length[1:] > 1e-8)
    curvature: list[float | None] = [None]*len(path)
    angles = np.arccos(np.clip(np.sum(step[:-1][usable]*step[1:][usable], axis=1)
                              / (length[:-1][usable]*length[1:][usable]), -1, 1))
    for index, number in zip(np.flatnonzero(usable)+1, 2*angles/(length[:-1][usable]+length[1:][usable])):
        curvature[int(index)] = float(number)
    return dict(contact_error_m=error.tolist(), normal_error_deg=normal_error.tolist(),
                curvature_inv_m=curvature, clearance_m=clearance,
                curvature_policy='interior_nodes_only_ignore_zero_translation_segments')


def display_trajectory(path: np.ndarray, spec: ContactProjectionSpec,
                       intrinsic: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Densify the actual decoder, never world chords for a contact cylinder."""
    parameter = np.unique(np.r_[np.linspace(0., 127., 513), np.arange(128)])
    if intrinsic is not None:
        states = np.column_stack([np.interp(parameter, np.arange(128), intrinsic[:, axis]) for axis in range(3)])
        dense = lift_contact_intrinsic(states, spec).numpy()
        # Endpoint clamp in the official output is retained for playback too.
        dense[0], dense[-1] = path[0], path[-1]
        rotation = _rotations(dense)
        poses = np.column_stack((dense[:, :3], _quaternions(rotation)))
    else:
        from scipy.spatial.transform import Rotation, Slerp
        rotation_source = _rotations(path)
        q = _quaternions(rotation_source)
        r = Slerp(np.arange(128), Rotation.from_quat(q[:, [1, 2, 3, 0]]))(parameter)
        position = np.column_stack([np.interp(parameter, np.arange(128), path[:, axis]) for axis in range(3)])
        rotation = r.as_matrix()
        poses = np.column_stack((position, r.as_quat()[:, [3, 0, 1, 2]]))
    return poses, poses[:, :3] + rotation @ spec.tool_offset


def serialize_variant(path: np.ndarray | None, spec: ContactProjectionSpec, metrics: dict,
                      intrinsic: np.ndarray | None = None, error: str | None = None) -> dict:
    if error or path is None or not np.isfinite(path).all():
        return dict(poses_wxyz=[], tool_points=[], display_poses_wxyz=[], display_tool_points=[],
                    display_progress=[],
                    metrics=metrics, diagnostics={}, error=error or 'decoder 未返回有效路径', execution_eligible=False)
    try:
        if path.shape != (128, 9):
            raise ValueError('路径必须为 [128,9]')
        rotation = _rotations(path)
        poses = np.column_stack((path[:, :3], _quaternions(rotation)))
        display, tip = display_trajectory(path, spec, intrinsic)
    except ValueError as failure:
        return serialize_variant(None, spec, metrics, error=str(failure))
    return dict(poses_wxyz=poses.tolist(), tool_points=(path[:, :3]+rotation@spec.tool_offset).tolist(),
                display_poses_wxyz=display.tolist(), display_tool_points=tip.tolist(), metrics=metrics,
                display_progress=(np.unique(np.r_[np.linspace(0., 127., 513), np.arange(128)])/127).tolist(),
                diagnostics=trajectory_diagnostics(path, spec, metrics.get('clearance_m')),
                error=None, execution_eligible=False)


class ContactVisualizationService:
    def __init__(self, *, experiment: Path = DEFAULT_EXPERIMENT, benchmark: Path = DEFAULT_BENCHMARK,
                 base_checkpoint: Path = DEFAULT_MODEL, device: str = 'auto',
                 execution: Any = None, execution_options: ExecutionConfig | None = None,
                 execution_seed: int = 1) -> None:
        self.experiment, self.benchmark = Path(experiment).resolve(), Path(benchmark).resolve()
        self.base_checkpoint = Path(base_checkpoint).resolve()
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('指定了 CUDA，但当前 PyTorch 无法访问 GPU')
        self.manifest, self.inputs = load_benchmark(self.benchmark)
        self.manifest_sha256 = sha256_file(self.benchmark/'manifest.json')
        self.index = [json.loads(line) for line in (self.benchmark/'index.jsonl').read_text().splitlines() if line.strip()]
        self.rows = {(str(case), str(scenario)): index for index, (case, scenario)
                     in enumerate(zip(self.inputs['condition_id'], self.inputs['scenario']))}
        if len(self.rows) != len(self.inputs['condition_id']):
            raise RuntimeError('benchmark 存在重复 case/scenario')
        self.models = {'base': None}
        self.models.update({p.name: p/'best.pt' for p in sorted((self.experiment/'models').glob('*')) if (p/'best.pt').is_file()})
        self.execution = execution
        self.execution_options = ExecutionConfig.from_value(execution_options or ExecutionConfig(
            speed_mps=.35, max_acceleration_mps2=1., max_angular_speed_rps=.5, max_angular_acceleration_rps2=1.))
        self.execution_seed = int(execution_seed)
        self._runtime: ContactInferenceRuntime | None = None
        self._runtime_model: str | None = None
        self._lock = threading.RLock()
        self._plan: dict | None = None
        self._execution_condition: dict | None = None
        self._execution_plan_id: str | None = None
        self._execution_trial: dict | None = None

    def catalog(self) -> dict:
        return dict(models=[dict(id=key, label=key, checkpoint=str(path or self.base_checkpoint)) for key, path in self.models.items()],
                    families=sorted({item['family'] for item in self.index}), scenarios=list(SCENARIOS),
                    default_model=DEFAULT_MODEL_ID if DEFAULT_MODEL_ID in self.models else 'base',
                    default_condition_id=DEFAULT_CONDITION if (DEFAULT_CONDITION, 'original') in self.rows else self.index[0]['condition_id'],
                    benchmark_conditions=len(self.index), device=self.device, decoders=list(DECODERS),
                    execution=self.execution.describe() if self.execution else dict(available=False, reason='执行器未配置'),
                    contact_execution_options=asdict(self.execution_options),
                    latent_seed=LATENT_SEED, contact_force_control=False)

    def list_conditions(self, request: dict) -> dict:
        items = self.index
        for key in ('family', 'kind', 'difficulty'):
            value = request.get(key, 'all')
            if value != 'all':
                items = [item for item in items if item[key] == value]
        offset = _integer(request.get('offset', 0), 0, len(self.index), 'offset')
        limit = _integer(request.get('limit', 100), 1, 200, 'limit')
        return dict(items=[{key: item[key] for key in ('condition_id', 'map_id', 'family', 'kind', 'difficulty')}
                           for item in items[offset:offset+limit]], total=len(items), offset=offset)

    def _editable(self) -> None:
        if self.execution and self.execution.status()['state'] in {'starting', 'running', 'stopping'}:
            raise RuntimeError('MuJoCo 正在执行或停止，请等待完成后更换规划')

    def invalidate_plan(self, request: dict) -> dict:
        with self._lock:
            self._editable()
            if self._plan and request.get('plan_id') == self._plan['result']['plan_id']:
                self._plan = None
            return dict(invalidated=True)

    def _case(self, request: dict) -> tuple[str, str, str, int, dict]:
        model = request.get('model_id', DEFAULT_MODEL_ID)
        scenario = request.get('scenario', 'original')
        case = request.get('condition_id', DEFAULT_CONDITION)
        if model not in self.models or scenario not in SCENARIOS or (case, scenario) not in self.rows:
            raise ValueError('未知模型、场景或任务条件')
        row = self.rows[(case, scenario)]
        relative = str(self.inputs['input_shard'][row])
        shard = _safe_path(self.benchmark, relative)
        _check_hash(shard, self.manifest['input_files_sha256'][relative])
        records = _read_scores(shard)
        record = _parse_input_record(records[int(self.inputs['input_row'][row])])
        if any(record[key] != str(self.inputs[key][row]) for key in ('condition_id', 'map_id', 'family', 'kind', 'difficulty')):
            raise RuntimeError('条件身份与冻结 runtime cache 不一致')
        condition = parse_input_condition(record['conditions'][scenario])
        if json_hash(condition) != str(self.inputs['input_sha256'][row]):
            raise RuntimeError('任务条件与冻结 input_sha256 不一致')
        return model, case, scenario, row, condition

    def _contract(self, model: str) -> tuple[Path, dict]:
        directory = self.experiment/'evaluation'/model
        contract = _read_json(directory/'contract.json')
        if (contract['model_id'] != model or contract['benchmark_manifest_sha256'] != self.manifest_sha256
                or json_hash({k: v for k, v in contract.items() if k not in {'contract_sha256', 'created_utc'}}) != contract['contract_sha256']):
            raise RuntimeError('模型评测契约或 benchmark 身份不一致')
        return directory, contract

    def replay(self, request: dict) -> dict:
        with self._lock:
            self._editable()
            self._plan = None
            model, case, scenario, row, condition = self._case(request)
            spec = ContactProjectionSpec.from_condition(condition)
            directory, contract = self._contract(model)
            stem = str(self.inputs['map_id'][row]).rsplit('/', 1)[-1]
            raw_path = directory/'raw'/f'{stem}.npz'
            raw_meta = _read_json(raw_path.with_suffix('.json'))
            if raw_meta['contract_sha256'] != contract['contract_sha256']:
                raise RuntimeError('raw 与模型契约不匹配')
            _check_hash(raw_path, raw_meta['raw_sha256'])
            with np.load(raw_path, allow_pickle=False) as archive:
                paths, rows, indices = archive['world_pose9'], archive['input_rows'], archive['candidate_indices']
            mask = rows == row
            if not mask.any() or paths.shape != (len(rows), 128, 9) or indices.shape != rows.shape:
                raise RuntimeError('冻结预测分片不含当前条件或格式错误')
            score_path = directory/'scores'/f'{stem}.jsonl.gz'
            score_meta = _read_json(directory/'scores'/f'{stem}.json')
            if (score_meta['raw_sha256'] != raw_meta['raw_sha256'] or score_meta['contract_sha256'] != contract['contract_sha256']):
                raise RuntimeError('评分未绑定当前 raw/contract')
            _check_hash(score_path, score_meta['scores_sha256'])
            scores = {(int(item['candidate_index']), item['projection']): item['metrics']
                      for item in _read_scores(score_path) if item['condition_id'] == case and item['scenario'] == scenario
                      and item['input_sha256'] == str(self.inputs['input_sha256'][row])}
            smooth_directory = self.experiment/'smooth_projection_supplement/evaluation'/model
            smooth_contract = _read_json(smooth_directory/'contract.json')
            smooth_meta = _read_json(smooth_directory/'scores'/f'{stem}.json')
            if (smooth_contract['parent_contract_sha256'] != contract['contract_sha256']
                    or json_hash({k: v for k, v in smooth_contract.items() if k not in {'contract_sha256', 'created_utc'}}) != smooth_contract['contract_sha256']
                    or smooth_contract['benchmark_manifest_sha256'] != self.manifest_sha256
                    or smooth_meta['contract_sha256'] != smooth_contract['contract_sha256']
                    or smooth_meta['raw_sha256'] != raw_meta['raw_sha256']):
                raise RuntimeError('平滑投影未绑定当前 raw/contract')
            smooth_path = smooth_directory/'projected'/f'{stem}.npz'
            smooth_score_path = smooth_directory/'scores'/f'{stem}.jsonl.gz'
            _check_hash(smooth_path, smooth_meta['projected_sha256'])
            _check_hash(smooth_score_path, smooth_meta['scores_sha256'])
            with np.load(smooth_path, allow_pickle=False) as archive:
                if not np.array_equal(archive['input_rows'], rows) or not np.array_equal(archive['candidate_indices'], indices):
                    raise RuntimeError('平滑路径与 raw 的候选顺序不匹配')
                smooth_world, smooth_intrinsic = archive['world_pose9'], archive['intrinsic_states']
            smooth_scores = {int(item['candidate_index']): item['metrics'] for item in _read_scores(smooth_score_path)
                             if item['condition_id'] == case and item['scenario'] == scenario
                             and item['input_sha256'] == str(self.inputs['input_sha256'][row])}
            # Only reconstruction of the unsaved analytic decoder is needed.
            _check_hash(Path(__file__).with_name('contact_projection.py'), contract['evaluation_sources']['contact_projection.py'])
            candidates, raw_by_index = [], {}
            for offset in np.flatnonzero(mask):
                index, raw = int(indices[offset]), paths[offset]
                raw_by_index[index] = raw.copy()
                variants = {'off': serialize_variant(raw, spec, scores[(index, 'off')])}
                try:
                    analytic = project_contact_pose9(raw, spec)
                    variants['on'] = serialize_variant(analytic.world_pose9.numpy(), spec, scores[(index, 'on')], analytic.intrinsic_states.numpy())
                except ValueError as error:
                    variants['on'] = serialize_variant(None, spec, scores[(index, 'on')], error=str(error))
                variants['on_smooth'] = serialize_variant(smooth_world[offset], spec, smooth_scores[index], smooth_intrinsic[offset])
                variants['on_smooth']['execution_eligible'] = bool(smooth_scores[index].get('candidate_success') and
                                                                  smooth_scores[index].get('dense_validation_completed') and
                                                                  variants['on_smooth']['error'] is None)
                candidates.append(dict(index=index, variants=variants))
            candidates.sort(key=lambda item: item['index'])
            return self._bind_plan(model, case, scenario, row, condition, candidates, raw_by_index,
                                   dict(source='frozen_saved_predictions', raw_sha256=raw_meta['raw_sha256'],
                                        parent_contract_sha256=contract['contract_sha256'],
                                        smooth_contract_sha256=smooth_contract['contract_sha256'],
                                        base_sha256=contract['base_sha256'], adapter_sha256=contract['adapter_sha256'],
                                        metrics_source='frozen_saved_scores', sampling_steps=contract['ddim_steps']))

    def sample(self, request: dict) -> dict:
        with self._lock:
            self._editable()
            self._plan = None
            model, case, scenario, row, condition = self._case(request)
            count = _integer(request.get('count', 8), 1, 8, 'count')
            steps = _integer(request.get('sampling_steps', 25), 1, 100, 'sampling_steps')
            if self._runtime_model != model:
                self._runtime = None
                self._runtime_model = None
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                self._runtime = ContactInferenceRuntime(self.base_checkpoint, self.models[model], self.device)
                self._runtime_model = model
            started = time.perf_counter()
            paths = self._runtime.sample(self.inputs, [row]*count, list(range(count)), sampling_steps=steps)
            spec = ContactProjectionSpec.from_condition(condition)
            candidates = []
            for index, raw in enumerate(paths):
                variants = {'off': serialize_variant(raw, spec, evaluate_contact_path(raw, condition))}
                for decoder, project in (('on', project_contact_pose9), ('on_smooth', project_smooth_contact_pose9)):
                    try:
                        value = project(raw, spec)
                        metrics = evaluate_contact_path(value.world_pose9, condition, projected_intrinsic=value.intrinsic_states)
                        variants[decoder] = serialize_variant(value.world_pose9.numpy(), spec, metrics, value.intrinsic_states.numpy())
                        variants[decoder]['execution_eligible'] = bool(decoder == 'on_smooth' and metrics.get('candidate_success')
                                                                     and metrics.get('dense_validation_completed'))
                    except ValueError as error:
                        variants[decoder] = serialize_variant(None, spec, dict(candidate_success=False, failures=['decoder_error']), error=str(error))
                candidates.append(dict(index=index, variants=variants))
            return self._bind_plan(model, case, scenario, row, condition, candidates,
                                   {index: path.copy() for index, path in enumerate(paths)},
                                   dict(source='online_shared_latent_ddim', metrics_source='fresh_full_urdf_audit',
                                        base_sha256=self._runtime.metadata['base_sha256'],
                                        adapter_sha256=sha256_file(self.models[model]) if self.models[model] else None,
                                        sampling_steps=steps, wall_time_s=time.perf_counter()-started))

    def _bind_plan(self, model: str, case: str, scenario: str, row: int, condition: dict,
                   candidates: list, raw_by_index: dict, provenance: dict) -> dict:
        for candidate in candidates:
            for decoder, value in candidate['variants'].items():
                audit = execution_policy(decoder, value['metrics'], value['diagnostics'], value['error'])
                value.update(execution_audit=audit, execution_eligible=audit['eligible'])
        eligible = [item for item in candidates if item['variants']['on_smooth']['execution_eligible']]
        selected = min(eligible, key=lambda item: (item['variants']['on_smooth']['metrics'].get('path_length_m', float('inf')), item['index'])) if eligible else None
        selected_decoder = 'on_smooth'
        if selected is None:
            selected = next((item for item in candidates if item['variants']['on']['execution_eligible']), None)
            selected_decoder = 'on' if selected else 'on_smooth'
        result = dict(schema_version='contact_visualization_v001', plan_id=uuid.uuid4().hex,
                      model_id=model, condition_id=case, scenario=scenario, condition=copy.deepcopy(condition),
                      candidates=candidates, selected_candidate_index=selected['index'] if selected else None,
                      selected_decoder=selected_decoder,
                      provenance=provenance | dict(benchmark_manifest_sha256=self.manifest_sha256,
                                                  input_sha256=str(self.inputs['input_sha256'][row]),
                                                  latent_seed=LATENT_SEED, inference_version=INFERENCE_VERSION),
                      playback_parameter='normalized_node_index_not_physical_time',
                      execution_geometry='selected_decoder_specific_reference', contact_force_control=False,
                      loading_status='loaded', saved_candidate_count=len(candidates),
                      candidate_protocol_note=('冻结协议只在预选100/1000张地图保存8个候选，其余地图仅保存候选0；在线推理可生成1–8个候选。'
                                               if provenance['source'] == 'frozen_saved_predictions' else '在线生成候选，未写回冻结实验。'))
        self._plan = dict(result=copy.deepcopy(result), raw=raw_by_index, condition=copy.deepcopy(condition))
        return result

    def _require_plan(self, request: dict) -> dict:
        if not self._plan or request.get('plan_id') != self._plan['result']['plan_id']:
            raise ValueError('规划已失效，请重新载入或生成轨迹')
        return self._plan

    def preview_execution(self, request: dict) -> dict:
        from .contact_execution import build_contact_scene
        with self._lock:
            self._editable()
            plan = self._require_plan(request)
            if self.execution is None:
                raise RuntimeError('执行器未配置')
            poses = next((value['poses_wxyz'] for item in plan['result']['candidates']
                          for value in item['variants'].values() if value['poses_wxyz']), None)
            if poses is None:
                raise ValueError('没有可用于预览的有效接触路径')
            self.execution.preview(build_contact_scene(plan['condition']), poses[0], poses[-1])
            self._execution_condition = copy.deepcopy(plan['condition'])
            self._execution_plan_id = plan['result']['plan_id']
            self._execution_trial = None
            return self.execution_status()

    def start_execution(self, request: dict) -> dict:
        from .contact_execution import build_contact_scene, prepare_contact_reference, prepare_contact_polyline_reference
        with self._lock:
            self._editable()
            plan = self._require_plan(request)
            decoder = request.get('decoder', 'on_smooth')
            if decoder not in {'on', 'on_smooth'}:
                raise ValueError('接触闭环执行支持 on 和 on_smooth；raw 未满足接触约束时只能查看')
            index = _integer(request.get('candidate_index', plan['result']['selected_candidate_index']), 0, 7, 'candidate_index')
            candidate = next((item for item in plan['result']['candidates'] if item['index'] == index), None)
            variant = candidate['variants'][decoder] if candidate else None
            if not variant or not variant['execution_eligible']:
                reasons = ', '.join(variant['execution_audit']['blocking_failures']) if variant else 'unknown_candidate'
                raise ValueError('该候选有曲率以外的执行拦截原因: '+reasons)
            allow_curvature = request.get('allow_curvature_failure', False)
            if not isinstance(allow_curvature, bool):
                raise ValueError('allow_curvature_failure 必须是布尔值')
            if variant['execution_audit']['curvature_warning'] and not allow_curvature:
                raise ValueError('曲率超过8 m^-1；请选择允许曲率超标的诊断飞行模式')
            if self.execution is None:
                raise RuntimeError('执行器未配置')
            overrides = request.get('options', {})
            editable_options = {'speed_mps', 'max_acceleration_mps2', 'max_angular_speed_rps', 'max_angular_acceleration_rps2'}
            if not isinstance(overrides, dict) or set(overrides)-editable_options:
                raise ValueError('只允许调整参考平移/旋转速度与加速度上限')
            config = ExecutionConfig.from_value(asdict(self.execution_options) | overrides)
            condition = copy.deepcopy(plan['condition'])
            spec = ContactProjectionSpec.from_condition(condition)
            decoded = (project_contact_pose9 if decoder == 'on' else project_smooth_contact_pose9)(plan['raw'][index], spec)
            regenerated = np.column_stack((decoded.world_pose9.numpy()[:, :3], _quaternions(_rotations(decoded.world_pose9.numpy()))))
            saved = np.asarray(variant['poses_wxyz'])
            if (not np.allclose(regenerated[:, :3], saved[:, :3], atol=2e-6, rtol=0)
                    or not np.allclose(np.abs(np.sum(regenerated[:, 3:]*saved[:, 3:], axis=1)), 1., atol=2e-6, rtol=0)):
                raise RuntimeError('当前 decoder 重建与已保存路径不一致，禁止执行')
            if decoder == 'on':
                reference = prepare_contact_polyline_reference(condition, decoded.intrinsic_states.numpy(), config,
                                                              allow_curvature_failure=allow_curvature)
            else:
                reference = prepare_contact_reference(condition, decoded.control_points.numpy(), config,
                                                      allow_curvature_failure=allow_curvature)
            trial = dict(decoder=decoder, candidate_index=index, mode=variant['execution_audit']['mode'],
                         source_metrics=copy.deepcopy(variant['metrics']),
                         formal_source_success=variant['execution_audit']['formal_source_success'],
                         geometry=variant['execution_audit']['geometry'],
                         allow_curvature_failure=allow_curvature, timing=reference.describe())
            reference.execution_trial = trial
            self.execution.start(build_contact_scene(condition), saved, reference=reference,
                                 candidates=[saved], seed=self.execution_seed, options=config)
            self._execution_condition = condition
            self._execution_plan_id = plan['result']['plan_id']
            self._execution_trial = trial
            return self.execution_status()

    def execution_status(self) -> dict:
        from .contact_execution import contact_execution_metrics
        result = self.execution.status() if self.execution else dict(state='unavailable', executed_poses=[])
        result['plan_id'] = self._execution_plan_id
        if self._execution_trial:
            result['execution_trial'] = copy.deepcopy(self._execution_trial)
            if result.get('reference_audit', {}).get('curvature_warning'):
                result['execution_trial']['mode'] = 'curvature_diagnostic'
        outcome = result.get('flight_outcome', {})
        result['flight_success'] = outcome.get('success', result.get('success'))
        poses = result.get('executed_poses', [])
        if self._execution_condition and poses:
            try:
                result['contact_metrics'] = contact_execution_metrics(self._execution_condition, poses)
                result['contact_metrics']['task_success'] = bool((result.get('state') == 'completed' or outcome.get('terminal_state') == 'completed')
                                                                and result['flight_success']
                                                                and result.get('contact_count', 0) == 0
                                                                and result['contact_metrics'].get('task_success'))
            except (ValueError, RuntimeError, ImportError, OSError) as error:
                result['contact_metrics'] = dict(task_success=False, evaluation_error=str(error))
            result['task_success'] = result['contact_metrics']['task_success']
        result['contact_force_control'] = False
        return result

    def stop_execution(self, request: dict) -> dict:
        if self.execution:
            self.execution.stop()
        return self.execution_status()

    def close(self) -> None:
        if self.execution:
            self.execution.close()


class ContactWebHandler(DiTWebHandler):
    service: ContactVisualizationService

    def do_GET(self) -> None:
        if urlsplit(self.path).path == '/api/execution/status':
            self._json(200, self.service.execution_status())
        else:
            super().do_GET()

    def do_POST(self) -> None:
        routes = {'/api/conditions': self.service.list_conditions, '/api/replay': self.service.replay,
                  '/api/sample': self.service.sample, '/api/plan/invalidate': self.service.invalidate_plan,
                  '/api/execution/preview': self.service.preview_execution,
                  '/api/execution/start': self.service.start_execution, '/api/execution/stop': self.service.stop_execution}
        operation = routes.get(urlsplit(self.path).path)
        if operation is None:
            self._json(404, dict(error='not found'))
            return
        try:
            self._json(200, operation(self._payload()))
        except (ValueError, KeyError, TypeError, RuntimeError, ImportError, OSError, StopIteration) as error:
            self._json(400, dict(error=str(error)))

    def log_message(self, format: str, *args: Any) -> None:
        if urlsplit(self.path).path != '/api/execution/status':
            super().log_message(format, *args)


def handler_type(service: ContactVisualizationService) -> type[ContactWebHandler]:
    class BoundHandler(ContactWebHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(WEB_ROOT), **kwargs)
    BoundHandler.service = service
    return BoundHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8793)
    parser.add_argument('--experiment', type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument('--benchmark', type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--collector-root', type=Path, default=DEFAULT_COLLECTOR)
    parser.add_argument('--mujoco-root', type=Path, default=DEFAULT_MUJOCO_ROOT)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--execution-seed', type=int, default=1)
    parser.add_argument('--speed', type=float, default=.35)
    parser.add_argument('--max-accel', type=float, default=1.)
    parser.add_argument('--angular-speed', type=float, default=.5)
    parser.add_argument('--angular-accel', type=float, default=1.)
    parser.add_argument('--mppi-samples', type=int, default=192)
    parser.add_argument('--mppi-horizon', type=int, default=32)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--no-realtime', action='store_true')
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    os.environ['HNUTER_COLLECTOR_ROOT'] = str(args.collector_root.resolve())
    options = ExecutionConfig(speed_mps=args.speed, max_acceleration_mps2=args.max_accel,
                              max_angular_speed_rps=args.angular_speed, max_angular_acceleration_rps2=args.angular_accel,
                              num_samples=args.mppi_samples, horizon=args.mppi_horizon)
    options.validate()
    executor = MujocoExecutionManager(args.mujoco_root, viewer=not args.headless, realtime=not args.no_realtime)
    service = ContactVisualizationService(experiment=args.experiment, benchmark=args.benchmark,
                                          base_checkpoint=args.checkpoint, device=args.device, execution=executor,
                                          execution_options=options, execution_seed=args.execution_seed)
    server = ThreadingHTTPServer((args.host, args.port), handler_type(service))
    url = f'http://{args.host}:{args.port}/'
    print(f'HNUTER Contact Adapter Lab: {url}\nReplay is read-only; online inference device={service.device}', flush=True)
    if args.open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == '__main__':
    main()
