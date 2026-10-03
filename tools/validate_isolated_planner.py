#!/usr/bin/env python3
"""Replay fixed LiDAR-token NPZ conditions through the isolated planner.

Input keys: tokens[32,10], mask[32], origin[3] (or start[7]), goal[3 or 7].
Run via /isaac-sim/python.sh with --import-parent-torch to reproduce the tested
Isaac parent / separate venv Torch arrangement. No simulation is launched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hnuter_dit.planner_process import PlannerProcess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='cuda')
    parser.add_argument('--python', default=None, help='Worker Python; defaults to HNUTER_DIT_PYTHON or /opt/nvblox-venv/bin/python')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--import-parent-torch', action='store_true')
    args = parser.parse_args()
    parent_torch = None
    if args.import_parent_torch:
        import torch
        parent_torch = dict(version=torch.__version__, module_path=torch.__file__)
    with np.load(args.condition, allow_pickle=False) as frame:
        tokens, mask = frame['tokens'].copy(), frame['mask'].copy()
        start = frame['start'].copy() if 'start' in frame else np.r_[frame['origin'], [1., 0., 0., 0.]]
        goal = frame['goal'].copy()
        if goal.shape == (3,):
            goal = np.r_[goal, [1., 0., 0., 0.]]
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version='hnuter_isolated_planner_replay_v1',
        condition_path=str(args.condition.resolve()),
        condition_sha256=hashlib.sha256(args.condition.read_bytes()).hexdigest(),
        seed=args.seed, count=args.count, parent_torch=parent_torch,
        parent_inference_model_imported='hnuter_dit.model' in sys.modules,
        start_pose_wxyz=start.tolist(), goal_pose_wxyz=goal.tolist(),
        active_tokens=int(mask.sum()), requests={},
        limitations=['Recorded token replay only; does not validate closed-loop flight or collision freedom.',
            'If the input supplies origin[3] instead of start[7], the start attitude is assumed identity.',
            'The baseline is the first request and the online request runs second; these are not steady-state latency statistics.',
            'Online priors are hand-designed; the unchanged checkpoint refines translation and endpoint SLERP constrains attitude.'])
    client = None
    try:
        began = time.perf_counter()
        client = PlannerProcess(args.output/'worker', python=args.python, device=args.device)
        report['worker'] = client.describe()
        report['startup_seconds'] = time.perf_counter()-began
        for mode in ('baseline', 'online'):
            began = time.perf_counter()
            paths, metadata = client.sample(mode, tokens, mask, start, goal, args.seed, args.count)
            metadata['ipc_wall_seconds'] = time.perf_counter()-began
            metadata['lengths_m'] = np.linalg.norm(np.diff(paths[..., :3], axis=1), axis=2).sum(axis=1).tolist()
            metadata['finite'] = bool(np.isfinite(paths).all())
            metadata['endpoint_position_error_m'] = float(max(np.abs(paths[:, 0, :3]-start[:3]).max(),
                                                              np.abs(paths[:, -1, :3]-goal[:3]).max()))
            report['requests'][mode] = metadata
            np.savez_compressed(args.output/f'{mode}.npz', paths=paths, start=start, goal=goal, tokens=tokens, mask=mask)
            print(json.dumps(dict(mode=mode, inference_seconds=metadata['inference_seconds'],
                                  lengths_m=metadata['lengths_m'], model_evaluations=metadata['model_evaluations'])), flush=True)
    finally:
        try:
            if client is not None:
                report['close'] = client.close()
        finally:
            (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
