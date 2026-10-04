"""NumPy-only synchronous IPC client for process-isolated DiT inference."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time

import numpy as np


def _atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False) + '\n')
    temporary.replace(path)


def _worker_environment():
    env = os.environ.copy()
    # Isaac's Kit/ml_archive paths and shared libraries must never override the
    # worker venv. Torch's own wheel RPATH supplies its CUDA dependencies.
    for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD'):
        env.pop(key, None)
    env['LD_LIBRARY_PATH'] = env.get('HNUTER_DIT_LIBRARY_PATH', '')
    env['PYTHONNOUSERSITE'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.setdefault('OMP_NUM_THREADS', '4')
    env.setdefault('MKL_NUM_THREADS', '4')
    return env


class PlannerProcess:
    """Serialize one request at a time; call sample from a background thread.

    Every instance uses a private mailbox, even with the same output directory.
    Worker errors, unexpected exits and timeouts are surfaced to the caller.
    Logs and request/response evidence remain available after close.
    """
    _worker_script = Path(__file__).with_name('planner_worker.py')

    def __init__(self, output_dir=None, *, python=None, device='cuda',
                 startup_timeout=120., request_timeout=120., temporal_checkpoint=None):
        if device not in ('cpu', 'cuda', 'auto'):
            raise ValueError('device must be cpu, cuda, or auto')
        if not np.isfinite(startup_timeout) or startup_timeout <= 0 or not np.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError('planner timeouts must be finite and positive')
        executable = str(python or os.environ.get('HNUTER_DIT_PYTHON', '/opt/nvblox-venv/bin/python'))
        if not Path(executable).is_file() or not os.access(executable, os.X_OK):
            raise RuntimeError('DiT worker Python is missing or not executable: ' + executable +
                '. Use the perception image built with the simulation repository tools/build_perception_image.sh, '
                'or set HNUTER_DIT_PYTHON to an isolated Python with torch, numpy and scipy.')
        parent = None if output_dir is None else Path(output_dir).resolve()
        if parent is not None:
            parent.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(tempfile.mkdtemp(prefix='dit-planner-', dir=parent))
        # Docker's root/user mapping may differ from the host user's UID. Keep
        # this run's non-secret diagnostic mailbox readable from the bind mount.
        self.output_dir.chmod(0o755)
        self._closed = False
        self._sequence = 0
        self._status = None
        self._request_lock = threading.Lock()
        self._request_timeout = float(request_timeout)
        self.runtime_kind = 'temporal' if temporal_checkpoint is not None else 'legacy'
        config = dict(device=device, runtime_kind=self.runtime_kind)
        if temporal_checkpoint is not None:
            checkpoint = Path(temporal_checkpoint).resolve()
            if not checkpoint.is_file():
                raise ValueError('temporal checkpoint does not exist: ' + str(checkpoint))
            config.update(checkpoint_path=str(checkpoint),
                          expected_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        _atomic_json(self.output_dir/'config.json', config)
        self._log = (self.output_dir/'worker.log').open('w')
        try:
            self._process = subprocess.Popen([executable, str(self._worker_script), str(self.output_dir)],
                env=_worker_environment(), stdout=self._log, stderr=subprocess.STDOUT)
        except BaseException:
            self._closed = True
            self._log.close()
            raise
        try:
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                status = self._poll()
                if status and status.get('state') == 'ready':
                    self._description = status['model']
                    return
                time.sleep(.02)
            raise RuntimeError('DiT worker startup timed out; see ' + str(self.output_dir/'worker.log'))
        except BaseException:
            self._terminate()
            self._closed = True
            self._log.close()
            raise

    def _terminate(self):
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5.)

    def _poll(self, *, allow_stopped=False):
        path = self.output_dir/'status.json'
        if path.exists():
            self._status = json.loads(path.read_text())
        status = self._status
        if status and status.get('state') == 'error':
            raise RuntimeError('DiT worker failed: ' + status.get('error', 'unknown') + '; see ' + str(self.output_dir/'worker.log'))
        stopped = status and status.get('state') == 'stopped'
        if stopped and not allow_stopped:
            raise RuntimeError('DiT worker stopped unexpectedly; see ' + str(self.output_dir/'worker.log'))
        if self._process.poll() is not None and (not stopped or self._process.returncode != 0):
            raise RuntimeError('DiT worker exited with code ' + str(self._process.returncode) + '; see ' + str(self.output_dir/'worker.log'))
        return status

    def describe(self):
        return dict(self._description, process_isolated=True, worker_directory=str(self.output_dir))

    def sample(self, mode, tokens, mask, start, goal, seed, count=8):
        if self.runtime_kind != 'legacy':
            raise ValueError('a temporal worker cannot sample the legacy modes')
        if mode not in ('baseline', 'online'):
            raise ValueError('mode must be baseline or online')
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or not 1 <= count <= 16:
            raise ValueError('count must be an integer in [1,16]')
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError('seed must be an integer')
        tokens, mask = np.asarray(tokens, dtype=np.float32), np.asarray(mask, dtype=bool)
        start, goal = np.asarray(start, dtype=np.float32), np.asarray(goal, dtype=np.float32)
        if tokens.shape != (32, 10) or mask.shape != (32,) or not np.isfinite(tokens).all():
            raise ValueError('expected finite tokens[32,10] and mask[32]')
        if any(pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-8 for pose in (start, goal)):
            raise ValueError('start and goal must be finite xyz+wxyz poses')
        response, metadata = self._request(dict(mode=mode, tokens=tokens, mask=mask,
            start=start, goal=goal, seed=np.int64(seed), count=np.int64(count)), ('paths',))
        paths = response['paths']
        if paths.shape != (count, 128, 7) or not np.isfinite(paths).all():
            raise RuntimeError('DiT worker returned invalid poses')
        return paths, metadata

    def sample_temporal(self, encoding, seed, steps=20):
        """Send only causal sensor conditioning, never dynamic obstacle truth."""
        if self.runtime_kind != 'temporal':
            raise ValueError('temporal sampling requires an explicit temporal checkpoint')
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError('seed must be an integer')
        if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or not 1 <= steps <= 100:
            raise ValueError('temporal sampling steps must be an integer in [1,100]')
        tokens = np.asarray(encoding['history_tokens'], dtype=np.float32)
        mask = np.asarray(encoding['history_mask'])
        times = np.asarray(encoding['history_times'], dtype=np.float64)
        condition = np.asarray(encoding['condition'], dtype=np.float32)
        if (tokens.shape != (4,32,10) or mask.shape != (4,32) or mask.dtype != np.dtype(bool) or condition.shape != (12,) or
                times.shape != (4,) or not np.isfinite(tokens).all() or not np.isfinite(condition).all() or
                not np.allclose(times, [-.6,-.4,-.2,0.], atol=1e-8, rtol=0)):
            raise ValueError('invalid temporal four-frame conditioning or relative timestamps')
        arrays, metadata = self._request(dict(mode='temporal', history_tokens=tokens,
            history_mask=mask, history_times=times, condition=condition,
            seed=np.int64(seed), count=np.int64(1), steps=np.int64(steps)),
            ('positions', 'velocities', 'accelerations', 'future_times', 'control_points'))
        shapes = dict(positions=(65,3), velocities=(65,3), accelerations=(65,3),
                      future_times=(65,), control_points=(16,3))
        for key, shape in shapes.items():
            if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                raise RuntimeError('Temporal worker returned invalid ' + key)
        if not np.allclose(arrays['future_times'], np.arange(65)*.125, atol=1e-8, rtol=0):
            raise RuntimeError('Temporal worker changed the fixed prediction times')
        return dict(arrays, metadata=metadata, inference_seconds=metadata.get('inference_seconds'))

    def _request(self, payload, fields):
        """Shared atomic mailbox lifecycle for both strict model contracts."""
        with self._request_lock:
            if self._closed:
                raise RuntimeError('DiT planner process is closed')
            self._poll()
            sequence = self._sequence + 1
            temporary = self.output_dir/'request.tmp'
            with temporary.open('wb') as stream:
                np.savez(stream, sequence=np.int64(sequence), **payload)
            temporary.replace(self.output_dir/'request.npz')
            self._sequence = sequence
            deadline = time.monotonic() + self._request_timeout
            while time.monotonic() < deadline:
                if self._closed:
                    raise RuntimeError('DiT planner closed while waiting for a response')
                self._poll()
                response_path = self.output_dir/'response.npz'
                if response_path.exists():
                    with np.load(response_path, allow_pickle=False) as response:
                        response_sequence = int(response['sequence'])
                        if response_sequence == sequence:
                            metadata = json.loads(str(response['metadata']))
                            if metadata.get('error'):
                                raise RuntimeError('DiT request failed: ' + metadata['error'])
                            if any(field not in response for field in fields):
                                raise RuntimeError('DiT worker response is missing required arrays')
                            return {field: response[field].copy() for field in fields}, metadata
                        if response_sequence > sequence:
                            raise RuntimeError('DiT IPC response sequence is ahead of the request')
                time.sleep(.01)
            # A timed-out CUDA request may still be running; do not reuse it or
            # let a subsequent request mistake stale work for a fresh answer.
            self._terminate()
            self._closed = True
            self._log.close()
            raise RuntimeError('DiT request timed out; see ' + str(self.output_dir/'worker.log'))

    def close(self, timeout=20.):
        if self._closed:
            return self._status
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError('close timeout must be finite and positive')
        _atomic_json(self.output_dir/'stop.json', {'sequence': self._sequence})
        try:
            self._process.wait(timeout=timeout)
            status = self._poll(allow_stopped=True)
            if self._process.returncode != 0 or not status or status.get('state') != 'stopped':
                raise RuntimeError('DiT worker did not stop cleanly; see ' + str(self.output_dir/'worker.log'))
            return status
        except subprocess.TimeoutExpired as exc:
            self._terminate()
            raise RuntimeError('DiT worker shutdown timed out') from exc
        finally:
            self._closed = True
            self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
