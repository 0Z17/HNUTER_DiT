"""Run one owned experiment group with CPU/GPU temperature hysteresis.

Only the child session created here can receive STOP/CONT. No system settings
are changed. A stopped guard leaves a hot child stopped, unless explicitly told
to terminate it. The status file records the child identity for recovery.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any


def process_identity(pid: int) -> dict[str, Any] | None:
    try:
        root = Path('/proc') / str(pid)
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=fields[0], pgid=int(fields[2]), sid=int(fields[3]),
                    start_ticks=int(fields[19]), uid=root.stat().st_uid)
    except (OSError, ValueError):
        return None


def matching_identity(saved: dict[str, Any]) -> bool:
    current = process_identity(saved['pid'])
    return current is not None and current['state'] != 'Z' and all(
        current[key] == saved[key] for key in ('pid','uid','pgid','sid','start_ticks'))


def remaining_owned_members(root: dict[str, Any]) -> list[dict[str, Any]]:
    leader = process_identity(root['pid'])
    if leader is not None and leader['start_ticks'] != root['start_ticks']:
        return []
    result = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        member = process_identity(int(path.name))
        if (member is not None and member['state'] != 'Z' and member['uid'] == root['uid'] == os.getuid()
                and member['pgid'] == root['pgid'] and member['sid'] == root['sid']
                and member['start_ticks'] >= root['start_ticks']):
            result.append(member)
    return result


def stop_remaining_members(root: dict[str, Any]) -> list[dict[str, Any]]:
    members = remaining_owned_members(root)
    for member in members:
        if matching_identity(member):
            try:
                os.kill(member['pid'], signal.SIGSTOP)
            except ProcessLookupError:
                pass
    return members


def cpu_temperatures() -> list[dict[str, Any]]:
    values = []
    for directory in Path('/sys/class/hwmon').glob('hwmon*'):
        try:
            driver = (directory / 'name').read_text().strip()
        except OSError:
            continue
        if driver not in {'k10temp', 'coretemp', 'zenpower'}:
            continue
        for path in directory.glob('temp*_input'):
            try:
                temperature = float(path.read_text()) / 1000
            except (OSError, ValueError):
                continue
            if math.isfinite(temperature) and 0 < temperature < 150:
                values.append(dict(path=str(path), driver=driver, celsius=temperature))
    return values


def gpu_temperature(index: int) -> float | None:
    try:
        result = subprocess.run(
            ['nvidia-smi', f'--id={index}', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
            check=True, capture_output=True, text=True, timeout=3)
        value = float(result.stdout.strip())
        return value if math.isfinite(value) and 0 < value < 150 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def should_pause(cpu: float | None, gpu: float | None, *, use_gpu: bool,
                 cpu_pause: float, gpu_pause: float) -> bool:
    return cpu is None or cpu >= cpu_pause or (use_gpu and (gpu is None or gpu >= gpu_pause))


def can_resume(cpu: float | None, gpu: float | None, *, use_gpu: bool,
               cpu_resume: float, gpu_resume: float) -> bool:
    return cpu is not None and cpu <= cpu_resume and (not use_gpu or (gpu is not None and gpu <= gpu_resume))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--cpu-pause-c', type=float, default=90.)
    parser.add_argument('--cpu-resume-c', type=float, default=80.)
    parser.add_argument('--gpu-pause-c', type=float, default=83.)
    parser.add_argument('--gpu-resume-c', type=float, default=75.)
    parser.add_argument('--cool-seconds', type=float, default=10.)
    parser.add_argument('--poll-seconds', type=float, default=1.)
    parser.add_argument('--terminate-on-shutdown', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a child command is required after --')
    for low, high in ((args.cpu_resume_c, args.cpu_pause_c), (args.gpu_resume_c, args.gpu_pause_c)):
        if not 0 < low < high < 150:
            parser.error('resume must be lower than pause')
    if args.cool_seconds < 0 or args.poll_seconds <= 0:
        parser.error('invalid monitoring timing')
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.state_dir / 'guard.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('another guard owns this state directory')
    lock.seek(0); lock.truncate(); lock.write(str(os.getpid()) + '\n'); lock.flush()
    previous_status = args.state_dir / 'status.json'
    if previous_status.exists():
        previous = json.loads(previous_status.read_text())
        if previous.get('child') and remaining_owned_members(previous['child']):
            parser.error('previous owned child group still exists; resolve its recorded PID before launching a second group')
    if not cpu_temperatures() or (args.gpu is not None and gpu_temperature(args.gpu) is None):
        parser.error('required temperature sensor unavailable; command was not launched')
    log = (args.state_dir / 'events.jsonl').open('a', buffering=1)
    def event(kind: str, **extra: Any) -> None:
        log.write(json.dumps(dict(event=kind, time_utc=dt.datetime.now(dt.timezone.utc).isoformat(), **extra)) + '\n')
    child = subprocess.Popen(command, start_new_session=True)
    identity = process_identity(child.pid)
    if identity is None or identity['uid'] != os.getuid() or identity['pgid'] != child.pid or identity['sid'] != child.pid:
        child.terminate()
        raise RuntimeError('child is not an owned dedicated process group')
    paused = False
    pause_started = cool_started = None
    pause_count = resume_count = samples = 0
    total_paused = 0.
    peak_cpu = peak_gpu = None
    stopping = False
    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    state: dict[str, Any] = dict(guard_pid=os.getpid(), child=identity, command=command,
        pause_count=0, resume_count=0, samples=0, paused_seconds=0.)
    try:
        event('started', guard_pid=os.getpid(), child=identity, command=command,
              cpu_pause_c=args.cpu_pause_c, cpu_resume_c=args.cpu_resume_c,
              gpu_pause_c=args.gpu_pause_c, gpu_resume_c=args.gpu_resume_c,
              gpu_index=args.gpu, cool_seconds=args.cool_seconds)
        while not stopping and child.poll() is None:
            current_identity = process_identity(child.pid)
            if current_identity is None and child.poll() is not None:
                break
            if current_identity is None or any(current_identity[key] != identity[key] for key in ('pid', 'uid', 'pgid', 'sid', 'start_ticks')):
                raise RuntimeError('child identity changed')
            readings = cpu_temperatures()
            cpu = max((r['celsius'] for r in readings), default=None)
            gpu = gpu_temperature(args.gpu) if args.gpu is not None else None
            now = time.monotonic()
            samples += 1
            if cpu is not None:
                peak_cpu = cpu if peak_cpu is None else max(peak_cpu, cpu)
            if gpu is not None:
                peak_gpu = gpu if peak_gpu is None else max(peak_gpu, gpu)
            if not paused and should_pause(cpu, gpu, use_gpu=args.gpu is not None,
                                           cpu_pause=args.cpu_pause_c, gpu_pause=args.gpu_pause_c):
                os.killpg(child.pid, signal.SIGSTOP)
                paused, pause_started, cool_started = True, now, None
                pause_count += 1
                event('paused', cpu_c=cpu, gpu_c=gpu)
            if paused:
                if can_resume(cpu, gpu, use_gpu=args.gpu is not None,
                              cpu_resume=args.cpu_resume_c, gpu_resume=args.gpu_resume_c):
                    if cool_started is None:
                        cool_started = now
                    if now - cool_started >= args.cool_seconds:
                        os.killpg(child.pid, signal.SIGCONT)
                        duration = now - pause_started
                        total_paused += duration
                        paused, pause_started, cool_started = False, None, None
                        resume_count += 1
                        event('resumed', cpu_c=cpu, gpu_c=gpu, paused_seconds=duration)
                else:
                    cool_started = None
            state = dict(status='paused_cooling' if paused else 'running', guard_pid=os.getpid(),
                         child=identity, command=command, cpu_c=cpu, gpu_c=gpu,
                         peak_cpu_c=peak_cpu, peak_gpu_c=peak_gpu, samples=samples,
                         pause_count=pause_count, resume_count=resume_count,
                         paused_seconds=total_paused + (now - pause_started if paused else 0),
                         updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            atomic_json(args.state_dir / 'status.json', state)
            time.sleep(args.poll_seconds)
    except BaseException as error:
        # A failed sensor/logger/monitor must never leave a hot owned job running.
        current = process_identity(child.pid)
        matches = current is not None and all(current[key] == identity[key]
                    for key in ('pid', 'uid', 'pgid', 'sid', 'start_ticks'))
        stopped = False
        if matches and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGSTOP)
                stopped = True
            except ProcessLookupError:
                pass
        remaining = stop_remaining_members(identity)
        stopped |= bool(remaining)
        state.update(status='monitor_failed_child_paused' if stopped else 'monitor_failed',
                     child=identity, remaining_members=remaining, error_type=type(error).__name__, error=str(error))
        try:
            atomic_json(args.state_dir / 'status.json', state)
            event('monitor_failed', child_paused=stopped, error=str(error))
        except OSError:
            pass
        import sys
        print(f'temperature monitor failed; child_paused={stopped}: {error}', file=sys.stderr, flush=True)
        log.close()
        return 1
    if stopping and child.poll() is None:
        if args.terminate_on_shutdown:
            os.killpg(child.pid, signal.SIGTERM)
            if paused:
                os.killpg(child.pid, signal.SIGCONT)
            state.update(status='termination_requested')
            event('termination_requested')
        else:
            if not paused:
                os.killpg(child.pid, signal.SIGSTOP)
            state.update(status='guard_stopped_child_paused')
            event('guard_stopped_child_paused', reason='thermal_fail_closed')
        code = 130
    else:
        code = child.wait()
        remaining = stop_remaining_members(identity)
        if remaining:
            state.update(status='leader_exited_members_paused', exit_code=code, remaining_members=remaining)
            code = code or 1
            event('leader_exited_members_paused', members=remaining)
        else:
            state.update(status='complete' if code == 0 else 'failed', exit_code=code)
        event('child_exited', exit_code=code)
    atomic_json(args.state_dir / 'status.json', state)
    log.close()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
