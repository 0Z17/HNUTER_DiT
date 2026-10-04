#!/usr/bin/env python3
"""Encode recorded Isaac RGB frames with measured flight telemetry captions.

Requires Pillow and ffmpeg (or imageio-ffmpeg). No simulator, model inference,
trajectory replay or pose manipulation occurs during this export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get('schema') != 'hnuter_actual_flight_video_v1' or data.get('pose_replay') is not False:
        raise ValueError('not an actual controlled-flight capture manifest')
    frames = data.get('frames', [])
    if not frames or len(frames) != data.get('frame_count') or not data.get('complete_sequence'):
        raise ValueError('capture contains no frames or an incomplete sequence')
    fps, dt, interval = data['fps'], data['physics_dt'], data['step_interval']
    if fps not in (10, 20) or abs(fps*dt*interval-1.) > 1e-10:
        raise ValueError('capture clock does not match video framerate')
    for index, frame in enumerate(frames):
        image = (path.parent / frame['filename']).resolve()
        if not image.is_relative_to(path.parent) or not image.is_file():
            raise ValueError('frame path must remain inside the capture directory')
        if (frame['index'] != index or frame['physics_step'] != index*interval
                or abs(frame['simulation_time']-index/fps) > 1e-9):
            raise ValueError('frame sequence and simulation timestamps disagree')
        if digest(image) != frame['sha256']:
            raise ValueError('frame digest mismatch: ' + frame['filename'])
        with Image.open(image) as pixels:
            if pixels.size != (data['width'], data['height']):
                raise ValueError('frame resolution differs from manifest')
    return data


def validate_flight_identity(manifest, data, report_path):
    """Bind model labels to the run that recorded these exact camera frames."""
    flight = json.loads(Path(report_path).read_text())
    recorded = flight.get('video', {})
    if recorded.get('manifest_sha256') != digest(manifest):
        raise ValueError('flight report does not identify this exact video manifest')
    for field in ('frame_count', 'fps', 'width', 'height', 'last_simulation_time'):
        if recorded.get(field) != data.get(field):
            raise ValueError('flight report and capture disagree on ' + field)
    last = data['frames'][-1]
    steps = flight.get('steps')
    if (not isinstance(steps, int) or steps < last['physics_step']
            or steps-last['physics_step'] >= data['step_interval']
            or abs(flight.get('physics_dt', 0)-data['physics_dt']) > 1e-12):
        raise ValueError('video does not cover the recorded flight physics steps')
    # Failed physical/planning runs remain exportable; their measured contact
    # and planner status are essential evidence and are shown in captions.
    return flight


def locate_ffmpeg(explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError('--ffmpeg must be an executable file')
        return str(path)
    found = shutil.which('ffmpeg')
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError('Install ffmpeg or imageio-ffmpeg, or pass --ffmpeg /path/to/executable') from exc


def caption(image, frame, *, label, fps):
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
        title = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
    except OSError:
        font = title = ImageFont.load_default()
    draw.rectangle((0, 0, image.width, 34), fill=(10, 18, 28))
    draw.text((12, 7), label, font=title, fill=(234, 241, 250))
    y = image.height-76
    draw.rectangle((0, y, image.width, image.height), fill=(10, 18, 28))
    p = frame['position']
    lines = [
        f"Simulation {frame['simulation_time']:5.1f} s | step {frame['physics_step']:05d} | "
        f"goal distance {frame['goal_distance_m']:.2f} m | contact: {'YES' if frame['collision_detected'] else 'none'}",
        f"Measured xyz: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) m | planner: {frame['planner_status']}",
        f"Actual physics + rendered camera | {fps} fps, 1x simulation time (not wall-clock real time)",
    ]
    for row, text in enumerate(lines):
        draw.text((12, y+5+row*23), text, font=font, fill=(234, 241, 250))
    return image


def export(manifest, output, *, ffmpeg=None, report=None):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('export destination already exists; choose a new output path')
    data = validate_manifest(manifest)
    if data['width'] % 2 or data['height'] % 2:
        raise ValueError('H264 yuv420p output requires even dimensions')
    report_path = Path(report).resolve() if report is not None else manifest.parent.parent / 'report.json'
    if not report_path.is_file():
        raise ValueError('flight report is required to identify the executed model')
    flight = validate_flight_identity(manifest, data, report_path)
    model = flight.get('planner', {}).get('checkpoint', {})
    representation = model.get('input_representation', 'box')
    label = f"HNUTER | temporal DiT / {representation} | industrial {flight.get('dynamic_profile', 'scene')}"
    output.parent.mkdir(parents=True, exist_ok=True)
    executable = locate_ffmpeg(ffmpeg)
    version = subprocess.check_output([executable, '-version'], text=True).splitlines()[0]
    temporary = output.with_name(output.stem+'.encoding.mp4')
    if temporary.exists():
        raise ValueError('temporary encoding file already exists')
    command = [executable, '-hide_banner', '-loglevel', 'warning', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f"{data['width']}x{data['height']}", '-r', str(data['fps']),
        '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
        try:
            for frame in data['frames']:
                with Image.open(manifest.parent / frame['filename']) as pixels:
                    annotated = caption(pixels, frame, label=label, fps=data['fps'])
                    process.stdin.write(annotated.tobytes())
            process.stdin.close()
            code = process.wait()
            if code:
                errors.seek(0)
                raise RuntimeError('ffmpeg failed: ' + errors.read().decode(errors='replace'))
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait()
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(output)
    evidence = dict(schema='hnuter_actual_flight_mp4_v1', output=str(output), sha256=digest(output),
        manifest=str(manifest), manifest_sha256=digest(manifest), report=str(report_path), report_sha256=digest(report_path),
        checkpoint_sha256=model.get('checkpoint_sha256'), input_representation=representation,
        fps=data['fps'], frame_count=data['frame_count'], encoded_duration_seconds=data['frame_count']/data['fps'],
        first_simulation_time=data['first_simulation_time'], last_simulation_time=data['last_simulation_time'],
        ffmpeg_version=version, width=data['width'], height=data['height'],
        source=data['source'], playback=data['playback'], pose_replay=False,
        overlays='Measured frame state and planner status; no generated or interpolated flight frames.')
    output.with_suffix('.json').write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--ffmpeg')
    args = parser.parse_args(argv)
    print(json.dumps(export(args.manifest, args.output, ffmpeg=args.ffmpeg, report=args.report), indent=2))


if __name__ == '__main__':
    main()
