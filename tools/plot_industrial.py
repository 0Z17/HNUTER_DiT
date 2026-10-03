#!/usr/bin/env python3
"""Plot measured industrial flights without importing Isaac, Torch or CUDA.

Example:
    python tools/plot_industrial.py results/industrial/baseline \
        results/industrial/online --output docs/industrial_comparison.png

Requires NumPy and Matplotlib. Inputs are report.json and trajectory.json from
hnuter_dit.industrial_runner. A fixture/stub stays visibly marked as diagnostic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def load_run(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    rows = json.loads((directory / "trajectory.json").read_text())
    if not rows:
        raise ValueError(f"{directory}: trajectory.json contains no measured samples")
    times = np.asarray([row["t"] for row in rows], dtype=float)
    positions = np.asarray([row["position"] for row in rows], dtype=float)
    velocities = np.asarray([row["velocity"] for row in rows], dtype=float)
    goal = np.asarray(report["goal_position"], dtype=float)
    if positions.shape != (len(times), 3) or velocities.shape != positions.shape or goal.shape != (3,):
        raise ValueError(f"{directory}: expected measured position/velocity and goal xyz")
    if not np.isfinite(times).all() or not np.isfinite(goal).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"{directory}: goal and increasing sample times must be finite")
    # Invalid physical samples remain gaps, never interpolated or replaced by targets.
    positions[~np.isfinite(positions).all(axis=1)] = np.nan
    velocities[~np.isfinite(velocities).all(axis=1)] = np.nan
    planner = report.get("planner") or {}
    fixture = bool(report.get("fixture_smoke") or report.get("test_hold_stub")
                   or planner.get("fixture_smoke") or planner.get("test_hold_stub")
                   or planner.get("model_used") is False)
    obstacle = [(row["t"], row["obstacle"]["actual_position"]) for row in rows
                if "obstacle" in row and "actual_position" in row["obstacle"]]
    return dict(directory=directory, report=report, planner=planner, rows=rows,
                t=times, position=positions, velocity=velocities, goal=goal,
                fixture=fixture, obstacle=obstacle)


def timings(plans, key):
    points = []
    for event in plans:
        value = (event.get("model") or {}).get(key) if key == "inference_seconds" else event.get(key)
        # Scan time is the simulation timestamp of the planning input; latency
        # itself is wall time. Do not mix it with simulated observation age.
        t = event.get("scan_time")
        if t is not None and value is not None:
            pair = np.asarray([t, value], dtype=float)
            if np.isfinite(pair).all() and pair[1] >= 0:
                points.append(pair)
    return np.asarray(points).reshape(-1, 2)


def phases(runs, axes):
    schedules = {tuple(run["report"].get("obstacle", {}).get("motion_times_seconds", [3, 5, 13, 15]))
                 for run in runs if run["report"].get("dynamic_obstacle")}
    maximum = max(float(run["t"][-1]) for run in runs)
    colors = ("#e9b44c", "#d95d62", "#6aa9ba")
    names = ("enter", "occupied", "exit")
    for schedule in sorted(schedules):
        if len(schedule) != 4 or not np.isfinite(schedule).all():
            continue
        for index, (left, right) in enumerate(zip(schedule[:-1], schedule[1:])):
            if left >= maximum:
                continue
            for axis in axes:
                axis.axvspan(left, min(right, maximum), color=colors[index], alpha=.12, zorder=0)
            axes[0].text((left + min(right, maximum)) / 2, .92,
                         f"{names[index]}\n{left:g}–{right:g} s", ha="center", va="top",
                         transform=axes[0].get_xaxis_transform(), fontsize=8, color="#555555")
    return bool(schedules)


def plot_runs(runs, output, dpi=180):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 13, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    header_height = 1.05 + .25 * len(runs)
    height = 9.2 + header_height
    fig, grid = plt.subplots(2, 2, figsize=(15, height))
    xy, error, speed, latency = grid.ravel()
    fig.subplots_adjust(left=.07, right=.98, bottom=.16, top=1-header_height/height,
                        hspace=.32, wspace=.22)
    fig.suptitle("Industrial flight evaluation · measured physics", x=.07, ha="left", y=.978,
                 fontsize=18, fontweight="bold")
    colors = plt.get_cmap("tab10")
    has_timing = False
    all_fixture = all(run["fixture"] for run in runs)
    for index, run in enumerate(runs):
        color = colors(index % 10)
        report = run["report"]
        label = f"{index+1}: {report.get('mode', 'unknown')} · seed {report.get('seed', '?')}"
        result = "PASS" if report.get("passed") else "FAIL"
        if run["fixture"]:
            result = f"DIAGNOSTIC STUB — model not used; mission {result}"
        elif not report.get("completed"):
            result += " · incomplete run"
        elif run["planner"].get("failed"):
            result += " · planner failed"
        summary = f"{label}  |  {result}  |  {run['directory'].name}"
        fig.text(.07, .937-index*.25/height, summary, color=color, fontsize=10, fontweight="bold")
        t, position, goal = run["t"], run["position"], run["goal"]
        xy.plot(position[:, 0], position[:, 1], color=color, lw=2.2, label=label, zorder=3)
        xy.scatter(position[0, 0], position[0, 1], s=48, marker="o", facecolor="white",
                   edgecolor=color, linewidth=1.7, zorder=4)
        xy.scatter(goal[0], goal[1], s=150, marker="*", color=color, edgecolor="white", linewidth=.6, zorder=5)
        if run["obstacle"]:
            centers = np.asarray([item[1] for item in run["obstacle"]], dtype=float)
            xy.plot(centers[:, 0], centers[:, 1], ls="--", color=color, alpha=.65, lw=1.5)
            xy.scatter(*centers[0, :2], marker="s", s=30, facecolor="white", edgecolor=color, zorder=4)
            xy.scatter(*centers[-1, :2], marker="s", s=30, color=color, zorder=4)
        error.plot(t, np.linalg.norm(position-goal, axis=1), color=color, lw=1.8, label=label)
        speed.plot(t, np.linalg.norm(run["velocity"], axis=1), color=color, lw=1.8, label=label)
        first_collision = (report.get("collision") or {}).get("first_collision")
        if first_collision:
            impact_time = first_collision["simulation_time"]
            impact = position[np.argmin(np.abs(t-impact_time))]
            xy.scatter(*impact[:2], marker="x", s=85, linewidth=2.2, color=color, zorder=6)
            for axis in (error, speed):
                axis.axvline(impact_time, color=color, ls=":", alpha=.8)
        # A diagnostic's synthetic planner timings are never performance evidence.
        if not run["fixture"]:
            plans = run["planner"].get("plans", [])
            for key, style, marker, title in (("planning_seconds", "-", "o", "total"),
                                               ("inference_seconds", "--", "s", "inference")):
                values = timings(plans, key)
                if len(values):
                    latency.plot(values[:, 0], values[:, 1], color=color, ls=style, marker=marker,
                                 markersize=4, lw=1.4, label=f"{index+1} {title}")
                    has_timing = True

    xy.set(title="Actual XY trajectory", xlabel="World X (m)", ylabel="World Y (m)")
    xy.set_aspect("equal", adjustable="datalim")
    xy.margins(.15)
    xy.legend(loc="best", fontsize=8)
    symbols = [Line2D([], [], color="#555555", marker="o", markerfacecolor="white", ls="", label="initial pose"),
               Line2D([], [], color="#555555", marker="*", ls="", markersize=10, label="goal"),
               Line2D([], [], color="#555555", marker="s", ls="--", label="actual box center"),
               Line2D([], [], color="#555555", marker="x", ls=":", label="first collision")]
    fig.legend(handles=symbols, loc="lower left", bbox_to_anchor=(.065, .064), ncol=4, frameon=False, fontsize=9)
    error.set(title="Distance to goal", ylabel="3D position error (m)", xlabel="Simulation time (s)")
    speed.set(title="Actual speed", ylabel="World linear speed (m/s)", xlabel="Simulation time (s)")
    latency.set(title="Planning latency · wall time", ylabel="Latency (wall seconds)",
                xlabel="Input scan time (simulation seconds)")
    for tolerance in {run["report"].get("goal_distance_tolerance_m", .35) for run in runs}:
        if tolerance is not None and np.isfinite(tolerance):
            error.axhline(tolerance, color="#777777", ls=":", lw=1)
    for tolerance in {run["report"].get("goal_speed_tolerance_mps", .10) for run in runs}:
        if tolerance is not None and np.isfinite(tolerance):
            speed.axhline(tolerance, color="#777777", ls=":", lw=1)
    maximum = max(float(run["t"][-1]) for run in runs)
    for axis in (error, speed, latency):
        axis.set_xlim(0, max(.001, maximum))
        axis.set_ylim(bottom=0)
    dynamic = phases(runs, (error, speed, latency))
    if has_timing:
        latency.legend(loc="best", fontsize=8, ncol=2)
    else:
        latency.text(.5, .5, "No model timing: diagnostic stub" if all_fixture else "No planner timing recorded",
                     transform=latency.transAxes, ha="center", va="center", color="#777777", fontsize=12)
    for axis in grid.ravel():
        axis.grid(color="#cbd1d7", alpha=.55, linewidth=.6)
        axis.ticklabel_format(style="plain", useOffset=False)
    notes = "Dotted thresholds: goal distance / speed. Mission pass also requires continuous settling and complete collision truth."
    if dynamic:
        notes += "\nShading: commanded obstacle phases; dashed XY path shows measured box motion. Timing includes completed planning results only."
    else:
        notes += "\nTiming includes completed planning results only. Total planning and model inference are measured wall durations."
    if any(run["fixture"] for run in runs):
        notes += "\nDIAGNOSTIC STUB: validates the physical fixture only; no learned flight or obstacle-avoidance claim."
    fig.text(.07, .045, notes, fontsize=8.3, color="#555555", va="top", linespacing=1.5)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", help="Run directories containing report.json and trajectory.json")
    parser.add_argument("--output", type=Path, required=True, help="Destination .png file")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args(argv)
    if args.output.suffix.lower() != ".png" or not 60 <= args.dpi <= 600:
        parser.error("output must end in .png and dpi must be between 60 and 600")
    try:
        runs = [load_run(directory) for directory in args.runs]
        plot_runs(runs, args.output, args.dpi)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Cannot plot industrial evaluation: {exc}\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
