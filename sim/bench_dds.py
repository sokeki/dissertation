#!/usr/bin/env python3
"""
Measure the timing asymmetry between the MuJoCo and ROS2 backends.

    source /opt/ros/humble/setup.bash
    sim/venv/bin/python sim/bench_dds.py [--json out.json]

The unified HandInterface hides a real difference: MujocoHand.step() integrates
one 2 ms timestep and returns as fast as the CPU allows, while Ros2Hand.step()
blocks until the hand publishes. A client loop therefore runs at very different
rates on the two backends, and the ROS2 rate is set by the publisher, not by the
client. This script puts numbers on that.

Cold-start figures are measured in FRESH SUBPROCESSES on purpose: within one
interpreter, rclpy's context and DDS participant stay warm after the first
init(), so an in-process measurement understates first-connect cost by ~40x.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import mujoco

SIM_DIR = Path(__file__).resolve().parent
MODEL = SIM_DIR / "models/allegro_3f/allegro_3f.xml"
MOCK = SIM_DIR / "mock_hand.py"

BASE_DOMAIN = 80


def _env(domain: int) -> dict:
    env = dict(os.environ)
    env["ROS_DOMAIN_ID"] = str(domain)
    env.setdefault("ROS_LOCALHOST_ONLY", "1")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def spawn_mock(domain: int, rate: float, namespace: str = "allegroHand_0"):
    proc = subprocess.Popen(
        [sys.executable, str(MOCK), "--namespace", namespace, "--rate", str(rate)],
        cwd=str(SIM_DIR), env=_env(domain),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"mock died: {proc.stdout.read()}")
        if "mock hand on" in (proc.stdout.readline() or ""):
            return proc
    raise RuntimeError("mock never started")


def bench_mujoco(iters: int = 20_000) -> dict:
    """Loop rate a client achieves against MujocoHand."""
    sys.path.insert(0, str(SIM_DIR))
    from mujoco_hand import MujocoHand

    hand = MujocoHand(str(MODEL), show_viewer=False)
    hand.set_joint_targets([0.0, 0.4, 0.4] * 3)
    for _ in range(500):  # warm caches
        hand.step()

    samples = []
    t0 = time.monotonic()
    for _ in range(iters):
        t = time.perf_counter()
        hand.step()
        hand.get_joint_positions()
        samples.append(time.perf_counter() - t)
    wall = time.monotonic() - t0
    hand.close()

    samples.sort()
    return {
        "backend": "MujocoHand",
        "loop_hz": iters / wall,
        "mean_step_ms": statistics.fmean(samples) * 1e3,
        "p50_ms": samples[len(samples) // 2] * 1e3,
        "p99_ms": samples[int(len(samples) * 0.99)] * 1e3,
        "sim_timestep_ms": 2.0,
    }


# Run inside a fresh interpreter so the cold rclpy cost is real.
_COLD_PROBE = r"""
import json, os, sys, time
sys.path.insert(0, {sim!r})
from ros2_hand import Ros2Hand, RclpyTransport

t0 = time.perf_counter()
hand = Ros2Hand(transport=RclpyTransport(namespace={ns!r}), state_timeout=30.0)
ctor = time.perf_counter() - t0

t1 = time.perf_counter()
hand.step()
first = time.perf_counter() - t1

for _ in range(50):
    hand.step()

n = {n}
samples = []
t2 = time.monotonic()
for _ in range(n):
    t = time.perf_counter()
    hand.step()
    hand.get_joint_positions()
    samples.append(time.perf_counter() - t)
wall = time.monotonic() - t2
hand.close()

samples.sort()
print("RESULT" + json.dumps({{
    "ctor_ms": ctor * 1e3,
    "first_step_ms": first * 1e3,
    "loop_hz": n / wall,
    "mean_step_ms": sum(samples) / len(samples) * 1e3,
    "p50_ms": samples[len(samples)//2] * 1e3,
    "p99_ms": samples[int(len(samples)*0.99)] * 1e3,
}}))
"""


def bench_ros2(domain: int, rate: float, n: int = 400) -> dict:
    proc = spawn_mock(domain, rate)
    try:
        script = _COLD_PROBE.format(sim=str(SIM_DIR), ns="allegroHand_0", n=n)
        out = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(SIM_DIR), env=_env(domain),
            capture_output=True, text=True, timeout=180,
        )
        if out.returncode != 0:
            raise RuntimeError(f"probe failed:\n{out.stdout}\n{out.stderr}")
        line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")][0]
        result = json.loads(line[len("RESULT"):])
        result.update({"backend": "Ros2Hand", "publisher_hz": rate})
        return result
    finally:
        proc.kill()
        proc.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--json", type=Path, help="also write results as JSON")
    parser.add_argument("--rates", type=float, nargs="+",
                        default=[100.0, 333.0, 500.0, 1000.0],
                        help="publisher rates to sweep")
    args = parser.parse_args()

    if not MODEL.exists():
        sys.exit(f"{MODEL} not built; run python sim/build_3f.py")

    results = {"mujoco": bench_mujoco(), "ros2": []}

    print("=" * 72)
    print("MujocoHand: step() is CPU-bound, not rate-limited")
    print("=" * 72)
    mj = results["mujoco"]
    print(f"  client loop rate     {mj['loop_hz']:10.0f} Hz")
    print(f"  mean step()          {mj['mean_step_ms']:10.4f} ms  "
          f"(p50 {mj['p50_ms']:.4f}, p99 {mj['p99_ms']:.4f})")
    print(f"  simulated per step   {mj['sim_timestep_ms']:10.1f} ms")
    print(f"  => runs {mj['loop_hz'] * mj['sim_timestep_ms'] / 1000:.1f}x "
          "faster than real time")

    print()
    print("=" * 72)
    print("Ros2Hand over real DDS: step() paces to the publisher")
    print("=" * 72)
    print(f"  {'pub Hz':>8} {'loop Hz':>10} {'mean ms':>9} {'p50':>8} "
          f"{'p99':>8} {'ctor ms':>9} {'1st step ms':>12}")
    for i, rate in enumerate(args.rates):
        r = bench_ros2(BASE_DOMAIN + i, rate)
        results["ros2"].append(r)
        print(f"  {rate:8.0f} {r['loop_hz']:10.1f} {r['mean_step_ms']:9.3f} "
              f"{r['p50_ms']:8.3f} {r['p99_ms']:8.3f} {r['ctor_ms']:9.1f} "
              f"{r['first_step_ms']:12.3f}")

    print()
    slowest = min(r["loop_hz"] for r in results["ros2"])
    fastest = max(r["loop_hz"] for r in results["ros2"])
    print(f"  ROS2 client loop spans {slowest:.0f}-{fastest:.0f} Hz across "
          f"{args.rates[0]:.0f}-{args.rates[-1]:.0f} Hz publishers,")
    print(f"  vs {mj['loop_hz']:.0f} Hz for MuJoCo: a "
          f"{mj['loop_hz'] / fastest:.0f}-{mj['loop_hz'] / slowest:.0f}x "
          "difference in client loop rate for identical client code.")

    if args.json:
        # Unlike the grasp matrix, these numbers are NOT reproducible: they are
        # wall-clock measurements and vary with machine, load and DDS
        # implementation. The file is a record of one measurement on one machine,
        # so it carries a platform block and will legitimately differ on re-run.
        def git(*a):
            try:
                return subprocess.run(("git", "-C", str(SIM_DIR.parent), *a),
                                      capture_output=True, text=True,
                                      check=True).stdout.strip()
            except (subprocess.CalledProcessError, FileNotFoundError):
                return "unknown"

        payload = {
            "note": "Generated by sim/bench_dds.py -- do not edit by hand. "
                    "Wall-clock measurements: NOT reproducible across machines.",
            "platform": {
                "python": platform.python_version(),
                "machine": platform.machine(),
                "system": f"{platform.system()} {platform.release()}",
                "cpu_count": os.cpu_count(),
            },
            "provenance": {
                "git_commit": git("rev-parse", "HEAD"),
                # Scoped to sim/: see the equivalent note in grasp_eval.py. An
                # unscoped check counts this output file as an uncommitted change.
                "code_dirty": bool(git("status", "--porcelain", "--", "sim")),
                "mujoco": mujoco.__version__,
                "ros_distro": os.environ.get("ROS_DISTRO", "unset"),
                "rmw": os.environ.get("RMW_IMPLEMENTATION", "default"),
            },
            "publisher_rates_hz": args.rates,
            "results": results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
