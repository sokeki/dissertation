# Hand backend tests

```bash
# Fast, deterministic, no ROS2 needed: 40 tests, ~2 s
sim/venv/bin/python -m pytest sim/tests/ -q

# Full suite including real DDS: 57 tests, ~16 s
source /opt/ros/humble/setup.bash
sim/venv/bin/python -m pytest sim/tests/ -q
```

| File | Tests | Covers |
|---|---:|---|
| `test_equivalence.py` | 10 | One client, two backends, equivalent trajectories |
| `test_joint_mapping.py` | 15 | Name-based ordering, loud failure on mismatch |
| `test_rclpy_adapter.py` | 15 | `RclpyTransport` wiring, against a stub `rclpy` |
| `test_live_dds.py` | 17 | Real DDS: discovery, QoS, timing, process death |

## Environment

ROS2 Humble is at `/opt/ros/humble`. A spawned non-interactive shell does not
read `.bashrc`, so each shell needs `source /opt/ros/humble/setup.bash` first.

**The venv imports `rclpy` and `mujoco` together**, so there is no
two-interpreter problem. It is *not* `--system-site-packages`
(`include-system-site-packages = false`); `setup.bash` exports a `PYTHONPATH`
pointing at ROS's `dist-packages`, and `PYTHONPATH` is honoured independently of
venv site isolation — that flag only suppresses the base interpreter's
`site-packages`. The ABI matches because `rclpy`'s
`_rclpy_pybind11.cpython-310-*.so` needs CPython 3.10 and the venv is 3.10.12,
from the same interpreter Humble targets. A venv on 3.11+ would fail.

Two side effects: `PYTHONPATH` precedes venv `site-packages`, so a package
shipped by both resolves to ROS's copy (nothing collides today — `numpy` still
resolves to the venv's 2.2.6); and sourcing ROS makes pytest autoload seven ROS
plugins, one of which aborted collection on a missing PyYAML. Disabled in
`sim/pytest.ini`.

## Live DDS results

Everything from the loopback suite that could be re-expressed over DDS was, with
`mock_hand.py` as a real subprocess on an isolated `ROS_DOMAIN_ID` and
`ROS_LOCALHOST_ONLY=1`.

**Passed unchanged (claim carried over directly):** state received, canonical
joint names, commands reaching the hand and moving it, name-based mapping
against a reverse-order publisher, missing-joint loud failure, empty-names loud
failure, opt-in override.

**Needed adjustment for real timing, not for correctness:** the loopback tests
step a fixed count and assert, because loopback is deterministic — one `poll()`
is exactly one control period. Over DDS, step count no longer implies elapsed
time, so the convergence tests became deadline loops (`while time.monotonic() <
deadline`). No assertion was weakened; the tolerance stayed the same.

To provoke the error paths over a real graph, `mock_hand.py` gained
`--publish-order reversed`, `--omit-names`, `--drop-joint`, and `--reliability`.
The loopback `StubTransport` can inject a malformed sample directly; a real
publisher has to be *told* to misbehave.

**Nothing that passed on loopback failed over DDS.** What did break was the test
harness (the pytest plugin autoload above, which aborted collection entirely
before a single test ran) and two of my own predictions — see below.

## QoS: the inference is now measured

Matrix run against the live mock (`test_qos_matching_matrix`):

| Publisher | Subscription | Data flows? |
|---|---|---|
| RELIABLE | RELIABLE | yes |
| RELIABLE | BEST_EFFORT | yes |
| BEST_EFFORT | BEST_EFFORT | yes |
| **BEST_EFFORT** | **RELIABLE** | **no — `step()` times out** |

The predicted failure is real: DDS will not match a RELIABLE *request* against a
BEST_EFFORT *offer*, so no data flows at all and the symptom is a `step()`
timeout rather than degraded data.

**So the subscription default was changed to BEST_EFFORT**, because it is the
only setting that matches a publisher of either kind, and Wonik's actual QoS is
still unknown. The `joint_cmd` publisher stays RELIABLE — a dropped position
target leaves the hand holding a stale pose, and commands are low-rate. Both are
configurable via `reliability=` / `cmd_reliability=`.

## Timing: the abstraction's real cost

From `sim/bench_dds.py` (cold figures measured in fresh interpreters, since
rclpy's context stays warm within a process):

| Backend | Client loop rate | mean `step()` | p99 |
|---|---:|---:|---:|
| `MujocoHand` | **7023 Hz** | 0.142 ms | 0.205 ms |
| `Ros2Hand`, 100 Hz publisher | 100.0 Hz | 9.999 ms | 10.13 ms |
| `Ros2Hand`, 333 Hz publisher | 333.0 Hz | 3.003 ms | 3.12 ms |
| `Ros2Hand`, 500 Hz publisher | 500.0 Hz | 2.000 ms | 2.16 ms |
| `Ros2Hand`, 1000 Hz publisher | 999.9 Hz | 1.000 ms | 1.14 ms |

Identical client code runs **7–70× faster on MuJoCo**, and the ROS2 rate is set
entirely by the publisher — the match is exact to three significant figures,
which is the pacing mechanism working as designed rather than a coincidence.

MuJoCo also runs 14× faster than real time (0.142 ms per 2 ms timestep), so a
client that assumes wall-clock progress will behave differently on the two
backends even though both honour the interface. Anything timing-sensitive should
be expressed in control cycles, not seconds, or should read the elapsed
simulated/message time explicitly.

## DDS behaviour

- **Discovery latency lands in the constructor, not the first `step()`.** I
  predicted the opposite. Measured: `RclpyTransport.__init__` takes ~150–190 ms
  cold (`rclpy.init` plus participant and subscription setup), after which the
  first `step()` against a live publisher costs 0.26–6.8 ms — indistinguishable
  from steady state. Within one process later constructors take ~5 ms, so this
  is a per-process cost, not per-hand.
- **A longer first-`step()` timeout is still needed, for a different reason:**
  publisher *liveness*. If the client is constructed before the hand node is up
  (normal in a launch file, where ordering isn't guaranteed), the first `step()`
  absorbs the publisher's entire startup. ~200 ms for the mock's interpreter
  start; a real driver enumerating CAN could take seconds. Hence
  `STARTUP_TIMEOUT` (15 s) vs `STEADY_TIMEOUT` (2 s).
- **Node death surfaces promptly.** Killing the mock mid-run raises
  `HandStateTimeout` within the configured timeout (<3 s with `state_timeout=1.0`,
  the slack covering messages already in flight). This is exactly the failure a
  no-op `step()` could not detect.
- **Restart recovers without rebuilding the client.** DDS re-discovers a
  replacement publisher on the same topic; a driver restart does not require
  restarting the client.

## What the equivalence test asserts

Not sample-by-sample equality — the backends genuinely differ (MuJoCo has
gravity droop and a second-order servo response; the mock is a kinematic
first-order lag), and per-sample difference peaks near 0.5 rad mid-transient.

It asserts: identical joint names, both converging on every commanded pose,
settled poses agreeing across backends (worst case 0.072 rad against a 0.10 rad
tolerance), every significant motion agreeing in direction (27/27) and
magnitude, and per-joint correlation above 0.9 (measured min 0.923).

Two guards keep that honest: `test_tolerance_is_not_vacuous` asserts the
smallest commanded motion (0.25 rad) stays well above the tolerance, so a
backend that failed to move cannot pass; and
`test_a_mismapped_backend_would_fail_this_suite` proves a deliberate finger-swap
is detected. Both were checked by mutation — making `_decode` trust index order
fails 5 tests, making `step()` a no-op fails 8.

## Still unverified

The contract itself, not the transport. `RclpyTransport` is now proven against a
real graph, but the graph is our own mock. Outstanding:

1. Whether Wonik's node publishes `joint_states.name` at all. Their ROS1 node
   sets it (`allegro_node.cpp:33`) but resizes before filling, so an empty array
   is plausible. We refuse to guess by default.
2. Whether `joint_cmd` is still consumed by index. `allegro_node_grasp.cpp:139`
   ignores the inbound `name` array. We publish canonical order, so we are
   correct either way.
3. The real publish rate, and whether the 3F has a ROS2 driver with this
   contract at all — we have the 3F ROS1 package and second-hand knowledge of
   the 4F ROS2 one.
4. Wonik's actual QoS, which decides whether the BEST_EFFORT default was
   necessary or merely harmless.
