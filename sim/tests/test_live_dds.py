"""
RclpyTransport against a real ROS2 graph, with the mock hand in its own process.

This is the file that closes the gap the loopback tests cannot: DDS discovery,
QoS negotiation, real timing, and process death. Everything here spawns
mock_hand.py as a genuine subprocess, so messages cross a real DDS transport
rather than a Python function call.

Skipped automatically when ROS2 is unavailable, so the fast suite still runs
anywhere. To run these:

    source /opt/ros/humble/setup.bash
    sim/venv/bin/python -m pytest sim/tests/test_live_dds.py -q

Each test gets its own ROS_DOMAIN_ID so concurrent runs, and any real hand on
the default domain, cannot interfere.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SIM_DIR = Path(__file__).resolve().parent.parent
MOCK_SCRIPT = SIM_DIR / "mock_hand.py"

rclpy = pytest.importorskip("rclpy", reason="ROS2 not available")

from ros2_hand import (  # noqa: E402  (after importorskip by design)
    CANONICAL_JOINT_NAMES_3F,
    HandStateTimeout,
    JointNameMismatch,
    RclpyTransport,
    Ros2Hand,
)

# Discovery over DDS is not instant, so the first message can take far longer
# than steady state. Measured in test_discovery_latency; this bound is generous
# on purpose, since a flaky first-connect would make the whole suite unreliable.
STARTUP_TIMEOUT = 15.0
STEADY_TIMEOUT = 2.0

# Domain IDs are per-test to isolate DDS traffic. 0 is avoided since that is
# where a real hand would live.
_next_domain = iter(range(60, 100))


@pytest.fixture
def domain_id():
    return next(_next_domain)


class MockProcess:
    """A mock_hand.py subprocess, plus the env needed to talk to it."""

    def __init__(self, proc: subprocess.Popen, namespace: str, env: dict):
        self.proc = proc
        self.namespace = namespace
        self.env = env

    def kill(self, sig=signal.SIGKILL) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(sig)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None


@pytest.fixture
def spawn_mock(domain_id):
    """Factory fixture launching mock_hand.py as a separate process."""
    started: list[MockProcess] = []

    def _spawn(*args: str, namespace: str = "allegroHand_0", rate: float = 333.0,
               wait_for_banner: bool = True):
        env = dict(os.environ)
        env["ROS_DOMAIN_ID"] = str(domain_id)
        # Loopback only: keeps test traffic off any physical network.
        env.setdefault("ROS_LOCALHOST_ONLY", "1")
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            [sys.executable, str(MOCK_SCRIPT),
             "--namespace", namespace, "--rate", str(rate), *args],
            cwd=str(SIM_DIR), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        mock = MockProcess(proc, namespace, env)
        started.append(mock)

        # Wait for the node's banner so we know it got as far as constructing.
        # Tests that need a genuinely cold start pass wait_for_banner=False --
        # otherwise this wait means the publisher is already up and streaming by
        # the time the test calls step(), and "cold start" measures nothing.
        if wait_for_banner:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    pytest.fail(f"mock node died at startup:\n{proc.stdout.read()}")
                line = proc.stdout.readline()
                if line and "mock hand on" in line:
                    break
            else:
                pytest.fail("mock node never printed its startup banner")
        return mock

    yield _spawn

    for mock in started:
        mock.kill()


@pytest.fixture
def hand_factory(domain_id):
    """
    Factory for Ros2Hand instances on a live graph, cleaned up afterwards.

    ROS_DOMAIN_ID must be set in this process's environment before rclpy.init(),
    which is why it is set here rather than in the subprocess env alone.
    """
    created: list[Ros2Hand] = []
    saved = {k: os.environ.get(k) for k in ("ROS_DOMAIN_ID", "ROS_LOCALHOST_ONLY")}
    os.environ["ROS_DOMAIN_ID"] = str(domain_id)
    os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")

    def _make(namespace: str = "allegroHand_0", **kwargs):
        timeout = kwargs.pop("state_timeout", STARTUP_TIMEOUT)
        transport = RclpyTransport(namespace=namespace, **kwargs)
        hand = Ros2Hand(
            namespace=namespace, transport=transport, state_timeout=timeout,
        )
        created.append(hand)
        return hand

    yield _make

    for hand in created:
        try:
            hand.close()
        except Exception:
            pass
    if rclpy.ok():
        rclpy.shutdown()
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# 1. The loopback suite's claims, re-run over real DDS
# ---------------------------------------------------------------------------

def test_state_is_received_over_dds(spawn_mock, hand_factory):
    """The baseline: does anything arrive at all across a process boundary."""
    spawn_mock()
    hand = hand_factory()
    hand.step()
    positions = hand.get_joint_positions()
    assert len(positions) == 9


def test_joint_names_match_the_canonical_order(spawn_mock, hand_factory):
    spawn_mock()
    hand = hand_factory()
    hand.step()
    assert list(hand.joint_names) == list(CANONICAL_JOINT_NAMES_3F)


def test_commands_reach_the_hand_and_move_it(spawn_mock, hand_factory):
    """Round trip: publish joint_cmd, observe joint_states converge to it."""
    spawn_mock()
    hand = hand_factory()
    hand.step()

    target = [0.0, 0.6, 0.6, 0.0, 0.6, 0.6, 0.0, 0.6, 0.6]
    hand.set_joint_targets(target)

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        hand.step()
        if max(abs(a - b) for a, b in zip(hand.get_joint_positions(), target)) < 0.01:
            return
    pytest.fail(f"never converged; last = {hand.get_joint_positions()}")


def test_mapping_by_name_over_dds(spawn_mock, hand_factory):
    """
    The ordering claim over a real transport.

    The mock publishes joint_states in reverse order while consuming commands by
    index, exactly as the hardware does. Only correct name-based mapping on the
    read path plus canonical ordering on the write path gets this right.
    """
    spawn_mock("--publish-order", "reversed")
    hand = hand_factory()
    hand.step()

    target = [0.0, 0.5, -0.5, 0.0, 0.4, -0.4, 0.0, 0.3, -0.3]
    hand.set_joint_targets(target)

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        hand.step()
        if max(abs(a - b) for a, b in zip(hand.get_joint_positions(), target)) < 0.02:
            return
    pytest.fail(
        f"reversed-order publisher not mapped correctly; "
        f"got {hand.get_joint_positions()} want {target}"
    )


def test_missing_joint_fails_loudly_over_dds(spawn_mock, hand_factory):
    spawn_mock("--drop-joint", "joint_8_0")
    hand = hand_factory()
    hand.step()
    with pytest.raises(JointNameMismatch, match="joint_8_0"):
        hand.get_joint_positions()


def test_omitted_names_fail_loudly_over_dds(spawn_mock, hand_factory):
    spawn_mock("--omit-names")
    hand = hand_factory()
    hand.step()
    with pytest.raises(JointNameMismatch, match="empty name array"):
        hand.get_joint_positions()


def test_omitted_names_accepted_when_opted_in(spawn_mock, hand_factory):
    spawn_mock("--omit-names")
    hand = hand_factory()
    hand.allow_unnamed_states = True
    hand.step()
    assert len(hand.get_joint_positions()) == 9


# ---------------------------------------------------------------------------
# 2. QoS: characterise the matching failure rather than predicting it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "publisher_reliability,subscriber_reliability,should_match",
    [
        ("reliable", "reliable", True),
        ("reliable", "best_effort", True),
        ("best_effort", "best_effort", True),
        # The predicted failure: DDS refuses to match a RELIABLE request
        # against a BEST_EFFORT offer, so no data flows at all.
        ("best_effort", "reliable", False),
    ],
)
def test_qos_matching_matrix(
    spawn_mock, hand_factory, publisher_reliability,
    subscriber_reliability, should_match,
):
    spawn_mock("--reliability", publisher_reliability)
    hand = hand_factory(
        reliability=subscriber_reliability,
        state_timeout=STARTUP_TIMEOUT if should_match else 5.0,
    )

    if should_match:
        hand.step()
        assert len(hand.get_joint_positions()) == 9
    else:
        with pytest.raises(HandStateTimeout):
            hand.step()


def test_invalid_reliability_is_rejected():
    from ros2_hand import qos_profile

    with pytest.raises(ValueError, match="reliability must be one of"):
        qos_profile(1, "eventually")


# ---------------------------------------------------------------------------
# 3 & 4. DDS-specific behaviour: discovery, first-step cost, node death
# ---------------------------------------------------------------------------

def test_discovery_cost_is_paid_in_the_constructor_not_the_first_step(
    spawn_mock, hand_factory, record_property
):
    """
    Where the DDS startup cost actually lands.

    I predicted the first step() would be slower than steady state because
    discovery had to complete first. Measured, that is WRONG: against an
    already-running publisher the first step() takes ~2.5 ms, indistinguishable
    from the ~3.0 ms steady interval, while RclpyTransport's constructor takes
    ~190 ms. rclpy.init() plus participant creation and subscription matching
    all happen during construction, so by the time step() is first called data
    is already flowing.

    The practical consequence is the opposite of what the naming implied: a long
    first-step timeout is not needed for discovery. It is needed for publisher
    LIVENESS -- see the next test.

    NOTE the ~190 ms constructor figure is a COLD-PROCESS cost, recorded here
    only for information. Within one pytest session rclpy's context and the DDS
    participant stay warm after the first test, so later constructors take ~5 ms.
    bench_dds.py measures the cold number in a fresh interpreter, which is why
    this test does not assert on it.
    """
    spawn_mock()

    t0 = time.monotonic()
    hand = hand_factory()
    ctor = time.monotonic() - t0

    t1 = time.monotonic()
    hand.step()
    first = time.monotonic() - t1

    for _ in range(20):  # let the stream settle
        hand.step()
    steady = []
    for _ in range(100):
        t = time.monotonic()
        hand.step()
        steady.append(time.monotonic() - t)
    mean_steady = sum(steady) / len(steady)

    record_property("transport_ctor_s", round(ctor, 6))
    record_property("first_step_s", round(first, 6))
    record_property("mean_steady_step_s", round(mean_steady, 6))
    print(f"\nctor {ctor * 1e3:.1f} ms; first step {first * 1e3:.2f} ms; "
          f"steady mean {mean_steady * 1e3:.2f} ms")

    # The corrected claim, and the part that holds warm or cold: against a live
    # publisher the first step is already at steady-state cost.
    assert first < 3 * mean_steady, (
        "against a live publisher the first step should already be at "
        f"steady-state cost, got {first * 1e3:.2f} ms vs "
        f"{mean_steady * 1e3:.2f} ms"
    )
    assert mean_steady < STEADY_TIMEOUT


def test_first_step_absorbs_publisher_startup_when_the_hand_is_not_up_yet(
    spawn_mock, hand_factory, record_property
):
    """
    The real reason STARTUP_TIMEOUT must exceed STEADY_TIMEOUT.

    Constructing the client before the hand node exists is the normal case in a
    launch file, where startup order is not guaranteed. The first step() then
    has to absorb the publisher's entire startup, which is unbounded from our
    side: here it is the mock's Python interpreter start (~200 ms), but a real
    driver enumerating CAN hardware could take seconds.
    """
    hand = hand_factory(state_timeout=STARTUP_TIMEOUT)  # constructed first
    # Deliberately do NOT wait for the banner: the point is that step() is
    # called while the publisher is still starting up.
    spawn_mock(wait_for_banner=False)

    t0 = time.monotonic()
    hand.step()
    first = time.monotonic() - t0

    steady = []
    for _ in range(50):
        t = time.monotonic()
        hand.step()
        steady.append(time.monotonic() - t)
    mean_steady = sum(steady) / len(steady)

    record_property("first_step_cold_s", round(first, 6))
    print(f"\ncold first step {first * 1e3:.1f} ms; "
          f"steady mean {mean_steady * 1e3:.2f} ms")

    assert first > 5 * mean_steady, (
        "expected the first step to absorb publisher startup"
    )


def test_steady_state_step_paces_to_the_publisher(spawn_mock, hand_factory):
    """
    step() should return at roughly the publisher's rate, not faster.

    This is the mechanism behind the timing asymmetry between backends: a client
    loop against ROS2 is paced by the hand, not by the CPU.
    """
    rate = 100.0
    spawn_mock(rate=rate)
    hand = hand_factory()
    hand.step()
    for _ in range(10):
        hand.step()

    n = 100
    t0 = time.monotonic()
    for _ in range(n):
        hand.step()
    measured = n / (time.monotonic() - t0)

    print(f"\npublisher {rate:g} Hz -> client loop {measured:.1f} Hz")
    # Generous bounds: this asserts pacing exists, not a precise rate.
    assert 0.5 * rate < measured < 1.6 * rate


def test_step_raises_promptly_when_the_node_dies(spawn_mock, hand_factory):
    """
    A hand that disappears mid-run must surface as HandStateTimeout.

    This is the failure a no-op step() could not detect: with no state arriving,
    a client would otherwise loop forever believing the hand simply had not
    moved yet.
    """
    mock = spawn_mock()
    hand = hand_factory(state_timeout=1.0)
    hand.step()
    for _ in range(5):
        hand.step()

    mock.kill()

    t0 = time.monotonic()
    with pytest.raises(HandStateTimeout):
        for _ in range(10_000):
            hand.step()
    elapsed = time.monotonic() - t0

    print(f"\ndetected dead node after {elapsed:.2f} s "
          f"(state_timeout=1.0)")
    # Must be bounded by the timeout, not hang. Allow one timeout's slack for
    # in-flight messages already queued when the process was killed.
    assert elapsed < 3.0


def test_recovery_after_the_node_restarts(spawn_mock, hand_factory):
    """
    DDS should re-discover a replacement publisher without rebuilding the hand.

    Worth knowing for the real system: a driver restart need not mean restarting
    the client.
    """
    mock = spawn_mock()
    hand = hand_factory(state_timeout=2.0)
    hand.step()
    mock.kill()

    with pytest.raises(HandStateTimeout):
        for _ in range(10_000):
            hand.step()

    spawn_mock()  # a fresh node on the same topic
    hand.state_timeout = STARTUP_TIMEOUT
    hand.step()
    assert len(hand.get_joint_positions()) == 9
