"""
ROS2 backend for the unified hand interface.

Drives Wonik's Allegro Hand node over its JointState topic contract, so client
code written against HandInterface can target the real hand without change.

--------------------------------------------------------------------------
TOPIC CONTRACT: WHAT IS READ FROM SOURCE vs WHAT IS INFERRED
--------------------------------------------------------------------------
We do not have Wonik's ROS2 package (allegro_hand_ros2_v5) locally -- only the
3F ROS1 package in wonik_3f_reference/. Everything below is marked with where
it came from, because the untested assumptions are the ones most likely to bite
when real hardware appears.

READ FROM THE 3F ROS1 SOURCE (wonik_3f_reference/src/allegro_hand_controllers):

  * Topic names, from src/allegro_node.h:33-35 --
        allegroHand/joint_states   (published by hand)
        allegroHand/joint_cmd      (subscribed by hand)
    These match the 4F ROS2 names, so the naming survived the ROS1->ROS2 port.

  * Joint names and canonical order, from src/allegro_node.cpp:9-14 --
        joint_0_0 .. joint_8_0  (DOF_JOINTS == 9 for the 3F)
    Identical to the names in the URDF and therefore to our MuJoCo model.

  * joint_states DOES populate the name array: src/allegro_node.cpp:33 sets
    current_joint_state.name[i] = jointNames[i]. So name-based mapping on the
    read path is supported, not a guess.

  * joint_cmd is consumed BY INDEX, NOT BY NAME. src/allegro_node_grasp.cpp:139
        for (int i = 0; i < DOF_JOINTS; i++)
            desired_position[i] = msg.position[i];
    The name array on an inbound command is ignored entirely. This makes the
    contract ASYMMETRIC and it drives the design below: on the write path we
    MUST emit canonical index order, and a name array is decoration. We still
    populate it, for `ros2 topic echo` readability and so a future node that
    does respect names sees consistent data -- but it buys no safety.

  * Namespacing pattern, from launch/allegro_hand.launch:66-67 -- topics are
    remapped into an enumerated allegroHand_<NUM>/ namespace, hence the
    `namespace` constructor argument.

INFERRED / UNVERIFIED -- re-check against the real node before trusting:

  * QoS profile. ROS1 used queue size 1 (allegro_node.cpp:70). We use depth-1
    KEEP_LAST with rclpy's default RELIABLE reliability. If Wonik's ROS2 node
    publishes joint_states BEST_EFFORT, a RELIABLE subscription will not match
    it and we will receive nothing -- which surfaces as a step() timeout, not
    silent breakage. See RclpyTransport.QOS_NOTE.
  * Control rate. The 4F ROS2 README quotes 1 kHz. The 3F ROS1 header defines
    ALLEGRO_CONTROL_TIME_INTERVAL 0.002 (500 Hz), and the publish rate is not
    necessarily the control rate. We therefore never hardcode a rate: step()
    derives its pacing from message arrival (see below).
  * Whether the 3F has a ROS2 driver at all. Wonik ship ROS2 for the 4F; the
    3F package we have is ROS1. If the 3F ROS2 node differs, topic names are
    the first thing to re-check.
  * allegroHand/tactile_sensors (std_msgs/Int32MultiArray) is part of the 4F
    ROS2 contract but is not exposed here -- HandInterface has no tactile
    concept yet, and inventing one unused is worse than leaving it out.

--------------------------------------------------------------------------
WHY THERE IS A TRANSPORT SEAM
--------------------------------------------------------------------------
Ros2Hand talks to a Transport rather than to rclpy directly. Two reasons, one
practical and one about testing:

  * rclpy and mujoco cannot be imported into the same interpreter here (see
    README notes / the module docstring of tests/conftest.py). Making rclpy a
    lazily-imported detail of one Transport implementation means this module
    imports fine without ROS2 installed at all.
  * It lets the equivalence tests run the real Ros2Hand logic -- name mapping,
    ordering, step() semantics, error paths -- against an in-process mock hand
    with no ROS graph, no DDS, and no wall-clock sleeping.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from hand_interface import HandInterface

# Canonical joint order for the 3F hand, from allegro_node.cpp:9-14.
# This is the order the hardware indexes joint_cmd positions by.
CANONICAL_JOINT_NAMES_3F = (
    "joint_0_0", "joint_1_0", "joint_2_0",
    "joint_3_0", "joint_4_0", "joint_5_0",
    "joint_6_0", "joint_7_0", "joint_8_0",
)

DEFAULT_NAMESPACE = "allegroHand_0"
CMD_TOPIC = "joint_cmd"
STATE_TOPIC = "joint_states"


def require_rclpy():
    """
    Import rclpy with an actionable message if it is missing.

    Worth the wrapper because a bare ModuleNotFoundError here is misleading:
    the usual cause is not a missing pip package but an unsourced ROS2 install,
    or running under the MuJoCo venv, which cannot see rclpy at all.
    """
    try:
        import rclpy
        from rclpy.node import Node  # noqa: F401
        from rclpy.qos import QoSProfile  # noqa: F401
        from sensor_msgs.msg import JointState  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"rclpy is unavailable ({exc}). This code path needs a sourced ROS2 "
            "install:\n"
            "    source /opt/ros/humble/setup.bash\n"
            "and must run under the ROS2 interpreter, not the MuJoCo venv "
            "(rclpy ships with ROS, not PyPI, so pip cannot supply it).\n"
            "To test the backend without ROS2, use LoopbackTransport with "
            "mock_hand.MockHandDynamics -- see sim/tests/README.md."
        ) from exc
    return rclpy


RELIABILITY_CHOICES = ("best_effort", "reliable")


def qos_profile(depth: int = 1, reliability: str = "best_effort"):
    """
    Build a QoSProfile. Separated out so Ros2Hand and the mock node cannot
    drift apart, and so tests can construct the same profiles.
    """
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    if reliability not in RELIABILITY_CHOICES:
        raise ValueError(
            f"reliability must be one of {RELIABILITY_CHOICES}, got {reliability!r}"
        )
    policy = (
        ReliabilityPolicy.RELIABLE if reliability == "reliable"
        else ReliabilityPolicy.BEST_EFFORT
    )
    return QoSProfile(depth=depth, reliability=policy)


class HandStateTimeout(RuntimeError):
    """No fresh joint_states arrived within the timeout."""


class JointNameMismatch(ValueError):
    """The hand's joint_states names do not cover our expected joints."""


@dataclass(frozen=True)
class StateSample:
    """One received joint_states message, plus a local arrival sequence number."""

    seq: int
    names: tuple[str, ...]
    positions: tuple[float, ...]
    stamp: float | None = None


class Transport(ABC):
    """
    Message plumbing for Ros2Hand, so the backend logic is testable off-graph.

    Implementations are responsible only for moving messages; all joint
    ordering and validation lives in Ros2Hand.
    """

    @abstractmethod
    def publish_cmd(self, names: Sequence[str], positions: Sequence[float]) -> None:
        """Publish a joint_cmd. `positions` is already in canonical order."""

    @abstractmethod
    def poll(self, timeout: float) -> None:
        """Process inbound messages for up to `timeout` seconds."""

    @property
    @abstractmethod
    def latest(self) -> StateSample | None:
        """Most recent joint_states received, or None if none yet."""

    def close(self) -> None:
        pass


class Ros2Hand(HandInterface):
    """
    HandInterface implementation backed by Wonik's JointState topics.

    Args:
        joint_names: canonical joint order. Defaults to the 3F hand's 9 joints.
            This is the order set_joint_targets/get_joint_positions use, and
            the order positions are published in (which the hardware requires
            -- it reads joint_cmd by index).
        namespace: topic namespace, e.g. "allegroHand_0". Wonik's launch files
            enumerate these so several hands can coexist.
        transport: injected Transport. Defaults to RclpyTransport.
        state_timeout: how long step() waits for a fresh joint_states before
            raising HandStateTimeout.
        allow_unnamed_states: if True, accept a joint_states whose name array is
            empty by assuming it is already in canonical order. Defaults to
            False: silently mis-assigning finger joints is the exact failure
            this class exists to prevent. Only enable against a node you have
            confirmed publishes canonical order.
    """

    def __init__(
        self,
        joint_names: Sequence[str] = CANONICAL_JOINT_NAMES_3F,
        namespace: str = DEFAULT_NAMESPACE,
        transport: Transport | None = None,
        state_timeout: float = 1.0,
        allow_unnamed_states: bool = False,
    ):
        self._joint_names = tuple(joint_names)
        if len(set(self._joint_names)) != len(self._joint_names):
            raise ValueError(f"duplicate joint names: {self._joint_names}")

        self.namespace = namespace
        self.state_timeout = state_timeout
        self.allow_unnamed_states = allow_unnamed_states

        if transport is None:
            transport = RclpyTransport(namespace=namespace)
        self._transport = transport

        # Cache the name->canonical permutation, keyed by the incoming name
        # tuple, so a stable publisher costs one lookup per message rather than
        # a rebuild. Re-derived automatically if the publisher's order changes.
        self._order_cache: dict[tuple[str, ...], tuple[int, ...]] = {}
        self._last_seq = -1

    @property
    def joint_names(self) -> Sequence[str]:
        return self._joint_names

    def _permutation(self, names: tuple[str, ...]) -> tuple[int, ...]:
        """
        Indices into an incoming position array, in our canonical joint order.

        Extra joints in the message are ignored -- an aggregated /joint_states
        carrying an arm plus the hand is legitimate. Missing ones are fatal.
        """
        cached = self._order_cache.get(names)
        if cached is not None:
            return cached

        lookup = {name: i for i, name in enumerate(names)}
        missing = [n for n in self._joint_names if n not in lookup]
        if missing:
            raise JointNameMismatch(
                f"joint_states from {self.namespace}/{STATE_TOPIC} is missing "
                f"{len(missing)} expected joint(s): {missing}. "
                f"Received names: {list(names)}. Expected: {list(self._joint_names)}"
            )

        perm = tuple(lookup[n] for n in self._joint_names)
        self._order_cache[names] = perm
        return perm

    def _decode(self, sample: StateSample) -> tuple[float, ...]:
        """Map a received sample into canonical joint order."""
        names, positions = sample.names, sample.positions

        if not names:
            if not self.allow_unnamed_states:
                raise JointNameMismatch(
                    f"joint_states from {self.namespace}/{STATE_TOPIC} has an "
                    "empty name array, so positions cannot be mapped by name. "
                    "Pass allow_unnamed_states=True to assume canonical order "
                    "instead -- only if you have verified the publisher's order."
                )
            if len(positions) != len(self._joint_names):
                raise JointNameMismatch(
                    f"unnamed joint_states has {len(positions)} positions, "
                    f"expected {len(self._joint_names)}"
                )
            return tuple(float(p) for p in positions)

        if len(positions) != len(names):
            raise JointNameMismatch(
                f"joint_states has {len(names)} names but "
                f"{len(positions)} positions"
            )

        perm = self._permutation(names)
        return tuple(float(positions[i]) for i in perm)

    def get_joint_positions(self) -> Sequence[float]:
        sample = self._transport.latest
        if sample is None:
            raise HandStateTimeout(
                f"no joint_states received yet on "
                f"{self.namespace}/{STATE_TOPIC}; call step() first, or check "
                "the hand node is running and its QoS is compatible"
            )
        return list(self._decode(sample))

    def set_joint_targets(self, targets: Sequence[float]) -> None:
        targets = list(targets)
        if len(targets) != len(self._joint_names):
            raise ValueError(
                f"Expected {len(self._joint_names)} targets, got {len(targets)}"
            )
        # Canonical order is mandatory here: the hand reads joint_cmd by index
        # and ignores these names (allegro_node_grasp.cpp:139).
        self._transport.publish_cmd(self._joint_names, targets)

    def step(self) -> None:
        """
        Block until a joint_states newer than the last one we saw arrives.

        WHY BLOCK, rather than no-op or sleep(dt):

        HandInterface's contract is that a client can loop
            set_joint_targets(...); step()
        and have get_joint_positions() afterwards reflect a state that has
        moved on. MujocoHand satisfies that by construction, because mj_step
        integrates. The ROS2 backend has no simulation to advance -- Wonik's
        node runs its own control loop -- so the only way to honour the same
        post-condition is to wait for the hand's next state publication.

        The alternatives are both worse:

          * No-op. get_joint_positions() would return whatever the last
            callback happened to leave behind, so a client loop would spin as
            fast as the CPU allows, flooding joint_cmd, and would read the same
            stale state many times in a row. Worse, a client that loops until a
            pose is reached would either spin uselessly or, if no state ever
            arrived, never notice -- a dead hand node would look like a hand
            that simply is not moving yet.
          * sleep(fixed dt). Requires guessing the hand's rate, which we
            explicitly do not know (the 4F README says 1 kHz, the 3F ROS1
            header implies 500 Hz, and neither is necessarily the publish
            rate). It would also drift out of phase with the real publisher.

        Blocking on arrival makes the loop self-pacing at whatever rate the
        hand actually publishes, gives the same "state has advanced" guarantee
        as mj_step, and turns a dead or QoS-mismatched node into a prompt, loud
        HandStateTimeout instead of a silent hang.

        The cost is that step() is not real-time bounded -- it is as slow as the
        hand is. That is the honest behaviour for a backend whose clock belongs
        to someone else, and it is why state_timeout exists.
        """
        deadline = time.monotonic() + self.state_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._transport.poll(remaining)
            sample = self._transport.latest
            if sample is not None and sample.seq > self._last_seq:
                self._last_seq = sample.seq
                return

        raise HandStateTimeout(
            f"no fresh joint_states on {self.namespace}/{STATE_TOPIC} within "
            f"{self.state_timeout}s. Is the hand node running? If it is, check "
            "QoS compatibility (see RclpyTransport.QOS_NOTE)."
        )

    def close(self) -> None:
        self._transport.close()


class RclpyTransport(Transport):
    """
    Real ROS2 transport. rclpy is imported lazily so this module stays
    importable in the MuJoCo venv, where rclpy does not exist.

    NOTE: unlike the rest of this file, this class has not been exercised
    against a live ROS2 graph -- ROS2 is not installed in this environment.
    tests/test_rclpy_adapter.py verifies its wiring against a stub rclpy
    (topic names, namespacing, message fields, callback plumbing), which covers
    the API-shape mistakes but NOT DDS discovery, QoS negotiation or real
    timing. Treat first contact with hardware as the real test.
    """

    QOS_NOTE = (
        "Default: depth-1 KEEP_LAST, BEST_EFFORT on the state subscription and "
        "RELIABLE on the command publisher. Measured against a live graph: a "
        "RELIABLE subscription does NOT receive from a BEST_EFFORT publisher "
        "(the request/offered contract is violated, so DDS never matches them "
        "and step() times out), whereas a BEST_EFFORT subscription receives "
        "from a publisher of either reliability. Since Wonik's actual QoS is "
        "unknown, BEST_EFFORT is the subscription default because it is the "
        "one that matches either way. Override with `reliability=`."
    )

    def __init__(
        self,
        namespace: str = DEFAULT_NAMESPACE,
        node_name: str = "hand_interface_client",
        qos_depth: int = 1,
        reliability: str = "best_effort",
        cmd_reliability: str = "reliable",
    ):
        """
        Args:
            qos_depth: KEEP_LAST history depth. 1 matches ROS1's queue size.
            reliability: "best_effort" or "reliable", for the joint_states
                subscription. See QOS_NOTE -- best_effort matches a publisher of
                either kind, reliable only matches a RELIABLE publisher.
            cmd_reliability: reliability for the joint_cmd publisher. RELIABLE by
                default: a dropped position command leaves the hand holding a
                stale target, and commands are low-rate compared to state.
        """
        rclpy = require_rclpy()
        from rclpy.node import Node
        from sensor_msgs.msg import JointState

        self._rclpy = rclpy
        self._JointState = JointState

        # Do not clobber an rclpy context the host application already owns.
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()

        self._node: Node = rclpy.create_node(node_name)

        self.cmd_topic = f"{namespace}/{CMD_TOPIC}"
        self.state_topic = f"{namespace}/{STATE_TOPIC}"
        self.reliability = reliability
        self.cmd_reliability = cmd_reliability

        self._pub = self._node.create_publisher(
            JointState, self.cmd_topic, qos_profile(qos_depth, cmd_reliability)
        )
        self._sub = self._node.create_subscription(
            JointState, self.state_topic, self._on_state,
            qos_profile(qos_depth, reliability),
        )

        self._latest: StateSample | None = None
        self._seq = 0

    def _on_state(self, msg) -> None:
        self._seq += 1
        stamp = None
        header = getattr(msg, "header", None)
        if header is not None and getattr(header, "stamp", None) is not None:
            stamp = header.stamp.sec + header.stamp.nanosec * 1e-9
        self._latest = StateSample(
            seq=self._seq,
            names=tuple(msg.name),
            positions=tuple(msg.position),
            stamp=stamp,
        )

    def publish_cmd(self, names: Sequence[str], positions: Sequence[float]) -> None:
        msg = self._JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        # Advisory only -- the hand reads positions by index and ignores these.
        msg.name = list(names)
        msg.position = [float(p) for p in positions]
        self._pub.publish(msg)

    def poll(self, timeout: float) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=max(timeout, 0.0))

    @property
    def latest(self) -> StateSample | None:
        return self._latest

    def close(self) -> None:
        self._node.destroy_node()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()


class LoopbackTransport(Transport):
    """
    In-process transport wired straight to a MockHandDynamics.

    No ROS graph, no DDS, no sleeping: poll() advances the mock by one control
    period and synthesises the joint_states message the real node would have
    published. That makes tests deterministic and hardware-free, and exercises
    all of Ros2Hand's real logic.

    `publish_order` deliberately controls the order the synthesised state's
    name array is emitted in, so tests can prove Ros2Hand maps by name instead
    of trusting index alignment.
    """

    def __init__(self, mock, publish_order: Sequence[str] | None = None,
                 include_names: bool = True):
        self._mock = mock
        self._publish_order = (
            tuple(publish_order) if publish_order is not None
            else tuple(mock.joint_names)
        )
        unknown = set(self._publish_order) - set(mock.joint_names)
        if unknown:
            raise ValueError(f"publish_order names not on the mock: {sorted(unknown)}")
        self._include_names = include_names
        self._latest: StateSample | None = None
        self._seq = 0
        self.published_cmds: list[tuple[tuple[str, ...], tuple[float, ...]]] = []
        self._emit()

    def _emit(self) -> None:
        state = dict(zip(self._mock.joint_names, self._mock.positions))
        self._seq += 1
        self._latest = StateSample(
            seq=self._seq,
            names=self._publish_order if self._include_names else (),
            positions=tuple(state[n] for n in self._publish_order),
        )

    def publish_cmd(self, names: Sequence[str], positions: Sequence[float]) -> None:
        self.published_cmds.append((tuple(names), tuple(positions)))
        # Mirrors the hardware: consumed by index, names ignored.
        self._mock.set_target_by_index(positions)

    def poll(self, timeout: float) -> None:
        del timeout  # deterministic: one control period per poll, no waiting
        self._mock.advance()
        self._emit()

    @property
    def latest(self) -> StateSample | None:
        return self._latest
