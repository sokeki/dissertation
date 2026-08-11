"""
Mock Allegro hand, for exercising the ROS2 backend without hardware.

Split in two deliberately:

  * MockHandDynamics -- pure Python, no ROS imports. First-order lag towards
    the commanded position, optional joint limits. Usable inside the MuJoCo venv
    via LoopbackTransport, which is how the tests run.
  * MockHandNode -- a thin rclpy wrapper that publishes/subscribes the real
    topics, for driving Ros2Hand over an actual ROS2 graph.

The dynamics are the part worth testing, and they are the part that does not
need ROS2 installed. Keeping them separate is what lets the equivalence tests
run at all in this environment (see tests/conftest.py).

Run as a node (needs a sourced ROS2 install; not available in this repo's
environment -- untested against a live graph):

    python3 -m mock_hand --namespace allegroHand_0 --rate 333
"""

from __future__ import annotations

import math
from typing import Sequence

from ros2_hand import (
    CANONICAL_JOINT_NAMES_3F,
    CMD_TOPIC,
    DEFAULT_NAMESPACE,
    STATE_TOPIC,
    require_rclpy,
)

# Publish period. The real rate is uncertain (the 4F ROS2 README says 1 kHz,
# the 3F ROS1 header implies 500 Hz), and nothing in the design depends on it,
# because Ros2Hand.step() paces off message arrival rather than a known rate.
# 3 ms is a plausible middle value for the state publish rate.
DEFAULT_DT = 0.003

# Time constant of the first-order lag. Chosen so the mock's settling time is
# the same order as the MuJoCo model's measured ~0.15 s, which keeps the
# cross-backend trajectory comparison meaningful rather than comparing a step
# function against a physical response.
DEFAULT_TAU = 0.045


class MockHandDynamics:
    """
    Plausible joint_states for a hand that is tracking a position command.

    First-order lag: each joint moves a fixed fraction of its remaining error
    per control period, which is the discrete solution of
    tau * dq/dt = (target - q). No gravity, no contact, no coupling -- this is a
    transport-and-plumbing mock, not a physics model. What it is for is
    exercising the message contract; what it deliberately does not do is
    reproduce MuJoCo's dynamics.

    Note set_target_by_index: the real node reads joint_cmd positions by index
    and ignores the message's name array (allegro_node_grasp.cpp:139). The mock
    copies that behaviour on purpose -- a mock that helpfully mapped by name
    would hide exactly the ordering bug we want to be able to catch.
    """

    def __init__(
        self,
        joint_names: Sequence[str] = CANONICAL_JOINT_NAMES_3F,
        dt: float = DEFAULT_DT,
        tau: float = DEFAULT_TAU,
        limits: Sequence[tuple[float, float]] | None = None,
        initial: Sequence[float] | None = None,
    ):
        self.joint_names = tuple(joint_names)
        n = len(self.joint_names)
        self.dt = dt
        self.tau = tau

        if limits is not None and len(limits) != n:
            raise ValueError(f"expected {n} limit pairs, got {len(limits)}")
        self.limits = [tuple(map(float, lh)) for lh in limits] if limits else None

        start = [0.0] * n if initial is None else [float(v) for v in initial]
        if len(start) != n:
            raise ValueError(f"expected {n} initial positions, got {len(start)}")
        self.positions = self._clamp(start)
        self.target = list(self.positions)

    def _clamp(self, values: Sequence[float]) -> list[float]:
        if self.limits is None:
            return [float(v) for v in values]
        return [
            min(max(float(v), lo), hi)
            for v, (lo, hi) in zip(values, self.limits)
        ]

    def set_target_by_index(self, positions: Sequence[float]) -> None:
        """Accept a joint_cmd. Positional, names ignored -- as the hardware does."""
        if len(positions) != len(self.joint_names):
            raise ValueError(
                f"joint_cmd has {len(positions)} positions, "
                f"expected {len(self.joint_names)}"
            )
        self.target = self._clamp(positions)

    def advance(self, dt: float | None = None) -> None:
        """Advance one control period towards the target."""
        step = self.dt if dt is None else dt
        alpha = 1.0 - math.exp(-step / self.tau)
        self.positions = self._clamp([
            q + alpha * (t - q) for q, t in zip(self.positions, self.target)
        ])

    def state(self) -> tuple[tuple[str, ...], tuple[float, ...]]:
        return self.joint_names, tuple(self.positions)


class MockHandNode:
    """
    rclpy node presenting MockHandDynamics on Wonik's topic contract.

    UNVERIFIED against a live graph -- ROS2 is not installed here. The dynamics
    it wraps are covered by tests; this wrapper is not.
    """

    def __init__(
        self,
        namespace: str = DEFAULT_NAMESPACE,
        dynamics: MockHandDynamics | None = None,
        node_name: str = "mock_allegro_hand",
        qos_depth: int = 1,
        reliability: str = "reliable",
        publish_order: Sequence[str] | None = None,
        include_names: bool = True,
        drop_joints: Sequence[str] = (),
    ):
        """
        Args:
            reliability: QoS reliability for the joint_states publisher. Being
                able to run this as BEST_EFFORT is what lets the tests
                characterise the QoS matching failure mode over a real graph
                instead of predicting it.
            publish_order: emit joint_states in this order rather than canonical,
                to prove the client maps by name over real DDS.
            include_names: if False, publish an empty name array -- the
                degenerate case Ros2Hand refuses to guess at.
            drop_joints: omit these joints from joint_states entirely, to
                exercise the loud-failure path over real DDS.
        """
        rclpy = require_rclpy()
        from sensor_msgs.msg import JointState

        from ros2_hand import qos_profile

        self._rclpy = rclpy
        self._JointState = JointState
        self.dynamics = dynamics or MockHandDynamics()

        self._publish_order = tuple(
            publish_order if publish_order is not None else self.dynamics.joint_names
        )
        unknown = set(self._publish_order) - set(self.dynamics.joint_names)
        if unknown:
            raise ValueError(f"publish_order has unknown joints: {sorted(unknown)}")
        self._include_names = include_names
        self._drop = set(drop_joints)

        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()

        self._node = rclpy.create_node(node_name)
        self._pub = self._node.create_publisher(
            JointState, f"{namespace}/{STATE_TOPIC}",
            qos_profile(qos_depth, reliability),
        )
        # Commands are received RELIABLE regardless: a BEST_EFFORT command
        # subscription would silently drop position targets.
        self._node.create_subscription(
            JointState, f"{namespace}/{CMD_TOPIC}", self._on_cmd,
            qos_profile(qos_depth, "reliable"),
        )
        self._timer = self._node.create_timer(self.dynamics.dt, self._tick)
        self.published = 0

    def _on_cmd(self, msg) -> None:
        self.dynamics.set_target_by_index(msg.position)

    def _tick(self) -> None:
        self.dynamics.advance()
        names, positions = self.dynamics.state()
        by_name = dict(zip(names, positions))
        emitted = [n for n in self._publish_order if n not in self._drop]

        msg = self._JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = list(emitted) if self._include_names else []
        msg.position = [by_name[n] for n in emitted]
        self._pub.publish(msg)
        self.published += 1

    def spin(self) -> None:
        self._rclpy.spin(self._node)

    def close(self) -> None:
        self._node.destroy_node()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Mock Allegro 3F hand node.")
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--rate", type=float, default=1.0 / DEFAULT_DT,
                        help="joint_states publish rate in Hz")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU,
                        help="first-order lag time constant, seconds")
    parser.add_argument("--reliability", default="reliable",
                        choices=("reliable", "best_effort"),
                        help="QoS reliability of the joint_states publisher")
    parser.add_argument("--publish-order", default="canonical",
                        choices=("canonical", "reversed"),
                        help="order to emit joint_states names in")
    parser.add_argument("--omit-names", action="store_true",
                        help="publish an empty JointState.name array")
    parser.add_argument("--drop-joint", action="append", default=[],
                        metavar="NAME", help="omit a joint from joint_states")
    args = parser.parse_args(argv)

    dynamics = MockHandDynamics(dt=1.0 / args.rate, tau=args.tau)
    order = (
        tuple(reversed(dynamics.joint_names)) if args.publish_order == "reversed"
        else dynamics.joint_names
    )

    node = MockHandNode(
        namespace=args.namespace,
        dynamics=dynamics,
        reliability=args.reliability,
        publish_order=order,
        include_names=not args.omit_names,
        drop_joints=args.drop_joint,
    )
    print(
        f"mock hand on {args.namespace}/ at {args.rate:g} Hz, "
        f"{args.reliability}, order={args.publish_order}"
        f"{', names omitted' if args.omit_names else ''}"
        f"{', dropping ' + ','.join(args.drop_joint) if args.drop_joint else ''}"
        "; ctrl-c to stop",
        flush=True,
    )
    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.close()


if __name__ == "__main__":
    main()
