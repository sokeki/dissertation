"""
Unified hand control interface.

This defines the abstract contract that any hand controller must satisfy,
regardless of whether it's driving a simulated hand in MuJoCo or a real
hand over ROS2, and regardless of whether that hand has 3 or 4 fingers.

Higher-level code (grasp demos, evaluation scripts) should only ever talk
to this interface, never to MuJoCo or ROS2 directly. That's what makes the
interface "unified": the same demo script works unmodified against any
backend that implements this class correctly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Finger:
    """
    One finger's joints, as indices into the hand's joint_names order.

    Morphology that joint names alone cannot express. A grasp controller has to
    know which joints belong to the same finger and which of them curl it,
    because a 3-finger and a 4-finger hand do not close the same way; without
    this it can only command every joint identically, which is not a grasp.

    spread_joint is the base rotation that aims the finger rather than curling
    it (abduction on the fingers, opposition on a thumb), or None if the finger
    has none. flexion_joints are the curling joints, ordered proximal to distal.
    """

    name: str
    spread_joint: int | None
    flexion_joints: tuple[int, ...]


class HandInterface(ABC):
    """Abstract base class for any controllable robotic hand."""

    @property
    def fingers(self) -> Sequence[Finger]:
        """
        The hand's fingers. Override in any backend used for grasping.

        Not abstract, because pose-replay clients (see hand_client) do not need
        it and predate it; a backend that cannot describe its morphology should
        fail loudly here rather than force every backend to invent an answer.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not describe its fingers; a grasp "
            "controller needs them to know which joints curl which finger"
        )

    def get_joint_efforts(self) -> Sequence[float]:
        """
        Applied torque at each joint, N*m, in joint_names order.

        Not invented for simulation: this is already part of the real hand's
        contract. Wonik's node fills sensor_msgs/JointState.effort on
        allegroHand/joint_states (allegro_node.cpp:110,
        current_joint_state.effort[i] = desired_torque[i]), so a ROS2 backend can
        supply it from the same message it already reads positions from.

        Needed because a grasp controller that only sees position cannot tell
        "touching firmly" from "crushing": both look like a position the finger
        failed to reach. See grasp_controller for what goes wrong without it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not report joint efforts"
        )

    @property
    def joint_limits(self) -> Sequence[tuple[float, float]]:
        """
        (lower, upper) for each joint, in joint_names order.

        Needed so a controller can express "close this finger" in terms of each
        joint's own travel instead of hardcoded angles, which is what lets one
        controller drive hands with different ranges.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose joint limits"
        )

    @property
    @abstractmethod
    def joint_names(self) -> Sequence[str]:
        """
        Ordered list of joint names this hand controls.

        For the 4F Allegro hand this will have 16 entries (4 per finger);
        for the 3F hand, however many the real hardware specifies. Calling
        code should never assume a fixed length -- always check this.
        """
        raise NotImplementedError

    @abstractmethod
    def get_joint_positions(self) -> Sequence[float]:
        """
        Return the current position of every joint, in the same order as
        joint_names. Units should be radians, matching ROS2's JointState
        convention, so sim and real values are directly comparable.
        """
        raise NotImplementedError

    @abstractmethod
    def set_joint_targets(self, targets: Sequence[float]) -> None:
        """
        Command every joint to move toward the given target positions.

        'targets' must be the same length as joint_names, and in the same
        order. This call should return immediately -- it requests a move,
        it does not block until the move completes.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self) -> None:
        """
        Advance the controller by one control cycle.

        For MuJoCo, this calls mj_step(). For the real hand, this is likely
        a no-op or a small sleep, since the real hardware's control loop
        runs independently in its own ROS2 node. Having both backends
        implement step() lets demo code use the same loop structure either
        way: set_joint_targets(...), then step(), repeatedly.
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Optional cleanup hook (closing a viewer window, shutting down a
        ROS2 node, etc). Default implementation does nothing, so backends
        that don't need cleanup aren't forced to override this.
        """
        pass
