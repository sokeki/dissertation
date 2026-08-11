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
from typing import Sequence


class HandInterface(ABC):
    """Abstract base class for any controllable robotic hand."""

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
