"""
Client code written purely against HandInterface.

This module is the point of the whole interface exercise: nothing in it knows
whether it is driving MuJoCo or a real hand over ROS2. It imports neither
mujoco nor rclpy. The equivalence tests run this same function against both
backends and compare the results, so if anything backend-specific leaks in
here, that claim stops meaning anything.
"""

from __future__ import annotations

from typing import Sequence

from hand_interface import HandInterface

# A short scripted sequence: open, curl one finger, spread, curl all three,
# close back to open. Joint order is joint_0_0 .. joint_8_0, i.e. three fingers
# of (abduction, proximal flexion, distal flexion).
#
# Every one of the nine joints moves at some point, deliberately. The three
# abduction joints (0, 3, 6) are the easiest to leave static, and if they never
# move, a backend that mis-mapped them would still pass every assertion. Their
# limits are asymmetric and differ per finger (joint_0_0 is -1.65..0.1,
# joint_3_0 is -0.1..1.65, joint_6_0 is -0.6..0.6), so the values below are not
# symmetric either -- they are chosen to stay inside each joint's own range.
# Every pose-to-pose change of a given joint is at least 0.25 rad. That is not
# cosmetic: the equivalence test tolerates ~0.10 rad of difference between
# backends (MuJoCo droops against gravity, the mock does not), so any commanded
# motion smaller than a few times that would let a backend which simply failed
# to move still look correct. tests/test_equivalence.py asserts this property
# holds, so shrinking a motion here will fail the suite rather than quietly
# weakening it.
#
# Flexion is capped at 0.3 rad to keep the sequence CONTACT-FREE. The 3F is an
# opposed hand: fingers 1 and 2 flex toward -y while the thumb flexes toward +y,
# so a uniform curl closes them onto each other and self-collides from ~0.4 rad
# onward (measured: 0 contacts at 0.3, 14 at 0.4). That is correct hardware
# behaviour -- it is the hand closing on empty air -- but it makes those poses
# useless for cross-backend comparison, because the mock has no contact model at
# all and cannot reproduce contact-limited motion. Grasping deliberately lives in
# that closed regime; see grasp_controller.py. test_equivalence asserts the
# contact-free property directly, so raising these values fails the suite.
DEMO_POSES: tuple[tuple[float, ...], ...] = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (-0.3, 0.3, 0.3, 0.3, 0.0, 0.0, 0.3, 0.0, 0.0),
    (-0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3),
    (0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.0, 0.3, 0.3),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)


def run_pose_sequence(
    hand: HandInterface,
    poses: Sequence[Sequence[float]] = DEMO_POSES,
    steps_per_pose: int = 200,
) -> dict:
    """
    Command each pose in turn, holding it for `steps_per_pose` control cycles.

    Returns the recorded trajectory and the settled position at the end of each
    pose. Uses only the HandInterface contract: joint_names,
    set_joint_targets, step, get_joint_positions.
    """
    n = len(hand.joint_names)
    for i, pose in enumerate(poses):
        if len(pose) != n:
            raise ValueError(
                f"pose {i} has {len(pose)} values, but this hand has {n} joints"
            )

    trajectory: list[list[float]] = []
    settled: list[list[float]] = []

    for pose in poses:
        hand.set_joint_targets(pose)
        for _ in range(steps_per_pose):
            hand.step()
            trajectory.append(list(hand.get_joint_positions()))
        settled.append(list(hand.get_joint_positions()))

    return {
        "joint_names": list(hand.joint_names),
        "commanded": [list(p) for p in poses],
        "settled": settled,
        "trajectory": trajectory,
    }
