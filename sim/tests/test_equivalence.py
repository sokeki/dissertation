"""
The project's central claim, as an assertion.

Claim: one piece of client code, written only against HandInterface, drives
both the MuJoCo model and the ROS2 backend, and produces equivalent joint
trajectories.

WHAT "EQUIVALENT" CAN AND CANNOT MEAN HERE

It cannot mean sample-by-sample equality. The two backends genuinely differ:

  * MuJoCo integrates rigid-body dynamics at a 2 ms timestep with gravity,
    contact and a kp=100 position servo. It settles with a steady-state droop
    of up to ~0.06 rad, because gravity torque is balanced against spring
    torque at a small offset.
  * The mock is a kinematic first-order lag at a 3 ms period with no gravity,
    so it settles exactly on the command.

Measured, the per-sample trajectory difference peaks around 0.5 rad mid-
transient. Asserting on that would be asserting the mock reproduces MuJoCo's
physics, which is not the claim and not desirable -- the mock exists to test the
message contract.

So equivalence is asserted on the things that must hold for the interface claim
to be true, each robust to the differing time bases:

  1. Both backends expose the same joint names, in the same order.
  2. Both converge to every commanded pose (within a documented tolerance).
  3. The settled poses agree with each other.
  4. Every significant joint motion goes in the same direction on both, with a
     comparable magnitude -- a time-base-independent statement about shape.
  5. Per-joint trajectories are strongly correlated.

Fine-grained joint-ordering correctness is proved separately, in
test_joint_mapping.py, using distinct per-joint values. It is left out of here
on purpose: DEMO_POSES commands some joints to equal values (both curled
fingers reach 0.6), so this file could not distinguish those from each other.
"""

from pathlib import Path

import numpy as np
import pytest

from hand_client import DEMO_POSES, run_pose_sequence
from mock_hand import MockHandDynamics
from mujoco_hand import MujocoHand
from ros2_hand import LoopbackTransport, Ros2Hand

MODEL = "models/allegro_3f/allegro_3f.xml"

# MuJoCo's measured steady-state droop against gravity at kp=100 peaks at
# 0.072 rad over DEMO_POSES (worst at the 0.9 rad flexion command, where
# gravity torque is largest). 0.10 leaves headroom for a re-tuned gain or a
# changed TARGET_MASS_KG without being so loose that the test stops meaning
# anything: the smallest commanded motion in DEMO_POSES is 0.25 rad, i.e. 2.5x
# this, and test_tolerance_is_not_vacuous asserts that margin holds.
SETTLE_TOL_RAD = 0.10

# Displacements smaller than this are dominated by droop rather than by the
# commanded motion, so direction comparison on them would be noise.
SIGNIFICANT_MOTION_RAD = 0.05


@pytest.fixture(scope="module")
def model_path():
    # Resolved relative to sim/, not to pytest's rootpath, so the suite works
    # whether it is invoked from sim/ or from the repo root.
    path = Path(__file__).resolve().parent.parent / MODEL
    if not path.exists():
        pytest.skip(f"{MODEL} not built; run python sim/build_3f.py")
    return str(path)


@pytest.fixture(scope="module")
def mujoco_result(model_path):
    hand = MujocoHand(model_path, show_viewer=False)
    try:
        return run_pose_sequence(hand)
    finally:
        hand.close()


@pytest.fixture(scope="module")
def ros2_result(model_path):
    """
    Ros2Hand over the in-process mock.

    The mock's joint limits are taken from the MuJoCo model rather than
    hardcoded, so the two backends are constrained identically and the mock
    cannot report a pose the real hand could not reach.

    Its control period is also matched to the MuJoCo timestep. That is not
    fudging the comparison: the mock's period is a property of the mock, not of
    the real hand (whose rate we do not know -- see ros2_hand's contract notes),
    and leaving them mismatched at 3 ms vs 2 ms means the two runs cover
    different amounts of simulated time per step. The transients then sit at
    different sample indices and the correlation check measures that offset
    rather than trajectory shape -- it cost ~0.03 of correlation on the worst
    joint. Matching the period removes the confound so the check measures what
    it claims to.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(model_path)
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                          int(model.actuator_trnid[i, 0]))
        for i in range(model.nu)
    ]
    limits = [
        (float(model.jnt_range[int(model.actuator_trnid[i, 0]), 0]),
         float(model.jnt_range[int(model.actuator_trnid[i, 0]), 1]))
        for i in range(model.nu)
    ]
    mock = MockHandDynamics(
        joint_names=names, limits=limits, dt=float(model.opt.timestep)
    )
    hand = Ros2Hand(joint_names=names, transport=LoopbackTransport(mock))
    try:
        return run_pose_sequence(hand)
    finally:
        hand.close()


def test_client_code_is_backend_agnostic():
    """
    The shared client must not import either backend.

    If hand_client ever imports mujoco or rclpy, "the same client code runs on
    both" degrades into a claim about a module that secretly knows which one it
    is talking to.
    """
    import hand_client

    source = open(hand_client.__file__).read()
    assert "import mujoco" not in source
    assert "import rclpy" not in source
    assert "mujoco_hand" not in source
    assert "ros2_hand" not in source


def test_both_backends_report_identical_joint_names(mujoco_result, ros2_result):
    """Claim 1. Without this, 'the same client code' cannot address both."""
    assert mujoco_result["joint_names"] == ros2_result["joint_names"]
    assert mujoco_result["joint_names"] == [
        "joint_0_0", "joint_1_0", "joint_2_0",
        "joint_3_0", "joint_4_0", "joint_5_0",
        "joint_6_0", "joint_7_0", "joint_8_0",
    ]


def test_same_client_code_ran_the_same_commands(mujoco_result, ros2_result):
    """Guards the premise: both runs must have issued identical commands."""
    assert mujoco_result["commanded"] == ros2_result["commanded"]
    assert mujoco_result["commanded"] == [list(p) for p in DEMO_POSES]


@pytest.mark.parametrize("backend", ["mujoco", "ros2"])
def test_backend_converges_to_every_commanded_pose(
    backend, mujoco_result, ros2_result
):
    """Claim 2."""
    result = {"mujoco": mujoco_result, "ros2": ros2_result}[backend]
    settled = np.array(result["settled"])
    commanded = np.array(result["commanded"])

    err = np.abs(settled - commanded)
    worst = np.unravel_index(err.argmax(), err.shape)
    assert err.max() < SETTLE_TOL_RAD, (
        f"{backend} failed to reach pose {worst[0]} on "
        f"{result['joint_names'][worst[1]]}: err {err.max():.4f} rad"
    )


def test_settled_poses_agree_across_backends(mujoco_result, ros2_result):
    """Claim 3 -- the core of the equivalence claim."""
    a = np.array(mujoco_result["settled"])
    b = np.array(ros2_result["settled"])
    assert a.shape == b.shape

    diff = np.abs(a - b)
    worst = np.unravel_index(diff.argmax(), diff.shape)
    assert diff.max() < SETTLE_TOL_RAD, (
        f"backends disagree at pose {worst[0]} on "
        f"{mujoco_result['joint_names'][worst[1]]}: {diff.max():.4f} rad"
    )


def test_every_significant_motion_agrees_in_direction(mujoco_result, ros2_result):
    """
    Claim 4. Time-base independent: compares pose-to-pose displacement rather
    than per-sample values, so it is unaffected by the backends' different
    control periods.

    A finger moving the wrong way, or a swapped pair of joints with differing
    commands, shows up here even when the endpoints happen to look plausible.
    """
    a = np.diff(np.array(mujoco_result["settled"]), axis=0)
    b = np.diff(np.array(ros2_result["settled"]), axis=0)

    significant = np.abs(b) > SIGNIFICANT_MOTION_RAD
    assert significant.sum() > 0, "sequence commands no significant motion"

    same_direction = np.sign(a[significant]) == np.sign(b[significant])
    assert same_direction.all(), (
        f"{(~same_direction).sum()} of {significant.sum()} significant motions "
        "moved in opposite directions between backends"
    )
    # Magnitudes must also be comparable, not merely same-signed. The bound is
    # 2x the settle tolerance because a displacement is a difference of two
    # settled poses, so it carries both poses' droop.
    assert np.abs(a - b)[significant].max() < 2 * SETTLE_TOL_RAD


def test_per_joint_trajectories_are_strongly_correlated(mujoco_result, ros2_result):
    """
    Claim 5. Correlation, not equality: it tests that each joint follows the
    same shape over the run without requiring the transients to match.
    """
    a = np.array(mujoco_result["trajectory"])
    b = np.array(ros2_result["trajectory"])
    assert a.shape == b.shape

    for j, name in enumerate(mujoco_result["joint_names"]):
        # Every joint moves in DEMO_POSES, so no joint has zero variance and
        # the correlation is always well defined.
        assert a[:, j].std() > 1e-6 and b[:, j].std() > 1e-6, (
            f"{name} never moves; the sequence cannot validate its mapping"
        )
        # Measured minimum over DEMO_POSES is 0.923 (worst joint); the residual
        # gap from 1.0 is the different transient shape -- MuJoCo's second-order
        # servo response against the mock's first-order lag -- which is expected
        # and not something the mock should be reproducing.
        corr = np.corrcoef(a[:, j], b[:, j])[0, 1]
        assert corr > 0.9, f"{name} trajectories correlate only {corr:.3f}"


def test_pose_sequence_is_contact_free(model_path):
    """
    The precondition the cross-backend comparison rests on.

    The mock has no contact model, so it cannot reproduce contact-limited
    motion. If DEMO_POSES commanded the fingers into each other -- which on this
    opposed 3F hand happens from about 0.4 rad of uniform flexion -- MuJoCo would
    stall against self-collision while the mock sailed to the target, and the
    disagreement would be blamed on gravity droop. Asserted here so raising a
    flexion value in DEMO_POSES fails loudly instead.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    for i, pose in enumerate(DEMO_POSES):
        data.ctrl[:] = pose
        for _ in range(1200):
            mujoco.mj_step(model, data)
        assert data.ncon == 0, (
            f"pose {i} produces {data.ncon} contacts; the sequence must stay "
            "contact-free for the mock comparison to be valid"
        )


def test_tolerance_is_not_vacuous():
    """
    Keeps the equivalence tolerance honest.

    A tolerance loose enough to swallow the commanded motions would make every
    assertion above pass trivially. Assert the smallest commanded displacement
    in the sequence is comfortably larger than the tolerance, so a backend that
    simply failed to move, or moved to the wrong pose, still fails.
    """
    poses = np.array(DEMO_POSES)
    steps = np.abs(np.diff(poses, axis=0))
    smallest = steps[steps > 1e-9].min()
    assert smallest > 2 * SETTLE_TOL_RAD, (
        f"smallest commanded motion {smallest:.3f} rad is not comfortably "
        f"above the {SETTLE_TOL_RAD} rad tolerance; the test could pass "
        "without either backend moving correctly"
    )


def test_a_mismapped_backend_would_fail_this_suite(model_path):
    """
    Negative control: prove the equivalence assertions can actually fail.

    A passing equivalence suite is only meaningful if a broken backend fails
    it. Here the mock publishes its state in a permuted order while claiming
    canonical names, which is precisely the silent finger-swap the design is
    meant to prevent, and the settled-agreement assertion must catch it.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(model_path)
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                          int(model.actuator_trnid[i, 0]))
        for i in range(model.nu)
    ]

    class MislabellingTransport(LoopbackTransport):
        """Reports finger 1's positions under finger 2's names, and vice versa."""

        def _emit(self):
            super()._emit()
            p = list(self._latest.positions)
            p[1], p[4] = p[4], p[1]
            p[2], p[5] = p[5], p[2]
            object.__setattr__(self._latest, "positions", tuple(p))

    mock = MockHandDynamics(joint_names=names)
    hand = Ros2Hand(joint_names=names, transport=MislabellingTransport(mock))
    broken = run_pose_sequence(hand, steps_per_pose=200)

    a = np.array(broken["settled"])
    c = np.array(broken["commanded"])
    assert np.abs(a - c).max() > SETTLE_TOL_RAD, (
        "a deliberately mis-mapped backend passed the convergence check, so "
        "the equivalence suite cannot detect finger swaps"
    )
