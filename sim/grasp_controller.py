"""
A grasp controller written against HandInterface only.

Imports neither mujoco nor rclpy, and never asks how many joints or fingers the
hand has. It runs unmodified on the 9-DOF 3F model and the 16-DOF 4F Menagerie
model.

--------------------------------------------------------------------------
HOW A GRASP GENERALISES ACROSS MORPHOLOGIES
--------------------------------------------------------------------------
The design problem is that a 3-finger and a 4-finger hand do not close the same
way, so "replay a joint trajectory" cannot transfer -- a 9-vector means nothing
to a 16-DOF hand, and even per-joint normalisation would close a thumb into a
palm. Three abstractions were considered:

1. Joint-space trajectory replay. Rejected: hardcodes the joint count, which the
   brief rules out, and encodes one hand's geometry.

2. Fingertip-pose targets with inverse kinematics. This is the most capable
   option and the most morphology-neutral in principle: "put three fingertips on
   the object surface" transfers to any hand with three or more fingers.
   Rejected here because it needs an IK solver plus per-hand reachability, and
   because HandInterface is a position-command interface with no Cartesian
   concept -- adding one would mean the abstraction under test was doing the work
   rather than the interface. Worth revisiting if grasp quality becomes the
   goal rather than interface transfer.

3. Per-finger normalised closure with force-regulated stopping. Chosen.

Closure is one scalar per finger, c in [0, 1], mapped onto that finger's own
flexion joints between their neutral and fully-flexed limits. The hand supplies
the finger grouping and the joint limits (HandInterface.fingers,
HandInterface.joint_limits), so the same command means "curl this finger fully"
on a 3-jointed 3F finger and a 3-jointed 4F finger below its base rotation
alike, without the controller knowing either number.

--------------------------------------------------------------------------
WHY FORCE REGULATION, AND WHAT REPLACED WHAT
--------------------------------------------------------------------------
The first version closed each finger until it STALLED -- until the gap between
commanded and measured position exceeded a threshold learned from free-space lag.
That worked as contact detection and transferred across both hands, but it was a
bad way to decide how hard to squeeze, and the measured result was damning: grip
forces of 142 N median and 1299 N peak against a 0.98 N object. Stalling only
tells you the finger has stopped; by the time the gap is unambiguous the servo is
already commanding a large error, and force is whatever kp times that error
happens to be. Success was partly achieved by crushing, so the success rate was
not measuring grasp quality.

What is used instead: each finger drives its own applied joint torque toward a
setpoint. Advance closure while torque is below setpoint, retract while above.
Contact is never explicitly detected -- it does not need to be. In free space
torque stays low and the finger keeps closing; on contact torque rises and the
finger settles where the setpoint is met. The loop is a bounded integrator on
force, so the steady state is a force, not a position.

WHY THIS DOES NOT REINTRODUCE A HAND-SPECIFIC CONSTANT. The setpoint
(GRIP_EFFORT_NM) is a TASK quantity -- how hard to hold a 100 g object -- and is
the same value for both hands. Servo gain, which is what leaked before, is
absorbed automatically: reaching a given torque needs a position offset of
tau/kp, and the loop finds that offset by feedback rather than being told the
gain. A hand with kp=1 simply ends up with a larger offset than one with kp=100
and the same grip force. That is strictly better than the calibration it
replaces, which measured a per-hand quantity and could only ever bound error, not
force.

REJECTED ALTERNATIVES:

  * Capping torque with the actuator's forcerange. Bounds the worst case without
    making the controller force-aware -- the finger still commands maximum torque,
    just a lower maximum -- so grip force would be set by whatever limit the model
    happens to declare. It is also not portable: our 3F declares 15 N*m (itself
    probably a placeholder) while Menagerie's 4F declares no limit at all, so the
    same controller would squeeze completely differently on each. Kept only as a
    passive safety net, not as the mechanism.

  * Stopping on a measured contact-force threshold. Closer to right, but a
    threshold still leaves the final force undetermined: it stops when force
    EXCEEDS a bound, so the force at rest is the bound plus whatever the servo
    winds up before it reacts. Regulating to a setpoint pins the value instead of
    bounding it, for the same feedback signal.

  * Backing off a fixed fraction of closure once contact is established. Needs a
    per-hand fraction to translate into force, which is exactly the leak this
    change exists to avoid.

WHERE THE FORCE SIGNAL COMES FROM. HandInterface.get_joint_efforts, which the
real hand already publishes in JointState.effort. Its weakness is that joint
torque is not fingertip force: the two differ by a lever arm that changes with
pose, so a fixed torque setpoint gives a force that varies somewhat with finger
configuration. Bounding fingertip force directly would need either tactile
sensing or a kinematic Jacobian, neither of which HandInterface exposes. The
evaluation harness measures actual contact forces independently, so this
approximation shows up in the reported numbers rather than hiding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from hand_interface import HandInterface


# Target applied joint torque during a grasp, N*m. Same value for both hands:
# this is a property of the task, not of the hardware.
#
# DERIVATION, for the 100 g object used throughout the evaluation (W = 0.98 N):
# resisting that weight by friction alone at n contacts with coefficient mu needs
# a normal force per contact of at least W/(mu*n). At the worst friction tested
# (mu=0.3) and the minimum contact count the success criterion accepts (n=2), that
# is 0.98/(0.3*2) = 1.6 N. A safety factor of ~3 for the fact that contact normals
# are not aligned with the load gives ~5 N per contact. At a fingertip lever arm
# of roughly 50 mm (measured: the 3F medial link is 54 mm, the 4F's 38 mm), 5 N
# corresponds to 0.25 N*m of joint torque.
#
# TARGET FORCE RANGE this is meant to produce: single-digit newtons of total
# normal force, i.e. roughly 2-20 N for a 100 g object -- a few times the object's
# weight, which is what a person uses to hold a small ball without deforming it.
# grasp_eval reports the measured distribution against a 20 N sane bound, so
# whether this derivation actually lands there is a reported number, not a claim.
GRIP_EFFORT_NM = 0.25


@dataclass
class GraspConfig:
    """Parameters of a closing motion. Deliberately all explicit."""

    # Torque setpoint each finger regulates to. See GRIP_EFFORT_NM.
    grip_effort: float = GRIP_EFFORT_NM

    # Maximum closure change per control cycle. The loop scales down from this as
    # torque approaches the setpoint, so it is a speed limit, not a fixed step.
    closure_rate: float = 0.002

    # Gain of the retraction response when torque overshoots the setpoint,
    # relative to closure_rate. Above 1 so the loop backs off faster than it
    # advances, which keeps overshoot from persisting: squeezing too hard is the
    # failure being fixed, so the asymmetry is deliberate.
    retract_gain: float = 2.0

    # Torque within this fraction of the setpoint counts as regulated.
    effort_tol_frac: float = 0.15

    # Consecutive regulated cycles before a finger is considered done.
    hold_cycles: int = 50

    # Where the spread joints are held, as a fraction from each joint's neutral
    # pose toward its upper limit. 0 keeps the fingers in their neutral splay.
    # This is the parameter that does NOT transfer between the two hands -- see
    # the note in the module docstring of grasp_eval.py.
    spread_fraction: float = 0.0

    # Hard cap on closing cycles, so a hand that never regulates still terminates.
    max_cycles: int = 3000

    # Cycles to hold the final grasp before the harness measures it, letting
    # contact forces and the object settle.
    settle_cycles: int = 400


@dataclass
class GraspResult:
    """What the closing motion did. Purely from HandInterface observations."""

    cycles: int
    closure: list[float]
    regulated: list[bool]
    final_effort: list[float]
    peak_effort: list[float]
    setpoint: float = 0.0
    joint_names: list[str] = field(default_factory=list)
    finger_names: list[str] = field(default_factory=list)

    @property
    def all_regulated(self) -> bool:
        return bool(self.regulated) and all(self.regulated)


def _neutral(lo: float, hi: float) -> float:
    """The open pose: zero if the joint's travel includes it, else the nearest
    limit. The 4F thumb's opposition joint (0.263..1.396) has no zero, so this
    cannot assume one."""
    return min(max(0.0, lo), hi)


class GraspController:
    """
    Closes a hand around whatever is in front of it, one closure scalar per
    finger, stopping each finger when it stalls.
    """

    def __init__(self, hand: HandInterface, config: GraspConfig | None = None):
        self.hand = hand
        self.config = config or GraspConfig()
        self.fingers = list(hand.fingers)
        self.limits = list(hand.joint_limits)
        if not self.fingers:
            raise ValueError("hand reports no fingers")

        n = len(hand.joint_names)
        # Start from the open pose, spread joints at their configured preshape.
        self._targets = [_neutral(*self.limits[i]) for i in range(n)]
        for finger in self.fingers:
            if finger.spread_joint is not None:
                lo, hi = self.limits[finger.spread_joint]
                neutral = _neutral(lo, hi)
                self._targets[finger.spread_joint] = (
                    neutral + self.config.spread_fraction * (hi - neutral)
                )

    def open_pose(self) -> list[float]:
        return list(self._targets)

    def _flexion_target(self, joint: int, closure: float) -> float:
        """Map a closure fraction onto one joint's own travel.

        Closing is toward the upper limit on both Allegro variants: every flexion
        joint on both has a positive upper limit and a near-zero or negative
        lower one. A hand that flexed the other way would need a per-joint sign,
        which is exactly the kind of morphology detail HandInterface.fingers
        exists to carry.
        """
        lo, hi = self.limits[joint]
        neutral = _neutral(lo, hi)
        return neutral + closure * (hi - neutral)

    def close(self, record: list | None = None) -> GraspResult:
        """
        Close each finger until its applied torque reaches the setpoint.

        Per finger, per cycle: measure the largest applied torque across its
        flexion joints, then move closure toward whichever direction reduces the
        gap to the setpoint. Advancing slows as the setpoint is approached so the
        loop settles instead of hunting; retracting is faster than advancing
        (retract_gain) because overshooting the setpoint is the failure mode this
        design exists to prevent.

        No contact detection and no stall threshold: a finger in free space sees
        low torque and keeps closing, a finger on the object sees torque rise and
        stops where the setpoint is met. Terminates when every finger has held
        within tolerance for hold_cycles, or at max_cycles.
        """
        cfg = self.config
        nf = len(self.fingers)
        closure = [0.0] * nf
        in_band = [0] * nf
        regulated = [False] * nf
        peak = [0.0] * nf
        setpoint = cfg.grip_effort
        tol = setpoint * cfg.effort_tol_frac

        self.hand.set_joint_targets(self._targets)

        cycle = 0
        while cycle < cfg.max_cycles:
            efforts = list(self.hand.get_joint_efforts())

            for fi, finger in enumerate(self.fingers):
                eff = max(
                    (abs(efforts[j]) for j in finger.flexion_joints), default=0.0
                )
                peak[fi] = max(peak[fi], eff)
                gap = setpoint - eff

                if abs(gap) <= tol:
                    in_band[fi] += 1
                    if in_band[fi] >= cfg.hold_cycles:
                        regulated[fi] = True
                    continue
                in_band[fi] = 0
                regulated[fi] = False

                if gap > 0:
                    # Below setpoint: keep closing, decelerating as torque builds
                    # so the approach is asymptotic rather than a step onto the
                    # object.
                    step = cfg.closure_rate * min(1.0, gap / max(setpoint, 1e-9))
                    closure[fi] = min(1.0, closure[fi] + step)
                else:
                    step = cfg.closure_rate * cfg.retract_gain * min(
                        1.0, -gap / max(setpoint, 1e-9)
                    )
                    closure[fi] = max(0.0, closure[fi] - step)

                for j in finger.flexion_joints:
                    self._targets[j] = self._flexion_target(j, closure[fi])

            self.hand.set_joint_targets(self._targets)
            self.hand.step()
            if record is not None:
                record.append(list(self.hand.get_joint_positions()))
            cycle += 1

            if all(regulated):
                break

        # Hold, so contact forces and the object settle before measurement. The
        # loop keeps running here: releasing regulation during the hold would let
        # the torque drift back to whatever the frozen position commands.
        for _ in range(cfg.settle_cycles):
            efforts = list(self.hand.get_joint_efforts())
            for fi, finger in enumerate(self.fingers):
                eff = max(
                    (abs(efforts[j]) for j in finger.flexion_joints), default=0.0
                )
                peak[fi] = max(peak[fi], eff)
                gap = setpoint - eff
                if abs(gap) > tol:
                    if gap > 0:
                        closure[fi] = min(1.0, closure[fi] + cfg.closure_rate
                                          * min(1.0, gap / max(setpoint, 1e-9)))
                    else:
                        closure[fi] = max(0.0, closure[fi] - cfg.closure_rate
                                          * cfg.retract_gain
                                          * min(1.0, -gap / max(setpoint, 1e-9)))
                    for j in finger.flexion_joints:
                        self._targets[j] = self._flexion_target(j, closure[fi])
            self.hand.set_joint_targets(self._targets)
            self.hand.step()
            if record is not None:
                record.append(list(self.hand.get_joint_positions()))

        efforts = list(self.hand.get_joint_efforts())
        final = [
            max((abs(efforts[j]) for j in f.flexion_joints), default=0.0)
            for f in self.fingers
        ]
        return GraspResult(
            cycles=cycle,
            closure=closure,
            regulated=regulated,
            final_effort=[round(e, 4) for e in final],
            peak_effort=[round(e, 4) for e in peak],
            setpoint=setpoint,
            joint_names=list(self.hand.joint_names),
            finger_names=[f.name for f in self.fingers],
        )
