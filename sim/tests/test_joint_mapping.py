"""
Joint ordering and the loud-failure requirement.

JointState carries a name array and there is no guarantee the hand publishes in
the order we index by, so every one of these tests is about refusing to trust
index alignment. A silent mis-mapping here would command the wrong finger,
which is the worst possible failure mode: plausible-looking motion that is
subtly wrong.
"""

import pytest

from mock_hand import MockHandDynamics
from ros2_hand import (
    CANONICAL_JOINT_NAMES_3F,
    JointNameMismatch,
    HandStateTimeout,
    LoopbackTransport,
    Ros2Hand,
    StateSample,
    Transport,
)


class StubTransport(Transport):
    """Transport that replays a fixed StateSample, for exercising decode paths."""

    def __init__(self, sample=None):
        self.sample = sample
        self.published = []

    def publish_cmd(self, names, positions):
        self.published.append((tuple(names), tuple(positions)))

    def poll(self, timeout):
        pass

    @property
    def latest(self):
        return self.sample


def make_sample(names, positions, seq=1):
    return StateSample(seq=seq, names=tuple(names), positions=tuple(positions))


def test_positions_are_mapped_by_name_not_by_index():
    """A hand publishing in a different order must still read out correctly."""
    # Reverse the publish order, and give each joint a value keyed to its name
    # so a mis-map is detectable rather than coincidentally right.
    canonical = list(CANONICAL_JOINT_NAMES_3F)
    value = {name: 0.1 * (i + 1) for i, name in enumerate(canonical)}
    shuffled = list(reversed(canonical))

    hand = Ros2Hand(transport=StubTransport(
        make_sample(shuffled, [value[n] for n in shuffled])
    ))

    got = hand.get_joint_positions()
    assert got == pytest.approx([value[n] for n in canonical])
    # Sanity: had we trusted index order, we would have got the reverse.
    assert got != pytest.approx([value[n] for n in shuffled])


def test_interleaved_order_is_mapped_correctly():
    """Not just reversal -- an arbitrary permutation must map correctly too."""
    canonical = list(CANONICAL_JOINT_NAMES_3F)
    value = {name: float(i) for i, name in enumerate(canonical)}
    permuted = [canonical[i] for i in (4, 0, 8, 2, 6, 1, 7, 3, 5)]

    hand = Ros2Hand(transport=StubTransport(
        make_sample(permuted, [value[n] for n in permuted])
    ))
    assert hand.get_joint_positions() == pytest.approx(list(range(9)))


def test_missing_joint_fails_loudly():
    """A joint_states short of one of our joints must raise, not pad or guess."""
    names = list(CANONICAL_JOINT_NAMES_3F)[:-1]
    hand = Ros2Hand(transport=StubTransport(make_sample(names, [0.0] * len(names))))

    with pytest.raises(JointNameMismatch) as exc:
        hand.get_joint_positions()
    # The message must name the offender, or debugging on hardware is guesswork.
    assert "joint_8_0" in str(exc.value)


def test_extra_joints_are_tolerated():
    """An aggregated /joint_states carrying arm joints too is legitimate."""
    canonical = list(CANONICAL_JOINT_NAMES_3F)
    names = ["arm_shoulder", *canonical, "arm_wrist"]
    positions = [99.0, *[0.25] * len(canonical), 98.0]

    hand = Ros2Hand(transport=StubTransport(make_sample(names, positions)))
    assert hand.get_joint_positions() == pytest.approx([0.25] * 9)


def test_empty_name_array_fails_loudly_by_default():
    """
    Refuse to guess when the publisher sends no names.

    Wonik's node does populate names (allegro_node.cpp:33), but it resizes the
    array before filling it, so a partially-initialised publisher sending empty
    names is a plausible real failure. Guessing canonical order here would be a
    silent finger mis-assignment.
    """
    hand = Ros2Hand(transport=StubTransport(make_sample([], [0.3] * 9)))
    with pytest.raises(JointNameMismatch, match="empty name array"):
        hand.get_joint_positions()


def test_empty_name_array_allowed_when_explicitly_opted_in():
    hand = Ros2Hand(
        transport=StubTransport(make_sample([], [0.3] * 9)),
        allow_unnamed_states=True,
    )
    assert hand.get_joint_positions() == pytest.approx([0.3] * 9)


def test_unnamed_state_with_wrong_length_still_fails():
    hand = Ros2Hand(
        transport=StubTransport(make_sample([], [0.3] * 7)),
        allow_unnamed_states=True,
    )
    with pytest.raises(JointNameMismatch):
        hand.get_joint_positions()


def test_names_positions_length_mismatch_fails():
    hand = Ros2Hand(transport=StubTransport(
        make_sample(CANONICAL_JOINT_NAMES_3F, [0.0] * 5)
    ))
    with pytest.raises(JointNameMismatch, match="names but"):
        hand.get_joint_positions()


def test_targets_are_published_in_canonical_order():
    """
    The hand reads joint_cmd by index (allegro_node_grasp.cpp:139), so the
    published position array must be in canonical order regardless of anything
    else, and the name array must agree with it.
    """
    transport = StubTransport()
    hand = Ros2Hand(transport=transport)
    targets = [0.1 * i for i in range(9)]
    hand.set_joint_targets(targets)

    names, positions = transport.published[-1]
    assert names == CANONICAL_JOINT_NAMES_3F
    assert positions == pytest.approx(targets)


def test_wrong_number_of_targets_is_rejected():
    hand = Ros2Hand(transport=StubTransport())
    with pytest.raises(ValueError, match="Expected 9 targets"):
        hand.set_joint_targets([0.0] * 8)


def test_reading_before_any_state_arrives_raises():
    hand = Ros2Hand(transport=StubTransport(None))
    with pytest.raises(HandStateTimeout):
        hand.get_joint_positions()


def test_duplicate_joint_names_rejected_at_construction():
    with pytest.raises(ValueError, match="duplicate joint names"):
        Ros2Hand(joint_names=["a", "b", "a"], transport=StubTransport())


def test_end_to_end_mapping_through_a_reordering_mock():
    """
    The integration version of the ordering claim: a mock that publishes its
    state in a non-canonical order, driven through the real Ros2Hand.

    The mock consumes commands by index like the hardware, so this only comes
    out right if the write path uses canonical order AND the read path maps by
    name.
    """
    mock = MockHandDynamics(tau=0.01, dt=0.01)
    reversed_order = list(reversed(CANONICAL_JOINT_NAMES_3F))
    hand = Ros2Hand(transport=LoopbackTransport(mock, publish_order=reversed_order))

    target = [0.0, 0.5, -0.5, 0.0, 0.4, -0.4, 0.0, 0.3, -0.3]
    hand.set_joint_targets(target)
    for _ in range(400):
        hand.step()

    assert hand.get_joint_positions() == pytest.approx(target, abs=1e-3)
    # And the mock's own internal order agrees, i.e. nothing was reversed twice.
    assert list(mock.positions) == pytest.approx(target, abs=1e-3)


def test_step_times_out_when_no_state_arrives():
    """A dead hand node must surface promptly, not hang or look idle."""

    class SilentTransport(StubTransport):
        def poll(self, timeout):
            pass  # never delivers anything

    hand = Ros2Hand(transport=SilentTransport(None), state_timeout=0.05)
    with pytest.raises(HandStateTimeout, match="no fresh joint_states"):
        hand.step()


def test_step_requires_a_genuinely_fresh_sample():
    """
    A stale sample that never updates must not satisfy step().

    This is what separates blocking-step from no-op-step: repeatedly returning
    the same state would let a client loop believe the hand is advancing.
    """

    class FrozenTransport(StubTransport):
        def poll(self, timeout):
            pass  # sample exists, but its seq never increases

    hand = Ros2Hand(
        transport=FrozenTransport(make_sample(CANONICAL_JOINT_NAMES_3F, [0.0] * 9)),
        state_timeout=0.05,
    )
    hand.step()  # first call consumes the existing sample
    with pytest.raises(HandStateTimeout):
        hand.step()  # second must not accept the same one again
