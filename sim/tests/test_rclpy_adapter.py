"""
Wiring checks for RclpyTransport against a stub rclpy.

WHY A STUB. ROS2 is not installed in this environment (see conftest.py), so
RclpyTransport cannot be run against a live graph. Rather than leave it wholly
unexercised, these tests inject a minimal fake rclpy and sensor_msgs into
sys.modules and assert the adapter's observable contract: which topics it
touches, in which namespace, with what QoS depth, which message fields it
populates, and that inbound messages reach Ros2Hand correctly.

WHAT THIS CATCHES: wrong topic names, a missing or malformed namespace, calling
the rclpy API wrongly, forgetting to populate JointState.position, dropping the
header stamp, mis-wiring the subscription callback, and lifecycle mistakes
around init/shutdown.

WHAT THIS CANNOT CATCH, and what therefore remains genuinely untested until
hardware or a real ROS2 install is available:
  * DDS discovery, and whether our QoS actually matches Wonik's publisher.
  * Real timing, message latency, and the true publish rate.
  * Any behaviour of the real node that differs from the contract we inferred.
A green run here is evidence the adapter is wired as intended, NOT evidence it
talks to an Allegro hand.
"""

import sys
import types

import pytest

from ros2_hand import CANONICAL_JOINT_NAMES_3F, Ros2Hand


class FakeStamp:
    def __init__(self, sec=1, nanosec=500_000_000):
        self.sec = sec
        self.nanosec = nanosec


class FakeHeader:
    def __init__(self):
        self.stamp = FakeStamp()


class FakeJointState:
    """Stands in for sensor_msgs.msg.JointState."""

    def __init__(self):
        self.header = FakeHeader()
        self.name = []
        self.position = []
        self.velocity = []
        self.effort = []


class FakeClock:
    def now(self):
        return types.SimpleNamespace(to_msg=lambda: FakeStamp())


class FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


class FakeNode:
    def __init__(self, name):
        self.name = name
        self.publishers = {}
        self.subscriptions = {}
        self.destroyed = False

    def create_publisher(self, msg_type, topic, qos):
        pub = FakePublisher()
        self.publishers[topic] = (msg_type, qos, pub)
        return pub

    def create_subscription(self, msg_type, topic, callback, qos):
        self.subscriptions[topic] = (msg_type, qos, callback)
        return object()

    def get_clock(self):
        return FakeClock()

    def destroy_node(self):
        self.destroyed = True


class FakeReliabilityPolicy:
    """Stands in for rclpy.qos.ReliabilityPolicy."""

    RELIABLE = "RELIABLE"
    BEST_EFFORT = "BEST_EFFORT"


class FakeQoSProfile:
    def __init__(self, depth, reliability=FakeReliabilityPolicy.RELIABLE):
        self.depth = depth
        self.reliability = reliability


@pytest.fixture
def fake_rclpy(monkeypatch):
    """Install a stub rclpy/sensor_msgs into sys.modules for the test duration."""
    state = {"initialised": False, "nodes": [], "spins": [], "shutdowns": 0}

    rclpy = types.ModuleType("rclpy")

    def init(*args, **kwargs):
        state["initialised"] = True

    def ok():
        return state["initialised"]

    def create_node(name):
        node = FakeNode(name)
        state["nodes"].append(node)
        return node

    def spin_once(node, timeout_sec=None):
        state["spins"].append((node, timeout_sec))

    def shutdown():
        state["initialised"] = False
        state["shutdowns"] += 1

    rclpy.init = init
    rclpy.ok = ok
    rclpy.create_node = create_node
    rclpy.spin_once = spin_once
    rclpy.shutdown = shutdown

    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = FakeNode
    qos_mod = types.ModuleType("rclpy.qos")
    qos_mod.QoSProfile = FakeQoSProfile
    qos_mod.ReliabilityPolicy = FakeReliabilityPolicy

    sensor_msgs = types.ModuleType("sensor_msgs")
    msg_mod = types.ModuleType("sensor_msgs.msg")
    msg_mod.JointState = FakeJointState
    sensor_msgs.msg = msg_mod

    for name, mod in [
        ("rclpy", rclpy),
        ("rclpy.node", node_mod),
        ("rclpy.qos", qos_mod),
        ("sensor_msgs", sensor_msgs),
        ("sensor_msgs.msg", msg_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    return state


@pytest.fixture
def transport(fake_rclpy):
    from ros2_hand import RclpyTransport

    return RclpyTransport(namespace="allegroHand_0"), fake_rclpy


def test_topics_use_the_documented_names_and_namespace(transport):
    t, _ = transport
    assert t.cmd_topic == "allegroHand_0/joint_cmd"
    assert t.state_topic == "allegroHand_0/joint_states"


def test_namespace_is_configurable_for_multiple_hands(fake_rclpy):
    """Wonik's launch files enumerate allegroHand_0, allegroHand_1, ..."""
    from ros2_hand import RclpyTransport

    second = RclpyTransport(namespace="allegroHand_1")
    assert second.cmd_topic == "allegroHand_1/joint_cmd"
    assert second.state_topic == "allegroHand_1/joint_states"


def test_publisher_and_subscription_are_created_on_the_right_topics(transport):
    t, state = transport
    node = state["nodes"][-1]
    assert set(node.publishers) == {"allegroHand_0/joint_cmd"}
    assert set(node.subscriptions) == {"allegroHand_0/joint_states"}


def test_qos_depth_is_one(transport):
    """Matches ROS1 queue size 1; see RclpyTransport.QOS_NOTE for the caveat."""
    t, state = transport
    node = state["nodes"][-1]
    _, pub_qos, _ = node.publishers["allegroHand_0/joint_cmd"]
    _, sub_qos, _ = node.subscriptions["allegroHand_0/joint_states"]
    assert pub_qos.depth == 1
    assert sub_qos.depth == 1


def test_default_reliability_is_best_effort_on_state_reliable_on_cmd(transport):
    """
    Pins the defaults the live QoS matrix justified.

    A BEST_EFFORT subscription matches a publisher of either reliability, so it
    is the safe default while Wonik's actual QoS is unknown. Commands go out
    RELIABLE because a dropped target leaves the hand holding a stale pose.
    """
    t, state = transport
    node = state["nodes"][-1]
    _, pub_qos, _ = node.publishers["allegroHand_0/joint_cmd"]
    _, sub_qos, _ = node.subscriptions["allegroHand_0/joint_states"]
    assert sub_qos.reliability == FakeReliabilityPolicy.BEST_EFFORT
    assert pub_qos.reliability == FakeReliabilityPolicy.RELIABLE


def test_reliability_is_configurable(fake_rclpy):
    from ros2_hand import RclpyTransport

    RclpyTransport(reliability="reliable", cmd_reliability="best_effort")
    node = fake_rclpy["nodes"][-1]
    _, pub_qos, _ = node.publishers["allegroHand_0/joint_cmd"]
    _, sub_qos, _ = node.subscriptions["allegroHand_0/joint_states"]
    assert sub_qos.reliability == FakeReliabilityPolicy.RELIABLE
    assert pub_qos.reliability == FakeReliabilityPolicy.BEST_EFFORT


def test_published_command_populates_positions_names_and_stamp(transport):
    t, state = transport
    targets = [0.1 * i for i in range(9)]
    t.publish_cmd(CANONICAL_JOINT_NAMES_3F, targets)

    _, _, pub = state["nodes"][-1].publishers["allegroHand_0/joint_cmd"]
    msg = pub.published[-1]
    assert msg.position == pytest.approx(targets)
    assert msg.name == list(CANONICAL_JOINT_NAMES_3F)
    assert msg.header.stamp is not None


def test_inbound_state_reaches_the_hand_and_is_mapped_by_name(transport):
    """End-to-end through the real Ros2Hand, over the stubbed transport."""
    t, state = transport
    hand = Ros2Hand(transport=t)
    _, _, callback = state["nodes"][-1].subscriptions["allegroHand_0/joint_states"]

    canonical = list(CANONICAL_JOINT_NAMES_3F)
    value = {n: 0.1 * (i + 1) for i, n in enumerate(canonical)}
    shuffled = list(reversed(canonical))

    msg = FakeJointState()
    msg.name = shuffled
    msg.position = [value[n] for n in shuffled]
    callback(msg)

    assert hand.get_joint_positions() == pytest.approx(
        [value[n] for n in canonical]
    )


def test_step_consumes_one_fresh_message_and_spins(transport):
    t, state = transport
    hand = Ros2Hand(transport=t, state_timeout=0.1)
    _, _, callback = state["nodes"][-1].subscriptions["allegroHand_0/joint_states"]

    msg = FakeJointState()
    msg.name = list(CANONICAL_JOINT_NAMES_3F)
    msg.position = [0.0] * 9
    callback(msg)

    hand.step()
    assert state["spins"], "step() must spin the node to service callbacks"


def test_stamp_is_decoded_from_the_header(transport):
    t, state = transport
    _, _, callback = state["nodes"][-1].subscriptions["allegroHand_0/joint_states"]

    msg = FakeJointState()
    msg.name = list(CANONICAL_JOINT_NAMES_3F)
    msg.position = [0.0] * 9
    msg.header.stamp = FakeStamp(sec=7, nanosec=250_000_000)
    callback(msg)

    assert t.latest.stamp == pytest.approx(7.25)


def test_sequence_number_increases_per_message(transport):
    """step()'s freshness check depends on this being monotonic."""
    t, state = transport
    _, _, callback = state["nodes"][-1].subscriptions["allegroHand_0/joint_states"]

    seqs = []
    for _ in range(3):
        msg = FakeJointState()
        msg.name = list(CANONICAL_JOINT_NAMES_3F)
        msg.position = [0.0] * 9
        callback(msg)
        seqs.append(t.latest.seq)

    assert seqs == sorted(set(seqs)) and len(seqs) == 3


def test_init_is_called_when_no_context_exists(fake_rclpy):
    from ros2_hand import RclpyTransport

    assert not fake_rclpy["initialised"]
    RclpyTransport()
    assert fake_rclpy["initialised"]


def test_existing_rclpy_context_is_not_shut_down_on_close(fake_rclpy):
    """
    Do not tear down a context the host application owns.

    A node embedded in a larger ROS2 application must not call rclpy.shutdown()
    on close, or it kills every other node in the process.
    """
    from ros2_hand import RclpyTransport

    fake_rclpy["initialised"] = True  # pretend the host already called init()
    t = RclpyTransport()
    t.close()

    assert fake_rclpy["shutdowns"] == 0
    assert fake_rclpy["initialised"]
    assert fake_rclpy["nodes"][-1].destroyed


def test_owned_context_is_shut_down_on_close(fake_rclpy):
    from ros2_hand import RclpyTransport

    t = RclpyTransport()
    t.close()
    assert fake_rclpy["shutdowns"] == 1


def test_module_imports_without_ros2_but_transport_fails_loudly(monkeypatch):
    """
    The reason rclpy is a lazy import.

    ros2_hand must import in the MuJoCo venv, where rclpy does not exist --
    otherwise the equivalence test could not run at all. But constructing a
    RclpyTransport there must fail with a clear ImportError rather than
    something obscure.

    Blocks rclpy at the finder level so the assertion holds even on a machine
    where ROS2 *is* installed.
    """
    import importlib

    class BlockRclpy:
        def find_module(self, name, path=None):
            return self if name == "rclpy" or name.startswith("rclpy.") else None

        def find_spec(self, name, path=None, target=None):
            if name == "rclpy" or name.startswith("rclpy."):
                raise ImportError(f"{name} blocked for test")
            return None

    for name in [n for n in sys.modules if n == "rclpy" or n.startswith("rclpy.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "meta_path", [BlockRclpy(), *sys.meta_path])

    import ros2_hand

    importlib.reload(ros2_hand)  # must not raise with no ROS2 present
    assert ros2_hand.Ros2Hand is not None

    with pytest.raises(ImportError):
        ros2_hand.RclpyTransport()

    # Leave the module object in a clean state for other tests.
    monkeypatch.undo()
    importlib.reload(ros2_hand)
