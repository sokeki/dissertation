"""
Test configuration, and the note on which interpreter runs what.

ENVIRONMENT: BOTH IMPORTS WORK IN ONE INTERPRETER

ROS2 Humble is installed at /opt/ros/humble. An earlier probe in this project
concluded it was absent; that probe ran before the install existed (/opt/ros has
a later mtime), so both observations were accurate when made. Note that a
spawned non-interactive shell does not read .bashrc, so `source
/opt/ros/humble/setup.bash` is required per shell before ROS2 is visible.

The two-interpreter problem anticipated in the brief does not exist: the MuJoCo
venv can import rclpy and mujoco together. Why, since it is load-bearing:

  * It is NOT --system-site-packages. venv/pyvenv.cfg has
    include-system-site-packages = false.
  * setup.bash exports PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:
    /opt/ros/humble/local/lib/python3.10/dist-packages, and PYTHONPATH is
    honoured by the interpreter independently of venv site isolation. That flag
    only suppresses the BASE interpreter's site-packages
    (/usr/lib/python3/dist-packages), not PYTHONPATH.
  * The ABI matches. rclpy's extension is
    _rclpy_pybind11.cpython-310-x86_64-linux-gnu.so, and the venv is Python
    3.10.12, built from the same /usr/bin/python3.10 that ROS2 Humble targets.
    A venv on 3.11+ would import-fail on those .so files.

Two consequences worth knowing rather than discovering later:

  * PYTHONPATH sits AHEAD of the venv's site-packages in sys.path, so a package
    shipped by both would resolve to ROS's copy. In practice nothing collides
    here -- numpy still resolves to the venv's 2.2.6 because ROS ships no numpy
    -- but a future `pip install` of something ROS also provides would silently
    lose to /opt/ros.
  * Sourcing ROS2 makes pytest autoload seven ROS plugins, one of which
    (launch_testing) hard-fails on a missing PyYAML and aborted collection
    entirely. Disabled in sim/pytest.ini.

WHAT RUNS WHERE

  * Loopback/stub tests (test_equivalence, test_joint_mapping,
    test_rclpy_adapter): no ROS2 needed, fast and deterministic. Kept that way
    on purpose -- see the Transport seam in ros2_hand.py.
  * test_live_dds.py: real DDS, mock_hand.py in a separate process. Skipped
    automatically when rclpy is unimportable.

The Transport seam still earns its place even though rclpy is now available: it
is what makes the equivalence test deterministic and sub-second, and it lets the
error paths be provoked directly rather than via an adversarial publisher.
"""

import sys
from pathlib import Path

# Tests import the modules flat (`from ros2_hand import ...`), matching the
# existing style of mujoco_hand.py importing hand_interface directly.
SIM_DIR = Path(__file__).resolve().parent.parent
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))
