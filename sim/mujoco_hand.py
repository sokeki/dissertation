"""
MuJoCo backend for the unified hand interface.

This wraps a MuJoCo model (e.g. the Allegro hand from mujoco_menagerie) and
exposes it through the same HandInterface that the future ROS2/real-hand
backend will also implement. Demo code written against HandInterface
doesn't need to know this is MuJoCo underneath.
"""

import mujoco
import mujoco.viewer

from hand_interface import HandInterface


class MujocoHand(HandInterface):
    def __init__(self, mjcf_path: str, show_viewer: bool = True):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)

        # Build the joint-name list directly from the model, rather than
        # hardcoding it, so this same class works for both the 4F and 3F
        # hands without modification, just point it at a different MJCF file.
        self._joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]

        self._viewer = None
        if show_viewer:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    @property
    def joint_names(self):
        return self._joint_names

    def get_joint_positions(self):
        # data.ctrl gives commanded targets; for *actual* joint position we
        # read data.qpos. For a hand with one joint per actuator (as here),
        # qpos lines up with the actuator order directly.
        return list(self.data.qpos[: self.model.nu])

    def set_joint_targets(self, targets):
        if len(targets) != len(self._joint_names):
            raise ValueError(
                f"Expected {len(self._joint_names)} targets, got {len(targets)}"
            )
        for i, value in enumerate(targets):
            self.data.ctrl[i] = value

    def step(self):
        mujoco.mj_step(self.model, self.data)
        if self._viewer is not None:
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
