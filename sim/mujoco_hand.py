"""
MuJoCo backend for the unified hand interface.

This wraps a MuJoCo model (e.g. the Allegro hand from mujoco_menagerie) and
exposes it through the same HandInterface that the future ROS2/real-hand
backend will also implement. Demo code written against HandInterface
doesn't need to know this is MuJoCo underneath.
"""

import mujoco
import mujoco.viewer

from hand_interface import Finger, HandInterface


class MujocoHand(HandInterface):
    def __init__(self, mjcf_path: str = None, show_viewer: bool = True,
                 model: "mujoco.MjModel" = None):
        """
        Args:
            mjcf_path: path to an MJCF file.
            model: an already-compiled MjModel, used instead of mjcf_path. The
                grasp scenes are assembled in memory (hand XML plus an object
                body) rather than written to disk, so they arrive this way.
        """
        if (mjcf_path is None) == (model is None):
            raise ValueError("pass exactly one of mjcf_path or model")
        self.model = model if model is not None else \
            mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)

        # Build the joint-name list directly from the model, rather than
        # hardcoding it, so this same class works for both the 4F and 3F
        # hands without modification, just point it at a different MJCF file.
        #
        # Report the names of the JOINTS each actuator drives, not the actuator
        # names. The interface's whole purpose is that client code can address
        # the same joints on any backend, and the ROS2 side speaks the URDF
        # joint names (joint_0_0 ...). Reporting actuator names here would make
        # the two backends silently non-interchangeable by name -- ours happen
        # to be called act_0_0 ..., so nothing would match up.
        self._joint_ids = []
        for i in range(self.model.nu):
            if self.model.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
                raise ValueError(
                    f"actuator {i} is not a direct joint transmission; this "
                    "backend assumes one actuator per joint"
                )
            self._joint_ids.append(int(self.model.actuator_trnid[i, 0]))

        self._joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            for jid in self._joint_ids
        ]
        # qpos is not guaranteed to be laid out in actuator order (a free joint
        # or an unactuated joint would break the assumption), so address each
        # joint through its own qpos offset.
        self._qpos_adr = [int(self.model.jnt_qposadr[jid]) for jid in self._joint_ids]

        self._limits = [
            (float(self.model.jnt_range[jid, 0]), float(self.model.jnt_range[jid, 1]))
            for jid in self._joint_ids
        ]
        self._fingers = self._derive_fingers()

        self._viewer = None
        if show_viewer:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _derive_fingers(self):
        """
        Group the actuated joints into fingers using the kinematic tree.

        Derived rather than configured, so the same code describes both hands.
        Every actuated joint is traced up to the child-of-root body its chain
        hangs from -- one such body per finger -- and ordered within the chain by
        depth, giving proximal-to-distal order for free.

        The shallowest joint in each chain is taken as the spread joint. That
        holds for both Allegro variants: the 3F's joint_N_0 and the 4F's ffj0 /
        thj0 are all base rotations, with the curling joints below them. It is a
        structural assumption, not a name match, so it survives the two hands'
        completely different naming schemes -- but it IS an assumption, and a hand
        whose first joint curled would need to override this.
        """
        chains: dict[int, list[tuple[int, int]]] = {}
        for act, jid in enumerate(self._joint_ids):
            body = int(self.model.jnt_bodyid[jid])
            path = []
            while body > 0:
                path.append(body)
                body = int(self.model.body_parentid[body])
            path.reverse()
            root_child = path[0] if len(path) == 1 else path[1]
            chains.setdefault(root_child, []).append((len(path), act))

        fingers = []
        for root_child in sorted(chains):
            ordered = [act for _, act in sorted(chains[root_child])]
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, root_child)
            fingers.append(Finger(
                name=name,
                spread_joint=ordered[0],
                flexion_joints=tuple(ordered[1:]),
            ))
        return tuple(fingers)

    @property
    def joint_names(self):
        return self._joint_names

    @property
    def fingers(self):
        return self._fingers

    @property
    def joint_limits(self):
        return self._limits

    def get_joint_positions(self):
        # data.ctrl gives commanded targets; for *actual* joint position we
        # read data.qpos, indexed through each joint's own qpos address so this
        # stays correct if the model ever gains an unactuated or free joint.
        return [float(self.data.qpos[adr]) for adr in self._qpos_adr]

    def get_joint_efforts(self):
        # actuator_force is the torque the position servo is actually applying,
        # which is what the real hand reports in JointState.effort. One actuator
        # per joint is already enforced in __init__, so the indices line up.
        return [float(self.data.actuator_force[i]) for i in range(self.model.nu)]

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
