"""
Build a grasp scene: a hand model plus one graspable object.

The scene is assembled in memory by parsing the hand's MJCF and injecting an
object body, rather than by writing a file that <include>s the hand. That keeps
mesh paths working for both hands without writing anything into the
mujoco_menagerie checkout, and lets the object's parameters vary per run.

Object mass and friction are explicit parameters with no defaults inherited from
MuJoCo, because both dominate whether a grasp holds and a silent default would
make the results unreproducible.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco

HANDS = {
    "3F": Path("models/allegro_3f/allegro_3f.xml"),
    "4F": Path("mujoco_menagerie/wonik_allegro/right_hand.xml"),
}


@dataclass(frozen=True)
class ObjectSpec:
    """A graspable object. Every physical parameter stated outright."""

    shape: str = "sphere"          # "sphere" or "box"
    size: float = 0.030            # sphere radius, or box half-extent, metres
    mass: float = 0.100            # kg
    friction: float = 1.0          # sliding friction coefficient
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)  # from the grasp centre

    def label(self) -> str:
        return (f"{self.shape}{self.size * 1000:.0f}mm"
                f"/{self.mass * 1000:.0f}g/mu{self.friction:g}")


def _asset_dict(hand_xml: Path) -> dict:
    """Load the meshes a hand model references, for compiling from a string."""
    tree = ET.parse(hand_xml)
    meshdir = tree.getroot().find("compiler")
    sub = meshdir.get("meshdir", ".") if meshdir is not None else "."
    root = (hand_xml.parent / sub).resolve()
    return {p.name: p.read_bytes() for p in root.glob("*.stl")}


def grasp_centre(hand_xml: Path, closure: float = 0.55) -> tuple:
    """
    Where this hand's fingertips converge, found by closing the empty hand.

    Measured rather than hardcoded, so the object is placed correctly for either
    morphology without per-hand constants. The centroid of the fingertip bodies
    at partial closure is a reasonable proxy for where a grasped object sits.

    closure is deliberately partial: at full closure the empty fingers collide
    with each other and the centroid collapses toward the palm.
    """
    import numpy as np

    from mujoco_hand import MujocoHand

    model = mujoco.MjModel.from_xml_string(
        hand_xml.read_text(), _asset_dict(hand_xml)
    )
    hand = MujocoHand(model=model, show_viewer=False)
    limits = hand.joint_limits

    targets = list(hand.get_joint_positions())
    tips = []
    for finger in hand.fingers:
        for j in finger.flexion_joints:
            lo, hi = limits[j]
            neutral = min(max(0.0, lo), hi)
            targets[j] = neutral + closure * (hi - neutral)
        # deepest body in the chain = the fingertip
        deepest, best = None, -1
        for j in finger.flexion_joints:
            jid = hand._joint_ids[j]
            body = int(model.jnt_bodyid[jid])
            depth, b = 0, body
            while b > 0:
                depth += 1
                b = int(model.body_parentid[b])
            if depth > best:
                best, deepest = depth, body
        tips.append(deepest)

    hand.set_joint_targets(targets)
    for _ in range(1500):
        hand.step()
    pts = np.array([hand.data.xpos[b] for b in tips])
    return tuple(float(v) for v in pts.mean(axis=0))


def build_scene(hand_xml: Path, spec: ObjectSpec, centre=None):
    """
    Compile a hand-plus-object model.

    Returns (model, object_body_name).
    """
    if centre is None:
        centre = grasp_centre(hand_xml)
    pos = tuple(c + o for c, o in zip(centre, spec.offset))

    root = ET.fromstring(hand_xml.read_text())
    world = root.find("worldbody")

    # Set the friction on the HAND's collision geoms too, not just the object.
    #
    # MuJoCo combines two geoms' friction by elementwise maximum, so friction set
    # on the object alone is silently overridden by whichever surface is grippier.
    # Both hand models leave their geoms at the default mu=1, so the first version
    # of this sweep produced byte-identical results at mu=0.3, 0.6 and 1.0 -- the
    # parameter was doing nothing at all and the table looked like friction simply
    # did not matter. Setting both surfaces makes the swept value the one that
    # actually governs hand-object contact.
    #
    # Side effect: it also sets finger-to-finger friction, which is not what is
    # being studied but is harmless here, since self-contact is incidental to
    # these grasps rather than load-bearing.
    for geom in root.iter("geom"):
        cls = geom.get("class") or ""
        contype = geom.get("contype")
        is_visual = contype == "0" or "visual" in cls
        if not is_visual:
            geom.set("friction", f"{spec.friction:.6g} 0.005 0.0001")
    # Menagerie carries its collision parameters in a default class rather than
    # on the geoms, so patch the class too or the geoms inherit mu=1 back.
    for default in root.iter("default"):
        if default.get("class") in ("collision", "allegro_right"):
            dg = default.find("geom")
            if dg is None:
                dg = ET.SubElement(default, "geom")
            dg.set("friction", f"{spec.friction:.6g} 0.005 0.0001")

    body = ET.SubElement(world, "body", {
        "name": "object",
        "pos": " ".join(f"{v:.6g}" for v in pos),
    })
    ET.SubElement(body, "freejoint", {"name": "object_free"})
    geom = {
        "name": "object_geom",
        "type": spec.shape,
        "mass": f"{spec.mass:.6g}",
        # condim 6 so the contact model resists twisting as well as sliding;
        # with condim 3 a sphere spins out of a fingertip grasp regardless of mu.
        "condim": "6",
        "friction": f"{spec.friction:.6g} 0.005 0.0001",
        "rgba": "0.8 0.3 0.3 1",
    }
    if spec.shape == "sphere":
        geom["size"] = f"{spec.size:.6g}"
    else:
        geom["size"] = " ".join([f"{spec.size:.6g}"] * 3)
    ET.SubElement(body, "geom", geom)

    # elliptic friction cone with a high impratio: the pyramidal default lets
    # objects creep under tangential load, which shows up as slow slip.
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("cone", "elliptic")
    option.set("impratio", "10")

    xml = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml, _asset_dict(hand_xml)), "object"
