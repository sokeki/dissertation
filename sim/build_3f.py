#!/usr/bin/env python3
"""
Build a MuJoCo model of the Wonik Allegro Hand V5 3F from Wonik's ROS URDF.

Run:  python sim/build_3f.py
Out:  sim/models/allegro_3f/allegro_3f.xml  (+ assets/*.stl)

The conversion is a two-stage compile rather than a direct XML translation,
because of three things about how MuJoCo treats URDF:

1. MuJoCo's URDF parser ignores <actuator>, and MJCF cannot <include> a URDF,
   so there is no way to attach actuators to a URDF-sourced model in place.
   Instead we compile the URDF, export the equivalent native MJCF with
   mj_saveLastXML, and inject the actuators into that.

2. The URDF's inertials are placeholders -- every link, from the palm down to
   the fingertip, is 0.4154 kg with a uniform 1e-4 tensor. Those numbers are
   physically inconsistent with the link geometry and make the solver diverge,
   so we discard them and let MuJoCo compute inertia from the meshes instead
   (inertiafromgeom="true"). Deriving mass from geometry alone overshoots the
   real hand ~2x, though, because MuJoCo fills each mesh with solid material
   at a uniform density while the real hand is a hollow shell around motors
   and voids. We correct for that by calibrating one global density so the
   model's total mass matches the published figure -- see TARGET_MASS_KG.

3. palm_link has no joint, so it welds to the world body. MuJoCo's
   parent/child contact filter deliberately does not apply when the parent is
   the world body, so the finger bases are *not* filtered against the palm and
   start ~24 mm interpenetrated -- the fingers get blasted apart on the first
   step. We add explicit <contact><exclude> pairs, the same fix
   mujoco_menagerie applies to the 4F hand.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

REPO = Path(__file__).resolve().parent.parent
URDF_REPO = REPO / "wonik_3f_reference"
URDF_PKG = URDF_REPO / "src/allegro_hand_description"
URDF_SRC = URDF_PKG / "allegro_hand_description_3F.urdf"
MESH_SRC = URDF_PKG / "meshes"
LICENSE_SRC = URDF_REPO / "LICENSE"

OUT_DIR = REPO / "sim/models/allegro_3f"
OUT_XML = OUT_DIR / "allegro_3f.xml"
ASSET_DIR = OUT_DIR / "assets"
STAGED_URDF = OUT_DIR / "_staged.urdf"
MANIFEST = OUT_DIR / "manifest.json"
LICENSE_OUT = OUT_DIR / "LICENSE.wonik"

# Upstream source of the URDF and meshes, pinned so this build is reproducible
# even if the upstream repo moves, is force-pushed, or disappears. Verified at
# build time; pass --allow-upstream-drift to build against a different commit
# (and then update this constant if the new one is intended).
UPSTREAM_REPO = "https://github.com/Wonikrobotics-git/allegro_hand_ros_v5-3Finger.git"
UPSTREAM_COMMIT = "40608536f0f2ca43fa78078ebf1a3fd07098427f"

# Finger bases, whose parent is the world-welded palm. See docstring note 3.
# The 3F hand is 9 DOF: three fingers, three actuated joints each.
PALM = "palm_link"
FINGER_BASES = ("link_0_0", "link_3_0", "link_6_0")

# Collision primitive to fit per mesh, replacing MuJoCo's convex hull of the
# visual mesh. See fit_primitives for why, and for what it does not fix.
# f4 is the fingertip: a capsule, since its rounded end is the main contact
# surface and a box corner would misrepresent it. Everything else is a box,
# matching mujoco_menagerie's treatment of the 4F hand.
#
# Sizes are NOT fitted automatically. An axis-aligned fit to the mesh vertices
# was tried first and made things worse: the finger-base mesh (f1) is authored
# with a large offset that includes palm-side housing, spanning 64 mm in z for a
# link whose kinematic length is 42 mm, so its AABB came out at 178 cm^3 -- more
# than double its own 84 cm^3 convex hull. Fitting a box to that inflates rather
# than corrects, and it drove the finger base from 33 g to 128 g.
#
# So each primitive is authored from two sources instead:
#   * length along the link's long axis = the joint-to-joint distance from the
#     URDF (link_0_0 -> 41.75 mm in +y; link_1_0 and link_2_0 -> 54.25 mm in +z),
#     which is the one dimension the kinematics pins exactly;
#   * cross-section = the mesh's p25-p75 vertex spread on the other two axes,
#     which excludes the offset housing material without guessing.
#
# Menagerie's numbers are deliberately NOT copied. Its cross-section
# (19.6 x 27.5 mm) is the V4 4F finger; measured here, the V5 3F proximal link is
# ~30 x 29 mm, so the V5 fingers are genuinely chunkier and copying V4 would have
# modelled the wrong hardware. What is borrowed is the approach: boxes for
# phalanges, a capsule for the fingertip, collision on group 3.
#
# Resulting volume vs the true triangle volume: 1.24x (tip) to 1.7x (base),
# 1.46x overall, against ~5.2x for the hulls MuJoCo was using. Since total mass
# is pinned to TARGET_MASS_KG, this changes the distribution between links, not
# the total.
PRIMITIVES = {
    # palm: slab under the finger row. p5-p95 in x/z, trimmed to exclude the
    # finger-base housings that the mesh includes at high y.
    "body": {"type": "box", "size": "0.030 0.029 0.0375", "pos": "0 0.005 -0.0425"},
    # finger base: long axis +y, 41.75 mm to the flexion joint.
    "f1": {"type": "box", "size": "0.012 0.020875 0.012", "pos": "0 0.020875 0"},
    # proximal: long axis +z, 54.25 mm to the medial joint.
    "f2": {"type": "box", "size": "0.014 0.013 0.027125", "pos": "0 0 0.027125"},
    # medial: long axis +z. NOTE this one is shorter than its 54.25 mm
    # joint-to-joint distance. The validator below caught the discrepancy: the
    # mesh only reaches z=51.2 mm, because the last few mm of the physical link
    # are covered by the fingertip cap rather than the medial shell. Clamped to
    # the mesh, since a box longer than the part it represents would put the
    # medial link's contact surface where there is no material.
    "f3": {"type": "box", "size": "0.013 0.010 0.0256", "pos": "0 0 0.0256"},
    # fingertip: the primary contact surface, so a capsule. Radius from the
    # mesh's x/y spread (12.5 mm); length and offset set to reproduce its z span
    # of -19..+11 mm exactly.
    "f4": {"type": "capsule", "size": "0.0125 0.0025", "pos": "0 0 -0.004"},
}

# Menagerie's convention: group 3 for collision, so the viewer's default view
# shows the visual meshes (group 1) and collision can be toggled on separately.
COLLISION_GROUP = "3"

# Total mass of the real hand, used to calibrate geom density (see calibrate).
# Nominally the published Allegro Hand V5 (3F) figure of 1,050 g.
#
# UNVERIFIED -- TWO SEPARATE DOUBTS, THE SECOND MORE SERIOUS:
#
# (a) Source quality. The figure comes from secondary spec aggregators and
#     reseller listings, not from Wonik's own datasheet, which we do not have.
#
# (b) Which variant it describes. Some listings present 1,050 g as "the V5
#     spec" without saying whether it is the 3F or the 4F. If it is actually
#     the 4F number, the 3F should be roughly one finger lighter: at this
#     model's per-finger mass of 162.7 g that would be ~890 g, meaning this
#     model is calibrated ~18% heavy. Circumstantial support for that reading:
#     the older 16-DOF V4 is quoted at 1.08 kg, which is suspiciously close to
#     1,050 g for a hand carrying an entire extra finger.
#
# Re-verify against the Wonik user manual if it becomes available, and prefer a
# figure that explicitly names the 3F variant.
#
# IF THIS NUMBER IS WRONG, the fix is this one constant: mass is linear in
# density, so the solve just rescales. But everything derived moves with it --
# every per-link mass and inertia tensor scales by the same factor, and the
# tuned KP below no longer corresponds to the measured tracking error, since
# gravity torque changes while joint damping does not. Re-run the kp sweep and
# update both this constant and KP's comment together. Use --target-mass to try
# an alternative without editing the file.
TARGET_MASS_KG = 1.050

# Total mass must land within this of the target, or the build fails. The
# density solve is exact (mass is linear in density), so this only needs to
# absorb float noise; it is not a modelling fudge factor.
MASS_TOL_KG = 1e-6

# Density used for the calibration probe compile. Its value is arbitrary and
# does not reach the output -- we compile once at this density, measure the
# resulting total, and solve for the density that hits TARGET_MASS_KG. Probing
# rather than scaling MuJoCo's default avoids hardcoding that default.
PROBE_DENSITY = 1000.0

# JOINT DYNAMICS OVERRIDE -- the URDF's third set of placeholders.
#
# Every one of the nine joints carries an identical <dynamics damping="3"
# friction="10"/> and <limit effort="15" velocity="7"/>. That uniformity is the
# same tell as the 0.4154 kg / 1e-4 inertials: a real hand's proximal and distal
# joints do not share a friction figure to three significant figures.
#
# frictionloss=10 N*m is the damaging one. It is 67% of the joint's own stated
# 15 N*m effort limit, which no working actuator would spend on static friction,
# and it makes the joint immovable below 10 N*m. Consequences measured before
# this override:
#   * a position servo needs 10/kp = 0.1 rad of error just to break friction, so
#     the grasp controller's free-space calibration returned 0.135 rad -- it was
#     measuring friction breakaway, not contact lag;
#   * commanding past that produced 13.5 N*m, 90% of the torque limit, and
#     contact forces of 142-1299 N against a 0.98 N object;
#   * no force-aware controller can regulate grip below 10 N*m, so the crushing
#     was not fixable in the controller alone.
#
# Replaced with mujoco_menagerie's values for the same hardware family (the 4F
# model uses damping=0.1, frictionloss=0), which is the closest thing to an
# independent reference we have. Zero rather than a small non-zero friction
# because we have no measurement to justify any particular value, and zero is the
# assumption that does not silently impose a force floor.
#
# STILL UNVERIFIED: the 15 N*m effort limit itself is probably also placeholder.
# Published Allegro fingertip force is a few newtons, which at a ~50 mm lever is
# well under 1 N*m, so 15 N*m looks ~20x high. Left as-is because the
# force-regulated controller no longer relies on the limit to bound grip force --
# but it should be replaced with a measured figure if one becomes available.
JOINT_DAMPING = 0.1
JOINT_FRICTIONLOSS = 0.0

# Position servo gain.
#
# Was 100, chosen when the URDF's damping=3 made the servo heavily overdamped.
# Removing the placeholder frictionloss and dropping damping to 0.1 changes that
# completely: at kp=100 the joints now overshoot by 0.85 rad, which is unusable,
# so the gain had to come down regardless of the grasp work.
#
# kp=1 matches mujoco_menagerie's 4F model for the same hardware family, which is
# the nearest independent reference. Measured here: 0.032 rad overshoot, 0.032 rad
# steady-state error against gravity at a 0.3 rad target, and free-space actuator
# effort of ~0.4 N*m rather than the 11 N*m kp=100 now draws.
#
# A side benefit worth stating, since the last grasp comparison was confounded by
# it: the two hand models now have identical servo tuning (kp=1, damping 0.1,
# frictionloss 0), so a difference in grasp success between them is attributable
# to morphology rather than to one hand being commanded 100x more stiffly.
KP = 1.0

EXPECTED_ACTUATORS = 9


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upstream_state() -> dict:
    """Read the pinned source repo's commit and whether its tree is dirty."""
    def git(*args: str) -> str:
        return subprocess.run(
            ("git", "-C", str(URDF_REPO), *args),
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    try:
        return {
            "repo": UPSTREAM_REPO,
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        sys.exit(f"cannot read upstream git state at {URDF_REPO}: {exc}")


def check_upstream(state: dict, allow_drift: bool) -> None:
    """
    Fail if the source repo is not at the pinned commit.

    The meshes and joint limits are build inputs, so a silent upstream change
    would silently change the model's dynamics. A dirty tree is also refused:
    the commit hash would no longer describe what was actually compiled.
    """
    problems = []
    if state["commit"] != UPSTREAM_COMMIT:
        problems.append(
            f"commit is {state['commit'][:12]}, pinned to {UPSTREAM_COMMIT[:12]}"
        )
    if state["dirty"]:
        problems.append("working tree has uncommitted changes")
    if not problems:
        return

    detail = "; ".join(problems)
    if allow_drift:
        print(f"WARNING: upstream drift allowed: {detail}")
        return
    sys.exit(
        f"upstream {URDF_REPO.name}: {detail}.\n"
        "Re-run with --allow-upstream-drift to build anyway, then update "
        "UPSTREAM_COMMIT if the new state is intended."
    )


def stage() -> None:
    """Copy meshes, licence and a compiler-patched URDF into the output dir."""
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    shutil.copytree(MESH_SRC, ASSET_DIR)

    # The staged meshes are Wonik's, BSD-2-Clause, which requires the copyright
    # notice to travel with any redistribution. Copying it into the output dir
    # keeps the model directory self-contained and compliant if committed.
    shutil.copy2(LICENSE_SRC, LICENSE_OUT)

    urdf = URDF_SRC.read_text()
    # strippath turns package:// URIs into bare filenames resolved via meshdir.
    # discardvisual is kept off so the exported MJCF still has render meshes.
    # fusestatic must be off: it would otherwise absorb palm_link into
    # worldbody and each _tip into its distal link, leaving no palm body to
    # write contact exclusions against and no fingertip bodies to query.
    patch = (
        '  <mujoco>\n'
        '    <compiler meshdir="assets" strippath="true" discardvisual="false"'
        ' inertiafromgeom="true" balanceinertia="true" fusestatic="false"/>\n'
        '  </mujoco>\n'
    )
    anchor = "<robot "
    insert = urdf.index(">", urdf.index(anchor)) + 1
    STAGED_URDF.write_text(urdf[:insert] + "\n" + patch + urdf[insert:])


def collision_primitives(raw_mjcf: str) -> dict:
    """
    Return the collision primitive for each collision mesh, replacing its hull.

    WHY REPLACE THE HULLS. MuJoCo collides meshes by their convex hull, and
    Wonik's visual meshes hull badly: against the true triangle volume the hull
    inflates the palm 2.3x and the finger base 6.0x. An inflated hull puts
    contact points in the wrong place and makes fingers fatter than the real
    hardware -- precisely the regime a grasp evaluation probes. mujoco_menagerie
    replaces the 4F hand's collision geometry with primitives for the same
    reason; this follows that approach (boxes for phalanges, capsule for the
    fingertip, collision on group 3) with sizes authored for the V5 3F. See
    PRIMITIVES for where each number comes from and why Menagerie's own sizes
    are not reused.

    A SECOND DEFICIENCY, found while measuring the first: the meshes are not
    clean. They carry non-manifold edges (20 in the palm, 1-6 per finger link),
    and MuJoCo's mesh-derived volumes disagree with the true triangle volumes by
    1.2-6x -- for the palm it implies 1093 cm^3 against a 503 cm^3 bounding box,
    which no solid can occupy. A clean 20 mm control cube compiles to exactly
    8.00 cm^3, so the fault is in the meshes, not the toolchain. Because the mass
    calibration derives every link mass from these volumes, moving to primitives
    makes the inertia auditable as well as the contacts.

    WHAT THIS DOES NOT FIX. One box per link cannot represent palm concavity, so
    an object still cannot nestle into the palm; Menagerie's single palm box has
    the same limitation. It makes the palm a correctly-sized slab rather than an
    inflated blob, which is what matters for fingertip grasps. Power grasps that
    rely on palm shape remain out of scope, and this is the main reason to read
    the results below as fingertip-grasp results.

    This function validates the table against the meshes rather than trusting
    it: every primitive must sit inside the mesh's actual vertex extent, so a
    typo cannot silently produce geometry larger than the part it represents.
    """
    import numpy as np

    model = mujoco.MjModel.from_xml_string(raw_mjcf, _asset_dict())
    used: dict[str, dict] = {}

    for g in range(model.ngeom):
        # Collision geoms only: the visual duplicates keep their meshes.
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        mid = int(model.geom_dataid[g])
        if mid < 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)
        if name not in PRIMITIVES:
            sys.exit(f"no collision primitive authored for mesh {name!r}")
        attrs = dict(PRIMITIVES[name])

        start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        verts = model.mesh_vert[start:start + count].astype(float)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[g])
        local = verts @ rot.reshape(3, 3).T + model.geom_pos[g]
        mesh_lo, mesh_hi = local.min(axis=0), local.max(axis=0)

        size = np.array([float(v) for v in attrs["size"].split()])
        pos = np.array([float(v) for v in attrs["pos"].split()])
        if attrs["type"] == "capsule":
            radius, cyl_half = size
            half = np.array([radius, radius, radius + cyl_half])
        else:
            half = size

        # Allow 1 mm of slack: the fingertip capsule legitimately bulges a hair
        # past the mesh where a rounded cap meets a flat percentile bound.
        slack = 1e-3
        if np.any(pos - half < mesh_lo - slack) or np.any(pos + half > mesh_hi + slack):
            sys.exit(
                f"primitive for {name} extends outside the mesh it replaces:\n"
                f"  primitive {np.round(pos - half, 4)} .. {np.round(pos + half, 4)}\n"
                f"  mesh      {np.round(mesh_lo, 4)} .. {np.round(mesh_hi, 4)}"
            )

        used[name] = attrs

    missing = set(PRIMITIVES) - set(used)
    if missing:
        sys.exit(f"PRIMITIVES has unused entries: {sorted(missing)}")
    return used


def _asset_dict() -> dict:
    return {p.name: p.read_bytes() for p in ASSET_DIR.glob("*.stl")}


def export_mjcf() -> str:
    """Compile the staged URDF and return it as native MJCF text."""
    model = mujoco.MjModel.from_xml_path(str(STAGED_URDF))
    mujoco.mj_saveLastXML(str(OUT_XML), model)
    STAGED_URDF.unlink()
    return OUT_XML.read_text()


def build_tree(raw_mjcf: str, density: float,
               primitives: dict | None = None) -> ET.ElementTree:
    """
    Assemble the finished MJCF: swap collision meshes for primitives, set their
    density, exclude the palm/finger-base contacts, and add one position
    actuator per joint.

    Mass and inertia both come from the geoms, so a single density controls the
    whole model's scale while the geometry keeps the relative distribution
    between links. The exported visual geoms already carry density="0" (they
    are render-only duplicates of the collision meshes); skipping those is what
    keeps them from double-counting into the inertia.
    """
    root = ET.fromstring(raw_mjcf)

    for geom in root.iter("geom"):
        if geom.get("density") == "0":
            continue

        if primitives is not None:
            mesh = geom.get("mesh")
            if mesh is None or mesh not in primitives:
                sys.exit(f"collision geom has no fitted primitive: {geom.attrib}")
            # Rebuild the geom from scratch: the mesh attribute, and the pos/quat
            # that placed the mesh, must all go, or the primitive inherits an
            # orientation meant for a different shape.
            geom.attrib.clear()
            geom.attrib.update(primitives[mesh])
            geom.set("group", COLLISION_GROUP)

        geom.set("density", f"{density:.10g}")

    contact = ET.SubElement(root, "contact")
    for base in FINGER_BASES:
        ET.SubElement(contact, "exclude", {"body1": PALM, "body2": base})

    for joint in root.iter("joint"):
        joint.set("damping", f"{JOINT_DAMPING:g}")
        joint.set("frictionloss", f"{JOINT_FRICTIONLOSS:g}")

    actuator = ET.SubElement(root, "actuator")
    for joint in root.iter("joint"):
        name = joint.get("name")
        lower, upper = joint.get("range").split()
        ET.SubElement(
            actuator,
            "position",
            {
                "name": name.replace("joint", "act"),
                "joint": name,
                "kp": str(KP),
                "ctrlrange": f"{lower} {upper}",
                # Carries the URDF's effort limit through as a torque clamp.
                "forcerange": joint.get("actuatorfrcrange"),
            },
        )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    return tree


def compile_tree(tree: ET.ElementTree) -> mujoco.MjModel:
    tree.write(OUT_XML, encoding="unicode")
    return mujoco.MjModel.from_xml_path(str(OUT_XML))


def calibrate(raw_mjcf: str, target: float) -> tuple[mujoco.MjModel, float]:
    """
    Solve for the single geom density that makes total mass match `target`.

    Mass is linear in density for fixed geometry, so one probe compile
    determines the answer exactly -- no iteration needed. Inertia is likewise
    linear in density, but rather than assume that we verify it below against
    the probe model.

    WHAT THIS DOES AND DOES NOT FIX. It pins the total and keeps the relative
    distribution that mesh volume implies. It cannot recover the real hand's
    distribution, because one density cannot represent a hand whose links are
    unequally hollow. The direction of that error is worth being precise about,
    since it is easy to state backwards:

        Pinning the total forces the uniform density to be the
        volume-weighted mean of the real per-link densities,
            rho_uniform = sum(rho_i * V_i) / sum(V_i).
        Each link's model mass is rho_uniform * V_i against a real mass of
        rho_i * V_i, so
            model_i / real_i = rho_uniform / rho_i.
        A link DENSER than the mean is therefore UNDERestimated by the model,
        and a link less dense than the mean is overestimated.

    So if the palm is the dense part (motors, PCB, cabling) then the model's
    palm is too LIGHT and its fingertips too heavy -- not the reverse. Which
    links actually sit above the mean is unknown without per-link masses from
    Wonik, so no claim is made here about which way this particular model
    skews; only that the errors must go in opposite directions and sum to zero
    by construction. Per-link masses from the manual would remove the guess
    entirely, at which point this whole function should be replaced by explicit
    per-body <inertial> elements.
    """
    primitives = collision_primitives(raw_mjcf)
    probe = compile_tree(build_tree(raw_mjcf, PROBE_DENSITY, primitives))
    probe_mass = float(probe.body_mass.sum())
    factor = target / probe_mass
    density = PROBE_DENSITY * factor

    model = compile_tree(build_tree(raw_mjcf, density, primitives))

    print(f"mass calibration: {probe_mass:.4f} kg at density {PROBE_DENSITY:g}")
    print(f"  target      {target:.4f} kg")
    print(f"  factor      {factor:.6f}  ->  density {density:.2f}")

    # Verify the claim that inertia scaled by the same factor as mass, rather
    # than trusting it. Compare every body's tensor against the probe's.
    scaled = probe.body_inertia * factor
    nonzero = scaled > 0
    inertia_err = abs(model.body_inertia[nonzero] / scaled[nonzero] - 1.0).max()
    mass_err = abs(model.body_mass.sum() - target)
    print(f"  total mass  {model.body_mass.sum():.6f} kg  (err {mass_err:.2e})")
    print(f"  inertia scaled linearly: max rel dev {inertia_err:.2e}")

    if mass_err > MASS_TOL_KG:
        sys.exit(
            f"total mass {model.body_mass.sum():.6f} kg is more than "
            f"{MASS_TOL_KG:g} kg from target {target:.6f} kg"
        )
    if inertia_err > 1e-6:
        sys.exit(
            f"inertia did not scale linearly with mass (max rel dev "
            f"{inertia_err:.2e}); the density solve assumes it does"
        )
    return model, density


def check(model: mujoco.MjModel) -> None:
    data = mujoco.MjData(model)
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]

    mujoco.mj_forward(model, data)
    penetrating = data.ncon

    print(f"wrote {OUT_XML.relative_to(REPO)}")
    print(f"  bodies      {model.nbody}")
    print(f"  joints      {model.njnt}")
    print(f"  actuators   {model.nu}  {names}")
    print(f"  contacts at rest pose  {penetrating}")
    print("  per-link mass:")
    for i in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"    {name:16s} {model.body_mass[i] * 1000:7.1f} g")

    if model.nu != EXPECTED_ACTUATORS:
        sys.exit(f"expected {EXPECTED_ACTUATORS} actuators, got {model.nu}")
    if penetrating:
        sys.exit(f"model starts with {penetrating} contacts; expected none")


def write_manifest(model: mujoco.MjModel, target: float, density: float,
                   state: dict) -> None:
    """
    Record the exact inputs this model was built from, next to the output.

    Deliberately contains no timestamp: the manifest should be byte-identical
    for an identical rebuild, so it can be committed and diffed rather than
    churning on every run.
    """
    manifest = {
        "note": "Generated by sim/build_3f.py -- do not edit by hand.",
        "upstream": state,
        "inputs": {
            "urdf": {
                "path": str(URDF_SRC.relative_to(REPO)),
                "sha256": sha256(URDF_SRC),
            },
            "meshes": {
                p.name: sha256(p) for p in sorted(ASSET_DIR.glob("*.stl"))
            },
        },
        "calibration": {
            "target_mass_kg": target,
            "target_is_published_default": target == TARGET_MASS_KG,
            "probe_density": PROBE_DENSITY,
            "solved_density": round(density, 6),
            "total_mass_kg": round(float(model.body_mass.sum()), 9),
            "per_link_mass_g": {
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i):
                    round(float(model.body_mass[i]) * 1000, 3)
                for i in range(1, model.nbody)
            },
        },
        "actuators": {"count": model.nu, "kp": KP},
        "tooling": {"mujoco": mujoco.__version__},
        "builder": {
            "script": str(Path(__file__).resolve().relative_to(REPO)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "licence": {
            "meshes": "BSD-2-Clause, Copyright (c) 2024 WonikRobotics_official",
            "notice": LICENSE_OUT.name,
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--target-mass", type=float, default=TARGET_MASS_KG, metavar="KG",
        help="total mass to calibrate to (default: %(default)s, the published "
             "V5 3F figure -- see TARGET_MASS_KG for why it is uncertain)",
    )
    parser.add_argument(
        "--allow-upstream-drift", action="store_true",
        help="build even if the source repo is not at the pinned commit",
    )
    args = parser.parse_args(argv)

    state = upstream_state()
    check_upstream(state, args.allow_upstream_drift)
    stage()
    model, density = calibrate(export_mjcf(), args.target_mass)
    check(model)
    write_manifest(model, args.target_mass, density, state)
    print(f"  manifest    {MANIFEST.relative_to(REPO)}")


if __name__ == "__main__":
    main()
