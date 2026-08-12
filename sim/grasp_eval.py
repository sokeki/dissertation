"""
Grasp evaluation harness: runs the matrix and reports measured outcomes.

    sim/venv/bin/python sim/grasp_eval.py            # full matrix
    sim/venv/bin/python sim/grasp_eval.py --quick    # smaller sweep
    sim/venv/bin/python sim/grasp_eval.py --json out.json

SEPARATION OF CONCERNS. grasp_controller.py sees only HandInterface. This file
reaches into MuJoCo directly to count contacts, attribute them to fingers and
apply disturbances. That split is deliberate: the controller must be
backend-agnostic for the transfer claim to mean anything, while the evaluator is
free to use ground truth the controller cannot have. Anything measured here is
therefore an independent check on the controller, not a readout of its own
beliefs -- notably, the controller infers contact from tracking error, and this
file measures whether contact actually occurred.

--------------------------------------------------------------------------
SUCCESS CRITERION, AND WHY THIS ONE
--------------------------------------------------------------------------
A grasp counts as a success only if BOTH hold:

  (1) At least two distinct fingers are in contact with the object after the
      closing motion settles.
  (2) The object moves less than SLIP_TOL_M from its settled pose through a
      disturbance sweep: 1 g applied along each of +/-x, +/-y, +/-z in turn.

Why (1): a single contact cannot resist a wrench in any direction; a hand
touching an object with one fingertip has not grasped it. Requiring two distinct
FINGERS rather than two contacts also rules out the degenerate case of one
finger registering several contact points on the same surface, which a naive
contact count would score as a grasp.

Why (2): the hand here is welded to the world with no arm, so there is no lift
test available. Rotating gravity is the equivalent statement -- it asks whether
the grasp would survive the hand being reoriented, which is what a lift would
test. Sweeping all six axis directions means the criterion is not satisfied by an
object merely resting on top of the fingers, which is the main way a grasp
evaluation flatters itself: a ball sitting in an upward-facing hand passes any
single-direction test while being supported entirely by gravity.

Why these are strict enough to be worth reporting: (2) rejects resting contact,
(1) rejects incidental touching, and the two together cannot be passed by a hand
that closed on nothing. The threshold SLIP_TOL_M = 0.02 is a fifth of the
smallest object diameter tested, so an object that has rolled free of the fingers
cannot stay inside it.

What the criterion does NOT test: grasp quality under load beyond 1 g, resistance
to torque about the grasp axis, or whether the grasp would survive being carried
by a moving arm. It also says nothing about whether the contacts are well
distributed -- a pinch and an enveloping grasp score alike.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from grasp_controller import GRIP_EFFORT_NM, GraspConfig, GraspController
from grasp_scene import HANDS, ObjectSpec, build_scene, grasp_centre
from mujoco_hand import MujocoHand

REPO = Path(__file__).resolve().parent.parent
OBJECT_MASS_KG = 0.100

SLIP_TOL_M = 0.020

# Upper bound on a defensible total normal force for the 100 g (0.98 N) object.
# 20 N is ~20x the object's weight: firm enough to be robust, and in the range a
# person uses to hold a small ball without deforming it. Reported as a headline
# count of how many configurations exceed it, because the previous stall-based
# controller exceeded it by up to 65x and that invalidated the whole table.
SANE_FORCE_MAX_N = 20.0
MIN_FINGERS = 2
DISTURBANCE_G = 9.81
DISTURBANCE_CYCLES = 500
GRAVITY_DIRECTIONS = (
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
)


@dataclass
class GraspMetrics:
    hand: str
    object: str
    shape: str
    size: float
    mass: float
    friction: float
    offset: tuple

    initial_penetration: int = 0
    standoff_m: float = 0.0
    contact_during_calibration: int = 0
    converge_cycles: int = 0
    converge_time_s: float = 0.0
    closure: list = field(default_factory=list)
    regulated: list = field(default_factory=list)
    final_effort_Nm: list = field(default_factory=list)
    peak_effort_Nm: list = field(default_factory=list)

    n_contacts: int = 0
    n_fingers: int = 0
    contact_fingers: list = field(default_factory=list)
    contact_points: list = field(default_factory=list)
    contact_spread_m: float = 0.0
    normal_force_N: float = 0.0

    max_disturbance_slip_m: float = 0.0
    worst_direction: str = ""
    contacts_after_disturbance: int = 0

    success: bool = False
    failure_mode: str = ""


def _finger_of_body(model, body_id: int, finger_names: list[str]) -> str | None:
    """Attribute a body to the finger whose chain it belongs to."""
    path = []
    b = body_id
    while b > 0:
        path.append(b)
        b = int(model.body_parentid[b])
    path.reverse()
    if not path:
        return None
    root_child = path[0] if len(path) == 1 else path[1]
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root_child)
    return name if name in finger_names else None


def _object_contacts(model, data, obj_body: int, finger_names: list[str]):
    """Contacts between the object and the hand, with finger attribution."""
    out = []
    for c in range(data.ncon):
        con = data.contact[c]
        b1 = int(model.geom_bodyid[con.geom1])
        b2 = int(model.geom_bodyid[con.geom2])
        if obj_body not in (b1, b2):
            continue
        other = b2 if b1 == obj_body else b1
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, c, force)
        out.append({
            "finger": _finger_of_body(model, other, finger_names),
            "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other),
            "pos": [float(v) for v in con.pos],
            "normal_force": float(abs(force[0])),
        })
    return out


def run_grasp(hand_tag: str, spec: ObjectSpec, config: GraspConfig | None = None,
              centre=None) -> GraspMetrics:
    hand_xml = HANDS[hand_tag]
    model, obj_name = build_scene(hand_xml, spec, centre=centre)
    hand = MujocoHand(model=model, show_viewer=False)
    data = hand.data
    obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
    obj_qpos = int(model.jnt_qposadr[model.body_jntadr[obj_body]])
    finger_names = [f.name for f in hand.fingers]

    m = GraspMetrics(
        hand=hand_tag, object=spec.label(), shape=spec.shape, size=spec.size,
        mass=spec.mass, friction=spec.friction, offset=tuple(spec.offset),
    )

    # PROTOCOL: gravity is OFF while the hand closes, and ON for the disturbance
    # sweep that decides success.
    #
    # This is not a convenience. The hand is welded to the world with no arm, so
    # nothing can present the object to it: with gravity on, an unsupported object
    # simply falls out of the workspace during the approach. That is what happened
    # on the first run -- the 4F scored 4.93 m of "slip", which is exactly one
    # second of free fall (0.5*9.81*1^2), and its contact error was identical to
    # free space at every cycle because it never touched the object at all. The
    # 3F only escaped this because its fingers happen to point up, so the object
    # landed on them; comparing the two hands under that protocol would have been
    # comparing hand orientations, not grasps.
    #
    # Closing in zero-g isolates the question actually being asked -- can the
    # closed grasp hold the object -- and the six-direction 1 g sweep afterwards
    # answers it under full load. What this protocol does NOT test is the approach
    # problem: reaching without knocking the object away, which a real system must
    # solve and which needs an arm to study.
    base_gravity = model.opt.gravity.copy()
    model.opt.gravity[:] = 0.0

    controller = GraspController(hand, config)
    hand.set_joint_targets(controller.open_pose())
    mujoco.mj_forward(model, data)

    # An object that already intersects the OPEN hand does not fit in its
    # aperture, which is a real capability limit of the hand and reported as such
    # ("exceeds-aperture") rather than excluded as a setup artefact. Measured
    # open-hand aperture at the grasp centre: 70 mm diameter for the 3F, 110 mm
    # for the 4F. An earlier version tried backing the object off along the
    # approach axis to make these placeable, but that only converts a hand that
    # cannot open wide enough into a hand doing a distant fingertip pinch, which
    # misrepresents the limit instead of reporting it.
    m.initial_penetration = len(
        _object_contacts(model, data, obj_body, finger_names)
    )
    for _ in range(200):
        hand.step()

    result = controller.close()
    # Contact during calibration would invalidate the learned stall threshold,
    # since the baseline is meant to be free-space lag. Recorded, not assumed.
    m.contact_during_calibration = m.initial_penetration
    model.opt.gravity[:] = base_gravity
    m.converge_cycles = result.cycles
    m.converge_time_s = result.cycles * float(model.opt.timestep)
    m.closure = [round(c, 3) for c in result.closure]
    m.regulated = list(result.regulated)
    m.final_effort_Nm = list(result.final_effort)
    m.peak_effort_Nm = list(result.peak_effort)

    contacts = _object_contacts(model, data, obj_body, finger_names)
    m.n_contacts = len(contacts)
    fingers_touching = sorted({c["finger"] for c in contacts if c["finger"]})
    m.n_fingers = len(fingers_touching)
    m.contact_fingers = fingers_touching
    m.contact_points = [[round(v, 4) for v in c["pos"]] for c in contacts]
    m.normal_force_N = round(sum(c["normal_force"] for c in contacts), 3)
    if len(contacts) >= 2:
        pts = np.array([c["pos"] for c in contacts])
        m.contact_spread_m = round(
            float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).max()), 4
        )

    # Disturbance sweep. Gravity is restored afterwards so each direction starts
    # from the settled grasp rather than compounding the previous one's slip.
    settled = data.qpos[obj_qpos:obj_qpos + 3].copy()
    saved = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())
    worst, worst_dir = 0.0, ""
    contacts_after = 0
    for direction in GRAVITY_DIRECTIONS:
        data.qpos[:], data.qvel[:], data.ctrl[:] = (
            saved[0].copy(), saved[1].copy(), saved[2].copy()
        )
        mujoco.mj_forward(model, data)
        model.opt.gravity[:] = np.array(direction) * DISTURBANCE_G
        for _ in range(DISTURBANCE_CYCLES):
            hand.step()
        slip = float(np.linalg.norm(data.qpos[obj_qpos:obj_qpos + 3] - settled))
        if slip > worst:
            worst, worst_dir = slip, f"{direction}"
        contacts_after = max(
            contacts_after,
            len(_object_contacts(model, data, obj_body, finger_names)),
        )
    model.opt.gravity[:] = base_gravity

    m.max_disturbance_slip_m = round(worst, 4)
    m.worst_direction = worst_dir
    m.contacts_after_disturbance = contacts_after

    m.success = (m.n_fingers >= MIN_FINGERS and worst < SLIP_TOL_M)
    if m.success:
        m.failure_mode = "-"
    elif m.initial_penetration:
        m.failure_mode = "exceeds-aperture"
    elif m.n_contacts == 0:
        m.failure_mode = "never-contacted"
    elif m.n_fingers < MIN_FINGERS:
        m.failure_mode = f"only-{m.n_fingers}-finger"
    else:
        m.failure_mode = "slipped-in-disturbance"
    hand.close()
    return m


def matrix(quick: bool = False):
    sizes = (0.030, 0.040) if quick else (0.020, 0.030, 0.040, 0.050)
    frictions = (1.0,) if quick else (0.3, 0.6, 1.0)
    offsets = ((0, 0, 0),) if quick else (
        (0, 0, 0), (0.010, 0, 0), (0, 0.010, 0), (0, 0, 0.010), (0, 0, -0.010),
    )
    shapes = ("sphere",) if quick else ("sphere", "box")
    return list(itertools.product(HANDS, shapes, sizes, frictions, offsets))


def provenance(spreads: dict, config: GraspConfig, centres: dict,
               rows: list) -> dict:
    """
    Wrap the result rows in a record of what produced them.

    WHY. The first version wrote a bare list of 240 rows with nothing saying
    which run it was. Identifying a file then meant inferring the spread setting
    from its 4F success rate, and that ambiguity caused a real error: a
    common-spread result file was compared against a per-hand-spread figure in the
    README and read as a contradiction. A results file a reader has to
    reverse-engineer is worse than none.

    Deliberately carries NO timestamp, matching build_3f.py's manifest: the grasp
    matrix is deterministic, so an unchanged re-run produces a byte-identical file
    that diffs cleanly instead of churning. Git records when it was committed.
    """
    def git(*args: str) -> str:
        try:
            return subprocess.run(("git", "-C", str(REPO), *args),
                                  capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    ok = [r for r in rows if r["success"]]
    touched = [r for r in rows if r["n_contacts"] > 0]
    forces = sorted(r["normal_force_N"] for r in touched)

    return {
        "note": "Generated by sim/grasp_eval.py -- do not edit by hand.",
        "run": {
            "spread_fraction": spreads,
            "n_configs": len(rows),
            "object_mass_kg": OBJECT_MASS_KG,
            "grasp_centres": {t: [round(v, 5) for v in c]
                              for t, c in centres.items()},
        },
        "controller": {
            k: v for k, v in asdict(config).items() if k != "spread_fraction"
        },
        "grip_effort_setpoint_Nm": GRIP_EFFORT_NM,
        "success_criterion": {
            "min_distinct_fingers": MIN_FINGERS,
            "slip_tol_m": SLIP_TOL_M,
            "disturbance_g": DISTURBANCE_G,
            "disturbance_cycles": DISTURBANCE_CYCLES,
            "gravity_directions": [list(d) for d in GRAVITY_DIRECTIONS],
            "sane_force_bound_N": SANE_FORCE_MAX_N,
        },
        "summary": {
            "success_total": f"{len(ok)}/{len(rows)}",
            "success_by_hand": {
                t: f"{sum(1 for r in ok if r['hand'] == t)}/"
                   f"{sum(1 for r in rows if r['hand'] == t)}"
                for t in HANDS
            },
            "grip_force_N": {
                "median": round(forces[len(forces) // 2], 2) if forces else None,
                "max": round(forces[-1], 2) if forces else None,
                "over_sane_bound": f"{sum(1 for v in forces if v > SANE_FORCE_MAX_N)}"
                                   f"/{len(forces)}",
            },
        },
        "provenance": {
            "git_commit": git("rev-parse", "HEAD"),
            # Scoped to sim/, i.e. the code and model that produced this result.
            # An unscoped `git status --porcelain` includes UNTRACKED files, so it
            # counted this very file as dirt: on the first generation into an
            # as-yet-uncommitted results/ directory it reported dirty=true no
            # matter what order things were run in. Self-referential and useless.
            # Scoping also means an edit to the top-level README, which cannot
            # affect a result, no longer flags it.
            "code_dirty": bool(git("status", "--porcelain", "--", "sim")),
            "mujoco": mujoco.__version__,
            "hand_models": {
                t: str(p.relative_to(REPO)) for t, p in HANDS.items()
            },
        },
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Grasp evaluation matrix.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--spread", default="0.0",
        help="GraspConfig.spread_fraction: a single value for both hands "
             "(e.g. 0.0), or per-hand (e.g. '3F=0.0,4F=0.6')",
    )
    args = parser.parse_args()

    if "=" in args.spread:
        spreads = {k: float(v) for k, v in
                   (part.split("=") for part in args.spread.split(","))}
    else:
        spreads = {tag: float(args.spread) for tag in HANDS}
    missing = set(HANDS) - set(spreads)
    if missing:
        parser.error(f"--spread missing hands: {sorted(missing)}")

    centres = {tag: grasp_centre(path) for tag, path in HANDS.items()}
    for tag, c in centres.items():
        print(f"grasp centre {tag}: {np.round(c, 4)}")

    rows = []
    combos = matrix(args.quick)
    print(f"\nrunning {len(combos)} configurations "
          f"(spread_fraction={spreads})\n")

    header = (f"{'hand':<5} {'shape':<7} {'size':>5} {'mu':>4} {'offset':>16} "
              f"{'ncon':>5} {'nfing':>6} {'force':>7} {'conv_s':>7} "
              f"{'slip_mm':>8} {'ok':>3}  failure")
    print(header)
    print("-" * len(header))
    for tag, shape, size, mu, off in combos:
        spec = ObjectSpec(shape=shape, size=size, mass=OBJECT_MASS_KG,
                          friction=mu, offset=off)
        r = run_grasp(tag, spec, GraspConfig(spread_fraction=spreads[tag]),
                      centre=centres[tag])
        rows.append(asdict(r))
        print(f"{tag:<5} {shape:<7} {size*1000:5.0f} {mu:4.1f} "
              f"{str(tuple(int(v*1000) for v in off)):>16} "
              f"{r.n_contacts:5d} {r.n_fingers:6d} {r.normal_force_N:7.2f} "
              f"{r.converge_time_s:7.3f} {r.max_disturbance_slip_m*1000:8.1f} "
              f"{'Y' if r.success else 'n':>3}  {r.failure_mode}")

    # Configurations where the object spawned already intersecting the hand are
    # scene-setup artefacts, not grasp failures: the object was placed somewhere
    # the hand already occupies, so no controller could have succeeded. Reported
    # separately and excluded from the denominator rather than quietly counted as
    # failures, which would flatter or penalise a hand for its own bulk.
    def valid(r):
        # Every configuration is a real outcome now; exceeds-aperture is a hand
        # capability limit, not a harness artefact, so nothing is excluded.
        return True

    ok_rows = [r for r in rows if r["success"]]
    valid_rows = [r for r in rows if valid(r)]
    invalid = len(rows) - len(valid_rows)
    print(f"\noverall: {len(ok_rows)}/{len(valid_rows)} of valid configs "
          f"({100 * len(ok_rows) / max(len(valid_rows), 1):.0f}%), "
          f"{invalid} excluded")
    for tag in HANDS:
        sub = [r for r in rows if r["hand"] == tag and valid(r)]
        ok = sum(1 for r in sub if r["success"])
        excl = sum(1 for r in rows if r["hand"] == tag and not valid(r))
        print(f"  {tag}: {ok}/{len(sub)} "
              f"({100 * ok / max(len(sub), 1):.0f}%), {excl} excluded")

    print("\nsuccess rate by object size (valid configs):")
    for tag in HANDS:
        parts = []
        for size in sorted({r["size"] for r in rows}):
            sub = [r for r in rows
                   if r["hand"] == tag and valid(r) and r["size"] == size]
            ok = sum(1 for r in sub if r["success"])
            parts.append(f"{size*1000:.0f}mm {ok}/{len(sub)}")
        print(f"  {tag}: " + "  ".join(parts))

    print("\nsuccess rate by friction (valid configs):")
    for tag in HANDS:
        parts = []
        for mu in sorted({r["friction"] for r in rows}):
            sub = [r for r in rows
                   if r["hand"] == tag and valid(r) and r["friction"] == mu]
            ok = sum(1 for r in sub if r["success"])
            parts.append(f"mu={mu} {ok}/{len(sub)}")
        print(f"  {tag}: " + "  ".join(parts))

    modes = {}
    for r in rows:
        if not r["success"] and valid(r):
            modes[r["failure_mode"]] = modes.get(r["failure_mode"], 0) + 1
    if modes:
        print("\nfailure modes (valid configs): " + ", ".join(
            f"{k}={v}" for k, v in sorted(modes.items(), key=lambda x: -x[1])))

    # HEADLINE METRIC: grip force distribution. Reported for every configuration
    # that made contact, not only successes, so a controller that crushes its way
    # to a good success rate cannot hide behind it.
    touched = [r for r in rows if r["n_contacts"] > 0]
    print(f"\n{'=' * 60}\nGRIP FORCE (headline)\n{'=' * 60}")
    print(f"object weight: {0.100 * 9.81:.2f} N;  sane bound: "
          f"{SANE_FORCE_MAX_N:g} N total normal force")
    for label, subset in (("all configs that made contact", touched),
                          ("successes only", ok_rows)):
        if not subset:
            continue
        f = sorted(r["normal_force_N"] for r in subset)
        over = sum(1 for v in f if v > SANE_FORCE_MAX_N)
        print(f"  {label} (n={len(f)}): median {f[len(f)//2]:.1f} N, "
              f"max {f[-1]:.1f} N, "
              f"{over}/{len(f)} exceed {SANE_FORCE_MAX_N:g} N "
              f"({100 * over / len(f):.0f}%)")
    for tag in HANDS:
        sub = [r for r in touched if r["hand"] == tag]
        if not sub:
            continue
        f = sorted(r["normal_force_N"] for r in sub)
        over = sum(1 for v in f if v > SANE_FORCE_MAX_N)
        print(f"  {tag}: median {f[len(f)//2]:.1f} N, max {f[-1]:.1f} N, "
              f"{over}/{len(f)} over bound")
    peaks = [max(r["peak_effort_Nm"]) for r in touched if r["peak_effort_Nm"]]
    if peaks:
        print(f"  peak joint torque across run: max {max(peaks):.3f} N*m "
              f"(setpoint {0.25:g} N*m)")

    if args.json:
        cfg = GraspConfig(spread_fraction=0.0)  # non-spread fields are shared
        payload = provenance(spreads, cfg, centres, rows)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
