# Results

Raw evaluation output backing the figures in the top-level README. Generated
files, committed deliberately so the numbers in the write-up can be checked
against the data that produced them.

Do not edit these by hand. Regenerate them.

| File | What it is | Reproducible |
|---|---|---|
| `grasp_tuned.json` | Grasp matrix, per-hand spread. The headline run | Yes, byte-identical |
| `grasp_common.json` | Grasp matrix, common spread=0 | Yes, byte-identical |
| `timing.json` | MuJoCo vs ROS2 loop rates over real DDS | No, see below |

## Regenerating

Run from the repo root, after `sim/build_3f.py`:

```bash
sim/venv/bin/python sim/grasp_eval.py --spread '3F=0.0,4F=0.6' \
    --json results/grasp_tuned.json
sim/venv/bin/python sim/grasp_eval.py --spread 0.0 \
    --json results/grasp_common.json

source /opt/ros/humble/setup.bash
sim/venv/bin/python sim/bench_dds.py --json results/timing.json
```

## Structure

Each file is an object, not a bare array. The result rows are under `results`;
everything else records what produced them:

- `run` - the spread setting, config count, object mass, measured grasp centres
- `controller` - every `GraspConfig` field
- `success_criterion` - the thresholds a success is judged against
- `summary` - success rates and grip-force distribution, precomputed
- `provenance` - git commit, dirty flag, MuJoCo version, hand model paths

Read `run.spread_fraction` before comparing a file against any figure. The two
grasp files differ by more than a factor of two on the 4F, and that difference is
a finding, not noise. An earlier version of these files was a bare array of rows
with no such header, and identifying a run meant inferring the spread setting
from its 4F success rate. That ambiguity caused a real error: `grasp_common.json`
was read as contradicting a per-hand-spread figure in the top-level README.

## Reproducibility

The grasp matrix is deterministic. Re-running with the same code and the same
arguments produces a byte-identical file, verified by md5. The files carry no
timestamp for that reason, matching `sim/models/allegro_3f/manifest.json`: an
unchanged re-run should diff cleanly rather than churn. Git records when they
were committed.

`timing.json` is different. It is wall-clock measurement, so it varies with
machine, system load and DDS implementation, and will differ on every run. It
carries a `platform` block for that reason and should be read as a record of one
measurement on one machine, not as a reproducible artifact. The ratio it
demonstrates (MuJoCo's loop rate being two to three orders of magnitude above the
ROS2 rate) is robust; the exact figures are not.

If `provenance.git_dirty` is `true`, the working tree had uncommitted changes when
the file was generated, so `provenance.git_commit` does not fully describe the
code that produced it. Prefer regenerating from a clean tree.
