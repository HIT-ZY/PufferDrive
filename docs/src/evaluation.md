# Evaluations and benchmarks

Driving is a safety-critical multi-agent application, making careful evaluation and risk assessment essential. Mistakes in the real world are costly, so simulations are used to catch errors before deployment. To support rapid iteration, evaluations should ideally run efficiently. This is why we also paid attention to optimizing the speed of the evaluations. This page contains an overview of the available benchmarks and evals.

## Evaluate all 150 test scenes

Run the standalone evaluator from the repository root:

```bash
python scripts/evaluate_test.py \
  --checkpoint /path/to/model.pt
```

The defaults are 150 maps, `cuda:0`, test directory
`resources/drive/binaries/testing`, and output directory
`/mnt/cpfs-c-300t/mnt/cpfs-wlc-rdma-300t/results`.
The existing resource symlink should resolve the test directory to
`/mnt/cpfs-c-300t/mnt/cpfs-wlc-rdma-300t/GPUDrive_mini/resources/drive/binaries/testing`.
Verify it with `readlink -f resources/drive/binaries/testing`.
Override paths with `--test-dir` and `--output-dir`, or use `--device cpu`.

By default, `--replay-ego custom` runs two passes in sequence:

- `self_play`: the policy controls eligible agents, including the selected ego.
- `custom_replay`: the policy controls exactly one selected vehicle per scene;
  other created agents follow their logged trajectories.

Both groups report `safe_completion_rate`, `completion_rate`, `collision_rate`,
`offroad_rate`, `score`, and `dnf_rate`. WOSAC is explicitly skipped in custom
mode because this mode discards the original track-index metadata.

The ego is selected deterministically from vehicles valid at frame 0, with a
finite initial state, positive width and length, and a goal farther away than
`max(2 metres, goal_radius)`. Among eligible vehicles, selection prefers those
not marked as expert, then takes the lowest index in the current object list.
If all eligible vehicles are marked as expert, the first of those is selected.
A scene without an eligible vehicle causes an error. Selection does not rotate
through vehicles or change randomly between rollouts. With the default one
rollout per scene, custom replay evaluates 150 ego episodes.

Existing BIN files need no reconversion for custom replay, even if their original
SDC or prediction indices are out of range. The evaluator creates temporary map
copies, writes the selected ego index into the SDC slot, and clears prediction
indices. Object IDs, trajectories, roads, and the source files remain unchanged.
Both driving passes use these copies, and they are deleted after evaluation.
The selected vehicle is a custom ego, not a recovered dataset SDC.

### Converting raw JSON for custom replay

If the BIN files do not exist yet, use the existing converter:

```bash
python pufferlib/ocean/drive/drive.py \
  --data-folder /mnt/cpfs-c-300t/mnt/cpfs-wlc-rdma-300t/GPUDrive_mini/testing \
  --output-dir /mnt/cpfs-c-300t/mnt/cpfs-wlc-rdma-300t/GPUDrive_mini/resources/drive/binaries/testing \
  --discard-track-metadata \
  --workers 8
```

`--discard-track-metadata` writes an unset SDC index and no prediction indices;
it is intended for custom replay when the original mapping is unavailable.
It does not modify source JSON. Omit this flag to preserve correctly mapped
metadata for the original SDC/WOSAC protocol. Conversion does not change resource
links. Missing input or failed conversions raise an error. Running the converter
without arguments keeps its original training defaults.

The evaluator requires exactly `map_000.bin` through `map_149.bin`, with distinct
scenario IDs and 91-frame reference trajectories. Use `--num-maps N` for a
different dataset size. Raw JSON files cannot be used directly.

### Original SDC and WOSAC protocol

For data with verified original track mappings, enable all three passes:

```bash
python scripts/evaluate_test.py \
  --checkpoint /path/to/model.pt \
  --replay-ego sdc
```

This runs `self_play`, `human_replay` (the dataset SDC only), and `wosac`, in that
order. Human replay reports the same six driving metrics; WOSAC reports
`realism_meta_score`, `ade`, `min_ade`, `kinematic_metrics`,
`interactive_metrics`, and `map_based_metrics`.

SDC and `tracks_to_predict` metadata must use positions in the current objects
list, not object IDs or positions in an earlier, unfiltered track list. The
converter rejects out-of-range indices unless metadata is explicitly discarded.
If objects were filtered without updating metadata, recover the original
track-index-to-object-ID mapping from the source scenario and remap both fields.
An object with `id=35` is not necessarily the original track at index 35.
Passing a bounds check alone cannot establish that the mapping is correct.
WOSAC also requires future reference trajectories and valid prediction targets.

### Execution and saved results

The script covers maps in index order without random map sampling or partial
scenes. Each pass creates and closes its own environments and uses independent
settings. The INI's evaluation enable flags do not control this script.
Defaults are 8 complete scenes per batch, one episode per scene in each driving
pass, and 32 rollouts per scene when WOSAC is enabled. Use `--batch-size`,
`--driving-rollouts`, and `--wosac-rollouts` to change those counts.
If a checkpoint uses a different architecture, action space, or reward/goal
configuration, supply its INI with `--config /path/to/drive.ini`; it is merged
over `pufferlib/config/default.ini`.

A uniquely named `test_metrics_<UTC timestamp>_<id>.json` contains metrics,
checkpoint/config paths, effective settings, coverage, and per-scene results.
The report uses `schema_version: 3`. In custom mode, `metrics`, `per_scene`, and
`coverage` contain `self_play`, `custom_replay`, and `wosac` entries.
`ego_selection` records each scene's selected object index and ID, expert flag,
and original SDC index. For example, read custom replay success from
`report["metrics"]["custom_replay"]["safe_completion_rate"]`.
Skipped WOSAC metrics are `null`, its coverage is 0, and `skipped.wosac` explains
why. With `--replay-ego sdc`, `human_replay` replaces `custom_replay` and all
three groups run.

`agent_episodes` records separate counts for the two driving passes.
Driving rates are weighted by agent episode counts within each pass; completion
is computed from total completed/assigned goals. Results from the two driving
modes are never pooled. Enabled WOSAC metrics are averaged equally across scenes,
retaining full precision.

Both driving passes start at step 0 for the 90 transitions of a 91-frame scene,
with the configured driving rewards and goal behavior. WOSAC uses the separate
`wosac_*` settings (by default step 10, `control_wosac`, and stop at the goal).
Each pass resets LSTM state at rollout start, with no per-agent reset on respawn,
matching the existing evaluator.

WOSAC retains the final simulation state until the next explicit reset
(`defer_reset=True`). From frame 10, it records 81 states through frame 90 with
80 policy steps. Driving passes use automatic reset and collect their episode
logs at the boundary. In `goal_behavior=2` (stop), each assigned goal is counted
as completed only once, even if the agent stays at the goal for many steps.

Missing scenes, missing results from an enabled pass, or invalid metrics cause
a nonzero exit and `status: "failed"`; partial results and coverage remain
available. `status: "complete"` means all enabled passes covered the requested
dataset: two in custom mode, three in SDC mode. No WandB account or rendering is
required. Rebuild the C extension after native code updates with
`python setup.py build_ext --inplace --force`.

## Safe goal completion

`safe_completion_rate` is the fraction of evaluated agent episodes that completed
all assigned goals without any vehicle collision or off-road event during the
entire scene episode. It is an agent-weighted rate in `[0, 1]`, not a per-step rate
or a product of the separate collision, off-road, and completion rates.

An individual agent respawn does not erase earlier violations; violations after
reaching a goal also disqualify the episode. For generated goals, every assigned
goal must be completed, including the final goal (unlike `score`, which permits
one unfinished goal when multiple goals were assigned).

The metric is available in environment `info` logs and in
`Evaluator.self_play_stats` / `Evaluator.human_replay_stats`. Training logs it as
`environment/safe_completion_rate`; evaluation logs it to WandB as
`eval/sp_safe_completion_rate` or `eval/hr_safe_completion_rate`.
When combining batches, weight each batch's rate by its logged `n` agent count.

Rebuild the C extension after updating: `python setup.py build_ext --inplace --force`.

## Sanity maps 🐛

Quickly test the training on curated, lightweight scenarios without downloading the full dataset. Each sanity map tests a specific behavior.

```bash
puffer sanity puffer_drive --wandb --wandb-name sanity-demo --sanity-maps forward_goal_in_front s_curve
```

Or run them all at once:

```bash
puffer sanity puffer_drive --wandb --wandb-name sanity-all
```

- Tip: turn learning-rate annealing off for these short runs (`--train.anneal_lr False`) to keep the sanity checks from decaying the optimizer mid-run.

Available maps:

- `forward_goal_in_front`: Straight approach to a goal in view.
- `reverse_goal_behind`: Backward start with a behind-the-ego goal.
- `two_agent_forward_goal_in_front`: Two agents advancing to forward goals.
- `two_agent_reverse_goal_behind`: Two agents reversing to rear goals.
- `simple_turn`: Single, gentle turn to a nearby goal.
- `s_curve`: S-shaped path with alternating curvature.
- `u_turn`: U-shaped turn to a goal behind the start.
- `one_or_two_point_turn`: Tight turn requiring a small reversal.
- `three_or_four_point_turn`: Even tighter turn needing multiple reversals.
- `goal_out_of_sight`: Goal starts without direct path; needs some planning.

![Sanity map gallery placeholder](images/maps_screenshot.png)

## Distributional realism benchmark 📊

We provide a PufferDrive implementation of the Waymo Open Sim Agents Challenge (WOSAC) for fast, easy evaluation of how well your trained agent matches distributional properties of human behavior.

```bash
puffer eval puffer_drive --eval.wosac-realism-eval True
```

Add `--load-model-path <path_to_checkpoint>.pt` to score a trained policy, instead of a random baseline.

See [the WOSAC benchmark page](wosac.md) for the metric pipeline and all the details.

## Human-compatibility benchmark 🤝

You may be interested in how compatible your agent is with human partners. For this purpose, we support an eval where your policy only controls the self-driving car (SDC). The rest of the agents in the scene are stepped using the logs. While it is not a perfect eval since the human partners here are static, it will still give you a sense of how closely aligned your agent's behavior is to how people drive. You can run it like this:

```bash
puffer eval puffer_drive --eval.human-replay-eval True --load-model-path <path_to_checkpoint>.pt
```

During this evaluation the self-driving car (SDC) is controlled by your policy while other agents replay log trajectories.
