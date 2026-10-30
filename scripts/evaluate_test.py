#!/usr/bin/env python3
"""Evaluate one checkpoint: sanity maps, test self-play, WOSAC, then human replay."""

import argparse
import ast
import configparser
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics
import struct
import sys
import tempfile
import uuid


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_DIR = Path("resources/drive/binaries/testing")
DEFAULT_OUTPUT_DIR = Path("/mnt/cpfs-wlc-rdma-300t/mnt/cpfs-wlc-rdma-300t/results")
DEFAULT_SANITY_DIR = Path("resources/drive/sanity/sanity_jsons")
SANITY_MAPS = (
    "forward_goal_in_front", "reverse_goal_behind", "two_agent_forward_goal_in_front",
    "two_agent_reverse_goal_behind", "simple_turn", "s_curve", "u_turn",
    "one_or_two_point_turn", "three_or_four_point_turn", "goal_out_of_sight",
)
STAGES = ("sanity", "self_play", "wosac", "human_replay")
DRIVING_METRICS = (
    "safe_completion_rate", "completion_rate", "collision_rate",
    "offroad_rate", "score", "dnf_rate",
    "episode_return", "episode_length", "collisions_per_agent", "offroad_per_agent",
    "lane_alignment_rate", "speed_at_goal", "goals_reached_this_episode",
    "goals_sampled_this_episode", "perc_controlled", "perc_other",
)
DRIVING_TOTALS = ("agent_episodes", "goals_reached_total", "goals_assigned_total")
WOSAC_FEATURES = (
    "linear_speed", "linear_acceleration", "angular_speed", "angular_acceleration",
    "distance_to_nearest_object", "time_to_collision", "collision_indication",
    "distance_to_road_edge", "offroad_indication",
)
WOSAC_METRICS = (
    "realism_meta_score", "ade", "min_ade",
    "kinematic_metrics", "interactive_metrics", "map_based_metrics",
    *("likelihood_" + key for key in WOSAC_FEATURES),
    "num_collisions_sim", "num_collisions_ref", "num_offroad_sim", "num_offroad_ref",
    "num_agents_per_scene",
)
WOSAC_TOTALS = ("realism_meta_score_std", "total_num_agents", "total_unique_scenarios")
# Without vehicle prediction targets, TTC and the scores requiring it are undefined.
WOSAC_TTC_DEPENDENT = ("likelihood_time_to_collision", "interactive_metrics", "realism_meta_score")
WOSAC_NO_VEHICLE_REASON = "No vehicle among the original prediction targets; TTC is not applicable."
METRIC_GROUPS = {
    "sanity": (*DRIVING_METRICS, *DRIVING_TOTALS, "solved_maps", "evaluated_maps", "solved_rate"),
    "self_play": (*DRIVING_METRICS, *DRIVING_TOTALS),
    "wosac": (*WOSAC_METRICS, *WOSAC_TOTALS),
    "human_replay": (*DRIVING_METRICS, *DRIVING_TOTALS),
}


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained .pt state_dict")
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR,
                        help="Directory of map_NNN.bin files (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory for the JSON report (default: %(default)s)")
    parser.add_argument("--num-maps", type=positive_int, default=1000,
                        help="Number of test maps (default: 1000)")
    parser.add_argument("--batch-size", type=positive_int, default=8, help="Complete scenes per batch")
    parser.add_argument("--driving-rollouts", type=positive_int, default=1,
                        help="Rollouts per scene for each driving pass (default: 1)")
    parser.add_argument("--wosac-rollouts", type=positive_int, default=32)
    parser.add_argument("--device", default="cuda:0", help="Device, e.g. cuda:0 or cpu (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES),
                        help="Stages to run, always in sanity/self_play/wosac/human_replay order")
    parser.add_argument("--sanity-dir", type=Path, default=DEFAULT_SANITY_DIR,
                        help="Directory of named sanity JSON scenes (default: %(default)s)")
    parser.add_argument("--sanity-maps", nargs="+", choices=SANITY_MAPS, default=list(SANITY_MAPS))
    parser.add_argument("--sanity-rollouts", type=positive_int, default=10,
                        help="Inference-only episodes per sanity map (default: 10)")
    parser.add_argument("--config", type=Path, default=ROOT / "pufferlib/config/ocean/drive.ini",
                        help="Training INI (architecture and driving reward/goal settings)")
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model-type", choices=("mlp", "lstm"), default=None,
                            help="Drive architecture: two-layer MLP or existing LSTM")
    model_group.add_argument("--rnn-name", choices=("None", "Recurrent"), default=None,
                        help="Override the INI's rnn_name: None disables recurrence, Recurrent enables LSTM")
    options = parser.parse_args(argv)
    if len(set(options.sanity_maps)) != len(options.sanity_maps):
        parser.error("--sanity-maps must not contain duplicates")
    return options


def load_settings(path):
    """Use the same default/Drive INI precedence and value conversion as pufferl."""
    parser = configparser.ConfigParser()
    default = ROOT / "pufferlib/config/default.ini"
    if not path.is_file():
        raise FileNotFoundError(f"Missing config: {path}")
    parser.read([str(default), str(path)])
    config = {}
    for section in parser.sections():
        values = {}
        for key, raw in parser[section].items():
            try:
                values[key] = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                values[key] = raw
        if section == "base":
            config.update(values)
        else:
            config[section] = values
    config["train"]["use_rnn"] = config["rnn_name"] is not None
    return config


def validate_maps(directory, num_maps, check_indices=True):
    """Check all files and scenario identities before allocating simulator resources."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing binary test directory: {directory}. "
                                "Convert test JSON files with pufferlib/ocean/drive/drive.py first.")
    paths = [directory / f"map_{i:03d}.bin" for i in range(num_maps)]
    actual = set(directory.glob("map_*.bin"))
    if actual != set(paths):
        missing = sorted(p.name for p in set(paths) - actual)
        extra = sorted(p.name for p in actual - set(paths))
        raise ValueError(
            f"Expected exactly {num_maps} maps numbered 0..{num_maps - 1} in {directory}; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    scenario_ids = []
    for path in paths:
        with path.open("rb") as handle:
            header = handle.read(16)
        if len(header) != 16:
            raise ValueError(f"Truncated map header: {path}")
        scenario_id = header.rstrip(b"\0").decode("utf-8")
        if not scenario_id or scenario_id in scenario_ids:
            raise ValueError(f"Empty or duplicate scenario ID in {path}: {scenario_id!r}")
        if check_indices:
            validate_sdc(path)
        scenario_ids.append(scenario_id)
    return scenario_ids


def validate_sdc(path):
    """Validate SDC and WOSAC object indices before initializing the C simulator."""
    with path.open("rb") as handle:
        handle.seek(16)
        header = handle.read(8)
        if len(header) != 8:
            raise ValueError(f"Truncated SDC metadata in {path}")
        sdc_index, num_tracks = struct.unpack("2i", header)
        if num_tracks < 0 or 28 + 4 * num_tracks > path.stat().st_size:
            raise ValueError(f"Invalid tracks_to_predict metadata in {path}")
        track_indices = struct.unpack(f"{num_tracks}i", handle.read(4 * num_tracks))
        num_objects = struct.unpack("i", handle.read(4))[0]
        if not 0 <= sdc_index < num_objects:
            raise ValueError(f"Human-replay requires a valid SDC in {path}: "
                             f"sdc_track_index={sdc_index}, num_objects={num_objects}")
        invalid = [index for index in track_indices if not 0 <= index < num_objects]
        if invalid:
            raise ValueError(f"WOSAC requires valid tracks_to_predict indices in {path}: "
                             f"invalid={invalid}, num_objects={num_objects}. "
                             "Repair the JSON metadata mapping and reconvert the scene.")


def inspect_map(path):
    """Read object identities/validity without invoking C on unvalidated metadata."""
    data = path.read_bytes()
    if len(data) < 32:
        raise ValueError(f"Truncated map: {path}")
    sdc, count = struct.unpack_from("2i", data, 16)
    offset = 24 + 4 * count
    if count < 0 or offset + 8 > len(data):
        raise ValueError(f"Invalid track metadata length: {path}")
    tracks = struct.unpack_from(f"{count}i", data, 24)
    objects, roads = struct.unpack_from("2i", data, offset)
    if objects < 1 or roads < 0:
        raise ValueError(f"Invalid object counts: {path}")
    offset += 8
    entities = []
    for index in range(objects + roads):
        if offset + 16 > len(data):
            raise ValueError(f"Truncated entity {index}: {path}")
        _, kind, object_id, frames = struct.unpack_from("4i", data, offset)
        dynamic = index < objects
        if frames < 1 or (dynamic and (frames != 91 or kind not in (1, 2, 3))):
            raise ValueError(f"Unsupported entity type/trajectory length at {index}: {path}")
        arrays = offset + 16
        scalars = arrays + (8 if dynamic else 3) * frames * 4
        if scalars + 28 > len(data):
            raise ValueError(f"Truncated entity data at {index}: {path}")
        if dynamic:
            entities.append({
                "index": index, "id": object_id, "kind": kind,
                "valid": struct.unpack_from(f"{frames}i", data, arrays + 7 * frames * 4),
                "x": struct.unpack_from(f"{frames}f", data, arrays),
                "y": struct.unpack_from(f"{frames}f", data, arrays + frames * 4),
                "heading": struct.unpack_from(f"{frames}f", data, arrays + 6 * frames * 4),
                "width": struct.unpack_from("f", data, scalars)[0],
                "length": struct.unpack_from("f", data, scalars + 4)[0],
            })
        offset = scalars + 28
    if len({obj["id"] for obj in entities}) != objects:
        raise ValueError(f"Duplicate object IDs: {path}")
    return sdc, tracks, entities


def validate_stage_maps(directory, num_maps, mode, init_steps=0):
    ids = validate_maps(directory, num_maps, check_indices=False)
    targets = {}
    egos = []
    for map_id, scenario_id in enumerate(ids):
        path = directory / f"map_{map_id:03d}.bin"
        sdc, tracks, objects = inspect_map(path)
        # C initializes a nonnegative SDC before its normal validity/type filter.
        if sdc < -1 or sdc >= len(objects):
            raise ValueError(f"Invalid SDC index {sdc} for {len(objects)} objects: {path}")
        if sdc >= 0 and (objects[sdc]["kind"] != 1 or objects[sdc]["valid"][init_steps] != 1):
            raise ValueError(f"SDC must be a vehicle valid at frame {init_steps}: {path}")
        if mode == "human_replay":
            if sdc < 0:
                raise ValueError(f"Human-replay requires a valid original SDC: {path}. "
                                 "Check the source metadata and reconvert the scene.")
            egos.append({"map_id": map_id, "scenario_id": scenario_id,
                         "ego_object_index": sdc, "ego_object_id": objects[sdc]["id"]})
        if mode == "wosac":
            if not tracks or len(set(tracks)) != len(tracks) or any(i < 0 or i >= len(objects) for i in tracks):
                raise ValueError(f"WOSAC requires nonempty, unique, valid tracks_to_predict indices: {path}")
            targets[scenario_id] = []
            for index in tracks:
                obj = objects[index]
                if obj["valid"][init_steps] != 1 or not any(obj["valid"][init_steps + 1:]):
                    raise ValueError(f"WOSAC target {index} lacks valid initial/future reference frames: {path}")
                if not all(math.isfinite(obj[k][t]) for k in ("x", "y", "heading")
                           for t in range(init_steps, 91) if obj["valid"][t]):
                    raise ValueError(f"Non-finite WOSAC reference for target {index}: {path}")
                if not all(math.isfinite(obj[k]) and obj[k] > 0 for k in ("width", "length")):
                    raise ValueError(f"Invalid WOSAC target dimensions at {index}: {path}")
                targets[scenario_id].append(obj["id"])
    return ids, targets, egos


def prepare_sanity_maps(options, destination):
    """Use the existing JSON converter, keeping source maps and checkpoints untouched."""
    from pufferlib.ocean.drive.drive import save_map_binary

    destination.mkdir(parents=True, exist_ok=True)
    names = list(options.sanity_maps)
    for index, name in enumerate(names):
        source = options.sanity_dir / f"{name}.json"
        if not source.is_file():
            raise FileNotFoundError(f"Missing sanity map: {source}. Restore the sanity resources or set --sanity-dir.")
        scene = json.loads(source.read_text())
        # Curated JSONs may share or omit IDs. Give each temporary scene a unique short ID.
        scene["scenario_id"] = f"sanity_{index:03d}"
        # Handcrafted maps can retain indices from a larger source scene. This
        # stage controls eligible actors, so neither Waymo role is needed here.
        scene["metadata"] = {**scene.get("metadata", {}),
                             "sdc_track_index": -1, "tracks_to_predict": []}
        save_map_binary(scene, destination / f"map_{index:03d}.bin", index)
    ids, _, _ = validate_stage_maps(destination, len(names), "sanity")
    return ids, names


def aggregate_driving(rows):
    n = sum(row["n"] for row in rows)
    if n <= 0:
        raise ValueError("No completed agent episodes")
    result = {key: sum(row[key] * row["n"] for row in rows) / n
              for key in DRIVING_METRICS if key != "completion_rate"}
    goals = sum(row["goals_sampled_this_episode"] * row["n"] for row in rows)
    reached = sum(row["goals_reached_this_episode"] * row["n"] for row in rows)
    if goals <= 0:
        raise ValueError("No assigned goals in driving evaluation")
    result["completion_rate"] = reached / goals
    result.update(agent_episodes=n, goals_reached_total=reached, goals_assigned_total=goals)
    return result


def aggregate_wosac(rows):
    if not rows:
        raise ValueError("No WOSAC scene results")
    for row in rows:
        check_metrics(row, WOSAC_METRICS, allow_none=WOSAC_TTC_DEPENDENT)
    values = {key: [row[key] for row in rows if row[key] is not None] for key in WOSAC_METRICS}
    result = {key: sum(items) / len(items) if items else None for key, items in values.items()}
    scores = values["realism_meta_score"]
    result.update(realism_meta_score_std=statistics.pstdev(scores) if scores else None,
                  total_num_agents=sum(row["num_agents_per_scene"] for row in rows),
                  total_unique_scenarios=len(rows))
    return result


def wosac_metric_coverage(rows):
    counts = {key: sum(row[key] is not None for row in rows) for key in WOSAC_METRICS}
    counts["realism_meta_score_std"] = counts["realism_meta_score"]
    return counts


def prepare_wosac_row(raw, has_vehicle_targets):
    """Convert only the known no-vehicle TTC case to JSON null, with its reason."""
    not_applicable = {} if has_vehicle_targets else {
        key: WOSAC_NO_VEHICLE_REASON for key in WOSAC_TTC_DEPENDENT}
    row = {key: None if key in not_applicable else float(raw[key]) for key in WOSAC_METRICS}
    check_metrics(row, WOSAC_METRICS, allow_none=not_applicable)
    row["not_applicable"] = not_applicable
    return row


def check_metrics(row, keys, allow_none=()):
    for key in keys:
        if key in row and row[key] is None and key in allow_none:
            continue
        if key not in row or row[key] is None or not math.isfinite(float(row[key])):
            raise ValueError(f"Missing or non-finite metric {key}: {row}")


def load_policy(config, env, checkpoint, device):
    import torch
    from pufferlib.ocean import torch as policy_module
    from pufferlib.model_config import apply_model_type

    apply_model_type(config)
    policy = getattr(policy_module, config["policy_name"])(env, **config["policy"])
    if config["rnn_name"] is not None:
        policy = getattr(policy_module, config["rnn_name"])(env, policy, **config["rnn"])
    weights = torch.load(checkpoint, map_location=device, weights_only=True)
    weights = {key.removeprefix("module."): value for key, value in weights.items()}
    policy.load_state_dict(weights, strict=True)
    return policy.to(device).eval()


def rollout_driving(env, policy, config, seed):
    """Match Evaluator's recurrent-state behavior; never render or resample maps."""
    import numpy as np
    import torch
    from pufferlib.ocean.drive import binding
    from pufferlib.pytorch import sample_logits

    obs, _ = env.reset(seed=seed)
    device = config["train"]["device"]
    state = {}
    if config["train"]["use_rnn"]:
        state = {
            "lstm_h": torch.zeros(env.num_agents, policy.hidden_size, device=device),
            "lstm_c": torch.zeros(env.num_agents, policy.hidden_size, device=device),
        }
    with torch.no_grad():
        for _ in range(env.episode_length - env.init_steps - 1):
            logits, _ = policy.forward_eval(torch.as_tensor(obs, device=device), state)
            actions, _, _ = sample_logits(logits)
            actions = actions.cpu().numpy().reshape(env.action_space.shape)
            if isinstance(logits, torch.distributions.Normal):
                actions = np.clip(actions, env.action_space.low, env.action_space.high)
            obs, _, _, truncations, info = env.step(actions, per_env_logs=True)
            if truncations.all():
                if len(info) != env.num_envs or not all(row.get("n", 0) > 0 for row in info):
                    raise RuntimeError("Missing scene logs at episode boundary")
                # Per-scene logging does not consume C's aggregate counters. Drain them
                # so repeated rollouts do not average previous episodes a second time.
                binding.vec_log(env.c_envs, env.num_agents)
                return info
    raise RuntimeError("Driving rollout ended without an episode boundary")


def make_env_config(config, options, mode="self_play"):
    if mode not in METRIC_GROUPS:
        raise ValueError(f"Unknown evaluation mode: {mode}")
    env = copy.deepcopy(config["env"])
    env.update(map_dir=str(getattr(options, "prepared_test_dir", options.test_dir)), num_maps=options.num_maps,
               resample_frequency=0, report_interval=1, termination_mode=0,
               episode_length=91, init_steps=0, control_mode="control_agents",
               init_mode="create_all_valid", seed=options.seed)
    env["defer_reset"] = mode == "wosac"
    if mode == "human_replay":
        env["control_mode"] = "control_sdc_only"
    elif mode == "wosac":
        for field in ("init_steps", "control_mode", "init_mode", "goal_behavior", "goal_radius"):
            env[field] = config["eval"][f"wosac_{field}"]
    if not 0 <= env["init_steps"] < env["episode_length"] - 2:
        raise ValueError("Invalid evaluation init_steps")
    return env


def run_stage(options, config, scenario_ids, report, mode, map_names=None, targets=None, egos=None):
    """One isolated pass; no optimizer, training or checkpoint writes."""
    import numpy as np
    import torch
    from pufferlib.ocean.drive.drive import Drive

    if str(options.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu")
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    stage_config = copy.deepcopy(config)
    stage_config["train"]["device"] = options.device
    stage_config["eval"]["wosac_num_rollouts"] = options.wosac_rollouts
    env_config = make_env_config(stage_config, options, mode)
    report["settings"][f"{mode}_env"] = env_config
    rows = report["per_scene"][mode]
    policy = None
    evaluator = None
    if mode == "wosac":
        from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator
        evaluator = WOSACEvaluator(stage_config)

    for start in range(0, options.num_maps, options.batch_size):
        indices = list(range(start, min(start + options.batch_size, options.num_maps)))
        expected_ids = [scenario_ids[i] for i in indices]
        env = Drive(**env_config, map_ids=indices)
        try:
            if env.map_ids != indices or env.scenario_ids != expected_ids:
                raise RuntimeError("Explicit map selection failed; rebuild the C extension")
            if policy is None:
                policy = load_policy(stage_config, env, options.checkpoint, options.device)
            if mode == "wosac":
                env.reset(seed=options.seed + start)
                truth = evaluator.collect_ground_truth_trajectories(env)
                # Do not silently score only a subset when a target was not created by C.
                has_vehicle_targets = {}
                for scene_id in expected_ids:
                    mask = ((truth["scenario_id"][:, 0] == scene_id)
                            & truth["is_track_to_predict"][:, 0])
                    actual_ids = set(truth["id"][mask, 0].tolist())
                    if actual_ids != set(targets[scene_id]):
                        raise RuntimeError(f"WOSAC target coverage mismatch in {scene_id}: "
                                           f"expected={targets[scene_id]}, loaded={sorted(actual_ids)}")
                    has_vehicle_targets[scene_id] = bool(truth["is_vehicle"][mask].any())
                trajectories = evaluator.collect_simulated_trajectories(stage_config, env, policy)
                results = evaluator.compute_metrics(
                    truth, trajectories, env.get_global_agent_state(), env.get_road_edge_polylines(),
                    aggregate_results=False, drop_last_scenario=False, round_results=False,
                    allow_undefined_ttc=True,
                )
                if set(results.index) != set(expected_ids) or len(results) != len(indices):
                    missing = sorted(set(expected_ids) - set(results.index))
                    raise RuntimeError(f"Incomplete WOSAC results; missing scenarios: {missing}")
                for map_id, scene_id in zip(indices, expected_ids):
                    row = prepare_wosac_row(results.loc[scene_id].to_dict(), has_vehicle_targets[scene_id])
                    if row["num_agents_per_scene"] != len(targets[scene_id]):
                        raise RuntimeError(f"Incomplete WOSAC target count in {scene_id}")
                    rows.append({"map_id": map_id, "scenario_id": scene_id, **row})
                report["metrics"][mode] = aggregate_wosac(rows)
                report["metric_coverage"][mode] = wosac_metric_coverage(rows)
            else:
                rollouts = options.sanity_rollouts if mode == "sanity" else options.driving_rollouts
                for rollout in range(rollouts):
                    logs = rollout_driving(env, policy, stage_config, options.seed + start + rollout)
                    if len(logs) != len(indices):
                        raise RuntimeError(f"Missing scene logs in {mode}")
                    for map_id, scene_id, log in zip(indices, expected_ids, logs):
                        check_metrics(log, (*DRIVING_METRICS, "n"))
                        if log["n"] <= 0:
                            raise RuntimeError(f"No completed agent episodes in map {map_id}")
                        if mode == "human_replay" and log["n"] != 1:
                            raise RuntimeError(f"{mode} must control exactly one ego in map {map_id}")
                        row = {"map_id": map_id, "scenario_id": scene_id, "rollout": rollout,
                               **{key: float(log[key]) for key in (*DRIVING_METRICS, "n")}}
                        if map_names is not None:
                            row["map_name"] = map_names[map_id]
                        if egos is not None:
                            row.update(egos[map_id])
                        rows.append(row)
                report["metrics"][mode] = aggregate_driving(rows)
                if mode == "sanity":
                    summarize_sanity(report, options.sanity_rollouts)
        finally:
            env.close()
        print(f"{mode}: {indices[-1] + 1}/{options.num_maps} scenes", flush=True)


def summarize_sanity(report, rollouts):
    by_map = {}
    for row in report["per_scene"]["sanity"]:
        by_map.setdefault(row["map_name"], []).append(row)
    summaries = []
    for name, rows in by_map.items():
        metrics = aggregate_driving(rows)
        complete = len(rows) == rollouts and len({row["rollout"] for row in rows}) == rollouts
        summaries.append({"map_name": name, "metrics": metrics, "rollouts_completed": len(rows),
                          "solved": metrics["score"] >= 0.95 if complete else None})
    report["sanity_results"] = summaries
    completed = [row for row in summaries if row["solved"] is not None]
    solved = sum(row["solved"] for row in completed)
    report["metrics"]["sanity"].update(solved_maps=solved, evaluated_maps=len(completed),
                                      solved_rate=solved / len(completed) if completed else None)


def evaluate(options, config, report):
    """Run selected stages in fixed order; one stage's failure never hides other results."""
    report["settings"] = {
        "device": options.device, "seed": options.seed, "batch_size": options.batch_size,
        "driving_rollouts": options.driving_rollouts, "sanity_rollouts": options.sanity_rollouts,
        "wosac_rollouts": options.wosac_rollouts, "inference_only": True,
        "policy_name": config["policy_name"], "rnn_name": config["rnn_name"],
        "model_type": config.get("model_type"),
        "policy": config["policy"], "rnn": config["rnn"],
        "driving_aggregation": "agent-episode weighted; completion uses total goal counts",
        "wosac_aggregation": "equal weight per scene; original tracks_to_predict only; population std",
        "human_replay_control": "original dataset SDC only; other created agents replay logged trajectories",
        "sanity_solved_rule": "per-map mean score >= 0.95; not a safe-completion threshold",
        "lstm_reset": "zero at rollout start only; no history warm-up or per-agent reset on respawn",
    }
    for mode in STAGES:
        status = report["stages"][mode]
        if mode not in options.stages:
            status["status"] = "skipped"
            report["skipped"][mode] = "Not selected by --stages"
            continue
        print(f"\nStarting {mode}", flush=True)
        status.update(status="running", started_at=datetime.now(timezone.utc).isoformat())
        stage_options = copy.copy(options)
        names, targets, egos = None, None, None
        try:
            with tempfile.TemporaryDirectory(prefix=f"pufferdrive-{mode}-") as directory:
                if mode == "sanity":
                    stage_options.num_maps = len(options.sanity_maps)
                    stage_options.prepared_test_dir = Path(directory)
                    ids, names = prepare_sanity_maps(options, Path(directory))
                else:
                    init_steps = config["eval"]["wosac_init_steps"] if mode == "wosac" else 0
                    ids, targets, egos = validate_stage_maps(options.test_dir, options.num_maps, mode, init_steps)
                    if mode == "human_replay":
                        report["ego_selection"] = egos
                    if mode == "wosac":
                        report["wosac_targets"] = targets
                run_stage(stage_options, config, ids, report, mode, names, targets, egos or None)
                rows = report["per_scene"][mode]
                repeat = (1 if mode == "wosac" else options.sanity_rollouts if mode == "sanity"
                          else options.driving_rollouts)
                if (len(rows) != stage_options.num_maps * repeat
                        or {row["scenario_id"] for row in rows} != set(ids)):
                    raise RuntimeError(f"Incomplete {mode} coverage")
            status["status"] = "complete"
        except Exception as exc:
            status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"{mode}: {status['error']}", file=sys.stderr, flush=True)
        rows = report["per_scene"][mode]
        if rows:
            try:
                summary = aggregate_wosac(rows) if mode == "wosac" else aggregate_driving(rows)
                check_metrics(summary, WOSAC_METRICS if mode == "wosac" else DRIVING_METRICS,
                              allow_none=WOSAC_TTC_DEPENDENT if mode == "wosac" else ())
                report["metrics"][mode] = summary
                if mode == "wosac":
                    report["metric_coverage"][mode] = wosac_metric_coverage(rows)
                if mode == "sanity":
                    summarize_sanity(report, options.sanity_rollouts)
            except Exception as exc:
                # Partial results can also be unaggregatable. Keep the failure
                # inside this stage and do not publish an earlier batch's summary.
                error = f"{type(exc).__name__}: {exc}"
                status.update(status="failed", aggregation_error=error)
                status.setdefault("error", error)
                report["metrics"][mode] = {key: None for key in METRIC_GROUPS[mode]}
                if mode == "sanity":
                    report["sanity_results"] = []
                print(f"{mode}: aggregation failed: {error}", file=sys.stderr, flush=True)
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        status["coverage"] = len({row["scenario_id"] for row in rows})
        # Save progress to the same run file after each stage, including failures.
        if "report_path" in report:
            write_report(report, Path(report["report_path"]))


def write_report(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def save_report(report, directory):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"test_metrics_{stamp}_{uuid.uuid4().hex[:8]}.json"
    write_report(report, path)
    return path


def new_report(options):
    groups = METRIC_GROUPS
    stages = {}
    for mode in groups:
        stages[mode] = {"status": "pending" if mode in options.stages else "skipped",
                        "requested_scenarios": len(options.sanity_maps) if mode == "sanity" else options.num_maps,
                        "coverage": 0}
    return {
        "schema_version": 5, "replay_ego": "sdc",
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(options.checkpoint), "test_dir": str(options.test_dir),
        "sanity_dir": str(options.sanity_dir), "sanity_maps": options.sanity_maps,
        "config": str(options.config), "requested_scenarios": options.num_maps,
        "stage_order": list(groups), "stages": stages,
        "metrics": {mode: {key: None for key in keys} for mode, keys in groups.items()},
        "metric_coverage": {"wosac": wosac_metric_coverage([])},
        "per_scene": {mode: [] for mode in groups}, "sanity_results": [],
        "skipped": {mode: "Not selected by --stages" for mode, stage in stages.items()
                    if stage["status"] == "skipped"},
        "metric_notes": {
            "driving": "Rates and means are agent-episode weighted; completion is total reached / assigned goals.",
            "episode_return": "Mean sum of the rewards emitted at each step over a controlled-agent scene episode.",
            "episode_length": "Mean native episode length in simulation steps.",
            "collisions_per_agent": "Mean number of steps classified as vehicle collision, not distinct collision events.",
            "offroad_per_agent": "Mean number of steps classified as offroad, not distinct offroad events.",
            "lane_alignment_rate": "Fraction aligned on the final logged step; not a time average.",
            "speed_at_goal": "Last recorded goal speed (m/s), averaged over all agent episodes; non-finishers contribute zero.",
            "goals_reached_this_episode": "Mean native completed-goal counter per agent episode; respawn repeats follow native counting rules.",
            "goals_sampled_this_episode": "Mean assigned-goal counter per agent episode.",
            "perc_controlled": "Agent-episode weighted controlled fraction of created scene actors, in [0, 1].",
            "perc_other": "Agent-episode weighted background fraction of created scene actors, in [0, 1].",
            "sanity": "Inference only; solved if per-map mean score >= 0.95 after all requested rollouts.",
            "wosac": "Project WOSAC implementation with local wosac.ini weights; original prediction targets only. "
                     "ADE/minADE are in metres and excluded from realism_meta_score. likelihood_* are exponentiated log-likelihood scores.",
            "wosac_not_applicable": "Scenes without vehicle prediction targets retain all other metrics. TTC, "
                                    "interactive_metrics and realism_meta_score are null with a per-scene reason. "
                                    "Summaries average each metric over its applicable scenes; original composite "
                                    "weights are unchanged. metric_coverage lists each metric's valid-scene count. "
                                    "Stage coverage counts processed scenes, including those with not-applicable metrics.",
            "num_collisions_sim": "Scene mean per-agent probability of any collision over simulated rollouts (not an event count).",
            "num_collisions_ref": "Scene fraction of reference agents with any collision.",
            "num_offroad_sim": "Scene mean per-agent probability of any offroad event over simulated rollouts.",
            "num_offroad_ref": "Scene fraction of reference agents with any offroad event.",
            "realism_meta_score_std": "Population standard deviation across evaluated scenes (zero for one scene).",
            "total_num_agents": "Scored WOSAC objects summed across scenes, not multiplied by rollout count.",
        },
    }


def print_report(report, path):
    print(f"\nStatus: {report['status']}")
    for mode, metrics in report["metrics"].items():
        stage = report["stages"][mode]
        print(f"\n{mode}: {stage['status']} ({stage['coverage']}/{stage['requested_scenarios']} scenes)")
        if "error" in stage:
            print(f"  Error: {stage['error']}")
        if mode in report["skipped"]:
            print(f"  {report['skipped'][mode]}")
            continue
        for key, value in metrics.items():
            coverage = report["metric_coverage"].get(mode, {})
            display = f"{value:.6f}" if value is not None else (
                "N/A" if key in coverage and stage["coverage"] > 0 else "unavailable")
            suffix = f" ({coverage[key]}/{stage['coverage']} scenes)" if key in coverage else ""
            print(f"  {key:38s}: {display}{suffix}")
        if mode == "sanity":
            for item in report["sanity_results"]:
                result = "incomplete" if item["solved"] is None else "solved" if item["solved"] else "unsolved"
                print(f"  {item['map_name']}: {result}")
                for key, value in item["metrics"].items():
                    print(f"    {key:36s}: {value:.6f}")
    print(f"Saved: {path}")


def main(argv=None):
    options = parse_args(argv)
    for name in ("checkpoint", "test_dir", "output_dir", "config", "sanity_dir"):
        setattr(options, name, getattr(options, name).expanduser().resolve())
    report = new_report(options)
    path = save_report(report, options.output_dir)
    report["report_path"] = str(path)
    try:
        if not options.checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {options.checkpoint}")
        config = load_settings(options.config)
        if options.model_type is not None or config.get("model_type") is not None:
            from pufferlib.model_config import apply_model_type
            apply_model_type(config, options.model_type)
        if options.rnn_name is not None:
            config.pop("model_type", None)
            config["rnn_name"] = None if options.rnn_name == "None" else options.rnn_name
            config["train"]["use_rnn"] = config["rnn_name"] is not None
        evaluate(options, config, report)
        report["status"] = ("failed" if any(stage["status"] not in ("complete", "skipped")
                                           for stage in report["stages"].values()) else "complete")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        for stage in report["stages"].values():
            if stage["status"] in ("pending", "running"):
                stage.update(status="failed", error=report["error"])
        print(report["error"], file=sys.stderr)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["coverage"] = {
        mode: len({row["scenario_id"] for row in rows}) for mode, rows in report["per_scene"].items()
    }
    report["agent_episodes"] = {
        mode: sum(row["n"] for row in rows) for mode, rows in report["per_scene"].items() if mode != "wosac"
    }
    write_report(report, path)
    print_report(report, path)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
