#!/usr/bin/env python3
"""Evaluate a checkpoint on every test map, print metrics, and save a JSON report."""

import argparse
import ast
import configparser
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import struct
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_DIR = Path("resources/drive/binaries/testing")
DEFAULT_OUTPUT_DIR = Path("/mnt/cpfs-c-300t/mnt/cpfs-wlc-rdma-300t/results")
DRIVING_METRICS = (
    "safe_completion_rate", "completion_rate", "collision_rate",
    "offroad_rate", "score", "dnf_rate",
)
WOSAC_METRICS = (
    "realism_meta_score", "ade", "min_ade",
    "kinematic_metrics", "interactive_metrics", "map_based_metrics",
)
METRIC_GROUPS = {"self_play": DRIVING_METRICS, "human_replay": DRIVING_METRICS, "wosac": WOSAC_METRICS}


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
    parser.add_argument("--num-maps", type=positive_int, default=150)
    parser.add_argument("--batch-size", type=positive_int, default=8, help="Complete scenes per batch")
    parser.add_argument("--driving-rollouts", type=positive_int, default=1,
                        help="Rollouts per scene for each of self-play and human-replay (default: 1)")
    parser.add_argument("--wosac-rollouts", type=positive_int, default=32)
    parser.add_argument("--device", default="cuda:0", help="Device, e.g. cuda:0 or cpu (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=ROOT / "pufferlib/config/ocean/drive.ini",
                        help="Training INI (architecture and driving reward/goal settings)")
    return parser.parse_args(argv)


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


def validate_maps(directory, num_maps):
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
        validate_sdc(path)
        scenario_ids.append(scenario_id)
    return scenario_ids


def validate_sdc(path):
    """Reject missing/out-of-range SDC indices before C initializes human-replay scenes."""
    with path.open("rb") as handle:
        handle.seek(16)
        header = handle.read(8)
        if len(header) != 8:
            raise ValueError(f"Truncated SDC metadata in {path}")
        sdc_index, num_tracks = struct.unpack("2i", header)
        if num_tracks < 0 or 28 + 4 * num_tracks > path.stat().st_size:
            raise ValueError(f"Invalid tracks_to_predict metadata in {path}")
        handle.seek(4 * num_tracks, 1)
        num_objects = struct.unpack("i", handle.read(4))[0]
        if not 0 <= sdc_index < num_objects:
            raise ValueError(f"Human-replay requires a valid SDC in {path}: "
                             f"sdc_track_index={sdc_index}, num_objects={num_objects}")


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
    return result


def aggregate_wosac(rows):
    if not rows:
        raise ValueError("No WOSAC scene results")
    return {key: sum(row[key] for row in rows) / len(rows) for key in WOSAC_METRICS}


def check_metrics(row, keys):
    for key in keys:
        if key not in row or not math.isfinite(float(row[key])):
            raise ValueError(f"Missing or non-finite metric {key}: {row}")


def load_policy(config, env, checkpoint, device):
    import torch
    from pufferlib.ocean import torch as policy_module

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
    env.update(map_dir=str(options.test_dir), num_maps=options.num_maps,
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


def evaluate(options, config, scenario_ids, report):
    # Keep --help, input validation, and aggregation tests independent of RL dependencies.
    import numpy as np
    import torch
    from pufferlib.ocean.drive.drive import Drive
    from pufferlib.ocean.benchmark.evaluator import WOSACEvaluator

    if str(options.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu")
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    config["train"]["device"] = options.device
    config["eval"]["wosac_num_rollouts"] = options.wosac_rollouts
    env_configs = {mode: make_env_config(config, options, mode) for mode in METRIC_GROUPS}
    report["settings"] = {
        "device": options.device, "seed": options.seed, "batch_size": options.batch_size,
        "driving_rollouts": options.driving_rollouts, "wosac_rollouts": options.wosac_rollouts,
        **{f"{mode}_env": env_config for mode, env_config in env_configs.items()},
        "policy_name": config["policy_name"], "rnn_name": config["rnn_name"],
        "policy": config["policy"], "rnn": config["rnn"],
        "driving_aggregation": "agent-episode weighted; completion uses total goal counts",
        "human_replay_control": "dataset SDC only; other agents replay logged trajectories",
        "wosac_aggregation": "equal weight per scene; tracks_to_predict only",
        "lstm_reset": "at rollout start only, matching the existing Evaluator",
    }
    policy = None
    wosac_rows = report["per_scene"]["wosac"]

    for mode in ("self_play", "human_replay"):
        driving_rows = report["per_scene"][mode]
        for start in range(0, options.num_maps, options.batch_size):
            indices = list(range(start, min(start + options.batch_size, options.num_maps)))
            expected_ids = [scenario_ids[i] for i in indices]
            env = Drive(**env_configs[mode], map_ids=indices)
            try:
                if env.map_ids != indices or env.scenario_ids != expected_ids:
                    raise RuntimeError("Explicit map selection failed; rebuild the C extension")
                if policy is None:
                    policy = load_policy(config, env, options.checkpoint, options.device)
                for rollout in range(options.driving_rollouts):
                    logs = rollout_driving(env, policy, config, options.seed + start + rollout)
                    for map_id, scenario_id, log in zip(indices, expected_ids, logs):
                        keys = (*DRIVING_METRICS, "n", "goals_sampled_this_episode", "goals_reached_this_episode")
                        check_metrics(log, keys)
                        if mode == "human_replay" and log["n"] != 1:
                            raise RuntimeError(f"Human-replay must control exactly one SDC in map {map_id}")
                        driving_rows.append({"map_id": map_id, "scenario_id": scenario_id, "rollout": rollout,
                                             **{key: float(log[key]) for key in keys}})
            finally:
                env.close()
            report["metrics"][mode] = aggregate_driving(driving_rows)
            print(f"{mode}: {indices[-1] + 1}/{options.num_maps} scenes", flush=True)

    evaluator = WOSACEvaluator(config)
    for start in range(0, options.num_maps, options.batch_size):
        indices = list(range(start, min(start + options.batch_size, options.num_maps)))
        expected_ids = [scenario_ids[i] for i in indices]
        env = Drive(**env_configs["wosac"], map_ids=indices)
        try:
            if env.map_ids != indices or env.scenario_ids != expected_ids:
                raise RuntimeError("Explicit map selection failed; rebuild the C extension")
            env.reset(seed=options.seed + start)
            ground_truth = evaluator.collect_ground_truth_trajectories(env)
            trajectories = evaluator.collect_simulated_trajectories(config, env, policy)
            results = evaluator.compute_metrics(
                ground_truth, trajectories, env.get_global_agent_state(), env.get_road_edge_polylines(),
                aggregate_results=False, drop_last_scenario=False, round_results=False,
            )
            if set(results.index) != set(expected_ids) or len(results) != len(indices):
                missing = sorted(set(expected_ids) - set(results.index))
                raise RuntimeError(f"Incomplete WOSAC results; missing scenarios: {missing}. "
                                   "Check tracks_to_predict and valid reference trajectories.")
            for map_id, scenario_id in zip(indices, expected_ids):
                row = results.loc[scenario_id].to_dict()
                check_metrics(row, (*WOSAC_METRICS, "num_agents_per_scene"))
                wosac_rows.append({"map_id": map_id, "scenario_id": scenario_id,
                                  **{key: float(value) for key, value in row.items()}})
        finally:
            env.close()
        report["metrics"]["wosac"] = aggregate_wosac(wosac_rows)
        print(f"\nWOSAC: {indices[-1] + 1}/{options.num_maps} scenes", flush=True)


def save_report(report, directory):
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"test_metrics_{stamp}_{uuid.uuid4().hex[:8]}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)
    return path


def main(argv=None):
    options = parse_args(argv)
    for name in ("checkpoint", "test_dir", "output_dir", "config"):
        setattr(options, name, getattr(options, name).expanduser().resolve())
    report = {
        "schema_version": 2,
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(options.checkpoint), "test_dir": str(options.test_dir),
        "config": str(options.config), "requested_scenarios": options.num_maps,
        "metrics": {mode: {key: None for key in keys} for mode, keys in METRIC_GROUPS.items()},
        "per_scene": {mode: [] for mode in METRIC_GROUPS},
    }
    try:
        if not options.checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {options.checkpoint}")
        ids = validate_maps(options.test_dir, options.num_maps)
        config = load_settings(options.config)
        evaluate(options, config, ids, report)
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["coverage"] = {
        mode: len({row["scenario_id"] for row in rows}) for mode, rows in report["per_scene"].items()
    }
    report["agent_episodes"] = {
        mode: sum(row["n"] for row in report["per_scene"][mode]) for mode in ("self_play", "human_replay")
    }
    path = save_report(report, options.output_dir)
    print(f"\nStatus: {report['status']}")
    for mode, metrics in report["metrics"].items():
        print(f"\n{mode}:")
        for key, value in metrics.items():
            print(f"  {key:26s}: {value:.6f}" if value is not None else f"  {key:26s}: unavailable")
    print(f"Coverage: {report['coverage']} / {options.num_maps} requested scenes")
    print(f"Saved: {path}")
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
