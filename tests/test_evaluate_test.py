"""Run with: python -m unittest discover -s tests -p test_evaluate_test.py -v"""

import contextlib
import importlib.util
import importlib
import io
import json
from pathlib import Path
import struct
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate_test", ROOT / "scripts/evaluate_test.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def driving_row(n=1, success=1, goals=1, reached=1):
    return {**{key: success for key in evaluation.DRIVING_METRICS}, "n": n,
            "goals_sampled_this_episode": goals, "goals_reached_this_episode": reached}


class EvaluationTests(unittest.TestCase):
    def test_default_dataset_and_rollouts(self):
        options = evaluation.parse_args(["--checkpoint", "model.pt"])
        self.assertEqual(options.test_dir, Path("resources/drive/binaries/testing"))
        self.assertEqual(options.num_maps, 1000)
        self.assertEqual(options.wosac_rollouts, 32)
        self.assertEqual(options.stages, list(evaluation.STAGES))
        self.assertEqual(options.sanity_rollouts, 10)

    def test_rnn_override_reaches_all_stages_and_report(self):
        cases = (
            ("Recurrent", None, "Recurrent"),
            ("None", None, None),
            ("Recurrent", "None", None),
            ("None", "Recurrent", "Recurrent"),
        )
        source = (ROOT / "pufferlib/config/ocean/drive.ini").read_text()
        for configured, override, expected in cases:
            with self.subTest(configured=configured, override=override), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint = root / "model.pt"
                checkpoint.touch()
                config_path = root / "drive.ini"
                config_text = source.replace("rnn_name = Recurrent", f"rnn_name = {configured}")
                config_path.write_text(config_text)
                seen = []

                def fake_stage(options, config, ids, report, mode, *args):
                    seen.append(mode)
                    self.assertEqual(config["rnn_name"], expected)
                    self.assertEqual(config["train"]["use_rnn"], expected is not None)
                    row = {"scenario_id": ids[0], "rollout": 0}
                    if mode == "wosac":
                        row.update({key: 0.5 for key in evaluation.WOSAC_METRICS})
                        row["num_agents_per_scene"] = 1
                    else:
                        row.update(driving_row())
                    if mode == "sanity":
                        row["map_name"] = options.sanity_maps[0]
                    report["per_scene"][mode].append(row)

                args = ["--checkpoint", str(checkpoint), "--config", str(config_path),
                        "--output-dir", str(root / "out"), "--num-maps", "1",
                        "--sanity-maps", evaluation.SANITY_MAPS[0], "--sanity-rollouts", "1"]
                if override is not None:
                    args.extend(["--rnn-name", override])
                with patch.object(evaluation, "run_stage", side_effect=fake_stage), \
                     patch.object(evaluation, "prepare_sanity_maps",
                                  return_value=(["scene_0"], [evaluation.SANITY_MAPS[0]])), \
                     patch.object(evaluation, "validate_stage_maps", return_value=(["scene_0"], [], [])), \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    code = evaluation.main(args)
                report = json.loads(next((root / "out").glob("*.json")).read_text())
                self.assertEqual(code, 0, report["stages"])
                self.assertEqual(seen, list(evaluation.STAGES))
                self.assertEqual(report["settings"]["rnn_name"], expected)
                self.assertEqual(config_path.read_text(), config_text)

    def test_invalid_rnn_override_is_rejected(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            evaluation.parse_args(["--checkpoint", "model.pt", "--rnn-name", "LSTM"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--rnn-name", stderr.getvalue())
        self.assertIn("invalid choice", stderr.getvalue())

    def test_agent_weighted_metrics_and_goal_weighted_completion(self):
        rows = [driving_row(1, 1, 3, 2), driving_row(3, 0, 1, 0)]
        rows[0]["episode_return"], rows[1]["episode_return"] = 10, 2
        rows[0]["collisions_per_agent"], rows[1]["collisions_per_agent"] = 4, 0
        result = evaluation.aggregate_driving(rows)
        self.assertEqual(result["safe_completion_rate"], 0.25)
        self.assertAlmostEqual(result["completion_rate"], 2 / 6)
        self.assertEqual(result["episode_return"], 4)
        self.assertEqual(result["collisions_per_agent"], 1)
        self.assertEqual(result["agent_episodes"], 4)
        self.assertEqual(result["goals_assigned_total"], 6)
        self.assertEqual(result["goals_reached_total"], 2)

    def test_wosac_scene_weighting(self):
        rows = [{key: value for key in evaluation.WOSAC_METRICS} for value in (0.1, 0.9)]
        rows[0]["num_agents_per_scene"] = 1
        rows[1]["num_agents_per_scene"] = 20
        self.assertAlmostEqual(evaluation.aggregate_wosac(rows)["realism_meta_score"], 0.5)
        self.assertAlmostEqual(evaluation.aggregate_wosac(rows)["likelihood_linear_speed"], 0.5)
        self.assertAlmostEqual(evaluation.aggregate_wosac(rows)["realism_meta_score_std"], 0.4)
        self.assertEqual(evaluation.aggregate_wosac(rows)["total_num_agents"], 21)
        self.assertEqual(evaluation.aggregate_wosac(rows[:1])["realism_meta_score_std"], 0)

    def test_wosac_averages_each_metric_over_applicable_scenes(self):
        complete = evaluation.prepare_wosac_row({key: 0.2 for key in evaluation.WOSAC_METRICS}, True)
        partial = evaluation.prepare_wosac_row({key: 0.8 for key in evaluation.WOSAC_METRICS}, False)
        result = evaluation.aggregate_wosac([complete, partial])
        counts = evaluation.wosac_metric_coverage([complete, partial])
        self.assertEqual(result["realism_meta_score"], 0.2)
        self.assertEqual(result["realism_meta_score_std"], 0.)
        self.assertEqual(result["ade"], 0.5)
        self.assertEqual(counts["realism_meta_score"], 1)
        self.assertEqual(counts["likelihood_time_to_collision"], 1)
        self.assertEqual(counts["ade"], 2)
        self.assertEqual(result["total_unique_scenarios"], 2)
        all_partial = evaluation.aggregate_wosac([partial])
        self.assertIsNone(all_partial["realism_meta_score"])
        self.assertIsNone(all_partial["realism_meta_score_std"])
        self.assertEqual(all_partial["ade"], 0.8)
        json.dumps(all_partial, allow_nan=False)

    def test_wosac_applicability_does_not_hide_invalid_metrics(self):
        raw = {key: 0.5 for key in evaluation.WOSAC_METRICS}
        raw["likelihood_time_to_collision"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite metric likelihood_time_to_collision"):
            evaluation.prepare_wosac_row(raw, True)
        with self.assertRaisesRegex(ValueError, "non-finite metric likelihood_time_to_collision"):
            evaluation.aggregate_wosac([raw])
        raw["ade"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite metric ade"):
            evaluation.prepare_wosac_row(raw, False)

    def test_all_nonvehicle_wosac_run_saves_nulls_and_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.pt").touch()
            write_fixture_map(root / "map_000.bin", 0, 2, tracks=(1,))

            def fake_stage(options, config, ids, report, mode, *args):
                row = evaluation.prepare_wosac_row({key: 0.5 for key in evaluation.WOSAC_METRICS}, False)
                row["num_agents_per_scene"] = 1
                report["per_scene"][mode].append({"map_id": 0, "scenario_id": ids[0], **row})

            output = io.StringIO()
            with patch.object(evaluation, "run_stage", side_effect=fake_stage), \
                 contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                code = evaluation.main([
                    "--checkpoint", str(root / "model.pt"), "--test-dir", str(root),
                    "--output-dir", str(root / "out"), "--num-maps", "1", "--stages", "wosac"])
            self.assertEqual(code, 0)
            report = json.loads(next((root / "out").glob("*.json")).read_text())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["stages"]["wosac"]["coverage"], 1)
            self.assertIsNone(report["metrics"]["wosac"]["realism_meta_score"])
            self.assertEqual(report["metric_coverage"]["wosac"]["realism_meta_score"], 0)
            self.assertEqual(report["metric_coverage"]["wosac"]["ade"], 1)
            self.assertIn("N/A (0/1 scenes)", output.getvalue())
            self.assertIn("likelihood_time_to_collision", report["per_scene"]["wosac"][0]["not_applicable"])

    def test_map_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for i in range(3):
                write_fixture_map(directory / f"map_{i:03d}.bin", i, 2)
            self.assertEqual(evaluation.validate_maps(directory, 3), ["scene_0", "scene_1", "scene_2"])
            with self.assertRaisesRegex(ValueError, "Expected exactly"):
                evaluation.validate_maps(directory, 4)
            with self.assertRaisesRegex(ValueError, "Expected exactly"):
                evaluation.validate_maps(directory, 2)
            with (directory / "map_002.bin").open("r+b") as handle:
                handle.write(b"scene_0".ljust(16, b"\0"))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                evaluation.validate_maps(directory, 3)

    def test_human_replay_requires_valid_sdc(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map_000.bin"
            for invalid_sdc in (-1, 2):
                write_fixture_map(path, 0, 2, sdc_index=invalid_sdc)
                with self.assertRaisesRegex(ValueError, "valid SDC"):
                    evaluation.validate_maps(Path(directory), 1)
            write_fixture_map(path, 0, 2, sdc_index=1)
            self.assertEqual(evaluation.validate_maps(Path(directory), 1), ["scene_0"])

    def test_wosac_rejects_invalid_track_indices_in_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map_000.bin"
            write_fixture_map(path, 0, 2, tracks=(34,))
            with self.assertRaisesRegex(ValueError, "WOSAC requires valid tracks_to_predict"):
                evaluation.validate_maps(Path(directory), 1)

    def test_report_serialization_and_unique_names(self):
        with tempfile.TemporaryDirectory() as directory:
            report = {"status": "complete", "metrics": {"ade": 1.2}}
            first = evaluation.save_report(report, Path(directory))
            second = evaluation.save_report(report, Path(directory))
            self.assertNotEqual(first, second)
            self.assertEqual(json.loads(first.read_text()), report)

    def test_missing_checkpoint_produces_failed_report(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ["--checkpoint", str(Path(directory) / "missing.pt"), "--test-dir", directory,
                    "--output-dir", str(Path(directory) / "out")]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(evaluation.main(args), 1)
            report = json.loads(next((Path(directory) / "out").glob("*.json")).read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(set(report["metrics"]), set(evaluation.STAGES))
            self.assertEqual(report["skipped"], {})
            self.assertTrue(all(value is None for group in report["metrics"].values() for value in group.values()))

    def test_nonfinite_metric_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            evaluation.check_metrics({"ade": float("nan")}, ["ade"])

    def test_four_stages_cover_every_map_in_order(self):
        self.check_passes()

    def test_wosac_failure_does_not_prevent_human_replay(self):
        self.check_passes(bad_tracks=True)

    def test_missing_wosac_target_is_reported_and_replay_continues(self):
        self.check_passes(missing_target=True)

    def test_nonvehicle_scene_keeps_other_scenes_and_later_batches(self):
        self.check_passes(nonvehicle_scenes=("scene_1",))

    def test_all_nonvehicle_scenes_still_complete_every_stage(self):
        self.check_passes(nonvehicle_scenes=("scene_0", "scene_1", "scene_2"))

    def check_passes(self, bad_tracks=False, missing_target=False, nonvehicle_scenes=()):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i in range(3):
                write_fixture_map(root / f"map_{i:03d}.bin", i, 2, sdc_index=1,
                                  tracks=(34,) if bad_tracks else (0, 1))
            options = evaluation.parse_args(["--checkpoint", "model.pt", "--test-dir", str(root),
                                            "--num-maps", "3", "--batch-size", "2", "--device", "cpu",
                                            "--driving-rollouts", "2", "--sanity-rollouts", "2",
                                            "--sanity-maps", *evaluation.SANITY_MAPS[:2]])
            config = evaluation.load_settings(ROOT / "pufferlib/config/ocean/drive.ini")
            original_config = json.dumps(config, sort_keys=True)
            report = evaluation.new_report(options)
            instances = []
            loads = []
            self_test = self

            def prepare_sanity(opts, destination):
                for i in range(2):
                    write_fixture_map(destination / f"map_{i:03d}.bin", i, 2, sdc_index=-1)
                return ["scene_0", "scene_1"], options.sanity_maps

            class FakeDrive:
                def __init__(self, map_ids, **kwargs):
                    if instances:
                        self_test.assertTrue(instances[-1].closed)
                    self.map_ids = map_ids
                    self.scenario_ids = [f"scene_{i}" for i in map_ids]
                    self.control_mode = kwargs["control_mode"]
                    self.kwargs = kwargs
                    self.closed = False
                    self_test.assertEqual(kwargs["resample_frequency"], 0)
                    self_test.assertEqual(kwargs["termination_mode"], 0)
                    if self.control_mode == "control_wosac":
                        self_test.assertEqual(kwargs["map_dir"], str(root))
                        self_test.assertEqual(kwargs["init_steps"], 10)
                        self_test.assertTrue(kwargs["defer_reset"])
                    else:
                        self_test.assertEqual(kwargs["init_steps"], 0)
                        self_test.assertFalse(kwargs["defer_reset"])
                    instances.append(self)

                def reset(self, **kwargs):
                    pass

                def close(self):
                    self.closed = True

                def get_global_agent_state(self):
                    return {}

                def get_road_edge_polylines(self):
                    return {}

            class FakeResults:
                def __init__(self, indices):
                    self.index = indices
                    self.loc = self

                def __len__(self):
                    return len(self.index)

                def __getitem__(self, key):
                    row = {**{name: 0.5 for name in evaluation.WOSAC_METRICS}, "num_agents_per_scene": 2}
                    if key in nonvehicle_scenes:
                        row.update({name: np.nan for name in evaluation.WOSAC_TTC_DEPENDENT})
                    return SimpleNamespace(to_dict=lambda: row)

            class FakeWosac:
                def __init__(self, config):
                    pass

                def collect_ground_truth_trajectories(self, env):
                    count = 1 if missing_target else 2
                    return {"scenario_id": np.array([[s] for s in env.scenario_ids for _ in range(count)]),
                            "id": np.array([[i] for _ in env.scenario_ids for i in range(count)]),
                            "is_vehicle": np.array([[s not in nonvehicle_scenes]
                                                    for s in env.scenario_ids for _ in range(count)]),
                            "is_track_to_predict": np.ones((len(env.scenario_ids) * count, 1), dtype=bool)}

                def collect_simulated_trajectories(self, config, env, policy):
                    return {}

                def compute_metrics(self, truth, *args, **kwargs):
                    self_test.assertFalse(kwargs["drop_last_scenario"])
                    self_test.assertFalse(kwargs["round_results"])
                    self_test.assertTrue(kwargs["allow_undefined_ttc"])
                    return FakeResults(list(dict.fromkeys(truth["scenario_id"][:, 0])))

            modules = {
                "torch": SimpleNamespace(manual_seed=lambda seed: None),
                "pufferlib.ocean.drive.drive": SimpleNamespace(Drive=FakeDrive),
                "pufferlib.ocean.benchmark.evaluator": SimpleNamespace(WOSACEvaluator=FakeWosac),
            }
            def fake_rollout(env, *args):
                n, success = (1, 0.25) if env.control_mode == "control_sdc_only" else (2, 1.0)
                return [driving_row(n=n, success=success) for _ in env.map_ids]

            def load_policy(*args):
                loads.append(args[1].control_mode)
                return object()

            with patch.dict("sys.modules", modules), patch.object(evaluation, "load_policy", side_effect=load_policy), \
                 patch.object(evaluation, "rollout_driving", side_effect=fake_rollout), \
                 patch.object(evaluation, "prepare_sanity_maps", side_effect=prepare_sanity), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                evaluation.evaluate(options, config, report)
            wosac_batches = 0 if bad_tracks else 1 if missing_target else 2
            self.assertEqual([env.control_mode for env in instances],
                             ["control_agents"] * 3 + ["control_wosac"] * wosac_batches + ["control_sdc_only"] * 2)
            self.assertTrue(all(env.closed for env in instances))
            self.assertEqual(json.dumps(config, sort_keys=True), original_config)
            self.assertEqual(report["stages"]["sanity"]["status"], "complete")
            self.assertEqual(report["stages"]["self_play"]["status"], "complete")
            self.assertEqual(report["stages"]["human_replay"]["status"], "complete")
            self.assertEqual(report["metrics"]["sanity"]["solved_maps"], 2)
            self.assertEqual(len(report["per_scene"]["self_play"]), 6)
            self.assertEqual(len(report["per_scene"]["human_replay"]), 6)
            self.assertEqual(report["metrics"]["human_replay"]["safe_completion_rate"], 0.25)
            self.assertTrue(all(row["ego_object_id"] == 1 for row in report["per_scene"]["human_replay"]))
            if bad_tracks or missing_target:
                self.assertEqual(report["stages"]["wosac"]["status"], "failed")
                self.assertEqual(report["per_scene"]["wosac"], [])
            else:
                self.assertEqual(report["stages"]["wosac"]["status"], "complete")
                self.assertEqual(len(report["per_scene"]["wosac"]), 3)
                self.assertEqual(report["metrics"]["wosac"]["total_num_agents"], 6)
                self.assertEqual(loads, ["control_agents", "control_agents", "control_wosac", "control_sdc_only"])
                self.assertEqual(report["metric_coverage"]["wosac"]["ade"], 3)
                self.assertEqual(report["metric_coverage"]["wosac"]["realism_meta_score"],
                                 3 - len(nonvehicle_scenes))
                for row in report["per_scene"]["wosac"]:
                    if row["scenario_id"] in nonvehicle_scenes:
                        for key in evaluation.WOSAC_TTC_DEPENDENT:
                            self.assertIsNone(row[key])
                            self.assertIn(key, row["not_applicable"])
                    self.assertEqual(row["ade"], 0.5)
                json.dumps(report, allow_nan=False)
            for mode, keys in evaluation.METRIC_GROUPS.items():
                self.assertEqual(set(report["metrics"][mode]), set(keys))

    def test_invalid_sdc_fails_without_replacing_vehicle_or_rewriting_map(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.pt").touch()
            path = root / "map_000.bin"
            for sdc in (-1, 35):
                with self.subTest(sdc=sdc):
                    # Two otherwise valid vehicles are present, but neither can replace the SDC.
                    write_fixture_map(path, 0, 2, sdc_index=sdc, tracks=(0,))
                    before = path.read_bytes()
                    output = root / f"out_{sdc}"
                    with patch.object(evaluation, "run_stage") as run, \
                         contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        code = evaluation.main(["--checkpoint", str(root / "model.pt"), "--test-dir", str(root),
                                                "--output-dir", str(output), "--num-maps", "1",
                                                "--stages", "human_replay"])
                    self.assertEqual(code, 1)
                    run.assert_not_called()
                    self.assertEqual(path.read_bytes(), before)
                    report = json.loads(next(output.glob("*.json")).read_text())
                    self.assertEqual(report["stages"]["human_replay"]["status"], "failed")
                    self.assertIn("SDC", report["stages"]["human_replay"]["error"])
                    self.assertEqual(report["coverage"]["human_replay"], 0)
                    self.assertEqual(report["per_scene"]["human_replay"], [])

    def test_human_replay_report_preserves_original_sdc_index_and_object_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            checkpoint.touch()
            path = root / "map_000.bin"
            write_fixture_map(path, 0, 2, sdc_index=1, tracks=(0,))
            # Object IDs deliberately differ from list indices.
            data = bytearray(path.read_bytes())
            object_size = 16 + 8 * 91 * 4 + 28
            struct.pack_into("i", data, 36 + 8, 35)
            struct.pack_into("i", data, 36 + object_size + 8, 75)
            path.write_bytes(data)
            before = path.read_bytes()
            def fake_stage(options, config, ids, report, mode, names, targets, egos):
                self.assertEqual(mode, "human_replay")
                self.assertEqual(options.test_dir, root)
                self.assertFalse(hasattr(options, "prepared_test_dir"))
                self.assertEqual(egos[0]["ego_object_index"], 1)
                self.assertEqual(egos[0]["ego_object_id"], 75)
                report["per_scene"][mode].append({**egos[0], "rollout": 0, **driving_row()})
            with patch.object(evaluation, "run_stage", side_effect=fake_stage), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = evaluation.main(["--checkpoint", str(checkpoint), "--test-dir", str(root),
                                        "--output-dir", str(root / "out"), "--num-maps", "1",
                                        "--stages", "human_replay"])
            self.assertEqual(code, 0)
            self.assertEqual(path.read_bytes(), before)
            report = json.loads(next((root / "out").glob("*.json")).read_text())
            self.assertEqual(report["replay_ego"], "sdc")
            self.assertEqual(set(report["metrics"]), set(evaluation.STAGES))
            self.assertEqual(report["ego_selection"][0]["ego_object_id"], 75)
            self.assertEqual(report["coverage"]["human_replay"], 1)

    def test_stage_validation_distinguishes_sdc_and_prediction_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "map_000.bin"
            write_fixture_map(path, 0, 2, sdc_index=-1, tracks=(0,))
            evaluation.validate_stage_maps(root, 1, "self_play")
            evaluation.validate_stage_maps(root, 1, "wosac", 10)
            with self.assertRaisesRegex(ValueError, "valid original SDC"):
                evaluation.validate_stage_maps(root, 1, "human_replay")
            write_fixture_map(path, 0, 2, sdc_index=0)
            evaluation.validate_stage_maps(root, 1, "human_replay")
            with self.assertRaisesRegex(ValueError, "nonempty"):
                evaluation.validate_stage_maps(root, 1, "wosac", 10)
            write_fixture_map(path, 0, 2, sdc_index=0, tracks=(0, 0))
            with self.assertRaisesRegex(ValueError, "unique"):
                evaluation.validate_stage_maps(root, 1, "wosac", 10)

    def test_missing_sanity_resource_does_not_hide_test_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.pt").touch()
            write_fixture_map(root / "map_000.bin", 0, 2)
            def fake_stage(options, config, ids, report, mode, *args):
                self.assertEqual(mode, "self_play")
                report["per_scene"][mode].append({"scenario_id": ids[0], "rollout": 0, **driving_row()})
            with patch.object(evaluation, "run_stage", side_effect=fake_stage), \
                 patch.object(evaluation, "prepare_sanity_maps", side_effect=FileNotFoundError("Missing sanity map")), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = evaluation.main(["--checkpoint", str(root / "model.pt"), "--test-dir", str(root),
                                        "--output-dir", str(root / "out"), "--num-maps", "1",
                                        "--stages", "self_play", "sanity"])
            self.assertEqual(code, 1)
            files = list((root / "out").glob("*.json"))
            self.assertEqual(len(files), 1)
            report = json.loads(files[0].read_text())
            self.assertEqual(report["stages"]["sanity"]["status"], "failed")
            self.assertEqual(report["stages"]["self_play"]["status"], "complete")
            self.assertEqual(report["metrics"]["self_play"]["agent_episodes"], 1)
            self.assertEqual(report["stage_order"], list(evaluation.STAGES))

    def test_partial_sanity_map_is_not_marked_solved(self):
        report = {"per_scene": {"sanity": [
            {"map_name": "simple_turn", "rollout": 0, **driving_row()},
            {"map_name": "s_curve", "rollout": 0, **driving_row(success=0.94)},
            {"map_name": "s_curve", "rollout": 1, **driving_row(success=0.94)},
        ]}, "metrics": {"sanity": {}}}
        evaluation.summarize_sanity(report, 2)
        self.assertIsNone(report["sanity_results"][0]["solved"])
        self.assertFalse(report["sanity_results"][1]["solved"])
        self.assertEqual(report["metrics"]["sanity"]["evaluated_maps"], 1)
        self.assertEqual(report["metrics"]["sanity"]["solved_rate"], 0)

    def test_failed_stage_keeps_partial_metrics_and_runs_next_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i in range(2):
                write_fixture_map(root / f"map_{i:03d}.bin", i, 2)
            options = evaluation.parse_args(["--checkpoint", "model.pt", "--test-dir", str(root),
                                            "--num-maps", "2", "--stages", "self_play", "human_replay"])
            report = evaluation.new_report(options)
            seen = []
            def fake_stage(options, config, ids, report, mode, *args):
                seen.append(mode)
                if mode == "self_play":
                    report["per_scene"][mode].append({"scenario_id": ids[0], **driving_row(n=2)})
                    raise RuntimeError("Failed second batch")
                for scenario_id in ids:
                    report["per_scene"][mode].append({"scenario_id": scenario_id, **driving_row()})
            config = evaluation.load_settings(ROOT / "pufferlib/config/ocean/drive.ini")
            with patch.object(evaluation, "run_stage", side_effect=fake_stage), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                evaluation.evaluate(options, config, report)
            self.assertEqual(seen, ["self_play", "human_replay"])
            self.assertEqual(report["stages"]["self_play"]["status"], "failed")
            self.assertEqual(report["stages"]["self_play"]["coverage"], 1)
            self.assertEqual(report["metrics"]["self_play"]["agent_episodes"], 2)
            self.assertEqual(report["stages"]["human_replay"]["status"], "complete")

    def test_aggregation_failure_does_not_stop_later_stages(self):
        for rollout_failed in (False, True):
            with self.subTest(rollout_failed=rollout_failed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "model.pt").touch()
                write_fixture_map(root / "map_000.bin", 0, 2)
                seen = []

                def fake_stage(options, config, ids, report, mode, *args):
                    seen.append(mode)
                    row = driving_row(goals=0, reached=0) if mode == "self_play" else driving_row()
                    report["per_scene"][mode].append({"scenario_id": ids[0], "rollout": 0, **row})
                    if mode == "self_play":
                        # A stale earlier-batch summary must not survive this failure.
                        report["metrics"][mode] = evaluation.aggregate_driving([driving_row()])
                        if rollout_failed:
                            raise RuntimeError("Rollout failed")

                with patch.object(evaluation, "run_stage", side_effect=fake_stage), \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    code = evaluation.main([
                        "--checkpoint", str(root / "model.pt"), "--test-dir", str(root),
                        "--output-dir", str(root / "out"), "--num-maps", "1",
                        "--stages", "self_play", "human_replay",
                    ])
                self.assertEqual(code, 1)
                report = json.loads(next((root / "out").glob("*.json")).read_text())
                self.assertEqual(seen, ["self_play", "human_replay"])
                self.assertEqual(report["stages"]["self_play"]["status"], "failed")
                self.assertIn("No assigned goals", report["stages"]["self_play"]["aggregation_error"])
                if rollout_failed:
                    self.assertIn("Rollout failed", report["stages"]["self_play"]["error"])
                self.assertTrue(all(value is None for value in report["metrics"]["self_play"].values()))
                self.assertEqual(len(report["per_scene"]["self_play"]), 1)
                self.assertEqual(report["stages"]["self_play"]["coverage"], 1)
                self.assertEqual(report["stages"]["human_replay"]["status"], "complete")


class WosacInputValidationTests(unittest.TestCase):
    def setUp(self):
        import numpy as np

        self.truth = {key: np.zeros((2, 1, 81), dtype=np.float32) for key in ("x", "y", "heading")}
        self.truth["x"][1] = 10.
        self.truth.update(
            valid=np.ones((2, 1, 81), dtype=bool), id=np.array([[35], [75]]),
            is_vehicle=np.ones((2, 1), dtype=bool), is_track_to_predict=np.ones((2, 1), dtype=bool),
            scenario_id=np.array([["scene_0"], ["scene_0"]]),
        )
        self.sim = {key: np.repeat(self.truth[key], 2, axis=1) for key in ("x", "y", "heading")}
        self.sim["id"] = np.broadcast_to(self.truth["id"][:, :, None], (2, 2, 81)).copy()
        self.state = {"length": np.full(2, 4.), "width": np.full(2, 2.)}
        self.edges = {"x": np.array([-200., 200.]), "y": np.array([-10., -10.]),
                      "lengths": np.array([2]), "scenario_id": np.array(["scene_0"])}

    def compute(self, benchmark, **kwargs):
        evaluator = benchmark.WOSACEvaluator({"train": {"device": "cpu"}, "eval": {"wosac_init_steps": 10}})
        return evaluator.compute_metrics(self.truth, self.sim, self.state, self.edges,
                                         drop_last_scenario=False, round_results=False, **kwargs)

    def test_nonfinite_simulation_cannot_be_hidden_by_scene_averaging(self):
        import numpy as np

        with benchmark_module() as benchmark:
            # Previously pandas skipped the broken object's ADE and reported 0
            # while still counting both objects in num_agents_per_scene.
            self.sim["x"][0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "Non-finite simulated x.*object 35.*rollout 0.*frame 10"):
                self.compute(benchmark)

    def test_nonfinite_interaction_partner_is_also_rejected(self):
        import numpy as np

        self.truth["is_track_to_predict"][1] = False
        with benchmark_module() as benchmark:
            for field in ("x", "y", "heading"):
                for source, label in ((self.sim, "simulated"), (self.truth, "reference")):
                    with self.subTest(field=field, source=label):
                        original = source[field][1, 0, 20]
                        source[field][1, 0, 20] = np.inf
                        with self.assertRaisesRegex(ValueError, f"Non-finite {label} {field}.*object 75"):
                            self.compute(benchmark)
                        source[field][1, 0, 20] = original

    def test_invalid_reference_frames_may_have_missing_coordinates(self):
        import numpy as np

        self.truth["valid"][0, 0, -1] = False
        for field in ("x", "y", "heading"):
            self.truth[field][0, 0, -1] = np.nan
        with benchmark_module() as benchmark:
            row = self.compute(benchmark).loc["scene_0"].to_dict()
        evaluation.check_metrics(row, evaluation.WOSAC_METRICS)
        self.assertEqual(row["ade"], 0.)

    def test_invalid_dimensions_are_rejected(self):
        import numpy as np

        with benchmark_module() as benchmark:
            for value in (0., -1., np.nan, np.inf):
                with self.subTest(value=value):
                    self.state["width"][1] = value
                    with self.assertRaisesRegex(ValueError, "finite, positive agent width"):
                        self.compute(benchmark)

    def test_wosac_reports_undefined_ttc_for_nonvehicle_only_targets(self):
        self.truth["is_vehicle"][:] = False
        with benchmark_module() as benchmark:
            with self.assertRaisesRegex(ValueError, "likelihood_time_to_collision.*scene_0.*vehicle"):
                self.compute(benchmark)

    def test_wosac_keeps_other_metrics_when_ttc_is_not_applicable(self):
        import numpy as np

        # A vehicle outside tracks_to_predict must not make TTC applicable.
        self.truth["is_track_to_predict"][0] = False
        self.truth["is_vehicle"][1] = False
        with benchmark_module() as benchmark:
            result = self.compute(benchmark, allow_undefined_ttc=True)
        self.assertEqual(list(result.index), ["scene_0"])
        raw = result.loc["scene_0"].to_dict()
        for key in evaluation.WOSAC_TTC_DEPENDENT:
            self.assertTrue(np.isnan(raw[key]), key)
        row = evaluation.prepare_wosac_row(raw, False)
        self.assertEqual(row["ade"], 0.)
        self.assertEqual(row["num_agents_per_scene"], 1)
        self.assertGreater(row["kinematic_metrics"], 0.)
        json.dumps(row, allow_nan=False)

    def test_wosac_mixed_scene_batch_retains_applicability(self):
        import numpy as np

        self.truth["scenario_id"][1] = "scene_1"
        self.truth["is_vehicle"][1] = False
        self.edges = {"x": np.tile(self.edges["x"], 2), "y": np.tile(self.edges["y"], 2),
                      "lengths": np.array([2, 2]), "scenario_id": np.array(["scene_0", "scene_1"])}
        with benchmark_module() as benchmark:
            result = self.compute(benchmark, allow_undefined_ttc=True)
        self.assertEqual(list(result.index), ["scene_0", "scene_1"])
        complete = evaluation.prepare_wosac_row(result.loc["scene_0"].to_dict(), True)
        partial = evaluation.prepare_wosac_row(result.loc["scene_1"].to_dict(), False)
        summary = evaluation.aggregate_wosac([complete, partial])
        self.assertEqual(summary["realism_meta_score"], complete["realism_meta_score"])
        self.assertEqual(summary["ade"], 0.)
        self.assertEqual(summary["total_num_agents"], 2)

    def test_allowing_undefined_ttc_does_not_hide_other_undefined_metrics(self):
        # One reference frame cannot define speed or acceleration.
        self.truth["valid"][:, :, 1:] = False
        with benchmark_module() as benchmark:
            with self.assertRaisesRegex(ValueError, "Undefined WOSAC metric likelihood_linear_speed"):
                self.compute(benchmark, allow_undefined_ttc=True)

    def test_wosac_mixed_targets_exclude_nonvehicles_only_from_ttc(self):
        self.truth["is_vehicle"][1] = False
        with benchmark_module() as benchmark:
            row = self.compute(benchmark).loc["scene_0"].to_dict()
        evaluation.check_metrics(row, evaluation.WOSAC_METRICS)
        self.assertEqual(row["num_agents_per_scene"], 2)
        self.assertEqual(row["ade"], 0.)


class ConverterMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Conversion is real Python code; only unused simulator imports are stubbed.
        importlib.import_module("numpy")  # Keep native module imports outside patch.dict restoration.
        package = ModuleType("pufferlib")
        package.PufferEnv = object
        modules = {"pufferlib": package, "gymnasium": ModuleType("gymnasium"),
                   "pufferlib.ocean.drive": SimpleNamespace(binding=None)}
        spec = importlib.util.spec_from_file_location("drive_converter", ROOT / "pufferlib/ocean/drive/drive.py")
        cls.converter = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", modules):
            spec.loader.exec_module(cls.converter)

    def test_reported_scene_indices_are_not_guessed_from_ids(self):
        ids = [0, 5, 7, 8, 12, 14, 15, 25, 27, 28, 33, 35, 38, 40, 42, 43, 46, 47, 49, 50, 73, 74, 75]
        scene = {"scenario_id": "reported_scene", "objects": [{"id": i} for i in ids],
                 "metadata": {"sdc_track_index": 35,
                              "tracks_to_predict": [{"track_index": 34}, {"track_index": 6}]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map_000.bin"
            path.write_bytes(b"existing binary")
            with self.assertRaisesRegex(ValueError, r"sdc_track_index=35.*tracks_to_predict\[0\].track_index=34"):
                self.converter.save_map_binary(scene, path, 0)
            self.assertEqual(path.read_bytes(), b"existing binary")

    def test_valid_indices_are_preserved_independently_of_object_ids(self):
        scene = {"scenario_id": "scene_0", "objects": [{"id": i} for i in (35, 0, 74)],
                 "metadata": {"sdc_track_index": 1, "tracks_to_predict": [{"track_index": 0}, {"track_index": 2}]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map_000.bin"
            self.converter.save_map_binary(scene, path, 0)
            header = path.read_bytes()[16:40]
            self.assertEqual(struct.unpack("6i", header), (1, 2, 0, 2, 3, 0))
            self.assertEqual(evaluation.validate_maps(Path(directory), 1), ["scene_0"])

    def test_invalid_wosac_reference_is_rejected_before_creating_file(self):
        scene = {"objects": [{"id": 34}, {"id": 6}],
                 "metadata": {"sdc_track_index": 0, "tracks_to_predict": [{"track_index": 34}]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map_000.bin"
            with self.assertRaisesRegex(ValueError, r"tracks_to_predict\[0\].track_index=34"):
                self.converter.save_map_binary(scene, path, 0)
            self.assertFalse(path.exists())

    def test_maps_without_waymo_metadata_keep_missing_sdc_sentinel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map_000.bin"
            self.converter.save_map_binary({"objects": [{"id": 1}]}, path, 0)
            self.assertEqual(struct.unpack("4i", path.read_bytes()[16:32]), (-1, 0, 1, 0))

    def test_load_map_and_worker_preserve_source_metadata(self):
        scene = {"scenario_id": "scene_0", "objects": [{"id": 35}, {"id": 75}],
                 "metadata": {"sdc_track_index": 1, "tracks_to_predict": [{"track_index": 0}]}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scene.json"
            source.write_text(json.dumps(scene))
            before = source.read_bytes()
            self.converter.load_map(source, 0, root / "direct.bin")
            result = self.converter._process_single_map((0, source, root / "worker.bin"))
            self.assertTrue(result[2], result[3])
            for name in ("direct.bin", "worker.bin"):
                self.assertEqual(struct.unpack_from("5i", (root / name).read_bytes(), 16), (1, 1, 0, 2, 0))
            self.assertEqual(source.read_bytes(), before)

    def test_sanity_conversion_uses_unique_ids_and_preserves_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = {"scenario_id": "shared_id", "objects": [{"id": 8, "type": "vehicle"}], "roads": []}
            source = json.dumps(scene)
            for name in ("simple_turn", "s_curve"):
                (root / f"{name}.json").write_text(source)
            options = SimpleNamespace(sanity_dir=root, sanity_maps=["s_curve", "simple_turn"])
            modules = {"pufferlib.ocean.drive.drive": self.converter}
            with patch.dict("sys.modules", modules):
                ids, names = evaluation.prepare_sanity_maps(options, root / "prepared")
                self.assertEqual(ids, ["sanity_000", "sanity_001"])
                self.assertEqual(names, options.sanity_maps)
                self.assertEqual((root / "simple_turn.json").read_text(), source)
                self.assertEqual(evaluation.inspect_map(root / "prepared/map_000.bin")[0], -1)
                options.sanity_maps = ["missing"]
                with self.assertRaisesRegex(FileNotFoundError, "Missing sanity map"):
                    evaluation.prepare_sanity_maps(options, root / "missing-output")

    def test_curated_sanity_does_not_use_stale_waymo_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "three_or_four_point_turn"
            source = root / f"{name}.json"
            source.write_text(json.dumps({
                "objects": [{"id": 999, "type": "vehicle"}], "roads": [],
                "metadata": {"sdc_track_index": 32, "tracks_to_predict": [{"track_index": 40}]},
            }))
            original = source.read_bytes()
            options = SimpleNamespace(sanity_dir=root, sanity_maps=[name])
            with patch.dict("sys.modules", {"pufferlib.ocean.drive.drive": self.converter}):
                evaluation.prepare_sanity_maps(options, root / "prepared")
            sdc, tracks, objects = evaluation.inspect_map(root / "prepared/map_000.bin")
            self.assertEqual((sdc, tracks), (-1, ()))
            self.assertEqual([obj["id"] for obj in objects], [999])
            self.assertEqual(source.read_bytes(), original)


def write_fixture_map(path, map_id, count, sdc_index=None, moving=False, speed=0., tracks=(),
                      goal_distance=None):
    """Tiny real binary map: valid stationary vehicles with distant goals."""
    with path.open("wb") as handle:
        handle.write(struct.pack("16s", f"scene_{map_id}".encode()))
        if sdc_index is None:
            sdc_index = 0 if count else -1
        handle.write(struct.pack("2i", sdc_index, len(tracks)))
        handle.write(struct.pack(f"{len(tracks)}i", *tracks))
        handle.write(struct.pack("2i", count, 1))
        for i in range(count):
            handle.write(struct.pack("4i", map_id, 1, i, 91))
            positions = [i * 10. + (0.1 * t if moving else 0.) for t in range(91)]
            for values in (positions, [0.] * 91, [0.] * 91,
                           [0.] * 91, [speed] * 91, [0.] * 91, [0.] * 91):
                handle.write(struct.pack("91f", *values))
            handle.write(struct.pack("91i", *([1] * 91)))
            goal_x = 100. if goal_distance is None else i * 10. + goal_distance
            handle.write(struct.pack("6fi", 2., 4., 1., goal_x, 0., 0., 0))
        # Lane geometry gives the real simulator a finite map extent.
        handle.write(struct.pack("4i", map_id, 4, 100, 2))
        handle.write(struct.pack("6f", -200., 200., -10., 10., 0., 0.))
        handle.write(struct.pack("6fi", 0., 0., 0., 0., 0., 0., 0))


class NativeMapSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        extensions = list((ROOT / "pufferlib/ocean/drive").glob("binding*.so"))
        if not extensions:
            raise unittest.SkipTest("Build the native binding first")
        spec = importlib.util.spec_from_file_location("binding", extensions[0])
        cls.binding = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.binding)

    def test_selected_maps_preserve_order_and_complete_agent_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            for i, count in enumerate([2, 3, 4]):
                write_fixture_map(Path(directory) / f"map_{i:03d}.bin", i, count)
            offsets, maps, n = self.binding.shared(
                map_dir=directory, num_agents=1, num_maps=3, init_mode=0, control_mode=1,
                init_steps=0, goal_behavior=0, goal_target_distance=30., max_controlled_agents=32,
                map_ids=[2, 0],
            )
            self.assertEqual(maps, [2, 0])
            self.assertEqual(n, 2)
            self.assertEqual(offsets, [0, 4, 6])  # Not truncated to num_agents=1.

    def test_selected_map_without_agents_fails_instead_of_skipping(self):
        with tempfile.TemporaryDirectory() as directory:
            write_fixture_map(Path(directory) / "map_000.bin", 0, 0)
            with self.assertRaisesRegex(ValueError, "No controllable agents"):
                self.binding.shared(
                    map_dir=directory, num_agents=1, num_maps=1, init_mode=0, control_mode=1,
                    init_steps=0, goal_behavior=0, goal_target_distance=30., max_controlled_agents=32,
                    map_ids=[0],
                )

    def test_wosac_retains_prediction_targets_beyond_actor_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            limit = self.binding.MAX_AGENTS
            targets = (0, 1, 2, limit + 1, limit + 3)
            write_fixture_map(Path(directory) / "map_000.bin", 0, limit + 5,
                              tracks=targets, goal_distance=100.)
            for mode, count in ((2, limit), (1, limit), (3, 1)):
                with self.subTest(control_mode=mode):
                    offsets, _, _ = self.binding.shared(
                        map_dir=directory, num_agents=1, num_maps=1, init_mode=0,
                        control_mode=mode, init_steps=10, goal_behavior=2,
                        goal_target_distance=30., max_controlled_agents=limit, map_ids=[0])
                    self.assertEqual(offsets, [0, count])
                    env = NativeDriveHarness(self.binding, directory, init_steps=10,
                                             num_agents=count, control_mode=mode)
                    try:
                        env.reset()
                        truth = env.get_ground_truth_trajectories()
                        ids = truth["id"][:, 0].tolist()
                        self.assertEqual(ids[0], 0)  # Original SDC stays first.
                        self.assertEqual(len(set(ids)), count)
                        if mode == 2:
                            loaded = truth["id"][truth["is_track_to_predict"]].tolist()
                            self.assertEqual(set(loaded), set(targets))
                        else:
                            self.assertEqual(ids, list(range(count)))
                    finally:
                        env.close()

    def test_real_scene_logs_export_completion_metrics(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            write_fixture_map(Path(directory) / "map_000.bin", 0, 2)
            obs_size = self.binding.EGO_FEATURES_CLASSIC + (self.binding.MAX_AGENTS - 1) * 7 + 128 * 7
            obs = np.zeros((2, obs_size), dtype=np.float32)
            actions = np.full((2, 1), 45, dtype=np.int32)
            rewards = np.zeros(2, dtype=np.float32)
            terminals = np.zeros(2, dtype=bool)
            truncations = np.zeros(2, dtype=bool)
            kwargs = evaluation.load_settings(ROOT / "pufferlib/config/ocean/drive.ini")["env"]
            kwargs.update(map_dir=directory, map_id=0, max_agents=2, human_agent_idx=0,
                          ini_file=str(ROOT / "pufferlib/config/ocean/drive.ini"),
                          action_type=0, dynamics_model=0, control_mode=1, init_mode=0,
                          episode_length=3, termination_mode=0, goal_radius=1000.)
            env = self.binding.env_init(obs, actions, rewards, terminals, truncations, 42, **kwargs)
            vec = self.binding.vectorize(env)
            try:
                self.binding.vec_reset(vec, 42)
                self.binding.vec_step(vec)
                self.binding.vec_step(vec)
                self.assertTrue(truncations.all())
                per_scene = self.binding.env_log(env, 0)
                aggregate = self.binding.vec_log(vec, 2)
                for result in (per_scene, aggregate):
                    evaluation.check_metrics(result, evaluation.DRIVING_METRICS)
                    self.assertEqual(result["n"], 2)
                    self.assertEqual(result["completion_rate"], 1.)
                    self.assertEqual(result["safe_completion_rate"], 1.)
                    self.assertEqual(result["collision_rate"], 0.)
                    self.assertEqual(result["offroad_rate"], 0.)
            finally:
                self.binding.vec_close(vec)

    def test_human_replay_controls_marked_sdc_and_replays_other_vehicle(self):
        self.check_human_replay()

    def test_human_replay_resumes_after_a_missing_log_frame(self):
        self.check_human_replay(missing_frame=True)

    def check_human_replay(self, missing_frame=False):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "map_000.bin"
            write_fixture_map(source, 0, 2, sdc_index=1, moving=True)
            # The replay partner accelerates; the policy-controlled SDC starts still.
            data = bytearray(source.read_bytes())
            arrays = 32 + 16  # Map header without prediction targets, then first entity header.
            struct.pack_into("91f", data, arrays, *(0.1 * t + 0.05 * t * t for t in range(91)))
            struct.pack_into("91f", data, arrays + 3 * 91 * 4, *(1. + t for t in range(91)))
            if missing_frame:
                struct.pack_into("i", data, arrays + 7 * 91 * 4 + 2 * 4, 0)
            source.write_bytes(data)
            offsets, maps, n = self.binding.shared(
                map_dir=directory, num_agents=1024, num_maps=1, init_mode=0, control_mode=3,
                init_steps=0, goal_behavior=0, goal_target_distance=30., max_controlled_agents=32,
                map_ids=[0],
            )
            self.assertEqual(offsets, [0, 1])
            obs_size = self.binding.EGO_FEATURES_CLASSIC + (self.binding.MAX_AGENTS - 1) * 7 + 128 * 7
            obs = np.zeros((1, obs_size), dtype=np.float32)
            actions = np.full((1, 1), 45, dtype=np.int32)  # No acceleration or steering.
            rewards = np.zeros(1, dtype=np.float32)
            terminals = np.zeros(1, dtype=bool)
            truncations = np.zeros(1, dtype=bool)
            kwargs = evaluation.load_settings(ROOT / "pufferlib/config/ocean/drive.ini")["env"]
            kwargs.update(map_dir=directory, map_id=0, max_agents=1, human_agent_idx=0,
                          ini_file=str(ROOT / "pufferlib/config/ocean/drive.ini"),
                          action_type=0, dynamics_model=0, control_mode=3, init_mode=0,
                          episode_length=5 if missing_frame else 3, termination_mode=0)
            env = self.binding.env_init(obs, actions, rewards, terminals, truncations, 42, **kwargs)
            vec = self.binding.vectorize(env)
            try:
                self.binding.vec_reset(vec, 42)
                partner_x = self.binding.EGO_FEATURES_CLASSIC
                self.assertAlmostEqual(float(obs[0, partner_x]), -10. * 0.02, places=5)
                self.assertAlmostEqual(float(obs[0, partner_x + 6]), 1. / 100., places=5)
                self.binding.vec_step(vec)
                # Object 1 stays still; object 0 replays both x(t=1)=0.15 and vx(t=1)=2.
                self.assertAlmostEqual(float(obs[0, partner_x]), -9.85 * 0.02, places=5)
                self.assertAlmostEqual(float(obs[0, partner_x + 6]), 2. / 100., places=5)
                self.binding.vec_step(vec)
                if missing_frame:
                    np.testing.assert_array_equal(obs[0, partner_x:partner_x + 7], 0.)
                    self.binding.vec_step(vec)
                    self.assertAlmostEqual(float(obs[0, partner_x]), -9.25 * 0.02, places=5)
                    self.assertAlmostEqual(float(obs[0, partner_x + 6]), 4. / 100., places=5)
                    self.binding.vec_step(vec)
                self.assertAlmostEqual(float(obs[0, partner_x + 6]), 1. / 100., places=5)
                log = self.binding.env_log(env, 0)
                self.assertEqual(log["n"], 1)
                self.assertEqual(log["perc_controlled"], 0.5)
                self.assertEqual(log["perc_other"], 0.5)
            finally:
                self.binding.vec_close(vec)

    def test_stop_goal_is_counted_once(self):
        with tempfile.TemporaryDirectory() as directory:
            write_fixture_map(Path(directory) / "map_000.bin", 0, 2)
            env = NativeDriveHarness(self.binding, directory, init_steps=0,
                                     episode_length=4, goal_radius=1000.)
            try:
                env.reset()
                for _ in range(3):
                    env.step(env.actions)
                self.assertTrue(env.truncations.all())
                row = self.binding.env_log(env.handle, 2)
                self.assertEqual(row["goals_sampled_this_episode"], 1.)
                self.assertEqual(row["goals_reached_this_episode"], 1.)
                self.assertEqual(row["completion_rate"], 1.)
            finally:
                env.close()

    def test_return_sums_emitted_rewards_across_goal_completion(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            write_fixture_map(Path(directory) / "map_000.bin", 0, 2, goal_distance=0.24)
            for goal_behavior in (0, 2):  # Respawn and stop.
                with self.subTest(goal_behavior=goal_behavior):
                    env = NativeDriveHarness(self.binding, directory, init_steps=0,
                                             episode_length=5, goal_radius=0.01,
                                             goal_behavior=goal_behavior)
                    try:
                        env.reset()
                        # Accelerate straight ahead; both agents reach the goal on step 3.
                        actions = np.full_like(env.actions, 84)
                        rewards = [env.step(actions)[1].copy() for _ in range(4)]
                        self.assertTrue((rewards[0] < 0).all())
                        self.assertTrue((rewards[1] < 0).all())
                        np.testing.assert_array_equal(rewards[2], 1.)
                        self.assertTrue(env.truncations.all())
                        row = self.binding.env_log(env.handle, 2)
                        self.assertEqual(row["goals_reached_this_episode"], 1.)
                        self.assertAlmostEqual(row["episode_return"],
                                               np.asarray(rewards).sum(axis=0).mean(), places=6)
                    finally:
                        env.close()

    def test_wosac_retains_last_frame_and_computes_real_metrics(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory, benchmark_module() as benchmark:
            write_fixture_map(Path(directory) / "map_000.bin", 0, 2, moving=True, speed=1., tracks=(0,))
            for init_steps in (0, 10):
                env = NativeDriveHarness(self.binding, directory, init_steps=init_steps)
                config = {"train": {"device": "cpu", "use_rnn": False},
                          "eval": {"wosac_init_steps": init_steps, "wosac_num_rollouts": 2}}
                evaluator = benchmark.WOSACEvaluator(config)
                policy = SimpleNamespace(forward_eval=lambda obs, state: (None, None))
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        simulated = evaluator.collect_simulated_trajectories(config, env, policy)
                    self.assertEqual(env.steps, 2 * (90 - init_steps))
                    self.assertTrue(env.truncations.all())
                    expected = np.arange(init_steps, 91) * 0.1
                    for rollout in range(2):
                        np.testing.assert_allclose(simulated["x"][0, rollout], expected, atol=2e-5)
                    # A post-terminal step must not advance or log the same episode again.
                    before = env.get_global_agent_state()["x"].copy()
                    logged_n = self.binding.env_log(env.handle, 2)["n"]
                    env.step(env.actions)
                    self.assertTrue(env.truncations.all())
                    np.testing.assert_array_equal(env.get_global_agent_state()["x"], before)
                    self.assertEqual(self.binding.env_log(env.handle, 2)["n"], logged_n)

                    truth = env.get_ground_truth_trajectories()
                    edges = {"x": np.array([-200., 200.]), "y": np.array([-10., -10.]),
                             "lengths": np.array([2]), "scenario_id": np.array(["scene_0"])}
                    result = evaluator.compute_metrics(truth, simulated, env.get_global_agent_state(), edges,
                                                       drop_last_scenario=False, round_results=False)
                    self.assertEqual(list(result.index), ["scene_0"])
                    row = result.loc["scene_0"].to_dict()
                    evaluation.check_metrics(row, evaluation.WOSAC_METRICS)
                    self.assertLess(row["ade"], 2e-5)
                    self.assertLess(row["min_ade"], 2e-5)
                finally:
                    env.close()


@contextlib.contextmanager
def benchmark_module():
    """Load real metric code without importing optional Gym/rendering dependencies."""
    import torch

    modules = {}
    for name, path in (("pufferlib", ROOT / "pufferlib"),
                       ("pufferlib.ocean", ROOT / "pufferlib/ocean"),
                       ("pufferlib.ocean.benchmark", ROOT / "pufferlib/ocean/benchmark")):
        package = ModuleType(name)
        package.__path__ = [str(path)]
        modules[name] = package
    modules["pufferlib"].pytorch = SimpleNamespace(
        sample_logits=lambda logits: (torch.full((2, 1), 45, dtype=torch.int32), None, None))
    # Plotting is not part of numerical evaluation and is never called here.
    modules["matplotlib"] = ModuleType("matplotlib")
    modules["matplotlib.pyplot"] = ModuleType("matplotlib.pyplot")
    with patch.dict("sys.modules", modules):
        yield importlib.import_module("pufferlib.ocean.benchmark.evaluator")


class NativeDriveHarness:
    """Minimal adapter around the real C simulator for CPU trajectory tests."""

    def __init__(self, binding, directory, init_steps, episode_length=91, goal_radius=2.,
                 goal_behavior=2, num_agents=2, control_mode=2):
        import numpy as np

        self.binding = binding
        self.driver_env = self
        self.defer_reset = True
        self.init_steps = init_steps
        self.steps = 0
        self.num_agents = num_agents
        obs_size = binding.EGO_FEATURES_CLASSIC + (binding.MAX_AGENTS - 1) * 7 + 128 * 7
        self.obs = np.zeros((num_agents, obs_size), dtype=np.float32)
        self.actions = np.full((num_agents, 1), 45, dtype=np.int32)
        self.rewards = np.zeros(num_agents, dtype=np.float32)
        self.terminals = np.zeros(num_agents, dtype=bool)
        self.truncations = np.zeros(num_agents, dtype=bool)
        self.observation_space = SimpleNamespace(shape=self.obs.shape)
        self.action_space = SimpleNamespace(shape=self.actions.shape)
        kwargs = evaluation.load_settings(ROOT / "pufferlib/config/ocean/drive.ini")["env"]
        kwargs.update(map_dir=directory, map_id=0, max_agents=num_agents, human_agent_idx=0,
                      ini_file=str(ROOT / "pufferlib/config/ocean/drive.ini"),
                      action_type=0, dynamics_model=0, control_mode=control_mode, init_mode=0,
                      init_steps=init_steps, episode_length=episode_length, termination_mode=0,
                      goal_behavior=goal_behavior, goal_radius=goal_radius, defer_reset=True)
        self.handle = binding.env_init(self.obs, self.actions, self.rewards, self.terminals,
                                       self.truncations, 42, **kwargs)
        self.vec = binding.vectorize(self.handle)

    def reset(self):
        self.binding.vec_reset(self.vec, 42)
        self.binding.vec_log(self.vec, self.num_agents)
        return self.obs, []

    def step(self, actions):
        self.terminals[:] = False
        self.truncations[:] = False
        self.actions[:] = actions
        self.binding.vec_step(self.vec)
        self.steps += 1
        return self.obs, self.rewards, self.terminals, self.truncations, []

    def get_global_agent_state(self):
        import numpy as np

        state = {key: np.zeros(self.num_agents, dtype=np.int32 if key == "id" else np.float32)
                 for key in ("x", "y", "z", "heading", "id", "length", "width")}
        self.binding.vec_get_global_agent_state(self.vec, *state.values())
        return state

    def get_ground_truth_trajectories(self):
        import numpy as np

        n = self.num_agents
        truth = {key: np.zeros((n, 91 - self.init_steps), dtype=np.float32)
                 for key in ("x", "y", "z", "heading")}
        truth.update(valid=np.zeros((n, 91 - self.init_steps), dtype=np.int32),
                     id=np.zeros(n, dtype=np.int32), is_vehicle=np.zeros(n, dtype=bool),
                     is_track_to_predict=np.zeros(n, dtype=bool), scenario_id=np.zeros(n, dtype="S16"))
        self.binding.vec_get_global_ground_truth_trajectories(self.vec, *truth.values())
        truth["scenario_id"] = truth["scenario_id"].astype(str)
        return {key: value[:, None] for key, value in truth.items()}

    def close(self):
        self.binding.vec_close(self.vec)


if __name__ == "__main__":
    unittest.main()
