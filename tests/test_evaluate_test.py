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
        self.assertEqual(options.num_maps, 150)
        self.assertEqual(options.wosac_rollouts, 32)
        self.assertEqual(options.replay_ego, "custom")

    def test_agent_weighted_metrics_and_goal_weighted_completion(self):
        rows = [driving_row(1, 1, 3, 2), driving_row(3, 0, 1, 0)]
        result = evaluation.aggregate_driving(rows)
        self.assertEqual(result["safe_completion_rate"], 0.25)
        self.assertAlmostEqual(result["completion_rate"], 2 / 6)

    def test_wosac_scene_weighting(self):
        rows = [{key: value for key in evaluation.WOSAC_METRICS} for value in (0.1, 0.9)]
        rows[0]["num_agents_per_scene"] = 1
        rows[1]["num_agents_per_scene"] = 20
        self.assertAlmostEqual(evaluation.aggregate_wosac(rows)["realism_meta_score"], 0.5)

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
            self.assertEqual(set(report["metrics"]), {"self_play", "custom_replay", "wosac"})
            self.assertIn("wosac", report["skipped"])
            self.assertTrue(all(value is None for group in report["metrics"].values() for value in group.values()))

    def test_nonfinite_metric_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            evaluation.check_metrics({"ade": float("nan")}, ["ade"])

    def test_three_passes_cover_every_map_and_keep_metrics_separate(self):
        self.check_passes("sdc")

    def test_custom_replay_runs_two_passes_without_wosac(self):
        self.check_passes("custom")

    def check_passes(self, replay_ego):
        options = SimpleNamespace(test_dir=Path("test"), checkpoint=Path("model.pt"), num_maps=3,
                                  batch_size=2, driving_rollouts=2, wosac_rollouts=4, seed=42, device="cpu",
                                  replay_ego=replay_ego)
        custom = replay_ego == "custom"
        replay_mode = "custom_replay" if custom else "human_replay"
        config = evaluation.load_settings(ROOT / "pufferlib/config/ocean/drive.ini")
        ids = ["scene_0", "scene_1", "scene_2"]
        instances = []

        class FakeDrive:
            def __init__(self, map_ids, **kwargs):
                self.map_ids = map_ids
                self.scenario_ids = [ids[i] for i in map_ids]
                self.control_mode = kwargs["control_mode"]
                self.closed = False
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
                return SimpleNamespace(to_dict=lambda: {"num_agents_per_scene": 2,
                    **{name: 0.5 for name in evaluation.WOSAC_METRICS}})

        class FakeWosac:
            def __init__(self, config):
                self_test.assertFalse(custom, "Custom replay must not instantiate WOSAC")

            def collect_ground_truth_trajectories(self, env):
                return env.scenario_ids

            def collect_simulated_trajectories(self, config, env, policy):
                return {}

            def compute_metrics(self, truth, *args, **kwargs):
                self_test.assertFalse(kwargs["drop_last_scenario"])
                self_test.assertFalse(kwargs["round_results"])
                return FakeResults(truth)

        self_test = self
        modules = {
            "torch": SimpleNamespace(manual_seed=lambda seed: None),
            "numpy": SimpleNamespace(random=SimpleNamespace(seed=lambda seed: None)),
            "pufferlib.ocean.drive.drive": SimpleNamespace(Drive=FakeDrive),
            "pufferlib.ocean.benchmark.evaluator": SimpleNamespace(WOSACEvaluator=FakeWosac),
        }
        report = {"metrics": {}, "per_scene": {mode: [] for mode in evaluation.metric_groups(options)},
                  "ego_selection": [{"ego_object_index": 1, "ego_object_id": 75} for _ in ids]}
        def fake_rollout(env, *args):
            n, success = (1, 0.25) if env.control_mode == "control_sdc_only" else (2, 1.0)
            return [driving_row(n=n, success=success) for _ in env.map_ids]

        with patch.dict("sys.modules", modules), patch.object(evaluation, "load_policy", return_value=object()), \
             patch.object(evaluation, "rollout_driving", side_effect=fake_rollout), \
             contextlib.redirect_stdout(io.StringIO()):
            evaluation.evaluate(options, config, ids, report)
        self.assertEqual([env.map_ids for env in instances], [[0, 1], [2]] * (2 if custom else 3))
        self.assertEqual([env.control_mode for env in instances],
                         ["control_agents"] * 2 + ["control_sdc_only"] * 2 + ([] if custom else ["control_wosac"] * 2))
        self.assertTrue(all(env.closed for env in instances))
        self.assertEqual(len(report["per_scene"]["self_play"]), 6)
        self.assertEqual(len(report["per_scene"][replay_mode]), 6)
        self.assertEqual(len(report["per_scene"]["wosac"]), 0 if custom else 3)
        self.assertEqual(report["metrics"]["self_play"]["safe_completion_rate"], 1.)
        self.assertEqual(report["metrics"][replay_mode]["safe_completion_rate"], 0.25)
        for mode, keys in evaluation.metric_groups(options).items():
            if custom and mode == "wosac":
                self.assertNotIn(mode, report["metrics"])
                continue
            self.assertEqual(set(report["metrics"][mode]), set(keys))

    def test_custom_selection_preserves_source_and_rejects_no_eligible_vehicle(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input"
            source.mkdir()
            path = source / "map_000.bin"
            write_fixture_map(path, 0, 2, sdc_index=35, tracks=(34, 6))
            before = path.read_bytes()
            destination = Path(directory) / "prepared"
            ids, selected = evaluation.prepare_custom_maps(source, destination, 1, 2.)
            self.assertEqual(ids, ["scene_0"])
            self.assertEqual(selected[0]["ego_object_index"], 0)
            self.assertEqual(selected[0]["original_sdc_track_index"], 35)
            self.assertEqual(path.read_bytes(), before)
            prepared = (destination / path.name).read_bytes()
            self.assertEqual(struct.unpack_from("2i", prepared, 16), (0, 0))
            self.assertEqual(prepared[24:], before[32:])  # Only metadata changed.
            with self.assertRaisesRegex(ValueError, "No valid vehicle"):
                evaluation.prepare_custom_maps(source, destination, 1, 1000.)

    def test_custom_run_records_selection_and_cleans_temporary_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            checkpoint.touch()
            write_fixture_map(root / "map_000.bin", 0, 2, sdc_index=35, tracks=(34,))
            temp_paths = []
            def fake_evaluate(options, config, ids, report):
                temp_paths.append(options.prepared_test_dir)
                evaluation.validate_maps(options.prepared_test_dir, 1)
                self.assertEqual(report["ego_selection"][0]["ego_object_index"], 0)
                self.assertIn("wosac", report["skipped"])
                for mode in ("self_play", "custom_replay"):
                    row = {"scenario_id": ids[0], **driving_row()}
                    report["per_scene"][mode].append(row)
                    report["metrics"][mode] = evaluation.aggregate_driving([row])
            with patch.object(evaluation, "evaluate", side_effect=fake_evaluate), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = evaluation.main(["--checkpoint", str(checkpoint), "--test-dir", str(root),
                                        "--output-dir", str(root / "out"), "--num-maps", "1"])
            self.assertEqual(code, 0)
            self.assertFalse(temp_paths[0].exists())
            report = json.loads(next((root / "out").glob("*.json")).read_text())
            self.assertEqual(report["coverage"], {"self_play": 1, "custom_replay": 1, "wosac": 0})
            self.assertTrue(all(v is None for v in report["metrics"]["wosac"].values()))


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

    def test_explicit_discard_for_custom_replay_does_not_mutate_json(self):
        scene = {"objects": [{"id": 35}], "metadata": {"sdc_track_index": 35,
                 "tracks_to_predict": [{"track_index": 34}]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map_000.bin"
            self.converter.save_map_binary(scene, path, 0, discard_track_metadata=True)
            self.assertEqual(struct.unpack("4i", path.read_bytes()[16:32]), (-1, 0, 1, 0))
            self.assertEqual(scene["metadata"]["sdc_track_index"], 35)


def write_fixture_map(path, map_id, count, sdc_index=None, moving=False, speed=0., tracks=()):
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
            handle.write(struct.pack("6fi", 2., 4., 1., 100., 0., 0., 0))
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
                    self.assertEqual(result["n"], 2)
                    self.assertEqual(result["completion_rate"], 1.)
                    self.assertEqual(result["safe_completion_rate"], 1.)
                    self.assertEqual(result["collision_rate"], 0.)
                    self.assertEqual(result["offroad_rate"], 0.)
            finally:
                self.binding.vec_close(vec)

    def test_human_replay_controls_marked_sdc_and_replays_other_vehicle(self):
        self.check_native_replay(custom=False)

    def test_custom_ego_controls_selected_vehicle_and_others_replay(self):
        self.check_native_replay(custom=True)

    def check_native_replay(self, custom):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "map_000.bin"
            write_fixture_map(source, 0, 2, sdc_index=35 if custom else 1, moving=True)
            if custom:
                with source.open("r+b") as handle:
                    handle.seek(32 + 16 + 8 * 91 * 4 + 24)
                    handle.write(struct.pack("i", 1))  # Prefer non-expert object 1 as ego.
                original = source.read_bytes()
                prepared = Path(directory) / "prepared"
                _, choices = evaluation.prepare_custom_maps(Path(directory), prepared, 1, 2.)
                self.assertEqual(choices[0]["ego_object_index"], 1)
                self.assertEqual(choices[0]["ego_object_id"], 1)
                self.assertEqual(source.read_bytes(), original)
                directory = str(prepared)
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
                          episode_length=3, termination_mode=0)
            env = self.binding.env_init(obs, actions, rewards, terminals, truncations, 42, **kwargs)
            vec = self.binding.vectorize(env)
            try:
                self.binding.vec_reset(vec, 42)
                partner_x = self.binding.EGO_FEATURES_CLASSIC
                self.assertAlmostEqual(float(obs[0, partner_x]), -10. * 0.02, places=5)
                self.binding.vec_step(vec)
                # Object 1 (SDC) stays still; object 0 follows its logged x(t=1)=0.1.
                self.assertAlmostEqual(float(obs[0, partner_x]), -9.9 * 0.02, places=5)
                self.binding.vec_step(vec)
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

    def __init__(self, binding, directory, init_steps, episode_length=91, goal_radius=2.):
        import numpy as np

        self.binding = binding
        self.driver_env = self
        self.defer_reset = True
        self.init_steps = init_steps
        self.steps = 0
        obs_size = binding.EGO_FEATURES_CLASSIC + (binding.MAX_AGENTS - 1) * 7 + 128 * 7
        self.obs = np.zeros((2, obs_size), dtype=np.float32)
        self.actions = np.full((2, 1), 45, dtype=np.int32)
        self.rewards = np.zeros(2, dtype=np.float32)
        self.terminals = np.zeros(2, dtype=bool)
        self.truncations = np.zeros(2, dtype=bool)
        self.observation_space = SimpleNamespace(shape=self.obs.shape)
        self.action_space = SimpleNamespace(shape=self.actions.shape)
        kwargs = evaluation.load_settings(ROOT / "pufferlib/config/ocean/drive.ini")["env"]
        kwargs.update(map_dir=directory, map_id=0, max_agents=2, human_agent_idx=0,
                      ini_file=str(ROOT / "pufferlib/config/ocean/drive.ini"),
                      action_type=0, dynamics_model=0, control_mode=2, init_mode=0,
                      init_steps=init_steps, episode_length=episode_length, termination_mode=0,
                      goal_behavior=2, goal_radius=goal_radius, defer_reset=True)
        self.handle = binding.env_init(self.obs, self.actions, self.rewards, self.terminals,
                                       self.truncations, 42, **kwargs)
        self.vec = binding.vectorize(self.handle)

    def reset(self):
        self.binding.vec_reset(self.vec, 42)
        self.binding.vec_log(self.vec, 2)
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

        state = {key: np.zeros(2, dtype=np.int32 if key == "id" else np.float32)
                 for key in ("x", "y", "z", "heading", "id", "length", "width")}
        self.binding.vec_get_global_agent_state(self.vec, *state.values())
        return state

    def get_ground_truth_trajectories(self):
        import numpy as np

        truth = {key: np.zeros((2, 91 - self.init_steps), dtype=np.float32)
                 for key in ("x", "y", "z", "heading")}
        truth.update(valid=np.zeros((2, 91 - self.init_steps), dtype=np.int32),
                     id=np.zeros(2, dtype=np.int32), is_vehicle=np.zeros(2, dtype=bool),
                     is_track_to_predict=np.zeros(2, dtype=bool), scenario_id=np.zeros(2, dtype="S16"))
        self.binding.vec_get_global_ground_truth_trajectories(self.vec, *truth.values())
        truth["scenario_id"] = truth["scenario_id"].astype(str)
        return {key: value[:, None] for key, value in truth.items()}

    def close(self):
        self.binding.vec_close(self.vec)


if __name__ == "__main__":
    unittest.main()
