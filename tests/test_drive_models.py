"""CPU model/CLI tests without optional simulator, Gym, or rendering dependencies.

Run: python -m unittest discover -s tests -p test_drive_models.py -v
"""

import argparse
import ast
from collections import defaultdict
import configparser
import contextlib
import glob
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
# Optimizers import Dynamo lazily. Load it before module isolation so restoring
# sys.modules does not cause PyTorch's native operator registry to load twice.
import torch._dynamo


ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def policy_modules():
    # Keep actual PyTorch/model/initialization code; only stub unused environment
    # dependencies so these tests do not require a built simulator or map dataset.
    modules = {}
    for name, path in (("pufferlib", ROOT / "pufferlib"),
                       ("pufferlib.ocean", ROOT / "pufferlib/ocean")):
        package = ModuleType(name)
        package.__path__ = [str(path)]
        modules[name] = package
    modules["pufferlib"].__version__ = "test"
    modules["pufferlib"].APIUsageError = ValueError
    modules["pufferlib.emulation"] = ModuleType("pufferlib.emulation")
    spaces = ModuleType("pufferlib.spaces")
    spaces.Box = type("Box", (), {})
    modules["pufferlib.spaces"] = spaces
    modules["pufferlib"].spaces = spaces
    with patch.dict(sys.modules, modules):
        importlib.import_module("pufferlib.pytorch")
        policy = importlib.import_module("pufferlib.ocean.torch")
        selector = importlib.import_module("pufferlib.model_config")
        yield policy, selector


def training_functions(selector):
    # Execute the real CLI/loader functions while avoiding pufferl's unrelated
    # CUDA kernel, logger and rendering imports at module scope.
    path = ROOT / "pufferlib/pufferl.py"
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("load_config", "load_policy")]
    namespace = dict(globals(), __file__=str(path), pufferlib=sys.modules["pufferlib"],
                     RichHelpFormatter=argparse.HelpFormatter,
                     apply_model_type=selector.apply_model_type)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def evaluation_module():
    spec = importlib.util.spec_from_file_location("drive_evaluation", ROOT / "scripts/evaluate_test.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_env():
    return SimpleNamespace(
        single_observation_space=SimpleNamespace(shape=(1121,)),
        single_action_space=SimpleNamespace(nvec=np.array([91])),
        max_partner_objects=31, partner_features=7,
        max_road_objects=128, road_features=7, ego_features=8,
    )


def observations(batch=3):
    obs = torch.randn(batch, 1121)
    obs[:, 8 + 31 * 7:].reshape(batch, 128, 7)[:, :, -1] = 0
    return obs


class DriveModelTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)

    def config(self, training, *arguments):
        with patch.object(sys, "argv", ["puffer", "--train.device", "cpu", *arguments]):
            return training.load_config("puffer_drive")

    def test_cli_selection_and_legacy_defaults(self):
        with policy_modules() as (_, selector):
            training = training_functions(selector)
            for arguments, policy, recurrent in (
                ((), "Drive", True),
                (("--rnn-name", "None"), "Drive", False),
                (("--model-type", "mlp"), "DriveMLP", False),
                (("--model-type", "lstm", "--rnn-name", "None"), "Drive", True),
            ):
                with self.subTest(arguments=arguments):
                    config = self.config(training, *arguments)
                    self.assertEqual(config["policy_name"], policy)
                    self.assertEqual(config["rnn_name"], "Recurrent" if recurrent else None)
                    self.assertEqual(config["train"]["use_rnn"], recurrent)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.config(training, "--model-type", "invalid")

    def test_mlp_forward_backward_and_no_memory(self):
        with policy_modules() as (models, _):
            model = models.DriveMLP(fake_env())
            layers = [layer for layer in model.mlp if isinstance(layer, torch.nn.Linear)]
            self.assertEqual([(layer.in_features, layer.out_features) for layer in layers],
                             [(256, 256), (256, 256)])
            obs = observations()
            logits, values = model.forward_train(obs)
            self.assertEqual(logits[0].shape, (3, 91))
            self.assertEqual(values.shape, (3, 1))
            state = {"lstm_h": torch.randn(3, 256), "lstm_c": torch.randn(3, 256)}
            eval_logits, eval_values = model.forward_eval(obs, state)
            torch.testing.assert_close(eval_logits[0], logits[0])
            torch.testing.assert_close(eval_values, values)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            before = [layer.weight.detach().clone() for layer in layers]
            loss = torch.nn.functional.cross_entropy(logits[0], torch.tensor([0, 1, 2]))
            loss = loss + (values - 1).square().mean()
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            optimizer.step()
            for previous, layer in zip(before, layers):
                self.assertFalse(torch.equal(previous, layer.weight))

    def test_training_checkpoints_load_in_evaluation(self):
        with policy_modules() as (models, selector), tempfile.TemporaryDirectory() as directory:
            training = training_functions(selector)
            evaluation = evaluation_module()
            for kind in ("mlp", "lstm"):
                with self.subTest(kind=kind):
                    config = self.config(training, "--model-type", kind)
                    env = fake_env()
                    model = training.load_policy(config, SimpleNamespace(driver_env=env))
                    if kind == "lstm":
                        self.assertIsInstance(model, models.Recurrent)
                        self.assertIsInstance(model.policy, models.Drive)
                        self.assertFalse(hasattr(model.policy, "mlp"))
                    else:
                        self.assertIsInstance(model, models.DriveMLP)
                    checkpoint = Path(directory) / f"{kind}.pt"
                    torch.save(model.state_dict(), checkpoint)
                    eval_config = evaluation.load_settings(ROOT / "pufferlib/config/ocean/drive.ini")
                    selector.apply_model_type(eval_config, kind)
                    restored = evaluation.load_policy(eval_config, env, checkpoint, "cpu")
                    state_a = {"lstm_h": None, "lstm_c": None}
                    state_b = {"lstm_h": None, "lstm_c": None}
                    for _ in range(2):
                        obs = observations()
                        logits_a, value_a = model.forward_eval(obs, state_a)
                        logits_b, value_b = restored.forward_eval(obs, state_b)
                        torch.testing.assert_close(logits_a[0], logits_b[0])
                        torch.testing.assert_close(value_a, value_b)
                    wrong_config = evaluation.load_settings(ROOT / "pufferlib/config/ocean/drive.ini")
                    selector.apply_model_type(wrong_config, "lstm" if kind == "mlp" else "mlp")
                    with self.assertRaises(RuntimeError):
                        evaluation.load_policy(wrong_config, env, checkpoint, "cpu")

    def test_evaluation_selection_reaches_report_and_stages(self):
        with policy_modules(), tempfile.TemporaryDirectory() as directory:
            evaluation = evaluation_module()
            root = Path(directory)
            checkpoint = root / "model.pt"
            checkpoint.touch()
            for kind in ("mlp", "lstm"):
                seen = []

                def run_stage(options, config, ids, report, mode, *args):
                    seen.append(mode)
                    self.assertEqual(config["policy_name"], "DriveMLP" if kind == "mlp" else "Drive")
                    self.assertEqual(config["train"]["use_rnn"], kind == "lstm")
                    row = {"scenario_id": ids[0], "rollout": 0}
                    if mode == "wosac":
                        row.update({key: 0.5 for key in evaluation.WOSAC_METRICS})
                        row["num_agents_per_scene"] = 1
                    else:
                        row.update({key: 1 for key in evaluation.DRIVING_METRICS})
                        row["n"] = 1
                    if mode == "sanity":
                        row["map_name"] = options.sanity_maps[0]
                    report["per_scene"][mode].append(row)

                with patch.object(evaluation, "run_stage", side_effect=run_stage), \
                     patch.object(evaluation, "prepare_sanity_maps",
                                  return_value=(["scene_0"], [evaluation.SANITY_MAPS[0]])), \
                     patch.object(evaluation, "validate_stage_maps", return_value=(["scene_0"], [], [])), \
                     contextlib.redirect_stdout(io.StringIO()):
                    code = evaluation.main([
                        "--checkpoint", str(checkpoint), "--model-type", kind,
                        "--output-dir", str(root / kind), "--num-maps", "1",
                        "--sanity-rollouts", "1", "--sanity-maps", evaluation.SANITY_MAPS[0],
                    ])
                self.assertEqual(code, 0)
                self.assertEqual(seen, list(evaluation.STAGES))
                report = json.loads(next((root / kind).glob("*.json")).read_text())
                self.assertEqual(report["settings"]["model_type"], kind)
                self.assertEqual(report["settings"]["rnn_name"], "Recurrent" if kind == "lstm" else None)

    def test_evaluation_rejects_conflicting_selectors(self):
        evaluation = evaluation_module()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evaluation.parse_args(["--checkpoint", "model.pt", "--model-type", "mlp",
                                   "--rnn-name", "Recurrent"])


if __name__ == "__main__":
    unittest.main()
