"""Transformer architecture tests plus the shared window/PPO/inference contract.

Run: python -m unittest tests.test_transformer_sequence -v
"""

import contextlib
import copy
import importlib
import io
import math
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from tests import test_lstm_sequence as sequence_tests
from tests.test_drive_models import (
    policy_modules, training_functions, evaluation_module, fake_env, observations,
)


class TransformerSequenceTests(sequence_tests.SequenceTests):
    # Run the same window reconstruction, respawn/map-reset, cross-rollout,
    # continuous action, PPO gradient, trainer and evaluation integration tests
    # with a different temporal model. This also covers N > rollout_horizon.
    model_type = "transformer"
    wrapper_name = "TransformerSequence"
    temporal_weight_name = "transformer.layers.0.self_attn.in_proj_weight"

    def config(self, selector, length=4, chunks=None):
        config = super().config(selector, length, chunks)
        config["policy"].update(input_size=8, hidden_size=16)
        config["transformer"]["dim_feedforward"] = 32
        selector.apply_model_type(config)
        return config

    def model(self, models, length=4, env=None, input_size=8, hidden_size=16):
        env = env or fake_env()
        return models.TransformerSequence(
            env, models.Drive(env, input_size=input_size, hidden_size=hidden_size),
            sequence_length=length, dim_feedforward=2 * hidden_size)

    def test_same_initialization_final_output_and_stateless_calls(self):
        with policy_modules() as (models, _):
            for length in (1, 4, 32):
                with self.subTest(length=length):
                    torch.manual_seed(77)
                    env = fake_env()
                    reference = models.Recurrent(env, models.Drive(env, input_size=8, hidden_size=16), 16, 16)
                    torch.manual_seed(77)
                    model = self.model(models, length)
                    for name, value in reference.policy.state_dict().items():
                        torch.testing.assert_close(model.policy.state_dict()[name], value, atol=0, rtol=0)
                    inputs = observations(2 * length).reshape(2, length, -1)
                    state = {}
                    logits, values = model(inputs, state)
                    # Independently check the selected token and the head input.
                    encoded = model.policy.encode_observations(inputs.flatten(0, 1)).reshape(2, length, 16)
                    features = model.transformer(encoded + model.position_encoding)[:, -1].tanh()
                    self.assertTrue((features.abs() <= 1).all())
                    expected, target = model.policy.decode_actions(features)
                    torch.testing.assert_close(logits[0], expected[0])
                    torch.testing.assert_close(values, target)
                    model.eval()
                    with torch.inference_mode():
                        model(observations(2 * length).reshape(2, length, -1), state)
                        again, v_again = model.forward_eval(inputs, state)
                    torch.testing.assert_close(again[0], logits[0], atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(v_again, values, atol=1e-6, rtol=1e-5)
                    self.assertEqual(state, {})
                    self.assertFalse(hasattr(model, "lstm"))
                    self.assertFalse(any("cache" in key or "lstm" in key for key in model.state_dict()))
                    self.assertNotIn("position_encoding", model.state_dict())
                    for bad in (inputs[:, -1], inputs[:, :0], inputs[:, :, :-1]):
                        with self.assertRaisesRegex(ValueError, "Expected observations"):
                            model(bad)
                    for key in ("lstm_h", "lstm_c", "past_key_values", "kv_cache"):
                        with self.assertRaisesRegex(ValueError, "hidden state or KV cache"):
                            model(inputs, {key: None})

    def test_default_parameter_count_and_initialization(self):
        with policy_modules() as (models, selector):
            training = training_functions(selector)
            counts = {}
            policies = {}
            for kind in ("lstm", "transformer"):
                torch.manual_seed(42)
                with patch.object(sys, "argv", ["puffer", "--model-type", kind, "--train.device", "cpu"]):
                    config = training.load_config("puffer_drive")
                model = training.load_policy(config, SimpleNamespace(driver_env=fake_env()))
                counts[kind] = sum(p.numel() for p in model.parameters())
                policies[kind] = model.policy.state_dict()
            self.assertEqual(counts, dict(lstm=614236, transformer=615516))
            for name, value in policies["lstm"].items():
                torch.testing.assert_close(policies["transformer"][name], value, atol=0, rtol=0)
            self.assertEqual(config["transformer"], dict(num_layers=1, num_heads=4, dim_feedforward=512))
            self.assertEqual(config["sequence_length"], 32)
            self.assertEqual(config["sequence_chunks"], 0)
            for name, module in model.transformer.named_modules():
                if isinstance(module, torch.nn.LayerNorm):
                    torch.testing.assert_close(module.weight, torch.ones_like(module.weight))
                    torch.testing.assert_close(module.bias, torch.zeros_like(module.bias))
                if isinstance(module, torch.nn.Dropout):
                    self.assertEqual(module.p, 0)
            for name, parameter in model.transformer.named_parameters():
                if name.endswith("bias"):
                    torch.testing.assert_close(parameter, torch.zeros_like(parameter))
                if parameter.ndim == 2:
                    self.assertLessEqual(parameter.abs().max().item(), math.sqrt(6 / sum(parameter.shape)))
            block = model.transformer.layers[0]
            self.assertTrue(block.norm_first and block.self_attn.batch_first)
            self.assertEqual(block.self_attn.dropout, 0)
            self.assertEqual(block.activation, torch.nn.functional.gelu)

    def test_nondefault_dimensions_and_independent_layers(self):
        with policy_modules() as (models, selector), tempfile.TemporaryDirectory() as directory:
            training = training_functions(selector)
            with patch.object(sys, "argv", ["puffer", "--model-type", "transformer",
                    "--sequence-length", "3", "--train.device", "cpu", "--policy.input-size", "8",
                    "--policy.hidden-size", "15", "--transformer.num-layers", "2",
                    "--transformer.num-heads", "3", "--transformer.dim-feedforward", "21"]):
                config = training.load_config("puffer_drive")
            model = training.load_policy(config, SimpleNamespace(driver_env=fake_env()))
            self.assertEqual(model.position_encoding.shape, (1, 3, 15))
            first, second = model.transformer.layers
            self.assertNotEqual(first.self_attn.in_proj_weight.data_ptr(), second.self_attn.in_proj_weight.data_ptr())
            self.assertFalse(torch.equal(first.self_attn.in_proj_weight, second.self_attn.in_proj_weight))
            self.assertFalse(torch.equal(first.linear1.weight, second.linear1.weight))
            self.assertEqual(first.linear1.out_features, 21)
            # Verify sine/cosine by the formula, including the odd-width tail.
            for pos in range(3):
                for dim in range(15):
                    angle = pos / (10000 ** (2 * (dim // 2) / 15))
                    self.assertAlmostEqual(model.position_encoding[0, pos, dim].item(),
                                           math.sin(angle) if dim % 2 == 0 else math.cos(angle), places=6)
            path = Path(directory) / "model.pt"
            torch.save(model.state_dict(), path)
            restored = evaluation_module().load_policy(copy.deepcopy(config), fake_env(), path, "cpu")
            config["load_model_path"] = str(path)
            from_training_loader = training.load_policy(config, SimpleNamespace(driver_env=fake_env()))
            inputs = observations(6).reshape(2, 3, -1)
            with torch.no_grad():
                expected, target = model(inputs)
                for loaded in (restored, from_training_loader):
                    actual, value = loaded(inputs)
                    torch.testing.assert_close(actual[0], expected[0], atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(value, target, atol=1e-6, rtol=1e-5)
            wrong = copy.deepcopy(config)
            wrong["transformer"]["num_layers"] = 1
            with self.assertRaises(RuntimeError):
                evaluation_module().load_policy(wrong, fake_env(), path, "cpu")

    def test_invalid_dimensions_are_rejected_before_construction(self):
        with policy_modules() as (models, selector):
            training = training_functions(selector)
            evaluation = evaluation_module()
            for field in ("num_layers", "num_heads", "dim_feedforward", "hidden_size"):
                for bad in (0, -1, 1.5, True):
                    with self.subTest(field=field, bad=bad):
                        flag = "--policy.hidden-size" if field == "hidden_size" else f"--transformer.{field.replace('_', '-')}"
                        with patch.object(sys, "argv", ["puffer", "--model-type", "transformer", flag, str(bad)]), \
                             self.assertRaisesRegex(ValueError, "positive integer"):
                            training.load_config("puffer_drive")
                        if field != "hidden_size":
                            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                                evaluation.parse_args(["--checkpoint", "test.pt", flag, str(bad)])
                            with self.assertRaisesRegex(ValueError, "positive integer"):
                                models.TransformerSequence(fake_env(), models.Drive(fake_env()), **{field: bad})
            config = self.config(selector)
            config["transformer"]["num_heads"] = 3
            with self.assertRaisesRegex(ValueError, "divisible"):
                selector.apply_model_type(config)
            with self.assertRaisesRegex(ValueError, "divisible"):
                models.TransformerSequence(fake_env(), models.Drive(fake_env()), num_heads=3)

    def test_bidirectional_attention_and_gradient_through_whole_window(self):
        with policy_modules() as (models, _):
            model = self.model(models)
            encoded = torch.randn(2, 4, 16, requires_grad=True)
            model.transformer(encoded)[:, 0, 0].sum().backward()
            # First-token attention reaches the last token: no causal mask.
            self.assertGreater(encoded.grad[:, -1].abs().sum().item(), 0)
            inputs = observations(8).reshape(2, 4, -1).requires_grad_()
            logits, values = model(inputs)
            (logits[0].square().sum() + values.square().sum()).backward()
            for step in range(4):
                self.assertGreater(inputs.grad[:, step].abs().sum().item(), 0)
            for name, param in model.named_parameters():
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)
                self.assertGreater(param.grad.abs().sum().item(), 0, name)
            # Tanh bounds features only; a large value-head bias is not clipped.
            with torch.no_grad():
                model.policy.value_fn.bias.fill_(5.)
                model.policy.value_fn.weight.zero_()
            _, values = model(inputs)
            torch.testing.assert_close(values, torch.full_like(values, 5.))

    def test_wosac_subprocess_transmits_nondefault_transformer_structure(self):
        with policy_modules() as (_, selector), tempfile.TemporaryDirectory() as directory:
            config = self.config(selector, 8, chunks=5)
            config["transformer"].update(num_layers=2, num_heads=2, dim_feedforward=24)
            selector.apply_model_type(config)
            config["train"].update(data_dir=directory, env="puffer_drive")
            utils = importlib.import_module("pufferlib.utils")
            for full_args in (config, None):
                with patch.object(utils.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="")) as run, \
                     contextlib.redirect_stdout(io.StringIO()):
                    utils.run_wosac_eval_in_subprocess(config["train"], SimpleNamespace(run_id="test"), 0, full_args)
                command = run.call_args.args[0]
                with patch.object(sys, "argv", ["puffer", *command[5:]]):
                    child = training_functions(selector).load_config("puffer_drive")
                self.assertEqual(child["model_type"], "transformer")
                self.assertEqual(child["transformer"], config["transformer"])
                self.assertEqual(child["policy"]["hidden_size"], 16)

    def test_training_logs_record_architecture(self):
        with policy_modules() as (_, selector):
            seq = importlib.import_module("pufferlib.sequence")
            Trainer = sequence_tests.load_trainer(seq)
            config = self.config(selector, chunks=3)["train"]
            trainer = SimpleNamespace(config=config, stats={}, losses={}, profile=[], sps=0,
                global_step=8, epoch=1, start_time=time.time(), ent_coef_initial=.005, total_epochs=3,
                optimizer=SimpleNamespace(param_groups=[dict(lr=.001)]), logger=Mock())
            with patch.dict(Trainer.mean_and_log.__globals__, dist_sum=lambda value, device: value):
                logs = Trainer.mean_and_log(trainer)
            self.assertEqual(logs["model_type"], "transformer")
            self.assertEqual(logs["sequence_length"], 4)
            self.assertEqual(logs["sequence_chunks"], 3)
            for name, value in config["transformer"].items():
                self.assertEqual(logs[f"transformer/{name}"], value)


if __name__ == "__main__":
    unittest.main()
