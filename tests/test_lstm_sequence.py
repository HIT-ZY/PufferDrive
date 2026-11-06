"""Sequence model, window reconstruction, PPO and inference integration tests.

Run: python -m unittest discover -s tests -p test_lstm_sequence.py -v
"""

import ast
from collections import defaultdict
import contextlib
import copy
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from tests.test_drive_models import ROOT, policy_modules, training_functions, evaluation_module, fake_env, observations


def load_trainer(sequence):
    """Run the real trainer with native advantages; omit optional UI imports."""
    libraries = list((ROOT / "pufferlib").glob("_C*.so"))
    if not libraries:
        raise unittest.SkipTest("Build the native advantage kernel first")
    torch.ops.load_library(str(libraries[0]))
    tree = ast.parse((ROOT / "pufferlib/pufferl.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in ("PuffeRL", "Profile", "compute_puff_advantage")]
    namespace = dict(globals(), **{name: getattr(sequence, name) for name in (
        "ObservationHistory", "make_history", "forward_with_history", "sequence_ppo_backward")},
        record=lambda method: method, ADVANTAGE_CUDA=False, pufferlib=sys.modules["pufferlib"],
        Utilization=lambda: None, NoLogger=lambda config: SimpleNamespace(run_id="test"),
        Evaluator=lambda *args: SimpleNamespace())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "pufferlib/pufferl.py", "exec"), namespace)
    return namespace["PuffeRL"]


class SequenceTests(unittest.TestCase):
    model_type = "lstm_seq"
    wrapper_name = "RecurrentSequence"
    temporal_weight_name = "lstm.weight_ih_l0"

    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)

    def config(self, selector, length=4, chunks=None):
        training = training_functions(selector)
        chunk_args = [] if chunks is None else ["--sequence-chunks", str(chunks)]
        with patch.object(sys, "argv", ["puffer", "--model-type", self.model_type, "--sequence-length", str(length),
                                       "--train.device", "cpu", *chunk_args]):
            return training.load_config("puffer_drive")

    def model(self, models, length=4, env=None, input_size=8, hidden_size=16):
        env = env or fake_env()
        return models.RecurrentSequence(env, models.Drive(env, input_size=input_size, hidden_size=hidden_size),
                                        input_size=hidden_size, hidden_size=hidden_size, sequence_length=length)

    def test_config_checkpoint_and_validation(self):
        with policy_modules() as (models, selector), tempfile.TemporaryDirectory() as directory:
            training = training_functions(selector)
            evaluation = evaluation_module()
            config = self.config(selector, 8)
            self.assertFalse(config["train"]["use_rnn"])
            self.assertTrue(config["train"]["use_sequence"])
            self.assertEqual(config["rnn_name"], self.wrapper_name)
            self.assertEqual(config["train"]["sequence_length"], 8)
            self.assertEqual(config["train"]["sequence_chunks"], 0)
            self.assertEqual(config["train"]["rollout_horizon"], 32)
            with patch.object(sys, "argv", ["puffer", "--model-type", self.model_type]):
                self.assertEqual(training.load_config("puffer_drive")["sequence_length"], 32)
            for bad in (0, -1, 1.5, True):
                with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "positive integer"):
                    self.config(selector, bad)
            for bad in ("0", "-1", "1.5"):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    evaluation.parse_args(["--checkpoint", "test.pt", "--sequence-length", bad])
            model = training.load_policy(config, SimpleNamespace(driver_env=fake_env()))
            path = Path(directory) / "model.pt"
            torch.save(model.state_dict(), path)
            restored = evaluation.load_policy(config, fake_env(), path, "cpu")
            inputs = observations(16).reshape(2, 8, -1)
            with torch.no_grad():
                a, va = model(inputs)
                b, vb = restored.forward_eval(inputs)
            torch.testing.assert_close(a[0], b[0])
            torch.testing.assert_close(va, vb)

    def test_chunk_config_cli_ini_and_validation(self):
        with policy_modules() as (_, selector), tempfile.TemporaryDirectory() as directory:
            training = training_functions(selector)
            evaluation = evaluation_module()
            for length in (1, 16, 32):
                for chunks in (0, 1, 8):
                    config = self.config(selector, length, chunks)
                    self.assertEqual(config["sequence_chunks"], chunks)
                    self.assertEqual(config["train"]["sequence_chunks"], chunks)
                    self.assertEqual(config["train"]["sequence_length"], length)
                    self.assertEqual(config["train"]["minibatch_size"], 32768)
                    self.assertEqual(config["train"]["max_minibatch_size"], 32768)
                    args = evaluation.parse_args(["--checkpoint", "test.pt", "--sequence-chunks", str(chunks)])
                    self.assertEqual(args.sequence_chunks, chunks)
            for bad in (-1, 1.5, True, "auto"):
                with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "non-negative integer"):
                    self.config(selector, chunks=bad)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    evaluation.parse_args(["--checkpoint", "test.pt", "--sequence-chunks", str(bad)])
            # Same INI value is accepted by both loaders; CLI can restore auto with 0.
            root = Path(directory)
            (root / "config/ocean").mkdir(parents=True)
            (root / "config/default.ini").write_text((ROOT / "pufferlib/config/default.ini").read_text())
            ini = root / "config/ocean/drive.ini"
            source = (ROOT / "pufferlib/config/ocean/drive.ini").read_text()
            ini.write_text(source.replace("sequence_chunks = 0", "sequence_chunks = 5"))
            for arguments, expected in (([], 5), (["--sequence-chunks", "0"], 0), (["--sequence-chunks", "3"], 3)):
                with patch.object(sys, "argv", ["puffer", "--model-type", self.model_type, *arguments]):
                    cfg = training.load_config("puffer_drive", config_dir=str(root))
                self.assertEqual(cfg["train"]["sequence_chunks"], expected)
            cfg = evaluation.load_settings(ini)
            selector.apply_model_type(cfg, self.model_type)
            self.assertEqual(cfg["train"]["sequence_chunks"], 5)

    def test_explicit_chunk_counts_are_balanced_and_independent_of_length(self):
        with policy_modules():
            seq = importlib.import_module("pufferlib.sequence")
            for length in (1, 16, 32):
                for count, chunks, sizes in ((6, 4, [2, 2, 1, 1]), (7, 3, [3, 2, 2]),
                                              (6, 1, [6]), (3, 100, [1, 1, 1])):
                    parts = list(seq.sequence_slices(count, length, 1, chunks))
                    self.assertEqual([p.stop - p.start for p in parts], sizes)
                    self.assertEqual([i for p in parts for i in range(p.start, p.stop)], list(range(count)))
            self.assertEqual([p.stop - p.start for p in seq.sequence_slices(6, 4, 20)], [5, 1])
            self.assertEqual([p.stop - p.start for p in seq.sequence_slices(3, 32, 1)], [1, 1, 1])

    def test_same_initialization_final_output_and_stateless_calls(self):
        with policy_modules() as (models, _):
            for length in (1, 4, 32):
                torch.manual_seed(77)
                env = fake_env()
                original = models.Recurrent(env, models.Drive(env, input_size=8, hidden_size=16), 16, 16)
                rng = torch.get_rng_state()
                torch.manual_seed(77)
                model = self.model(models, length)
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                self.assertEqual(sum(p.numel() for p in original.parameters()),
                                 sum(p.numel() for p in model.parameters()))
                for key, value in original.state_dict().items():
                    torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
                inputs = observations(2 * length).reshape(2, length, -1)
                state = {}
                logits, values = model(inputs, state)
                expected_logits, expected_values = original(inputs, {"lstm_h": None, "lstm_c": None})
                torch.testing.assert_close(logits[0], expected_logits[0].reshape(2, length, -1)[:, -1])
                torch.testing.assert_close(values[:, 0], expected_values[:, -1])
                model(observations(2 * length).reshape(2, length, -1), {})
                again, v_again = model.forward_eval(inputs, state)
                torch.testing.assert_close(logits[0], again[0])
                torch.testing.assert_close(values, v_again)
                self.assertEqual(state, {})
                with self.assertRaises(ValueError):
                    model(inputs[:, -1])
                with self.assertRaisesRegex(ValueError, "hidden state"):
                    model(inputs, {"lstm_h": None, "lstm_c": None})

    def test_history_reconstruction_matches_independent_windows(self):
        with policy_modules():
            seq = importlib.import_module("pufferlib.sequence")
            for length in (1, 4, 32):
                history = seq.ObservationHistory(2, length, (1,), device="cpu")
                horizon, segments = 3, 3
                stored = torch.zeros(segments * 2, horizon, 1)
                prefixes = torch.zeros(segments * 2, length - 1, 1)
                resets = torch.zeros(segments * 2, horizon, dtype=torch.bool)
                expected = torch.zeros(segments * 2, horizon, length, 1)
                oracle = [[], []]
                for step in range(horizon * segments):
                    obs = torch.tensor([[step], [100 + step]], dtype=torch.float32)
                    done = torch.tensor([step in (2, 5), step in (5, 7)])
                    # Two separate asynchronous agent batches must never share history.
                    for agent in (1, 0):
                        window = history.append(obs[agent:agent+1], done[agent:agent+1], slice(agent, agent+1))
                        if done[agent]:
                            oracle[agent] = []
                        oracle[agent].append(obs[agent].clone())
                        valid = oracle[agent][-length:]
                        reference = torch.stack([valid[0]] * (length - len(valid)) + valid)
                        torch.testing.assert_close(window[0], reference, atol=0, rtol=0)
                        row, t = (step // horizon) * 2 + agent, step % horizon
                        if t == 0:
                            prefixes[row] = window[0, :-1]
                        stored[row, t] = obs[agent]
                        resets[row, t] = done[agent]
                        expected[row, t] = reference
                order = torch.randperm(segments * 2 * horizon)
                rows, times = order // horizon, order % horizon
                actual = seq.reconstruct_windows(stored, prefixes, resets, rows, times)
                torch.testing.assert_close(actual, expected[rows, times], atol=0, rtol=0)
                history.reset()
                fresh = history.append(torch.tensor([[999.], [888.]]))
                torch.testing.assert_close(fresh[:, 0], fresh[:, -1], atol=0, rtol=0)

    def test_uneven_chunks_match_full_minibatch_loss_and_gradients(self):
        with policy_modules() as (models, _):
            seq = importlib.import_module("pufferlib.sequence")
            from pufferlib.pytorch import sample_logits
            model = self.model(models)
            reference = copy.deepcopy(model)
            obs = observations(6).reshape(2, 3, -1)
            prefixes = obs[:, :1].expand(-1, 3, -1).clone()
            resets = torch.tensor([[False, True, False], [False, False, False]])
            windows = seq.reconstruct_windows(obs, prefixes, resets, torch.arange(2).repeat_interleave(3),
                                              torch.arange(3).repeat(2))
            with torch.no_grad():
                logits, old_values = model(windows)
                actions, old_logprob, _ = sample_logits(logits)
            # Exercise both PPO and value clipping, not just ratio == 1.
            old_logprob += torch.tensor([0., .5, -.5, .1, -.2, .3])
            old_values = old_values.reshape(2, 3) + .4
            returns, advantages = torch.randn(2, 3), torch.randn(2, 3)
            importance = torch.tensor([[.7], [1.4]])
            config = dict(max_minibatch_size=20, clip_coef=.2, vf_clip_coef=.2, vf_coef=2.)
            logits, values = reference(windows)
            _, logprob, entropy = sample_logits(logits, action=actions)
            ratio = (logprob - old_logprob).exp().reshape(2, 3)
            adv = importance * (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            pg = torch.maximum(-adv * ratio, -adv * ratio.clamp(.8, 1.2)).mean()
            values = values.reshape(2, 3)
            clipped = old_values + (values - old_values).clamp(-.2, .2)
            vl = .5 * torch.maximum((values - returns).square(), (clipped - returns).square()).mean()
            (pg + 2 * vl - .005 * entropy.mean()).backward()
            for chunks, sizes in ((0, [5, 1]), (1, [6]), (4, [2, 2, 1, 1]), (99, [1] * 6)):
                with self.subTest(chunks=chunks):
                    model.zero_grad(set_to_none=True)
                    with patch.object(model, "forward", wraps=model.forward) as forward, \
                         patch.object(torch.autograd, "backward", wraps=torch.autograd.backward) as backward:
                        metrics, new_values, ratios = seq.sequence_ppo_backward(
                            model, obs, prefixes, resets, torch.arange(2), actions.reshape(2, 3, 1),
                            old_logprob.reshape(2, 3), old_values, returns, advantages, importance,
                            dict(config, sequence_chunks=chunks), .005, contextlib.nullcontext())
                    self.assertEqual(forward.call_count, len(sizes))
                    self.assertEqual(backward.call_count, len(sizes))
                    self.assertEqual([call.args[0].shape[0] for call in forward.call_args_list], sizes)
                    torch.testing.assert_close(torch.cat([call.args[0] for call in forward.call_args_list]),
                                               windows, atol=0, rtol=0)
                    self.assertAlmostEqual(metrics["policy_loss"], pg.item(), places=6)
                    self.assertAlmostEqual(metrics["value_loss"], vl.item(), places=6)
                    torch.testing.assert_close(new_values, values)
                    torch.testing.assert_close(ratios, ratio)
                    for (name, param), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
                        self.assertIsNotNone(param.grad, name)
                        self.assertGreater(param.grad.abs().sum().item(), 0, name)
                        torch.testing.assert_close(param.grad, expected.grad, atol=2e-6, rtol=2e-4, msg=name)

    def test_continuous_inference_chunks_preserve_order_and_outputs(self):
        with policy_modules() as (models, _):
            seq = importlib.import_module("pufferlib.sequence")
            env = fake_env()
            env.single_action_space = sys.modules["pufferlib.spaces"].Box()
            env.single_action_space.shape = (2,)
            model = self.model(models, env=env)
            obs = observations(3)
            with torch.no_grad():
                expected, target = model(obs[:, None].expand(-1, 4, -1))
                for chunks, sizes in ((0, [2, 1]), (1, [3]), (2, [2, 1]), (4, [1, 1, 1])):
                    history = seq.ObservationHistory(3, 4, (1121,), device="cpu")
                    with patch.object(model, "forward_eval", wraps=model.forward_eval) as forward:
                        actual, values = seq.forward_with_history(
                            model, obs, {}, history, max_minibatch_size=8, sequence_chunks=chunks)
                    self.assertEqual(forward.call_count, len(sizes))
                    self.assertEqual([call.args[0].shape for call in forward.call_args_list],
                                     [(size, 4, 1121) for size in sizes])
                    torch.testing.assert_close(actual.loc, expected.loc)
                    torch.testing.assert_close(actual.scale, expected.scale)
                    torch.testing.assert_close(values, target)

    def test_real_trainer_rollout_update_and_cross_batch_history(self):
        with policy_modules() as (models, selector), tempfile.TemporaryDirectory() as directory:
            seq = importlib.import_module("pufferlib.sequence")
            Trainer = load_trainer(seq)
            env = FakeVectorEnv()
            config = self.config(selector, chunks=2)["train"]
            config.update(batch_size=8, rollout_horizon=2, minibatch_size=8, max_minibatch_size=4,
                          total_timesteps=24, optimizer="adam", update_epochs=1, compile=False,
                          env="puffer_drive", data_dir=directory, checkpoint_interval=1,
                          eval=dict(eval_interval=100, human_replay_eval=False, self_play_eval=False,
                                    wosac_realism_eval=False))
            model = self.model(models)
            with patch.object(Trainer, "print_dashboard"), patch.object(Trainer, "mean_and_log", return_value={}):
                trainer = Trainer(config, env, model)
                trainer.evaluate()
                positions = torch.arange(8)
                windows = seq.reconstruct_windows(trainer.observations, trainer.sequence_prefixes,
                                                  trainer.sequence_resets, positions // 2, positions % 2)
                with torch.no_grad():
                    logits, values = model(windows)
                    from pufferlib.pytorch import sample_logits
                    _, logprob, _ = sample_logits(logits, action=trainer.actions.reshape(8, -1))
                torch.testing.assert_close((logprob.reshape(4, 2) - trainer.logprobs).exp(),
                                           torch.ones(4, 2), atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(values.reshape(4, 2), trainer.values)
                previous = trainer.observation_history.buffer.clone()
                before = {key: value.clone() for key, value in model.state_dict().items()}
                with patch.object(trainer.optimizer, "step", wraps=trainer.optimizer.step) as step, \
                     patch.object(model, "forward", wraps=model.forward) as forward:
                    trainer.train()
                self.assertEqual(forward.call_count, trainer.total_minibatches * 2)
                self.assertTrue(all(call.args[0].shape == (2, 4, 1121)
                                    for call in forward.call_args_list))
                self.assertEqual(step.call_count, 1)  # two logical minibatches, accumulated once
                self.assertFalse(torch.equal(before[self.temporal_weight_name],
                                             model.get_parameter(self.temporal_weight_name)))
                torch.testing.assert_close(trainer.observation_history.buffer, previous, atol=0, rtol=0)
                trainer.evaluate()
                # Step 4 starts a new rollout batch, not a new episode. Agent 1 has not reset yet.
                torch.testing.assert_close(trainer.sequence_prefixes[1], previous[1, 1:], atol=0, rtol=0)
                self.assertEqual(trainer.global_step, 16)
                self.assertFalse(hasattr(trainer, "lstm_h"))
                self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in trainer.losses.values()))
                state = torch.load(next(Path(directory).rglob("trainer_state.pt")), weights_only=True)
                self.assertEqual((state["model_type"], state["sequence_length"]), (self.model_type, 4))
                self.assertEqual(state["sequence_chunks"], 2)
                if self.model_type == "transformer":
                    self.assertEqual(state["transformer"], config["transformer"])

    def test_all_inference_entrypoints_use_the_same_windows(self):
        with policy_modules() as (models, selector):
            seq = importlib.import_module("pufferlib.sequence")
            config = self.config(selector, chunks=2)
            config["eval"].update(wosac_realism_eval=False, human_replay_eval=False)
            model = self.model(models)

            def load_function(path, name, cls=None):
                tree = ast.parse((ROOT / path).read_text())
                body = tree.body if cls is None else next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
                function = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
                namespace = dict(torch=torch, np=np, pufferlib=sys.modules["pufferlib"],
                                 make_history=seq.make_history, forward_with_history=seq.forward_with_history)
                exec(compile(ast.Module(body=[function], type_ignores=[]), path, "exec"), namespace)
                return namespace[name]

            rollout = load_function("pufferlib/ocean/benchmark/evaluator.py", "_run_rollout", "Evaluator")
            wosac = load_function("pufferlib/ocean/benchmark/evaluator.py", "collect_simulated_trajectories", "WOSACEvaluator")
            render = load_function("pufferlib/pufferl.py", "eval")
            evaluator = evaluation_module()
            driver_module = ModuleType("pufferlib.ocean.drive")
            driver_module.binding = SimpleNamespace(vec_log=Mock())
            expected = None
            for entrypoint in ("render", "self_play", "human_replay", "standalone", "wosac"):
                env = FakeInferenceEnv()
                captured = []
                original = model.forward_eval

                def capture(inputs, state):
                    self.assertNotIn("lstm_h", state)
                    self.assertNotIn("lstm_c", state)
                    captured.append(inputs.clone())
                    return original(inputs, state)

                with patch.object(model, "forward_eval", side_effect=capture), \
                     patch.dict(sys.modules, {"pufferlib.ocean.drive": driver_module}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    if entrypoint == "render":
                        render("puffer_drive", args=copy.deepcopy(config), vecenv=env, policy=model)
                    elif entrypoint in ("self_play", "human_replay"):
                        rollout(SimpleNamespace(configs=config, sim_steps=7), model, env, per_env_logs=True)
                    elif entrypoint == "standalone":
                        evaluator.rollout_driving(env, model, config, seed=42)
                    else:
                        wosac(SimpleNamespace(sim_steps=7, num_rollouts=2), config, env, model)
                self.assertEqual(len(captured), 24 if entrypoint == "wosac" else 12)
                self.assertTrue(all(part.shape == (1, 4, 1121) for part in captured))
                actual = torch.cat(captured).reshape(-1, 2, 4, 1121)
                if expected is None:
                    expected = actual
                torch.testing.assert_close(actual[:6], expected, atol=0, rtol=0)
                if entrypoint == "wosac":
                    torch.testing.assert_close(actual[6:], expected, atol=0, rtol=0)
            self.assertEqual(expected[2, 0, :, 0].tolist(), [2., 2., 2., 2.])
            self.assertEqual(expected[2, 1, :, 0].tolist(), [100., 100., 101., 102.])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_chunking_reduces_cuda_peak_training_memory(self):
        with policy_modules() as (models, _):
            seq = importlib.import_module("pufferlib.sequence")
            env = fake_env()
            model = self.model(models, length=8, env=env, input_size=64, hidden_size=256).cuda()
            obs = observations(64).reshape(8, 8, -1).cuda()
            prefixes = obs[:, :1].expand(-1, 7, -1).clone()
            resets = torch.zeros(8, 8, device="cuda", dtype=torch.bool)
            rows = torch.arange(8, device="cuda")
            actions = torch.zeros(8, 8, 1, device="cuda", dtype=torch.int32)
            old_logprobs = torch.full((8, 8), -4.5, device="cuda")
            old_values = torch.zeros(8, 8, device="cuda")
            returns, advantages = torch.randn(8, 8, device="cuda"), torch.randn(8, 8, device="cuda")
            importance = torch.ones(8, 1, device="cuda")
            config = dict(clip_coef=.2, vf_clip_coef=.2, vf_coef=2., max_minibatch_size=64 * 8)
            # Warm up forward/backward before comparing allocated memory.
            model(obs[:1])[1].sum().backward()
            peaks = []
            for chunks in (1, 16):
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                seq.sequence_ppo_backward(
                    model, obs, prefixes, resets, rows, actions, old_logprobs, old_values,
                    returns, advantages, importance, dict(config, sequence_chunks=chunks),
                    .005, contextlib.nullcontext())
                torch.cuda.synchronize()
                peaks.append(torch.cuda.max_memory_allocated())
            self.assertLess(peaks[1], peaks[0], f"Chunked/full CUDA memory: {peaks[1]}/{peaks[0]}")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_training_with_amp_and_cpu_offload(self):
        with policy_modules() as (models, selector), tempfile.TemporaryDirectory() as directory:
            seq = importlib.import_module("pufferlib.sequence")
            Trainer = load_trainer(seq)
            for offload, precision in ((False, "float32"), (True, "bfloat16")):
                if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
                    continue
                with self.subTest(offload=offload, precision=precision):
                    config = self.config(selector)["train"]
                    config.update(device="cuda", cpu_offload=offload, precision=precision,
                                  batch_size=8, rollout_horizon=2, minibatch_size=8, max_minibatch_size=4,
                                  total_timesteps=24, optimizer="adam", compile=False, env="puffer_drive",
                                  data_dir=directory, checkpoint_interval=100,
                                  eval=dict(eval_interval=100, human_replay_eval=False, self_play_eval=False,
                                            wosac_realism_eval=False))
                    model = self.model(models).cuda()
                    with patch.object(Trainer, "print_dashboard"), patch.object(Trainer, "mean_and_log", return_value={}):
                        trainer = Trainer(config, FakeVectorEnv(), model)
                        trainer.evaluate()
                        self.assertEqual(trainer.sequence_prefixes.device.type, "cpu" if offload else "cuda")
                        trainer.train()
                        trainer.evaluate()
                    self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in trainer.losses.values()))
                    self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                    # Use the configured chunk size when checking probabilities:
                    # different CUDA batch shapes can select different GEMM kernels.
                    from pufferlib.pytorch import sample_logits
                    with torch.no_grad(), trainer.amp_context:
                        positions = torch.arange(8, device=trainer.observations.device)
                        windows = seq.reconstruct_windows(trainer.observations, trainer.sequence_prefixes,
                                                          trainer.sequence_resets, positions // 2, positions % 2).cuda()
                        chunk_size = max(1, config["max_minibatch_size"] // config["sequence_length"])
                        parts = []
                        for inputs, actions in zip(windows.split(chunk_size),
                                                   trainer.actions.reshape(8, -1).split(chunk_size)):
                            logits, _ = model(inputs)
                            _, part, _ = sample_logits(logits, action=actions)
                            parts.append(part)
                        logprob = torch.cat(parts)
                    torch.testing.assert_close(logprob.reshape(4, 2).float(), trainer.logprobs,
                                               atol=1e-6, rtol=1e-6)

    def test_native_simulator_sequence_rollout(self):
        from tests.test_evaluate_test import NativeDriveHarness, write_fixture_map
        extensions = list((ROOT / "pufferlib/ocean/drive").glob("binding*.so"))
        if not extensions:
            self.skipTest("Build the native Drive simulator first")
        spec = importlib.util.spec_from_file_location("binding", extensions[0])
        binding = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(binding)
        with policy_modules() as (models, selector), tempfile.TemporaryDirectory() as directory:
            write_fixture_map(Path(directory) / "map_000.bin", 0, 2, moving=True, speed=1., tracks=(0,))
            env = NativeDriveHarness(binding, directory, init_steps=0, episode_length=7)
            try:
                seq = importlib.import_module("pufferlib.sequence")
                from pufferlib.pytorch import sample_logits
                model_env = fake_env()
                model_env.single_observation_space.shape = env.obs.shape[1:]
                model_env.ego_features = binding.EGO_FEATURES_CLASSIC
                model_env.max_partner_objects = binding.MAX_AGENTS - 1
                model = self.model(models, env=model_env)
                config = self.config(selector)
                obs, _ = env.reset()
                history = seq.make_history(config, obs, "cpu")
                done = None
                with torch.no_grad():
                    for _ in range(6):
                        logits, value = seq.forward_with_history(model, torch.as_tensor(obs), {}, history, done)
                        action, logprob, _ = sample_logits(logits)
                        self.assertTrue(torch.isfinite(value).all() and torch.isfinite(logprob).all())
                        obs, _, terminals, truncated, _ = env.step(action.numpy())
                        done = terminals | truncated
                self.assertTrue(env.truncations.all())
            finally:
                env.close()

    def test_wosac_subprocess_preserves_sequence_settings(self):
        with policy_modules() as (_, selector), tempfile.TemporaryDirectory() as directory:
            config = self.config(selector, 8, chunks=5)
            config["policy"]["hidden_size"] = 16
            config["train"].update(data_dir=directory, env="puffer_drive")
            utils = importlib.import_module("pufferlib.utils")
            with patch.object(utils.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="")) as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                utils.run_wosac_eval_in_subprocess(config["train"], SimpleNamespace(run_id="test"), 0, config)
            command = run.call_args.args[0]
            for flag, value in (("--model-type", self.model_type), ("--sequence-length", "8"),
                                ("--sequence-chunks", "5"),
                                ("--policy.hidden-size", "16"), ("--train.device", "cpu"),
                                ("--train.max-minibatch-size", str(config["train"]["max_minibatch_size"]))):
                self.assertEqual(command[command.index(flag) + 1], value)


class FakeVectorEnv:
    """Two vehicles, a mid-segment respawn and a later map/episode reset."""
    num_agents = 2
    agents_per_batch = 2
    single_observation_space = SimpleNamespace(shape=(1121,), dtype=np.dtype("float32"))
    single_action_space = SimpleNamespace(shape=(1,), dtype=np.dtype("int32"))
    action_space = SimpleNamespace(shape=(2, 1))

    def async_reset(self, seed):
        self.tick = 0

    def recv(self):
        obs = np.zeros((2, 1121), dtype=np.float32)
        obs[:, 0] = self.tick + np.array([0, 100])
        done = np.array([self.tick == 3, False], dtype=np.float32)
        truncated = np.full(2, self.tick == 5, dtype=np.float32)
        return obs, np.full(2, .1, dtype=np.float32), done, truncated, [], np.arange(2), np.ones(2)

    def send(self, action):
        self.tick += 1


class FakeInferenceEnv(FakeVectorEnv):
    episode_length = 7
    init_steps = 0
    num_envs = 1
    defer_reset = True
    render_mode = 1
    c_envs = None
    observation_space = SimpleNamespace(shape=(2, 1121))

    def __init__(self):
        self.driver_env = self

    def reset(self, seed=0):
        self.tick = 0
        return self.recv()[0], []

    def step(self, actions, per_env_logs=False):
        self.tick += 1
        obs, reward, _, _, *_ = self.recv()
        return obs, reward, np.array([self.tick == 2, False]), np.full(2, self.tick == 6), [{"n": 2}]

    def get_global_agent_state(self):
        return {key: np.zeros(2) for key in ("x", "y", "z", "heading", "id")}

    def render(self, **kwargs):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    unittest.main()
