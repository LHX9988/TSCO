"""Meaningful invariance, replay, masking, training and checkpoint checks."""
import importlib.util
import tempfile
from pathlib import Path
import unittest
import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from tsco.learning import (DQNAgent, PrioritizedReplay, GraphQuantileNetwork,
                               observation_dimensions, _tensor_batch, quantile_huber_loss)
from tsco.schema import Observation, Transition


def observation(n=3, seed=3):
    rng = np.random.default_rng(seed)
    return Observation(rng.normal(size=(n, 5)).astype(np.float32),
                       rng.normal(size=(n, n, 2)).astype(np.float32),
                       np.ones((n, n), dtype=bool), rng.normal(size=4).astype(np.float32),
                       rng.normal(size=6).astype(np.float32), np.ones(1 + 3*n, dtype=bool))


def config(**kwargs):
    # Explicit unit-test fixture values exercise code paths. They are not
    # manuscript settings or a recommended training configuration.
    settings = dict(seed=5, replay_capacity=64, warmup_steps=0, batch_size=4,
                    target_update=2, n_step=3, dropout=0.0, torch_threads=1,
                    hardware_embedding_dim=7, value_hidden_dims=[11],
                    gat_negative_slope=0.1, per_priority_epsilon=1e-5,
                    adam_betas=[0.8, 0.98], adam_eps=1e-7, adam_weight_decay=0.0,
                    per_beta_steps=100, anneal_rule="linear")
    settings.update(kwargs)
    return {"drl": settings}


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is optional for core grid tests")
class LearningTests(unittest.TestCase):
    def test_graph_permutation_equivariance_and_dynamic_size(self):
        obs = observation()
        agent = DQNAgent(obs, config())
        values = agent.action_values(obs)
        permutation = np.array([2, 0, 1])
        mask = np.concatenate(([obs.mask[0]], obs.mask[1:].reshape(3, 3)[permutation].ravel()))
        reordered = Observation(obs.nodes[permutation], obs.edges[permutation][:, permutation],
                                obs.adjacency[permutation][:, permutation], obs.task,
                                obs.global_features, mask)
        expected = np.concatenate(([values[0]], values[1:].reshape(3, 3)[permutation].ravel()))
        np.testing.assert_allclose(agent.action_values(reordered), expected, rtol=2e-5, atol=2e-6)
        self.assertEqual(agent.action_values(observation(5)).shape, (16,))

    def test_feasibility_mask_and_flat_size_guard(self):
        obs = observation()
        obs.mask[1:] = False
        for variant in ("tsco", "no_quantile", "flat_dqn"):
            agent = DQNAgent(obs, config(), variant)
            self.assertEqual(agent.act(obs, explore=True), 0)
        with self.assertRaisesRegex(ValueError, "graph size"):
            agent.act(observation(5))

    def test_n_step_terminal_flush_and_observation_copy(self):
        obs = observation()
        agent = DQNAgent(obs, config(gamma=0.5))
        for i, reward in enumerate((1, 2, 3)):
            agent.observe(Transition(obs, 0, reward, obs, i == 2))
        self.assertEqual(len(agent.replay), 3)
        self.assertEqual(len(agent.pending), 0)
        self.assertEqual(agent.steps, 3)
        np.testing.assert_allclose([agent.replay.data[i].reward for i in range(3)], [2.75, 3.5, 3.0])
        np.testing.assert_allclose([agent.replay.data[i].discount for i in range(3)], [.125, .25, .5])
        self.assertTrue(all(agent.replay.data[i].done for i in range(3)))
        saved = agent.replay.data[0].observation.nodes.copy()
        obs.nodes[:] = 999
        np.testing.assert_array_equal(agent.replay.data[0].observation.nodes, saved)

    def test_quantile_loss_exact_small_example(self):
        predicted = torch.tensor([[0., 0.]])
        target = torch.tensor([[1., 1.]])
        loss, errors = quantile_huber_loss(predicted, target)
        self.assertAlmostEqual(float(loss[0]), .25)
        self.assertAlmostEqual(float(errors[0]), 1.0)

    def test_prioritized_replay_sampling(self):
        replay = PrioritizedReplay(4, alpha=1., seed=3, priority_epsilon=1e-5)
        obs = observation()
        for i in range(4):
            replay.add(Transition(obs, 0, float(i), obs, False))
        replay.update(np.arange(4), np.array([1., 1., 100., 1.]))
        samples, indices, weights = replay.sample(1000)
        self.assertGreater(np.mean(indices == 2), .9)
        self.assertEqual(len(samples), 1000)
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.all(weights > 0))

    def test_learning_target_update_and_checkpoint(self):
        obs = observation()
        for variant in ("tsco", "no_quantile", "flat_dqn"):
            agent = DQNAgent(obs, config(save_replay=True), variant)
            for i in range(10):
                agent.observe(Transition(obs, 0, float(i % 3), obs, i == 9))
            before = [p.detach().clone() for p in agent.online.parameters()]
            result1 = agent.learn()
            result2 = agent.learn()
            self.assertTrue(np.isfinite(result1["loss"]))
            self.assertEqual(result2["optimizer_updates"], 2)
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, agent.online.parameters())))
            for key, value in agent.online.state_dict().items():
                torch.testing.assert_close(value, agent.target.state_dict()[key])
            values = agent.action_values(obs)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "policy.pt"
                agent.save(path, {"split": "train"})
                restored = DQNAgent.load(path)
                np.testing.assert_allclose(restored.action_values(obs), values, rtol=0, atol=0)
                self.assertEqual(len(restored.replay), len(agent.replay))
                self.assertEqual(restored.steps, 10)
                self.assertEqual(restored.metadata["split"], "train")

    def test_mixed_graph_size_replay(self):
        agent = DQNAgent(observation(2), config(n_step=1))
        for i in range(8):
            obs = observation(2 if i % 2 else 3)
            agent.observe(Transition(obs, 0, 1., obs, True))
        self.assertTrue(np.isfinite(agent.learn()["loss"]))

    def test_optimizer_trains_noisy_sigma_and_resamples_noise(self):
        obs = observation()
        for variant in ("tsco", "no_quantile"):
            agent = DQNAgent(obs, config(n_step=1), variant)
            for i in range(8):
                agent.observe(Transition(obs, i % len(obs.mask),
                                         float(2 + i), obs, i == 7))
            sigma_before = {name: parameter.detach().clone()
                            for name, parameter in agent.online.named_parameters()
                            if "sigma" in name}
            noise_before = {name: value.detach().clone()
                            for name, value in agent.online.named_buffers()
                            if "epsilon" in name}
            self.assertTrue(np.isfinite(agent.learn()["loss"]))
            sigma_after = {name: parameter
                           for name, parameter in agent.online.named_parameters()
                           if "sigma" in name}
            self.assertTrue(sigma_after)
            self.assertTrue(any(parameter.grad is not None and
                                torch.count_nonzero(parameter.grad).item() > 0
                                for parameter in sigma_after.values()))
            self.assertTrue(any(not torch.equal(sigma_before[name], parameter)
                                for name, parameter in sigma_after.items()))
            self.assertTrue(any(not torch.equal(noise_before[name], value)
                                for name, value in agent.online.named_buffers()
                                if name in noise_before))
            self.assertTrue(all(parameter.grad is None
                                for parameter in agent.target.parameters()))

    def test_missing_unreported_architecture_and_optimizer_values_fail_early(self):
        obs = observation()
        for name in ("hardware_embedding_dim", "value_hidden_dims", "gat_negative_slope",
                     "per_priority_epsilon", "adam_betas", "adam_eps", "adam_weight_decay"):
            settings = config()
            settings["drl"][name] = None
            with self.assertRaisesRegex(ValueError, "drl." + name):
                DQNAgent(obs, settings)

    def test_unspecified_beta_schedule_fails_when_training_starts(self):
        for name in ("per_beta_steps", "anneal_rule"):
            settings = config(warmup_steps=20_000)
            settings["drl"][name] = None
            agent = DQNAgent(observation(), settings)
            with self.assertRaisesRegex(ValueError, "drl." + name):
                agent.learn()
        # Uniform replay in Flat-DQN does not require a PER annealing schedule.
        flat = DQNAgent(observation(), config(per_beta_steps=None, anneal_rule=None), "flat_dqn")
        self.assertIsNone(flat.learn())


if __name__ == "__main__":
    unittest.main()
