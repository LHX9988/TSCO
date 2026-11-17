"""CS-GAQR-DQN and controlled architecture ablations.

Numerical values stated in the manuscript are used explicitly. Additional
network and optimizer values must be supplied in the configuration; the module
does not silently complete unreported settings.

The graph encoder is a dense, edge-aware GATv2 implementation: efficient for the
paper's 5--20 computing sites, with O(N**2) temporary attention storage. It does
not pretend to provide a sparse-O(|L|) implementation. Node/action parameters are
shared, so graph-policy inference supports different numbers of CPN sites.

Observation normalization belongs to the training-data pipeline. A checkpoint
stores architecture, optimizer and RNG state; replay is optional because the
nominal capacity (500,000 transitions) can be large. See ``save_replay``.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import math
import os

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:
    raise ImportError(
        "DRL commands require PyTorch. Install the project's learning extras "
        "or a CPU/CUDA PyTorch build appropriate for your machine."
    ) from exc

from .schema import Observation, Transition


def _setting(settings: dict, name: str, default: Any, *aliases: str) -> Any:
    for key in (name, *aliases):
        if key in settings:
            return settings[key]
    return default


def _required_setting(settings: dict, name: str) -> Any:
    if name not in settings or settings[name] is None:
        raise ValueError(
            f"drl.{name} is required but is not specified in the manuscript; "
            "provide the value used in your implementation."
        )
    return settings[name]


def _validate_unreported_settings(settings: dict, variant: str) -> dict:
    """Validate explicit values before allocating tensors or initializing RNGs."""
    names = ["adam_betas", "adam_eps", "adam_weight_decay"]
    if variant != "flat_dqn":
        names += ["hardware_embedding_dim", "value_hidden_dims",
                  "gat_negative_slope", "per_priority_epsilon"]
    values = {name: _required_setting(settings, name) for name in names}
    betas = values["adam_betas"]
    if not isinstance(betas, (list, tuple)) or len(betas) != 2 or not all(
            np.isfinite(b) and 0 <= b < 1 for b in betas):
        raise ValueError("drl.adam_betas must contain two finite values in [0,1)")
    for name, strict in (("adam_eps", True), ("adam_weight_decay", False)):
        value = values[name]
        if not np.isfinite(value) or (value <= 0 if strict else value < 0):
            raise ValueError(f"drl.{name} must be finite and {'positive' if strict else 'nonnegative'}")
    if variant != "flat_dqn":
        width = values["hardware_embedding_dim"]
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError("drl.hardware_embedding_dim must be a positive integer")
        widths = values["value_hidden_dims"]
        if not isinstance(widths, (list, tuple)) or not all(
                isinstance(w, int) and not isinstance(w, bool) and w > 0 for w in widths):
            raise ValueError("drl.value_hidden_dims must be an explicit list of positive widths; [] means a linear head")
        if not np.isfinite(values["gat_negative_slope"]) or values["gat_negative_slope"] < 0:
            raise ValueError("drl.gat_negative_slope must be finite and nonnegative")
        if not np.isfinite(values["per_priority_epsilon"]) or values["per_priority_epsilon"] <= 0:
            raise ValueError("drl.per_priority_epsilon must be finite and positive")
    return values


def copy_observation(obs: Observation) -> Observation:
    """Protect stored replay observations from mutable simulator arrays."""
    return Observation(
        np.array(obs.nodes, dtype=np.float32, copy=True),
        np.array(obs.edges, dtype=np.float32, copy=True),
        np.array(obs.adjacency, dtype=bool, copy=True),
        np.array(obs.task, dtype=np.float32, copy=True),
        np.array(obs.global_features, dtype=np.float32, copy=True),
        np.array(obs.mask, dtype=bool, copy=True),
    )


def observation_dimensions(obs: Observation) -> dict[str, int]:
    n = len(obs.nodes)
    if obs.nodes.ndim != 2 or n < 1:
        raise ValueError("Observation.nodes must have shape [N,F], N >= 1")
    if obs.edges.ndim != 3 or obs.edges.shape[:2] != (n, n):
        raise ValueError("Observation.edges must have shape [N,N,E]")
    if obs.adjacency.shape != (n, n):
        raise ValueError("Observation.adjacency must have shape [N,N]")
    if obs.task.ndim != 1 or obs.global_features.ndim != 1:
        raise ValueError("Task/global features must be vectors")
    if obs.mask.shape != (1 + 3 * n,) or not obs.mask[0]:
        raise ValueError("Action mask must have 1+3*N entries and retain defer")
    for values in (obs.nodes, obs.edges, obs.task, obs.global_features):
        if not np.all(np.isfinite(values)):
            raise ValueError("All policy features must be finite")
    return dict(nodes=n, node_dim=obs.nodes.shape[1], edge_dim=obs.edges.shape[2],
                task_dim=len(obs.task), global_dim=len(obs.global_features))


def _tensor_batch(observations: Iterable[Observation], device: torch.device) -> dict:
    obs = list(observations)
    if not obs:
        raise ValueError("Cannot form an empty observation batch")
    dimensions = [observation_dimensions(o) for o in obs]
    if any(d != dimensions[0] for d in dimensions):
        raise ValueError("A forward batch must have matching feature and graph sizes")
    arrays = {}
    for key, dtype in (("nodes", torch.float32), ("edges", torch.float32),
                       ("adjacency", torch.bool), ("task", torch.float32),
                       ("global_features", torch.float32), ("mask", torch.bool)):
        arrays[key] = torch.as_tensor(np.stack([getattr(o, key) for o in obs]),
                                     device=device, dtype=dtype)
    return arrays


class NoisyLinear(nn.Module):
    """Factorized Gaussian NoisyNet layer, with explicit inference control."""
    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.force_noise = False
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, sigma0 / math.sqrt(in_features))
        nn.init.constant_(self.bias_sigma, sigma0 / math.sqrt(out_features))
        self.reset_noise()

    @staticmethod
    def _scaled_noise(size: int, reference: torch.Tensor) -> torch.Tensor:
        x = torch.randn(size, device=reference.device, dtype=reference.dtype)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        eps_in = self._scaled_noise(self.in_features, self.weight_mu)
        eps_out = self._scaled_noise(self.out_features, self.weight_mu)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or self.force_noise:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(x, weight, bias)


class ResidualGATv2(nn.Module):
    """Dense GATv2; edge[i,j] describes a message from source j to target i."""
    def __init__(self, in_dim: int, edge_dim: int, heads: int = 4,
                 head_dim: int = 32, dropout: float = 0.1,
                 *, negative_slope: float):
        super().__init__()
        self.heads, self.head_dim, self.dropout = heads, head_dim, dropout
        self.negative_slope = negative_slope
        width = heads * head_dim
        self.left = nn.Linear(in_dim, width, bias=False)
        self.right = nn.Linear(in_dim, width, bias=False)
        self.edge = nn.Linear(edge_dim, width, bias=False)
        self.attention = nn.Parameter(torch.empty(heads, head_dim))
        self.residual = nn.Linear(in_dim, width, bias=False) if in_dim != width else nn.Identity()
        self.bias = nn.Parameter(torch.zeros(width))
        nn.init.xavier_uniform_(self.attention)

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor,
                adjacency: torch.Tensor) -> torch.Tensor:
        b, n, _ = nodes.shape
        x = F.dropout(nodes, self.dropout, training=self.training)
        left = self.left(x).reshape(b, n, self.heads, self.head_dim)
        right = self.right(x).reshape(b, n, self.heads, self.head_dim)
        edge = self.edge(edges).reshape(b, n, n, self.heads, self.head_dim)
        pair = F.leaky_relu(left[:, :, None] + right[:, None, :] + edge,
                            negative_slope=self.negative_slope)
        score = (pair * self.attention).sum(-1)  # [batch,destination,source,head]
        # Self loops ensure the normalization is defined for isolated sites.
        allowed = adjacency | torch.eye(n, device=nodes.device, dtype=torch.bool)[None]
        score = score.masked_fill(~allowed[..., None], -torch.inf)
        weights = torch.softmax(score, dim=2)
        weights = F.dropout(weights, self.dropout, training=self.training)
        aggregated = (weights[..., None] * right[:, None, :, :, :]).sum(dim=2)
        return F.elu(aggregated.reshape(b, n, -1) + self.residual(nodes) + self.bias)


class GraphQuantileNetwork(nn.Module):
    """Graph encoder and shared, hardware-conditioned dueling candidate head."""
    def __init__(self, dimensions: dict, n_quantiles: int = 32,
                 dropout: float = 0.1, sigma0: float = 0.5,
                 *, hardware_embedding_dim: int, value_hidden_dims: list[int],
                 gat_negative_slope: float):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.encoder1 = ResidualGATv2(dimensions["node_dim"], dimensions["edge_dim"],
                                     dropout=dropout, negative_slope=gat_negative_slope)
        self.encoder2 = ResidualGATv2(128, dimensions["edge_dim"], dropout=dropout,
                                     negative_slope=gat_negative_slope)
        self.task_encoder = nn.Sequential(nn.Linear(dimensions["task_dim"], 128),
                                         nn.ReLU(), nn.LayerNorm(128),
                                         nn.Linear(128, 128), nn.ReLU(), nn.LayerNorm(128))
        self.global_encoder = nn.Sequential(nn.Linear(dimensions["global_dim"], 128),
                                           nn.ReLU(), nn.LayerNorm(128),
                                           nn.Linear(128, 128), nn.ReLU(), nn.LayerNorm(128))
        self.hardware = nn.Embedding(4, hardware_embedding_dim)  # CPU/GPU/TPU + defer
        self.candidate = nn.Sequential(NoisyLinear(384 + hardware_embedding_dim, 256, sigma0), nn.ReLU(),
                                       NoisyLinear(256, 128, sigma0), nn.ReLU(),
                                       NoisyLinear(128, n_quantiles, sigma0))
        value_layers = []
        previous = 384  # pooled graph + 128-dimensional task/global encodings
        for width in value_hidden_dims:
            value_layers += [NoisyLinear(previous, width, sigma0), nn.ReLU()]
            previous = width
        value_layers.append(NoisyLinear(previous, n_quantiles, sigma0))
        self.value = nn.Sequential(*value_layers)

    def forward(self, batch: dict) -> torch.Tensor:
        encoded = self.encoder1(batch["nodes"], batch["edges"], batch["adjacency"])
        encoded = self.encoder2(encoded, batch["edges"], batch["adjacency"])
        b, n, _ = encoded.shape
        task = self.task_encoder(batch["task"])
        global_state = self.global_encoder(batch["global_features"])
        pooled = encoded.mean(dim=1)
        node_actions = encoded[:, :, None, :].expand(-1, -1, 3, -1).reshape(b, n * 3, 128)
        node_actions = torch.cat((pooled[:, None, :], node_actions), dim=1)
        hardware_ids = torch.cat((torch.tensor([3], device=encoded.device),
                                  torch.arange(3, device=encoded.device).repeat(n)))
        hardware = self.hardware(hardware_ids)[None].expand(b, -1, -1)
        count = 1 + 3 * n
        candidates = torch.cat((node_actions, task[:, None].expand(-1, count, -1),
                                global_state[:, None].expand(-1, count, -1), hardware), dim=-1)
        advantage = self.candidate(candidates)
        value = self.value(torch.cat((pooled, task, global_state), dim=-1))[:, None]
        mask = batch["mask"].to(advantage.dtype)[..., None]
        mean_advantage = (advantage * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
        return value + advantage - mean_advantage


class FlatDQNNetwork(nn.Module):
    """Fixed-size scalar DQN, intentionally without the graph/quantile machinery."""
    def __init__(self, dimensions: dict):
        super().__init__()
        self.n_nodes = dimensions["nodes"]
        n = self.n_nodes
        size = (n * dimensions["node_dim"] + n * n * dimensions["edge_dim"] +
                n * n + dimensions["task_dim"] + dimensions["global_dim"])
        self.network = nn.Sequential(nn.Linear(size, 256), nn.ReLU(),
                                     nn.Linear(256, 128), nn.ReLU(),
                                     nn.Linear(128, 1 + 3 * n))

    def forward(self, batch: dict) -> torch.Tensor:
        if batch["nodes"].shape[1] != self.n_nodes:
            raise ValueError("Flat-DQN is tied to the training graph size; use a graph policy for transfer")
        b = len(batch["nodes"])
        features = torch.cat((batch["nodes"].reshape(b, -1), batch["edges"].reshape(b, -1),
                              batch["adjacency"].float().reshape(b, -1), batch["task"],
                              batch["global_features"]), dim=1)
        return self.network(features)[..., None]


class PrioritizedReplay:
    """Sum-tree proportional replay, with O(log capacity) insertion/sampling."""
    def __init__(self, capacity: int, alpha: float = 0.6, seed: int = 0,
                 *, priority_epsilon: float | None = None):
        if capacity <= 0 or not 0 <= alpha <= 1:
            raise ValueError("Invalid replay capacity/alpha")
        self.capacity, self.alpha = int(capacity), float(alpha)
        if alpha > 0 and (priority_epsilon is None or not np.isfinite(priority_epsilon)
                          or priority_epsilon <= 0):
            raise ValueError("A positive per_priority_epsilon is required for prioritized replay")
        self.priority_epsilon = priority_epsilon
        self.leaf_count = 1 << (self.capacity - 1).bit_length()
        self.tree = np.zeros(2 * self.leaf_count, dtype=np.float64)
        self.data: list[Transition | None] = [None] * self.capacity
        self.position, self.size, self.maximum_priority = 0, 0, 1.0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def _set(self, index: int, value: float) -> None:
        node = self.leaf_count + index
        difference = value - self.tree[node]
        while node:
            self.tree[node] += difference
            node //= 2

    def add(self, transition: Transition) -> None:
        self.data[self.position] = transition
        self._set(self.position, self.maximum_priority ** self.alpha)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for index, priority in zip(indices, priorities):
            if not np.isfinite(priority) or priority < 0:
                raise ValueError("Replay priority must be finite and nonnegative")
            priority = max(float(priority), self.priority_epsilon or 0.0)
            self.maximum_priority = max(self.maximum_priority, priority)
            self._set(int(index), priority ** self.alpha)

    def sample(self, batch_size: int, beta: float = 0.4):
        if self.size == 0:
            raise ValueError("Cannot sample empty replay")
        if self.alpha == 0:
            indices = self.rng.integers(self.size, size=batch_size)
            weights = np.ones(batch_size, dtype=np.float32)
        else:
            total = self.tree[1]
            masses = (np.arange(batch_size) + self.rng.random(batch_size)) * total / batch_size
            indices = []
            for mass in masses:
                node = 1
                while node < self.leaf_count:
                    left = 2 * node
                    if mass < self.tree[left]:
                        node = left
                    else:
                        mass -= self.tree[left]
                        node = left + 1
                indices.append(node - self.leaf_count)
            indices = np.asarray(indices, dtype=int)
            probabilities = self.tree[self.leaf_count + indices] / total
            weights = (self.size * probabilities) ** (-beta)
            weights = (weights / weights.max()).astype(np.float32)
        return [self.data[i] for i in indices], np.asarray(indices), weights

    def state_dict(self) -> dict:
        return dict(capacity=self.capacity, alpha=self.alpha, tree=self.tree,
                    data=self.data, position=self.position, size=self.size,
                    maximum_priority=self.maximum_priority, rng=self.rng.bit_generator.state,
                    priority_epsilon=self.priority_epsilon)

    def load_state_dict(self, state: dict) -> None:
        if (state["capacity"] != self.capacity or state["alpha"] != self.alpha
                or state["priority_epsilon"] != self.priority_epsilon):
            raise ValueError("Replay checkpoint configuration differs")
        for name in ("tree", "data", "position", "size", "maximum_priority"):
            setattr(self, name, state[name])
        self.rng.bit_generator.state = state["rng"]


def quantile_huber_loss(predictions: torch.Tensor, targets: torch.Tensor,
                        kappa: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample mean quantile-Huber loss and mean absolute TD error."""
    if kappa <= 0:
        raise ValueError("Quantile Huber threshold must be positive")
    delta = targets[:, None, :] - predictions[:, :, None]
    absolute = delta.abs()
    huber = torch.where(absolute <= kappa, 0.5 * delta.square(),
                        kappa * (absolute - 0.5 * kappa))
    tau = (torch.arange(predictions.shape[1], device=predictions.device,
                        dtype=predictions.dtype) + 0.5) / predictions.shape[1]
    weight = (tau[None, :, None] - (delta.detach() < 0).to(delta.dtype)).abs()
    return (weight * huber / kappa).mean((1, 2)), absolute.detach().mean((1, 2))


class DQNAgent:
    """Training/inference façade used by the event-level CPN environment.

    ``observe`` counts real environment transitions. Trial closed-loop actions
    must not call it until the environment commits the corresponding transition.
    ``learn`` performs at most one optimizer update; callers control the ratio of
    updates to transitions. Terminal transitions flush all n-step prefixes.
    """
    def __init__(self, observation: Observation, config: dict, variant: str = "tsco",
                 device: str = "cpu"):
        aliases = {"w/o Q": "no_quantile", "w/o_q": "no_quantile", "wo_q": "no_quantile",
                   "Flat-DQN": "flat_dqn", "flat": "flat_dqn"}
        self.variant = aliases.get(variant, variant)
        if self.variant not in {"tsco", "no_quantile", "flat_dqn"}:
            raise ValueError(f"Unknown learning architecture: {variant}")
        self.config = deepcopy(config)
        settings = config.get("drl", {})
        unreported = _validate_unreported_settings(settings, self.variant)
        self.settings = settings
        self.device = torch.device(device)
        self.seed = int(settings.get("seed", config.get("seed", 0)))
        self.rng = np.random.default_rng(self.seed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        if settings.get("torch_threads") is not None:
            torch.set_num_threads(int(settings["torch_threads"]))
        torch.use_deterministic_algorithms(bool(settings.get("deterministic", True)))
        self.dimensions = observation_dimensions(observation)
        self.gamma = float(settings.get("gamma", 0.99))
        self.n_step = 1 if self.variant == "flat_dqn" else int(_setting(settings, "n_step", 3, "n_s"))
        self.n_quantiles = 1 if self.variant != "tsco" else int(_setting(settings, "n_quantiles", 32, "N_q"))
        self.batch_size = int(settings.get("batch_size", 256))
        self.warmup_steps = int(_setting(settings, "warmup_steps", 20_000, "warm_up", "warmup"))
        self.grad_clip = float(_setting(settings, "grad_clip", 10.0, "gradient_norm"))
        self.target_update = int(_setting(settings, "target_update", 5_000, "target_update_steps"))
        self.kappa = float(_setting(settings, "huber_kappa", 1.0, "kappa"))
        if self.n_step < 1 or not 0 < self.gamma <= 1 or self.target_update < 1:
            raise ValueError("Invalid n-step, discount or target update setting")
        if self.variant == "flat_dqn":
            self.online = FlatDQNNetwork(self.dimensions)
        else:
            self.online = GraphQuantileNetwork(
                self.dimensions, self.n_quantiles, float(settings.get("dropout", 0.1)),
                float(_setting(settings, "noisy_sigma", 0.5, "sigma0")),
                hardware_embedding_dim=unreported["hardware_embedding_dim"],
                value_hidden_dims=unreported["value_hidden_dims"],
                gat_negative_slope=unreported["gat_negative_slope"])
        self.online.to(self.device)
        self.target = deepcopy(self.online).to(self.device).eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=float(_setting(settings, "learning_rate", 1e-4, "lr")),
            betas=tuple(unreported["adam_betas"]), eps=unreported["adam_eps"],
            weight_decay=unreported["adam_weight_decay"])
        alpha = 0.0 if self.variant == "flat_dqn" else float(settings.get("per_alpha", 0.6))
        self.replay = PrioritizedReplay(int(settings.get("replay_capacity", 500_000)), alpha,
                                       self.seed, priority_epsilon=unreported.get("per_priority_epsilon"))
        self.pending: deque[Transition] = deque()
        self.steps, self.updates = 0, 0
        self.metadata: dict = {}

    @staticmethod
    def _noise(network: nn.Module, enabled: bool, reset: bool = False) -> None:
        """Set the eval-time noise override; training mode always uses noise.

        ``enabled=False`` clears ``force_noise`` and does not override the
        recursive ``network.train()`` flag in ``NoisyLinear.forward``.
        """
        for layer in network.modules():
            if isinstance(layer, NoisyLinear):
                layer.force_noise = enabled
                if reset:
                    layer.reset_noise()

    def _validate_dimensions(self, observation: Observation) -> None:
        dims = observation_dimensions(observation)
        for name in ("node_dim", "edge_dim", "task_dim", "global_dim"):
            if dims[name] != self.dimensions[name]:
                raise ValueError(f"Checkpoint {name} differs from observation")
        if self.variant == "flat_dqn" and dims["nodes"] != self.dimensions["nodes"]:
            raise ValueError("Flat-DQN cannot transfer to another CPN graph size")

    def action_values(self, observation: Observation, explore: bool = False) -> np.ndarray:
        """Raw expected values; masking is applied by ``act`` only."""
        self._validate_dimensions(observation)
        was_training = self.online.training
        self.online.eval()  # no dropout during rollout, including exploration
        self._noise(self.online, explore and self.variant != "flat_dqn", reset=explore)
        with torch.no_grad():
            values = self.online(_tensor_batch([observation], self.device)).mean(-1)[0]
        self._noise(self.online, False)
        self.online.train(was_training)
        return values.cpu().numpy()

    def act(self, observation: Observation, explore: bool = False,
            apply_mask: bool = True) -> int:
        self._validate_dimensions(observation)
        mask = np.asarray(observation.mask, dtype=bool) if apply_mask else np.ones_like(observation.mask, dtype=bool)
        if explore and self.variant == "flat_dqn":
            decay = max(1, int(self.settings.get("epsilon_decay_steps", 200_000)))
            final = float(self.settings.get("epsilon_final", 0.05))
            epsilon = 1.0 - (1.0 - final) * min(1.0, self.steps / decay)
            if self.rng.random() < epsilon:
                return int(self.rng.choice(np.flatnonzero(mask)))
        values = self.action_values(observation, explore=explore)
        return int(np.argmax(np.where(mask, values, -np.inf)))

    def _emit_prefix(self) -> None:
        reward, discount, final = 0.0, 1.0, None
        for transition in list(self.pending)[:self.n_step]:
            reward += discount * transition.reward
            discount *= self.gamma * float(transition.discount)
            final = transition
            if transition.done:
                break
        first = self.pending[0]
        assert final is not None
        self.replay.add(Transition(first.observation, first.action, float(reward),
                                   final.next_observation, final.done, float(discount)))
        self.pending.popleft()

    def observe(self, transition: Transition) -> None:
        self._validate_dimensions(transition.observation)
        self._validate_dimensions(transition.next_observation)
        if not 0 <= transition.action < len(transition.observation.mask):
            raise ValueError("Transition action index is out of range")
        if not np.isfinite(transition.reward) or not 0 <= transition.discount <= 1:
            raise ValueError("Invalid transition reward/discount")
        self.steps += 1
        stored = Transition(copy_observation(transition.observation), int(transition.action),
                            float(transition.reward), copy_observation(transition.next_observation),
                            bool(transition.done), float(transition.discount))
        self.pending.append(stored)
        if len(self.pending) >= self.n_step:
            self._emit_prefix()
        if transition.done:
            while self.pending:
                self._emit_prefix()

    def learn(self) -> dict | None:
        # The manuscript specifies beta's endpoints but neither annealing rule
        # nor duration. Validate them when training starts, even during warm-up.
        if self.replay.alpha > 0:
            rule = _required_setting(self.settings, "anneal_rule")
            beta_steps = _required_setting(self.settings, "per_beta_steps")
            if rule != "linear":
                raise ValueError("drl.anneal_rule must explicitly be 'linear' for the implemented schedule")
            if not isinstance(beta_steps, int) or isinstance(beta_steps, bool) or beta_steps <= 0:
                raise ValueError("drl.per_beta_steps must be a positive integer number of environment transitions")
        if len(self.replay) < self.batch_size or self.steps < self.warmup_steps:
            return None
        beta_start = float(self.settings.get("per_beta_start", 0.4))
        beta_end = float(self.settings.get("per_beta_end", 1.0))
        beta = (min(beta_end, beta_start + (beta_end - beta_start) * self.steps / beta_steps)
                if self.replay.alpha > 0 else 1.0)
        samples, indices, weights = self.replay.sample(self.batch_size, beta)
        self.optimizer.zero_grad(set_to_none=True)
        priorities = np.empty(self.batch_size, dtype=float)
        # Group graph sizes; shared encoders allow even mixed-size replay batches.
        groups: dict[tuple[int, int], list[int]] = {}
        for i, sample in enumerate(samples):
            key = (len(sample.observation.nodes), len(sample.next_observation.nodes))
            groups.setdefault(key, []).append(i)
        total_loss = 0.0
        for selected in groups.values():
            subset = [samples[i] for i in selected]
            current = _tensor_batch([s.observation for s in subset], self.device)
            following = _tensor_batch([s.next_observation for s in subset], self.device)
            actions = torch.tensor([s.action for s in subset], device=self.device, dtype=torch.long)
            rewards = torch.tensor([s.reward for s in subset], device=self.device, dtype=torch.float32)
            discounts = torch.tensor([s.discount * (not s.done) for s in subset],
                                     device=self.device, dtype=torch.float32)
            importance = torch.as_tensor(weights[selected], device=self.device)
            # Target construction excludes dropout and exploratory network noise.
            self.online.eval()
            self._noise(self.online, False)
            self.target.eval()
            self._noise(self.target, False)
            with torch.no_grad():
                target_quantiles = self.target(following)
                if self.variant == "flat_dqn":
                    next_values = target_quantiles.mean(-1)  # standard scalar DQN
                else:
                    next_values = self.online(following).mean(-1)  # double Q selection
                next_actions = next_values.masked_fill(~following["mask"], -torch.inf).argmax(1)
                future = target_quantiles[torch.arange(len(subset), device=self.device), next_actions]
                targets = rewards[:, None] + discounts[:, None] * future
            self.online.train()
            # train() activates NoisyLinear; False only clears its eval-time
            # override. Resampling here gives fresh noise for the TD update.
            self._noise(self.online, False, reset=True)
            predicted = self.online(current)[torch.arange(len(subset), device=self.device), actions]
            if self.variant == "tsco":
                losses, errors = quantile_huber_loss(predicted, targets, self.kappa)
            else:
                losses = F.smooth_l1_loss(predicted[:, 0], targets[:, 0], reduction="none", beta=self.kappa)
                errors = (targets[:, 0] - predicted[:, 0]).abs().detach()
            loss = (losses * importance).sum() / self.batch_size
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite learning loss; inspect input normalization/rewards")
            loss.backward()
            total_loss += float(loss.detach())
            priorities[selected] = errors.cpu().numpy() + (self.replay.priority_epsilon or 0.0)
        grad_norm = float(nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip))
        self.optimizer.step()
        self.updates += 1
        if self.replay.alpha > 0:
            self.replay.update(indices, priorities)
        if self.updates % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
            self.target.eval()
        return dict(loss=total_loss, grad_norm=grad_norm, beta=beta,
                    replay_size=len(self.replay), optimizer_updates=self.updates,
                    environment_transitions=self.steps)

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        replay_enabled = bool(self.settings.get("save_replay", False))
        payload = dict(format_version=1, config=self.config, variant=self.variant,
                       dimensions=self.dimensions, online=self.online.state_dict(),
                       target=self.target.state_dict(), optimizer=self.optimizer.state_dict(),
                       steps=self.steps, updates=self.updates,
                       numpy_rng=self.rng.bit_generator.state,
                       replay_rng=self.replay.rng.bit_generator.state,
                       torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                       metadata=deepcopy(metadata if metadata is not None else self.metadata),
                       replay=self.replay.state_dict() if replay_enabled else None,
                       pending=list(self.pending) if replay_enabled else [],
                       exact_resume=replay_enabled)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "DQNAgent":
        """Load a trusted project checkpoint. Standard checkpoints omit replay."""
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported checkpoint format")
        d = payload["dimensions"]
        n = d["nodes"]
        template = Observation(np.zeros((n, d["node_dim"]), np.float32),
                               np.zeros((n, n, d["edge_dim"]), np.float32),
                               np.eye(n, dtype=bool), np.zeros(d["task_dim"], np.float32),
                               np.zeros(d["global_dim"], np.float32), np.ones(1 + 3*n, dtype=bool))
        agent = cls(template, payload["config"], payload["variant"], device)
        agent.online.load_state_dict(payload["online"])
        agent.target.load_state_dict(payload["target"])
        agent.target.eval()
        agent.optimizer.load_state_dict(payload["optimizer"])
        agent.steps, agent.updates = payload["steps"], payload["updates"]
        agent.rng.bit_generator.state = payload["numpy_rng"]
        agent.replay.rng.bit_generator.state = payload["replay_rng"]
        torch.set_rng_state(payload["torch_rng"].cpu())
        if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        if payload.get("replay") is not None:
            agent.replay.load_state_dict(payload["replay"])
            agent.pending = deque(payload["pending"])
        agent.metadata = payload.get("metadata", {})
        agent.metadata["checkpoint_includes_replay"] = bool(payload.get("exact_resume"))
        return agent
