# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""A Rainbow DQN agent using JAX."""

from typing import Any, Callable, NamedTuple, Sequence

from bsuite.baselines import base
from bsuite.baselines.utils import prioritized_replay

import dm_env
from dm_env import specs
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax


class TrainingState(NamedTuple):
  params: hk.Params
  target_params: hk.Params
  opt_state: Any
  rng_key: jnp.ndarray
  step: int


class NoisyLinear(hk.Module):
  """A linear layer with trainable Gaussian parameter noise."""

  def __init__(self, output_size: int, std_init: float = 0.5, name=None):
    super().__init__(name=name)
    self._output_size = output_size
    self._std_init = std_init

  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    inputs = jnp.asarray(inputs)
    input_size = inputs.shape[-1]
    mu_range = 1. / np.sqrt(input_size)

    weight_mu = hk.get_parameter(
        'weight_mu',
        shape=[input_size, self._output_size],
        init=hk.initializers.RandomUniform(-mu_range, mu_range))
    weight_sigma = hk.get_parameter(
        'weight_sigma',
        shape=[input_size, self._output_size],
        init=hk.initializers.Constant(self._std_init / np.sqrt(input_size)))
    bias_mu = hk.get_parameter(
        'bias_mu',
        shape=[self._output_size],
        init=hk.initializers.RandomUniform(-mu_range, mu_range))
    bias_sigma = hk.get_parameter(
        'bias_sigma',
        shape=[self._output_size],
        init=hk.initializers.Constant(self._std_init / np.sqrt(self._output_size)))

    key_w, key_b = jax.random.split(hk.next_rng_key())
    weight_eps = jax.random.normal(key_w, shape=weight_mu.shape)
    bias_eps = jax.random.normal(key_b, shape=bias_mu.shape)
    weight = weight_mu + weight_sigma * weight_eps
    bias = bias_mu + bias_sigma * bias_eps
    return jnp.matmul(inputs, weight) + bias


class Rainbow(base.Agent):
  """A bsuite-style Rainbow agent with noisy nets and prioritized replay."""

  def __init__(
      self,
      obs_spec: specs.Array,
      action_spec: specs.DiscreteArray,
      network: Callable[[jnp.ndarray], jnp.ndarray],
      optimizer: optax.GradientTransformation,
      batch_size: int,
      rng: hk.PRNGSequence,
      replay_capacity: int,
      min_replay_size: int,
      sgd_period: int,
      target_update_period: int,
      discount: float,
      n_step: int,
      priority_exponent: float,
      importance_sampling_exponent: float,
      max_importance_sampling_exponent: float,
      importance_sampling_exponent_anneal_steps: int,
      uniform_sample_probability: float,
      priority_epsilon: float,
      n_atoms: int,
      v_min: float,
      v_max: float,
  ):
    if importance_sampling_exponent_anneal_steps <= 0:
      raise ValueError('importance_sampling_exponent_anneal_steps must be positive.')

    network = hk.transform(network)
    atoms = jnp.asarray(np.linspace(v_min, v_max, n_atoms), dtype=jnp.float32)
    delta_z = (v_max - v_min) / (n_atoms - 1)

    def project_distribution(
        rewards: jnp.ndarray,
        discounts: jnp.ndarray,
        next_pmfs: jnp.ndarray,
    ) -> jnp.ndarray:
      next_atoms = rewards[:, None] + discounts[:, None] * atoms[None, :]
      clipped_atoms = jnp.clip(next_atoms, min=v_min, max=v_max)
      b = (clipped_atoms - v_min) / delta_z
      lower = jnp.clip(jnp.floor(b), min=0, max=n_atoms - 1)
      upper = jnp.clip(jnp.ceil(b), min=0, max=n_atoms - 1)
      lower_mass = (upper + (lower == upper).astype(jnp.float32) - b) * next_pmfs
      upper_mass = (b - lower) * next_pmfs
      batch_size = rewards.shape[0]
      target_pmfs = jnp.zeros((batch_size, n_atoms), dtype=jnp.float32)
      batch_indices = jnp.arange(batch_size)[:, None]
      target_pmfs = target_pmfs.at[
          batch_indices, lower.astype(jnp.int32)].add(lower_mass)
      target_pmfs = target_pmfs.at[
          batch_indices, upper.astype(jnp.int32)].add(upper_mass)
      return target_pmfs

    def loss(
        params: hk.Params,
        target_params: hk.Params,
        transitions: Sequence[jnp.ndarray],
        loss_rng_key: jnp.ndarray,
    ):
      o_tm1, a_tm1, r_t, discount_t, o_t, weights = transitions
      key_tm1, key_selector, key_target = jax.random.split(loss_rng_key, 3)

      pmfs_tm1 = network.apply(params, key_tm1, o_tm1)
      chosen_pmfs = pmfs_tm1[jnp.arange(pmfs_tm1.shape[0]), a_tm1]

      selector_pmfs = network.apply(params, key_selector, o_t)
      selector_values = jnp.sum(selector_pmfs * atoms[None, None, :], axis=-1)
      next_actions = jnp.argmax(selector_values, axis=-1)

      target_pmfs_all = network.apply(target_params, key_target, o_t)
      next_pmfs = target_pmfs_all[jnp.arange(target_pmfs_all.shape[0]), next_actions]
      projected_pmfs = project_distribution(r_t, discount_t, next_pmfs)

      log_chosen_pmfs = jnp.log(jnp.clip(chosen_pmfs, a_min=1e-5, a_max=1.))
      per_sample_loss = -jnp.sum(projected_pmfs * log_chosen_pmfs, axis=-1)
      mean_loss = jnp.mean(weights * per_sample_loss)
      q_values = jnp.sum(chosen_pmfs * atoms[None, :], axis=-1)
      return mean_loss, (per_sample_loss, q_values)

    @jax.jit
    def sgd_step(
        state: TrainingState,
        transitions: Sequence[jnp.ndarray],
    ):
      new_rng_key, loss_rng_key = jax.random.split(state.rng_key)
      (loss_value, aux), gradients = jax.value_and_grad(loss, has_aux=True)(
          state.params, state.target_params, transitions, loss_rng_key)
      priorities, q_values = aux
      updates, new_opt_state = optimizer.update(
          gradients, state.opt_state, state.params)
      new_params = optax.apply_updates(state.params, updates)
      new_step = state.step + 1
      new_target_params = jax.lax.cond(
          new_step % target_update_period == 0,
          lambda _: new_params,
          lambda _: state.target_params,
          operand=None)
      new_state = TrainingState(
          params=new_params,
          target_params=new_target_params,
          opt_state=new_opt_state,
          rng_key=new_rng_key,
          step=new_step)
      return new_state, loss_value, priorities, q_values

    dummy_observation = np.zeros((1, *obs_spec.shape), jnp.float32)
    initial_params = network.init(next(rng), dummy_observation)
    initial_opt_state = optimizer.init(initial_params)

    self._state = TrainingState(
        params=initial_params,
        target_params=initial_params,
        opt_state=initial_opt_state,
        rng_key=next(rng),
        step=0)
    self._forward = jax.jit(network.apply)
    self._sgd_step = sgd_step
    self._policy_rng = rng
    self._atoms = atoms
    self._num_actions = action_spec.num_values
    self._batch_size = batch_size
    self._sgd_period = sgd_period
    self._min_replay_size = min_replay_size
    self._importance_sampling_exponent = importance_sampling_exponent
    self._max_importance_sampling_exponent = max_importance_sampling_exponent
    self._importance_sampling_exponent_anneal_steps = (
        importance_sampling_exponent_anneal_steps)
    self._total_steps = 0
    self._replay = prioritized_replay.PrioritizedReplay(
        capacity=replay_capacity,
        priority_exponent=priority_exponent,
        importance_sampling_exponent=importance_sampling_exponent,
        uniform_sample_probability=uniform_sample_probability,
        priority_epsilon=priority_epsilon,
        discount=discount,
        n_step=n_step)
    self._update_importance_sampling_exponent()

  def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
    """Selects actions greedily under a noisy value distribution."""
    observation = timestep.observation[None, ...]
    pmfs = self._forward(self._state.params, next(self._policy_rng), observation)
    q_values = np.asarray(jnp.sum(pmfs * self._atoms[None, None, :], axis=-1))[0]
    action = np.random.choice(np.flatnonzero(q_values == q_values.max()))
    return int(action)

  def update(
      self,
      timestep: dm_env.TimeStep,
      action: base.Action,
      new_timestep: dm_env.TimeStep,
  ):
    """Adds a transition to replay and periodically runs SGD."""
    self._replay.add([
        timestep.observation,
        action,
        np.float32(new_timestep.reward),
        np.float32(new_timestep.discount),
        new_timestep.observation,
    ])

    self._total_steps += 1
    self._update_importance_sampling_exponent()
    if self._total_steps % self._sgd_period != 0:
      return
    if self._replay.size < self._min_replay_size:
      return

    batch = self._replay.sample(self._batch_size)
    transitions = [*batch.transitions, batch.weights]
    self._state, _, priorities, _ = self._sgd_step(self._state, transitions)
    self._replay.update_priorities(
        batch.indices, np.asarray(jax.device_get(priorities)))

  def _update_importance_sampling_exponent(self):
    fraction = min(
        self._total_steps / self._importance_sampling_exponent_anneal_steps, 1.)
    exponent = self._importance_sampling_exponent + fraction * (
        self._max_importance_sampling_exponent -
        self._importance_sampling_exponent)
    self._replay.set_importance_sampling_exponent(exponent)


def default_agent(
    obs_spec: specs.Array,
    action_spec: specs.DiscreteArray,
    seed: int = 0,
) -> Rainbow:
  """Initializes a Rainbow agent with bsuite-scale defaults."""

  hidden_sizes = [64, 64]
  noisy_hidden_size = 64
  n_atoms = 51
  replay_capacity = 10000

  def network(inputs: jnp.ndarray) -> jnp.ndarray:
    x = hk.Flatten()(inputs)
    x = hk.nets.MLP(hidden_sizes, activate_final=True)(x)

    value = NoisyLinear(noisy_hidden_size, name='value_hidden')(x)
    value = jax.nn.relu(value)
    value = NoisyLinear(n_atoms, name='value_output')(value)
    value = value[:, None, :]

    advantage = NoisyLinear(noisy_hidden_size, name='adv_hidden')(x)
    advantage = jax.nn.relu(advantage)
    advantage = NoisyLinear(
        action_spec.num_values * n_atoms,
        name='adv_output')(advantage)
    advantage = jnp.reshape(
        advantage, (-1, action_spec.num_values, n_atoms))

    logits = value + advantage - jnp.mean(advantage, axis=1, keepdims=True)
    return jax.nn.softmax(logits, axis=-1)

  return Rainbow(
      obs_spec=obs_spec,
      action_spec=action_spec,
      network=network,
      optimizer=optax.adam(1e-3),
      batch_size=32,
      rng=hk.PRNGSequence(seed),
      replay_capacity=replay_capacity,
      min_replay_size=100,
      sgd_period=1,
      target_update_period=4,
      discount=0.99,
      n_step=3,
      priority_exponent=0.5,
      importance_sampling_exponent=0.4,
      max_importance_sampling_exponent=1.,
      importance_sampling_exponent_anneal_steps=replay_capacity,
      uniform_sample_probability=0.,
      priority_epsilon=1e-6,
      n_atoms=n_atoms,
      v_min=-10.,
      v_max=10.,
  )
