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
"""Tests for the Rainbow agent."""

from absl.testing import absltest

import dm_env
from dm_env import specs
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from bsuite.baselines.jax.rainbow.agent import Rainbow


class RainbowAgentTest(absltest.TestCase):

  def test_importance_sampling_exponent_anneals_to_one(self):
    obs_spec = specs.Array(shape=(1,), dtype=np.float32)
    action_spec = specs.DiscreteArray(num_values=2)

    def network(inputs: jnp.ndarray) -> jnp.ndarray:
      x = hk.Flatten()(inputs)
      logits = hk.Linear(action_spec.num_values * 3)(x)
      logits = jnp.reshape(logits, (-1, action_spec.num_values, 3))
      return jax.nn.softmax(logits, axis=-1)

    agent = Rainbow(
        obs_spec=obs_spec,
        action_spec=action_spec,
        network=network,
        optimizer=optax.adam(1e-3),
        batch_size=2,
        rng=hk.PRNGSequence(0),
        replay_capacity=8,
        min_replay_size=100,
        sgd_period=1,
        target_update_period=4,
        discount=0.99,
        n_step=1,
        priority_exponent=0.5,
        importance_sampling_exponent=0.4,
        max_importance_sampling_exponent=1.,
        importance_sampling_exponent_anneal_steps=4,
        uniform_sample_probability=0.,
        priority_epsilon=1e-6,
        n_atoms=3,
        v_min=-1.,
        v_max=1.)

    timestep = dm_env.restart(np.zeros((1,), dtype=np.float32))
    new_timestep = dm_env.transition(
        reward=1., observation=np.ones((1,), dtype=np.float32), discount=1.)

    self.assertAlmostEqual(agent._replay.importance_sampling_exponent, 0.4)

    expected = [0.55, 0.7, 0.85, 1.]
    for beta in expected:
      agent.update(timestep, 0, new_timestep)
      self.assertAlmostEqual(agent._replay.importance_sampling_exponent, beta)


if __name__ == '__main__':
  absltest.main()
