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
"""Breadcrumb trap environment for reward-directed exploration."""

from typing import Any, Dict, Optional

from bsuite.environments import base
from bsuite.experiments.breadcrumb_trap import sweep

import dm_env
from dm_env import specs
import numpy as np


class BreadcrumbTrap(base.Environment):
  """A negative trail leading into an exponentially growing empty subtree."""

  def __init__(self,
               trail_length: int,
               move_cost: float = 0.1,
               mapping_seed: Optional[int] = None):
    super().__init__()
    if trail_length <= 0:
      raise ValueError('trail_length must be positive.')
    if move_cost < 0:
      raise ValueError('move_cost must be non-negative.')

    self._trail_length = trail_length
    self._move_cost = move_cost
    self._depth = 2 * trail_length

    rng = np.random.RandomState(mapping_seed)
    self._trail_actions = tuple(rng.binomial(1, 0.5, size=trail_length))

    self._path = []
    self._path_depth = 0
    self._on_trail_path = True
    self._in_trap_subtree = False

    self._denoised_return = 0.
    self._trap_entries = 0
    self._max_trap_depth = 0
    self._total_negative_reward = 0.

    self.bsuite_num_episodes = sweep.NUM_EPISODES

  def _get_observation(self):
    obs = np.zeros(shape=(4,), dtype=np.float32)
    obs[0] = self._path_depth / self._depth
    if self._on_trail_path and self._path_depth < self._trail_length:
      obs[1] = 1.
    elif self._in_trap_subtree:
      obs[2] = 1.
    else:
      obs[3] = 1.
    return obs

  def _reset(self) -> dm_env.TimeStep:
    self._path = []
    self._path_depth = 0
    self._on_trail_path = True
    self._in_trap_subtree = False
    return dm_env.restart(self._get_observation())

  def _step(self, action: int) -> dm_env.TimeStep:
    reward = 0.

    if self._on_trail_path and self._path_depth < self._trail_length:
      expected_action = self._trail_actions[self._path_depth]
      if action == expected_action:
        reward = -self._move_cost
        self._total_negative_reward += reward
        if self._path_depth + 1 == self._trail_length:
          self._in_trap_subtree = True
          self._trap_entries += 1
      else:
        self._on_trail_path = False

    self._path.append(action)
    self._path_depth += 1
    if self._in_trap_subtree and self._path_depth > self._trail_length:
      self._max_trap_depth = max(self._max_trap_depth,
                                 self._path_depth - self._trail_length)

    observation = self._get_observation()
    self._denoised_return += reward
    if self._path_depth == self._depth:
      return dm_env.termination(reward=reward, observation=observation)
    return dm_env.transition(reward=reward, observation=observation)

  def observation_spec(self):
    return specs.Array(
        shape=(4,), dtype=np.float32, name='observation')

  def action_spec(self):
    return specs.DiscreteArray(2, name='action')

  @property
  def optimal_return(self):
    return 0.

  def bsuite_info(self) -> Dict[str, Any]:
    return dict(
        denoised_return=self._denoised_return,
        trap_entries=self._trap_entries,
        max_trap_depth=self._max_trap_depth,
        total_negative_reward=self._total_negative_reward,
    )
