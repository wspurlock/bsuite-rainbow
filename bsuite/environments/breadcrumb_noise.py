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
"""Breadcrumb environment with a noisy decoy reward."""

from typing import Any, Dict, Optional

from bsuite.environments import base
from bsuite.experiments.breadcrumb_noise import sweep

import dm_env
from dm_env import specs
import numpy as np


class BreadcrumbNoise(base.Environment):
  """A sparse tree-like task with a breadcrumb path and a noisy decoy."""

  def __init__(self,
               depth: int,
               breadcrumb_reward: float = 0.1,
               decoy_depth: int = 2,
               mapping_seed: Optional[int] = None):
    super().__init__()
    if depth <= decoy_depth:
      raise ValueError('depth must be greater than decoy_depth.')

    self._depth = depth
    self._breadcrumb_reward = breadcrumb_reward
    self._goal_reward = float(2 ** depth)
    self._decoy_depth = decoy_depth

    rng = np.random.RandomState(mapping_seed)
    self._breadcrumb_actions = tuple(rng.binomial(1, 0.5, size=depth))
    self._rng = np.random.RandomState(mapping_seed)

    self._path = []
    self._path_depth = 0
    self._on_breadcrumb_path = True

    self._denoised_return = 0.
    self._goal_hits = 0
    self._decoy_hits = 0
    self._max_breadcrumb_depth = 0

    self.bsuite_num_episodes = sweep.NUM_EPISODES

  def _get_observation(self):
    obs = np.zeros(shape=(2,), dtype=np.float32)
    obs[0] = self._path_depth / self._depth
    obs[1] = float(self._on_breadcrumb_path)
    return obs

  def _reset(self) -> dm_env.TimeStep:
    self._path = []
    self._path_depth = 0
    self._on_breadcrumb_path = True
    return dm_env.restart(self._get_observation())

  def _step(self, action: int) -> dm_env.TimeStep:
    reward = 0.
    expected_action = self._breadcrumb_actions[self._path_depth]

    if self._on_breadcrumb_path:
      if action == expected_action:
        reward = self._breadcrumb_reward
        self._max_breadcrumb_depth = max(self._max_breadcrumb_depth,
                                         self._path_depth + 1)
      else:
        self._on_breadcrumb_path = False
        if self._path_depth == self._decoy_depth:
          reward = self._rng.uniform(0., 2.)
          self._decoy_hits += 1

    self._path.append(action)
    self._path_depth += 1

    if self._path_depth == self._depth:
      if self._on_breadcrumb_path:
        reward += self._goal_reward
        self._goal_hits += 1
      self._denoised_return += reward
      return dm_env.termination(reward=reward, observation=self._get_observation())

    self._denoised_return += reward
    return dm_env.transition(reward=reward, observation=self._get_observation())

  def observation_spec(self):
    return specs.Array(
        shape=(2,), dtype=np.float32, name='observation')

  def action_spec(self):
    return specs.DiscreteArray(2, name='action')

  @property
  def optimal_return(self):
    return self._depth * self._breadcrumb_reward + self._goal_reward

  def bsuite_info(self) -> Dict[str, Any]:
    return dict(
        denoised_return=self._denoised_return,
        goal_hits=self._goal_hits,
        decoy_hits=self._decoy_hits,
        max_breadcrumb_depth=self._max_breadcrumb_depth,
    )
