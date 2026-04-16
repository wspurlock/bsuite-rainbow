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
"""Moat environment for reward-directed exploration."""

from typing import Any, Dict

from bsuite.environments import base
from bsuite.experiments.moat import sweep

import dm_env
from dm_env import specs
import numpy as np


class Moat(base.Environment):
  """A 1D chain with a moat, a neutral gap, and a positive goal."""

  def __init__(self,
               moat_length: int,
               moat_cost: float = 0.1,
               goal_reward: float = 1.0):
    super().__init__()
    if moat_length < 4:
      raise ValueError('moat_length must be at least 4.')
    if moat_cost < 0:
      raise ValueError('moat_cost must be non-negative.')
    if goal_reward <= 0:
      raise ValueError('goal_reward must be positive.')

    self._moat_length = moat_length
    self._moat_cost = moat_cost
    self._goal_reward = goal_reward

    self._goal_position = 0
    self._gap_position = 1
    self._moat_position = 2
    self._start_position = max(self._moat_position + 1, moat_length // 2)
    self._episode_len = moat_length

    self._position = self._start_position
    self._timestep = 0

    self._denoised_return = 0.
    self._goal_hits = 0
    self._total_moat_crossings = 0

    self.bsuite_num_episodes = sweep.NUM_EPISODES

  def _get_observation(self):
    obs = np.zeros(shape=(self._moat_length,), dtype=np.float32)
    obs[self._position] = 1.
    return obs

  def _reset(self) -> dm_env.TimeStep:
    self._position = self._start_position
    self._timestep = 0
    return dm_env.restart(self._get_observation())

  def _step(self, action: int) -> dm_env.TimeStep:
    self._timestep += 1
    old_position = self._position

    if action == 0:
      self._position = max(0, self._position - 1)
    else:
      self._position = min(self._moat_length - 1, self._position + 1)

    reward = 0.
    if old_position == self._moat_position + 1 and self._position == self._moat_position:
      reward -= self._moat_cost
      self._total_moat_crossings += 1
    elif old_position == self._gap_position and self._position == self._goal_position:
      reward += self._goal_reward

    observation = self._get_observation()
    if self._position == self._goal_position:
      self._goal_hits += 1
      self._denoised_return += reward
      return dm_env.termination(reward=reward, observation=observation)
    elif self._timestep == self._episode_len:
      self._denoised_return += reward
      return dm_env.termination(reward=reward, observation=observation)

    self._denoised_return += reward
    return dm_env.transition(reward=reward, observation=observation)

  def observation_spec(self):
    return specs.Array(
        shape=(self._moat_length,),
        dtype=np.float32,
        name='observation')

  def action_spec(self):
    return specs.DiscreteArray(2, name='action')

  @property
  def optimal_return(self):
    return self._goal_reward - self._moat_cost

  def bsuite_info(self) -> Dict[str, Any]:
    return dict(
        denoised_return=self._denoised_return,
        goal_hits=self._goal_hits,
        total_moat_crossings=self._total_moat_crossings,
    )
