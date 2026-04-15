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
"""Tests for bsuite.experiments.moat."""

from absl.testing import absltest
from bsuite.environments import moat
from dm_env import test_utils

import numpy as np


class InterfaceTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

  def make_object_under_test(self):
    return moat.Moat(moat_length=4)

  def make_action_sequence(self):
    valid_actions = [0, 1]
    rng = np.random.RandomState(42)

    for _ in range(100):
      yield rng.choice(valid_actions)


class MoatRewardTest(absltest.TestCase):

  def test_crossing_moat_reaches_positive_goal(self):
    env = moat.Moat(moat_length=5, moat_cost=0.5, goal_reward=1.0)
    timestep = env.reset()
    rewards = []
    while not timestep.last():
      timestep = env.step(0)
      rewards.append(timestep.reward)
    self.assertEqual(rewards, [-0.5, 0.0, 1.0])

  def test_multiple_crossings_are_counted(self):
    env = moat.Moat(moat_length=5)
    env.reset()

    env.step(0)
    self.assertEqual(env.bsuite_info()['total_moat_crossings'], 1)

    env.step(1)
    env.step(0)
    self.assertEqual(env.bsuite_info()['total_moat_crossings'], 2)

  def test_starts_in_middle_of_chain(self):
    env = moat.Moat(moat_length=7)
    timestep = env.reset()
    self.assertEqual(np.argmax(timestep.observation), 3)

  def test_smallest_chain_starts_right_of_moat(self):
    env = moat.Moat(moat_length=4)
    timestep = env.reset()
    self.assertEqual(np.argmax(timestep.observation), 3)


if __name__ == '__main__':
  absltest.main()
