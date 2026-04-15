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
"""Tests for bsuite.experiments.breadcrumb_trap."""

from absl.testing import absltest
from bsuite.environments import breadcrumb_trap
from dm_env import test_utils

import numpy as np


class InterfaceTest(test_utils.EnvironmentTestMixin, absltest.TestCase):

  def make_object_under_test(self):
    return breadcrumb_trap.BreadcrumbTrap(trail_length=3, mapping_seed=5)

  def make_action_sequence(self):
    valid_actions = [0, 1]
    rng = np.random.RandomState(42)

    for _ in range(100):
      yield rng.choice(valid_actions)


class BreadcrumbTrapRewardTest(absltest.TestCase):

  def test_negative_trail_then_empty_subtree(self):
    env = breadcrumb_trap.BreadcrumbTrap(
        trail_length=2, move_cost=0.5, mapping_seed=0)
    trail_actions = list(env._trail_actions)  # pylint: disable=protected-access

    timestep = env.reset()
    timestep = env.step(trail_actions[0])
    self.assertEqual(timestep.reward, -0.5)
    timestep = env.step(trail_actions[1])
    self.assertEqual(timestep.reward, -0.5)
    timestep = env.step(0)
    self.assertEqual(timestep.reward, 0.)

  def test_leaving_trail_continues_to_leaf(self):
    env = breadcrumb_trap.BreadcrumbTrap(trail_length=3, mapping_seed=0)
    wrong_action = 1 - env._trail_actions[0]  # pylint: disable=protected-access

    timestep = env.reset()
    timestep = env.step(wrong_action)
    self.assertFalse(timestep.last())
    self.assertEqual(timestep.reward, 0.)

    for _ in range(4):
      timestep = env.step(0)
      self.assertFalse(timestep.last())
    timestep = env.step(0)
    self.assertTrue(timestep.last())

  def test_invalid_lengths_raise(self):
    with self.assertRaises(ValueError):
      breadcrumb_trap.BreadcrumbTrap(trail_length=0)
    with self.assertRaises(ValueError):
      breadcrumb_trap.BreadcrumbTrap(
          trail_length=2, move_cost=-0.1)


if __name__ == '__main__':
  absltest.main()
