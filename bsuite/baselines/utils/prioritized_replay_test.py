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
"""Tests for prioritized replay."""

from absl.testing import absltest

from bsuite.baselines.utils import prioritized_replay
import numpy as np


class SegmentTreeTest(absltest.TestCase):

  def test_sum_tree_prefix_sum_non_power_of_two_capacity(self):
    tree = prioritized_replay.SumSegmentTree(capacity=3)
    for i in range(3):
      tree.update(i, 1.)

    indices = [tree.find_prefix_sum(i + 0.5) for i in range(3)]
    self.assertEqual(indices, [0, 1, 2])

  def test_min_tree_ignores_unused_leaves(self):
    tree = prioritized_replay.MinSegmentTree(capacity=3)
    tree.update(0, 2.)
    tree.update(1, 3.)

    self.assertEqual(tree.minimum(), 2.)


class PrioritizedReplayTest(absltest.TestCase):

  def test_sampling_non_power_of_two_capacity_uses_valid_indices(self):
    replay = prioritized_replay.PrioritizedReplay(
        capacity=3,
        priority_exponent=1.,
        importance_sampling_exponent=0.)

    for i in range(3):
      replay.add([np.asarray([i]), i, 1., 1., np.asarray([i + 1])])

    replay.update_priorities(
        np.asarray([0, 1, 2]),
        np.asarray([1., 1., 1.], dtype=np.float32))

    batch = replay.sample(50)
    self.assertTrue(np.all((0 <= batch.indices) & (batch.indices < 3)))


if __name__ == '__main__':
  absltest.main()
