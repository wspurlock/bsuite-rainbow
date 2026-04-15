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
"""Prioritized replay with n-step returns."""

import collections
from typing import Any, NamedTuple, Optional, Sequence

import numpy as np


class PrioritizedReplaySample(NamedTuple):
  transitions: Sequence[np.ndarray]
  indices: np.ndarray
  weights: np.ndarray


class SumSegmentTree:
  """A sum segment tree backed by a flat numpy array."""

  def __init__(self, capacity: int):
    self._capacity = capacity
    self._tree_capacity = 1
    while self._tree_capacity < capacity:
      self._tree_capacity *= 2
    self._tree = np.zeros(2 * self._tree_capacity - 1, dtype=np.float32)

  def update(self, index: int, value: float):
    tree_index = index + self._tree_capacity - 1
    self._tree[tree_index] = value
    while tree_index > 0:
      tree_index = (tree_index - 1) // 2
      left = 2 * tree_index + 1
      self._tree[tree_index] = self._tree[left] + self._tree[left + 1]

  def total(self) -> float:
    return float(self._tree[0])

  def get(self, index: int) -> float:
    return float(self._tree[index + self._tree_capacity - 1])

  def find_prefix_sum(self, value: float) -> int:
    index = 0
    while 2 * index + 1 < len(self._tree):
      left = 2 * index + 1
      if value <= self._tree[left]:
        index = left
      else:
        value -= self._tree[left]
        index = left + 1
    return index - (self._tree_capacity - 1)


class MinSegmentTree:
  """A min segment tree backed by a flat numpy array."""

  def __init__(self, capacity: int):
    self._capacity = capacity
    self._tree_capacity = 1
    while self._tree_capacity < capacity:
      self._tree_capacity *= 2
    self._tree = np.full(2 * self._tree_capacity - 1, np.inf, dtype=np.float32)

  def update(self, index: int, value: float):
    tree_index = index + self._tree_capacity - 1
    self._tree[tree_index] = value
    while tree_index > 0:
      tree_index = (tree_index - 1) // 2
      left = 2 * tree_index + 1
      self._tree[tree_index] = min(self._tree[left], self._tree[left + 1])

  def minimum(self) -> float:
    return float(self._tree[0])


class PrioritizedReplay:
  """Prioritized replay buffer with bsuite-compatible n-step returns."""

  _data: Optional[Sequence[np.ndarray]]

  def __init__(
      self,
      capacity: int,
      priority_exponent: float,
      importance_sampling_exponent: float,
      uniform_sample_probability: float = 0.,
      normalize_weights: bool = True,
      priority_epsilon: float = 1e-6,
      discount: float = 0.99,
      n_step: int = 1,
  ):
    self._data = None
    self._capacity = capacity
    self._num_added = 0
    self._priority_exponent = priority_exponent
    self._importance_sampling_exponent = importance_sampling_exponent
    self._uniform_sample_probability = uniform_sample_probability
    self._normalize_weights = normalize_weights
    self._priority_epsilon = priority_epsilon
    self._discount = discount
    self._n_step = n_step
    self._max_priority = 1.
    self._sum_tree = SumSegmentTree(capacity)
    self._min_tree = MinSegmentTree(capacity)
    self._n_step_buffer = collections.deque(maxlen=n_step)

  def add(self, items: Sequence[Any]):
    """Adds a transition [obs, action, reward, discount, next_obs]."""
    if len(items) != 5:
      raise ValueError('Expected five transition items.')

    transition = [
        np.asarray(items[0]),
        np.asarray(items[1]),
        np.float32(items[2]),
        np.float32(items[3]),
        np.asarray(items[4]),
    ]
    self._n_step_buffer.append(transition)

    if transition[3] == 0.:
      while self._n_step_buffer:
        self._add_aggregated_transition()
        self._n_step_buffer.popleft()
      return

    if len(self._n_step_buffer) < self._n_step:
      return

    self._add_aggregated_transition()
    self._n_step_buffer.popleft()

  def sample(self, size: int) -> PrioritizedReplaySample:
    if self.size == 0:
      raise ValueError('Cannot sample from an empty replay.')

    indices = np.zeros(size, dtype=np.int32)
    total_priority = self._sum_tree.total()
    segment = total_priority / size
    for i in range(size):
      if np.random.rand() < self._uniform_sample_probability:
        indices[i] = np.random.randint(self.size)
      else:
        low = i * segment
        high = (i + 1) * segment
        mass = np.random.uniform(low, high)
        indices[i] = self._sum_tree.find_prefix_sum(mass)

    transitions = [slot[indices] for slot in self._data]
    probabilities = np.asarray(
        [self._sum_tree.get(index) / total_priority for index in indices],
        dtype=np.float32)
    weights = (self.size * probabilities)**(-self._importance_sampling_exponent)

    if self._normalize_weights:
      min_probability = self._min_tree.minimum() / total_priority
      max_weight = (self.size * min_probability)**(
          -self._importance_sampling_exponent)
      weights /= max_weight

    return PrioritizedReplaySample(
        transitions=transitions,
        indices=indices,
        weights=weights.astype(np.float32))

  def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
    priorities = np.asarray(priorities, dtype=np.float32)
    priorities = np.abs(priorities) + self._priority_epsilon
    self._max_priority = max(self._max_priority, float(np.max(priorities)))

    for index, priority in zip(indices, priorities):
      value = float(priority**self._priority_exponent)
      self._sum_tree.update(int(index), value)
      self._min_tree.update(int(index), value)

  @property
  def size(self) -> int:
    return min(self._capacity, self._num_added)

  @property
  def importance_sampling_exponent(self) -> float:
    return self._importance_sampling_exponent

  def set_importance_sampling_exponent(self, exponent: float):
    self._importance_sampling_exponent = float(exponent)

  def _add_aggregated_transition(self):
    if self._data is None:
      self._preallocate(self._build_n_step_transition())

    transition = self._build_n_step_transition()
    slot = self._num_added % self._capacity
    for buffer, item in zip(self._data, transition):
      buffer[slot] = item

    priority = self._max_priority**self._priority_exponent
    self._sum_tree.update(slot, priority)
    self._min_tree.update(slot, priority)
    self._num_added += 1

  def _build_n_step_transition(self) -> Sequence[np.ndarray]:
    obs, action = self._n_step_buffer[0][:2]
    reward = np.float32(0.)
    cumulative_discount = np.float32(1.)
    next_obs = self._n_step_buffer[-1][4]

    for _, _, step_reward, step_discount, step_next_obs in self._n_step_buffer:
      reward += cumulative_discount * step_reward
      next_obs = step_next_obs
      cumulative_discount *= self._discount * step_discount
      if cumulative_discount == 0.:
        break

    return [obs, action, reward, cumulative_discount, next_obs]

  def _preallocate(self, items: Sequence[Any]):
    arrays = []
    for item in items:
      if item is None:
        raise ValueError('Cannot store `None` objects in replay.')
      arrays.append(np.asarray(item))

    self._data = [
        np.zeros((self._capacity,) + array.shape, dtype=array.dtype)
        for array in arrays
    ]
