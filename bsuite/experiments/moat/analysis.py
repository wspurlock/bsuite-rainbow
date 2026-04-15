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
"""Analysis for moat experiment."""

from typing import Optional, Sequence

from bsuite.experiments.moat import sweep
from bsuite.utils import plotting
import pandas as pd
import plotnine as gg

NUM_EPISODES = sweep.NUM_EPISODES
TAGS = sweep.TAGS


def score(df: pd.DataFrame) -> float:
  """Outputs the fraction of episodes that reached the goal."""
  max_eps = min(df.episode.max(), NUM_EPISODES)
  final = df[df.episode == max_eps]
  return (final.goal_hits / max_eps).mean()


def plot_learning(df: pd.DataFrame,
                  sweep_vars: Optional[Sequence[str]] = None) -> gg.ggplot:
  """Plots average goal-hit rate through time by moat length."""
  plot_df = df.copy()
  plot_df['goal_hit_rate'] = plot_df.goal_hits / plot_df.episode
  p = plotting.plot_regret_group_nosmooth(
      df_in=plot_df,
      group_col='moat_length',
      sweep_vars=sweep_vars,
      regret_col='goal_hit_rate',
      max_episode=NUM_EPISODES,
  )
  return p + gg.ylab('goal-hit rate')


def plot_seeds(df_in: pd.DataFrame,
               sweep_vars: Optional[Sequence[str]] = None) -> gg.ggplot:
  """Plot the goal-hit rate individually by run."""
  df = df_in.copy()
  df['goal_hit_rate'] = df.goal_hits.diff() / df.episode.diff()
  p = plotting.plot_individual_returns(
      df_in=df[df.episode > 1],
      max_episode=NUM_EPISODES,
      return_column='goal_hit_rate',
      colour_var='moat_length',
      yintercept=1.,
      sweep_vars=sweep_vars,
  )
  return p + gg.ylab('goal hits per episode')
