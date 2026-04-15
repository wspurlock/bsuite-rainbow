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
"""Run a Rainbow bsuite sweep with one subprocess pinned to each GPU."""

import os
import queue
import subprocess
import sys
import threading
from typing import Iterable, Sequence

from absl import app
from absl import flags

from bsuite import sweep

flags.DEFINE_string(
    'bsuite_id', 'SWEEP', 'BSuite identifier or sweep name to execute.')
flags.DEFINE_list(
    'gpus', None,
    'Comma-separated GPU ids to use, for example "0,1,2,3". One worker '
    'subprocess is launched per GPU.')
flags.DEFINE_string('save_path', '/tmp/bsuite', 'where to save bsuite results')
flags.DEFINE_enum('logging_mode', 'csv', ['csv', 'sqlite', 'terminal'],
                  'which form of logging to use for bsuite results')
flags.DEFINE_boolean('overwrite', False, 'overwrite csv logging if found')
flags.DEFINE_integer('num_episodes', None, 'Number of episodes to run for.')
flags.DEFINE_boolean('verbose', False, 'whether child runs log to std output')
flags.DEFINE_boolean(
    'jax_preallocate', False,
    'Whether each child process should allow JAX GPU memory preallocation.')

FLAGS = flags.FLAGS


def _resolve_bsuite_ids(bsuite_id: str) -> Sequence[str]:
  if bsuite_id in sweep.SWEEP:
    return (bsuite_id,)
  if hasattr(sweep, bsuite_id):
    return tuple(getattr(sweep, bsuite_id))
  raise ValueError(f'Invalid flag: bsuite_id={bsuite_id}.')


def _parse_gpus(gpus: Iterable[str]) -> Sequence[str]:
  parsed = tuple(gpu.strip() for gpu in gpus if gpu.strip())
  if not parsed:
    raise ValueError('At least one GPU id must be provided via --gpus.')
  return parsed


def _build_command(bsuite_id: str) -> Sequence[str]:
  command = [
      sys.executable,
      '-m',
      'bsuite.baselines.jax.rainbow.run',
      f'--bsuite_id={bsuite_id}',
      f'--save_path={FLAGS.save_path}',
      f'--logging_mode={FLAGS.logging_mode}',
      f'--overwrite={FLAGS.overwrite}',
      f'--verbose={FLAGS.verbose}',
  ]
  if FLAGS.num_episodes is not None:
    command.append(f'--num_episodes={FLAGS.num_episodes}')
  return tuple(command)


def _child_env(gpu_id: str) -> dict[str, str]:
  env = dict(os.environ)
  env['CUDA_VISIBLE_DEVICES'] = gpu_id
  if not FLAGS.jax_preallocate:
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
  return env


def _run_worker(gpu_id: str, work: 'queue.Queue[str]', failures: list[str]):
  while True:
    try:
      bsuite_id = work.get_nowait()
    except queue.Empty:
      return

    command = _build_command(bsuite_id)
    print(f'[GPU {gpu_id}] Starting {bsuite_id}')
    try:
      subprocess.run(
          command,
          check=True,
          env=_child_env(gpu_id),
      )
      print(f'[GPU {gpu_id}] Finished {bsuite_id}')
    except subprocess.CalledProcessError as error:
      failures.append(
          f'GPU {gpu_id} failed on {bsuite_id} with exit code '
          f'{error.returncode}')
      print(f'[GPU {gpu_id}] Failed {bsuite_id} ({error.returncode})')
    finally:
      work.task_done()


def main(_):
  gpu_ids = _parse_gpus(FLAGS.gpus or ())
  bsuite_ids = _resolve_bsuite_ids(FLAGS.bsuite_id)

  print('Experiment info')
  print('---------------')
  print(f'Num experiments: {len(bsuite_ids)}')
  print(f'GPUs: {", ".join(gpu_ids)}')
  print(f'Num worker processes: {len(gpu_ids)}')

  work: queue.Queue[str] = queue.Queue()
  for bsuite_id in bsuite_ids:
    work.put(bsuite_id)

  failures: list[str] = []
  threads = [
      threading.Thread(
          target=_run_worker,
          args=(gpu_id, work, failures),
          daemon=False)
      for gpu_id in gpu_ids
  ]

  for thread in threads:
    thread.start()
  for thread in threads:
    thread.join()

  if failures:
    raise RuntimeError('\n'.join(failures))


if __name__ == '__main__':
  app.run(main)
