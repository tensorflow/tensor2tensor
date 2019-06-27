# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Produces the training and dev data for --problem into --data_dir.

Produces sharded and shuffled TFRecord files of tensorflow.Example protocol
buffers for a variety of registered datasets.

All Problems are registered with @registry.register_problem or are in
_SUPPORTED_PROBLEM_GENERATORS in this file. Each entry maps a string name
(selectable on the command-line with --problem) to a function that takes 2
arguments - input_directory and mode (one of "train" or "dev") - and yields for
each training example a dictionary mapping string feature names to lists of
{string, int, float}. The generator will be run once for each mode.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
import random
import tempfile

import numpy as np

from tensor2tensor import problems as problems_lib  # pylint: disable=unused-import
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.envs import env_problem_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import usr_dir

try:
  # pylint: disable=g-import-not-at-top
  from tensor2tensor.data_generators import algorithmic_math
  from tensor2tensor.data_generators import audio
  from tensor2tensor.data_generators import snli
  from tensor2tensor.data_generators import wsj_parsing
  # pylint: enable=g-import-not-at-top
except ImportError:
  pass

# Improrting here to prevent pylint from ungrouped-imports warning.
import tensorflow as tf  # pylint: disable=g-import-not-at-top

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "", "Data directory.")
flags.DEFINE_string("tmp_dir", "/tmp/t2t_datagen",
                    "Temporary storage directory.")
flags.DEFINE_string("problem", "",
                    "The name of the problem to generate data for.")
flags.DEFINE_string("exclude_problems", "",
                    "Comma-separates list of problems to exclude.")
flags.DEFINE_integer(
    "num_shards", 0, "How many shards to use. Ignored for "
    "registered Problems.")
flags.DEFINE_integer("max_cases", 0,
                     "Maximum number of cases to generate (unbounded if 0).")
flags.DEFINE_integer(
    "env_problem_max_env_steps", 0,
    "Maximum number of steps to take for environment-based problems. "
    "Actions are chosen randomly")
flags.DEFINE_integer(
    "env_problem_batch_size", 0,
    "Number of environments to simulate for environment-based problems.")
flags.DEFINE_bool("only_list", False,
                  "If true, we only list the problems that will be generated.")
flags.DEFINE_integer("random_seed", 429459, "Random seed to use.")
flags.DEFINE_integer("task_id", -1, "For distributed data generation.")
flags.DEFINE_integer("task_id_start", -1, "For distributed data generation.")
flags.DEFINE_integer("task_id_end", -1, "For distributed data generation.")
flags.DEFINE_integer(
    "num_concurrent_processes", None,
    "Applies only to problems for which multiprocess_generate=True.")
flags.DEFINE_string(
    "t2t_usr_dir", "", "Path to a Python module that will be imported. The "
    "__init__.py file should include the necessary imports. "
    "The imported files should contain registrations, "
    "e.g. @registry.register_problem calls, that will then be "
    "available to t2t-datagen.")

# Mapping from problems that we can generate data for to their generators.
# pylint: disable=g-long-lambda
_SUPPORTED_PROBLEM_GENERATORS = {
    "algorithmic_algebra_inverse":
        (lambda: algorithmic_math.algebra_inverse(26, 0, 2, 100000),
         lambda: algorithmic_math.algebra_inverse(26, 3, 3, 10000),
         lambda: None),  # test set
    "parsing_english_ptb8k":
        (lambda: wsj_parsing.parsing_token_generator(
            FLAGS.data_dir, FLAGS.tmp_dir, True, 2**13, 2**9),
         lambda: wsj_parsing.parsing_token_generator(
             FLAGS.data_dir, FLAGS.tmp_dir, False, 2**13, 2**9),
         lambda: None),  # test set
    "parsing_english_ptb16k":
        (lambda: wsj_parsing.parsing_token_generator(
            FLAGS.data_dir, FLAGS.tmp_dir, True, 2**14, 2**9),
         lambda: wsj_parsing.parsing_token_generator(
             FLAGS.data_dir, FLAGS.tmp_dir, False, 2**14, 2**9),
         lambda: None),  # test set
    "inference_snli32k":
        (lambda: snli.snli_token_generator(FLAGS.tmp_dir, True, 2**15),
         lambda: snli.snli_token_generator(FLAGS.tmp_dir, False, 2**15),
         lambda: None),  # test set
    "audio_timit_characters_test": (lambda: audio.timit_generator(
        FLAGS.data_dir, FLAGS.tmp_dir, True, 1718
    ), lambda: audio.timit_generator(FLAGS.data_dir, FLAGS.tmp_dir, False, 626),
                                    lambda: None),  # test set
    "audio_timit_tokens_8k_test": (lambda: audio.timit_generator(
        FLAGS.data_dir,
        FLAGS.tmp_dir,
        True,
        1718,
        vocab_filename="vocab.endefr.%d" % 2**13,
        vocab_size=2**13), lambda: audio.timit_generator(
            FLAGS.data_dir,
            FLAGS.tmp_dir,
            False,
            626,
            vocab_filename="vocab.endefr.%d" % 2**13,
            vocab_size=2**13), lambda: None),  # test set
    "audio_timit_tokens_32k_test": (lambda: audio.timit_generator(
        FLAGS.data_dir,
        FLAGS.tmp_dir,
        True,
        1718,
        vocab_filename="vocab.endefr.%d" % 2**15,
        vocab_size=2**15), lambda: audio.timit_generator(
            FLAGS.data_dir,
            FLAGS.tmp_dir,
            False,
            626,
            vocab_filename="vocab.endefr.%d" % 2**15,
            vocab_size=2**15), lambda: None),  # test set
}

# pylint: enable=g-long-lambda


def set_random_seed():
  """Set the random seed from flag everywhere."""
  tf.set_random_seed(FLAGS.random_seed)
  random.seed(FLAGS.random_seed)
  np.random.seed(FLAGS.random_seed)


def main(_):
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)

  # Calculate the list of problems to generate.
  problems = sorted(
      list(_SUPPORTED_PROBLEM_GENERATORS) + registry.list_base_problems() +
      registry.list_env_problems())
  for exclude in FLAGS.exclude_problems.split(","):
    if exclude:
      problems = [p for p in problems if exclude not in p]
  if FLAGS.problem and FLAGS.problem[-1] == "*":
    problems = [p for p in problems if p.startswith(FLAGS.problem[:-1])]
  elif FLAGS.problem and "," in FLAGS.problem:
    problems = [p for p in problems if p in FLAGS.problem.split(",")]
  elif FLAGS.problem:
    problems = [p for p in problems if p == FLAGS.problem]
  else:
    problems = []

  # Remove TIMIT if paths are not given.
  if getattr(FLAGS, "timit_paths", None):
    problems = [p for p in problems if "timit" not in p]
  # Remove parsing if paths are not given.
  if getattr(FLAGS, "parsing_path", None):
    problems = [p for p in problems if "parsing_english_ptb" not in p]

  if not problems:
    problems_str = "\n  * ".join(
        sorted(
            list(_SUPPORTED_PROBLEM_GENERATORS) +
            registry.list_base_problems() + registry.list_env_problems()))
    error_msg = ("You must specify one of the supported problems to "
                 "generate data for:\n  * " + problems_str + "\n")
    error_msg += ("TIMIT and parsing need data_sets specified with "
                  "--timit_paths and --parsing_path.")
    raise ValueError(error_msg)

  if not FLAGS.data_dir:
    FLAGS.data_dir = tempfile.gettempdir()
    tf.logging.warning(
        "It is strongly recommended to specify --data_dir. "
        "Data will be written to default data_dir=%s.", FLAGS.data_dir)
  FLAGS.data_dir = os.path.expanduser(FLAGS.data_dir)
  tf.gfile.MakeDirs(FLAGS.data_dir)

  tf.logging.info("Generating problems:\n%s" %
                  registry.display_list_by_prefix(problems, starting_spaces=4))
  if FLAGS.only_list:
    return
  for problem in problems:
    set_random_seed()

    if problem in _SUPPORTED_PROBLEM_GENERATORS:
      generate_data_for_problem(problem)
    elif problem in registry.list_base_problems():
      generate_data_for_registered_problem(problem)
    elif problem in registry.list_env_problems():
      generate_data_for_env_problem(problem)
    else:
      tf.logging.error("Problem %s is not a supported problem for datagen.",
                       problem)


def generate_data_for_problem(problem):
  """Generate data for a problem in _SUPPORTED_PROBLEM_GENERATORS."""
  training_gen, dev_gen, test_gen = _SUPPORTED_PROBLEM_GENERATORS[problem]

  num_train_shards = FLAGS.num_shards or 10
  tf.logging.info("Generating training data for %s.", problem)
  train_output_files = generator_utils.train_data_filenames(
      problem + generator_utils.UNSHUFFLED_SUFFIX, FLAGS.data_dir,
      num_train_shards)
  generator_utils.generate_files(training_gen(), train_output_files,
                                 FLAGS.max_cases)
  num_dev_shards = int(num_train_shards * 0.1)
  tf.logging.info("Generating development data for %s.", problem)
  dev_output_files = generator_utils.dev_data_filenames(
      problem + generator_utils.UNSHUFFLED_SUFFIX, FLAGS.data_dir,
      num_dev_shards)
  generator_utils.generate_files(dev_gen(), dev_output_files)
  num_test_shards = int(num_train_shards * 0.1)
  test_output_files = []
  test_gen_data = test_gen()
  if test_gen_data is not None:
    tf.logging.info("Generating test data for %s.", problem)
    test_output_files = generator_utils.test_data_filenames(
        problem + generator_utils.UNSHUFFLED_SUFFIX, FLAGS.data_dir,
        num_test_shards)
    generator_utils.generate_files(test_gen_data, test_output_files)
  all_output_files = train_output_files + dev_output_files + test_output_files
  generator_utils.shuffle_dataset(all_output_files)


def generate_data_in_process(arg):
  problem_name, data_dir, tmp_dir, task_id = arg
  problem = registry.problem(problem_name)
  problem.generate_data(data_dir, tmp_dir, task_id)


def generate_data_for_env_problem(problem_name):
  """Generate data for `EnvProblem`s."""
  assert FLAGS.env_problem_max_env_steps > 0, ("--env_problem_max_env_steps "
                                               "should be greater than zero")
  assert FLAGS.env_problem_batch_size > 0, ("--env_problem_batch_size should be"
                                            " greather than zero")
  problem = registry.env_problem(problem_name)
  task_id = None if FLAGS.task_id < 0 else FLAGS.task_id
  data_dir = os.path.expanduser(FLAGS.data_dir)
  tmp_dir = os.path.expanduser(FLAGS.tmp_dir)
  # TODO(msaffar): Handle large values for env_problem_batch_size where we
  #  cannot create that many environments within the same process.
  problem.initialize(batch_size=FLAGS.env_problem_batch_size)
  env_problem_utils.play_env_problem_randomly(
      problem, num_steps=FLAGS.env_problem_max_env_steps)
  problem.generate_data(data_dir=data_dir, tmp_dir=tmp_dir, task_id=task_id)


def generate_data_for_registered_problem(problem_name):
  """Generate data for a registered problem."""
  tf.logging.info("Generating data for %s.", problem_name)
  if FLAGS.num_shards:
    raise ValueError("--num_shards should not be set for registered Problem.")
  problem = registry.problem(problem_name)
  task_id = None if FLAGS.task_id < 0 else FLAGS.task_id
  data_dir = os.path.expanduser(FLAGS.data_dir)
  tmp_dir = os.path.expanduser(FLAGS.tmp_dir)
  if task_id is None and problem.multiprocess_generate:
    if FLAGS.task_id_start != -1:
      assert FLAGS.task_id_end != -1
      task_id_start = FLAGS.task_id_start
      task_id_end = FLAGS.task_id_end
    else:
      task_id_start = 0
      task_id_end = problem.num_generate_tasks
    pool = multiprocessing.Pool(processes=FLAGS.num_concurrent_processes)
    problem.prepare_to_generate(data_dir, tmp_dir)
    args = [(problem_name, data_dir, tmp_dir, task_id)
            for task_id in range(task_id_start, task_id_end)]
    pool.map(generate_data_in_process, args)
  else:
    problem.generate_data(data_dir, tmp_dir, task_id)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
