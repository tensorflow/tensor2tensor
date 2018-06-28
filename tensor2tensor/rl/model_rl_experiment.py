# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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
r"""Training of model-based RL agents.

Example invocation:

python -m tensor2tensor.rl.model_rl_experiment \
    --output_dir=$HOME/t2t/rl_v1 \
    --loop_hparams_set=rl_modelrl_base \
    --loop_hparams='true_env_generator_num_steps=10000,epochs=3'
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import copy
import datetime
import math
import os
import time

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.layers import discretization
from tensor2tensor.rl import rl_trainer_lib
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("loop_hparams_set", "rl_modelrl_base",
                    "Which RL hparams set to use.")
flags.DEFINE_string("loop_hparams", "", "Overrides for overall loop HParams.")


def setup_directories(base_dir, subdirs):
  base_dir = os.path.expanduser(base_dir)
  tf.gfile.MakeDirs(base_dir)

  all_dirs = {}
  for subdir in subdirs:
    dir_name = os.path.join(base_dir, subdir)
    tf.gfile.MakeDirs(dir_name)
    all_dirs[subdir] = dir_name
  return all_dirs


def make_relative_timing_fn():
  """Make a function that logs the duration since it was made."""
  start_time = time.time()

  def format_relative_time():
    time_delta = time.time() - start_time
    return str(datetime.timedelta(seconds=time_delta))

  def log_relative_time():
    tf.logging.info("Timing: %s", format_relative_time())

  return log_relative_time


@contextlib.contextmanager
def temporary_flags(flag_settings):
  old_values = {}
  for flag_name, flag_value in flag_settings.items():
    old_values[flag_name] = getattr(FLAGS, flag_name)
    setattr(FLAGS, flag_name, flag_value)
  yield
  for flag_name, flag_value in old_values.items():
    setattr(FLAGS, flag_name, flag_value)


def generate_real_env_data(problem_name, agent_policy_path, hparams, data_dir,
                           tmp_dir, autoencoder_path=None, eval_phase=False):
  """Run the agent against the real environment and return mean reward."""
  tf.gfile.MakeDirs(data_dir)
  with temporary_flags({
      "problem": problem_name,
      "agent_policy_path": agent_policy_path,
      "autoencoder_path": autoencoder_path,
      "only_use_ae_for_policy": True,
  }):
    gym_problem = registry.problem(problem_name)
    gym_problem.settable_num_steps = hparams.true_env_generator_num_steps
    gym_problem.eval_phase = eval_phase
    gym_problem.generate_data(data_dir, tmp_dir)
    mean_reward = gym_problem.statistics.sum_of_rewards / \
                  (1.0 + gym_problem.statistics.number_of_dones)

  return mean_reward


def make_log_fn(epoch, log_relative_time_fn):

  def log(msg, *args):
    msg %= args
    tf.logging.info("%s Epoch %d: %s", ">>>>>>>", epoch, msg)
    log_relative_time_fn()

  return log


def train_autoencoder(problem_name, data_dir, output_dir, hparams, epoch):
  """Train autoencoder on problem_name."""
  train_steps = hparams.autoencoder_train_steps * (epoch + 2)
  with temporary_flags({
      "problem": problem_name,
      "data_dir": data_dir,
      "output_dir": output_dir,
      "model": "autoencoder_ordered_discrete",
      "hparams_set": "autoencoder_discrete_pong",
      "train_steps": train_steps,
      "eval_steps": 100,
  }):
    t2t_trainer.main([])


def train_agent(problem_name, agent_model_dir,
                event_dir, world_model_dir, epoch_data_dir, hparams,
                autoencoder_path=None, epoch=0):
  """Train the PPO agent in the simulated environment."""
  gym_problem = registry.problem(problem_name)
  ppo_hparams = trainer_lib.create_hparams(hparams.ppo_params)
  ppo_epochs_num = hparams.ppo_epochs_num
  ppo_hparams.epochs_num = ppo_epochs_num
  ppo_hparams.eval_every_epochs = 50
  ppo_hparams.save_models_every_epochs = ppo_epochs_num
  ppo_hparams.epoch_length = hparams.ppo_epoch_length
  ppo_hparams.num_agents = hparams.ppo_num_agents
  ppo_hparams.world_model_dir = world_model_dir
  ppo_hparams.add_hparam("force_beginning_resets", True)
  if hparams.ppo_learning_rate:
    ppo_hparams.learning_rate = hparams.ppo_learning_rate

  # Adding model hparams for model specific adjustments
  model_hparams = trainer_lib.create_hparams(hparams.generative_model_params)
  ppo_hparams.add_hparam("model_hparams", model_hparams)

  environment_spec = copy.copy(gym_problem.environment_spec)
  environment_spec.simulation_random_starts = hparams.simulation_random_starts
  environment_spec.intrinsic_reward_scale = hparams.intrinsic_reward_scale

  ppo_hparams.add_hparam("environment_spec", environment_spec)

  with temporary_flags({
      "problem": problem_name,
      "model": hparams.generative_model,
      "hparams_set": hparams.generative_model_params,
      "output_dir": world_model_dir,
      "data_dir": epoch_data_dir,
      "autoencoder_path": autoencoder_path,
  }):
    rl_trainer_lib.train(ppo_hparams, event_dir, agent_model_dir, epoch=epoch)


def evaluate_world_model(simulated_problem_name, problem_name, hparams,
                         world_model_dir, epoch_data_dir, tmp_dir,
                         autoencoder_path=None):
  """Generate simulated environment data and return reward accuracy."""
  gym_simulated_problem = registry.problem(simulated_problem_name)
  sim_steps = hparams.simulated_env_generator_num_steps
  gym_simulated_problem.settable_num_steps = sim_steps
  with temporary_flags({
      "problem": problem_name,
      "model": hparams.generative_model,
      "hparams_set": hparams.generative_model_params,
      "data_dir": epoch_data_dir,
      "output_dir": world_model_dir,
      "autoencoder_path": autoencoder_path,
  }):
    gym_simulated_problem.generate_data(epoch_data_dir, tmp_dir)
  n = max(1., gym_simulated_problem.statistics.number_of_dones)
  model_reward_accuracy = (
      gym_simulated_problem.statistics.successful_episode_reward_predictions
      / float(n))
  return model_reward_accuracy


def train_world_model(problem_name, data_dir, output_dir, hparams, epoch,
                      use_autoencoder=False):
  """Train the world model on problem_name."""
  train_steps = hparams.model_train_steps * (epoch + 2)
  with temporary_flags({
      "data_dir": data_dir,
      "output_dir": output_dir,
      "problem": problem_name,
      "model": hparams.generative_model,
      "hparams_set": hparams.generative_model_params,
      "eval_steps": 100,
      "train_steps": train_steps,
      # Hack: If training on autoencoded frames, autoencoder_path needs to be
      # set so that the problem reports the right sizes for frames.
      "autoencoder_path": "dummy" if use_autoencoder else None,
  }):
    t2t_trainer.main([])


def encode_dataset(model, dataset, problem, ae_hparams, autoencoder_path,
                   out_files):
  """Encode all frames in dataset with model and write them out to out_files."""
  batch_size = 8
  dataset = dataset.batch(batch_size)
  examples = dataset.make_one_shot_iterator().get_next()
  images = examples.pop("frame")
  images = tf.expand_dims(images, 1)

  encoded = model.encode(images)
  encoded_frame_height = int(
      math.ceil(problem.frame_height / 2**ae_hparams.num_hidden_layers))
  encoded_frame_width = int(
      math.ceil(problem.frame_width / 2**ae_hparams.num_hidden_layers))
  num_bits = 8
  encoded = tf.reshape(
      encoded, [-1, encoded_frame_height, encoded_frame_width, 3, num_bits])
  encoded = tf.cast(discretization.bit_to_int(encoded, num_bits), tf.uint8)

  pngs = tf.map_fn(tf.image.encode_png, encoded, dtype=tf.string,
                   back_prop=False)

  with tf.Session() as sess:
    autoencoder_saver = tf.train.Saver(tf.global_variables("autoencoder.*"))
    trainer_lib.restore_checkpoint(autoencoder_path, autoencoder_saver, sess,
                                   must_restore=True)

    def generator():
      """Generate examples."""
      while True:
        try:
          pngs_np, examples_np = sess.run([pngs, examples])
          rewards_np = [list(el) for el in examples_np["reward"]]
          actions_np = [list(el) for el in examples_np["action"]]
          pngs_np = [el for el in pngs_np]
          for action, reward, png in zip(actions_np, rewards_np, pngs_np):
            yield {
                "action": action,
                "reward": reward,
                "image/encoded": [png],
                "image/format": ["png"],
                "image/height": [encoded_frame_height],
                "image/width": [encoded_frame_width],
            }
        except tf.errors.OutOfRangeError:
          break

    generator_utils.generate_files(
        generator(), out_files,
        cycle_every_n=problem.total_number_of_frames // 10)


def encode_env_frames(problem_name, ae_problem_name, autoencoder_path,
                      epoch_data_dir):
  """Encode all frames from problem_name and write out as ae_problem_name."""
  with tf.Graph().as_default():
    ae_hparams = trainer_lib.create_hparams("autoencoder_discrete_pong",
                                            problem_name=problem_name)
    problem = ae_hparams.problem
    model = registry.model("autoencoder_ordered_discrete")(
        ae_hparams, tf.estimator.ModeKeys.EVAL)

    ae_problem = registry.problem(ae_problem_name)
    ae_training_paths = ae_problem.training_filepaths(epoch_data_dir, 10, True)
    ae_eval_paths = ae_problem.dev_filepaths(epoch_data_dir, 1, True)

    skip_train = False
    skip_eval = False
    for path in ae_training_paths:
      if tf.gfile.Exists(path):
        skip_train = True
        break
    for path in ae_eval_paths:
      if tf.gfile.Exists(path):
        skip_eval = True
        break

    # Encode train data
    if not skip_train:
      dataset = problem.dataset(tf.estimator.ModeKeys.TRAIN, epoch_data_dir,
                                shuffle_files=False, output_buffer_size=100,
                                preprocess=False)
      encode_dataset(model, dataset, problem, ae_hparams, autoencoder_path,
                     ae_training_paths)

    # Encode eval data
    if not skip_eval:
      dataset = problem.dataset(tf.estimator.ModeKeys.EVAL, epoch_data_dir,
                                shuffle_files=False, output_buffer_size=100,
                                preprocess=False)
      encode_dataset(model, dataset, problem, ae_hparams, autoencoder_path,
                     ae_eval_paths)


def check_problems(problem_names):
  for problem_name in problem_names:
    registry.problem(problem_name)


def training_loop(hparams, output_dir, report_fn=None, report_metric=None):
  """Run the main training loop."""
  if report_fn:
    assert report_metric is not None

  # Global state

  # Directories
  subdirectories = ["data", "tmp", "world_model", "ppo"]
  using_autoencoder = hparams.autoencoder_train_steps > 0
  if using_autoencoder:
    subdirectories.append("autoencoder")
  directories = setup_directories(output_dir, subdirectories)

  # Problems
  problem_name = "gym_discrete_problem_with_agent_on_%s" % hparams.game
  ae_problem_name = problem_name + "_ae"
  simulated_problem_name = (
      "gym_simulated_discrete_problem_with_agent_on_%s" % hparams.game)
  world_model_problem = ae_problem_name if using_autoencoder else problem_name
  check_problems([problem_name, world_model_problem, simulated_problem_name])

  # Autoencoder model dir
  autoencoder_model_dir = (FLAGS.autoencoder_path or
                           directories.get("autoencoder"))
  FLAGS.autoencoder_path = None

  # Timing log function
  log_relative_time = make_relative_timing_fn()

  # Per-epoch state
  epoch_metrics = []
  epoch_data_dirs = []

  # Collect data from the real environment with random policy
  data_dir = os.path.join(directories["data"], "random")
  epoch_data_dirs.append(data_dir)
  tf.logging.info("Generating real environment data with random policy")
  mean_reward = generate_real_env_data(
      problem_name, None, hparams, data_dir, directories["tmp"])
  tf.logging.info("Mean reward (random): %.4f", mean_reward)

  for epoch in range(hparams.epochs):
    is_final_epoch = (epoch + 1) == hparams.epochs
    log = make_log_fn(epoch, log_relative_time)

    # Combine all previously collected environment data
    epoch_data_dir = os.path.join(directories["data"], str(epoch))
    tf.gfile.MakeDirs(epoch_data_dir)
    # Because the data is being combined in every iteration, we only need to
    # copy from the previous directory.
    combine_training_data(registry.problem(problem_name),
                          epoch_data_dir,
                          epoch_data_dirs[-1:])
    epoch_data_dirs.append(epoch_data_dir)

    if using_autoencoder:
      # Train the Autoencoder on all prior environment frames
      log("Training Autoencoder")
      train_autoencoder(problem_name, epoch_data_dir, autoencoder_model_dir,
                        hparams, epoch)

      log("Autoencoding environment frames")
      encode_env_frames(problem_name, world_model_problem,
                        autoencoder_model_dir, epoch_data_dir)

    # Train world model
    log("Training world model")
    train_world_model(world_model_problem, epoch_data_dir,
                      directories["world_model"], hparams, epoch,
                      use_autoencoder=using_autoencoder)

    # Evaluate world model
    model_reward_accuracy = 0.
    if hparams.eval_world_model:
      log("Evaluating world model")
      model_reward_accuracy = evaluate_world_model(
          simulated_problem_name, world_model_problem, hparams,
          directories["world_model"],
          epoch_data_dir, directories["tmp"],
          autoencoder_path=autoencoder_model_dir)
      log("World model reward accuracy: %.4f", model_reward_accuracy)

    # Train PPO
    log("Training PPO")
    ppo_event_dir = os.path.join(directories["ppo"], str(epoch))
    ppo_model_dir = directories["ppo"]
    if not hparams.ppo_continue_training:
      ppo_model_dir = ppo_event_dir
    train_agent(simulated_problem_name, ppo_model_dir,
                ppo_event_dir, directories["world_model"], epoch_data_dir,
                hparams, autoencoder_path=autoencoder_model_dir, epoch=epoch)

    # Collect data from the real environment.
    log("Generating real environment data")
    if is_final_epoch:
      epoch_data_dir = os.path.join(epoch_data_dir, "final_eval")
    mean_reward = generate_real_env_data(
        problem_name, ppo_model_dir, hparams, epoch_data_dir,
        directories["tmp"], autoencoder_path=autoencoder_model_dir,
        eval_phase=is_final_epoch)
    log("Mean reward during generation: %.4f", mean_reward)

    # Report metrics.
    eval_metrics = {"model_reward_accuracy": model_reward_accuracy,
                    "mean_reward": mean_reward}
    epoch_metrics.append(eval_metrics)
    log("Eval metrics: %s", str(eval_metrics))
    if report_fn:
      report_fn(eval_metrics[report_metric], epoch)

  # Report the evaluation metrics from the final epoch
  return epoch_metrics[-1]


def combine_training_data(problem, final_data_dir, old_data_dirs,
                          copy_last_eval_set=True):
  """Add training data from old_data_dirs into final_data_dir."""
  for i, data_dir in enumerate(old_data_dirs):
    suffix = os.path.basename(data_dir)
    # Glob train files in old data_dir
    old_train_files = tf.gfile.Glob(
        problem.filepattern(data_dir, tf.estimator.ModeKeys.TRAIN))
    if (i + 1) == len(old_data_dirs) and copy_last_eval_set:
      old_train_files += tf.gfile.Glob(
          problem.filepattern(data_dir, tf.estimator.ModeKeys.EVAL))
    for fname in old_train_files:
      # Move them to the new data_dir with a suffix
      # Since the data is read based on a prefix filepattern, adding the suffix
      # should be fine.
      new_fname = os.path.join(final_data_dir,
                               os.path.basename(fname) + "." + suffix)
      if not tf.gfile.Exists(new_fname):
        tf.gfile.Copy(fname, new_fname)


@registry.register_hparams
def rl_modelrl_base():
  return tf.contrib.training.HParams(
      epochs=3,
      # Total frames used for training =
      # steps * (1 - 1/11) * epochs
      # 1/11 steps are used for evaluation data
      # 100k frames for training = 36666
      true_env_generator_num_steps=36666,
      generative_model="next_frame_basic",
      generative_model_params="next_frame",
      ppo_params="ppo_pong_base",
      autoencoder_train_steps=0,
      model_train_steps=50000,
      simulated_env_generator_num_steps=2000,
      simulation_random_starts=True,
      intrinsic_reward_scale=0.,
      ppo_epochs_num=200,  # This should be enough to see something
      # Our simulated envs do not know how to reset.
      # You should set ppo_time_limit to the value you believe that
      # the simulated env produces a reasonable output.
      ppo_time_limit=200,  # TODO(blazej): this param is unused
      # It makes sense to have ppo_time_limit=ppo_epoch_length,
      # though it is not necessary.
      ppo_epoch_length=60,
      ppo_num_agents=16,
      ppo_learning_rate=0.,
      # Whether the PPO agent should be restored from the previous iteration, or
      # should start fresh each time.
      ppo_continue_training=True,
      game="wrapped_long_pong",
      # Whether to evaluate the world model in each iteration of the loop to get
      # the model_reward_accuracy metric.
      eval_world_model=True,
  )


@registry.register_hparams
def rl_modelrl_base_stochastic():
  """Base setting with a stochastic next-frame model."""
  hparams = rl_modelrl_base()
  hparams.generative_model = "next_frame_stochastic"
  hparams.generative_model_params = "next_frame_stochastic_cutoff"
  return hparams


@registry.register_hparams
def rl_modelrl_medium():
  """Small set for larger testing."""
  hparams = rl_modelrl_base()
  hparams.true_env_generator_num_steps //= 2
  return hparams


@registry.register_hparams
def rl_modelrl_25k():
  """Small set for larger testing."""
  hparams = rl_modelrl_medium()
  hparams.true_env_generator_num_steps //= 2
  return hparams


@registry.register_hparams
def rl_modelrl_short():
  """Small set for larger testing."""
  hparams = rl_modelrl_base()
  hparams.true_env_generator_num_steps //= 5
  hparams.model_train_steps //= 10
  hparams.ppo_epochs_num //= 10
  return hparams


@registry.register_hparams
def rl_modelrl_tiny():
  """Tiny set for testing."""
  return rl_modelrl_base().override_from_dict(
      tf.contrib.training.HParams(
          epochs=2,
          true_env_generator_num_steps=100,
          simulated_env_generator_num_steps=100,
          model_train_steps=2,
          ppo_epochs_num=2,
          ppo_time_limit=5,
          ppo_epoch_length=5,
          ppo_num_agents=2,
      ).values())


@registry.register_hparams
def rl_modelrl_tiny_stochastic():
  """Tiny setting with a stochastic next-frame model."""
  hparams = rl_modelrl_tiny()
  hparams.generative_model = "next_frame_stochastic"
  hparams.generative_model_params = "next_frame_stochastic_tiny"
  return hparams


@registry.register_hparams
def rl_modelrl_l1_base():
  """Parameter set with L1 loss."""
  hparams = rl_modelrl_base()
  hparams.generative_model_params = "next_frame_l1"
  return hparams


@registry.register_hparams
def rl_modelrl_l1_medium():
  """Medium parameter set with L1 loss."""
  hparams = rl_modelrl_medium()
  hparams.generative_model_params = "next_frame_l1"
  return hparams


@registry.register_hparams
def rl_modelrl_l1_short():
  """Short parameter set with L1 loss."""
  hparams = rl_modelrl_short()
  hparams.generative_model_params = "next_frame_l1"
  return hparams


@registry.register_hparams
def rl_modelrl_l1_tiny():
  """Tiny parameter set with L1 loss."""
  hparams = rl_modelrl_tiny()
  hparams.generative_model_params = "next_frame_l1"
  return hparams


@registry.register_hparams
def rl_modelrl_l2_base():
  """Parameter set with L2 loss."""
  hparams = rl_modelrl_base()
  hparams.generative_model_params = "next_frame_l2"
  return hparams


@registry.register_hparams
def rl_modelrl_l2_medium():
  """Medium parameter set with L2 loss."""
  hparams = rl_modelrl_medium()
  hparams.generative_model_params = "next_frame_l2"
  return hparams


@registry.register_hparams
def rl_modelrl_l2_short():
  """Short parameter set with L2 loss."""
  hparams = rl_modelrl_short()
  hparams.generative_model_params = "next_frame_l2"
  return hparams


@registry.register_hparams
def rl_modelrl_l2_tiny():
  """Tiny parameter set with L2 loss."""
  hparams = rl_modelrl_tiny()
  hparams.generative_model_params = "next_frame_l2"
  return hparams


@registry.register_hparams
def rl_modelrl_ae_base():
  """Parameter set for autoencoders."""
  hparams = rl_modelrl_base()
  hparams.ppo_params = "ppo_pong_ae_base"
  hparams.generative_model_params = "next_frame_ae"
  hparams.autoencoder_train_steps = 50000
  return hparams


@registry.register_hparams
def rl_modelrl_ae_25k():
  hparams = rl_modelrl_ae_base()
  hparams.true_env_generator_num_steps //= 4
  return hparams


@registry.register_hparams
def rl_modelrl_ae_l1_base():
  """Parameter set for autoencoders and L1 loss."""
  hparams = rl_modelrl_ae_base()
  hparams.generative_model_params = "next_frame_l1"
  return hparams


@registry.register_hparams
def rl_modelrl_ae_l2_base():
  """Parameter set for autoencoders and L2 loss."""
  hparams = rl_modelrl_ae_base()
  hparams.generative_model_params = "next_frame_l2"
  return hparams


@registry.register_hparams
def rl_modelrl_ae_medium():
  """Medium parameter set for autoencoders."""
  hparams = rl_modelrl_ae_base()
  hparams.true_env_generator_num_steps //= 2
  return hparams


@registry.register_hparams
def rl_modelrl_ae_short():
  """Small parameter set for autoencoders."""
  hparams = rl_modelrl_ae_base()
  hparams.autoencoder_train_steps //= 10
  hparams.true_env_generator_num_steps //= 5
  hparams.model_train_steps //= 10
  hparams.ppo_epochs_num //= 10
  return hparams


@registry.register_hparams
def rl_modelrl_ae_tiny():
  """Tiny set for testing autoencoders."""
  hparams = rl_modelrl_tiny()
  hparams.ppo_params = "ppo_pong_ae_base"
  hparams.generative_model_params = "next_frame_ae"
  hparams.autoencoder_train_steps = 2
  return hparams


@registry.register_hparams
def rl_modelrl_breakout_tiny():
  """Tiny set for testing Breakout."""
  hparams = rl_modelrl_tiny()
  hparams.game = "wrapped_breakout"
  return hparams


@registry.register_hparams
def rl_modelrl_breakout_base():
  """Base set for testing Breakout."""
  hparams = rl_modelrl_base()
  hparams.game = "wrapped_breakout"
  return hparams


@registry.register_hparams
def rl_modelrl_breakout_ae_base():
  """Base set for testing Breakout with an autoencoder."""
  hparams = rl_modelrl_ae_base()
  hparams.game = "wrapped_breakout"
  return hparams


@registry.register_hparams
def rl_modelrl_breakout_medium():
  """Medium set for testing Breakout."""
  hparams = rl_modelrl_medium()
  hparams.game = "wrapped_breakout"
  return hparams


@registry.register_hparams
def rl_modelrl_breakout_ae_medium():
  """Medium set for testing Breakout with an autoencoder."""
  hparams = rl_modelrl_ae_medium()
  hparams.game = "wrapped_breakout"
  return hparams


@registry.register_hparams
def rl_modelrl_breakout_short():
  """Short set for testing Breakout."""
  hparams = rl_modelrl_short()
  hparams.game = "wrapped_breakout"
  return hparams


@registry.register_hparams
def rl_modelrl_breakout_ae_short():
  """Short set for testing Breakout with an autoencoder."""
  hparams = rl_modelrl_ae_short()
  hparams.game = "wrapped_breakout"
  return hparams


@registry.register_hparams
def rl_modelrl_freeway_tiny():
  """Tiny set for testing Freeway."""
  hparams = rl_modelrl_tiny()
  hparams.game = "freeway"
  return hparams


@registry.register_hparams
def rl_modelrl_freeway_base():
  """Base set for testing Freeway."""
  hparams = rl_modelrl_base()
  hparams.game = "freeway"
  return hparams


@registry.register_hparams
def rl_modelrl_freeway_ae_base():
  """Base set for testing Freeway with an autoencoder."""
  hparams = rl_modelrl_ae_base()
  hparams.game = "freeway"
  return hparams


@registry.register_hparams
def rl_modelrl_freeway_medium():
  """Medium set for testing Freeway."""
  hparams = rl_modelrl_medium()
  hparams.game = "freeway"
  return hparams


@registry.register_hparams
def rl_modelrl_freeway_ae_medium():
  """Medium set for testing Freeway with an autoencoder."""
  hparams = rl_modelrl_ae_medium()
  hparams.game = "freeway"
  return hparams


@registry.register_hparams
def rl_modelrl_freeway_short():
  """Short set for testing Freeway."""
  hparams = rl_modelrl_freeway_medium()
  hparams.true_env_generator_num_steps //= 5
  hparams.model_train_steps //= 2
  hparams.ppo_epochs_num //= 2
  hparams.intrinsic_reward_scale = 0.1
  return hparams


@registry.register_hparams
def rl_modelrl_freeway_ae_short():
  """Short set for testing Freeway with an autoencoder."""
  hparams = rl_modelrl_ae_short()
  hparams.game = "freeway"
  return hparams


@registry.register_hparams
def rl_modelrl_tiny_simulation_deterministic_starts():
  hp = rl_modelrl_tiny()
  hp.simulation_random_starts = False
  return hp


def create_loop_hparams():
  hparams = registry.hparams(FLAGS.loop_hparams_set)
  hparams.parse(FLAGS.loop_hparams)
  return hparams


def main(_):
  hp = create_loop_hparams()
  output_dir = FLAGS.output_dir
  training_loop(hp, output_dir)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
