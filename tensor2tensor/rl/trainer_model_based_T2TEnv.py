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

python -m tensor2tensor.rl.trainer_model_based \
    --output_dir=$HOME/t2t/rl_v1 \
    --loop_hparams_set=rlmb_base \
    --loop_hparams='num_real_env_frames=10000,epochs=3'
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import copy
import datetime
import math
import os
import re
import time

import numpy as np
import six


from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import gym_problems
from tensor2tensor.data_generators import gym_problems_specs
from tensor2tensor.layers import discretization
from tensor2tensor.rl import rl_trainer_lib
from tensor2tensor.rl.envs.tf_atari_wrappers import PyFuncWrapper
from tensor2tensor.rl.envs.utils import InitialFrameChooser
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("loop_hparams_set", "rlmb_base",
                    "Which RL hparams set to use.")
flags.DEFINE_string("loop_hparams", "", "Overrides for overall loop HParams.")
flags.DEFINE_string("job_dir_to_evaluate", "",
                    "Directory of a job to be evaluated.")
flags.DEFINE_string("eval_results_dir", "/tmp",
                    "Directory to store result of evaluation")

HP_SCOPES = ["loop", "model", "ppo"]


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
                           tmp_dir, autoencoder_path=None, eval_phase=False,
                           real_reward=False):
  """Run the agent against the real environment and return mean reward."""
  tf.gfile.MakeDirs(data_dir)
  with temporary_flags({
      "problem": problem_name,
      "agent_policy_path": agent_policy_path,
      "autoencoder_path": autoencoder_path,
  }):
    gym_problem = registry.problem(problem_name)
    if hparams.gather_ppo_real_env_data:
      env_steps_per_epoch = int(hparams.num_real_env_frames / hparams.epochs)
    else:
      env_steps_per_epoch = (
          hparams.num_real_env_frames / (hparams.epochs * (1. - 1./11.)))
    gym_problem.settable_num_steps = env_steps_per_epoch
    gym_problem.settable_eval_phase = eval_phase
    if real_reward and autoencoder_path is None:
      gym_problem._forced_collect_level = 1  # pylint: disable=protected-access
    gym_problem.generate_data(data_dir, tmp_dir)
    mean_reward = None
    if gym_problem.statistics.number_of_dones:
      mean_reward = (gym_problem.statistics.sum_of_rewards /
                     gym_problem.statistics.number_of_dones)

  return mean_reward


def make_log_fn(epoch, log_relative_time_fn):

  def log(msg, *args):
    msg %= args
    tf.logging.info("%s Epoch %d: %s", ">>>>>>>", epoch, msg)
    log_relative_time_fn()

  return log


def train_autoencoder(problem_name, data_dir, output_dir, hparams, epoch):
  """Train autoencoder on problem_name."""
  additional_steps = 1 + hparams.autoencoder_train_steps_initial_multiplier
  train_steps = hparams.autoencoder_train_steps * (epoch + additional_steps)
  with temporary_flags({
      "problem": problem_name,
      "data_dir": data_dir,
      "output_dir": output_dir,
      "model": "autoencoder_ordered_discrete",
      "hparams_set": hparams.autoencoder_hparams_set,
      "train_steps": train_steps,
      "eval_steps": 100,
  }):
    t2t_trainer.main([])


def _ppo_training_epochs(hparams, epoch, is_final_epoch, real_env_training):
  """Helper for PPO restarts."""
  if hparams.gather_ppo_real_env_data:
    assert hparams.real_ppo_epochs_num is 0, (
        "Should be put to 0 to enforce better readability")
    real_training_ppo_epochs_num = int(math.ceil(
        hparams.num_real_env_frames /
        (hparams.epochs*hparams.real_ppo_epoch_length)))
  else:
    real_training_ppo_epochs_num = hparams.real_ppo_epochs_num

  simulated_training_ppo_epochs_num = hparams.ppo_epochs_num

  if epoch == -1:
    assert real_env_training, (
        "Epoch -1 should only be used for PPO collection in real environment.")
    return real_training_ppo_epochs_num
  ppo_training_epochs = (epoch + 1) * (simulated_training_ppo_epochs_num
                                       + real_training_ppo_epochs_num)
  if is_final_epoch:  # Length of training in the final epoch is doubled.
    ppo_training_epochs += simulated_training_ppo_epochs_num
  if real_env_training:
    ppo_training_epochs += real_training_ppo_epochs_num
  return ppo_training_epochs


def train_agent(problem_name, agent_model_dir,
                event_dir, world_model_dir, epoch_data_dir, hparams, epoch=0,
                is_final_epoch=False):
  """Train the PPO agent in the simulated environment."""
  gym_problem = registry.problem(problem_name)
  ppo_hparams = trainer_lib.create_hparams(hparams.ppo_params)
  ppo_params_names = ["epochs_num", "epoch_length",
                      "learning_rate", "num_agents",
                      "optimization_epochs", "eval_every_epochs"]

  for param_name in ppo_params_names:
    ppo_param_name = "ppo_" + param_name
    if ppo_param_name in hparams:
      ppo_hparams.set_hparam(param_name, hparams.get(ppo_param_name))

  ppo_hparams.epochs_num = _ppo_training_epochs(hparams, epoch,
                                                is_final_epoch, False)
  ppo_hparams.save_models_every_epochs = 10
  ppo_hparams.world_model_dir = world_model_dir
  ppo_hparams.add_hparam("force_beginning_resets", True)

  # Adding model hparams for model specific adjustments
  model_hparams = trainer_lib.create_hparams(hparams.generative_model_params)
  ppo_hparams.add_hparam("model_hparams", model_hparams)

  environment_spec = copy.copy(gym_problem.environment_spec)
  environment_spec.simulation_random_starts = hparams.simulation_random_starts
  do_flip = hparams.simulation_flip_first_random_for_beginning
  environment_spec.simulation_flip_first_random_for_beginning = do_flip
  environment_spec.intrinsic_reward_scale = hparams.intrinsic_reward_scale

  ppo_hparams.add_hparam("environment_spec", environment_spec)
  ppo_hparams.add_hparam("initial_frame_chooser", InitialFrameChooser(
      environment_spec, mode=tf.estimator.ModeKeys.TRAIN
  ))

  with temporary_flags({
      "problem": problem_name,
      "model": hparams.generative_model,
      "hparams_set": hparams.generative_model_params,
      "output_dir": world_model_dir,
      "data_dir": epoch_data_dir,
  }):
    rl_trainer_lib.train(ppo_hparams, event_dir + "sim", agent_model_dir,
                         name_scope="ppo_sim%d" % (epoch + 1))

ppo_data_dumper_counter = 0
dumper_path = None


def ppo_data_dumper(observ, reward, done, action):
  """Save frames from PPO to a numpy file."""
  global ppo_data_dumper_counter, dumper_path
  file_path = "{}/frame_{}.npz".format(dumper_path, ppo_data_dumper_counter)
  if gym_problems.frame_dumper_use_disk:
    # np.savez_compressed can't create a tf.gfile, so we need to create it
    # beforehand.
    with tf.gfile.Open(file_path, mode="wb+") as gfile:
      gfile.write("1")
    with tf.gfile.Open(file_path, mode="wb+") as gfile:
      np.savez_compressed(
          gfile, observ=observ, reward=reward, done=done, action=action)
  else:
    data = {"observ": observ, "reward": reward, "done": done, "action": action}
    gym_problems.frame_dumper[file_path] = data
  ppo_data_dumper_counter += 1
  return 0.0


def train_agent_real_env(
    problem_name, agent_model_dir, event_dir, world_model_dir, epoch_data_dir,
    hparams, epoch=0, is_final_epoch=False):
  """Train the PPO agent in the real environment."""
  global dumper_path, ppo_data_dumper_counter

  gym_problem = registry.problem(problem_name)
  ppo_hparams = trainer_lib.create_hparams(hparams.ppo_params)
  ppo_params_names = ["epochs_num", "epoch_length",
                      "learning_rate", "num_agents", "eval_every_epochs",
                      "optimization_epochs", "effective_num_agents"]

  # This should be overridden.
  ppo_hparams.add_hparam("effective_num_agents", None)
  for param_name in ppo_params_names:
    ppo_param_name = "real_ppo_"+ param_name
    if ppo_param_name in hparams:
      ppo_hparams.set_hparam(param_name, hparams.get(ppo_param_name))

  ppo_hparams.epochs_num = _ppo_training_epochs(hparams, epoch,
                                                is_final_epoch, True)
  # We do not save model, as that resets frames that we need at restarts.
  # But we need to save at the last step, so we set it very high.
  ppo_hparams.save_models_every_epochs = 1000000

  environment_spec = copy.copy(gym_problem.environment_spec)

  if hparams.gather_ppo_real_env_data:
    # TODO(piotrmilos):This should be refactored
    assert hparams.real_ppo_num_agents == 1, (
        "It is required to use collect with pyfunc_wrapper")

    ppo_data_dumper_counter = 0
    dumper_path = os.path.join(epoch_data_dir, "dumper")
    tf.gfile.MakeDirs(dumper_path)
    dumper_spec = [PyFuncWrapper, {"process_fun": ppo_data_dumper}]
    environment_spec.wrappers.insert(2, dumper_spec)

  ppo_hparams.add_hparam("environment_spec", environment_spec)

  with temporary_flags({
      "problem": problem_name,
      "output_dir": world_model_dir,
      "data_dir": epoch_data_dir,
  }):
    rl_trainer_lib.train(ppo_hparams, event_dir + "real", agent_model_dir,
                         name_scope="ppo_real%d" % (epoch + 1))


def evaluate_world_model(simulated_problem_name, problem_name, hparams,
                         world_model_dir, epoch_data_dir, tmp_dir):
  """Generate simulated environment data and return reward accuracy."""
  gym_simulated_problem = registry.problem(simulated_problem_name)
  sim_steps = hparams.simulated_env_generator_num_steps
  gym_simulated_problem.settable_num_steps = sim_steps
  gym_simulated_problem.settable_rollout_fractions = \
      hparams.eval_rollout_fractions
  with temporary_flags({
      "problem": problem_name,
      "model": hparams.generative_model,
      "hparams_set": hparams.generative_model_params,
      "data_dir": epoch_data_dir,
      "output_dir": world_model_dir,
  }):
    gym_simulated_problem.generate_data(epoch_data_dir, tmp_dir)
  n = max(1., gym_simulated_problem.statistics.number_of_dones)
  model_reward_accuracy = [
      (frac, score / float(n))
      for (frac, score) in six.iteritems(
          gym_simulated_problem.statistics.
          successful_episode_reward_predictions
      )
  ]
  old_path = os.path.join(epoch_data_dir, "debug_frames_sim")
  new_path = os.path.join(epoch_data_dir, "debug_frames_sim_eval")
  if not tf.gfile.Exists(new_path):
    tf.gfile.Rename(old_path, new_path)
  return model_reward_accuracy


def train_world_model(problem_name, data_dir, output_dir, hparams, epoch):
  """Train the world model on problem_name."""
  train_steps = hparams.model_train_steps * (
      epoch + hparams.inital_epoch_train_steps_multiplier)
  model_hparams = trainer_lib.create_hparams(hparams.generative_model_params)
  learning_rate = model_hparams.learning_rate_constant
  if epoch > 0: learning_rate *= hparams.learning_rate_bump
  with temporary_flags({
      "data_dir": data_dir,
      "output_dir": output_dir,
      "problem": problem_name,
      "model": hparams.generative_model,
      "hparams_set": hparams.generative_model_params,
      "hparams": "learning_rate_constant=%.6f" % learning_rate,
      "eval_steps": 100,
      "local_eval_frequency": 2000,
      "train_steps": train_steps,
  }):
    t2t_trainer.main([])


def encode_dataset(model, dataset, problem, ae_hparams, autoencoder_path,
                   out_files):
  """Encode all frames in dataset with model and write them out to out_files."""
  batch_size = 8
  dataset = dataset.batch(batch_size)
  examples = dataset.make_one_shot_iterator().get_next()
  images = examples.pop("frame")
  images = tf.cast(images, tf.int32)

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
          rewards = examples_np["reward"].tolist()
          actions = examples_np["action"].tolist()
          frame_numbers = examples_np["frame_number"].tolist()
          for action, reward, frame_number, png in \
                  zip(actions, rewards, frame_numbers, pngs_np):
            yield {
                "action": action,
                "reward": reward,
                "frame_number": frame_number,
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


def encode_env_frames(problem_name, ae_problem_name, ae_hparams_set,
                      autoencoder_path, epoch_data_dir):
  """Encode all frames from problem_name and write out as ae_problem_name."""
  with tf.Graph().as_default():
    ae_hparams = trainer_lib.create_hparams(ae_hparams_set,
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
      encode_dataset(model, dataset, problem=problem, ae_hparams=ae_hparams,
                     autoencoder_path=autoencoder_path,
                     out_files=ae_training_paths)

    # Encode eval data
    if not skip_eval:
      dataset = problem.dataset(tf.estimator.ModeKeys.EVAL, epoch_data_dir,
                                shuffle_files=False, output_buffer_size=100,
                                preprocess=False)
      encode_dataset(model, dataset, problem=problem, ae_hparams=ae_hparams,
                     autoencoder_path=autoencoder_path, out_files=ae_eval_paths)


def check_problems(problem_names):
  for problem_name in problem_names:
    registry.problem(problem_name)


def setup_problems(hparams, using_autoencoder=False):
  """Register problems based on game name."""
  if hparams.game in gym_problems_specs.ATARI_GAMES:
    game_with_mode = hparams.game + "_deterministic-v4"
  else:
    game_with_mode = hparams.game
  game_problems_kwargs = {}
  # Problems
  if using_autoencoder:
    game_problems_kwargs["autoencoder_hparams"] = (
        hparams.autoencoder_hparams_set)
    problem_name = (
        "gym_discrete_problem_with_agent_on_%s_with_autoencoder"
        % game_with_mode)
    world_model_problem = (
        "gym_discrete_problem_with_agent_on_%s_autoencoded" % game_with_mode)
    simulated_problem_name = (
        "gym_simulated_discrete_problem_with_agent_on_%s_autoencoded"
        % game_with_mode)
    world_model_eval_problem_name = (
        "gym_simulated_discrete_problem_for_world_model_eval_with_agent_on_%s"
        "_autoencoded"
        % game_with_mode)
  else:
    problem_name = ("gym_discrete_problem_with_agent_on_%s" % game_with_mode)
    world_model_problem = problem_name
    simulated_problem_name = ("gym_simulated_discrete_problem_with_agent_on_%s"
                              % game_with_mode)
    world_model_eval_problem_name = (
        "gym_simulated_discrete_problem_for_world_model_eval_with_agent_on_%s"
        % game_with_mode)
  if problem_name not in registry.list_problems():
    game_problems_kwargs["resize_height_factor"] = hparams.resize_height_factor
    game_problems_kwargs["resize_width_factor"] = hparams.resize_width_factor
    game_problems_kwargs["grayscale"] = hparams.grayscale
    tf.logging.info("Game Problem %s not found; dynamically registering",
                    problem_name)
    gym_problems_specs.create_problems_for_game(
        hparams.game, game_mode="Deterministic-v4", **game_problems_kwargs)
  return (problem_name, world_model_problem, simulated_problem_name,
          world_model_eval_problem_name)


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

  (problem_name, world_model_problem, simulated_problem_name,
   world_model_eval_problem_name) = setup_problems(hparams, using_autoencoder)

  # Autoencoder model dir
  autoencoder_model_dir = directories.get("autoencoder")

  # Timing log function
  log_relative_time = make_relative_timing_fn()

  # Per-epoch state
  epoch_metrics = []
  epoch_data_dirs = []

  ppo_model_dir = None
  data_dir = os.path.join(directories["data"], "initial")
  epoch_data_dirs.append(data_dir)
  # Collect data from the real environment with PPO or random policy.
  if hparams.gather_ppo_real_env_data:
    ppo_model_dir = directories["ppo"]
    tf.logging.info("Initial training of PPO in real environment.")
    ppo_event_dir = os.path.join(directories["world_model"],
                                 "ppo_summaries/initial")
    train_agent_real_env(
        problem_name, ppo_model_dir,
        ppo_event_dir, directories["world_model"], data_dir,
        hparams, epoch=-1, is_final_epoch=False)

  tf.logging.info("Generating real environment data with %s policy",
                  "PPO" if hparams.gather_ppo_real_env_data else "random")
  mean_reward = generate_real_env_data(
      problem_name, ppo_model_dir, hparams, data_dir, directories["tmp"])
  tf.logging.info("Mean reward (random): {}".format(mean_reward))

  eval_metrics_event_dir = os.path.join(directories["world_model"],
                                        "eval_metrics_event_dir")
  eval_metrics_writer = tf.summary.FileWriter(eval_metrics_event_dir)
  model_reward_accuracy_summary = tf.Summary()
  for frac in hparams.eval_rollout_fractions:
    model_reward_accuracy_summary.value.add(
        tag="model_reward_accuracy_{}".format(frac),
        simple_value=None
    )
  mean_reward_summary = tf.Summary()
  mean_reward_summary.value.add(tag="mean_reward",
                                simple_value=None)
  mean_reward_gen_summary = tf.Summary()
  mean_reward_gen_summary.value.add(tag="mean_reward_during_generation",
                                    simple_value=None)

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
                        ae_hparams_set=hparams.autoencoder_hparams_set,
                        autoencoder_path=autoencoder_model_dir,
                        epoch_data_dir=epoch_data_dir)

    # Train world model
    log("Training world model")
    train_world_model(world_model_problem, epoch_data_dir,
                      directories["world_model"], hparams, epoch)

    # Evaluate world model
    model_reward_accuracy = []
    if hparams.eval_world_model:
      log("Evaluating world model")
      model_reward_accuracy = evaluate_world_model(
          world_model_eval_problem_name, world_model_problem, hparams,
          directories["world_model"],
          epoch_data_dir, directories["tmp"])
      log(
          "World model reward accuracy per rollout fraction: %s",
          model_reward_accuracy
      )

    # Train PPO
    log("Training PPO in simulated environment.")
    ppo_event_dir = os.path.join(directories["world_model"],
                                 "ppo_summaries", str(epoch))
    ppo_model_dir = directories["ppo"]
    if not hparams.ppo_continue_training:
      ppo_model_dir = ppo_event_dir
    train_agent(simulated_problem_name, ppo_model_dir,
                ppo_event_dir, directories["world_model"], epoch_data_dir,
                hparams, epoch=epoch, is_final_epoch=is_final_epoch)

    # Train PPO on real env (short)
    log("Training PPO in real environment.")
    train_agent_real_env(
        problem_name, ppo_model_dir,
        ppo_event_dir, directories["world_model"], epoch_data_dir,
        hparams, epoch=epoch, is_final_epoch=is_final_epoch)

    if hparams.stop_loop_early:
      return 0.0

    # Collect data from the real environment.
    generation_mean_reward = None
    if not is_final_epoch:
      log("Generating real environment data.")
      generation_mean_reward = generate_real_env_data(
          problem_name, ppo_model_dir, hparams, epoch_data_dir,
          directories["tmp"], autoencoder_path=autoencoder_model_dir,
          eval_phase=False)
      log("Mean clipped reward during generation: {}".format(
          generation_mean_reward))

    log("Evaluating in real environment.")
    eval_data_dir = os.path.join(epoch_data_dir, "eval")
    mean_reward = generate_real_env_data(
        problem_name, ppo_model_dir, hparams, eval_data_dir,
        directories["tmp"], autoencoder_path=autoencoder_model_dir,
        eval_phase=True, real_reward=True)
    log("Mean eval reward (unclipped): {}".format(mean_reward))

    # Summarize metrics.
    assert model_reward_accuracy is not None
    assert mean_reward is not None
    for ((_, accuracy), summary_value) in zip(
        model_reward_accuracy, model_reward_accuracy_summary.value
    ):
      summary_value.simple_value = accuracy
    mean_reward_summary.value[0].simple_value = mean_reward
    eval_metrics_writer.add_summary(model_reward_accuracy_summary, epoch)
    eval_metrics_writer.add_summary(mean_reward_summary, epoch)
    if generation_mean_reward is not None:
      mean_reward_gen_summary.value[0].simple_value = int(
          generation_mean_reward)
      eval_metrics_writer.add_summary(mean_reward_gen_summary, epoch)
    eval_metrics_writer.flush()

    # Report metrics
    eval_metrics = {"mean_reward": mean_reward}
    eval_metrics.update({
        "model_reward_accuracy_{}".format(frac): accuracy
        for (frac, accuracy) in model_reward_accuracy
    })
    epoch_metrics.append(eval_metrics)
    log("Eval metrics: %s", str(eval_metrics))
    if report_fn:
      report_fn(eval_metrics[report_metric], epoch)

  # Return the evaluation metrics from the final epoch
  return epoch_metrics[-1]


def extract_game_name(data_dir):
  files = tf.gfile.ListDirectory(data_dir)
  matches = [re.findall(r"on_(.*)_deterministic", f) for f in files]
  non_empty_matches = [m for m in matches if m]
  return non_empty_matches[0][0]


def compute_final_evaluation_on_real_environments(hparams, job_results_dir,
                                                  eval_output_file=None):
  """Runs evaluation of PPO policies on environment with real environments."""
  if eval_output_file is None:
    eval_output_file = os.path.join(
        FLAGS.eval_results_dir,
        "result_{}.txt".format(
            os.path.basename(os.path.normpath(job_results_dir))))
  directories = tf.gfile.ListDirectory(job_results_dir)
  results = {}
  tmp_dir = os.path.join(FLAGS.eval_results_dir, "eval_tmp")
  if tf.gfile.Exists(tmp_dir):
    tf.gfile.DeleteRecursively(tmp_dir)
  for directory in directories:
    ppo_model_dir = os.path.join(job_results_dir, directory, "ppo")
    data_dir = os.path.join(job_results_dir, directory, "data/initial")
    hparams.game = extract_game_name(data_dir)
    problem_name, _, _, _ = setup_problems(hparams)

    tf.logging.info("Evaluating in real environment game %s." % hparams.game)
    try:
      mean_reward = int(generate_real_env_data(
          problem_name, ppo_model_dir, hparams,
          os.path.join(tmp_dir, directory),
          "/tmp", autoencoder_path=None,
          eval_phase=True, real_reward=True))
      tf.logging.info(
          "Mean eval reward on {}: {}".format(hparams.game, mean_reward))
    except AttributeError:
      tf.logging.info("No PPO model for: {}".format(ppo_model_dir))
      mean_reward = None
    game_results = results.get(hparams.game, [])
    game_results.append(mean_reward)
    results[hparams.game] = game_results

  with open(eval_output_file, "w") as f:
    for game in sorted(six.iterkeys(results)):
      print("{}:".format(game), file=f, end="")
      for z in reversed(sorted(results[game])):
        print(" {}".format(z), file=f, end="")
      print("", file=f)


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
def rlmb_base():
  return tf.contrib.training.HParams(
      epochs=15,
      # Total frames used for training. This will be distributed evenly across
      # hparams.epochs.
      # This number should be divisible by real_ppo_epoch_length*epochs
      # for our frame accounting to be preceise.
      num_real_env_frames=96000,
      generative_model="next_frame_basic_deterministic",
      generative_model_params="next_frame_pixel_noise",
      ppo_params="ppo_pong_base",
      autoencoder_train_steps=0,
      autoencoder_train_steps_initial_multiplier=10,
      model_train_steps=15000,
      inital_epoch_train_steps_multiplier=3,
      simulated_env_generator_num_steps=2000,
      simulation_random_starts=True,  # Use random starts in PPO.
      # Flip the first random frame in PPO batch for the true beginning.
      simulation_flip_first_random_for_beginning=True,
      intrinsic_reward_scale=0.,
      ppo_epochs_num=1000,  # This should be enough to see something
      # Our simulated envs do not know how to reset.
      # You should set ppo_time_limit to the value you believe that
      # the simulated env produces a reasonable output.
      ppo_time_limit=200,  # TODO(blazej): this param is unused
      # It makes sense to have ppo_time_limit=ppo_epoch_length,
      # though it is not necessary.
      ppo_epoch_length=50,
      ppo_num_agents=16,
      # Do not eval since simulated batch env does not produce dones
      ppo_eval_every_epochs=0,
      ppo_learning_rate=1e-4,  # Will be changed, just so it exists.
      # Whether the PPO agent should be restored from the previous iteration, or
      # should start fresh each time.
      ppo_continue_training=True,
      # Resizing.
      resize_height_factor=2,
      resize_width_factor=2,
      grayscale=False,
      # Bump learning rate after first epoch by 3x.
      # We picked 3x because our default learning rate schedule decreases with
      # 1/square root of step; 1/sqrt(10k) = 0.01 and 1/sqrt(100k) ~ 0.0032
      # so by bumping it up 3x we about "go back" from 100k steps to 10k, which
      # is approximately as much as "going back 1 epoch" would be.
      # In your experiments, you want to optimize this rate to your schedule.
      learning_rate_bump=3.0,

      gather_ppo_real_env_data=True,
      real_ppo_epochs_num=0,
      # This needs to be divisible by real_ppo_effective_num_agents.
      real_ppo_epoch_length=16*200,
      real_ppo_num_agents=1,
      real_ppo_learning_rate=1e-4,
      real_ppo_continue_training=True,
      real_ppo_effective_num_agents=16,
      real_ppo_eval_every_epochs=0,

      game="pong",
      # Whether to evaluate the world model in each iteration of the loop to get
      # the model_reward_accuracy metric.
      eval_world_model=True,
      # Rollout fractions to report reward_accuracy on.
      eval_rollout_fractions=[0.25, 0.5, 1],
      stop_loop_early=False,  # To speed-up tests.
  )


@registry.register_hparams
def rlmb_basetest():
  """Base setting but quicker with only 2 epochs."""
  hparams = rlmb_base()
  hparams.game = "pong"
  hparams.epochs = 2
  hparams.num_real_env_frames = 3200
  hparams.model_train_steps = 100
  hparams.simulated_env_generator_num_steps = 20
  hparams.ppo_epochs_num = 2
  return hparams


@registry.register_hparams
def rlmb_noresize():
  hparams = rlmb_base()
  hparams.resize_height_factor = 1
  hparams.resize_width_factor = 1
  return hparams


@registry.register_hparams
def rlmb_quick():
  """Base setting but quicker with only 2 epochs."""
  hparams = rlmb_base()
  hparams.epochs = 2
  hparams.model_train_steps = 25000
  hparams.ppo_epochs_num = 700
  hparams.ppo_epoch_length = 50
  return hparams


@registry.register_hparams
def rlmb_quick_noresize():
  hparams = rlmb_base()
  hparams.resize_height_factor = 1
  hparams.resize_width_factor = 1
  return hparams


@registry.register_hparams
def rlmb_quick_sd():
  """Quick setting with stochastic discrete model."""
  hparams = rlmb_quick()
  hparams.generative_model = "next_frame_basic_stochastic_discrete"
  hparams.generative_model_params = "next_frame_basic_stochastic_discrete"
  return hparams


@registry.register_hparams
def rlmb_quick_sm():
  """Quick setting with sampling."""
  hparams = rlmb_quick()
  hparams.generative_model_params = "next_frame_sampling"
  return hparams


@registry.register_hparams
def rlmb_base_stochastic():
  """Base setting with a stochastic next-frame model."""
  hparams = rlmb_base()
  hparams.inital_epoch_train_steps_multiplier = 5
  hparams.generative_model = "next_frame_basic_stochastic"
  hparams.generative_model_params = "next_frame_basic_stochastic"
  return hparams


@registry.register_hparams
def rlmb_base_sampling_stochastic():
  """Base setting with a stochastic next-frame model."""
  hparams = rlmb_base()
  hparams.generative_model = "next_frame_basic_stochastic"
  hparams.generative_model_params = "next_frame_sampling_stochastic"
  return hparams


@registry.register_hparams
def rlmb_base_stochastic_discrete():
  """Base setting with stochastic discrete model."""
  hparams = rlmb_base()
  hparams.generative_model = "next_frame_basic_stochastic_discrete"
  hparams.generative_model_params = "next_frame_basic_stochastic_discrete"
  return hparams


@registry.register_hparams
def rlmb_base_stochastic_recurrent():
  """Base setting with recurrent model."""
  hparams = rlmb_base()
  hparams.generative_model = "next_frame_basic_recurrent"
  hparams.generative_model_params = "next_frame_basic_recurrent"
  return hparams


@registry.register_hparams
def rlmb_base_stochastic_discrete_noresize():
  """Base setting with stochastic discrete model."""
  hparams = rlmb_base()
  hparams.generative_model = "next_frame_basic_stochastic_discrete"
  hparams.generative_model_params = "next_frame_basic_stochastic_discrete"
  hparams.resize_height_factor = 1
  hparams.resize_width_factor = 1
  return hparams


@registry.register_hparams
def rlmb_base_sv2p():
  """Base setting with sv2p as world model."""
  hparams = rlmb_base()
  hparams.learning_rate_bump = 1.0
  hparams.generative_model = "next_frame_sv2p"
  hparams.generative_model_params = "next_frame_sv2p_atari"
  return hparams


@registry.register_hparams
def rlmb_base_sv2p_softmax():
  """Base setting with sv2p as world model with softmax."""
  hparams = rlmb_base_sv2p()
  hparams.generative_model_params = "next_frame_sv2p_atari_softmax"
  return hparams


@registry.register_hparams
def rlmb_base_sv2p_deterministic():
  """Base setting with deterministic sv2p as world model."""
  hparams = rlmb_base_sv2p()
  hparams.generative_model_params = "next_frame_sv2p_atari_deterministic"
  return hparams


@registry.register_hparams
def rlmb_base_sv2p_deterministic_softmax():
  """Base setting with deterministic sv2p as world model with softmax."""
  hparams = rlmb_base_sv2p_softmax()
  hparams.generative_model_params = (
      "next_frame_sv2p_atari_softmax_deterministic")
  return hparams


@registry.register_hparams
def rlmb_base_sv2p_flippy30():
  """Base setting with sv2p as world model."""
  hparams = rlmb_base()
  hparams.epochs = 30
  hparams.ppo_epochs_num = 1000
  hparams.model_train_steps = 15000
  hparams.learning_rate_bump = 1.0
  hparams.inital_epoch_train_steps_multiplier = 5
  hparams.generative_model = "next_frame_sv2p"
  hparams.generative_model_params = "next_frame_sv2p_atari"
  return hparams


@registry.register_hparams
def rlmb_base_sv2p_softmax_flippy30():
  """Base setting with sv2p as world model with softmax."""
  hparams = rlmb_base_sv2p_flippy30()
  hparams.generative_model_params = "next_frame_sv2p_atari_softmax"
  return hparams


@registry.register_hparams
def rlmb_base_sv2p_deterministic_flippy30():
  """Base setting with deterministic sv2p as world model."""
  hparams = rlmb_base_sv2p_flippy30()
  hparams.generative_model_params = "next_frame_sv2p_atari_deterministic"
  return hparams


@registry.register_hparams
def rlmb_base_sv2p_deterministic_softmax_flippy30():
  """Base setting with deterministic sv2p as world model with softmax."""
  hparams = rlmb_base_sv2p_softmax_flippy30()
  hparams.generative_model_params = (
      "next_frame_sv2p_atari_softmax_deterministic")
  return hparams


@registry.register_hparams
def rlmb_base_sampling():
  """Base setting with a stochastic next-frame model."""
  hparams = rlmb_base()
  hparams.generative_model_params = "next_frame_sampling"
  return hparams


@registry.register_hparams
def rlmb_base_sampling_noresize():
  hparams = rlmb_base_sampling()
  hparams.resize_height_factor = 1
  hparams.resize_width_factor = 1
  return hparams


@registry.register_hparams
def rlmb_flippy60():
  """Schedule with a lot of epochs (slow)."""
  hparams = rlmb_base_sampling()
  hparams.epochs = 60
  hparams.ppo_epochs_num = 500
  hparams.model_train_steps = 10000
  return hparams


@registry.register_hparams
def rlmb_flippy30():
  """Schedule with a lot of epochs (slow)."""
  hparams = rlmb_base_sampling()
  hparams.epochs = 30
  hparams.ppo_epochs_num = 1000
  hparams.model_train_steps = 15000
  return hparams


@registry.register_hparams
def rlmb_medium():
  """Small set for larger testing."""
  hparams = rlmb_base()
  hparams.num_real_env_frames //= 2
  return hparams


@registry.register_hparams
def rlmb_25k():
  """Small set for larger testing."""
  hparams = rlmb_medium()
  hparams.num_real_env_frames //= 2
  return hparams


@registry.register_hparams
def rlmb_short():
  """Small set for larger testing."""
  hparams = rlmb_base()
  hparams.num_real_env_frames //= 5
  hparams.model_train_steps //= 10
  hparams.ppo_epochs_num //= 10
  return hparams


@registry.register_hparams
def rlmb_model_only():
  hp = rlmb_base()
  hp.epochs = 1
  hp.ppo_epochs_num = 0
  return hp


@registry.register_hparams
def rlmb_tiny():
  """Tiny set for testing."""
  return rlmb_base_sampling().override_from_dict(
      tf.contrib.training.HParams(
          epochs=1,
          num_real_env_frames=128,
          simulated_env_generator_num_steps=64,
          model_train_steps=2,
          ppo_epochs_num=2,
          ppo_time_limit=5,
          ppo_epoch_length=5,
          ppo_num_agents=2,
          real_ppo_epoch_length=36,
          real_ppo_num_agents=1,
          real_ppo_epochs_num=0,
          real_ppo_effective_num_agents=2,
          generative_model_params="next_frame_tiny",
          stop_loop_early=True,
          resize_height_factor=2,
          resize_width_factor=2,
          game="pong",
      ).values())


@registry.register_hparams
def rlmb_tiny_stochastic():
  """Tiny setting with a stochastic next-frame model."""
  hparams = rlmb_tiny()
  hparams.epochs = 1  # Too slow with 2 for regular runs.
  hparams.generative_model = "next_frame_basic_stochastic"
  hparams.generative_model_params = "next_frame_basic_stochastic"
  return hparams


@registry.register_hparams
def rlmb_tiny_recurrent():
  """Tiny setting with a recurrent next-frame model."""
  hparams = rlmb_tiny()
  hparams.epochs = 1  # Too slow with 2 for regular runs.
  hparams.generative_model = "next_frame_basic_recurrent"
  hparams.generative_model_params = "next_frame_basic_recurrent"
  return hparams


@registry.register_hparams
def rlmb_tiny_sv2p():
  """Tiny setting with a tiny sv2p model."""
  hparams = rlmb_tiny()
  hparams.generative_model = "next_frame_sv2p"
  hparams.generative_model_params = "next_frame_sv2p_tiny"
  hparams.grayscale = False
  return hparams


@registry.register_hparams
def rlmb_ae_base():
  """Parameter set for autoencoders."""
  hparams = rlmb_base()
  hparams.ppo_params = "ppo_pong_ae_base"
  hparams.generative_model_params = "next_frame_ae"
  hparams.autoencoder_hparams_set = "autoencoder_discrete_pong"
  hparams.gather_ppo_real_env_data = False
  hparams.autoencoder_train_steps = 5000
  hparams.resize_height_factor = 1
  hparams.resize_width_factor = 1
  hparams.grayscale = False
  return hparams


@registry.register_hparams
def rlmb_ae_basetest():
  """Base AE setting but quicker with only 2 epochs."""
  hparams = rlmb_ae_base()
  hparams.game = "pong"
  hparams.epochs = 2
  hparams.num_real_env_frames = 3200
  hparams.model_train_steps = 100
  hparams.autoencoder_train_steps = 10
  hparams.simulated_env_generator_num_steps = 20
  hparams.ppo_epochs_num = 2
  return hparams


@registry.register_hparams
def rlmb_ae_tiny():
  """Tiny set for testing autoencoders."""
  hparams = rlmb_tiny()
  hparams.ppo_params = "ppo_pong_ae_base"
  hparams.generative_model_params = "next_frame_ae_tiny"
  hparams.autoencoder_hparams_set = "autoencoder_discrete_tiny"
  hparams.gather_ppo_real_env_data = False
  hparams.resize_height_factor = 1
  hparams.resize_width_factor = 1
  hparams.grayscale = False
  hparams.autoencoder_train_steps = 1
  hparams.autoencoder_train_steps_initial_multiplier = 0
  return hparams


@registry.register_hparams
def rlmb_tiny_simulation_deterministic_starts():
  hp = rlmb_tiny()
  hp.simulation_random_starts = False
  return hp


# RangedHParams for tuning
# ==============================================================================
# Note that the items here must be scoped with one of
# HP_SCOPES={loop, model, ppo}, which set hyperparameters for the top-level
# hparams, hp.generative_model_params, and hp.ppo_params, respectively.
@registry.register_ranged_hparams
def rlmb_grid(rhp):
  """Grid over games and frames, and 5 runs each for variance."""
  rhp.set_categorical("loop.game", ["breakout", "pong", "freeway"])
  base = 100000
  medium = base // 2
  small = medium // 2
  rhp.set_discrete("loop.num_real_env_frames", [base, medium, small])

  # Dummy parameter to get 5 runs for each configuration
  rhp.set_discrete("model.moe_loss_coef", list(range(5)))


@registry.register_ranged_hparams
def rlmb_variance(rhp):
  # Dummy parameter to get 5 runs for each configuration
  rhp.set_discrete("model.moe_loss_coef", list(range(5)))
  rhp.set_categorical("loop.game", ["breakout", "pong", "freeway"])


@registry.register_ranged_hparams
def rlmb_variance_nogame(rhp):
  # Dummy parameter to get 20 runs for current configuration.
  rhp.set_discrete("model.moe_loss_coef", list(range(20)))


@registry.register_ranged_hparams
def rlmb_three(rhp):
  rhp.set_discrete("model.moe_loss_coef", list(range(10)))
  rhp.set_categorical("loop.game", ["breakout", "pong", "boxing"])


@registry.register_ranged_hparams
def rlmb_test1(rhp):
  rhp.set_discrete("model.moe_loss_coef", list(range(10)))
  rhp.set_categorical("loop.game", ["breakout", "pong", "boxing"])
  rhp.set_discrete("loop.ppo_learning_rate", [5e-5, 1e-4, 2e-4])
  rhp.set_discrete("ppo.optimization_batch_size", [20, 40])
  rhp.set_discrete("loop.epochs", [3, 6])


@registry.register_ranged_hparams
def rlmb_scheduled_sampling(rhp):
  rhp.set_float("model.scheduled_sampling_prob", 0.0, 1.0)


@registry.register_ranged_hparams
def rlmb_all_games(rhp):
  rhp.set_discrete("model.moe_loss_coef", list(range(5)))
  rhp.set_categorical("loop.game", gym_problems_specs.ATARI_GAMES)


@registry.register_ranged_hparams
def rlmb_whitelisted_games(rhp):
  rhp.set_discrete("model.moe_loss_coef", list(range(10)))
  rhp.set_categorical("loop.game", gym_problems_specs.ATARI_WHITELIST_GAMES)


@registry.register_ranged_hparams
def rlmb_human_score_games(rhp):
  rhp.set_discrete("model.moe_loss_coef", list(range(10)))
  rhp.set_categorical("loop.game",
                      gym_problems_specs.ATARI_GAMES_WITH_HUMAN_SCORE)


@registry.register_ranged_hparams
def rlmb_curious_games10(rhp):
  rhp.set_discrete("model.moe_loss_coef", list(range(10)))
  rhp.set_categorical("loop.game", gym_problems_specs.ATARI_CURIOUS_GAMES)


@registry.register_ranged_hparams
def rlmb_curious_games5(rhp):
  rhp.set_discrete("model.moe_loss_coef", list(range(5)))
  rhp.set_categorical("loop.game", gym_problems_specs.ATARI_CURIOUS_GAMES)


@registry.register_ranged_hparams
def rlmb_debug_games(rhp):
  rhp.set_discrete("model.moe_loss_coef", list(range(10)))
  rhp.set_categorical("loop.game", gym_problems_specs.ATARI_DEBUG_GAMES)


@registry.register_ranged_hparams
def rlmb_ae_variance(rhp):
  # Dummy parameter to get 5 runs for each configuration
  rhp.set_discrete("model.moe_loss_coef", list(range(5)))
  rhp.set_categorical("loop.game", ["breakout", "pong", "freeway"])
  base = 100000
  small = base // 4
  rhp.set_discrete("loop.num_real_env_frames", [base, small])


@registry.register_ranged_hparams
def rlmb_ppolr_game(rhp):
  rhp.set_categorical("loop.game", ["breakout", "pong", "freeway"])
  base_lr = 1e-4
  rhp.set_float("loop.ppo_learning_rate", base_lr / 2, base_lr * 2)


@registry.register_ranged_hparams
def rlmb_ppolr(rhp):
  base_lr = 1e-4
  rhp.set_float("loop.ppo_learning_rate", base_lr / 2, base_lr * 2)


@registry.register_ranged_hparams
def rlmb_ae_ppo_lr(rhp):
  rhp.set_categorical("loop.game", ["breakout", "pong", "freeway"])
  base_lr = 1e-4
  rhp.set_float("loop.ppo_learning_rate", base_lr / 2, base_lr * 2)


@registry.register_ranged_hparams
def rlmb_dropout_range(rhp):
  rhp.set_float("model.dropout", 0.2, 0.4)


@registry.register_ranged_hparams
def rlmb_intrinsic_reward_scale(rhp):
  rhp.set_float("loop.intrinsic_reward_scale", 0.01, 10.)


@registry.register_ranged_hparams
def rlmb_l1l2cutoff_range(rhp):
  """Loss and loss-cutoff tuning grid."""
  rhp.set_float("model.video_modality_loss_cutoff", 1.4, 3.4)


@registry.register_ranged_hparams
def rlmb_xentcutoff_range(rhp):
  """Cross entropy cutoff tuning grid."""
  rhp.set_float("model.video_modality_loss_cutoff", 0.01, 0.05)


@registry.register_ranged_hparams
def rlmb_pixel_noise(rhp):
  """Input pixel noise tuning grid."""
  rhp.set_categorical("loop.generative_model_params",
                      ["next_frame_pixel_noise"])
  rhp.set_discrete("model.video_modality_input_noise",
                   [0.0025 * i for i in range(200)])


@registry.register_ranged_hparams
def rlmb_dummy_range(rhp):
  """Dummy tuning grid just to get the variance."""
  rhp.set_float("model.moe_loss_coef", 0.01, 0.02)


@registry.register_ranged_hparams
def rlmb_epochs_num(rhp):
  rhp.set_categorical("loop.game", gym_problems_specs.ATARI_WHITELIST_GAMES)
  rhp.set_discrete("model.moe_loss_coef", list(range(5)))
  rhp.set_discrete("loop.epochs", [3, 6, 12])


@registry.register_ranged_hparams
def rlmb_ppo_epochs_num(rhp):
  rhp.set_categorical("loop.game", gym_problems_specs.ATARI_WHITELIST_GAMES)
  rhp.set_discrete("model.moe_loss_coef", list(range(5)))
  rhp.set_discrete("loop.ppo_epochs_num", [200, 1000, 2000, 4000])


@registry.register_ranged_hparams
def rlmb_ppo_epoch_len(rhp):
  rhp.set_categorical("loop.game", gym_problems_specs.ATARI_WHITELIST_GAMES)
  rhp.set_discrete("model.moe_loss_coef", list(range(5)))
  rhp.set_discrete("loop.ppo_epoch_length", [25, 50, 100])


@registry.register_ranged_hparams
def rlmb_num_frames(rhp):
  rhp.set_categorical("loop.game", gym_problems_specs.ATARI_WHITELIST_GAMES)
  rhp.set_discrete("model.moe_loss_coef", list(range(5)))
  rhp.set_discrete("loop.num_real_env_frames",
                   [1000*el for el in [30, 100, 500, 1000]])


@registry.register_ranged_hparams
def rlmb_ppo_optimization_batch_size(rhp):
  rhp.set_categorical("loop.game", ["pong", "boxing", "seaquest"])
  rhp.set_discrete("model.moe_loss_coef", list(range(10)))
  rhp.set_discrete("ppo.optimization_batch_size", [4, 10, 20])


@registry.register_ranged_hparams
def rlmb_logits_clip(rhp):
  rhp.set_categorical("loop.game", ["pong", "boxing", "seaquest"])
  rhp.set_discrete("model.moe_loss_coef", list(range(10)))
  rhp.set_discrete("ppo.logits_clip", [0., 5.])


def merge_unscoped_hparams(scopes_and_hparams):
  """Merge multiple HParams into one with scopes."""
  merged_values = {}
  for (scope, hparams) in scopes_and_hparams:
    for key, value in six.iteritems(hparams.values()):
      scoped_key = "%s.%s" % (scope, key)
      merged_values[scoped_key] = value

  return tf.contrib.training.HParams(**merged_values)


def split_scoped_hparams(scopes, merged_hparams):
  """Split single HParams with scoped keys into multiple."""
  split_values = dict([(scope, dict()) for scope in scopes])
  merged_values = merged_hparams.values()
  for scoped_key, value in six.iteritems(merged_values):
    scope = scoped_key.split(".")[0]
    key = scoped_key[len(scope) + 1:]
    split_values[scope][key] = value

  return [
      tf.contrib.training.HParams(**split_values[scope]) for scope in scopes
  ]


def training_loop_hparams_from_scoped_overrides(scoped_overrides, trial_id):
  """Create HParams suitable for training loop from scoped HParams.

  Args:
    scoped_overrides: HParams, with keys all scoped by one of HP_SCOPES. These
      parameters are overrides for the base HParams created by
      create_loop_hparams.
    trial_id: str, trial identifier. This is used to register unique HParams
      names for the underlying model and ppo HParams.

  Returns:
    HParams suitable for passing to training_loop.
  """
  trial_hp_overrides = scoped_overrides.values()

  # Create loop, model, and ppo base HParams
  loop_hp = create_loop_hparams()
  model_hp_name = trial_hp_overrides.get(
      "loop.generative_model_params", loop_hp.generative_model_params)
  model_hp = registry.hparams(model_hp_name).parse(FLAGS.hparams)
  ppo_params_name = trial_hp_overrides.get(
      "loop.ppo_params", loop_hp.ppo_params)
  ppo_hp = registry.hparams(ppo_params_name)

  # Merge them and then override with the scoped overrides
  combined_hp = merge_unscoped_hparams(
      zip(HP_SCOPES, [loop_hp, model_hp, ppo_hp]))
  combined_hp.override_from_dict(trial_hp_overrides)

  # Split out the component hparams
  loop_hp, model_hp, ppo_hp = (
      split_scoped_hparams(HP_SCOPES, combined_hp))

  # Dynamic register the model hp and set the new name in loop_hp
  model_hp_name = "model_hp_%s" % str(trial_id)
  dynamic_register_hparams(model_hp_name, model_hp)
  loop_hp.generative_model_params = model_hp_name

  # Dynamic register the PPO hp and set the new name in loop_hp
  ppo_hp_name = "ppo_hp_%s" % str(trial_id)
  dynamic_register_hparams(ppo_hp_name, ppo_hp)
  loop_hp.ppo_params = ppo_hp_name

  return loop_hp


def dynamic_register_hparams(name, hparams):

  @registry.register_hparams(name)
  def new_hparams_set():
    return tf.contrib.training.HParams(**hparams.values())

  return new_hparams_set


def create_loop_hparams():
  hparams = registry.hparams(FLAGS.loop_hparams_set)
  hparams.parse(FLAGS.loop_hparams)
  return hparams


def main(_):
  hp = create_loop_hparams()
  if FLAGS.job_dir_to_evaluate:
    compute_final_evaluation_on_real_environments(hp, FLAGS.job_dir_to_evaluate)
  else:
    training_loop(hp, FLAGS.output_dir)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
