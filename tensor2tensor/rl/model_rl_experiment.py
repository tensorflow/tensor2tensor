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
    --rl_hparams_set=rl_modelrl_first \
    --rl_hparams='true_env_generator_num_steps=10000,epochs=3'
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import time

# Dependency imports

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.rl import rl_trainer_lib
from tensor2tensor.rl.envs.tf_atari_wrappers import MaxAndSkipWrapper
from tensor2tensor.rl.envs.tf_atari_wrappers import TimeLimitWrapper
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("rl_hparams_set", "rl_modelrl_first",
                    "Which RL hparams set to use.")
flags.DEFINE_string("rl_hparams", "", "Overrides for RL-specific HParams.")


def train(hparams, output_dir):
  """Training function."""
  prefix = output_dir
  data_dir = os.path.expanduser(prefix + "/data")
  tmp_dir = os.path.expanduser(prefix + "/tmp")
  output_dir = os.path.expanduser(prefix + "/output")
  tf.gfile.MakeDirs(data_dir)
  tf.gfile.MakeDirs(tmp_dir)
  tf.gfile.MakeDirs(output_dir)
  last_model_dir = ""
  start_time = time.time()
  line = ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    "
  epoch_metrics = []
  for iloop in range(hparams.epochs):
    # Generate environment frames
    time_delta = time.time() - start_time
    tf.logging.info("%s Step %d.1 - generate data from policy. Time: %s",
                    line, iloop, str(datetime.timedelta(seconds=time_delta)))
    FLAGS.problem = "gym_discrete_problem_with_agent_on_%s" % hparams.game
    FLAGS.agent_policy_path = last_model_dir
    gym_problem = registry.problem(FLAGS.problem)
    gym_problem.settable_num_steps = hparams.true_env_generator_num_steps
    iter_data_dir = os.path.join(data_dir, str(iloop))
    tf.gfile.MakeDirs(iter_data_dir)
    gym_problem.generate_data(iter_data_dir, tmp_dir)

    time_delta = time.time() - start_time
    tf.logging.info("%s Step %d.2 - generate env model. Time: %s",
                    line, iloop, str(datetime.timedelta(seconds=time_delta)))

    # Train env model
    FLAGS.data_dir = iter_data_dir
    FLAGS.output_dir = output_dir
    FLAGS.model = hparams.generative_model
    FLAGS.hparams_set = hparams.generative_model_params
    FLAGS.train_steps = hparams.model_train_steps * (iloop + 2)
    FLAGS.eval_steps = 10
    t2t_trainer.main([])

    # Evaluate and dump frames from env model
    time_delta = time.time() - start_time
    tf.logging.info("%s Step %d.3 - evaluate env model. Time: %s",
                    line, iloop, str(datetime.timedelta(seconds=time_delta)))
    gym_simulated_problem = registry.problem(
        "gym_simulated_discrete_problem_with_agent_on_%s" % hparams.game)
    sim_steps = hparams.simulated_env_generator_num_steps
    gym_simulated_problem.settable_num_steps = sim_steps
    gym_simulated_problem.generate_data(iter_data_dir, tmp_dir)
    model_reward_accuracy = (
        gym_simulated_problem.successful_episode_reward_predictions / sim_steps)

    # Train PPO agent
    time_delta = time.time() - start_time
    tf.logging.info("%s Step %d.4 - train PPO in model env. Time: %s",
                    line, iloop, str(datetime.timedelta(seconds=time_delta)))

    # Setup PPO hparams
    ppo_hparams = trainer_lib.create_hparams("ppo_atari_base",
                                             data_dir=output_dir)
    ppo_epochs_num = hparams.ppo_epochs_num
    ppo_hparams.epochs_num = ppo_epochs_num
    ppo_hparams.simulated_environment = True
    ppo_hparams.simulated_eval_environment = False
    ppo_hparams.eval_every_epochs = ppo_epochs_num
    ppo_hparams.save_models_every_epochs = ppo_epochs_num
    ppo_hparams.epoch_length = hparams.ppo_epoch_length
    ppo_hparams.num_agents = hparams.ppo_num_agents

    in_graph_wrappers = [
        (TimeLimitWrapper, {"timelimit": hparams.ppo_time_limit}),
        (MaxAndSkipWrapper, {"skip": 4})]
    in_graph_wrappers += gym_problem.in_graph_wrappers
    ppo_hparams.add_hparam("in_graph_wrappers", in_graph_wrappers)

    ppo_dir = generator_utils.make_tmp_dir(dir=data_dir, prefix="ppo_")
    eval_env_res = rl_trainer_lib.train(
        ppo_hparams, gym_simulated_problem.env_name, ppo_dir)
    last_model_dir = ppo_dir
    tf.logging.info("%s Phase %d finished - eval env result = %f. Time: %s",
                    line, iloop, eval_env_res,
                    str(datetime.timedelta(seconds=time_delta)))

    eval_metrics = {"model_reward_accuracy": model_reward_accuracy,
                    "eval_env_res": eval_env_res}
    epoch_metrics.append(eval_metrics)

  # Report the evaluation metrics from the final epoch
  return epoch_metrics[-1]


@registry.register_hparams
def rl_modelrl_tiny():
  # This is a tiny set for testing.
  return tf.contrib.training.HParams(
      epochs=2,
      true_env_generator_num_steps=20,
      generative_model="basic_conv_gen",
      generative_model_params="basic_conv",
      model_train_steps=10,
      simulated_env_generator_num_steps=20,
      ppo_epochs_num=2,
      # Our simulated envs do not know how to reset.
      # You should set ppo_time_limit to the value you believe that
      # the simulated env produces a reasonable output.
      ppo_time_limit=200,
      # It makes sense to have ppo_time_limit=ppo_epoch_length,
      # though it is not necessary.
      ppo_epoch_length=200,
      ppo_num_agents=1,
      game="wrapped_pong",
  )


@registry.register_hparams
def rl_modelrl_small():
  return tf.contrib.training.HParams(
      epochs=10,
      true_env_generator_num_steps=300,
      generative_model="basic_conv_gen",
      generative_model_params="basic_conv",
      model_train_steps=100,
      simulated_env_generator_num_steps=210,
      ppo_epochs_num=200,
      # Our simulated envs do not know how to reset.
      # You should set ppo_time_limit to the value you believe that
      # the simulated env produces a reasonable output.
      ppo_time_limit=200,
      # It makes sense to have ppo_time_limit=ppo_epoch_length,
      # though it is not necessary.
      ppo_epoch_length=200,
      ppo_num_agents=1,
      game="wrapped_pong",
  )


@registry.register_hparams
def rl_modelrl_first():
  return tf.contrib.training.HParams(
      epochs=10,
      true_env_generator_num_steps=60000,
      generative_model="basic_conv_gen",
      generative_model_params="basic_conv",
      model_train_steps=50000,
      simulated_env_generator_num_steps=2000,
      ppo_epochs_num=2000,  # This should be enough to see something
      ppo_time_limit=200,
      ppo_epoch_length=200,  # 200 worked with the standard pong.
      ppo_num_agents=1,
      game="wrapped_pong",
  )


@registry.register_hparams
def rl_modelrl_v1():
  return tf.contrib.training.HParams(
      epochs=10,
      true_env_generator_num_steps=50000,
      generative_model="basic_conv_gen",
      generative_model_params="basic_conv",
      model_train_steps=50000,
      simulated_env_generator_num_steps=300,
      ppo_epochs_num=2000,
      ppo_epoch_length=300,
      game="pong",
  )


def create_rl_hparams():
  hparams = registry.hparams(FLAGS.rl_hparams_set)()
  hparams.parse(FLAGS.rl_hparams)
  return hparams


def main(_):
  hp = create_rl_hparams()
  output_dir = FLAGS.output_dir
  train(hp, output_dir)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
