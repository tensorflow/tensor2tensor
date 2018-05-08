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
"""Training of model-based RL agents."""

import datetime
import os
import tempfile
import time

# Dependency imports

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.rl import rl_trainer_lib
from tensor2tensor.rl.envs.tf_atari_wrappers import ShiftRewardWrapper
from tensor2tensor.rl.envs.tf_atari_wrappers import TimeLimitWrapper
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS


def train(hparams, output_dir):
  """Training function."""
  prefix = output_dir
  data_dir = os.path.expanduser(prefix + "/data")
  tmp_dir = os.path.expanduser(prefix + "/tmp")
  output_dir = os.path.expanduser(prefix + "/output")
  tf.gfile.MakeDirs(data_dir)
  tf.gfile.MakeDirs(tmp_dir)
  tf.gfile.MakeDirs(output_dir)
  last_model = ""
  start_time = time.time()
  line = ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>    "
  for iloop in range(hparams.epochs):
    time_delta = time.time() - start_time
    print(line+"Step {}.1. - generate data from policy. "
          "Time: {}".format(iloop, str(datetime.timedelta(seconds=time_delta))))
    FLAGS.problem = "gym_discrete_problem_with_agent_on_%s" % hparams.game
    FLAGS.agent_policy_path = last_model
    gym_problem = registry.problem(FLAGS.problem)
    gym_problem.settable_num_steps = hparams.true_env_generator_num_steps
    iter_data_dir = os.path.join(data_dir, str(iloop))
    tf.gfile.MakeDirs(iter_data_dir)
    gym_problem.generate_data(iter_data_dir, tmp_dir)

    time_delta = time.time() - start_time
    print(line+"Step {}.2. - generate env model. "
          "Time: {}".format(iloop, str(datetime.timedelta(seconds=time_delta))))
    # 2. generate env model
    FLAGS.data_dir = iter_data_dir
    FLAGS.output_dir = output_dir
    FLAGS.model = hparams.generative_model
    FLAGS.hparams_set = hparams.generative_model_params
    FLAGS.train_steps = hparams.model_train_steps * (iloop + 2)
    FLAGS.eval_steps = 10
    t2t_trainer.main([])

    # Dump frames from env model.
    time_delta = time.time() - start_time
    print(line+"Step {}.3. - evalue env model. "
          "Time: {}".format(iloop, str(datetime.timedelta(seconds=time_delta))))
    gym_simulated_problem = registry.problem(
        "gym_simulated_discrete_problem_with_agent_on_%s" % hparams.game)
    sim_steps = hparams.simulated_env_generator_num_steps
    gym_simulated_problem.settable_num_steps = sim_steps
    gym_simulated_problem.generate_data(iter_data_dir, tmp_dir)

    # PPO.
    time_delta = time.time() - start_time
    print(line + "Step {}.4. - train PPO in model env."
          " Time: {}".format(iloop,
                             str(datetime.timedelta(seconds=time_delta))))
    ppo_epochs_num = hparams.ppo_epochs_num
    ppo_hparams = trainer_lib.create_hparams(
        "atari_base",
        "epochs_num={},simulated_environment=True,eval_every_epochs=0,"
        "save_models_every_epochs={}".format(ppo_epochs_num+1, ppo_epochs_num),
        data_dir=output_dir)
    ppo_hparams.epoch_length = hparams.ppo_epoch_length
    ppo_dir = tempfile.mkdtemp(dir=data_dir, prefix="ppo_")
    in_graph_wrappers = [
        (TimeLimitWrapper, {"timelimit": 150}),
        (ShiftRewardWrapper, {"add_value": -2})]
    in_graph_wrappers += gym_problem.in_graph_wrappers
    ppo_hparams.add_hparam("in_graph_wrappers", in_graph_wrappers)
    ppo_hparams.num_agents = 1
    rl_trainer_lib.train(ppo_hparams, "PongDeterministic-v4", ppo_dir)

    last_model = ppo_dir + "/model{}.ckpt".format(ppo_epochs_num)


def main(_):
  hparams = tf.contrib.training.HParams(
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
  train(hparams, FLAGS.output_dir)


if __name__ == "__main__":
  tf.app.run()
